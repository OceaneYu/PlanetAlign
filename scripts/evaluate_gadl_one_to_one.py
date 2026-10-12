#!/usr/bin/env python3
# 文件作用：固定配置训练 GADL，比较原始属性、未训练表示和最终表示，并诊断嵌入坍缩与同分排名。
# 核心依赖：PlanetAlign/algorithms/gadl/main.py → GADL.train()；PlanetAlign/metrics/ranking.py 的排名与诊断函数。
# 核心处理：_run_seed() 运行一个种子，测试节点对只用于训练结束后的评估，不选择检查点。
# 结果去向：main() 汇总多个种子并写 JSON/CSV；其他 GADL 脚本复用本文件的数据和评估辅助函数。

"""Reproducible one-to-one GADL evaluation and collapse diagnostics.

The script never uses evaluation pairs for training, early stopping, model
selection, or hyperparameter selection.  It runs one fixed configuration for
every requested seed and evaluates only after training has completed.
"""

import argparse
import csv
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Data

from PlanetAlign.algorithms import GADL
from PlanetAlign.algorithms.gadl.model import (
    edge_index_to_undirected_binary_adjacency,
)
from PlanetAlign.data import BaseData, Dataset
from PlanetAlign.metrics import pairwise_ranks, rank_diagnostics
from PlanetAlign.utils import get_anchor_pairs


METRIC_KS = (1, 5, 10, 50)
MODE_NAMES = ("raw_attribute_cosine", "untrained_gadl", "trained_gadl")
DIRECTION_NAMES = ("source_to_target", "target_to_source")
DELTA_SPECS = {
    "untrained_minus_raw_attribute": (
        "untrained_gadl", "raw_attribute_cosine"
    ),
    "trained_minus_untrained": ("trained_gadl", "untrained_gadl"),
    "trained_minus_raw_attribute": ("trained_gadl", "raw_attribute_cosine"),
}
COMPARISON_FIELDS = (
    *(f"Hits@{k}" for k in METRIC_KS),
    "MRR",
    "mean_rank",
    "true_similarity_mean",
    "wrong_similarity_mean",
    "similarity_std",
    "maximum_tie_rate",
    "average_maximum_candidates",
    "unique_top1_ratio",
)


def _summary(values):
    values = values.detach().to(torch.float64).flatten().cpu()
    if values.numel() == 0:
        return {"count": 0, "mean": None, "std": None,
                "min": None, "q25": None, "median": None,
                "q75": None, "max": None}
    quantiles = torch.quantile(values, torch.tensor(
        [0.25, 0.5, 0.75], dtype=torch.float64
    ))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "q25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q75": float(quantiles[2]),
        "max": float(values.max()),
    }


# 功能：按指定同分策略计算单向 Hits/MRR/平均名次，同时记录真假节点分数和并列诊断。
def _directional_metrics(
    similarity, pairs, tie_policy="optimistic", tie_atol=1e-8, tie_rtol=1e-7
):
    similarity = similarity.detach().cpu()
    pairs = pairs.detach().long().cpu()
    query_scores = similarity[pairs[:, 0]]
    truth_scores = query_scores[torch.arange(len(pairs)), pairs[:, 1]]
    ranks = pairwise_ranks(
        similarity, pairs, direction="ltr", tie_policy=tie_policy,
        atol=tie_atol, rtol=tie_rtol
    ).cpu()
    wrong_mask = torch.ones_like(query_scores, dtype=torch.bool)
    wrong_mask[torch.arange(len(pairs)), pairs[:, 1]] = False
    return {
        **{f"Hits@{k}": float((ranks <= k).float().mean()) for k in METRIC_KS},
        "MRR": float((1.0 / ranks).mean()),
        "mean_rank": float(ranks.mean()),
        "true_similarity": _summary(truth_scores),
        "wrong_similarity": _summary(query_scores[wrong_mask]),
        **rank_diagnostics(
            similarity, pairs, direction="ltr", atol=tie_atol, rtol=tie_rtol
        ),
    }


# 功能：交换两侧矩阵与节点对做双向评估，返回单向记录及均值。
def _bidirectional_metrics(
    similarity, pairs, tie_policy="optimistic", tie_atol=1e-8, tie_rtol=1e-7
):
    left_to_right = _directional_metrics(
        similarity, pairs, tie_policy, tie_atol, tie_rtol
    )
    right_to_left = _directional_metrics(
        similarity.T, pairs[:, [1, 0]], tie_policy, tie_atol, tie_rtol
    )
    average = {
        key: 0.5 * (left_to_right[key] + right_to_left[key])
        for key in [*(f"Hits@{k}" for k in METRIC_KS), "MRR", "mean_rank"]
    }
    return {"source_to_target": left_to_right,
            "target_to_source": right_to_left,
            "bidirectional_mean": average}


# 功能：在同一 S 上逐同分策略评估，并报告矩阵分布和是否近乎常量。
def _evaluate_mode(similarity, pairs, tie_policies, tie_atol, tie_rtol):
    similarity = similarity.detach().cpu()
    return {
        "tie_policies": {
            tie_policy: {
                "metrics": _bidirectional_metrics(
                    similarity, pairs, tie_policy, tie_atol, tie_rtol
                )
            }
            for tie_policy in tie_policies
        },
        "similarity": _summary(similarity),
        "similarity_collapsed": float(
            similarity.std(unbiased=False)
        ) < 1e-6,
    }


def _comparison_values(mode_result, direction, tie_policy="average"):
    metrics = mode_result["tie_policies"][tie_policy]["metrics"][direction]
    return {
        **{f"Hits@{k}": metrics[f"Hits@{k}"] for k in METRIC_KS},
        "MRR": metrics["MRR"],
        "mean_rank": metrics["mean_rank"],
        "true_similarity_mean": metrics["true_similarity"]["mean"],
        "wrong_similarity_mean": metrics["wrong_similarity"]["mean"],
        "similarity_std": mode_result["similarity"]["std"],
        "maximum_tie_rate": metrics["maximum_tie_rate"],
        "average_maximum_candidates": metrics["average_maximum_candidates"],
        "unique_top1_ratio": metrics["unique_top1_ratio"],
    }


# 输入来源：_run_seed() 传入原属性、初始表示和最终表示的评估记录，比较关系由 DELTA_SPECS 指定。
# 输出结果：按方向计算两模式在平均同分策略下的指标差值，交回单种子结果用于跨种子汇总。
def _paired_deltas(modes):
    deltas = {}
    for delta_name, (left_mode, right_mode) in DELTA_SPECS.items():
        deltas[delta_name] = {}
        for direction in DIRECTION_NAMES:
            left = _comparison_values(modes[left_mode], direction, "average")
            right = _comparison_values(modes[right_mode], direction, "average")
            deltas[delta_name][direction] = {
                field: left[field] - right[field]
                for field in COMPARISON_FIELDS
            }
    return deltas


# 功能：检查节点向量范数与各维方差，帮助识别所有节点表示变得相同的坍缩。
def _embedding_diagnostics(embedding):
    norms = torch.linalg.vector_norm(embedding.detach(), dim=1)
    dimension_variance = embedding.detach().to(torch.float64).var(
        dim=0, unbiased=False
    )
    mean_variance = float(dimension_variance.mean())
    return {
        "norm": _summary(norms),
        "dimension_variance": _summary(dimension_variance),
        "mean_dimension_variance": mean_variance,
        "collapsed": mean_variance < 1e-8,
    }


# 功能：区分真实边与非边，比较嵌入内积的分布，不修改模型。
def _reconstruction_diagnostics(graph, embedding):
    adjacency = edge_index_to_undirected_binary_adjacency(
        graph.edge_index, graph.num_nodes, dtype=embedding.dtype, device="cpu"
    ).to_dense().bool()
    logits = embedding.detach().cpu() @ embedding.detach().cpu().T
    positive = logits[adjacency]
    negative = logits[~adjacency]
    return {
        "positive_count": int(positive.numel()),
        "negative_count": int(negative.numel()),
        "negative_to_positive_ratio": (
            float(negative.numel() / positive.numel()) if positive.numel() else None
        ),
        "positive_logits": _summary(positive),
        "negative_logits": _summary(negative),
    }


def _peak_memory(device):
    if device.type == "cuda":
        return {
            "kind": "cuda_max_memory_allocated",
            "megabytes": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        }
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
    return {"kind": "process_peak_rss", "megabytes": maximum_rss / divisor}


def _git_state(repository):
    def run(*arguments):
        completed = subprocess.run(
            ["git", *arguments], cwd=repository,
            text=True, capture_output=True, check=False,
        )
        return completed.stdout.strip()

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


# 输出结果：创建已知节点置换的小图样例，用来检查算法能否恢复受控对应关系。
def _synthetic_permutation_dataset():
    generator = torch.Generator().manual_seed(20260807)
    num_nodes, feature_dim = 48, 16
    source_features = torch.randn(num_nodes, feature_dim, generator=generator)
    source_features = torch.nn.functional.normalize(source_features, dim=1)
    source_edges = []
    for node in range(num_nodes):
        source_edges.append((node, (node + 1) % num_nodes))
        source_edges.append((node, (node + 5) % num_nodes))
    source_edge_index = torch.tensor(source_edges, dtype=torch.long).T
    permutation = torch.randperm(num_nodes, generator=generator)
    target_features = torch.empty_like(source_features)
    target_features[permutation] = source_features
    target_edge_index = permutation[source_edge_index]
    graphs = [
        Data(name="synthetic_source", x=source_features,
             edge_index=source_edge_index, num_nodes=num_nodes),
        Data(name="synthetic_target", x=target_features,
             edge_index=target_edge_index, num_nodes=num_nodes),
    ]
    anchors = torch.stack([torch.arange(num_nodes), permutation], dim=1)
    return BaseData(graphs, anchors, name="synthetic_permutation",
                    train_ratio=0.0, seed=0)


# 功能：按命令行模式加载磁盘 Dataset 或生成受控置换样例。
def _load_dataset(args, seed):
    if args.dataset == "synthetic-permutation":
        return _synthetic_permutation_dataset()
    return Dataset(
        root=args.root, name=args.dataset,
        train_ratio=args.train_ratio, seed=seed,
    )


# 功能：按协议保留完整图或筛选 Douban 源图的指定边关系，报告处理前后的边数。
def _apply_douban_protocol(dataset, gids, protocol):
    counts = {
        "source_before": int(dataset.pyg_graphs[gids[0]].num_edges),
        "target_before": int(dataset.pyg_graphs[gids[1]].num_edges),
    }
    if protocol == "baseline-compatible":
        if dataset.name.lower() != "douban":
            raise ValueError("baseline-compatible protocol is only defined for Douban")
        source_graph = dataset.pyg_graphs[gids[0]]
        if source_graph.edge_attr is None or source_graph.edge_attr.shape[1] < 2:
            raise ValueError("Douban baseline protocol requires two-column edge_attr")
        relation_mask = source_graph.edge_attr[:, 1] > 0
        filtered_graph = source_graph.clone()
        filtered_graph.edge_index = source_graph.edge_index[:, relation_mask]
        filtered_graph.edge_attr = source_graph.edge_attr[relation_mask]
        graphs = list(dataset.pyg_graphs)
        graphs[gids[0]] = filtered_graph
        dataset.pyg_graphs = graphs
    counts.update({
        "source_after": int(dataset.pyg_graphs[gids[0]].num_edges),
        "target_after": int(dataset.pyg_graphs[gids[1]].num_edges),
    })
    return counts


def _graph_metadata(graph):
    return {
        "name": getattr(graph, "name", None),
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.num_edges),
        "attribute_dim": int(graph.x.shape[1]),
    }


# 核心思路：加载并应用图协议，调用 GADL.train()，对属性、初始与最终矩阵分别评估。
# 输出结果：单种子的分数、损失、嵌入/重构诊断、时间与内存记录交给 main() 汇总。
def _run_seed(args, repository, seed):
    dataset = _load_dataset(args, seed)
    edge_counts = _apply_douban_protocol(
        dataset, args.gids, args.douban_graph_protocol
    )
    graph1, graph2 = [dataset.pyg_graphs[gid] for gid in args.gids]
    model_parameters = {
        "hidden_dim": args.hidden_dim,
        "num_hidden_layers": args.num_hidden_layers,
        "spectral_rank": args.spectral_rank,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "descriptor_weight": args.descriptor_weight,
        "commutativity_weight": args.commutativity_weight,
        "functional_map_weight": args.functional_map_weight,
        "bijectivity_weight": args.bijectivity_weight,
        "orthogonality_weight": args.orthogonality_weight,
        "residual": args.residual,
        "seed": seed,
    }
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = GADL(**model_parameters).to(device)
    wall_start = time.time()
    similarity, _ = model.train(
        dataset, gids=args.gids, use_attr=True,
        total_epochs=args.total_epochs, save_log=False, verbose=False,
    )
    wall_time = time.time() - wall_start

    evaluation_pairs = get_anchor_pairs(
        dataset.test_data, args.gids[0], args.gids[1]
    )
    raw_similarity = (
        torch.nn.functional.normalize(graph1.x.detach().to(torch.float32), dim=1)
        @ torch.nn.functional.normalize(
            graph2.x.detach().to(torch.float32), dim=1
        ).T
    ).cpu()
    final_similarity = similarity.detach().cpu()
    modes = {
        "raw_attribute_cosine": _evaluate_mode(
            raw_similarity, evaluation_pairs, args.tie_policies,
            args.tie_atol, args.tie_rtol
        ),
        "untrained_gadl": _evaluate_mode(
            model.initial_S, evaluation_pairs, args.tie_policies,
            args.tie_atol, args.tie_rtol
        ),
        "trained_gadl": _evaluate_mode(
            final_similarity, evaluation_pairs, args.tie_policies,
            args.tie_atol, args.tie_rtol
        ),
    }
    paired_deltas = _paired_deltas(modes)

    spectra_finite = all(
        torch.isfinite(component).all().item()
        for spectrum in model.spectra.values()
        for component in spectrum.values()
    )
    result = {
        "seed": seed,
        "dataset": dataset.name,
        "protocol": args.douban_graph_protocol,
        "gids": list(args.gids),
        "parameters": {**model_parameters, "total_epochs": args.total_epochs},
        "graphs": [_graph_metadata(graph1), _graph_metadata(graph2)],
        "edge_counts": edge_counts,
        "anchors": {
            "all": int(dataset._anchor_links.shape[0]),
            "train": int(dataset.train_data.shape[0]),
            "test": int(dataset.test_data.shape[0]),
        },
        "effective_spectral_rank": model.effective_rank,
        "spectra_finite": spectra_finite,
        "loss": {
            "initial": model.initial_losses,
            "final": model.final_losses,
            "history": model.loss_history,
        },
        "time_seconds": {
            "wall": wall_time,
            "spectral": model.spectral_time,
            "training": model.training_time,
        },
        "peak_memory": _peak_memory(device),
        "modes": modes,
        "paired_deltas": paired_deltas,
        "tie_protocol": {
            "policies": args.tie_policies,
            "primary": "average",
            "paired_delta_policy": "average",
            "atol": args.tie_atol,
            "rtol": args.tie_rtol,
        },
        "initial": {
            "metrics": _bidirectional_metrics(model.initial_S, evaluation_pairs),
            "similarity": _summary(model.initial_S),
            "embeddings": {
                str(gid): _embedding_diagnostics(model.initial_embeddings[gid])
                for gid in args.gids
            },
        },
        "final": {
            "metrics": _bidirectional_metrics(final_similarity, evaluation_pairs),
            "similarity": _summary(final_similarity),
            "similarity_collapsed": float(final_similarity.std(unbiased=False)) < 1e-6,
            "embeddings": {
                str(gid): _embedding_diagnostics(model.raw_embeddings[gid])
                for gid in args.gids
            },
            "reconstruction": {
                str(args.gids[0]): _reconstruction_diagnostics(
                    graph1, model.raw_embeddings[args.gids[0]]
                ),
                str(args.gids[1]): _reconstruction_diagnostics(
                    graph2, model.raw_embeddings[args.gids[1]]
                ),
            },
        },
        "git": _git_state(repository),
    }
    return result


def _numeric_path(item, path):
    value = item
    for part in path.split("."):
        value = value[part]
    return float(value)


# 功能：将各种子的可比较指标整理成均值与标准差。
def _aggregate(results):
    def summarize(values):
        values = torch.tensor(values, dtype=torch.float64)
        return {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
        }

    aggregate = {"modes": {}, "paired_deltas": {}, "resources": {}}
    for mode_name in MODE_NAMES:
        aggregate["modes"][mode_name] = {}
        for tie_policy in results[0]["tie_protocol"]["policies"]:
            aggregate["modes"][mode_name][tie_policy] = {}
            for direction in DIRECTION_NAMES:
                aggregate["modes"][mode_name][tie_policy][direction] = {}
                for field in COMPARISON_FIELDS:
                    aggregate["modes"][mode_name][tie_policy][direction][field] = summarize([
                        _comparison_values(
                            result["modes"][mode_name], direction, tie_policy
                        )[field]
                        for result in results
                    ])
    for delta_name in DELTA_SPECS:
        aggregate["paired_deltas"][delta_name] = {}
        for direction in DIRECTION_NAMES:
            aggregate["paired_deltas"][delta_name][direction] = {
                field: summarize([
                    result["paired_deltas"][delta_name][direction][field]
                    for result in results
                ])
                for field in COMPARISON_FIELDS
            }
    for path in (
        "time_seconds.wall", "time_seconds.spectral",
        "time_seconds.training", "peak_memory.megabytes",
    ):
        aggregate["resources"][path] = summarize([
            _numeric_path(result, path) for result in results
        ])
    return aggregate


# 功能：将分模式、分同分策略的结果展平成 CSV，便于比较。
def _write_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        common = {
            "seed": result["seed"],
            "dataset": result["dataset"],
            "protocol": result["protocol"],
        }
        for mode_name in MODE_NAMES:
            for tie_policy in result["tie_protocol"]["policies"]:
                for direction in DIRECTION_NAMES:
                    rows.append({
                        **common,
                        "record_type": "mode",
                        "comparison": mode_name,
                        "tie_policy": tie_policy,
                        "direction": direction,
                        **_comparison_values(
                            result["modes"][mode_name], direction, tie_policy
                        ),
                    })
        for delta_name in DELTA_SPECS:
            for direction in DIRECTION_NAMES:
                rows.append({
                    **common,
                    "record_type": "paired_delta",
                    "comparison": delta_name,
                    "tie_policy": "average",
                    "direction": direction,
                    **result["paired_deltas"][delta_name][direction],
                })
    with path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# 功能：解析数据协议、种子、模型参数及同分评估策略。
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", default="data")
    parser.add_argument("--gids", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--seeds", "--seed", nargs="+", type=int, default=[0])
    parser.add_argument("--train-ratio", type=float, default=0.0)
    parser.add_argument("--total-epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--spectral-rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--descriptor-weight", type=float, default=1e-3)
    parser.add_argument("--commutativity-weight", type=float, default=1.0)
    parser.add_argument("--functional-map-weight", type=float, default=1.0)
    parser.add_argument("--bijectivity-weight", type=float, default=1e-1)
    parser.add_argument("--orthogonality-weight", type=float, default=1e-1)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--tie-policies", nargs="+",
        choices=["optimistic", "average", "pessimistic"],
        default=["optimistic", "average", "pessimistic"],
    )
    parser.add_argument("--tie-atol", type=float, default=1e-8)
    parser.add_argument("--tie-rtol", type=float, default=1e-7)
    parser.add_argument(
        "--douban-graph-protocol",
        choices=["full-graph", "baseline-compatible"],
        default="full-graph",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


# 功能：逐种子运行 _run_seed()，保存完整 JSON 与比较 CSV，不按测试分数选择训练轮次。
def main():
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    results = []
    print("Fixed configuration (no test-set selection):")
    print(json.dumps(vars(args), indent=2, default=str))
    for seed in args.seeds:
        result = _run_seed(args, repository, seed)
        results.append(result)
        for mode_name in MODE_NAMES:
            for tie_policy in args.tie_policies:
                ltr = result["modes"][mode_name]["tie_policies"][tie_policy]["metrics"]["source_to_target"]
                print(
                    f"seed={seed} protocol={result['protocol']} mode={mode_name} "
                    f"tie_policy={tie_policy} Hits@1={ltr['Hits@1']:.4f} "
                    f"Hits@5={ltr['Hits@5']:.4f} MRR={ltr['MRR']:.4f} "
                    f"mean_rank={ltr['mean_rank']:.2f} "
                    f"max_tie_rate={ltr['maximum_tie_rate']:.4f}"
                )
        print(f"wall={result['time_seconds']['wall']:.2f}s")

    payload = {
        "experiment": "GADL one-to-one performance attribution",
        "selection_policy": "fixed configuration; test pairs used only after training",
        "tie_protocol": {
            "policies": args.tie_policies,
            "primary": "average",
            "paired_delta_policy": "average",
            "atol": args.tie_atol,
            "rtol": args.tie_rtol,
        },
        "results": results,
        "aggregate": _aggregate(results),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as file_handle:
        json.dump(payload, file_handle, indent=2, allow_nan=False)
    if args.output_csv:
        _write_csv(args.output_csv, results)
    print(f"Wrote {args.output_json}")
    if args.output_csv:
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
