#!/usr/bin/env python3
# 文件作用：在一次连续 GADL 训练中观察指定轮次，记录分数、损失和表示变化。
# 核心依赖：PlanetAlign/algorithms/gadl/main.py → GADL.train()；scripts/evaluate_gadl_one_to_one.py 的加载和评估函数。
# 核心处理：临时包装目标函数与优化器步数计数，读取分离后的状态，不替换训练目标。
# 结果去向：main() 保存检查点与多种子汇总 JSON，并绘制训练轨迹。

"""Run checkpointed, full-configuration GADL one-to-one evaluation.

Checkpoint observations are attached to a single uninterrupted ``GADL.train``
call.  Epoch zero is captured before the first optimizer step; later epochs are
captured after exactly that many steps.  The observers only detach tensors and
do not alter the objective, gradients, optimizer, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from contextlib import ExitStack
from pathlib import Path
from statistics import mean, pstdev
from unittest.mock import patch

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import PlanetAlign.algorithms.gadl.main as gadl_main
from PlanetAlign.algorithms.gadl import GADL
from evaluate_gadl_one_to_one import (
    _apply_douban_protocol,
    _evaluate_mode,
    _graph_metadata,
    _load_dataset,
    _reconstruction_diagnostics,
)


CHECKPOINT_EPOCHS = (0, 1, 5, 20, 50, 100)
TIE_POLICIES = ("average", "optimistic", "pessimistic")


def _finite_scalar(value):
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _losses(losses):
    return {
        key: scalar
        for key, value in losses.items()
        if (scalar := _finite_scalar(value)) is not None
    }


def _embedding_variance(embedding):
    embedding = embedding.detach().float().cpu()
    per_dimension = embedding.var(dim=0, unbiased=False)
    norms = embedding.norm(dim=1)
    return {
        "mean_per_dimension_variance": float(per_dimension.mean()),
        "min_per_dimension_variance": float(per_dimension.min()),
        "max_per_dimension_variance": float(per_dimension.max()),
        "zero_variance_dimension_ratio": float((per_dimension <= 1e-12).float().mean()),
        "mean_node_norm": float(norms.mean()),
        "std_node_norm": float(norms.std(unbiased=False)),
        "all_finite": bool(torch.isfinite(embedding).all()),
    }


# 功能：用固定嵌入的内积分开统计真实边和非边的重构分数，不执行反向传播。
# 结果去向：_checkpoint() 把这些统计写入观察记录，用来解释重构损失与表示变化。
def _logit_summary(embedding, adjacency):
    embedding = embedding.detach()
    adjacency = adjacency.detach().to(embedding.device)
    if adjacency.is_sparse:
        adjacency = adjacency.to_dense()
    logits = embedding @ embedding.T
    positive = logits[adjacency > 0]
    negative = logits[adjacency <= 0]

    def summarize(values):
        values = values.float()
        return {
            "count": int(values.numel()),
            "mean": float(values.mean().cpu()),
            "std": float(values.std(unbiased=False).cpu()),
            "min": float(values.min().cpu()),
            "max": float(values.max().cpu()),
        }

    return {
        "positive": summarize(positive),
        "negative": summarize(negative),
        "positive_fraction": float((adjacency > 0).float().mean().cpu()),
    }


# 功能：从当前两侧嵌入计算矩阵、各同分策略分数、重构和方差，返回观察记录。
def _checkpoint(epoch, z_src, z_tgt, adjacency_src, adjacency_tgt, losses, pairs, args):
    z_src = z_src.detach()
    z_tgt = z_tgt.detach()
    similarity = F.normalize(z_src, dim=1) @ F.normalize(z_tgt, dim=1).T
    similarity_cpu = similarity.float().cpu()
    evaluations = _evaluate_mode(
        similarity_cpu,
        pairs.cpu(),
        TIE_POLICIES,
        args.tie_atol,
        args.tie_rtol,
    )
    return {
        "epoch": epoch,
        "primary_tie_policy": "average",
        "tie_policies": evaluations["tie_policies"],
        "similarity": {
            "mean": float(similarity_cpu.mean()),
            "std": float(similarity_cpu.std(unbiased=False)),
            "min": float(similarity_cpu.min()),
            "max": float(similarity_cpu.max()),
            "all_finite": bool(torch.isfinite(similarity_cpu).all()),
        },
        "embedding": {
            "source": _embedding_variance(z_src),
            "target": _embedding_variance(z_tgt),
        },
        "losses": _losses(losses),
        "reconstruction_logits": {
            "source": _logit_summary(z_src, adjacency_src),
            "target": _logit_summary(z_tgt, adjacency_tgt),
        },
    }


# 功能：在平均同分策略下计算最终记录减去初始记录的双向指标差值。
# 输出结果：差值交给 _run_seed() 的结果记录；排名指标 mean_rank 变小才表示名次改善。
def _metric_delta(final_checkpoint, initial_checkpoint):
    output = {}
    for direction in ("source_to_target", "target_to_source"):
        final = final_checkpoint["tie_policies"]["average"]["metrics"][direction]
        initial = initial_checkpoint["tie_policies"]["average"]["metrics"][direction]
        output[direction] = {
            key: final[key] - initial[key]
            for key in (
                "Hits@1",
                "Hits@5",
                "Hits@10",
                "Hits@50",
                "MRR",
                "mean_rank",
                "maximum_tie_rate",
                "average_maximum_candidates",
                "unique_top1_ratio",
            )
        }
    return output


# 功能：包装原 optimizer.step()，先执行原更新，再累计已完成步数供检查点定位。
def _optimizer_step_observer(counter, original_step):
    def observed_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        counter["completed"] += 1
        return result

    return observed_step


# 核心思路：在局部观察器范围内调用一次 GADL.train()，按实际优化步数收集指定检查点。
# 输出结果：完整轨迹、检查点、缺失轮次与资源信息交给 main()。
def _run_seed(args, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    dataset = _load_dataset(args, seed)
    protocol = _apply_douban_protocol(dataset, args.gids, args.douban_graph_protocol)
    pairs = dataset.test_data.detach().cpu()
    graphs = [dataset.pyg_graphs[gid] for gid in args.gids]
    model = GADL(
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        spectral_rank=args.spectral_rank,
        lr=args.lr,
        weight_decay=args.weight_decay,
        descriptor_weight=args.descriptor_weight,
        commutativity_weight=args.commutativity_weight,
        functional_map_weight=args.functional_map_weight,
        bijectivity_weight=args.bijectivity_weight,
        orthogonality_weight=args.orthogonality_weight,
        residual=args.residual,
        seed=seed,
    ).to(args.device)

    wanted = {epoch for epoch in CHECKPOINT_EPOCHS if epoch <= args.total_epochs}
    captured = {}
    counter = {"completed": 0}
    original_objective = gadl_main.gadl_objective

    # 输入来源：临时替换 PlanetAlign/algorithms/gadl/main.py 导入的 gadl_objective()，由 GADL.train() 在该调用点进入。
    # 阅读建议：原目标函数定义在 PlanetAlign/algorithms/gadl/model.py → gadl_objective()，包装器仍先执行它。
    # 功能：先执行原目标，再按已完成的 optimizer.step() 次数决定是否保存 _checkpoint()。
    # 输出结果：原损失字典原样返回训练循环；检查点仅写入 captured，不改变优化目标。
    def observed_objective(*objective_args, **objective_kwargs):
        result = original_objective(*objective_args, **objective_kwargs)
        epoch = counter["completed"]
        if epoch in wanted and epoch not in captured:
            captured[epoch] = _checkpoint(
                epoch,
                objective_kwargs.get("z1", objective_args[1]),
                objective_kwargs.get("z2", objective_args[2]),
                objective_kwargs.get("adjacency1", objective_args[3]),
                objective_kwargs.get("adjacency2", objective_args[4]),
                result,
                pairs,
                args,
            )
        return result

    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(gadl_main, "gadl_objective", observed_objective))
        stack.enter_context(
            patch.object(
                torch.optim.Adam,
                "step",
                _optimizer_step_observer(counter, torch.optim.Adam.step),
            )
        )
        model.train(
            dataset,
            args.gids,
            use_attr=True,
            total_epochs=args.total_epochs,
            save_log=True,
            verbose=args.verbose,
        )
    wall_time = time.perf_counter() - started

    if args.total_epochs in wanted and args.total_epochs not in captured:
        z_src, z_tgt = model.embeddings
        captured[args.total_epochs] = {
            **_checkpoint(
                args.total_epochs,
                z_src,
                z_tgt,
                torch.as_tensor(graphs[0].edge_index.new_zeros((graphs[0].num_nodes,) * 2)),
                torch.as_tensor(graphs[1].edge_index.new_zeros((graphs[1].num_nodes,) * 2)),
                model.final_losses,
                pairs,
                args,
            ),
            "reconstruction_logits": {
                "source": _reconstruction_diagnostics(graphs[0], z_src),
                "target": _reconstruction_diagnostics(graphs[1], z_tgt),
            },
        }

    missing = sorted(wanted.difference(captured))
    checkpoints = [captured[epoch] for epoch in sorted(captured)]
    finite = all(
        checkpoint["similarity"]["all_finite"]
        and checkpoint["embedding"]["source"]["all_finite"]
        and checkpoint["embedding"]["target"]["all_finite"]
        and all(math.isfinite(value) for value in checkpoint["losses"].values())
        for checkpoint in checkpoints
    )
    collapsed = any(
        checkpoint["embedding"][direction]["mean_per_dimension_variance"] <= args.collapse_tolerance
        for checkpoint in checkpoints
        for direction in ("source", "target")
    )
    peak_cuda = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    )
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    result = {
        "seed": seed,
        "protocol": protocol,
        "graph_metadata": [_graph_metadata(graph) for graph in graphs],
        "anchor_count": int(pairs.shape[0]),
        "optimizer_steps": counter["completed"],
        "checkpoint_epochs_requested": sorted(wanted),
        "checkpoint_epochs_missing": missing,
        "checkpoints": checkpoints,
        "epoch_100_minus_epoch_0": (
            _metric_delta(captured[args.total_epochs], captured[0])
            if args.total_epochs in captured and 0 in captured
            else None
        ),
        "loss_history": model.loss_history,
        "timing_seconds": {
            "wall": wall_time,
            "spectral": model.spectral_time,
            "training": model.training_time,
        },
        "memory": {
            "peak_rss_native_units": peak_rss,
            "peak_cuda_bytes": peak_cuda,
        },
        "health": {
            "all_finite": finite,
            "embedding_collapse_detected": collapsed,
            "collapse_variance_tolerance": args.collapse_tolerance,
            "eligible_for_additional_seeds": finite and not collapsed and not missing,
        },
    }
    return result


# 功能：按检查点和指标汇总各种子的均值与标准差。
def _aggregate(results):
    if not results:
        return {}
    aggregate = {}
    for epoch in CHECKPOINT_EPOCHS:
        selected = {
            result["seed"]: next(
                (item for item in result["checkpoints"] if item["epoch"] == epoch), None
            )
            for result in results
        }
        if any(value is None for value in selected.values()):
            continue
        aggregate[str(epoch)] = {}
        for direction in ("source_to_target", "target_to_source"):
            aggregate[str(epoch)][direction] = {}
            for metric in ("Hits@1", "Hits@5", "Hits@10", "Hits@50", "MRR", "mean_rank"):
                values = [
                    checkpoint["tie_policies"]["average"]["metrics"][direction][metric]
                    for checkpoint in selected.values()
                ]
                aggregate[str(epoch)][direction][metric] = {
                    "mean": mean(values),
                    "std": pstdev(values),
                    "values": values,
                }
    return aggregate


# 功能：将已有记录中的命中率、表示方差和损失画成轨迹图。
def _plot(result, output_path):
    checkpoints = result["checkpoints"]
    epochs = [item["epoch"] for item in checkpoints]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for direction, label in (
        ("source_to_target", "source to target"),
        ("target_to_source", "target to source"),
    ):
        metrics = [
            item["tie_policies"]["average"]["metrics"][direction]
            for item in checkpoints
        ]
        axes[0].plot(epochs, [item["Hits@1"] for item in metrics], marker="o", label=label)
        axes[1].plot(epochs, [item["MRR"] for item in metrics], marker="o", label=label)
    axes[0].set_title("Average-tie Hits@1")
    axes[1].set_title("Average-tie MRR")
    axes[0].legend()
    axes[1].legend()

    history = result["loss_history"]
    if history:
        if isinstance(history, dict):
            for name, values in history.items():
                if isinstance(values, list) and values:
                    axes[2].plot(range(1, len(values) + 1), values, label=name)
        elif isinstance(history, list) and isinstance(history[0], dict):
            for name in history[0]:
                values = [_finite_scalar(item[name]) for item in history if name in item]
                if values:
                    axes[2].plot(range(1, len(values) + 1), values, label=name)
    axes[2].set_title("Training losses")
    axes[2].set_yscale("symlog")
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Douban")
    parser.add_argument("--root", default="data")
    parser.add_argument("--gids", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--train-ratio", type=float, default=0.0)
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--spectral-rank", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--descriptor-weight", type=float, default=1e-3)
    parser.add_argument("--commutativity-weight", type=float, default=1.0)
    parser.add_argument("--functional-map-weight", type=float, default=1.0)
    parser.add_argument("--bijectivity-weight", type=float, default=0.1)
    parser.add_argument("--orthogonality-weight", type=float, default=0.1)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--douban-graph-protocol", default="baseline-compatible", choices=("baseline-compatible", "full-graph"))
    parser.add_argument("--tie-atol", type=float, default=1e-8)
    parser.add_argument("--tie-rtol", type=float, default=1e-7)
    parser.add_argument("--collapse-tolerance", type=float, default=1e-12)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path)
    return parser.parse_args()


# 功能：逐种子运行连续训练观察，写 JSON 与图像，不用检查点分数反向决定训练。
def main():
    args = parse_args()
    config = vars(args).copy()
    config["output_json"] = str(config["output_json"])
    config["output_plot"] = str(config["output_plot"]) if config["output_plot"] else None
    results = [_run_seed(args, seed) for seed in args.seeds]
    output = {
        "experiment": "GADL full training one-to-one checkpoint validation",
        "primary_tie_policy": "average",
        "explicit_tie_policies": list(TIE_POLICIES),
        "configuration": config,
        "results": results,
        "aggregate": _aggregate(results),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.output_plot:
        _plot(results[0], args.output_plot)
    print(json.dumps({
        "output_json": str(args.output_json),
        "seeds": args.seeds,
        "health": {result["seed"]: result["health"] for result in results},
        "timing_seconds": {result["seed"]: result["timing_seconds"] for result in results},
    }, indent=2))


if __name__ == "__main__":
    main()
