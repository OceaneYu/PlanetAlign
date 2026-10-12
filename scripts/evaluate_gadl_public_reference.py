#!/usr/bin/env python3
# 文件作用：逐项切换 GADL 的公开参考兼容行为，在固定训练方案下比较复现差异。
# 核心依赖：PlanetAlign/algorithms/gadl/profiles.py → ConfigurableGADL；scripts/evaluate_gadl_one_to_one.py 的图协议与评估。
# 结果去向：main() 保存各配置、种子、观察轮次的参考评估和平均同分评估 JSON。
# 阅读建议：先看 VARIANTS，再读 _run() → _checkpoint()，明确一次只比较哪些行为。

"""Evaluate controlled GADL public-reference compatibility variants."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn.functional as F

from evaluate_gadl_one_to_one import (
    _apply_douban_protocol,
    _embedding_diagnostics,
    _evaluate_mode,
    _graph_metadata,
    _load_dataset,
)
from PlanetAlign.algorithms.gadl.profiles import ConfigurableGADL


CHECKPOINTS = (0, 1, 5, 20, 50, 100)
REFERENCE_DISTANCE_EPSILON = 1e-8


VARIANTS = {
    "planetalign_current": {
        "commutativity_weight": 1.0,
        "reconstruction_mode": "unweighted_dense_bce",
        "functional_map_schedule": "constant",
        "activation_mode": "planetalign_prelu",
    },
    "beta_1e-2_only": {
        "commutativity_weight": 1e-2,
        "reconstruction_mode": "unweighted_dense_bce",
        "functional_map_schedule": "constant",
        "activation_mode": "planetalign_prelu",
    },
    "beta_1e-2_reference_weighted_bce": {
        "commutativity_weight": 1e-2,
        "reconstruction_mode": "reference_weighted_bce",
        "functional_map_schedule": "constant",
        "activation_mode": "planetalign_prelu",
    },
    "beta_weighted_reference_fm_decay": {
        "commutativity_weight": 1e-2,
        "reconstruction_mode": "reference_weighted_bce",
        "functional_map_schedule": "reference_linear_floor_half",
        "activation_mode": "planetalign_prelu",
    },
    "public_reference_compatible": {
        "commutativity_weight": 1e-2,
        "reconstruction_mode": "reference_weighted_bce",
        "functional_map_schedule": "reference_linear_floor_half",
        "activation_mode": "reference_identity",
    },
}


def _scalar(value):
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.detach().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _loss_diagnostics(losses):
    output = {
        key: scalar
        for key, value in losses.items()
        if (scalar := _scalar(value)) is not None
    }
    output["functional_map"] = output.get("fm12", 0.0) + output.get("fm21", 0.0)
    return output


def _embedding_diagnostic(embedding):
    result = _embedding_diagnostics(embedding.detach().cpu())
    result["mean_per_dimension_variance"] = float(
        embedding.detach().float().var(dim=0, unbiased=False).mean().cpu()
    )
    return result


def _logit_diagnostics(embedding, adjacency):
    target = adjacency.to_dense() if adjacency.is_sparse else adjacency
    target = target.to(embedding.device)
    logits = embedding.detach() @ embedding.detach().T
    positive = logits[target > 0].float()
    negative = logits[target <= 0].float()
    return {
        "positive_count": int(positive.numel()),
        "negative_count": int(negative.numel()),
        "positive_fraction": float((target > 0).float().mean().cpu()),
        "positive_mean": float(positive.mean().cpu()),
        "positive_std": float(positive.std(unbiased=False).cpu()),
        "negative_mean": float(negative.mean().cpu()),
        "negative_std": float(negative.std(unbiased=False).cpu()),
    }


# 功能：归一化嵌入并按参考的距离和乐观同分规则计算 Hits，供检查点评估。
def _reference_hits(embedding_src, embedding_tgt, pairs):
    source = F.normalize(embedding_src.detach(), p=2, dim=1).double()
    target = F.normalize(embedding_tgt.detach(), p=2, dim=1).double()
    distance = torch.cdist(source, target, p=2).cpu()
    pairs = pairs.cpu()
    hits = {1: 0, 5: 0, 10: 0, 50: 0, 100: 0}
    ranks = []
    tie_counts = []
    for source_id, target_id in pairs.tolist():
        row = distance[source_id]
        truth = row[target_id]
        rank_zero_based = int(torch.sum(row < (truth - REFERENCE_DISTANCE_EPSILON)))
        ranks.append(rank_zero_based + 1)
        tie_counts.append(int(torch.sum(torch.abs(row - truth) <= REFERENCE_DISTANCE_EPSILON)))
        for k in hits:
            if rank_zero_based < k:
                hits[k] += 1
    count = len(ranks)
    return {
        "protocol": "public-reference-compatible source-to-target distance ranking",
        "distance": "L2 distance after row-wise L2 normalization",
        "epsilon": REFERENCE_DISTANCE_EPSILON,
        "tie_behavior": "optimistic: count(distance < truth_distance - epsilon)",
        "Hits@1": hits[1] / count,
        "Hits@5": hits[5] / count,
        "Hits@10": hits[10] / count,
        "Hits@50": hits[50] / count,
        "Hits@100": hits[100] / count,
        "MRR_diagnostic_not_in_reference": mean(1.0 / rank for rank in ranks),
        "mean_rank_diagnostic_not_in_reference": mean(ranks),
        "truth_tie_rate": mean(value > 1 for value in tie_counts),
        "average_truth_tie_candidates": mean(tie_counts),
    }


# 功能：对同一当前嵌入同时做公开参考和 PlanetAlign 平均同分评估，并附损失与重构诊断。
def _checkpoint(state_epoch, phase, z1, z2, adjacency1, adjacency2, losses, pairs, args):
    similarity = F.normalize(z1.detach(), dim=1) @ F.normalize(z2.detach(), dim=1).T
    planetalign = _evaluate_mode(
        similarity.float().cpu(),
        pairs.cpu(),
        ("average",),
        args.tie_atol,
        args.tie_rtol,
    )["tie_policies"]["average"]
    return {
        "epoch": state_epoch,
        "phase": phase,
        "public_reference_evaluation": _reference_hits(z1, z2, pairs),
        "planetalign_average_tie_evaluation": planetalign,
        "losses": _loss_diagnostics(losses),
        "embedding": {
            "source": _embedding_diagnostic(z1),
            "target": _embedding_diagnostic(z2),
        },
        "reconstruction_logits": {
            "source": _logit_diagnostics(z1, adjacency1),
            "target": _logit_diagnostics(z2, adjacency2),
        },
        "self_S_std": float(similarity.float().std(unbiased=False).cpu()),
        "all_finite": bool(
            torch.isfinite(z1).all()
            and torch.isfinite(z2).all()
            and torch.isfinite(similarity).all()
            and all(math.isfinite(value) for value in _loss_diagnostics(losses).values())
        ),
    }


# 核心思路：按一个变体创建 ConfigurableGADL，通过 objective_observer 观察固定轮次。
# 输出结果：返回该变体/种子的检查点、权重计划、配置和时间记录。
def _run(args, variant_name, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    dataset = _load_dataset(args, seed)
    protocol = _apply_douban_protocol(dataset, args.gids, "baseline-compatible")
    pairs = dataset.test_data.detach().cpu()
    captured = {}

    # 输入来源：PlanetAlign/algorithms/gadl/profiles.py → ConfigurableGADL.train() 的 configured_objective 回调。
    # 功能：按 state_epoch 与 phase 捕获指定检查点，每轮只记录一次；这里的轮次表示已完成的更新数。
    # 结果去向：_checkpoint() 的记录放入 captured，最后随 _run() 返回，不参与模型选择。
    def observer(**event):
        epoch = event["state_epoch"]
        phase = event["phase"]
        should_capture = (
            epoch in CHECKPOINTS
            and epoch <= args.total_epochs
            and (phase in ("initial", "final") or epoch > 0)
            and epoch not in captured
        )
        if should_capture:
            captured[epoch] = _checkpoint(
                epoch,
                phase,
                event["z1"],
                event["z2"],
                event["adjacency1"],
                event["adjacency2"],
                event["losses"],
                pairs,
                args,
            )

    options = VARIANTS[variant_name]
    model = ConfigurableGADL(
        hidden_dim=512,
        num_hidden_layers=3,
        spectral_rank=300,
        lr=1e-4,
        weight_decay=5e-4,
        descriptor_weight=1e-3,
        commutativity_weight=options["commutativity_weight"],
        functional_map_weight=1.0,
        bijectivity_weight=0.1,
        orthogonality_weight=0.1,
        residual=False,
        seed=seed,
        reconstruction_mode=options["reconstruction_mode"],
        functional_map_schedule=options["functional_map_schedule"],
        activation_mode=options["activation_mode"],
        objective_observer=observer,
    ).to(args.device)
    started = time.perf_counter()
    model.train(
        dataset,
        args.gids,
        use_attr=True,
        total_epochs=args.total_epochs,
        save_log=True,
        verbose=False,
    )
    wall = time.perf_counter() - started
    checkpoints = [captured[key] for key in sorted(captured)]
    expected = [epoch for epoch in CHECKPOINTS if epoch <= args.total_epochs]
    missing = sorted(set(expected).difference(captured))
    collapsed = any(
        checkpoint["embedding"][side]["mean_per_dimension_variance"] <= args.collapse_tolerance
        for checkpoint in checkpoints
        for side in ("source", "target")
    )
    return {
        "variant": variant_name,
        "seed": seed,
        "configuration": options,
        "protocol": protocol,
        "graph_metadata": [_graph_metadata(dataset.pyg_graphs[gid]) for gid in args.gids],
        "anchor_count": int(pairs.shape[0]),
        "checkpoints": checkpoints,
        "missing_checkpoints": missing,
        "schedule_history": model.schedule_history,
        "timing_seconds": {
            "wall": wall,
            "spectral": model.spectral_time,
            "training": model.training_time,
        },
        "peak_memory": {
            "rss_native_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        },
        "health": {
            "all_finite": all(checkpoint["all_finite"] for checkpoint in checkpoints),
            "embedding_collapse_detected": collapsed,
            "eligible_for_additional_seeds": not missing and not collapsed and all(
                checkpoint["all_finite"] for checkpoint in checkpoints
            ),
        },
    }


# 功能：按变体和观察轮次汇总各种子评估值。
def _aggregate(results):
    output = {}
    for variant in sorted({result["variant"] for result in results}):
        selected = [result for result in results if result["variant"] == variant]
        output[variant] = {}
        for epoch in CHECKPOINTS:
            checkpoints = [
                next((item for item in result["checkpoints"] if item["epoch"] == epoch), None)
                for result in selected
            ]
            if not checkpoints or any(item is None for item in checkpoints):
                continue
            values = [item["public_reference_evaluation"]["Hits@1"] for item in checkpoints]
            output[variant][str(epoch)] = {
                "public_reference_Hits@1": {"mean": mean(values), "std": pstdev(values), "values": values}
            }
            for direction in ("source_to_target", "target_to_source"):
                metrics = [
                    item["planetalign_average_tie_evaluation"]["metrics"][direction]
                    for item in checkpoints
                ]
                output[variant][str(epoch)][direction] = {}
                for metric in ("Hits@1", "Hits@5", "Hits@10", "Hits@50", "MRR", "mean_rank"):
                    metric_values = [item[metric] for item in metrics]
                    output[variant][str(epoch)][direction][metric] = {
                        "mean": mean(metric_values),
                        "std": pstdev(metric_values),
                        "values": metric_values,
                    }
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Douban")
    parser.add_argument("--root", default="data")
    parser.add_argument("--gids", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--train-ratio", type=float, default=0.0)
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tie-atol", type=float, default=1e-8)
    parser.add_argument("--tie-rtol", type=float, default=1e-7)
    parser.add_argument("--collapse-tolerance", type=float, default=1e-12)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


# 功能：遍历请求变体与种子调用 _run()，保存原始与汇总 JSON。
def main():
    args = parse_args()
    results = [
        _run(args, variant, seed)
        for variant in args.variants
        for seed in args.seeds
    ]
    output = {
        "experiment": "GADL public-reference compatibility attribution",
        "source_audit": {
            "online_relation": "online_edge_label[0][1], 11502 symmetric adjacency entries",
            "reference_hyperparameters": {
                "hidden_dim": 512,
                "num_hidden_layers": 3,
                "spectral_rank": 300,
                "lr": 1e-4,
                "weight_decay": 5e-4,
                "alpha": 1e-3,
                "beta": 1e-2,
                "lambda_bij": 0.1,
                "lambda_orth": 0.1,
                "epochs": 100,
                "residual": False,
            },
            "beta_source_mismatch": "public compute_FM applies beta to an internal C^T C term and leaves Laplacian commutativity unweighted; requested PlanetAlign profile maps beta to commutativity",
            "activation": "PReLU is constructed in public GConv_GADL but not called by forward",
            "reference_evaluation": "row L2-normalize, torch.cdist p=2 double, source-to-target rank=count(distance < truth-epsilon), epsilon=1e-8",
        },
        "configuration": {
            "dataset": args.dataset,
            "gids": args.gids,
            "seeds": args.seeds,
            "variants": args.variants,
            "total_epochs": args.total_epochs,
            "device": args.device,
            "test_data_used_for_training_or_selection": False,
        },
        "results": results,
        "aggregate": _aggregate(results),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_json),
        "health": {
            f'{result["variant"]}:seed{result["seed"]}': result["health"]
            for result in results
        },
        "timing_seconds": {
            f'{result["variant"]}:seed{result["seed"]}': result["timing_seconds"]
            for result in results
        },
    }, indent=2))


if __name__ == "__main__":
    main()

