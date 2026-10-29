# 文件作用：比较 GADL 与当前 JOENA 作为相似度来源时，同一个 QuotientDecode 的多对多结果。
# 核心依赖：PlanetAlign/algorithms/gadl/main.py、PlanetAlign/algorithms/joena/main.py；PlanetAlign/m2m/quotient.py → quotient_decode()。
# 核心处理：_decode_without_gt() 先完成预测，之后才读取真实实体做评分与候选覆盖诊断。
# 结果去向：main() 写出原预测、对比 JSON 与报告；诊断辅助函数来自 scripts/diagnose_candidate_coverage.py。

"""Evaluate GADL as a node-similarity backbone for the existing M2M decoder.

The prediction path is deliberately blind to the entity ground truth:

    Dataset -> GADL -> normalized embeddings -> cosine S
            -> existing QuotientDecode -> predicted entities

The ground-truth JSON is opened only after both GADL and the comparison
backbone have completed candidate construction, partitioning, and decoding.
Ground truth is then used only for offline diagnostics and evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib")
)

import torch

from PlanetAlign.algorithms import GADL, JOENA
from PlanetAlign.data import Dataset
from PlanetAlign.m2m import evaluate_predictions, use_full_anchor_supervision
from PlanetAlign.m2m.quotient import quotient_decode
from PlanetAlign.utils import get_anchor_pairs

from diagnose_candidate_coverage import (
    build_local_context,
    diagnose_side,
    gt_side_groups,
    replay_current_final_partitions,
)


GIDS = (0, 1)
M2M_METRICS = ("Macro Set-F1", "Micro Set-F1", "AUIS", "AUCS")


def _distribution(values: torch.Tensor, quantiles: bool = True) -> Dict[str, Any]:
    flat = values.detach().to(torch.float32).cpu().reshape(-1)
    if flat.numel() == 0:
        return {"count": 0}
    result: Dict[str, Any] = {
        "count": int(flat.numel()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
    }
    if quantiles:
        q = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9]))
        result.update({"p10": float(q[0]), "median": float(q[1]), "p90": float(q[2])})
    return result


# 功能：检查 S 的形状、有限值、分布与每行高分，返回矩阵可用性诊断。
def similarity_diagnostics(similarity: torch.Tensor) -> Dict[str, Any]:
    S = similarity.detach().to(torch.float32).cpu()
    finite = torch.isfinite(S)
    top_count = min(10, S.shape[1])
    top = torch.topk(S, k=top_count, dim=1).values
    topk: Dict[str, Any] = {"top1": _distribution(top[:, 0])}
    for k in (5, 10):
        if top_count >= k:
            topk[f"top{k}_mean"] = _distribution(top[:, :k].mean(dim=1))
            topk[f"top{k}_boundary"] = _distribution(top[:, k - 1])
    return {
        "shape": list(S.shape),
        "dtype": str(S.dtype),
        "finite": bool(finite.all()),
        "nan_count": int(torch.isnan(S).sum()),
        "inf_count": int(torch.isinf(S).sum()),
        "scores": _distribution(S, quantiles=False),
        "row_top_scores": topk,
    }


# 功能：将真实实体两侧成员的所有组合标为 True，生成与 S 同形状的正确节点对掩码。
# 流程位置：仅由预测后的 separability_diagnostics() 用来区分分数，不传入 quotient_decode()。
def _true_pair_mask(
    entities: Mapping[str, Mapping[str, Iterable[int]]],
    shape: Tuple[int, int],
) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool)
    n_src, n_tgt = shape
    for item in entities.values():
        src = sorted({int(i) for i in item.get("src", []) if 0 <= int(i) < n_src})
        tgt = sorted({int(j) for j in item.get("tgt", []) if 0 <= int(j) < n_tgt})
        if src and tgt:
            src_idx = torch.tensor(src, dtype=torch.long)
            tgt_idx = torch.tensor(tgt, dtype=torch.long)
            mask[src_idx[:, None], tgt_idx[None, :]] = True
    return mask


# 功能：预测后用真实实体区分正确与错误节点对，比较分数差距与排序可分性。
def separability_diagnostics(
    similarity: torch.Tensor,
    entities: Mapping[str, Mapping[str, Iterable[int]]],
) -> Dict[str, Any]:
    S = similarity.detach().to(torch.float32).cpu()
    true_mask = _true_pair_mask(entities, tuple(S.shape))
    within = S[true_mask]
    outside = S[~true_mask]
    within_stats = _distribution(within)
    outside_stats = _distribution(outside, quantiles=False)
    mean_gap = float(within.mean() - outside.mean())
    pooled_std = torch.sqrt(
        0.5 * (within.var(unbiased=False) + outside.var(unbiased=False))
    )
    effect_size = float((within.mean() - outside.mean()) / pooled_std) if pooled_std > 0 else float("nan")

    outside_sorted = torch.sort(outside).values
    lower = torch.searchsorted(outside_sorted, within, right=False).to(torch.float64)
    upper = torch.searchsorted(outside_sorted, within, right=True).to(torch.float64)
    auc = float(((lower + upper) * 0.5 / outside.numel()).mean())
    return {
        "gt_usage": "offline diagnostic after prediction only",
        "within_entity_pair_scores": within_stats,
        "outside_entity_pair_scores": outside_stats,
        "mean_gap": mean_gap,
        "standardized_mean_gap": effect_size,
        "pairwise_auc": auc,
    }


# 功能：将 S、图、训练锚点送入 quotient_decode() 并重放候选/分组，无真实实体参数。
def _decode_without_gt(
    similarity: torch.Tensor,
    graph_src,
    graph_tgt,
    anchors: torch.Tensor,
    use_attr: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    predictions, decode_info = quotient_decode(
        similarity,
        graph_src,
        graph_tgt,
        use_attr=use_attr,
        anchors=anchors,
    )
    source_context = build_local_context(graph_src, similarity)
    target_context = build_local_context(graph_tgt, similarity.T.contiguous())
    final_src, final_tgt, final_partition_info = replay_current_final_partitions(
        similarity, graph_src, graph_tgt, anchors, use_attr
    )
    return {
        "predictions": predictions,
        "decode_info": decode_info,
        "source_context": source_context,
        "target_context": target_context,
        "final_src": final_src,
        "final_tgt": final_tgt,
        "final_partition_info": final_partition_info,
        "decode_and_candidate_seconds": time.perf_counter() - started,
    }


# 输入来源：main() 在完成盲预测后传入预测状态与真实实体。
# 功能：调用 scripts/diagnose_candidate_coverage.py → diagnose_side() 分析两侧候选覆盖。
def _offline_candidate_diagnostics(
    blind_state: Mapping[str, Any],
    entities: Mapping[str, Mapping[str, Sequence[int]]],
) -> Dict[str, Any]:
    source, _, _ = diagnose_side(
        "source",
        blind_state["source_context"],
        blind_state["final_src"],
        gt_side_groups(entities, "src"),
    )
    target, _, _ = diagnose_side(
        "target",
        blind_state["target_context"],
        blind_state["final_tgt"],
        gt_side_groups(entities, "tgt"),
    )
    return {"source": source, "target": target}


# 功能：复用 full-anchor JOENA 缓存，缺失时训练并保存；返回 S 与来源记录。
def _load_or_train_current_joena(
    args: argparse.Namespace,
    dataset: Dataset,
    use_attr: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    cache = args.existing_cache
    if cache is None:
        cache = Path("logs/m2m_diag/S_cache") / (
            f"{args.dataset}_e{args.existing_epochs}_s{args.seed}_fa.pt"
        )
    if cache.exists():
        S = torch.load(cache, map_location="cpu", weights_only=True)
        return S.detach().to(torch.float32).cpu(), {
            "backbone": "JOENA",
            "source": "existing full-anchor cache",
            "cache": str(cache.resolve()),
            "epochs": args.existing_epochs,
            "seed": args.seed,
        }

    torch.manual_seed(args.seed)
    model = JOENA(alpha=0.7).to(args.device)
    started = time.perf_counter()
    model.train(
        dataset=dataset,
        gids=list(GIDS),
        use_attr=use_attr,
        total_epochs=args.existing_epochs,
        save_log=False,
        verbose=False,
    )
    S = model.S.detach().to(torch.float32).cpu()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(S, cache)
    return S, {
        "backbone": "JOENA",
        "source": "fresh current-backbone training",
        "cache": str(cache.resolve()),
        "epochs": args.existing_epochs,
        "seed": args.seed,
        "training_seconds": time.perf_counter() - started,
    }


# 输入来源：main() 传入 similarity_diagnostics() 对 GADL 与 JOENA 的分数统计。
# 输出结果：均值差、标准差比和分数范围比交给对比报告；不根据这些诊断重缩放或选择矩阵。
def _scale_comparison(gadl: Mapping[str, Any], existing: Mapping[str, Any]) -> Dict[str, Any]:
    g = gadl["scores"]
    e = existing["scores"]
    std_ratio = float(g["std"] / e["std"]) if e["std"] else float("inf")
    range_g = float(g["max"] - g["min"])
    range_e = float(e["max"] - e["min"])
    range_ratio = range_g / range_e if range_e else float("inf")
    materially_different = not (0.5 <= std_ratio <= 2.0 and 0.5 <= range_ratio <= 2.0)
    return {
        "mean_difference_gadl_minus_existing": float(g["mean"] - e["mean"]),
        "std_ratio_gadl_over_existing": std_ratio,
        "range_ratio_gadl_over_existing": range_ratio,
        "materially_different": materially_different,
        "rule": "materially different when std or range ratio is outside [0.5, 2.0]",
        "decoder_note": "default profile-cosine candidates and zero relative match threshold are scale tolerant",
    }


# 功能：将两种基础矩阵的实体指标、可分性和尺度对比写为可读报告。
def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    g = report["backbones"]["GADL"]
    j = report["backbones"]["JOENA"]
    lines = [
        "# GADL to QuotientDecode minimal integration",
        "",
        "Prediction was completed before the GT entity JSON was opened.",
        "No oracle source groups or target cardinalities entered decoding.",
        "",
        "| Backbone | Macro Set-F1 | Micro Set-F1 | AUIS | AUCS |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in (("GADL", g), ("JOENA", j)):
        m = item["metrics"]
        lines.append(
            f"| {name} | {m['Macro Set-F1']:.4f} | {m['Micro Set-F1']:.4f} | "
            f"{m['AUIS']:.4f} | {m['AUCS']:.4f} |"
        )
    lines.extend([
        "",
        f"GADL S shape: {g['similarity']['shape']}",
        f"GADL separability AUC: {g['separability']['pairwise_auc']:.4f}",
        f"JOENA separability AUC: {j['separability']['pairwise_auc']:.4f}",
        "",
        "Decoder parameters were identical and left at QuotientDecode defaults.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/m2m_overlap_0.05"))
    parser.add_argument("--dataset", default="douban_m2m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gadl-epochs", type=int, default=100)
    parser.add_argument("--existing-epochs", type=int, default=10)
    parser.add_argument("--existing-cache", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/m2m_diag/gadl_m2m_backbone_douban_seed42.json"),
    )
    return parser.parse_args()


# 核心思路：加载图并恢复训练锚点，取得 GADL/JOENA 的 S，先解码再读真实实体评分。
# 输出结果：保存预测与诊断，使矩阵来源差异和解码参数一致性可检查。
def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    dataset = Dataset(
        root=args.root,
        name=args.dataset,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    use_full_anchor_supervision(dataset)
    graph_src, graph_tgt = dataset.pyg_graphs[GIDS[0]], dataset.pyg_graphs[GIDS[1]]
    use_attr = all(graph.x is not None for graph in (graph_src, graph_tgt))
    if not use_attr:
        raise ValueError("GADL M2M integration requires attributes on both graphs")
    anchors = get_anchor_pairs(dataset.train_data, *GIDS)

    gadl = GADL(seed=args.seed).to(args.device)
    started = time.perf_counter()
    gadl.train(
        dataset=dataset,
        gids=list(GIDS),
        use_attr=True,
        total_epochs=args.gadl_epochs,
        save_log=False,
        verbose=False,
    )
    gadl_seconds = time.perf_counter() - started
    S_gadl = gadl.S.detach().to(torch.float32).cpu()
    S_existing, existing_provenance = _load_or_train_current_joena(
        args, dataset, use_attr
    )
    expected_shape = (int(graph_src.num_nodes), int(graph_tgt.num_nodes))
    for name, S in (("GADL", S_gadl), ("JOENA", S_existing)):
        if tuple(S.shape) != expected_shape:
            raise ValueError(f"{name} S shape {tuple(S.shape)} != {expected_shape}")
        if not torch.isfinite(S).all():
            raise ValueError(f"{name} S contains NaN or Inf")

    # Every prediction-stage operation is complete before the GT JSON is read.
    blind_gadl = _decode_without_gt(S_gadl, graph_src, graph_tgt, anchors, use_attr)
    blind_existing = _decode_without_gt(
        S_existing, graph_src, graph_tgt, anchors, use_attr
    )

    gt_path = args.root / f"{args.dataset}_gt_many2many.json"
    with gt_path.open("r", encoding="utf-8") as handle:
        gt_payload = json.load(handle)
        entities = gt_payload["entities"]
    allow_overlap = gt_payload.get("metadata", {}).get("overlap_ratio", 0.0) > 0

    gadl_similarity = similarity_diagnostics(S_gadl)
    existing_similarity = similarity_diagnostics(S_existing)
    backbones: Dict[str, Dict[str, Any]] = {}
    for name, S, blind, provenance in (
        (
            "GADL",
            S_gadl,
            blind_gadl,
            {
                "backbone": "PlanetAlign GADL",
                "profile": "existing default implementation",
                "seed": args.seed,
                "epochs": args.gadl_epochs,
                "training_seconds": gadl_seconds,
                "spectral_seconds": gadl.spectral_time,
                "configuration": {
                    "hidden_dim": gadl.hidden_dim,
                    "num_hidden_layers": gadl.num_hidden_layers,
                    "spectral_rank": gadl.spectral_rank,
                    "effective_rank": gadl.effective_rank,
                    "lr": gadl.lr,
                    "weight_decay": gadl.weight_decay,
                    "descriptor_weight": gadl.descriptor_weight,
                    "commutativity_weight": gadl.commutativity_weight,
                    "functional_map_weight": gadl.functional_map_weight,
                    "bijectivity_weight": gadl.bijectivity_weight,
                    "orthogonality_weight": gadl.orthogonality_weight,
                    "residual": gadl.residual,
                },
                "final_losses": gadl.final_losses,
            },
        ),
        ("JOENA", S_existing, blind_existing, existing_provenance),
    ):
        metrics = evaluate_predictions(entities, blind["predictions"], metrics=M2M_METRICS,
                                       allow_overlap=allow_overlap)
        backbones[name] = {
            "provenance": provenance,
            "similarity": gadl_similarity if name == "GADL" else existing_similarity,
            "separability": separability_diagnostics(S, entities),
            "candidate_partition_diagnostics": _offline_candidate_diagnostics(
                blind, entities
            ),
            "decode_info": blind["decode_info"],
            "final_partition_info": blind["final_partition_info"],
            "decode_and_candidate_seconds": blind["decode_and_candidate_seconds"],
            "metrics": metrics,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.with_suffix("")
    tensor_path = stem.with_name(stem.name + "_tensors.pt")
    prediction_path = stem.with_name(stem.name + "_predictions.json")
    report_path = stem.with_suffix(".md")
    torch.save(
        {
            "GADL": {
                "S": S_gadl,
                "embeddings": {
                    int(gid): embedding.detach().to(torch.float32).cpu()
                    for gid, embedding in gadl.embeddings.items()
                },
            },
            "JOENA": {"S": S_existing},
        },
        tensor_path,
    )
    with prediction_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "GADL": blind_gadl["predictions"],
                "JOENA": blind_existing["predictions"],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    report = {
        "protocol": {
            "dataset": args.dataset,
            "root": str(args.root.resolve()),
            "gids": list(GIDS),
            "seed": args.seed,
            "nodes": [int(graph_src.num_nodes), int(graph_tgt.num_nodes)],
            "edges": [int(graph_src.edge_index.shape[1]), int(graph_tgt.edge_index.shape[1])],
            "attribute_dims": [int(graph_src.x.shape[1]), int(graph_tgt.x.shape[1])],
            "anchor_count": int(anchors.shape[0]),
            "decoder": "existing QuotientDecode defaults",
            "decoder_parameters_changed": False,
            "gt_json_opened_after_both_predictions": True,
            "gt_prediction_usage": "none",
            "gt_diagnostic_usage": "separability, candidate recovery, and final evaluation only",
            "oracle_protocol_called": False,
            "data_flow": [
                "Dataset",
                "GADL normalized node embeddings",
                "cosine similarity S",
                "existing QuotientDecode",
                "predicted entities",
                "evaluate_predictions",
            ],
        },
        "similarity_scale_comparison": _scale_comparison(
            gadl_similarity, existing_similarity
        ),
        "backbones": backbones,
        "artifacts": {
            "json": str(args.output.resolve()),
            "report": str(report_path.resolve()),
            "tensors": str(tensor_path.resolve()),
            "predictions": str(prediction_path.resolve()),
        },
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=True)
    _write_report(report_path, report)

    print(f"wrote {args.output}")
    print(f"wrote {report_path}")
    for name in ("GADL", "JOENA"):
        metrics = backbones[name]["metrics"]
        print(
            f"{name}: Macro Set-F1={metrics['Macro Set-F1']:.4f} "
            f"Micro Set-F1={metrics['Micro Set-F1']:.4f} "
            f"AUIS={metrics['AUIS']:.4f} AUCS={metrics['AUCS']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
