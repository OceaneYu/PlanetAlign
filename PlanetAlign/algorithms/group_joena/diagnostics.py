from __future__ import annotations

from statistics import median
from typing import Dict, Iterable, List, Mapping, Optional

import torch

from .assignment import SoftGroupAssignment


EntityMap = Dict[str, Dict[str, List[int]]]


def _safe_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def assignment_diagnostics(
    assignments: torch.Tensor,
    configured_group_count: Optional[int] = None,
    majority_threshold: float = 0.5,
    min_non_empty_ratio: float = 0.5,
    uniform_row_max_multiplier: float = 1.2,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Summarize hard and soft assignment collapse signals."""

    if assignments.dim() != 2:
        raise ValueError("assignments must be 2D")
    u = assignments.detach().cpu()
    n, k = u.shape
    configured = int(configured_group_count or k)
    hard = SoftGroupAssignment.hard_assignments(u)
    counts = torch.bincount(hard, minlength=k).to(torch.float32)
    non_empty = int((counts > 0).sum().item())
    row_max = u.max(dim=1).values if n else torch.empty(0)
    entropy = -(u.clamp_min(eps) * torch.log(u.clamp_min(eps))).sum(dim=1)
    singleton_count = int((counts == 1).sum().item())
    group_sizes = [int(x) for x in counts.tolist()]
    nonzero_sizes = [x for x in group_sizes if x > 0]
    max_group_size = max(nonzero_sizes) if nonzero_sizes else 0
    warnings: List[str] = []

    if n > 0 and max_group_size / n > majority_threshold:
        warnings.append("assignment_collapse_majority_group")
    if configured > 0 and non_empty / configured < min_non_empty_ratio:
        warnings.append("too_few_non_empty_groups")
    if k > 0 and row_max.numel() > 0 and _safe_float(row_max.mean()) <= uniform_row_max_multiplier / k:
        warnings.append("assignments_close_to_uniform")

    return {
        "node_count": int(n),
        "configured_group_count": configured,
        "non_empty_group_count": non_empty,
        "empty_group_count": max(0, configured - non_empty),
        "mean_group_size": float(sum(nonzero_sizes) / len(nonzero_sizes)) if nonzero_sizes else 0.0,
        "median_group_size": float(median(nonzero_sizes)) if nonzero_sizes else 0.0,
        "max_group_size": int(max_group_size),
        "singleton_ratio": float(singleton_count / configured) if configured else 0.0,
        "assignment_entropy": _safe_float(entropy.mean()) if entropy.numel() else 0.0,
        "mean_row_max_probability": _safe_float(row_max.mean()) if row_max.numel() else 0.0,
        "min_row_max_probability": _safe_float(row_max.min()) if row_max.numel() else 0.0,
        "max_row_max_probability": _safe_float(row_max.max()) if row_max.numel() else 0.0,
        "group_mass_distribution": [float(x) for x in u.sum(dim=0).tolist()],
        "hard_group_sizes": group_sizes,
        "warnings": warnings,
    }


def transport_diagnostics(
    transport: torch.Tensor,
    effective_threshold: float = 1e-3,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Summarize group transport concentration."""

    if transport.dim() != 2:
        raise ValueError("transport must be 2D")
    t = transport.detach().cpu().to(torch.float32)
    row_sum = t.sum(dim=1, keepdim=True).clamp_min(eps)
    col_sum = t.sum(dim=0, keepdim=True).clamp_min(eps)
    row_prob = t / row_sum
    col_prob = t / col_sum
    row_entropy = -(row_prob.clamp_min(eps) * torch.log(row_prob.clamp_min(eps))).sum(dim=1)
    col_entropy = -(col_prob.clamp_min(eps) * torch.log(col_prob.clamp_min(eps))).sum(dim=0)
    row_max = row_prob.max(dim=1).values if t.shape[1] else torch.empty(0)
    warnings: List[str] = []
    if row_max.numel() > 0 and _safe_float(row_max.mean()) <= 1.2 / max(1, t.shape[1]):
        warnings.append("transport_close_to_uniform")

    return {
        "shape": [int(t.shape[0]), int(t.shape[1])],
        "row_entropy_mean": _safe_float(row_entropy.mean()) if row_entropy.numel() else 0.0,
        "column_entropy_mean": _safe_float(col_entropy.mean()) if col_entropy.numel() else 0.0,
        "max_value": _safe_float(t.max()) if t.numel() else 0.0,
        "mean_value": _safe_float(t.mean()) if t.numel() else 0.0,
        "effective_matched_pairs": int((t > effective_threshold).sum().item()),
        "warnings": warnings,
    }


def score_diagnostics(
    scores: torch.Tensor,
    pred_entities: Optional[Mapping[str, Mapping[str, Iterable[int]]]] = None,
    gt_entities: Optional[Mapping[str, Mapping[str, Iterable[int]]]] = None,
    threshold: float = 0.0,
) -> Dict[str, object]:
    """Summarize node-level scores and prediction/GT cardinalities."""

    if scores.dim() != 2:
        raise ValueError("scores must be 2D")
    s = scores.detach().cpu().to(torch.float32)
    pred_positive = None
    if pred_entities is not None:
        pred_positive = sum(len(list(item.get("tgt", []))) for item in pred_entities.values())
    gt_positive = None
    if gt_entities is not None:
        gt_positive = sum(len(list(item.get("tgt", []))) for item in gt_entities.values())
    return {
        "shape": [int(s.shape[0]), int(s.shape[1])],
        "min": _safe_float(s.min()) if s.numel() else 0.0,
        "max": _safe_float(s.max()) if s.numel() else 0.0,
        "mean": _safe_float(s.mean()) if s.numel() else 0.0,
        "density_above_threshold": float((s > threshold).sum().item() / max(1, s.numel())),
        "predicted_positive_pair_count": pred_positive,
        "gt_positive_pair_count": gt_positive,
    }


def prediction_diagnostics(
    pred_entities: Mapping[str, Mapping[str, Iterable[int]]],
    gt_entities: Optional[Mapping[str, Mapping[str, Iterable[int]]]] = None,
) -> Dict[str, object]:
    """Summarize predicted group cardinalities and merge behavior."""

    sizes = [len(list(v.get("src", []))) + len(list(v.get("tgt", []))) for v in pred_entities.values()]
    tgt_sizes = [len(list(v.get("tgt", []))) for v in pred_entities.values()]
    singleton = sum(1 for size in sizes if size == 1)
    out: Dict[str, object] = {
        "predicted_group_count": int(len(pred_entities)),
        "mean_group_size": float(sum(sizes) / len(sizes)) if sizes else 0.0,
        "max_group_size": int(max(sizes)) if sizes else 0,
        "singleton_ratio": float(singleton / len(sizes)) if sizes else 0.0,
        "mean_target_size": float(sum(tgt_sizes) / len(tgt_sizes)) if tgt_sizes else 0.0,
        "max_target_size": int(max(tgt_sizes)) if tgt_sizes else 0,
    }
    if gt_entities is not None:
        gt_sizes = [len(list(v.get("src", []))) + len(list(v.get("tgt", []))) for v in gt_entities.values()]
        out["gt_group_count"] = int(len(gt_entities))
        out["gt_mean_group_size"] = float(sum(gt_sizes) / len(gt_sizes)) if gt_sizes else 0.0
        out["over_merge_ratio"] = float(
            sum(1 for size in sizes if gt_sizes and size > max(gt_sizes)) / max(1, len(sizes))
        )
        out["under_merge_ratio"] = float(
            sum(1 for size in sizes if size <= 1) / max(1, len(sizes))
        )
    return out
