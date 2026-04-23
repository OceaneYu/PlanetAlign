from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from .metrics_ACS import acs_score, concentration_accuracy_score
from .metrics_MSF1 import macro_f1_score, macro_set_f1
from .metrics_Micro_SF1 import global_set_f1, micro_f1_score
from .metrics_m2m_EGS import egs_score, egs_weighted_score, m2m_egs_score
from .metrics_m2m_SGS import m2m_sgs_score, sgs_score, sgs_weighted_score


EntityMap = Dict[str, Dict[str, List[int]]]

DEFAULT_MANY_TO_MANY_METRICS = ("ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS")


def _normalize_metric_name(metric: str) -> str:
    return metric.strip().lower().replace("_", "-").replace(" ", "")


def many_to_many_scores(
    gt_entities: EntityMap,
    pred_entities: EntityMap,
    metrics: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Compute a compact set of many-to-many alignment scores.

    Parameters
    ----------
    gt_entities
        Ground-truth entities in the JSON-compatible format used by the
        many-to-many metric files: ``{eid: {"src": [...], "tgt": [...]}}``.
    pred_entities
        Predicted entities in the same format.
    metrics
        Optional iterable of metric names. Defaults to ACS, MSF1, MicroF1,
        M2M-SGS, and M2M-EGS. Weighted variants can be requested with
        ``Weighted-M2M-SGS`` and ``Weighted-M2M-EGS``.
    """

    metric_fns: Dict[str, tuple[str, Callable[[EntityMap, EntityMap], float]]] = {
        "acs": ("ACS", acs_score),
        "msf1": ("MSF1", macro_f1_score),
        "macro-f1": ("MSF1", macro_f1_score),
        "macrof1": ("MSF1", macro_f1_score),
        "microf1": ("MicroF1", micro_f1_score),
        "micro-f1": ("MicroF1", micro_f1_score),
        "m2m-sgs": ("M2M-SGS", sgs_score),
        "sgs": ("M2M-SGS", sgs_score),
        "m2m-egs": ("M2M-EGS", egs_score),
        "egs": ("M2M-EGS", egs_score),
        "weighted-m2m-sgs": ("Weighted-M2M-SGS", sgs_weighted_score),
        "m2m-sgs-weighted": ("Weighted-M2M-SGS", sgs_weighted_score),
        "weighted-sgs": ("Weighted-M2M-SGS", sgs_weighted_score),
        "weighted-m2m-egs": ("Weighted-M2M-EGS", egs_weighted_score),
        "m2m-egs-weighted": ("Weighted-M2M-EGS", egs_weighted_score),
        "weighted-egs": ("Weighted-M2M-EGS", egs_weighted_score),
    }

    requested = DEFAULT_MANY_TO_MANY_METRICS if metrics is None else tuple(metrics)
    results: Dict[str, float] = {}
    invalid_metrics = []
    for metric in requested:
        key = _normalize_metric_name(metric)
        if key not in metric_fns:
            invalid_metrics.append(metric)
            continue
        out_name, fn = metric_fns[key]
        results[out_name] = float(fn(gt_entities, pred_entities))

    if invalid_metrics:
        valid = ", ".join(sorted({name for name, _ in metric_fns.values()}))
        raise ValueError(f"Invalid many-to-many metrics {invalid_metrics}. Valid metrics: {valid}")

    return results


def similarity_to_pred_entities(
    similarity: torch.Tensor,
    gt_entities: EntityMap,
    source_key: str = "src",
    target_key: str = "tgt",
    aggregation: str = "mean",
    top_k: Optional[int] = None,
) -> EntityMap:
    """Convert a pairwise similarity matrix into many-to-many predictions.

    For each ground-truth entity, the source-side nodes are treated as the
    query set. Their similarity rows are aggregated over all target nodes, then
    the top-k target nodes are selected. By default, k equals the size of that
    entity's ground-truth target set.
    """

    if similarity.dim() != 2:
        raise ValueError("similarity must be a 2D tensor of shape (num_src_nodes, num_tgt_nodes)")
    if aggregation not in {"mean", "sum", "max"}:
        raise ValueError("aggregation must be one of: 'mean', 'sum', 'max'")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative")

    num_src_nodes, num_tgt_nodes = similarity.shape
    pred_entities: EntityMap = {}

    for eid, gt_item in gt_entities.items():
        pred_item: Dict[str, Any] = {}
        for key, value in gt_item.items():
            pred_item[key] = [int(v) for v in value] if isinstance(value, list) else value

        src_nodes = [int(n) for n in gt_item.get(source_key, [])]
        src_nodes = [n for n in src_nodes if 0 <= n < num_src_nodes]
        target_size = len(gt_item.get(target_key, []))
        k = target_size if top_k is None else top_k
        k = min(k, num_tgt_nodes)

        if k == 0 or not src_nodes:
            pred_item[target_key] = []
            pred_entities[eid] = pred_item
            continue

        query_scores = similarity[src_nodes]
        if aggregation == "mean":
            target_scores = query_scores.mean(dim=0)
        elif aggregation == "sum":
            target_scores = query_scores.sum(dim=0)
        else:
            target_scores = query_scores.max(dim=0).values

        pred_item[target_key] = [int(i) for i in torch.topk(target_scores, k=k).indices.cpu().tolist()]
        pred_entities[eid] = pred_item

    return pred_entities


def pred_entities_from_similarity(
    similarity: torch.Tensor,
    gt_entities: EntityMap,
    source_key: str = "src",
    target_key: str = "tgt",
    aggregation: str = "mean",
    top_k: Optional[int] = None,
) -> EntityMap:
    """Alias for :func:`similarity_to_pred_entities`."""

    return similarity_to_pred_entities(
        similarity=similarity,
        gt_entities=gt_entities,
        source_key=source_key,
        target_key=target_key,
        aggregation=aggregation,
        top_k=top_k,
    )
