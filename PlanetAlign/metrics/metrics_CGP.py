"""CGP-v1: standard precision/recall/F1 of cross-graph node relations.

Only non-overlapping units are supported. Counts are computed without
materializing Cartesian products or matching predicted units to GT units.
"""

from collections import Counter
from typing import Dict, Iterable, Mapping, Union

from .metrics_MSF1 import InvalidPredictionError, _unit_sets


CGP_PROTOCOL = "CGP-v1"
CGP_METRICS = ("CGP-P", "CGP-R", "CGP-F1")


class NoEvaluableRelationsError(ValueError):
    """The GT has no cross-graph relations; exclude this evaluation from summaries."""


class UnsupportedCGPOverlapError(ValueError):
    """CGP-v1's equivalent counts do not support overlapping benchmarks."""


def _validated_sides(entities, *, ground_truth=False):
    # Reuse the existing membership/partition validator without changing MSF1.
    # Its prediction error class lets us distinguish overlap from bad node IDs.
    try:
        units = _unit_sets(entities, require_disjoint=True)
    except InvalidPredictionError as exc:
        # MSF1's suggestion to enable allow_overlap does not apply to CGP-v1.
        detail = str(exc).partition(". Use allow_overlap=True")[0]
        if ground_truth:
            detail = detail.replace("Invalid prediction", "Overlapping ground truth", 1)
            raise UnsupportedCGPOverlapError(
                f"CGP-v1 does not support overlapping ground truth. {detail}"
            ) from exc
        raise InvalidPredictionError(
            f"CGP-v1 does not support overlapping predictions. {detail}"
        ) from exc
    return [({node for side, node in unit if side == "src"},
             {node for side, node in unit if side == "tgt"}) for unit in units]


def cross_graph_pair_scores(
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    pred_entities: Mapping[str, Mapping[str, Iterable[int]]],
    *,
    allow_overlap: bool = False,
) -> Dict[str, Union[int, float]]:
    """Count and score the union of src x tgt relations, pooled per dataset.

    Evaluate a predicted relation (s, t) iff s is in the GT source union OR t
    is in the GT target union. Entire predicted memberships are retained;
    only relations with BOTH endpoints out of scope are excluded.

    Validates the full GT and full prediction before scope filtering. Shared
    same-side nodes across predicted units raise InvalidPredictionError.
    Overlapping GT or allow_overlap=True raises UnsupportedCGPOverlapError.
    Missing sides, single-sided units and empty units generate no relations.
    Duplicate members within one unit are set-valued; inputs are not mutated.

    Returns CGP-P/R/F1 plus integer TP/FP/FN/NumGTPairs/NumPredPairs. A zero
    denominator yields 0, except that zero GT relations raises
    NoEvaluableRelationsError rather than emitting an aggregatable zero score.

    Partition validation makes these counts exact: GT and predicted products
    are disjoint. GT node-owner lookups count bilateral intersections for TP.
    No pair set, unit-assignment matrix or unit-matching solver is created;
    memory scales with memberships, not with the number of Cartesian pairs.
    """
    if allow_overlap:
        raise UnsupportedCGPOverlapError("CGP-v1 does not support overlapping benchmarks.")

    gt = _validated_sides(gt_entities, ground_truth=True)
    pred = _validated_sides(pred_entities)
    num_gt_pairs = sum(len(src) * len(tgt) for src, tgt in gt)
    if not num_gt_pairs:
        raise NoEvaluableRelationsError(
            "无可评估真实关系 (CGP-v1: ground truth contains no cross-graph relations)."
        )

    # Include nodes from single-sided GT units in the test scope as specified.
    src_owner = {node: i for i, (src, _) in enumerate(gt) for node in src}
    tgt_owner = {node: i for i, (_, tgt) in enumerate(gt) for node in tgt}
    num_pred_pairs, tp = 0, 0
    for src, tgt in pred:
        src_hits = Counter(src_owner[s] for s in src if s in src_owner)
        tgt_hits = Counter(tgt_owner[t] for t in tgt if t in tgt_owner)
        outside_src = len(src) - sum(src_hits.values())
        outside_tgt = len(tgt) - sum(tgt_hits.values())
        num_pred_pairs += len(src) * len(tgt) - outside_src * outside_tgt
        tp += sum(count * tgt_hits.get(i, 0) for i, count in src_hits.items())

    fp, fn = num_pred_pairs - tp, num_gt_pairs - tp
    return {
        "CGP-P": tp / num_pred_pairs if num_pred_pairs else 0.0,
        "CGP-R": tp / num_gt_pairs,
        "CGP-F1": 2 * tp / (2 * tp + fp + fn),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "NumGTPairs": num_gt_pairs,
        "NumPredPairs": num_pred_pairs,
    }
