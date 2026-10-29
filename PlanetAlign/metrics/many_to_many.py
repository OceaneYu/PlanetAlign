# 文件作用：比较真实与预测实体集合，返回各多对多指标的分数。
# 数据来源：PlanetAlign/m2m/core.py → evaluate_predictions() 同时传入原始预测和供其他指标使用的编号适配结果。
# 核心依赖：PlanetAlign/metrics/metrics_MSF1.py、PlanetAlign/metrics/metrics_Micro_SF1.py、PlanetAlign/metrics/metrics_AUIS.py、PlanetAlign/metrics/metrics_AUCS.py 的评分函数。
# 阅读建议：先看 many_to_many_scores() 如何选指标和返回 scores，需要了解分数定义时再读具体指标。

from typing import Callable, Dict, Iterable, List, Optional

from .metrics_MSF1 import macro_f1_score
from .metrics_Micro_SF1 import global_set_f1, micro_f1_score
from .metrics_AUCS import aucs_score, weighted_aucs_score
from .metrics_AUIS import auis_score, weighted_auis_score


EntityMap = Dict[str, Dict[str, List[int]]]

DEFAULT_MANY_TO_MANY_METRICS = ("Macro Set-F1", "Micro Set-F1", "AUIS", "AUCS")


def _normalize_metric_name(metric: str) -> str:
    return metric.strip().lower().replace("_", "-").replace(" ", "")


# 功能：根据请求的指标名称调用评分函数，默认计算 Macro/Micro Set-F1 与 AUIS/AUCS。
# 输出结果：返回“指标名称→浮点分数”的 scores 字典，交回评估包装函数，再由实验入口汇总保存。
def many_to_many_scores(
    gt_entities: EntityMap,
    pred_entities: EntityMap,
    metrics: Optional[Iterable[str]] = None,
    *,
    aligned_pred_entities: Optional[EntityMap] = None,
    allow_overlap: bool = False,
) -> Dict[str, float]:
    """Compute a compact set of many-to-many alignment scores.

    Parameters
    ----------
    gt_entities
        Ground-truth entities in the JSON-compatible format used by the
        many-to-many metric files: ``{eid: {"src": [...], "tgt": [...]}}``.
    pred_entities
        Original predicted entities. MSF1-v2 always uses these groups directly.
    metrics
        Optional iterable of metric names. Defaults to Macro Set-F1,
        Micro Set-F1, AUIS, and AUCS. Weighted AUIS/AUCS are supplemental
        diagnostics and must be requested explicitly.
    aligned_pred_entities
        Optional existing entity-id alignment for Micro Set-F1 and AUIS/AUCS.
        If omitted, those metrics use pred_entities as before. Never used by MSF1.
    allow_overlap
        Explicit opt-in for overlapping benchmarks. MSF1 otherwise rejects
        same-side nodes shared by different units. Other metrics are unchanged.
    """

    metric_fns: Dict[str, tuple[str, Callable[[EntityMap, EntityMap], float]]] = {
        "macroset-f1": ("Macro Set-F1", macro_f1_score),
        "macro-set-f1": ("Macro Set-F1", macro_f1_score),
        "microset-f1": ("Micro Set-F1", micro_f1_score),
        "micro-set-f1": ("Micro Set-F1", micro_f1_score),
        "msf1": ("MSF1", macro_f1_score),
        "macro-f1": ("MSF1", macro_f1_score),
        "macrof1": ("MSF1", macro_f1_score),
        "microf1": ("MicroF1", micro_f1_score),
        "micro-f1": ("MicroF1", micro_f1_score),
        "auis": ("AUIS", auis_score),
        "aucs": ("AUCS", aucs_score),
        "weighted-auis": ("Weighted-AUIS", weighted_auis_score),
        "weighted-aucs": ("Weighted-AUCS", weighted_aucs_score),
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
        prediction = pred_entities if fn is macro_f1_score or aligned_pred_entities is None else aligned_pred_entities
        if fn is macro_f1_score:
            results[out_name] = float(fn(gt_entities, prediction, allow_overlap=allow_overlap))
        else:
            results[out_name] = float(fn(gt_entities, prediction))

    if invalid_metrics:
        valid = ", ".join(sorted({name for name, _ in metric_fns.values()}))
        raise ValueError(f"Invalid many-to-many metrics {invalid_metrics}. Valid metrics: {valid}")

    return results
