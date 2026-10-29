# 文件作用：借助真实实体做目标节点排序诊断，用于分析基础矩阵能提供多少匹配信息。
# 数据来源：scripts/run_m2m_experiments.py → evaluate_algorithm() 在开启 oracle 诊断时传入 S 与真实实体。
# 结果去向：oracle_grouped_topk_target_metrics() 返回带诊断前缀的目标侧分数，由实验入口单独记录。
# 阅读建议：先看 oracle_grouped_topk()；这里使用真实源分组和目标组大小，不是正常预测流程。

"""Explicitly oracle-assisted diagnostics for many-to-many alignment."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import torch

from PlanetAlign.metrics import global_set_f1
from PlanetAlign.metrics.metrics_MSF1 import f1_from_sets


EntityMap = Dict[str, Dict[str, List[int]]]

ORACLE_GROUPED_TOPK_PROTOCOL = "oracle_source_groups_and_target_cardinality"
ORACLE_GROUPED_TOPK_PREFIX = "OracleSrc+GT-K-"


# 核心思路：按真实源组汇总 S 的行，再按真实目标组大小选择 top-k 节点。
# 输出结果：实体字典供 oracle_grouped_topk_target_metrics() 诊断，不能当作盲预测结果。
def oracle_grouped_topk(
    similarity: torch.Tensor,
    gt_entities: Mapping[str, Mapping[str, Sequence[int]]],
    source_key: str = "src",
    target_key: str = "tgt",
    aggregation: str = "mean",
) -> EntityMap:
    """Produce an oracle-assisted target-ranking diagnostic from ``similarity``.

    This function uses the ground-truth source groups as its queries and the
    ground-truth target cardinality as the per-entity top-k. It is therefore
    only a diagnostic for target ranking conditional on those two pieces of
    ground truth. It is not blind or end-to-end M2M evaluation and must not be
    reported as a normal M2M result.

    The corresponding protocol metadata is
    ``oracle_source_groups_and_target_cardinality``.
    """

    if similarity.dim() != 2:
        raise ValueError("similarity must be a 2D tensor of shape (num_src_nodes, num_tgt_nodes)")
    if aggregation not in {"mean", "sum", "max"}:
        raise ValueError("aggregation must be one of: 'mean', 'sum', 'max'")

    similarity = similarity.detach().to(torch.float32).cpu()
    num_src_nodes, num_tgt_nodes = similarity.shape
    pred_entities: EntityMap = {}

    for eid, gt_item in gt_entities.items():
        src_nodes = [int(node) for node in gt_item.get(source_key, [])]
        src_nodes = [node for node in src_nodes if 0 <= node < num_src_nodes]
        k = min(len(gt_item.get(target_key, [])), num_tgt_nodes)

        if not src_nodes or k == 0:
            target_nodes: List[int] = []
        else:
            query_scores = similarity[src_nodes]
            if aggregation == "mean":
                target_scores = query_scores.mean(dim=0)
            elif aggregation == "sum":
                target_scores = query_scores.sum(dim=0)
            else:
                target_scores = query_scores.max(dim=0).values
            target_nodes = [int(node) for node in torch.topk(target_scores, k=k).indices.tolist()]

        pred_entities[str(eid)] = {
            source_key: src_nodes,
            target_key: target_nodes,
        }

    return pred_entities


# 功能：只计算该诊断适用的目标侧集合分数，返回明确标记为 oracle 的指标。
def oracle_grouped_topk_target_metrics(
    gt_entities: Mapping[str, Mapping[str, Sequence[int]]],
    pred_entities: Mapping[str, Mapping[str, Sequence[int]]],
    target_key: str = "tgt",
) -> Dict[str, float]:
    """Return only target-side metrics appropriate for the oracle diagnostic."""

    # 目标侧 oracle 诊断保持原来的按编号宏平均，不改成双侧、重新匹配的 MSF1-v2。
    target_f1 = [
        f1_from_sets(pred_entities.get(eid, {}).get(target_key, []), gt_entities[eid].get(target_key, []))[0]
        for eid in sorted(gt_entities)
    ]
    macro = sum(target_f1) / len(target_f1) if target_f1 else 0.0
    micro = global_set_f1(gt_entities, pred_entities, key=target_key)
    return {
        f"{ORACLE_GROUPED_TOPK_PREFIX}Target-MicroF1": float(micro["F1"]),
        f"{ORACLE_GROUPED_TOPK_PREFIX}Target-MSF1": float(macro),
        f"{ORACLE_GROUPED_TOPK_PREFIX}Target-Recall@GT-K": float(micro["Recall"]),
    }


__all__ = [
    "ORACLE_GROUPED_TOPK_PREFIX",
    "ORACLE_GROUPED_TOPK_PROTOCOL",
    "oracle_grouped_topk",
    "oracle_grouped_topk_target_metrics",
]
