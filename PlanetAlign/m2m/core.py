# 文件作用：统一实体字典格式，提供数据读写、训练锚点恢复和预测评分的辅助接口。
# 数据来源：基准接口读 .pt/JSON；实验入口将真实与预测分组传给 evaluate_predictions() 评分。
# 结果去向：evaluate_predictions() 将原始组交给 MSF1，将编号适配结果交给其余指标；由 PlanetAlign/metrics/many_to_many.py 汇总。
# 阅读建议：训练前看 use_full_anchor_supervision()，评分先看 evaluate_predictions() 的输入分流，再看旧编号适配。

"""Reusable interfaces for many-to-many graph alignment research.

This module collects the pieces needed by new many-to-many alignment methods:

- load a generated ``*_m2m.pt`` dataset and its ``*_gt_many2many.json`` file;
- normalize/load/save prediction files in the standard entity-map format;
- evaluate explicit group predictions.

Prediction format
-----------------
The canonical prediction object is JSON-compatible::

    {
        "e0": {"src": [0, 1], "tgt": [5, 6]},
        "e1": {"src": [2], "tgt": [7, 8, 9]}
    }

Prediction JSON files may either be this raw map or a wrapped payload with an
``"entities"`` field. The wrapped form is preferred because it can carry
dataset metadata::

    {
        "dataset": "cora_m2m",
        "graphs": ["cora1", "cora2"],
        "entities": {...},
        "metadata": {"method": "MyBaseline"}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from PlanetAlign.data import Dataset
from PlanetAlign.metrics import many_to_many_scores


EntityMap = Dict[str, Dict[str, List[int]]]


# 输入来源：scripts/run_quotient_compare.py → main() 传入刚加载的多对多 Dataset。
# 功能：把 .pt 中已保留的原始训练锚点全部恢复到 train_data，并清空 test_data，避免重复划分。
# 流程位置：应在生成的多对多基准加载后、基础算法训练前调用。
def use_full_anchor_supervision(dataset) -> None:
    """Give a generated M2M dataset its intended training supervision.

    The M2M builder stores the *original* train split (the paper's 20%
    protocol) in ``anchor_links`` and builds the entity ground truth from the
    original test split — supervision and evaluation are node-disjoint, an
    invariant now *enforced by the generator* (cross-set collision exclusion +
    a hard post-condition in ``many2many_builder``; the original flickr-lastfm
    data carries a duplicated anchor pair that previously leaked into entity
    e62 — fixed and regenerated 2026-07-12, see the build manifests'
    amendments). Loading such a dataset through ``Dataset(train_ratio=0.2)``
    therefore *double-splits* the supervision down to ~4% of the original
    anchors, which starves structure-reliant graphs (arenas/phone-email/italy
    collapse to Hits@1 ~ 0 while their originals reach 0.98/0.35/0.10).

    Call this right after constructing the Dataset: all of ``anchor_links``
    becomes ``train_data`` and ``test_data`` is emptied (node-level Hits has no
    held-out pairs under this protocol; the benchmark's evaluation is the
    blind entity map).
    """
    dataset.train_data = dataset._anchor_links.clone()
    dataset.test_data = dataset._anchor_links[:0]
DEFAULT_ENTITY_SIDES = ("src", "tgt")


# 功能：把 Dataset、真实实体、文件路径和构建信息放在同一个返回对象里。
@dataclass
class LoadedM2MBenchmark:
    """A generated many-to-many benchmark loaded from disk."""

    dataset: Dataset
    entities: EntityMap
    root: Path
    name: str
    graphs: List[str]
    metadata: Dict[str, Any]
    gt_path: Path


# 输出结果：按基准命名约定拼出 <name>_gt_many2many.json 的路径，不读取文件。
def ground_truth_path(root: Union[str, Path], name: str) -> Path:
    """Return the standard ground-truth JSON path for an M2M dataset."""

    return Path(root) / f"{name}_gt_many2many.json"


# 功能：把实体编号转为字符串、节点编号转为整数，去重并按可选节点总数检查范围。
# 输出结果：返回规范化字典，供 load_entity_map()、save_entity_map() 和评估函数使用。
def normalize_entity_map(
    entities: Mapping[str, Mapping[str, Iterable[int]]],
    sides: Sequence[str] = DEFAULT_ENTITY_SIDES,
    num_nodes: Optional[Mapping[str, int]] = None,
    fill_missing_sides: bool = True,
) -> EntityMap:
    """Validate and normalize an entity map.

    The returned map has string entity ids, integer node ids, duplicate nodes
    removed within each side, and side keys in ``sides``. If ``num_nodes`` is
    provided, node ids are checked to be within range for the corresponding
    side.
    """

    if not isinstance(entities, Mapping):
        raise TypeError("entities must be a mapping from entity id to side-node maps")

    normalized: EntityMap = {}
    for raw_eid, item in entities.items():
        if not isinstance(item, Mapping):
            raise TypeError(f"entity {raw_eid!r} must map side names to node lists")

        eid = str(raw_eid)
        if eid in normalized:
            raise ValueError(f"entity ids collide after string normalization: {raw_eid!r}")
        normalized[eid] = {}
        for side in sides:
            if side not in item:
                if fill_missing_sides:
                    normalized[eid][side] = []
                    continue
                raise ValueError(f"entity {eid!r} is missing side {side!r}")

            value = item[side]
            if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
                raise TypeError(f"entity {eid!r} side {side!r} must be an iterable of node ids")

            seen = set()
            nodes: List[int] = []
            for raw_node in value:
                node = int(raw_node)
                limit = None if num_nodes is None else num_nodes.get(side)
                if limit is not None and not 0 <= node < limit:
                    raise ValueError(
                        f"entity {eid!r} side {side!r} has node {node}, "
                        f"outside valid range [0, {limit})"
                    )
                if node not in seen:
                    seen.add(node)
                    nodes.append(node)
            normalized[eid][side] = nodes
    return normalized


# 输入来源：读取原始实体字典或带 entities 字段的 JSON，再交给 normalize_entity_map()。
def load_entity_map(
    path: Union[str, Path],
    sides: Sequence[str] = DEFAULT_ENTITY_SIDES,
    num_nodes: Optional[Mapping[str, int]] = None,
) -> EntityMap:
    """Load a raw or wrapped entity-map JSON file."""

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    entities = payload["entities"] if isinstance(payload, Mapping) and "entities" in payload else payload
    return normalize_entity_map(entities, sides=sides, num_nodes=num_nodes)


# 功能：校验预测字典并写入 JSON，可附加数据集名称、图名称和元信息。
# 输出结果：返回写入路径；与 load_entity_map() 组成预测文件的读写接口。
def save_entity_map(
    entities: Mapping[str, Mapping[str, Iterable[int]]],
    path: Union[str, Path],
    dataset_name: Optional[str] = None,
    graphs: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    sides: Sequence[str] = DEFAULT_ENTITY_SIDES,
) -> Path:
    """Save predictions in the preferred wrapped JSON format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_entity_map(entities, sides=sides)
    payload: Dict[str, Any] = {"entities": normalized}
    if dataset_name is not None:
        payload["dataset"] = dataset_name
    if graphs is not None:
        payload["graphs"] = list(graphs)
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# 功能：调用 PlanetAlign/data/dataset.py → Dataset 加载图，同时读取真实实体 JSON。
# 输出结果：返回 LoadedM2MBenchmark；此函数不自动调用 use_full_anchor_supervision()。
def load_m2m_benchmark(
    root: Union[str, Path],
    name: str,
    train_ratio: float = 0.2,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    gids: Tuple[int, int] = (0, 1),
) -> LoadedM2MBenchmark:
    """Load a generated M2M ``.pt`` dataset and its ground-truth JSON."""

    root = Path(root)
    dataset = Dataset(root=root, name=name, train_ratio=train_ratio, seed=seed, dtype=dtype)
    gt_path = ground_truth_path(root, name)
    if not gt_path.exists():
        raise FileNotFoundError(f"many-to-many ground-truth JSON not found: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    side_limits = {
        "src": int(dataset.pyg_graphs[gids[0]].num_nodes),
        "tgt": int(dataset.pyg_graphs[gids[1]].num_nodes),
    }
    entities = normalize_entity_map(payload["entities"], num_nodes=side_limits)
    return LoadedM2MBenchmark(
        dataset=dataset,
        entities=entities,
        root=root,
        name=name,
        graphs=list(payload.get("graphs", [])),
        metadata=dict(payload.get("metadata", {})),
        gt_path=gt_path,
    )


# 功能：按真实实体编号补齐缺失预测为两侧空列表，可选择保留额外预测。
def complete_prediction_entities(
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    pred_entities: Mapping[str, Mapping[str, Iterable[int]]],
    sides: Sequence[str] = DEFAULT_ENTITY_SIDES,
    keep_extra: bool = False,
) -> EntityMap:
    """Return predictions aligned to the GT entity ids.

    Missing predicted entities are treated as empty predictions. Extra
    predicted entities are ignored by the official scores because they are not
    keyed by a ground-truth entity.
    """

    gt = normalize_entity_map(gt_entities, sides=sides)
    pred = normalize_entity_map(pred_entities, sides=sides)
    completed: EntityMap = {}
    for eid in gt:
        completed[eid] = pred.get(eid, {side: [] for side in sides})
        for side in sides:
            completed[eid].setdefault(side, [])
    if keep_extra:
        for eid, item in pred.items():
            if eid not in completed:
                completed[eid] = item
    return completed


# 功能：把一个实体转换成 (图侧, 节点编号) 的集合，避免两图相同编号被误当成同一个节点。
# 结果去向：align_prediction_to_ground_truth() 用这些集合的重叠量匹配真实与预测实体编号。
def _entity_nodes(item: Mapping[str, Iterable[int]], sides: Sequence[str]) -> set[tuple[str, int]]:
    nodes = set()
    for side in sides:
        for node in item.get(side, []):
            nodes.add((str(side), int(node)))
    return nodes


# 流程位置：预测已完成后，为评分对应真实组和预测组的编号，不重新给节点分组。
# 输入来源：evaluate_predictions() 等评分调用方传入真实与预测实体字典。
# 核心思路：按共有成员多少排序，每个真实组与预测组至多对应一次；漏预测补空，额外预测保留。
# 输出结果：编号适配结果供 Micro-SF1、AUIS、AUCS 评分；MSF1-v2 不使用此结果。
def align_prediction_to_ground_truth(
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    pred_entities: Mapping[str, Mapping[str, Iterable[int]]],
    sides: Sequence[str] = DEFAULT_ENTITY_SIDES,
) -> EntityMap:
    """Relabel predicted group ids by node-overlap with GT groups.

    Micro Set-F1 is keyed by entity id, while
    many group discovery methods produce arbitrary cluster ids. This adapter
    performs a one-to-one greedy matching from predicted groups to GT groups so
    that a pure group-id permutation does not lower keyed metrics. Unmatched
    prediction groups are retained under ``__extra__`` ids so cluster-level
    AUIS/AUCS still see over-segmentation or extra clusters.
    """

    gt = normalize_entity_map(gt_entities, sides=sides)
    pred = normalize_entity_map(pred_entities, sides=sides)
    gt_nodes = {eid: _entity_nodes(item, sides) for eid, item in gt.items()}
    pred_nodes = {eid: _entity_nodes(item, sides) for eid, item in pred.items()}

    candidates: List[Tuple[int, float, str, str]] = []
    for geid, g_nodes in gt_nodes.items():
        for peid, p_nodes in pred_nodes.items():
            overlap = len(g_nodes & p_nodes)
            if overlap <= 0:
                continue
            union = len(g_nodes | p_nodes)
            jaccard = overlap / union if union else 0.0
            candidates.append((overlap, jaccard, geid, peid))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    matched_gt = set()
    matched_pred = set()
    aligned: EntityMap = {}
    for _, _, geid, peid in candidates:
        if geid in matched_gt or peid in matched_pred:
            continue
        aligned[geid] = pred[peid]
        matched_gt.add(geid)
        matched_pred.add(peid)

    for geid in gt:
        if geid not in aligned:
            aligned[geid] = {side: [] for side in sides}

    extra_index = 0
    for peid, item in pred.items():
        if peid in matched_pred:
            continue
        extra_eid = f"__extra__{extra_index}:{peid}"
        aligned[extra_eid] = item
        extra_index += 1
    return aligned


# 功能：只评分已有预测；MSF1 用原始组，其他指标仍按选项对应编号或补齐预测。
# 输入输出：接收真实实体 gt_entities、预测实体 pred_entities 和指标选项，返回 scores 分数字典。
# 调用关系：scripts/run_quotient_compare.py → main() 在预测后调用；内部交给 PlanetAlign/metrics/many_to_many.py → many_to_many_scores()。
def evaluate_predictions(
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    pred_entities: Mapping[str, Mapping[str, Iterable[int]]],
    metrics: Optional[Iterable[str]] = None,
    sides: Sequence[str] = DEFAULT_ENTITY_SIDES,
    match_entities: bool = True,
    *,
    allow_overlap: bool = False,
) -> Dict[str, float]:
    """Evaluate explicit group predictions; MSF1 defaults to non-overlapping units.

    Pass allow_overlap=True only for an overlapping benchmark. Invalid MSF1
    predictions raise InvalidPredictionError instead of being repaired.
    """

    gt = normalize_entity_map(gt_entities, sides=sides)
    raw_pred = normalize_entity_map(pred_entities, sides=sides)
    if match_entities:
        pred = align_prediction_to_ground_truth(gt, raw_pred, sides=sides)
    else:
        pred = complete_prediction_entities(gt, raw_pred, sides=sides, keep_extra=True)
    return many_to_many_scores(gt, raw_pred, metrics=metrics, aligned_pred_entities=pred,
                               allow_overlap=allow_overlap)


__all__ = [
    "EntityMap",
    "LoadedM2MBenchmark",
    "align_prediction_to_ground_truth",
    "complete_prediction_entities",
    "evaluate_predictions",
    "ground_truth_path",
    "load_entity_map",
    "load_m2m_benchmark",
    "normalize_entity_map",
    "save_entity_map",
]
