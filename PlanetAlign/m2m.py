"""Reusable interfaces for many-to-many graph alignment research.

This module collects the pieces needed by new many-to-many alignment methods:

- load a generated ``*_m2m.pt`` dataset and its ``*_gt_many2many.json`` file;
- normalize/load/save prediction files in the standard entity-map format;
- evaluate either explicit group predictions or a pairwise similarity matrix;
- provide a lightweight baseline mixin for new algorithms.

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

from PlanetAlign.algorithms.base_model import BaseModel
from PlanetAlign.data import Dataset
from PlanetAlign.metrics import many_to_many_scores, similarity_to_pred_entities


EntityMap = Dict[str, Dict[str, List[int]]]


def use_full_anchor_supervision(dataset) -> None:
    """Give a generated M2M dataset its intended training supervision.

    The M2M builder stores the *original* train split (the paper's 20%
    protocol) in ``anchor_links`` and builds the entity ground truth from the
    original test split — supervision and evaluation are disjoint by
    construction (verified: anchor nodes never appear in GT entities, all 9
    datasets). Loading such a dataset through ``Dataset(train_ratio=0.2)``
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


def ground_truth_path(root: Union[str, Path], name: str) -> Path:
    """Return the standard ground-truth JSON path for an M2M dataset."""

    return Path(root) / f"{name}_gt_many2many.json"


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


def _entity_nodes(item: Mapping[str, Iterable[int]], sides: Sequence[str]) -> set[tuple[str, int]]:
    nodes = set()
    for side in sides:
        for node in item.get(side, []):
            nodes.add((str(side), int(node)))
    return nodes


def align_prediction_to_ground_truth(
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    pred_entities: Mapping[str, Mapping[str, Iterable[int]]],
    sides: Sequence[str] = DEFAULT_ENTITY_SIDES,
) -> EntityMap:
    """Relabel predicted group ids by node-overlap with GT groups.

    The many-to-many metrics ACS/MSF1/MicroF1 are keyed by entity id, while
    many group discovery methods produce arbitrary cluster ids. This adapter
    performs a one-to-one greedy matching from predicted groups to GT groups so
    that a pure group-id permutation does not lower keyed metrics. Unmatched
    prediction groups are retained under ``__extra__`` ids so cluster-level
    metrics such as SGS/EGS still see over-segmentation or extra clusters.
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


def evaluate_predictions(
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    pred_entities: Mapping[str, Mapping[str, Iterable[int]]],
    metrics: Optional[Iterable[str]] = None,
    sides: Sequence[str] = DEFAULT_ENTITY_SIDES,
    match_entities: bool = True,
) -> Dict[str, float]:
    """Evaluate explicit many-to-many group predictions."""

    gt = normalize_entity_map(gt_entities, sides=sides)
    if match_entities:
        pred = align_prediction_to_ground_truth(gt, pred_entities, sides=sides)
    else:
        pred = complete_prediction_entities(gt, pred_entities, sides=sides, keep_extra=True)
    return many_to_many_scores(gt, pred, metrics=metrics)


def predictions_from_similarity(
    similarity: torch.Tensor,
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    source_key: str = "src",
    target_key: str = "tgt",
    aggregation: str = "mean",
    top_k: Optional[int] = None,
) -> EntityMap:
    """Convert a pairwise similarity matrix into standard M2M predictions."""

    gt = normalize_entity_map(gt_entities, sides=(source_key, target_key))
    return similarity_to_pred_entities(
        similarity=similarity,
        gt_entities=gt,
        source_key=source_key,
        target_key=target_key,
        aggregation=aggregation,
        top_k=top_k,
    )


def evaluate_similarity(
    similarity: torch.Tensor,
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    metrics: Optional[Iterable[str]] = None,
    source_key: str = "src",
    target_key: str = "tgt",
    aggregation: str = "mean",
    top_k: Optional[int] = None,
    return_predictions: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], EntityMap]]:
    """Evaluate a node-level similarity matrix under M2M metrics."""

    pred = predictions_from_similarity(
        similarity=similarity,
        gt_entities=gt_entities,
        source_key=source_key,
        target_key=target_key,
        aggregation=aggregation,
        top_k=top_k,
    )
    scores = evaluate_predictions(gt_entities, pred, metrics=metrics, sides=(source_key, target_key))
    if return_predictions:
        return scores, pred
    return scores


class ManyToManyBaseline(BaseModel):
    """Base class for new pairwise many-to-many alignment baselines.

    Subclasses should implement ``train()`` following ``BaseModel`` and set
    ``self.S`` to a dense ``[num_src_nodes, num_tgt_nodes]`` similarity matrix.
    They then inherit ``predict_many_to_many()`` and ``test_many_to_many()``.
    Algorithms that directly produce group predictions may override
    ``predict_many_to_many()``.
    """

    def predict_many_to_many(
        self,
        gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
        source_key: str = "src",
        target_key: str = "tgt",
        aggregation: str = "mean",
        top_k: Optional[int] = None,
    ) -> EntityMap:
        if self.S is None:
            raise RuntimeError("Model is not trained yet, call train() before predicting")
        return predictions_from_similarity(
            similarity=self.S.detach().to(torch.float32).cpu(),
            gt_entities=gt_entities,
            source_key=source_key,
            target_key=target_key,
            aggregation=aggregation,
            top_k=top_k,
        )

    def test_many_to_many(
        self,
        gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
        metrics: Optional[Iterable[str]] = None,
        source_key: str = "src",
        target_key: str = "tgt",
        aggregation: str = "mean",
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        pred = self.predict_many_to_many(
            gt_entities=gt_entities,
            source_key=source_key,
            target_key=target_key,
            aggregation=aggregation,
            top_k=top_k,
        )
        return evaluate_predictions(gt_entities, pred, metrics=metrics, sides=(source_key, target_key))


def evaluate_model(
    model: BaseModel,
    dataset: Dataset,
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    gids: Tuple[int, int] = (0, 1),
    train_kwargs: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Iterable[str]] = None,
    one_to_one_metrics: Optional[Iterable[str]] = ("Hits@1", "Hits@10", "MRR"),
    save_log: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train a baseline model and evaluate both 1-1 and M2M metrics.

    This helper expects the model to either set ``self.S`` during ``train()``
    or return a dense similarity matrix as the first return value.
    """

    kwargs = dict(train_kwargs or {})
    ret = model.train(dataset=dataset, gids=list(gids), save_log=save_log, verbose=verbose, **kwargs)
    if model.S is None:
        if isinstance(ret, tuple) and ret and torch.is_tensor(ret[0]) and ret[0].dim() == 2:
            model.S = ret[0]
        elif torch.is_tensor(ret) and ret.dim() == 2:
            model.S = ret
        else:
            raise RuntimeError("model did not set self.S or return a 2D similarity matrix")

    scores: Dict[str, Any] = {}
    if one_to_one_metrics is not None:
        scores.update(model.test(dataset=dataset, gids=list(gids), metrics=list(one_to_one_metrics)))
    scores.update(evaluate_similarity(model.S.detach().to(torch.float32).cpu(), gt_entities, metrics=metrics))
    return scores


__all__ = [
    "EntityMap",
    "LoadedM2MBenchmark",
    "ManyToManyBaseline",
    "align_prediction_to_ground_truth",
    "complete_prediction_entities",
    "evaluate_model",
    "evaluate_predictions",
    "evaluate_similarity",
    "ground_truth_path",
    "load_entity_map",
    "load_m2m_benchmark",
    "normalize_entity_map",
    "predictions_from_similarity",
    "save_entity_map",
]
