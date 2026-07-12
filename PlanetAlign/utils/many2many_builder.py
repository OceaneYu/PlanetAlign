"""Many-to-many graph alignment benchmark builder.

Converts an existing one-to-one alignment dataset (e.g. Douban, ACM-DBLP) into
a many-to-many benchmark via node splitting. Implements the four properties:

    1. Mixed granularity: entities are split as 1-1 / 1-N / N-1 / N-N according
       to a configurable ratio.
    2. Internal cohesion: nodes coming from the same split share intra-group
       edges with high probability.
    3. External structure at group level: neighbours of the original anchor
       node are redistributed across its split nodes so that the group (not any
       single node) preserves the original connectivity pattern.
    4. Fuzzy boundary: a small fraction of nodes are allowed to appear in more
       than one ground-truth entity (controlled by ``overlap_ratio``).

Output
------
- A PyTorch ``.pt`` file that follows ``PlanetAlign.data.Dataset`` schema
  (``graphs``, ``number_of_nodes``, ``edges``, ``node_attributes``,
  ``edge_attributes``, ``anchor_links``). ``anchor_links`` contains only the
  retained 1-1 pairs, so legacy one-to-one trainers keep working.
- ``gt_many2many.json`` in the format expected by the ACS / MSF1 / MicroF1 /
  M2M-SGS / M2M-EGS metrics: ``{"entities": {eid: {"src": [...], "tgt": [...]}}}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from PlanetAlign.data import Dataset

EntityMap = Dict[str, Dict[str, List[int]]]

DEFAULT_SPLIT_RATIOS: Dict[str, float] = {
    "1-1": 0.70,
    "1-many": 0.10,
    "many-1": 0.10,
    "many-many": 0.10,
}


@dataclass
class ManyToManyBenchmark:
    """Result container produced by :func:`build_many_to_many_benchmark`."""

    graphs: List[Data]
    entities: EntityMap
    train_anchors: torch.Tensor  # [K, 2] preserved 1-1 pairs (for legacy models)
    metadata: Dict[str, object] = field(default_factory=dict)
    name: str = "m2m"
    graph_names: Tuple[str, ...] = ("src", "tgt")

    def save(self, root: Union[str, Path]) -> Tuple[Path, Path]:
        """Persist the benchmark in the repo's standard formats.

        Writes two files under ``root``:
            - ``{name}.pt``: compatible with ``PlanetAlign.data.Dataset``.
            - ``{name}_gt_many2many.json``: used by the many-to-many metrics.
        """
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        pt_path = root / f"{self.name}.pt"
        json_path = root / f"{self.name}_gt_many2many.json"

        data_dict = {
            "graphs": list(self.graph_names),
            "number_of_nodes": [int(g.num_nodes) for g in self.graphs],
            "edges": [g.edge_index for g in self.graphs],
            "anchor_links": self.train_anchors,
        }
        if all(g.x is not None for g in self.graphs):
            data_dict["node_attributes"] = [g.x for g in self.graphs]
        if all(g.edge_attr is not None for g in self.graphs):
            data_dict["edge_attributes"] = [g.edge_attr for g in self.graphs]
        torch.save(data_dict, pt_path)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": self.name,
                    "graphs": list(self.graph_names),
                    "num_nodes": [int(g.num_nodes) for g in self.graphs],
                    "entities": self.entities,
                    "metadata": self.metadata,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return pt_path, json_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assign_entity_types(
    num_entities: int,
    split_ratios: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Deterministically allocate entity types according to the requested ratios.

    We use quota allocation + shuffle (instead of per-entity sampling) so that
    the resulting proportions exactly match ``split_ratios`` when the counts
    divide evenly, and are within one element otherwise.
    """
    types = list(split_ratios.keys())
    total = sum(split_ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0 (got {total:.4f})")

    quotas = [int(round(split_ratios[t] * num_entities)) for t in types]
    diff = num_entities - sum(quotas)
    # Fix any rounding drift by adjusting the largest bucket.
    if diff != 0:
        largest = int(np.argmax(quotas))
        quotas[largest] += diff

    labels = np.concatenate([np.full(q, i, dtype=np.int64) for i, q in enumerate(quotas)])
    rng.shuffle(labels)
    return np.array([types[i] for i in labels], dtype=object)


def _edge_index_to_neighbor_lists(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    neigh: List[List[int]] = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        neigh[u].append(v)
    return neigh


def _split_count(
    kind: str,
    max_expansion: int,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """Pick (k_src, k_tgt) — number of split nodes on each side."""
    if max_expansion < 2:
        raise ValueError("max_expansion must be >= 2")
    low, high = 2, max_expansion
    if kind == "1-1":
        return 1, 1
    if kind == "1-many":
        return 1, int(rng.integers(low, high + 1))
    if kind == "many-1":
        return int(rng.integers(low, high + 1)), 1
    if kind == "many-many":
        return int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))
    raise ValueError(f"unknown split type: {kind}")


def _split_one_node(
    orig_node: int,
    k: int,
    neighbors: List[int],
    next_new_id: int,
    internal_density: float,
    rng: np.random.Generator,
) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split a single node ``orig_node`` into ``k`` split nodes.

    Returns
    -------
    members : list of node ids (length ``k``) forming this entity group.
    external_edges : list of undirected pairs (member_id, neighbor) describing
        how the original neighbours get redistributed across the new members.
    internal_edges : list of undirected pairs between members (within group).

    Notes
    -----
    - Uses the convention that ``orig_node`` itself becomes member 0. New
      members receive fresh ids starting from ``next_new_id``.
    - Each original neighbour is independently attached to each member with
      probability ``1/k``, then at least one member is forced to keep the
      edge so that group-level connectivity is never lost.
    """
    if k == 1:
        return [orig_node], [(orig_node, w) for w in neighbors], []

    members = [orig_node] + list(range(next_new_id, next_new_id + k - 1))

    # External redistribution. Each neighbour is placed on each member with
    # probability 1/k (Bernoulli). We ensure every neighbour ends up on at
    # least one member so the group-level edge to w is preserved.
    external_edges: List[Tuple[int, int]] = []
    p = 1.0 / k
    for w in neighbors:
        mask = rng.random(k) < p
        if not mask.any():
            mask[rng.integers(0, k)] = True
        for i, keep in enumerate(mask):
            if keep:
                external_edges.append((members[i], w))

    # Internal edges (dense intra-group connectivity).
    internal_edges: List[Tuple[int, int]] = []
    for i in range(k):
        for j in range(i + 1, k):
            if rng.random() < internal_density:
                internal_edges.append((members[i], members[j]))
    # Guarantee the group is connected: if none sampled, add a chain.
    if k > 1 and not internal_edges:
        for i in range(k - 1):
            internal_edges.append((members[i], members[i + 1]))

    return members, external_edges, internal_edges


def _build_side(
    graph: Data,
    selected: Dict[int, int],  # orig_node -> k (split factor, may be 1)
    internal_density: float,
    rng: np.random.Generator,
) -> Tuple[Data, Dict[int, List[int]]]:
    """Apply node splitting to a single graph.

    Parameters
    ----------
    graph : Data
        Input PyG graph.
    selected : dict
        Mapping from original anchor node id to its split factor ``k``.
    internal_density : float
    rng : numpy Generator

    Returns
    -------
    new_graph : Data
        Expanded graph. Original node ids [0, num_nodes) keep their meaning;
        split nodes occupy ids [num_nodes, new_num_nodes).
    group_map : dict
        Mapping from anchor node id to the full list of member ids in its
        group (always includes the original id as the first element).
    """
    num_nodes = graph.num_nodes
    neigh = _edge_index_to_neighbor_lists(graph.edge_index, num_nodes)

    # Edges NOT incident to a split node stay verbatim; edges incident to a
    # split node are rewritten by ``_split_one_node``.
    split_set = {u for u, k in selected.items() if k > 1}
    kept_edges: List[Tuple[int, int]] = []
    if split_set:
        src = graph.edge_index[0].tolist()
        dst = graph.edge_index[1].tolist()
        for u, v in zip(src, dst):
            if u in split_set or v in split_set:
                continue
            kept_edges.append((u, v))
    else:
        kept_edges = list(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))

    group_map: Dict[int, List[int]] = {}
    new_edges: List[Tuple[int, int]] = []
    next_new_id = num_nodes
    attr_copy_from: List[int] = []  # orig_node id for each newly appended node

    for orig_node, k in selected.items():
        members, ext, intra = _split_one_node(
            orig_node=orig_node,
            k=k,
            neighbors=neigh[orig_node],
            next_new_id=next_new_id,
            internal_density=internal_density,
            rng=rng,
        )
        group_map[orig_node] = members
        if k > 1:
            next_new_id += k - 1
            attr_copy_from.extend([orig_node] * (k - 1))
        new_edges.extend(ext)
        new_edges.extend(intra)

    combined = kept_edges + new_edges
    if combined:
        ei = torch.tensor(combined, dtype=torch.long).t().contiguous()
        ei = to_undirected(ei, num_nodes=next_new_id)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)

    x = graph.x
    if x is not None and attr_copy_from:
        extra_x = x[torch.tensor(attr_copy_from, dtype=torch.long)]
        x = torch.cat([x, extra_x], dim=0)

    # Edge attributes: the original edge_attr is tied to the original edge
    # ordering, which we no longer preserve after splitting. If present, we
    # fall back to a simple strategy: drop edge_attr when we actually split
    # anything, and keep it only when the graph is unchanged.
    edge_attr = graph.edge_attr if not split_set else None

    new_graph = Data(
        name=getattr(graph, "name", "graph"),
        num_nodes=next_new_id,
        x=x,
        edge_index=ei,
        edge_attr=edge_attr,
    )
    return new_graph, group_map


def _apply_overlap(
    entities: EntityMap,
    overlap_ratio: float,
    total_nodes_by_side: Dict[str, int],
    rng: np.random.Generator,
) -> int:
    """Add fuzzy boundaries: push a few nodes into additional entities.

    ``overlap_ratio`` is interpreted per side: that fraction of nodes across
    all entities on one side become multi-group members. Returns the number
    of (node, side) pairs that were inserted.
    """
    if overlap_ratio <= 0.0:
        return 0
    eids = list(entities.keys())
    if len(eids) < 2:
        return 0

    inserted = 0
    for side, total in total_nodes_by_side.items():
        # Build a flat pool of (eid, node) from the current groups.
        pool: List[Tuple[str, int]] = []
        for eid in eids:
            for n in entities[eid].get(side, []):
                pool.append((eid, n))
        if not pool:
            continue
        target = int(round(overlap_ratio * total))
        target = min(target, len(pool))
        idx = rng.choice(len(pool), size=target, replace=False)
        for i in idx:
            src_eid, node = pool[int(i)]
            # Pick a different entity to share into.
            candidates = [e for e in eids if e != src_eid]
            if not candidates:
                continue
            tgt_eid = candidates[int(rng.integers(0, len(candidates)))]
            bag = entities[tgt_eid].setdefault(side, [])
            if node not in bag:
                bag.append(node)
                inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_many_to_many_benchmark(
    dataset: Dataset,
    gids: Tuple[int, int] = (0, 1),
    split_ratios: Optional[Dict[str, float]] = None,
    max_expansion: int = 4,
    internal_density: float = 0.8,
    overlap_ratio: float = 0.05,
    anchor_source: str = "test",
    train_anchor_source: str = "train",
    name: Optional[str] = None,
    seed: int = 42,
) -> ManyToManyBenchmark:
    """Construct a many-to-many alignment benchmark from a 1-1 dataset.

    Parameters
    ----------
    dataset : PlanetAlign.data.Dataset
        A loaded one-to-one alignment dataset (e.g. ``Douban``).
    gids : tuple of int
        Graph indices used as (source, target). Defaults to ``(0, 1)``.
    split_ratios : dict, optional
        Proportion of entities assigned to each type. Defaults to
        ``{"1-1": 0.70, "1-many": 0.10, "many-1": 0.10, "many-many": 0.10}``.
    max_expansion : int
        Upper bound on the number of split nodes on each side
        (inclusive). The sampled ``k`` lies in ``[2, max_expansion]``.
    internal_density : float
        Probability of connecting any two nodes inside the same split group.
    overlap_ratio : float
        Fraction of nodes (per side) that are allowed to appear in a second
        entity, modeling fuzzy group boundaries. 0 disables the feature.
    anchor_source : {"test", "all"}
        Where to draw the pool of anchors to convert into many-to-many
        entities. ``"test"`` uses ``dataset.test_data`` (standard choice);
        ``"all"`` uses every anchor link.
    train_anchor_source : {"train", "one_to_one", "none"}
        What to store in ``anchor_links`` of the new ``.pt`` file.
        ``"train"`` keeps the original training split unchanged (nodes are
        still valid ids after splitting). ``"one_to_one"`` keeps only the
        entities whose type ended up as 1-1. ``"none"`` writes an empty tensor.
    name : str, optional
        Dataset name stamped into the saved files.
    seed : int
        Random seed for reproducibility.
    """
    if split_ratios is None:
        split_ratios = dict(DEFAULT_SPLIT_RATIOS)
    if anchor_source not in {"test", "all"}:
        raise ValueError("anchor_source must be 'test' or 'all'")
    if train_anchor_source not in {"train", "one_to_one", "none"}:
        raise ValueError("train_anchor_source must be 'train', 'one_to_one', or 'none'")

    rng = np.random.default_rng(seed)

    g_src = dataset.pyg_graphs[gids[0]]
    g_tgt = dataset.pyg_graphs[gids[1]]

    if anchor_source == "test": #只取测试集的锚点对来构建多对多数据集
        anchors = dataset.test_data
    else:
        anchors = torch.cat([dataset.train_data, dataset.test_data], dim=0)
    anchors_np = anchors[:, [gids[0], gids[1]]].cpu().numpy()

    # Endpoints that will serve as training supervision must never become
    # evaluation entities. Original datasets can contain duplicate anchor rows
    # or nodes shared across anchor pairs (e.g. flickr-lastfm carries the pair
    # (4227, 11939) twice); the pair-level train/test split then puts copies of
    # the same node on both sides, and building an entity from the test copy
    # leaks a training pair into the ground truth.
    excluded_src: set = set()
    excluded_tgt: set = set()
    if train_anchor_source == "train":
        train_np = dataset.train_data[:, [gids[0], gids[1]]].cpu().numpy()
        excluded_src = {int(u) for u, _ in train_np}
        excluded_tgt = {int(v) for _, v in train_np}

    # De-duplicate anchors per side. If the same node appears in multiple
    # anchor pairs we must keep it in only one entity to avoid ambiguous
    # splits; later the overlap mechanism handles planned reuse.
    seen_src: set = set()
    seen_tgt: set = set()
    dedup_pairs: List[Tuple[int, int]] = []
    dropped_train_collisions = 0
    for u, v in anchors_np:
        u, v = int(u), int(v)
        if u in excluded_src or v in excluded_tgt:
            dropped_train_collisions += 1
            continue
        if u in seen_src or v in seen_tgt:
            continue
        seen_src.add(u)
        seen_tgt.add(v)
        dedup_pairs.append((u, v))

    num_entities = len(dedup_pairs)
    if num_entities == 0:
        raise ValueError("No usable anchors to build a many-to-many benchmark.")

    type_labels = _assign_entity_types(num_entities, split_ratios, rng)

    # Pick split factor (k) for every anchor on each side.
    src_splits: Dict[int, int] = {}
    tgt_splits: Dict[int, int] = {}
    entity_specs: List[Tuple[str, int, int, str]] = []
    for i, (u, v) in enumerate(dedup_pairs):
        kind = type_labels[i]
        k_src, k_tgt = _split_count(kind, max_expansion, rng)
        src_splits[u] = k_src
        tgt_splits[v] = k_tgt
        entity_specs.append((f"e{i}", u, v, kind))

    # Build the two expanded graphs.
    new_src, group_src = _build_side(g_src, src_splits, internal_density, rng)
    new_tgt, group_tgt = _build_side(g_tgt, tgt_splits, internal_density, rng)

    # Assemble entities with the final node id lists.
    entities: EntityMap = {}
    counts = {"1-1": 0, "1-many": 0, "many-1": 0, "many-many": 0}
    for eid, u, v, kind in entity_specs:
        entities[eid] = {
            "src": list(map(int, group_src[u])),
            "tgt": list(map(int, group_tgt[v])),
        }
        counts[kind] += 1

    overlap_inserted = _apply_overlap(
        entities,
        overlap_ratio,
        total_nodes_by_side={"src": new_src.num_nodes, "tgt": new_tgt.num_nodes},
        rng=rng,
    )

    # Decide what to put in the legacy anchor_links field.
    if train_anchor_source == "train":
        train_anchors = dataset.train_data[:, [gids[0], gids[1]]].clone()
    elif train_anchor_source == "one_to_one":
        one_to_one = [(u, v) for (eid, u, v, kind) in entity_specs if kind == "1-1"]
        train_anchors = (
            torch.tensor(one_to_one, dtype=torch.long)
            if one_to_one
            else torch.zeros((0, 2), dtype=torch.long)
        )
    else:
        train_anchors = torch.zeros((0, 2), dtype=torch.long)

    # Hard post-condition: supervision and evaluation must be node-disjoint.
    # (Scoped to "train" mode; "one_to_one" reuses entity nodes by design and
    # is unsuitable for blind evaluation.) This makes the flickr-lastfm class
    # of leakage impossible to regress silently.
    if train_anchor_source == "train" and train_anchors.numel():
        ta_src = {int(x) for x in train_anchors[:, 0].tolist()}
        ta_tgt = {int(y) for y in train_anchors[:, 1].tolist()}
        for eid, item in entities.items():
            bad_s = [n for n in item["src"] if int(n) in ta_src]
            bad_t = [n for n in item["tgt"] if int(n) in ta_tgt]
            if bad_s or bad_t:
                raise ValueError(
                    f"train/test leakage: entity {eid} shares nodes with train "
                    f"anchors (src={bad_s}, tgt={bad_t}); generator invariant violated")

    metadata = {
        "source_dataset": getattr(dataset, "name", "unknown"),
        "split_ratios": split_ratios,
        "max_expansion": int(max_expansion),
        "internal_density": float(internal_density),
        "overlap_ratio": float(overlap_ratio),
        "num_entities": int(num_entities),
        "entity_type_counts": counts,
        "overlap_insertions": int(overlap_inserted),
        "dropped_train_collisions": int(dropped_train_collisions),
        "seed": int(seed),
        "orig_num_nodes": [int(g_src.num_nodes), int(g_tgt.num_nodes)],
        "new_num_nodes": [int(new_src.num_nodes), int(new_tgt.num_nodes)],
    }

    name = name or f"{getattr(dataset, 'name', 'dataset')}_m2m"
    return ManyToManyBenchmark(
        graphs=[new_src, new_tgt],
        entities=entities,
        train_anchors=train_anchors,
        metadata=metadata,
        name=name,
        graph_names=(
            getattr(g_src, "name", "src"),
            getattr(g_tgt, "name", "tgt"),
        ),
    )


__all__ = [
    "ManyToManyBenchmark",
    "DEFAULT_SPLIT_RATIOS",
    "build_many_to_many_benchmark",
]
