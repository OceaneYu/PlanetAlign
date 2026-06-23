"""Blind (non-leaking) many-to-many evaluation.

The default M2M adapter in :mod:`PlanetAlign.m2m`
(``similarity_to_pred_entities``) hands the method two pieces of ground truth:

1. the **source-side grouping** (it iterates ``gt_entities`` and uses each GT
   entity's source nodes as the query set), and
2. the **target group size** (``top_k`` defaults to ``len(gt target set)``).

Under that setup the task collapses to node ranking, so any innovation in group
discovery is invisible and a strong 1-1 method like JOENA already looks good.
This module provides a *blind* alternative: the method only sees a node-level
similarity matrix and the two graphs, must **discover groups on both sides
itself** and **match them itself**, and is scored on the resulting entity map.

Group discovery uses only observable graph signal (attribute equality on
adjacent nodes, or neighbourhood overlap), never the ground-truth entity map, so
it is a legitimate honest baseline rather than an oracle.

Pipeline
--------
``blind_predict(S, g_src, g_tgt)`` ->
    1. ``discover_groups`` on each graph (union-find over cohesion edges);
    2. ``decode_entity_map`` pools S over the discovered groups and matches each
       source group to its best target group above a threshold.

``evaluate_blind(S, gt, g_src, g_tgt)`` runs the pipeline and scores the
prediction with the standard M2M metrics. Nothing from ``gt`` enters the
prediction path.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from PlanetAlign.metrics import many_to_many_scores
from PlanetAlign.m2m import align_prediction_to_ground_truth, EntityMap


# ---------------------------------------------------------------------------
# Group discovery (observable signal only)
# ---------------------------------------------------------------------------
def _cohesion_pairs(
    graph,
    use_attr: bool,
    attr_tau: float,
    struct_tau: float,
) -> List[Tuple[int, int]]:
    """Within-group candidate edges from observable structure.

    Attribute mode: adjacent endpoints whose raw attributes are near-identical
    (split group-mates share attributes verbatim). Structural fallback: adjacent
    endpoints with high neighbourhood Jaccard overlap.
    """
    ei = graph.edge_index
    if ei.numel() == 0:
        return []
    src, dst = ei[0], ei[1]
    keep = src < dst
    src, dst = src[keep], dst[keep]
    if src.numel() == 0:
        return []

    if use_attr and graph.x is not None:
        xn = F.normalize(graph.x.to(torch.float32), p=2, dim=1)
        sim = (xn[src] * xn[dst]).sum(dim=1)
        mask = sim >= attr_tau
        s_keep, d_keep = src[mask].tolist(), dst[mask].tolist()
        return list(zip(s_keep, d_keep))

    # Structural fallback: neighbourhood Jaccard.
    num_nodes = graph.num_nodes
    neigh: List[set] = [set() for _ in range(num_nodes)]
    e_src, e_dst = graph.edge_index[0].tolist(), graph.edge_index[1].tolist()
    for u, v in zip(e_src, e_dst):
        neigh[u].add(v)
        neigh[v].add(u)
    pairs: List[Tuple[int, int]] = []
    for u, v in zip(src.tolist(), dst.tolist()):
        nu, nv = neigh[u] - {v}, neigh[v] - {u}
        if not nu and not nv:
            continue
        inter = len(nu & nv)
        union = len(nu | nv)
        if union > 0 and inter / union >= struct_tau:
            pairs.append((u, v))
    return pairs


def _components(num_nodes: int, pairs: Iterable[Tuple[int, int]]) -> List[List[int]]:
    """Union-find connected components; isolated nodes become singletons."""
    parent = list(range(num_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in pairs:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv

    groups: Dict[int, List[int]] = {}
    for n in range(num_nodes):
        groups.setdefault(find(n), []).append(n)
    return list(groups.values())


def discover_groups(
    graph,
    use_attr: bool = True,
    attr_tau: float = 0.99,
    struct_tau: float = 0.5,
) -> List[List[int]]:
    """Partition a graph's nodes into groups via union-find over cohesion edges.

    Cohesion edges come from observable structure/attributes only. Every node
    belongs to exactly one returned group; nodes with no cohesion edge form
    singleton groups.

    Note: attribute equality over-merges on low-dimensional attributes (it
    chains unrelated same-attribute nodes). For weak-attribute graphs prefer
    :func:`discover_groups_by_profile`, which uses the cross-graph alignment.
    """
    return _components(int(graph.num_nodes), _cohesion_pairs(graph, use_attr, attr_tau, struct_tau))


def discover_groups_by_profile(
    profiles: torch.Tensor,
    graph,
    tau: float = 0.1,
) -> List[List[int]]:
    """Partition nodes into groups via their cross-graph alignment profiles.

    ``profiles`` is a ``[num_nodes, d]`` tensor of per-node alignment vectors —
    rows of the similarity matrix for the source side, columns (i.e. rows of
    ``S.T``) for the target side. Two adjacent nodes are merged when the cosine
    of their (L2-normalized) profiles is ``>= tau``.

    Group-mates map to the same region of the other graph, so their profiles
    are near-identical, while non-group-mates — even attribute-identical,
    adjacent ones — map elsewhere and have near-orthogonal profiles. This makes
    grouping robust on weak-attribute graphs where attribute/structure signals
    over-merge.
    """
    ei = graph.edge_index
    if ei.numel() == 0:
        return _components(int(graph.num_nodes), [])
    src, dst = ei[0], ei[1]
    keep = src < dst
    src, dst = src[keep], dst[keep]

    p = F.normalize(profiles.to(torch.float32), p=2, dim=1)
    sim = (p[src] * p[dst]).sum(dim=1)
    mask = sim >= tau
    pairs = list(zip(src[mask].tolist(), dst[mask].tolist()))
    return _components(int(graph.num_nodes), pairs)


# ---------------------------------------------------------------------------
# Group matching / decoding
# ---------------------------------------------------------------------------
def _group_labels(groups: Sequence[Sequence[int]], num_nodes: int) -> torch.Tensor:
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    for gid, members in enumerate(groups):
        for n in members:
            labels[n] = gid
    return labels


def _pooled_group_scores(
    similarity: torch.Tensor,
    src_groups: Sequence[Sequence[int]],
    tgt_groups: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Mean similarity between every (source group, target group) pair.

    Computed in O(n1 * n2) with two scatter reductions instead of a dense
    group-by-group matmul, so it scales when most groups are singletons.
    """
    n1, n2 = similarity.shape
    g1, g2 = len(src_groups), len(tgt_groups)
    src_lab = _group_labels(src_groups, n1).to(similarity.device)
    tgt_lab = _group_labels(tgt_groups, n2).to(similarity.device)

    # Row-pool: R[a, j] = mean over source members of group a of S[member, j].
    src_size = torch.zeros(g1, device=similarity.device).index_add_(
        0, src_lab, torch.ones(n1, device=similarity.device)
    ).clamp(min=1.0)
    R = torch.zeros(g1, n2, dtype=similarity.dtype, device=similarity.device)
    R.index_add_(0, src_lab, similarity)
    R = R / src_size.unsqueeze(1)

    # Col-pool: T[a, b] = mean over target members of group b of R[a, member].
    tgt_size = torch.zeros(g2, device=similarity.device).index_add_(
        0, tgt_lab, torch.ones(n2, device=similarity.device)
    ).clamp(min=1.0)
    Tt = torch.zeros(g2, g1, dtype=similarity.dtype, device=similarity.device)
    Tt.index_add_(0, tgt_lab, R.T)
    T = (Tt / tgt_size.unsqueeze(1)).T
    return T  # [g1, g2]


def decode_entity_map(
    similarity: torch.Tensor,
    src_groups: Sequence[Sequence[int]],
    tgt_groups: Sequence[Sequence[int]],
    threshold: float = 0.0,
    relative_threshold: float = 0.0,
) -> EntityMap:
    """Match each source group to its best target group from ``similarity``.

    A predicted entity is ``{"src": <source group>, "tgt": <best target group>}``.
    Source groups whose best pooled score is below ``threshold`` (absolute) or
    below ``relative_threshold * global_max`` are emitted with an empty target
    side. Ground-truth groups are never consulted.
    """
    if not src_groups or not tgt_groups:
        return {f"p{i}": {"src": list(map(int, g)), "tgt": []} for i, g in enumerate(src_groups)}

    T = _pooled_group_scores(similarity, src_groups, tgt_groups)
    best_val, best_idx = T.max(dim=1)
    cutoff = max(float(threshold), relative_threshold * float(T.max()))

    pred: EntityMap = {}
    for a, src_members in enumerate(src_groups):
        eid = f"p{a}"
        if float(best_val[a]) <= cutoff:
            pred[eid] = {"src": list(map(int, src_members)), "tgt": []}
        else:
            b = int(best_idx[a])
            pred[eid] = {"src": list(map(int, src_members)), "tgt": list(map(int, tgt_groups[b]))}
    return pred


def blind_predict(
    similarity: torch.Tensor,
    graph_src,
    graph_tgt,
    use_attr: bool = True,
    attr_tau: float = 0.99,
    struct_tau: float = 0.5,
    threshold: float = 0.0,
    relative_threshold: float = 0.0,
) -> EntityMap:
    """Discover groups on both graphs and decode an entity map, without GT."""
    S = similarity.detach().to(torch.float32).cpu()
    src_groups = discover_groups(graph_src, use_attr, attr_tau, struct_tau)
    tgt_groups = discover_groups(graph_tgt, use_attr, attr_tau, struct_tau)
    return decode_entity_map(S, src_groups, tgt_groups, threshold, relative_threshold)


def evaluate_blind(
    similarity: torch.Tensor,
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    graph_src,
    graph_tgt,
    metrics: Optional[Iterable[str]] = None,
    use_attr: bool = True,
    attr_tau: float = 0.99,
    struct_tau: float = 0.5,
    threshold: float = 0.0,
    relative_threshold: float = 0.0,
    return_predictions: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], EntityMap]]:
    """Blind M2M evaluation: the method self-produces groups, then is scored.

    Unlike :func:`PlanetAlign.m2m.evaluate_similarity`, no source grouping or
    target-size oracle is leaked into the prediction.
    """
    pred = blind_predict(
        similarity, graph_src, graph_tgt,
        use_attr=use_attr, attr_tau=attr_tau, struct_tau=struct_tau,
        threshold=threshold, relative_threshold=relative_threshold,
    )
    aligned = align_prediction_to_ground_truth(gt_entities, pred)
    scores = many_to_many_scores(gt_entities, aligned, metrics=metrics)
    if return_predictions:
        return scores, pred
    return scores


__all__ = [
    "discover_groups",
    "discover_groups_by_profile",
    "decode_entity_map",
    "blind_predict",
    "evaluate_blind",
]
