from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from .assignment import SoftGroupAssignment
from .types import GroupAlignmentPrediction

EntityMap = Dict[str, Dict[str, List[int]]]


def hard_groups(assignments: torch.Tensor) -> List[List[int]]:
    """Convert no-overlap soft assignments into hard node groups."""

    hard = SoftGroupAssignment.hard_assignments(assignments)
    groups: List[List[int]] = [[] for _ in range(assignments.shape[1])]
    for node, gid in enumerate(hard.detach().cpu().tolist()):
        groups[int(gid)].append(int(node))
    return groups


def node_scores_from_groups(src_assignments: torch.Tensor, transport: torch.Tensor, tgt_assignments: torch.Tensor) -> torch.Tensor:
    """Compute S = U_s @ T @ U_t.T."""

    if src_assignments.dim() != 2 or tgt_assignments.dim() != 2 or transport.dim() != 2:
        raise ValueError("assignments and transport must be 2D")
    if src_assignments.shape[1] != transport.shape[0] or tgt_assignments.shape[1] != transport.shape[1]:
        raise ValueError(
            "shape mismatch for S = U_s @ T @ U_t.T: "
            f"{tuple(src_assignments.shape)}, {tuple(transport.shape)}, {tuple(tgt_assignments.shape)}"
        )
    return src_assignments @ transport @ tgt_assignments.T


def group_prediction_from_alignment(
    src_assignments: torch.Tensor,
    tgt_assignments: torch.Tensor,
    transport: torch.Tensor,
    min_transport: Optional[float] = None,
    source_key: str = "src",
    target_key: str = "tgt",
) -> GroupAlignmentPrediction:
    """Build a GT-free prediction from hard source/target groups and ``T``.

    Each non-empty source group is matched to the strongest target group in its
    transport row. This is the model's inference output. Benchmark labels are
    intentionally not accepted here.
    """

    src_groups = hard_groups(src_assignments)
    tgt_groups = hard_groups(tgt_assignments)
    transport_cpu = transport.detach().cpu()
    scores = node_scores_from_groups(src_assignments, transport, tgt_assignments).detach().cpu()

    entities: EntityMap = {}
    group_pairs: List[Tuple[int, int]] = []
    for sgid, src_nodes in enumerate(src_groups):
        if not src_nodes or transport_cpu.shape[1] == 0:
            continue
        tgid = int(torch.argmax(transport_cpu[sgid]).item())
        value = float(transport_cpu[sgid, tgid].item())
        if min_transport is not None and value < min_transport:
            continue
        tgt_nodes = tgt_groups[tgid] if 0 <= tgid < len(tgt_groups) else []
        eid = f"g{sgid}"
        entities[eid] = {
            source_key: [int(n) for n in src_nodes],
            target_key: [int(n) for n in tgt_nodes],
        }
        group_pairs.append((int(sgid), int(tgid)))

    return GroupAlignmentPrediction(
        source_groups=src_groups,
        target_groups=tgt_groups,
        group_pairs=group_pairs,
        source_assignments=src_assignments.detach().cpu(),
        target_assignments=tgt_assignments.detach().cpu(),
        group_alignment=transport.detach().cpu(),
        node_alignment_scores=scores,
        entities=entities,
    )


def predict_entities_from_group_alignment(
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    src_assignments: torch.Tensor,
    tgt_assignments: torch.Tensor,
    transport: torch.Tensor,
    target_size_mode: str = "gt",
    source_key: str = "src",
    target_key: str = "tgt",
) -> GroupAlignmentPrediction:
    """Build benchmark-keyed M2M predictions from discovered groups and ``T``.

    This is an evaluation adapter, not the model's GT-free inference API. It
    uses benchmark entities only to choose query source nodes and output entity
    ids. ``target_size_mode='gt'`` additionally uses the GT target-set size for
    top-k formatting; ``target_size_mode='group'`` avoids that size oracle.
    """

    if target_size_mode not in {"gt", "group"}:
        raise ValueError("target_size_mode must be 'gt' or 'group'")

    src_groups = hard_groups(src_assignments)
    tgt_groups = hard_groups(tgt_assignments)
    src_hard = src_assignments.argmax(dim=1).detach().cpu()
    tgt_hard = tgt_assignments.argmax(dim=1).detach().cpu()
    scores = node_scores_from_groups(src_assignments, transport, tgt_assignments).detach().cpu()
    transport_cpu = transport.detach().cpu()

    entities: EntityMap = {}
    group_pairs_set = set()
    for eid, item in gt_entities.items():
        src_nodes = [int(n) for n in item.get(source_key, []) if 0 <= int(n) < src_assignments.shape[0]]
        gt_tgt_nodes = [int(n) for n in item.get(target_key, []) if 0 <= int(n) < tgt_assignments.shape[0]]
        if not src_nodes:
            entities[str(eid)] = {source_key: [], target_key: []}
            continue

        src_group_ids = sorted({int(src_hard[n].item()) for n in src_nodes})
        matched_tgt_groups = []
        for sgid in src_group_ids:
            if transport_cpu.shape[1] == 0:
                continue
            tgid = int(torch.argmax(transport_cpu[sgid]).item())
            matched_tgt_groups.append(tgid)
            group_pairs_set.add((sgid, tgid))

        candidates = sorted({node for gid in matched_tgt_groups for node in tgt_groups[gid]})
        if target_size_mode == "gt":
            k = len(gt_tgt_nodes)
            if k == 0:
                pred_tgt = []
            elif candidates:
                candidate_scores = []
                src_idx = torch.tensor(src_nodes, dtype=torch.long)
                for node in candidates:
                    candidate_scores.append((float(scores[src_idx, node].mean().item()), node))
                candidate_scores.sort(key=lambda x: x[0], reverse=True)
                pred_tgt = [node for _, node in candidate_scores[:k]]
            else:
                mean_scores = scores[torch.tensor(src_nodes, dtype=torch.long)].mean(dim=0)
                pred_tgt = [int(i) for i in torch.topk(mean_scores, k=min(k, scores.shape[1])).indices.tolist()]
        else:
            pred_tgt = candidates

        entities[str(eid)] = {
            source_key: src_nodes,
            target_key: [int(n) for n in pred_tgt],
        }

    return GroupAlignmentPrediction(
        source_groups=src_groups,
        target_groups=tgt_groups,
        group_pairs=sorted(group_pairs_set),
        source_assignments=src_assignments.detach().cpu(),
        target_assignments=tgt_assignments.detach().cpu(),
        group_alignment=transport.detach().cpu(),
        node_alignment_scores=scores,
        entities=entities,
    )
