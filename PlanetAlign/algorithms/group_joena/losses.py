from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

EntityMap = Dict[str, Dict[str, List[int]]]


def assignment_entropy(assignments: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean assignment entropy; lower is sparser in no-overlap mode."""

    if assignments.dim() != 2:
        raise ValueError("assignments must be 2D")
    return -(assignments.clamp_min(eps) * torch.log(assignments.clamp_min(eps))).sum(dim=1).mean()


def internal_cohesion_loss(
    graph: Data,
    assignments: torch.Tensor,
    threshold: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize soft groups whose internal edge density is below threshold."""

    if graph.edge_index.numel() == 0:
        return assignments.new_zeros(())
    src, dst = graph.edge_index[0].to(assignments.device), graph.edge_index[1].to(assignments.device)
    edge_mass = assignments[src] * assignments[dst]
    numerator = edge_mass.sum(dim=0)
    masses = assignments.sum(dim=0)
    self_mass = (assignments * assignments).sum(dim=0)
    denominator = (masses * masses - self_mass).clamp_min(eps)
    density = numerator / denominator
    valid = masses > 1.0 + eps
    if not valid.any():
        return assignments.new_zeros(())
    return F.relu(threshold - density[valid]).mean()


def separation_loss(
    group_embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 0.8,
) -> torch.Tensor:
    """Margin loss to discourage prototype/group embedding collapse."""

    valid = group_embeddings[valid_mask]
    if valid.shape[0] < 2:
        return group_embeddings.new_zeros(())
    norm = F.normalize(valid, p=2, dim=1)
    sim = norm @ norm.T
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return F.relu(sim[mask] - margin).mean()


def supervised_group_balanced_loss(
    scores: torch.Tensor,
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    source_key: str = "src",
    target_key: str = "tgt",
) -> torch.Tensor:
    """Group-normalized supervised alignment loss.

    Each entity contributes approximately equally, regardless of Cartesian
    product size. This is disabled by default because full GT supervision can
    leak test labels in benchmark experiments.
    """

    if scores.dim() != 2:
        raise ValueError("scores must be a 2D node similarity matrix")
    log_prob = torch.log_softmax(scores, dim=1)
    losses = []
    for item in gt_entities.values():
        src_nodes = [int(n) for n in item.get(source_key, []) if 0 <= int(n) < scores.shape[0]]
        tgt_nodes = [int(n) for n in item.get(target_key, []) if 0 <= int(n) < scores.shape[1]]
        if not src_nodes or not tgt_nodes:
            continue
        src_idx = torch.tensor(src_nodes, dtype=torch.long, device=scores.device)
        tgt_idx = torch.tensor(tgt_nodes, dtype=torch.long, device=scores.device)
        losses.append(-log_prob[src_idx][:, tgt_idx].mean())
    if not losses:
        return scores.new_zeros(())
    return torch.stack(losses).mean()


def reconstruction_loss(
    graph: Data,
    assignments: torch.Tensor,
    group_adjacency: torch.Tensor,
    neg_ratio: float = 1.0,
    seed: int = 42,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sparse Bernoulli-Poisson reconstruction loss with negative sampling."""

    if neg_ratio <= 0 or graph.edge_index.numel() == 0:
        return assignments.new_zeros(())
    device = assignments.device
    src = graph.edge_index[0].to(device)
    dst = graph.edge_index[1].to(device)
    pos_rate = (assignments[src] @ group_adjacency * assignments[dst]).sum(dim=1).clamp_min(eps)
    pos_prob = (1.0 - torch.exp(-pos_rate)).clamp(min=eps, max=1.0 - eps)
    pos_loss = -torch.log(pos_prob).mean()

    num_neg = max(1, int(src.numel() * neg_ratio))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    neg_src = torch.randint(0, graph.num_nodes, (num_neg,), generator=gen, device="cpu").to(device)
    neg_dst = torch.randint(0, graph.num_nodes, (num_neg,), generator=gen, device="cpu").to(device)
    neg_rate = (assignments[neg_src] @ group_adjacency * assignments[neg_dst]).sum(dim=1).clamp_min(eps)
    neg_prob = (1.0 - torch.exp(-neg_rate)).clamp(min=eps, max=1.0 - eps)
    neg_loss = -torch.log(1.0 - neg_prob).mean()
    return pos_loss + neg_loss
