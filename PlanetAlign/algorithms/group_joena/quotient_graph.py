from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .types import QuotientGraph


class QuotientGraphBuilder(nn.Module):
    """Build a split-invariant group-level graph from soft assignments."""

    def __init__(
        self,
        aggregation: str = "soft_or",
        soft_or_gamma: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if aggregation not in {"soft_or", "normalized_sum"}:
            raise ValueError("aggregation must be 'soft_or' or 'normalized_sum'")
        if soft_or_gamma <= 0:
            raise ValueError("soft_or_gamma must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.aggregation = aggregation
        self.soft_or_gamma = float(soft_or_gamma)
        self.eps = float(eps)

    def forward(self, graph: Data, embeddings: torch.Tensor, assignments: torch.Tensor) -> QuotientGraph:
        if embeddings.dim() != 2:
            raise ValueError("embeddings must be 2D")
        if assignments.dim() != 2:
            raise ValueError("assignments must be 2D")
        if embeddings.shape[0] != graph.num_nodes or assignments.shape[0] != graph.num_nodes:
            raise ValueError(
                "graph.num_nodes, embeddings rows, and assignment rows must match: "
                f"{graph.num_nodes}, {embeddings.shape[0]}, {assignments.shape[0]}"
            )

        masses = assignments.sum(dim=0)
        valid_mask = masses > self.eps
        denom = masses.clamp_min(self.eps).unsqueeze(1)
        group_embeddings = assignments.T @ embeddings / denom
        raw_adj = self._soft_edge_counts(graph, assignments)
        raw_adj = raw_adj - torch.diag(torch.diag(raw_adj))

        if self.aggregation == "soft_or":
            group_adj = 1.0 - torch.exp(-self.soft_or_gamma * raw_adj)
        else:
            norm = masses.view(-1, 1) * masses.view(1, -1)
            group_adj = raw_adj / norm.clamp_min(self.eps)
        group_adj = torch.clamp(group_adj, min=0.0, max=1.0)
        group_adj = group_adj * valid_mask.view(-1, 1).to(group_adj.dtype)
        group_adj = group_adj * valid_mask.view(1, -1).to(group_adj.dtype)

        debug: Dict[str, float] = {
            "num_valid_groups": float(valid_mask.sum().item()),
            "mean_group_mass": float(masses.mean().detach().cpu().item()),
            "max_group_mass": float(masses.max().detach().cpu().item()) if masses.numel() else 0.0,
        }
        return QuotientGraph(
            embeddings=group_embeddings,
            adjacency=group_adj,
            masses=masses,
            valid_mask=valid_mask,
            assignments=assignments,
            debug=debug,
        )

    def _soft_edge_counts(self, graph: Data, assignments: torch.Tensor) -> torch.Tensor:
        num_nodes = int(graph.num_nodes)
        num_groups = assignments.shape[1]
        if graph.edge_index.numel() == 0:
            return torch.zeros(num_groups, num_groups, dtype=assignments.dtype, device=assignments.device)

        edge_index = graph.edge_index.to(assignments.device).long()
        values = torch.ones(edge_index.shape[1], dtype=assignments.dtype, device=assignments.device)
        adj = torch.sparse_coo_tensor(
            edge_index,
            values,
            size=(num_nodes, num_nodes),
            dtype=assignments.dtype,
            device=assignments.device,
        ).coalesce()
        au = torch.sparse.mm(adj, assignments)
        return assignments.T @ au
