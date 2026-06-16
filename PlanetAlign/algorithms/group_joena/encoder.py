from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

from PlanetAlign.algorithms.joena.model import MLP
from PlanetAlign.utils import get_batch_rwr_scores


def degree_feature(graph: Data, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a simple normalized degree feature for anchor-free fallback."""

    deg = torch.zeros(graph.num_nodes, dtype=dtype, device=device)
    if graph.edge_index.numel() > 0:
        src = graph.edge_index[0].to(device)
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=dtype))
    scale = deg.max().clamp_min(1.0)
    return (deg / scale).view(-1, 1)


def build_node_input(
    graph: Data,
    anchor_nodes: torch.Tensor,
    use_attr: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build the same style of node input used by JOENA.

    JOENA relies on RWR scores from training anchors. We keep that signal. When
    no anchors are available, a normalized degree feature is used as a minimal
    structural fallback so the model can still run in smoke tests.
    """

    pieces = []
    anchor_nodes = anchor_nodes.detach().long().to(device)
    if anchor_nodes.numel() > 0:
        rwr = get_batch_rwr_scores(graph, anchor_nodes, device=device).to(dtype)
        pieces.append(rwr)
    else:
        pieces.append(degree_feature(graph, dtype=dtype, device=device))

    if use_attr:
        if graph.x is None:
            raise ValueError("use_attr=True but graph.x is None")
        if graph.x.shape[0] != graph.num_nodes:
            raise ValueError(
                f"node attribute shape {tuple(graph.x.shape)} does not match num_nodes={graph.num_nodes}"
            )
        pieces.insert(0, graph.x.to(dtype).to(device))

    return torch.cat(pieces, dim=1)


class NodeEncoder(nn.Module):
    """Shared JOENA-style MLP encoder for the two graphs."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dtype: torch.dtype):
        super().__init__()
        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(dtype)

    def forward(self, x_src: torch.Tensor, x_tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_src.dim() != 2 or x_tgt.dim() != 2:
            raise ValueError("NodeEncoder expects 2D source and target input matrices")
        if x_src.shape[1] != x_tgt.shape[1]:
            raise ValueError(
                f"source/target input dimensions differ: {x_src.shape[1]} vs {x_tgt.shape[1]}"
            )
        return self.mlp(x_src, x_tgt)
