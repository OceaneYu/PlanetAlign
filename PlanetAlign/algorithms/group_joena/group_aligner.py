from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from PlanetAlign.algorithms.joena.utils import sinkhorn_stable

from .types import GroupAlignmentResult, QuotientGraph


class GroupAligner(nn.Module):
    """Align quotient graphs with a weighted fused Gromov-Wasserstein objective.

    The feature term is ``<M, T>`` and the structure term is the squared-loss
    GW objective over quotient adjacencies. ``alpha`` is converted to JOENA's
    positive GW/feature weight ratio ``alpha / (1 - alpha) * sqrt(k)``.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        gamma_p: float = 1e-2,
        in_iter: int = 5,
        out_iter: int = 10,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if not 0 <= alpha < 1:
            raise ValueError("alpha must be in [0, 1)")
        if gamma_p <= 0:
            raise ValueError("gamma_p must be positive")
        self.alpha = float(alpha)
        self.gamma_p = float(gamma_p)
        self.in_iter = int(in_iter)
        self.out_iter = int(out_iter)
        self.eps = float(eps)
        self.dtype = dtype

    def forward(self, src: QuotientGraph, tgt: QuotientGraph) -> GroupAlignmentResult:
        src_idx = torch.where(src.valid_mask)[0]
        tgt_idx = torch.where(tgt.valid_mask)[0]
        if src_idx.numel() == 0 or tgt_idx.numel() == 0:
            raise ValueError("GroupAligner received no valid source or target groups")

        z_src = src.embeddings[src_idx]
        z_tgt = tgt.embeddings[tgt_idx]
        a_src = src.adjacency[src_idx][:, src_idx]
        a_tgt = tgt.adjacency[tgt_idx][:, tgt_idx]

        feature_cost = 1.0 - (F.normalize(z_src, p=2, dim=1) @ F.normalize(z_tgt, p=2, dim=1).T)
        feature_cost = feature_cost.clamp_min(0.0).to(self.dtype)
        gw_weight = self.alpha / max(1.0 - self.alpha, self.eps) * min(src_idx.numel(), tgt_idx.numel()) ** 0.5

        transport_valid = sinkhorn_stable(
            feature_cost,
            a_src.to(self.dtype),
            a_tgt.to(self.dtype),
            gw_weight=gw_weight,
            gamma_p=self.gamma_p,
            threshold_lambda=0.0,
            in_iter=self.in_iter,
            out_iter=self.out_iter,
            dtype=self.dtype,
            device=feature_cost.device,
        )
        transport_valid = torch.nan_to_num(transport_valid, nan=0.0, posinf=0.0, neginf=0.0)
        transport_valid = transport_valid / transport_valid.sum().clamp_min(self.eps)

        full_transport = torch.zeros(
            src.embeddings.shape[0],
            tgt.embeddings.shape[0],
            dtype=self.dtype,
            device=feature_cost.device,
        )
        full_transport[src_idx.unsqueeze(1), tgt_idx.unsqueeze(0)] = transport_valid

        structure_loss = self._gw_loss(a_src.to(self.dtype), a_tgt.to(self.dtype), transport_valid)
        alignment_loss = torch.sum(feature_cost * transport_valid) + gw_weight * structure_loss
        return GroupAlignmentResult(
            transport=full_transport,
            feature_cost=feature_cost,
            alignment_loss=alignment_loss,
            structure_loss=structure_loss,
            valid_source=src_idx,
            valid_target=tgt_idx,
        )

    def _gw_loss(self, adj_src: torch.Tensor, adj_tgt: torch.Tensor, transport: torch.Tensor) -> torch.Tensor:
        n_src, n_tgt = transport.shape
        a = transport.sum(dim=1)
        b = transport.sum(dim=0)
        left = (adj_src ** 2) @ a.view(-1, 1) @ torch.ones((1, n_tgt), dtype=transport.dtype, device=transport.device)
        right = torch.ones((n_src, 1), dtype=transport.dtype, device=transport.device) @ b.view(1, -1) @ (adj_tgt ** 2)
        cross = 2.0 * adj_src @ transport @ adj_tgt.T
        return torch.sum((left + right - cross) * transport)
