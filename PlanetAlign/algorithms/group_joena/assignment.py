from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation without extra dependencies."""

    shifted = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(shifted, dim=dim, descending=True).values
    range_shape = [1] * shifted.dim()
    range_shape[dim] = shifted.shape[dim]
    k = torch.arange(1, shifted.shape[dim] + 1, device=shifted.device, dtype=shifted.dtype).view(range_shape)
    cumsum = zs.cumsum(dim)
    support = 1 + k * zs > cumsum
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cumsum.gather(dim, support_size.long() - 1) - 1) / support_size.to(shifted.dtype)
    return torch.clamp(shifted - tau, min=0)


class SoftGroupAssignment(nn.Module):
    """Temperature-controlled node-to-group assignment via learned prototypes."""

    def __init__(
        self,
        num_groups: int,
        embedding_dim: int,
        method: str = "softmax",
        temperature: float = 0.5,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if num_groups < 1:
            raise ValueError("num_groups must be positive")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if method not in {"softmax", "sparsemax"}:
            raise ValueError("method must be 'softmax' or 'sparsemax'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.num_groups = int(num_groups)
        self.embedding_dim = int(embedding_dim)
        self.method = method
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.prototypes = nn.Parameter(torch.empty(num_groups, embedding_dim, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        init = torch.randn(self.num_groups, self.embedding_dim, generator=generator, dtype=self.prototypes.dtype)
        with torch.no_grad():
            self.prototypes.copy_(F.normalize(init, p=2, dim=1))

    def reset_from_embeddings(self, embeddings: torch.Tensor) -> None:
        """Initialize prototypes from evenly spaced nodes for reproducibility."""

        if embeddings.dim() != 2:
            raise ValueError("embeddings must be a 2D tensor")
        n, dim = embeddings.shape
        if dim != self.embedding_dim:
            raise ValueError(f"embedding dim mismatch: got {dim}, expected {self.embedding_dim}")
        if n == 0:
            raise ValueError("cannot initialize assignment prototypes from an empty embedding matrix")
        idx = torch.linspace(0, n - 1, steps=self.num_groups, device=embeddings.device).round().long()
        with torch.no_grad():
            chosen = embeddings[idx].detach().to(self.prototypes.dtype)
            self.prototypes.copy_(F.normalize(chosen, p=2, dim=1))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() != 2:
            raise ValueError("embeddings must be 2D [num_nodes, embedding_dim]")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embedding dim mismatch: got {embeddings.shape[1]}, expected {self.embedding_dim}"
            )
        emb = F.normalize(embeddings, p=2, dim=1)
        proto = F.normalize(self.prototypes, p=2, dim=1)
        logits = emb @ proto.T / self.temperature
        if self.method == "softmax":
            out = torch.softmax(logits, dim=1)
        else:
            out = sparsemax(logits, dim=1)
            row_sum = out.sum(dim=1, keepdim=True).clamp_min(1e-12)
            out = out / row_sum
        return out

    @staticmethod
    def hard_assignments(assignments: torch.Tensor) -> torch.Tensor:
        if assignments.dim() != 2:
            raise ValueError("assignments must be 2D")
        return assignments.argmax(dim=1)
