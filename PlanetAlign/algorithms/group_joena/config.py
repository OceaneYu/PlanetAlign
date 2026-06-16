from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class GroupJOENAConfig:
    """Configuration for Group-JOENA.

    ``num_groups`` is the preferred way to specify the latent entity count.
    Using the true benchmark entity count must be marked explicitly by passing
    ``oracle_num_groups=True`` to ``fit()``/``train()``. When no group count is
    supplied, the model falls back to the number of one-to-one training anchors
    as a heuristic.
    """

    hidden_dim: int = 128
    out_dim: int = 128
    num_groups: Optional[int] = None
    assignment_method: str = "softmax"
    assignment_temperature: float = 0.5
    quotient_aggregation: str = "soft_or"
    soft_or_gamma: float = 1.0
    cohesion_threshold: float = 0.25
    alignment_weight: float = 1.0
    supervised_weight: float = 0.0
    cohesion_weight: float = 0.1
    reconstruction_weight: float = 0.05
    sparsity_weight: float = 0.01
    separation_weight: float = 0.01
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    eval_interval: int = 5
    seed: int = 42
    device: str = "cpu"
    group_alignment_alpha: float = 0.7
    gamma_p: float = 1e-2
    sinkhorn_in_iter: int = 5
    sinkhorn_out_iter: int = 10
    reconstruction_neg_ratio: float = 1.0
    separation_margin: float = 0.8
    collapse_majority_threshold: float = 0.5
    min_non_empty_group_ratio: float = 0.5
    uniform_row_max_multiplier: float = 1.2
    transport_effective_threshold: float = 1e-3
    score_density_threshold: float = 0.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.out_dim <= 0:
            raise ValueError("hidden_dim and out_dim must be positive")
        if self.num_groups is not None and self.num_groups < 1:
            raise ValueError("num_groups must be None or a positive integer")
        if self.assignment_method not in {"softmax", "sparsemax"}:
            raise ValueError("assignment_method must be 'softmax' or 'sparsemax'")
        if self.assignment_temperature <= 0:
            raise ValueError("assignment_temperature must be positive")
        if self.quotient_aggregation not in {"soft_or", "normalized_sum"}:
            raise ValueError("quotient_aggregation must be 'soft_or' or 'normalized_sum'")
        if self.soft_or_gamma <= 0:
            raise ValueError("soft_or_gamma must be positive")
        if not 0 <= self.cohesion_threshold <= 1:
            raise ValueError("cohesion_threshold must be in [0, 1]")
        for name in (
            "alignment_weight",
            "supervised_weight",
            "cohesion_weight",
            "reconstruction_weight",
            "sparsity_weight",
            "separation_weight",
            "weight_decay",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs < 1 or self.eval_interval < 1:
            raise ValueError("max_epochs and eval_interval must be positive")
        if not 0 <= self.group_alignment_alpha < 1:
            raise ValueError("group_alignment_alpha must be in [0, 1)")
        if self.gamma_p <= 0:
            raise ValueError("gamma_p must be positive")
        if self.sinkhorn_in_iter < 1 or self.sinkhorn_out_iter < 1:
            raise ValueError("sinkhorn iterations must be positive")
        if self.reconstruction_neg_ratio < 0:
            raise ValueError("reconstruction_neg_ratio must be non-negative")
        if not 0 < self.collapse_majority_threshold <= 1:
            raise ValueError("collapse_majority_threshold must be in (0, 1]")
        if not 0 <= self.min_non_empty_group_ratio <= 1:
            raise ValueError("min_non_empty_group_ratio must be in [0, 1]")
        if self.uniform_row_max_multiplier <= 0:
            raise ValueError("uniform_row_max_multiplier must be positive")
        if self.transport_effective_threshold < 0:
            raise ValueError("transport_effective_threshold must be non-negative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
