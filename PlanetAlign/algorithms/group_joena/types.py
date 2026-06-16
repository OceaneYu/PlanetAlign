from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


@dataclass
class QuotientGraph:
    """Group-level graph induced by a node-to-group assignment."""

    embeddings: torch.Tensor
    adjacency: torch.Tensor
    masses: torch.Tensor
    valid_mask: torch.Tensor
    assignments: torch.Tensor
    debug: Dict[str, float] = field(default_factory=dict)


@dataclass
class GroupAlignmentResult:
    """Result of aligning source and target quotient graphs."""

    transport: torch.Tensor
    feature_cost: torch.Tensor
    alignment_loss: torch.Tensor
    structure_loss: torch.Tensor
    valid_source: torch.Tensor
    valid_target: torch.Tensor


@dataclass
class GroupAlignmentPrediction:
    """Group-mediated prediction object retained for analysis."""

    source_groups: List[List[int]]
    target_groups: List[List[int]]
    group_pairs: List[Tuple[int, int]]
    source_assignments: torch.Tensor
    target_assignments: torch.Tensor
    group_alignment: torch.Tensor
    node_alignment_scores: torch.Tensor
    entities: Dict[str, Dict[str, List[int]]]
