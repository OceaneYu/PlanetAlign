"""Many-to-many decoding on top of a node-level aligner.

This module holds compositions of *model + decoder*: an alignment model (e.g.
:class:`PlanetAlign.algorithms.JOENA`) that produces a node-level similarity
matrix ``S``, followed by the blind group-decoding in
:mod:`PlanetAlign.m2m_blind`. These are deliberately **not** in
:mod:`PlanetAlign.algorithms`, which is reserved for standalone alignment
algorithms — ``JOENAGroupDecode`` adds no new representation learning, it only
reuses JOENA's ``S`` and decodes groups from it.

The decoding rationale (validated empirically):

- A group-cohesion *embedding* regularizer on JOENA is a no-op: with attributes
  the encoder already collapses group members, and the bottleneck is the readout.
- *Attribute/embedding* clustering for group discovery over-merges on weak-
  attribute graphs (e.g. airport, attr_dim=4): unrelated same-attribute nodes
  get chained into giant blobs, and the base rate of within- vs cross-group
  adjacent pairs is ~25:1 against.
- But the **cross-graph alignment profile** separates them cleanly: group-mates
  map to the same region of the other graph, so their rows of the similarity
  matrix ``S`` are near-identical (cosine ~0.4 within-group) while non-group-
  mates — even attribute-identical, adjacent ones — map elsewhere (cosine ~0.0).

So the decoder discovers groups from ``S`` itself, not from attributes:

1. Run JOENA to obtain a node-level similarity matrix ``S`` (n1 x n2).
2. Discover source groups by union-find over *adjacent* node pairs whose S-row
   profiles are similar (``row_tau``); target groups symmetrically from S-columns
   (``col_tau``).
3. Match each source group to its best target group by pooled similarity to
   produce a many-to-many entity map.

The prediction path never consults the ground-truth entity map, so this is
designed for the blind protocol in :mod:`PlanetAlign.m2m_blind` rather than the
group/size-leaking adapter in :mod:`PlanetAlign.m2m`.
"""

from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch

from PlanetAlign.data import Dataset
from PlanetAlign.algorithms.base_model import BaseModel
from PlanetAlign.algorithms.joena import JOENA
from PlanetAlign.m2m_blind import (
    decode_entity_map,
    discover_groups_by_profile,
    discover_groups,
)
from PlanetAlign.m2m import align_prediction_to_ground_truth, EntityMap
from PlanetAlign.metrics import many_to_many_scores


class JOENAGroupDecode(BaseModel):
    """Joint group decoding on top of JOENA's alignment.

    Parameters
    ----------
    row_tau, col_tau : float
        Profile-cosine thresholds for merging adjacent source / target nodes
        into a group. Higher values give tighter groups (better MSF1/MicroF1,
        more over-segmentation / lower ACS). Default 0.1.
    match_threshold : float
        Absolute pooled-similarity cutoff below which a source group is left
        with an empty target side. Default 0.0.
    relative_match_threshold : float
        Pooled-similarity cutoff as a fraction of the global maximum. Default 0.0.
    group_source : {"profile", "attr"}
        How groups are discovered. ``"profile"`` (default) uses the S-row/S-col
        alignment profiles — robust on weak-attribute graphs. ``"attr"`` falls
        back to attribute/structure cohesion (provided for comparison).
    alpha, gamma_p, init_lambda, hid_dim, out_dim, lr
        Forwarded to the internal :class:`JOENA`.
    """

    def __init__(self,
                 row_tau: float = 0.1,
                 col_tau: float = 0.1,
                 match_threshold: float = 0.0,
                 relative_match_threshold: float = 0.0,
                 group_source: str = "profile",
                 alpha: float = 0.7,
                 gamma_p: float = 1e-2,
                 init_lambda: float = 1.0,
                 hid_dim: int = 128,
                 out_dim: int = 128,
                 lr: float = 1e-4,
                 dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        assert 0.0 <= row_tau <= 1.0 and 0.0 <= col_tau <= 1.0
        assert group_source in {"profile", "attr"}

        self.row_tau = row_tau
        self.col_tau = col_tau
        self.match_threshold = match_threshold
        self.relative_match_threshold = relative_match_threshold
        self.group_source = group_source
        self._joena_kwargs = dict(alpha=alpha, gamma_p=gamma_p, init_lambda=init_lambda,
                                  hid_dim=hid_dim, out_dim=out_dim, lr=lr)

        # Populated by train().
        self._graph_src = None
        self._graph_tgt = None
        self.src_groups: Optional[List[List[int]]] = None
        self.tgt_groups: Optional[List[List[int]]] = None

    # ------------------------------------------------------------------
    def train(self,
              dataset: Dataset,
              gids: Union[Tuple[int, int], List[int]],
              use_attr: bool = True,
              total_epochs: int = 100,
              save_log: bool = True,
              verbose: bool = True):
        """Train the internal JOENA aligner and cache the graphs for decoding."""
        gid1, gid2 = gids
        joena = JOENA(dtype=self.dtype, **self._joena_kwargs).to(self.device)
        S, logger = joena.train(dataset=dataset, gids=gids, use_attr=use_attr,
                                total_epochs=total_epochs, save_log=save_log, verbose=verbose)
        self.S = S.detach().to(self.dtype)
        self._graph_src = dataset.pyg_graphs[gid1]
        self._graph_tgt = dataset.pyg_graphs[gid2]
        self._use_attr = use_attr
        return self.S, logger

    # ------------------------------------------------------------------
    def _discover(self) -> Tuple[List[List[int]], List[List[int]]]:
        if self.S is None:
            raise RuntimeError("Model is not trained yet, call train() first")
        S = self.S.detach().to(torch.float32).cpu()
        if self.group_source == "profile":
            src_groups = discover_groups_by_profile(S, self._graph_src, tau=self.row_tau)
            tgt_groups = discover_groups_by_profile(S.T.contiguous(), self._graph_tgt, tau=self.col_tau)
        else:
            src_groups = discover_groups(self._graph_src, use_attr=self._use_attr)
            tgt_groups = discover_groups(self._graph_tgt, use_attr=self._use_attr)
        self.src_groups, self.tgt_groups = src_groups, tgt_groups
        return src_groups, tgt_groups

    def predict_entities(self) -> EntityMap:
        """Produce a many-to-many entity map from ``S`` alone (no ground truth)."""
        src_groups, tgt_groups = self._discover()
        S = self.S.detach().to(torch.float32).cpu()
        return decode_entity_map(
            S, src_groups, tgt_groups,
            threshold=self.match_threshold,
            relative_threshold=self.relative_match_threshold,
        )

    def test_blind(self,
                   gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
                   metrics: Optional[Iterable[str]] = None) -> Dict[str, float]:
        """Decode an entity map and score it under the blind M2M protocol."""
        pred = self.predict_entities()
        aligned = align_prediction_to_ground_truth(gt_entities, pred)
        return many_to_many_scores(gt_entities, aligned, metrics=metrics)
