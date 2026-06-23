"""JOENAGroup — group-cohesive embedding learning for many-to-many alignment.

This is a thin extension of JOENA (``PlanetAlign.algorithms.joena``).  JOENA
learns node embeddings whose *optimal-transport plan* aligns anchors one-to-one.
On the many-to-many benchmarks produced by
``PlanetAlign.utils.many2many_builder``, the right answer is no longer a single
node but a whole *group* (the split members of an entity).  JOENA's embedding
objective has no notion of groups, so it spreads a fixed transport mass over the
group instead of mapping the source group onto the entire target group.

JOENAGroup keeps JOENA's encoder (``MLP``) and fused Gromov-Wasserstein loss
(``FusedGWLoss``) verbatim and adds one term: an **intra-graph group-cohesion
regularizer** that pulls together the embeddings of nodes that the benchmark
construction makes co-grouped.

Why this works (and why it is not GT leakage)
---------------------------------------------
The builder splits one node into ``k`` members that

1. **share the original node's attributes verbatim** (``extra_x = x[orig]``), and
2. are **densely interconnected** (``internal_density`` high, with a connectivity
   guarantee).

Both signals are observable from the input graph alone — no ground-truth entity
map is consulted.  We therefore mark an edge ``(u, v)`` as a *cohesion edge* when
``u`` and ``v`` are adjacent and either their raw attributes are near-identical
(attribute mode) or they share a large fraction of neighbours (structural
fallback).  Single-member (1-1) entities produce no cohesion edges and are left
untouched, so the regularizer is automatically targeted at the multi-member
groups that the many-to-many metrics care about.

Pulling group members to (almost) the same embedding means that whenever one
member is close to a source group, all of its group-mates are too — so querying
with a source group retrieves the *entire* target group.  That directly targets
MSF1 / MicroF1 / M2M-EGS, the metrics that node-level 1-1 alignment leaves on the
table.

Outputs
-------
``self.S`` is set according to ``output``:

- ``"plan"`` (default): JOENA's transport plan — drop-in comparable to JOENA,
  keeps Hits@1 strong.
- ``"embed"``: cosine similarity of the group-cohesive embeddings.  Not mass
  constrained, so it can give every target group-mate a high score; usually the
  better choice for the many-to-many metrics.
- ``"blend"``: ``blend_beta * plan + (1 - blend_beta) * embed`` after per-matrix
  max-normalization.

Both matrices are always stashed on ``self.plan_S`` / ``self.embed_S`` for
analysis regardless of ``output``.
"""

from typing import List, Optional, Tuple, Union

import os
import time

import torch
import torch.nn.functional as F
import psutil

from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_anchor_pairs, get_batch_rwr_scores
from PlanetAlign.metrics import hits_ks_scores, mrr_score
from PlanetAlign.algorithms.base_model import BaseModel
from PlanetAlign.algorithms.joena.model import MLP, FusedGWLoss


class JOENAGroup(BaseModel):
    """Group-cohesive variant of JOENA.

    Parameters
    ----------
    alpha, gamma_p, init_lambda, hid_dim, out_dim, lr
        Identical to :class:`PlanetAlign.algorithms.JOENA`.
    mu : float, optional
        Weight of the intra-graph group-cohesion loss.  ``0`` recovers plain
        JOENA.  Default 10.0.
    coh_attr_tau : float, optional
        Cohesion threshold in attribute mode: an edge is a cohesion edge when the
        cosine similarity of its endpoints' raw attributes is ``>= coh_attr_tau``.
        Split group-mates share attributes exactly, so a value close to 1 keeps
        precision high.  Default 0.99.
    coh_struct_tau : float, optional
        Cohesion threshold in structural-fallback mode (no attributes): an edge is
        kept when the Jaccard overlap of its endpoints' neighbourhoods is
        ``>= coh_struct_tau``.  Default 0.5.
    output : {"plan", "embed", "blend"}, optional
        Which similarity matrix to expose as ``self.S``.  Default ``"plan"``.
    blend_beta : float, optional
        Mixing weight for ``output="blend"``.  Default 0.5.
    dtype : torch.dtype, optional
        ``torch.float32`` or ``torch.float64``.  Default ``torch.float32``.
    """

    def __init__(self,
                 alpha: float = 0.7,
                 gamma_p: float = 1e-2,
                 init_lambda: float = 1.0,
                 hid_dim: int = 128,
                 out_dim: int = 128,
                 lr: float = 1e-4,
                 mu: float = 10.0,
                 coh_attr_tau: float = 0.99,
                 coh_struct_tau: float = 0.5,
                 output: str = "plan",
                 blend_beta: float = 0.5,
                 dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        assert mu >= 0.0, "mu must be non-negative"
        assert 0.0 <= coh_attr_tau <= 1.0
        assert 0.0 <= coh_struct_tau <= 1.0
        assert output in {"plan", "embed", "blend"}
        assert 0.0 <= blend_beta <= 1.0

        self.alpha = alpha
        self.gamma_p = gamma_p
        self.init_lambda = init_lambda
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.lr = lr
        self.mu = mu
        self.coh_attr_tau = coh_attr_tau
        self.coh_struct_tau = coh_struct_tau
        self.output = output
        self.blend_beta = blend_beta

        # Populated by train().
        self.plan_S: Optional[torch.Tensor] = None
        self.embed_S: Optional[torch.Tensor] = None
        self.emb1: Optional[torch.Tensor] = None
        self.emb2: Optional[torch.Tensor] = None
        self.num_cohesion_edges: Tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    def train(self,
              dataset: Dataset,
              gids: Union[Tuple[int, int], List[int]],
              use_attr: bool = True,
              total_epochs: int = 100,
              save_log: bool = True,
              verbose: bool = True):
        self.check_inputs(dataset, gids, plain_method=False, use_attr=use_attr, pairwise=True, supervised=True)
        gid1, gid2 = gids

        logger = self.init_training_logger(
            dataset, use_attr,
            additional_headers=['memory', 'infer_time', 'coh_loss', 'coh_edges_1', 'coh_edges_2'],
            save_log=save_log,
        )
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        n1, n2 = graph1.num_nodes, graph2.num_nodes
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)

        # --- Input features: identical to JOENA (attributes + RWR). ---
        rwr_t0 = time.time()
        rwr_emb1 = get_batch_rwr_scores(graph1, anchor_links[:, 0], device=self.device).cpu().to(self.dtype)
        rwr_emb2 = get_batch_rwr_scores(graph2, anchor_links[:, 1], device=self.device).cpu().to(self.dtype)
        rwr_time = time.time() - rwr_t0
        if use_attr:
            input_emb1 = torch.cat((graph1.x.to(self.dtype), rwr_emb1), dim=1)
            input_emb2 = torch.cat((graph2.x.to(self.dtype), rwr_emb2), dim=1)
        else:
            input_emb1, input_emb2 = rwr_emb1, rwr_emb2
        input_emb1, input_emb2 = input_emb1.to(self.device), input_emb2.to(self.device)

        # --- Group-cohesion edges (the group prior). Computed once. ---
        coh1 = self._cohesion_edges(graph1, use_attr)
        coh2 = self._cohesion_edges(graph2, use_attr)
        self.num_cohesion_edges = (int(coh1.shape[1]), int(coh2.shape[1]))
        if verbose:
            print(f"  [JOENAGroup] cohesion edges: src={self.num_cohesion_edges[0]} "
                  f"tgt={self.num_cohesion_edges[1]} (mu={self.mu})")

        gw_weight = self.alpha / (1 - self.alpha) * min(n1, n2) ** 0.5

        model = MLP(input_dim=input_emb1.shape[1],
                    hidden_dim=self.hid_dim,
                    output_dim=self.out_dim).to(self.dtype).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = FusedGWLoss(graph1, graph2, gw_weight=gw_weight, gamma_p=self.gamma_p,
                                init_lambda=self.init_lambda, in_iter=5, out_iter=10,
                                dtype=self.dtype).to(self.device)

        S = torch.ones(n1, n2, dtype=self.dtype).to(self.device) / (n1 * n2)
        out1 = out2 = None
        for epoch in range(total_epochs):
            t0 = time.time()
            model.train()
            optimizer.zero_grad()
            ref_t0 = time.time()
            out1, out2 = model(input_emb1, input_emb2)
            ot_loss, S, _ = criterion(out1=out1, out2=out2)
            coh_loss = self._cohesion_loss(out1, coh1) + self._cohesion_loss(out2, coh2)
            loss = ot_loss + self.mu * coh_loss
            refer_time = rwr_time + (time.time() - ref_t0)
            loss.backward()
            optimizer.step()
            t1 = time.time()

            with torch.no_grad():
                model.eval()
                hits, mrr = hits_ks_scores(S, test_pairs, mode='mean'), mrr_score(S, test_pairs, mode='mean')
                mem_gb = process.memory_info().rss / 1024 ** 3
                logger.log(epoch=epoch + 1,
                           loss=loss.item(),
                           epoch_time=t1 - t0,
                           hits=hits,
                           mrr=mrr,
                           memory=round(mem_gb, 4),
                           infer_time=round(refer_time, 4),
                           coh_loss=round(float(coh_loss), 6),
                           coh_edges_1=self.num_cohesion_edges[0],
                           coh_edges_2=self.num_cohesion_edges[1],
                           verbose=verbose)

        # --- Build the two candidate similarity matrices. ---
        with torch.no_grad():
            model.eval()
            out1, out2 = model(input_emb1, input_emb2)
            self.plan_S = S.detach().to(self.dtype).cpu()
            self.embed_S = (out1 @ out2.T).detach().to(self.dtype).cpu()
            self.emb1 = out1.detach().to(self.dtype).cpu()
            self.emb2 = out2.detach().to(self.dtype).cpu()

        self.S = self._select_output().to(self.device)
        return self.S, logger

    # ------------------------------------------------------------------
    def _select_output(self) -> torch.Tensor:
        if self.output == "plan":
            return self.plan_S.clone()
        if self.output == "embed":
            return self.embed_S.clone()
        plan = self._max_normalize(self.plan_S)
        embed = self._max_normalize(self.embed_S)
        return self.blend_beta * plan + (1.0 - self.blend_beta) * embed

    @staticmethod
    def _max_normalize(mat: torch.Tensor) -> torch.Tensor:
        peak = mat.abs().max()
        return mat / peak if peak > 0 else mat

    # ------------------------------------------------------------------
    def _cohesion_edges(self, graph, use_attr: bool) -> torch.Tensor:
        """Return a ``[2, E']`` tensor of within-group candidate edges.

        Attribute mode (preferred): adjacent endpoints whose raw attributes are
        near-identical.  Structural fallback: adjacent endpoints with high
        neighbourhood Jaccard overlap.
        """
        ei = graph.edge_index
        if ei.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        src, dst = ei[0], ei[1]
        keep_dir = src < dst                       # dedup undirected
        src, dst = src[keep_dir], dst[keep_dir]
        if src.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        if use_attr and graph.x is not None:
            xn = F.normalize(graph.x.to(self.dtype), p=2, dim=1)
            sim = (xn[src] * xn[dst]).sum(dim=1)
            keep = sim >= self.coh_attr_tau
        else:
            keep = self._structural_cohesion_mask(graph, src, dst)

        edges = torch.stack([src[keep], dst[keep]], dim=0)
        return edges.to(self.device)

    def _structural_cohesion_mask(self, graph, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        num_nodes = graph.num_nodes
        neigh: List[set] = [set() for _ in range(num_nodes)]
        e_src = graph.edge_index[0].tolist()
        e_dst = graph.edge_index[1].tolist()
        for u, v in zip(e_src, e_dst):
            neigh[u].add(v)
            neigh[v].add(u)
        keep = torch.zeros(src.shape[0], dtype=torch.bool)
        src_l, dst_l = src.tolist(), dst.tolist()
        for i, (u, v) in enumerate(zip(src_l, dst_l)):
            nu, nv = neigh[u] - {v}, neigh[v] - {u}
            if not nu and not nv:
                continue
            inter = len(nu & nv)
            union = len(nu | nv)
            if union > 0 and inter / union >= self.coh_struct_tau:
                keep[i] = True
        return keep

    def _cohesion_loss(self, emb: torch.Tensor, coh_edges: torch.Tensor) -> torch.Tensor:
        """Mean ``1 - cos`` over cohesion edges (embeddings are L2-normalized)."""
        if self.mu == 0.0 or coh_edges.numel() == 0:
            return emb.new_zeros(())
        u = emb[coh_edges[0]]
        v = emb[coh_edges[1]]
        return (1.0 - (u * v).sum(dim=1)).mean()
