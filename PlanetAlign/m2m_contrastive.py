"""JOENA-PC: JOENA with symmetry-broken inputs and a profile-contrastive term.

Why this exists (measured chain, see docs/m2m_quotient_align_design.md §5.5):
the readout ceiling on Douban is caused by *globally duplicated coupling rows*
across unrelated nodes (random-pair profile-cosine null q999 = 1.0), which in
turn traces to an input symmetry: JOENA feeds ``[attributes, anchor-RWR]``, and
on Douban 98% of nodes share their exact attribute row with others (buckets up
to 907) while nodes far from all 44 train anchors have near-zero RWR rows —
identical inputs give identical MLP outputs and identical coupling rows
*forever*, so no loss can separate them.

Two coordinated changes:

1. **Symmetry breaking (inputs).** Append observable positional features that
   differ between unrelated attribute-twins but stay coherent for true
   siblings: one-hop mean-propagated attributes (unrelated twins have unrelated
   neighbourhoods; siblings' neighbourhoods are pieces of one original node's)
   and log-degree.
2. **Profile-contrastive term (objective).** On the differentiable surrogate
   profile ``p_i = softmax(out1_i @ out2^T / temp)`` (the Sinkhorn coupling is
   computed under ``no_grad`` in JOENA, but its rows are monotone in these
   logits):
   - *uniformity*: random node pairs' profiles are pushed toward orthogonality
     (mean squared cosine) — random pairs are non-co-referent with probability
     ~1, so this dissolves spurious duplicates (alignment-and-uniformity, Wang
     & Isola 2020);
   - *bootstrap alignment* (optional): adjacent pairs whose *coupling* profile
     cosine is already very high (>= ``pos_bar``, high-precision on the
     benchmarks) are pulled together, reinforcing true-sibling collinearity.

Training path otherwise mirrors ``PlanetAlign.algorithms.JOENA`` (same RWR
features, MLP, FusedGWLoss, epochs, seed) so results are directly comparable.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple, Union

import psutil
import torch
import torch.nn.functional as F

from PlanetAlign.algorithms.base_model import BaseModel
from PlanetAlign.algorithms.joena.model import MLP, FusedGWLoss
from PlanetAlign.data import Dataset
from PlanetAlign.metrics import hits_ks_scores, mrr_score
from PlanetAlign.utils import get_anchor_pairs, get_batch_rwr_scores


def _mean_propagated(graph, x: torch.Tensor) -> torch.Tensor:
    """One-hop mean neighbourhood attributes: D^{-1}(A+I) x (dense, small graphs)."""
    n = int(graph.num_nodes)
    A = torch.zeros(n, n, dtype=x.dtype)
    ei = graph.edge_index
    A[ei[0], ei[1]] = 1.0
    A = ((A + A.T) > 0).to(x.dtype)
    A = A + torch.eye(n, dtype=x.dtype)
    deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (A @ x) / deg


def _log_degree(graph) -> torch.Tensor:
    n = int(graph.num_nodes)
    deg = torch.zeros(n)
    ei = graph.edge_index
    deg.index_add_(0, ei[0], torch.ones(ei.shape[1]))
    d = torch.log1p(deg)
    return (d / d.max().clamp(min=1.0)).unsqueeze(1)


def _adjacent_pairs(graph) -> Tuple[torch.Tensor, torch.Tensor]:
    ei = graph.edge_index
    src, dst = ei[0], ei[1]
    keep = src < dst
    return src[keep], dst[keep]


# ---------------------------------------------------------------------------
# Group-level marginals (experimental): the balanced-marginal tension lives in
# the DATA term — uniform node marginals force each target column to absorb
# exactly 1/n2, so an entity's k source nodes cannot co-concentrate on its m
# target nodes whenever k/n1 > m/n2. With bootstrapped partitions, giving each
# group one unit of mass (split among members, blended with uniform by ``rho``)
# removes the k-vs-m mismatch at the group level.
# ---------------------------------------------------------------------------
def _sinkhorn_marginals(inter_c, intra_c1, intra_c2, a, b, threshold_lambda,
                        in_iter, out_iter, gw_weight, gamma_p, dtype, device):
    """Copy of algorithms.joena.utils.sinkhorn_stable with custom marginals."""
    n1, n2 = inter_c.shape
    f = torch.ones(n1).to(dtype).to(device) / n1
    g = torch.ones(n2).to(dtype).to(device) / n2
    s = torch.ones((n1, n2)).to(dtype).to(device) / (n1 * n2)

    def soft_min_row(z_in, eps):
        hard_min = torch.min(z_in, dim=1, keepdim=True)[0]
        return (hard_min - eps * torch.log(
            torch.sum(torch.exp(-(z_in - hard_min) / eps), dim=1, keepdim=True))).squeeze(-1)

    def soft_min_col(z_in, eps):
        hard_min = torch.min(z_in, dim=0, keepdim=True)[0]
        return (hard_min - eps * torch.log(
            torch.sum(torch.exp(-(z_in - hard_min) / eps), dim=0, keepdim=True))).squeeze(0)

    for _ in range(out_iter):
        a_hat = torch.sum(s - threshold_lambda, dim=1)
        b_hat = torch.sum(s - threshold_lambda, dim=0)
        temp = (intra_c1 ** 2 @ a_hat.view(-1, 1) @ torch.ones((1, n2)).to(dtype).to(device) +
                torch.ones((n1, 1)).to(dtype).to(device) @ b_hat.view(1, -1) @ intra_c2 ** 2)
        L = temp - 2 * intra_c1 @ (s - threshold_lambda) @ intra_c2.T
        Q = inter_c + gw_weight * L
        for _ in range(in_iter):
            f = soft_min_row(Q - g.view(1, -1), gamma_p) + gamma_p * torch.log(a)
            g = soft_min_col(Q - f.view(-1, 1), gamma_p) + gamma_p * torch.log(b)
        s = 0.05 * s + 0.95 * torch.exp((f.view(-1, 1) + g.view(-1, 1).T - Q) / gamma_p)
    return s


class _GroupMarginalFusedGWLoss(FusedGWLoss):
    """FusedGWLoss whose Sinkhorn solve uses externally-set marginals."""

    def set_marginals(self, a: torch.Tensor, b: torch.Tensor) -> None:
        self._a, self._b = a, b

    def forward(self, out1, out2):
        inter_c = torch.exp(-(out1 @ out2.T))
        intra_c1 = torch.exp(-(out1 @ out1.T)) * self.adj1
        intra_c2 = torch.exp(-(out2 @ out2.T)) * self.adj2
        with torch.no_grad():
            s = _sinkhorn_marginals(inter_c, intra_c1, intra_c2, self._a, self._b,
                                    self.threshold_lambda, self.in_iter, self.out_iter,
                                    self.gw_weight, self.gamma_p, self.dtype, self.device)
            self.threshold_lambda = (self.lambda_step
                                     * self._update_lambda(inter_c, intra_c1, intra_c2, s)
                                     + (1 - self.lambda_step) * self.threshold_lambda)
        s_hat = s - self.threshold_lambda
        w_loss = torch.sum(inter_c * s_hat)
        a_m = torch.sum(s_hat, dim=1)
        b_m = torch.sum(s_hat, dim=0)
        gw_loss = torch.sum(
            (intra_c1 ** 2 @ a_m.view(-1, 1) @ torch.ones((1, self.n2)).to(self.dtype).to(self.device) +
             torch.ones((self.n1, 1)).to(self.dtype).to(self.device) @ b_m.view(1, -1) @ intra_c2 ** 2 -
             2 * intra_c1 @ s_hat @ intra_c2.T) * s_hat)
        return w_loss + self.gw_weight * gw_loss + 20, s, self.threshold_lambda


def _group_uniform_marginal(groups, n: int, rho: float, dtype) -> torch.Tensor:
    """Blend of uniform (1-rho) and one-unit-per-group (rho) node masses."""
    m = torch.full((n,), (1.0 - rho) / n, dtype=dtype)
    g = len(groups)
    for members in groups:
        w = rho / (g * len(members))
        for node in members:
            m[node] += w
    return m


class JOENAPC(BaseModel):
    """JOENA + symmetry-broken inputs + profile-contrastive objective.

    Parameters beyond JOENA's: ``break_symmetry`` toggles the positional input
    features; ``lambda_unif`` / ``lambda_align`` weight the uniformity and
    bootstrap-alignment terms (0 disables); ``temp`` is the surrogate-profile
    softmax temperature; ``pos_bar`` the coupling-cosine bar for bootstrap
    positives; ``num_neg_pairs`` the per-epoch random-pair sample size;
    ``warmup_epochs`` delays the bootstrap term until S stabilizes.
    """

    def __init__(self,
                 alpha: float = 0.7,
                 gamma_p: float = 1e-2,
                 init_lambda: float = 1.0,
                 hid_dim: int = 128,
                 out_dim: int = 128,
                 lr: float = 1e-4,
                 break_symmetry: bool = True,
                 lambda_unif: float = 1.0,
                 lambda_align: float = 1.0,
                 temp: float = 0.1,
                 pos_bar: float = 0.95,
                 num_neg_pairs: int = 4096,
                 warmup_epochs: int = 2,
                 group_marginals: bool = False,
                 marginal_rho: float = 0.5,
                 marginal_warmup: int = 3,
                 dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        self.alpha = alpha
        self.gamma_p = gamma_p
        self.init_lambda = init_lambda
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.lr = lr
        self.break_symmetry = break_symmetry
        self.lambda_unif = lambda_unif
        self.lambda_align = lambda_align
        self.temp = temp
        self.pos_bar = pos_bar
        self.num_neg_pairs = num_neg_pairs
        self.warmup_epochs = warmup_epochs
        self.group_marginals = group_marginals
        self.marginal_rho = marginal_rho
        self.marginal_warmup = marginal_warmup

    # ------------------------------------------------------------------
    def _uniformity(self, profiles: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Mean squared cosine over random profile pairs (push to orthogonal)."""
        n = profiles.shape[0]
        m = min(self.num_neg_pairs, n * (n - 1) // 2)
        i = torch.randint(0, n, (m,), generator=gen)
        j = torch.randint(0, n, (m,), generator=gen)
        keep = i != j
        if int(keep.sum()) == 0:
            return profiles.new_zeros(())
        p = F.normalize(profiles, p=2, dim=1)
        cos = (p[i[keep]] * p[j[keep]]).sum(dim=1)
        return (cos ** 2).mean()

    @staticmethod
    def _bootstrap_positives(S: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                             bar: float, cap: int = 8192) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adjacent pairs whose (no-grad) coupling rows already agree strongly."""
        with torch.no_grad():
            p = F.normalize(S.to(torch.float32), p=2, dim=1)
            cos = (p[u] * p[v]).sum(dim=1)
            mask = cos >= bar
        uu, vv = u[mask], v[mask]
        if uu.numel() > cap:
            sel = torch.randperm(uu.numel())[:cap]
            uu, vv = uu[sel], vv[sel]
        return uu, vv

    # ------------------------------------------------------------------
    def train(self,
              dataset: Dataset,
              gids: Union[Tuple[int, int], List[int]],
              use_attr: bool = True,
              total_epochs: int = 100,
              save_log: bool = True,
              verbose: bool = True):
        self.check_inputs(dataset, gids, plain_method=False, use_attr=use_attr,
                          pairwise=True, supervised=True)
        gid1, gid2 = gids
        logger = self.init_training_logger(dataset, use_attr,
                                           additional_headers=['memory', 'infer_time'],
                                           save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        n1, n2 = graph1.num_nodes, graph2.num_nodes
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)

        rwr_t0 = time.time()
        rwr1 = get_batch_rwr_scores(graph1, anchor_links[:, 0], device=self.device).cpu().to(self.dtype)
        rwr2 = get_batch_rwr_scores(graph2, anchor_links[:, 1], device=self.device).cpu().to(self.dtype)
        rwr_time = time.time() - rwr_t0

        feats1: List[torch.Tensor] = [rwr1]
        feats2: List[torch.Tensor] = [rwr2]
        if use_attr:
            x1, x2 = graph1.x.to(self.dtype), graph2.x.to(self.dtype)
            feats1.insert(0, x1)
            feats2.insert(0, x2)
        if self.break_symmetry:
            # Positional features that differ between unrelated attribute-twins.
            if use_attr:
                feats1.append(_mean_propagated(graph1, x1))
                feats2.append(_mean_propagated(graph2, x2))
            feats1.append(_mean_propagated(graph1, rwr1))
            feats2.append(_mean_propagated(graph2, rwr2))
            feats1.append(_log_degree(graph1).to(self.dtype))
            feats2.append(_log_degree(graph2).to(self.dtype))
        input_emb1 = torch.concatenate(feats1, dim=1).to(self.device)
        input_emb2 = torch.concatenate(feats2, dim=1).to(self.device)

        gw_weight = self.alpha / (1 - self.alpha) * min(n1, n2) ** 0.5
        model = MLP(input_dim=input_emb1.shape[1], hidden_dim=self.hid_dim,
                    output_dim=self.out_dim).to(self.dtype).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_cls = _GroupMarginalFusedGWLoss if self.group_marginals else FusedGWLoss
        criterion = loss_cls(graph1, graph2, gw_weight=gw_weight, gamma_p=self.gamma_p,
                             init_lambda=self.init_lambda, in_iter=5, out_iter=10,
                             dtype=self.dtype).to(self.device)
        if self.group_marginals:
            criterion.set_marginals(
                torch.ones(n1, dtype=self.dtype, device=self.device) / n1,
                torch.ones(n2, dtype=self.dtype, device=self.device) / n2)

        u1, v1 = _adjacent_pairs(graph1)
        u2, v2 = _adjacent_pairs(graph2)
        gen = torch.Generator().manual_seed(0)

        S = torch.ones(n1, n2, dtype=self.dtype, device=self.device) / (n1 * n2)
        for epoch in range(total_epochs):
            refer_time = rwr_time
            t0 = time.time()
            model.train()
            optimizer.zero_grad()
            ref_t0 = time.time()
            out1, out2 = model(input_emb1, input_emb2)
            loss, S, _ = criterion(out1=out1, out2=out2)

            # Differentiable surrogate profiles (rows/cols of the logits).
            z = out1 @ out2.T
            if self.lambda_unif > 0:
                p_src = torch.softmax(z / self.temp, dim=1)
                p_tgt = torch.softmax(z.T / self.temp, dim=1)
                loss = loss + self.lambda_unif * (self._uniformity(p_src, gen)
                                                  + self._uniformity(p_tgt, gen))
            if self.lambda_align > 0 and epoch >= self.warmup_epochs:
                p_src = torch.softmax(z / self.temp, dim=1)
                uu, vv = self._bootstrap_positives(S, u1, v1, self.pos_bar)
                if uu.numel():
                    ps = F.normalize(p_src, p=2, dim=1)
                    loss = loss + self.lambda_align * (1 - (ps[uu] * ps[vv]).sum(dim=1)).mean()
                p_tgt = torch.softmax(z.T / self.temp, dim=1)
                uu, vv = self._bootstrap_positives(S.T.contiguous(), u2, v2, self.pos_bar)
                if uu.numel():
                    pt = F.normalize(p_tgt, p=2, dim=1)
                    loss = loss + self.lambda_align * (1 - (pt[uu] * pt[vv]).sum(dim=1)).mean()

            refer_time += time.time() - ref_t0
            loss.backward()
            optimizer.step()
            t1 = time.time()

            with torch.no_grad():
                model.eval()
                hits = hits_ks_scores(S, test_pairs, mode='mean')
                mrr = mrr_score(S, test_pairs, mode='mean')
                mem_gb = process.memory_info().rss / 1024 ** 3
                logger.log(epoch=epoch + 1, loss=loss.item(), epoch_time=t1 - t0,
                           hits=hits, mrr=mrr, memory=round(mem_gb, 4),
                           infer_time=round(refer_time, 4), verbose=verbose)

            if self.group_marginals and epoch + 1 >= self.marginal_warmup:
                # Bootstrap partitions from the current coupling and refresh
                # the group-uniform marginals for the next epoch's solve.
                from PlanetAlign.m2m_quotient import (
                    merge_average_linkage, otsu_threshold, _pair_cosines)
                with torch.no_grad():
                    Sf = S.detach().to(torch.float32).cpu()
                    sims1 = _pair_cosines(Sf, u1, v1) if u1.numel() else torch.zeros(0)
                    grp1 = merge_average_linkage(n1, u1, v1, sims1, Sf,
                                                 otsu_threshold(sims1))
                    StT = Sf.T.contiguous()
                    sims2 = _pair_cosines(StT, u2, v2) if u2.numel() else torch.zeros(0)
                    grp2 = merge_average_linkage(n2, u2, v2, sims2, StT,
                                                 otsu_threshold(sims2))
                    criterion.set_marginals(
                        _group_uniform_marginal(grp1, n1, self.marginal_rho, self.dtype).to(self.device),
                        _group_uniform_marginal(grp2, n2, self.marginal_rho, self.dtype).to(self.device))

        self.S = S.detach()
        return S.detach(), logger
