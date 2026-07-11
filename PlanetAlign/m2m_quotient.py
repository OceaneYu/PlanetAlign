"""Quotient decoding: a principled many-to-many readout over a 1-1 aligner.

Design rationale (each choice answers one failure mode documented in
``docs/m2m_why_one_to_one_fails.md``):

1. **Learning object = equivalence classes.** A many-to-many alignment is a
   partition of each graph plus a matching between the partitions (the quotient
   graphs). The node-level similarity ``S`` cannot express intra-graph
   co-reference; the partition can.
2. **Evidence = alignment profiles.** Group-mates map to the same region of the
   other graph, so their rows of ``S`` are near-collinear while non-group-mates
   (even attribute-identical, adjacent ones) are near-orthogonal. This signal is
   already present in any trained 1-1 aligner — no new module is needed.
3. **Adaptive threshold (Otsu).** The within/cross-group profile-cosine
   distribution is bimodal; a per-graph Otsu threshold replaces the brittle
   fixed ``tau``.
4. **Average-linkage gate (anti-chaining).** Plain union-find merges by
   transitivity: A~B and B~C chain A with C even when A and C are dissimilar.
   Candidate edges are processed by descending cosine and a merge additionally
   requires the two group *centroids* to agree, which blocks chains.
5. **Exclusivity at the right level (Hungarian on the quotient).** Ground-truth
   entities pair one source group with one target group, so at the quotient
   level a one-to-one prior is *correct again*. A global rectangular assignment
   replaces per-group greedy argmax, which can stack many source groups onto
   one attractive target group.
6. **Alternating refinement.** Once target groups exist, source profiles can be
   pooled from ``n2`` dimensions down to ``g2`` group columns — denoised
   evidence — and the decode/match repeated to a (partition) fixed point.
7. **Anchor-supervised evidence selection.** Grouping evidence differs in
   reliability per graph: verbatim attribute cohesion is near-perfect on
   strong-attribute graphs (Cora, 1433-dim) but over-merges on weak ones (the
   25:1 base-rate problem); profile cohesion behaves oppositely — and is itself
   weakened exactly when the OT marginals are balanced (n1 ≈ n2 locks ``S``
   toward a permutation, so group-mates cannot share target columns). No
   single unsupervised contrast can arbitrate, because each candidate optimizes
   its own signal. The arbiter that is both legitimate and task-aligned is the
   **training anchors**: the 1-1 anchor pairs are the supervision every aligner
   already trains on, upgraded here to group level — a candidate partition pair
   is scored by the fraction of train anchors (x, y) whose source group is
   matched to the target group containing y. The entity-level ground truth is
   never consulted. Without anchors, a profile-contrast arbiter is the fallback.
8. **Null-calibrated threshold candidate (RMT).** Random-pair profile cosines
   form the no-signal null; its upper quantile is a calibrated merge threshold.
   Measured motivation: on Cora, Otsu lands at ~0.43-0.46 — *above* the
   within-group median 0.35 — while the null q999 is 0.066; the profile signal
   was there (within-vs-cross AUC 0.998), Otsu just mis-thresholded it.
9. **Global candidates (dead on vanilla JOENA, revived by JOENA-PC).** On
   Douban 59.6% of multi-node source groups are internally disconnected — a
   hard ceiling for any adjacency-restricted merge. Attr-bucket x
   near-duplicate-profile candidates were first *measured dead*: precision 1.9%
   (154/7917 true), with no gate able to rescue it — whole rows of the vanilla
   coupling are duplicated across unrelated nodes (null q999 = 1.0), traced to
   an input symmetry (identical ``[attr, anchor-RWR]`` inputs give identical
   rows forever). :mod:`PlanetAlign.m2m_contrastive` (JOENA-PC) breaks the
   symmetry at training time (propagated positional features + a
   profile-uniformity term); on its S the null bulk dissolves (q99 1.0 -> 0.88)
   and this same candidate *wins* the anchor arbitration (Douban blind MicroF1
   0.633 -> 0.677). Default ``global_candidate=None`` (auto) enables it only
   when the anchor arbiter is active (>= 20 anchors) — risky evidence requires
   a reliable arbiter, which correctly rejected it on the vanilla S.
10. **Aligner-level arbitration (falsified at multi-seed; superseded by the
   coupling ensemble).** :func:`entity_anchor_agreement` scores a *final*
   entity map by chance-corrected train-anchor agreement. It ranked branches
   correctly on all four benchmarks at seed 42 — but multi-seed testing (seeds
   0/1/2) showed that was luck: on Douban it picks the right branch 1/4 times
   and at seed 0 it is *confidently wrong* (agreement 0.64 for the branch that
   decodes 0.15 worse). Root cause: anchors are the preserved 1-1 pairs, so
   branch-level agreement measures the 1-1 slice of quality — exactly what the
   M2M-oriented branches trade away. The shipped default is instead the
   **coupling ensemble** (``--model ensemble``): average the three branches'
   couplings and decode once; it matches the per-seed oracle branch on
   douban/cora/pems08 (0.639/0.992/0.651 vs oracle 0.639/0.992/0.658) with no
   selector at all, trailing only on airport (0.768 vs 0.826) where one branch
   strongly dominates. ``--model auto`` remains available with this caveat.

Everything here is post-hoc over a frozen ``S``: the empirical finding is that
the bottleneck of 1-1 methods on M2M is the readout, not the representation.
The prediction path never touches the ground-truth entity map, so results are
valid under the blind protocol in :mod:`PlanetAlign.m2m_blind`.

Precondition (measured scope boundary). QuotientDecode is a *readout over an
aligned* ``S``: its evidence is that group-mates have collinear ``S`` rows,
which requires ``S`` to carry non-trivial alignment signal. On datasets where
the base aligner fails at the node level (JOENA ``Hits@1`` ~ 0 on italy /
phone-email / arenas, even at 200 epochs — weak/plain-attribute, near-regular
graphs), the rows are noise and every readout collapses (MicroF1 0.05-0.15).
That is an upstream 1-1 failure, orthogonal to this module; fix the base
aligner first. A cheap train-anchor ``Hits@1`` check is a suitable guard.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from PlanetAlign.algorithms.base_model import BaseModel
from PlanetAlign.algorithms.joena import JOENA
from PlanetAlign.data import Dataset
from PlanetAlign.m2m import EntityMap, align_prediction_to_ground_truth
from PlanetAlign.metrics import many_to_many_scores
from PlanetAlign.utils import get_anchor_pairs


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _adjacent_pairs(graph) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unique undirected adjacent pairs (u < v) of a PyG graph."""
    ei = graph.edge_index
    if ei.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    src, dst = ei[0], ei[1]
    keep = src < dst
    return src[keep], dst[keep]


def _pair_cosines(profiles: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    p = F.normalize(profiles.to(torch.float32), p=2, dim=1)
    return (p[u] * p[v]).sum(dim=1)


def pool_columns(S: torch.Tensor, groups: Sequence[Sequence[int]]) -> torch.Tensor:
    """Mean-pool the columns of ``S`` by group: returns ``[n_rows, n_groups]``."""
    n_rows, n_cols = S.shape
    g = len(groups)
    labels = torch.zeros(n_cols, dtype=torch.long)
    for gid, members in enumerate(groups):
        for node in members:
            labels[node] = gid
    size = torch.zeros(g).index_add_(0, labels, torch.ones(n_cols)).clamp(min=1.0)
    pooled = torch.zeros(n_rows, g, dtype=S.dtype)
    pooled.index_add_(1, labels, S)
    return pooled / size.unsqueeze(0)


def _pool_columns_reduce(S: torch.Tensor, groups: Sequence[Sequence[int]],
                         reduce: str) -> torch.Tensor:
    """Column pooling by group with ``sum`` / ``mean`` / ``amax`` reduction."""
    n_rows, n_cols = S.shape
    labels = torch.zeros(n_cols, dtype=torch.long)
    for gid, members in enumerate(groups):
        for node in members:
            labels[node] = gid
    g = len(groups)
    if reduce == "amax":
        pooled = torch.full((n_rows, g), float("-inf"), dtype=S.dtype)
        pooled = pooled.index_reduce_(1, labels, S, "amax", include_self=True)
        return pooled
    pooled = torch.zeros(n_rows, g, dtype=S.dtype)
    pooled.index_add_(1, labels, S)
    if reduce == "mean":
        size = torch.zeros(g).index_add_(0, labels, torch.ones(n_cols)).clamp(min=1.0)
        pooled = pooled / size.unsqueeze(0)
    return pooled


def quotient_scores(S: torch.Tensor,
                    src_groups: Sequence[Sequence[int]],
                    tgt_groups: Sequence[Sequence[int]],
                    mode: str = "mean") -> torch.Tensor:
    """Group-to-group scores for every (source group, target group) pair.

    Modes (all vectorized, no per-block loops):

    - ``mean``: mean of the S block — the original readout. Dilutes 1-to-many
      blocks (a correct k x m block averages mass/(km)).
    - ``coverage``: bidirectional containment, scale-free in [0, 1]. For pair
      (A, B): covA = mean over i in A of (mass of S[i, B] / mass of S[i, :]),
      covB symmetric over columns; score = sqrt(covA * covB). Measures "A's
      mass lands in B AND B's mass comes from A", the entity semantics.

    Measured (18 dataset-seed cells, paired on identical S): coverage beats
    mean on balanced-marginal data (cora +0.0016 x4 seeds, pems08 +0.003..
    +0.025, ppi +0.004..0.008) and loses consistently on unbalanced Douban
    (-0.007..-0.015, row/col-mass normalization distorts when column
    capacities differ ~3x). Default stays ``mean``; ``coverage`` is the
    documented option for balanced regimes. max/sum/hybrid variants were
    measured with no net value and removed.
    """
    if mode == "mean":
        R = pool_columns(S, tgt_groups)              # [n1, g2]
        return pool_columns(R.T.contiguous(), src_groups).T.contiguous()
    if mode == "coverage":
        eps = 1e-12
        Rsum = _pool_columns_reduce(S, tgt_groups, "sum")                  # [n1, g2]
        cov_rows = Rsum / S.sum(dim=1, keepdim=True).clamp(min=eps)
        covA = _pool_columns_reduce(cov_rows.T.contiguous(), src_groups, "mean").T.contiguous()
        Csum = _pool_columns_reduce(S.T.contiguous(), src_groups, "sum")   # [n2, g1]
        cov_cols = Csum / S.sum(dim=0, keepdim=True).clamp(min=eps).T
        covB = _pool_columns_reduce(cov_cols.T.contiguous(), tgt_groups, "mean")  # [g1, g2]
        return torch.sqrt(covA.clamp(min=0) * covB.clamp(min=0))
    raise ValueError(f"unknown score mode: {mode}")


def quotient_adjacency(graph, groups: Sequence[Sequence[int]]) -> torch.Tensor:
    """Row-normalized quotient-graph adjacency (with self-loops): [g, g]."""
    g = len(groups)
    labels = _group_labels(groups, int(graph.num_nodes))
    A = torch.eye(g)
    ei = graph.edge_index
    if ei.numel():
        a, b = labels[ei[0]], labels[ei[1]]
        keep = a != b
        A[a[keep], b[keep]] = 1.0
        A[b[keep], a[keep]] = 1.0
    return A / A.sum(dim=1, keepdim=True).clamp(min=1.0)


def neighbor_consistency_refine(T: torch.Tensor,
                                Aq1: torch.Tensor,
                                Aq2: torch.Tensor,
                                beta: float,
                                iters: int = 1) -> torch.Tensor:
    """Group-level alignment-consistency smoothing of the score matrix.

    The consistency principle ("neighbors of matches should match") is wrong at
    node level under M2M but *correct at the quotient level*: if group A
    matches group B, A's neighbor groups should match B's neighbor groups.
    One propagation step: T <- (1-beta) * T + beta * Aq1 @ T @ Aq2^T, on a
    min-max normalized T.
    """
    if beta <= 0:
        return T
    lo, hi = float(T.min()), float(T.max())
    Tn = (T - lo) / (hi - lo) if hi > lo else T
    for _ in range(iters):
        Tn = (1 - beta) * Tn + beta * (Aq1 @ Tn @ Aq2.T)
    return Tn


def evict_outliers(groups: List[List[int]],
                   profiles: torch.Tensor,
                   tau: float) -> List[List[int]]:
    """Merge-split refinement: evict members that disagree with their group.

    Merging is greedy and irreversible; this one-pass correction removes any
    member whose profile cosine to its group's leave-self-out centroid falls
    below ``tau`` (the same threshold that justified the merges). Evicted
    nodes become singletons.
    """
    p = F.normalize(profiles.to(torch.float32), p=2, dim=1)
    out: List[List[int]] = []
    for members in groups:
        if len(members) < 2:
            out.append(list(members))
            continue
        idx = torch.tensor(members, dtype=torch.long)
        total = p[idx].sum(dim=0)
        keep, evicted = [], []
        for node in members:
            rest = total - p[node]
            denom = float(rest.norm())
            cos = float(p[node] @ rest) / denom if denom > 0 else 0.0
            (keep if cos >= tau else evicted).append(node)
        if len(keep) >= 2:
            out.append(keep)
            out.extend([e] for e in evicted)
        else:
            out.append(list(members))     # group would dissolve; keep as-is
    return out


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------
def otsu_threshold(values: torch.Tensor,
                   lo: float = 0.05,
                   hi: float = 0.6,
                   bins: int = 64,
                   fallback: float = 0.1) -> float:
    """Otsu's threshold over a 1-D sample, clamped to ``[lo, hi]``.

    The within-group vs cross-group profile cosines form a bimodal mixture;
    Otsu maximizes the between-class variance of the split. Degenerate samples
    (too few pairs, no spread) fall back to ``fallback``.
    """
    vals = values.detach().to(torch.float32).flatten()
    vals = vals[torch.isfinite(vals)]
    if vals.numel() < 10 or float(vals.std()) < 1e-6:
        return fallback
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax - vmin < 1e-6:
        return fallback
    hist = torch.histc(vals, bins=bins, min=vmin, max=vmax)
    total = hist.sum()
    centers = torch.linspace(vmin, vmax, bins)
    w0 = torch.cumsum(hist, 0)
    w1 = total - w0
    mu_cum = torch.cumsum(hist * centers, 0)
    mu_total = mu_cum[-1]
    mu0 = mu_cum / w0.clamp(min=1e-9)
    mu1 = (mu_total - mu_cum) / w1.clamp(min=1e-9)
    between = w0 * w1 * (mu0 - mu1) ** 2
    between[(w0 == 0) | (w1 == 0)] = -1.0
    # The empty valley between two tight modes makes `between` flat over many
    # bins; argmax alone would sit at the valley's low edge. Take the plateau
    # midpoint instead, the standard Otsu tie-break.
    plateau = torch.nonzero(between >= between.max() - 1e-9).flatten()
    t = float(centers[int(plateau[len(plateau) // 2])])
    return min(max(t, lo), hi)


# ---------------------------------------------------------------------------
# Anti-chaining agglomeration
# ---------------------------------------------------------------------------
def merge_average_linkage(num_nodes: int,
                          u: torch.Tensor,
                          v: torch.Tensor,
                          sims: torch.Tensor,
                          profiles: torch.Tensor,
                          tau: Union[float, torch.Tensor]) -> List[List[int]]:
    """Union-find over candidate pairs with an average-linkage centroid gate.

    Candidate edges are processed in descending profile-cosine order. A merge
    happens only if (a) the edge cosine passes its ``tau`` and (b) the cosine of
    the two current group centroids also passes it. (b) is what blocks
    transitive chaining: in plain union-find, A~B and B~C force {A,B,C} even
    when cos(A, C) is low.

    ``tau`` may be a scalar or a per-edge tensor — the latter lets adjacency
    pairs and (stricter) global candidate pairs share one merge pass.
    """
    p = F.normalize(profiles.to(torch.float32), p=2, dim=1)
    parent = list(range(num_nodes))
    csum = p.clone()                      # centroid running sums, indexed by root
    per_edge = isinstance(tau, torch.Tensor)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = torch.argsort(-sims)
    for idx in order.tolist():
        t = float(tau[idx]) if per_edge else tau
        if float(sims[idx]) < t:
            if per_edge:
                continue
            break
        ru, rv = find(int(u[idx])), find(int(v[idx]))
        if ru == rv:
            continue
        a, b = csum[ru], csum[rv]
        denom = float(a.norm()) * float(b.norm())
        cos_ab = float(a @ b) / denom if denom > 0 else 0.0
        if cos_ab < t:
            continue
        parent[rv] = ru
        csum[ru] = a + b

    groups: Dict[int, List[int]] = {}
    for node in range(num_nodes):
        groups.setdefault(find(node), []).append(node)
    return list(groups.values())


# ---------------------------------------------------------------------------
# RMT-motivated evidence: null-calibrated threshold and global candidates
# ---------------------------------------------------------------------------
def null_threshold(profiles: torch.Tensor,
                   num_null: int = 20000,
                   q: float = 0.999,
                   seed: int = 0,
                   degenerate_above: float = 0.9) -> Optional[float]:
    """Merge threshold calibrated on the random-pair null distribution.

    In high dimension, profiles of unrelated nodes have near-zero cosine with
    fluctuations shrinking like 1/sqrt(effective dim); the empirical ``q``
    quantile of random-pair cosines turns "merge" into a calibrated hypothesis
    test. Returns ``None`` when the null itself is heavy at 1.0 (duplicate-row
    degeneracy, e.g. hub-aligned Sinkhorn rows) — the candidate is then skipped.
    """
    n = profiles.shape[0]
    if n < 3:
        return None
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, n, (num_null,), generator=g)
    j = torch.randint(0, n, (num_null,), generator=g)
    keep = i != j
    if int(keep.sum()) < 100:
        return None
    cos = _pair_cosines(profiles, i[keep], j[keep])
    t = float(torch.quantile(cos, q))
    return None if t > degenerate_above else t


def global_candidate_pairs(profiles: torch.Tensor,
                           x: Optional[torch.Tensor],
                           tau_global: float = 0.98,
                           hub_load_limit: int = 16,
                           topk: int = 8,
                           max_bucket: int = 2000,
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-adjacent merge candidates: attr-identical pairs with near-duplicate
    profiles, excluding hub-degenerate rows.

    Motivated by a measured failure: on Douban 59.6% of multi-node source
    groups are internally *disconnected*, so adjacency-restricted merging has a
    hard ceiling. The benchmark builder copies split-node attributes verbatim,
    so true siblings live in the same exact-attribute bucket and have profile
    cosine ~1.0. Three gates keep this safe on attr-degenerate graphs:
    exact-attr bucket membership, profile cosine >= ``tau_global``, and a hub
    guard dropping rows whose argmax column is shared by > ``hub_load_limit``
    rows (Sinkhorn hub rows are near-duplicates of each other without being
    co-referent).
    """
    empty = (torch.zeros(0, dtype=torch.long),) * 2 + (torch.zeros(0),)
    if x is None:
        return empty

    # Hub guard from the profiles' argmax column load.
    top_col = profiles.argmax(dim=1)
    load = torch.bincount(top_col, minlength=int(profiles.shape[1]))
    ok = load[top_col] <= hub_load_limit

    buckets: Dict[Tuple, List[int]] = {}
    for node in range(x.shape[0]):
        if not bool(ok[node]):
            continue
        key = tuple(x[node].nonzero().flatten().tolist())
        buckets.setdefault(key, []).append(node)

    p = F.normalize(profiles.to(torch.float32), p=2, dim=1)
    us, vs, ss = [], [], []
    for members in buckets.values():
        if len(members) < 2 or len(members) > max_bucket:
            continue
        idx = torch.tensor(members, dtype=torch.long)
        C = p[idx] @ p[idx].T
        C.fill_diagonal_(-1.0)
        k = min(topk, len(members) - 1)
        vals, nbrs = C.topk(k, dim=1)
        for a in range(len(members)):
            for b_pos in range(k):
                if float(vals[a, b_pos]) < tau_global:
                    break
                b = int(nbrs[a, b_pos])
                if a < b:
                    us.append(members[a]); vs.append(members[b]); ss.append(float(vals[a, b_pos]))
    if not us:
        return empty
    return (torch.tensor(us, dtype=torch.long), torch.tensor(vs, dtype=torch.long),
            torch.tensor(ss))


# ---------------------------------------------------------------------------
# Blind evidence selection
# ---------------------------------------------------------------------------
def partition_contrast(profiles: torch.Tensor,
                       u: torch.Tensor,
                       v: torch.Tensor,
                       groups: Sequence[Sequence[int]]) -> float:
    """Profile contrast of a partition over the graph's adjacent pairs.

    ``mean cos(profiles) over within-group adjacent pairs`` minus the same mean
    over cross-group pairs. The validated co-reference signal (profile
    collinearity) arbitrates between candidate partitions without touching any
    ground truth. Partitions with no within pairs (all singletons) score -inf;
    a single all-in-one blob has no cross pairs and scores just its (typically
    low) within mean.
    """
    if u.numel() == 0:
        return float("-inf")
    num_nodes = profiles.shape[0]
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    for gid, members in enumerate(groups):
        for node in members:
            labels[node] = gid
    cos = _pair_cosines(profiles, u, v)
    within_mask = labels[u] == labels[v]
    within, cross = cos[within_mask], cos[~within_mask]
    if within.numel() == 0:
        return float("-inf")
    if cross.numel() == 0:
        return float(within.mean())
    return float(within.mean() - cross.mean())


def _candidate_partitions(profiles: torch.Tensor,
                          graph,
                          u: torch.Tensor,
                          v: torch.Tensor,
                          tau: Optional[float],
                          anti_chaining: bool,
                          evidence_selection: bool,
                          use_attr: bool,
                          null_candidate: bool = True,
                          global_candidate: bool = True,
                          allowed: Optional[str] = None,
                          ) -> Tuple[List[Tuple[str, List[List[int]]]], float]:
    """Candidate partitions for one side.

    Up to four evidences, all arbitrated downstream by the anchor criterion:
    ``profile@tau`` (Otsu or fixed), ``profile-null`` (random-pair-null
    calibrated threshold), ``profile-g`` (adjacency plus gated global
    candidates, breaking the disconnected-group ceiling), and ``attr``.
    """
    sims = _pair_cosines(profiles, u, v) if u.numel() else torch.zeros(0)

    def profile_partition(t: float) -> List[List[int]]:
        if anti_chaining:
            return merge_average_linkage(int(graph.num_nodes), u, v, sims, profiles, t)
        from PlanetAlign.m2m_blind import discover_groups_by_profile
        return discover_groups_by_profile(profiles, graph, tau=t)

    tau_main = tau if tau is not None else otsu_threshold(sims)
    n = int(graph.num_nodes)
    x = getattr(graph, "x", None)
    candidates: List[Tuple[str, List[List[int]]]] = []

    def want(kind: str) -> bool:
        return allowed is None or allowed == kind

    if want("profile"):
        candidates.append((f"profile@{tau_main:.3f}", profile_partition(tau_main)))
    if evidence_selection and null_candidate and want("profile-null"):
        t_null = null_threshold(profiles)
        if t_null is not None and abs(t_null - tau_main) > 1e-3:
            candidates.append((f"profile-null@{t_null:.3f}", profile_partition(t_null)))
    if evidence_selection and global_candidate and x is not None and want("profile-g"):
        gu, gv, gs = global_candidate_pairs(profiles, x)
        if gu.numel():
            all_u = torch.cat([u, gu]); all_v = torch.cat([v, gv])
            all_s = torch.cat([sims, gs])
            tau_vec = torch.cat([torch.full((u.numel(),), float(tau_main)),
                                 torch.full((gu.numel(),), max(float(tau_main), 0.98))])
            candidates.append((f"profile-g@{tau_main:.3f}+{gu.numel()}",
                               merge_average_linkage(n, all_u, all_v, all_s, profiles, tau_vec)))
    if (evidence_selection and use_attr and x is not None and want("attr")):
        from PlanetAlign.m2m_blind import discover_groups
        candidates.append(("attr", discover_groups(graph, use_attr=True)))
    if not candidates:  # allowed kind unavailable this round — fall back to profile
        candidates.append((f"profile@{tau_main:.3f}", profile_partition(tau_main)))
    return candidates, tau_main


def _group_labels(groups: Sequence[Sequence[int]], num_nodes: int) -> torch.Tensor:
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    for gid, members in enumerate(groups):
        for node in members:
            labels[node] = gid
    return labels


def anchor_agreement(S: torch.Tensor,
                     src_groups: Sequence[Sequence[int]],
                     tgt_groups: Sequence[Sequence[int]],
                     anchors: torch.Tensor,
                     matcher: str = "hungarian",
                     relative_match_threshold: float = 0.0,
                     ) -> Tuple[float, float]:
    """Chance-corrected fraction of train anchors whose groups match together.

    Anchors are the task's given 1-1 supervision (the same pairs the base
    aligner trains on), upgraded to group level for readout model selection.
    Raw agreement saturates in two opposite ways: giant merged blobs swallow
    anchors trivially, and near-singleton partitions satisfy anchors without
    merging anything. The chance term — the probability that a random target
    node falls into the matched group, ``|matched target group| / n2`` — cancels
    the blob effect; coarseness preferences (see the caller) handle the rest.

    Returns ``(corrected, raw)``.
    """
    if anchors is None or anchors.numel() == 0:
        return float("nan"), float("nan")
    T = quotient_scores(S, src_groups, tgt_groups)
    match = (hungarian_match if matcher == "hungarian" else greedy_match)(
        T, relative_threshold=relative_match_threshold)
    lab_s = _group_labels(src_groups, S.shape[0])
    lab_t = _group_labels(tgt_groups, S.shape[1])
    n2 = S.shape[1]
    hits, chance = 0, 0.0
    for x, y in anchors.tolist():
        b = match.get(int(lab_s[int(x)]))
        if b is None:
            continue
        chance += len(tgt_groups[b]) / n2
        if b == int(lab_t[int(y)]):
            hits += 1
    m = anchors.shape[0]
    raw = hits / m
    return raw - chance / m, raw


def entity_anchor_agreement(pred: "EntityMap",
                            anchors: torch.Tensor,
                            n2: int) -> Tuple[float, float]:
    """Chance-corrected anchor agreement of a *final* entity map.

    Decode-agnostic version of :func:`anchor_agreement`: an anchor (x, y)
    agrees when some predicted entity contains x on the source side and y on
    the target side. Used to arbitrate between whole aligners (e.g. JOENA vs
    JOENA-PC) — model selection on the task's given supervision only.
    Returns ``(corrected, raw)``.
    """
    if anchors is None or anchors.numel() == 0:
        return float("nan"), float("nan")
    src_to_ent: Dict[int, str] = {}
    for eid, item in pred.items():
        for x in item.get("src", []):
            src_to_ent.setdefault(int(x), eid)
    hits, chance = 0, 0.0
    for x, y in anchors.tolist():
        eid = src_to_ent.get(int(x))
        if eid is None:
            continue
        tgt = pred[eid].get("tgt", [])
        chance += len(tgt) / max(n2, 1)
        if int(y) in set(int(t) for t in tgt):
            hits += 1
    m = anchors.shape[0]
    raw = hits / m
    return raw - chance / m, raw


def _select_partitions(S: torch.Tensor,
                       cands_s: List[Tuple[str, List[List[int]]]],
                       cands_t: List[Tuple[str, List[List[int]]]],
                       anchors: Optional[torch.Tensor],
                       matcher: str,
                       relative_match_threshold: float,
                       profiles_src: torch.Tensor,
                       profiles_tgt: torch.Tensor,
                       us: torch.Tensor, vs: torch.Tensor,
                       ut: torch.Tensor, vt: torch.Tensor,
                       ) -> Tuple[List[List[int]], List[List[int]], str, str, float]:
    """Pick the (source, target) partition pair.

    With anchors: maximize group-level anchor agreement (ties → profile
    contrast). Without: per-side profile contrast (unsupervised fallback).
    """
    if len(cands_s) == 1 and len(cands_t) == 1:
        return cands_s[0][1], cands_t[0][1], cands_s[0][0], cands_t[0][0], float("nan")

    # With too few anchors the agreement estimate is noise (measured on
    # pems08: 6 anchors mis-arbitrate, the contrast fallback chooses better).
    MIN_ANCHORS = 20
    if anchors is not None and anchors.shape[0] < MIN_ANCHORS:
        anchors = None

    if anchors is not None and anchors.numel() > 0:
        scored = []
        for ns, gs in cands_s:
            for nt, gt_ in cands_t:
                corrected, raw = anchor_agreement(S, gs, gt_, anchors, matcher,
                                                  relative_match_threshold)
                scored.append((corrected, raw, gs, gt_, ns, nt))
        # Occam among the anchor-consistent: candidates within a small tolerance
        # of the best corrected agreement are tied; among ties prefer the
        # coarsest partition pair (fewest groups that still explain the anchors).
        best_corrected = max(s[0] for s in scored)
        tied = [s for s in scored if s[0] >= best_corrected - 0.02]
        corrected, raw, gs, gt_, ns, nt = min(tied, key=lambda s: len(s[2]) + len(s[3]))
        return gs, gt_, ns, nt, corrected

    pick_s = max(cands_s, key=lambda c: partition_contrast(profiles_src, us, vs, c[1]))
    pick_t = max(cands_t, key=lambda c: partition_contrast(profiles_tgt, ut, vt, c[1]))
    return pick_s[1], pick_t[1], pick_s[0], pick_t[0], float("nan")


# ---------------------------------------------------------------------------
# Quotient matching
# ---------------------------------------------------------------------------
def hungarian_match(T: torch.Tensor, relative_threshold: float = 0.0) -> Dict[int, int]:
    """Global one-to-one assignment between source and target groups.

    Solves the rectangular linear assignment problem on ``-T`` and drops
    assigned pairs whose pooled score is ``<= relative_threshold * T.max()``.
    Unassigned / dropped source groups stay unmatched (empty target side).
    """
    from scipy.optimize import linear_sum_assignment

    Tn = T.detach().to(torch.float32).cpu().numpy()
    rows, cols = linear_sum_assignment(-Tn)
    cutoff = relative_threshold * float(T.max()) if T.numel() else 0.0
    return {int(a): int(b) for a, b in zip(rows, cols) if Tn[a, b] > cutoff}


def greedy_match(T: torch.Tensor, relative_threshold: float = 0.0) -> Dict[int, int]:
    """Per-source-group argmax matching (the pre-quotient baseline readout)."""
    if T.numel() == 0:
        return {}
    best_val, best_idx = T.max(dim=1)
    cutoff = relative_threshold * float(T.max())
    return {a: int(best_idx[a]) for a in range(T.shape[0]) if float(best_val[a]) > cutoff}


def _partition_signature(groups: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(g)) for g in groups))


def quotient_decode(S: torch.Tensor,
                    graph_src,
                    graph_tgt,
                    tau: Optional[float] = None,
                    max_iters: int = 2,
                    relative_match_threshold: float = 0.0,
                    matcher: str = "hungarian",
                    anti_chaining: bool = True,
                    evidence_selection: bool = True,
                    use_attr: bool = True,
                    null_candidate: bool = True,
                    global_candidate: Optional[bool] = None,
                    anchors: Optional[torch.Tensor] = None,
                    overlap_expand_tau: Optional[float] = None,
                    score_mode: str = "mean",
                    neighbor_beta: float = 0.0,
                    evict: bool = True,
                    ) -> Tuple[EntityMap, Dict[str, object]]:
    """Decode a many-to-many entity map from a node-level similarity matrix.

    Parameters
    ----------
    S : ``[n1, n2]`` similarity/transport matrix from any 1-1 aligner.
    tau : profile-cosine merge threshold; ``None`` selects it per side and per
        iteration with Otsu's method.
    max_iters : decode/match alternations. Iteration 1 uses raw profiles
        (rows/columns of ``S``); later iterations use group-pooled profiles.
    matcher : ``"hungarian"`` (global 1-1 on the quotient) or ``"greedy"``.
    anti_chaining : apply the average-linkage centroid gate.
    evidence_selection : decode candidate partitions per side (profile@Otsu and
        attribute cohesion) and select the pair with the highest group-level
        anchor agreement (see ``anchors``); falls back to profile contrast when
        no anchors are given. The entity-level GT is never consulted.
    use_attr : allow the attribute-cohesion candidate (requires ``graph.x``).
    anchors : ``[m, 2]`` train anchor pairs — the task's given 1-1 supervision,
        used only to arbitrate between candidate partitions.
    overlap_expand_tau : if set, after matching each node may join one extra
        entity whose source-group centroid its profile matches above this
        threshold (handles fuzzy-boundary benchmarks; off by default).
    """
    S = S.detach().to(torch.float32).cpu()
    n1, n2 = S.shape
    if global_candidate is None:
        # Risky evidence needs a reliable arbiter: enable global candidates
        # only when the anchor criterion is active (>= MIN_ANCHORS pairs).
        global_candidate = anchors is not None and anchors.shape[0] >= 20
    us, vs = _adjacent_pairs(graph_src)
    ut, vt = _adjacent_pairs(graph_tgt)

    profiles_src: torch.Tensor = S
    profiles_tgt: torch.Tensor = S.T.contiguous()
    src_groups: List[List[int]] = [[i] for i in range(n1)]
    tgt_groups: List[List[int]] = [[j] for j in range(n2)]
    info: Dict[str, object] = {"taus": [], "group_counts": [], "evidence": [],
                               "anchor_agreement": [], "iters": 0}
    prev_sig: Optional[Tuple] = None
    allowed_s: Optional[str] = None   # evidence type locked after iteration 1
    allowed_t: Optional[str] = None

    for it in range(max_iters):
        cands_s, tau_s = _candidate_partitions(profiles_src, graph_src, us, vs, tau,
                                               anti_chaining, evidence_selection, use_attr,
                                               null_candidate, global_candidate,
                                               allowed=allowed_s)
        cands_t, tau_t = _candidate_partitions(profiles_tgt, graph_tgt, ut, vt, tau,
                                               anti_chaining, evidence_selection, use_attr,
                                               null_candidate, global_candidate,
                                               allowed=allowed_t)
        src_groups, tgt_groups, ev_s, ev_t, agree = _select_partitions(
            S, cands_s, cands_t, anchors, matcher, relative_match_threshold,
            profiles_src, profiles_tgt, us, vs, ut, vt)
        allowed_s, allowed_t = ev_s.split("@")[0], ev_t.split("@")[0]

        if evict:
            # Merge-split refinement: one leave-self-out eviction pass at the
            # same threshold that justified the merges. Only valid when the
            # partition's own evidence is the profile (evicting attr-evidence
            # groups by profile cosine mismatches evidences — measured -0.043
            # on Cora, whose true members have low profile cosine under the
            # permutation lock).
            if allowed_s == "profile":
                src_groups = evict_outliers(src_groups, profiles_src, tau_s)
            if allowed_t == "profile":
                tgt_groups = evict_outliers(tgt_groups, profiles_tgt, tau_t)

        info["taus"].append((round(tau_s, 4), round(tau_t, 4)))
        info["group_counts"].append((len(src_groups), len(tgt_groups)))
        info["evidence"].append((ev_s, ev_t))
        info["anchor_agreement"].append(None if agree != agree else round(agree, 4))
        info["iters"] = it + 1

        sig = (_partition_signature(src_groups), _partition_signature(tgt_groups))
        if sig == prev_sig:
            break
        prev_sig = sig

        # Denoised profiles for the next round: pool over the other side's groups.
        profiles_src = pool_columns(S, tgt_groups)                    # [n1, g2]
        profiles_tgt = pool_columns(S.T.contiguous(), src_groups)     # [n2, g1]

    T = quotient_scores(S, src_groups, tgt_groups, mode=score_mode)
    if neighbor_beta > 0:
        T = neighbor_consistency_refine(
            T, quotient_adjacency(graph_src, src_groups),
            quotient_adjacency(graph_tgt, tgt_groups), beta=neighbor_beta)
    match = (hungarian_match if matcher == "hungarian" else greedy_match)(
        T, relative_threshold=relative_match_threshold)

    pred: EntityMap = {}
    for a, members in enumerate(src_groups):
        b = match.get(a)
        pred[f"p{a}"] = {
            "src": [int(x) for x in members],
            "tgt": [int(y) for y in tgt_groups[b]] if b is not None else [],
        }

    if overlap_expand_tau is not None and len(src_groups) > 1:
        centroids = torch.stack([
            F.normalize(F.normalize(profiles_src, p=2, dim=1)[g].sum(dim=0), p=2, dim=0)
            for g in (torch.tensor(g_, dtype=torch.long) for g_ in src_groups)
        ])
        prof_n = F.normalize(profiles_src, p=2, dim=1)
        cos = prof_n @ centroids.T                                    # [n1, g1]
        member_of = torch.full((n1,), -1, dtype=torch.long)
        for a, g_ in enumerate(src_groups):
            for node in g_:
                member_of[node] = a
        cos[torch.arange(n1), member_of] = -1.0                       # exclude own group
        best_val, best_a = cos.max(dim=1)
        for node in range(n1):
            if float(best_val[node]) >= overlap_expand_tau:
                pred[f"p{int(best_a[node])}"]["src"].append(int(node))

    info["matched_groups"] = len(match)
    return pred, info


def evaluate_quotient_blind(S: torch.Tensor,
                            gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
                            graph_src,
                            graph_tgt,
                            metrics: Optional[Iterable[str]] = None,
                            return_predictions: bool = False,
                            **decode_kwargs):
    """Blind scoring of :func:`quotient_decode` (GT never enters prediction)."""
    pred, info = quotient_decode(S, graph_src, graph_tgt, **decode_kwargs)
    aligned = align_prediction_to_ground_truth(gt_entities, pred)
    scores = many_to_many_scores(gt_entities, aligned, metrics=metrics)
    if return_predictions:
        return scores, pred, info
    return scores


# ---------------------------------------------------------------------------
# Model wrapper (JOENA + quotient decode), mirroring JOENAGroupDecode
# ---------------------------------------------------------------------------
class JOENAQuotientDecode(BaseModel):
    """JOENA aligner + quotient decoding readout.

    Same training path as :class:`PlanetAlign.m2m_decode.JOENAGroupDecode`;
    only the readout differs (adaptive taus, anti-chaining agglomeration,
    Hungarian quotient matching, alternating refinement).
    """

    def __init__(self,
                 tau: Optional[float] = None,
                 max_iters: int = 2,
                 relative_match_threshold: float = 0.0,
                 matcher: str = "hungarian",
                 anti_chaining: bool = True,
                 evidence_selection: bool = True,
                 null_candidate: bool = True,
                 global_candidate: Optional[bool] = None,
                 overlap_expand_tau: Optional[float] = None,
                 alpha: float = 0.7,
                 gamma_p: float = 1e-2,
                 init_lambda: float = 1.0,
                 hid_dim: int = 128,
                 out_dim: int = 128,
                 lr: float = 1e-4,
                 dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        assert matcher in {"hungarian", "greedy"}
        self.decode_kwargs = dict(
            tau=tau, max_iters=max_iters,
            relative_match_threshold=relative_match_threshold,
            matcher=matcher, anti_chaining=anti_chaining,
            evidence_selection=evidence_selection,
            null_candidate=null_candidate,
            global_candidate=global_candidate,
            overlap_expand_tau=overlap_expand_tau,
        )
        self._use_attr = True
        self._joena_kwargs = dict(alpha=alpha, gamma_p=gamma_p, init_lambda=init_lambda,
                                  hid_dim=hid_dim, out_dim=out_dim, lr=lr)
        self._graph_src = None
        self._graph_tgt = None
        self._anchors: Optional[torch.Tensor] = None
        self.decode_info: Optional[Dict[str, object]] = None

    def train(self,
              dataset: Dataset,
              gids: Union[Tuple[int, int], List[int]],
              use_attr: bool = True,
              total_epochs: int = 100,
              save_log: bool = True,
              verbose: bool = True):
        gid1, gid2 = gids
        joena = JOENA(dtype=self.dtype, **self._joena_kwargs).to(self.device)
        S, logger = joena.train(dataset=dataset, gids=gids, use_attr=use_attr,
                                total_epochs=total_epochs, save_log=save_log, verbose=verbose)
        self.S = S.detach().to(self.dtype)
        self._graph_src = dataset.pyg_graphs[gid1]
        self._graph_tgt = dataset.pyg_graphs[gid2]
        self._use_attr = use_attr
        self._anchors = get_anchor_pairs(dataset.train_data, gid1, gid2)
        return self.S, logger

    def predict_entities(self) -> EntityMap:
        if self.S is None:
            raise RuntimeError("Model is not trained yet, call train() first")
        pred, info = quotient_decode(self.S, self._graph_src, self._graph_tgt,
                                     use_attr=self._use_attr, anchors=self._anchors,
                                     **self.decode_kwargs)
        self.decode_info = info
        return pred

    def test_blind(self,
                   gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
                   metrics: Optional[Iterable[str]] = None) -> Dict[str, float]:
        pred = self.predict_entities()
        aligned = align_prediction_to_ground_truth(gt_entities, pred)
        return many_to_many_scores(gt_entities, aligned, metrics=metrics)
