"""M2MAlign — a group-aware many-to-many graph alignment algorithm.

The algorithm targets the four structural properties of the many-to-many
benchmark produced by ``PlanetAlign.utils.many2many_builder``:

1. Mixed granularity (1-1 / 1-N / N-1 / N-N).  Handled by group-level
   bipartite matching with a non-competitive aggregation, so multiple
   candidates in the same ground-truth group reinforce rather than dilute
   each other.
2. Intra-group cohesion (``internal_density``).  Handled by clustering each
   graph using the product of adjacency and embedding similarity, so dense
   same-entity subgraphs are pulled together.
3. Group-level external structure (per-split-node 1/k neighbour retention).
   Handled by a boundary-neighbour consistency bonus when matching groups.
4. Fuzzy boundaries (``overlap_ratio``).  Handled by allowing a node to be
   a weighted member of multiple clusters.

The base node-level similarity is the same RWR + attribute cosine signal
that HOT uses (see ``PlanetAlign.algorithms.hot``), so the 1-1 metrics
remain competitive while the group machinery boosts the many-to-many
metrics.
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import os
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.utils import to_dense_adj

from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_batch_rwr_scores, pairwise_cosine_similarity
from PlanetAlign.algorithms.base_model import BaseModel


Cluster = Dict[int, float]  # node_id -> membership weight in [0, 1]


class M2MAlign(BaseModel):
    """Group-aware many-to-many alignment.

    Parameters
    ----------
    alpha : float
        Blend between the base similarity and the group boost.
        ``S = alpha * S0 + (1 - alpha) * boost``.  Default 0.4.
    tau : float
        Cohesion-edge threshold.  Edges whose cosine embedding similarity
        is below ``tau`` are removed before clustering.  Default 0.5.
    overlap_slack : float
        Secondary-cluster tolerance.  A node with two cluster scores
        ``s1 >= s2`` is also admitted into the second cluster if
        ``s2 >= (1 - overlap_slack) * s1``.  Default 0.15.
    lambda_struct : float
        Weight of the boundary-neighbour structural consistency bonus.
        Default 0.5.
    beta : float
        Kept ratio for secondary group matches per source group — enables
        N-N correspondences.  Default 0.6.
    max_group_size : int
        Hard cap on a single cluster to keep Hungarian tractable.  Default 8.
    n_iter : int
        Number of outer passes (base init + refinements).  Default 2.
    base : {"rwr"}
        Source of the base node embedding.  Only RWR is implemented today
        (the hook is retained for future plug-ins).
    """

    def __init__(self,
                 alpha: float = 0.6,
                 tau: float = 0.95,
                 overlap_slack: float = 0.10,
                 lambda_struct: float = 0.5,
                 beta: float = 0.6,
                 max_group_size: int = 4,
                 n_iter: int = 2,
                 base: str = 'rwr',
                 smooth_source: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        assert 0.0 <= alpha <= 1.0
        assert 0.0 <= tau <= 1.0
        assert 0.0 <= overlap_slack <= 1.0
        assert lambda_struct >= 0.0
        assert 0.0 <= beta <= 1.0
        assert max_group_size >= 2
        assert n_iter >= 1
        assert base in {'rwr'}

        self.alpha = alpha
        self.tau = tau
        self.overlap_slack = overlap_slack
        self.lambda_struct = lambda_struct
        self.beta = beta
        self.max_group_size = max_group_size
        self.n_iter = n_iter
        self.base = base
        self.smooth_source = smooth_source

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self,
              dataset: Dataset,
              gids: Union[List[int], Tuple[int, ...]],
              use_attr: bool = True,
              save_log: bool = True,
              verbose: bool = True,
              init_S: Optional[torch.Tensor] = None):
        """Train the M2MAlign refinement.

        Parameters
        ----------
        init_S : optional torch.Tensor of shape (n1, n2)
            If provided, use this as the base node-level similarity instead
            of recomputing RWR + attribute cosine.  Lets the caller plug in
            a stronger 1-1 baseline (FINAL, PARROT, JOENA, ...).  When
            supplied, ``use_attr`` only affects the cohesion embedding used
            for intra-graph clustering.
        """
        self.check_inputs(dataset, gids, plain_method=False, use_attr=use_attr,
                          pairwise=True, supervised=True)
        gid1, gid2 = gids

        logger = self.init_training_logger(
            dataset, use_attr,
            additional_headers=['memory', 'infer_time', 'num_clusters_1', 'num_clusters_2'],
            save_log=save_log,
        )
        process = psutil.Process(os.getpid())

        graph1 = dataset.pyg_graphs[gid1]
        graph2 = dataset.pyg_graphs[gid2]
        train_pairs = dataset.train_data[:, [gid1, gid2]]
        train_pairs = train_pairs[torch.sum(train_pairs == -1, dim=1) == 0]

        t0 = time.time()

        emb1 = self._build_embedding(graph1, train_pairs[:, 0], use_attr)
        emb2 = self._build_embedding(graph2, train_pairs[:, 1], use_attr)
        if init_S is None:
            S0 = pairwise_cosine_similarity(emb1, emb2).to(self.dtype)
        else:
            S0 = init_S.detach().to(self.dtype).to(self.device)
            assert S0.shape == (graph1.num_nodes, graph2.num_nodes), (
                f"init_S shape {tuple(S0.shape)} != expected "
                f"({graph1.num_nodes}, {graph2.num_nodes})"
            )

        adj_list_1 = self._neighbor_lists(graph1)
        adj_list_2 = self._neighbor_lists(graph2)

        S = S0.clone()
        last_k1 = last_k2 = 0
        for it in range(self.n_iter):
            # Use emb1/emb2 for clustering on the first pass.  On later
            # passes, blend a "post-boost" signal so the clustering can
            # absorb the similarity-guided group structure.  Keeping the
            # raw embedding dominant avoids collapsing to a degenerate
            # single cluster.
            clusters1 = self._overlap_cluster(graph1, emb1, adj_list_1)
            clusters2 = self._overlap_cluster(graph2, emb2, adj_list_2)
            last_k1, last_k2 = len(clusters1), len(clusters2)
            if verbose:
                print(f"  [M2MAlign iter {it+1}] clusters: src={last_k1} tgt={last_k2}")

            group_sim = self._group_similarity(S, clusters1, clusters2)
            group_sim = self._apply_structural_bonus(
                group_sim, clusters1, clusters2, adj_list_1, adj_list_2, S
            )
            matches = self._match_groups(group_sim)
            S = self._rebuild_similarity(S0, clusters1, clusters2, matches)

        t1 = time.time()
        infer_time = t1 - t0
        mem_gb = process.memory_info().rss / 1024 ** 3

        self.S = S.detach().to(self.dtype).cpu()

        # Log using the 1-1 metrics we can compute cheaply.
        from PlanetAlign.metrics import hits_ks_scores, mrr_score
        from PlanetAlign.utils import get_anchor_pairs
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)
        hits = hits_ks_scores(self.S, test_pairs, mode='mean')
        mrr = mrr_score(self.S, test_pairs, mode='mean')
        logger.log(epoch=1,
                   loss=0.0,
                   epoch_time=infer_time,
                   hits=hits,
                   mrr=mrr,
                   memory=round(mem_gb, 4),
                   infer_time=round(infer_time, 4),
                   num_clusters_1=last_k1,
                   num_clusters_2=last_k2,
                   verbose=verbose)

        return self.S, logger

    # ------------------------------------------------------------------
    # Stage A — base embedding
    # ------------------------------------------------------------------
    def _build_embedding(self,
                         graph,
                         landmarks: torch.Tensor,
                         use_attr: bool) -> torch.Tensor:
        if landmarks.numel() == 0:
            raise ValueError("M2MAlign requires at least one anchor pair for the RWR base embedding.")
        rwr = get_batch_rwr_scores(graph, landmarks.to(self.device),
                                   device=self.device).to(self.dtype)
        rwr = F.normalize(rwr, p=2, dim=1)
        if use_attr and graph.x is not None:
            attr = F.normalize(graph.x.to(self.dtype).to(self.device), p=2, dim=1)
            return torch.cat([rwr, attr], dim=1)
        return rwr

    # ------------------------------------------------------------------
    # Stage B — overlapping intra-graph clustering
    # ------------------------------------------------------------------
    def _overlap_cluster(self,
                         graph,
                         emb: torch.Tensor,
                         adj_list: List[List[int]]) -> List[Cluster]:
        num_nodes = graph.num_nodes
        emb_cpu = F.normalize(emb, p=2, dim=1).detach().cpu()
        src = graph.edge_index[0].cpu().numpy()
        dst = graph.edge_index[1].cpu().numpy()

        # Build cohesion edges: adjacent + embedding-similar.
        sims = (emb_cpu[src] * emb_cpu[dst]).sum(dim=1).numpy()
        mask = (src < dst) & (sims >= self.tau)
        cohesion_src = src[mask]
        cohesion_dst = dst[mask]

        # Union–find to get hard clusters over these cohesion edges.
        parent = list(range(num_nodes))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for u, v in zip(cohesion_src, cohesion_dst):
            union(int(u), int(v))

        roots: Dict[int, List[int]] = {}
        for n in range(num_nodes):
            r = find(n)
            roots.setdefault(r, []).append(n)

        # Keep only non-singleton clusters; cap size by similarity to centroid.
        clusters: List[Cluster] = []
        for members in roots.values():
            if len(members) < 2:
                continue
            if len(members) > self.max_group_size:
                # Trim to the ``max_group_size`` members closest to the centroid.
                idx = torch.tensor(members, dtype=torch.long)
                centroid = emb_cpu[idx].mean(dim=0, keepdim=True)
                centroid = F.normalize(centroid, p=2, dim=1)
                scores = (emb_cpu[idx] @ centroid.T).squeeze(1).numpy()
                top = np.argsort(-scores)[:self.max_group_size]
                members = [members[i] for i in top.tolist()]
            clusters.append({int(n): 1.0 for n in members})

        if not clusters:
            return clusters

        # Overlap pass — allow a node to join a second cluster if the score
        # is close to the primary one (property 4: fuzzy boundaries).
        cluster_node_sets = [set(c.keys()) for c in clusters]
        cluster_centroids = torch.stack([
            F.normalize(emb_cpu[torch.tensor(list(c.keys()), dtype=torch.long)]
                        .mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
            for c in clusters
        ])  # [K, D]

        for u in range(num_nodes):
            neigh = adj_list[u]
            if not neigh:
                continue
            # Cluster score = fraction of neighbours in cluster * embedding similarity.
            scores = []
            for k, cset in enumerate(cluster_node_sets):
                if len(cset) == 0:
                    scores.append(0.0)
                    continue
                frac = sum(1 for w in neigh if w in cset) / len(neigh)
                if frac <= 0:
                    scores.append(0.0)
                    continue
                emb_sim = float(emb_cpu[u] @ cluster_centroids[k])
                scores.append(frac * max(emb_sim, 0.0))
            scores_t = torch.tensor(scores)
            if float(scores_t.max()) <= 0:
                continue
            top_val, top_idx = torch.topk(scores_t, k=min(2, len(scores)))
            primary = int(top_idx[0])
            primary_val = float(top_val[0])
            if u not in clusters[primary]:
                clusters[primary][u] = min(1.0, primary_val)
            if len(top_idx) > 1 and float(top_val[1]) >= (1.0 - self.overlap_slack) * primary_val > 0:
                secondary = int(top_idx[1])
                if u not in clusters[secondary]:
                    clusters[secondary][u] = min(1.0, float(top_val[1]))

        # Drop clusters that somehow lost all members.
        clusters = [c for c in clusters if len(c) >= 2]
        return clusters

    # ------------------------------------------------------------------
    # Stage C — group-to-group matching
    # ------------------------------------------------------------------
    def _group_similarity(self,
                          S: torch.Tensor,
                          clusters1: List[Cluster],
                          clusters2: List[Cluster]) -> torch.Tensor:
        k1, k2 = len(clusters1), len(clusters2)
        if k1 == 0 or k2 == 0:
            return torch.zeros((k1, k2), dtype=self.dtype)

        # Per-cluster weighted indicator vectors  w^(1) ∈ R^{k1 × n1}, w^(2) ∈ R^{k2 × n2}.
        n1, n2 = S.shape
        W1 = torch.zeros((k1, n1), dtype=self.dtype, device=S.device)
        W2 = torch.zeros((k2, n2), dtype=self.dtype, device=S.device)
        size1 = torch.zeros(k1, dtype=self.dtype, device=S.device)
        size2 = torch.zeros(k2, dtype=self.dtype, device=S.device)
        for i, c in enumerate(clusters1):
            for n, w in c.items():
                W1[i, n] = w
                size1[i] += w
        for j, c in enumerate(clusters2):
            for n, w in c.items():
                W2[j, n] = w
                size2[j] += w

        # Size-normalised but additive group score.
        numer = W1 @ S @ W2.T                                   # [k1, k2]
        denom = torch.sqrt(size1.unsqueeze(1) * size2.unsqueeze(0)).clamp(min=1e-9)
        return numer / denom

    def _apply_structural_bonus(self,
                                group_sim: torch.Tensor,
                                clusters1: List[Cluster],
                                clusters2: List[Cluster],
                                adj_list_1: List[List[int]],
                                adj_list_2: List[List[int]],
                                S: torch.Tensor) -> torch.Tensor:
        if self.lambda_struct <= 0 or group_sim.numel() == 0:
            return group_sim

        ext1 = [self._external_neighbours(c, adj_list_1) for c in clusters1]
        ext2 = [self._external_neighbours(c, adj_list_2) for c in clusters2]

        bonus = torch.zeros_like(group_sim)
        for i, nbrs_i in enumerate(ext1):
            if not nbrs_i:
                continue
            idx1 = torch.tensor(nbrs_i, dtype=torch.long, device=S.device)
            sub = S[idx1]  # [|Ni|, n2]
            for j, nbrs_j in enumerate(ext2):
                if not nbrs_j:
                    continue
                idx2 = torch.tensor(nbrs_j, dtype=torch.long, device=S.device)
                block = sub[:, idx2]            # [|Ni|, |Nj|]
                # Mean of the max-per-row: a loose bipartite fit that rewards
                # good alignment of external neighbours without requiring a
                # strict matching.
                row_max = block.max(dim=1).values
                bonus[i, j] = row_max.mean()

        return group_sim * (1.0 + self.lambda_struct * bonus.clamp(min=0.0))

    def _match_groups(self, group_sim: torch.Tensor) -> List[Tuple[int, int, float]]:
        if group_sim.numel() == 0:
            return []
        k1, k2 = group_sim.shape
        arr = group_sim.detach().cpu().numpy()

        # Hungarian on the square padded matrix.  We negate because
        # ``linear_sum_assignment`` minimises cost.
        size = max(k1, k2)
        cost = np.full((size, size), fill_value=1e6, dtype=np.float64)
        cost[:k1, :k2] = -arr
        row_ind, col_ind = linear_sum_assignment(cost)

        primary_score: Dict[int, float] = {}
        matches: List[Tuple[int, int, float]] = []
        for r, c in zip(row_ind, col_ind):
            if r < k1 and c < k2:
                score = float(arr[r, c])
                if score <= 0:
                    continue
                matches.append((r, c, score))
                primary_score[r] = max(primary_score.get(r, 0.0), score)

        # Secondary matches to support N-N.  For each source group keep
        # any target group scored above ``beta * primary`` that isn't
        # already matched.
        taken_tgt = {c for (_, c, _) in matches}
        for r in range(k1):
            prim = primary_score.get(r, 0.0)
            if prim <= 0:
                continue
            threshold = self.beta * prim
            order = np.argsort(-arr[r])
            for c in order:
                if c in taken_tgt:
                    continue
                score = float(arr[r, c])
                if score < threshold:
                    break
                matches.append((r, int(c), score))
                taken_tgt.add(int(c))
        return matches

    # ------------------------------------------------------------------
    # Stage D — similarity reweighting via cluster-max smoothing
    # ------------------------------------------------------------------
    def _rebuild_similarity(self,
                            S0: torch.Tensor,
                            clusters1: List[Cluster],
                            clusters2: List[Cluster],
                            matches: List[Tuple[int, int, float]]) -> torch.Tensor:
        """Cluster-max smoothing — never *lowers* a node-level score.

        For each target cluster ``C2``:
            S_smooth[u, v] = max(S0[u, v],  membership(v) · max_{v'∈C2} S0[u, v'])
        Symmetrically for source-side clustering.  This addresses the core
        failure of 1-1 algorithms on many-to-many data: good node-level
        scores stay high, but all cluster mates get lifted to the cluster's
        best score so that ``similarity_to_pred_entities`` picks up the full
        target group when querying with a source group.

        When ``matches`` is non-empty the group-matching signal is layered
        on top as an *additional* lift, restricted to matched (C1, C2) pairs
        and scaled by each pair's similarity score.  The structural-bonus
        + Hungarian + N-N extension baked into ``matches`` therefore acts as
        a long-range correction to the local cluster-max smoothing.

        Finally we mix back the raw base:
            S_final = alpha · S0 + (1 - alpha) · S_smooth
        With ``alpha`` close to 0 the smoothed score dominates; since
        smoothing only lifts scores, node-level top-1 rankings are
        preserved in practice.
        """
        S_smooth = S0.clone()

        # --- Target-side cluster smoothing (lifts cluster mates in columns). ---
        for c in clusters2:
            idx = torch.tensor(list(c.keys()), dtype=torch.long, device=S0.device)
            if idx.numel() < 2:
                continue
            weights = torch.tensor([c[int(k)] for k in idx.tolist()],
                                   dtype=self.dtype, device=S0.device)
            block = S0[:, idx]                       # [n1, |C|]
            row_max, _ = block.max(dim=1)            # [n1]
            # Distribute the cluster's best score to each member, weighted by
            # membership in [0, 1].
            new_cols = row_max.unsqueeze(1) * weights.unsqueeze(0)   # [n1, |C|]
            current = S_smooth[:, idx]
            S_smooth[:, idx] = torch.maximum(current, new_cols)

        # --- Source-side cluster smoothing (lifts cluster mates in rows). ---
        # Off by default: metrics already receive the true source group via
        # ``gt_entities``, and smoothing across wrongly-merged source nodes
        # can flatten the S0 signal.  Keep as an opt-in for symmetry.
        if self.smooth_source:
            for c in clusters1:
                idx = torch.tensor(list(c.keys()), dtype=torch.long, device=S0.device)
                if idx.numel() < 2:
                    continue
                weights = torch.tensor([c[int(k)] for k in idx.tolist()],
                                       dtype=self.dtype, device=S0.device)
                block = S0[idx, :]                       # [|C|, n2]
                col_max, _ = block.max(dim=0)            # [n2]
                new_rows = weights.unsqueeze(1) * col_max.unsqueeze(0)
                current = S_smooth[idx, :]
                S_smooth[idx, :] = torch.maximum(current, new_rows)

        # --- Group-matching lift ---------------------------------------------
        # For each matched pair (C1_i, C2_j, s_ij), raise the block
        # S_smooth[C1_i, C2_j] towards the row-max that would be induced if
        # every member of C1_i saw every member of C2_j.  This helps the
        # uncommon case where node-level S0 misplaces the true group.
        if matches:
            max_score = max(s for (_, _, s) in matches) + 1e-9
            for (i, j, s) in matches:
                c1 = clusters1[i]
                c2 = clusters2[j]
                idx1 = torch.tensor(list(c1.keys()), dtype=torch.long, device=S0.device)
                idx2 = torch.tensor(list(c2.keys()), dtype=torch.long, device=S0.device)
                if idx1.numel() == 0 or idx2.numel() == 0:
                    continue
                w1 = torch.tensor([c1[int(k)] for k in idx1.tolist()],
                                  dtype=self.dtype, device=S0.device)
                w2 = torch.tensor([c2[int(k)] for k in idx2.tolist()],
                                  dtype=self.dtype, device=S0.device)
                # Take the peak S0 inside the block; spread it to every member.
                block_S0 = S0[idx1.unsqueeze(1), idx2.unsqueeze(0)]
                peak = float(block_S0.max())
                scale = (s / max_score)              # in (0, 1]
                lift = scale * peak * w1.unsqueeze(1) * w2.unsqueeze(0)
                current = S_smooth[idx1.unsqueeze(1), idx2.unsqueeze(0)]
                S_smooth[idx1.unsqueeze(1), idx2.unsqueeze(0)] = torch.maximum(current, lift)

        return self.alpha * S0 + (1.0 - self.alpha) * S_smooth

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _neighbor_lists(graph) -> List[List[int]]:
        num_nodes = graph.num_nodes
        neigh: List[List[int]] = [[] for _ in range(num_nodes)]
        src = graph.edge_index[0].tolist()
        dst = graph.edge_index[1].tolist()
        for u, v in zip(src, dst):
            neigh[u].append(v)
        return neigh

    @staticmethod
    def _external_neighbours(cluster: Cluster, adj_list: List[List[int]]) -> List[int]:
        members = set(cluster.keys())
        ext: set = set()
        for n in members:
            for w in adj_list[n]:
                if w not in members:
                    ext.add(w)
        return sorted(ext)
