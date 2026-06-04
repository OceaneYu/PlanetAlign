import os
import time
from typing import Dict, List, Optional, Tuple, Union

import psutil
import torch
import torch.nn.functional as F

from PlanetAlign.algorithms.base_model import BaseModel
from PlanetAlign.data import Dataset
from PlanetAlign.metrics import hits_ks_scores, mrr_score, similarity_to_pred_entities
from PlanetAlign.utils import get_anchor_pairs

from .model import TGAENetwork


class TGAE(BaseModel):
    """Transferable Graph Autoencoder for pairwise network alignment.

    This class adapts the T-GAE graph autoencoder into PlanetAlign's standard
    ``BaseModel`` interface. It trains the shared encoder by reconstructing
    each input graph, then aligns nodes by embedding distance. The
    ``predict_many_to_many`` method ports T-GAE's local target-side expansion
    idea for many-to-many evaluation.
    """

    def __init__(self,
                 num_hidden_layers: int = 8,
                 hidden_dim: int | List[int] = 16,
                 output_dim: int = 8,
                 lr: float = 1e-3,
                 weight_decay: float = 5e-4,
                 anchor_loss_weight: float = 0.0,
                 anchor_temperature: float = 0.1,
                 similarity: str = "cosine",
                 reconstruction_neg_ratio: float = 1.0,
                 dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        assert num_hidden_layers >= 1
        assert output_dim >= 1
        assert lr > 0
        assert weight_decay >= 0
        assert anchor_loss_weight >= 0
        assert anchor_temperature > 0
        assert similarity in {"cosine", "neg_exp_dist"}
        assert reconstruction_neg_ratio > 0
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.anchor_loss_weight = anchor_loss_weight
        self.anchor_temperature = anchor_temperature
        self.similarity = similarity
        self.reconstruction_neg_ratio = reconstruction_neg_ratio
        self.model: Optional[TGAENetwork] = None
        self.embeddings: Dict[int, torch.Tensor] = {}
        self._last_gids: Tuple[int, int] | None = None

    def train(self,
              dataset: Dataset,
              gids: Union[List[int], Tuple[int, ...]],
              use_attr: bool = False,
              total_epochs: int = 50,
              eval_interval: int = 1,
              save_log: bool = True,
              verbose: bool = True):
        self.check_inputs(dataset, gids, plain_method=False, use_attr=use_attr,
                          pairwise=True, supervised=True)
        assert total_epochs >= 1
        assert eval_interval >= 1
        gid1, gid2 = gids
        self._last_gids = (gid1, gid2)

        logger = self.init_training_logger(
            dataset,
            use_attr,
            additional_headers=["memory", "infer_time"],
            save_log=save_log,
        )
        process = psutil.Process(os.getpid())

        caches = []
        input_dim = None
        for gid in gids:
            graph = dataset.pyg_graphs[gid]
            features = self._build_features(graph, use_attr).to(self.device)
            if input_dim is None:
                input_dim = features.shape[1]
            elif features.shape[1] != input_dim:
                raise ValueError("TGAE requires the aligned graphs to have the same input feature dimension")

            adj_norm = self._normalized_sparse_adj(graph).to(self.device)
            pos_edges = graph.edge_index.detach().long().to(self.device)
            non_loop = pos_edges[0] != pos_edges[1]
            pos_edges = pos_edges[:, non_loop]
            caches.append((gid, features, adj_norm, pos_edges, graph.num_nodes))

        self.model = TGAENetwork(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_hidden_layers=self.num_hidden_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        train_t0 = time.time()
        train_pairs = get_anchor_pairs(dataset.train_data, gid1, gid2).to(self.device)
        for epoch in range(1, total_epochs + 1):
            t0 = time.time()
            loss = torch.zeros((), dtype=self.dtype, device=self.device)
            z_by_gid = {}
            for gid, features, adj_norm, pos_edges, num_nodes in caches:
                z = self.model(features, adj_norm)
                z_by_gid[gid] = z
                loss = loss + self._sampled_reconstruction_loss(z, pos_edges, num_nodes)
            loss = loss / len(caches)
            if self.anchor_loss_weight > 0 and train_pairs.numel() > 0:
                loss = loss + self.anchor_loss_weight * self._anchor_alignment_loss(
                    z_by_gid[gid1],
                    z_by_gid[gid2],
                    train_pairs,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_time = time.time() - t0
            infer_time = time.time() - train_t0
            if epoch % eval_interval == 0 or epoch == total_epochs:
                self._refresh_alignment(dataset, gid1, gid2, caches)
                test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)
                hits = hits_ks_scores(self.S, test_pairs, mode="mean")
                mrr = mrr_score(self.S, test_pairs, mode="mean")
                mem_gb = process.memory_info().rss / 1024 ** 3
                logger.log(
                    epoch=epoch,
                    loss=float(loss.item()),
                    epoch_time=epoch_time,
                    hits=hits,
                    mrr=mrr,
                    memory=round(mem_gb, 4),
                    infer_time=round(infer_time, 4),
                    verbose=verbose,
                )
            elif verbose:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f} | EpochTime: {epoch_time:.2f}s")

        self._refresh_alignment(dataset, gid1, gid2, caches)
        return self.S, logger

    def predict_many_to_many(self,
                             gt_entities: Dict[str, Dict[str, List[int]]],
                             source_key: str = "src",
                             target_key: str = "tgt",
                             mode: str = "local_expand",
                             relax_ratio: float = 1.03,
                             target_dup_sim_ratio: float = 1.03,
                             max_extra_targets: int = 3) -> Dict[str, Dict[str, List[int]]]:
        """Build many-to-many predictions from the trained similarity matrix.

        ``mode='local_expand'`` follows the T-GAE M2M extension: choose a
        strongest target-side backbone match, then add nearby target nodes that
        are also close to the source group. ``mode='gt_size_topk'`` uses the
        repository's common top-k conversion for comparison with other models.
        """
        if self.S is None:
            raise RuntimeError("Model is not trained yet, call train() first")
        if mode == "gt_size_topk":
            return similarity_to_pred_entities(
                self.S,
                gt_entities,
                source_key=source_key,
                target_key=target_key,
            )
        if mode != "local_expand":
            raise ValueError("mode must be either 'local_expand' or 'gt_size_topk'")
        if relax_ratio < 1.0:
            raise ValueError("relax_ratio must be >= 1.0")
        if target_dup_sim_ratio < 1.0:
            raise ValueError("target_dup_sim_ratio must be >= 1.0")
        if max_extra_targets < 0:
            raise ValueError("max_extra_targets must be non-negative")

        similarity = self.S.detach().cpu()
        if len(self.embeddings) < 2:
            raise RuntimeError("Embeddings are unavailable")
        if self._last_gids is None:
            raise RuntimeError("Model is not trained yet, call train() first")
        target_emb = self.embeddings[self._last_gids[1]].detach().cpu()
        target_dist = torch.cdist(target_emb, target_emb, p=2)
        pred_entities: Dict[str, Dict[str, List[int]]] = {}

        for eid, gt_item in gt_entities.items():
            pred_item = {
                key: [int(v) for v in value] if isinstance(value, list) else value
                for key, value in gt_item.items()
            }
            src_nodes = [int(n) for n in gt_item.get(source_key, [])]
            src_nodes = [n for n in src_nodes if 0 <= n < similarity.shape[0]]
            if not src_nodes:
                pred_item[target_key] = []
                pred_entities[eid] = pred_item
                continue

            group_scores = similarity[src_nodes].mean(dim=0)
            base_target = int(torch.argmax(group_scores).item())
            targets = [base_target]
            if max_extra_targets > 0:
                source_dist = -torch.log(group_scores.clamp_min(1e-12))
                base_source_dist = float(source_dist[base_target].item())
                base_target_dist = target_dist[base_target]
                order = torch.argsort(base_target_dist).tolist()
                for cand in order:
                    cand = int(cand)
                    if cand == base_target:
                        continue
                    close_to_source = float(source_dist[cand].item()) <= base_source_dist * relax_ratio
                    nearest_target_dist = float(base_target_dist[cand].item())
                    if nearest_target_dist <= 0:
                        close_to_target = False
                    else:
                        best_non_self = base_target_dist[base_target_dist > 0]
                        ref_dist = float(best_non_self.min().item()) if best_non_self.numel() else nearest_target_dist
                        close_to_target = nearest_target_dist <= ref_dist * target_dup_sim_ratio
                    if close_to_source and close_to_target:
                        targets.append(cand)
                    if len(targets) >= max_extra_targets + 1:
                        break

            pred_item[target_key] = targets
            pred_entities[eid] = pred_item

        return pred_entities

    def _refresh_alignment(self, dataset, gid1, gid2, caches):
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            embeddings = {}
            for gid, features, adj_norm, _, _ in caches:
                embeddings[gid] = self.model(features, adj_norm).detach().to(torch.float32).cpu()
            emb1 = embeddings[gid1]
            emb2 = embeddings[gid2]
            if self.similarity == "cosine":
                self.S = F.normalize(emb1, p=2, dim=1) @ F.normalize(emb2, p=2, dim=1).T
            else:
                self.S = torch.exp(-torch.cdist(emb1, emb2, p=2))
            self.embeddings = embeddings
        self.model.train()

    def _sampled_reconstruction_loss(self,
                                     emb: torch.Tensor,
                                     pos_edges: torch.Tensor,
                                     num_nodes: int) -> torch.Tensor:
        num_pos = pos_edges.shape[1]
        num_neg = max(1, int(num_pos * self.reconstruction_neg_ratio))
        neg_src = torch.randint(0, num_nodes, (num_neg,), device=emb.device)
        neg_dst = torch.randint(0, num_nodes, (num_neg,), device=emb.device)
        non_loop = neg_src != neg_dst
        if not torch.all(non_loop):
            neg_src = neg_src[non_loop]
            neg_dst = neg_dst[non_loop]
            while neg_src.numel() < num_neg:
                extra = num_neg - neg_src.numel()
                src = torch.randint(0, num_nodes, (extra,), device=emb.device)
                dst = torch.randint(0, num_nodes, (extra,), device=emb.device)
                mask = src != dst
                neg_src = torch.cat([neg_src, src[mask]])
                neg_dst = torch.cat([neg_dst, dst[mask]])
            neg_src = neg_src[:num_neg]
            neg_dst = neg_dst[:num_neg]

        pos_logits = (emb[pos_edges[0]] * emb[pos_edges[1]]).sum(dim=1)
        neg_logits = (emb[neg_src] * emb[neg_dst]).sum(dim=1)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits),
        ])
        return F.binary_cross_entropy_with_logits(logits, labels)

    def _anchor_alignment_loss(self,
                               emb1: torch.Tensor,
                               emb2: torch.Tensor,
                               train_pairs: torch.Tensor) -> torch.Tensor:
        src = train_pairs[:, 0].long()
        tgt = train_pairs[:, 1].long()
        z1 = F.normalize(emb1, p=2, dim=1)
        z2 = F.normalize(emb2, p=2, dim=1)

        logits_ltr = z1[src] @ z2.T / self.anchor_temperature
        loss_ltr = F.cross_entropy(logits_ltr, tgt)

        unique_tgt, inverse = torch.unique(tgt, sorted=True, return_inverse=True)
        target_to_src = torch.full(
            (unique_tgt.numel(),),
            -1,
            dtype=torch.long,
            device=train_pairs.device,
        )
        # Keep the first source for each target if train data contains duplicates.
        for row_idx, bucket_idx in enumerate(inverse.tolist()):
            if target_to_src[bucket_idx] < 0:
                target_to_src[bucket_idx] = src[row_idx]
        logits_rtl = z2[unique_tgt] @ z1.T / self.anchor_temperature
        loss_rtl = F.cross_entropy(logits_rtl, target_to_src)
        return 0.5 * (loss_ltr + loss_rtl)

    def _build_features(self, graph, use_attr: bool) -> torch.Tensor:
        structural = self._structural_features(graph).to(self.dtype)
        if use_attr and graph.x is not None:
            attr = graph.x.to(self.dtype).cpu()
            attr = F.normalize(attr, p=2, dim=1)
            return torch.cat([structural, attr], dim=1)
        return structural

    def _structural_features(self, graph) -> torch.Tensor:
        num_nodes = graph.num_nodes
        src = graph.edge_index[0].detach().cpu().long()
        dst = graph.edge_index[1].detach().cpu().long()
        values = torch.ones(src.numel(), dtype=torch.float32)
        adj = torch.sparse_coo_tensor(
            torch.stack([src, dst]),
            values,
            (num_nodes, num_nodes),
        ).coalesce()

        deg = torch.sparse.sum(adj, dim=1).to_dense().float()
        neigh_deg_sum = torch.sparse.mm(adj, deg.unsqueeze(1)).squeeze(1)
        safe_deg = deg.clamp_min(1.0)
        neigh_deg_mean = neigh_deg_sum / safe_deg

        neigh_deg_sq_sum = torch.sparse.mm(adj, (deg ** 2).unsqueeze(1)).squeeze(1)
        neigh_deg_var = (neigh_deg_sq_sum / safe_deg) - neigh_deg_mean ** 2
        neigh_deg_std = torch.sqrt(torch.clamp(neigh_deg_var, min=0.0))

        two_hop_signal = torch.sparse.mm(adj, neigh_deg_mean.unsqueeze(1)).squeeze(1) / safe_deg
        reciprocal_signal = torch.zeros(num_nodes, dtype=torch.float32)
        edge_pairs = set(zip(src.tolist(), dst.tolist()))
        for u, v in edge_pairs:
            if (v, u) in edge_pairs:
                reciprocal_signal[u] += 1.0

        features = torch.stack([
            torch.log1p(deg),
            torch.log1p(neigh_deg_mean),
            torch.log1p(neigh_deg_std),
            torch.log1p(two_hop_signal),
            deg / max(float(num_nodes - 1), 1.0),
            neigh_deg_mean / max(float(num_nodes - 1), 1.0),
            reciprocal_signal / safe_deg,
        ], dim=1)
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (features - mean) / std

    def _normalized_sparse_adj(self, graph) -> torch.Tensor:
        num_nodes = graph.num_nodes
        src = graph.edge_index[0].detach().cpu().long()
        dst = graph.edge_index[1].detach().cpu().long()
        loops = torch.arange(num_nodes, dtype=torch.long)
        row = torch.cat([src, loops])
        col = torch.cat([dst, loops])
        values = torch.ones(row.numel(), dtype=self.dtype)

        deg = torch.zeros(num_nodes, dtype=self.dtype)
        deg.index_add_(0, row, values)
        deg_inv_sqrt = deg.clamp_min(1).pow(-0.5)
        norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(
            torch.stack([row, col]),
            norm_values,
            (num_nodes, num_nodes),
            dtype=self.dtype,
        ).coalesce()
