# 文件作用：实现 TGAE 图自编码对齐，训练共享编码器重构两图并产生节点相似度。
# 核心依赖：PlanetAlign/algorithms/tgae/model.py → TGAENetwork；PlanetAlign/utils/anchors.py → get_anchor_pairs() 提取可选监督。
# 结果去向：train() 返回 self.S 与日志，scripts/run_m2m_experiments.py → evaluate_algorithm() 可将它交给实体解码。
# 阅读建议：先看 train() → _refresh_alignment()，再看重构损失与 _build_features()。

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import psutil
import torch
import torch.nn.functional as F

from PlanetAlign.algorithms.base_model import BaseModel
from PlanetAlign.data import Dataset
from PlanetAlign.metrics import hits_ks_scores, mrr_score
from PlanetAlign.utils import get_anchor_pairs

from .model import TGAENetwork


# 功能：提供共享图编码器与可选训练锚点损失的一对一矩阵接口。
class TGAE(BaseModel):
    """Transferable Graph Autoencoder for pairwise network alignment.

    This class adapts the T-GAE graph autoencoder into PlanetAlign's standard
    ``BaseModel`` interface. It trains the shared encoder by reconstructing
    each input graph, then aligns nodes by embedding distance. Its final
    similarity matrix ``S`` can be passed to a blind M2M decoder.
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

    # 输入来源：Dataset 的图边与可选属性生成编码输入；可选监督只使用 train_data，test_data 用于日志。
    # 核心思路：缓存图特征、归一化邻接与边，按边重构训练网络，并可叠加训练锚点的对齐损失。
    # 输出结果：_refresh_alignment() 更新 self.S，最终返回矩阵与日志。
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

    # 功能：在评估模式下重算两侧嵌入，将向量距离的负指数保存为 S。
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

    # 功能：将真实边和随机负例边的嵌入内积分成正负标签，计算重构损失。
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

    # 功能：以训练锚点构造双向分类目标，让对应节点嵌入相近。
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

    # 功能：组合结构统计与可选归一化节点属性，供共享编码器输入。
    def _build_features(self, graph, use_attr: bool) -> torch.Tensor:
        structural = self._structural_features(graph).to(self.dtype)
        if use_attr and graph.x is not None:
            attr = graph.x.to(self.dtype).cpu()
            attr = F.normalize(attr, p=2, dim=1)
            return torch.cat([structural, attr], dim=1)
        return structural

    # 功能：按度数、邻居度数统计与局部连接等构造可跨图比较的标准化结构特征。
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

    # 输出结果：加自环并归一化的稀疏邻接矩阵，供图编码器聚合邻居。
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
