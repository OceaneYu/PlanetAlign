# 文件作用：实现 GADL 的图预处理、双通道编码网络和训练目标。
# 数据来源：PlanetAlign/algorithms/gadl/main.py → GADL.train() 传入图边、节点属性及训练权重。
# 核心处理：低通分支汇总邻居共性，高通分支保留差异；同时约束图重构与跨图谱空间映射。
# 结果去向：gadl_objective() 返回损失字典给训练循环；encode_pair() 的嵌入用于生成最终 S。

"""Mathematical components for Graph Alignment via Dual-Pass Learning.

This module is an independent implementation of the GADL formulation.  It
contains no dataset loading, assignment, evaluation, or reference-repository
compatibility code.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as scipy_sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


# 功能：去自环、对称化并合并重复边，返回无向二值稀疏邻接矩阵。
def edge_index_to_undirected_binary_adjacency(
        edge_index: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype = torch.float32,
        device=None) -> torch.Tensor:
    """Build a loop-free, undirected, binary sparse adjacency matrix."""
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index must be a torch.Tensor")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if edge_index.is_floating_point():
        if not torch.isfinite(edge_index).all():
            raise ValueError("edge_index contains non-finite values")
        if not torch.equal(edge_index, edge_index.round()):
            raise ValueError("edge_index must contain integer node indices")

    target_device = torch.device(device) if device is not None else edge_index.device
    indices = edge_index.detach().to(device=target_device, dtype=torch.long)
    if indices.numel() > 0:
        if torch.any(indices < 0) or torch.any(indices >= num_nodes):
            raise ValueError("edge_index contains a node index outside [0, num_nodes)")
        indices = indices[:, indices[0] != indices[1]]
        indices = torch.cat([indices, indices.flip(0)], dim=1)

    values = torch.ones(indices.shape[1], dtype=dtype, device=target_device)
    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_nodes),
        dtype=dtype,
        device=target_device,
    ).coalesce()

    # Coalescing sums duplicate and already-bidirectional edges.  Resetting all
    # surviving values to one restores binary adjacency semantics.
    if adjacency._nnz() > 0:
        adjacency = torch.sparse_coo_tensor(
            adjacency.indices(),
            torch.ones_like(adjacency.values()),
            adjacency.size(),
            dtype=dtype,
            device=target_device,
        ).coalesce()
    return adjacency


# 功能：构建归一化低通与高通图滤波器，供 DualPassEncoder 的两个分支使用。
def build_dual_pass_filters(
        edge_index: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype = torch.float32,
        device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct normalized low-pass and high-pass GADL graph filters."""
    adjacency = edge_index_to_undirected_binary_adjacency(
        edge_index, num_nodes, dtype=dtype, device=device
    )
    target_device = adjacency.device
    nodes = torch.arange(num_nodes, device=target_device)
    identity_indices = torch.stack([nodes, nodes])
    identity_values = torch.ones(num_nodes, dtype=dtype, device=target_device)

    augmented_indices = torch.cat([adjacency.indices(), identity_indices], dim=1)
    augmented_values = torch.cat([adjacency.values(), identity_values])
    augmented = torch.sparse_coo_tensor(
        augmented_indices,
        augmented_values,
        (num_nodes, num_nodes),
        dtype=dtype,
        device=target_device,
    ).coalesce()

    degrees = torch.sparse.sum(augmented, dim=1).to_dense()
    if not torch.isfinite(degrees).all() or torch.any(degrees <= 0):
        raise ValueError("augmented adjacency produced invalid node degrees")
    inv_sqrt_degree = degrees.rsqrt()
    row, col = augmented.indices()
    normalized_values = (
        inv_sqrt_degree[row] * augmented.values() * inv_sqrt_degree[col]
    )

    half_identity = torch.full(
        (num_nodes,), 0.5, dtype=dtype, device=target_device
    )
    common_indices = torch.cat([identity_indices, augmented.indices()], dim=1)
    low_values = torch.cat([half_identity, 0.5 * normalized_values])
    high_values = torch.cat([half_identity, -0.5 * normalized_values])

    low_pass = torch.sparse_coo_tensor(
        common_indices, low_values, (num_nodes, num_nodes),
        dtype=dtype, device=target_device,
    ).coalesce()
    high_pass = torch.sparse_coo_tensor(
        common_indices, high_values, (num_nodes, num_nodes),
        dtype=dtype, device=target_device,
    ).coalesce()

    if not torch.isfinite(low_pass.values()).all():
        raise ValueError("low-pass filter contains non-finite values")
    if not torch.isfinite(high_pass.values()).all():
        raise ValueError("high-pass filter contains non-finite values")
    return low_pass, high_pass


# 功能：按两图节点数限制共享谱维度，避免请求超过图所能提供的特征向量。
def resolve_effective_rank(spectral_rank: int, n_src: int, n_tgt: int) -> int:
    """Choose one valid spectral rank shared by both aligned graphs."""
    if not isinstance(spectral_rank, int) or isinstance(spectral_rank, bool):
        raise TypeError("spectral_rank must be an integer")
    if spectral_rank < 1:
        raise ValueError("spectral_rank must be at least 1")
    if min(n_src, n_tgt) < 2:
        raise ValueError("GADL requires at least two nodes in each graph")
    return min(spectral_rank, n_src - 1, n_tgt - 1)


# 功能：求归一化图拉普拉斯的低频特征值和特征向量；按规模选择稠密或稀疏求解。
# 输出结果：谱基交给 GADL.train() 的函数映射损失计算。
def normalized_laplacian_eigenbasis(
        edge_index: torch.Tensor,
        num_nodes: int,
        spectral_rank: int,
        dtype: torch.dtype = torch.float32,
        dense_threshold: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the smallest eigenpairs of the normalized graph Laplacian."""
    if num_nodes < 2:
        raise ValueError("normalized Laplacian eigendecomposition needs >= 2 nodes")
    effective_rank = min(spectral_rank, num_nodes - 1)
    if effective_rank < 1:
        raise ValueError("effective spectral rank must be at least 1")

    adjacency = edge_index_to_undirected_binary_adjacency(
        edge_index.detach().cpu(), num_nodes, dtype=torch.float64, device="cpu"
    )

    if num_nodes <= dense_threshold:
        dense_adjacency = adjacency.to_dense()
        degrees = dense_adjacency.sum(dim=1)
        active = degrees > 0
        inv_sqrt_degree = torch.zeros_like(degrees)
        inv_sqrt_degree[active] = degrees[active].rsqrt()
        normalized_adjacency = (
            inv_sqrt_degree[:, None]
            * dense_adjacency
            * inv_sqrt_degree[None, :]
        )
        laplacian = torch.diag(active.to(torch.float64)) - normalized_adjacency
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        eigenvalues = eigenvalues[:effective_rank]
        eigenvectors = eigenvectors[:, :effective_rank]
    else:
        coalesced = adjacency.coalesce()
        indices = coalesced.indices().numpy()
        values = coalesced.values().numpy()
        scipy_adjacency = scipy_sparse.coo_matrix(
            (values, (indices[0], indices[1])),
            shape=(num_nodes, num_nodes),
        )
        laplacian = csgraph.laplacian(scipy_adjacency, normed=True)
        # ARPACK otherwise chooses an implicit random starting vector outside
        # PyTorch's seeded RNG.  A fixed non-constant vector makes repeated
        # runs reproducible without changing the eigensystem being solved.
        start_vector = np.linspace(1.0, 2.0, num_nodes, dtype=np.float64)
        start_vector /= np.linalg.norm(start_vector)
        values_np, vectors_np = eigsh(
            laplacian, k=effective_rank, which="SM", v0=start_vector
        )
        order = np.argsort(values_np)
        eigenvalues = torch.from_numpy(values_np[order].copy())
        eigenvectors = torch.from_numpy(vectors_np[:, order].copy())

    eigenvalues = eigenvalues.to(dtype=dtype)
    eigenvectors = eigenvectors.to(dtype=dtype)
    if not torch.isfinite(eigenvalues).all() or not torch.isfinite(eigenvectors).all():
        raise RuntimeError("normalized Laplacian eigendecomposition is non-finite")
    return eigenvalues, eigenvectors


# 输入来源：SpectralFilterBranch.forward() 传入当前线性层输出和固定图滤波器。
# 输出结果：按稀疏或普通矩阵乘法得到邻居传播后的特征，交回分支继续残差与激活处理。
def _apply_filter(filter_matrix: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    if filter_matrix.layout == torch.sparse_coo:
        return torch.sparse.mm(filter_matrix, features)
    return filter_matrix @ features


# 功能：使用一个固定图滤波器的可学习线性与激活层，可加入残差连接。
class SpectralFilterBranch(nn.Module):
    """A learnable graph-convolution branch driven by one fixed graph filter."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_hidden_layers: int, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for layer_index in range(num_hidden_layers):
            layer_input = input_dim if layer_index == 0 else hidden_dim
            self.layers.append(nn.Linear(layer_input, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))
        self.residual_projection = (
            nn.Linear(input_dim, hidden_dim, bias=False) if residual else None
        )

    # 功能：交替应用图滤波与可学习层，返回单通道节点表示。
    def forward(self, features: torch.Tensor,
                filter_matrix: torch.Tensor) -> torch.Tensor:
        hidden = features
        residual = (
            self.residual_projection(features)
            if self.residual_projection is not None else None
        )
        for layer, activation in zip(self.layers, self.activations):
            hidden = _apply_filter(filter_matrix, layer(hidden))
            if residual is not None:
                hidden = hidden + residual
            hidden = activation(hidden)
        return hidden


# 功能：分别学习低通、高通表示，再拼接为节点嵌入。
class DualPassEncoder(nn.Module):
    """Shared encoder with independent low-pass and high-pass parameters."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_hidden_layers: int, residual: bool = False):
        super().__init__()
        self.low_branch = SpectralFilterBranch(
            input_dim, hidden_dim, num_hidden_layers, residual=residual
        )
        self.high_branch = SpectralFilterBranch(
            input_dim, hidden_dim, num_hidden_layers, residual=residual
        )

    # 输出结果：返回两种结构频率表示的拼接，供 GADLModel.encode_pair() 使用。
    def forward(self, features: torch.Tensor, low_pass: torch.Tensor,
                high_pass: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError("node features must be a two-dimensional tensor")
        if not torch.isfinite(features).all():
            raise ValueError("node features contain non-finite values")
        low_embedding = self.low_branch(features, low_pass)
        high_embedding = self.high_branch(features, high_pass)
        return torch.cat([low_embedding, high_embedding], dim=1)


# 功能：维护共享双通道编码器和两方向的谱空间映射参数。
class GADLModel(nn.Module):
    """Shared dual-pass encoder and bidirectional functional maps."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_hidden_layers: int, spectral_rank: int,
                 residual: bool = False):
        super().__init__()
        self.encoder = DualPassEncoder(
            input_dim, hidden_dim, num_hidden_layers, residual=residual
        )
        self.C12 = nn.Parameter(torch.eye(spectral_rank))
        self.C21 = nn.Parameter(torch.eye(spectral_rank))

    # 功能：用同一个 encoder 分别编码两张图，返回两侧嵌入。
    def encode_pair(self, x1: torch.Tensor, low1: torch.Tensor,
                    high1: torch.Tensor, x2: torch.Tensor,
                    low2: torch.Tensor, high2: torch.Tensor):
        # One encoder instance is deliberately applied to both graphs.
        z1 = self.encoder(x1, low1, high1)
        z2 = self.encoder(x2, low2, high2)
        return z1, z2

    # 功能：利用谱基和嵌入计算双向映射、互逆与正交约束，返回各项损失。
    def functional_map_losses(
            self,
            z1: torch.Tensor,
            z2: torch.Tensor,
            phi1: torch.Tensor,
            phi2: torch.Tensor,
            lambda1: torch.Tensor,
            lambda2: torch.Tensor,
            descriptor_weight: float,
            commutativity_weight: float) -> Dict[str, torch.Tensor]:
        f1 = phi1.T @ z1
        f2 = phi2.T @ z2

        descriptor12 = torch.linalg.matrix_norm(self.C12 @ f1 - f2) ** 2
        descriptor21 = torch.linalg.matrix_norm(self.C21 @ f2 - f1) ** 2
        commute12 = (
            lambda2[:, None] * self.C12 - self.C12 * lambda1[None, :]
        )
        commute21 = (
            lambda1[:, None] * self.C21 - self.C21 * lambda2[None, :]
        )
        fm12 = (
            descriptor_weight * descriptor12
            + commutativity_weight * torch.linalg.matrix_norm(commute12) ** 2
        )
        fm21 = (
            descriptor_weight * descriptor21
            + commutativity_weight * torch.linalg.matrix_norm(commute21) ** 2
        )

        identity = torch.eye(
            self.C12.shape[0], dtype=self.C12.dtype, device=self.C12.device
        )
        bijectivity = (
            torch.linalg.matrix_norm(self.C12 @ self.C21 - identity) ** 2
            + torch.linalg.matrix_norm(self.C21 @ self.C12 - identity) ** 2
        )
        orthogonality = (
            torch.linalg.matrix_norm(self.C12 @ self.C12.T - identity) ** 2
            + torch.linalg.matrix_norm(self.C21.T @ self.C21 - identity) ** 2
        )
        return {
            "fm12": fm12,
            "fm21": fm21,
            "bijectivity": bijectivity,
            "orthogonality": orthogonality,
        }


# 功能：用节点嵌入内积重构邻接矩阵，以二元交叉熵衡量边预测误差。
def dense_reconstruction_loss(
        embeddings: torch.Tensor,
        adjacency: torch.Tensor) -> torch.Tensor:
    """Element-wise BCE for the inner-product graph decoder."""
    target = adjacency.to_dense() if adjacency.layout == torch.sparse_coo else adjacency
    target = target.to(dtype=embeddings.dtype, device=embeddings.device)
    logits = embeddings @ embeddings.T
    return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")


# 功能：合并两图重构与函数映射约束，返回含 total 的损失字典供 GADL.train() 反向传播。
def gadl_objective(
        model: GADLModel,
        z1: torch.Tensor,
        z2: torch.Tensor,
        adjacency1: torch.Tensor,
        adjacency2: torch.Tensor,
        phi1: torch.Tensor,
        phi2: torch.Tensor,
        lambda1: torch.Tensor,
        lambda2: torch.Tensor,
        descriptor_weight: float,
        commutativity_weight: float,
        functional_map_weight: float,
        bijectivity_weight: float,
        orthogonality_weight: float) -> Dict[str, torch.Tensor]:
    """Compute the complete first-version GADL training objective."""
    reconstruction = (
        dense_reconstruction_loss(z1, adjacency1)
        + dense_reconstruction_loss(z2, adjacency2)
    )
    map_losses = model.functional_map_losses(
        z1, z2, phi1, phi2, lambda1, lambda2,
        descriptor_weight=descriptor_weight,
        commutativity_weight=commutativity_weight,
    )
    total = (
        reconstruction
        + functional_map_weight * (map_losses["fm12"] + map_losses["fm21"])
        + bijectivity_weight * map_losses["bijectivity"]
        + orthogonality_weight * map_losses["orthogonality"]
    )
    return {"total": total, "reconstruction": reconstruction, **map_losses}
