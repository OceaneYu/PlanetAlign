# 文件作用：实现 GADL，通过共享的双通道图编码器学习一对一节点相似度。
# 数据来源：scripts/evaluate_gadl_one_to_one.py 等评估脚本或 PlanetAlign/m2m/base.py → train_base_S() 传入 Dataset。
# 核心依赖：PlanetAlign/algorithms/gadl/model.py 的图滤波、拉普拉斯特征基、GADLModel 与 gadl_objective()。
# 结果去向：train() 保存并返回 self.S 与日志；节点评估用 BaseModel.test()，多对多解码由入口另行调用。

"""PlanetAlign interface for the GADL one-to-one graph aligner."""

import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from PlanetAlign.algorithms.base_model import BaseModel
from PlanetAlign.data import Dataset

from .model import (
    GADLModel,
    build_dual_pass_filters,
    edge_index_to_undirected_binary_adjacency,
    gadl_objective,
    normalized_laplacian_eigenbasis,
    resolve_effective_rank,
)


# 功能：使用图边与同维节点属性进行无监督训练，训练锚点不进入优化目标。
class GADL(BaseModel):
    """Unsupervised one-to-one alignment with dual-pass spectral encoding.

    GADL requires same-dimensional node attributes on both graphs.  Training
    uses only ``graph.x`` and ``graph.edge_index``; anchor links are reserved
    for the inherited :meth:`BaseModel.test` evaluation path.
    """

    # 功能：检查维度、学习率、损失权重与 seed，保存配置并准备训练结果的空容器。
    # 流程位置：创建 GADL 对象时执行；实际网络、图滤波器和谱基直到 train() 才构建。
    def __init__(self,
                 hidden_dim: int = 512,
                 num_hidden_layers: int = 3,
                 spectral_rank: int = 300,
                 lr: float = 1e-4,
                 weight_decay: float = 5e-4,
                 descriptor_weight: float = 1e-3,
                 commutativity_weight: float = 1.0,
                 functional_map_weight: float = 1.0,
                 bijectivity_weight: float = 1e-1,
                 orthogonality_weight: float = 1e-1,
                 residual: bool = False,
                 seed: int = 0,
                 dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be positive")
        if spectral_rank < 1:
            raise ValueError("spectral_rank must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")
        for name, value in {
            "weight_decay": weight_decay,
            "descriptor_weight": descriptor_weight,
            "commutativity_weight": commutativity_weight,
            "functional_map_weight": functional_map_weight,
            "bijectivity_weight": bijectivity_weight,
            "orthogonality_weight": orthogonality_weight,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.spectral_rank = spectral_rank
        self.lr = lr
        self.weight_decay = weight_decay
        self.descriptor_weight = descriptor_weight
        self.commutativity_weight = commutativity_weight
        self.functional_map_weight = functional_map_weight
        self.bijectivity_weight = bijectivity_weight
        self.orthogonality_weight = orthogonality_weight
        self.residual = residual
        self.seed = seed
        self.model: Optional[GADLModel] = None
        self.embeddings: Dict[int, torch.Tensor] = {}
        self.raw_embeddings: Dict[int, torch.Tensor] = {}
        self.initial_embeddings: Dict[int, torch.Tensor] = {}
        self.initial_S: Optional[torch.Tensor] = None
        self.initial_losses: Dict[str, float] = {}
        self.final_losses: Dict[str, float] = {}
        self.loss_history = []
        self.spectral_time = 0.0
        self.training_time = 0.0
        self.spectra = {}
        self.effective_rank: Optional[int] = None

    # 输入来源：Dataset 中两图的 x 与 edge_index，要求属性维度一致；锚点不进入训练目标。
    # 核心思路：构建低通/高通滤波器与谱基，训练共享编码器，组合图重构和双向函数映射损失。
    # 输出结果：归一化两侧嵌入的内积作为 S；保存训练损失历史，返回矩阵与日志。
    def train(self,
              dataset: Dataset,
              gids: Union[List[int], Tuple[int, ...]],
              use_attr: bool = True,
              total_epochs: int = 100,
              save_log: bool = True,
              verbose: bool = True):
        if len(gids) != 2:
            raise ValueError("GADL supports exactly two graphs")
        if not use_attr:
            raise ValueError("GADL requires use_attr=True and node attributes")
        if total_epochs < 1:
            raise ValueError("total_epochs must be at least 1")

        self.check_inputs(
            dataset, gids, plain_method=False, use_attr=True,
            pairwise=True, supervised=False,
        )
        gid1, gid2 = gids
        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        if graph1.x is None or graph2.x is None:
            raise ValueError("GADL requires node attributes on both graphs")
        if graph1.x.dim() != 2 or graph2.x.dim() != 2:
            raise ValueError("GADL node attributes must be two-dimensional")
        if graph1.x.shape[1] != graph2.x.shape[1]:
            raise ValueError(
                "GADL requires both graphs to have the same node attribute dimension"
            )
        if not torch.isfinite(graph1.x).all() or not torch.isfinite(graph2.x).all():
            raise ValueError("GADL node attributes must contain only finite values")

        self.effective_rank = resolve_effective_rank(
            self.spectral_rank, graph1.num_nodes, graph2.num_nodes
        )
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        x1 = graph1.x.detach().to(device=self.device, dtype=self.dtype)
        x2 = graph2.x.detach().to(device=self.device, dtype=self.dtype)
        low1, high1 = build_dual_pass_filters(
            graph1.edge_index, graph1.num_nodes, dtype=self.dtype, device=self.device
        )
        low2, high2 = build_dual_pass_filters(
            graph2.edge_index, graph2.num_nodes, dtype=self.dtype, device=self.device
        )
        adjacency1 = edge_index_to_undirected_binary_adjacency(
            graph1.edge_index, graph1.num_nodes, dtype=self.dtype, device=self.device
        )
        adjacency2 = edge_index_to_undirected_binary_adjacency(
            graph2.edge_index, graph2.num_nodes, dtype=self.dtype, device=self.device
        )

        spectral_start = time.time()
        lambda1, phi1 = normalized_laplacian_eigenbasis(
            graph1.edge_index, graph1.num_nodes, self.effective_rank, dtype=self.dtype
        )
        lambda2, phi2 = normalized_laplacian_eigenbasis(
            graph2.edge_index, graph2.num_nodes, self.effective_rank, dtype=self.dtype
        )
        lambda1 = lambda1.to(self.device)
        lambda2 = lambda2.to(self.device)
        phi1 = phi1.to(self.device)
        phi2 = phi2.to(self.device)
        self.spectral_time = time.time() - spectral_start
        self.spectra = {
            gid1: {"eigenvalues": lambda1.detach(), "eigenvectors": phi1.detach()},
            gid2: {"eigenvalues": lambda2.detach(), "eigenvectors": phi2.detach()},
        }

        self.model = GADLModel(
            input_dim=x1.shape[1],
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            spectral_rank=self.effective_rank,
            residual=self.residual,
        ).to(device=self.device, dtype=self.dtype)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_component_names = [
            "reconstruction", "fm12", "fm21", "bijectivity", "orthogonality"
        ]
        logger = self.init_training_logger(
            dataset, use_attr, save_log=save_log,
            additional_headers=loss_component_names,
        )

        with torch.no_grad():
            initial_z1, initial_z2 = self.model.encode_pair(
                x1, low1, high1, x2, low2, high2
            )
            initial_losses = gadl_objective(
                self.model, initial_z1, initial_z2, adjacency1, adjacency2,
                phi1, phi2, lambda1, lambda2,
                descriptor_weight=self.descriptor_weight,
                commutativity_weight=self.commutativity_weight,
                functional_map_weight=self.functional_map_weight,
                bijectivity_weight=self.bijectivity_weight,
                orthogonality_weight=self.orthogonality_weight,
            )
            initial_normalized1 = F.normalize(initial_z1, p=2, dim=1)
            initial_normalized2 = F.normalize(initial_z2, p=2, dim=1)
            self.initial_S = (initial_normalized1 @ initial_normalized2.T).detach()
            self.initial_embeddings = {
                gid1: initial_z1.detach(),
                gid2: initial_z2.detach(),
            }
            self.initial_losses = {
                name: float(value.detach()) for name, value in initial_losses.items()
            }

        training_start = time.time()
        for epoch in range(1, total_epochs + 1):
            epoch_start = time.time()
            self.model.train()
            optimizer.zero_grad()
            z1, z2 = self.model.encode_pair(x1, low1, high1, x2, low2, high2)
            losses = gadl_objective(
                self.model, z1, z2, adjacency1, adjacency2,
                phi1, phi2, lambda1, lambda2,
                descriptor_weight=self.descriptor_weight,
                commutativity_weight=self.commutativity_weight,
                functional_map_weight=self.functional_map_weight,
                bijectivity_weight=self.bijectivity_weight,
                orthogonality_weight=self.orthogonality_weight,
            )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("GADL training produced a non-finite loss")
            losses["total"].backward()
            optimizer.step()

            loss_row = {
                name: float(value.detach()) for name, value in losses.items()
            }
            loss_row["epoch"] = epoch
            self.loss_history.append(loss_row)

            # GADL is unsupervised.  The logger records loss only; NaN explicitly
            # marks metrics that were not consulted during training.
            logger.log(
                epoch=epoch,
                loss=float(losses["total"].detach()),
                epoch_time=time.time() - epoch_start,
                hits={},
                mrr=float("nan"),
                verbose=verbose,
                **{name: loss_row[name] for name in loss_component_names},
            )
        self.training_time = time.time() - training_start

        self.model.eval()
        with torch.no_grad():
            z1, z2 = self.model.encode_pair(x1, low1, high1, x2, low2, high2)
            final_losses = gadl_objective(
                self.model, z1, z2, adjacency1, adjacency2,
                phi1, phi2, lambda1, lambda2,
                descriptor_weight=self.descriptor_weight,
                commutativity_weight=self.commutativity_weight,
                functional_map_weight=self.functional_map_weight,
                bijectivity_weight=self.bijectivity_weight,
                orthogonality_weight=self.orthogonality_weight,
            )
            normalized1 = F.normalize(z1, p=2, dim=1)
            normalized2 = F.normalize(z2, p=2, dim=1)
            self.S = (normalized1 @ normalized2.T).detach()
            self.embeddings = {
                gid1: normalized1.detach(),
                gid2: normalized2.detach(),
            }
            self.raw_embeddings = {gid1: z1.detach(), gid2: z2.detach()}
            self.final_losses = {
                name: float(value.detach()) for name, value in final_losses.items()
            }

        expected_shape = (graph1.num_nodes, graph2.num_nodes)
        if tuple(self.S.shape) != expected_shape:
            raise RuntimeError(
                f"GADL produced similarity shape {tuple(self.S.shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(self.S).all():
            raise RuntimeError("GADL produced a non-finite similarity matrix")
        return self.S, logger
