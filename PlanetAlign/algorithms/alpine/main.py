# 文件作用：将 Alpine 的结构图匹配接入 PlanetAlign，输出原生的一对一节点对。
# 数据来源：调用者把两侧 PyG 图装在 Dataset 中传入 Alpine.train()，训练不使用锚点或属性。
# 核心处理：较小图作为查询，_run_alpine() 优化结构匹配后用 Hungarian 转成硬节点对。
# 结果去向：train() 返回 mapping_pairs 与日志；test() 直接比较这些节点对，不使用排名矩阵 S。
# 阅读建议：先读 Alpine.train()/test()，原有许可与来源说明保留在下方。

"""PlanetAlign integration of the official Alpine graph aligner.

The numerical path is adapted from the MIT-licensed Alpine implementation at
https://github.com/constantinosskitsas/Alpine (commit ce676811).  Alpine's
prediction is its final Hungarian assignment, so this module deliberately does
not turn that assignment into a one-hot ranking matrix.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances

from PlanetAlign.algorithms.base_model import BaseModel
from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_anchor_pairs


# 功能：报告稠密工作矩阵超过调用者设置的规模上限。
class AlpineResourceLimitError(RuntimeError):
    """Raised when Alpine's dense working set exceeds the configured guard."""


# 功能：保留原节点编号，将 PyG 边转为 NetworkX 无向图供 Alpine 使用。
def _pyg_to_undirected_nx(graph) -> nx.Graph:
    """Convert PyG indices to a simple graph without renumbering nodes."""
    result = nx.Graph()
    result.add_nodes_from(range(int(graph.num_nodes)))
    result.add_edges_from(
        (int(src), int(dst))
        for src, dst in graph.edge_index.detach().cpu().T.tolist()
    )
    return result


def _structural_features(graph: nx.Graph, weight: int) -> np.ndarray:
    """Reproduce Alpine's released structural features."""
    nodes = list(graph.nodes())
    degrees = dict(graph.degree())
    ego_nodes = {node: list(nx.ego_graph(graph, node).nodes) for node in nodes}

    def neighbor_mean(values: Dict[int, float], node: int) -> float:
        neighbors = [values[other] for other in ego_nodes[node] if other != node]
        return float(np.mean(neighbors)) if neighbors else 0.0

    if weight == 2:
        features = np.zeros((len(nodes), 2))
        features[:, 0] = [degrees[node] for node in nodes]
        features[:, 1] = [neighbor_mean(degrees, node) for node in nodes]
        return np.nan_to_num(features)

    clustering = nx.clustering(graph)
    features = np.zeros((len(nodes), 7))
    features[:, 0] = [degrees[node] for node in nodes]
    features[:, 1] = [clustering[node] for node in nodes]
    features[:, 2] = [neighbor_mean(degrees, node) for node in nodes]
    features[:, 3] = [neighbor_mean(clustering, node) for node in nodes]
    return np.nan_to_num(features)


def _sinkhorn_knopp(
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
    cost: torch.Tensor,
    reg: float = 1.0,
    max_iters: int = 500,
    tolerance: float = 1e-5,
    eval_frequency: int = 10,
) -> torch.Tensor:
    """Released Alpine Sinkhorn-Knopp update, reduced to its used path."""
    rows, cols = cost.shape
    u = torch.ones(rows, dtype=source_mass.dtype) / rows
    v = torch.ones(cols, dtype=target_mass.dtype) / cols
    kernel = torch.exp(cost / -reg)

    iteration = 1
    error = 1.0
    while error > tolerance and iteration <= max_iters:
        previous_u, previous_v = u, v
        kernel_t_u = torch.matmul(u, kernel)
        v = torch.div(target_mass, kernel_t_u)
        kernel_v = torch.matmul(kernel, v)
        u = torch.div(source_mass, kernel_v)

        numerical_error = (
            torch.any(kernel_t_u == 0)
            or torch.any(torch.isnan(u))
            or torch.any(torch.isnan(v))
            or torch.any(torch.isinf(u))
            or torch.any(torch.isinf(v))
        )
        if numerical_error:
            u, v = previous_u, previous_v
            break

        if iteration % eval_frequency == 0:
            target_hat = torch.matmul(u, kernel) * v
            if len(source_mass) != len(target_mass):
                source_hat = torch.matmul(v, kernel.T) * u
                error = (
                    (source_mass - source_hat).pow(2).sum().item() / 2
                    + (target_mass - target_hat).pow(2).sum().item() / 2
                )
            else:
                error = (target_mass - target_hat).pow(2).sum().item()
        iteration += 1

    return u.reshape(-1, 1) * kernel * v.reshape(1, -1)


def _optimize(
    query_adj: torch.Tensor,
    target_adj: torch.Tensor,
    feature_cost: torch.Tensor,
    outer_iters: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]], float]:
    """Run Alpine's partial-QAP continuation and Hungarian projection."""
    query_nodes = len(query_adj)
    target_nodes = len(target_adj)
    identity_pad = torch.zeros((query_nodes, query_nodes + 1), dtype=torch.float64)
    identity_pad[torch.arange(query_nodes), torch.arange(query_nodes)] = 1

    plan = torch.ones((query_nodes + 1, target_nodes), dtype=torch.float64)
    plan[:-1, :] *= 1 / target_nodes
    plan[-1, :] *= (target_nodes - query_nodes) / target_nodes
    matrix_ones = torch.ones_like(plan)
    target_mass = torch.ones(target_nodes, dtype=torch.float64)
    source_mass = torch.ones(query_nodes + 1, dtype=torch.float64)
    source_mass[-1] = target_nodes - query_nodes

    for continuation_step in range(outer_iters):
        for frank_wolfe_step in range(1, 11):
            residual = (
                query_adj
                - identity_pad @ plan @ target_adj @ plan.T @ identity_pad.T
            )
            derivative = (
                -4
                * identity_pad.T
                @ residual
                @ identity_pad
                @ plan
                @ target_adj
                + continuation_step * (matrix_ones - 2 * plan)
                + feature_cost
            )
            direction = _sinkhorn_knopp(source_mass, target_mass, derivative)
            alpha = 2 / float(2 + frank_wolfe_step)
            plan[:query_nodes, :target_nodes] += alpha * (
                direction[:query_nodes, :target_nodes]
                - plan[:query_nodes, :target_nodes]
            )

    plan = plan[:-1]
    row_ids, column_ids = linear_sum_assignment(plan.numpy(), maximize=True)
    pairs = [(int(row), int(column)) for row, column in zip(row_ids, column_ids)]
    permutation = np.zeros((query_nodes, target_nodes))
    permutation[row_ids, column_ids] = 1
    target_np = target_adj.numpy()
    residual = query_adj.numpy() - permutation @ target_np @ permutation.T
    frobenius_squared = float(np.linalg.norm(residual, "fro") ** 2)
    return plan, pairs, frobenius_squared


# 流程位置：Alpine.train() 将定向后的查询图和目标图传入，调用适配后的参考数值流程。
def _run_alpine(
    query_graph: nx.Graph,
    target_graph: nx.Graph,
    mu: float,
    weight: int,
    outer_iters: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]], float]:
    """Run the verified public Alpine path on an oriented graph pair."""
    query_nodes = query_graph.number_of_nodes()
    target_nodes = target_graph.number_of_nodes()
    if query_nodes > target_nodes:
        raise ValueError("Alpine requires the query graph to be no larger than the target")

    for node in list(nx.isolates(query_graph)):
        query_graph.add_edge(node, node)
    for node in list(nx.isolates(target_graph)):
        target_graph.add_edge(node, node)

    query_graph.add_node(query_nodes)
    query_graph.add_edge(query_nodes, query_nodes)
    query_with_dummy = torch.tensor(
        nx.to_numpy_array(query_graph), dtype=torch.float64
    )
    target_adj = torch.tensor(
        nx.to_numpy_array(target_graph), dtype=torch.float64
    )
    query_features = _structural_features(query_graph, weight)
    target_features = _structural_features(target_graph, weight)
    feature_cost = torch.tensor(
        euclidean_distances(query_features, target_features),
        dtype=torch.float64,
    )
    return _optimize(
        query_with_dummy[:query_nodes, :query_nodes],
        target_adj,
        mu * feature_cost,
        outer_iters,
    )


# 功能：提供无监督的结构部分匹配，允许两图大小不同，但查询节点须匹配不同目标节点。
class Alpine(BaseModel):
    """Partial unlabeled graph alignment using native Hungarian pairs.

    Alpine is an unsupervised, structure-only, pairwise method.  It always
    orients the smaller input graph as query and maps every query node to a
    distinct target node.  ``test`` evaluates those hard pairs directly on
    held-out ground truth and reports only Accuracy/Hits@1.

    Parameters
    ----------
    mu : float, optional
        Weight of the structural-feature cost. Default is 1.0.
    weight : int, optional
        ``2`` selects the paper/default degree features; other values select
        the released seven-column feature representation. Default is 2.
    niter : int, optional
        Number of gradual-concavity continuation steps. Default is 10.
    max_dense_work_entries : int or None, optional
        Resource guard based on an estimate of the dense working set. The
        default 45 million covers the verified small/medium suite. Set to
        ``None`` to opt into larger runs at the caller's own resource risk.
    """

    def __init__(
        self,
        mu: float = 1.0,
        weight: int = 2,
        niter: int = 10,
        max_dense_work_entries: Optional[int] = 45_000_000,
    ):
        super().__init__(dtype=torch.float64)
        if mu < 0:
            raise ValueError("mu must be non-negative")
        if niter < 1:
            raise ValueError("niter must be positive")
        if max_dense_work_entries is not None and max_dense_work_entries < 1:
            raise ValueError("max_dense_work_entries must be positive or None")
        self.mu = mu
        self.weight = weight
        self.niter = niter
        self.max_dense_work_entries = max_dense_work_entries
        self.transport_plan: Optional[torch.Tensor] = None
        self.mapping_pairs: Optional[List[Tuple[int, int]]] = None
        self.query_gid: Optional[int] = None
        self.target_gid: Optional[int] = None
        self.frobenius_squared: Optional[float] = None
        self.runtime_seconds: Optional[float] = None
        self.last_evaluation: Optional[dict] = None
        self._trained_gids: Optional[Tuple[int, int]] = None

    @staticmethod
    def _dense_work_entries(query_nodes: int, target_nodes: int) -> int:
        return (
            query_nodes * query_nodes
            + target_nodes * target_nodes
            + 4 * (query_nodes + 1) * target_nodes
        )

    # 核心思路：检查 CPU 与规模限制，定向两图后调用 _run_alpine()，再把节点对映射回原图方向。
    # 输出结果：保存 mapping_pairs 与传输计划，返回节点对和日志，不伪造 one-hot 排名矩阵。
    def train(
        self,
        dataset: Dataset,
        gids: Union[Tuple[int, int], List[int]],
        use_attr: bool = False,
        save_log: bool = True,
        verbose: bool = True,
    ):
        """Align two PlanetAlign PyG graphs without consuming anchor links."""
        self.check_inputs(
            dataset,
            gids,
            plain_method=True,
            use_attr=use_attr,
            pairwise=True,
            supervised=False,
        )
        if torch.device(self.device).type != "cpu":
            raise ValueError("Alpine's verified dense float64 implementation is CPU-only")

        gid1, gid2 = int(gids[0]), int(gids[1])
        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        if int(graph1.num_nodes) <= int(graph2.num_nodes):
            query_gid, target_gid = gid1, gid2
            query_pyg, target_pyg = graph1, graph2
        else:
            query_gid, target_gid = gid2, gid1
            query_pyg, target_pyg = graph2, graph1

        query_nodes = int(query_pyg.num_nodes)
        target_nodes = int(target_pyg.num_nodes)
        estimated_entries = self._dense_work_entries(query_nodes, target_nodes)
        if (
            self.max_dense_work_entries is not None
            and estimated_entries > self.max_dense_work_entries
        ):
            raise AlpineResourceLimitError(
                "SKIPPED_RESOURCE: Alpine estimated "
                f"{estimated_entries:,} dense work entries for {query_nodes}x"
                f"{target_nodes} nodes, above max_dense_work_entries="
                f"{self.max_dense_work_entries:,}; pass None to override"
            )

        logger = self.init_training_logger(
            dataset,
            use_attr,
            additional_headers=["infer_time"],
            save_log=save_log,
        )
        query_graph = _pyg_to_undirected_nx(query_pyg)
        target_graph = _pyg_to_undirected_nx(target_pyg)
        started = time.perf_counter()
        plan, pairs, frobenius = _run_alpine(
            query_graph,
            target_graph,
            mu=self.mu,
            weight=self.weight,
            outer_iters=self.niter,
        )
        runtime = time.perf_counter() - started

        self.S = None
        self.transport_plan = plan
        self.mapping_pairs = pairs
        self.query_gid = query_gid
        self.target_gid = target_gid
        self.frobenius_squared = frobenius
        self.runtime_seconds = runtime
        self.last_evaluation = None
        self._trained_gids = (gid1, gid2)
        logger.log(
            epoch=1,
            loss=frobenius,
            epoch_time=runtime,
            hits={},
            mrr=float("nan"),
            infer_time=round(runtime, 4),
            verbose=verbose,
        )
        return self.mapping_pairs, logger

    # 功能：用 Dataset.test_data 的真实节点对检查 mapping_pairs，返回 Accuracy 或 Hits@1。
    # 流程位置：评估原生硬匹配，因此不沿用 BaseModel.test() 的矩阵排名流程。
    def test(
        self,
        dataset: Dataset,
        gids: Union[Tuple[int, int], List[int]],
        metrics: Optional[Union[Tuple[str, ...], List[str]]] = None,
        **_ranking_options,
    ) -> dict:
        """Compare native Hungarian pairs directly with held-out test GT."""
        if self.mapping_pairs is None or self.query_gid is None:
            raise AssertionError("Model is not trained yet, call train() method first")
        if set(map(int, gids)) != set(self._trained_gids or ()):
            raise ValueError("gids must identify the same graph pair used for training")

        requested = {"Accuracy", "Hits@1"} if metrics is None else set(metrics)
        unsupported = requested - {"Accuracy", "Hits@1"}
        if unsupported:
            raise ValueError(
                "Alpine exposes hard Hungarian pairs, so ranking metrics are not "
                f"defined: {sorted(unsupported)}"
            )

        gid1, gid2 = int(gids[0]), int(gids[1])
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)
        if test_pairs.numel() == 0:
            raise ValueError("held-out test GT contains no comparable pairs")
        oriented_gt = [(int(left), int(right)) for left, right in test_pairs.tolist()]
        if self.query_gid == gid2:
            oriented_gt = [(right, left) for left, right in oriented_gt]
        elif self.query_gid != gid1:
            raise RuntimeError("stored Alpine query graph is inconsistent with gids")

        predicted = set(self.mapping_pairs)
        correct = sum(pair in predicted for pair in oriented_gt)
        accuracy = correct / len(oriented_gt)
        self.last_evaluation = {
            "comparison": "exact_native_hungarian_pair_membership",
            "test_gt_pairs": len(oriented_gt),
            "correct_pairs": correct,
            "Accuracy": accuracy,
            "Hits@1": accuracy,
        }
        return {
            name: accuracy
            for name in ("Accuracy", "Hits@1")
            if name in requested
        }
