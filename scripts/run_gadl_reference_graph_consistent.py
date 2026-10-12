#!/usr/bin/env python3
# 文件作用：在隔离的公开 GADL 副本上保持图与谱输入一致，观察原训练过程和评估协议。
# 数据来源：main() 从参数指定的参考目录加载 .mat、谱和源代码，prepare_and_assert_graphs() 核对输入。
# 核心处理：临时包装参考训练函数记录检查点与损失轨迹，原数值训练路径继续执行。
# 结果去向：main() 保存捕获张量、结果 JSON 与报告；这不是 PlanetAlign 的普通 GADL 入口。
# 阅读建议：main() → prepare_and_assert_graphs() → _run_seed()，区分参考副本文件与正式库代码。

"""Run a graph-consistent, instrumented copy of the public GADL reference.

The public source is imported from an isolated /private/tmp copy and is never
edited. Instrumentation observes the existing training loop and recomputes
diagnostics from fixed model states; it does not alter the model, optimizer,
loss, hyperparameters, or public evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time
import types
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import scipy.io
import scipy.sparse as sp
from scipy.sparse import csgraph
import torch
import torch.nn.functional as F


EPSILON = 1e-8
KS = (1, 5, 10, 50)
SEEDS = (0, 1, 2)
PAPER = {"Hits@1": 0.5331, "Hits@5": 0.7361, "Hits@10": 0.8067, "Hits@50": 0.9418}
MIXED_OPTIMISTIC_H1 = 0.9442


def _distribution(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().cpu().double().reshape(-1)
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "median": float(values.quantile(0.5)),
        "p90": float(values.quantile(0.9)),
        "max": float(values.max()),
    }


def _symmetrized_relation_one(mat: Mapping[str, Any], key: str) -> torch.Tensor:
    relation = sp.csr_matrix(mat[key][0][1])
    symmetrized = relation + relation.T
    symmetrized.data[:] = 1
    symmetrized.eliminate_zeros()
    return torch.from_numpy(symmetrized.toarray()).int()


def _relation_one(mat: Mapping[str, Any], key: str) -> torch.Tensor:
    relation = sp.csr_matrix(mat[key][0][1])
    relation.data[:] = 1
    relation.eliminate_zeros()
    return torch.from_numpy(relation.toarray()).int()


def _reference_laplacian(adjacency: torch.Tensor) -> sp.coo_matrix:
    sparse = adjacency.to_sparse_coo().coalesce().cpu()
    indices = sparse.indices().numpy()
    values = sparse.values().numpy()
    scipy_adjacency = sp.coo_matrix(
        (values, (indices[0], indices[1])), shape=tuple(adjacency.shape)
    )
    return sp.coo_matrix(csgraph.laplacian(scipy_adjacency, normed=True))


# 功能：比较实际与期望矩阵的非零位置及最大数值误差，不一致时阻止继续使用这套参考输入。
# 结果去向：prepare_and_assert_graphs() 收集断言记录，供 main() 写入输入一致性报告。
def _matrix_assertion(actual: sp.spmatrix, expected: sp.spmatrix, name: str) -> Dict[str, Any]:
    actual = actual.tocsr()
    expected = expected.tocsr()
    difference = (actual - expected).tocoo()
    max_error = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    frobenius = float(np.linalg.norm(difference.data))
    pattern_difference = (actual.astype(bool) != expected.astype(bool)).nnz
    passed = bool(pattern_difference == 0 and max_error <= 1e-10)
    result = {
        "name": name,
        "shape": list(actual.shape),
        "actual_nnz": int(actual.nnz),
        "expected_nnz": int(expected.nnz),
        "pattern_difference": int(pattern_difference),
        "max_absolute_error": max_error,
        "frobenius_error": frobenius,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"{name} consistency assertion failed: {result}")
    return result


# 功能：核对谱文件在指定拉普拉斯上的残差与有限值，超出约定时停止运行。
def _eigenpair_assertion(
    laplacian: sp.spmatrix,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    name: str,
) -> Dict[str, Any]:
    matrix = laplacian.tocsr().astype(np.float64)
    values = eigenvalues.detach().cpu().double().numpy()
    vectors = eigenvectors.detach().cpu().double().numpy()
    lhs = matrix @ vectors
    residual = lhs - vectors * values.reshape(1, -1)
    relative = float(np.linalg.norm(residual) / max(np.linalg.norm(lhs), 1e-30))
    maximum = float(np.max(np.abs(residual)))
    passed = bool(relative <= 1e-4 and np.isfinite(relative))
    result = {
        "name": name,
        "relative_residual_frobenius": relative,
        "max_absolute_residual": maximum,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"{name} eigenpair assertion failed: {result}")
    return result


# 功能：准备参考图输入，检查训练邻接、拉普拉斯与谱基是否来自相同图口径。
def prepare_and_assert_graphs(runtime_root: Path) -> Dict[str, Any]:
    mat = scipy.io.loadmat(runtime_root / "data" / "douban.mat")
    online = _symmetrized_relation_one(mat, "online_edge_label")
    offline = _relation_one(mat, "offline_edge_label")
    if online.shape != (3906, 3906) or int(torch.count_nonzero(online)) != 14_430:
        raise RuntimeError("unexpected reconstructed Online graph")
    if offline.shape != (1118, 1118) or int(torch.count_nonzero(offline)) != 3_022:
        raise RuntimeError("unexpected reconstructed Offline graph")
    if torch.count_nonzero(torch.diag(online)) or torch.count_nonzero(torch.diag(offline)):
        raise RuntimeError("reconstructed adjacency unexpectedly contains self-loops")
    if not torch.equal(online, online.T) or not torch.equal(offline, offline.T):
        raise RuntimeError("reconstructed adjacency is not symmetric")

    torch.save(online, runtime_root / "data" / "online.pt")
    torch.save(offline, runtime_root / "data" / "offline.pt")
    eig_root = runtime_root / "eigs" / "Douban Online_Offline"
    bundled_l1 = torch.load(eig_root / "L1.pth", weights_only=False)
    bundled_l2 = torch.load(eig_root / "L2.pth", weights_only=False)
    lam1 = torch.load(eig_root / "lam1.pth", weights_only=False)
    lam2 = torch.load(eig_root / "lam2.pth", weights_only=False)
    phi1 = torch.load(eig_root / "phi1.pth", weights_only=False)
    phi2 = torch.load(eig_root / "phi2.pth", weights_only=False)
    computed_l1 = _reference_laplacian(online)
    computed_l2 = _reference_laplacian(offline)
    assertions = {
        "online_laplacian": _matrix_assertion(computed_l1, bundled_l1, "Online L == bundled L1"),
        "offline_laplacian": _matrix_assertion(computed_l2, bundled_l2, "Offline L == bundled L2"),
        "online_eigenpairs": _eigenpair_assertion(bundled_l1, lam1, phi1, "Online bundled eigenpairs"),
        "offline_eigenpairs": _eigenpair_assertion(bundled_l2, lam2, phi2, "Offline bundled eigenpairs"),
    }
    return {
        "online": {
            "shape": list(online.shape),
            "dtype": str(online.dtype),
            "nonzero_entries": int(torch.count_nonzero(online)),
            "self_loops": int(torch.count_nonzero(torch.diag(online))),
            "symmetric": bool(torch.equal(online, online.T)),
            "construction": "binary(relation1 union relation1.T)",
        },
        "offline": {
            "shape": list(offline.shape),
            "dtype": str(offline.dtype),
            "nonzero_entries": int(torch.count_nonzero(offline)),
            "self_loops": int(torch.count_nonzero(torch.diag(offline))),
            "symmetric": bool(torch.equal(offline, offline.T)),
            "construction": "offline relation1, already symmetric",
        },
        "assertions": assertions,
        "all_assertions_passed": bool(all(item["passed"] for item in assertions.values())),
        "conclusion": "encoder/reconstruction adjacency and functional-map spectral basis come from the same graph",
    }


def _embedding_diagnostics(embedding: torch.Tensor) -> Dict[str, Any]:
    embedding = embedding.detach().cpu().float()
    return {
        "norm": _distribution(torch.linalg.vector_norm(embedding, dim=1)),
        "mean_dimension_variance": float(embedding.var(dim=0, unbiased=False).mean()),
        "maximum_dimension_variance": float(embedding.var(dim=0, unbiased=False).max()),
        "finite": bool(torch.isfinite(embedding).all()),
    }


def _rank_metrics(ranks: torch.Tensor) -> Dict[str, float]:
    ranks = ranks.detach().cpu().double()
    result = {f"Hits@{k}": float((ranks <= k).double().mean()) for k in KS}
    result["MRR"] = float((1.0 / ranks).mean())
    result["Mean Rank"] = float(ranks.mean())
    return result


# 功能：对捕获距离比较参考乐观、argsort、平均与悲观同分排名，返回指标及并列统计。
def evaluate_distance(distance: torch.Tensor, test_idx: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    distance = distance.detach().cpu().double()
    source = torch.as_tensor(test_idx, dtype=torch.long)
    truth_labels = torch.as_tensor(labels, dtype=torch.long)
    rows = distance[source]
    truth = rows[torch.arange(len(source)), truth_labels]
    less = (rows < (truth[:, None] - EPSILON)).sum(dim=1)
    equal = (torch.abs(rows - truth[:, None]) <= EPSILON).sum(dim=1)
    optimistic = less.double() + 1.0
    average = less.double() + (equal.double() + 1.0) / 2.0
    pessimistic = less.double() + equal.double()
    order = torch.argsort(rows, dim=1)
    argsort_rank = (order == truth_labels[:, None]).nonzero(as_tuple=False)[:, 1].double() + 1.0
    minimum = rows.min(dim=1).values
    minimum_ties = (torch.abs(rows - minimum[:, None]) <= EPSILON).sum(dim=1)
    truth_in_minimum = torch.abs(truth - minimum) <= EPSILON
    truth_unique = truth_in_minimum & (minimum_ties == 1)
    return {
        "public_optimistic": _rank_metrics(optimistic),
        "tgae_argsort": _rank_metrics(argsort_rank),
        "average_tie": _rank_metrics(average),
        "pessimistic_tie": _rank_metrics(pessimistic),
        "tie_diagnostics": {
            "truth_unique_nearest_neighbor_ratio": float(truth_unique.double().mean()),
            "truth_in_minimum_tie_group_ratio": float(truth_in_minimum.double().mean()),
            "minimum_tie_candidate_count": _distribution(minimum_ties),
            "maximum_tie_count": int(minimum_ties.max()),
        },
    }


def _reconstruction_loss(embedding: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    prediction = torch.sigmoid(embedding @ embedding.T)
    adjacency = adjacency.to(device=embedding.device, dtype=prediction.dtype).to_dense()
    n = adjacency.shape[0]
    positive = adjacency.sum()
    pos_weight = (n * n - positive) / positive
    norm = n * n / ((n * n - positive) * 2.0)
    weights = torch.ones_like(adjacency)
    weights[adjacency == 1] = pos_weight
    return norm * F.binary_cross_entropy(prediction.reshape(-1), adjacency.reshape(-1), weight=weights.reshape(-1))


# 功能：在固定模型状态下捕获嵌入与距离，重算评估/损失并写张量工件，不执行优化。
def _checkpoint(
    model: torch.nn.Module,
    train_adj: Mapping[str, List[torch.Tensor]],
    train_features: Mapping[str, List[torch.Tensor]],
    test_pairs: np.ndarray,
    args: Any,
    epoch: int,
    artifact_path: Path,
) -> Dict[str, Any]:
    keys = list(train_adj.keys())
    adjacency1 = train_adj[keys[0]][0]
    adjacency2 = train_adj[keys[1]][0]
    encoder_adjacency1 = sp.coo_matrix(adjacency1.cpu().numpy())
    encoder_adjacency2 = sp.coo_matrix(adjacency2.cpu().numpy())
    x1 = train_features[keys[0]][0]
    x2 = train_features[keys[1]][0]
    eig_root = Path("eigs") / "Douban Online_Offline"
    phi1 = torch.load(eig_root / "phi1.pth")
    phi2 = torch.load(eig_root / "phi2.pth")
    lam1 = torch.load(eig_root / "lam1.pth")
    lam2 = torch.load(eig_root / "lam2.pth")
    model.eval()
    with torch.no_grad():
        z1_high, z1_low, z2_high, z2_low = model(
            x1, encoder_adjacency1, x2, encoder_adjacency2
        )
        z1 = torch.cat([z1_high, z1_low], dim=1)
        z2 = torch.cat([z2_high, z2_low], dim=1)
        fm_loss, map_loss = model.DF_map(z1, z2, phi1, phi2, lam1, lam2, args)
        reconstruction1 = _reconstruction_loss(z1, adjacency1)
        reconstruction2 = _reconstruction_loss(z2, adjacency2)
        normalized1 = F.normalize(z1, p=2, dim=1)
        normalized2 = F.normalize(z2, p=2, dim=1)
        distance = torch.cdist(normalized1.double(), normalized2.double(), p=2)
    test_idx = test_pairs[0, :].astype(np.int32)
    labels = test_pairs[1, :].astype(np.int32)
    schedule = max(1.0 * (1 - epoch / args.epoch), 0.5) if args.use_fm_decay else 1.0
    reconstruction = reconstruction1 + reconstruction2
    total = reconstruction + schedule * (fm_loss + map_loss)
    torch.save(
        {
            "epoch": epoch,
            "source_embedding": z1.detach().cpu(),
            "target_embedding": z2.detach().cpu(),
            "source_normalized_embedding": normalized1.detach().cpu(),
            "target_normalized_embedding": normalized2.detach().cpu(),
            "distance": distance.detach().cpu(),
            "test_idx": test_idx,
            "labels": labels,
        },
        artifact_path,
    )
    return {
        "epoch": epoch,
        "artifact": str(artifact_path),
        "evaluation": evaluate_distance(distance, test_idx, labels),
        "losses": {
            "total": float(total),
            "reconstruction": float(reconstruction),
            "reconstruction_source": float(reconstruction1),
            "reconstruction_target": float(reconstruction2),
            "functional_map": float(fm_loss),
            "map_regularization": float(map_loss),
            "fm_schedule_weight": float(schedule),
        },
        "embedding": {
            "source": _embedding_diagnostics(z1),
            "target": _embedding_diagnostics(z2),
        },
        "distance_finite": bool(torch.isfinite(distance).all()),
    }


def _make_args(main_module: Any, load_data: Any) -> Any:
    previous = sys.argv
    try:
        sys.argv = [
            "main_GADL.py",
            "--dataset",
            "Douban Online_Offline",
            "--data_path",
            "data",
            "--device",
            "0",
        ]
        args = main_module.parse_args()
    finally:
        sys.argv = previous
    args, _ = load_data.load_config("config.yaml", args.dataset, args)
    return args


# 核心思路：设定种子后包装参考 run_GADL、映射与重构观察器，再调用参考 main() 继续原训练。
# 输出结果：该种子的固定轮次记录、轨迹与资源信息交给 main()。
def _run_seed(
    seed: int,
    main_module: Any,
    load_data: Any,
    output_dir: Path,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    args = _make_args(main_module, load_data)
    captured: Dict[str, Any] = {}
    original_run = main_module.run_GADL

    # 输入来源：_run_seed() 临时替换参考模块的 run_GADL，参考 main() 调用它时进入此包装器。
    # 核心思路：训练前后取检查点，临时观察函数映射与重构损失，再调用 original_run() 执行原训练。
    # 结果去向：轨迹与检查点写入 captured；finally 恢复观察方法，外层再恢复原 run_GADL。
    def instrumented_run(data: str, model: torch.nn.Module, train_adj: Any, train_features: Any, run_args: Any, device: Any, test_pairs: Any) -> None:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        captured["epoch_0"] = _checkpoint(
            model, train_adj, train_features, test_pairs, run_args, 0, seed_dir / "epoch_0.pt"
        )
        fm_records: List[Tuple[float, float]] = []
        bce_records: List[float] = []
        original_df_map = model.DF_map
        original_bce = main_module.F.binary_cross_entropy

        # 功能：调用模型原 DF_map() 后记录分离出的标量损失，原返回值继续交给参考训练。
        def observed_df_map(self: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            result = original_df_map(*call_args, **call_kwargs)
            fm_records.append((float(result[0].detach()), float(result[1].detach())))
            return result

        # 功能：调用原二元交叉熵后记录标量，原损失张量继续参与参考训练的反向传播。
        def observed_bce(*call_args: Any, **call_kwargs: Any) -> Any:
            result = original_bce(*call_args, **call_kwargs)
            bce_records.append(float(result.detach()))
            return result

        model.DF_map = types.MethodType(observed_df_map, model)
        main_module.F.binary_cross_entropy = observed_bce
        started = time.perf_counter()
        try:
            original_run(data, model, train_adj, train_features, run_args, device, test_pairs)
        finally:
            captured["training_wall_seconds"] = time.perf_counter() - started
            model.DF_map = original_df_map
            main_module.F.binary_cross_entropy = original_bce
        captured["epoch_100"] = _checkpoint(
            model, train_adj, train_features, test_pairs, run_args, 100, seed_dir / "epoch_100.pt"
        )
        if len(fm_records) != run_args.epoch or len(bce_records) != 2 * run_args.epoch:
            raise RuntimeError(
                f"incomplete training observations: fm={len(fm_records)}, bce={len(bce_records)}"
            )
        keys = list(train_adj.keys())
        adjacency1 = train_adj[keys[0]][0]
        adjacency2 = train_adj[keys[1]][0]
        n1, n2 = adjacency1.shape[0], adjacency2.shape[0]
        positive1, positive2 = float(adjacency1.sum()), float(adjacency2.sum())
        norm1 = n1 * n1 / ((n1 * n1 - positive1) * 2.0)
        norm2 = n2 * n2 / ((n2 * n2 - positive2) * 2.0)
        trajectory = []
        for step in range(1, run_args.epoch + 1):
            fm_loss, map_loss = fm_records[step - 1]
            reconstruction = norm1 * bce_records[2 * (step - 1)] + norm2 * bce_records[2 * (step - 1) + 1]
            schedule = max(1.0 * (1 - step / run_args.epoch), 0.5) if run_args.use_fm_decay else 1.0
            trajectory.append(
                {
                    "epoch": step,
                    "reconstruction": reconstruction,
                    "functional_map": fm_loss,
                    "map_regularization": map_loss,
                    "fm_schedule_weight": schedule,
                    "total": reconstruction + schedule * (fm_loss + map_loss),
                }
            )
        captured["training_loss_trajectory"] = trajectory

    main_module.run_GADL = instrumented_run
    try:
        main_module.main(args)
    finally:
        main_module.run_GADL = original_run

    health = {
        "epoch_0_finite": bool(captured["epoch_0"]["distance_finite"]),
        "epoch_100_finite": bool(captured["epoch_100"]["distance_finite"]),
        "losses_finite": bool(
            all(math.isfinite(item[key]) for item in captured["training_loss_trajectory"] for key in ("reconstruction", "functional_map", "map_regularization", "total"))
        ),
        "embedding_collapse": bool(
            min(
                captured["epoch_100"]["embedding"]["source"]["mean_dimension_variance"],
                captured["epoch_100"]["embedding"]["target"]["mean_dimension_variance"],
            )
            <= 1e-12
        ),
    }
    health["healthy"] = bool(
        health["epoch_0_finite"]
        and health["epoch_100_finite"]
        and health["losses_finite"]
        and not health["embedding_collapse"]
    )
    captured.update(
        {
            "seed": seed,
            "configuration": {
                "input_dim": args.input_dim,
                "hidden_dim": args.hidden_dim,
                "num_hidden_layers": args.num_hidden_layers,
                "spectral_rank": args.k,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epoch,
                "alpha": args.alpha,
                "beta": args.beta,
                "l_bij": args.l_bij,
                "l_ort": args.l_ort,
                "residual": args.residual,
                "use_fm_decay": args.use_fm_decay,
            },
            "health": health,
            "peak_memory": {
                "rss_native_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            },
        }
    )
    return captured


def _delta(after: Mapping[str, float], before: Mapping[str, float]) -> Dict[str, float]:
    return {key: float(after[key] - before[key]) for key in after if key in before}


def _aggregate(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


# 功能：按检查点汇总各种子的对应指标，不选最佳轮次。
def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {"epoch_0": {}, "epoch_100": {}, "epoch_100_minus_epoch_0": {}}
    policies = ("public_optimistic", "tgae_argsort", "average_tie", "pessimistic_tie")
    metrics = ("Hits@1", "Hits@5", "Hits@10", "Hits@50", "MRR", "Mean Rank")
    for checkpoint in ("epoch_0", "epoch_100"):
        for policy in policies:
            output[checkpoint][policy] = {
                metric: _aggregate(result[checkpoint]["evaluation"][policy][metric] for result in results)
                for metric in metrics
            }
    for policy in policies:
        output["epoch_100_minus_epoch_0"][policy] = {
            metric: _aggregate(
                result["epoch_100"]["evaluation"][policy][metric]
                - result["epoch_0"]["evaluation"][policy][metric]
                for result in results
            )
            for metric in metrics
        }
    output["tie_diagnostics_epoch_100"] = {
        key: _aggregate(result["epoch_100"]["evaluation"]["tie_diagnostics"][key] for result in results)
        for key in ("truth_unique_nearest_neighbor_ratio", "truth_in_minimum_tie_group_ratio", "maximum_tie_count")
    }
    output["minimum_tie_candidate_mean_epoch_100"] = _aggregate(
        result["epoch_100"]["evaluation"]["tie_diagnostics"]["minimum_tie_candidate_count"]["mean"]
        for result in results
    )
    return output


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


# 功能：将已生成的结果写为人可读报告，不重新训练或评估。
def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    aggregate = payload["aggregate"]
    lines = [
        "# GADL graph-consistent public-reference reproduction",
        "",
        "All pre-training graph/spectral assertions passed. Encoder/reconstruction adjacency and bundled functional-map bases come from the same graph.",
        "",
        "| Evaluator | H@1 | H@5 | H@10 | H@50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in ("public_optimistic", "tgae_argsort", "average_tie", "pessimistic_tie"):
        metrics = aggregate["epoch_100"][policy]
        lines.append(
            f"| {policy} | {_percent(metrics['Hits@1']['mean'])} +/- {_percent(metrics['Hits@1']['std'])} | "
            f"{_percent(metrics['Hits@5']['mean'])} | {_percent(metrics['Hits@10']['mean'])} | {_percent(metrics['Hits@50']['mean'])} |"
        )
    lines.extend(
        [
            "",
            f"Mixed public optimistic H@1: {_percent(MIXED_OPTIMISTIC_H1)}",
            f"Graph-consistent public optimistic H@1: {_percent(aggregate['epoch_100']['public_optimistic']['Hits@1']['mean'])}",
            f"Difference: {payload['comparisons']['graph_consistent_minus_mixed_optimistic_Hits@1_percentage_points']:.2f} percentage points",
            "",
            "No checkpoint selection or hyperparameter search was performed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("/private/tmp/gadl-reference-run.6L34Eb"),
    )
    parser.add_argument(
        "--dependency-root", type=Path, default=Path("/private/tmp/gadl-reference-deps")
    )
    parser.add_argument(
        "--shim-root", type=Path, default=Path("/private/tmp/gadl-reference-shims")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/gadl_one_to_one/gadl_reference_graph_consistent_artifacts"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/gadl_one_to_one/gadl_reference_graph_consistent.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("logs/gadl_one_to_one/gadl_reference_graph_consistent.md"),
    )
    return parser.parse_args()


# 功能：准备隔离目录和图断言，导入参考代码，逐种子运行并写出 JSON、捕获工件和报告。
def main() -> None:
    args = parse_args()
    workspace = Path.cwd()
    args.output_dir = (workspace / args.output_dir).resolve()
    args.output_json = (workspace / args.output_json).resolve()
    args.output_report = (workspace / args.output_report).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    consistency = prepare_and_assert_graphs(args.runtime_root)
    if not consistency["all_assertions_passed"]:
        raise RuntimeError("graph consistency assertions failed")

    if not hasattr(sp.spmatrix, "A"):
        sp.spmatrix.A = property(lambda self: self.toarray())
    original_torch_load = torch.load
    torch.load = lambda *call_args, **call_kwargs: original_torch_load(
        *call_args, **({"weights_only": False} | call_kwargs)
    )
    sys.path.insert(0, str(args.shim_root))
    sys.path.insert(1, str(args.runtime_root))
    sys.path.append(str(args.dependency_root))
    os.chdir(args.runtime_root)
    try:
        import load_data
        import main_GADL

        load_data.load_adj.__defaults__ = ("data",)
        results = []
        seed_zero = _run_seed(0, main_GADL, load_data, args.output_dir)
        results.append(seed_zero)
        if not seed_zero["health"]["healthy"]:
            raise RuntimeError(f"seed 0 failed health checks: {seed_zero['health']}")
        for seed in (1, 2):
            result = _run_seed(seed, main_GADL, load_data, args.output_dir)
            if not result["health"]["healthy"]:
                raise RuntimeError(f"seed {seed} failed health checks: {result['health']}")
            results.append(result)
    finally:
        os.chdir(workspace)
        torch.load = original_torch_load

    aggregate = aggregate_results(results)
    public_h1 = aggregate["epoch_100"]["public_optimistic"]["Hits@1"]["mean"]
    evaluator_names = ("public_optimistic", "tgae_argsort", "average_tie")
    comparisons = {
        "graph_consistent_minus_mixed_optimistic_Hits@1_percentage_points": 100.0 * (public_h1 - MIXED_OPTIMISTIC_H1),
        "optimistic_tie_inflation_vs_argsort_percentage_points": 100.0 * (
            public_h1 - aggregate["epoch_100"]["tgae_argsort"]["Hits@1"]["mean"]
        ),
        "optimistic_tie_inflation_vs_average_percentage_points": 100.0 * (
            public_h1 - aggregate["epoch_100"]["average_tie"]["Hits@1"]["mean"]
        ),
        "paper_deltas_epoch_100": {
            evaluator: {
                metric: 100.0 * (aggregate["epoch_100"][evaluator][metric]["mean"] - paper_value)
                for metric, paper_value in PAPER.items()
            }
            for evaluator in evaluator_names
        },
    }
    closest = min(
        evaluator_names,
        key=lambda name: abs(aggregate["epoch_100"][name]["Hits@1"]["mean"] - PAPER["Hits@1"]),
    )
    comparisons["closest_to_paper_by_Hits@1"] = closest
    payload = {
        "experiment": "gadl_graph_consistent_public_reference_reproduction",
        "seeds": list(SEEDS),
        "graph_consistency": consistency,
        "results": results,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "constraints": {
            "reference_source_modified": False,
            "planetalign_modified": False,
            "hyperparameter_search": False,
            "checkpoint_selection": False,
            "m2m_used": False,
        },
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_report(args.output_report, payload)
    print(json.dumps({"comparisons": comparisons, "epoch_100": aggregate["epoch_100"]}, indent=2))


if __name__ == "__main__":
    main()
