#!/usr/bin/env python3
# 文件作用：检查 GADL 公开参考数据的图关系、谱文件及最终距离矩阵的同分评估协议。
# 数据来源：main() 从 --reference-root 读取参考 .mat、谱文件和捕获的距离矩阵。
# 核心处理：独立重建候选图与拉普拉斯，核对特征基，再比较距离排名的同分策略。
# 结果去向：诊断 JSON 供研究比较；本文件不调用 PlanetAlign 的 GADL 训练。

"""Diagnose the public GADL Douban reference data and evaluation protocol.

This script is intentionally independent from PlanetAlign's GADL and ranking
modules. It reads the reference repository and a captured final distance
matrix, but never modifies reference or production code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import scipy.io
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.stats import pearsonr, spearmanr
import torch


EPSILON = 1e-8
KS = (1, 5, 10, 50)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _dense(adjacency: Any) -> np.ndarray:
    if sp.issparse(adjacency):
        return adjacency.toarray()
    return np.asarray(adjacency)


def _binary_csr(adjacency: Any) -> sp.csr_matrix:
    matrix = sp.csr_matrix(adjacency)
    matrix.data = (matrix.data != 0).astype(np.int32)
    matrix.eliminate_zeros()
    return matrix.astype(np.int32)


def _full_union(relations: Iterable[Any]) -> sp.csr_matrix:
    relations = list(relations)
    result = _binary_csr(relations[0])
    for relation in relations[1:]:
        result = result + _binary_csr(relation)
    result.data[:] = 1
    result.eliminate_zeros()
    return result


def _symmetrize_binary(adjacency: Any) -> sp.csr_matrix:
    result = _binary_csr(adjacency)
    result = result + result.T
    result.data[:] = 1
    result.eliminate_zeros()
    return result.astype(np.int32)


def _distribution(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def _adjacency_summary(adjacency: sp.csr_matrix) -> Dict[str, Any]:
    diagonal = np.asarray(adjacency.diagonal())
    asymmetry = adjacency - adjacency.T
    return {
        "shape": list(adjacency.shape),
        "dtype": str(adjacency.dtype),
        "nnz_entries": int(adjacency.nnz),
        "sum": float(adjacency.sum()),
        "diagonal_nonzero": int(np.count_nonzero(diagonal)),
        "off_diagonal_nnz": int(adjacency.nnz - np.count_nonzero(diagonal)),
        "undirected_upper_triangle_edges_excluding_self_loops": int(sp.triu(adjacency, k=1).nnz),
        "symmetric": bool(asymmetry.nnz == 0),
    }


def _edge_set(adjacency: sp.csr_matrix) -> set[Tuple[int, int]]:
    upper = sp.triu(adjacency, k=1).tocoo()
    return set(zip(upper.row.tolist(), upper.col.tolist()))


def _safe_correlation(function: Any, left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(function(left, right).statistic)


# 功能：在真实对应节点诱导的两图子图上比较边重叠、度数与相关性。
def aligned_subgraph_diagnostics(
    online: sp.csr_matrix,
    offline: sp.csr_matrix,
    ground_truth: np.ndarray,
) -> Dict[str, Any]:
    pairs = np.asarray(ground_truth, dtype=np.int64) - 1
    online_ids = pairs[:, 0]
    offline_ids = pairs[:, 1]
    online_induced = online[online_ids][:, online_ids].tocsr()
    offline_induced = offline[offline_ids][:, offline_ids].tocsr()
    online_edges = _edge_set(online_induced)
    offline_edges = _edge_set(offline_induced)
    intersection = online_edges & offline_edges
    union = online_edges | offline_edges

    induced_online_degree = np.asarray(online_induced.sum(axis=1)).reshape(-1)
    induced_offline_degree = np.asarray(offline_induced.sum(axis=1)).reshape(-1)
    full_online_degree = np.asarray(online.sum(axis=1)).reshape(-1)[online_ids]
    full_offline_degree = np.asarray(offline.sum(axis=1)).reshape(-1)[offline_ids]
    signed_difference = induced_online_degree - induced_offline_degree
    absolute_difference = np.abs(signed_difference)

    return {
        "ground_truth": {
            "count": int(len(pairs)),
            "online_id_unique": int(np.unique(online_ids).size),
            "offline_id_unique": int(np.unique(offline_ids).size),
            "online_id_range_zero_based": [int(online_ids.min()), int(online_ids.max())],
            "offline_id_range_zero_based": [int(offline_ids.min()), int(offline_ids.max())],
        },
        "induced_subgraphs": {
            "online_entries": int(online_induced.nnz),
            "offline_entries": int(offline_induced.nnz),
            "online_undirected_edges": int(len(online_edges)),
            "offline_undirected_edges": int(len(offline_edges)),
            "edge_intersection": int(len(intersection)),
            "edge_union": int(len(union)),
            "jaccard": float(len(intersection) / len(union)) if union else 1.0,
            "online_to_offline_edge_recall": float(len(intersection) / len(online_edges)) if online_edges else 1.0,
            "offline_to_online_edge_recall": float(len(intersection) / len(offline_edges)) if offline_edges else 1.0,
            "edge_symmetric_difference": int(len(online_edges ^ offline_edges)),
            "exactly_equal_after_ground_truth_permutation": bool(online_edges == offline_edges),
        },
        "degree_comparison": {
            "induced_pearson": _safe_correlation(pearsonr, induced_online_degree, induced_offline_degree),
            "induced_spearman": _safe_correlation(spearmanr, induced_online_degree, induced_offline_degree),
            "full_graph_anchor_degree_pearson": _safe_correlation(pearsonr, full_online_degree, full_offline_degree),
            "full_graph_anchor_degree_spearman": _safe_correlation(spearmanr, full_online_degree, full_offline_degree),
            "induced_online_degree": _distribution(induced_online_degree),
            "induced_offline_degree": _distribution(induced_offline_degree),
            "signed_online_minus_offline": _distribution(signed_difference),
            "absolute_difference": _distribution(absolute_difference),
            "equal_degree_ratio": float(np.mean(signed_difference == 0)),
        },
        "classification_evidence": {
            "different_total_node_counts": bool(online.shape[0] != offline.shape[0]),
            "aligned_subgraphs_same_size_but_not_isomorphic_under_ground_truth": bool(online_edges != offline_edges),
            "classification": "B_real_online_offline_graph_pair",
            "reason": "Different graph sizes and low edge overlap after applying the provided real-user correspondence are inconsistent with a perturb-and-permute copy.",
        },
    }


def reference_laplacian(adjacency: sp.csr_matrix) -> sp.coo_matrix:
    # Exact source path: torch dense/sparse COO -> scipy COO ->
    # scipy.sparse.csgraph.laplacian(normed=True). No binarization and no
    # explicit self-loop insertion/removal occur in compute_laplacian_eigenbasis.
    coo = sp.coo_matrix(adjacency)
    return sp.coo_matrix(csgraph.laplacian(coo, normed=True))


def _sparse_pattern(matrix: sp.spmatrix) -> set[Tuple[int, int]]:
    coo = matrix.tocoo()
    nonzero = coo.data != 0
    return set(zip(coo.row[nonzero].tolist(), coo.col[nonzero].tolist()))


# 功能：比较候选与已有稀疏矩阵的非零位置、数值误差和对角线。
def matrix_comparison(candidate: sp.spmatrix, bundled: sp.spmatrix) -> Dict[str, Any]:
    candidate = candidate.tocsr()
    bundled = bundled.tocsr()
    difference = (candidate - bundled).tocoo()
    absolute = np.abs(difference.data)
    candidate_pattern = _sparse_pattern(candidate)
    bundled_pattern = _sparse_pattern(bundled)
    diagonal_candidate = np.asarray(candidate.diagonal())
    diagonal_bundled = np.asarray(bundled.diagonal())
    row_sums = np.asarray(candidate.sum(axis=1)).reshape(-1)
    symmetry = candidate - candidate.T
    return {
        "shape": list(candidate.shape),
        "candidate_dtype": str(candidate.dtype),
        "bundled_dtype": str(bundled.dtype),
        "candidate_nnz": int(candidate.nnz),
        "bundled_nnz": int(bundled.nnz),
        "nonzero_pattern_equal": bool(candidate_pattern == bundled_pattern),
        "pattern_only_in_candidate": int(len(candidate_pattern - bundled_pattern)),
        "pattern_only_in_bundled": int(len(bundled_pattern - candidate_pattern)),
        "max_absolute_error": float(absolute.max()) if absolute.size else 0.0,
        "mean_absolute_error_dense_domain": float(absolute.sum() / (candidate.shape[0] * candidate.shape[1])),
        "frobenius_error": float(np.linalg.norm(difference.data)),
        "allclose_atol_1e-10_rtol_1e-8": bool(
            candidate_pattern == bundled_pattern
            and np.allclose(candidate.data, bundled.data, atol=1e-10, rtol=1e-8)
        ),
        "candidate_diagonal": _distribution(diagonal_candidate),
        "bundled_diagonal": _distribution(diagonal_bundled),
        "candidate_row_sum": _distribution(row_sums),
        "candidate_symmetric": bool(symmetry.nnz == 0 or np.max(np.abs(symmetry.data)) <= 1e-12),
    }


# 功能：检查谱值顺序、特征向量正交性及拉普拉斯特征方程残差。
def eigenpair_diagnostics(
    laplacian: sp.spmatrix,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
) -> Dict[str, Any]:
    matrix = laplacian.tocsr().astype(np.float64)
    values = eigenvalues.detach().cpu().double().numpy()
    vectors = eigenvectors.detach().cpu().double().numpy()
    residual = matrix @ vectors - vectors * values.reshape(1, -1)
    lhs = matrix @ vectors
    per_vector = np.linalg.norm(residual, axis=0)
    gram = vectors.T @ vectors
    identity = np.eye(vectors.shape[1])
    gaps = np.diff(values)
    return {
        "laplacian_shape": list(matrix.shape),
        "lambda_shape": list(values.shape),
        "phi_shape": list(vectors.shape),
        "lambda_sorted_non_decreasing": bool(np.all(gaps >= -1e-7)),
        "lambda_range": [float(values.min()), float(values.max())],
        "near_repeated_gap_le_1e-6_count": int(np.sum(np.abs(gaps) <= 1e-6)),
        "residual_frobenius": float(np.linalg.norm(residual)),
        "relative_residual_frobenius": float(np.linalg.norm(residual) / max(np.linalg.norm(lhs), 1e-30)),
        "residual_max_absolute": float(np.max(np.abs(residual))),
        "per_eigenvector_residual": _distribution(per_vector),
        "orthonormality_frobenius": float(np.linalg.norm(gram - identity)),
        "valid_eigenpairs_relative_residual_le_1e-4": bool(
            np.linalg.norm(residual) / max(np.linalg.norm(lhs), 1e-30) <= 1e-4
        ),
        "note": "Residual validation is invariant to eigenvector sign and to basis rotations inside repeated eigenspaces.",
    }


# 功能：重建不同参考图口径，和缓存拉普拉斯/谱基比较，返回各候选的诊断记录。
def spectral_diagnostics(reference_root: Path, mat: Mapping[str, Any]) -> Dict[str, Any]:
    online_relations = [_binary_csr(item) for item in mat["online_edge_label"][0]]
    offline_relations = [_binary_csr(item) for item in mat["offline_edge_label"][0]]
    candidates = {
        "online_relation_0": online_relations[0],
        "online_relation_0_symmetrized": _symmetrize_binary(online_relations[0]),
        "online_relation_1": online_relations[1],
        "online_relation_1_symmetrized": _symmetrize_binary(online_relations[1]),
        "online_union_of_relations": _full_union(online_relations),
        "online_full_field": _binary_csr(mat["online"]),
        "offline_relation_0": offline_relations[0],
        "offline_relation_0_symmetrized": _symmetrize_binary(offline_relations[0]),
        "offline_relation_1": offline_relations[1],
        "offline_relation_1_symmetrized": _symmetrize_binary(offline_relations[1]),
        "offline_union_of_relations": _full_union(offline_relations),
        "offline_full_field": _binary_csr(mat["offline"]),
    }
    eig_root = reference_root / "eigs" / "Douban Online_Offline"
    bundled_l1 = torch.load(eig_root / "L1.pth", weights_only=False)
    bundled_l2 = torch.load(eig_root / "L2.pth", weights_only=False)
    lam1 = torch.load(eig_root / "lam1.pth", weights_only=False)
    lam2 = torch.load(eig_root / "lam2.pth", weights_only=False)
    phi1 = torch.load(eig_root / "phi1.pth", weights_only=False)
    phi2 = torch.load(eig_root / "phi2.pth", weights_only=False)

    comparisons: Dict[str, Any] = {}
    for name, adjacency in candidates.items():
        side = "online" if name.startswith("online") else "offline"
        bundled = bundled_l1 if side == "online" else bundled_l2
        comparisons[name] = {
            "adjacency": _adjacency_summary(adjacency),
            "laplacian": matrix_comparison(reference_laplacian(adjacency), bundled),
        }

    matching = [
        name
        for name, result in comparisons.items()
        if result["laplacian"]["allclose_atol_1e-10_rtol_1e-8"]
    ]
    relation_union_matches_full = {
        "online": bool((candidates["online_union_of_relations"] != candidates["online_full_field"]).nnz == 0),
        "offline": bool((candidates["offline_union_of_relations"] != candidates["offline_full_field"]).nnz == 0),
    }
    return {
        "source_trace": {
            "loader": "load_douban reads relation-1, but import_data discards those adjacency returns and load_adj reads missing online.pt/offline.pt",
            "artifact_generation": "main_GADL.run_GADL calls compute_laplacian_eigenbasis(A, k=300, norm_type='sym') only when eigs directory is absent",
            "conversion": "torch tensor -> sparse COO/coalesce/CPU -> scipy.sparse.coo_matrix preserving values and dtype",
            "binarization": "none inside compute_laplacian_eigenbasis",
            "self_loops": "none added; scipy csgraph.laplacian ignores adjacency diagonal when constructing L",
            "laplacian": "scipy.sparse.csgraph.laplacian(A, normed=True)",
            "eigendecomposition": "scipy.sparse.linalg.eigsh(L, k=300, which='SM'); lambda/phi converted to torch.float32",
        },
        "relation_union_matches_full_field": relation_union_matches_full,
        "bundled": {
            "L1": {"shape": list(bundled_l1.shape), "dtype": str(bundled_l1.dtype), "nnz": int(bundled_l1.nnz)},
            "L2": {"shape": list(bundled_l2.shape), "dtype": str(bundled_l2.dtype), "nnz": int(bundled_l2.nnz)},
        },
        "candidate_comparisons": comparisons,
        "matching_candidates": matching,
        "classification": (
            "bundled_L_matches_full_graph"
            if "online_full_field" in matching and "offline_full_field" in matching
            else "bundled_L_matches_symmetrized_relation_1_online_and_relation_1_offline"
            if "online_relation_1_symmetrized" in matching and "offline_relation_1" in matching
            else "bundled_L_matches_relation_1"
            if "online_relation_1" in matching and "offline_relation_1" in matching
            else "bundled_L_matches_other_or_no_candidate"
        ),
        "eigenpairs": {
            "online": eigenpair_diagnostics(bundled_l1, lam1, phi1),
            "offline": eigenpair_diagnostics(bundled_l2, lam2, phi2),
        },
    }


def _ranking_metrics(ranks: torch.Tensor) -> Dict[str, float]:
    ranks = ranks.double()
    result = {f"Hits@{k}": float((ranks <= k).double().mean()) for k in KS}
    result["MRR"] = float((1.0 / ranks).mean())
    result["Mean Rank"] = float(ranks.mean())
    return result


# 功能：读取捕获的距离矩阵，统计严格、近似同分、平均与悲观排名及最高分并列影响。
def distance_tie_diagnostics(distance_path: Path) -> Dict[str, Any]:
    payload = torch.load(distance_path, weights_only=False)
    distance = payload["distance"].detach().cpu().double()
    test_idx = torch.as_tensor(payload["test_idx"], dtype=torch.long)
    labels = torch.as_tensor(payload["labels"], dtype=torch.long)
    rows = distance[test_idx]
    truth = rows[torch.arange(len(test_idx)), labels]

    strictly_less_epsilon = (rows < (truth[:, None] - EPSILON)).sum(dim=1)
    epsilon_equal_truth = (torch.abs(rows - truth[:, None]) <= EPSILON).sum(dim=1)
    exact_less = (rows < truth[:, None]).sum(dim=1)
    exact_equal_truth = (rows == truth[:, None]).sum(dim=1)

    optimistic = strictly_less_epsilon.double() + 1.0
    average = strictly_less_epsilon.double() + (epsilon_equal_truth.double() + 1.0) / 2.0
    pessimistic = strictly_less_epsilon.double() + epsilon_equal_truth.double()
    exact_average = exact_less.double() + (exact_equal_truth.double() + 1.0) / 2.0
    exact_pessimistic = exact_less.double() + exact_equal_truth.double()

    order = torch.argsort(rows, dim=1)
    argsort_rank = (order == labels[:, None]).nonzero(as_tuple=False)[:, 1].double() + 1.0

    row_minimum = rows.min(dim=1).values
    min_ties_epsilon = (torch.abs(rows - row_minimum[:, None]) <= EPSILON).sum(dim=1)
    min_ties_exact = (rows == row_minimum[:, None]).sum(dim=1)
    truth_equals_minimum_exact = truth == row_minimum
    truth_in_minimum_epsilon = torch.abs(truth - row_minimum) <= EPSILON
    unique_truth_nearest_epsilon = truth_in_minimum_epsilon & (min_ties_epsilon == 1)

    sorted_rows = torch.sort(rows, dim=1).values
    adjacent = torch.diff(sorted_rows, dim=1)
    any_exact_tie = (adjacent == 0).any(dim=1)
    any_epsilon_tie = (adjacent <= EPSILON).any(dim=1)

    policies = {
        "public_reference_optimistic_epsilon_1e-8": _ranking_metrics(optimistic),
        "average_tie_epsilon_1e-8": _ranking_metrics(average),
        "pessimistic_tie_epsilon_1e-8": _ranking_metrics(pessimistic),
        "torch_argsort_deterministic_current_runtime": _ranking_metrics(argsort_rank),
        "average_tie_exact_value": _ranking_metrics(exact_average),
        "pessimistic_tie_exact_value": _ranking_metrics(exact_pessimistic),
    }
    optimistic_h1 = policies["public_reference_optimistic_epsilon_1e-8"]["Hits@1"]
    return {
        "capture_metadata": payload.get("metadata", {}),
        "distance": {
            "shape": list(distance.shape),
            "dtype": str(distance.dtype),
            "finite": bool(torch.isfinite(distance).all()),
            "test_pair_count": int(len(test_idx)),
            "epsilon": EPSILON,
        },
        "policies": policies,
        "tie_inflation_percentage_points": {
            "optimistic_minus_average": 100.0 * (optimistic_h1 - policies["average_tie_epsilon_1e-8"]["Hits@1"]),
            "optimistic_minus_pessimistic": 100.0 * (optimistic_h1 - policies["pessimistic_tie_epsilon_1e-8"]["Hits@1"]),
            "optimistic_minus_torch_argsort": 100.0 * (optimistic_h1 - policies["torch_argsort_deterministic_current_runtime"]["Hits@1"]),
        },
        "truth_and_minimum": {
            "truth_distance_exactly_equals_row_minimum_ratio": float(truth_equals_minimum_exact.double().mean()),
            "truth_in_epsilon_minimum_tie_group_ratio": float(truth_in_minimum_epsilon.double().mean()),
            "truth_is_unique_nearest_neighbor_ratio": float(unique_truth_nearest_epsilon.double().mean()),
            "minimum_tie_candidates_epsilon": _distribution(min_ties_epsilon.numpy()),
            "minimum_tie_candidates_exact": _distribution(min_ties_exact.numpy()),
            "truth_tie_candidates_epsilon": _distribution(epsilon_equal_truth.numpy()),
            "truth_tie_candidates_exact": _distribution(exact_equal_truth.numpy()),
            "rows_with_any_exact_tie_ratio": float(any_exact_tie.double().mean()),
            "rows_with_any_epsilon_tie_ratio": float(any_epsilon_tie.double().mean()),
        },
        "epsilon_effect": {
            "average_Hits@1_epsilon_minus_exact_percentage_points": 100.0 * (
                policies["average_tie_epsilon_1e-8"]["Hits@1"]
                - policies["average_tie_exact_value"]["Hits@1"]
            ),
            "pessimistic_Hits@1_epsilon_minus_exact_percentage_points": 100.0 * (
                policies["pessimistic_tie_epsilon_1e-8"]["Hits@1"]
                - policies["pessimistic_tie_exact_value"]["Hits@1"]
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path(".reference/GADL"))
    parser.add_argument(
        "--distance-path",
        type=Path,
        default=Path("logs/gadl_one_to_one/gadl_reference_final_distance_seed0.pt"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/gadl_one_to_one/gadl_reference_protocol_diagnosis.json"),
    )
    return parser.parse_args()


# 功能：读取参考材料，执行图、谱与同分诊断，再将结果写入 --output-json。
def main() -> None:
    args = parse_args()
    mat = scipy.io.loadmat(args.reference_root / "data" / "douban.mat")
    online_relations = [_binary_csr(item) for item in mat["online_edge_label"][0]]
    offline_relations = [_binary_csr(item) for item in mat["offline_edge_label"][0]]
    online_full = _binary_csr(mat["online"])
    offline_full = _binary_csr(mat["offline"])

    spectral = spectral_diagnostics(args.reference_root, mat)
    tie = distance_tie_diagnostics(args.distance_path)
    reconstructed_encoder = {
        "online": _adjacency_summary(online_relations[1]),
        "offline": _adjacency_summary(offline_relations[1]),
    }
    spectral_matching = spectral["matching_candidates"]
    mixed = not (
        "online_relation_1" in spectral_matching and "offline_relation_1" in spectral_matching
    )

    payload = {
        "experiment": "gadl_reference_protocol_diagnosis",
        "reference_root": str(args.reference_root.resolve()),
        "data_identity": {
            "mat_keys": sorted(key for key in mat if not key.startswith("__")),
            "online_full": _adjacency_summary(online_full),
            "offline_full": _adjacency_summary(offline_full),
            "online_relation_0": _adjacency_summary(online_relations[0]),
            "online_relation_1": _adjacency_summary(online_relations[1]),
            "offline_relation_0": _adjacency_summary(offline_relations[0]),
            "offline_relation_1": _adjacency_summary(offline_relations[1]),
            "aligned_subgraph_analysis": aligned_subgraph_diagnostics(
                online_full, offline_full, mat["ground_truth"]
            ),
        },
        "spectral_artifact_analysis": spectral,
        "reconstructed_run_protocol": {
            "encoder_and_reconstruction_adjacency": reconstructed_encoder,
            "functional_map_spectral_matching_candidates": spectral_matching,
            "mixed_graph_protocol": mixed,
            "valid_reference_reproduction": not mixed,
            "classification": "mixed_graph_protocol" if mixed else "internally_consistent_relation_1_protocol",
        },
        "distance_tie_audit": tie,
        "tgae_evaluator_comparison": {
            "official_repository": "https://github.com/Jason-Tree/T-GAE",
            "file": "subgraphMatching.py",
            "tgae": "torch.argsort(dist_list), then find the ground-truth label position; no epsilon and no strict-less shortcut",
            "gadl": "count(dist < truth_dist - 1e-8); this assigns every epsilon tie the optimistic first position",
            "test_pair_direction_same": True,
            "gadl_changed_tie_behavior": True,
            "effect_on_this_distance_matrix_percentage_points": tie["tie_inflation_percentage_points"]["optimistic_minus_torch_argsort"],
        },
        "final_classification": {
            "douban_mat": "B_real_online_offline_graph_pair",
            "missing_online_pt_most_likely": (
                "symmetrized relation-1 adjacency with 14,430 entries, because bundled L1 matches relation-1 union its transpose"
                if "online_relation_1_symmetrized" in spectral_matching
                else "unknown: none of the tested candidates matches bundled L1"
            ),
            "missing_offline_pt_most_likely": "offline relation-1/full field; those candidates are identical",
            "reconstructed_relation_1_run_is_mixed": mixed,
            "paper_reported_Hits@1": 0.5331,
            "evaluator_corrected_argsort_Hits@1_on_same_mixed_D": tie["policies"]["torch_argsort_deterministic_current_runtime"]["Hits@1"],
            "fully_graph_corrected_reference_Hits@1": None if mixed else tie["policies"]["torch_argsort_deterministic_current_runtime"]["Hits@1"],
            "remaining_unknowns": [
                "author original online.pt/offline.pt checksums",
                "the checkpoint or exact random seed used for Table 2",
                "Table 2 evaluator implementation, because the paper says greedy while reporting Hit@K",
                "training logs proving whether bundled eigs and encoder adjacency came from the same graph",
            ],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True, default=_json_default)
    )
    print(json.dumps(payload["final_classification"], indent=2, default=_json_default))


if __name__ == "__main__":
    main()
