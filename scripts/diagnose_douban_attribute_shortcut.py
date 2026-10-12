#!/usr/bin/env python3
# 文件作用：不训练模型，检查 Douban 原始属性是否足以直接匹配，并核对参考数据与排名同分影响。
# 数据来源：main() 读取 PlanetAlign 的 .pt、参考 .mat 和 Dataset 的锚点划分。
# 结果去向：属性一致性、节点对差异与多种同分策略结果写入 --output-json，供诊断复现偏差。

"""Audit the Douban raw-attribute shortcut without training a model."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_anchor_pairs


KS = (1, 5, 10, 50)
HASH_RULE = (
    "SHA-256 of each C-contiguous little-endian float64 row; no quantization"
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_array(values):
    return np.asarray(values, dtype=np.dtype("<f8"), order="C")


def _array_sha256(values):
    return hashlib.sha256(_canonical_array(values).tobytes()).hexdigest()


def _row_hashes(values):
    rows = _canonical_array(values)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in rows]


def _summary(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if values.numel() == 0:
        return {
            "count": 0, "mean": None, "std": None, "min": None,
            "q25": None, "median": None, "q75": None, "max": None,
        }
    quantiles = torch.quantile(
        values, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "q25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q75": float(quantiles[2]),
        "max": float(values.max()),
    }


def _metrics_from_ranks(ranks):
    ranks = torch.as_tensor(ranks, dtype=torch.float64)
    if ranks.numel() == 0:
        return {**{f"Hits@{k}": None for k in KS}, "MRR": None,
                "mean_rank": None}
    return {
        **{f"Hits@{k}": float((ranks <= k).to(torch.float64).mean())
           for k in KS},
        "MRR": float((1.0 / ranks).mean()),
        "mean_rank": float(ranks.mean()),
    }


# 功能：对指定方向独立计算乐观、平均、悲观排名与确定 argmax 命中，并交叉检查逐行结果。
def _directional_audit(similarity, pairs, atol=1e-8, rtol=1e-7):
    pairs = pairs.long().cpu()
    if pairs.numel() == 0:
        empty = _metrics_from_ranks(torch.empty(0))
        return {
            "optimistic_strict_greater": empty,
            "average_tie_rank": empty,
            "pessimistic_tie_rank": empty,
            "deterministic_argmax_hits1": None,
            "explicit_loop_matches_vectorized": True,
        }
    scores = similarity[pairs[:, 0]].to(torch.float64).cpu()
    row_ids = torch.arange(len(pairs))
    truth = scores[row_ids, pairs[:, 1]]
    greater = scores > truth[:, None]
    equal = torch.isclose(scores, truth[:, None], atol=atol, rtol=rtol)
    greater_count = greater.sum(dim=1).to(torch.float64)
    equal_count = equal.sum(dim=1).to(torch.float64)
    optimistic = greater_count + 1.0
    average = greater_count + (equal_count + 1.0) / 2.0
    pessimistic = greater_count + equal_count
    explicit = []
    for index, target in enumerate(pairs[:, 1].tolist()):
        truth_score = float(scores[index, target])
        explicit.append(
            1 + sum(float(candidate) > truth_score
                    for candidate in scores[index].tolist())
        )
    return {
        "optimistic_strict_greater": _metrics_from_ranks(optimistic),
        "average_tie_rank": _metrics_from_ranks(average),
        "pessimistic_tie_rank": _metrics_from_ranks(pessimistic),
        "deterministic_argmax_hits1": float(
            (scores.argmax(dim=1) == pairs[:, 1]).to(torch.float64).mean()
        ),
        "explicit_loop_matches_vectorized": bool(
            torch.equal(torch.tensor(explicit, dtype=torch.float64), optimistic)
        ),
    }


def _bidirectional_audit(similarity, pairs):
    return {
        "source_to_target": _directional_audit(similarity, pairs),
        "target_to_source": _directional_audit(
            similarity.T, pairs[:, [1, 0]]
        ),
    }


def _positive_scale(source, target, atol=1e-8, rtol=1e-7):
    source = source.to(torch.float64)
    target = target.to(torch.float64)
    source_support = source.abs() > atol
    target_support = target.abs() > atol
    if not torch.equal(source_support, target_support):
        return False
    if not bool(source_support.any()):
        return True
    ratios = source[source_support] / target[target_support]
    scale = ratios[0]
    return bool(
        scale > 0
        and torch.allclose(ratios, torch.full_like(ratios, scale),
                           atol=atol, rtol=rtol)
        and torch.allclose(source, scale * target, atol=atol, rtol=rtol)
    )


def _matrix_metadata(values):
    values = np.asarray(values)
    nonzero = int(np.count_nonzero(values))
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "min": float(values.min()),
        "max": float(values.max()),
        "nonzero": nonzero,
        "sparsity": float(1.0 - nonzero / values.size),
        "canonical_array_sha256": _array_sha256(values),
    }


def _edge_set(edges):
    edges = np.asarray(edges)
    if edges.ndim != 2:
        raise ValueError("Edges must be a two-dimensional array")
    if edges.shape[1] == 2:
        edge_rows = edges
    elif edges.shape[0] == 2:
        edge_rows = edges.T
    else:
        raise ValueError("Edges must have shape [E, 2] or [2, E]")
    return set(map(tuple, edge_rows.tolist()))


# 功能：核对 .pt 与参考 .mat 的节点属性、锚点和边关系，返回差异与文件哈希。
def _reference_comparison(pt_path, mat_path, stored, mat):
    mat_features = [
        mat["online_node_label"].toarray(),
        mat["offline_node_label"].toarray(),
    ]
    feature_comparison = []
    for pt_values, mat_values in zip(stored["node_attributes"], mat_features):
        pt_array = pt_values.cpu().numpy()
        feature_comparison.append({
            "exact_equal": bool(np.array_equal(pt_array, mat_values)),
            "allclose": bool(np.allclose(pt_array, mat_values)),
            "planetalign": _matrix_metadata(pt_array),
            "reference": _matrix_metadata(mat_values),
        })

    pt_anchors = stored["anchor_links"].cpu().numpy().astype(np.int64)
    mat_anchors = np.asarray(mat["ground_truth"], dtype=np.int64) - 1
    pt_anchor_set = set(map(tuple, pt_anchors.tolist()))
    mat_anchor_set = set(map(tuple, mat_anchors.tolist()))
    graph_comparison = []
    for gid, name in enumerate(("online", "offline")):
        pt_edges = stored["edges"][gid].cpu().numpy()
        mat_adjacency = mat[name].tocoo()
        mat_edges = np.stack([mat_adjacency.row, mat_adjacency.col], axis=1)
        relation_results = []
        pt_attributes = stored["edge_attributes"][gid].cpu().numpy()
        for relation_id, relation in enumerate(mat[f"{name}_edge_label"][0]):
            relation = relation.tocoo()
            mat_relation_edges = np.stack(
                [relation.row, relation.col], axis=1
            ) if relation.nnz else np.empty((0, 2), dtype=np.int64)
            pt_relation_edges = pt_edges[:, pt_attributes[:, relation_id] > 0].T
            relation_results.append({
                "relation": relation_id,
                "planetalign_edges": int(len(pt_relation_edges)),
                "reference_edges": int(len(mat_relation_edges)),
                "edge_set_equal": _edge_set(pt_relation_edges)
                == _edge_set(mat_relation_edges),
            })
        graph_comparison.append({
            "graph": name,
            "planetalign_edges": int(pt_edges.shape[1]),
            "reference_edges": int(mat_adjacency.nnz),
            "edge_set_equal": _edge_set(pt_edges.T) == _edge_set(mat_edges),
            "relations": relation_results,
        })

    reference_directory = mat_path.parent
    expected_adjacencies = [
        reference_directory / "online.pt",
        reference_directory / "offline.pt",
    ]
    return {
        "files": {
            "planetalign": {
                "path": str(pt_path.resolve()),
                "sha256": _sha256_file(pt_path),
            },
            "reference": {
                "path": str(mat_path.resolve()),
                "sha256": _sha256_file(mat_path),
            },
        },
        "features": feature_comparison,
        "anchors": {
            "planetalign_shape": list(pt_anchors.shape),
            "reference_shape": list(mat_anchors.shape),
            "same_order": bool(np.array_equal(pt_anchors, mat_anchors)),
            "same_set": pt_anchor_set == mat_anchor_set,
            "planetalign_only": len(pt_anchor_set - mat_anchor_set),
            "reference_only": len(mat_anchor_set - pt_anchor_set),
        },
        "graphs": graph_comparison,
        "normalization": {
            "planetalign": "none; Dataset only casts stored values to dtype",
            "reference": "none; loader densifies and casts node labels to float32",
        },
        "node_reordering_evidence": (
            "No node reordering detected: feature rows and edge endpoint sets are "
            "identical. Only anchor row order differs."
        ),
        "reference_missing_files": [
            str(path.resolve()) for path in expected_adjacencies
            if not path.exists()
        ],
        "reference_loader_note": (
            "load_douban reads relation adjacencies from douban.mat, but "
            "import_data ignores those a1/a2 values and calls load_adj for "
            "online.pt and offline.pt; those files are absent in this checkout."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--reference-mat", type=Path,
        default=Path(".reference/GADL/data/douban.mat"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json", type=Path,
        default=Path(
            "logs/gadl_one_to_one/douban_attribute_shortcut_diagnosis.json"
        ),
    )
    return parser.parse_args()


# 功能：构造原属性相似度，统计重复行、对应属性与排名并列，再保存完整审计记录。
def main():
    args = parse_args()
    pt_path = args.root / "douban.pt"
    stored = torch.load(pt_path, weights_only=True)
    mat = loadmat(args.reference_mat)
    dataset = Dataset(
        root=args.root, name="douban", train_ratio=args.train_ratio,
        seed=args.seed, dtype=torch.float64,
    )
    source_graph, target_graph = dataset.pyg_graphs
    source = source_graph.x.cpu()
    target = target_graph.x.cpu()
    all_pairs = dataset._anchor_links[:, [0, 1]].long().cpu()
    train_pairs = get_anchor_pairs(dataset.train_data, 0, 1).long().cpu()
    test_pairs = get_anchor_pairs(dataset.test_data, 0, 1).long().cpu()
    similarity = F.normalize(source, dim=1) @ F.normalize(target, dim=1).T

    source_hashes = _row_hashes(source.numpy())
    target_hashes = _row_hashes(target.numpy())
    source_counts = Counter(source_hashes)
    target_counts = Counter(target_hashes)
    source_hash_set = set(source_hashes)
    target_hash_set = set(target_hashes)

    source_anchor_set = set(all_pairs[:, 0].tolist())
    target_anchor_set = set(all_pairs[:, 1].tolist())
    source_non_anchor = set(range(source_graph.num_nodes)) - source_anchor_set
    target_non_anchor = set(range(target_graph.num_nodes)) - target_anchor_set

    source_query = similarity[all_pairs[:, 0]]
    source_truth = source_query[
        torch.arange(len(all_pairs)), all_pairs[:, 1]
    ]
    source_max = source_query.max(dim=1).values
    source_max_counts = torch.isclose(
        source_query, source_max[:, None], atol=1e-8, rtol=1e-7
    ).sum(dim=1)
    target_query = similarity.T[all_pairs[:, 1]]
    target_truth = target_query[
        torch.arange(len(all_pairs)), all_pairs[:, 0]
    ]
    target_max = target_query.max(dim=1).values
    target_max_counts = torch.isclose(
        target_query, target_max[:, None], atol=1e-8, rtol=1e-7
    ).sum(dim=1)

    records = []
    l1_values, l2_values, max_values = [], [], []
    cosine_values, norm_ratios = [], []
    exact_values, allclose_values, scale_values, same_id_values = [], [], [], []
    for index, (source_id, target_id) in enumerate(all_pairs.tolist()):
        source_vector = source[source_id]
        target_vector = target[target_id]
        difference = source_vector - target_vector
        source_norm = torch.linalg.vector_norm(source_vector)
        target_norm = torch.linalg.vector_norm(target_vector)
        exact = torch.equal(source_vector, target_vector)
        allclose = torch.allclose(source_vector, target_vector)
        l1 = float(difference.abs().sum())
        l2 = float(torch.linalg.vector_norm(difference))
        maximum = float(difference.abs().max())
        cosine = float(similarity[source_id, target_id])
        norm_ratio = float(source_norm / target_norm) if target_norm else None
        positive_scale = _positive_scale(source_vector, target_vector)
        source_exact_count = target_counts[source_hashes[source_id]]
        target_exact_count = source_counts[target_hashes[target_id]]
        records.append({
            "anchor_index": index,
            "source_id": source_id,
            "target_id": target_id,
            "source_id_equals_target_id": source_id == target_id,
            "exact_equal": exact,
            "allclose": allclose,
            "l1_distance": l1,
            "l2_distance": l2,
            "maximum_absolute_difference": maximum,
            "cosine_similarity": cosine,
            "source_to_target_norm_ratio": norm_ratio,
            "positive_proportional_scaling": positive_scale,
            "source_anchor_exact_matches_in_target": source_exact_count,
            "target_anchor_exact_matches_in_source": target_exact_count,
            "source_to_target_max_cosine_candidates": int(
                source_max_counts[index]
            ),
            "target_to_source_max_cosine_candidates": int(
                target_max_counts[index]
            ),
            "source_truth_is_unique_top1": bool(
                torch.isclose(source_truth[index], source_max[index])
                and source_max_counts[index] == 1
            ),
            "target_truth_is_unique_top1": bool(
                torch.isclose(target_truth[index], target_max[index])
                and target_max_counts[index] == 1
            ),
        })
        l1_values.append(l1)
        l2_values.append(l2)
        max_values.append(maximum)
        cosine_values.append(cosine)
        if norm_ratio is not None:
            norm_ratios.append(norm_ratio)
        exact_values.append(exact)
        allclose_values.append(allclose)
        scale_values.append(positive_scale)
        same_id_values.append(source_id == target_id)

    generator = torch.Generator().manual_seed(args.seed + 20260808)
    shuffled_pairs = all_pairs.clone()
    shuffled_pairs[:, 1] = shuffled_pairs[
        torch.randperm(len(shuffled_pairs), generator=generator), 1
    ]
    target_permutation = torch.randperm(
        target_graph.num_nodes, generator=generator
    )
    permuted_target = torch.empty_like(target)
    permuted_target[target_permutation] = target
    permuted_edge_index = target_permutation[target_graph.edge_index]
    synchronized_pairs = all_pairs.clone()
    synchronized_pairs[:, 1] = target_permutation[synchronized_pairs[:, 1]]
    synchronized_similarity = (
        F.normalize(source, dim=1) @ F.normalize(permuted_target, dim=1).T
    )

    split_metrics = {
        "all_anchors": _bidirectional_audit(similarity, all_pairs),
        "train_data": _bidirectional_audit(similarity, train_pairs),
        "test_data": _bidirectional_audit(similarity, test_pairs),
    }
    sanity = {
        "explicit_anchor_ranks": split_metrics["all_anchors"],
        "randomly_shuffled_target_ids": _bidirectional_audit(
            similarity, shuffled_pairs
        ),
        "synchronized_target_node_permutation": {
            "metrics": _bidirectional_audit(
                synchronized_similarity, synchronized_pairs
            ),
            "target_x_permuted": True,
            "target_edge_index_permuted": True,
            "anchor_target_ids_permuted": True,
            "edge_count_preserved": bool(
                permuted_edge_index.shape == target_graph.edge_index.shape
            ),
        },
        "target_x_only_permutation_without_anchor_update": (
            _bidirectional_audit(synchronized_similarity, all_pairs)
        ),
        "tie_rule": {
            "planetalign_attribution_script": (
                "rank = count(score > truth_score) + 1; all maximum ties "
                "receive optimistic rank 1"
            ),
            "reference_gadl": (
                "rank = count(distance < truth_distance - 1e-8); all minimum "
                "distance ties receive optimistic rank 1"
            ),
            "additional_reported_rules": [
                "average rank within tie block",
                "pessimistic last rank within tie block",
                "torch.argmax deterministic first-index winner",
            ],
        },
        "split_metrics": split_metrics,
    }

    cross_hashes = source_hash_set & target_hash_set
    cross_exact_pairs = sum(
        source_counts[row_hash] * target_counts[row_hash]
        for row_hash in cross_hashes
    )
    source_non_anchor_with_cross_match = sum(
        source_hashes[node] in target_hash_set for node in source_non_anchor
    )
    target_non_anchor_with_cross_match = sum(
        target_hashes[node] in source_hash_set for node in target_non_anchor
    )

    result = {
        "audit": "PlanetAlign Douban raw-attribute shortcut diagnosis",
        "configuration": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "hash_rule": HASH_RULE,
            "cosine_tie_tolerance": {"atol": 1e-8, "rtol": 1e-7},
        },
        "data_flow": {
            "planetalign": [
                "Google Drive prebuilt data/douban.pt",
                "Dataset.load_dataset reads node_attributes without reordering",
                "Dataset casts graph.x to requested dtype; no normalization",
                "anchor_links loaded independently from the same .pt file",
                "train_test_split permutes anchor rows only",
                "get_anchor_pairs selects columns; it does not alter node IDs",
            ],
            "reference": [
                ".reference/GADL/data/douban.mat",
                "load_douban reads online/offline_node_label",
                "sparse labels are densified and cast to float32",
                "ground_truth is transposed and converted from 1-based to 0-based",
                "extract_test_data uses explicit source IDs and target labels",
            ],
            "conversion_provenance": (
                "No douban.mat-to-douban.pt conversion script was found in "
                "PlanetAlign; the repository distributes a prebuilt .pt file."
            ),
        },
        "graphs": {
            "source": _matrix_metadata(source.numpy()),
            "target": _matrix_metadata(target.numpy()),
            "source_nodes": source_graph.num_nodes,
            "target_nodes": target_graph.num_nodes,
            "source_edges": source_graph.num_edges,
            "target_edges": target_graph.num_edges,
            "anchors": len(all_pairs),
            "train_anchors": len(train_pairs),
            "test_anchors": len(test_pairs),
        },
        "anchor_attribute_comparison": {
            "records": records,
            "exact_equal_ratio": sum(exact_values) / len(exact_values),
            "allclose_ratio": sum(allclose_values) / len(allclose_values),
            "cosine_equal_one_ratio": float(
                torch.isclose(torch.tensor(cosine_values), torch.ones(len(cosine_values))).to(torch.float64).mean()
            ),
            "source_id_equals_target_id_ratio": sum(same_id_values) / len(same_id_values),
            "positive_proportional_scaling_ratio": sum(scale_values) / len(scale_values),
            "l1_distance": _summary(l1_values),
            "l2_distance": _summary(l2_values),
            "maximum_absolute_difference": _summary(max_values),
            "cosine_similarity": _summary(cosine_values),
            "norm_ratio": _summary(norm_ratios),
        },
        "attribute_uniqueness": {
            "hash_rule": HASH_RULE,
            "source_unique_rows": len(source_counts),
            "target_unique_rows": len(target_counts),
            "source_duplicate_excess_rows": source_graph.num_nodes - len(source_counts),
            "target_duplicate_excess_rows": target_graph.num_nodes - len(target_counts),
            "source_rows_in_duplicate_buckets": sum(
                count for count in source_counts.values() if count > 1
            ),
            "target_rows_in_duplicate_buckets": sum(
                count for count in target_counts.values() if count > 1
            ),
            "cross_graph_exact_hash_intersection": len(cross_hashes),
            "cross_graph_exact_pair_count": cross_exact_pairs,
            "source_anchor_exact_matches_in_target": _summary([
                record["source_anchor_exact_matches_in_target"]
                for record in records
            ]),
            "target_anchor_exact_matches_in_source": _summary([
                record["target_anchor_exact_matches_in_source"]
                for record in records
            ]),
            "source_to_target_max_cosine_candidate_count": _summary(
                source_max_counts
            ),
            "target_to_source_max_cosine_candidate_count": _summary(
                target_max_counts
            ),
            "source_to_target_max_tie_ratio": float(
                (source_max_counts > 1).to(torch.float64).mean()
            ),
            "target_to_source_max_tie_ratio": float(
                (target_max_counts > 1).to(torch.float64).mean()
            ),
            "source_truth_unique_top1_ratio": sum(
                record["source_truth_is_unique_top1"] for record in records
            ) / len(records),
            "target_truth_unique_top1_ratio": sum(
                record["target_truth_is_unique_top1"] for record in records
            ) / len(records),
            "non_anchor_cross_graph_exact_matches": {
                "source_non_anchor_count": len(source_non_anchor),
                "source_non_anchor_with_target_exact_match": (
                    source_non_anchor_with_cross_match
                ),
                "target_non_anchor_count": len(target_non_anchor),
                "target_non_anchor_with_source_exact_match": (
                    target_non_anchor_with_cross_match
                ),
                "cross_exact_pairs_excluding_true_anchors": (
                    cross_exact_pairs - len(all_pairs)
                ),
            },
        },
        "evaluation_sanity": sanity,
        "planetalign_vs_reference": _reference_comparison(
            pt_path, args.reference_mat, stored, mat
        ),
        "classification": {
            "primary": "evaluation implementation issue: optimistic tie ranking",
            "secondary": (
                "expected dataset attribute shortcut: shared one-hot location "
                "features create large exact-match buckets"
            ),
            "not_supported": [
                "PlanetAlign node renumbering error",
                "PlanetAlign feature construction using anchors at load time",
                "different PlanetAlign and reference feature matrices",
            ],
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({
        "output": str(args.output_json),
        "anchors": len(all_pairs),
        "exact_equal_ratio": result["anchor_attribute_comparison"]["exact_equal_ratio"],
        "source_unique_top1_ratio": result["attribute_uniqueness"]["source_truth_unique_top1_ratio"],
        "target_unique_top1_ratio": result["attribute_uniqueness"]["target_truth_unique_top1_ratio"],
        "source_max_tie_ratio": result["attribute_uniqueness"]["source_to_target_max_tie_ratio"],
        "target_max_tie_ratio": result["attribute_uniqueness"]["target_to_source_max_tie_ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
