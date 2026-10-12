# 文件作用：追踪 QuotientDecode 的候选筛选与合并阶段，定位真实同组节点在哪里丢失或误合并。
# 核心依赖：PlanetAlign/m2m/base.py 生成固定 S，PlanetAlign/m2m/quotient.py 的候选、选择和轮廓汇总函数。
# 核心处理：先不接触真实实体重放分组，再用真实实体离线统计各阶段精确率、召回率与连通率。
# 结果去向：main() 保存分阶段 JSON/CSV；scripts/evaluate_gadl_m2m_backbone.py 复用这些诊断函数。
# 阅读建议：main() → build_local_context()/replay_current_final_partitions() → diagnose_side()。

"""Diagnose local-candidate filtering in the current QuotientDecode pipeline.

The diagnostic has four stages on each graph side:

    raw one-hop candidates
    -> candidates passing the current Otsu profile threshold
    -> candidates passing the running-centroid gate and triggering union
    -> the current full QuotientDecode partition after alternating refinement,
       evidence selection, and outlier handling

Candidate construction, thresholding, union, and final partition prediction are
finished before the ground-truth file is opened.  Ground truth is used only for
offline labels, score-distribution analysis, and metrics.

The script trains its base afresh by default.  ``--reuse-owned-cache`` may reuse
only this script's cache after checking the resolved root, dataset hash, git
revision, base, seed, and full-anchor protocol.  It never reads the repository's
legacy shared S_cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

from PlanetAlign.data import Dataset
from PlanetAlign.m2m import use_full_anchor_supervision
from PlanetAlign.m2m.base import arbitrate_sharpen, base_configs, train_base_S
from PlanetAlign.m2m.quotient import (
    _adjacent_pairs,
    _candidate_partitions,
    _partition_signature,
    _select_partitions,
    evict_outliers,
    otsu_threshold,
    pool_columns,
)
from PlanetAlign.utils import get_anchor_pairs


Pair = Tuple[int, int]
GIDS = (0, 1)


def canonical_pair(u: int, v: int) -> Pair:
    return (u, v) if u < v else (v, u)


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().to(torch.float32).cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def unique_pairs(u: torch.Tensor, v: torch.Tensor) -> List[Pair]:
    return sorted({canonical_pair(int(a), int(b)) for a, b in zip(u.tolist(), v.tolist())})


def normalized_pair_cosines(
    normalized_profiles: torch.Tensor,
    pairs: Sequence[Pair],
    chunk_size: int = 20_000,
) -> torch.Tensor:
    if not pairs:
        return torch.zeros(0, dtype=torch.float32)
    chunks: List[torch.Tensor] = []
    for start in range(0, len(pairs), chunk_size):
        block = pairs[start : start + chunk_size]
        u = torch.tensor([pair[0] for pair in block], dtype=torch.long)
        v = torch.tensor([pair[1] for pair in block], dtype=torch.long)
        chunks.append((normalized_profiles[u] * normalized_profiles[v]).sum(dim=1))
    return torch.cat(chunks)


def graph_components(num_nodes: int, pairs: Iterable[Pair]) -> List[List[int]]:
    parent = list(range(num_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in pairs:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[rv] = ru
    groups: Dict[int, List[int]] = defaultdict(list)
    for node in range(num_nodes):
        groups[find(node)].append(node)
    return list(groups.values())


# 功能：重放组中心门控合并，分别保留过阈值、被接纳、被拒绝与重复连接的候选。
def trace_average_linkage(
    num_nodes: int,
    profiles: torch.Tensor,
    pairs: Sequence[Pair],
    similarities: torch.Tensor,
    tau: float,
) -> Dict[str, object]:
    """Replay merge_average_linkage and retain the exact decision stages."""
    if len(pairs) != int(similarities.numel()):
        raise ValueError("pairs and similarities must have the same length")
    p = F.normalize(profiles.detach().to(torch.float32).cpu(), p=2, dim=1)
    parent = list(range(num_nodes))
    centroid_sum = p.clone()

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    threshold_pairs: List[Pair] = []
    accepted_pairs: List[Pair] = []
    centroid_rejected: List[Pair] = []
    redundant: List[Pair] = []
    order = torch.argsort(-similarities)
    for idx in order.tolist():
        if float(similarities[idx]) < tau:
            break
        u, v = pairs[idx]
        threshold_pairs.append((u, v))
        ru, rv = find(u), find(v)
        if ru == rv:
            redundant.append((u, v))
            continue
        a, b = centroid_sum[ru], centroid_sum[rv]
        denom = float(a.norm()) * float(b.norm())
        centroid_cosine = float(a @ b) / denom if denom > 0 else 0.0
        if centroid_cosine < tau:
            centroid_rejected.append((u, v))
            continue
        parent[rv] = ru
        centroid_sum[ru] = a + b
        accepted_pairs.append((u, v))

    groups: Dict[int, List[int]] = defaultdict(list)
    for node in range(num_nodes):
        groups[find(node)].append(node)
    return {
        "threshold_pairs": sorted(set(threshold_pairs)),
        "accepted_pairs": sorted(set(accepted_pairs)),
        "centroid_rejected_pairs": sorted(set(centroid_rejected)),
        "redundant_pairs": sorted(set(redundant)),
        "union_groups": list(groups.values()),
    }


# 功能：复用 PlanetAlign/m2m/quotient.py 的候选与选择逻辑重放最终分组，不接收真实实体。
def replay_current_final_partitions(
    S: torch.Tensor,
    graph_src,
    graph_tgt,
    anchors: torch.Tensor,
    use_attr: bool,
    max_iters: int = 2,
) -> Tuple[List[List[int]], List[List[int]], Dict[str, object]]:
    """Replay the current default QuotientDecode grouping path without GT."""
    S = S.detach().to(torch.float32).cpu()
    n1, n2 = S.shape
    global_candidate = anchors is not None and anchors.shape[0] >= 20
    us, vs = _adjacent_pairs(graph_src)
    ut, vt = _adjacent_pairs(graph_tgt)
    profiles_src = S
    profiles_tgt = S.T.contiguous()
    src_groups: List[List[int]] = [[node] for node in range(n1)]
    tgt_groups: List[List[int]] = [[node] for node in range(n2)]
    allowed_s: Optional[str] = None
    allowed_t: Optional[str] = None
    previous_signature: Optional[Tuple] = None
    info: Dict[str, object] = {"taus": [], "evidence": [], "group_counts": [], "iters": 0}

    for iteration in range(max_iters):
        cands_s, tau_s = _candidate_partitions(
            profiles_src,
            graph_src,
            us,
            vs,
            None,
            True,
            True,
            use_attr,
            True,
            global_candidate,
            allowed=allowed_s,
        )
        cands_t, tau_t = _candidate_partitions(
            profiles_tgt,
            graph_tgt,
            ut,
            vt,
            None,
            True,
            True,
            use_attr,
            True,
            global_candidate,
            allowed=allowed_t,
        )
        src_groups, tgt_groups, evidence_s, evidence_t, _ = _select_partitions(
            S,
            cands_s,
            cands_t,
            anchors,
            "hungarian",
            0.0,
            profiles_src,
            profiles_tgt,
            us,
            vs,
            ut,
            vt,
        )
        allowed_s = evidence_s.split("@")[0]
        allowed_t = evidence_t.split("@")[0]
        if allowed_s == "profile":
            src_groups = evict_outliers(src_groups, profiles_src, tau_s)
        if allowed_t == "profile":
            tgt_groups = evict_outliers(tgt_groups, profiles_tgt, tau_t)

        # Match quotient_decode's public diagnostic payload exactly.
        info["taus"].append((round(tau_s, 4), round(tau_t, 4)))
        info["evidence"].append((evidence_s, evidence_t))
        info["group_counts"].append((len(src_groups), len(tgt_groups)))
        info["iters"] = iteration + 1
        signature = (_partition_signature(src_groups), _partition_signature(tgt_groups))
        if signature == previous_signature:
            break
        previous_signature = signature
        profiles_src = pool_columns(S, tgt_groups)
        profiles_tgt = pool_columns(S.T.contiguous(), src_groups)

    return src_groups, tgt_groups, info


# 功能：验证缓存来源或调用 train_base_S()/arbitrate_sharpen() 生成固定矩阵，记录参数与哈希。
def load_or_train_fixed_s(
    args: argparse.Namespace,
    dataset: Dataset,
    graph_src,
    graph_tgt,
    anchors: torch.Tensor,
    data_sha256: str,
    root_fingerprint: str,
) -> Tuple[torch.Tensor, Dict[str, object], Path]:
    head = git_head()
    safe_base = "".join(char if char.isalnum() or char in "-_" else "_" for char in args.base)
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"{args.dataset}__root-{root_fingerprint[:12]}__data-{data_sha256[:12]}"
        f"__git-{head[:12]}__base-{safe_base}__seed-{args.seed}__full-anchors.pt"
    )
    expected = {
        "resolved_root": str(args.root.resolve()),
        "dataset": args.dataset,
        "data_sha256": data_sha256,
        "git_head": head,
        "seed": args.seed,
        "base": args.base,
        "full_anchor_supervision": True,
    }
    if args.reuse_owned_cache and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        mismatch = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatch:
            raise ValueError(f"diagnostic cache provenance mismatch: {mismatch}")
        chosen = payload["S"].detach().to(torch.float32).cpu()
        provenance = dict(metadata)
        provenance["run_load_mode"] = "explicitly_reused_provenance_checked_cache"
        provenance["cache_path"] = str(cache_path.resolve())
        return chosen, provenance, cache_path

    started = time.perf_counter()
    raw = train_base_S(args.base, dataset, gids=GIDS, seed=args.seed)
    train_seconds = time.perf_counter() - started
    has_attr = all(graph.x is not None for graph in (graph_src, graph_tgt))
    chosen, temperature, table = arbitrate_sharpen(
        raw, graph_src, graph_tgt, anchors, use_attr=has_attr
    )
    provenance: Dict[str, object] = {
        **expected,
        "source": "fresh_train_base_S_then_anchor_arbitrate_sharpen",
        "train_seconds": train_seconds,
        "anchor_count": int(anchors.shape[0]),
        "use_attr": has_attr,
        "sharpen_temperature": temperature,
        "sharpen_arbitration": [
            {"temperature": temp, "corrected": corrected, "raw": raw_agreement}
            for temp, corrected, raw_agreement in table
        ],
        "raw_S_sha256": sha256_tensor(raw),
        "S_sha256": sha256_tensor(chosen),
        "cache_path": str(cache_path.resolve()),
    }
    torch.save({"S": chosen, "S_raw": raw, "metadata": provenance}, cache_path)
    return chosen, provenance, cache_path


# 功能：从一侧图和轮廓建立邻接候选、Otsu 阈值和合并追踪结果，供后续离线分析。
def build_local_context(graph, profiles: torch.Tensor) -> Dict[str, object]:
    """Complete all local candidate stages without access to GT."""
    normalized = F.normalize(profiles.detach().to(torch.float32).cpu(), p=2, dim=1)
    local_u, local_v = _adjacent_pairs(graph)
    pairs = unique_pairs(local_u, local_v)
    similarities = normalized_pair_cosines(normalized, pairs)
    tau = otsu_threshold(similarities)
    trace = trace_average_linkage(
        int(graph.num_nodes), normalized, pairs, similarities, tau
    )
    return {
        "num_nodes": int(graph.num_nodes),
        "pairs": pairs,
        "similarities": similarities,
        "tau": tau,
        "threshold_pairs": trace["threshold_pairs"],
        "accepted_pairs": trace["accepted_pairs"],
        "union_groups": trace["union_groups"],
        "centroid_rejected_count": len(trace["centroid_rejected_pairs"]),
        "redundant_count": len(trace["redundant_pairs"]),
    }


# 功能：从真实实体提取指定侧分组并检查重叠；仅用于诊断标签。
def gt_side_groups(
    entities: Mapping[str, Mapping[str, Sequence[int]]], side: str
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    seen: Set[int] = set()
    for eid, entity in entities.items():
        members = [int(node) for node in entity[side]]
        overlap = seen.intersection(members)
        if overlap:
            raise ValueError(f"{side} GT overlap at nodes {sorted(overlap)[:5]}")
        seen.update(members)
        groups[str(eid)] = members
    return groups


# 输出结果：真实同组节点对、节点组标签和多成员组，供阶段指标计算。
def true_pairs_and_labels(
    num_nodes: int,
    groups: Mapping[str, Sequence[int]],
) -> Tuple[Set[Pair], List[str], List[List[int]]]:
    # The builder splits only evaluation anchors. All other same-side nodes are
    # singleton entities for this grouping diagnostic.
    labels = [f"__singleton_{node}" for node in range(num_nodes)]
    true_pairs: Set[Pair] = set()
    multi_groups: List[List[int]] = []
    for eid, raw_members in groups.items():
        members = sorted(map(int, raw_members))
        for node in members:
            labels[node] = eid
        for pos, u in enumerate(members):
            true_pairs.update((u, v) for v in members[pos + 1 :])
        if len(members) > 1:
            multi_groups.append(members)
    return true_pairs, labels, multi_groups


# 功能：只保留真实组内的候选连接，检查每个多成员组能否内部连通。
def strict_group_connectivity_rate(
    multi_groups: Sequence[Sequence[int]], pairs: Set[Pair]
) -> Tuple[int, int, float]:
    """Connectivity within each true-entity-induced candidate subgraph."""
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    for u, v in pairs:
        adjacency[u].add(v)
        adjacency[v].add(u)
    connected = 0
    for raw_members in multi_groups:
        members = list(map(int, raw_members))
        allowed = set(members)
        reached = {members[0]}
        stack = [members[0]]
        while stack:
            node = stack.pop()
            for nxt in adjacency[node]:
                if nxt in allowed and nxt not in reached:
                    reached.add(nxt)
                    stack.append(nxt)
        connected += int(len(reached) == len(members))
    total = len(multi_groups)
    return connected, total, connected / total if total else float("nan")


# 功能：比较阶段候选对与真实同组对，统计精确率、召回率、组连通率与最大连通块。
def edge_stage_metrics(
    num_nodes: int,
    pairs: Set[Pair],
    true_pairs: Set[Pair],
    labels: Sequence[str],
    multi_groups: Sequence[Sequence[int]],
) -> Dict[str, object]:
    correct = len(pairs & true_pairs)
    errors = len(pairs) - correct
    connected, group_count, connectivity = strict_group_connectivity_rate(multi_groups, pairs)
    components = graph_components(num_nodes, pairs)
    largest = max(components, key=len) if components else []
    max_entities = max(
        (len({labels[node] for node in component}) for component in components),
        default=0,
    )
    return {
        "edge_count": len(pairs),
        "correct_edges": correct,
        "cross_entity_error_edges": errors,
        "precision": correct / len(pairs) if pairs else float("nan"),
        "recall": correct / len(true_pairs) if true_pairs else float("nan"),
        "true_group_connectivity_rate": connectivity,
        "connected_true_groups": connected,
        "multi_node_true_groups": group_count,
        "max_component_nodes": len(largest),
        "max_component_entities": max_entities,
    }


# 功能：对最终分组统计内部正确/错误节点对和真实组覆盖，区别于单条候选边统计。
def partition_stage_metrics(
    num_nodes: int,
    partition: Sequence[Sequence[int]],
    true_pairs: Set[Pair],
    labels: Sequence[str],
    multi_groups: Sequence[Sequence[int]],
) -> Dict[str, object]:
    predicted_group = [-1] * num_nodes
    predicted_pairs = 0
    max_nodes = 0
    max_entities = 0
    for gid, raw_members in enumerate(partition):
        members = list(map(int, raw_members))
        predicted_pairs += len(members) * (len(members) - 1) // 2
        max_nodes = max(max_nodes, len(members))
        max_entities = max(max_entities, len({labels[node] for node in members}))
        for node in members:
            predicted_group[node] = gid
    if any(gid < 0 for gid in predicted_group):
        raise ValueError("final partition does not cover every graph node")

    correct = sum(predicted_group[u] == predicted_group[v] for u, v in true_pairs)
    connected = sum(
        len({predicted_group[int(node)] for node in members}) == 1
        for members in multi_groups
    )
    group_count = len(multi_groups)
    errors = predicted_pairs - correct
    return {
        "group_count": len(partition),
        "edge_count": predicted_pairs,
        "correct_edges": correct,
        "cross_entity_error_edges": errors,
        "precision": correct / predicted_pairs if predicted_pairs else float("nan"),
        "recall": correct / len(true_pairs) if true_pairs else float("nan"),
        "true_group_connectivity_rate": connected / group_count if group_count else float("nan"),
        "connected_true_groups": connected,
        "multi_node_true_groups": group_count,
        "max_component_nodes": max_nodes,
        "max_component_entities": max_entities,
    }


def distribution_summary(values: torch.Tensor) -> Dict[str, object]:
    values = values.detach().to(torch.float32).cpu().flatten()
    if not values.numel():
        return {"count": 0}
    quantile_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    quantiles = torch.quantile(values, torch.tensor(quantile_levels))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()) if values.numel() > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": {
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, quantiles.tolist())
        },
    }


# 输入来源：score_analysis() 传入候选对的轮廓分数，以及预测完成后由真实组得到的同组标签。
# 核心思路：按分数降序累计命中与误合并，同分候选一起纳入，避免曲线受同分排序顺序影响。
# 输出结果：每个阈值的精确率、召回率与计数行，供离线 CSV 分析；不用于选择预测阈值。
def precision_recall_rows(
    similarities: torch.Tensor, is_true: torch.Tensor
) -> List[Dict[str, object]]:
    similarities = similarities.detach().to(torch.float32).cpu()
    is_true = is_true.detach().to(torch.bool).cpu()
    positives = int(is_true.sum())
    negatives = int((~is_true).sum())
    order = torch.argsort(similarities, descending=True)
    rows: List[Dict[str, object]] = []
    tp = fp = 0
    position = 0
    while position < len(order):
        score = float(similarities[order[position]])
        end = position
        while end < len(order) and float(similarities[order[end]]) == score:
            if bool(is_true[order[end]]):
                tp += 1
            else:
                fp += 1
            end += 1
        fn = positives - tp
        tn = negatives - fp
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / positives if positives else float("nan")
        rows.append(
            {
                "threshold": score,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "tpr": recall,
                "fpr": fp / negatives if negatives else float("nan"),
            }
        )
        position = end
    return rows


# 功能：对真实同组与跨组候选比较分数分布，形成阈值与精确率/召回率表。
def score_analysis(
    pairs: Sequence[Pair],
    similarities: torch.Tensor,
    true_pairs: Set[Pair],
    tau: float,
) -> Tuple[Dict[str, object], torch.Tensor, List[Dict[str, object]]]:
    is_true = torch.tensor([pair in true_pairs for pair in pairs], dtype=torch.bool)
    true_scores = similarities[is_true]
    error_scores = similarities[~is_true]
    tp = int((true_scores >= tau).sum())
    fp = int((error_scores >= tau).sum())
    pr_rows = precision_recall_rows(similarities, is_true)
    recall_80 = [row for row in pr_rows if float(row["recall"]) >= 0.80]
    fpr_20 = [row for row in pr_rows if float(row["fpr"]) <= 0.20]
    best_recall_80 = min(recall_80, key=lambda row: (float(row["fpr"]), -float(row["precision"]))) if recall_80 else None
    best_fpr_20 = max(fpr_20, key=lambda row: (float(row["recall"]), float(row["precision"]))) if fpr_20 else None
    # Compute exact AP robustly under tied score blocks: each newly recovered
    # positive receives the precision at the block's common threshold.
    previous_tp = 0
    ap_sum = 0.0
    for row in pr_rows:
        new_tp = int(row["tp"]) - previous_tp
        ap_sum += new_tp * float(row["precision"])
        previous_tp = int(row["tp"])
    average_precision = ap_sum / len(true_scores) if len(true_scores) else float("nan")
    analysis = {
        "otsu_threshold": tau,
        "true_same_entity_scores": distribution_summary(true_scores),
        "cross_entity_scores": distribution_summary(error_scores),
        "otsu_operating_point": {
            "tp": tp,
            "fp": fp,
            "tpr": tp / len(true_scores) if len(true_scores) else float("nan"),
            "fpr": fp / len(error_scores) if len(error_scores) else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
        },
        "average_precision": average_precision,
        "oracle_diagnostic": {
            "exists_threshold_tpr_ge_0.80_fpr_le_0.20": any(
                float(row["recall"]) >= 0.80 and float(row["fpr"]) <= 0.20
                for row in pr_rows
            ),
            "exists_threshold_tpr_ge_0.90_fpr_le_0.10": any(
                float(row["recall"]) >= 0.90 and float(row["fpr"]) <= 0.10
                for row in pr_rows
            ),
            "lowest_fpr_at_tpr_ge_0.80": best_recall_80,
            "highest_tpr_at_fpr_le_0.20": best_fpr_20,
        },
    }
    return analysis, is_true, pr_rows


# 输入来源：main() 或 scripts/evaluate_gadl_m2m_backbone.py 传入已完成的候选上下文、分组及真实标签。
# 输出结果：阶段摘要与明细行，交给调用者写入 JSON/CSV。
def diagnose_side(
    side: str,
    context: Mapping[str, object],
    final_partition: Sequence[Sequence[int]],
    gt_groups: Mapping[str, Sequence[int]],
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    num_nodes = int(context["num_nodes"])
    pairs: Sequence[Pair] = context["pairs"]
    similarities: torch.Tensor = context["similarities"]
    true_pairs, labels, multi_groups = true_pairs_and_labels(num_nodes, gt_groups)
    raw_pairs = set(pairs)
    threshold_pairs = set(context["threshold_pairs"])
    accepted_pairs = set(context["accepted_pairs"])
    stages = {
        "raw_candidates": edge_stage_metrics(
            num_nodes, raw_pairs, true_pairs, labels, multi_groups
        ),
        "profile_threshold": edge_stage_metrics(
            num_nodes, threshold_pairs, true_pairs, labels, multi_groups
        ),
        "centroid_union": edge_stage_metrics(
            num_nodes, accepted_pairs, true_pairs, labels, multi_groups
        ),
        "final_partition": partition_stage_metrics(
            num_nodes, final_partition, true_pairs, labels, multi_groups
        ),
    }
    analysis, is_true, pr_rows = score_analysis(
        pairs, similarities, true_pairs, float(context["tau"])
    )
    score_rows = [
        {
            "side": side,
            "u": u,
            "v": v,
            "is_true_same_entity": int(bool(label)),
            "profile_cosine": float(score),
            "passes_otsu": int(float(score) >= float(context["tau"])),
        }
        for (u, v), score, label in zip(pairs, similarities.tolist(), is_true.tolist())
    ]
    for row in pr_rows:
        row["side"] = side
    result = {
        "side": side,
        "num_nodes": num_nodes,
        "local_profile_tau": float(context["tau"]),
        "true_pair_count": len(true_pairs),
        "multi_node_true_groups": len(multi_groups),
        "centroid_rejected_count": int(context["centroid_rejected_count"]),
        "redundant_after_union_count": int(context["redundant_count"]),
        "stages": stages,
        "final_partition_connectivity": stages["final_partition"]["true_group_connectivity_rate"],
        "profile_score_analysis": analysis,
    }
    return result, score_rows, pr_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def print_side_summary(result: Mapping[str, object]) -> None:
    stages = result["stages"]
    print(f"\n[{result['side']}] Otsu tau={result['local_profile_tau']:.6f}")
    print(
        f"{'stage':<20}{'edges':>10}{'correct':>10}{'errors':>10}"
        f"{'precision':>12}{'recall':>10}{'connect':>10}{'maxE':>8}"
    )
    for name in ("raw_candidates", "profile_threshold", "centroid_union", "final_partition"):
        metrics = stages[name]
        print(
            f"{name:<20}{metrics['edge_count']:>10d}{metrics['correct_edges']:>10d}"
            f"{metrics['cross_entity_error_edges']:>10d}{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>10.4f}{metrics['true_group_connectivity_rate']:>10.4f}"
            f"{metrics['max_component_entities']:>8d}"
        )
    score = result["profile_score_analysis"]
    print(f"score true={score['true_same_entity_scores']}")
    print(f"score error={score['cross_entity_scores']}")
    print(f"Otsu operating point={score['otsu_operating_point']}")
    print(f"oracle diagnostic={score['oracle_diagnostic']}")


# 核心思路：加载或训练 S，在读取真实实体前完成两侧候选和分组，再调用 diagnose_side() 离线评估。
# 输出结果：记录数据/矩阵来源及分阶段覆盖统计，写 JSON 和 CSV。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/m2m_no_overlap"))
    parser.add_argument("--dataset", default="douban_m2m")
    parser.add_argument("--base", default="PARROT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=None,
        help="JSON output path; score and PR CSV files use the same stem",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("logs/m2m_diag/candidate_threshold_cache"),
    )
    parser.add_argument(
        "--reuse-owned-cache",
        action="store_true",
        help="reuse only this script's provenance-checked independent cache",
    )
    args = parser.parse_args()
    if args.base not in base_configs():
        raise ValueError(f"unknown base {args.base!r}; choices={sorted(base_configs())}")
    if args.output is None:
        args.output = Path("logs/m2m_diag/candidate_threshold") / (
            f"{args.dataset}__base-{args.base}__seed-{args.seed}.json"
        )

    data_path = args.root / f"{args.dataset}.pt"
    gt_path = args.root / f"{args.dataset}_gt_many2many.json"
    if not data_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"missing dataset inputs: {data_path} or {gt_path}")
    data_sha256 = sha256_file(data_path)
    gt_sha256 = sha256_file(gt_path)
    root_fingerprint = hashlib.sha256(str(args.root.resolve()).encode("utf-8")).hexdigest()

    torch.manual_seed(args.seed)
    dataset = Dataset(
        root=args.root,
        name=args.dataset,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    use_full_anchor_supervision(dataset)
    graph_src, graph_tgt = dataset.pyg_graphs[GIDS[0]], dataset.pyg_graphs[GIDS[1]]
    anchors = get_anchor_pairs(dataset.train_data, GIDS[0], GIDS[1])
    S, provenance, cache_path = load_or_train_fixed_s(
        args,
        dataset,
        graph_src,
        graph_tgt,
        anchors,
        data_sha256,
        root_fingerprint,
    )
    if tuple(S.shape) != (int(graph_src.num_nodes), int(graph_tgt.num_nodes)):
        raise ValueError(f"S shape {tuple(S.shape)} does not match graph node counts")

    # Finish every prediction-stage calculation before GT is opened.
    started = time.perf_counter()
    source_context = build_local_context(graph_src, S)
    target_context = build_local_context(graph_tgt, S.T.contiguous())
    has_attr = all(graph.x is not None for graph in (graph_src, graph_tgt))
    final_src, final_tgt, final_info = replay_current_final_partitions(
        S, graph_src, graph_tgt, anchors, has_attr
    )

    with gt_path.open("r", encoding="utf-8") as handle:
        entities = json.load(handle)["entities"]
    source, source_scores, source_pr = diagnose_side(
        "source", source_context, final_src, gt_side_groups(entities, "src")
    )
    target, target_scores, target_pr = diagnose_side(
        "target", target_context, final_tgt, gt_side_groups(entities, "tgt")
    )
    elapsed = time.perf_counter() - started

    scores_csv = args.output.with_name(f"{args.output.stem}__local_scores.csv")
    pr_csv = args.output.with_name(f"{args.output.stem}__pr_curve.csv")
    write_csv(
        scores_csv,
        [*source_scores, *target_scores],
        ["side", "u", "v", "is_true_same_entity", "profile_cosine", "passes_otsu"],
    )
    write_csv(
        pr_csv,
        [*source_pr, *target_pr],
        ["side", "threshold", "tp", "fp", "fn", "tn", "precision", "recall", "tpr", "fpr"],
    )

    report = {
        "protocol": {
            "root": str(args.root.resolve()),
            "dataset": args.dataset,
            "base": args.base,
            "seed": args.seed,
            "use_full_anchor_supervision": True,
            "train_anchor_count": int(anchors.shape[0]),
            "candidate_rule": "current one-hop local candidates only",
            "final_partition": "current default QuotientDecode grouping path",
            "ground_truth_usage": "offline statistics only; opened after all prediction stages",
            "connectivity_definition": "strict true-entity-induced candidate connectivity; final partition requires co-clustering",
            "outside_gt_nodes": "singleton same-side entities",
            "oracle_curve_policy": "diagnostic only; never used to select the prediction threshold",
        },
        "inputs": {
            "data_path": str(data_path.resolve()),
            "data_sha256": data_sha256,
            "gt_path": str(gt_path.resolve()),
            "gt_sha256": gt_sha256,
            "git_head": git_head(),
        },
        "S_provenance": provenance,
        "S_cache_path": str(cache_path.resolve()),
        "artifacts": {
            "json": str(args.output.resolve()),
            "local_scores_csv": str(scores_csv.resolve()),
            "pr_curve_csv": str(pr_csv.resolve()),
        },
        "final_partition_info": final_info,
        "diagnostic_seconds": elapsed,
        "source": source,
        "target": target,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=True)

    print(f"\ndataset={args.dataset} base={args.base} seed={args.seed}")
    print(f"S source={provenance['source']}")
    print(f"S sha256={provenance['S_sha256']}")
    print(f"final partition={final_info}")
    print_side_summary(source)
    print_side_summary(target)
    print(f"\nwrote {args.output}")
    print(f"wrote {scores_csv}")
    print(f"wrote {pr_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
