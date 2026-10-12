#!/usr/bin/env python3
# 文件作用：比较 GADL 论文声明与公开参考的模型/评估口径，额外实现全局贪心一对一评估。
# 核心依赖：PlanetAlign/algorithms/gadl/paper_profile.py → paper_declared_gadl()；scripts/evaluate_gadl_public_reference.py → _run()。
# 核心处理：固定 seed 0，在同一检查点比较多种评估，匹配函数不使用真实对应作为选择依据。
# 结果去向：main() 保存论文/公开配置及评估差异 JSON。

"""Locate the gap between GADL paper and public-reference Douban results.

The script runs seed 0 only and never performs checkpoint selection.  It keeps
the existing public and PlanetAlign evaluators untouched, adding a third,
explicit global greedy one-to-one evaluator around the same checkpoint tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_gadl_public_reference as reference_experiment
from PlanetAlign.algorithms.gadl.paper_profile import (
    PAPER_DECLARED_PROFILE,
    PUBLIC_REFERENCE_DIFFERENCES,
    paper_declared_gadl,
)
from PlanetAlign.algorithms.gadl.profiles import ConfigurableGADL


CHECKPOINTS = (0, 1, 5, 20, 50, 100)
PUBLIC_REFERENCE_MODEL_PROFILE = {
    "hidden_dim": 512,
    "num_hidden_layers": 3,
    "spectral_rank": 300,
    "lr": 1e-4,
    "weight_decay": 5e-4,
    "descriptor_weight": 1e-3,
    "commutativity_weight": 1e-2,
    "functional_map_weight": 1.0,
    "bijectivity_weight": 0.1,
    "orthogonality_weight": 0.1,
    "residual": False,
    "reconstruction_mode": "reference_weighted_bce",
    "functional_map_schedule": "reference_linear_floor_half",
    "activation_mode": "reference_identity",
}


# 核心思路：按整个矩阵分数从高到低选择双方都未使用的节点对，同分按稳定行优先顺序。
# 输出结果：两方向节点映射，生成匹配时不接收真实锚点。
def greedy_one_to_one_matching(similarity: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Globally greedily select highest-scoring unmatched source/target pairs.

    Equal scores use stable row-major order because the paper does not specify
    a tie rule.  Ground-truth anchors are never consulted during matching.
    """

    scores = similarity.detach().double().cpu().numpy()
    n_src, n_tgt = scores.shape
    order = np.argsort(-scores.reshape(-1), kind="stable")
    src_to_tgt = np.full(n_src, -1, dtype=np.int64)
    tgt_to_src = np.full(n_tgt, -1, dtype=np.int64)
    used_src = np.zeros(n_src, dtype=bool)
    used_tgt = np.zeros(n_tgt, dtype=bool)
    required = min(n_src, n_tgt)
    selected = 0
    for flat_index in order:
        src = int(flat_index // n_tgt)
        tgt = int(flat_index % n_tgt)
        if used_src[src] or used_tgt[tgt]:
            continue
        used_src[src] = True
        used_tgt[tgt] = True
        src_to_tgt[src] = tgt
        tgt_to_src[tgt] = src
        selected += 1
        if selected == required:
            break
    return src_to_tgt, tgt_to_src


# 功能：先调用 greedy_one_to_one_matching() 得到硬匹配，再用传入真实节点对计算命中。
def greedy_evaluation(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    pairs: torch.Tensor,
) -> Dict[str, Any]:
    similarity = F.normalize(z_src.detach(), dim=1) @ F.normalize(z_tgt.detach(), dim=1).T
    src_to_tgt, tgt_to_src = greedy_one_to_one_matching(similarity)
    anchor_pairs = pairs.detach().long().cpu().numpy()
    src_ids = anchor_pairs[:, 0]
    tgt_ids = anchor_pairs[:, 1]
    correct = src_to_tgt[src_ids] == tgt_ids
    reverse_correct = tgt_to_src[tgt_ids] == src_ids
    return {
        "definition": "global greedy highest-similarity unmatched pairs",
        "similarity": "cosine_on_l2_normalized_embeddings",
        "tie_break": "stable_row_major_source_then_target; paper_unspecified",
        "anchor_count": int(len(anchor_pairs)),
        "selected_pair_count": int(min(similarity.shape)),
        "source_to_target_hits_at_1": float(correct.mean()),
        "target_to_source_hits_at_1": float(reverse_correct.mean()),
        "correct_anchor_count": int(correct.sum()),
        "unmatched_source_anchor_count": int((src_to_tgt[src_ids] < 0).sum()),
        "note": "Greedy produces one assignment, so paper-style Hits@5/10/50 are undefined.",
    }


# 输出结果：将复现入口传来的参数交给 paper_declared_gadl() 创建模型。
def _paper_model_factory(**kwargs: Any) -> ConfigurableGADL:
    passthrough = {
        "seed": kwargs["seed"],
        "objective_observer": kwargs["objective_observer"],
    }
    return paper_declared_gadl(**passthrough)


# 功能：临时包装公开参考实验入口的模型、图协议和检查点评估，固定完整图并附全局贪心结果。
def _run_full_graph(
    args: argparse.Namespace,
    variant_name: str,
    model_factory: Any,
    full_configuration: Dict[str, Any],
) -> Dict[str, Any]:
    original_apply_protocol = reference_experiment._apply_douban_protocol
    original_checkpoint = reference_experiment._checkpoint
    original_model = reference_experiment.ConfigurableGADL
    original_variants = dict(reference_experiment.VARIANTS)

    def apply_full_graph(dataset: Any, gids: Iterable[int], ignored_protocol: str) -> Dict[str, Any]:
        del ignored_protocol
        return original_apply_protocol(dataset, gids, "full-graph")

    def checkpoint_with_greedy(*checkpoint_args: Any, **checkpoint_kwargs: Any) -> Dict[str, Any]:
        checkpoint = original_checkpoint(*checkpoint_args, **checkpoint_kwargs)
        z_src = checkpoint_args[2]
        z_tgt = checkpoint_args[3]
        pairs = checkpoint_args[7]
        checkpoint["paper_greedy_one_to_one_evaluation"] = greedy_evaluation(z_src, z_tgt, pairs)
        return checkpoint

    modes = {
        "commutativity_weight": full_configuration["commutativity_weight"],
        "reconstruction_mode": full_configuration["reconstruction_mode"],
        "functional_map_schedule": full_configuration["functional_map_schedule"],
        "activation_mode": full_configuration["activation_mode"],
    }
    try:
        reference_experiment._apply_douban_protocol = apply_full_graph
        reference_experiment._checkpoint = checkpoint_with_greedy
        reference_experiment.ConfigurableGADL = model_factory
        reference_experiment.VARIANTS[variant_name] = modes
        result = reference_experiment._run(args, variant_name, seed=0)
    finally:
        reference_experiment._apply_douban_protocol = original_apply_protocol
        reference_experiment._checkpoint = original_checkpoint
        reference_experiment.ConfigurableGADL = original_model
        reference_experiment.VARIANTS.clear()
        reference_experiment.VARIANTS.update(original_variants)

    result["configuration"] = dict(full_configuration)
    result["protocol"]["declared_online_edge_entries"] = 16_328
    return result


def _checkpoint_at(result: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    return next(item for item in result["checkpoints"] if item["epoch"] == epoch)


def _find_metric(mapping: Dict[str, Any], candidates: Iterable[str]) -> Optional[float]:
    for key in candidates:
        if key in mapping:
            return float(mapping[key])
    for value in mapping.values():
        if isinstance(value, dict):
            found = _find_metric(value, candidates)
            if found is not None:
                return found
    return None


def _public_h1(checkpoint: Dict[str, Any]) -> Optional[float]:
    return _find_metric(
        checkpoint["public_reference_evaluation"],
        ("Hits@1", "hits_at_1", "hits@1", "hit_at_1", "H@1", "1"),
    )


def _planetalign_source_h1(checkpoint: Dict[str, Any]) -> Optional[float]:
    return _find_metric(
        checkpoint["planetalign_average_tie_evaluation"].get("metrics", {}).get("source_to_target", {}),
        ("Hits@1", "hits_at_1", "hits@1", "hit_at_1", "H@1", "1"),
    )


def _load_filtered_public_h1(path: Path, epoch: int = 100) -> Optional[float]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    for result in payload.get("results", []):
        if result.get("variant") == "public_reference_compatible" and result.get("seed") == 0:
            return _public_h1(_checkpoint_at(result, epoch))
    return None


# 功能：抽取同一检查点的参考、平均同分和全局贪心指标，便于并排比较。
def _evaluator_comparison(result: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for checkpoint in result["checkpoints"]:
        public_h1 = _public_h1(checkpoint)
        planetalign_h1 = _planetalign_source_h1(checkpoint)
        greedy_h1 = checkpoint["paper_greedy_one_to_one_evaluation"]["source_to_target_hits_at_1"]
        output[str(checkpoint["epoch"])] = {
            "public_reference_h1": public_h1,
            "planetalign_average_tie_source_to_target_h1": planetalign_h1,
            "paper_greedy_one_to_one_h1": greedy_h1,
            "planetalign_minus_public": None if public_h1 is None or planetalign_h1 is None else planetalign_h1 - public_h1,
            "greedy_minus_public": None if public_h1 is None else greedy_h1 - public_h1,
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Douban")
    parser.add_argument("--root", default="data")
    parser.add_argument("--gids", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--train-ratio", type=float, default=0.0)
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tie-atol", type=float, default=1e-8)
    parser.add_argument("--tie-rtol", type=float, default=1e-7)
    parser.add_argument("--collapse-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--public-filtered-json",
        type=Path,
        default=REPO_ROOT / "logs/gadl_one_to_one/douban_public_reference_attribution_seed0.json",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


# 功能：运行固定协议并核对检查点，写出不同配置与评估器的对比，不做测试分数选点。
def main() -> None:
    args = parse_args()
    if args.dataset.lower() != "douban":
        raise ValueError("paper-declared-profile is defined only for Douban Online-Offline")
    if args.total_epochs != PAPER_DECLARED_PROFILE["total_epochs"]:
        raise ValueError("paper-declared-profile requires exactly 100 epochs")

    public_full = _run_full_graph(
        args,
        "public_reference_full_graph_control",
        ConfigurableGADL,
        {**PUBLIC_REFERENCE_MODEL_PROFILE, "graph_protocol": "full-graph", "total_epochs": 100},
    )
    paper_full = _run_full_graph(
        args,
        "paper_declared_full_graph",
        _paper_model_factory,
        PAPER_DECLARED_PROFILE,
    )

    public_filtered_h1 = _load_filtered_public_h1(args.public_filtered_json)
    public_full_h1 = _public_h1(_checkpoint_at(public_full, 100))
    paper_full_h1 = _public_h1(_checkpoint_at(paper_full, 100))
    payload = {
        "experiment": "gadl_table_4_gap_final_location",
        "paper_version_audit": {
            "document": "arXiv:2509.09597v3, 20 June 2026",
            "reported_result_location": "Table 2 in v3; Table 4 is dataset statistics",
            "reported_douban_online_offline_hits": {
                "1": 0.5331,
                "5": 0.7361,
                "10": 0.8067,
                "50": 0.9418,
            },
            "evaluation_text": "cosine similarity plus greedy unmatched-pair selection",
            "ambiguity": "Greedy yields one assignment, but the paper reports Hit@5/10/50; tie behavior is not specified.",
        },
        "paper_declared_profile": PAPER_DECLARED_PROFILE,
        "differences_from_public_reference": PUBLIC_REFERENCE_DIFFERENCES,
        "seed": 0,
        "checkpoints": list(CHECKPOINTS),
        "results": [public_full, paper_full],
        "evaluator_comparison": {
            "public_reference_full_graph_control": _evaluator_comparison(public_full),
            "paper_declared_full_graph": _evaluator_comparison(paper_full),
        },
        "paired_attribution_epoch_100": {
            "public_reference_filtered_h1": public_filtered_h1,
            "public_reference_full_h1": public_full_h1,
            "paper_declared_full_h1": paper_full_h1,
            "data_protocol_delta_public_full_minus_filtered": (
                None if public_filtered_h1 is None or public_full_h1 is None else public_full_h1 - public_filtered_h1
            ),
            "model_configuration_delta_paper_minus_public_on_full_graph": (
                None if public_full_h1 is None or paper_full_h1 is None else paper_full_h1 - public_full_h1
            ),
            "unexplained_gap_to_paper_53_31": None if paper_full_h1 is None else 0.5331 - paper_full_h1,
        },
        "constraints": {
            "checkpoint_selection": False,
            "test_metric_tuning": False,
            "m2m_used": False,
        },
    }
    if not all(math.isfinite(float(x)) for x in [public_full_h1, paper_full_h1] if x is not None):
        raise RuntimeError("non-finite final evaluation")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload["paired_attribution_epoch_100"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
