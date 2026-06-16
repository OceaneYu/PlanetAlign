"""Run the minimal Group-JOENA Phase A correctness audit.

Example:
    conda run -n m2m python scripts/group_joena_phase_a_audit.py

Optional pems08 smoke:
    conda run -n m2m python scripts/group_joena_phase_a_audit.py --run-pems08
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch
from torch_geometric.data import Data

from PlanetAlign.algorithms.group_joena import GroupJOENA
from PlanetAlign.algorithms.group_joena.group_aligner import GroupAligner
from PlanetAlign.algorithms.group_joena.inference import (
    node_scores_from_groups,
    predict_entities_from_group_alignment,
)
from PlanetAlign.algorithms.group_joena.quotient_graph import QuotientGraphBuilder
from PlanetAlign.algorithms.group_joena.types import QuotientGraph
from PlanetAlign.data import BaseData, Dataset
from PlanetAlign.m2m import evaluate_predictions, evaluate_similarity


GIDS = [0, 1]
GT = {
    "e0": {"src": [0], "tgt": [0, 1]},
    "e1": {"src": [1, 2], "tgt": [2]},
    "e2": {"src": [3], "tgt": [3]},
}


def tiny_assignments() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    us = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ut = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return us, ut, torch.eye(3)


def tiny_dataset(train_ratio: float = 0.75) -> BaseData:
    src_edges = torch.tensor([[0, 1, 2, 1, 2, 0, 3], [1, 2, 1, 0, 3, 2, 2]], dtype=torch.long)
    tgt_edges = torch.tensor([[0, 1, 2, 0, 1, 2, 3], [1, 0, 3, 2, 2, 0, 2]], dtype=torch.long)
    src_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tgt_x = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    anchors = torch.tensor([[0, 0], [1, 2], [2, 2], [3, 3]], dtype=torch.long)
    return BaseData(
        graphs=[
            Data(name="tiny_src", num_nodes=4, x=src_x, edge_index=src_edges),
            Data(name="tiny_tgt", num_nodes=4, x=tgt_x, edge_index=tgt_edges),
        ],
        anchor_links=anchors,
        name="tiny_m2m",
        train_ratio=train_ratio,
        seed=0,
    )


def quotient(embeddings: torch.Tensor, adjacency: torch.Tensor) -> QuotientGraph:
    assignments = torch.eye(embeddings.shape[0])
    masses = assignments.sum(dim=0)
    return QuotientGraph(
        embeddings=embeddings,
        adjacency=adjacency,
        masses=masses,
        valid_mask=masses > 0,
        assignments=assignments,
    )


def rounded_scores(scores: Mapping[str, float]) -> Dict[str, float]:
    return {key: round(float(value), 6) for key, value in scores.items()}


def metric_sanity() -> Dict[str, Dict[str, float]]:
    cases = {
        "A_perfect": GT,
        "B_permuted_ids": {
            "x1": {"src": [1, 2], "tgt": [2]},
            "x2": {"src": [3], "tgt": [3]},
            "x0": {"src": [0], "tgt": [0, 1]},
        },
        "C_single_giant": {"all": {"src": [0, 1, 2, 3], "tgt": [0, 1, 2, 3]}},
        "D_all_singleton": {f"s{i}": {"src": [i], "tgt": [i]} for i in range(4)},
        "E_wrong": {
            "w0": {"src": [0], "tgt": [3]},
            "w1": {"src": [1, 2], "tgt": [0]},
            "w2": {"src": [3], "tgt": [1]},
        },
        "F_empty": {},
    }
    return {name: rounded_scores(evaluate_predictions(GT, pred)) for name, pred in cases.items()}


def oracle_a() -> Dict[str, Any]:
    us, ut, t = tiny_assignments()
    scores = node_scores_from_groups(us, t, ut)
    similarity_scores, similarity_pred = evaluate_similarity(scores, GT, return_predictions=True)
    group_pred = predict_entities_from_group_alignment(GT, us, ut, t, target_size_mode="group")
    return {
        "similarity_metrics": rounded_scores(similarity_scores),
        "group_metrics": rounded_scores(evaluate_predictions(GT, group_pred.entities)),
        "prediction": similarity_pred,
    }


def oracle_b() -> Dict[str, Any]:
    us, ut, _ = tiny_assignments()
    graph_s = Data(num_nodes=4, edge_index=torch.tensor([[0, 1, 2, 2], [1, 0, 3, 1]]))
    graph_t = Data(num_nodes=4, edge_index=torch.tensor([[0, 1, 2, 2], [1, 0, 3, 0]]))
    h_s = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [0.0, 0.0]])
    h_t = torch.tensor([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
    builder = QuotientGraphBuilder(aggregation="normalized_sum")
    q_s = builder(graph_s, h_s, us)
    q_t = builder(graph_t, h_t, ut)
    result = GroupAligner(alpha=0.2, gamma_p=0.05, in_iter=10, out_iter=20)(q_s, q_t)
    return {
        "row_argmax": [int(x) for x in result.transport.argmax(dim=1).tolist()],
        "transport": [[round(float(x), 6) for x in row] for row in result.transport.tolist()],
    }


def fgw_sensitivity() -> Dict[str, float]:
    features = torch.eye(3)
    adj_path = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    adj_star = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    q_src = quotient(features, adj_path)
    q_same = quotient(features, adj_path)
    q_struct = quotient(features, adj_star)
    t_same = GroupAligner(alpha=0.7, gamma_p=0.05, in_iter=10, out_iter=20)(q_src, q_same).transport
    t_struct = GroupAligner(alpha=0.7, gamma_p=0.05, in_iter=10, out_iter=20)(q_src, q_struct).transport
    t_alpha0 = GroupAligner(alpha=0.0, gamma_p=0.05, in_iter=10, out_iter=20)(q_src, q_struct).transport
    t_feat = GroupAligner(alpha=0.0, gamma_p=0.05, in_iter=10, out_iter=20)(
        q_src, quotient(features[[2, 1, 0]], adj_path)
    ).transport
    return {
        "structure_delta_alpha_0_7": round(float((t_same - t_struct).abs().sum()), 6),
        "structure_delta_alpha_0": round(float((t_same - t_alpha0).abs().sum()), 6),
        "feature_swap_delta_alpha_0": round(float((t_same - t_feat).abs().sum()), 6),
    }


def train_tiny(oracle_k: bool) -> Dict[str, Any]:
    dataset = tiny_dataset(train_ratio=0.75 if oracle_k else 0.5)
    kwargs: Dict[str, Any] = {
        "hidden_dim": 16,
        "out_dim": 16,
        "max_epochs": 80,
        "eval_interval": 80,
        "supervised_weight": 5.0,
        "reconstruction_weight": 0.0,
        "cohesion_weight": 0.0,
        "separation_weight": 0.0,
        "sparsity_weight": 0.0,
        "group_alignment_alpha": 0.2,
        "learning_rate": 0.02,
        "weight_decay": 0.0,
        "assignment_temperature": 0.2,
        "seed": 9,
    }
    if oracle_k:
        kwargs["num_groups"] = len(GT)
    model = GroupJOENA(**kwargs)
    model.fit(
        dataset,
        gids=GIDS,
        train_entities=GT,
        oracle_num_groups=oracle_k,
        save_log=False,
        verbose=False,
    )
    pred = model.predict_many_to_many(GT, target_size_mode="group")
    return {
        "oracle_num_groups": oracle_k,
        "num_groups": model.num_groups_,
        "metrics": rounded_scores(evaluate_predictions(GT, pred)),
        "prediction": pred,
        "diagnostics": model.diagnostics(pred_entities=pred, gt_entities=GT),
    }


def oracle_c() -> Dict[str, Any]:
    dataset = tiny_dataset()
    model = GroupJOENA(
        num_groups=3,
        hidden_dim=16,
        out_dim=16,
        max_epochs=80,
        eval_interval=80,
        supervised_weight=5.0,
        reconstruction_weight=0.0,
        cohesion_weight=0.0,
        separation_weight=0.0,
        sparsity_weight=0.0,
        group_alignment_alpha=0.2,
        learning_rate=0.02,
        weight_decay=0.0,
        assignment_temperature=0.2,
        seed=9,
    )
    model.fit(
        dataset,
        gids=GIDS,
        train_entities=GT,
        oracle_num_groups=True,
        save_log=False,
        verbose=False,
    )
    true_t = torch.eye(3)
    pred = predict_entities_from_group_alignment(
        GT,
        model.source_assignments,
        model.target_assignments,
        true_t,
        target_size_mode="group",
    )
    return {
        "metrics": rounded_scores(evaluate_predictions(GT, pred.entities)),
        "prediction": pred.entities,
        "diagnostics": model.diagnostics(pred_entities=pred.entities, gt_entities=GT),
    }


def pems08_smoke(root: Path, name: str) -> Dict[str, Any]:
    gt_path = root / f"{name}_gt_many2many.json"
    if not gt_path.exists() or not (root / f"{name}.pt").exists():
        return {"status": "skipped", "reason": f"missing {name} in {root}"}
    dataset = Dataset(root=root, name=name, train_ratio=0.2, seed=42)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_entities = json.load(f)["entities"]
    model = GroupJOENA(
        num_groups=len(gt_entities),
        hidden_dim=32,
        out_dim=32,
        max_epochs=5,
        eval_interval=5,
        reconstruction_weight=0.0,
    )
    model.fit(
        dataset,
        gids=GIDS,
        num_groups=len(gt_entities),
        oracle_num_groups=True,
        save_log=False,
        verbose=False,
    )
    pred = model.predict_many_to_many(gt_entities, target_size_mode="group")
    return {
        "status": "ok",
        "metrics": rounded_scores(evaluate_predictions(gt_entities, pred)),
        "diagnostics": model.diagnostics(pred_entities=pred, gt_entities=gt_entities),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Group-JOENA Phase A correctness audit")
    parser.add_argument("--run-pems08", action="store_true", help="Also run a quick pems08_m2m smoke test")
    parser.add_argument("--m2m-root", type=Path, default=Path("data/m2m_no_overlap"))
    parser.add_argument("--dataset", default="pems08_m2m")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report: Dict[str, Any] = {
        "metric_sanity": metric_sanity(),
        "oracle_a_true_U_true_T": oracle_a(),
        "oracle_b_true_U_learned_T": oracle_b(),
        "fgw_sensitivity": fgw_sensitivity(),
        "tiny_oracle_k": train_tiny(oracle_k=True),
        "tiny_fallback_k": train_tiny(oracle_k=False),
        "oracle_c_learned_U_true_T": oracle_c(),
    }

    if args.run_pems08:
        report["pems08_quick"] = pems08_smoke(args.m2m_root, args.dataset)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
