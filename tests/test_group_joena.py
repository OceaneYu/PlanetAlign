import json
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from PlanetAlign.algorithms.group_joena import GroupJOENA, GroupJOENAConfig
from PlanetAlign.algorithms.group_joena.assignment import SoftGroupAssignment
from PlanetAlign.algorithms.group_joena.diagnostics import (
    assignment_diagnostics,
    prediction_diagnostics,
    score_diagnostics,
    transport_diagnostics,
)
from PlanetAlign.algorithms.group_joena.group_aligner import GroupAligner
from PlanetAlign.algorithms.group_joena.inference import (
    group_prediction_from_alignment,
    node_scores_from_groups,
    predict_entities_from_group_alignment,
)
from PlanetAlign.algorithms.group_joena.losses import assignment_entropy, internal_cohesion_loss
from PlanetAlign.algorithms.group_joena.quotient_graph import QuotientGraphBuilder
from PlanetAlign.algorithms.group_joena.types import QuotientGraph
from PlanetAlign.data import BaseData, Dataset
from PlanetAlign.m2m import evaluate_predictions, evaluate_similarity


PHASE_A_GT = {
    "e0": {"src": [0], "tgt": [0, 1]},
    "e1": {"src": [1, 2], "tgt": [2]},
    "e2": {"src": [3], "tgt": [3]},
}


def _tiny_assignments():
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


def _tiny_dataset(train_ratio=0.75):
    src_edges = torch.tensor(
        [
            [0, 1, 2, 1, 2, 0, 3],
            [1, 2, 1, 0, 3, 2, 2],
        ],
        dtype=torch.long,
    )
    tgt_edges = torch.tensor(
        [
            [0, 1, 2, 0, 1, 2, 3],
            [1, 0, 3, 2, 2, 0, 2],
        ],
        dtype=torch.long,
    )
    src_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tgt_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
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


def _quotient(embeddings, adjacency, assignments=None):
    k = embeddings.shape[0]
    if assignments is None:
        assignments = torch.eye(k)
    masses = assignments.sum(dim=0)
    return QuotientGraph(
        embeddings=embeddings,
        adjacency=adjacency,
        masses=masses,
        valid_mask=masses > 0,
        assignments=assignments,
    )


class GroupJOENATest(unittest.TestCase):
    def test_assignment_rows_are_normalized(self):
        module = SoftGroupAssignment(num_groups=3, embedding_dim=4, temperature=0.7)
        x = torch.randn(5, 4)

        u = module(x)

        self.assertEqual(u.shape, (5, 3))
        self.assertTrue(torch.allclose(u.sum(dim=1), torch.ones(5), atol=1e-5))

    def test_hard_assignment_has_one_group_per_node(self):
        u = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])

        hard = SoftGroupAssignment.hard_assignments(u)

        self.assertEqual(hard.shape, (3,))
        self.assertTrue(torch.all((0 <= hard) & (hard < 2)))

    def test_quotient_graph_shape_and_soft_or_range(self):
        edge_index = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]])
        graph = Data(num_nodes=3, edge_index=edge_index)
        embeddings = torch.randn(3, 4)
        assignments = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        builder = QuotientGraphBuilder(aggregation="soft_or", soft_or_gamma=1.0)

        q = builder(graph, embeddings, assignments)

        self.assertEqual(q.embeddings.shape, (2, 4))
        self.assertEqual(q.adjacency.shape, (2, 2))
        self.assertGreaterEqual(float(q.adjacency.min()), 0.0)
        self.assertLessEqual(float(q.adjacency.max()), 1.0)
        self.assertFalse(torch.isnan(q.adjacency).any())

    def test_empty_group_does_not_create_nan(self):
        edge_index = torch.tensor([[0, 1], [1, 0]])
        graph = Data(num_nodes=2, edge_index=edge_index)
        embeddings = torch.randn(2, 3)
        assignments = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        q = QuotientGraphBuilder()(graph, embeddings, assignments)

        self.assertFalse(q.valid_mask[1].item())
        self.assertFalse(torch.isnan(q.embeddings).any())
        self.assertFalse(torch.isnan(q.adjacency).any())

    def test_node_score_factorization_shape(self):
        us = torch.softmax(torch.randn(4, 3), dim=1)
        ut = torch.softmax(torch.randn(5, 2), dim=1)
        t = torch.softmax(torch.randn(3, 2).reshape(-1), dim=0).reshape(3, 2)

        s = node_scores_from_groups(us, t, ut)

        self.assertEqual(s.shape, (4, 5))

    def test_losses_are_finite(self):
        edge_index = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]])
        graph = Data(num_nodes=3, edge_index=edge_index)
        assignments = torch.softmax(torch.randn(3, 2), dim=1)

        loss = assignment_entropy(assignments) + internal_cohesion_loss(graph, assignments)

        self.assertTrue(torch.isfinite(loss))

    def test_inference_recovers_one_to_many_without_unrelated_node(self):
        gt = {"e0": {"src": [0], "tgt": [0, 1, 2]}, "e1": {"src": [1], "tgt": [3]}}
        us = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        ut = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        t = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

        pred = predict_entities_from_group_alignment(gt, us, ut, t, target_size_mode="group")

        self.assertEqual(set(pred.entities["e0"]["tgt"]), {0, 1, 2})
        self.assertNotIn(3, pred.entities["e0"]["tgt"])

    def test_align_prediction_does_not_need_ground_truth(self):
        us, ut, t = _tiny_assignments()

        pred = group_prediction_from_alignment(us, ut, t)

        self.assertEqual(set(pred.entities["g0"]["src"]), {0})
        self.assertEqual(set(pred.entities["g0"]["tgt"]), {0, 1})
        self.assertEqual(set(pred.entities["g1"]["src"]), {1, 2})
        self.assertEqual(set(pred.entities["g1"]["tgt"]), {2})

    def test_metric_sanity_cases_are_deterministic(self):
        gt = PHASE_A_GT
        cases = {
            "perfect": gt,
            "permuted_ids": {
                "x1": {"src": [1, 2], "tgt": [2]},
                "x2": {"src": [3], "tgt": [3]},
                "x0": {"src": [0], "tgt": [0, 1]},
            },
            "single_giant": {"all": {"src": [0, 1, 2, 3], "tgt": [0, 1, 2, 3]}},
            "all_singleton": {
                f"s{i}": {"src": [i], "tgt": [i]} for i in range(4)
            },
            "wrong": {
                "w0": {"src": [0], "tgt": [3]},
                "w1": {"src": [1, 2], "tgt": [0]},
                "w2": {"src": [3], "tgt": [1]},
            },
            "empty": {},
        }
        table = {name: evaluate_predictions(gt, pred) for name, pred in cases.items()}

        for value in table["perfect"].values():
            self.assertAlmostEqual(value, 1.0)
        for value in table["permuted_ids"].values():
            self.assertAlmostEqual(value, 1.0)
        self.assertLess(table["single_giant"]["MSF1"], 1.0)
        self.assertEqual(table["single_giant"]["M2M-EGS"], 1.0)
        self.assertEqual(table["empty"]["ACS"], 0.0)
        self.assertLess(table["wrong"]["MicroF1"], 1.0)

    def test_oracle_a_true_assignments_and_true_transport_are_optimal(self):
        us, ut, t = _tiny_assignments()

        scores = node_scores_from_groups(us, t, ut)
        metrics, pred = evaluate_similarity(scores, PHASE_A_GT, return_predictions=True)
        group_pred = predict_entities_from_group_alignment(PHASE_A_GT, us, ut, t, target_size_mode="group")
        group_metrics = evaluate_predictions(PHASE_A_GT, group_pred.entities)

        self.assertEqual(set(pred["e0"]["tgt"]), {0, 1})
        for value in metrics.values():
            self.assertAlmostEqual(value, 1.0)
        for value in group_metrics.values():
            self.assertAlmostEqual(value, 1.0)

    def test_oracle_b_true_assignments_learns_transport_on_tiny_case(self):
        us, ut, _ = _tiny_assignments()
        graph_s = Data(num_nodes=4, edge_index=torch.tensor([[0, 1, 2, 2], [1, 0, 3, 1]]))
        graph_t = Data(num_nodes=4, edge_index=torch.tensor([[0, 1, 2, 2], [1, 0, 3, 0]]))
        h_s = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [0.0, 0.0]])
        h_t = torch.tensor([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
        builder = QuotientGraphBuilder(aggregation="normalized_sum")
        q_s = builder(graph_s, h_s, us)
        q_t = builder(graph_t, h_t, ut)

        result = GroupAligner(alpha=0.2, gamma_p=0.05, in_iter=10, out_iter=20)(q_s, q_t)
        row_match = result.transport.argmax(dim=1)

        self.assertEqual(row_match.tolist(), [0, 1, 2])

    def test_oracle_c_true_transport_with_distinct_features_discovers_groups(self):
        dataset = _tiny_dataset()
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
            gids=[0, 1],
            train_entities=PHASE_A_GT,
            oracle_num_groups=True,
            save_log=False,
            verbose=False,
        )
        true_t = torch.eye(3)
        scores = node_scores_from_groups(model.source_assignments, true_t, model.target_assignments)
        pred = predict_entities_from_group_alignment(
            PHASE_A_GT,
            model.source_assignments,
            model.target_assignments,
            true_t,
            target_size_mode="group",
        )
        metrics = evaluate_predictions(PHASE_A_GT, pred.entities)

        self.assertGreater(float(scores[0, 0]), float(scores[0, 2]))
        for value in metrics.values():
            self.assertAlmostEqual(value, 1.0)

    def test_group_aligner_has_structure_and_feature_sensitivity(self):
        features = torch.eye(3)
        adj_path = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        adj_star = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        q_src = _quotient(features, adj_path)
        q_tgt_same = _quotient(features, adj_path)
        q_tgt_struct = _quotient(features, adj_star)
        align_struct = GroupAligner(alpha=0.7, gamma_p=0.05, in_iter=10, out_iter=20)
        align_feature = GroupAligner(alpha=0.0, gamma_p=0.05, in_iter=10, out_iter=20)

        t_same = align_struct(q_src, q_tgt_same).transport
        t_struct = align_struct(q_src, q_tgt_struct).transport
        t_alpha0 = align_feature(q_src, q_tgt_struct).transport
        q_tgt_swapped = _quotient(features[[2, 1, 0]], adj_path)
        t_feat_swapped = align_feature(q_src, q_tgt_swapped).transport

        self.assertGreater(float((t_same - t_struct).abs().sum()), 1e-4)
        self.assertLess(float((t_same - t_alpha0).abs().sum()), float((t_same - t_struct).abs().sum()))
        self.assertGreater(float((t_same - t_feat_swapped).abs().sum()), 1e-4)

    def test_quotient_split_allocation_invariance_and_group_size_effect(self):
        assignments = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        embeddings = torch.randn(3, 2)
        graph_a = Data(num_nodes=3, edge_index=torch.tensor([[0, 2], [2, 0]]))
        graph_b = Data(num_nodes=3, edge_index=torch.tensor([[1, 2], [2, 1]]))
        builder = QuotientGraphBuilder(aggregation="soft_or", soft_or_gamma=1.0)

        q_a = builder(graph_a, embeddings, assignments)
        q_b = builder(graph_b, embeddings, assignments)

        self.assertTrue(torch.allclose(q_a.adjacency, q_b.adjacency, atol=1e-6))

        assignments_big = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        graph_big = Data(num_nodes=4, edge_index=torch.tensor([[0, 3], [3, 0]]))
        emb_big = torch.randn(4, 2)
        soft_or = QuotientGraphBuilder(aggregation="soft_or", soft_or_gamma=1.0)(graph_big, emb_big, assignments_big)
        normalized = QuotientGraphBuilder(aggregation="normalized_sum")(graph_big, emb_big, assignments_big)

        self.assertGreaterEqual(float(soft_or.adjacency[0, 1]), float(normalized.adjacency[0, 1]))
        self.assertLessEqual(float(soft_or.adjacency[0, 1]), 1.0)

    def test_diagnostics_and_gradient_flow_on_tiny_step(self):
        dataset = _tiny_dataset()
        model = GroupJOENA(
            num_groups=3,
            hidden_dim=8,
            out_dim=8,
            max_epochs=1,
            eval_interval=1,
            supervised_weight=1.0,
            reconstruction_weight=0.0,
            cohesion_weight=0.0,
            separation_weight=0.0,
            sparsity_weight=0.0,
            group_alignment_alpha=0.2,
            seed=7,
        )

        model.fit(
            dataset,
            gids=[0, 1],
            train_entities=PHASE_A_GT,
            oracle_num_groups=True,
            save_log=False,
            verbose=False,
        )
        diag = model.diagnostics(gt_entities=PHASE_A_GT)

        self.assertTrue(model.loss_history[-1]["total_loss"] == model.loss_history[-1]["total_loss"])
        self.assertGreaterEqual(diag["source_assignment"]["non_empty_group_count"], 1)
        self.assertIn("encoder", diag["gradient_norms"])
        self.assertGreaterEqual(diag["gradient_norms"]["encoder"], 0.0)
        self.assertIn("effective_matched_pairs", diag["transport"])

    def test_tiny_oracle_k_overfit_reaches_phase_a_gate(self):
        dataset = _tiny_dataset()
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
            gids=[0, 1],
            train_entities=PHASE_A_GT,
            oracle_num_groups=True,
            save_log=False,
            verbose=False,
        )
        pred = model.predict_many_to_many(PHASE_A_GT, target_size_mode="group")
        scores = evaluate_predictions(PHASE_A_GT, pred)

        for key in ("ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"):
            self.assertGreaterEqual(scores[key], 0.90)

    def test_diagnostic_helpers_have_expected_keys(self):
        us, ut, t = _tiny_assignments()
        s = node_scores_from_groups(us, t, ut)

        self.assertIn("non_empty_group_count", assignment_diagnostics(us))
        self.assertIn("effective_matched_pairs", transport_diagnostics(t))
        self.assertIn("density_above_threshold", score_diagnostics(s))
        self.assertIn("predicted_group_count", prediction_diagnostics(PHASE_A_GT))

    def test_invalid_num_groups_raises(self):
        with self.assertRaises(ValueError):
            GroupJOENAConfig(num_groups=0)

    def test_pems08_smoke_if_data_exists(self):
        root = Path("data/m2m_no_overlap")
        name = "pems08_m2m"
        gt_path = root / f"{name}_gt_many2many.json"
        if not gt_path.exists() or not (root / f"{name}.pt").exists():
            self.skipTest("local m2m_no_overlap pems08 benchmark is not available")

        dataset = Dataset(root=root, name=name, train_ratio=0.2, seed=42)
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_entities = json.load(f)["entities"]
        model = GroupJOENA(
            num_groups=len(gt_entities),
            hidden_dim=16,
            out_dim=16,
            max_epochs=1,
            eval_interval=1,
            reconstruction_weight=0.0,
            cohesion_weight=0.0,
            separation_weight=0.0,
            sparsity_weight=0.0,
        ).to("cpu")

        model.train(
            dataset,
            gids=[0, 1],
            num_groups=len(gt_entities),
            oracle_num_groups=True,
            save_log=False,
            verbose=False,
        )
        scores = model.test_many_to_many(gt_entities)

        self.assertEqual(model.S.shape, (dataset.pyg_graphs[0].num_nodes, dataset.pyg_graphs[1].num_nodes))
        self.assertEqual(set(scores), {"ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"})


if __name__ == "__main__":
    unittest.main()
