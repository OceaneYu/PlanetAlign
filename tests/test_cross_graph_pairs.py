"""CGP-v1 examples and an independent explicit-relation-set oracle."""

import copy
import itertools
import random
import unittest
from unittest import mock

from PlanetAlign.m2m import evaluate_predictions
from PlanetAlign.metrics import (
    CGP_METRICS, InvalidPredictionError, NoEvaluableRelationsError,
    UnsupportedCGPOverlapError, cross_graph_pair_scores, many_to_many_scores,
)


def unit(src=(), tgt=()):
    return {"src": list(src), "tgt": list(tgt)}


def explicit_reference(gt, pred):
    """Small cases only: independent set unions, OR scope, set differences."""
    def relations(entities):
        return {(int(s), int(t)) for u in entities.values()
                for s in u.get("src", []) for t in u.get("tgt", [])}

    truth = relations(gt)
    if not truth:
        return None
    e_src = {int(s) for u in gt.values() for s in u.get("src", [])}
    e_tgt = {int(t) for u in gt.values() for t in u.get("tgt", [])}
    evaluated = {(s, t) for s, t in relations(pred) if s in e_src or t in e_tgt}
    tp, fp, fn = len(truth & evaluated), len(evaluated - truth), len(truth - evaluated)
    return {"CGP-P": tp / (tp + fp) if tp + fp else 0.0,
            "CGP-R": tp / (tp + fn), "CGP-F1": 2 * tp / (2 * tp + fp + fn),
            "TP": tp, "FP": fp, "FN": fn,
            "NumGTPairs": len(truth), "NumPredPairs": len(evaluated)}


class CrossGraphPairTest(unittest.TestCase):
    def setUp(self):
        self.gt = {"A": unit([1], [5]), "B": unit([2], [6])}

    def test_perfect_prediction(self):
        result = cross_graph_pair_scores(self.gt, self.gt)
        self.assertEqual([result[k] for k in CGP_METRICS], [1.0, 1.0, 1.0])
        self.assertEqual(result["TP"], 2)

    def test_swapped_one_to_one_correspondences_are_zero(self):
        pred = {"P": unit([1], [6]), "Q": unit([2], [5])}
        result = cross_graph_pair_scores(self.gt, pred)
        self.assertEqual([result[k] for k in CGP_METRICS], [0.0, 0.0, 0.0])
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (0, 2, 2))

    def test_split_product_loses_off_diagonal_relations(self):
        result = cross_graph_pair_scores({"A": unit([1, 2], [5, 6])}, self.gt)
        self.assertEqual(result["CGP-P"], 1)
        self.assertEqual(result["CGP-R"], 0.5)
        self.assertAlmostEqual(result["CGP-F1"], 2 / 3)
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (2, 0, 2))

    def test_merge_adds_off_diagonal_false_positives(self):
        result = cross_graph_pair_scores(self.gt, {"P": unit([1, 2], [5, 6])})
        self.assertEqual(result["CGP-P"], 0.5)
        self.assertEqual(result["CGP-R"], 1)
        self.assertAlmostEqual(result["CGP-F1"], 2 / 3)
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (2, 2, 0))

    def test_empty_and_single_sided_predictions_are_zero(self):
        for pred in ({}, {"P": unit(), "Q": unit()},
                     {"P": unit([1, 2]), "Q": unit((), [5, 6])},
                     {"P": {"src": [1, 2]}, "Q": {"tgt": [5, 6]}}):
            with self.subTest(pred=pred):
                result = cross_graph_pair_scores(self.gt, pred)
                self.assertEqual([result[k] for k in CGP_METRICS], [0.0, 0.0, 0.0])
                self.assertEqual((result["NumPredPairs"], result["FN"]), (0, 2))

    def test_scope_uses_or_and_excludes_only_both_outside_pairs(self):
        result = cross_graph_pair_scores({"A": unit([1], [5])}, {"P": unit([1, 99], [5, 200])})
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (1, 2, 0))
        self.assertEqual((result["NumGTPairs"], result["NumPredPairs"]), (1, 3))
        self.assertAlmostEqual(result["CGP-P"], 1 / 3)
        self.assertEqual(result["CGP-R"], 1)
        self.assertEqual(result["CGP-F1"], 0.5)

    def test_either_endpoint_alone_brings_a_wrong_relation_into_scope(self):
        for pred in ({"P": unit([1], [200])}, {"P": unit([99], [5])}):
            with self.subTest(pred=pred):
                result = cross_graph_pair_scores({"A": unit([1], [5])}, pred)
                self.assertEqual((result["TP"], result["FP"], result["FN"]), (0, 1, 1))

    def test_pure_background_group_does_not_change_any_counts(self):
        pred = {**self.gt, "background": unit([99, 100], [200, 201])}
        self.assertEqual(cross_graph_pair_scores(self.gt, pred), cross_graph_pair_scores(self.gt, self.gt))

    def test_ids_order_and_duplicate_members_do_not_change_score_or_inputs(self):
        gt = {"A": unit([1, 2], [5, 6])}
        expected = cross_graph_pair_scores(gt, self.gt)
        for order in itertools.permutations(self.gt.values()):
            pred = {f"renamed-{i}": {s: list(reversed(ns)) * 2 for s, ns in v.items()}
                    for i, v in enumerate(order)}
            renamed_gt = {99: unit([2, 1, 1], [6, 5, 6])}
            before = copy.deepcopy((renamed_gt, pred))
            self.assertEqual(cross_graph_pair_scores(renamed_gt, pred), expected)
            self.assertEqual((renamed_gt, pred), before)

    def test_duplicate_units_and_cross_unit_reuse_are_illegal(self):
        for pred in (
            {"P": unit([1], [5]), "Q": unit([1], [5])},
            {"P": unit([1], [5]), "Q": unit([1, 2], [6])},
            {"P": unit([1], [5]), "Q": unit([2], [5, 6])},
            {"P": unit([1, 2], [5, 6]), "Q": unit([1, 2], [5, 6])},
        ):
            before = copy.deepcopy(pred)
            with self.subTest(pred=pred), self.assertRaises(InvalidPredictionError):
                cross_graph_pair_scores(self.gt, pred)
            self.assertEqual(pred, before)

    def test_partition_validation_precedes_pair_and_scope_filtering(self):
        for pred in ({"P": unit([99]), "Q": unit([99])},
                     {"P": unit([99], [200]), "Q": unit([99], [201])}):
            with self.subTest(pred=pred), self.assertRaisesRegex(InvalidPredictionError, "src node 99"):
                cross_graph_pair_scores(self.gt, pred)

    def test_relations_are_ordered_and_graph_sides_are_distinct(self):
        gt = {"A": unit([1], [2]), "source_scope": unit([2]), "target_scope": unit((), [1])}
        result = cross_graph_pair_scores(gt, {"P": unit([2], [1])})
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (0, 1, 1))
        diagonal = {"A": unit([7], [7])}
        self.assertEqual(cross_graph_pair_scores(diagonal, diagonal)["TP"], 1)
        both_directions = {"A": unit([1], [2]), "B": unit([2], [1])}
        self.assertEqual(cross_graph_pair_scores(both_directions, both_directions)["TP"], 2)

    def test_single_sided_gt_nodes_remain_in_test_scope(self):
        gt = {"A": unit([1], [5]), "source_only": unit([3]), "target_only": unit((), [7])}
        result = cross_graph_pair_scores(gt, {"P": unit([3], [99]), "Q": unit([98], [7])})
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (0, 2, 1))

    def test_counts_are_pooled_not_macro_averaged(self):
        gt = {"small": unit([1], [5]), "large": unit([2, 3], [6, 7])}
        result = cross_graph_pair_scores(gt, {"P": gt["small"]})
        self.assertEqual(result["CGP-P"], 1)
        self.assertEqual(result["CGP-R"], 1 / 5)
        self.assertAlmostEqual(result["CGP-F1"], 1 / 3)

    def test_no_gt_relations_is_not_a_normal_zero_score(self):
        for gt in ({}, {"empty": unit()}, {"A": unit([1]), "B": unit((), [5])}):
            with self.subTest(gt=gt), self.assertRaisesRegex(NoEvaluableRelationsError, "无可评估真实关系"):
                cross_graph_pair_scores(gt, {"P": unit([1], [5])})

    def test_overlapping_gt_or_mode_is_explicitly_unsupported(self):
        for gt in ({"A": unit([1], [5]), "B": unit([1], [6])},
                   {"A": unit([1], [5]), "B": unit([2], [5])}):
            with self.subTest(gt=gt), self.assertRaisesRegex(UnsupportedCGPOverlapError, "overlapping ground truth"):
                cross_graph_pair_scores(gt, {})
        with self.assertRaises(UnsupportedCGPOverlapError):
            cross_graph_pair_scores(self.gt, self.gt, allow_overlap=True)

    def test_large_product_uses_integer_counts_without_enumeration(self):
        # 4.9 billion pairs would be infeasible as a Python set; only 140k
        # memberships are supplied. Also guards against 32-bit count overflow.
        n = 70_000
        gt = {"A": unit(range(n), range(n))}
        result = cross_graph_pair_scores(gt, gt)
        self.assertEqual(result["TP"], n * n)
        self.assertIsInstance(result["NumPredPairs"], int)
        self.assertEqual(result["CGP-F1"], 1)
        scoped = cross_graph_pair_scores({"one": unit([0], [0])}, gt)
        self.assertEqual(scoped["NumPredPairs"], 2 * n - 1)
        self.assertEqual(scoped["FP"], 2 * n - 2)
        self.assertEqual(scoped["TP"], 1)

    def test_random_non_overlapping_cases_match_explicit_sets(self):
        rng = random.Random(20261108)

        def entities(count, node_count):
            groups = {str(i): unit() for i in range(count)}
            for side in ("src", "tgt"):
                for node in range(node_count):
                    i = rng.randrange(count + 1)
                    if i < count:
                        groups[str(i)][side].extend([node] * rng.randint(1, 2))
            return groups

        for _ in range(1000):
            gt = entities(rng.randrange(6), 7)
            pred = entities(rng.randrange(7), 10)
            expected = explicit_reference(gt, pred)
            if expected is None:
                with self.assertRaises(NoEvaluableRelationsError):
                    cross_graph_pair_scores(gt, pred)
            else:
                self.assertEqual(cross_graph_pair_scores(gt, pred), expected)


class CrossGraphPairRoutingTest(unittest.TestCase):
    def setUp(self):
        self.gt = {"A": unit([1], [5]), "B": unit([2], [6])}
        self.pred = {"P": unit([1], [6]), "Q": unit([2], [5])}

    def test_three_metrics_share_one_count_and_ignore_aligned_predictions(self):
        with mock.patch("PlanetAlign.metrics.many_to_many.cross_graph_pair_scores",
                        wraps=cross_graph_pair_scores) as compute:
            result = many_to_many_scores(self.gt, self.pred,
                                         metrics=["CGP-P", "cgp_r", " CGP-F1 ", "cgp-p"],
                                         aligned_pred_entities=self.gt)
        self.assertEqual(compute.call_count, 1)
        self.assertEqual(result, {k: 0.0 for k in CGP_METRICS})

    def test_public_cgp_only_calls_no_group_matching_or_completion(self):
        with mock.patch("PlanetAlign.m2m.core.align_prediction_to_ground_truth", side_effect=AssertionError), \
                mock.patch("PlanetAlign.m2m.core.complete_prediction_entities", side_effect=AssertionError), \
                mock.patch("PlanetAlign.metrics.metrics_MSF1.min_weight_full_bipartite_matching", side_effect=AssertionError), \
                mock.patch("scipy.optimize.linear_sum_assignment", side_effect=AssertionError):
            for match in (True, False):
                result = evaluate_predictions(self.gt, self.pred, metrics=iter(CGP_METRICS), match_entities=match)
                self.assertEqual(result, {k: 0.0 for k in CGP_METRICS})

    def test_combined_evaluation_preserves_existing_metric_results(self):
        old = many_to_many_scores(self.gt, self.pred)
        combined = many_to_many_scores(self.gt, self.pred, metrics=[*old, *CGP_METRICS])
        self.assertEqual({k: combined[k] for k in old}, old)
        old = evaluate_predictions(self.gt, self.pred)
        combined = evaluate_predictions(self.gt, self.pred, metrics=[*old, *CGP_METRICS])
        self.assertEqual({k: combined[k] for k in old}, old)
        self.assertEqual(set(old), {"Macro Set-F1", "Micro Set-F1", "AUIS", "AUCS"})

    def test_wrappers_propagate_invalid_empty_gt_and_overlap_status(self):
        for wrapper in (evaluate_predictions, many_to_many_scores):
            with self.subTest(wrapper=wrapper.__name__):
                with self.assertRaises(NoEvaluableRelationsError):
                    wrapper({}, {}, metrics=CGP_METRICS)
                with self.assertRaises(UnsupportedCGPOverlapError):
                    wrapper(self.gt, self.pred, metrics=CGP_METRICS, allow_overlap=True)
                with self.assertRaises(InvalidPredictionError):
                    wrapper(self.gt, {"P": self.gt["A"], "Q": self.gt["A"]}, metrics=CGP_METRICS)

    def test_cgp_rejects_side_projection(self):
        with self.assertRaisesRegex(ValueError, "both src and tgt"):
            evaluate_predictions(self.gt, self.pred, metrics=CGP_METRICS, sides=("src",))

    def test_empty_gt_is_reported_before_legacy_metrics_in_mixed_request(self):
        for wrapper in (evaluate_predictions, many_to_many_scores):
            with self.subTest(wrapper=wrapper.__name__), self.assertRaises(NoEvaluableRelationsError):
                wrapper({}, {}, metrics=["AUCS", "CGP-F1"])

    def test_blind_scoring_wrapper_does_not_match_units_for_cgp(self):
        from PlanetAlign.m2m.blind import evaluate_blind
        with mock.patch("PlanetAlign.m2m.blind.blind_predict", return_value=self.pred), \
                mock.patch("PlanetAlign.m2m.blind.align_prediction_to_ground_truth", side_effect=AssertionError):
            scores, pred = evaluate_blind(None, self.gt, None, None,
                                          metrics=iter(CGP_METRICS), return_predictions=True)
        self.assertIs(pred, self.pred)
        self.assertEqual(scores, {k: 0.0 for k in CGP_METRICS})


if __name__ == "__main__":
    unittest.main()
