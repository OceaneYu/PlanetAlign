# 文件作用：用固定小例子核对 MSF1-v2 的双侧成员分数、全局匹配和测试范围。
# 核心依赖：PlanetAlign/metrics/metrics_MSF1.py → macro_set_f1()；不运行训练或修改真实数据。
# 阅读建议：先读正确、部分正确、混组三个例子，再看拆分/合并与其他指标的隔离测试。

import copy
import itertools
import random
import unittest
from functools import lru_cache
from unittest import mock

from PlanetAlign.m2m import evaluate_predictions
from PlanetAlign.m2m.diagnostics import oracle_grouped_topk_target_metrics
from PlanetAlign.metrics import InvalidPredictionError, macro_f1_score, macro_set_f1, many_to_many_scores


def unit(src=(), tgt=()):
    return {"src": list(src), "tgt": list(tgt)}


class BilateralMacroSetF1Test(unittest.TestCase):
    def test_perfect_prediction_does_not_require_matching_entity_ids(self):
        self.assertEqual(macro_f1_score({"A": unit([1], [5])}, {"P": unit([1], [5])}), 1.0)

    def test_partial_members_receive_partial_credit(self):
        gt = {"A": unit([1], [5])}
        self.assertEqual(macro_f1_score(gt, {"P": unit([2], [5])}), 0.5)
        self.assertAlmostEqual(macro_f1_score(gt, {"P": unit([1], [])}), 2 / 3)

    def test_swapped_cross_graph_correspondences_are_not_perfect(self):
        gt = {"A": unit([1], [5]), "B": unit([2], [6])}
        pred = {"P": unit([1], [6]), "Q": unit([2], [5])}
        self.assertEqual(macro_f1_score(gt, pred), 0.5)

    def test_global_assignment_maximizes_total_f1(self):
        # F1 矩阵为 [[0.6, 8/15], [8/15, 0.2]]；贪心为 0.4，最优宏平均为 8/15。
        gt = {"A": unit([0], range(9)), "B": unit([1], range(9, 13))}
        pred = {
            "P": unit([0], [*range(5), *range(9, 13)]),
            "Q": unit([1], range(5, 9)),
        }
        self.assertAlmostEqual(macro_f1_score(gt, pred), 8 / 15)

    def test_ids_dictionary_order_and_member_order_do_not_change_score(self):
        gt = [unit([0], range(9)), unit([1], range(9, 13))]
        pred = [unit([0], [*range(5), *range(9, 13)]), unit([1], range(5, 9))]
        scores = []
        for gs, ps in itertools.product(itertools.permutations(gt), itertools.permutations(pred)):
            truth = {f"renamed-{i}": g for i, g in enumerate(gs)}
            prediction = {
                f"__extra__{i}:random": {side: list(reversed(nodes)) for side, nodes in p.items()}
                for i, p in enumerate(ps)
            }
            scores.append(macro_f1_score(truth, prediction))
        self.assertEqual(len(set(scores)), 1)
        self.assertAlmostEqual(scores[0], 8 / 15)

    def test_split_and_merge_are_penalized(self):
        whole = {"A": unit([1, 2], [5, 6])}
        pieces = {"P": unit([1], [5]), "Q": unit([2], [6])}
        with self.subTest(case="split"):
            self.assertAlmostEqual(macro_f1_score(whole, pieces), 2 / 3)
        with self.subTest(case="merge"):
            self.assertAlmostEqual(macro_f1_score(pieces, whole), 1 / 3)

    def test_missing_units_contribute_zero_and_duplicates_are_invalid(self):
        gt = {"A": unit([1], [5]), "B": unit([2], [6])}
        self.assertEqual(macro_f1_score(gt, {"P": gt["A"]}), 0.5)
        pred = {"P": gt["A"], "Q": gt["B"], "duplicate": gt["A"]}
        with self.assertRaises(InvalidPredictionError):
            macro_f1_score(gt, pred)

    def test_empty_inputs_and_empty_gt_units_yield_zero(self):
        for gt, pred in (
            ({}, {}),
            ({}, {"background": unit([99], [99])}),
            ({"A": unit([1], [5])}, {}),
            ({"empty": unit()}, {"empty": unit()}),
        ):
            with self.subTest(gt=gt, pred=pred):
                self.assertEqual(macro_f1_score(gt, pred), 0.0)

    def test_pure_training_background_and_empty_prediction_groups_are_ignored(self):
        gt = {"A": unit([1], [5])}
        pred = {
            "correct": unit([1], [5]),
            "training": unit([9], [19]),
            "background": unit([99, 100], [999]),
            "empty": unit(),
        }
        details = macro_set_f1(gt, pred)
        self.assertEqual(details["MSF1"], 1.0)
        self.assertEqual(details["NumPredEntities"], 1)
        self.assertEqual(details["IgnoredPredEntities"], 3)

    def test_mixed_group_keeps_all_training_or_background_members_as_fp(self):
        gt = {"A": unit([1], [5])}
        pred = {"mixed": unit([1, 99], [5]), "unlabeled": unit([100], [999])}
        self.assertEqual(macro_f1_score(gt, pred), 0.8)

    def test_single_side_groups_are_retained_and_count_as_extra_units(self):
        gt = {"A": unit([1], [5])}
        pred = {"source_only": unit([1], []), "target_only": unit([], [5])}
        details = macro_set_f1(gt, pred)
        self.assertAlmostEqual(details["MSF1"], 2 / 3)
        self.assertEqual(details["NumPredEntities"], 2)
        self.assertEqual(details["ExtraPredEntities"], 1)

    def test_same_number_on_different_graph_sides_is_not_the_same_node(self):
        self.assertAlmostEqual(macro_f1_score({"A": unit([7], [7])}, {"P": unit([7], [])}), 2 / 3)
        self.assertEqual(macro_f1_score({"A": unit([1], [5])}, {"P": unit([5], [1])}), 0.0)

    def test_duplicate_members_are_sets_and_inputs_are_not_mutated(self):
        gt = {"A": unit([1, 1], [5, 5])}
        pred = {"P": unit([1, 1, 1], [5, 5])}
        before = copy.deepcopy((gt, pred))
        self.assertEqual(macro_f1_score(gt, pred), 1.0)
        self.assertEqual((gt, pred), before)

    def test_overlapping_gt_units_are_supported(self):
        gt = {"A": unit([1, 2], [5]), "B": unit([2], [6])}
        with self.assertRaisesRegex(ValueError, "Invalid ground truth.*src node 2"):
            macro_f1_score(gt, gt)
        self.assertEqual(macro_f1_score(gt, {"P": gt["B"], "Q": gt["A"]}, allow_overlap=True), 1.0)

    def test_macro_weights_units_equally_despite_different_sizes(self):
        gt = {"small": unit([1], [5]), "large": unit(range(10, 20), range(30, 40))}
        self.assertEqual(macro_f1_score(gt, {"P": gt["small"]}), 0.5)
        self.assertEqual(macro_f1_score(gt, {"P": gt["large"]}), 0.5)

    def test_duplicate_perfect_or_merged_predictions_are_rejected(self):
        a, b = unit([1], [5]), unit([2], [6])
        for gt, p in (({"A": a}, a), ({"A": a, "B": b}, unit([1, 2], [5, 6]))):
            pred = {"P1": p, "P2": copy.deepcopy(p)}
            before = copy.deepcopy(pred)
            with self.subTest(gt=gt), self.assertRaisesRegex(InvalidPredictionError, "src node 1.*'P1'.*'P2'"):
                macro_set_f1(gt, pred)
            self.assertEqual(pred, before)

    def test_partial_overlap_on_either_side_is_rejected(self):
        gt = {"A": unit([1], [5]), "B": unit([2], [6])}
        for side, pred in (
            ("src", {"P": unit([1], [5]), "Q": unit([1, 2], [6])}),
            ("tgt", {"P": unit([1], [5]), "Q": unit([2], [5, 6])}),
        ):
            with self.subTest(side=side), self.assertRaisesRegex(InvalidPredictionError, side + " node"):
                macro_set_f1(gt, pred)

    def test_full_prediction_is_validated_before_scope_filtering(self):
        for gt in ({"A": unit([1], [5])}, {}):
            for pred in (
                {"background1": unit([99]), "background2": unit([99])},
                {"mixed": unit([1, 99], [5]), "background": unit([99], [999])},
            ):
                with self.subTest(gt=gt, pred=pred), self.assertRaisesRegex(InvalidPredictionError, "src node 99"):
                    macro_set_f1(gt, pred)

    def test_same_number_on_opposite_sides_in_different_units_is_valid(self):
        gt = {"A": unit([1], [2]), "B": unit([2], [1])}
        self.assertEqual(macro_f1_score(gt, gt), 1.0)

    def test_multiple_empty_predictions_are_valid_and_score_zero(self):
        details = macro_set_f1({"A": unit([1], [5])}, {"P": unit(), "Q": unit()})
        self.assertEqual(details["MSF1"], 0.0)
        self.assertEqual(details["IgnoredPredEntities"], 2)
        self.assertEqual(details["MissingPredEntities"], 1)

    def test_overlap_mode_is_explicit_and_has_no_duplicate_penalty(self):
        a, b = unit([1], [5]), unit([2], [6])
        self.assertEqual(macro_f1_score({"A": a}, {"P": a, "Q": a}, allow_overlap=True), 1.0)
        merged = unit([1, 2], [5, 6])
        # Document the limitation of the separate overlap mode, not an official
        # non-overlapping submission (which must reject these two copies).
        self.assertAlmostEqual(macro_f1_score({"A": a, "B": b}, {"P": merged}, allow_overlap=True), 1 / 3)
        self.assertAlmostEqual(macro_f1_score({"A": a, "B": b}, {"P": merged, "Q": merged}, allow_overlap=True), 2 / 3)

    def test_random_small_cases_match_independent_exhaustive_assignment(self):
        rng = random.Random(20261028)

        def entities(count, node_count, overlap):
            groups = {str(i): unit() for i in range(count)}
            for side in ("src", "tgt"):
                for node in range(node_count):
                    if overlap:
                        chosen = [i for i in range(count) if rng.random() < 0.3]
                    else:
                        chosen = [rng.randrange(count + 1)]
                    for i in chosen:
                        if i < count:
                            groups[str(i)][side].append(node)
            return groups

        def reference(gt, pred):
            def tagged(item):
                return {(side, n) for side in ("src", "tgt") for n in item[side]}
            gs = [tagged(v) for v in gt.values()]
            scope = set().union(*gs)
            ps = [p for p in map(tagged, pred.values()) if p & scope]
            f1 = [[2 * len(g & p) / (len(g) + len(p)) for p in ps] for g in gs]

            @lru_cache(None)
            def best(i, used):
                if i == len(gs):
                    return 0.0
                return max([best(i + 1, used)] + [
                    f1[i][j] + best(i + 1, used | (1 << j))
                    for j in range(len(ps)) if not used & (1 << j)
                ])
            return best(0, 0) / len(gs) if gs else 0.0

        for overlap in (False, True):
            for _ in range(200):
                gt = entities(rng.randrange(6), 5, overlap)
                pred = entities(rng.randrange(7), 7, overlap)
                actual = macro_f1_score(gt, pred, allow_overlap=overlap)
                self.assertAlmostEqual(actual, reference(gt, pred), places=13)
                self.assertTrue(0 <= actual <= 1)
                self.assertEqual(actual, macro_f1_score(
                    dict(reversed(list(gt.items()))), dict(reversed(list(pred.items()))),
                    allow_overlap=overlap))


class MacroSetF1RoutingTest(unittest.TestCase):
    def test_all_public_msf1_routes_reject_illegal_predictions(self):
        gt = {"A": unit([1], [5])}
        pred = {"P": gt["A"], "Q": gt["A"]}
        for alias in ("Macro Set-F1", "macro-set-f1", "MSF1", "macro-f1", "macrof1"):
            with self.subTest(alias=alias):
                with self.assertRaises(InvalidPredictionError):
                    many_to_many_scores(gt, pred, metrics=[alias], aligned_pred_entities=gt)
                for match in (False, True):
                    with self.assertRaises(InvalidPredictionError):
                        evaluate_predictions(gt, pred, metrics=[alias], match_entities=match)

    def test_overlap_mode_is_forwarded_by_public_wrappers(self):
        gt = {"A": unit([1, 2], [5]), "B": unit([2], [6])}
        self.assertEqual(evaluate_predictions(gt, gt, metrics=["MSF1"], allow_overlap=True), {"MSF1": 1.0})
        self.assertEqual(many_to_many_scores(gt, gt, metrics=["MSF1"], allow_overlap=True), {"MSF1": 1.0})

    def test_colliding_entity_ids_cannot_silently_remove_a_duplicate(self):
        gt = {"A": unit([1], [5])}
        with self.assertRaisesRegex(ValueError, "entity ids collide"):
            evaluate_predictions(gt, {1: gt["A"], "1": gt["A"]}, metrics=["MSF1"])

    def test_quotient_overlap_variants_are_not_legal_non_overlap_predictions(self):
        import torch
        from torch_geometric.data import Data
        from PlanetAlign.m2m.quotient import quotient_decode

        graph = Data(num_nodes=2, edge_index=torch.empty((2, 0), dtype=torch.long))
        similarity = torch.tensor([[0.9, 0.8], [0.9, 0.8]])
        gt = {"A": unit([0], [0]), "B": unit([1], [1])}
        pred, _ = quotient_decode(similarity, graph, graph)
        self.assertEqual(evaluate_predictions(gt, pred, metrics=["MSF1"])["MSF1"], 1.0)
        for options in ({"matcher": "greedy"}, {"overlap_expand_tau": 0.35},
                        {"matcher": "greedy", "overlap_expand_tau": 0.35}):
            pred, _ = quotient_decode(similarity, graph, graph, **options)
            with self.subTest(options=options), self.assertRaises(InvalidPredictionError):
                evaluate_predictions(gt, pred, metrics=["MSF1"])

    def test_every_msf1_alias_receives_raw_predictions(self):
        gt = {"A": unit([1], [5])}
        raw = {"P": unit([1, 99], [5])}
        aligned = {"A": unit([1], [5])}
        for alias in ("Macro Set-F1", "macro-set-f1", "MSF1", "macro-f1", "macrof1"):
            with self.subTest(alias=alias):
                scores = many_to_many_scores(gt, raw, metrics=[alias], aligned_pred_entities=aligned)
                self.assertEqual(list(scores.values()), [0.8])

    def test_msf1_bypasses_greedy_results_while_other_metrics_keep_old_behavior(self):
        gt = {"A": unit([0, 1], [0, 1]), "B": unit([2, 3], [2, 3])}
        pred = {"P": unit([0, 1], [2, 3]), "Q": unit([2, 3], [0, 1])}
        for prediction, expected_micro in ((pred, 0.0), (dict(reversed(list(pred.items()))), 1.0)):
            scores = evaluate_predictions(gt, prediction)
            self.assertEqual(scores["Macro Set-F1"], 0.5)
            # 其他指标的既有定义和平局行为留待后续独立处理。
            self.assertEqual(scores["Micro Set-F1"], expected_micro)
            self.assertAlmostEqual(scores["AUIS"], 1 / 3)
            self.assertEqual(scores["AUCS"], 0.0)
        with mock.patch("PlanetAlign.m2m.core.align_prediction_to_ground_truth", return_value={}):
            self.assertEqual(evaluate_predictions(gt, pred, metrics=["MSF1"])["MSF1"], 0.5)
        self.assertEqual(evaluate_predictions(gt, pred, metrics=["MSF1"], match_entities=False)["MSF1"], 0.5)

    def test_oracle_target_msf1_keeps_its_original_target_only_definition(self):
        gt = {"A": unit([1], [5])}
        pred = {"A": unit([2], [5])}
        self.assertEqual(oracle_grouped_topk_target_metrics(gt, pred)["OracleSrc+GT-K-Target-MSF1"], 1.0)
        self.assertEqual(macro_f1_score(gt, pred), 0.5)


if __name__ == "__main__":
    unittest.main()
