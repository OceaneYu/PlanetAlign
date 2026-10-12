# 文件作用：验证完美/缺失预测、指标固定数值、边界情况和加权指标显式请求。
# 核心依赖：PlanetAlign/metrics/many_to_many.py 与 PlanetAlign/metrics/metrics_AUIS.py、PlanetAlign/metrics/metrics_AUCS.py；样例在本文件构造，不需完整基准。
# 阅读建议：先读测试类的准备数据和对应断言，可用 python -m unittest discover -s tests 运行。

import unittest

import PlanetAlign.metrics as metrics_module
from PlanetAlign.metrics import (
    aucs_score,
    auis_score,
    many_to_many_scores,
    weighted_aucs_score,
    weighted_auis_score,
)


# 功能：用固定实体字典检查统一评分与 AUIS/AUCS，防止指标定义被悄悄改变。
class ManyToManyMetricsTest(unittest.TestCase):
    def setUp(self):
        self.gt_entities = {
            "e0": {"src": [0], "tgt": [1, 2]},
            "e1": {"src": [1], "tgt": [0]},
        }
        self.pred_entities = {
            "e0": {"src": [0], "tgt": [1, 2]},
            "e1": {"src": [1], "tgt": [0]},
        }

    def test_perfect_prediction_scores_are_one(self):
        scores = many_to_many_scores(self.gt_entities, self.pred_entities)

        self.assertEqual(
            set(scores),
            {"Macro Set-F1", "Micro Set-F1", "AUIS", "AUCS"},
        )
        for value in scores.values():
            self.assertAlmostEqual(value, 1.0)

    def test_missing_entity_is_treated_as_empty_prediction(self):
        pred_missing = {"e0": {"src": [0], "tgt": [1]}}

        scores = many_to_many_scores(self.gt_entities, pred_missing)

        self.assertLess(scores["Macro Set-F1"], 1.0)
        self.assertLess(scores["Micro Set-F1"], 1.0)

    # 功能：用手工分组核对固定 AUIS/AUCS 数值，保护宏平均和加权口径。
    def test_integrity_and_compactness_regression_values(self):
        gt = {
            "a": {"src": [0, 1], "tgt": [0]},
            "b": {"src": [2], "tgt": [1, 2, 3]},
        }
        fragmented = {
            "p0": {"src": [0], "tgt": []},
            "p1": {"src": [1], "tgt": [0]},
            "p2": {"src": [2], "tgt": [1]},
            "p3": {"src": [], "tgt": [2, 3]},
        }

        self.assertAlmostEqual(auis_score(gt, fragmented), 1.0 / 3.0)
        self.assertAlmostEqual(weighted_auis_score(gt, fragmented), 1.0 / 3.0)
        self.assertAlmostEqual(aucs_score(gt, fragmented), 7.0 / 12.0)
        self.assertAlmostEqual(weighted_aucs_score(gt, fragmented), 25.0 / 42.0)

    def test_existing_edge_case_behavior_is_preserved(self):
        gt = {
            "a": {"src": [0, 1], "tgt": [0]},
            "b": {"src": [2], "tgt": [1, 2, 3]},
        }
        no_overlap = {"p0": {"src": [99], "tgt": [99]}}
        singleton_gt = {"a": {"src": [0], "tgt": []}}

        self.assertEqual(auis_score(gt, no_overlap), 0.0)
        self.assertEqual(weighted_auis_score(gt, no_overlap), 0.0)
        self.assertEqual(aucs_score(gt, no_overlap), 1.0)
        self.assertEqual(weighted_aucs_score(gt, no_overlap), 1.0)
        self.assertEqual(auis_score(singleton_gt, {}), 1.0)
        self.assertEqual(aucs_score(singleton_gt, {}), 1.0)

    def test_weighted_metrics_are_explicit_only(self):
        default_scores = many_to_many_scores(self.gt_entities, self.pred_entities)
        supplemental = many_to_many_scores(
            self.gt_entities,
            self.pred_entities,
            metrics=["Weighted-AUIS", "Weighted-AUCS"],
        )

        self.assertNotIn("Weighted-AUIS", default_scores)
        self.assertNotIn("Weighted-AUCS", default_scores)
        self.assertEqual(set(supplemental), {"Weighted-AUIS", "Weighted-AUCS"})

    def test_public_imports_are_available(self):
        self.assertTrue(callable(auis_score))
        self.assertTrue(callable(weighted_auis_score))
        self.assertTrue(callable(aucs_score))
        self.assertTrue(callable(weighted_aucs_score))
        self.assertTrue(callable(many_to_many_scores))
        self.assertFalse(hasattr(metrics_module, "similarity_to_pred_entities"))
        self.assertFalse(hasattr(metrics_module, "pred_entities_from_similarity"))


if __name__ == "__main__":
    unittest.main()
