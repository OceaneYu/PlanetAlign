# 文件作用：验证节点编号规范化、缺失预测、组编号变化及公共 API 边界。
# 核心依赖：PlanetAlign/m2m/core.py 的实体格式、JSON 与评估接口；样例在本文件构造，不需完整基准。
# 阅读建议：先读测试类的准备数据和对应断言，可用 python -m unittest discover -s tests 运行。

import tempfile
import unittest
from pathlib import Path

import PlanetAlign.m2m as m2m_module
from PlanetAlign.m2m import (
    align_prediction_to_ground_truth,
    evaluate_predictions,
    load_entity_map,
    normalize_entity_map,
    save_entity_map,
)


# 功能：用小型实体字典与临时 JSON 检查调用约定。
class ManyToManyInterfaceTest(unittest.TestCase):
    # 功能：准备两个真实实体，供各测试独立比较预测集合。
    def setUp(self):
        self.gt_entities = {
            "e0": {"src": [0], "tgt": [1, 2]},
            "e1": {"src": [1], "tgt": [0]},
        }

    def test_normalize_entity_map_deduplicates_and_casts_ints(self):
        raw = {"0": {"src": ["1", 1, 2], "tgt": [3, "3"]}}

        normalized = normalize_entity_map(raw)

        self.assertEqual(normalized, {"0": {"src": [1, 2], "tgt": [3]}})

    def test_save_and_load_wrapped_prediction_json(self):
        pred = {"e0": {"src": [0], "tgt": [1, 2]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.json"

            save_entity_map(pred, path, dataset_name="toy_m2m", metadata={"method": "dummy"})
            loaded = load_entity_map(path)

        self.assertEqual(loaded, {"e0": {"src": [0], "tgt": [1, 2]}})

    def test_missing_prediction_entity_is_scored_as_empty(self):
        pred = {"e0": {"src": [0], "tgt": [1, 2]}}

        scores = evaluate_predictions(self.gt_entities, pred)

        self.assertLess(scores["Macro Set-F1"], 1.0)
        self.assertLess(scores["Micro Set-F1"], 1.0)

    # 功能：仅改变预测组编号，确认适配器恢复对应且所有分数仍为 1。
    def test_group_id_permutation_is_matched_by_adapter(self):
        gt = {
            "e0": {"src": [0], "tgt": [0, 1]},
            "e1": {"src": [1, 2], "tgt": [2]},
            "e2": {"src": [3], "tgt": [3]},
        }
        pred = {
            "cluster_b": {"src": [1, 2], "tgt": [2]},
            "cluster_c": {"src": [3], "tgt": [3]},
            "cluster_a": {"src": [0], "tgt": [0, 1]},
        }

        aligned = align_prediction_to_ground_truth(gt, pred)
        scores = evaluate_predictions(gt, pred)

        self.assertEqual(set(aligned["e0"]["tgt"]), {0, 1})
        for value in scores.values():
            self.assertAlmostEqual(value, 1.0)

    # 功能：确认借助真实分组的旧相似度接口未从公共包导出。
    def test_gt_assisted_similarity_interfaces_are_not_public(self):
        for name in (
            "predictions_from_similarity",
            "evaluate_similarity",
            "ManyToManyBaseline",
            "evaluate_model",
        ):
            self.assertFalse(hasattr(m2m_module, name), name)


if __name__ == "__main__":
    unittest.main()
