# 文件作用：验证参考加权重构公式、映射权重计划和显式兼容配置。
# 核心依赖：PlanetAlign/algorithms/gadl/profiles.py；样例在本文件构造，不需完整基准。
# 阅读建议：先读测试类的准备数据和对应断言，可用 python -m unittest discover -s tests 运行。

import unittest

import torch
import torch.nn.functional as F

from PlanetAlign.algorithms.gadl.profiles import (
    ConfigurableGADL,
    PUBLIC_REFERENCE_COMPATIBLE_PROFILE,
    functional_map_schedule_weight,
    reference_weighted_reconstruction_loss,
)


# 功能：比较加权二元交叉熵与参考公式，并检查单类别输入拒绝。
class ReferenceReconstructionTest(unittest.TestCase):
    def test_weighted_bce_matches_public_formula(self):
        embeddings = torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]])
        adjacency = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        logits = embeddings @ embeddings.T
        positives = adjacency.sum()
        total = adjacency.numel()
        pos_weight = (total - positives) / positives
        norm = total / ((total - positives) * 2)
        weights = torch.ones_like(adjacency)
        weights[adjacency == 1] = pos_weight
        expected = norm * F.binary_cross_entropy(
            torch.sigmoid(logits).reshape(-1),
            adjacency.reshape(-1),
            weight=weights.reshape(-1),
        )
        actual = reference_weighted_reconstruction_loss(embeddings, adjacency.to_sparse())
        self.assertTrue(torch.allclose(actual, expected))

    def test_weighted_bce_rejects_single_class(self):
        with self.assertRaises(ValueError):
            reference_weighted_reconstruction_loss(torch.ones(2, 2), torch.zeros(2, 2))


# 功能：检查固定权重与带下限的参考衰减计划。
class ReferenceScheduleTest(unittest.TestCase):
    def test_constant_schedule(self):
        self.assertEqual(functional_map_schedule_weight("constant", 100, 100), 1.0)

    def test_reference_schedule(self):
        self.assertEqual(functional_map_schedule_weight("reference_linear_floor_half", 0, 100), 1.0)
        self.assertAlmostEqual(functional_map_schedule_weight("reference_linear_floor_half", 1, 100), 0.99)
        self.assertEqual(functional_map_schedule_weight("reference_linear_floor_half", 50, 100), 0.5)
        self.assertEqual(functional_map_schedule_weight("reference_linear_floor_half", 100, 100), 0.5)


# 功能：确认默认行为保留、公开参考配置需要显式选择。
class ConfigurableProfileTest(unittest.TestCase):
    def test_base_defaults_preserve_planetalign_behavior(self):
        model = ConfigurableGADL()
        self.assertEqual(model.reconstruction_mode, "unweighted_dense_bce")
        self.assertEqual(model.functional_map_schedule, "constant")
        self.assertEqual(model.activation_mode, "planetalign_prelu")

    def test_public_profile_is_explicit(self):
        model = ConfigurableGADL.public_reference_compatible()
        self.assertEqual(model.reconstruction_mode, "reference_weighted_bce")
        self.assertEqual(model.functional_map_schedule, "reference_linear_floor_half")
        self.assertEqual(model.activation_mode, "reference_identity")
        self.assertEqual(model.commutativity_weight, 1e-2)
        self.assertEqual(PUBLIC_REFERENCE_COMPATIBLE_PROFILE["hidden_dim"], 512)


if __name__ == "__main__":
    unittest.main()
