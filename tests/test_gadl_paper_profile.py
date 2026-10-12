# 文件作用：验证论文参数声明与全局贪心一对一匹配的稳定同分行为。
# 核心依赖：PlanetAlign/algorithms/gadl/paper_profile.py 与 scripts/evaluate_gadl_paper_declared.py；样例在本文件构造，不需完整基准。
# 阅读建议：先读测试类的准备数据和对应断言，可用 python -m unittest discover -s tests 运行。

import unittest

import torch

from PlanetAlign.algorithms.gadl.paper_profile import (
    PAPER_DECLARED_PROFILE,
    PUBLIC_REFERENCE_DIFFERENCES,
    paper_declared_gadl,
)
from scripts.evaluate_gadl_paper_declared import greedy_one_to_one_matching


# 功能：检查论文配置、默认配置不受影响及无锚点输入的稳定硬匹配。
class PaperDeclaredProfileTest(unittest.TestCase):
    def test_profile_matches_paper_declaration(self):
        self.assertEqual(PAPER_DECLARED_PROFILE["graph_protocol"], "full-graph")
        self.assertEqual(PAPER_DECLARED_PROFILE["online_edge_entries"], 16_328)
        self.assertEqual(PAPER_DECLARED_PROFILE["hidden_dim"], 256)
        self.assertEqual(PAPER_DECLARED_PROFILE["num_hidden_layers"], 6)
        self.assertEqual(PAPER_DECLARED_PROFILE["spectral_rank"], 300)
        self.assertEqual(PAPER_DECLARED_PROFILE["lr"], 1e-3)
        self.assertEqual(PAPER_DECLARED_PROFILE["reconstruction_mode"], "unweighted_dense_bce")
        self.assertEqual(PAPER_DECLARED_PROFILE["functional_map_schedule"], "constant")
        self.assertEqual(PAPER_DECLARED_PROFILE["activation_mode"], "planetalign_prelu")

    def test_profile_does_not_change_gadl_defaults(self):
        model = paper_declared_gadl(seed=7)
        self.assertEqual(model.hidden_dim, 256)
        self.assertEqual(model.num_hidden_layers, 6)
        self.assertEqual(model.reconstruction_mode, "unweighted_dense_bce")
        self.assertEqual(model.functional_map_schedule, "constant")

    def test_public_reference_differences_are_explicit(self):
        self.assertEqual(PUBLIC_REFERENCE_DIFFERENCES["online_edge_entries"]["public_reference"], 11_502)
        self.assertEqual(PUBLIC_REFERENCE_DIFFERENCES["hidden_dim"]["public_reference"], 512)
        self.assertEqual(PUBLIC_REFERENCE_DIFFERENCES["num_hidden_layers"]["public_reference"], 3)

    def test_global_greedy_matching(self):
        scores = torch.tensor([[0.90, 0.80], [0.85, 0.10], [0.20, 0.70]])
        source_to_target, target_to_source = greedy_one_to_one_matching(scores)
        self.assertEqual(source_to_target.tolist(), [0, -1, 1])
        self.assertEqual(target_to_source.tolist(), [0, 2])

    def test_global_greedy_ties_are_stable_and_anchor_independent(self):
        scores = torch.ones(2, 2)
        source_to_target, target_to_source = greedy_one_to_one_matching(scores)
        self.assertEqual(source_to_target.tolist(), [0, 1])
        self.assertEqual(target_to_source.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
