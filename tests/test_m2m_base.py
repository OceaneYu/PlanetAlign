# 文件作用：验证矩阵温度处理、锚点选择与算法结果类型声明。
# 核心依赖：PlanetAlign/m2m/base.py → sharpen()/arbitrate_sharpen()/base_configs()；样例在本文件构造，不需完整基准。
# 阅读建议：先读测试类的准备数据和对应断言，可用 python -m unittest discover -s tests 运行。

"""Unit tests for PlanetAlign.m2m.base — base-aligner axis + sharpen arbitration."""

import unittest

import torch
from torch_geometric.data import Data

from PlanetAlign.m2m.base import SHARPEN_TEMPS, arbitrate_sharpen, base_configs, sharpen


def _graph(num_nodes, edges):
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).T
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)
    return Data(edge_index=ei, num_nodes=num_nodes)


# 功能：检查不变换选项、每行和为 1 与低温更突出的性质。
class SharpenTest(unittest.TestCase):
    def test_none_is_identity(self):
        S = torch.rand(5, 6)
        self.assertTrue(torch.equal(sharpen(S, None), S))

    def test_temp_is_row_stochastic(self):
        S = torch.rand(4, 7)
        out = sharpen(S, 0.1)
        self.assertEqual(out.shape, S.shape)
        self.assertTrue(torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-5))

    def test_lower_temp_is_sharper(self):
        # A single dominant entry per row -> lower T concentrates more mass on it.
        S = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.8, 0.2]])
        hot = sharpen(S, 0.01).max(dim=1).values
        warm = sharpen(S, 1.0).max(dim=1).values
        self.assertTrue(torch.all(hot > warm))


# 功能：用块状矩阵与训练锚点检查候选选择及同分时优先原矩阵。
class ArbitrateSharpenTest(unittest.TestCase):
    def _block_problem(self):
        # Two groups of 2 source nodes each mapping to 2 target nodes; a clean
        # block-diagonal coupling is a near-permutation that the anchors confirm.
        S = torch.tensor([
            [0.9, 0.8, 0.0, 0.0],
            [0.85, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.8],
            [0.0, 0.0, 0.85, 0.9],
        ])
        g = _graph(4, [(0, 1), (2, 3)])
        anchors = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
        return S, g, anchors

    def test_returns_valid_choice(self):
        S, g, anchors = self._block_problem()
        S_best, temp, table = arbitrate_sharpen(S, g, g, anchors, use_attr=False)
        self.assertEqual(S_best.shape, S.shape)
        self.assertIn(temp, SHARPEN_TEMPS)
        self.assertEqual(len(table), len(SHARPEN_TEMPS))
        # Every row is (temp, corrected, raw); corrected in [-1, 1].
        for t, corrected, raw in table:
            self.assertTrue(-1.0 <= corrected <= 1.0 or corrected != corrected)

    def test_coupling_prefers_raw_on_ties(self):
        # When raw already achieves the best anchor agreement, the Occam tie-break
        # keeps raw (temp=None) rather than an equivalent sharpened variant.
        S, g, anchors = self._block_problem()
        _, temp, table = arbitrate_sharpen(S, g, g, anchors, use_attr=False)
        best = max(c for _, c, _ in table if c == c)
        raw_corrected = next(c for t, c, _ in table if t is None)
        self.assertAlmostEqual(raw_corrected, best, places=6)
        self.assertIsNone(temp)


# 功能：检查 PARROT 属于传输型、嵌入算法声明正确的矩阵来源类型。
class BaseConfigTest(unittest.TestCase):
    def test_parrot_is_a_coupling_base(self):
        cfgs = base_configs()
        self.assertIn("PARROT", cfgs)
        self.assertEqual(cfgs["PARROT"]["kind"], "coupling")
        self.assertEqual(cfgs["PARROT"]["mode"], "self.S")

    def test_embedding_bases_declared(self):
        cfgs = base_configs()
        self.assertEqual(cfgs["BRIGHT"]["kind"], "embedding")
        self.assertEqual(cfgs["BRIGHT"]["mode"], "embs-cos")


if __name__ == "__main__":
    unittest.main()
