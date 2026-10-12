# 文件作用：验证乐观/平均/悲观同分排名、方向、容差和无同分兼容性。
# 核心依赖：PlanetAlign/metrics/ranking.py → pairwise_ranks()，及 PlanetAlign/metrics/hits.py、PlanetAlign/metrics/mrr.py；样例在本文件构造，不需完整基准。
# 阅读建议：先读测试类的准备数据和对应断言，可用 python -m unittest discover -s tests 运行。

import unittest

import torch

from PlanetAlign.metrics import (
    hits_ks_scores,
    mean_rank_score,
    mrr_score,
    pairwise_ranks,
)


# 功能：用可手算的分数矩阵检查排名，并交叉验证 Hits、MRR 与 MeanRank。
class TieAwareRankingTest(unittest.TestCase):
    def test_unique_maximum_is_identical_for_all_policies(self):
        similarity = torch.tensor([[0.9, 0.2], [0.1, 0.8]])
        pairs = torch.tensor([[0, 0], [1, 1]])
        for policy in ("optimistic", "average", "pessimistic"):
            self.assertTrue(torch.equal(
                pairwise_ranks(similarity, pairs, "ltr", policy),
                torch.ones(2, dtype=torch.float64),
            ))
            self.assertEqual(hits_ks_scores(
                similarity, pairs, ks=[1], mode="ltr", tie_policy=policy
            )[1], 1.0)
            self.assertEqual(mrr_score(
                similarity, pairs, mode="ltr", tie_policy=policy
            ), 1.0)
            self.assertEqual(mean_rank_score(
                similarity, pairs, mode="ltr", tie_policy=policy
            ), 1.0)

    # 功能：真实候选与一个错误候选并列第一，检查三种策略分别给出 1、1.5、2。
    def test_two_candidates_tied_first(self):
        similarity = torch.tensor([[1.0, 1.0, 0.0]])
        pairs = torch.tensor([[0, 0]])
        expected = {"optimistic": 1.0, "average": 1.5, "pessimistic": 2.0}
        for policy, rank in expected.items():
            self.assertEqual(
                pairwise_ranks(similarity, pairs, tie_policy=policy).item(), rank
            )
            self.assertAlmostEqual(mrr_score(
                similarity, pairs, mode="ltr", tie_policy=policy
            ), 1.0 / rank)

    def test_multiple_candidates_tied_from_first_to_n(self):
        similarity = torch.tensor([[1.0, 1.0, 1.0, 0.2]])
        pairs = torch.tensor([[0, 1]])
        self.assertEqual(pairwise_ranks(
            similarity, pairs, tie_policy="optimistic"
        ).item(), 1.0)
        self.assertEqual(pairwise_ranks(
            similarity, pairs, tie_policy="average"
        ).item(), 2.0)
        self.assertEqual(pairwise_ranks(
            similarity, pairs, tie_policy="pessimistic"
        ).item(), 3.0)

    def test_tie_below_first_position(self):
        similarity = torch.tensor([[1.0, 0.5, 0.5, 0.0]])
        pairs = torch.tensor([[0, 1]])
        expected = {"optimistic": 2.0, "average": 2.5, "pessimistic": 3.0}
        for policy, rank in expected.items():
            self.assertEqual(
                pairwise_ranks(similarity, pairs, tie_policy=policy).item(), rank
            )

    def test_both_directions_and_all_metrics(self):
        similarity = torch.tensor([
            [1.0, 0.2, 0.1],
            [1.0, 0.9, 0.8],
        ])
        pairs = torch.tensor([[0, 0], [1, 1]])
        ltr = pairwise_ranks(
            similarity, pairs, direction="ltr", tie_policy="average"
        )
        rtl = pairwise_ranks(
            similarity, pairs, direction="rtl", tie_policy="average"
        )
        self.assertTrue(torch.equal(ltr, torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(rtl, torch.tensor([1.5, 1.0])))
        self.assertAlmostEqual(hits_ks_scores(
            similarity, pairs, ks=[1, 2], mode="mean", tie_policy="average"
        )[1], 0.5)
        self.assertAlmostEqual(mrr_score(
            similarity, pairs, mode="mean", tie_policy="average"
        ), ((1.0 + 0.5) / 2 + (1 / 1.5 + 1.0) / 2) / 2)
        self.assertAlmostEqual(mean_rank_score(
            similarity, pairs, mode="mean", tie_policy="average"
        ), (1.5 + 1.25) / 2)

    def test_tolerance_controls_approximate_ties(self):
        similarity = torch.tensor([[1.0, 1.0 + 5e-9]], dtype=torch.float64)
        pairs = torch.tensor([[0, 0]])
        self.assertEqual(pairwise_ranks(
            similarity, pairs, tie_policy="average", atol=1e-8, rtol=0
        ).item(), 1.5)
        self.assertEqual(pairwise_ranks(
            similarity, pairs, tie_policy="average", atol=0, rtol=0
        ).item(), 2.0)

    def test_continuous_scores_are_unchanged_across_policies(self):
        similarity = torch.tensor([
            [0.91, 0.32, 0.14],
            [0.27, 0.83, 0.49],
            [0.41, 0.12, 0.72],
        ])
        pairs = torch.tensor([[0, 0], [1, 1], [2, 2]])
        ranks = [
            pairwise_ranks(similarity, pairs, tie_policy=policy)
            for policy in ("optimistic", "average", "pessimistic")
        ]
        self.assertTrue(torch.equal(ranks[0], ranks[1]))
        self.assertTrue(torch.equal(ranks[1], ranks[2]))

    def test_default_is_optimistic(self):
        similarity = torch.tensor([[1.0, 1.0]])
        pairs = torch.tensor([[0, 0]])
        self.assertEqual(
            hits_ks_scores(similarity, pairs, ks=[1], mode="ltr")[1], 1.0
        )
        self.assertEqual(mrr_score(similarity, pairs, mode="ltr"), 1.0)
        self.assertEqual(mean_rank_score(similarity, pairs, mode="ltr"), 1.0)


if __name__ == "__main__":
    unittest.main()
