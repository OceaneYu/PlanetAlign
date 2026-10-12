# 文件作用：验证跨图矩阵表示限制、单目标读出上限、传输边际与 oracle 依赖。
# 核心依赖：scripts/diagnose_m2m_failure.py 与 PlanetAlign/m2m/diagnostics.py；样例在本文件构造，不需完整基准。
# 阅读建议：先读测试类的准备数据和对应断言，可用 python -m unittest discover -s tests 运行。

"""Failure-mode assertions behind docs/m2m_why_one_to_one_fails.md.

These tests encode — as fast, deterministic checks — the three structural
reasons a one-to-one network-alignment method cannot solve many-to-many
alignment. They do not train any model; they exercise the same probe helpers
the full diagnostic (scripts/diagnose_m2m_failure.py) uses, plus the metric
diagnostic helpers in PlanetAlign.

The claims:
  A. Representation gap: the output is a cross-graph S in R^{n1 x n2}; it has no
     slot for intra-graph co-reference, which the M2M ground truth requires.
  B. One-to-one readout ceiling: a per-source-node argmax names <= |src_e|
     targets for an entity, so target-set recall is capped below 1 whenever
     |tgt_e| > |src_e|.
  C-OT. The OT marginal (mass-conservation) constraint forbids the many-to-one
     concentration that N-1 entities need, while an unconstrained similarity
     collapses uncontrollably onto hubs.
  D. Oracle protocol: the diagnostic sizes each prediction to the GT target set,
     so it intentionally uses the answer's cardinality regardless of S.
"""

import importlib.util
import unittest
from pathlib import Path

import torch

from PlanetAlign.m2m.diagnostics import oracle_grouped_topk
from PlanetAlign.metrics import many_to_many_scores

# Load the diagnostic module by path (scripts/ is not an importable package).
_DIAG_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_m2m_failure.py"
_spec = importlib.util.spec_from_file_location("diagnose_m2m_failure", _DIAG_PATH)
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


# 功能：说明跨图分数矩阵不直接表达同一图内哪些节点同属实体。
class RepresentationGapTest(unittest.TestCase):
    """A. S is cross-graph only; M2M needs intra-graph co-reference."""

    def test_similarity_is_cross_graph_block_only(self):
        # A model emits S of shape (n1, n2). There is no (n1, n1) intra-graph
        # block, so "source node 0 and source node 1 are the same entity"
        # is unrepresentable.
        n1, n2 = 4, 3
        S = torch.rand(n1, n2)
        self.assertEqual(tuple(S.shape), (n1, n2))
        # The two source nodes of an N-N entity live on the same axis (rows);
        # nothing in S relates a row to another row.
        self.assertNotEqual(S.shape[0], S.shape[1])  # generally n1 != n2

    def test_gt_flags_intra_graph_coreference(self):
        gt = {
            "e0": {"src": [0], "tgt": [0]},            # 1-1
            "e1": {"src": [1, 2], "tgt": [1]},         # N-1: intra-graph on src
            "e2": {"src": [3], "tgt": [2, 3]},         # 1-N: intra-graph on tgt
        }
        s = diag.gt_structure(gt)
        self.assertEqual(s["entities_needing_intra_graph_coreference"], 2)
        self.assertGreater(s["frac_needing_intra_graph_coreference"], 0.0)


# 功能：用一对多实体说明每行只选一个目标会漏掉其余目标。
class OneToOneCeilingTest(unittest.TestCase):
    """B. A per-source-node argmax readout caps target-set recall below 1."""

    def test_argmax_recall_ceiling_below_one_for_one_to_many(self):
        gt = {
            "e0": {"src": [0], "tgt": [0, 1, 2]},      # 1 src can name <= 1 of 3
            "e1": {"src": [1], "tgt": [3]},            # 1-1, fully recoverable
        }
        s = diag.gt_structure(gt)
        # ceiling = sum_e min(|src|,|tgt|) / sum_e |tgt| = (1 + 1) / (3 + 1)
        self.assertAlmostEqual(s["argmax_readout_microrecall_ceiling"], 0.5, places=6)
        self.assertLess(s["argmax_readout_microrecall_ceiling"], 1.0)

    def test_one_to_one_decode_misses_extra_targets(self):
        # Entity e0 needs targets {0,1}. A strict one-to-one (top_k=1) readout
        # recovers at most one of them, so MicroF1 < 1; the explicit oracle
        # diagnostic that is handed k=|tgt| can reach 1.
        gt = {"e0": {"src": [0], "tgt": [0, 1]}}
        S = torch.tensor([[0.9, 0.8, 0.1]])  # row for the single source node
        one_to_one = {"e0": {"src": [0], "tgt": [int(S[0].argmax())]}}
        oracle = oracle_grouped_topk(S, gt)
        f1_oto = many_to_many_scores(gt, one_to_one, metrics=["MicroF1"])["MicroF1"]
        f1_oracle = many_to_many_scores(gt, oracle, metrics=["MicroF1"])["MicroF1"]
        self.assertLess(f1_oto, 1.0)
        self.assertGreater(f1_oracle, f1_oto)


# 功能：比较固定行列总量与无约束矩阵对重复目标选择的影响。
class OTMarginalConstraintTest(unittest.TestCase):
    """C-OT. Mass conservation forbids many-to-one; unconstrained S collapses."""

    def test_doubly_stochastic_plan_cannot_concentrate(self):
        # A doubly-stochastic coupling spreads column mass uniformly: no target
        # absorbs many sources. Identity-like permutation => one source/target.
        n = 6
        plan = torch.eye(n) * 0.9 + torch.full((n, n), 0.1 / n)
        probe = diag.marginal_probe(plan)
        self.assertLessEqual(probe["argmax_sharing"]["max_sources_per_target"], 1)
        self.assertLess(probe["marginal"]["col_sum_cov"], 0.3)

    def test_unconstrained_similarity_collapses_onto_hub(self):
        # No marginal constraint: every source's best match is the same hub
        # column. max_sources_per_target == n1, col-sum mass is highly skewed.
        n1, n2 = 8, 5
        S = torch.full((n1, n2), 0.1)
        S[:, 2] = 0.9  # column 2 is the hub everyone points at
        probe = diag.marginal_probe(S)
        self.assertEqual(probe["argmax_sharing"]["max_sources_per_target"], n1)
        self.assertGreater(probe["marginal"]["col_sum_cov"], 0.5)


# 功能：改变真实分组或目标数检查 oracle 输出，说明它不是盲预测。
class OracleProtocolTest(unittest.TestCase):
    """D. The diagnostic deliberately uses GT source groups and target size."""

    def test_uniform_similarity_still_emits_gt_target_count(self):
        # Even with a totally uninformative (uniform) S, the diagnostic returns
        # exactly |gt tgt| targets. This proves GT cardinality is in the oracle
        # protocol; it is not a test of formal M2M capability.
        gt = {
            "e0": {"src": [0], "tgt": [1, 2, 3]},
            "e1": {"src": [1], "tgt": [0]},
        }
        S = torch.ones(2, 4)  # uniform: carries no alignment information
        pred = oracle_grouped_topk(S, gt)
        self.assertEqual(len(pred["e0"]["tgt"]), 3)
        self.assertEqual(len(pred["e1"]["tgt"]), 1)

    def test_changing_gt_source_group_changes_oracle_prediction(self):
        S = torch.tensor([[0.9, 0.1], [0.0, 1.0]])
        gt_single = {"e0": {"src": [0], "tgt": [0]}}
        gt_grouped = {"e0": {"src": [0, 1], "tgt": [0]}}

        pred_single = oracle_grouped_topk(S, gt_single)
        pred_grouped = oracle_grouped_topk(S, gt_grouped)

        self.assertNotEqual(pred_single["e0"]["tgt"], pred_grouped["e0"]["tgt"])


if __name__ == "__main__":
    unittest.main()
