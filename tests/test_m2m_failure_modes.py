"""Failure-mode assertions behind docs/m2m_why_one_to_one_fails.md.

These tests encode — as fast, deterministic checks — the three structural
reasons a one-to-one network-alignment method cannot solve many-to-many
alignment. They do not train any model; they exercise the same probe helpers
the full diagnostic (scripts/diagnose_m2m_failure.py) uses, plus the metric
adapters in PlanetAlign.

The claims:
  A. Representation gap: the output is a cross-graph S in R^{n1 x n2}; it has no
     slot for intra-graph co-reference, which the M2M ground truth requires.
  B. One-to-one readout ceiling: a per-source-node argmax names <= |src_e|
     targets for an entity, so target-set recall is capped below 1 whenever
     |tgt_e| > |src_e|.
  C-OT. The OT marginal (mass-conservation) constraint forbids the many-to-one
     concentration that N-1 entities need, while an unconstrained similarity
     collapses uncontrollably onto hubs.
  D. Evaluation leak: the default adapter sizes each prediction to the GT target
     set, so it leaks the answer's cardinality regardless of S.
"""

import importlib.util
import unittest
from pathlib import Path

import torch

from PlanetAlign.metrics import (
    many_to_many_scores,
    similarity_to_pred_entities,
)

# Load the diagnostic module by path (scripts/ is not an importable package).
_DIAG_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_m2m_failure.py"
_spec = importlib.util.spec_from_file_location("diagnose_m2m_failure", _DIAG_PATH)
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


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
        # recovers at most one of them, so MicroF1 < 1; the leaking adapter that
        # is handed k=|tgt| can reach 1.
        gt = {"e0": {"src": [0], "tgt": [0, 1]}}
        S = torch.tensor([[0.9, 0.8, 0.1]])  # row for the single source node
        one_to_one = similarity_to_pred_entities(S, gt, top_k=1)
        leaked = similarity_to_pred_entities(S, gt)  # top_k defaults to |tgt|=2
        f1_oto = many_to_many_scores(gt, one_to_one, metrics=["MicroF1"])["MicroF1"]
        f1_leak = many_to_many_scores(gt, leaked, metrics=["MicroF1"])["MicroF1"]
        self.assertLess(f1_oto, 1.0)
        self.assertGreater(f1_leak, f1_oto)


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


class EvaluationLeakTest(unittest.TestCase):
    """D. The default adapter leaks the GT target-set size."""

    def test_default_adapter_emits_exactly_gt_target_count(self):
        # Even with a totally uninformative (uniform) S, the adapter returns
        # exactly |gt tgt| targets per entity, i.e. it leaks the cardinality.
        gt = {
            "e0": {"src": [0], "tgt": [1, 2, 3]},
            "e1": {"src": [1], "tgt": [0]},
        }
        S = torch.ones(2, 4)  # uniform: carries no alignment information
        pred = similarity_to_pred_entities(S, gt)
        self.assertEqual(len(pred["e0"]["tgt"]), 3)
        self.assertEqual(len(pred["e1"]["tgt"]), 1)


if __name__ == "__main__":
    unittest.main()
