"""Unit tests for PlanetAlign.m2m_quotient — one test per design claim."""

import unittest

import torch
from torch_geometric.data import Data

from PlanetAlign.m2m_blind import discover_groups_by_profile
from PlanetAlign.m2m_quotient import (
    global_candidate_pairs,
    greedy_match,
    hungarian_match,
    merge_average_linkage,
    null_threshold,
    otsu_threshold,
    partition_contrast,
    quotient_decode,
    quotient_scores,
)


def _graph(num_nodes, edges):
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).T
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)
    return Data(edge_index=ei, num_nodes=num_nodes)


class OtsuThresholdTest(unittest.TestCase):
    def test_separates_bimodal_sample(self):
        vals = torch.cat([torch.full((50,), 0.02), torch.full((50,), 0.45)])
        t = otsu_threshold(vals + 0.01 * torch.randn(100))
        self.assertGreater(t, 0.1)
        self.assertLess(t, 0.45)

    def test_degenerate_sample_falls_back(self):
        self.assertEqual(otsu_threshold(torch.full((100,), 0.3)), 0.1)
        self.assertEqual(otsu_threshold(torch.zeros(3)), 0.1)


class AntiChainingTest(unittest.TestCase):
    """A~B and B~C pass the edge gate, but A and C are dissimilar."""

    def setUp(self):
        # Profiles: A=(1,0), B=(1,1)/sqrt2, C=(0,1). cos(A,B)=cos(B,C)=0.707, cos(A,C)=0.
        self.profiles = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        self.u = torch.tensor([0, 1])
        self.v = torch.tensor([1, 2])
        p = torch.nn.functional.normalize(self.profiles, p=2, dim=1)
        self.sims = (p[self.u] * p[self.v]).sum(dim=1)  # both ~0.707

    def test_plain_union_find_chains(self):
        graph = _graph(3, [(0, 1), (1, 2)])
        groups = discover_groups_by_profile(self.profiles, graph, tau=0.5)
        self.assertEqual(len(groups), 1)  # chained into one blob

    def test_average_linkage_blocks_chain(self):
        groups = merge_average_linkage(3, self.u, self.v, self.sims, self.profiles, tau=0.5)
        sizes = sorted(len(g) for g in groups)
        # First (strongest) merge happens; the second is blocked because the
        # merged centroid no longer agrees with the remaining node at tau=0.5.
        self.assertEqual(sizes, [1, 2])


class HungarianUniquenessTest(unittest.TestCase):
    """Two source groups both prefer target 0; greedy stacks, Hungarian doesn't."""

    def setUp(self):
        self.T = torch.tensor([[0.9, 0.5], [0.8, 0.7]])

    def test_greedy_duplicates_target(self):
        m = greedy_match(self.T)
        self.assertEqual(m[0], 0)
        self.assertEqual(m[1], 0)  # duplicated claim

    def test_hungarian_is_exclusive_and_higher_total(self):
        m = hungarian_match(self.T)
        self.assertEqual(len(set(m.values())), len(m))  # no duplicates
        self.assertEqual(m, {0: 0, 1: 1})  # total 1.6 > greedy-unique 1.4


class PartitionContrastTest(unittest.TestCase):
    """The blind arbiter prefers the partition matching the profile structure."""

    def test_correct_partition_scores_higher(self):
        # Nodes 0,1 share a profile; node 2 is orthogonal. Edges (0,1), (1,2).
        profiles = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        u, v = torch.tensor([0, 1]), torch.tensor([1, 2])
        good = partition_contrast(profiles, u, v, [[0, 1], [2]])   # within=1, cross=0
        blob = partition_contrast(profiles, u, v, [[0, 1, 2]])     # all within, mean=0.5
        singles = partition_contrast(profiles, u, v, [[0], [1], [2]])
        self.assertGreater(good, blob)
        self.assertEqual(singles, float("-inf"))


class NullThresholdTest(unittest.TestCase):
    """Random-pair null calibration and its degeneracy guard."""

    def test_threshold_scales_with_dimension(self):
        torch.manual_seed(0)
        lo_dim = null_threshold(torch.randn(300, 8), num_null=5000)
        hi_dim = null_threshold(torch.randn(300, 512), num_null=5000)
        # cosine fluctuations shrink like 1/sqrt(d): higher dim => lower threshold
        self.assertGreater(lo_dim, hi_dim)
        self.assertGreater(hi_dim, 0.0)
        self.assertLess(hi_dim, 0.3)

    def test_duplicate_row_degeneracy_returns_none(self):
        # All rows identical => null cosines all 1.0 => unusable as a threshold.
        profiles = torch.ones(50, 16)
        self.assertIsNone(null_threshold(profiles, num_null=2000))


class GlobalCandidateTest(unittest.TestCase):
    """Attr-bucket x profile gates propose non-adjacent siblings; hubs excluded."""

    def test_disconnected_siblings_are_proposed_and_merged(self):
        # Nodes 0 and 2 are siblings (same attrs, same profile) but NOT adjacent.
        profiles = torch.tensor([
            [0.9, 0.1, 0.0],   # 0: sibling A
            [0.0, 0.1, 0.9],   # 1: unrelated, different attrs
            [0.9, 0.1, 0.0],   # 2: sibling A'
            [0.0, 0.9, 0.1],   # 3: unrelated
        ])
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        u, v, s = global_candidate_pairs(profiles, x, tau_global=0.98, hub_load_limit=16)
        self.assertEqual((int(u[0]), int(v[0])), (0, 2))
        # And the per-edge-tau merge accepts them.
        groups = merge_average_linkage(4, u, v, s, profiles, torch.full((u.numel(),), 0.98))
        self.assertIn([0, 2], [sorted(g) for g in groups])

    def test_hub_rows_are_excluded(self):
        # 10 rows all argmax on column 0 (hub) with identical attrs: with a
        # hub_load_limit below 10, no candidate pairs may be proposed.
        profiles = torch.zeros(10, 4)
        profiles[:, 0] = 1.0
        x = torch.ones(10, 3)
        u, v, s = global_candidate_pairs(profiles, x, hub_load_limit=5)
        self.assertEqual(u.numel(), 0)


class EntityAnchorAgreementTest(unittest.TestCase):
    """Aligner-level arbitration criterion on a final entity map."""

    def test_counts_anchor_pairs_landing_in_same_entity(self):
        from PlanetAlign.m2m_quotient import entity_anchor_agreement
        pred = {"p0": {"src": [0, 1], "tgt": [5]}, "p1": {"src": [2], "tgt": [6, 7]}}
        anchors = torch.tensor([[0, 5], [2, 6], [1, 9]])  # 2 of 3 agree
        corr, raw = entity_anchor_agreement(pred, anchors, n2=10)
        self.assertAlmostEqual(raw, 2 / 3, places=9)
        self.assertLess(corr, raw)  # chance term subtracted


class AnchorArbiterTest(unittest.TestCase):
    """When profiles carry no group signal, anchors + Occam pick the attribute
    partition (the Cora failure mode)."""

    def test_uninformative_profiles_select_attr_evidence(self):
        # G1: true group {0,1} shares attrs verbatim; 2 and 3 are distinct.
        g1 = _graph(4, [(0, 1), (1, 2), (2, 3)])
        g1.x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        g2 = _graph(3, [(0, 1), (1, 2)])
        g2.x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        S = torch.full((4, 3), 0.5)          # profiles uninformative everywhere
        # >= 20 anchors required for the arbiter (scarce anchors are noise —
        # measured on pems08); repeat the two consistent anchors.
        anchors = torch.tensor([[2, 1], [3, 2]]).repeat(10, 1)
        _, info = quotient_decode(S, g1, g2, anchors=anchors, max_iters=1)
        self.assertEqual(info["evidence"][0][0], "attr")


class EndToEndToyTest(unittest.TestCase):
    """Perfect recovery on a toy M2M instance with block-structured S."""

    def setUp(self):
        # G1: entity X = {0,1,2} (triangle), entity Y = {3}.
        # G2: entity X = {0}, entity Y = {1,2} (edge).
        self.g1 = _graph(4, [(0, 1), (1, 2), (0, 2), (2, 3)])
        self.g2 = _graph(3, [(1, 2), (0, 1)])
        S = torch.full((4, 3), 0.02)
        S[0:3, 0] = 0.9      # X-rows point at target 0
        S[3, 1] = 0.8        # Y-row points at targets 1,2
        S[3, 2] = 0.8
        self.S = S

    def test_quotient_decode_recovers_entities(self):
        pred, info = quotient_decode(self.S, self.g1, self.g2, max_iters=2)
        by_src = {tuple(sorted(v["src"])): sorted(v["tgt"]) for v in pred.values()}
        self.assertEqual(by_src[(0, 1, 2)], [0])
        self.assertEqual(by_src[(3,)], [1, 2])
        self.assertGreaterEqual(info["iters"], 1)

    def test_quotient_scores_shape(self):
        T = quotient_scores(self.S, [[0, 1, 2], [3]], [[0], [1, 2]])
        self.assertEqual(tuple(T.shape), (2, 2))
        self.assertGreater(float(T[0, 0]), float(T[0, 1]))
        self.assertGreater(float(T[1, 1]), float(T[1, 0]))


if __name__ == "__main__":
    unittest.main()


class GroupUniformMarginalTest(unittest.TestCase):
    """Group-level Sinkhorn marginals: one unit per group, blended with uniform."""

    def test_mass_sums_to_one_and_groups_get_equal_mass(self):
        from PlanetAlign.m2m_contrastive import _group_uniform_marginal
        groups = [[0, 1, 2], [3], [4, 5]]
        m = _group_uniform_marginal(groups, n=6, rho=1.0, dtype=torch.float32)
        self.assertAlmostEqual(float(m.sum()), 1.0, places=6)
        for g in groups:  # each group carries 1/3 regardless of size
            self.assertAlmostEqual(float(m[torch.tensor(g)].sum()), 1 / 3, places=6)

    def test_rho_zero_is_uniform(self):
        from PlanetAlign.m2m_contrastive import _group_uniform_marginal
        m = _group_uniform_marginal([[0, 1], [2]], n=3, rho=0.0, dtype=torch.float32)
        self.assertTrue(torch.allclose(m, torch.full((3,), 1 / 3)))
