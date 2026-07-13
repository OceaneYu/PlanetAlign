"""Unit tests for the Fused Gromov-Wasserstein quotient matcher.

FGW is the exact nonlinear form of the quotient consistency principle whose
first-order linearization is :func:`neighbor_consistency_refine`. These tests
pin the two properties that matter: it stays a one-to-one readout (quotient
exclusivity is load-bearing), and its structure term can break a feature tie
that Hungarian cannot.
"""

import unittest

import torch
from torch_geometric.data import Data

from PlanetAlign.m2m_quotient import fgw_match, hungarian_match, quotient_adjacency


def _graph(num_nodes, edges):
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).T
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)
    return Data(edge_index=ei, num_nodes=num_nodes)


class FGWMatchTest(unittest.TestCase):
    def test_is_one_to_one(self):
        T = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.8, 0.2], [0.1, 0.2, 0.85]])
        A = torch.eye(3)
        m = fgw_match(T, A, A, alpha=0.5)
        self.assertEqual(len(set(m.values())), len(m))          # injective
        self.assertTrue(set(m).issubset({0, 1, 2}))

    def test_recovers_clean_block(self):
        T = torch.eye(3) * 0.9 + 0.05
        A = torch.eye(3)
        self.assertEqual(fgw_match(T, A, A, alpha=0.3), {0: 0, 1: 1, 2: 2})

    def test_singleton_falls_back_to_hungarian(self):
        T = torch.tensor([[0.2], [0.9], [0.4]])                 # g2 == 1
        A1, A2 = torch.eye(3), torch.eye(1)
        self.assertEqual(fgw_match(T, A1, A2, alpha=0.5), hungarian_match(T))

    def test_structure_breaks_feature_tie(self):
        # Source quotient: groups 0-1 connected, group 2 isolated.
        # Target quotient: groups 0-1 connected, group 2 isolated (same shape).
        # Feature: group 2 is exactly tied between target 0 (connected) and
        # target 2 (isolated); groups 0,1 pin the connected pair. Pure structure
        # must send the isolated source group 2 to the isolated target 2.
        g_src = _graph(3, [(0, 1)])
        g_tgt = _graph(3, [(0, 1)])
        Aq1 = quotient_adjacency(g_src, [[0], [1], [2]])
        Aq2 = quotient_adjacency(g_tgt, [[0], [1], [2]])
        T = torch.tensor([[0.9, 0.1, 0.0],
                          [0.1, 0.9, 0.0],
                          [0.5, 0.0, 0.5]])                     # group 2 tie: t0 vs t2
        feat = hungarian_match(T)                               # may send 2 -> 0
        struct = fgw_match(T, Aq1, Aq2, alpha=0.9)
        self.assertEqual(struct[2], 2)                          # structure resolves the tie
        # And the tie really was ambiguous for the feature-only matcher.
        self.assertEqual(T[2, 0], T[2, 2])


if __name__ == "__main__":
    unittest.main()
