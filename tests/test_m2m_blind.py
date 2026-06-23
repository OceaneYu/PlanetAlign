import unittest

import torch
from torch_geometric.data import Data

from PlanetAlign.m2m_blind import discover_groups, decode_entity_map, blind_predict, evaluate_blind


def _two_triangles_graph(offset_attr: float) -> Data:
    # Two attribute-distinct triangles: {0,1,2} and {3,4,5}.
    edge_index = torch.tensor(
        [[0, 1, 2, 0, 3, 4, 5, 3], [1, 2, 0, 2, 4, 5, 3, 5]], dtype=torch.long
    )
    x = torch.tensor(
        [[1.0, 0.0]] * 3 + [[0.0, 1.0 + offset_attr]] * 3, dtype=torch.float32
    )
    return Data(num_nodes=6, edge_index=edge_index, x=x)


class M2MBlindTest(unittest.TestCase):
    def test_discover_groups_recovers_attr_cliques(self):
        g = _two_triangles_graph(0.0)
        groups = discover_groups(g, use_attr=True, attr_tau=0.99)
        sizes = sorted(len(c) for c in groups)
        self.assertEqual(sizes, [3, 3])

    def test_decode_matches_best_target_group(self):
        # 2 src groups, 2 tgt groups; identity-ish similarity favours matching
        # group A->A, B->B.
        src_groups = [[0, 1], [2, 3]]
        tgt_groups = [[0, 1], [2, 3]]
        S = torch.tensor(
            [
                [0.9, 0.9, 0.1, 0.1],
                [0.9, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.9],
                [0.1, 0.1, 0.9, 0.9],
            ]
        )
        pred = decode_entity_map(S, src_groups, tgt_groups)
        self.assertEqual(pred["p0"]["tgt"], [0, 1])
        self.assertEqual(pred["p1"]["tgt"], [2, 3])

    def test_evaluate_blind_does_not_consult_gt_for_grouping(self):
        g_src = _two_triangles_graph(0.0)
        g_tgt = _two_triangles_graph(0.0)
        S = torch.eye(6)
        gt = {"e0": {"src": [0, 1, 2], "tgt": [0, 1, 2]}, "e1": {"src": [3, 4, 5], "tgt": [3, 4, 5]}}
        scores, pred = evaluate_blind(S, gt, g_src, g_tgt, return_predictions=True)
        # discovery is structural/attribute only; prediction recovers the cliques
        self.assertTrue(all(len(v["src"]) == 3 for v in pred.values() if v["src"]))
        self.assertGreater(scores["MSF1"], 0.99)


if __name__ == "__main__":
    unittest.main()
