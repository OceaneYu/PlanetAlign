import unittest

import torch
from torch_geometric.data import Data

from PlanetAlign.m2m_decode import JOENAGroupDecode
from PlanetAlign.data import BaseData


def _tiny_dataset() -> BaseData:
    # Two 4-node graphs with attributes; anchors on the diagonal.
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]], dtype=torch.long)
    x = torch.eye(4, dtype=torch.float32)
    g1 = Data(name="src", num_nodes=4, edge_index=edge_index, x=x)
    g2 = Data(name="tgt", num_nodes=4, edge_index=edge_index.clone(), x=x.clone())
    anchors = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.long)
    return BaseData(graphs=[g1, g2], anchor_links=anchors, name="tiny_gd", train_ratio=0.5, seed=0)


class JOENAGroupDecodeTest(unittest.TestCase):
    def setUp(self):
        self.dataset = _tiny_dataset()
        self.model = JOENAGroupDecode(row_tau=0.1, col_tau=0.1, hid_dim=8, out_dim=8).to("cpu")
        self.model.train(self.dataset, gids=[0, 1], use_attr=True,
                         total_epochs=2, save_log=False, verbose=False)

    def test_train_sets_similarity_and_graphs(self):
        self.assertEqual(tuple(self.model.S.shape), (4, 4))
        self.assertTrue(torch.isfinite(self.model.S).all())

    def test_predict_entities_is_blind_and_well_formed(self):
        pred = self.model.predict_entities()
        self.assertGreater(len(pred), 0)
        all_src, all_tgt = [], []
        for item in pred.values():
            self.assertIn("src", item)
            self.assertIn("tgt", item)
            all_src += item["src"]
            all_tgt += item["tgt"]
        # every source node is covered exactly once (groups partition the nodes)
        self.assertEqual(sorted(all_src), [0, 1, 2, 3])
        self.assertTrue(set(all_tgt).issubset({0, 1, 2, 3}))

    def test_test_blind_returns_metrics(self):
        gt = {"e0": {"src": [0], "tgt": [0]}, "e1": {"src": [1], "tgt": [1]}}
        scores = self.model.test_blind(gt, metrics=["MSF1", "MicroF1"])
        self.assertIn("MSF1", scores)
        self.assertIn("MicroF1", scores)


if __name__ == "__main__":
    unittest.main()
