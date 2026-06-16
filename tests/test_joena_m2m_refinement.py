import unittest

import torch
from torch_geometric.data import Data

from PlanetAlign.algorithms import M2MAlign
from PlanetAlign.data import BaseData


def _tiny_dataset() -> BaseData:
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 2],
            [1, 0, 3, 2, 2, 0],
        ],
        dtype=torch.long,
    )
    graph1 = Data(name="src", num_nodes=4, edge_index=edge_index)
    graph2 = Data(name="tgt", num_nodes=4, edge_index=edge_index.clone())
    anchors = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.long)
    return BaseData(
        graphs=[graph1, graph2],
        anchor_links=anchors,
        name="tiny_init_s",
        train_ratio=0.5,
        seed=0,
    )


class JOENAM2MRefinementTest(unittest.TestCase):
    def test_init_s_changes_m2m_align_output(self):
        dataset = _tiny_dataset()
        init_a = torch.eye(4, dtype=torch.float32)
        init_b = torch.flip(torch.eye(4, dtype=torch.float32), dims=[1])

        model_a = M2MAlign(alpha=0.5, tau=0.0, n_iter=1).to("cpu")
        model_b = M2MAlign(alpha=0.5, tau=0.0, n_iter=1).to("cpu")

        s_a, _ = model_a.train(dataset, gids=[0, 1], use_attr=False, save_log=False, verbose=False, init_S=init_a)
        s_b, _ = model_b.train(dataset, gids=[0, 1], use_attr=False, save_log=False, verbose=False, init_S=init_b)

        self.assertEqual(tuple(s_a.shape), (4, 4))
        self.assertEqual(tuple(s_b.shape), (4, 4))
        self.assertFalse(torch.allclose(s_a, s_b))

    def test_bad_init_s_shape_raises(self):
        dataset = _tiny_dataset()
        model = M2MAlign(alpha=0.5, tau=0.0, n_iter=1).to("cpu")

        with self.assertRaises(AssertionError):
            model.train(
                dataset,
                gids=[0, 1],
                use_attr=False,
                save_log=False,
                verbose=False,
                init_S=torch.zeros(3, 4),
            )


if __name__ == "__main__":
    unittest.main()
