import unittest

import torch
from torch_geometric.data import Data

from PlanetAlign.algorithms import JOENAM2MAlign, M2MAlign
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

    def test_joena_m2m_align_wrapper_trains_and_keeps_base_s(self):
        dataset = _tiny_dataset()
        model = JOENAM2MAlign(
            joena_hid_dim=8,
            joena_out_dim=8,
            m2m_alpha=0.9,
            m2m_tau=0.0,
            m2m_n_iter=1,
        ).to("cpu")

        s, _ = model.train(
            dataset,
            gids=[0, 1],
            use_attr=False,
            total_epochs=1,
            save_log=False,
            verbose=False,
        )

        self.assertEqual(tuple(s.shape), (4, 4))
        self.assertEqual(tuple(model.base_S.shape), (4, 4))
        self.assertEqual(tuple(model.refined_raw_S.shape), (4, 4))
        self.assertTrue(torch.isfinite(s).all())
        self.assertIn("joena_time_s", model.timing_)
        self.assertIn("m2m_refine_time_s", model.timing_)

    def test_wrapper_preserves_base_topk_when_enabled(self):
        model = JOENAM2MAlign()
        base = torch.tensor([[0.9, 0.8, 0.1], [0.2, 0.8, 0.7]])
        refined = torch.tensor([[0.95, 0.1, 0.85], [0.3, 0.75, 0.7]])

        safe = model._preserve_base_topk(base, refined, k=2)

        self.assertEqual(set(torch.topk(safe[0], k=2).indices.tolist()), set(torch.topk(base[0], k=2).indices.tolist()))
        self.assertEqual(set(torch.topk(safe[1], k=2).indices.tolist()), set(torch.topk(refined[1], k=2).indices.tolist()))
        self.assertEqual(model.preserved_rows_, 1)
        self.assertTrue(torch.allclose(safe[0], base[0]))
        self.assertTrue(torch.allclose(safe[1], refined[1]))

    def test_invalid_preserve_base_topk_raises(self):
        with self.assertRaises(ValueError):
            JOENAM2MAlign(preserve_base_topk=-1)


if __name__ == "__main__":
    unittest.main()
