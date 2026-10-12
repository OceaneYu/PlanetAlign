# 文件作用：验证原生节点对、两图方向、锚点无关性与稠密规模限制。
# 核心依赖：PlanetAlign/algorithms/alpine/main.py → Alpine.train()/test()；样例在本文件构造，不需完整基准。
# 阅读建议：先读测试类的准备数据和对应断言，可用 python -m unittest discover -s tests 运行。

import unittest

import torch
from torch_geometric.data import Data

from PlanetAlign.algorithms import Alpine
from PlanetAlign.algorithms.alpine import AlpineResourceLimitError
from PlanetAlign.data import BaseData


def _undirected_edge_index(edges):
    directed = [edge for src, dst in edges for edge in ((src, dst), (dst, src))]
    return torch.tensor(directed, dtype=torch.long).T.contiguous()


def _toy_dataset(anchor_links=None):
    small = Data(
        name="small",
        num_nodes=4,
        edge_index=_undirected_edge_index(
            [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        ),
    )
    large = Data(
        name="large",
        num_nodes=8,
        edge_index=_undirected_edge_index(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (0, 2),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
            ]
        ),
    )
    if anchor_links is None:
        anchor_links = torch.tensor(
            [[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.long
        )
    return BaseData(
        [large, small], anchor_links, "alpine-toy", train_ratio=0.5, seed=0
    )


# 功能：用小图检查硬匹配接口，不把原生节点对误当成排名矩阵。
class AlpineTest(unittest.TestCase):
    def test_public_import_native_pairs_and_reversed_orientation(self):
        dataset = _toy_dataset()
        model = Alpine()
        pairs, logger = model.train(
            dataset, gids=[0, 1], use_attr=False, save_log=False, verbose=False
        )

        self.assertEqual(model.query_gid, 1)
        self.assertEqual(model.target_gid, 0)
        self.assertIsNone(model.S)
        self.assertEqual(pairs, model.mapping_pairs)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(len({left for left, _ in pairs}), 4)
        self.assertEqual(len({right for _, right in pairs}), 4)
        self.assertEqual(model.transport_plan.shape, (4, 8))
        self.assertEqual(model.frobenius_squared, 0.0)
        self.assertEqual(len(logger.get_logs()), 1)

        metrics = model.test(dataset, gids=[0, 1])
        self.assertEqual(metrics, {"Accuracy": 1.0, "Hits@1": 1.0})
        with self.assertRaises(ValueError):
            model.test(dataset, gids=[0, 1], metrics=["MRR"])

    # 功能：在固定图上改变训练/测试锚点，确认预测节点对保持一致。
    def test_predictions_do_not_depend_on_train_or_test_anchors(self):
        original = _toy_dataset()
        alternate = _toy_dataset(
            torch.tensor([[0, 3], [1, 2], [2, 1], [3, 0]], dtype=torch.long)
        )
        first, _ = Alpine().train(original, [0, 1], save_log=False, verbose=False)
        second, _ = Alpine().train(alternate, [0, 1], save_log=False, verbose=False)
        self.assertEqual(first, second)

    def test_resource_guard_is_explicit_and_overridable(self):
        dataset = _toy_dataset()
        with self.assertRaisesRegex(AlpineResourceLimitError, "SKIPPED_RESOURCE"):
            Alpine(max_dense_work_entries=1).train(
                dataset, [0, 1], save_log=False, verbose=False
            )


if __name__ == "__main__":
    unittest.main()
