# 文件作用：验证重复锚点和重叠成员都不能把训练节点带入真实评估实体。
# 核心依赖：PlanetAlign/utils/many2many_builder.py → build_many_to_many_benchmark()；样例在本文件构造，不需完整基准。
# 阅读建议：先读测试类的准备数据和对应断言，可用 python -m unittest discover -s tests 运行。

"""Regression test for the flickr-lastfm train/test leakage class.

Original datasets may carry duplicate anchor rows (flickr-lastfm holds the
pair (4227, 11939) twice) or nodes shared across anchor pairs. The pair-level
train/test split can then place copies of the same node in both splits; the
builder must never turn such a test pair into an evaluation entity while the
train copy serves as supervision.
"""

import unittest
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from PlanetAlign.utils.many2many_builder import build_many_to_many_benchmark


def _stub_dataset(train_pairs, test_pairs, n1=12, n2=12):
    edges = [(i, i + 1) for i in range(n1 - 1)]
    ei = torch.tensor(edges, dtype=torch.long).T
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    g1 = Data(edge_index=ei.clone(), num_nodes=n1)
    g2 = Data(edge_index=ei.clone(), num_nodes=n2)
    return SimpleNamespace(
        pyg_graphs=[g1, g2],
        train_data=torch.tensor(train_pairs, dtype=torch.long),
        test_data=torch.tensor(test_pairs, dtype=torch.long),
        name="stub",
    )


# 功能：用手工 train_data/test_data 重现节点冲突并检查构建器排除结果。
class BuilderLeakageTest(unittest.TestCase):
    # 功能：同时制造完整锚点重复和单侧节点重复，检查所有实体与训练节点隔离。
    def test_duplicated_anchor_pair_cannot_leak_into_entities(self):
        # Pair (5, 6) is duplicated across the split: one copy is training
        # supervision, one copy lands in test. Node 3 additionally collides on
        # the src side only ((3, 9) in train vs (3, 8) in test).
        ds = _stub_dataset(
            train_pairs=[[5, 6], [3, 9]],
            test_pairs=[[5, 6], [3, 8], [0, 0], [1, 1], [2, 2], [7, 7]],
        )
        bench = build_many_to_many_benchmark(
            ds, gids=(0, 1), overlap_ratio=0.0, seed=0)

        ta_src = set(bench.train_anchors[:, 0].tolist())
        ta_tgt = set(bench.train_anchors[:, 1].tolist())
        for eid, e in bench.entities.items():
            self.assertFalse(set(e["src"]) & ta_src, f"{eid} leaks src")
            self.assertFalse(set(e["tgt"]) & ta_tgt, f"{eid} leaks tgt")
        self.assertEqual(bench.metadata["dropped_train_collisions"], 2)
        self.assertEqual(bench.metadata["num_entities"], 4)

    # 功能：开启重叠机制后再检查实体成员，防止后处理重新引入训练节点。
    def test_overlap_insertion_cannot_reintroduce_anchor_nodes(self):
        ds = _stub_dataset(
            train_pairs=[[5, 6]],
            test_pairs=[[5, 6], [0, 0], [1, 1], [2, 2], [7, 7], [8, 8]],
        )
        bench = build_many_to_many_benchmark(
            ds, gids=(0, 1), overlap_ratio=0.3, seed=0)
        ta_src = set(bench.train_anchors[:, 0].tolist())
        ta_tgt = set(bench.train_anchors[:, 1].tolist())
        for eid, e in bench.entities.items():
            self.assertFalse(set(e["src"]) & ta_src)
            self.assertFalse(set(e["tgt"]) & ta_tgt)
