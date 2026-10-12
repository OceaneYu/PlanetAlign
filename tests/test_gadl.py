# 文件作用：验证双通道滤波、共享编码、函数映射、谱维度和无监督接口。
# 核心依赖：PlanetAlign/algorithms/gadl/main.py、PlanetAlign/algorithms/gadl/model.py；样例在本文件构造，不需完整基准。
# 阅读建议：先读测试类的准备数据和对应断言，可用 python -m unittest discover -s tests 运行。

import os
import tempfile
import unittest

import torch
from torch_geometric.data import Data

from PlanetAlign.algorithms import GADL
from PlanetAlign.algorithms.gadl.model import (
    DualPassEncoder,
    GADLModel,
    build_dual_pass_filters,
    edge_index_to_undirected_binary_adjacency,
    gadl_objective,
    normalized_laplacian_eigenbasis,
    resolve_effective_rank,
)
from PlanetAlign.data import BaseData


def _cycle_edges(num_nodes=4):
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    return torch.tensor(edges, dtype=torch.long).T


def _graph(num_nodes=4, feature_dim=3, features=True):
    x = None
    if features:
        values = torch.arange(num_nodes * feature_dim, dtype=torch.float32)
        x = values.reshape(num_nodes, feature_dim) / max(1, values.numel())
    return Data(x=x, edge_index=_cycle_edges(num_nodes), num_nodes=num_nodes)


def _dataset(feature_dims=(3, 3), train_ratio=0.0, anchors=None):
    graphs = [_graph(feature_dim=feature_dims[0]), _graph(feature_dim=feature_dims[1])]
    if anchors is None:
        anchors = torch.arange(4, dtype=torch.long).repeat(2, 1).T
    return BaseData(
        graphs=graphs,
        anchor_links=anchors,
        name="gadl_toy",
        train_ratio=train_ratio,
        seed=7,
    )


def _small_model(**kwargs):
    defaults = dict(
        hidden_dim=4,
        num_hidden_layers=1,
        spectral_rank=2,
        lr=1e-3,
        seed=13,
    )
    defaults.update(kwargs)
    return GADL(**defaults)


# 功能：检查低/高通滤波器的有限值、对称性与相加为单位矩阵。
class DualPassFilterTest(unittest.TestCase):
    def test_filters_are_symmetric_finite_and_sum_to_identity(self):
        # Duplicates, both directions, a self-loop, and an isolated node are
        # intentional inputs to the graph adapter.
        edge_index = torch.tensor(
            [[0, 1, 0, 1, 2, 2], [1, 0, 1, 0, 2, 2]], dtype=torch.long
        )
        low, high = build_dual_pass_filters(edge_index, num_nodes=4)
        low_dense, high_dense = low.to_dense(), high.to_dense()

        self.assertEqual(low_dense.shape, (4, 4))
        self.assertEqual(high_dense.shape, (4, 4))
        self.assertTrue(torch.allclose(low_dense, low_dense.T))
        self.assertTrue(torch.allclose(high_dense, high_dense.T))
        self.assertTrue(torch.isfinite(low_dense).all())
        self.assertTrue(torch.isfinite(high_dense).all())
        self.assertTrue(torch.allclose(low_dense + high_dense, torch.eye(4), atol=1e-6))

        adjacency = edge_index_to_undirected_binary_adjacency(edge_index, 4).to_dense()
        self.assertEqual(adjacency[0, 1].item(), 1.0)
        self.assertEqual(adjacency[2, 2].item(), 0.0)
        self.assertEqual(adjacency[3].sum().item(), 0.0)

    def test_non_finite_edge_index_is_rejected(self):
        edge_index = torch.tensor([[0.0, float("nan")], [1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            edge_index_to_undirected_binary_adjacency(edge_index, 2)


# 功能：检查共享编码器、独立分支、映射参数与完整目标的有限梯度。
class EncoderAndMapTest(unittest.TestCase):
    def setUp(self):
        self.edge_index = _cycle_edges()
        self.low, self.high = build_dual_pass_filters(self.edge_index, 4)
        self.x = _graph().x

    def test_encoder_shape_and_independent_branches(self):
        encoder = DualPassEncoder(3, 5, 2)
        output = encoder(self.x, self.low, self.high)
        self.assertEqual(output.shape, (4, 10))

        low_parameters = {id(p) for p in encoder.low_branch.parameters()}
        high_parameters = {id(p) for p in encoder.high_branch.parameters()}
        self.assertTrue(low_parameters)
        self.assertTrue(high_parameters)
        self.assertTrue(low_parameters.isdisjoint(high_parameters))

    def test_pair_encoding_uses_one_shared_encoder(self):
        model = GADLModel(3, 4, 1, spectral_rank=2)
        z1, z2 = model.encode_pair(
            self.x, self.low, self.high,
            self.x, self.low, self.high,
        )
        self.assertIsInstance(model.encoder, DualPassEncoder)
        self.assertFalse(hasattr(model, "encoder2"))
        self.assertTrue(torch.equal(z1, z2))

    def test_functional_maps_are_unbiased_learnable_parameters(self):
        model = GADLModel(3, 4, 1, spectral_rank=2)
        named = dict(model.named_parameters())
        self.assertEqual(model.C12.shape, (2, 2))
        self.assertEqual(model.C21.shape, (2, 2))
        self.assertTrue(model.C12.requires_grad)
        self.assertTrue(model.C21.requires_grad)
        self.assertIs(named["C12"], model.C12)
        self.assertIs(named["C21"], model.C21)
        self.assertFalse(hasattr(model, "C12_bias"))
        self.assertFalse(hasattr(model, "C21_bias"))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        optimized_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertIn(id(model.C12), optimized_parameters)
        self.assertIn(id(model.C21), optimized_parameters)

    def test_model_parameters_follow_requested_dtype(self):
        model = GADLModel(3, 4, 1, spectral_rank=2).to(dtype=torch.float64)
        low, high = build_dual_pass_filters(
            self.edge_index, 4, dtype=torch.float64
        )
        output = model.encoder(self.x.to(torch.float64), low, high)
        self.assertEqual(model.C12.dtype, torch.float64)
        self.assertEqual(model.C21.dtype, torch.float64)
        self.assertEqual(output.dtype, torch.float64)

    def test_identity_maps_zero_all_functional_regularizers(self):
        model = GADLModel(3, 4, 1, spectral_rank=2)
        z = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )
        phi = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float32
        )
        eigenvalues = torch.tensor([0.0, 1.0])
        losses = model.functional_map_losses(
            z, z, phi, phi, eigenvalues, eigenvalues,
            descriptor_weight=1.0,
            commutativity_weight=1.0,
        )
        for value in losses.values():
            self.assertAlmostEqual(value.item(), 0.0, places=6)

    def test_complete_objective_backpropagates_finite_gradients(self):
        model = GADLModel(3, 4, 1, spectral_rank=2)
        z1, z2 = model.encode_pair(
            self.x, self.low, self.high,
            self.x.flip(0), self.low, self.high,
        )
        eigenvalues, phi = normalized_laplacian_eigenbasis(
            self.edge_index, 4, spectral_rank=2
        )
        adjacency = edge_index_to_undirected_binary_adjacency(self.edge_index, 4)
        losses = gadl_objective(
            model, z1, z2, adjacency, adjacency,
            phi, phi, eigenvalues, eigenvalues,
            descriptor_weight=1e-3,
            commutativity_weight=1.0,
            functional_map_weight=1.0,
            bijectivity_weight=0.1,
            orthogonality_weight=0.1,
        )
        losses["total"].backward()

        self.assertTrue(torch.isfinite(losses["total"]))
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

        c12_before = model.C12.detach().clone()
        c21_before = model.C21.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        optimizer.step()
        self.assertFalse(torch.equal(c12_before, model.C12.detach()))
        self.assertFalse(torch.equal(c21_before, model.C21.detach()))


# 功能：检查两图共享谱维度与小图维度缩减。
class SpectralBasisTest(unittest.TestCase):
    def test_rank_is_shared_and_safely_reduced(self):
        self.assertEqual(resolve_effective_rank(300, 4, 3), 2)
        eigenvalues, eigenvectors = normalized_laplacian_eigenbasis(
            _cycle_edges(3), 3, spectral_rank=300
        )
        self.assertEqual(eigenvalues.shape, (2,))
        self.assertEqual(eigenvectors.shape, (3, 2))
        self.assertTrue(torch.all(eigenvalues[1:] >= eigenvalues[:-1]))

    def test_two_node_graph_accepts_requested_rank_at_node_count(self):
        edges = torch.tensor([[0], [1]], dtype=torch.long)
        eigenvalues, eigenvectors = normalized_laplacian_eigenbasis(
            edges, 2, spectral_rank=2
        )
        self.assertEqual(eigenvalues.shape, (1,))
        self.assertEqual(eigenvectors.shape, (2, 1))


# 功能：用小图训练检查矩阵、属性要求与不依赖参考目录的公共接口。
class GADLInterfaceTest(unittest.TestCase):
    def test_public_training_produces_finite_similarity_and_metrics(self):
        dataset = _dataset(train_ratio=0.0)
        model = _small_model().to("cpu")
        similarity, logger = model.train(
            dataset, gids=[0, 1], total_epochs=1,
            save_log=False, verbose=False,
        )

        self.assertIs(similarity, model.S)
        self.assertEqual(similarity.shape, (4, 4))
        self.assertTrue(torch.isfinite(similarity).all())
        self.assertTrue(torch.all(similarity <= 1.0 + 1e-6))
        self.assertTrue(torch.all(similarity >= -1.0 - 1e-6))
        self.assertTrue(torch.allclose(similarity.diag(), torch.ones(4), atol=1e-5))
        self.assertEqual(len(logger.logs), 1)

        scores = model.test(dataset, gids=[0, 1], metrics=["Hits@1", "MRR"])
        self.assertEqual(set(scores), {"Hits@1", "MRR"})
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in scores.values()))

    # 功能：让训练锚点访问触发错误，确认 GADL 的优化过程不读取 train_data。
    def test_training_does_not_access_train_data(self):
        class AnchorAccessTrap:
            def __len__(self):
                raise AssertionError("GADL accessed train_data")

        dataset = _dataset()
        dataset.train_data = AnchorAccessTrap()
        dataset.test_data = AnchorAccessTrap()
        _small_model().train(
            dataset, gids=[0, 1], total_epochs=1,
            save_log=False, verbose=False,
        )

    def test_effective_rank_controls_all_functional_map_dimensions(self):
        dataset = _dataset()
        model = _small_model(spectral_rank=300)
        model.train(
            dataset, gids=[0, 1], total_epochs=1,
            save_log=False, verbose=False,
        )
        self.assertEqual(model.effective_rank, 3)
        self.assertEqual(model.model.C12.shape, (3, 3))
        self.assertEqual(model.model.C21.shape, (3, 3))
        for spectrum in model.spectra.values():
            self.assertEqual(spectrum["eigenvalues"].shape, (3,))
            self.assertEqual(spectrum["eigenvectors"].shape, (4, 3))

    # 功能：固定图、属性与种子，只改锚点，确认训练矩阵不变。
    def test_anchor_changes_do_not_change_fixed_seed_training(self):
        anchors1 = torch.arange(4, dtype=torch.long).repeat(2, 1).T
        anchors2 = torch.tensor([[0, 1], [1, 3], [2, 0], [3, 2]])
        dataset1 = _dataset(train_ratio=0.5, anchors=anchors1)
        dataset2 = _dataset(train_ratio=0.5, anchors=anchors2)
        model1, model2 = _small_model(), _small_model()
        s1, _ = model1.train(
            dataset1, [0, 1], total_epochs=1, save_log=False, verbose=False
        )
        s2, _ = model2.train(
            dataset2, [0, 1], total_epochs=1, save_log=False, verbose=False
        )
        self.assertTrue(torch.equal(s1, s2))

    def test_missing_attributes_and_mismatched_dimensions_are_rejected(self):
        missing = BaseData(
            graphs=[_graph(features=False), _graph()],
            anchor_links=torch.arange(4).repeat(2, 1).T,
            name="missing_attr",
        )
        with self.assertRaisesRegex(ValueError, "Attributes are not available"):
            _small_model().train(
                missing, [0, 1], total_epochs=1, save_log=False, verbose=False
            )

        mismatched = _dataset(feature_dims=(2, 3))
        with self.assertRaisesRegex(ValueError, "same node attribute dimension"):
            _small_model().train(
                mismatched, [0, 1], total_epochs=1, save_log=False, verbose=False
            )

    def test_pairwise_and_use_attr_contracts_are_explicit(self):
        dataset = _dataset()
        with self.assertRaisesRegex(ValueError, "exactly two"):
            _small_model().train(
                dataset, [0], total_epochs=1, save_log=False, verbose=False
            )
        with self.assertRaisesRegex(ValueError, "use_attr=True"):
            _small_model().train(
                dataset, [0, 1], use_attr=False, total_epochs=1,
                save_log=False, verbose=False,
            )

    def test_training_works_without_reference_directory_or_repo_cwd(self):
        dataset = _dataset()
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                similarity, _ = _small_model().train(
                    dataset, [0, 1], total_epochs=1,
                    save_log=False, verbose=False,
                )
            finally:
                os.chdir(original_cwd)
        self.assertEqual(similarity.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
