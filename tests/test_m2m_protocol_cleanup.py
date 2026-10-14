# 文件作用：验证默认盲解码、显式 oracle 诊断前缀及 TGAE 矩阵接入。
# 核心依赖：scripts/run_m2m_experiments.py → evaluate_algorithm() 与 PlanetAlign/m2m/quotient.py；样例在本文件构造，不需完整基准。
# 阅读建议：先读测试类的准备数据和对应断言，可用 python -m unittest discover -s tests 运行。

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch_geometric.data import Data

import PlanetAlign
import PlanetAlign.metrics as metrics_module
from PlanetAlign.algorithms import TGAE
from PlanetAlign.data import BaseData
from PlanetAlign.m2m.diagnostics import (
    ORACLE_GROUPED_TOPK_PROTOCOL,
    oracle_grouped_topk,
    oracle_grouped_topk_target_metrics,
)
from PlanetAlign.m2m import evaluate_predictions
from PlanetAlign.m2m.quotient import quotient_decode


_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_m2m_experiments.py"
_spec = importlib.util.spec_from_file_location("run_m2m_experiments", _RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


# 功能：返回固定 S 的算法替身，让协议测试不需要执行真实训练。
class _DummyAlgorithm:
    def __init__(self, similarity):
        self._similarity = similarity
        self.S = None
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def train(self, **kwargs):
        return self._similarity, None

    def test(self, **kwargs):
        return {"Hits@1": 0.5, "Hits@10": 1.0, "MRR": 0.75}


# 功能：提供两张小图和训练锚点的最小数据替身。
class _DummyDataset:
    def __init__(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.pyg_graphs = [
            Data(num_nodes=2, edge_index=edge_index, x=None),
            Data(num_nodes=2, edge_index=edge_index, x=None),
        ]
        self.train_data = torch.tensor([[0, 0]], dtype=torch.long)


# 功能：用 mock 分别替换预测与评分，检查真实答案仅交给评分函数，以及输出字段保持一致。
class RunnerProtocolTest(unittest.TestCase):
    def setUp(self):
        self.similarity = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        self.dataset = _DummyDataset()
        self.gt = {
            "e0": {"src": [0], "tgt": [0]},
            "e1": {"src": [1], "tgt": [1]},
        }
        self.config = {
            "factory": lambda: _DummyAlgorithm(self.similarity),
            "train_kwargs": {"use_attr": False},
            "mode": "S",
        }
        self.blind_scores = {
            "Macro Set-F1": 0.2,
            "Micro Set-F1": 0.3,
            "AUIS": 0.4,
            "AUCS": 0.5,
        }

    # 功能：将固定替身数据与配置传给 evaluate_algorithm()，供两个协议测试共用。
    def _run(self, include_oracle_diagnostic=False):
        return runner.evaluate_algorithm(
            dataset_name="toy_m2m",
            dataset=self.dataset,
            gt_entities=self.gt,
            algo_name="Dummy",
            config=self.config,
            profile="quick",
            seed=42,
            verbose=False,
            auto_disable_attr=True,
            include_oracle_diagnostic=include_oracle_diagnostic,
        )

    # 功能：把 oracle 函数设为调用即报错，验证默认路径只执行盲解码。
    def test_default_runner_uses_blind_quotient_and_never_calls_oracle(self):
        with mock.patch.object(
            runner,
            "quotient_decode",
            return_value=({}, {"iters": 1}),
        ) as decode, mock.patch.object(
            runner, "evaluate_predictions", return_value=self.blind_scores,
        ) as evaluate, mock.patch.object(
            runner,
            "oracle_grouped_topk",
            side_effect=AssertionError("oracle path must be opt-in"),
        ) as oracle:
            record = self._run()

        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["m2m_protocol"], "blind_quotient_decode")
        decode.assert_called_once()
        self.assertEqual(len(decode.call_args.args), 3)
        torch.testing.assert_close(decode.call_args.args[0], self.similarity)
        self.assertIs(decode.call_args.args[1], self.dataset.pyg_graphs[0])
        self.assertIs(decode.call_args.args[2], self.dataset.pyg_graphs[1])
        self.assertEqual(set(decode.call_args.kwargs), {"use_attr", "anchors"})
        evaluate.assert_called_once_with(self.gt, {}, metrics=runner.M2M_METRICS)
        oracle.assert_not_called()
        self.assertEqual(record["Blind-Micro Set-F1"], 0.3)
        for metric_name in runner.M2M_METRICS:
            self.assertNotIn(metric_name, record)
            self.assertNotIn(metric_name, runner.CSV_FIELDS)
            self.assertIn(f"Blind-{metric_name}", runner.CSV_FIELDS)
        self.assertNotIn("oracle_protocol", record)

    def test_oracle_flag_adds_only_prefixed_target_diagnostics(self):
        with mock.patch.object(
            runner,
            "quotient_decode",
            return_value=({}, {"iters": 1}),
        ), mock.patch.object(runner, "evaluate_predictions", return_value=self.blind_scores):
            record = self._run(include_oracle_diagnostic=True)

        self.assertEqual(record["oracle_protocol"], ORACLE_GROUPED_TOPK_PROTOCOL)
        oracle_keys = [key for key in record if key.startswith("OracleSrc+GT-K-")]
        self.assertEqual(
            set(oracle_keys),
            {
                "OracleSrc+GT-K-Target-MicroF1",
                "OracleSrc+GT-K-Target-MSF1",
                "OracleSrc+GT-K-Target-Recall@GT-K",
            },
        )
        self.assertFalse(any("AUIS" in key or "AUCS" in key for key in oracle_keys))


# 功能：检查旧接口移除后的导入，以及 TGAE 的 S 可以直接进入 QuotientDecode。
class PublicApiAndImportTest(unittest.TestCase):
    def test_removed_public_apis_do_not_break_package_or_tgae_imports(self):
        self.assertIsNotNone(PlanetAlign)
        self.assertFalse(hasattr(metrics_module, "similarity_to_pred_entities"))
        self.assertFalse(hasattr(metrics_module, "pred_entities_from_similarity"))
        self.assertTrue(callable(TGAE))
        self.assertFalse(hasattr(TGAE, "predict_many_to_many"))

    def test_oracle_metrics_are_target_only(self):
        gt = {"e0": {"src": [0], "tgt": [0, 1]}}
        pred = oracle_grouped_topk(torch.ones(1, 3), gt)
        scores = oracle_grouped_topk_target_metrics(gt, pred)
        self.assertEqual(
            set(scores),
            {
                "OracleSrc+GT-K-Target-MicroF1",
                "OracleSrc+GT-K-Target-MSF1",
                "OracleSrc+GT-K-Target-Recall@GT-K",
            },
        )

    def test_tgae_similarity_runs_through_blind_quotient(self):
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
            dtype=torch.long,
        )
        graphs = [
            Data(name="src", num_nodes=4, edge_index=edge_index, x=None),
            Data(name="tgt", num_nodes=4, edge_index=edge_index, x=None),
        ]
        dataset = BaseData(
            graphs=graphs,
            anchor_links=torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.long),
            name="tgae_protocol_smoke",
            train_ratio=0.5,
            seed=0,
        )
        model = TGAE(num_hidden_layers=1, hidden_dim=4, output_dim=2)

        similarity, _ = model.train(
            dataset=dataset,
            gids=[0, 1],
            use_attr=False,
            total_epochs=1,
            save_log=False,
            verbose=False,
        )
        gt = {
            f"e{node}": {"src": [node], "tgt": [node]}
            for node in range(4)
        }
        pred, _ = quotient_decode(
            similarity,
            graphs[0],
            graphs[1],
            use_attr=False,
            anchors=dataset.train_data,
        )
        scores = evaluate_predictions(gt, pred)

        self.assertEqual(tuple(similarity.shape), (4, 4))
        self.assertTrue(torch.isfinite(similarity).all())
        self.assertIn("Micro Set-F1", scores)


if __name__ == "__main__":
    unittest.main()
