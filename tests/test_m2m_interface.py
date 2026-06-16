import tempfile
import unittest
from pathlib import Path

import torch

from PlanetAlign.m2m import (
    ManyToManyBaseline,
    align_prediction_to_ground_truth,
    evaluate_predictions,
    evaluate_similarity,
    load_entity_map,
    normalize_entity_map,
    save_entity_map,
)


class DummyManyToManyBaseline(ManyToManyBaseline):
    def train(self, dataset, gids, *args, **kwargs):
        self.S = torch.tensor([[0.1, 0.9, 0.8], [0.7, 0.2, 0.1]])
        return self.S, None


class ManyToManyInterfaceTest(unittest.TestCase):
    def setUp(self):
        self.gt_entities = {
            "e0": {"src": [0], "tgt": [1, 2]},
            "e1": {"src": [1], "tgt": [0]},
        }

    def test_normalize_entity_map_deduplicates_and_casts_ints(self):
        raw = {"0": {"src": ["1", 1, 2], "tgt": [3, "3"]}}

        normalized = normalize_entity_map(raw)

        self.assertEqual(normalized, {"0": {"src": [1, 2], "tgt": [3]}})

    def test_save_and_load_wrapped_prediction_json(self):
        pred = {"e0": {"src": [0], "tgt": [1, 2]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.json"

            save_entity_map(pred, path, dataset_name="toy_m2m", metadata={"method": "dummy"})
            loaded = load_entity_map(path)

        self.assertEqual(loaded, {"e0": {"src": [0], "tgt": [1, 2]}})

    def test_missing_prediction_entity_is_scored_as_empty(self):
        pred = {"e0": {"src": [0], "tgt": [1, 2]}}

        scores = evaluate_predictions(self.gt_entities, pred)

        self.assertLess(scores["MSF1"], 1.0)
        self.assertLess(scores["MicroF1"], 1.0)

    def test_group_id_permutation_is_matched_by_adapter(self):
        gt = {
            "e0": {"src": [0], "tgt": [0, 1]},
            "e1": {"src": [1, 2], "tgt": [2]},
            "e2": {"src": [3], "tgt": [3]},
        }
        pred = {
            "cluster_b": {"src": [1, 2], "tgt": [2]},
            "cluster_c": {"src": [3], "tgt": [3]},
            "cluster_a": {"src": [0], "tgt": [0, 1]},
        }

        aligned = align_prediction_to_ground_truth(gt, pred)
        scores = evaluate_predictions(gt, pred)

        self.assertEqual(set(aligned["e0"]["tgt"]), {0, 1})
        for value in scores.values():
            self.assertAlmostEqual(value, 1.0)

    def test_evaluate_similarity_can_return_predictions(self):
        similarity = torch.tensor([[0.1, 0.9, 0.8], [0.7, 0.2, 0.1]])

        scores, pred = evaluate_similarity(
            similarity,
            self.gt_entities,
            return_predictions=True,
        )

        self.assertAlmostEqual(scores["MSF1"], 1.0)
        self.assertEqual(set(pred["e0"]["tgt"]), {1, 2})
        self.assertEqual(pred["e1"]["tgt"], [0])

    def test_baseline_mixin_predicts_from_similarity(self):
        model = DummyManyToManyBaseline()
        model.S = torch.tensor([[0.1, 0.9, 0.8], [0.7, 0.2, 0.1]])

        scores = model.test_many_to_many(self.gt_entities)

        self.assertAlmostEqual(scores["MicroF1"], 1.0)


if __name__ == "__main__":
    unittest.main()
