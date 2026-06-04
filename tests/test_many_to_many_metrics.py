import unittest

import torch

from PlanetAlign.metrics import (
    acs_score,
    many_to_many_scores,
    similarity_to_pred_entities,
)


class ManyToManyMetricsTest(unittest.TestCase):
    def setUp(self):
        self.gt_entities = {
            "e0": {"src": [0], "tgt": [1, 2]},
            "e1": {"src": [1], "tgt": [0]},
        }
        self.pred_entities = {
            "e0": {"src": [0], "tgt": [1, 2]},
            "e1": {"src": [1], "tgt": [0]},
        }

    def test_perfect_prediction_scores_are_one(self):
        scores = many_to_many_scores(self.gt_entities, self.pred_entities)

        self.assertEqual(
            set(scores),
            {"ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"},
        )
        for value in scores.values():
            self.assertAlmostEqual(value, 1.0)

    def test_missing_entity_is_treated_as_empty_prediction(self):
        pred_missing = {"e0": {"src": [0], "tgt": [1]}}

        scores = many_to_many_scores(self.gt_entities, pred_missing)

        self.assertLess(scores["ACS"], 1.0)
        self.assertLess(scores["MSF1"], 1.0)
        self.assertLess(scores["MicroF1"], 1.0)

    def test_public_imports_are_available(self):
        self.assertTrue(callable(acs_score))
        self.assertTrue(callable(many_to_many_scores))

    def test_similarity_to_pred_entities_uses_target_set_size(self):
        similarity = torch.tensor([[0.1, 0.9, 0.8], [0.7, 0.2, 0.1]])

        pred_entities = similarity_to_pred_entities(similarity, self.gt_entities)

        self.assertEqual(pred_entities["e0"]["src"], [0])
        self.assertEqual(set(pred_entities["e0"]["tgt"]), {1, 2})
        self.assertEqual(pred_entities["e1"]["src"], [1])
        self.assertEqual(pred_entities["e1"]["tgt"], [0])


if __name__ == "__main__":
    unittest.main()
