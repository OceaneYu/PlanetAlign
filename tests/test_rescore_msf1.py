import json
import tempfile
import unittest
from pathlib import Path

from scripts.rescore_msf1 import main, prediction_maps, rescore


class RescoreMSF1Test(unittest.TestCase):
    def setUp(self):
        self.a = {"src": [1], "tgt": [5]}
        self.b = {"src": [2], "tgt": [6]}

    def test_saved_prediction_formats_and_empty_prediction(self):
        pred = {"P": self.a}
        for payload in (pred, {"entities": pred, "dataset": "toy"}):
            self.assertEqual(prediction_maps(payload), {"prediction": pred})
        self.assertEqual(prediction_maps({"model1": pred, "model2": {}}),
                         {"model1": pred, "model2": {}})
        self.assertEqual(rescore({"A": self.a}, prediction_maps({}))[0]["MSF1"], 0)

    def test_batch_reports_invalid_prediction_without_score_and_continues(self):
        rows = rescore({"A": self.a}, {
            "duplicate": {"P": self.a, "Q": self.a}, "valid": {"P": self.a},
        })
        self.assertEqual(rows[0]["status"], "invalid_prediction")
        self.assertNotIn("MSF1", rows[0])
        self.assertEqual(rows[1]["MSF1"], 1)

    def test_comparison_uses_same_predictions_with_previous_denominator(self):
        row = rescore({"A": {"src": [1, 2], "tgt": [5, 6]}},
                      {"split": {"P": self.a, "Q": self.b}})[0]
        self.assertAlmostEqual(row["MSF1"], 2 / 3)
        self.assertAlmostEqual(row["previous_bilateral_max_K_L"], 1 / 3)

    def test_cli_preserves_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            gt, pred, out = [Path(directory) / f"{name}.json" for name in ("gt", "pred", "out")]
            gt.write_text(json.dumps({"entities": {"A": self.a}, "metadata": {"overlap_ratio": 0}}))
            pred.write_text(json.dumps({"P": self.a, "Q": self.a}))
            before = (gt.read_bytes(), pred.read_bytes())
            argv = ["--gt", str(gt), "--predictions", str(pred), "--out", str(out)]
            self.assertEqual(main(argv), 1)
            report = json.loads(out.read_text())
            self.assertEqual(report["records"][0]["status"], "invalid_prediction")
            self.assertFalse(report["training_performed"])
            self.assertEqual((gt.read_bytes(), pred.read_bytes()), before)
            with self.assertRaises(SystemExit):
                main(argv)

    def test_cli_cannot_disable_non_overlap_validation_for_known_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            gt, pred, out = [Path(directory) / f"{name}.json" for name in ("gt", "pred", "out")]
            gt.write_text(json.dumps({"entities": {"A": self.a}, "metadata": {"overlap_ratio": 0}}))
            pred.write_text(json.dumps({"P": self.a, "Q": self.a}))
            with self.assertRaises(SystemExit):
                main(["--gt", str(gt), "--predictions", str(pred), "--out", str(out), "--allow-overlap"])
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
