import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.rescore_cgp import main, rescore


class RescoreCGPTest(unittest.TestCase):
    def setUp(self):
        self.gt = {"A": {"src": [1], "tgt": [5]}}

    def test_valid_and_invalid_saved_predictions_are_separate_records(self):
        rows = rescore(self.gt, {"invalid": {"P": self.gt["A"], "Q": self.gt["A"]},
                                "valid": self.gt})
        self.assertEqual(rows[0]["status"], "invalid_prediction")
        self.assertNotIn("CGP-F1", rows[0])
        self.assertEqual(rows[1]["status"], "ok")
        self.assertEqual(rows[1]["CGP-F1"], 1)

    def test_no_true_relations_are_reported_without_performance_scores(self):
        rows = rescore({"A": {"src": [1], "tgt": []}}, {"model": self.gt})
        self.assertEqual(rows[0]["status"], "no_evaluable_gt")
        self.assertIn("无可评估真实关系", rows[0]["reason"])
        self.assertEqual(rows[0]["NumGTPairs"], 0)
        self.assertFalse(any(k.startswith("CGP-") for k in rows[0]))

    def test_overlapping_gt_and_declared_mode_are_unsupported(self):
        overlapping_gt = {**self.gt, "B": {"src": [1], "tgt": [6]}}
        for gt, allow in ((overlapping_gt, False), (self.gt, True)):
            with self.subTest(gt=gt, allow=allow):
                row = rescore(gt, {"model": {}}, allow_overlap=allow)[0]
                self.assertEqual(row["status"], "unsupported_overlap")
                self.assertNotIn("CGP-F1", row)

    def test_cli_scores_original_predictions_and_never_trains_decodes_or_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            gt, pred, out = [Path(directory) / f"{name}.json" for name in ("gt", "pred", "out")]
            gt.write_text(json.dumps({"entities": self.gt, "metadata": {"overlap_ratio": 0}}))
            pred.write_text(json.dumps({"model": {"P": {"src": [1, 99], "tgt": [5, 200]}}}))
            before = gt.read_bytes(), pred.read_bytes()
            argv = ["--gt", str(gt), "--predictions", str(pred), "--out", str(out)]
            with mock.patch("PlanetAlign.m2m.base.train_base_S", side_effect=AssertionError), \
                    mock.patch("PlanetAlign.m2m.quotient.quotient_decode", side_effect=AssertionError), \
                    mock.patch("PlanetAlign.m2m.core.align_prediction_to_ground_truth", side_effect=AssertionError), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 0)
            report = json.loads(out.read_text())
            self.assertEqual(report["protocol"], "CGP-v1")
            self.assertEqual(report["num_evaluated"], 1)
            self.assertFalse(report["training_performed"])
            self.assertFalse(report["decoding_performed"])
            self.assertEqual(report["records"][0]["CGP-F1"], 0.5)
            self.assertEqual(report["records"][0]["NumPredPairs"], 3)
            self.assertEqual((gt.read_bytes(), pred.read_bytes()), before)
            output_before = out.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(argv)
            self.assertEqual(out.read_bytes(), output_before)

    def test_cli_excludes_empty_gt_from_evaluated_count(self):
        with tempfile.TemporaryDirectory() as directory:
            gt, pred, out = [Path(directory) / f"{name}.json" for name in ("gt", "pred", "out")]
            gt.write_text(json.dumps({"entities": {}}))
            pred.write_text(json.dumps({"valid_empty": {}, "other": self.gt}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--gt", str(gt), "--predictions", str(pred), "--out", str(out)]), 1)
            report = json.loads(out.read_text())
            self.assertEqual(report["num_evaluated"], 0)
            self.assertEqual(report["num_not_evaluated"], 2)
            self.assertTrue(all(r["status"] == "no_evaluable_gt" for r in report["records"]))
            self.assertFalse(any("CGP-F1" in r for r in report["records"]))

    def test_cli_reports_missing_prediction_without_creating_results(self):
        with tempfile.TemporaryDirectory() as directory:
            gt, pred, out = [Path(directory) / f"{name}.json" for name in ("gt", "missing", "out")]
            gt.write_text(json.dumps({"entities": self.gt}))
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit):
                main(["--gt", str(gt), "--predictions", str(pred), "--out", str(out)])
            self.assertIn("Missing input file", errors.getvalue())
            self.assertFalse(out.exists())

    def test_cli_honors_declared_overlap_even_if_selected_gt_has_no_shared_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            gt, pred, out = [Path(directory) / f"{name}.json" for name in ("gt", "pred", "out")]
            gt.write_text(json.dumps({"entities": self.gt, "metadata": {"overlap_ratio": 0.05}}))
            pred.write_text(json.dumps({"entities": self.gt}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--gt", str(gt), "--predictions", str(pred), "--out", str(out)]), 1)
            report = json.loads(out.read_text())
            self.assertEqual(report["benchmark_mode"], "overlap_unsupported")
            self.assertEqual(report["records"][0]["status"], "unsupported_overlap")
            self.assertEqual(report["num_evaluated"], 0)


if __name__ == "__main__":
    unittest.main()
