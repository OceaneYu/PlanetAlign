"""Score saved original predictions with CGP-v1; never trains or decodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PlanetAlign.metrics import (
    CGP_PROTOCOL, InvalidPredictionError, NoEvaluableRelationsError,
    UnsupportedCGPOverlapError, cross_graph_pair_scores,
)
from scripts.rescore_msf1 import prediction_maps


def rescore(gt, predictions, *, allow_overlap=False):
    records = []
    for method, pred in predictions.items():
        row = {"method": method, "num_raw_pred_units": len(pred)}
        try:
            scores = cross_graph_pair_scores(gt, pred, allow_overlap=allow_overlap)
        except NoEvaluableRelationsError as exc:
            row.update(status="no_evaluable_gt", reason=str(exc), NumGTPairs=0)
        except UnsupportedCGPOverlapError as exc:
            row.update(status="unsupported_overlap", reason=str(exc))
        except InvalidPredictionError as exc:
            row.update(status="invalid_prediction", reason=str(exc))
        else:
            row.update(status="ok", **scores)
        records.append(row)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"Refusing to overwrite existing output: {args.out}")
    for path in (args.gt, args.predictions):
        if not path.is_file():
            parser.error(f"Missing input file: {path}; no training or decoding will be performed")

    gt_bytes, pred_bytes = args.gt.read_bytes(), args.predictions.read_bytes()
    gt_payload = json.loads(gt_bytes)
    gt = gt_payload.get("entities", gt_payload)
    declared_overlap = gt_payload.get("metadata", {}).get("overlap_ratio", 0.0) > 0
    records = rescore(gt, prediction_maps(json.loads(pred_bytes)), allow_overlap=declared_overlap)
    evaluated = [row for row in records if row["status"] == "ok"]
    report = {
        "protocol": CGP_PROTOCOL,
        "benchmark_mode": "overlap_unsupported" if declared_overlap else "non_overlap",
        "source": "saved_original_predictions",
        "training_performed": False,
        "decoding_performed": False,
        "gt_file": str(args.gt.resolve()),
        "gt_sha256": hashlib.sha256(gt_bytes).hexdigest(),
        "predictions_file": str(args.predictions.resolve()),
        "predictions_sha256": hashlib.sha256(pred_bytes).hexdigest(),
        "num_evaluated": len(evaluated),
        "num_not_evaluated": len(records) - len(evaluated),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for row in records:
        if row["status"] == "ok":
            print(f"{row['method']}: CGP-P={row['CGP-P']:.9f}, CGP-R={row['CGP-R']:.9f}, "
                  f"CGP-F1={row['CGP-F1']:.9f}; TP={row['TP']}, FP={row['FP']}, FN={row['FN']}")
        else:
            print(f"{row['method']}: {row['status']}: {row['reason']}")
    return int(len(evaluated) != len(records))


if __name__ == "__main__":
    raise SystemExit(main())
