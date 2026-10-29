"""Rescore saved entity predictions with MSF1-v2-K; never trains or decodes.

Accepts a raw entity map, {"entities": ...}, or {"method": entity_map, ...}.
Writes a new JSON report and refuses to overwrite any existing file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PlanetAlign.metrics import InvalidPredictionError, MSF1_PROTOCOL, macro_set_f1


def _is_entity_map(value):
    return isinstance(value, dict) and all(
        isinstance(unit, dict) and any(side in unit for side in ("src", "tgt"))
        for unit in value.values()
    )


def prediction_maps(payload):
    if isinstance(payload, dict) and "entities" in payload and _is_entity_map(payload["entities"]):
        return {"prediction": payload["entities"]}
    if _is_entity_map(payload):
        return {"prediction": payload}
    if isinstance(payload, dict) and all(_is_entity_map(v) for v in payload.values()):
        return payload
    raise ValueError("Expected an entity map, an entities wrapper, or named method entity maps")


def rescore(gt, predictions, *, allow_overlap=False):
    records = []
    for name, pred in predictions.items():
        row = {"method": name, "num_raw_pred_units": len(pred)}
        try:
            details = macro_set_f1(gt, pred, allow_overlap=allow_overlap)
        except InvalidPredictionError as exc:
            row.update(status="invalid_prediction", error=str(exc))
        else:
            k, l = details["NumEntities"], details["NumPredEntities"]
            # This is the previous bilateral max(K,L) normalization on the
            # SAME predictions, not the historical target-only MSF1 metric.
            previous = details["MSF1"] * k / max(k, l) if max(k, l) else 0.0
            row.update(status="ok", **details, previous_bilateral_max_K_L=previous,
                       delta_from_max_K_L=details["MSF1"] - previous)
        records.append(row)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-overlap", action="store_true",
                        help="explicitly use the separate overlapping-benchmark mode")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"Refusing to overwrite existing output: {args.out}")

    gt_bytes, pred_bytes = args.gt.read_bytes(), args.predictions.read_bytes()
    gt_payload = json.loads(gt_bytes)
    gt = gt_payload.get("entities", gt_payload)
    ratio = gt_payload.get("metadata", {}).get("overlap_ratio")
    if ratio is not None and (ratio > 0) != args.allow_overlap:
        parser.error("--allow-overlap must agree with GT metadata.overlap_ratio")
    rows = rescore(gt, prediction_maps(json.loads(pred_bytes)), allow_overlap=args.allow_overlap)
    report = {
        "msf1_protocol": MSF1_PROTOCOL,
        "benchmark_mode": "overlap" if args.allow_overlap else "non_overlap",
        "source": "saved_predictions",
        "training_performed": False,
        "decoding_performed": False,
        "gt_file": str(args.gt.resolve()),
        "gt_sha256": hashlib.sha256(gt_bytes).hexdigest(),
        "predictions_file": str(args.predictions.resolve()),
        "predictions_sha256": hashlib.sha256(pred_bytes).hexdigest(),
        "records": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for row in rows:
        if row["status"] == "ok":
            print(f"{row['method']}: MSF1={row['MSF1']:.9f}, "
                  f"K={row['NumEntities']:.0f}, L={row['NumPredEntities']:.0f}")
        else:
            print(f"{row['method']}: invalid_prediction: {row['error']}")
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
