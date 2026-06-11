"""Batch-build many-to-many benchmarks from PlanetAlign one-to-one datasets.

Examples
--------
Build every ``data/*.pt`` dataset without fuzzy group overlap:

    python scripts/build_m2m_benchmarks.py --overlap-ratio 0

Build every dataset with 5% fuzzy group overlap:

    python scripts/build_m2m_benchmarks.py --output-root data/m2m_overlap --overlap-ratio 0.05

Build selected datasets:

    python scripts/build_m2m_benchmarks.py --datasets douban cora dbp15k_zh-en --overlap-ratio 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

# PlanetAlign.utils imports visual helpers, which may import Matplotlib. Keep
# the CLI quiet on machines where the default Matplotlib cache is read-only.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

from PlanetAlign.data import Dataset
from PlanetAlign.metrics import many_to_many_scores
from PlanetAlign.utils import DEFAULT_SPLIT_RATIOS, build_many_to_many_benchmark


def _parse_split_ratios(raw: str) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(
                "split ratios must look like 1-1=0.7,1-many=0.1,many-1=0.1,many-many=0.1"
            )
        key, value = chunk.split("=", 1)
        ratios[key.strip()] = float(value)

    expected = set(DEFAULT_SPLIT_RATIOS)
    if set(ratios) != expected:
        missing = ", ".join(sorted(expected - set(ratios)))
        extra = ", ".join(sorted(set(ratios) - expected))
        pieces = []
        if missing:
            pieces.append(f"missing: {missing}")
        if extra:
            pieces.append(f"unknown: {extra}")
        raise argparse.ArgumentTypeError("; ".join(pieces))

    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"split ratios must sum to 1.0, got {total:.6f}")
    return ratios


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise argparse.ArgumentTypeError("dtype must be float32 or float64")


def _find_dataset_file(input_root: Path, item: str) -> Path:
    candidate = Path(item)
    if candidate.suffix == ".pt" or candidate.parent != Path("."):
        path = candidate if candidate.is_absolute() else input_root / candidate
        if path.exists():
            return path
        raise FileNotFoundError(f"dataset file not found: {path}")

    exact = input_root / f"{item}.pt"
    if exact.exists():
        return exact

    lowered = item.lower()
    matches = [p for p in input_root.glob("*.pt") if p.stem.lower() == lowered]
    if len(matches) == 1:
        return matches[0]
    if matches:
        joined = ", ".join(str(p) for p in matches)
        raise FileNotFoundError(f"ambiguous dataset name {item!r}: {joined}")
    raise FileNotFoundError(f"dataset {item!r} not found under {input_root}")


def _iter_dataset_files(input_root: Path, requested: Optional[List[str]]) -> Iterable[Path]:
    if requested:
        seen = set()
        for item in requested:
            path = _find_dataset_file(input_root, item)
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                yield path
        return

    for path in sorted(input_root.glob("*.pt")):
        yield path


def _load_dataset(path: Path, train_ratio: float, seed: int, dtype: torch.dtype) -> Dataset:
    return Dataset(root=path.parent, name=path.stem, train_ratio=train_ratio, seed=seed, dtype=dtype)


def _format_scores(scores: Dict[str, float]) -> str:
    keys = ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]
    return ", ".join(f"{key}={scores[key]:.4f}" for key in keys if key in scores)


def build_one(path: Path, args: argparse.Namespace) -> Dict[str, object]:
    source_name = path.stem
    output_name = args.name_pattern.format(name=source_name)
    pt_path = args.output_root / f"{output_name}.pt"
    json_path = args.output_root / f"{output_name}_gt_many2many.json"

    record: Dict[str, object] = {
        "source_path": str(path),
        "source_dataset": source_name,
        "output_name": output_name,
        "pt_path": str(pt_path),
        "json_path": str(json_path),
        "status": "pending",
    }

    if not args.overwrite and (pt_path.exists() or json_path.exists()):
        record["status"] = "skipped"
        record["reason"] = "output exists; pass --overwrite to rebuild"
        print(f"[skip] {source_name}: output exists")
        return record

    started = time.perf_counter()
    print(f"[build] {source_name} -> {output_name}")

    dataset = _load_dataset(path, train_ratio=args.train_ratio, seed=args.seed, dtype=args.dtype)
    m2m = build_many_to_many_benchmark(
        dataset=dataset,
        gids=tuple(args.gids),
        split_ratios=args.split_ratios,
        max_expansion=args.max_expansion,
        internal_density=args.internal_density,
        overlap_ratio=args.overlap_ratio,
        anchor_source=args.anchor_source,
        train_anchor_source=args.train_anchor_source,
        name=output_name,
        seed=args.seed,
    )
    saved_pt, saved_json = m2m.save(args.output_root)

    record.update(
        {
            "status": "ok",
            "elapsed_s": round(time.perf_counter() - started, 3),
            "pt_path": str(saved_pt),
            "json_path": str(saved_json),
            "metadata": m2m.metadata,
        }
    )

    print(
        "       "
        f"entities={m2m.metadata['num_entities']}, "
        f"types={m2m.metadata['entity_type_counts']}, "
        f"overlap_insertions={m2m.metadata['overlap_insertions']}, "
        f"nodes={m2m.metadata['orig_num_nodes']}->{m2m.metadata['new_num_nodes']}"
    )

    if args.check:
        # Re-open through the standard Dataset class to catch schema mistakes.
        _ = Dataset(root=args.output_root, name=output_name, train_ratio=args.train_ratio, seed=args.seed, dtype=args.dtype)
        gt = m2m.entities
        perfect = many_to_many_scores(gt, {eid: dict(entity) for eid, entity in gt.items()})
        empty = many_to_many_scores(gt, {eid: {"src": [], "tgt": []} for eid in gt})
        record["perfect_scores"] = perfect
        record["empty_scores"] = empty
        print(f"       perfect: {_format_scores(perfect)}")
        print(f"       empty  : {_format_scores(empty)}")

    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one-to-one PlanetAlign .pt datasets into many-to-many benchmarks."
    )
    parser.add_argument("--input-root", type=Path, default=Path("data"), help="Directory containing source .pt files.")
    parser.add_argument("--output-root", type=Path, default=Path("data/m2m"), help="Directory for generated files.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset stems or .pt paths to convert. Omit to convert every .pt directly under --input-root.",
    )
    parser.add_argument(
        "--name-pattern",
        default="{name}_m2m",
        help="Output dataset name pattern. Use {name} for the source stem.",
    )
    parser.add_argument("--gids", type=int, nargs=2, default=[0, 1], help="Graph ids used as source and target.")
    parser.add_argument("--train-ratio", type=float, default=0.2, help="Train ratio used when loading source datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dtype", type=_dtype_from_name, default=torch.float32, help="float32 or float64.")
    parser.add_argument(
        "--split-ratios",
        type=_parse_split_ratios,
        default=dict(DEFAULT_SPLIT_RATIOS),
        help="Comma-separated ratios, e.g. 1-1=0.7,1-many=0.1,many-1=0.1,many-many=0.1.",
    )
    parser.add_argument("--max-expansion", type=int, default=4, help="Maximum split nodes per side.")
    parser.add_argument("--internal-density", type=float, default=0.8, help="Internal edge probability among split nodes.")
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.0,
        help="Fuzzy-boundary ratio. Use 0 to disable; use 0.05 for the 5%% overlap setting.",
    )
    parser.add_argument(
        "--anchor-source",
        choices=["test", "all"],
        default="test",
        help="Anchor pool converted into many-to-many entities.",
    )
    parser.add_argument(
        "--train-anchor-source",
        choices=["train", "one_to_one", "none"],
        default="train",
        help="What to store as legacy 1-to-1 anchor_links in the generated .pt file.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing outputs.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a dataset fails.")
    parser.add_argument("--check", action="store_true", help="Run Dataset reload and metric sanity checks after each build.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.expanduser()
    args.output_root = args.output_root.expanduser()
    args.output_root.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    dataset_files = list(_iter_dataset_files(args.input_root, args.datasets))
    if not dataset_files:
        print(f"No .pt datasets found under {args.input_root}", file=sys.stderr)
        return 1

    print(f"Found {len(dataset_files)} dataset(s). Output root: {args.output_root}")
    for path in dataset_files:
        try:
            records.append(build_one(path, args))
        except Exception as exc:  # pragma: no cover - intentionally user-facing batch behavior.
            record = {
                "source_path": str(path),
                "source_dataset": path.stem,
                "status": "failed",
                "error": repr(exc),
            }
            records.append(record)
            print(f"[fail] {path.stem}: {exc}", file=sys.stderr)
            if not args.keep_going:
                break

    manifest = {
        "args": {
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
            "datasets": args.datasets,
            "name_pattern": args.name_pattern,
            "gids": args.gids,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "dtype": str(args.dtype).replace("torch.", ""),
            "split_ratios": args.split_ratios,
            "max_expansion": args.max_expansion,
            "internal_density": args.internal_density,
            "overlap_ratio": args.overlap_ratio,
            "anchor_source": args.anchor_source,
            "train_anchor_source": args.train_anchor_source,
            "overwrite": args.overwrite,
            "check": args.check,
        },
        "records": records,
    }
    manifest_path = args.output_root / "m2m_build_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in records if r["status"] == "ok")
    skipped = sum(1 for r in records if r["status"] == "skipped")
    failed = sum(1 for r in records if r["status"] == "failed")
    print(f"\nDone. ok={ok}, skipped={skipped}, failed={failed}")
    print(f"Manifest: {manifest_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
