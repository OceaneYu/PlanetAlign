"""Run many-to-many benchmarks on selected datasets and algorithms.

The script generalizes the old Douban-only runners. It loads generated
``*_m2m.pt`` datasets plus their ``*_gt_many2many.json`` files, trains selected
PlanetAlign algorithms, converts each similarity matrix into many-to-many
groups, and writes JSON/CSV summaries.

Recommended first sweep:

    python scripts/run_m2m_experiments.py \
      --m2m-root data/m2m_no_overlap \
      --datasets pems08_m2m airport_m2m cora_m2m \
      --algorithms all \
      --profile quick

Run all datasets in a directory:

    python scripts/run_m2m_experiments.py \
      --m2m-root data/m2m_overlap_0.05 \
      --datasets all \
      --algorithms all \
      --profile quick

Run selected algorithms:

    python scripts/run_m2m_experiments.py \
      --m2m-root data/m2m_no_overlap \
      --datasets pems08_m2m \
      --algorithms IsoRank FINAL JOENA M2MAlign TGAE

Run every algorithm except especially slow ones:

    python -u scripts/run_m2m_experiments.py \
      --m2m-root data/m2m_no_overlap \
      --datasets pems08_m2m airport_m2m cora_m2m \
      --algorithms all \
      --exclude HOT SLOTAlign
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

import PlanetAlign
from PlanetAlign.data import Dataset
from PlanetAlign.metrics import many_to_many_scores, similarity_to_pred_entities
from PlanetAlign.m2m import evaluate_predictions
from PlanetAlign.utils import pairwise_cosine_similarity


GIDS = [0, 1]
RECOMMENDED_DATASETS = ["pems08_m2m", "airport_m2m", "cora_m2m"]
ONE_TO_ONE_METRICS = ["Hits@1", "Hits@10", "MRR"]
M2M_METRICS = ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]
CSV_FIELDS = [
    "dataset",
    "algorithm",
    "status",
    "profile",
    "time_s",
    "Hits@1",
    "Hits@10",
    "MRR",
    "ACS",
    "MSF1",
    "MicroF1",
    "M2M-SGS",
    "M2M-EGS",
    "selected_preserve_base_topk",
    "preserved_rows",
    "base_topk_mass",
    "base_entropy",
    "TGAE-local-ACS",
    "TGAE-local-MSF1",
    "TGAE-local-MicroF1",
    "TGAE-local-M2M-SGS",
    "TGAE-local-M2M-EGS",
    "error",
]


def _sim_from_embeddings(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    emb1 = emb1.detach().to(torch.float32).cpu()
    emb2 = emb2.detach().to(torch.float32).cpu()
    return pairwise_cosine_similarity(emb1, emb2)


def _sim_from_inner_product(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    return emb1.detach().to(torch.float32).cpu() @ emb2.detach().to(torch.float32).cpu().T


def _extract_s(algo: Any, ret: Tuple[Any, ...], mode: str, dataset: Dataset, gids: List[int]) -> torch.Tensor:
    gid1, gid2 = gids
    if mode == "self.S":
        s = algo.S
        if s is None:
            raise RuntimeError("algorithm did not populate self.S")
        return s.detach().to(torch.float32).cpu()
    if mode == "S":
        return ret[0].detach().to(torch.float32).cpu()
    if mode == "embs-cos":
        return _sim_from_embeddings(ret[0], ret[1])
    if mode == "embs-dot-dict":
        emb_dict = ret[0]
        return _sim_from_inner_product(emb_dict[gid1], emb_dict[gid2])
    raise ValueError(f"unknown sim extraction mode: {mode}")


def _has_node_attributes(dataset: Dataset, gids: List[int]) -> bool:
    return all(dataset.pyg_graphs[gid].x is not None for gid in gids)


def _quick(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return dict(kwargs)


def _full(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return dict(kwargs)


def _algo_configs(profile: str) -> Dict[str, Dict[str, Any]]:
    if profile == "quick":
        return {
            "IsoRank": {
                "factory": lambda: PlanetAlign.algorithms.IsoRank(alpha=0.4),
                "train_kwargs": _quick({"use_attr": False, "total_epochs": 10}),
                "mode": "S",
            },
            "FINAL": {
                "factory": lambda: PlanetAlign.algorithms.FINAL(alpha=0.9),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 10}),
                "mode": "self.S",
            },
            "IONE": {
                "factory": lambda: PlanetAlign.algorithms.IONE(out_dim=64),
                "train_kwargs": _quick({"use_attr": False, "total_epochs": 5}),
                "mode": "S",
            },
            "REGAL": {
                "factory": lambda: PlanetAlign.algorithms.REGAL(),
                "train_kwargs": _quick({"use_attr": True}),
                "mode": "embs-cos",
            },
            "CrossMNA": {
                "factory": lambda: PlanetAlign.algorithms.CrossMNA(),
                "train_kwargs": _quick({"use_attr": False, "total_epochs": 10}),
                "mode": "embs-dot-dict",
            },
            "NetTrans": {
                "factory": lambda: PlanetAlign.algorithms.NetTrans(),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 5}),
                "mode": "embs-cos",
            },
            "BRIGHT": {
                "factory": lambda: PlanetAlign.algorithms.BRIGHT(),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 10}),
                "mode": "embs-cos",
            },
            "NeXtAlign": {
                "factory": lambda: PlanetAlign.algorithms.NeXtAlign(),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 5}),
                "mode": "embs-cos",
            },
            "PARROT": {
                "factory": lambda: PlanetAlign.algorithms.PARROT(alpha=0.5),
                "train_kwargs": _quick({"use_attr": True}),
                "mode": "self.S",
            },
            "SLOTAlign": {
                "factory": lambda: PlanetAlign.algorithms.SLOTAlign(bases=4),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 10, "joint_epochs": 5}),
                "mode": "self.S",
            },
            "WLAlign": {
                "factory": lambda: PlanetAlign.algorithms.WLAlign(),
                "train_kwargs": _quick({"use_attr": False, "total_epochs": 5, "struct_epochs": 10}),
                "mode": "embs-cos",
            },
            "WAlign": {
                "factory": lambda: PlanetAlign.algorithms.WAlign(),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 5}),
                "mode": "embs-cos",
            },
            "HOT": {
                "factory": lambda: PlanetAlign.algorithms.HOT(alpha=0.5, lp=0.1),
                "train_kwargs": _quick({"use_attr": True, "in_iters": 2, "out_iters": 5}),
                "mode": "self.S",
            },
            "JOENA": {
                "factory": lambda: PlanetAlign.algorithms.JOENA(alpha=0.7),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 10}),
                "mode": "self.S",
            },
            "JOENAM2MAlign": {
                "factory": lambda: PlanetAlign.algorithms.JOENAM2MAlign(m2m_alpha=0.9),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 10}),
                "mode": "self.S",
            },
            "M2MAlign": {
                "factory": lambda: PlanetAlign.algorithms.M2MAlign(),
                "train_kwargs": _quick({"use_attr": True}),
                "mode": "self.S",
            },
            "GroupJOENA": {
                "factory": lambda: PlanetAlign.algorithms.GroupJOENA(
                    hidden_dim=32,
                    out_dim=32,
                    max_epochs=5,
                    eval_interval=5,
                    reconstruction_weight=0.0,
                ),
                "train_kwargs": _quick({"use_attr": True}),
                "mode": "self.S",
                "needs_num_groups": True,
                "oracle_num_groups": True,
            },
            "TGAE": {
                "factory": lambda: PlanetAlign.algorithms.TGAE(
                    num_hidden_layers=3,
                    hidden_dim=16,
                    output_dim=16,
                    lr=1e-3,
                ),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 5, "eval_interval": 5}),
                "mode": "self.S",
            },
            "DualMatch": {"skip": "train() is a stub in this repository"},
            "MEAformer": {"skip": "train() is a stub in this repository"},
        }

    if profile == "full":
        return {
            "IsoRank": {
                "factory": lambda: PlanetAlign.algorithms.IsoRank(alpha=0.4),
                "train_kwargs": _full({"use_attr": False, "total_epochs": 50}),
                "mode": "S",
            },
            "FINAL": {
                "factory": lambda: PlanetAlign.algorithms.FINAL(alpha=0.9),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 50}),
                "mode": "self.S",
            },
            "IONE": {
                "factory": lambda: PlanetAlign.algorithms.IONE(out_dim=100),
                "train_kwargs": _full({"use_attr": False, "total_epochs": 30}),
                "mode": "S",
            },
            "REGAL": {
                "factory": lambda: PlanetAlign.algorithms.REGAL(),
                "train_kwargs": _full({"use_attr": True}),
                "mode": "embs-cos",
            },
            "CrossMNA": {
                "factory": lambda: PlanetAlign.algorithms.CrossMNA(),
                "train_kwargs": _full({"use_attr": False, "total_epochs": 100}),
                "mode": "embs-dot-dict",
            },
            "NetTrans": {
                "factory": lambda: PlanetAlign.algorithms.NetTrans(),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 30}),
                "mode": "embs-cos",
            },
            "BRIGHT": {
                "factory": lambda: PlanetAlign.algorithms.BRIGHT(),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 100}),
                "mode": "embs-cos",
            },
            "NeXtAlign": {
                "factory": lambda: PlanetAlign.algorithms.NeXtAlign(),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 30}),
                "mode": "embs-cos",
            },
            "PARROT": {
                "factory": lambda: PlanetAlign.algorithms.PARROT(alpha=0.5),
                "train_kwargs": _full({"use_attr": True}),
                "mode": "self.S",
            },
            "SLOTAlign": {
                "factory": lambda: PlanetAlign.algorithms.SLOTAlign(bases=4),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 200, "joint_epochs": 50}),
                "mode": "self.S",
            },
            "WLAlign": {
                "factory": lambda: PlanetAlign.algorithms.WLAlign(),
                "train_kwargs": _full({"use_attr": False, "total_epochs": 30, "struct_epochs": 50}),
                "mode": "embs-cos",
            },
            "WAlign": {
                "factory": lambda: PlanetAlign.algorithms.WAlign(),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 20}),
                "mode": "embs-cos",
            },
            "HOT": {
                "factory": lambda: PlanetAlign.algorithms.HOT(alpha=0.5, lp=0.1),
                "train_kwargs": _full({"use_attr": True, "in_iters": 5, "out_iters": 30}),
                "mode": "self.S",
            },
            "JOENA": {
                "factory": lambda: PlanetAlign.algorithms.JOENA(alpha=0.7),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 50}),
                "mode": "self.S",
            },
            "JOENAM2MAlign": {
                "factory": lambda: PlanetAlign.algorithms.JOENAM2MAlign(m2m_alpha=0.9),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 50}),
                "mode": "self.S",
            },
            "M2MAlign": {
                "factory": lambda: PlanetAlign.algorithms.M2MAlign(),
                "train_kwargs": _full({"use_attr": True}),
                "mode": "self.S",
            },
            "GroupJOENA": {
                "factory": lambda: PlanetAlign.algorithms.GroupJOENA(
                    hidden_dim=128,
                    out_dim=128,
                    max_epochs=20,
                    eval_interval=5,
                ),
                "train_kwargs": _full({"use_attr": True}),
                "mode": "self.S",
                "needs_num_groups": True,
                "oracle_num_groups": True,
            },
            "TGAE": {
                "factory": lambda: PlanetAlign.algorithms.TGAE(
                    num_hidden_layers=4,
                    hidden_dim=16,
                    output_dim=16,
                    lr=1e-3,
                ),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 20, "eval_interval": 5}),
                "mode": "self.S",
            },
            "DualMatch": {"skip": "train() is a stub in this repository"},
            "MEAformer": {"skip": "train() is a stub in this repository"},
        }

    raise ValueError(f"unknown profile: {profile}")


def _available_dataset_names(root: Path) -> List[str]:
    return sorted(path.stem for path in root.glob("*_m2m.pt"))


def _resolve_datasets(root: Path, requested: Optional[List[str]]) -> List[str]:
    available = _available_dataset_names(root)
    if not requested:
        requested = list(RECOMMENDED_DATASETS)
    if requested == ["all"]:
        return available
    missing = [name for name in requested if name not in available]
    if missing:
        raise FileNotFoundError(f"missing datasets in {root}: {', '.join(missing)}")
    return requested


def _resolve_algorithms(
    configs: Dict[str, Dict[str, Any]],
    requested: Optional[List[str]],
    excluded: Optional[List[str]],
) -> List[str]:
    available = list(configs.keys())
    if not requested or requested == ["all"]:
        resolved = available
    else:
        lowered = {name.lower(): name for name in available}
        resolved = []
        missing = []
        for raw in requested:
            name = lowered.get(raw.lower())
            if name is None:
                missing.append(raw)
            else:
                resolved.append(name)
        if missing:
            raise ValueError(f"unknown algorithms: {', '.join(missing)}")

    if excluded:
        lowered = {name.lower(): name for name in available}
        exclude_set = set()
        missing_excluded = []
        for raw in excluded:
            name = lowered.get(raw.lower())
            if name is None:
                missing_excluded.append(raw)
            else:
                exclude_set.add(name)
        if missing_excluded:
            raise ValueError(f"unknown excluded algorithms: {', '.join(missing_excluded)}")
        resolved = [name for name in resolved if name not in exclude_set]
    return resolved


def _load_gt(root: Path, dataset_name: str) -> Dict[str, Any]:
    gt_path = root / f"{dataset_name}_gt_many2many.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"ground-truth JSON not found: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def evaluate_algorithm(
    dataset_name: str,
    dataset: Dataset,
    gt_entities: Dict[str, Dict[str, List[int]]],
    algo_name: str,
    config: Dict[str, Any],
    profile: str,
    seed: int,
    verbose: bool,
    auto_disable_attr: bool,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "dataset": dataset_name,
        "algorithm": algo_name,
        "status": "pending",
        "profile": profile,
    }

    if "skip" in config:
        record.update(status="skipped", error=config["skip"])
        return record

    torch.manual_seed(seed)
    train_kwargs = dict(config["train_kwargs"])
    if auto_disable_attr and train_kwargs.get("use_attr") and not _has_node_attributes(dataset, GIDS):
        train_kwargs["use_attr"] = False
        record["attr_note"] = "use_attr was disabled because this dataset has no node attributes"

    started = time.perf_counter()
    try:
        algo = config["factory"]().to("cpu")
        if config.get("needs_num_groups"):
            train_kwargs.setdefault("num_groups", len(gt_entities))
        if config.get("oracle_num_groups"):
            train_kwargs.setdefault("oracle_num_groups", True)
        ret = algo.train(
            dataset=dataset,
            gids=GIDS,
            save_log=False,
            verbose=verbose,
            **train_kwargs,
        )
        if not isinstance(ret, tuple):
            ret = (ret,)
        s = _extract_s(algo, ret, config["mode"], dataset, GIDS)
        algo.S = s.to(algo.device)

        one_to_one = algo.test(dataset=dataset, gids=GIDS, metrics=ONE_TO_ONE_METRICS)
        if algo_name == "GroupJOENA":
            pred = algo.predict_many_to_many(gt_entities, target_size_mode="group")
            m2m = evaluate_predictions(gt_entities, pred, metrics=M2M_METRICS)
        else:
            pred = similarity_to_pred_entities(s, gt_entities)
            m2m = many_to_many_scores(gt_entities, pred, metrics=M2M_METRICS)

        logged_train_kwargs = {
            key: value for key, value in train_kwargs.items()
            if key != "gt_entities"
        }
        if "gt_entities" in train_kwargs:
            logged_train_kwargs["num_gt_entities"] = len(train_kwargs["gt_entities"])

        record.update(
            {
                "status": "ok",
                "time_s": round(time.perf_counter() - started, 4),
                "train_kwargs": logged_train_kwargs,
                **one_to_one,
                **m2m,
            }
        )
        if hasattr(algo, "selected_preserve_base_topk_"):
            record["selected_preserve_base_topk"] = getattr(algo, "selected_preserve_base_topk_")
        if hasattr(algo, "preserved_rows_"):
            record["preserved_rows"] = getattr(algo, "preserved_rows_")
        base_confidence = getattr(algo, "base_confidence_", None)
        if base_confidence:
            record["base_topk_mass"] = base_confidence.get("topk_mass")
            record["base_entropy"] = base_confidence.get("entropy")

        if algo_name == "TGAE":
            local_pred = algo.predict_many_to_many(
                gt_entities,
                mode="local_expand",
                relax_ratio=1.03,
                target_dup_sim_ratio=1.03,
                max_extra_targets=3,
            )
            local_scores = many_to_many_scores(gt_entities, local_pred, metrics=M2M_METRICS)
            record.update(_flatten_metrics("TGAE-local-", local_scores))

    except Exception as exc:  # pragma: no cover - intentionally keeps long sweeps alive.
        logged_train_kwargs = {
            key: value for key, value in train_kwargs.items()
            if key != "gt_entities"
        }
        if "gt_entities" in train_kwargs:
            logged_train_kwargs["num_gt_entities"] = len(train_kwargs["gt_entities"])
        record.update(
            {
                "status": "error",
                "time_s": round(time.perf_counter() - started, 4),
                "train_kwargs": logged_train_kwargs,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return record


def _write_outputs(records: List[Dict[str, Any]], out_dir: Path, tag: str, args: argparse.Namespace) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{tag}.json"
    csv_path = out_dir / f"{tag}.csv"
    payload = {
        "args": {
            "m2m_root": str(args.m2m_root),
            "datasets": args.datasets,
            "algorithms": args.algorithms,
            "exclude": args.exclude,
            "profile": args.profile,
            "seed": args.seed,
        },
        "records": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in CSV_FIELDS})
    return json_path, csv_path


def _print_summary(records: List[Dict[str, Any]]) -> None:
    print("\nSummary")
    print("=" * 96)
    header = f"{'dataset':<18}{'algorithm':<14}{'status':<9}{'Hits@1':>9}{'MSF1':>9}{'MicroF1':>10}{'M2M-SGS':>10}{'time(s)':>9}"
    print(header)
    print("-" * len(header))
    for r in records:
        if r.get("status") == "ok":
            print(
                f"{r['dataset']:<18}{r['algorithm']:<14}{r['status']:<9}"
                f"{r.get('Hits@1', 0):>9.4f}{r.get('MSF1', 0):>9.4f}"
                f"{r.get('MicroF1', 0):>10.4f}{r.get('M2M-SGS', 0):>10.4f}"
                f"{r.get('time_s', 0):>9.1f}"
            )
        else:
            err = str(r.get("error", ""))[:40]
            print(f"{r['dataset']:<18}{r['algorithm']:<14}{r['status']:<9}{err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M2M experiments on selected datasets and algorithms.")
    parser.add_argument("--m2m-root", type=Path, default=Path("data/m2m_no_overlap"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(RECOMMENDED_DATASETS),
        help="Dataset names, or 'all'. Default: pems08_m2m airport_m2m cora_m2m.",
    )
    parser.add_argument("--algorithms", nargs="+", default=["all"], help="Algorithm names, or 'all'.")
    parser.add_argument("--exclude", nargs="+", default=None, help="Algorithm names to exclude from the run.")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/m2m_sweeps"))
    parser.add_argument("--tag", default=None, help="Output filename stem. Defaults to root/profile/dataset summary.")
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch algorithm logs.")
    parser.add_argument(
        "--no-auto-disable-attr",
        action="store_true",
        help="Do not turn use_attr off automatically on datasets without node attributes.",
    )
    parser.add_argument("--list-algorithms", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.m2m_root = args.m2m_root.expanduser()
    configs = _algo_configs(args.profile)

    if args.list_algorithms:
        print("\n".join(configs.keys()))
        return 0
    if args.list_datasets:
        print("\n".join(_available_dataset_names(args.m2m_root)))
        return 0

    dataset_names = _resolve_datasets(args.m2m_root, args.datasets)
    algorithm_names = _resolve_algorithms(configs, args.algorithms, args.exclude)
    print(f"M2M root   : {args.m2m_root}")
    print(f"Datasets   : {', '.join(dataset_names)}")
    print(f"Algorithms : {', '.join(algorithm_names)}")
    print(f"Profile    : {args.profile}")

    records: List[Dict[str, Any]] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 96)
        print(f"Dataset: {dataset_name}")
        print("=" * 96)
        dataset = Dataset(root=args.m2m_root, name=dataset_name, train_ratio=args.train_ratio, seed=args.seed)
        gt_payload = _load_gt(args.m2m_root, dataset_name)
        gt_entities = gt_payload["entities"]
        print(
            f"nodes={[g.num_nodes for g in dataset.pyg_graphs]} "
            f"entities={len(gt_entities)} "
            f"overlap={gt_payload.get('metadata', {}).get('overlap_ratio')}"
        )

        for algo_name in algorithm_names:
            print(f"[run] {dataset_name} / {algo_name}")
            record = evaluate_algorithm(
                dataset_name=dataset_name,
                dataset=dataset,
                gt_entities=gt_entities,
                algo_name=algo_name,
                config=configs[algo_name],
                profile=args.profile,
                seed=args.seed,
                verbose=args.verbose,
                auto_disable_attr=not args.no_auto_disable_attr,
            )
            records.append(record)
            status = record["status"]
            if status == "ok":
                print(
                    f"      ok {record['time_s']:.1f}s "
                    f"Hits@1={record['Hits@1']:.4f} "
                    f"MSF1={record['MSF1']:.4f} "
                    f"MicroF1={record['MicroF1']:.4f}"
                )
            else:
                print(f"      {status}: {record.get('error', '')}")

    tag = args.tag
    if tag is None:
        root_tag = args.m2m_root.name.replace("/", "_")
        ds_tag = "recommended3" if dataset_names == RECOMMENDED_DATASETS else f"{len(dataset_names)}datasets"
        algo_tag = "all" if args.algorithms == ["all"] else f"{len(algorithm_names)}algorithms"
        tag = f"{root_tag}_{ds_tag}_{algo_tag}_{args.profile}"

    json_path, csv_path = _write_outputs(records, args.out_dir, tag, args)
    _print_summary(records)
    print(f"\nJSON results: {json_path}")
    print(f"CSV results : {csv_path}")

    errors = sum(1 for record in records if record.get("status") == "error")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
