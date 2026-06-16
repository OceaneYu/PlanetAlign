"""Compare JOENA, M2MAlign, and JOENA-initialized M2MAlign on one M2M benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

from PlanetAlign.algorithms import JOENA, M2MAlign
from PlanetAlign.data import Dataset
from PlanetAlign.metrics import hits_ks_scores
from PlanetAlign.m2m import evaluate_predictions, predictions_from_similarity
from PlanetAlign.utils import get_anchor_pairs


GIDS = [0, 1]
M2M_METRICS = ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]
CSV_FIELDS = ["method", "time_s", "Hits@1", "ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]


def load_gt(root: Path, dataset_name: str) -> Dict[str, Dict[str, List[int]]]:
    gt_path = root / f"{dataset_name}_gt_many2many.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"ground-truth JSON not found: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["entities"]


def matrix_stats(name: str, matrix: torch.Tensor) -> Dict[str, Any]:
    s = matrix.detach().to(torch.float32).cpu()
    stats = {
        "name": name,
        "shape": [int(s.shape[0]), int(s.shape[1])],
        "min": float(s.min().item()) if s.numel() else 0.0,
        "max": float(s.max().item()) if s.numel() else 0.0,
        "mean": float(s.mean().item()) if s.numel() else 0.0,
        "sum": float(s.sum().item()) if s.numel() else 0.0,
        "finite": bool(torch.isfinite(s).all().item()),
    }
    print(
        f"[matrix] {name:<28} shape={tuple(stats['shape'])} "
        f"min={stats['min']:.6g} max={stats['max']:.6g} "
        f"mean={stats['mean']:.6g} sum={stats['sum']:.6g} finite={stats['finite']}"
    )
    return stats


def assert_score_matrix(name: str, matrix: torch.Tensor, dataset: Dataset) -> None:
    expected = (int(dataset.pyg_graphs[GIDS[0]].num_nodes), int(dataset.pyg_graphs[GIDS[1]].num_nodes))
    if tuple(matrix.shape) != expected:
        raise AssertionError(f"{name} shape {tuple(matrix.shape)} != expected source-target shape {expected}")
    if not torch.isfinite(matrix.detach().to(torch.float32).cpu()).all():
        raise AssertionError(f"{name} contains NaN or Inf")


def evaluate_scores(
    method: str,
    elapsed: float,
    scores: torch.Tensor,
    dataset: Dataset,
    gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
    top_k: int,
) -> Dict[str, Any]:
    assert_score_matrix(method, scores, dataset)
    test_pairs = get_anchor_pairs(dataset.test_data, GIDS[0], GIDS[1])
    hits = hits_ks_scores(scores.detach().to(torch.float32).cpu(), test_pairs, ks=[1], mode="mean")
    pred = predictions_from_similarity(
        similarity=scores.detach().to(torch.float32).cpu(),
        gt_entities=gt_entities,
        top_k=top_k,
    )
    m2m = evaluate_predictions(gt_entities, pred, metrics=M2M_METRICS)
    return {
        "method": method,
        "time_s": round(float(elapsed), 4),
        "Hits@1": float(hits.get(1, 0.0)),
        **m2m,
    }


def run_joena(dataset: Dataset, use_attr: bool, epochs: int) -> Tuple[JOENA, torch.Tensor, float]:
    model = JOENA(alpha=0.7).to("cpu")
    started = time.perf_counter()
    model.train(dataset=dataset, gids=GIDS, use_attr=use_attr, total_epochs=epochs, save_log=False, verbose=False)
    elapsed = time.perf_counter() - started
    if model.S is None:
        raise RuntimeError("JOENA did not populate model.S")
    return model, model.S.detach().to(torch.float32).cpu(), elapsed


def m2m_align_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "alpha": args.m2m_alpha,
        "tau": args.m2m_tau,
        "overlap_slack": args.m2m_overlap_slack,
        "lambda_struct": args.m2m_lambda_struct,
        "beta": args.m2m_beta,
        "max_group_size": args.m2m_max_group_size,
        "n_iter": args.m2m_n_iter,
        "smooth_source": args.m2m_smooth_source,
    }


def run_m2m_align(
    dataset: Dataset,
    use_attr: bool,
    kwargs: Mapping[str, Any],
    init_s: torch.Tensor | None = None,
) -> Tuple[M2MAlign, torch.Tensor, float]:
    model = M2MAlign(**dict(kwargs)).to("cpu")
    started = time.perf_counter()
    if init_s is None:
        model.train(dataset=dataset, gids=GIDS, use_attr=use_attr, save_log=False, verbose=False)
    else:
        model.train(dataset=dataset, gids=GIDS, use_attr=use_attr, save_log=False, verbose=False, init_S=init_s)
    elapsed = time.perf_counter() - started
    if model.S is None:
        raise RuntimeError("M2MAlign did not populate model.S")
    return model, model.S.detach().to(torch.float32).cpu(), elapsed


def write_outputs(records: List[Dict[str, Any]], matrix_info: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[Path, Path]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.tag}.json"
    csv_path = out_dir / f"{args.tag}.csv"
    payload = {
        "args": {
            "m2m_root": str(args.m2m_root),
            "dataset": args.dataset,
            "top_k": args.top_k,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "joena_epochs": args.joena_epochs,
            "use_attr": args.use_attr,
            "m2m_align": m2m_align_kwargs(args),
        },
        "matrix_info": matrix_info,
        "records": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in CSV_FIELDS})
    return json_path, csv_path


def print_table(records: List[Dict[str, Any]]) -> None:
    header = f"{'method':<30}{'time_s':>9}{'Hits@1':>9}{'ACS':>9}{'MSF1':>9}{'MicroF1':>10}{'M2M-SGS':>10}{'M2M-EGS':>10}"
    print("\nResults")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in records:
        print(
            f"{row['method']:<30}{row['time_s']:>9.4f}{row['Hits@1']:>9.4f}"
            f"{row['ACS']:>9.4f}{row['MSF1']:>9.4f}{row['MicroF1']:>10.4f}"
            f"{row['M2M-SGS']:>10.4f}{row['M2M-EGS']:>10.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JOENA -> M2MAlign refinement on a single M2M dataset.")
    parser.add_argument("--m2m-root", type=Path, default=Path("data/m2m_no_overlap"))
    parser.add_argument("--dataset", default="pems08_m2m")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--tag", default="joena_m2m_refinement_pems08")
    parser.add_argument("--out-dir", type=Path, default=Path("logs/m2m_sweeps"))
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--joena-epochs", type=int, default=10)
    parser.add_argument("--use-attr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--m2m-alpha", type=float, default=0.6)
    parser.add_argument("--m2m-tau", type=float, default=0.95)
    parser.add_argument("--m2m-overlap-slack", type=float, default=0.10)
    parser.add_argument("--m2m-lambda-struct", type=float, default=0.5)
    parser.add_argument("--m2m-beta", type=float, default=0.6)
    parser.add_argument("--m2m-max-group-size", type=int, default=4)
    parser.add_argument("--m2m-n-iter", type=int, default=2)
    parser.add_argument("--m2m-smooth-source", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    dataset = Dataset(root=args.m2m_root, name=args.dataset, train_ratio=args.train_ratio, seed=args.seed)
    gt_entities = load_gt(args.m2m_root, args.dataset)
    expected_shape = (int(dataset.pyg_graphs[0].num_nodes), int(dataset.pyg_graphs[1].num_nodes))
    print(f"Dataset: {args.dataset}")
    print(f"Nodes  : source={expected_shape[0]} target={expected_shape[1]}")
    print(f"GT     : entities={len(gt_entities)}")
    print(f"Adapter: fixed top_k={args.top_k}; no GT target-size oracle")
    print(f"M2MAlign params: {m2m_align_kwargs(args)}")

    records: List[Dict[str, Any]] = []
    matrices: List[Dict[str, Any]] = []
    m2m_kwargs = m2m_align_kwargs(args)

    _, s_joena, time_joena = run_joena(dataset, args.use_attr, args.joena_epochs)
    assert_score_matrix("JOENA", s_joena, dataset)
    matrices.append(matrix_stats("JOENA S0", s_joena))
    records.append(evaluate_scores("JOENA", time_joena, s_joena, dataset, gt_entities, args.top_k))

    _, s_m2m_default, time_m2m_default = run_m2m_align(dataset, args.use_attr, m2m_kwargs, init_s=None)
    assert_score_matrix("M2MAlign default", s_m2m_default, dataset)
    matrices.append(matrix_stats("M2MAlign default S", s_m2m_default))
    records.append(evaluate_scores("M2MAlign default", time_m2m_default, s_m2m_default, dataset, gt_entities, args.top_k))

    if tuple(s_joena.shape) != expected_shape:
        raise AssertionError(f"JOENA init_S shape {tuple(s_joena.shape)} != expected {expected_shape}")
    _, s_refined, time_refined = run_m2m_align(dataset, args.use_attr, m2m_kwargs, init_s=s_joena)
    assert_score_matrix("JOENA -> M2MAlign", s_refined, dataset)
    matrices.append(matrix_stats("JOENA -> M2MAlign S", s_refined))
    records.append(evaluate_scores("JOENA -> M2MAlign", time_refined, s_refined, dataset, gt_entities, args.top_k))

    print_table(records)
    json_path, csv_path = write_outputs(records, matrices, args)
    print(f"\nJSON results: {json_path}")
    print(f"CSV results : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
