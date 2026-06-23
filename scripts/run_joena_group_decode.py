"""Run JOENAGroupDecode on a many-to-many benchmark.

Trains JOENA once, then decodes groups from its alignment profiles and scores
the result under the *blind* (non-leaking) protocol. For reference it also prints
the *leaky* protocol (which hands the method the GT source groups and target
sizes via ``similarity_to_pred_entities``) on the same matrix.

Examples
--------
Single run::

    python scripts/run_joena_group_decode.py --dataset airport_m2m \
        --row-tau 0.1 --col-tau 0.1 --epochs 10

Threshold sweep (trains once, decodes at each tau)::

    python scripts/run_joena_group_decode.py --dataset airport_m2m \
        --epochs 10 --sweep 0.05,0.1,0.2,0.3,0.5 --out-csv logs/airport_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

import torch

from PlanetAlign.algorithms import JOENAGroupDecode
from PlanetAlign.data import Dataset
from PlanetAlign.metrics import hits_ks_scores
from PlanetAlign.m2m import evaluate_similarity
from PlanetAlign.utils import get_anchor_pairs


GIDS = [0, 1]
M2M_METRICS = ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]
CSV_FIELDS = [
    "dataset", "protocol", "row_tau", "col_tau", "src_groups", "tgt_groups",
    "Hits@1", "ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS", "time_s",
]


def load_gt(root: Path, name: str) -> Dict[str, Dict[str, List[int]]]:
    gt_path = root / f"{name}_gt_many2many.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"ground-truth JSON not found: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)["entities"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run JOENAGroupDecode on one M2M dataset.")
    p.add_argument("--m2m-root", type=Path, default=Path("data/m2m_no_overlap"))
    p.add_argument("--dataset", default="airport_m2m")
    p.add_argument("--row-tau", type=float, default=0.1, help="source-side profile threshold")
    p.add_argument("--col-tau", type=float, default=0.1, help="target-side profile threshold")
    p.add_argument("--sweep", default=None,
                   help="comma-separated taus; overrides --row-tau/--col-tau, reuses one trained model")
    p.add_argument("--match-threshold", type=float, default=0.0)
    p.add_argument("--relative-match-threshold", type=float, default=0.0)
    p.add_argument("--group-source", choices=["profile", "attr"], default="profile")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--train-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-attr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dataset = Dataset(root=args.m2m_root, name=args.dataset,
                      train_ratio=args.train_ratio, seed=args.seed)
    gt = load_gt(args.m2m_root, args.dataset)
    n1, n2 = int(dataset.pyg_graphs[0].num_nodes), int(dataset.pyg_graphs[1].num_nodes)
    print(f"Dataset      : {args.dataset}  (src={n1} tgt={n2}, GT entities={len(gt)})")
    print(f"Group source : {args.group_source}   use_attr={args.use_attr}   epochs={args.epochs}")

    taus = ([(float(t), float(t)) for t in args.sweep.split(",")]
            if args.sweep else [(args.row_tau, args.col_tau)])

    # Train JOENA once; reuse the matrix for every tau.
    model = JOENAGroupDecode(
        row_tau=taus[0][0], col_tau=taus[0][1],
        match_threshold=args.match_threshold,
        relative_match_threshold=args.relative_match_threshold,
        group_source=args.group_source,
        alpha=args.alpha,
    ).to(args.device)

    t0 = time.perf_counter()
    model.train(dataset=dataset, gids=GIDS, use_attr=args.use_attr,
                total_epochs=args.epochs, save_log=False, verbose=args.verbose)
    train_time = time.perf_counter() - t0

    S = model.S.detach().to(torch.float32).cpu()
    test_pairs = get_anchor_pairs(dataset.test_data, GIDS[0], GIDS[1])
    hits1 = float(hits_ks_scores(S, test_pairs, ks=[1], mode="mean").get(1, 0.0))
    leaky = evaluate_similarity(S, gt, metrics=M2M_METRICS)  # GT source groups + sizes

    records: List[Dict[str, Any]] = []
    records.append({
        "dataset": args.dataset, "protocol": "leaky-ref",
        "row_tau": "", "col_tau": "", "src_groups": "", "tgt_groups": "",
        "Hits@1": hits1, **leaky, "time_s": round(train_time, 3),
    })

    for row_tau, col_tau in taus:
        model.row_tau, model.col_tau = row_tau, col_tau
        td = time.perf_counter()
        blind = model.test_blind(gt, metrics=M2M_METRICS)
        decode_time = time.perf_counter() - td
        records.append({
            "dataset": args.dataset, "protocol": f"blind[{args.group_source}]",
            "row_tau": row_tau, "col_tau": col_tau,
            "src_groups": len(model.src_groups), "tgt_groups": len(model.tgt_groups),
            "Hits@1": hits1, **blind, "time_s": round(decode_time, 3),
        })

    # Pretty table.
    hdr = (f"{'protocol':<18}{'row_tau':>8}{'col_tau':>8}{'srcG':>7}{'tgtG':>7}"
           f"{'Hits@1':>9}{'ACS':>9}{'MSF1':>9}{'MicroF1':>9}{'SGS':>9}{'EGS':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in records:
        print(f"{r['protocol']:<18}{str(r['row_tau']):>8}{str(r['col_tau']):>8}"
              f"{str(r['src_groups']):>7}{str(r['tgt_groups']):>7}"
              f"{r['Hits@1']:>9.4f}{r['ACS']:>9.4f}{r['MSF1']:>9.4f}"
              f"{r['MicroF1']:>9.4f}{r['M2M-SGS']:>9.4f}{r['M2M-EGS']:>9.4f}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            for r in records:
                w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
        print(f"\nCSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
