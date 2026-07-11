"""Multi-seed sweep of QuotientDecode readout variants over cached couplings.

Evaluates decode-only variants (group scoring mode, quotient neighbor
consistency, outlier eviction) on the ensemble S (average of the three cached
branch couplings) for every (dataset, seed) cell with complete caches. All
blind; every variant sees identical S per cell, so differences are attributable
to the readout change alone — and multi-seed from the start (the arbitration
lesson).

    python scripts/sweep_readout_variants.py --stage scores
    python scripts/sweep_readout_variants.py --stage refine --score-mode <winner>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

import torch

from PlanetAlign.data import Dataset
from PlanetAlign.m2m_quotient import evaluate_quotient_blind
from PlanetAlign.utils import get_anchor_pairs

BRANCH_TAGS = ["", "_pc-u1.0-a1.0-t0.1", "_pc-u1.0-a1.0-t0.1-gm0.5"]
CELLS = [
    ("douban_m2m", "data/m2m_overlap_0.05", [42, 0, 1, 2]),
    ("cora_m2m", "data/m2m_no_overlap", [42, 0, 1, 2]),
    ("airport_m2m", "data/m2m_no_overlap", [42, 0, 1, 2]),
    ("pems08_m2m", "data/m2m_no_overlap", [42, 0, 1, 2]),
    ("ppi_m2m", "data/m2m_no_overlap", [42, 0]),
]
CACHE = Path("logs/m2m_diag/S_cache")


def load_cells():
    cells = []
    for name, root, seeds in CELLS:
        root = Path(root)
        gt = json.load(open(root / f"{name}_gt_many2many.json"))["entities"]
        for seed in seeds:
            paths = [CACHE / f"{name}{t}_e10_s{seed}.pt" for t in BRANCH_TAGS]
            if not all(p.exists() for p in paths):
                continue
            ds = Dataset(root=root, name=name, train_ratio=0.2, seed=seed)
            S = torch.stack([torch.load(p, weights_only=True) for p in paths]).mean(dim=0)
            cells.append(dict(name=name, seed=seed, S=S, gt=gt,
                              g1=ds.pyg_graphs[0], g2=ds.pyg_graphs[1],
                              anchors=get_anchor_pairs(ds.train_data, 0, 1)))
    return cells


def run_variant(cells, label, **kw):
    per_ds = {}
    t0 = time.perf_counter()
    for c in cells:
        sc = evaluate_quotient_blind(c["S"], c["gt"], c["g1"], c["g2"],
                                     metrics=["MicroF1"], use_attr=True,
                                     anchors=c["anchors"], **kw)
        per_ds.setdefault(c["name"], []).append(sc["MicroF1"])
    means = {ds: statistics.mean(v) for ds, v in per_ds.items()}
    overall = statistics.mean(means.values())
    row = "  ".join(f"{ds.split('_')[0]}={means[ds]:.4f}" for ds in means)
    print(f"{label:<34} {row}  | overall={overall:.4f} ({time.perf_counter()-t0:.0f}s)")
    return means, overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["scores", "refine"], default="scores")
    p.add_argument("--score-mode", default="mean")
    args = p.parse_args()

    cells = load_cells()
    print(f"loaded {len(cells)} (dataset, seed) cells\n")

    if args.stage == "scores":
        for mode in ["mean", "max", "sum", "coverage"]:
            run_variant(cells, f"score={mode}", score_mode=mode)
    else:
        base = args.score_mode
        run_variant(cells, f"score={base} (base)", score_mode=base)
        for beta in [0.1, 0.2, 0.4]:
            run_variant(cells, f"score={base} +nbr(beta={beta})",
                        score_mode=base, neighbor_beta=beta)
        run_variant(cells, f"score={base} +evict", score_mode=base, evict=True)
        run_variant(cells, f"score={base} +nbr(0.2)+evict",
                    score_mode=base, neighbor_beta=0.2, evict=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
