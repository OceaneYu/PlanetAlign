"""Paired test: FGW quotient matcher vs Hungarian, over a frozen PARROT S.

The quotient-level consistency principle is the Gromov-Wasserstein structure
term; the existing linear ``neighbor_consistency_refine`` is its first-order
linearization. This script asks whether the exact nonlinear form (FGW) beats
plain Hungarian (feature-only) and the linear refine, on the same PARROT S.

PARROT is deterministic and the decode is deterministic, so one run per dataset
is the whole answer. For each dataset we also log the chance-corrected anchor
agreement of each variant, to check whether alpha can be selected *blind*
(same signal as every other choice in the system).

    python scripts/test_fgw_matcher.py                 # all 9, seed 42
    python scripts/test_fgw_matcher.py --datasets cora_m2m ppi_m2m
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

import torch

from PlanetAlign.data import Dataset
from PlanetAlign.m2m import use_full_anchor_supervision
from PlanetAlign.m2m_base import train_base_S, arbitrate_sharpen
from PlanetAlign.m2m_quotient import (entity_anchor_agreement, evaluate_quotient_blind,
                                      quotient_decode)
from PlanetAlign.utils import get_anchor_pairs

DATASETS = ["douban_m2m", "cora_m2m", "airport_m2m", "pems08_m2m", "ppi_m2m",
            "arenas_m2m", "phone-email_m2m", "italy_m2m", "foursquare-twitter_m2m"]
ALPHAS = [0.1, 0.25, 0.5, 0.75]


def score(S, g_src, g_tgt, anchors, gt, has_attr, **decode_kw):
    sc, pred, _ = evaluate_quotient_blind(
        S, gt, g_src, g_tgt, metrics=["MicroF1"], return_predictions=True,
        use_attr=has_attr, anchors=anchors, **decode_kw)
    corr, _ = entity_anchor_agreement(pred, anchors, int(g_tgt.num_nodes))
    return sc["MicroF1"], corr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/m2m_no_overlap"))
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("logs/m2m_diag/fgw_matcher.json"))
    args = p.parse_args()

    cols = ["hungarian", "linear(b.15)"] + [f"fgw a{a}" for a in ALPHAS]
    hdr = f"{'dataset':<24}" + "".join(f"{c:>14}" for c in cols) + "   fgw-anchor-pick"
    print(hdr + "\n" + "-" * len(hdr))
    results = []
    wins = losses = ties = 0
    for name in args.datasets:
        ds = Dataset(root=args.root, name=name, train_ratio=0.2, seed=args.seed)
        use_full_anchor_supervision(ds)
        gt = json.load(open(args.root / f"{name}_gt_many2many.json"))["entities"]
        g_src, g_tgt = ds.pyg_graphs[0], ds.pyg_graphs[1]
        has_attr = all(g.x is not None for g in (g_src, g_tgt))
        anchors = get_anchor_pairs(ds.train_data, 0, 1)
        S = train_base_S("PARROT", ds, seed=args.seed)
        S, _, _ = arbitrate_sharpen(S, g_src, g_tgt, anchors, use_attr=has_attr)

        base_mf1, base_corr = score(S, g_src, g_tgt, anchors, gt, has_attr, matcher="hungarian")
        lin_mf1, _ = score(S, g_src, g_tgt, anchors, gt, has_attr,
                           matcher="hungarian", neighbor_beta=0.15)
        fgw = {}
        for a in ALPHAS:
            mf1, corr = score(S, g_src, g_tgt, anchors, gt, has_attr, matcher="fgw", fgw_alpha=a)
            fgw[a] = (mf1, corr)
        # Blind alpha pick by chance-corrected anchor agreement.
        a_pick = max(ALPHAS, key=lambda a: fgw[a][1])
        fgw_picked_mf1 = fgw[a_pick][0]
        delta = fgw_picked_mf1 - base_mf1
        wins += delta > 1e-4; losses += delta < -1e-4; ties += abs(delta) <= 1e-4

        row = f"{name:<24}{base_mf1:>14.4f}{lin_mf1:>14.4f}" + \
              "".join(f"{fgw[a][0]:>14.4f}" for a in ALPHAS) + \
              f"   a{a_pick}->{fgw_picked_mf1:.4f}({'+' if delta>=0 else ''}{delta:.4f})"
        print(row)
        results.append(dict(dataset=name, hungarian=round(base_mf1, 4),
                            linear=round(lin_mf1, 4),
                            fgw={f"a{a}": [round(fgw[a][0], 4), round(fgw[a][1], 4)] for a in ALPHAS},
                            anchor_pick_alpha=a_pick, fgw_picked_mf1=round(fgw_picked_mf1, 4),
                            delta_vs_hungarian=round(delta, 4)))

    print("-" * len(hdr))
    print(f"blind-alpha FGW vs Hungarian:  {wins} wins / {losses} losses / {ties} ties")
    # Oracle: best fgw alpha per dataset (upper bound, not blind-selectable).
    oracle_delta = sum(max(r["fgw"][k][0] for k in r["fgw"]) - r["hungarian"] for r in results)
    print(f"oracle-alpha FGW total gain over Hungarian: {oracle_delta:+.4f} "
          f"(mean {oracle_delta/len(results):+.4f})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
