"""Is the base aligner an anchor-arbitrable axis? — the unification experiment.

For every dataset and seed, train each base in the roster, sharpen-arbitrate it
blind, and record two numbers per base:

* ``corrected`` — chance-corrected *train-anchor* entity agreement (blind: the
  only signal the selector is allowed to use), and
* ``blind_microf1`` — the oracle metric the selector is trying to maximize.

The unified system picks ``argmax(corrected)``. This script asks whether that
blind pick tracks ``argmax(blind_microf1)`` — i.e. whether the base can be
chosen without ever looking at the entity GT. It is the direct test of the claim
in docs §5.11.1 that "one base (PARROT) wins, arbitrated on anchors".

    python scripts/validate_base_selection.py --seeds 42 0 1 2 \
        --bases PARROT JOENA --datasets douban_m2m cora_m2m arenas_m2m ...
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

import torch

from PlanetAlign.data import Dataset
from PlanetAlign.m2m import use_full_anchor_supervision
from PlanetAlign.m2m_base import train_base_S, arbitrate_sharpen
from PlanetAlign.m2m_quotient import (entity_anchor_agreement, evaluate_quotient_blind,
                                      quotient_decode)
from PlanetAlign.utils import get_anchor_pairs

GIDS = (0, 1)
DEFAULT_DATASETS = ["douban_m2m", "cora_m2m", "airport_m2m", "pems08_m2m", "ppi_m2m",
                    "arenas_m2m", "phone-email_m2m", "italy_m2m", "foursquare-twitter_m2m"]


def run_one(root: Path, name: str, bases: List[str], seed: int) -> Dict:
    ds = Dataset(root=root, name=name, train_ratio=0.2, seed=seed)
    use_full_anchor_supervision(ds)
    gt = json.load(open(root / f"{name}_gt_many2many.json"))["entities"]
    g_src, g_tgt = ds.pyg_graphs[0], ds.pyg_graphs[1]
    has_attr = all(g.x is not None for g in (g_src, g_tgt))
    anchors = get_anchor_pairs(ds.train_data, 0, 1)
    n2 = int(g_tgt.num_nodes)
    out = {"dataset": name, "seed": seed, "nodes": [int(g_src.num_nodes), int(g_tgt.num_nodes)],
           "attrs": has_attr, "anchors": int(anchors.shape[0]), "entities": len(gt), "bases": {}}
    for b in bases:
        t0 = time.perf_counter()
        try:
            S = train_base_S(b, ds, seed=seed)
            S_best, T, _ = arbitrate_sharpen(S, g_src, g_tgt, anchors, use_attr=has_attr)
            pred, _ = quotient_decode(S_best, g_src, g_tgt, use_attr=has_attr, anchors=anchors)
            corrected, raw = entity_anchor_agreement(pred, anchors, n2)
            sc = evaluate_quotient_blind(S_best, gt, g_src, g_tgt, metrics=["MicroF1"],
                                         use_attr=has_attr, anchors=anchors)
            out["bases"][b] = {"temp": T, "corrected": round(corrected, 4), "raw": round(raw, 4),
                               "blind_microf1": round(sc["MicroF1"], 4), "time_s": round(time.perf_counter()-t0, 1)}
        except Exception as exc:  # keep the grid going; a base can OOM/diverge
            out["bases"][b] = {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/m2m_no_overlap"))
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--bases", nargs="+", default=["PARROT", "JOENA"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--out", type=Path, default=Path("logs/m2m_diag/base_selection.json"))
    args = p.parse_args()

    results = []
    hdr = f"{'dataset':<24}{'seed':>5}  " + "".join(f"{b+' corr/blind':>22}" for b in args.bases) + "   anchor-pick  oracle-pick"
    print(hdr + "\n" + "-" * len(hdr))
    for name in args.datasets:
        for seed in args.seeds:
            r = run_one(args.root, name, args.bases, seed)
            results.append(r)
            cells = ""
            for b in args.bases:
                d = r["bases"][b]
                cells += (f"{d['error'][:20]:>22}" if "error" in d
                          else f"{d['corrected']:>10.4f}/{d['blind_microf1']:<10.4f}"[:22].rjust(22))
            ok = [b for b in args.bases if "error" not in r["bases"][b]]
            apick = max(ok, key=lambda b: (r["bases"][b]["corrected"], r["bases"][b]["raw"])) if ok else "-"
            opick = max(ok, key=lambda b: r["bases"][b]["blind_microf1"]) if ok else "-"
            flag = "" if apick == opick else "  <-- MISMATCH"
            print(f"{name:<24}{seed:>5}  {cells}   {apick:<11} {opick:<11}{flag}")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
