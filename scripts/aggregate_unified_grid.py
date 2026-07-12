"""Aggregate the multi-seed unified-system grid into the decision table.

Reads the three result files produced by the base-selection experiments and
reports, per dataset, mean +/- std blind MicroF1 for:

* PARROT (universal base candidate),
* the JOENA-family ensemble (strong-attribute datasets only),
* anchor-arbitrated {PARROT, ENSEMBLE} (the selector-free unified system), and
* always-PARROT (the simplest unified system).

The point of the table is the honest keep-only-if-better decision: does adding
the ensemble + anchor arbitration beat just using PARROT everywhere?
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

DIAG = Path("logs/m2m_diag")
STRONG = ["douban_m2m", "cora_m2m", "airport_m2m", "pems08_m2m", "ppi_m2m"]
WEAK = ["arenas_m2m", "phone-email_m2m", "italy_m2m", "foursquare-twitter_m2m"]


def load(path: Path) -> list:
    return json.load(open(path)) if path.exists() else []


def ms(xs: List[float]) -> str:
    if not xs:
        return "     -    "
    return f"{mean(xs):.4f}±{pstdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.4f}     "


def main() -> int:
    # PARROT blind MicroF1 per (dataset, seed) from the two PARROT sweeps.
    parrot: Dict[str, Dict[int, float]] = {}
    for rec in load(DIAG / "parrot_all_s42.json") + load(DIAG / "parrot_weak_s012.json"):
        d, s = rec["dataset"], rec["seed"]
        parrot.setdefault(d, {})[s] = rec["bases"]["PARROT"]["blind_microf1"]

    # PARROT + ENSEMBLE per (dataset, seed) from the head-to-head (strong-attr).
    pve = load(DIAG / "parrot_vs_ensemble_all.json")
    ens: Dict[str, Dict[int, float]] = {}
    arb: Dict[str, Dict[int, float]] = {}      # anchor-arbitrated blind MicroF1
    picks: Dict[str, List[str]] = {}
    mism = 0
    for rec in pve:
        d, s = rec["dataset"], rec["seed"]
        parrot.setdefault(d, {})[s] = rec["parrot_mf1"]      # authoritative (same protocol)
        if rec.get("ens_mf1") is None:
            continue
        ens.setdefault(d, {})[s] = rec["ens_mf1"]
        chosen = "PARROT" if rec["parrot_corr"] >= rec["ens_corr"] else "ENSEMBLE"
        arb.setdefault(d, {})[s] = rec["parrot_mf1"] if chosen == "PARROT" else rec["ens_mf1"]
        picks.setdefault(d, []).append(chosen[0])
        if rec["anchor_pick"] != rec["oracle_pick"]:
            mism += 1

    hdr = f"{'dataset':<24}{'PARROT':>16}{'ENSEMBLE':>16}{'ARBITRATED':>16}{'anchor-picks':>14}"
    print(hdr + "\n" + "-" * len(hdr))
    parrot_means, arb_means = [], []
    for d in STRONG + WEAK:
        p = list(parrot.get(d, {}).values())
        e = list(ens.get(d, {}).values())
        a = list(arb.get(d, {}).values()) or p     # weak-attr: no ensemble -> PARROT
        parrot_means.append(mean(p) if p else 0.0)
        arb_means.append(mean(a) if a else 0.0)
        pk = "".join(picks.get(d, ["P"] * len(p)))
        print(f"{d:<24}{ms(p):>16}{ms(e):>16}{ms(a):>16}{pk:>14}")

    print("-" * len(hdr))
    print(f"{'MEAN (9 datasets)':<24}{mean(parrot_means):>16.4f}{'':>16}{mean(arb_means):>16.4f}")
    print(f"\nanchor/oracle base mismatches (strong-attr, all seeds): {mism}")
    print("P=PARROT chosen, E=ENSEMBLE chosen (by chance-corrected anchor agreement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
