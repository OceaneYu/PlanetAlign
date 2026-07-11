"""Probe alternative base aligners on datasets where JOENA fails.

QuotientDecode is aligner-agnostic: it only needs a similarity matrix ``S``
with non-trivial alignment signal. On italy / phone-email / arenas (weak or no
attributes, sparse structure) JOENA's anchor-RWR feature pipeline collapses
(Hits@1 ~ 0 even at 200 epochs). This script asks whether any *other* family
in the library aligns those graphs — consistency (IsoRank, FINAL), embedding
(REGAL, BRIGHT, NeXtAlign, WAlign, CrossMNA) — and, for each viable base,
what the blind M2M readout achieves on top of it.

    python scripts/probe_base_aligners.py --dataset italy_m2m
    python scripts/probe_base_aligners.py --dataset foursquare-twitter_m2m \
        --algorithms IsoRank REGAL
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

import torch

import PlanetAlign
from PlanetAlign.data import Dataset
from PlanetAlign.m2m import use_full_anchor_supervision
from PlanetAlign.m2m_quotient import evaluate_quotient_blind
from PlanetAlign.metrics import hits_ks_scores
from PlanetAlign.utils import get_anchor_pairs, pairwise_cosine_similarity

GIDS = [0, 1]
CACHE = Path("logs/m2m_diag/S_cache")


def configs() -> Dict[str, Dict[str, Any]]:
    """Quick-profile configs (mirrors scripts/run_m2m_experiments.py)."""
    A = PlanetAlign.algorithms
    return {
        "IsoRank":   dict(factory=lambda: A.IsoRank(alpha=0.4),
                          kwargs=dict(use_attr=False, total_epochs=10), mode="S"),
        "FINAL":     dict(factory=lambda: A.FINAL(alpha=0.9),
                          kwargs=dict(use_attr=True, total_epochs=10), mode="self.S"),
        "REGAL":     dict(factory=lambda: A.REGAL(),
                          kwargs=dict(use_attr=True), mode="embs-cos"),
        "BRIGHT":    dict(factory=lambda: A.BRIGHT(),
                          kwargs=dict(use_attr=True, total_epochs=10), mode="embs-cos"),
        "NeXtAlign": dict(factory=lambda: A.NeXtAlign(),
                          kwargs=dict(use_attr=True, total_epochs=5), mode="embs-cos"),
        "WAlign":    dict(factory=lambda: A.WAlign(),
                          kwargs=dict(use_attr=True, total_epochs=5), mode="embs-cos"),
        "CrossMNA":  dict(factory=lambda: A.CrossMNA(),
                          kwargs=dict(use_attr=False, total_epochs=10), mode="embs-dot-dict"),
        "JOENA":     dict(factory=lambda: A.JOENA(alpha=0.7),
                          kwargs=dict(use_attr=True, total_epochs=10), mode="self.S"),
    }


def extract_s(algo, ret, mode):
    if not isinstance(ret, tuple):
        ret = (ret,)
    if mode == "self.S":
        return algo.S.detach().to(torch.float32).cpu()
    if mode == "S":
        return ret[0].detach().to(torch.float32).cpu()
    if mode == "embs-cos":
        return pairwise_cosine_similarity(ret[0].detach().to(torch.float32).cpu(),
                                          ret[1].detach().to(torch.float32).cpu())
    if mode == "embs-dot-dict":
        d = ret[0]
        return (d[GIDS[0]].detach().to(torch.float32).cpu()
                @ d[GIDS[1]].detach().to(torch.float32).cpu().T)
    raise ValueError(mode)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/m2m_no_overlap"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--algorithms", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.2)
    p.add_argument("--no-full-anchors", action="store_true",
                   help="reproduce the old double-split protocol (~4%% supervision)")
    args = p.parse_args()

    ds = Dataset(root=args.root, name=args.dataset, train_ratio=args.train_ratio, seed=args.seed)
    if not args.no_full_anchors:
        # anchor_links already IS the original 20%% train split; the entity GT
        # comes from the original test split (disjoint, verified).
        use_full_anchor_supervision(ds)
    gt = json.load(open(args.root / f"{args.dataset}_gt_many2many.json"))["entities"]
    g1, g2 = ds.pyg_graphs[GIDS[0]], ds.pyg_graphs[GIDS[1]]
    has_attr = all(g.x is not None for g in (g1, g2))
    anchors = get_anchor_pairs(ds.train_data, GIDS[0], GIDS[1])
    test_pairs = anchors  # in-sample diagnostic only (no held-out pairs in this protocol)
    print(f"dataset={args.dataset} nodes={[g1.num_nodes, g2.num_nodes]} attrs={has_attr} "
          f"train_anchors={anchors.shape[0]} entities={len(gt)}")

    cfgs = configs()
    names = args.algorithms or [n for n in cfgs if n != "JOENA"]
    hdr = f"{'base':<11}{'Hits@1*':>8}{'Hits@10*':>9}{'blind MicroF1':>15}{'time(s)':>9}   (*in-sample train-anchor Hits)"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for name in names:
        cfg = cfgs[name]
        kw = dict(cfg["kwargs"])
        if kw.get("use_attr") and not has_attr:
            kw["use_attr"] = False
        t0 = time.perf_counter()
        try:
            torch.manual_seed(args.seed)
            algo = cfg["factory"]().to("cpu")
            ret = algo.train(dataset=ds, gids=GIDS, save_log=False, verbose=False, **kw)
            S = extract_s(algo, ret, cfg["mode"])
            hits = hits_ks_scores(S, test_pairs, ks=[1, 10], mode="mean")
            sc = evaluate_quotient_blind(S, gt, g1, g2, metrics=["MicroF1"],
                                         use_attr=has_attr, anchors=anchors)
            CACHE.mkdir(parents=True, exist_ok=True)
            torch.save(S, CACHE / f"{args.dataset}_base-{name}_e-quick_s{args.seed}_fa.pt")
            print(f"{name:<11}{hits[1]:>8.4f}{hits[10]:>9.4f}{sc['MicroF1']:>15.4f}"
                  f"{time.perf_counter()-t0:>9.1f}")
        except Exception as exc:
            print(f"{name:<11}  ERROR {type(exc).__name__}: {str(exc)[:70]} "
                  f"({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
