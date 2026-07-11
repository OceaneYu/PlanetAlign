"""Head-to-head M2M readout comparison over the SAME trained aligner.

Trains JOENA once per dataset, freezes its similarity matrix ``S``, then scores
every readout on that same matrix under the blind protocol. Because the
representation is held fixed, any metric difference is attributable to the
readout alone — the controlled experiment behind ``docs/m2m_quotient_align_design.md``.

Readouts
--------
- ``attr-blind``       : attribute-cohesion grouping + greedy match (m2m_blind default)
- ``profile-greedy``   : fixed-tau profile grouping + greedy match (= JOENAGroupDecode)
- ``quotient``         : full QuotientDecode (Otsu taus, anti-chaining, Hungarian, 2 iters)
- ablations           : quotient minus one component at a time

Usage
-----
    python scripts/run_quotient_compare.py \
        --root data/m2m_overlap_0.05 --dataset douban_m2m
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

import torch

from PlanetAlign.algorithms import JOENA
from PlanetAlign.data import Dataset
from PlanetAlign.m2m_blind import evaluate_blind, discover_groups_by_profile, decode_entity_map
from PlanetAlign.m2m import align_prediction_to_ground_truth
from PlanetAlign.m2m_quotient import evaluate_quotient_blind
from PlanetAlign.metrics import many_to_many_scores
from PlanetAlign.utils import get_anchor_pairs

GIDS = [0, 1]
M2M_METRICS = ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]


def profile_greedy_scores(S, gt, g_src, g_tgt, tau=0.1):
    """The JOENAGroupDecode readout: fixed-tau profile union-find + greedy match."""
    src_groups = discover_groups_by_profile(S, g_src, tau=tau)
    tgt_groups = discover_groups_by_profile(S.T.contiguous(), g_tgt, tau=tau)
    pred = decode_entity_map(S, src_groups, tgt_groups)
    aligned = align_prediction_to_ground_truth(gt, pred)
    scores = many_to_many_scores(gt, aligned, metrics=M2M_METRICS)
    return scores, (len(src_groups), len(tgt_groups))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/m2m_overlap_0.05"))
    p.add_argument("--dataset", default="douban_m2m")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.2)
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--model", choices=["joena", "joena-pc", "auto", "ensemble"], default="joena")
    p.add_argument("--pc-lambda-unif", type=float, default=1.0)
    p.add_argument("--pc-lambda-align", type=float, default=1.0)
    p.add_argument("--pc-no-break-symmetry", action="store_true")
    p.add_argument("--pc-temp", type=float, default=0.1)
    p.add_argument("--pc-group-marginals", action="store_true")
    p.add_argument("--pc-marginal-rho", type=float, default=0.5)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ds = Dataset(root=args.root, name=args.dataset, train_ratio=args.train_ratio, seed=args.seed)
    gt = json.load(open(args.root / f"{args.dataset}_gt_many2many.json"))["entities"]
    g_src, g_tgt = ds.pyg_graphs[GIDS[0]], ds.pyg_graphs[GIDS[1]]
    use_attr = all(g.x is not None for g in (g_src, g_tgt))
    print(f"dataset={args.dataset} nodes={[g_src.num_nodes, g_tgt.num_nodes]} "
          f"entities={len(gt)} use_attr={use_attr}")

    def get_S(model_name: str) -> torch.Tensor:
        # "joena-pc-gm" (an auto branch) forces group marginals at the frozen
        # rho; plain "joena-pc" follows the CLI flags (defaults = frozen recipe).
        use_gm = args.pc_group_marginals or model_name == "joena-pc-gm"
        if model_name == "joena":
            tag = ""
        else:
            tag = (f"_pc-u{args.pc_lambda_unif}-a{args.pc_lambda_align}"
                   f"-t{args.pc_temp}{'-nosym' if args.pc_no_break_symmetry else ''}"
                   f"{f'-gm{args.pc_marginal_rho}' if use_gm else ''}")
        cache = Path("logs/m2m_diag/S_cache") / f"{args.dataset}{tag}_e{args.epochs}_s{args.seed}.pt"
        if cache.exists():
            s_ = torch.load(cache, map_location="cpu", weights_only=True)
            print(f"[{model_name}] loaded cached S {list(s_.shape)}")
            return s_
        t0 = time.perf_counter()
        if model_name == "joena":
            aligner = JOENA(alpha=0.7).to("cpu")
        else:
            from PlanetAlign.m2m_contrastive import JOENAPC
            aligner = JOENAPC(alpha=0.7,
                              lambda_unif=args.pc_lambda_unif,
                              lambda_align=args.pc_lambda_align,
                              temp=args.pc_temp,
                              break_symmetry=not args.pc_no_break_symmetry,
                              group_marginals=use_gm,
                              marginal_rho=args.pc_marginal_rho).to("cpu")
        torch.manual_seed(args.seed)
        aligner.train(dataset=ds, gids=GIDS, use_attr=use_attr,
                      total_epochs=args.epochs, save_log=False, verbose=False)
        s_ = aligner.S.detach().to(torch.float32).cpu()
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(s_, cache)
        print(f"[{model_name}] trained in {time.perf_counter()-t0:.1f}s; cached")
        return s_

    anchors = get_anchor_pairs(ds.train_data, GIDS[0], GIDS[1])
    model_tag = "" if args.model == "joena" else f"_{args.model.replace('-', '_')}"

    if args.model == "ensemble":
        # Selector-free system: average the three branches' couplings (same
        # scale, ~doubly-stochastic) and decode once. Multi-seed measurement:
        # matches the oracle branch on douban/cora/pems08 without any
        # arbitration; airport (one strongly dominant branch) trails by ~0.06.
        S = torch.stack([get_S(m) for m in ["joena", "joena-pc", "joena-pc-gm"]]).mean(dim=0)
        print("  [ensemble] averaged 3 branch couplings")
    elif args.model == "auto":
        # Aligner-level arbitration over three branches, train anchors only.
        # Rule: (>= 20 anchors) keep branches whose chance-corrected agreement
        # is within 0.02 of the best; among those take the strict max raw
        # agreement; if raw also ties, prefer the most refined branch (the
        # anchors then genuinely cannot distinguish them). < 20 anchors ->
        # conservative default = joena (no validated blind signal there;
        # measured: 6 anchors mis-rank, and train-anchor MRR anti-correlates).
        from PlanetAlign.m2m_quotient import entity_anchor_agreement
        BRANCHES = ["joena", "joena-pc", "joena-pc-gm"]   # refined last
        branches = {}
        for m in BRANCHES:
            S_m = get_S(m)
            sc, pred, info = evaluate_quotient_blind(
                S_m, gt, g_src, g_tgt, metrics=M2M_METRICS,
                return_predictions=True, use_attr=use_attr, anchors=anchors)
            corr, raw = entity_anchor_agreement(pred, anchors, int(g_tgt.num_nodes))
            branches[m] = (S_m, sc, corr, raw)
            print(f"  [branch {m:<12}] MicroF1={sc['MicroF1']:.4f} "
                  f"anchor corrected={corr:.4f} raw={raw:.4f}")
        if anchors.shape[0] < 20:
            chosen = "joena"
        else:
            best_corr = max(b[2] for b in branches.values())
            tied = [m for m in BRANCHES if branches[m][2] >= best_corr - 0.02]
            best_raw = max(branches[m][3] for m in tied)
            tied = [m for m in tied if branches[m][3] >= best_raw - 1e-9]
            chosen = tied[-1]                              # most refined among full ties
        S = branches[chosen][0]
        print(f"  [auto] chose {chosen}")
    else:
        S = get_S(args.model)

    hits1 = float(__import__('PlanetAlign').metrics.hits_ks_scores(
        S, get_anchor_pairs(ds.test_data, GIDS[0], GIDS[1]), ks=[1], mode="mean").get(1, 0.0))
    print(f"Hits@1={hits1:.4f}")
    anchors = get_anchor_pairs(ds.train_data, GIDS[0], GIDS[1])
    print(f"train_anchors={anchors.shape[0]}")

    records: List[Dict[str, Any]] = []

    def add(name: str, scores: Dict[str, float], extra: Dict[str, Any] = None, dt: float = 0.0):
        rec = {"readout": name, **scores, "decode_s": round(dt, 2), **(extra or {})}
        records.append(rec)
        print(f"  {name:<28} MicroF1={scores['MicroF1']:.4f} MSF1={scores['MSF1']:.4f} "
              f"ACS={scores['ACS']:.4f} ({dt:.1f}s)")

    # 1) attribute-cohesion blind (the diagnostic's default readout)
    t = time.perf_counter()
    sc = evaluate_blind(S, gt, g_src, g_tgt, metrics=M2M_METRICS, use_attr=use_attr)
    add("attr-blind", sc, dt=time.perf_counter() - t)

    # 2) JOENAGroupDecode readout (fixed tau 0.1, greedy)
    t = time.perf_counter()
    sc, counts = profile_greedy_scores(S, gt, g_src, g_tgt, tau=0.1)
    add("profile-greedy(tau=0.1)", sc, {"groups": counts}, time.perf_counter() - t)

    # 3) full quotient decode (anchor-arbitrated evidence selection)
    t = time.perf_counter()
    sc, pred, info = evaluate_quotient_blind(S, gt, g_src, g_tgt, metrics=M2M_METRICS,
                                             return_predictions=True, use_attr=use_attr,
                                             anchors=anchors)
    add("quotient(full)", sc, {"info": info}, time.perf_counter() - t)

    if not args.skip_ablations:
        for name, kw in [
            ("quotient - hungarian",    dict(matcher="greedy", anchors=anchors)),
            ("quotient - anti-chain",   dict(anti_chaining=False, anchors=anchors)),
            ("quotient - iteration",    dict(max_iters=1, anchors=anchors)),
            ("quotient - otsu(tau=.1)", dict(tau=0.1, anchors=anchors)),
            ("quotient - evidence-sel", dict(evidence_selection=False, anchors=anchors)),
            ("quotient - anchors",      dict(anchors=None)),
            ("quotient - null-cand",    dict(null_candidate=False, anchors=anchors)),
            ("quotient + global-cand",  dict(global_candidate=True, anchors=anchors)),
            ("quotient + overlap(.35)", dict(overlap_expand_tau=0.35, anchors=anchors)),
        ]:
            t = time.perf_counter()
            sc, _, info = evaluate_quotient_blind(S, gt, g_src, g_tgt, metrics=M2M_METRICS,
                                                  return_predictions=True, use_attr=use_attr, **kw)
            add(name, sc, {"info": info}, time.perf_counter() - t)

    out = args.out or Path(f"logs/m2m_diag/quotient_compare_{args.dataset}{model_tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "root": str(args.root), "seed": args.seed,
                   "epochs": args.epochs, "records": records}, f, ensure_ascii=False, indent=2,
                  default=str)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
