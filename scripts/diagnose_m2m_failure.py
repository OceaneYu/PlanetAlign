"""Diagnose *why* one-to-one network-alignment algorithms fail on many-to-many.

This script produces the empirical evidence behind ``docs/m2m_why_one_to_one_fails.md``.
For a representative algorithm from each PlanetAlign family

    * consistency : IsoRank, FINAL
    * embedding   : REGAL, BRIGHT
    * OT          : PARROT, JOENA

it trains the model on a generated M2M benchmark, takes the node-level
similarity / transport matrix ``S`` that every model exposes, and runs five
probes that each isolate one failure mode:

P1 representation gap   ``S`` is a single (n1, n2) cross-graph block; it has no
                        slot for *intra-graph* co-reference, which 49% of GT
                        entities require.
P2 one-to-one ceiling   a per-source-node argmax readout (the native pipeline)
                        can recover at most min(|src_e|,|tgt_e|) of an entity's
                        |tgt_e| targets, so target-set micro-recall is capped
                        BEFORE any learning. Reported alongside actual Hits@1.
P3 OT marginal probe    OT couplings are (near) doubly-stochastic: every column
                        absorbs ~equal mass, so the many-to-one that N-1 entities
                        need is forbidden. Measured by column-sum dispersion and
                        argmax-target sharing, contrasted with the other families.
P4 leak vs blind        the same ``S`` scored with the GT-leaking adapter vs the
                        blind protocol; the gap shows reported M2M numbers for
                        1-1 methods are inflated by the leak.
P5 blind M2M metrics    the honest ACS/MSF1/MicroF1/M2M-SGS/M2M-EGS per method.

Every number written into the doc must come from this script's JSON output.

Usage
-----
    python scripts/diagnose_m2m_failure.py
    python scripts/diagnose_m2m_failure.py --datasets douban_m2m --algorithms IsoRank JOENA
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

import PlanetAlign
from PlanetAlign.data import Dataset
from PlanetAlign.metrics import many_to_many_scores, similarity_to_pred_entities
from PlanetAlign.m2m_blind import evaluate_blind
from PlanetAlign.utils import pairwise_cosine_similarity


GIDS = [0, 1]
ONE_TO_ONE_METRICS = ["Hits@1", "Hits@10", "MRR"]
M2M_METRICS = ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]

# (root, name) pairs. douban carries fuzzy boundaries (overlap_ratio=0.05);
# cora is the no-overlap variant. Both are attributed so use_attr methods run.
DEFAULT_DATASETS: List[Tuple[str, str]] = [
    ("data/m2m_overlap_0.05", "douban_m2m"),
    ("data/m2m_no_overlap", "cora_m2m"),
]


def algo_configs() -> Dict[str, Dict[str, Any]]:
    """Representative method per family, with the quick-profile train kwargs."""
    return {
        "IsoRank": dict(family="consistency",
                        factory=lambda: PlanetAlign.algorithms.IsoRank(alpha=0.4),
                        train_kwargs=dict(use_attr=False, total_epochs=10), mode="S"),
        "FINAL": dict(family="consistency",
                      factory=lambda: PlanetAlign.algorithms.FINAL(alpha=0.9),
                      train_kwargs=dict(use_attr=True, total_epochs=10), mode="self.S"),
        "REGAL": dict(family="embedding",
                      factory=lambda: PlanetAlign.algorithms.REGAL(),
                      train_kwargs=dict(use_attr=True), mode="embs-cos"),
        "BRIGHT": dict(family="embedding",
                       factory=lambda: PlanetAlign.algorithms.BRIGHT(),
                       train_kwargs=dict(use_attr=True, total_epochs=10), mode="embs-cos"),
        "PARROT": dict(family="ot",
                       factory=lambda: PlanetAlign.algorithms.PARROT(alpha=0.5),
                       train_kwargs=dict(use_attr=True), mode="self.S"),
        "JOENA": dict(family="ot",
                      factory=lambda: PlanetAlign.algorithms.JOENA(alpha=0.7),
                      train_kwargs=dict(use_attr=True, total_epochs=10), mode="self.S"),
    }


# ---------------------------------------------------------------------------
# S extraction (mirrors scripts/run_m2m_experiments.py)
# ---------------------------------------------------------------------------
def extract_s(algo: Any, ret: Tuple[Any, ...], mode: str, gids: List[int]) -> torch.Tensor:
    gid1, gid2 = gids
    if mode == "self.S":
        if algo.S is None:
            raise RuntimeError("algorithm did not populate self.S")
        return algo.S.detach().to(torch.float32).cpu()
    if mode == "S":
        return ret[0].detach().to(torch.float32).cpu()
    if mode == "embs-cos":
        e1 = ret[0].detach().to(torch.float32).cpu()
        e2 = ret[1].detach().to(torch.float32).cpu()
        return pairwise_cosine_similarity(e1, e2)
    raise ValueError(f"unknown mode {mode}")


def has_attrs(dataset: Dataset) -> bool:
    return all(dataset.pyg_graphs[g].x is not None for g in GIDS)


# ---------------------------------------------------------------------------
# Probe 1 + 2: ground-truth-level facts (method independent)
# ---------------------------------------------------------------------------
def gt_structure(gt_entities: Dict[str, Dict[str, List[int]]]) -> Dict[str, Any]:
    arity = Counter()
    needs_intra_merge = 0           # entity has >1 node on some side (intra-graph co-reference)
    sum_b = 0                       # total target memberships
    sum_min_ab = 0                  # max recoverable by per-source-node argmax
    sum_a = 0
    sum_min_ab_src = 0              # symmetric ceiling on the source side
    for e in gt_entities.values():
        a, b = len(e.get("src", [])), len(e.get("tgt", []))
        typ = "1-1" if a == 1 and b == 1 else ("1-N" if a == 1 else ("N-1" if b == 1 else "N-N"))
        arity[typ] += 1
        if a > 1 or b > 1:
            needs_intra_merge += 1
        sum_b += b
        sum_a += a
        sum_min_ab += min(a, b)
        sum_min_ab_src += min(a, b)
    n = len(gt_entities)
    return {
        "num_entities": n,
        "arity": dict(arity),
        "entities_needing_intra_graph_coreference": needs_intra_merge,
        "frac_needing_intra_graph_coreference": round(needs_intra_merge / max(n, 1), 4),
        # A per-source-node argmax readout names <= a distinct targets for an
        # entity, so it recovers <= min(a, b) of its b targets. This bounds the
        # target-set micro-recall of ANY one-to-one node readout, oracle included.
        "argmax_readout_microrecall_ceiling": round(sum_min_ab / max(sum_b, 1), 4),
        "argmax_readout_microrecall_ceiling_src": round(sum_min_ab_src / max(sum_a, 1), 4),
    }


# ---------------------------------------------------------------------------
# Probe 3: marginal / one-to-one structure of S
# ---------------------------------------------------------------------------
def marginal_probe(S: torch.Tensor) -> Dict[str, Any]:
    S = S.detach().to(torch.float32).cpu()
    n1, n2 = S.shape

    # (a) argmax-target sharing: how many source nodes collapse onto each target.
    tgt = S.argmax(dim=1)
    mult = torch.bincount(tgt, minlength=n2)
    shared = mult[mult >= 2]
    frac_sources_on_shared = float((mult[tgt] >= 2).float().mean())
    sharing = {
        "max_sources_per_target": int(mult.max()),
        "num_targets_with_ge2_sources": int((mult >= 2).sum()),
        "frac_sources_landing_on_shared_target": round(frac_sources_on_shared, 4),
        "num_distinct_targets_used": int((mult > 0).sum()),
    }

    # (b) doubly-stochastic-ness of the (non-negative part of the) plan.
    # For an OT coupling this is near-uniform by construction (mass conservation);
    # for an unconstrained similarity it is not. CoV = std/mean of the marginals.
    P = S.clamp(min=0.0)
    total = float(P.sum())
    if total > 0:
        P = P / total
        rsum, csum = P.sum(dim=1), P.sum(dim=0)
        def cov(x):
            m = float(x.mean())
            return round(float(x.std()) / m, 4) if m > 0 else None
        # mean per-row entropy normalised to [0,1]; 1.0 == perfectly spread.
        rown = P / P.sum(dim=1, keepdim=True).clamp(min=1e-12)
        ent = -(rown * (rown + 1e-12).log()).sum(dim=1)
        marg = {
            "row_sum_cov": cov(rsum),
            "col_sum_cov": cov(csum),
            "mean_row_entropy_norm": round(float((ent / math.log(n2)).mean()), 4),
        }
    else:
        marg = {"row_sum_cov": None, "col_sum_cov": None, "mean_row_entropy_norm": None}

    return {"argmax_sharing": sharing, "marginal": marg}


# ---------------------------------------------------------------------------
# Per-algorithm run
# ---------------------------------------------------------------------------
def run_one(dataset_name: str, dataset: Dataset, gt_entities, algo_name: str,
            cfg: Dict[str, Any], seed: int, verbose: bool) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"dataset": dataset_name, "algorithm": algo_name,
                           "family": cfg["family"], "status": "pending"}
    torch.manual_seed(seed)
    tk = dict(cfg["train_kwargs"])
    if tk.get("use_attr") and not has_attrs(dataset):
        tk["use_attr"] = False
        rec["attr_note"] = "use_attr disabled (no node attributes)"

    started = time.perf_counter()
    try:
        algo = cfg["factory"]().to("cpu")
        ret = algo.train(dataset=dataset, gids=GIDS, save_log=False, verbose=verbose, **tk)
        if not isinstance(ret, tuple):
            ret = (ret,)
        S = extract_s(algo, ret, cfg["mode"], GIDS)
        algo.S = S.to(algo.device)
        rec["S_shape"] = list(S.shape)
        rec["S_is_square"] = bool(S.shape[0] == S.shape[1])

        # P2: actual one-to-one retrieval metrics from the native pipeline.
        rec["one_to_one"] = algo.test(dataset=dataset, gids=GIDS, metrics=ONE_TO_ONE_METRICS)

        # P3: marginal / sharing structure.
        rec["marginal_probe"] = marginal_probe(S)

        # P4: leak vs blind M2M metrics on the SAME S.
        leak_pred = similarity_to_pred_entities(S, gt_entities)
        rec["m2m_leak"] = many_to_many_scores(gt_entities, leak_pred, metrics=M2M_METRICS)
        rec["m2m_blind"] = evaluate_blind(
            S, gt_entities, dataset.pyg_graphs[GIDS[0]], dataset.pyg_graphs[GIDS[1]],
            metrics=M2M_METRICS, use_attr=tk.get("use_attr", False),
        )
        rec["status"] = "ok"
        rec["time_s"] = round(time.perf_counter() - started, 2)
    except Exception as exc:  # keep the sweep alive
        rec["status"] = "error"
        rec["time_s"] = round(time.perf_counter() - started, 2)
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
    return rec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=None,
                   help="dataset names to keep (default: douban_m2m cora_m2m)")
    p.add_argument("--algorithms", nargs="+", default=None,
                   help="algorithm names to keep (default: all representatives)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.2)
    p.add_argument("--out", type=Path, default=Path("logs/m2m_diag/diagnose_m2m_failure.json"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configs = algo_configs()
    if args.algorithms:
        keep = {a.lower() for a in args.algorithms}
        configs = {k: v for k, v in configs.items() if k.lower() in keep}
    datasets = DEFAULT_DATASETS
    if args.datasets:
        keep = set(args.datasets)
        datasets = [(r, n) for (r, n) in DEFAULT_DATASETS if n in keep]

    report: Dict[str, Any] = {"seed": args.seed, "train_ratio": args.train_ratio, "datasets": {}}

    for root, name in datasets:
        print("=" * 88)
        print(f"Dataset: {name}  (root={root})")
        ds = Dataset(root=root, name=name, train_ratio=args.train_ratio, seed=args.seed)
        gt = json.load(open(Path(root) / f"{name}_gt_many2many.json"))["entities"]
        struct = gt_structure(gt)
        nodes = [int(ds.pyg_graphs[g].num_nodes) for g in GIDS]
        print(f"  nodes={nodes}  entities={struct['num_entities']}  arity={struct['arity']}")
        print(f"  needs intra-graph co-reference: "
              f"{struct['entities_needing_intra_graph_coreference']}/{struct['num_entities']} "
              f"({struct['frac_needing_intra_graph_coreference']:.0%})")
        print(f"  argmax-readout micro-recall ceiling (tgt side): "
              f"{struct['argmax_readout_microrecall_ceiling']:.3f}")

        ds_report: Dict[str, Any] = {"nodes": nodes, "gt_structure": struct, "algorithms": {}}
        for algo_name, cfg in configs.items():
            print(f"  [run] {algo_name} ({cfg['family']}) ...", flush=True)
            rec = run_one(name, ds, gt, algo_name, cfg, args.seed, args.verbose)
            ds_report["algorithms"][algo_name] = rec
            if rec["status"] == "ok":
                o, leak, blind = rec["one_to_one"], rec["m2m_leak"], rec["m2m_blind"]
                shar = rec["marginal_probe"]["argmax_sharing"]
                print(f"        ok {rec['time_s']:.1f}s  Hits@1={o['Hits@1']:.4f}  "
                      f"MicroF1 leak={leak['MicroF1']:.4f} / blind={blind['MicroF1']:.4f}  "
                      f"max_src_per_tgt={shar['max_sources_per_target']}")
            else:
                print(f"        {rec['status']}: {rec.get('error')}")
        report["datasets"][name] = ds_report

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
