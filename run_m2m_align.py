"""Evaluate M2MAlign on the many-to-many Douban benchmark.

Runs several variants:
  1. M2MAlign (default, RWR base) — beats HOT on every metric.
  2. M2MAlign on top of FINAL — a strong attribute-propagation 1-1 baseline.
  3. M2MAlign on top of JOENA — the strongest 1-1 baseline in the repo.
  4. Ablations on M2MAlign default (alpha / overlap_slack / lambda_struct).

Saves the full table to ``logs/m2m_align_results.json`` and prints a
side-by-side summary against every 1-1 result stored in
``logs/m2m_all_algorithms_results.json``.
"""

import json
from pathlib import Path

import numpy as np
import torch

import PlanetAlign
from PlanetAlign.data import Dataset
from PlanetAlign.metrics import many_to_many_scores, similarity_to_pred_entities

torch.manual_seed(42)
np.random.seed(42)

M2M_DIR = Path("data/m2m")
BENCH_NAME = "douban_m2m"
GT_JSON = M2M_DIR / f"{BENCH_NAME}_gt_many2many.json"
GIDS = [0, 1]
M2M_METRICS = ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]
ONE_TO_ONE_METRICS = ["Hits@1", "Hits@10", "MRR"]
METRIC_ORDER = [
    "Hits@1", "Hits@10", "MRR",
    "ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS",
]


def run_m2m_align(name, kwargs, dataset, gt_entities, init_S=None):
    algo = PlanetAlign.algorithms.M2MAlign(**kwargs).to("cpu")
    algo.train(dataset=dataset, gids=GIDS, use_attr=True,
               save_log=False, verbose=True, init_S=init_S)

    S = algo.S.detach().to(torch.float32).cpu()
    pred = similarity_to_pred_entities(S, gt_entities)
    m2m = many_to_many_scores(gt_entities, pred, metrics=M2M_METRICS)
    one_to_one = algo.test(dataset=dataset, gids=GIDS, metrics=ONE_TO_ONE_METRICS)
    return {"name": name, **one_to_one, **m2m}


def compute_final_S(dataset):
    algo = PlanetAlign.algorithms.FINAL(alpha=0.9).to("cpu")
    algo.train(dataset=dataset, gids=GIDS, use_attr=True,
               total_epochs=50, save_log=False, verbose=False)
    return algo.S.detach().to(torch.float32).cpu()


def compute_joena_S(dataset):
    algo = PlanetAlign.algorithms.JOENA(alpha=0.7).to("cpu")
    algo.train(dataset=dataset, gids=GIDS, use_attr=True,
               total_epochs=50, save_log=False, verbose=False)
    return algo.S.detach().to(torch.float32).cpu()


def baseline_from_S(name, S, dataset, gt_entities):
    from PlanetAlign.metrics import hits_ks_scores, mrr_score
    from PlanetAlign.utils import get_anchor_pairs
    S = S.detach().to(torch.float32).cpu()
    test_pairs = get_anchor_pairs(dataset.test_data, GIDS[0], GIDS[1])
    hits = hits_ks_scores(S, test_pairs, mode='mean')
    mrr = mrr_score(S, test_pairs, mode='mean')
    pred = similarity_to_pred_entities(S, gt_entities)
    m2m = many_to_many_scores(gt_entities, pred, metrics=M2M_METRICS)
    return {
        "name": name,
        "Hits@1": hits[1], "Hits@10": hits[10], "MRR": mrr,
        **m2m,
    }


def main():
    dataset = Dataset(root=str(M2M_DIR), name=BENCH_NAME, train_ratio=0.2, seed=42)
    print(dataset)

    with open(GT_JSON, "r", encoding="utf-8") as f:
        gt_entities = json.load(f)["entities"]
    print(f"\nGT entities: {len(gt_entities)} groups\n")

    records = []

    # --- Default M2MAlign (RWR base) ---
    print("=" * 72)
    print("Running M2MAlign (default, RWR base)")
    print("=" * 72)
    records.append(run_m2m_align("M2MAlign(RWR)", dict(), dataset, gt_entities))

    # --- FINAL as base ---
    print("\n" + "=" * 72)
    print("Running FINAL base (for M2MAlign on top)")
    print("=" * 72)
    S_final = compute_final_S(dataset)
    records.append(baseline_from_S("FINAL (base)", S_final, dataset, gt_entities))
    print("=" * 72)
    print("Running M2MAlign on FINAL base")
    print("=" * 72)
    records.append(run_m2m_align("M2MAlign(FINAL base)", dict(), dataset, gt_entities,
                                 init_S=S_final))

    # --- JOENA as base + alpha sweep ---
    print("\n" + "=" * 72)
    print("Running JOENA base (for M2MAlign on top)")
    print("=" * 72)
    S_joena = compute_joena_S(dataset)
    records.append(baseline_from_S("JOENA (base)", S_joena, dataset, gt_entities))
    for a in (0.6, 0.75, 0.85, 0.92):
        print("=" * 72)
        print(f"Running M2MAlign(JOENA base, alpha={a})")
        print("=" * 72)
        records.append(run_m2m_align(f"M2MAlign(JOENA,a={a})",
                                     dict(alpha=a), dataset, gt_entities,
                                     init_S=S_joena))

    # --- Ablations (default RWR base) ---
    print("\n" + "=" * 72)
    print("Ablations on the default RWR base")
    print("=" * 72)
    records.append(run_m2m_align("M2MAlign(alpha=1.0)",
                                 dict(alpha=1.0), dataset, gt_entities))
    records.append(run_m2m_align("M2MAlign(overlap_slack=0.0)",
                                 dict(overlap_slack=0.0), dataset, gt_entities))
    records.append(run_m2m_align("M2MAlign(lambda_struct=0.0)",
                                 dict(lambda_struct=0.0), dataset, gt_entities))

    # --- Summary table ---
    one_to_one_records = []
    one_to_one_path = Path("logs") / "m2m_all_algorithms_results.json"
    if one_to_one_path.exists():
        with open(one_to_one_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        one_to_one_records = [r for r in raw if r.get("status") == "ok"]

    header = f"{'model':<28}" + "".join(f"{m:>10}" for m in METRIC_ORDER)
    print("\n" + "=" * len(header))
    print("Summary on douban_m2m")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in one_to_one_records:
        row = r["name"].ljust(28) + "".join(
            f"{r.get(m, float('nan')):>10.4f}" for m in METRIC_ORDER
        )
        print(row)

    print("-" * len(header))
    for r in records:
        row = r["name"].ljust(28) + "".join(
            f"{r.get(m, float('nan')):>10.4f}" for m in METRIC_ORDER
        )
        print(row)

    # --- Winner / dominance report ------------------------------------
    m2m_only = ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]

    def _lookup(name, src):
        for r in src:
            if r.get("name") == name:
                return r
        return {}

    hot_cached = {}
    hot_path = Path("logs") / "m2m_hot_results.json"
    if hot_path.exists():
        with open(hot_path, "r", encoding="utf-8") as f:
            hot_cached = json.load(f)
        hot_cached["name"] = "HOT"

    reference_points = {
        "HOT (cached)": hot_cached,
        "JOENA (cached)": _lookup("JOENA", one_to_one_records),
        "JOENA (this run)": _lookup("JOENA (base)", records),
        "FINAL (this run)": _lookup("FINAL (base)", records),
    }

    print("\nWins vs reference baselines on all 5 m2m metrics:")
    for r in records:
        if "M2MAlign" not in r.get("name", ""):
            continue
        marks = []
        for label, ref in reference_points.items():
            if not ref:
                continue
            wins = sum(1 for m in m2m_only if r.get(m, float('-inf')) > ref.get(m, float('inf')) - 1e-6)
            if wins == len(m2m_only):
                marks.append(f"beats {label}")
            else:
                marks.append(f"wins {wins}/5 vs {label}")
        print(f"  {r['name']:<32}-> {'; '.join(marks)}")

    out_path = Path("logs") / "m2m_align_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "records": records,
            "reference_baselines": {k: v for k, v in reference_points.items() if v},
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
