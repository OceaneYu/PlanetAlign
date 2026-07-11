"""Random-matrix-theory diagnosis of the alignment matrix S.

Questions this script answers with numbers (GT is used for *diagnosis only*,
never inside any candidate algorithm step):

Q1  Spectrum: does S decompose into a Marchenko-Pastur-like noise bulk plus
    signal spikes, and does the spike count (Gavish-Donoho threshold) track the
    true entity count?
Q2  Profile-cosine separation: how well do within-group vs cross-group adjacent
    pairs separate, against a random-pair null, for four profile variants:
    raw S rows | SVD-denoised | target-graph smoothed | denoised+smoothed?
Q3  Informativeness test: can the adjacent-vs-null comparison *detect* the Cora
    failure mode (profiles carry no group signal) without ground truth?

Outputs a compact table per dataset and caches S to logs/m2m_diag/S_cache/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

import torch
import torch.nn.functional as F

from PlanetAlign.algorithms import JOENA
from PlanetAlign.data import Dataset

GIDS = [0, 1]
CACHE = Path("logs/m2m_diag/S_cache")


# ---------------------------------------------------------------------------
def get_S(root: Path, name: str, epochs: int, seed: int, train_ratio: float) -> Tuple[torch.Tensor, Dataset]:
    ds = Dataset(root=root, name=name, train_ratio=train_ratio, seed=seed)
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = CACHE / f"{name}_e{epochs}_s{seed}.pt"
    if tag.exists():
        S = torch.load(tag, map_location="cpu", weights_only=True)
        print(f"  loaded cached S {list(S.shape)} from {tag}")
        return S, ds
    torch.manual_seed(seed)
    use_attr = all(g.x is not None for g in (ds.pyg_graphs[0], ds.pyg_graphs[1]))
    t0 = time.perf_counter()
    joena = JOENA(alpha=0.7).to("cpu")
    joena.train(dataset=ds, gids=GIDS, use_attr=use_attr, total_epochs=epochs,
                save_log=False, verbose=False)
    S = joena.S.detach().to(torch.float32).cpu()
    torch.save(S, tag)
    print(f"  trained JOENA in {time.perf_counter()-t0:.1f}s; cached to {tag}")
    return S, ds


# ---------------------------------------------------------------------------
def gavish_donoho_threshold(svals: torch.Tensor, m: int, n: int) -> float:
    """Optimal hard threshold for singular values, unknown noise (GD 2014).

    tau* = omega(beta) * median(singular values), beta = min/max dimension
    ratio, with the standard polynomial approximation of omega.
    """
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return float(omega * svals.median())


def spectrum_report(S: torch.Tensor) -> Dict[str, float]:
    m, n = S.shape
    svals = torch.linalg.svdvals(S)
    tau = gavish_donoho_threshold(svals, m, n)
    spikes = int((svals > tau).sum())
    total2 = float((svals ** 2).sum())
    erank = float((svals.sum() ** 2) / total2)                # participation ratio
    stable_rank = float(total2 / (svals[0] ** 2))
    signal_energy = float((svals[svals > tau] ** 2).sum() / total2)
    return {"svals": svals, "gd_tau": tau, "spikes": spikes, "erank": round(erank, 1),
            "stable_rank": round(stable_rank, 1), "signal_energy": round(signal_energy, 4)}


def denoise(S: torch.Tensor, rank: int) -> torch.Tensor:
    U, sv, Vh = torch.linalg.svd(S, full_matrices=False)
    r = max(1, min(rank, sv.shape[0]))
    return (U[:, :r] * sv[:r]) @ Vh[:r]


def norm_adj(graph) -> torch.Tensor:
    """Symmetric-normalized (A + I) as a dense filter matrix."""
    n = int(graph.num_nodes)
    A = torch.zeros(n, n)
    ei = graph.edge_index
    A[ei[0], ei[1]] = 1.0
    A = ((A + A.T) > 0).float() + torch.eye(n)
    d = A.sum(dim=1).clamp(min=1.0).pow(-0.5)
    return d.unsqueeze(1) * A * d.unsqueeze(0)


# ---------------------------------------------------------------------------
def adjacent_pairs(graph) -> Tuple[torch.Tensor, torch.Tensor]:
    ei = graph.edge_index
    src, dst = ei[0], ei[1]
    keep = src < dst
    return src[keep], dst[keep]


def pair_cos(P: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    Pn = F.normalize(P, p=2, dim=1)
    return (Pn[u] * Pn[v]).sum(dim=1)


def rank_auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """AUC of separating pos from neg by value (Mann-Whitney)."""
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    all_vals = torch.cat([pos, neg])
    ranks = all_vals.argsort().argsort().float() + 1
    rpos = ranks[: pos.numel()]
    auc = (rpos.sum() - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel())
    return float(auc)


def profile_separation(P: torch.Tensor, graph, entity_sets: List[set], num_null: int = 20000,
                       seed: int = 0) -> Dict[str, float]:
    """Within/cross/null cosine stats for one profile matrix (GT for labels only)."""
    u, v = adjacent_pairs(graph)
    cos_adj = pair_cos(P, u, v)
    within_mask = torch.tensor([bool(entity_sets[int(a)] & entity_sets[int(b)])
                                for a, b in zip(u.tolist(), v.tolist())])
    within, cross = cos_adj[within_mask], cos_adj[~within_mask]

    g = torch.Generator().manual_seed(seed)
    n = P.shape[0]
    i = torch.randint(0, n, (num_null,), generator=g)
    j = torch.randint(0, n, (num_null,), generator=g)
    keep = i != j
    null = pair_cos(P, i[keep], j[keep])

    q999 = float(torch.quantile(null, 0.999))
    frac_adj_above = float((cos_adj > q999).float().mean())
    return {
        "within_med": round(float(within.median()), 4) if within.numel() else float("nan"),
        "cross_med": round(float(cross.median()), 4) if cross.numel() else float("nan"),
        "null_med": round(float(null.median()), 4),
        "null_q999": round(q999, 4),
        "auc_within_vs_cross": round(rank_auc(within, cross), 4),
        "frac_adj_above_null_q999": round(frac_adj_above, 4),
        "n_within": int(within.numel()), "n_cross": int(cross.numel()),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.2)
    args = p.parse_args()

    targets = [
        (Path("data/m2m_overlap_0.05"), "douban_m2m"),
        (Path("data/m2m_no_overlap"), "cora_m2m"),
    ]
    report = {}
    for root, name in targets:
        print("=" * 90)
        print(f"Dataset: {name}")
        S, ds = get_S(root, name, args.epochs, args.seed, args.train_ratio)
        gt = json.load(open(root / f"{name}_gt_many2many.json"))["entities"]
        n1, n2 = S.shape

        # GT node->entity sets, per side (diagnosis only).
        src_sets = [set() for _ in range(n1)]
        tgt_sets = [set() for _ in range(n2)]
        for eid, e in gt.items():
            for x in e.get("src", []):
                src_sets[int(x)].add(eid)
            for y in e.get("tgt", []):
                tgt_sets[int(y)].add(eid)

        # Q1: spectrum
        spec = spectrum_report(S)
        print(f"  Q1 spectrum: spikes(GD)={spec['spikes']}  true_entities={len(gt)}  "
              f"erank={spec['erank']}  stable_rank={spec['stable_rank']}  "
              f"signal_energy={spec['signal_energy']}")

        # profile variants
        t0 = time.perf_counter()
        S_dn = denoise(S, spec["spikes"])
        A2 = norm_adj(ds.pyg_graphs[GIDS[1]])
        A1 = norm_adj(ds.pyg_graphs[GIDS[0]])
        variants_src = {
            "raw": S,
            "denoised": S_dn,
            "smoothed": S @ A2,
            "dn+smooth": S_dn @ A2,
        }
        variants_tgt = {
            "raw": S.T.contiguous(),
            "denoised": S_dn.T.contiguous(),
            "smoothed": S.T.contiguous() @ A1,
            "dn+smooth": S_dn.T.contiguous() @ A1,
        }
        print(f"  variants built in {time.perf_counter()-t0:.1f}s")

        ds_rep = {"spectrum": {k: v for k, v in spec.items() if k != "svals"},
                  "true_entities": len(gt), "src": {}, "tgt": {}}
        for side, variants, graph, sets_ in [
            ("src", variants_src, ds.pyg_graphs[GIDS[0]], src_sets),
            ("tgt", variants_tgt, ds.pyg_graphs[GIDS[1]], tgt_sets),
        ]:
            print(f"  Q2/Q3 [{side}]  {'variant':<10}{'within':>8}{'cross':>8}{'null':>8}"
                  f"{'nullq999':>9}{'AUC':>7}{'fracAdj>q':>10}")
            for vname, P in variants.items():
                r = profile_separation(P, graph, sets_)
                ds_rep[side][vname] = r
                print(f"           {vname:<12}{r['within_med']:>8.4f}{r['cross_med']:>8.4f}"
                      f"{r['null_med']:>8.4f}{r['null_q999']:>9.4f}"
                      f"{r['auc_within_vs_cross']:>7.3f}{r['frac_adj_above_null_q999']:>10.4f}")
        report[name] = ds_rep

    out = Path("logs/m2m_diag/rmt_diagnosis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
