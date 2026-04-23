"""Evaluate every one-to-one PlanetAlign algorithm on the many-to-many benchmark.

For each algorithm:
    1. Load the M2M dataset (same .pt schema; anchor_links = 1-1 split + carried-over train anchors).
    2. algo.train(dataset, gids, ...) — captures the returned similarity/embeddings.
    3. Derive a full (n1, n2) similarity matrix S and assign algo.S so .test() works.
    4. 1-1 metrics: algo.test(metrics=['Hits@1','Hits@10','MRR']).
    5. M2M metrics: similarity_to_pred_entities(S, gt) -> many_to_many_scores.

Each algorithm is wrapped in try/except so one failure does not abort the sweep.
Most algorithms in this repo don't assign self.S in train() — we patch it here.
"""

import json
import time
import traceback
from pathlib import Path

import torch

import PlanetAlign
from PlanetAlign.data import Dataset
from PlanetAlign.utils import pairwise_cosine_similarity
from PlanetAlign.metrics import many_to_many_scores, similarity_to_pred_entities

M2M_DIR = Path("data/m2m")
BENCH_NAME = "douban_m2m"
GT_JSON = M2M_DIR / f"{BENCH_NAME}_gt_many2many.json"

GIDS = [0, 1]
ONE_TO_ONE_METRICS = ["Hits@1", "Hits@10", "MRR"]
M2M_METRICS = ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]


def _sim_from_embeddings(emb1, emb2):
    emb1 = emb1.detach().to(torch.float32).cpu()
    emb2 = emb2.detach().to(torch.float32).cpu()
    return pairwise_cosine_similarity(emb1, emb2)


def _sim_from_inner_product(emb1, emb2):
    return (emb1.detach().to(torch.float32).cpu()
            @ emb2.detach().to(torch.float32).cpu().T)


def _sim_from_hot(sim_tensor_dict, cluster_nodes_dict, n1, n2):
    """HOT returns per-cluster similarities + per-cluster node lists.
    Reassemble a dense (n1, n2) matrix by scattering each cluster's tensor."""
    S = torch.full((n1, n2), -1e9, dtype=torch.float32)
    for cid, sim_tensor in sim_tensor_dict.items():
        nodes = cluster_nodes_dict[cid]  # list[2] of per-graph node indices
        idx1 = nodes[0].long().cpu()
        idx2 = nodes[1].long().cpu()
        block = sim_tensor.detach().to(torch.float32).cpu()
        S[idx1.unsqueeze(1), idx2.unsqueeze(0)] = torch.maximum(
            S[idx1.unsqueeze(1), idx2.unsqueeze(0)], block
        )
    finite = torch.isfinite(S) & (S > -1e8)
    if finite.any():
        S[~finite] = S[finite].min().item() - 1.0
    else:
        S = torch.zeros_like(S)
    return S


# -- Strategy entries: (name, factory, train_kwargs, sim_mode) ---------------
# sim_mode tells us how to turn the return value of train() into a full (n1, n2) S.
#   'S'          -> return is (S, logger); S is already the similarity matrix
#   'embs-cos'   -> (emb1, emb2, logger); S = cosine similarity
#   'embs-dot'   -> (emb_dict, logger); S = emb_dict[gid1] @ emb_dict[gid2].T
#   'hot'        -> (sim_tensor_dict, logger); reassemble via cluster_nodes_dict

ALGO_CONFIGS = [
    ("IsoRank",   lambda: PlanetAlign.algorithms.IsoRank(alpha=0.4),     dict(use_attr=False, total_epochs=50), 'S'),
    ("FINAL",     lambda: PlanetAlign.algorithms.FINAL(alpha=0.9),       dict(use_attr=True,  total_epochs=50), 'self.S'),
    ("IONE",      lambda: PlanetAlign.algorithms.IONE(out_dim=100),      dict(use_attr=False, total_epochs=30), 'S'),
    ("REGAL",     lambda: PlanetAlign.algorithms.REGAL(),                dict(use_attr=True), 'embs-cos'),
    ("CrossMNA",  lambda: PlanetAlign.algorithms.CrossMNA(),             dict(use_attr=False, total_epochs=100), 'embs-dot-dict'),
    ("NetTrans",  lambda: PlanetAlign.algorithms.NetTrans(),             dict(use_attr=True,  total_epochs=30), 'embs-cos'),
    ("BRIGHT",    lambda: PlanetAlign.algorithms.BRIGHT(),               dict(use_attr=True,  total_epochs=100), 'embs-cos'),
    ("NeXtAlign", lambda: PlanetAlign.algorithms.NeXtAlign(),            dict(use_attr=True,  total_epochs=30), 'embs-cos'),
    ("PARROT",    lambda: PlanetAlign.algorithms.PARROT(alpha=0.5),      dict(use_attr=True), 'self.S'),
    ("SLOTAlign", lambda: PlanetAlign.algorithms.SLOTAlign(bases=4),     dict(use_attr=True,  total_epochs=200, joint_epochs=50), 'self.S'),
    ("WLAlign",   lambda: PlanetAlign.algorithms.WLAlign(),              dict(use_attr=False, total_epochs=30, struct_epochs=50), 'embs-cos'),
    ("WAlign",    lambda: PlanetAlign.algorithms.WAlign(),               dict(use_attr=True,  total_epochs=20), 'embs-cos'),
    ("HOT",       lambda: PlanetAlign.algorithms.HOT(alpha=0.5, lp=0.1), dict(use_attr=True,  in_iters=5, out_iters=30), 'hot'),
    ("JOENA",     lambda: PlanetAlign.algorithms.JOENA(alpha=0.7),       dict(use_attr=True,  total_epochs=50), 'self.S'),
    # DualMatch and MEAformer are empty stubs in this repo — skipped.
]


def _extract_S(algo, ret, mode, dataset, gids):
    """Return a 2D (n1, n2) float32 similarity tensor."""
    gid1, gid2 = gids
    n1 = dataset.pyg_graphs[gid1].num_nodes
    n2 = dataset.pyg_graphs[gid2].num_nodes

    if mode == 'self.S':
        S = algo.S
        if S is None:
            raise RuntimeError("algo.S is None after train()")
        return S.detach().to(torch.float32).cpu()

    if mode == 'S':
        S = ret[0]
        return S.detach().to(torch.float32).cpu()

    if mode == 'embs-cos':
        emb1, emb2 = ret[0], ret[1]
        return _sim_from_embeddings(emb1, emb2)

    if mode == 'embs-dot-dict':
        emb_dict = ret[0]
        return _sim_from_inner_product(emb_dict[gid1], emb_dict[gid2])

    if mode == 'hot':
        # HOT's train returns (sim_tensor_dict, logger) but cluster_nodes_dict is
        # local to train. We re-run the clustering by storing it as a side channel:
        # the HOT code keeps cluster_nodes_dict local, so we patch by wrapping train.
        raise RuntimeError("HOT requires a special wrapper — see _run_hot")

    raise ValueError(f"Unknown sim mode: {mode}")


def _run_hot(algo, dataset, gids, train_kwargs):
    """HOT keeps cluster_nodes_dict local to train(); monkey-patch to capture it."""
    import types

    captured = {}
    orig_train = algo.train

    def patched_train(self, dataset, gids, use_attr=True, in_iters=5, out_iters=50,
                      save_log=True, verbose=True):
        # Lift the internals by importing the source and running it with a hook.
        # Simpler: monkey-patch numpy sum to capture, but messy. Instead, parse
        # the result by re-running HOT's internal clustering here is overkill.
        # Workaround: temporarily replace the dict assignment by overriding
        # `torch.maximum` is not needed; instead, we rely on the fact that
        # sim_tensor_dict keys are cluster IDs and each sim_tensor has shape
        # matching the cluster node counts per graph. The cluster_nodes_dict is
        # computed from c[j][i]. Since HOT doesn't expose it, we replicate its
        # construction via the public hot.mot module.
        raise NotImplementedError

    # Instead of patching, just run train and derive S via a shortcut: HOT logs
    # hits/mrr during training using multi_align_hits_ks_scores. Under the hood
    # it concatenates per-cluster similarities mapped to global node indices.
    # We replicate that here by importing HOT internals.
    from PlanetAlign.algorithms.hot import main as hot_module  # noqa: F401
    # Fall back: just call train and accept that HOT requires a bespoke path.
    ret = algo.train(dataset=dataset, gids=gids, **train_kwargs)
    return ret


def evaluate_algorithm(name, factory, train_kwargs, mode, dataset, gt_entities):
    print("=" * 72)
    print(f"Running {name}  train_kwargs={train_kwargs}")
    print("=" * 72)
    t0 = time.time()
    algo = factory().to("cpu")

    ret = algo.train(dataset=dataset, gids=GIDS,
                     save_log=False, verbose=False, **train_kwargs)

    # Normalise return to a list.
    if not isinstance(ret, tuple):
        ret = (ret,)

    S = _extract_S(algo, ret, mode, dataset, GIDS)

    # Make .test() work regardless of what the algorithm did internally.
    algo.S = S.to(algo.device)

    one_to_one = algo.test(dataset=dataset, gids=GIDS, metrics=ONE_TO_ONE_METRICS)
    pred = similarity_to_pred_entities(S, gt_entities)
    m2m = many_to_many_scores(gt_entities, pred, metrics=M2M_METRICS)
    dt = time.time() - t0
    print(f"  [ok] {dt:.1f}s  1-1={one_to_one}  m2m={m2m}")
    return {"name": name, "time_s": dt, "status": "ok", **one_to_one, **m2m}


def main():
    dataset = Dataset(root=str(M2M_DIR), name=BENCH_NAME, train_ratio=0.2, seed=42)
    print(dataset)

    with open(GT_JSON, "r", encoding="utf-8") as f:
        payload = json.load(f)
    gt_entities = payload["entities"]
    print(f"\nGT entities loaded: {len(gt_entities)} groups from {GT_JSON}\n")

    records = []
    for name, factory, train_kwargs, mode in ALGO_CONFIGS:
        if mode == 'hot':
            # Skip HOT for now (needs internal cluster reassembly; tackle separately).
            print(f"Skipping {name} (needs bespoke cluster reassembly — see notes)")
            records.append({"name": name, "status": "skipped",
                            "error": "HOT outputs clustered sim_tensor_dict; wrapper TBD"})
            continue
        try:
            rec = evaluate_algorithm(name, factory, train_kwargs, mode, dataset, gt_entities)
        except Exception as e:
            dt = 0.0
            print(f"  [error] {type(e).__name__}: {e}")
            traceback.print_exc()
            rec = {"name": name, "status": "error", "error": f"{type(e).__name__}: {e}"}
        records.append(rec)

    # Print summary tables.
    print("\n" + "=" * 100)
    print("Summary: one-to-one metrics on M2M benchmark")
    print("=" * 100)
    h = f"{'Model':<12}{'Hits@1':>10}{'Hits@10':>10}{'MRR':>10}{'time(s)':>10}   status"
    print(h); print("-" * len(h))
    for r in records:
        if r.get("status") == "ok":
            print(f"{r['name']:<12}{r['Hits@1']:>10.4f}{r['Hits@10']:>10.4f}{r['MRR']:>10.4f}"
                  f"{r['time_s']:>10.1f}   ok")
        else:
            print(f"{r['name']:<12}{'-':>10}{'-':>10}{'-':>10}{'-':>10}   {r.get('status','?')}: {r.get('error','')[:80]}")

    print("\n" + "=" * 100)
    print("Summary: many-to-many metrics on M2M benchmark")
    print("=" * 100)
    h = f"{'Model':<12}{'ACS':>10}{'MSF1':>10}{'MicroF1':>10}{'M2M-SGS':>10}{'M2M-EGS':>10}   status"
    print(h); print("-" * len(h))
    for r in records:
        if r.get("status") == "ok":
            print(f"{r['name']:<12}{r['ACS']:>10.4f}{r['MSF1']:>10.4f}{r['MicroF1']:>10.4f}"
                  f"{r['M2M-SGS']:>10.4f}{r['M2M-EGS']:>10.4f}   ok")
        else:
            print(f"{r['name']:<12}{'-':>10}{'-':>10}{'-':>10}{'-':>10}{'-':>10}   {r.get('status','?')}")

    out_path = Path("logs") / "m2m_all_algorithms_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
