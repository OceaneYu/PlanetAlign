"""Evaluate JOENA on the many-to-many Douban benchmark with the new M2M metrics."""

import json
from pathlib import Path

import torch

import PlanetAlign
from PlanetAlign.data import Dataset
from PlanetAlign.metrics import many_to_many_scores, similarity_to_pred_entities

M2M_DIR = Path("data/m2m")
BENCH_NAME = "douban_m2m"
GT_JSON = M2M_DIR / f"{BENCH_NAME}_gt_many2many.json"
GIDS = [0, 1]
M2M_METRICS = ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]
ONE_TO_ONE_METRICS = ["Hits@1", "Hits@10", "MRR"]


def main():
    dataset = Dataset(root=str(M2M_DIR), name=BENCH_NAME, train_ratio=0.2, seed=42)
    print(dataset)

    with open(GT_JSON, "r", encoding="utf-8") as f:
        gt_entities = json.load(f)["entities"]
    print(f"\nGT entities: {len(gt_entities)} groups")

    joena = PlanetAlign.algorithms.JOENA(alpha=0.7).to("cpu")
    joena.train(dataset=dataset, gids=GIDS, use_attr=True,
                total_epochs=100, save_log=False, verbose=True)

    S = joena.S.detach().to(torch.float32).cpu()
    pred = similarity_to_pred_entities(S, gt_entities)
    m2m = many_to_many_scores(gt_entities, pred, metrics=M2M_METRICS)
    one_to_one = joena.test(dataset=dataset, gids=GIDS, metrics=ONE_TO_ONE_METRICS)

    print("\n" + "=" * 60)
    print("JOENA on douban_m2m (1-1 reference metrics)")
    print("=" * 60)
    for k in ONE_TO_ONE_METRICS:
        print(f"  {k:<10}{one_to_one[k]:.4f}")

    print("\n" + "=" * 60)
    print("JOENA on douban_m2m (new many-to-many metrics)")
    print("=" * 60)
    for k in M2M_METRICS:
        print(f"  {k:<10}{m2m[k]:.4f}")

    out_path = Path("logs") / "m2m_joena_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({**one_to_one, **m2m}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
