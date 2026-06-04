"""Evaluate TGAE on the PlanetAlign Douban many-to-many benchmark."""

import json
from pathlib import Path

import torch

import PlanetAlign
from PlanetAlign.data import Dataset
from PlanetAlign.metrics import many_to_many_scores


M2M_DIR = Path("data/m2m")
BENCH_NAME = "douban_m2m"
GT_JSON = M2M_DIR / f"{BENCH_NAME}_gt_many2many.json"
GIDS = [0, 1]
ONE_TO_ONE_METRICS = ["Hits@1", "Hits@10", "MRR"]
M2M_METRICS = ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]


def main():
    torch.manual_seed(42)
    dataset = Dataset(root=str(M2M_DIR), name=BENCH_NAME, train_ratio=0.2, seed=42)
    with open(GT_JSON, "r", encoding="utf-8") as f:
        gt_entities = json.load(f)["entities"]

    model = PlanetAlign.algorithms.TGAE(
        num_hidden_layers=4,
        hidden_dim=16,
        output_dim=16,
        lr=1e-3,
    ).to("cpu")
    model.train(
        dataset=dataset,
        gids=GIDS,
        use_attr=True,
        total_epochs=20,
        eval_interval=5,
        save_log=False,
        verbose=True,
    )

    one_to_one = model.test(dataset=dataset, gids=GIDS, metrics=ONE_TO_ONE_METRICS)
    pred_local = model.predict_many_to_many(
        gt_entities,
        mode="local_expand",
        relax_ratio=1.03,
        target_dup_sim_ratio=1.03,
        max_extra_targets=3,
    )
    pred_topk = model.predict_many_to_many(gt_entities, mode="gt_size_topk")
    m2m_local = many_to_many_scores(gt_entities, pred_local, metrics=M2M_METRICS)
    m2m_topk = many_to_many_scores(gt_entities, pred_topk, metrics=M2M_METRICS)

    result = {
        "name": "TGAE",
        **one_to_one,
        "local_expand": m2m_local,
        "gt_size_topk": m2m_topk,
    }

    out_path = Path("logs") / "m2m_tgae_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nTGAE one-to-one:")
    for key, value in one_to_one.items():
        print(f"  {key:<8} {value:.4f}")

    print("\nTGAE many-to-many (local_expand):")
    for key, value in m2m_local.items():
        print(f"  {key:<8} {value:.4f}")

    print("\nTGAE many-to-many (gt_size_topk, comparable to other run_m2m_* scripts):")
    for key, value in m2m_topk.items():
        print(f"  {key:<8} {value:.4f}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
