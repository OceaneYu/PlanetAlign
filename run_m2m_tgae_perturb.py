"""Tune TGAE on a many-to-many graph perturbation/permutation benchmark.

This follows the original T-GAE graph-matching assumption more closely than
aligning the two real Douban domains directly: we take one many-to-many graph,
create a lightly perturbed and randomly permuted target graph, and transform
the many-to-many entity ground truth through the same permutation.

对原始的多对多图进行轻微的扰动，创建一个目标图，然后使用tgae算法学习从原始图到扰动图的映射。
"""

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import PlanetAlign
from PlanetAlign.data import BaseData, Dataset
from PlanetAlign.metrics import many_to_many_scores


M2M_DIR = Path("data/m2m")
BENCH_NAME = "douban_m2m"
GT_JSON = M2M_DIR / f"{BENCH_NAME}_gt_many2many.json"
OUT_PATH = Path("logs") / "m2m_tgae_perturb_results.json"

ONE_TO_ONE_METRICS = ["Hits@1", "Hits@10", "MRR"]
M2M_METRICS = ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]


def _undirected_pairs(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
    pairs = set()
    for u, v in edge_index.T.tolist():
        u = int(u)
        v = int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        pairs.add((a, b))
    return sorted(pairs)


def _pairs_to_edge_index(pairs: Iterable[Tuple[int, int]], num_nodes: int) -> torch.Tensor:
    edges = []
    for u, v in pairs:
        edges.append((u, v))
        edges.append((v, u))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).T
    return to_undirected(edge_index, num_nodes=num_nodes)


def _perturb_pairs(
    pairs: List[Tuple[int, int]],
    num_nodes: int,
    perturb_ratio: float,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    pair_set = set(pairs)
    num_edits = int(len(pair_set) * perturb_ratio)
    if num_edits == 0:
        return sorted(pair_set)

    num_delete = num_edits // 2
    num_add = num_edits - num_delete

    delete_edges = rng.sample(sorted(pair_set), k=min(num_delete, len(pair_set)))
    for edge in delete_edges:
        pair_set.remove(edge)

    added = 0
    max_tries = max(num_add * 100, 1000)
    tries = 0
    while added < num_add and tries < max_tries:
        tries += 1
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in pair_set:
            continue
        pair_set.add((a, b))
        added += 1

    return sorted(pair_set)


def build_permuted_m2m_dataset(
    perturb_ratio: float = 0.01,
    attr_noise: float = 0.0,
    train_ratio: float = 0.2,
    seed: int = 42,
    base_gid: int = 1,
) -> tuple[BaseData, Dict[str, Dict[str, List[int]]]]:
    source_dataset = Dataset(root=str(M2M_DIR), name=BENCH_NAME, train_ratio=0.2, seed=seed)
    source_graph = source_dataset.pyg_graphs[base_gid]
    num_nodes = source_graph.num_nodes

    rng = random.Random(seed)
    torch_rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=torch_rng)

    source_edge_index = to_undirected(source_graph.edge_index.cpu(), num_nodes=num_nodes)
    source_pairs = _undirected_pairs(source_edge_index)
    target_pairs = _perturb_pairs(source_pairs, num_nodes, perturb_ratio, rng)
    target_pairs = [(int(perm[u]), int(perm[v])) for u, v in target_pairs]
    target_edge_index = _pairs_to_edge_index(target_pairs, num_nodes)

    source_x = source_graph.x.detach().cpu() if source_graph.x is not None else None
    target_x = None
    if source_x is not None:
        target_x = torch.empty_like(source_x)
        target_x[perm] = source_x
        if attr_noise > 0:
            target_x = target_x + attr_noise * torch.randn(
                target_x.shape,
                generator=torch_rng,
                dtype=target_x.dtype,
            )

    graphs = [
        Data(name="m2m_source", num_nodes=num_nodes, x=source_x, edge_index=source_edge_index),
        Data(name="m2m_target_permuted", num_nodes=num_nodes, x=target_x, edge_index=target_edge_index),
    ]
    anchor_links = torch.stack([torch.arange(num_nodes), perm], dim=1)

    with open(GT_JSON, "r", encoding="utf-8") as f:
        source_entities = json.load(f)["entities"]
    entity_key = "src" if base_gid == 0 else "tgt"
    gt_entities = {}
    for eid, item in source_entities.items():
        src_nodes = [int(n) for n in item[entity_key] if 0 <= int(n) < num_nodes]
        if not src_nodes:
            continue
        gt_entities[eid] = {
            "src": src_nodes,
            "tgt": [int(perm[n]) for n in src_nodes],
        }

    dataset = BaseData(
        graphs=graphs,
        anchor_links=anchor_links,
        name=f"{BENCH_NAME}_perturb_{perturb_ratio}",
        train_ratio=train_ratio,
        seed=seed,
    )
    return dataset, gt_entities


def evaluate_config(dataset, gt_entities, config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    model = PlanetAlign.algorithms.TGAE(
        num_hidden_layers=config["num_hidden_layers"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        lr=config["lr"],
        anchor_loss_weight=config["anchor_loss_weight"],
        anchor_temperature=config["anchor_temperature"],
        similarity=config["similarity"],
        reconstruction_neg_ratio=config["reconstruction_neg_ratio"],
    ).to("cpu")
    model.train(
        dataset=dataset,
        gids=[0, 1],
        use_attr=True,
        total_epochs=config["epochs"],
        eval_interval=config["eval_interval"],
        save_log=False,
        verbose=False,
    )
    one_to_one = model.test(dataset=dataset, gids=[0, 1], metrics=ONE_TO_ONE_METRICS)

    pred_local = model.predict_many_to_many(
        gt_entities,
        mode="local_expand",
        relax_ratio=config["relax_ratio"],
        target_dup_sim_ratio=config["target_dup_sim_ratio"],
        max_extra_targets=config["max_extra_targets"],
    )
    pred_topk = model.predict_many_to_many(gt_entities, mode="gt_size_topk")
    m2m_local = many_to_many_scores(gt_entities, pred_local, metrics=M2M_METRICS)
    m2m_topk = many_to_many_scores(gt_entities, pred_topk, metrics=M2M_METRICS)

    return {
        "config": config,
        **one_to_one,
        "local_expand": m2m_local,
        "gt_size_topk": m2m_topk,
    }


def main():
    base = {
        "seed": 42,
        "num_hidden_layers": 4,
        "hidden_dim": 32,
        "output_dim": 32,
        "lr": 1e-3,
        "anchor_temperature": 0.07,
        "similarity": "cosine",
        "reconstruction_neg_ratio": 1.0,
        "epochs": 30,
        "eval_interval": 30,
        "relax_ratio": 1.05,
        "target_dup_sim_ratio": 1.05,
        "max_extra_targets": 4,
    }

    experiments = []
    for perturb_ratio in (0.0, 0.01, 0.03):
        dataset, gt_entities = build_permuted_m2m_dataset(
            perturb_ratio=perturb_ratio,
            attr_noise=0.0,
            train_ratio=0.2,
            seed=42,
            base_gid=1,
        )
        for anchor_loss_weight in (0.0, 0.5, 1.0):
            config = {
                **base,
                "perturb_ratio": perturb_ratio,
                "anchor_loss_weight": anchor_loss_weight,
            }
            print(
                f"Running perturb={perturb_ratio:.2f} "
                f"anchor_loss={anchor_loss_weight:.2f}",
                flush=True,
            )
            result = evaluate_config(dataset, gt_entities, config)
            experiments.append(result)
            print(
                "  "
                f"Hits@1={result['Hits@1']:.4f} "
                f"Hits@10={result['Hits@10']:.4f} "
                f"MRR={result['MRR']:.4f} "
                f"MSF1(topk)={result['gt_size_topk']['MSF1']:.4f} "
                f"M2M-SGS(topk)={result['gt_size_topk']['M2M-SGS']:.4f}",
                flush=True,
            )

    best = max(
        experiments,
        key=lambda r: (
            r["gt_size_topk"]["MSF1"],
            r["Hits@1"],
            r["gt_size_topk"]["M2M-SGS"],
        ),
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"best": best, "experiments": experiments}, f, indent=2)

    print("\nBest config:")
    print(json.dumps(best["config"], indent=2))
    print("\nBest one-to-one:")
    for key in ONE_TO_ONE_METRICS:
        print(f"  {key:<8} {best[key]:.4f}")
    print("\nBest many-to-many gt_size_topk:")
    for key, value in best["gt_size_topk"].items():
        print(f"  {key:<8} {value:.4f}")
    print("\nBest many-to-many local_expand:")
    for key, value in best["local_expand"].items():
        print(f"  {key:<8} {value:.4f}")
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
