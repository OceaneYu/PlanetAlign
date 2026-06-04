"""Build a many-to-many alignment benchmark from the Douban 1-1 dataset.

Usage:
    python run_m2m_benchmark.py

Outputs (under ./data/m2m/):
    douban_m2m.pt               <- PlanetAlign.data.Dataset compatible
    douban_m2m_gt_many2many.json <- ACS / MSF1 / MicroF1 / M2M-SGS / M2M-EGS GT

The script also runs a sanity-check by computing the many-to-many metrics
for two baseline predictions (perfect and empty) to verify the pipeline.
"""

import json
from pathlib import Path

import torch

import PlanetAlign
from PlanetAlign.utils import build_many_to_many_benchmark
from PlanetAlign.metrics import many_to_many_scores

OUT_DIR = Path("data/m2m")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the 1-1 source dataset.
dataset = PlanetAlign.datasets.Douban(
    root="data/",
    download=False,
    train_ratio=0.2,
    seed=42,
)
print(dataset)

# 2. Build the many-to-many benchmark.
m2m = build_many_to_many_benchmark(
    dataset=dataset,
    gids=(0, 1),
    split_ratios={"1-1": 0.70, "1-many": 0.10, "many-1": 0.10, "many-many": 0.10},
    max_expansion=4,          # up to 4 split nodes per side
    internal_density=0.8,     # intra-group edge probability
    overlap_ratio=0.05,       # 5% of nodes may appear in multiple groups
    anchor_source="test",     # use test anchors as the entity pool
    train_anchor_source="train",  # keep original 1-1 train anchors usable
    name="douban_m2m",
    seed=42,
)

pt_path, json_path = m2m.save(OUT_DIR)
print(f"\nSaved many-to-many benchmark:")
print(f"  graphs + anchors  -> {pt_path}")
print(f"  ground-truth JSON -> {json_path}")

# 3. Report high-level stats.
meta = m2m.metadata
print("\n=== Benchmark stats ===")
print(f"Source dataset      : {meta['source_dataset']}")
print(f"Entity type counts  : {meta['entity_type_counts']}")
print(f"Total entities      : {meta['num_entities']}")
print(f"Overlap insertions  : {meta['overlap_insertions']} (overlap_ratio={meta['overlap_ratio']})")
print(f"Src graph nodes     : {meta['orig_num_nodes'][0]} -> {meta['new_num_nodes'][0]}")
print(f"Tgt graph nodes     : {meta['orig_num_nodes'][1]} -> {meta['new_num_nodes'][1]}")
print(f"Src graph edges     : {m2m.graphs[0].edge_index.size(1)}")
print(f"Tgt graph edges     : {m2m.graphs[1].edge_index.size(1)}")

# 4. Sanity check: compute m2m metrics for a perfect and an empty prediction.
gt = m2m.entities
with open(json_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
assert payload["entities"] == gt

perfect_pred = {eid: dict(ent) for eid, ent in gt.items()}
empty_pred = {eid: {"src": [], "tgt": []} for eid in gt}

print("\n=== Metric sanity check (perfect vs empty predictions) ===")
perfect_scores = many_to_many_scores(gt, perfect_pred)
empty_scores = many_to_many_scores(gt, empty_pred)
print(f"{'Metric':<12}{'Perfect':>12}{'Empty':>12}")
for k in ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]:
    print(f"{k:<12}{perfect_scores[k]:>12.4f}{empty_scores[k]:>12.4f}")
# Note: when overlap_ratio > 0, M2M-EGS cannot reach 1.0 even for a perfect
# prediction, because a shared node makes group g intersect the cluster of
# another group h. That is the intended signal of fuzzy boundaries.

# 5. Extra sanity: rebuild with overlap_ratio=0 and confirm all 5 metrics
# reach 1.0 under the perfect prediction.
no_overlap = build_many_to_many_benchmark(
    dataset=dataset,
    gids=(0, 1),
    split_ratios={"1-1": 0.70, "1-many": 0.10, "many-1": 0.10, "many-many": 0.10},
    max_expansion=4,
    internal_density=0.8,
    overlap_ratio=0.0,
    anchor_source="test",
    seed=42,
)
clean_gt = no_overlap.entities
clean_perfect = many_to_many_scores(clean_gt, {e: dict(v) for e, v in clean_gt.items()})
print("\n=== No-overlap benchmark (perfect prediction should score 1.0 everywhere) ===")
for k in ["ACS", "MSF1", "MicroF1", "M2M-SGS", "M2M-EGS"]:
    print(f"{k:<12}{clean_perfect[k]:>12.4f}")

# 6. Show a few example entities to make the construction tangible.
print("\n=== Sample entities ===")
sample_keys = list(gt.keys())[:5]
for eid in sample_keys:
    print(f"  {eid}: src={gt[eid]['src']}  tgt={gt[eid]['tgt']}")
