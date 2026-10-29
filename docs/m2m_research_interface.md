# Many-to-Many Graph Alignment Research Interface

This note summarizes the reusable interfaces for many-to-many graph alignment
experiments in PlanetAlign.

## Data Loading

Generated many-to-many datasets use two files:

```text
{name}.pt
{name}_gt_many2many.json
```

Load both together:

```python
from PlanetAlign.m2m import load_m2m_benchmark

bench = load_m2m_benchmark(
    root="data/m2m_no_overlap",
    name="cora_m2m",
    train_ratio=0.2,
    seed=42,
)

dataset = bench.dataset
gt_entities = bench.entities
```

## Prediction Format

The canonical prediction format is:

```python
pred_entities = {
    "e0": {"src": [0, 1], "tgt": [5, 6]},
    "e1": {"src": [2], "tgt": [7, 8, 9]},
}
```

Prediction JSON files can be saved as:

```python
from PlanetAlign.m2m import save_entity_map

save_entity_map(
    pred_entities,
    "outputs/my_method_cora_m2m_predictions.json",
    dataset_name="cora_m2m",
    metadata={"method": "my_method"},
)
```

They can be loaded again with:

```python
from PlanetAlign.m2m import load_entity_map

pred_entities = load_entity_map("outputs/my_method_cora_m2m_predictions.json")
```

## Evaluation

Evaluate explicit group predictions:

```python
from PlanetAlign.m2m import evaluate_predictions

scores = evaluate_predictions(gt_entities, pred_entities)
```

MSF1 uses the raw groups and performs its own globally optimal matching,
independent of entity ids. The other metrics retain their existing entity-id
adapter; `match_entities=False` changes only those metrics. MSF1 defaults to
the non-overlapping contract. For an overlapping benchmark, explicitly pass
`allow_overlap=True` (the runner obtains this from `metadata.overlap_ratio`).

Evaluate a node-level similarity matrix with the blind quotient decoder:

```python
from PlanetAlign.m2m import evaluate_predictions
from PlanetAlign.m2m.quotient import quotient_decode
from PlanetAlign.utils import get_anchor_pairs

anchors = get_anchor_pairs(dataset.train_data, 0, 1)
pred_entities, decode_info = quotient_decode(
    S,
    dataset.pyg_graphs[0],
    dataset.pyg_graphs[1],
    anchors=anchors,
)
scores = evaluate_predictions(gt_entities, pred_entities)
```

`gt_entities` is used only after `quotient_decode()` has produced its groups,
to relabel and score the prediction. It does not enter group discovery or
matching.

Default metrics:

```text
Macro Set-F1
Micro Set-F1
AUIS
AUCS
```

`Macro Set-F1` is now **MSF1-v2**: globally matched bilateral unit-member F1, divided by the fixed number of true test units `K` (`MSF1-v2-K`). By default, same-side nodes shared by different prediction units raise `InvalidPredictionError`, including duplicates and out-of-scope groups. Overlapping benchmarks must explicitly pass `allow_overlap=True`; this separate mode retains its duplicate-prediction limitation. A prediction participates only if it contains a test GT node, but its entire source/target membership is retained, including wrong training/background members. Purely out-of-scope groups are ignored. MSF1 uses raw predictions; the other metrics retain their previous entity-id adaptation. Historical MSF1 results are not directly comparable. See [the exact protocol and examples](MSF1.md).

## Similarity-producing Baselines

An existing or new alignment model can participate by producing `self.S`, a
dense similarity matrix with shape:

```text
[num_source_nodes, num_target_nodes]
```

Train the model through its normal `BaseModel` interface, then pass the matrix
to the same blind decoder:

```python
model.train(dataset=dataset, gids=[0, 1])
pred_entities, decode_info = quotient_decode(
    model.S,
    dataset.pyg_graphs[0],
    dataset.pyg_graphs[1],
    anchors=anchors,
)
scores = evaluate_predictions(gt_entities, pred_entities)
```

If a method directly predicts groups, evaluate that explicit entity map with
`evaluate_predictions()`.
