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

By default, `evaluate_predictions()` first relabels predicted group ids by
node-overlap with GT groups. This makes the metric adapter invariant to a pure
cluster-id permutation. Pass `match_entities=False` if you need strict eid
matching.

Evaluate a node-level similarity matrix with the blind quotient decoder:

```python
from PlanetAlign.m2m.quotient import evaluate_quotient_blind
from PlanetAlign.utils import get_anchor_pairs

anchors = get_anchor_pairs(dataset.train_data, 0, 1)
scores, pred_entities, decode_info = evaluate_quotient_blind(
    S,
    gt_entities,
    dataset.pyg_graphs[0],
    dataset.pyg_graphs[1],
    anchors=anchors,
    return_predictions=True,
)
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
scores = evaluate_quotient_blind(
    model.S,
    gt_entities,
    dataset.pyg_graphs[0],
    dataset.pyg_graphs[1],
    anchors=anchors,
)
```

If a method directly predicts groups, evaluate that explicit entity map with
`evaluate_predictions()`.
