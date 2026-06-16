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

Evaluate a node-level similarity matrix:

```python
from PlanetAlign.m2m import evaluate_similarity

scores, pred_entities = evaluate_similarity(
    similarity=S,
    gt_entities=gt_entities,
    return_predictions=True,
)
```

Default metrics:

```text
ACS
MSF1
MicroF1
M2M-SGS
M2M-EGS
```

## Baseline Interface

For a new baseline, inherit `ManyToManyBaseline`, implement `train()`, and set
`self.S` to a dense similarity matrix with shape:

```text
[num_source_nodes, num_target_nodes]
```

Example:

```python
import torch
from PlanetAlign.m2m import ManyToManyBaseline


class MyM2MBaseline(ManyToManyBaseline):
    def train(self, dataset, gids, use_attr=True, **kwargs):
        graph1 = dataset.pyg_graphs[gids[0]]
        graph2 = dataset.pyg_graphs[gids[1]]

        self.S = torch.zeros(graph1.num_nodes, graph2.num_nodes)
        return self.S, None
```

Then evaluate:

```python
model = MyM2MBaseline().to("cpu")
model.train(dataset, gids=[0, 1])
scores = model.test_many_to_many(gt_entities)
```

If your method directly predicts groups, override `predict_many_to_many()`.
