# Group-JOENA

Group-JOENA is a many-to-many graph alignment baseline for node-splitting
benchmarks. It is designed for cases where one real entity may appear as
different numbers of nodes across two graphs.

## Interface

The algorithm follows the existing PlanetAlign model interface:

```python
from PlanetAlign.algorithms import GroupJOENA
from PlanetAlign.m2m import load_m2m_benchmark

bench = load_m2m_benchmark("data/m2m_no_overlap", "pems08_m2m")

model = GroupJOENA(
    num_groups=len(bench.entities),
    # This is an oracle-K benchmark setting because the true entity count is used.
    # Do not report it as fully unsupervised.
    max_epochs=5,
    hidden_dim=32,
    out_dim=32,
).to("cpu")

model.fit(
    dataset=bench.dataset,
    gids=[0, 1],
    use_attr=True,
    num_groups=len(bench.entities),
    oracle_num_groups=True,
    save_log=False,
)

prediction = model.align()  # GT-free group prediction

# Evaluation adapter: GT is used only here.
pred_entities = model.predict_many_to_many(bench.entities, target_size_mode="group")
scores = model.test_many_to_many(bench.entities)
```

`train()` is still available as the PlanetAlign compatibility wrapper and
delegates to `fit()`.

## Algorithm

Group-JOENA uses a group-mediated factorization:

```text
S = U_s @ T @ U_t.T
```

where:

- `U_s`: soft source node-to-group assignment;
- `U_t`: soft target node-to-group assignment;
- `T`: group-to-group transport matrix;
- `S`: final node-level many-to-many score matrix.

The current implementation has four stages:

1. **Node encoding**: reuse JOENA-style RWR features, optional node attributes,
   and a shared MLP encoder.
2. **Soft group assignment**: assign nodes to latent groups with softmax or
   sparsemax over learned prototypes.
3. **Split-invariant quotient graph**: aggregate node embeddings into group
   embeddings and aggregate edges with `soft_or` or `normalized_sum`.
4. **Group-level alignment**: align quotient graphs with a weighted fused
   Gromov-Wasserstein objective over group features and quotient adjacencies,
   then compute `S = U_s @ T @ U_t.T`.

## Losses

The training objective is modular:

```text
L =
    w_align * L_group_align
  + w_supervised * L_supervised
  + w_cohesion * L_cohesion
  + w_reconstruction * L_reconstruction
  + w_sparse * L_sparse
  + w_separation * L_separation
```

Default `supervised_weight` is `0.0`. Supervised loss is enabled only when
`train_entities` is passed explicitly. `gt_entities` is no longer used to
resolve group count or run inference.

## Run Commands

Quick smoke run:

```bash
conda run -n m2m python -u scripts/run_m2m_experiments.py \
  --m2m-root data/m2m_no_overlap \
  --datasets pems08_m2m \
  --algorithms GroupJOENA \
  --profile quick
```

Run with selected baselines:

```bash
conda run -n m2m python -u scripts/run_m2m_experiments.py \
  --m2m-root data/m2m_no_overlap \
  --datasets pems08_m2m airport_m2m cora_m2m \
  --algorithms GroupJOENA JOENA PARROT NetTrans M2MAlign \
  --profile quick
```

## Current Assumptions

- First version targets `m2m_no_overlap`.
- `num_groups` should be supplied explicitly. If it is the true benchmark
  entity count, pass `oracle_num_groups=True`.
- If neither is provided, the model falls back to the number of training
  anchors and emits a warning.
- The current output uses hard no-overlap group ids at inference time.
- Current generated benchmarks do not carry a many-to-many entity
  train/validation/test split. The usual `.train_data`/`.test_data` split is
  the original one-to-one anchor split, so Group-JOENA smoke scores should not
  be described as many-to-many generalization unless such an entity split is
  added.

## Limitations

- Full dense `S` is materialized, so very large graph pairs may require more
  memory.
- Overlap-aware multi-membership inference is not implemented yet.
- True entity-count `num_groups` is an oracle benchmark setting. For a fully
  unsupervised setting, add a group-count estimator or tune `num_groups`
  without looking at test GT.
- Quick profile is for correctness/smoke testing, not final performance.

## Phase A Audit

Run the deterministic correctness audit:

```bash
conda run -n m2m python scripts/group_joena_phase_a_audit.py
```

Include the local `pems08_m2m` quick smoke if the generated data exists:

```bash
conda run -n m2m python scripts/group_joena_phase_a_audit.py --run-pems08
```
