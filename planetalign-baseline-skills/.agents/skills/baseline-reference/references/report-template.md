# <Baseline> baseline qualification report

## 1. Identity and provenance

- Method:
- Paper:
- Authors:
- Venue / year:
- Official repository:
- Upstream commit:
- License:
- Paper file:
- Qualification date:

## 2. Method summary

Briefly describe the algorithm, required inputs, supervision, training/inference procedure, and output alignment representation.

## 3. Paper-reported one-to-one results

| Source | Dataset | Protocol | Metric | Reported | Seed / aggregation | Notes |
|---|---|---|---:|---:|---|---|

## 4. Official implementation reconstruction

- Entry point:
- Data:
- Config:
- Dependencies:
- Hardware:
- Exact commands:
- Ambiguities / paper-code discrepancies:

## 5. Paper reproduction

| Dataset | Metric | Paper | Reproduced | Abs. gap | Comparable? |
|---|---:|---:|---:|---:|---|

**Reproduction status:** PASS / PARTIAL / FAIL

Rationale:

Evidence:
- logs:
- configs:
- raw results:

## 6. Changes to the official repository copy

Separate:
- environment/path/API compatibility changes;
- instrumentation;
- PlanetAlign adapter changes;
- any algorithm-affecting changes.

Patch:
`patches/codex.patch`

## 7. PlanetAlign canonical one-to-one protocol

Summarize the protocol discovered from the current host code. Point to the authoritative loader, runner, evaluator, and metric definitions.

## 8. PlanetAlign benchmark

| Dataset | Metric(s) | Result | Runtime | Resource notes | Status |
|---|---|---:|---:|---|---|

Unsupported cases:

## 9. Protocol-compatible baseline comparison

Only include results verified to use the same current protocol.

| Method | Dataset | Metric | Result | Evidence |
|---|---|---:|---:|---|

## 10. Issues and limitations

- reproducibility:
- dependency/runtime:
- data availability:
- algorithm assumptions:
- benchmark compatibility:

## 11. Recommendation

**Reference qualification:** PASS / PARTIAL / FAIL

**Recommend formal PlanetAlign integration:** YES / CONDITIONAL / NO

Reason:

Formal integration has **not** been performed by this workflow.
