---
name: baseline-reference
description: Reproduce, verify, and benchmark an external one-to-one graph-alignment baseline from its GitHub repository and paper inside PlanetAlign. Use when the user provides a baseline repository/paper or asks to test, qualify, or archive a 1-to-1 baseline. This workflow is reference-only: never integrate into PlanetAlign/algorithms or modify host project source files.
---

# Baseline reference qualification

## Goal

Given:
- a GitHub repository URL for an external graph-alignment method; and
- the corresponding paper PDF or paper path/URL,

qualify the method as a PlanetAlign one-to-one baseline while keeping all baseline-specific work inside `.reference/<baseline>/`.

The workflow has two independent validation stages:

1. **Paper reproduction**: run the official implementation on an original paper setting and compare against the paper's reported result.
2. **PlanetAlign benchmark**: adapt the verified implementation inside `.reference` and evaluate it using the host repository's current canonical one-to-one protocol.

Finish by generating a standardized report. Do **not** integrate the method into `PlanetAlign/algorithms/`.

## Hard boundary

During this skill:

- All new or modified task files must be under `.reference/<baseline>/`.
- Do not modify `PlanetAlign/algorithms/`, runners, evaluators, dataset loaders, registries, configs, tests, or any other host source file.
- The host repository's canonical one-to-one loader/split/evaluator/runner is the source of truth. Adapt the external method to the protocol; never change the protocol to make the method fit.
- Test ground truth must not enter the prediction path unless the host canonical protocol explicitly allows it.
- If a host-source modification appears necessary, stop that route, document the incompatibility, and keep the reference result as `PARTIAL` rather than editing the host.
- Run the guard script after each major phase. If it reports a host write-boundary violation, stop and report it.

## Inputs

Required:
- `repo`: GitHub repository URL.
- `paper`: local PDF path or downloadable PDF URL.

Optional:
- `name`: explicit baseline name. Otherwise derive it from the repository slug.

Do not ask for experiment details that can be discovered from the paper, official repository, or current PlanetAlign code. Work autonomously unless a genuinely unavailable credential/private resource blocks progress.

## Initialize

From the PlanetAlign repository root:

```bash
python tools/baselinectl.py init --repo "<repo>" --paper "<paper>" [--name "<name>"]
```

Read the printed baseline directory and manifest. Resume an existing matching workspace instead of recreating it.

Expected workspace:

```text
.reference/<baseline>/
├── source/
├── paper/
├── patches/
├── reproduction/
│   ├── configs/
│   ├── logs/
│   └── results/
├── planetalign_benchmark/
│   ├── adapter/
│   ├── configs/
│   ├── logs/
│   └── results/
├── baseline_manifest.json
└── REPORT.md
```

## Phase 1 — Establish provenance

Record:

- paper title, authors, venue/year;
- official repository URL;
- cloned upstream commit hash;
- repository license if discoverable;
- relevant release/tag if the paper specifies one;
- environment requirements;
- paper datasets, splits, seeds, metrics, hyperparameters, preprocessing, and checkpoints relevant to the reported one-to-one results.

Treat the paper and official code as primary evidence. If README and paper conflict, note the conflict and inspect code/configs to determine what the released implementation actually does.

## Phase 2 — Extract paper claims

Create `reproduction/paper_claims.md` and a machine-readable result file such as `reproduction/paper_claims.json`.

For every result selected for reproduction, record:

- table/figure/section;
- dataset;
- metric;
- reported value;
- split/protocol;
- seed or aggregation;
- model/configuration;
- any ambiguity affecting comparability.

Prefer at least one central result whose original data/configuration is obtainable and whose metric semantics can be matched exactly.

## Phase 3 — Understand and smoke-test the official repository

Identify:

- executable entry points;
- expected data format;
- training/inference boundary;
- output similarity/alignment representation;
- dependencies and hardware assumptions.

Run the smallest meaningful smoke test before expensive experiments.

You may modify the cloned copy under `.reference/<baseline>/source/` for:
- environment compatibility;
- path fixes;
- API/version compatibility;
- reproducibility instrumentation;
- benchmark adapters.

Do not silently change the algorithm's mathematical objective or evaluation semantics.

When source changes are made, preserve them with:

```bash
python tools/baselinectl.py save-patch --name "<baseline>"
```

Explain algorithm-affecting changes separately in the report.

## Phase 4 — Reproduce an original paper result

Use the official data/config/checkpoint when available.

Save:
- exact commands;
- resolved configuration;
- stdout/stderr logs;
- environment/version information;
- raw and normalized result files.

Compare reproduced and paper-reported values explicitly.

Classify the reproduction:

- `PASS`: at least one key paper setting is semantically comparable and the reproduced result agrees within the paper's stated uncertainty, or within about 1 percentage point absolute when the paper provides no uncertainty and no better method-specific tolerance is available.
- `PARTIAL`: the implementation is demonstrably functional, but only a subset/different environment/config is reproducible, or a key result differs beyond the default tolerance with a plausible documented explanation.
- `FAIL`: no trustworthy semantically valid reproduction/sanity check can be obtained, or the discrepancy is severe and unexplained.

Do not upgrade a result to `PASS` merely because the program exits successfully.

Record the phase:

```bash
python tools/baselinectl.py phase --name "<baseline>" --phase reproduction --status PASS|PARTIAL|FAIL --summary "<short summary>"
python tools/baselinectl.py guard --name "<baseline>"
```

A reproduction `FAIL` may still be analyzed, but do not present the method as a verified baseline.

## Phase 5 — Discover PlanetAlign's canonical one-to-one protocol

Inspect the current host repository and identify the actual canonical:

- dataset loader/preprocessing;
- train/test anchor split;
- random seed handling;
- algorithm interface/BaseModel contract;
- prediction representation;
- evaluator and tie handling;
- metrics;
- runner;
- result/logging format.

Prefer executable code over stale documentation.

Write the discovery to:

`planetalign_benchmark/PROTOCOL.md`

Do not hard-code historical PlanetAlign protocol assumptions from this skill.

## Phase 6 — Build an isolated PlanetAlign adapter

Implement any adapter only under:

`planetalign_benchmark/adapter/`

The adapter may import stable host loaders/evaluators, but the host project must not import the adapter.

Requirements:

- preserve the external algorithm's intended semantics;
- translate PlanetAlign inputs into the baseline's native form;
- translate baseline outputs into PlanetAlign's canonical prediction form;
- do not leak test GT into inference;
- do not patch the host evaluator;
- explicitly mark unsupported datasets/required modalities rather than inventing substitute inputs.

First run a small smoke test, then run the canonical one-to-one benchmark on all applicable datasets.

After the benchmark:

```bash
python tools/baselinectl.py phase --name "<baseline>" --phase planetalign_benchmark --status PASS|PARTIAL|FAIL --summary "<short summary>"
python tools/baselinectl.py guard --name "<baseline>"
python tools/baselinectl.py save-patch --name "<baseline>"
```

## Phase 7 — Compare against existing baselines

Only compare numbers that share the same current canonical protocol.

If existing baseline results cannot be established as protocol-compatible, state that they are not directly comparable rather than mixing tables.

Report, when available:
- canonical one-to-one metrics;
- runtime;
- peak GPU memory or resource notes;
- seed/variance information;
- unsupported datasets and why.

## Phase 8 — Generate REPORT.md

Use the template in `references/report-template.md`.

The report must answer:

1. What method/paper/repository is this?
2. What one-to-one results does the paper report?
3. What exact official implementation/configuration was tested?
4. What was reproduced?
5. What is the reproduction gap and `PASS/PARTIAL/FAIL` judgement?
6. What source changes were required?
7. What is the current PlanetAlign canonical protocol?
8. What are the PlanetAlign one-to-one benchmark results?
9. How do protocol-compatible existing baselines compare?
10. What runtime/resource/compatibility issues exist?
11. Is this method recommended for formal integration, and why?

The final recommendation is advisory only.

Finalize:

```bash
python tools/baselinectl.py finalize-reference --name "<baseline>" --status PASS|PARTIAL|FAIL
python tools/baselinectl.py guard --name "<baseline>"
```

## Stop condition

Stop after the reference report is complete.

Do not copy, migrate, register, or integrate the algorithm into `PlanetAlign/algorithms/`, even when the result is `PASS`.

If the user later wants formal integration, they must explicitly invoke `$baseline-integrate` for the named baseline.
