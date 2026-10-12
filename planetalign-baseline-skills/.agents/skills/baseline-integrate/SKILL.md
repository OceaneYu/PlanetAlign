---
name: baseline-integrate
description: Formally integrate a previously qualified external graph-alignment baseline from .reference into PlanetAlign/algorithms as a self-contained implementation, then verify migration equivalence. Use only when the user explicitly invokes this skill for a named baseline after reviewing its reference report.
---

# Formal baseline integration

This is a separate, explicit workflow. Never use it as a continuation of `baseline-reference` unless the user explicitly invokes `$baseline-integrate`.

## Input

Required:
- the existing baseline name under `.reference/<baseline>/`.

Preflight:

```bash
python tools/baselinectl.py integration-preflight --name "<baseline>"
```

Read:
- `.reference/<baseline>/REPORT.md`
- `.reference/<baseline>/baseline_manifest.json`
- relevant source and adapter code.

If the reference status is `FAIL`, stop and explain why formal integration is unsafe.

`PARTIAL` may proceed only when the report contains enough evidence to trust the algorithm implementation and the user has explicitly requested integration.

## Integration objective

Create a self-contained PlanetAlign implementation under the project's existing algorithm architecture, typically:

```text
PlanetAlign/algorithms/<baseline>/
```

The formal implementation must:
- run without `.reference/`;
- not import or execute files from `.reference/`;
- preserve the verified external algorithm's semantics;
- conform to the current PlanetAlign algorithm/BaseModel interface;
- use the existing canonical one-to-one protocol unchanged;
- follow current repository conventions.

## Allowed host changes

Prefer changes inside the new algorithm directory.

Minimal edits outside it are allowed only when required by the existing architecture, for example:
- algorithm registry/export;
- canonical config registration;
- targeted tests.

Do not modify:
- existing baseline algorithms to accommodate the new method;
- canonical dataset semantics;
- evaluator metric semantics;
- train/test split semantics;
- test-GT visibility.

Keep every outside edit minimal and list it in the integration report.

## Migration procedure

1. Identify the smallest verified source subset required by the algorithm.
2. Migrate/restructure it into PlanetAlign style without depending on the original repository layout.
3. Preserve licenses/notices when required.
4. Implement the standard algorithm interface.
5. Add minimal tests for import/construction and the critical prediction path.
6. Run the same canonical benchmark setting that was used in `.reference`.
7. Compare pre-integration and post-integration results.

For deterministic methods under the same inputs/seed, investigate any material numerical difference.

For stochastic methods, compare under the same seed policy and use the reference run's observed variation where available.

Do not declare success while an unexplained migration regression remains.

## Verification

Run:

```bash
python tools/baselinectl.py verify-integration --name "<baseline>" --algorithm-path "PlanetAlign/algorithms/<baseline>"
```

This verifies that the formal implementation does not reference `.reference`.

Then run the repository's relevant tests and canonical one-to-one runner.

Write:

`.reference/<baseline>/integration/REPORT.md`

including:
- migrated components;
- files changed outside the algorithm directory;
- dependency changes;
- pre/post integration benchmark comparison;
- tests executed;
- remaining differences;
- final integration status.

Finalize:

```bash
python tools/baselinectl.py finalize-integration --name "<baseline>" --status PASS|PARTIAL|FAIL
```

Do not commit or push unless the user separately asks for that action.
