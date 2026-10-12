# PlanetAlign baseline qualification skills

This bundle implements two separate Codex skills:

- `$baseline-reference`: fully automated external baseline reproduction + isolated PlanetAlign one-to-one benchmarking under `.reference/`.
- `$baseline-integrate`: explicit-only formal migration into `PlanetAlign/algorithms/`.

The separation is intentional. A successful reference qualification never automatically triggers formal integration.

## Install into PlanetAlign

Copy these paths into the PlanetAlign repository root:

```text
.agents/skills/baseline-reference/
.agents/skills/baseline-integrate/
tools/baselinectl.py
```

Optionally merge `AGENTS.baseline-snippet.md` into the repository's root `AGENTS.md`.

Make the helper executable if desired:

```bash
chmod +x tools/baselinectl.py
```

## Normal use

Reference qualification can be invoked explicitly:

```text
$baseline-reference

repo: https://github.com/ORG/REPO
paper: /path/to/paper.pdf
```

Because the reference skill permits implicit invocation, a clear request such as "测试这个 baseline，repo 是 ...，论文是 ..." can also match it.

Formal integration is deliberately explicit-only:

```text
$baseline-integrate REPO
```

The integration skill's `agents/openai.yaml` sets:

```yaml
policy:
  allow_implicit_invocation: false
```

## State model

Reference:
`IN_PROGRESS -> PASS | PARTIAL | FAIL`

Integration:
`NOT_STARTED -> PASS | PARTIAL | FAIL`

A `FAIL` reference is blocked by the integration preflight. A `PARTIAL` reference can be integrated only after explicit invocation and review of its documented limitations.

## Important implementation notes

- `baselinectl.py` uses only the Python standard library plus the system `git` executable.
- `.reference/` is added to `.git/info/exclude` locally so cloned repositories and logs do not pollute normal `git status`; the root `.gitignore` is not modified by initialization.
- The reference guard snapshots pre-existing host working-tree changes and flags new/changed host paths outside `.reference/`.
- The guard intentionally stops and reports violations; it does not auto-revert files, because auto-revert could destroy unrelated user changes.
- The integration verifier rejects source files/symlinks that depend on `.reference`.

## Recommended first validation

Before trusting the workflow on a real baseline, run it once with a small known baseline and verify:

1. cloning/paper copy;
2. paper-result extraction;
3. original reproduction;
4. isolated canonical benchmark;
5. host-write guard;
6. report generation;
7. explicit integration gate;
8. pre/post migration equivalence.
