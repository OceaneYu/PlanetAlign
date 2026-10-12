## External baseline workflow

- External one-to-one graph-alignment baselines are qualified with the repo-local `baseline-reference` skill.
- Reference qualification must keep all task writes under `.reference/<baseline>/`; do not modify host project source during that phase.
- The current executable PlanetAlign one-to-one loader/split/runner/evaluator is the canonical protocol. Do not change protocol semantics to accommodate a baseline.
- Formal integration is a separate action and must use the explicitly invoked `$baseline-integrate` skill.
- Never integrate a baseline merely because reference qualification succeeded.
- Formal implementations must be self-contained and must not import from `.reference`.
