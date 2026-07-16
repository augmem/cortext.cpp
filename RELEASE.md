# Cortext v1.2.2 Release Artifact

This tracked artifact accompanies the `v1.2.2` tag.

## Release State

- Version: `1.2.2`
- Merged feature baseline before release metadata: `0298ce679876b27ed42925cbdfe4adc7d7c55279`
- Reviewed source tree: `ffdc8abdde25741103c4ee0fa6148f73d5879ee9`
- Product surface: C++ facade, C ABI, Python, JavaScript/TypeScript, Dart,
  Go, WASM, CLI, and Hermes provider integration.

## Highlights

- Replaces ambiguous consolidation booleans with one explicit consolidation
  state while preserving knob-derived behavior.
- Makes consolidation recommendations edge-triggered from persisted
  throughput state and F/S/T, without arbitrary count or elapsed-time
  triggers.
- Routes forced consolidation through maintenance-only processing so it does
  not create searchable input or perturb normal throughput observations.
- Keeps unchanged consolidation idempotent and protects original semantic
  long-term candidates from association and graph crowding.
- Repairs long-term retrieval recovery so SQL fallback paging preserves
  semantic-vector families and current reconstruction surfaces.
- Carries retrieval relevance through maintained bindings and keeps ephemeral
  queries durably non-mutating.

## Verification

- Pull requests [#3](https://github.com/augmem/cortext/pull/3) and
  [#4](https://github.com/augmem/cortext/pull/4) merged with all twelve
  exact-head GitHub checks successful and all review discussions resolved.
- The exact 1,915-message matched replay remained `2/7` top-12 needle hits in
  both control and recommendation-driven treatment; treatment made one
  throughput-derived consolidation call and preserved all 1,915 long-term
  memories and current embeddings with zero ephemeral durable delta.
- The registered 18-pair normal-path artifact passed all eight zero-margin
  latency, throughput, RSS, and database-size gates.
- Direct failed-cache retrieval improved materially at the measured 1,915-row
  and 8,000-row populations. These measurements do not claim production
  latency, seven-of-seven recall, or unbounded high-cardinality scaling.

## Packaging

- PyPI and npm artifacts contain six native targets each.
- Registry packages retain on-demand model bootstrap and do not embed the
  AIST model payload.

## Version Surfaces

The release surfaces report `1.2.2`:

- `CMakeLists.txt`
- `build.zig` and `build.zig.zon`
- `bindings/python/pyproject.toml`
- `bindings/javascript/package.json`
- `bindings/dart/pubspec.yaml`

## Publish Targets

- GitHub release: `v1.2.2`
- PyPI: `augmem.cortext==1.2.2`
- npm: `@augmem/cortext@1.2.2`
