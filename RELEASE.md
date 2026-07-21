# Cortext v1.2.3 Release Artifact

This tracked artifact accompanies the `v1.2.3` tag.

## Release State

- Version: `1.2.3`
- Merged feature baseline before release metadata: `2353640ed3b71b1d57074a54ee348d018cb2bacf`
- Release source: the integrated commit addressed by the annotated `v1.2.3`
  tag; the GitHub release and registry artifacts are rebuilt from that exact
  commit.
- Product surface: C++ facade, C ABI, Python, JavaScript/TypeScript, Dart,
  Go, WASM, CLI, and Hermes provider integration.

## Highlights

- Keeps Natural and Durable ingestion on one SQLite-authoritative operation
  path; Durable adds only its post-commit flush/checkpoint barrier.
- Cuts the existing SQLite HNSW route over to an F/S/T-derived bounded
  retrieval sawtooth with a separately bounded consolidation activation
  snapshot.
- Rearms consolidation hints after material cumulative throughput drawdown
  while keeping stable jitter quiet.
- Preserves exact SQL fallback when sparse candidate generation underfills
  after eligibility or family filtering, and repairs restart, eviction,
  suppression, supersession, and canonical-entry edge cases.
- Keeps all routing and work-budget behavior modality- and opaque
  `source_id`-agnostic without changing the public API or C API.

## Verification

- Pull request [#6](https://github.com/augmem/cortext/pull/6) merged at
  `2353640ed3b71b1d57074a54ee348d018cb2bacf` with all review discussions
  resolved and cross-platform native, AIST, WASM, Zig, Windows CLI, and
  Linux sanitizer validation owned by exact-head GitHub checks.
- The exact Debug LLVM 21 media-enabled suite passed 39,305 assertions in 712
  cases across four disjoint shards at fixed seed 424242; Python passed 159
  tests plus 55 subtests.
- The production-shaped nine-point matrix passed 4,608/4,608 exact top-1
  controls with recall@16 and semantic coverage of 1.0, deterministic ties,
  zero clustered misses, and source/modality invariance.
- The 30,380-packet mature run passed 511/512 exact top-1, recall@16
  0.999265, semantic coverage 0.998571, deterministic ties, 19/20 material
  cycle resets, and no retrieval work-bound violations.
- Paper traceability passed 70/70 and the manuscript records every algorithm
  and experiment change.
- These results do not claim whole-engine raw-time reset, bounded whole-engine
  restart, Durable plateau, or production-wide boundedness.

## Packaging

- PyPI and npm artifacts contain six native targets each.
- Registry packages retain on-demand model bootstrap and do not embed the
  AIST model payload.

## Version Surfaces

The release surfaces report `1.2.3`:

- `CMakeLists.txt`
- `build.zig` and `build.zig.zon`
- `bindings/python/pyproject.toml`
- `bindings/javascript/package.json`
- `bindings/dart/pubspec.yaml`

## Publish Targets

- GitHub release: `v1.2.3`
- PyPI: `augmem.cortext==1.2.3`
- npm: `@augmem/cortext@1.2.3`
