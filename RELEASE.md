# Cortext v1.2.0 Release Artifact

This tracked artifact accompanies the `v1.2.0` tag.

## Release State

- Version: `1.2.0`
- Runtime: C++20 multimodal memory engine with SQLite persistence and local
  model runtimes.
- Product surface: C++ facade, C ABI, Python, JavaScript/TypeScript, Dart,
  Go, WASM, CLI, and Hermes provider integration.

## Highlights

- Added explicit retention policies: `Natural` (default), `Durable`,
  `Boundary`, and `Ephemeral` across the core processing APIs and maintained
  bindings.
- Kept legacy C ingress paths durable while exposing retention through JSON
  options and language bindings.
- Added the Hermes provider for silent turn, tool-result, and multimodal
  memory ingestion with durable storage and ephemeral recall.
- Isolated Ephemeral retrieval probes from open Natural accumulator state,
  pacing, interrupt bookkeeping, spike bypass, and episode finalization.
- Added CI coverage for native, sanitizer, AIST, WASM, and Zig CLI targets.

## Version Surfaces

The release surfaces report `1.2.0`:

- `CMakeLists.txt`
- `build.zig` and `build.zig.zon`
- `bindings/python/pyproject.toml`
- `bindings/javascript/package.json`
- `bindings/dart/pubspec.yaml`

## Verification

- GitHub Actions run `29175197131` passed native, sanitizer, AIST, WASM, and
  all Zig CLI jobs for the merged retention line.
- Package artifacts are built from this tagged commit with the repository
  Python and JavaScript packaging scripts before publishing.

## Publish Targets

- GitHub release: `v1.2.0`
- PyPI: `augmem.cortext==1.2.0`
- npm: `@augmem/cortext@1.2.0`
