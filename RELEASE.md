# Cortext v1.1.6 Release Artifact

This file is the tracked release artifact for the v1.1.6 line. The `v1.1.6`
tag should point at the commit that contains this artifact.

## Release State

- Version: `1.1.6`
- Runtime: hard-cut production engine, limited to embedding-backed memory,
  durable graph associations, bounded retrieval, and shallow consolidation.
- Product surface: adds the public `cortext_cli` command-line tool, built as a
  normal tool surface through CMake and as an installable Zig artifact.
- Removed research stack: decoder, provider registry, semantic extractor,
  summarizer, static label bank, fact layer, label-bucket graph, and
  mode-selected deep consolidation remain preserved in git but not shipped.
- Public API status: C++ facade, C ABI, and binding entry points remain the
  release surface for this tag.

## Version Surfaces

The tracked package surfaces report `1.1.6`:

- `CMakeLists.txt`
- `bindings/python/pyproject.toml`
- `bindings/javascript/package.json`
- `bindings/dart/pubspec.yaml`
- `build.zig`
- `build.zig.zon`

`cortext_version()` and every FFI package should report the same version.

## Fix Train

v1.1.6 ships the release-hardening fixes completed after v1.0 plus the patch
train through the public CLI release:

- Fixed graph-retrieval temporal decay overflow/stale-recency behavior and
  recorded the frozen replay A/B result.
- Restored bounded SQLite WAL behavior: default file-backed auto-checkpointing,
  `SQLITE_CHECKPOINT_TRUNCATE` for full checkpoints, and objstore blob garbage
  collection during eviction.
- Fixed GitHub Actions drift, split AIST model coverage into a dedicated cached
  job, and added ASan+UBSan CI coverage over the non-AIST suite.
- Validated media ingress before filters, encoders, or payload copies touch
  caller buffers.
- Made accumulator windows explicitly volatile staging state and stopped
  restoring or flushing legacy aggregate-only accumulator snapshots.
- Compiled ablation environment hooks out of default builds behind
  `CORTEXT_EXPERIMENT_HOOKS=OFF` while documenting the operational environment
  variables.
- Removed stale label-classifier/human-label tooling, fixed eval env-file
  discovery, and removed stale vendored submodule entries.
- Added direct tests for previously untested tail pipeline operations, made
  can-fail tests assert state, scrubbed external `CORTEXT_*` env vars during
  test startup, and switched temp DB naming to unique pid/time/counter paths.
- Updated the paper with negative experiment results, pattern-separation
  definitions, current consolidation framing, and tracked eval artifact copies.
- Added root `NOTICE` and `third_party/VERSIONS` inventory.
- Documented store/transaction single-owner semantics and one-writer-per-DB
  runtime ownership; schema migrations now take `BEGIN IMMEDIATE` before
  reading applied migration IDs.
- Fixed ephemeral public `ProcessText` retrieval so no-storage queries still
  run retrieval and hydrate returned context.
- Added embedding-band belief revision: durable corrections write
  knob-derived, modality-agnostic `supersedes` edges and retrieval demotes
  superseded stale memories without deleting history.
- Added `cortext_cli` with durable `remember`, ephemeral `recall`,
  consolidation, and REPL commands; CMake builds it under
  `CORTEXT_BUILD_TOOLS=ON`, and Zig installs it to `zig-out/bin`.

## Current Evidence

The v1.1.6 public evidence inherits the v1.0 hosted Meta Multi-Session Chat
eval, the 128k RAG ablation, and the post-optimization full replay verification.
The cited aggregate artifacts are tracked under `docs/paper/artifacts/`; full
protocol details are recorded in `docs/paper/sections/9_experimental.qmd`,
`docs/paper/sections/11_optimization.qmd`, and the generated manuscript at
`docs/paper/_manuscript/index.md`.

Additional v1.1.6 local verification recorded during the fix train:

- Full `ctest --test-dir build-test -R cortext_tests --output-on-failure`
  passed after the test-integrity work.
- Full `ctest --test-dir build-test -R cortext_tests --output-on-failure`
  passed after the store ownership and migration-lock work in 557.72 seconds.
- Focused AIST, sanitizer, media-validation, WAL, objstore-eviction, migration,
  paper-render, JSON/JSONL artifact, and Zig smoke checks passed at their
  respective fix points.

## Build Gate

```bash
cmake -S . -B build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORTEXT_DISABLE_OPENTELEMETRY=ON \
  -DCORTEXT_BUILD_TOOLS=ON
cmake --build build/release -j
./build/release/tools/cli/cortext_cli --help
./build/release/tests/cortext_tests '~[aist]' --reporter compact
```

Run AIST-tagged tests separately on a host with the model installed:

```bash
./build/release/tests/cortext_tests '[aist]' --reporter compact
```

## FFI Gate

```bash
cmake --preset ffi-release-node
cmake --build --preset ffi-release-node --target cortext cortext_node -j
(cd bindings/go && go test)
(cd bindings/dart && dart pub get && dart analyze && dart test)
python3 -m py_compile bindings/python/cortext/__init__.py
node --check bindings/javascript/index.js bindings/wasm/cortext.js examples/web/main.js
node -e "const c=require('./bindings/javascript'); if (c.version() !== '1.1.6') process.exit(1)"
```

## Zig CLI Gate

```bash
zig build -Doptimize=ReleaseFast -Dshared=false -Dcli=true -Dfetch-aist-model=false
test -x zig-out/bin/cortext_cli || test -s zig-out/bin/cortext_cli.exe
zig build -Dtarget=x86_64-linux-gnu -Doptimize=ReleaseFast -Dshared=false -Dcli=true -Dfetch-aist-model=false
zig build -Dtarget=x86_64-windows-gnu -Doptimize=ReleaseFast -Dshared=false -Dcli=true -Dfetch-aist-model=false
zig build -Dtarget=x86_64-macos -Doptimize=ReleaseFast -Dshared=false -Dcli=true -Dfetch-aist-model=false
zig build -Dtarget=aarch64-macos -Doptimize=ReleaseFast -Dshared=false -Dcli=true -Dfetch-aist-model=false
```

## Browser WebAssembly Gate

```bash
./build-wasm.sh
test -s build-wasm/dist/wasm/cortext.js
test -s build-wasm/dist/wasm/cortext.wasm
```

## CI Gate

The GitHub workflow must pass with optional OpenTelemetry disabled for
deterministic model-free CI where applicable:

- Ubuntu native release build and non-AIST tests.
- Arch Linux native release build and non-AIST tests.
- Dedicated cached AIST model job running `[aist]`.
- Debug ASan+UBSan non-AIST job.
- Browser WebAssembly bundle build.
- Zig host and Linux cross-build smoke checks.
- Zig-installed native `cortext_cli` checked on Ubuntu and Windows runners;
  target artifacts checked for Linux, Windows, macOS x86_64, and macOS
  aarch64.

## Documentation Gate

When algorithm behavior or experiment conclusions change, update
`docs/paper/sections/` and regenerate:

```bash
QUARTO_DISABLE_GIT=1 QUARTO_DISABLE_GITHUB=1 quarto render docs/paper
```

For packaging/API changes, update the root README and all affected binding
READMEs before tagging.
