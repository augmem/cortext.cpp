# Cortext v1.0 Release Artifact

This file is the tracked release artifact for the v1.0 line. The `v1.0` and
`v1.0.0` tags should point at the commit that contains this artifact.

## Release State

- Version: `1.0.0`
- Runtime: hard-cut production engine, limited to embedding-backed memory,
  durable graph associations, bounded retrieval, and shallow consolidation.
- Removed research stack: decoder, provider registry, semantic extractor,
  summarizer, static label bank, fact layer, label-bucket graph, and
  mode-selected deep consolidation.
- Public API status: C++ facade, C ABI, and binding entry points remain the
  release surface for this tag.

## Version Surfaces

The tracked package surfaces now report `1.0.0`:

- `CMakeLists.txt`
- `bindings/python/pyproject.toml`
- `bindings/javascript/package.json`
- `bindings/dart/pubspec.yaml`
- `build.zig`
- `build.zig.zon`

`cortext_version()` and every FFI package should report the same version.

## Current Evidence

The current v1.0 public evidence is the hosted Meta Multi-Session Chat eval on
the public `nayohan/multi_session_chat` mirror, plus the 128k RAG ablation and
post-optimization full replay verification.

### Hosted Frontier Judge

- Artifact:
  `eval_runs/msc_frontier_late_200dlg_gpt55_20260630T053427Z/judge_openai_gpt55_four_system_clean.json`
- Replay: 9,130 text turns from 708 public MSC rows.
- Protocol: daily source-time consolidation at 02:00 UTC, 5,000-event warmup,
  500-event probe stride, 9 probes, 3 hosted `gpt-5.5` blind judgments per
  probe, 27/27 judgments complete.
- Result: Cortext won 21/27, traditional chat+RAG won 0/27, full-history upper
  bound won 1/27, hosted compaction rollup won 5/27.
- Mean context: Cortext 998 tokens versus traditional chat+RAG 49,196 tokens,
  a 97.97% reduction with probe-bootstrap 95% CI [97.77%, 98.17%].
- Caveat: this supports token-reduction, noise-reduction, and blind-win claims;
  it is not a completed sufficiency-match claim.

### 128k RAG Ablation

- Artifact:
  `eval_runs/msc_rag_ablation_128k_gpt55_20260630T_actual/judge_openai_gpt55_rag_ablation_128k.json`
- Protocol: same 9 MSC probes, 3 hosted `gpt-5.5` blind judgments per probe,
  128,000-token judge-context cap, 27/27 judgments complete.
- Systems: Cortext native, semantic vector RAG, lexical keyword RAG,
  rolling-window chat, 16k hybrid chat+vector RAG, hosted compaction rollup.
- Result: Cortext won 19/27, hosted compaction rollup won 8/27, and all four
  RAG-style packet variants won 0/27.
- Prompt cap: max estimated judge prompt 116,425 tokens under the 128,000-token
  cap.
- OpenAI usage: 1,871,994 prompt tokens and 39,001 completion tokens across 27
  judge requests.

### Full MSC Latency Verification

- Artifact:
  `build/graph_profile/full_msc_verify_rerun_20260630T142345Z/summary.json`
- Replay: same 9,130-turn MSC slice, same default knobs, daily consolidation,
  5,000-event warmup, and 500-event probe stride.
- Completion: 9 probes, 714 consolidations, 10,099 memories, 9,364 long-term
  memories, 21 working-memory rows, and 17,719 associations.
- Behavior: recursive non-timing probe content matched the saved frontier
  artifact and the prior post-optimization replay.
- Latency: judged probe total latency was 50.4-80.2 ms, mean 63.5 ms;
  `GraphRetrieve.total` was 22.2-29.2 ms, mean 25.9 ms.

Full protocol details are recorded in `docs/paper/sections/9_experimental.qmd`,
`docs/paper/sections/11_optimization.qmd`, and the generated manuscript at
`docs/paper/_manuscript/index.md`.

## Build Gate

```bash
cmake -S . -B build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORTEXT_DISABLE_OPENTELEMETRY=ON
cmake --build build/release -j
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
node -e "const c=require('./bindings/javascript'); if (c.version() !== '1.0.0') process.exit(1)"
```

## Browser WebAssembly Gate

```bash
./build-wasm.sh
test -s build-wasm/dist/wasm/cortext.js
test -s build-wasm/dist/wasm/cortext.wasm
```

## CI Gate

The GitHub workflow must pass with optional OpenTelemetry disabled for
deterministic model-free CI:

- Ubuntu native release build and non-AIST tests.
- Arch Linux native release build and non-AIST tests.
- Browser WebAssembly bundle build.
- Zig host and Linux cross-build smoke checks.

## Documentation Gate

When algorithm behavior or experiment conclusions change, update
`docs/paper/sections/` and regenerate:

```bash
QUARTO_DISABLE_GIT=1 QUARTO_DISABLE_GITHUB=1 quarto render docs/paper
```

For packaging/API changes, update the root README and all affected binding
READMEs before tagging.
