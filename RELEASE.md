# Cortext v2.0.0 Release Artifact

This tracked artifact prepares the `v2.0.0` release metadata. It does not create
the tag or publish the release.

## Release State

- Version: `2.0.0`. This is a major release because the C ABI changed:
  `procedural_enabled` was removed from `cortext_config`.
- Merged feature baseline before release metadata: `65de8462` (PR #32, on top
  of PRs #28–#31 and #33).
- Release source: the integrated commit that the annotated `v2.0.0` tag
  points to. The GitHub release and the shared native assets are rebuilt from
  that exact commit.
- Product surface: the C++ facade, the C ABI, the optional N-API/`ffi/node`
  addon, the WASM build, and the CLI. Language packages are published from
  the standalone sibling repositories (`cortext.py`, `cortext.ts`, `cortext.go`,
  `cortext.dart` and `cortext.wasm`), each tracking this engine release.

## Highlights

- **Engine v5** (#28): a pruned engine, an FTS5 keyword channel, realtime
  retrieval on 443k-turn stores, standing memories and a per-call evidence
  budget, and modality-agnostic multimodal memories.
  - The engine clock never runs behind the signals it holds.
  - A late-interaction reranker (`AIST-87M-LI.gguf`) ships as Git shards
    beside AIST.
- **Per-blob content times** (#30): `Context::Memory::content_timestamps`
  gives the stored signal time of every content blob, including blobs
  attached at query time. The C JSON has a `"timestamp"` on each content entry.
- **Reference-time retrieval** (#31): a query can carry a reference time, past
  or future (`cortext::RetrievalTime`; C `has_reference_ms` / `reference_ms` /
  `as_of`).
  - A temporal candidate channel recalls memories near that time.
  - `as_of` limits recall to what was stored by then.
  - Migration 55 adds the time indexes.
- **AIST position features** (#29) are checked against the training mirror by
  cosine, so the check also holds on x86 q8 kernels.
- **WASM, Node, CLI, examples and docs** (#33):
  - The WASM build creates contexts again: exception catching, an Emscripten
    filesystem object-store backend, a larger stack, and no WAL or
    checkpointer thread.
  - The WASM build exports `HEAPU8` and `HEAPF32`.
  - A WASM smoke test now runs in CI.
  - The Node addon, CLI and web demo cover media, the evidence budget and the
    reference time.
- **README benchmarks** (#32): the README reports the v5 results, including
  the same-pipeline LoCoMo comparison with Mem0 OSS and Graphiti.

## Breaking Changes

- **`procedural_enabled`** is removed from `Cortext::Config` and from the C
  `cortext_config` struct. This is a C ABI layout change, so bindings must
  rebuild.
- **Soft Anchors** are removed: the config stats fields, `Memory::SoftAnchor`
  and the matching C/JSON fields are gone.
- **The fixed 3 KiB per-memory trim is gone.** Whole memories are returned
  unless an evidence budget is set.
- **The late-interaction model is required.** When AIST is present,
  `AIST-87M-LI.gguf` must resolve. The engine looks in this order:
  `CORTEXT_LI_MODEL_PATH`, beside AIST, then the linked shards.
- **New trailing `Process*` parameters:** `EvidenceBudget` and
  `RetrievalTime`. Both are defaulted, so existing C++ callers still compile.
- **`RegisterBuiltInExtensionsOnDb` returns `int`** instead of `void`.
- **Store migrations 35–55** run on open. Stores written before migration 54
  have no late-interaction codes until they are re-ingested.

## Verification

- PRs #28–#33 were merged with exact-head GitHub Actions CI green: native,
  architecture-native, AIST embed, WASM browser (with the new Node smoke
  test), sanitizers, and Zig CLI install/target jobs including Windows.
- Packet parity was checked across the merges:
  - The upstream merge in #28 matched local `main` on all three parity sets
    (0/48, 0/20 and 0/250 differ).
  - #30 and #31 are byte-identical to `main` when no reference time is set.
- #31 was built and run together with #30 before it merged: the full suite
  passed, 557 test cases.
- Benchmark results for this engine line are in the README Benchmarks section.

## Packaging

- Release assets remain core-owned. The GitHub release includes
  `cortext-assets-2.0.0.tar.gz` and its `.sha256`, built by
  `.github/workflows/release-assets.yml` from the tagged tree.
- The asset tree carries every tracked model asset, now including the
  late-interaction reranker shard and its manifest entry. The inventory check
  verifies each asset's checksum after reassembly.
- TypeScript/Node prebuilds consume this exact `v2.0.0` tag through the
  standalone `augmem/cortext.ts` release pipeline.

## Version Surfaces

The engine release surfaces report `2.0.0`:

- `CMakeLists.txt` project version
- `build.zig` `cortext_version` and library package version
- `build.zig.zon` package version
- `models/manifest.json` `cortext_version`
- `README.md` asset download example (`ASSETS_VERSION`)

Language package versions are owned by their standalone sibling repositories.

## Publish Targets

- GitHub release tag: `v2.0.0`, with the engine and
  `cortext-assets-2.0.0.tar.gz`.
- Language packages: see the sibling repositories `augmem/cortext.py`,
  `augmem/cortext.ts`, `augmem/cortext.go`, `augmem/cortext.dart` and
  `augmem/cortext.wasm`.
