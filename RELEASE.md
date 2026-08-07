# Cortext v1.3.1 Release Artifact

This tracked artifact prepares the `v1.3.1` release metadata. It does not create
the tag or publish the release.

## Release State

- Version: `1.3.1`
- Merged feature baseline before release metadata:
  `b26bc30ab76bd50731382a15a503c7891d12612a` (PR #13 — Node text-media
  N-API bridge).
- Release source: the integrated commit addressed by the annotated `v1.3.1`
  tag; the GitHub release and shared native assets are rebuilt from that exact
  commit.
- Product surface: C++ facade, stable C ABI, optional N-API/`ffi/node` addon,
  WASM build, and CLI. Language packages are published from standalone sibling
  repositories (`cortext.py`, `cortext.ts`, `cortext.go`, `cortext.dart`, and
  `cortext.wasm`), not vendored in this engine tree.

## Highlights

- Adds the Node-API `processTextWithMediaJson` bridge for text processing with
  optional binary media, MIME type, and process options. The bridge consumes
  the merged C/C++ text-media API without adding a second native implementation.
- Retains text-with-media processing in the C++ and C APIs, including JSON
  wrappers and WASM exports. Text remains the canonical embedding/processing
  input while non-empty caller media is copied and persisted with its MIME type.
- Supports arbitrary binary text payloads: custom MIME payloads are treated as
  opaque bytes and hydrate unchanged. `Media{}`, `NULL`, or zero-size media
  omits the stored payload, while legacy calls retain canonical text storage.
- Removes the in-tree Python, TypeScript/Node, Go, Dart, and WASM bindings and
  their package builders. The shared native API, native libraries, and model
  shards remain engine-owned; standalone sibling packages consume those release
  assets rather than maintaining another copy of the bindings.
- Keeps the default AIST q8_0 model embed and Git-friendly under-100 MiB model
  shards, with manifest checksums for native and thin/embed-off consumers.
- Tracks optional surname/forename CSV shards under `data/surnames/chunks/`
  with `data/surnames/manifest.json`; full CSVs remain local-only.

## Verification

- Pull request [#11](https://github.com/augmem/cortext.cpp/pull/11) merged at
  `b6709817959b81e4fbbd4b0eb9be0a8102d80eb1` with exact-head GitHub Actions CI
  green: native, architecture-native, AIST embed, WASM browser, sanitizers,
  and Zig CLI install/target jobs including Windows.
- PR #11 covered text-media persistence, opaque binary hydration, C error
  handling, NULL/empty-media behavior, and legacy payload behavior.
- Pull request [#13](https://github.com/augmem/cortext.cpp/pull/13) is merged
  at `b26bc30ab76bd50731382a15a503c7891d12612a`, with the Node text-media
  N-API bridge at feature head `9686c5a10cca850f73406a7bbd9fa1a38859fb40`.
- The release branch is based directly on that merged `origin/main` commit;
  release metadata is the only change after the merge.

## Packaging

- GitHub Release includes `cortext-assets-1.3.1.tar.gz` and its `.sha256`,
  built by `.github/workflows/release-assets.yml` from the tagged tree.
- Default native libraries embed the shipping AIST q8_0 shards; the release
  asset tree also carries model shards for thin/embed-off consumers.
- Node prebuilds are built by the standalone `augmem/cortext.ts` release
  pipeline from this exact `v1.3.1` tag. That pipeline configures CMake with
  `CORTEXT_BUILD_NODE_BINDINGS=ON`, builds `cortext_node` on its host matrix,
  and publishes the resulting `cortext.node` artifacts; this repository does
  not publish Node prebuilds.
- Language package versions and registry publishing remain owned by the
  standalone sibling repositories, not this engine repository.

## Version Surfaces

The engine release surfaces report `1.3.1`:

- `CMakeLists.txt` project version
- `build.zig` `cortext_version` and library package version
- `build.zig.zon` package version
- `models/manifest.json` `cortext_version`

Language package versions are owned by their standalone sibling repositories.

## Publish Targets

- GitHub release tag: `v1.3.1` (engine + `cortext-assets-1.3.1.tar.gz`)
- Language packages: see sibling `augmem/cortext.py`, `augmem/cortext.ts`,
  `augmem/cortext.go`, `augmem/cortext.dart`, and `augmem/cortext.wasm`
  repositories.
