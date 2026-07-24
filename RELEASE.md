# Cortext v1.2.4 Release Artifact

This tracked artifact accompanies the `v1.2.4` tag.

## Release State

- Version: `1.2.4`
- Merged feature baseline before release metadata: `170d866a57665b6e48f1bb446276f01ca653fcc2`
  (PR #8 — shared release assets + AIST model embed)
- Release source: the integrated commit addressed by the annotated `v1.2.4`
  tag; the GitHub release and registry artifacts are rebuilt from that exact
  commit.
- Product surface: C++ facade, C ABI, Python, JavaScript/TypeScript, Dart,
  Go, WASM, CLI, and Hermes provider integration.

## Highlights

- Publishes multi-arch shared libraries and Git-friendly AIST model shards as
  GitHub Release assets (`cortext-assets-<version>.tar.gz`) so language
  bindings can consume one canonical bundle.
- Default builds link AIST model shards into `libcortext` (assemble at load)
  so shared libraries work without a separate download or
  `CORTEXT_AIST_MODEL_PATH`; opt out with `-DCORTEXT_EMBED_AIST_MODEL=OFF`.
- Tracks under-100 MiB model parts in-repo (no LFS) under
  `models/AIST-87M-GGUF/chunks/` with manifest checksums.
- Documents binding install layout and keeps thin/embed-off paths supported.

## Verification

- Pull request [#8](https://github.com/augmem/cortext/pull/8) merged at
  `170d866a57665b6e48f1bb446276f01ca653fcc2` with review threads resolved and
  exact-head CI green (native, AIST embed job, WASM, Zig matrix including
  Windows, sanitizers).

## Packaging

- GitHub Release includes `cortext-assets-<version>.tar.gz` (+ sha256) built by
  `.github/workflows/release-assets.yml` (natives + model shards).
- Default native libraries embed the shipping AIST q8_0 shards.
- PyPI and npm artifacts remain optional publish targets for this tag.

## Version Surfaces

The release surfaces report `1.2.4`:

- `CMakeLists.txt`
- `build.zig` and `build.zig.zon`
- `bindings/python/pyproject.toml`
- `bindings/javascript/package.json`
- `bindings/dart/pubspec.yaml`

## Publish Targets

- GitHub release: `v1.2.4`
- Shared assets: `cortext-assets-1.2.4.tar.gz` / `.tar.gz.sha256`
- PyPI: `augmem.cortext==1.2.4` (when published)
- npm: `@augmem/cortext@1.2.4` (when published)
