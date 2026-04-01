# BLAKE3 Migration & Performance Notes

## Why BLAKE3

* BLAKE3 is a tree-structured hash with SIMD- and thread-friendly design, delivering multi-GB/s throughput on commodity CPUs while remaining cryptographically sound \[[BLAKE3]].
* The algorithm outputs 256-bit digests by default, matching `OBJSTORE_ID_SIZE`, so SQLite schemas and backend layouts remain unchanged.
* Streaming APIs map naturally onto the object manager: we continue hashing chunks as they flow through staged writers, preserving the single-pass write pipeline.

> [BLAKE3]: https://github.com/BLAKE3-team/BLAKE3

## Build Integration

* Sources live under `third_party/blake3/` (official upstream C implementation, v1.8.2).
* `src/CMakeLists.txt` wires the portable files in for every platform and conditionally enables SIMD backends:
  * x86/x86\_64 builds (Clang/GCC) compile the SSE2, SSE4.1, AVX2, and AVX-512 units with per-file `-m` flags.
  * AArch64 builds add the NEON variant.
  * Other targets fall back to the portable scalar implementation, and we define `BLAKE3_NO_*` switches when an architecture lacks the compiled units.
* `include/objstore/blake3.h` exposes a minimal wrapper (`objstore_blake3_*`) so callers never depend on upstream types directly.
* `third_party/blake3/blake3.h` is installed alongside the public headers to keep downstream builds self-contained.

## Performance Expectations

* BLAKE3 eliminates the previous software-SHA256 bottleneck: hashing is now faster than our file backend I/O on modern CPUs, so throughput is once again gated by storage and fsync policy.
* Portable builds (no SIMD) still exceed the prior SHA256 path in throughput because BLAKE3’s core permutation is lighter even without vectorization.
* SIMD-enabled builds scale with available CPU features:
  * SSE4.1/AVX2 lanes provide 4–8-way parallelism inside a single thread.
  * AVX-512 widens to 16 lanes, which is helpful for high-core-count servers.
* Benchmark TODOs:
  * Re-run `benchmarks/file_backend_bench` for large (≥1 GiB) workloads and document updated MiB/s / ops/sec numbers.
  * Repeat medium/small suites to validate that hashing no longer dominates.
  * Capture WASM/OPFS numbers to confirm the portable path still meets <100 KB binary goals.

## Operational Considerations

* IDs have changed — existing databases that stored SHA256-derived IDs must be rehydrated (export/import or rehash payloads) before upgrading.
* Integrity verification now recomputes BLAKE3 digests; `objstore_object_verify` and smoke tests were refreshed accordingly.
* The transaction log remains backward compatible: entries still store 32-byte IDs plus payload sizes, so no serialization changes were needed.
* Future work:
  * Evaluate background prefetching to keep SIMD units saturated on spinning disks.
  * Consider exposing the BLAKE3 XOF interface for applications that need variable-length keys without rehashing payloads multiple times.
