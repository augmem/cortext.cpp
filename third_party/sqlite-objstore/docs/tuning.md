# File Backend Tuning Guide

This guide summarizes the high-impact knobs for the native file backend across desktop and server platforms. Use it alongside `docs/perf/file-backend.md` and `docs/architecture.md` when characterizing new hardware.

## 0. Build Configuration (Critical)

**Always use Release builds for performance testing**:

```bash
# Create optimized build
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target objstore_file_bench -j$(nproc)

# Run benchmarks
./build-release/benchmarks/objstore_file_bench --workload all --sync off
```

| Build Type | Flags          | Large Object Write | Notes                             |
| ---------- | -------------- | ------------------ | --------------------------------- |
| Debug      | `-O0` or none  | \~56 MiB/s         | **Not suitable for benchmarking** |
| Release    | `-O3 -DNDEBUG` | \~194 MiB/s        | Required for meaningful results   |

**Why this matters**: The BLAKE3 implementation depends on compiler optimizations to enable SIMD lanes and aggressive inlining. Debug builds disable those optimizations, erasing the hashing gains and understating backend throughput. See `docs/perf/blake3-migration.md` for architectural details.

## 1. Streaming Chunk Size

* Default: **64 KiB** (`PRAGMA objstore_chunk_size = 0` or leaving the field unset). The backend allocates a single aligned buffer of this size per transaction and reuses it for every staged write.
* When to increase: extremely fast NVMe devices that regularly handle multi‑GiB writes can benefit from 128 KiB chunks, but stay within the documented 1 MiB limit.
* When to decrease: memory-constrained deployments (embedded, WASM) may drop to 4–16 KiB to preserve RAM at the cost of additional syscalls.
* Command example:
  ```
  ./build/benchmarks/objstore_file_bench \
      --objects /tmp/bench/objects \
      --chunk 65536 \
      --workload all
  ```

## 2. Sync Mode Matrix

Run the benchmark once per sync mode to quantify the durability/throughput trade-off on the target filesystem:

| Mode       | Guarantees                                 | Typical Use                                         |
| ---------- | ------------------------------------------ | --------------------------------------------------- |
| `full`     | Payload + manifest `fsync()` before commit | Default for production and crash-safety validation  |
| `metadata` | Manifest `fsync()` only                    | SSDs where payload durability is handled externally |
| `off`      | No `fsync()`                               | Development, cache-hot smoke tests                  |

Use the built-in `--sync-matrix` flag to automate the comparisons:

```
./build/benchmarks/objstore_file_bench \
    --objects bench-output/objects \
    --sync-matrix \
    --runs 3 \
    --no-cache-clear
```

The harness will print a header before each sync mode so it is easy to capture deltas in CI logs.

## 3. Size Hints & Preallocation

* `objstore_stream_reader` now exposes `size_hint`. Unity tests demonstrate how to populate it when the payload size is known (see `tests/object_manager_test.c`).
* The file backend converts hints into `fallocate()` (Linux) or `ftruncate()` preallocation as soon as a staged writer is created. Direct `put()` calls use the same helper before streaming begins.
* Benefits:
  * fewer metadata updates on ext4/xfs
  * better small-object latency on APFS because sparse expansion is avoided
  * accurate `ENOSPC` detection before half the transaction has streamed
* Recommendation: always set `size_hint` when reading from SQLite BLOBs (`sqlite3_value_bytes()` already provides the size).

## 4. Manifest Buffering

* PUT/DEL entries accumulate in an in-memory `sqlite3_str` buffer during the transaction. `manifest.log` is written and synced exactly once inside `commit_staged()`.
* This removes the 10 ms per-object penalty from the previous “fsync per append” behavior.
* Keep `OBJSTORE_SYNC_METADATA` in mind for deployments that only need manifest durability (payload data is already fsynced during staging).

## 5. Platform Notes

* **APFS / macOS**: `ftruncate()` is used for preallocation. Keep an eye on Spotlight or Time Machine as they can influence cold-cache results. Use `sudo purge` between runs for reliable numbers.
* **ext4 / xfs (Linux)**: `fallocate()` is attempted first; if the filesystem does not support it the backend falls back to `ftruncate()`. Always clear the page cache (`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`) when capturing official numbers.
* **NTFS / Windows**: `ftruncate()` zero-fills the file, so chunk sizes above 64 KiB can increase CPU usage on spinning disks. Consider `OBJSTORE_SYNC_METADATA` for better latency if power-loss protection is available.

## 6. Benchmark Workflow Checklist

1. Pick a clean objects root (`rm -rf bench-output && mkdir bench-output`).
2. Run cache-hot sanity check:
   ```
   ./build/benchmarks/objstore_file_bench \
       --objects bench-output/objects \
       --runs 1 \
       --no-cache-clear
   ```
3. Collect cold-cache data with `--runs 3` (or more) and `--sync-matrix`.
4. Compare small-object ops/sec against the ≥10 k ops/sec goal; if still low, inspect `docs/tuning.md` chunk size section and rerun with `--chunk 32768` for comparison.
5. Record sync-mode deltas and include the CLI invocation plus environment details (filesystem, OS version) in the performance report.

Following this checklist ensures the improvements from Phase 05.7—manifest buffering, size hints, and the new default chunk size—translate into predictable throughput on every supported platform.

## 7. Performance Expectations & Bottlenecks

Understanding what limits performance in each workload helps set realistic targets:

### Large Objects (≥1 GiB)

**Current focus**: Backend I/O (BLAKE3 hashing is now faster than the storage stack)

| Metric           | Historical (SHA256 Release) | Target w/ BLAKE3                     |
| ---------------- | --------------------------- | ------------------------------------ |
| Write throughput | 190-200 MiB/s               | I/O limited (re-benchmark)           |
| Read throughput  | 10-15 GiB/s                 | 10-15 GiB/s                          |
| Latency (1 GiB)  | \~5.3 seconds               | \~1-2 seconds once storage saturates |

Action item: re-run large-object suites with the BLAKE3 build to capture new MiB/s ceilings and update this table.

### Medium Objects (10 MiB)

**Bottleneck**: Backend I/O / SQLite coordination (hashing overhead removed)

| Metric           | Current (Release) |
| ---------------- | ----------------- |
| Write throughput | \~192 MiB/s       |
| Read throughput  | 10-15 GiB/s       |
| Latency (p50)    | \~52 ms           |

### Small Objects (1 KiB)

**Bottleneck**: Filesystem metadata operations (creating 10,000 files)

| Metric           | Current (Release) | Target          |
| ---------------- | ----------------- | --------------- |
| Write throughput | \~5.5 MiB/s       | -               |
| Operations/sec   | \~5,600 ops/sec   | ≥10,000 ops/sec |
| Latency (p50)    | \~0.17 ms         | <0.1 ms         |

For small-object performance improvements, consider:

* Batch multiple objects in a single transaction
* Use faster filesystems (e.g., tmpfs for development)
* Adjust shard width to reduce directory fanout
* Future: object packing/bundling (out of scope for v1)
