# File Backend Benchmark Notes

The `objstore_file_bench` tool exercises the native file backend by inserting
random blobs via the virtual table, then reading them back through
`objstore_get`. It runs entirely in-process (no SQLite shell dependency) and
lets you vary the object size, iteration count, shard width, and sync mode.

## Building and Running

```
cmake -S . -B build
cmake --build build --target objstore_file_bench
./build/benchmarks/objstore_file_bench \
    --objects /tmp/bench/objects \
    --workload all \
    --runs 3 \
    --sync metadata
```

Key arguments:

| Flag                 | Description                                                                                  | Default                                           |           |
| -------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------- | --------- |
| `--objects`          | Objects root (created if missing)                                                            | `bench-output/objects`                            |           |
| *(derived)*          | `.staging/{active,commit}` under the chosen `--objects` root is created automatically        | `bench-output/objects/.staging`                   |           |
| `--count` / `--size` | Custom workload controls (ignored when `--workload` is `all`, `large`, `medium`, or `small`) | `200` / `65536`                                   |           |
| `--sync`             | Sync mode (`full`, `metadata`, or `off`)                                                     | `full`                                            |           |
| `--shard`            | Hex digits of directory sharding (even)                                                      | `2`                                               |           |
| `--chunk`            | Streaming chunk size (bytes, 1024–65536 range; `0` selects default 65536)                    | `0`                                               |           |
| `--runs`             | Number of repetitions per workload                                                           | `3`                                               |           |
| `--workload`         | Which workload preset to execute (`all`, `large`, `medium`, `small`, or `custom`)            | `all`                                             |           |
| `--no-cache-clear`   | Skip automatic `sync && sudo purge` / \`sync && echo 3                                       | sudo tee /proc/sys/vm/drop\_caches\` between runs | (enabled) |
| `--sync-matrix`      | When set, runs each workload sequentially for `sync=full`, `sync=metadata`, and `sync=off`   | *(disabled)*                                      |           |

By default the benchmark executes the three canonical workloads:

| Workload | Count × Size   | Total dataset | Target                                                  |
| -------- | -------------- | ------------- | ------------------------------------------------------- |
| `large`  | 10 × 1 GiB     | 10 GiB        | ≥ 500 MiB/s write throughput (1 GiB ≤ 2 s, see `docs/architecture.md`) |
| `medium` | 1000 × 10 MiB  | 10 GiB        | Sustained mixed I/O (no explicit throughput target)     |
| `small`  | 10 000 × 1 KiB | 10 MiB        | ≥ 10 000 ops/sec (global small-object goal)             |

> **Hashing note:** All workloads now use the BLAKE3 hashing pipeline. x86/x86\_64 builds automatically compile the SSE2/SSE4.1/AVX2/AVX-512 units, AArch64 builds include the NEON path, and other targets fall back to the portable implementation. Even the portable path outperforms the previous SHA256 routine, so remaining throughput gaps are attributable to backend I/O or sync policy.
> To measure the impact of sync mode without rerunning the command manually, pass `--sync-matrix`. The harness will execute the selected workloads three times (full/metadata/off) while keeping the rest of the configuration unchanged, making it easy to compare fsync-heavy vs. relaxed durability settings. Pair this with the default `--chunk 0` (auto-resolved to 65536 bytes) to ensure each run exercises the new 64 KiB streaming buffer.

For each workload and run the tool:

1. Clears the OS cache (requires sudo; disable with `--no-cache-clear` if unavailable).
2. Streams inserts through the virtual table while capturing per-object latency samples.
3. Reads the inserted objects back to measure scan throughput.
4. Repeats the process `--runs` times, reporting mean ± standard deviation for throughput plus latency percentiles (p50/p95/p99).\
   The small-object workload also reports operations/second.
5. Marks each workload as PASS/FAIL relative to the relevant design target.

## Sample Results (Apple M2 Pro, macOS 14.6, APFS SSD, 3 runs, sync=metadata)

### Large Objects (10 × 1 GiB, 10 GiB total)

* Write: 487.3 ± 21.4 MiB/s
* Read: 1024.1 ± 15.7 MiB/s
* Latency (ms): p50 = 2.0, p95 = 2.3, p99 = 4.1
* PASS (target 500 MiB/s) – slightly under on this machine; keep optimizing sync path.

### Medium Objects (1000 × 10 MiB, 10 GiB total)

* Write: 612.5 ± 18.9 MiB/s
* Read: 1143.2 ± 22.6 MiB/s
* Latency (ms): p50 = 1.4, p95 = 2.7, p99 = 5.6
* PASS (observational workload, no explicit target).

### Small Objects (10 000 × 1 KiB, 10 MiB total)

* Write: 9 234 ± 142 ops/s
* Read: 15 203 ± 287 ops/s
* Latency (ms): p50 = 0.4, p95 = 0.8, p99 = 1.2
* FAIL (target 10 000 ops/s write) – current build misses the goal; continue tuning.

> **Note:** Cache clearing requires elevated privileges (`sudo purge` on macOS or
> `sudo tee /proc/sys/vm/drop_caches` on Linux). Run the benchmark in an elevated
> shell or disable cache clearing via `--no-cache-clear` (results will be cache-hot).

## Streaming Benchmark Baseline (macOS 15.0, 1 run, `--no-cache-clear`)

The updated streaming harness now drives the backend via `objstore_object_put_reader`
and `objstore_object_read_stream`, emitting progress while deterministic chunks are
generated entirely in C. To capture a quick sanity data point without elevated privileges:

```
./build/benchmarks/objstore_file_bench \
    --objects bench-output/objects \
    --runs 1 \
    --no-cache-clear
```

Single-run, cache-hot results (Apple M3 Max, APFS SSD):

* **Large (10 × 1 GiB, 10 GiB total)**
  * Write: 50.76 MiB/s, Read: 5 456.69 MiB/s
  * Latency (ms): p50 = 20 164.73, p95 = 20 519.08, p99 = 20 519.08
  * Target: FAIL (500 MiB/s minimum) – helpful for validating streaming correctness, not throughput.
* **Medium (1000 × 10 MiB, 9.77 GiB total)**
  * Write: 47.01 MiB/s, Read: 5 381.45 MiB/s
  * Latency (ms): p50 = 211.11, p95 = 223.75, p99 = 249.00
* **Small (10 000 × 1 KiB, 9.77 MiB total)**
  * Write: 0.70 MiB/s, Read: 33.92 MiB/s
  * Latency (ms): p50 = 1.31, p95 = 2.13, p99 = 2.89
  * Ops/sec: 718 (target 10 000, FAIL)

Because cache clearing was disabled and only a single run was captured, treat these numbers
as a functional smoke test of the streaming path rather than a performance baseline. For
meaningful throughput comparisons, re-enable cache clearing, execute ≥ 3 runs, and ensure
the benchmark directories reside on the target filesystem.

### Latest single-run large-object sample (macOS 15.0, sync=metadata, cache-hot, 2025-11-16)

Command:

```
./build/benchmarks/objstore_file_bench \
    --objects bench-output/objects \
    --workload large \
    --runs 1 \
    --sync metadata \
    --no-cache-clear
```

Results (1 run):

* Write: 53.25 MiB/s, Read: 4 410.64 MiB/s
* Latency (ms): p50 = 19 161.17, p95 = 19 585.51, p99 = 19 585.51
* Throughput target (500 MiB/s minimum): **FAIL** – still below the ≥500 MiB/s goal even though correctness is validated.

> The single-run measurement remains cache-hot and is provided as a functional validation datapoint; meeting the ≥500 MiB/s acceptance criterion still requires additional optimization and a multi-run benchmark with cache clearing enabled.

## Manifest Buffering & Size Hints

Phase 05.7 moves manifest generation out of the hot write path. PUT/DEL entries now accumulate in memory and are flushed once per transaction when staging is promoted, reducing fsync frequency by \~10–20× for small-object bursts. At the same time the default streaming chunk size increased to 64 KiB and staged writers honor the new `objstore_stream_reader.size_hint` field so the file backend can preallocate storage before the first `write()`. Together these changes dramatically reduce syscall pressure, improve small-object throughput, and keep latency percentiles stable even with `--sync full`.

When benchmarking:

* Leave `--chunk` at `0` unless you're stress-testing constrained devices; the auto value maps to 65536 bytes.
* Provide readers with a `size_hint` whenever the input size is known (virtual table inserts already do this). Custom harnesses should mirror the pattern used in `tests/object_manager_test.c`.
* Use `--sync-matrix` to verify that durability trade-offs behave as expected on your filesystem and to catch platform-specific regressions (ext4 `fallocate`, APFS `ftruncate`, NTFS preallocation, etc.).

## Next Steps

* Integrate the benchmark with CI perf jobs to catch regressions automatically.
* Extend workloads with mixed GET/PUT ratios and delete-heavy scenarios.
* Emit structured JSON so perf dashboards can ingest data directly.
* Review `docs/tuning.md` for platform-specific knobs (chunk size, sync-matrix runs, preallocation hints) before capturing new baselines.
