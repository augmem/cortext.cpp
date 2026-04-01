# Performance, Stress, and Crash Validation

This project ships dedicated harnesses to measure throughput, memory usage, and
crash behaviour across the supported backends. The harnesses live under
`tests/perf/` and `tests/crash/` so CI can execute them alongside the regular
Unity suite.

## Running the suites

```bash
# Build perf binaries (micro/stress/stream) and run the short suite
cmake -S . -B build-perf -DOBJSTORE_ENABLE_PERF_TESTS=ON
cmake --build build-perf
ctest --test-dir build-perf -L perf

# Execute the crash harness (skips on Windows)
cmake --build build
ctest --test-dir build -R crash_kill_during_commit
```

The perf-label tests keep their runtime below \~15 seconds locally so they can
run in developer loops. For deeper analysis you can execute each binary
directly to change payload sizes, transaction counts, or backends.

## Snapshot results (macOS, Apple M3 Pro)

| Harness                    | Backend        | Workload                 | Key metrics                                                                                               |
| -------------------------- | -------------- | ------------------------ | --------------------------------------------------------------------------------------------------------- |
| `objstore_perf_micro`      | SQLite         | 64 ops, 4 KiB payloads   | insert 0.004 s (60.9 MiB/s), read 0.001 s (183.8 MiB/s), delete 0.001 s (1.5 MiB/s)                       |
| `objstore_perf_stress`     | SQLite         | 512 ops mixed (1–8 KiB)  | 0.15 s total, 3.5 k ops/s, 1.01 MiB written, 0.78 MiB read, 105 live objects at end                       |
| `objstore_perf_stream_1gb` | File           | 1 GiB zero payload       | write 13.88 s @ 73.8 MiB/s, read 0.074 s @ 13.8 GiB/s, chunk size 64 KiB, peak RSS 2.5 MiB                |
| `objstore_crash_kill`      | SQLite backend | Force kill during commit | Child process dies via `SIGKILL`; reopening the DB shows `objstore` + `objstore_data` row counts remain 0 |

Results were captured with the default configurations (`OBJSTORE_SYNC_FULL`,
64 KiB chunking). File-backend streaming used the macOS `/usr/bin/time -l`
wrapper to capture RSS and context-switch data:

```
stream_1gb backend=file bytes=1073741824 write_seconds=13.882 write_mib_per_s=73.76 \
    read_seconds=0.074 read_mib_per_s=13824.01 chunk_size=65536
maximum resident set size: 2.5 MiB
```

## Design targets & invariants

`docs/architecture.md` lists the global goals that drove the original design.
The headline performance/behavioral targets to keep an eye on when analysing the
harness output are:

- **Large objects (≈1 GiB)** – streaming writes should converge on ≥500 MiB/s so
  storing 1 GiB takes ≤2 s on NVMe/SSD-class storage. Reads routinely saturate
  the underlying filesystem (10–15 GiB/s on modern desktops).
- **Small objects (1 KiB)** – CRUD workloads should approach ≥10 000 ops/sec.
  This is primarily a filesystem metadata challenge, so shard width and sync
  mode have an outsized impact.
- **Streaming bounds** – the 64 KiB default chunk keeps RSS low (≤3 MiB during
  the 1 GiB streaming run) and matches the buffer allocated inside every staged
  writer. When deviating from the default, adjust both the harness CLI and the
  backend knobs so results stay apples-to-apples.
- **Backend-first commits** – if a perf or crash harness reports success, it
  implicitly validated that backend staging promoted before SQLite committed.
  Failures that occur after the backend writes can leave orphaned blobs; track
  them with metadata sweeps after stress tests.

## Custom workloads

Each harness accepts CLI flags so you can tailor experiments:

* `objstore_perf_micro --backend file --ops 1024 --payload 131072`
* `objstore_perf_stress --backend sqlite --ops 5000 --min 4096 --max 65536 --txn-batch 128`
* `objstore_perf_stream_1gb --backend file --bytes 17179869184 --storage-root /tmp/objstore-bench`

The crash harness is intentionally opinionated: it uses the SQLite backend,
stages thirty-two 2 MiB inserts inside a transaction, then terminates the child
with `SIGKILL`. The parent reopen/validates that no rows were committed, which
exercises the deferred-write queue and commit hooks.

## Coverage and reporting

Use `scripts/run-coverage.sh` to build an instrumented configuration, run
`ctest`, and generate both textual and HTML coverage reports:

```bash
scripts/run-coverage.sh
# -> build-coverage/coverage/summary.txt
# -> build-coverage/coverage/html/index.html
```

Coverage uses Clang’s `-fprofile-instr-generate -fcoverage-mapping`
instrumentation, so make sure `llvm-cov` and `llvm-profdata` are available on
`PATH`.
