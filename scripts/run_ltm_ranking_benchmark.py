#!/usr/bin/env python3
"""Paired, isolated performance benchmark for LTM retrieval ranking changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import re
from typing import Any

SCHEMA_VERSION = 5
METRIC_DIRECTIONS = {
    "retrieval_p50_ms": "lower-is-better",
    "retrieval_p95_ms": "lower-is-better",
    "ingestion_p50_ms": "lower-is-better",
    "ingestion_p95_ms": "lower-is-better",
    "retrieval_ops_per_second": "higher-is-better",
    "ingestion_ops_per_second": "higher-is-better",
    "peak_rss_bytes": "lower-is-better",
    "database_bytes": "lower-is-better",
}
CANDIDATE_BUILD_KEYS = (
    "BUILD_SHARED_LIBS",
    "BUILD_TESTING",
    "CMAKE_BUILD_TYPE",
    "CMAKE_CXX_COMPILER",
    "CORTEXT_BUILD_EXAMPLES",
    "CORTEXT_BUILD_NODE_BINDINGS",
)
PACKAGE_RUNTIME_FILES = ("index.js", "model-bootstrap.js", "package.json")
WORKER_SOURCE = r'''"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const packageRoot = process.env.CORTEXT_BENCH_PACKAGE_ROOT;
const dbPath = process.env.CORTEXT_BENCH_DB_PATH;
const modelPath = process.env.CORTEXT_AIST_MODEL_PATH;
if (!packageRoot || !dbPath || !modelPath) {
  throw new Error("benchmark worker is missing package, database, or model path");
}

const { Cortext } = require(packageRoot);
const messageCount = 512;
const retrievalCount = 32;
const needle = "The same-session recall test returned teal.";
const boilerplate =
  "All 46 unit tests pass. File updated successfully. [tool call] npm test completed.";
const longFiller = `${boilerplate} Generic build status and file offset metadata. `.repeat(12);
const queries = [
  "What color did the same-session recall test return?",
  "Which archived observation described the recall test color?",
  "Recall the distinctive color from the same-session test.",
  "What did the old memory say about the teal result?",
];

function messageAt(index) {
  if (index >= 88 && index < 96) {
    return `Archived observation ${index}: ${needle}`;
  }
  if (index >= 176 && index < 184) {
    return `${longFiller} ${needle} ${longFiller}`;
  }
  if (index % 11 === 0) {
    return `${boilerplate} Repeated acknowledgement family alpha.`;
  }
  if (index % 7 === 0) {
    return `${boilerplate} Repeated acknowledgement family beta.`;
  }
  return `${boilerplate} Run ${index % 17}; tally ${index % 13}; offset ${index}.`;
}

function percentile(values, fraction) {
  const ordered = [...values].sort((left, right) => left - right);
  const index = Math.min(
    ordered.length - 1,
    Math.max(0, Math.ceil(fraction * ordered.length) - 1)
  );
  return ordered[index];
}

function sha256File(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function databaseBytes(filePath) {
  let total = 0;
  for (const suffix of ["", "-wal", "-shm"]) {
    const candidate = `${filePath}${suffix}`;
    if (fs.existsSync(candidate)) {
      total += fs.statSync(candidate).size;
    }
  }
  return total;
}

const corpus = Array.from({ length: messageCount }, (_, index) => messageAt(index));
const corpusDigest = crypto
  .createHash("sha256")
  .update(JSON.stringify({ corpus, queries, retrievalCount }))
  .digest("hex");
const engine = new Cortext(
  { focus: 0.45, sensitivity: 0.5, stability: 0.5 },
  dbPath
);

const ingestionMs = [];
let ingestionTotalNs = 0n;
for (let index = 0; index < corpus.length; index += 1) {
  const start = process.hrtime.bigint();
  engine.processText(corpus[index], `benchmark/source/${index % 19}`, {
    retention: "durable",
  });
  const elapsed = process.hrtime.bigint() - start;
  ingestionTotalNs += elapsed;
  ingestionMs.push(Number(elapsed) / 1e6);
}

const retrievalMs = [];
let retrievalTotalNs = 0n;
for (let index = 0; index < retrievalCount; index += 1) {
  const start = process.hrtime.bigint();
  engine.processText(queries[index % queries.length], `benchmark/query/${index % 3}`, {
    retention: "ephemeral",
  });
  const elapsed = process.hrtime.bigint() - start;
  retrievalTotalNs += elapsed;
  retrievalMs.push(Number(elapsed) / 1e6);
}
engine.flush();

const maxRssKiB = process.resourceUsage().maxRSS;
const output = {
  sample: {
    retrieval_p50_ms: percentile(retrievalMs, 0.50),
    retrieval_p95_ms: percentile(retrievalMs, 0.95),
    ingestion_p50_ms: percentile(ingestionMs, 0.50),
    ingestion_p95_ms: percentile(ingestionMs, 0.95),
    retrieval_ops_per_second:
      retrievalCount / (Number(retrievalTotalNs) / 1e9),
    ingestion_ops_per_second:
      messageCount / (Number(ingestionTotalNs) / 1e9),
    peak_rss_bytes: Math.round(maxRssKiB * 1024),
    database_bytes: databaseBytes(dbPath),
  },
  identity: {
    corpus_sha256: corpusDigest,
    model_sha256: sha256File(modelPath),
  },
  runtime: {
    shared_objects:
      process.report && typeof process.report.getReport === "function"
        ? process.report.getReport().sharedObjects
        : [],
  },
};
process.stdout.write(`${JSON.stringify(output)}\n`);
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def script_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def package_runtime_sha256(package_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in PACKAGE_RUNTIME_FILES:
        path = package_root / relative
        if not path.is_file():
            raise RuntimeError(f"package runtime file does not exist: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_js_package_root(
    *,
    explicit: str | None = None,
    source_root: Path | None = None,
    fallback: Path | None = None,
) -> Path:
    """Locate the JS package root after in-tree bindings were removed.

    Preference order:
    1. explicit --candidate-package-root / --package-root
    2. legacy in-tree bindings/javascript (older checkouts)
    3. sibling checkout ../cortext.ts next to the engine source root
    4. fallback package root (usually the baseline package)
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    if source_root is not None:
        root = source_root.expanduser().resolve()
        candidates.append(root / "bindings" / "javascript")
        candidates.append(root.parent / "cortext.ts")
    if fallback is not None:
        candidates.append(fallback.expanduser().resolve())

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if (path / "package.json").is_file():
            return path
    raise RuntimeError(
        "JavaScript package root not found. Pass --candidate-package-root "
        "pointing at a cortext.ts checkout (in-tree bindings/javascript was removed)."
    )


def candidate_source_sha256(source_root: Path) -> str:
    pathspecs = [
        "CMakeLists.txt",
        "cmake",
        "include",
        "src",
        "ffi/node",
        "third_party",
    ]
    listed = subprocess.check_output(
        [
            "git",
            "-C",
            str(source_root),
            "ls-files",
            "-co",
            "--exclude-standard",
            "--",
            *pathspecs,
        ],
        text=True,
    ).splitlines()
    digest = hashlib.sha256()
    for relative in sorted(set(listed)):
        path = source_root / relative
        if not path.is_file():
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verified_tree_patch_sha256(source_root: Path, base: str, tree: str) -> str:
    subprocess.run(
        ["git", "-C", str(source_root), "cat-file", "-e", f"{tree}^{{tree}}"],
        check=True,
    )
    patch = subprocess.check_output(
        ["git", "-C", str(source_root), "diff", "--binary", base, tree]
    )
    return hashlib.sha256(patch).hexdigest()


def resolve_model_path() -> Path:
    configured = os.environ.get("CORTEXT_AIST_MODEL_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
        raise RuntimeError(f"CORTEXT_AIST_MODEL_PATH is not a file: {path}")

    cache = (
        Path.home()
        / "Library"
        / "Caches"
        / "augmem"
        / "cortext"
        / "models"
        / "AIST-87M-GGUF"
        / "AIST-87M_q8_0.gguf"
    )
    if cache.is_file():
        return cache
    raise RuntimeError("set CORTEXT_AIST_MODEL_PATH to the verified benchmark model")


def resolve_addon(package_root: Path, addon: str | None) -> Path:
    if addon:
        path = Path(addon).expanduser().resolve()
    else:
        architecture = platform.machine().lower()
        node_architecture = {
            "aarch64": "arm64",
            "arm64": "arm64",
            "amd64": "x64",
            "x86_64": "x64",
        }.get(architecture, architecture)
        tag = f"{sys.platform}-{node_architecture}"
        path = package_root / "prebuilds" / tag / "cortext.node"
    if not path.is_file():
        raise RuntimeError(f"Node addon does not exist: {path}")
    return path


def cmake_cache_record(cache_path: Path) -> dict[str, str]:
    if not cache_path.is_file():
        raise RuntimeError(f"candidate CMake cache does not exist: {cache_path}")
    values: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("//", "#")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        if ":" not in key_and_type:
            continue
        key, _ = key_and_type.split(":", 1)
        if key in CANDIDATE_BUILD_KEYS:
            values[key] = value
    missing = sorted(set(CANDIDATE_BUILD_KEYS) - values.keys())
    if missing:
        raise RuntimeError(f"candidate CMake cache is missing keys: {missing}")
    return {key: values[key] for key in CANDIDATE_BUILD_KEYS}


def compiler_sha256(build_config: dict[str, str]) -> str:
    compiler = Path(build_config["CMAKE_CXX_COMPILER"]).expanduser().resolve()
    if not compiler.is_file():
        raise RuntimeError(f"configured candidate compiler does not exist: {compiler}")
    return sha256_file(compiler)


def _macos_rpaths(addon: Path) -> list[str]:
    output = subprocess.check_output(["otool", "-l", str(addon)], text=True)
    return re.findall(r"\n\s*path\s+(\S+)\s+\(offset\s+\d+\)", output)


def _expand_macos_loader_path(value: str, addon: Path) -> Path | None:
    if value.startswith("@loader_path"):
        suffix = value[len("@loader_path") :].lstrip("/")
        return (addon.parent / suffix).resolve()
    if value.startswith("/"):
        return Path(value).resolve()
    return None


def verify_candidate_runtime(addon: Path, runtime: Path) -> str:
    runtime = runtime.resolve()
    if not runtime.is_file():
        raise RuntimeError(f"candidate core runtime does not exist: {runtime}")
    if sys.platform == "darwin":
        dependencies = subprocess.check_output(["otool", "-L", str(addon)], text=True)
        dependency = next(
            (
                line.strip().split(" ", 1)[0]
                for line in dependencies.splitlines()[1:]
                if line.strip().split(" ", 1)[0].endswith(f"/{runtime.name}")
            ),
            None,
        )
        if dependency is None:
            raise RuntimeError(
                f"candidate addon does not declare the core runtime {runtime.name}"
            )
        resolved: list[Path] = []
        if dependency.startswith("@rpath/"):
            suffix = dependency[len("@rpath/") :]
            for rpath in _macos_rpaths(addon):
                root = _expand_macos_loader_path(rpath, addon)
                if root is not None:
                    resolved.append((root / suffix).resolve())
        elif dependency.startswith("@loader_path/"):
            resolved.append((addon.parent / dependency[len("@loader_path/") :]).resolve())
        elif dependency.startswith("/"):
            resolved.append(Path(dependency).resolve())
        if runtime not in resolved:
            raise RuntimeError(
                "candidate runtime is not in the addon's resolved dependency paths: "
                f"dependency={dependency}, resolved={resolved}, runtime={runtime}"
            )
        return dependency
    if sys.platform.startswith("linux"):
        dependencies = subprocess.check_output(["ldd", str(addon)], text=True)
        for line in dependencies.splitlines():
            if "=>" not in line:
                continue
            name, resolved = (part.strip() for part in line.split("=>", 1))
            resolved_path = resolved.split(" ", 1)[0]
            if name == runtime.name and Path(resolved_path).resolve() == runtime:
                return name
        raise RuntimeError(f"candidate addon does not resolve to runtime {runtime}")
    raise RuntimeError(f"candidate runtime verification is unsupported on {sys.platform}")


def machine_id() -> str:
    return "|".join(
        [
            platform.system(),
            platform.release(),
            platform.machine(),
            platform.processor(),
        ]
    )


def node_version() -> str:
    return subprocess.check_output(["node", "--version"], text=True).strip()


def run_sample(
    package_root: Path,
    addon: Path,
    model: Path,
    expected_runtime: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str] | None]:
    with tempfile.TemporaryDirectory(prefix="cortext-ltm-ranking-") as directory:
        root = Path(directory)
        worker = root / "worker.js"
        worker.write_text(WORKER_SOURCE, encoding="utf-8")
        db_path = root / "sample.sqlite"
        environment = os.environ.copy()
        for key in (
            "DYLD_LIBRARY_PATH",
            "DYLD_FALLBACK_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
        ):
            environment.pop(key, None)
        environment.update(
            {
                "CORTEXT_BENCH_PACKAGE_ROOT": str(package_root),
                "CORTEXT_BENCH_DB_PATH": str(db_path),
                "CORTEXT_NODE_ADDON_PATH": str(addon),
                "CORTEXT_AIST_MODEL_PATH": str(model),
            }
        )
        result = subprocess.run(
            ["node", "--no-warnings", str(worker)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        decoded = json.loads(result.stdout)
        runtime_identity = None
        if expected_runtime is not None:
            expected = expected_runtime.resolve()
            loaded = {
                Path(value).resolve()
                for value in decoded["runtime"]["shared_objects"]
                if isinstance(value, str) and Path(value).is_absolute()
            }
            if expected not in loaded:
                raise RuntimeError(
                    "candidate worker did not report the expected loaded runtime: "
                    f"expected={expected}, loaded={sorted(str(path) for path in loaded)}"
                )
            runtime_identity = {
                "path": str(expected),
                "sha256": sha256_file(expected),
            }
        return decoded["sample"], decoded["identity"], runtime_identity


def assert_identity(expected: dict[str, str] | None, actual: dict[str, str]) -> dict[str, str]:
    if expected is not None and actual != expected:
        raise RuntimeError(f"benchmark identity changed: expected {expected}, got {actual}")
    return actual


def environment_record(identity: dict[str, str]) -> dict[str, Any]:
    return {
        "model_sha256": identity["model_sha256"],
        "corpus_sha256": identity["corpus_sha256"],
        "machine_id": machine_id(),
        "node_version": node_version(),
    }


def protocol_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "warmup_pairs": args.warmups,
        "measured_pairs": args.pairs,
        "order_seed": args.order_seed,
        "confidence": 0.95,
        "non_inferiority_margin": 0,
        "fresh_process": True,
        "fresh_database": True,
        "candidate_runtime_verified_per_sample": True,
    }


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def bootstrap_upper_bound(values: list[float], seed: int) -> float:
    rng = random.Random(seed)
    resampled = []
    for _ in range(20_000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        resampled.append(median(sample))
    resampled.sort()
    index = min(len(resampled) - 1, math.ceil(0.95 * len(resampled)) - 1)
    return float(resampled[index])


def metric_records(raw_pairs: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    output = []
    for offset, (name, direction) in enumerate(METRIC_DIRECTIONS.items()):
        if direction == "lower-is-better":
            differences = [
                float(pair["candidate"][name]) - float(pair["baseline"][name])
                for pair in raw_pairs
            ]
        else:
            differences = [
                float(pair["baseline"][name]) - float(pair["candidate"][name])
                for pair in raw_pairs
            ]
        point = median(differences)
        upper = bootstrap_upper_bound(differences, seed + offset)
        output.append(
            {
                "name": name,
                "direction": direction,
                "paired_median_difference": point,
                "one_sided_95_upper_bound": upper,
                "margin": 0,
                "pass": point <= 0 and upper <= 0,
            }
        )
    return output


def schema_path(output_path: Path) -> Path:
    configured = os.environ.get("CORTEXT_LTM_BENCHMARK_SCHEMA")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().with_name("ltm-ranking-performance.schema.json")


def validate_artifact(artifact: dict[str, Any], output_path: Path) -> None:
    schema_file = schema_path(output_path)
    if not schema_file.is_file():
        raise RuntimeError(f"performance schema does not exist: {schema_file}")
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    try:
        import jsonschema
    except ModuleNotFoundError as error:
        requirements = Path(__file__).resolve().with_name(
            "requirements-ltm-ranking-benchmark.txt"
        )
        raise RuntimeError(
            "install the benchmark validator dependency with "
            f"'{sys.executable} -m pip install -r {requirements}'"
        ) from error
    jsonschema.Draft202012Validator(schema).validate(artifact)


def write_artifact(artifact: dict[str, Any], output_path: Path) -> None:
    validate_artifact(artifact, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def initial_order(seed: int) -> bool:
    return bool(random.Random(seed).getrandbits(1))


def capture(args: argparse.Namespace) -> None:
    package_root = Path(args.package_root).expanduser().resolve()
    addon = resolve_addon(package_root, args.addon)
    model = resolve_model_path()
    identity = None
    for _ in range(args.warmups):
        _, observed, _ = run_sample(package_root, addon, model)
        identity = assert_identity(identity, observed)

    raw_samples = []
    for _ in range(args.pairs):
        sample, observed, _ = run_sample(package_root, addon, model)
        identity = assert_identity(identity, observed)
        raw_samples.append(sample)

    assert identity is not None
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "phase": "baseline",
        "target_sha": args.target_sha,
        "benchmark_script_sha256": script_sha256(),
        "artifact_identity": {
            "baseline_addon_sha256": sha256_file(addon),
            "baseline_package_sha256": package_runtime_sha256(package_root),
        },
        "environment": environment_record(identity),
        "protocol": protocol_record(args),
        "raw_samples": raw_samples,
    }
    write_artifact(artifact, Path(args.output).expanduser().resolve())


def compare(args: argparse.Namespace) -> None:
    baseline_path = Path(args.baseline).expanduser().resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    validate_artifact(baseline, baseline_path)
    current_script = script_sha256()
    if baseline["benchmark_script_sha256"] != current_script:
        raise RuntimeError("benchmark script digest differs from the frozen baseline")

    package_root = Path(args.package_root).expanduser().resolve()
    baseline_addon = resolve_addon(package_root, None)
    candidate_addon = resolve_addon(package_root, args.candidate_addon)
    candidate_runtime = Path(args.candidate_runtime).expanduser().resolve()
    candidate_runtime_dependency = verify_candidate_runtime(
        candidate_addon, candidate_runtime
    )
    candidate_cmake_cache = Path(args.candidate_cmake_cache).expanduser().resolve()
    candidate_build_config = cmake_cache_record(candidate_cmake_cache)
    candidate_source_root = Path(args.candidate_source_root).expanduser().resolve()
    candidate_package_root = resolve_js_package_root(
        explicit=getattr(args, "candidate_package_root", None),
        source_root=candidate_source_root,
        fallback=package_root,
    )
    baseline_addon_sha256 = sha256_file(baseline_addon)
    if baseline["artifact_identity"]["baseline_addon_sha256"] != baseline_addon_sha256:
        raise RuntimeError("baseline addon digest differs from the frozen baseline")
    baseline_package_sha256 = package_runtime_sha256(package_root)
    if (
        baseline["artifact_identity"]["baseline_package_sha256"]
        != baseline_package_sha256
    ):
        raise RuntimeError("baseline JavaScript package differs from the frozen baseline")
    model = resolve_model_path()
    identity = None
    baseline_first = initial_order(args.order_seed)

    def run_pair(pair_index: int) -> dict[str, Any]:
        nonlocal identity
        order_baseline_first = baseline_first if pair_index % 2 == 0 else not baseline_first
        order = "baseline-first" if order_baseline_first else "candidate-first"
        samples: dict[str, dict[str, Any]] = {}
        targets = (
            [
                ("baseline", package_root, baseline_addon),
                ("candidate", candidate_package_root, candidate_addon),
            ]
            if order_baseline_first
            else [
                ("candidate", candidate_package_root, candidate_addon),
                ("baseline", package_root, baseline_addon),
            ]
        )
        for label, sample_package_root, addon in targets:
            expected_runtime = candidate_runtime if label == "candidate" else None
            sample, observed, runtime_identity = run_sample(
                sample_package_root, addon, model, expected_runtime
            )
            identity = assert_identity(identity, observed)
            samples[label] = sample
            if runtime_identity is not None:
                samples["candidate_runtime"] = runtime_identity
        return {
            "pair": pair_index + 1,
            "order": order,
            "baseline": samples["baseline"],
            "candidate": samples["candidate"],
            "candidate_runtime": samples["candidate_runtime"],
        }

    for warmup in range(args.warmups):
        run_pair(warmup)
    raw_pairs = [run_pair(index) for index in range(args.pairs)]
    assert identity is not None
    environment = environment_record(identity)
    if environment != baseline["environment"]:
        raise RuntimeError(
            "candidate environment differs from baseline: "
            f"expected {baseline['environment']}, got {environment}"
        )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "phase": "candidate",
        "target_sha": baseline["target_sha"],
        "benchmark_script_sha256": current_script,
        "artifact_identity": {
            "baseline_addon_sha256": baseline_addon_sha256,
            "baseline_package_sha256": baseline_package_sha256,
            "candidate_addon_sha256": sha256_file(candidate_addon),
            "candidate_package_sha256": package_runtime_sha256(
                candidate_package_root
            ),
            "candidate_runtime_sha256": sha256_file(candidate_runtime),
            "candidate_runtime_dependency": candidate_runtime_dependency,
            "candidate_cmake_cache_sha256": sha256_file(candidate_cmake_cache),
            "candidate_build_config": candidate_build_config,
            "candidate_compiler_sha256": compiler_sha256(candidate_build_config),
            "candidate_tree": args.candidate_tree,
            "candidate_patch_sha256": verified_tree_patch_sha256(
                candidate_source_root, baseline["target_sha"], args.candidate_tree
            ),
            "candidate_source_sha256": candidate_source_sha256(candidate_source_root),
        },
        "environment": environment,
        "protocol": protocol_record(args),
        "raw_pairs": raw_pairs,
        "metrics": metric_records(raw_pairs, args.order_seed),
    }
    write_artifact(artifact, Path(args.output).expanduser().resolve())


def bind(args: argparse.Namespace) -> None:
    artifact_path = Path(args.artifact).expanduser().resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    validate_artifact(artifact, artifact_path)
    if artifact["phase"] != "candidate":
        raise RuntimeError("only candidate artifacts can be rebound")
    if artifact["benchmark_script_sha256"] != script_sha256():
        raise RuntimeError("candidate artifact benchmark script digest changed")

    source_root = Path(args.candidate_source_root).expanduser().resolve()
    candidate_package_root = resolve_js_package_root(
        explicit=getattr(args, "candidate_package_root", None),
        source_root=source_root,
    )
    addon = Path(args.candidate_addon).expanduser().resolve()
    runtime = Path(args.candidate_runtime).expanduser().resolve()
    cmake_cache = Path(args.candidate_cmake_cache).expanduser().resolve()
    identity = artifact["artifact_identity"]
    if sha256_file(addon) != identity["candidate_addon_sha256"]:
        raise RuntimeError("candidate addon changed after measurement")
    if (
        package_runtime_sha256(candidate_package_root)
        != identity["candidate_package_sha256"]
    ):
        raise RuntimeError("candidate JavaScript package changed after measurement")
    source_digest = candidate_source_sha256(source_root)
    if source_digest != identity["candidate_source_sha256"]:
        raise RuntimeError("candidate build sources changed after measurement")
    dependency = verify_candidate_runtime(addon, runtime)
    if dependency != identity["candidate_runtime_dependency"]:
        raise RuntimeError("candidate addon runtime dependency changed after measurement")
    if sha256_file(runtime) != identity["candidate_runtime_sha256"]:
        raise RuntimeError("candidate core runtime changed after measurement")
    if sha256_file(cmake_cache) != identity["candidate_cmake_cache_sha256"]:
        raise RuntimeError("candidate CMake cache changed after measurement")
    build_config = cmake_cache_record(cmake_cache)
    if build_config != identity["candidate_build_config"]:
        raise RuntimeError("candidate build configuration changed after measurement")
    if compiler_sha256(build_config) != identity["candidate_compiler_sha256"]:
        raise RuntimeError("candidate compiler changed after measurement")
    expected_runtime_path = str(runtime.resolve())
    expected_runtime_sha256 = sha256_file(runtime)
    for pair in artifact["raw_pairs"]:
        observed = pair["candidate_runtime"]
        if observed["path"] != expected_runtime_path:
            raise RuntimeError("measured candidate runtime path changed")
        if observed["sha256"] != expected_runtime_sha256:
            raise RuntimeError("measured candidate runtime digest changed")
    identity["candidate_tree"] = args.candidate_tree
    identity["candidate_patch_sha256"] = verified_tree_patch_sha256(
        source_root, artifact["target_sha"], args.candidate_tree
    )
    write_artifact(artifact, artifact_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--package-root", required=True)
    capture_parser.add_argument("--addon", required=True)
    capture_parser.add_argument("--target-sha", required=True)
    capture_parser.add_argument("--pairs", type=int, choices=(9, 18, 36, 72), required=True)
    capture_parser.add_argument("--warmups", type=int, choices=(2,), required=True)
    capture_parser.add_argument("--order-seed", type=int, choices=(20260714,), required=True)
    capture_parser.add_argument("--output", required=True)
    capture_parser.set_defaults(func=capture)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--package-root", required=True)
    compare_parser.add_argument("--candidate-addon", required=True)
    compare_parser.add_argument("--candidate-runtime", required=True)
    compare_parser.add_argument("--candidate-cmake-cache", required=True)
    compare_parser.add_argument("--candidate-tree", required=True)
    compare_parser.add_argument("--candidate-source-root", required=True)
    compare_parser.add_argument(
        "--candidate-package-root",
        default=None,
        help=(
            "JS package root for the candidate measurement (cortext.ts checkout). "
            "Defaults to bindings/javascript, sibling ../cortext.ts, or --package-root."
        ),
    )
    compare_parser.add_argument("--pairs", type=int, choices=(9, 18, 36, 72), required=True)
    compare_parser.add_argument("--warmups", type=int, choices=(2,), required=True)
    compare_parser.add_argument("--order-seed", type=int, choices=(20260714,), required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.set_defaults(func=compare)

    bind_parser = subparsers.add_parser("bind")
    bind_parser.add_argument("--artifact", required=True)
    bind_parser.add_argument("--candidate-addon", required=True)
    bind_parser.add_argument("--candidate-runtime", required=True)
    bind_parser.add_argument("--candidate-cmake-cache", required=True)
    bind_parser.add_argument("--candidate-tree", required=True)
    bind_parser.add_argument("--candidate-source-root", required=True)
    bind_parser.add_argument(
        "--candidate-package-root",
        default=None,
        help=(
            "JS package root for the candidate artifact (cortext.ts checkout). "
            "Defaults to bindings/javascript or sibling ../cortext.ts."
        ),
    )
    bind_parser.set_defaults(func=bind)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.target_sha) != 40 if args.command == "capture" else False:
        raise RuntimeError("target SHA must be a full 40-character Git object id")
    if args.command in ("compare", "bind"):
        if len(args.candidate_tree) != 40:
            raise RuntimeError("candidate tree must be a full 40-character Git object id")
        int(args.candidate_tree, 16)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
