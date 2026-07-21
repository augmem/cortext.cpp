#!/usr/bin/env python3
"""Build the sanitized nine-point active-signal-ring ablation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


POINTS = {
    "midpoint": (0.5, 0.5, 0.5),
    "all-low": (0.0, 0.0, 0.0),
    "all-high": (1.0, 1.0, 1.0),
    "focus-low": (0.0, 0.5, 0.5),
    "focus-high": (1.0, 0.5, 0.5),
    "sensitivity-low": (0.5, 0.0, 0.5),
    "sensitivity-high": (0.5, 1.0, 0.5),
    "stability-low": (0.5, 0.5, 0.0),
    "stability-high": (0.5, 0.5, 1.0),
}
RING_MUTATION_COUNTER = (
    "SignalProcessor.commit_table_row_count.cortext_active_signal_embeddings"
)
SOURCE_MANIFEST_PATHS = (
    "CMakeLists.txt",
    "examples/benchmark/chat_replay_live_run.cpp",
    "src/operations/active_signal_embedding_ring_internal.hpp",
    "src/operations/bounded_activation_shadow_internal.hpp",
    "src/operations/emotion_cascade.cpp",
    "src/operations/graph_retrieval.cpp",
    "src/operations/memory_storage.cpp",
    "src/operations/sparse_retrieval_knobs_internal.hpp",
    "src/operations/sparse_retrieval_route_internal.cpp",
    "src/operations/sparse_retrieval_route_internal.hpp",
    "src/operations/sparse_retrieval_route_sqlite_internal.cpp",
    "src/operations/sparse_retrieval_route_sqlite_internal.hpp",
    "src/signal_processor.cpp",
    "src/store.cpp",
    "src/store/commit_profile_internal.hpp",
    "src/store/schema.cpp",
    "tests/CMakeLists.txt",
    "tests/core_knobs.test.cpp",
    "tests/migration_core.test.cpp",
    "tests/operations_emotion_cascade.test.cpp",
    "tests/operations_memory_storage.test.cpp",
    "tests/operations_graph_retrieval.test.cpp",
    "tests/signal_processor.test.cpp",
    "tools/audit_public_retrieval_control.py",
    "tools/audit_public_retrieval_control_test.py",
    "tools/audit_storage_cost_profile.py",
    "tools/audit_storage_cost_profile_test.py",
    "tools/build_signal_centroid_ring_ablation.py",
    "tools/build_signal_centroid_ring_ablation_test.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def expected_capacity(focus: float, sensitivity: float, stability: float) -> int:
    value = 64 + 64 * focus + 32 * sensitivity + 32 * stability
    # Match C++ std::lround exactly. The clamped production expression is
    # nonnegative, so half-away-from-zero is floor(value + 0.5).
    return math.floor(value + 0.5)


def source_code_manifest(repo: Path) -> tuple[str, list[dict[str, str]]]:
    """Fingerprint the exact implementation/evidence files owned by this claim."""
    entries = []
    digest = hashlib.sha256()
    for relative in SOURCE_MANIFEST_PATHS:
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"source manifest path does not exist: {relative}")
        file_digest = file_sha256(path)
        entries.append({"path": relative, "sha256": file_digest})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest))
    return digest.hexdigest(), entries


def build(
    raw_dir: Path,
    audit_dir: Path,
    binary: Path,
    repo: Path,
) -> dict[str, Any]:
    runs = []
    for name, knobs in POINTS.items():
        profile_path = raw_dir / f"{name}.json"
        audit_path = audit_dir / f"{name}-audit.json"
        profile = load_json(profile_path)
        audit = load_json(audit_path)
        actual_knobs = tuple(float(profile[key]) for key in (
            "focus", "sensitivity", "stability"
        ))
        if actual_knobs != knobs:
            raise ValueError(f"{name} knob tuple differs from the contract")
        ring = profile.get("active_signal_embedding_ring")
        if not isinstance(ring, dict):
            raise ValueError(f"{name} lacks active signal ring parameters")
        capacity = expected_capacity(*knobs)
        if ring.get("capacity") != capacity:
            raise ValueError(f"{name} ring capacity is not knob-derived")
        if ring.get("modality_branching") is not False:
            raise ValueError(f"{name} introduces modality branching")
        if ring.get("source_id_branching") is not False:
            raise ValueError(f"{name} introduces source-id branching")
        rows = profile.get("working_set_curve")
        if not isinstance(rows, list) or len(rows) != 2000:
            raise ValueError(f"{name} lacks complete 2000-event profile rows")
        event_indices = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{name} has an invalid profile row")
            event_index = row.get("event_index")
            if (not isinstance(event_index, int) or isinstance(event_index, bool)
                    or event_index < 0 or event_index >= 2000
                    or event_index in event_indices):
                raise ValueError(f"{name} has invalid or duplicate event coverage")
            event_indices.add(event_index)
        if event_indices != set(range(2000)):
            raise ValueError(f"{name} has incomplete event coverage")
        mutation_values = []
        for row in rows:
            operations = row.get("operation_ms")
            if not isinstance(operations, dict):
                raise ValueError(f"{name} has an invalid operation map")
            if RING_MUTATION_COUNTER not in operations:
                continue
            value = operations[RING_MUTATION_COUNTER]
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} has an invalid ring mutation counter")
            mutation_values.append(float(value))
        if not mutation_values:
            raise ValueError(f"{name} lacks the ring mutation counter")
        maximum_mutations = max(mutation_values)
        if profile.get("processed_events") != 2000:
            raise ValueError(f"{name} did not process exactly 2000 events")
        if audit.get("passed") is not True:
            raise ValueError(f"{name} public retrieval audit failed")
        if audit.get("private_profile_sha256") != file_sha256(profile_path):
            raise ValueError(f"{name} retrieval audit is not bound to its profile")
        if audit.get("query_count") != 512:
            raise ValueError(f"{name} did not execute exactly 512 queries")
        source_digest_count = audit.get("source_digest_count")
        if not isinstance(source_digest_count, int) or source_digest_count < 4:
            raise ValueError(f"{name} lacks four opaque source identifiers")
        for field in (
            "exact_top1", "exact_identity_recall_at_k", "semantic_coverage"
        ):
            if audit.get(field) != 1.0:
                raise ValueError(f"{name} {field} is not exactly 1.0")
        if audit.get("deterministic_tie_order") is not True:
            raise ValueError(f"{name} deterministic ties did not pass")
        runs.append({
            "name": name,
            "focus": knobs[0],
            "sensitivity": knobs[1],
            "stability": knobs[2],
            "capacity": capacity,
            "processed_events": profile.get("processed_events"),
            "wall_ms": profile.get("wall_ms"),
            "maximum_ring_row_mutations_per_event": maximum_mutations,
            "query_count": audit.get("query_count"),
            "opaque_source_digest_count": source_digest_count,
            "exact_top1": audit.get("exact_top1"),
            "exact_identity_recall_at_16": audit.get("exact_identity_recall_at_k"),
            "semantic_coverage": audit.get("semantic_coverage"),
            "deterministic_tie_order": audit.get("deterministic_tie_order"),
            "passed": True,
            "profile_sha256": file_sha256(profile_path),
            "audit_sha256": file_sha256(audit_path),
        })
    source_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    source_manifest_sha256, source_manifest = source_code_manifest(repo)
    return {
        "schema": "cortext_signal_centroid_ring_knob_ablation_v4",
        "capacity_formula": (
            "std::lround(64 + 64F + 32S + 32T) after independent [0,1] "
            "clamping; half-integer ties round away from zero"
        ),
        "all_low_capacity": 64,
        "default_capacity": 128,
        "all_high_capacity": 192,
        "structural_grid": (
            "27 F/S/T points are covered by the core formula regression; "
            "these nine production-shaped runs isolate the midpoint, endpoints, "
            "and each knob axis"
        ),
        "production_shaped_query_count_per_run": 512,
        "modality_branching": False,
        "source_id_branching": False,
        "source_head": source_head,
        "source_code_manifest_sha256": source_manifest_sha256,
        "source_code_manifest": source_manifest,
        "benchmark_binary_sha256": file_sha256(binary),
        "all_passed": all(run["passed"] for run in runs),
        "runs": runs,
        "nonclaims": [
            "whole-engine plateau",
            "emotional-cascade traversal boundedness",
            "merge",
            "release",
            "deployment",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.raw_dir, args.audit_dir, args.binary, args.repo)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
