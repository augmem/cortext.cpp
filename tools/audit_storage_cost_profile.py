#!/usr/bin/env python3
"""Audit replay flatness and emit content-free behavior/database digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
import statistics
from pathlib import Path
from typing import Any, Sequence


STORAGE_OPERATIONS = (
    "cortext::operations::MemoryStorage",
    "MemoryStorage.supersession_edges",
)
LOGICAL_TABLES = {
    "memories": "memory_id",
    "associations": "source_memory_id, target_memory_id, edge_type",
    "signals": "signal_id",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def normalized_sql_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"blob_sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("logical database contains a non-finite float")
        return value
    return value


def logical_database_digest(db_path: Path) -> tuple[str, dict[str, int]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables: dict[str, Any] = {}
        counts: dict[str, int] = {}
        for table, ordering in LOGICAL_TABLES.items():
            cursor = connection.execute(f"SELECT * FROM {table} ORDER BY {ordering}")
            columns = [column[0] for column in cursor.description]
            rows = [
                [normalized_sql_value(value) for value in row]
                for row in cursor.fetchall()
            ]
            counts[table] = len(rows)
            tables[table] = {"columns": columns, "rows": rows}
        return digest(tables), counts
    finally:
        connection.close()


def behavior_digest(rows: Sequence[dict[str, Any]]) -> str:
    behavior = []
    for row in rows:
        if "behavior" not in row:
            raise ValueError("profile lacks behavior-oracle rows")
        public_behavior = json.loads(json.dumps(row["behavior"]))
        output = public_behavior.get("output")
        if isinstance(output, dict):
            # These public diagnostics measure execution duration and are not
            # behavioral outputs. Older profiles included them, so normalize
            # both old and new artifacts before content-addressing behavior.
            output.pop("soft_anchor_last_update_us", None)
            output.pop("soft_anchor_mean_update_us", None)
        behavior.append(
            {
                "event_index": row["event_index"],
                "modality": row["modality"],
                "behavior": public_behavior,
            }
        )
    return digest(behavior)


def theil_sen(xs: Sequence[float], ys: Sequence[float]) -> float:
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs))
        for j in range(i + 1, len(xs))
        if xs[j] != xs[i]
    ]
    if not slopes:
        raise ValueError("at least two distinct windows are required")
    return statistics.median(slopes)


def bootstrap_upper_slope(
    xs: Sequence[float], ys: Sequence[float], repetitions: int = 2000
) -> float:
    fitted_slope = theil_sen(xs, ys)
    intercept = statistics.median(
        y - fitted_slope * x for x, y in zip(xs, ys, strict=True)
    )
    residuals = [
        y - (intercept + fitted_slope * x)
        for x, y in zip(xs, ys, strict=True)
    ]
    rng = random.Random(0)
    slopes = []
    for _ in range(repetitions):
        sampled = [rng.choice(residuals) for _ in residuals]
        synthetic = [
            intercept + fitted_slope * x + residual
            for x, residual in zip(xs, sampled, strict=True)
        ]
        slopes.append(theil_sen(xs, synthetic))
    slopes.sort()
    return slopes[math.ceil(0.95 * len(slopes)) - 1]


def windows(values: Sequence[float], start: int, size: int) -> list[float]:
    return [
        statistics.mean(values[offset : offset + size])
        for offset in range(start, len(values) - size + 1, size)
    ]


def operation_values(
    rows: Sequence[dict[str, Any]], operation: str
) -> list[float]:
    values = []
    for row in rows:
        operation_ms = row.get("operation_ms")
        if not isinstance(operation_ms, dict) or operation not in operation_ms:
            raise ValueError(f"profile lacks full timing for {operation}")
        values.append(float(operation_ms[operation]))
    return values


def flatness_result(
    rows: Sequence[dict[str, Any]], operation: str, window_size: int
) -> dict[str, Any]:
    warmup = math.ceil(0.20 * len(rows))
    means = windows(operation_values(rows, operation), warmup, window_size)
    if len(means) < 10:
        return {
            "operation": operation,
            "passed": False,
            "reason": "at least ten post-warmup windows are required",
            "window_count": len(means),
        }
    xs = [float(warmup + window_size * index + window_size / 2) for index in range(len(means))]
    slope = theil_sen(xs, means)
    upper = bootstrap_upper_slope(xs, means)
    early = statistics.mean(means[:5])
    late = statistics.mean(means[-5:])
    ratio = late / early if early > 0.0 else math.inf
    return {
        "operation": operation,
        "passed": slope <= 0.01 and upper <= 0.02 and ratio <= 1.05,
        "window_count": len(means),
        "theil_sen_ms_per_message": slope,
        "bootstrap_95_upper_ms_per_message": upper,
        "final_five_over_first_five": ratio,
        "first_five_mean_ms": early,
        "final_five_mean_ms": late,
    }


def throughput_result(
    rows: Sequence[dict[str, Any]], window_size: int
) -> dict[str, Any]:
    warmup = math.ceil(0.20 * len(rows))
    process_means = windows(
        [float(row["process_ms"]) for row in rows], warmup, window_size
    )
    latency_means = windows(
        [float(row["latency_ms"]) for row in rows], warmup, window_size
    )
    if len(process_means) < 10 or len(latency_means) < 10:
        return {
            "passed": False,
            "reason": "at least ten post-warmup windows are required",
            "window_count": min(len(process_means), len(latency_means)),
        }
    messages_per_second = [
        1000.0 / latency_ms if latency_ms > 0.0 else math.inf
        for latency_ms in latency_means
    ]
    early_process = statistics.mean(process_means[:5])
    late_process = statistics.mean(process_means[-5:])
    early_rate = statistics.mean(messages_per_second[:5])
    late_rate = statistics.mean(messages_per_second[-5:])
    process_ratio = late_process / early_process if early_process > 0.0 else math.inf
    rate_ratio = late_rate / early_rate if early_rate > 0.0 else 0.0
    return {
        "passed": 0.90 <= process_ratio <= 1.10 and 0.90 <= rate_ratio <= 1.10,
        "window_count": len(process_means),
        "first_five_mean_process_ms": early_process,
        "final_five_mean_process_ms": late_process,
        "final_five_over_first_five_process": process_ratio,
        "first_five_mean_messages_per_second": early_rate,
        "final_five_mean_messages_per_second": late_rate,
        "final_five_over_first_five_messages_per_second": rate_ratio,
    }


def profile_summary(profile_path: Path, db_path: Path, window_size: int) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    rows = profile.get("working_set_curve")
    if not isinstance(rows, list) or not rows:
        raise ValueError("profile has no working-set curve")
    db_hash, counts = logical_database_digest(db_path)
    flatness = [flatness_result(rows, name, window_size) for name in STORAGE_OPERATIONS]
    throughput = throughput_result(rows, window_size)
    return {
        "schema": "cortext_storage_cost_profile_audit_v2",
        "processed_events": len(rows),
        "behavior_sha256": behavior_digest(rows),
        "logical_database_sha256": db_hash,
        "logical_table_counts": counts,
        "mean_process_ms": float(profile["mean_process_ms"]),
        "mean_total_ms": float(profile["mean_total_ms"]),
        "wall_ms": int(profile["wall_ms"]),
        "flatness": flatness,
        "flatness_passed": all(result["passed"] for result in flatness),
        "throughput": throughput,
        "throughput_passed": throughput["passed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    result = profile_summary(args.profile, args.db, args.window_size)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    passed = result["flatness_passed"] and result["throughput_passed"]
    return 0 if args.report_only or passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
