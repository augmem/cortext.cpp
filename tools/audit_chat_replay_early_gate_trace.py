#!/usr/bin/env python3
"""Audit where an existing chat-replay early-judge manifest would fail."""

from __future__ import annotations

import argparse
import json
import pathlib


def check_floor(name: str, value: float, floor: float) -> dict:
    return {
        "name": name,
        "value": value,
        "floor": floor,
        "status": "pass" if value >= floor else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--quality-gate-min-milestone", type=int, default=24)
    parser.add_argument("--min-quality-delta", type=float, default=-0.5)
    parser.add_argument("--min-win-rate", type=float)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rows = []
    first_failure = None
    for record in manifest.get("completed", []):
        milestone = int(record.get("milestone", 0) or 0)
        metrics = record.get("metrics", {})
        checks = []
        if milestone >= args.quality_gate_min_milestone:
            checks.append(
                check_floor(
                    "cortext_quality_delta_vs_traditional_chat_rag",
                    float(
                        metrics.get(
                            "cortext_quality_delta_vs_traditional_chat_rag", 0.0
                        )
                        or 0.0
                    ),
                    args.min_quality_delta,
                )
            )
            if args.min_win_rate is not None:
                checks.append(
                    check_floor(
                        "cortext_win_rate",
                        float(metrics.get("cortext_win_rate", 0.0) or 0.0),
                        args.min_win_rate,
                    )
                )
        else:
            checks.append(
                {
                    "name": "quality_gate_deferred",
                    "value": milestone,
                    "floor": args.quality_gate_min_milestone,
                    "status": "pass",
                }
            )
        status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
        row = {
            "milestone": milestone,
            "original_fail_fast_status": record.get("fail_fast_status"),
            "audited_status": status,
            "cortext_quality_delta_vs_traditional_chat_rag": metrics.get(
                "cortext_quality_delta_vs_traditional_chat_rag"
            ),
            "cortext_win_rate": metrics.get("cortext_win_rate"),
            "checks": checks,
        }
        rows.append(row)
        if first_failure is None and status == "fail":
            first_failure = row

    output = {
        "schema": "cortext_chat_replay_early_gate_trace_audit_v1",
        "source_manifest": str(args.manifest),
        "quality_gate_min_milestone": args.quality_gate_min_milestone,
        "min_quality_delta": args.min_quality_delta,
        "min_win_rate": args.min_win_rate,
        "first_failure": first_failure,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(args.out)
    return 0 if first_failure is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
