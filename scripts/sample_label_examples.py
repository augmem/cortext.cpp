#!/usr/bin/env python3
"""
Deterministically subsample span-label JSONL files with optional per-type caps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def stable_rank(key: str) -> bytes:
    return hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()


def parse_label_caps(values: list[str]) -> dict[str, int]:
    caps: dict[str, int] = {}
    for value in values:
        label, _, raw_cap = value.partition("=")
        if not label or not raw_cap:
            raise SystemExit(f"Invalid --label-cap '{value}', expected label=N")
        caps[label] = int(raw_cap)
    return caps


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministically subsample label JSONL examples.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--label-cap",
        action="append",
        default=[],
        help="Per-type cap in the form label=N. Repeatable.",
    )
    parser.add_argument(
        "--overall-cap",
        type=int,
        default=0,
        help="Optional total cap after label capping. 0 keeps all rows.",
    )
    args = parser.parse_args()

    label_caps = parse_label_caps(args.label_cap)
    rows = load_rows(Path(args.input))
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("type_label", ""))].append(row)

    kept: list[dict] = []
    for label, bucket in buckets.items():
        bucket.sort(key=lambda row: stable_rank(str(row.get("id", ""))))
        cap = label_caps.get(label, 0)
        if cap > 0:
            kept.extend(bucket[:cap])
        else:
            kept.extend(bucket)

    kept.sort(key=lambda row: stable_rank(str(row.get("id", ""))))
    if args.overall_cap > 0:
        kept = kept[: args.overall_cap]

    write_rows(Path(args.output), kept)
    print(f"[OK] kept {len(kept)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
