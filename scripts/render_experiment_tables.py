#!/usr/bin/env python3
"""
Render compact markdown tables from harness summary CSVs and optional scorer JSONs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_COLUMNS = [
    "name",
    "turns",
    "consolidation_runs",
    "retrieval_semantic_overlap_mean",
    "interrupt_precision",
    "interrupt_recall",
    "duration_s",
]


def load_rows(summary_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in summary_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)
    return rows


def render_markdown(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render study markdown tables from summary CSVs.")
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Summary CSV path. May be supplied multiple times.",
    )
    parser.add_argument(
        "--columns",
        default=",".join(DEFAULT_COLUMNS),
        help="Comma-separated column list.",
    )
    parser.add_argument(
        "--score-json",
        action="append",
        default=[],
        help="Optional scorer summary JSON path. May be supplied multiple times.",
    )
    parser.add_argument("--title", default="", help="Optional markdown title.")
    parser.add_argument("--out", default="", help="Optional output markdown path.")
    args = parser.parse_args()

    summary_paths = [Path(item) for item in args.summary]
    rows = load_rows(summary_paths)
    columns = [item.strip() for item in args.columns.split(",") if item.strip()]

    parts = []
    if args.title:
        parts.append(f"## {args.title}")
        parts.append("")
    if rows:
        parts.append(render_markdown(rows, columns))
    for score_path in args.score_json:
        payload = json.loads(Path(score_path).read_text(encoding="utf-8"))
        parts.append("")
        parts.append(f"### {Path(score_path).stem}")
        parts.append("")
        parts.append("```json")
        parts.append(json.dumps(payload, indent=2))
        parts.append("```")

    output = "\n".join(parts).strip() + "\n"
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
