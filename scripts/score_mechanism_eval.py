#!/usr/bin/env python3
"""
Score predictions on the synthetic mechanism pack.

Prediction schema:
  {
    "conversation_id": "...",
    "query_id": "...",
    "current_answer": "...",
    "historical_answer": "...",
    "belief_answer": "...",
    "abstained": false,
    "stale_first": false,
    "top_source_class": "direct_user_update"
  }
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def exactish(pred: object, gold: object) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score synthetic mechanism-eval predictions.")
    parser.add_argument("--answer-key", required=True, help="Path to valid.answer_key.jsonl.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl.")
    parser.add_argument("--out", default="", help="Optional JSON summary output path.")
    args = parser.parse_args()

    gold_rows = load_jsonl(Path(args.answer_key))
    pred_rows = load_jsonl(Path(args.predictions))
    pred_map = {
        (str(row.get("conversation_id", "")), str(row.get("query_id", ""))): row
        for row in pred_rows
    }

    totals = defaultdict(int)
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for gold in gold_rows:
        key = (str(gold.get("conversation_id", "")), str(gold.get("query_id", "")))
        pred = pred_map.get(key, {})
        family = str(gold.get("scenario_family", "unknown"))
        current_ok = exactish(pred.get("current_answer", ""), gold.get("expected_current", ""))
        historical_ok = exactish(
            pred.get("historical_answer", ""), gold.get("expected_historical", "")
        )
        belief_ok = exactish(pred.get("belief_answer", ""), gold.get("expected_belief", ""))
        abstain_ok = bool(pred.get("abstained", False)) == bool(gold.get("expected_abstain", False))
        stale_fail = bool(pred.get("stale_first", False))
        provenance_fail = normalize_text(pred.get("top_source_class", "")) not in {
            "",
            normalize_text(gold.get("provenance_class", "")),
        }

        totals["n"] += 1
        totals["current_correct"] += int(current_ok)
        totals["historical_correct"] += int(historical_ok)
        totals["belief_correct"] += int(belief_ok)
        totals["abstain_correct"] += int(abstain_ok)
        totals["stale_failures"] += int(stale_fail)
        totals["provenance_failures"] += int(provenance_fail)

        by_family[family]["n"] += 1
        by_family[family]["current_correct"] += int(current_ok)
        by_family[family]["historical_correct"] += int(historical_ok)
        by_family[family]["belief_correct"] += int(belief_ok)
        by_family[family]["stale_failures"] += int(stale_fail)

    summary = {
        "n": totals["n"],
        "current_accuracy": totals["current_correct"] / totals["n"] if totals["n"] else 0.0,
        "historical_accuracy": totals["historical_correct"] / totals["n"] if totals["n"] else 0.0,
        "belief_accuracy": totals["belief_correct"] / totals["n"] if totals["n"] else 0.0,
        "abstention_accuracy": totals["abstain_correct"] / totals["n"] if totals["n"] else 0.0,
        "stale_resurfacing_rate": totals["stale_failures"] / totals["n"] if totals["n"] else 0.0,
        "provenance_failure_rate": (
            totals["provenance_failures"] / totals["n"] if totals["n"] else 0.0
        ),
        "by_family": {
            family: {
                "n": values["n"],
                "current_accuracy": values["current_correct"] / values["n"] if values["n"] else 0.0,
                "historical_accuracy": values["historical_correct"] / values["n"] if values["n"] else 0.0,
                "belief_accuracy": values["belief_correct"] / values["n"] if values["n"] else 0.0,
                "stale_resurfacing_rate": values["stale_failures"] / values["n"] if values["n"] else 0.0,
            }
            for family, values in sorted(by_family.items())
        },
    }

    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
