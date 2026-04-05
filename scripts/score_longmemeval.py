#!/usr/bin/env python3
"""
Score LongMemEval predictions against the external answer key sidecar.

Prediction schema:
  {"conversation_id": "...", "query_id": "...", "answer": "...", "abstained": false}
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


def match_answer(prediction: str, gold_answers: list[str]) -> bool:
    norm_pred = normalize_text(prediction)
    if not norm_pred:
        return False
    for gold in gold_answers:
        norm_gold = normalize_text(gold)
        if norm_gold and (norm_pred == norm_gold or norm_gold in norm_pred or norm_pred in norm_gold):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Score LongMemEval predictions.")
    parser.add_argument("--answer-key", required=True, help="Path to *.answer_key.jsonl.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl.")
    parser.add_argument("--out", default="", help="Optional JSON summary output path.")
    args = parser.parse_args()

    gold_rows = load_jsonl(Path(args.answer_key))
    pred_rows = load_jsonl(Path(args.predictions))
    pred_map = {
        (str(row.get("conversation_id", "")), str(row.get("query_id", row.get("conversation_id", "")))): row
        for row in pred_rows
    }

    totals = {"n": 0, "correct": 0, "abstain_required": 0, "abstain_correct": 0}
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})

    for gold in gold_rows:
        key = (
            str(gold.get("conversation_id", "")),
            str(gold.get("query_id", gold.get("conversation_id", ""))),
        )
        pred = pred_map.get(key, {})
        prediction = str(pred.get("answer", ""))
        abstained = bool(pred.get("abstained", False))
        gold_answers = [str(item) for item in gold.get("answers", [])]
        requires_abstention = bool(gold.get("requires_abstention", False))
        correct = False
        if requires_abstention:
            totals["abstain_required"] += 1
            correct = abstained
            if correct:
                totals["abstain_correct"] += 1
        else:
            correct = match_answer(prediction, gold_answers)
        totals["n"] += 1
        if correct:
            totals["correct"] += 1
        question_type = str(gold.get("question_type", "unknown"))
        by_type[question_type]["n"] += 1
        if correct:
            by_type[question_type]["correct"] += 1

    summary = {
        "n": totals["n"],
        "accuracy": (totals["correct"] / totals["n"]) if totals["n"] else 0.0,
        "abstention_accuracy": (
            totals["abstain_correct"] / totals["abstain_required"]
            if totals["abstain_required"]
            else 0.0
        ),
        "by_question_type": {
            key: {
                "n": value["n"],
                "accuracy": (value["correct"] / value["n"]) if value["n"] else 0.0,
            }
            for key, value in sorted(by_type.items())
        },
    }

    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
