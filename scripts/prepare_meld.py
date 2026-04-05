#!/usr/bin/env python3
"""
Convert MELD CSVs into Cortext JSONL conversation files.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def split_source_name(split: str) -> str:
    return {"train": "train_sent_emo.csv", "valid": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}[split]


def convert_split(raw_path: Path, out_dir: Path, split: str) -> int:
    grouped: dict[str, list[dict]] = defaultdict(list)
    meta_rows: dict[str, list[dict]] = defaultdict(list)
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dialogue_id = normalize_text(row.get("Dialogue_ID", ""))
            if not dialogue_id:
                continue
            grouped[dialogue_id].append(row)
            meta_rows[dialogue_id].append(
                {
                    "emotion": normalize_text(row.get("Emotion", "")).lower(),
                    "sentiment": normalize_text(row.get("Sentiment", "")).lower(),
                    "speaker": normalize_text(row.get("Speaker", "")),
                    "utterance_id": normalize_text(row.get("Utterance_ID", "")),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{split}.jsonl"
    meta_path = out_dir / f"{split}.metadata.jsonl"
    count = 0
    with jsonl_path.open("w", encoding="utf-8") as out, meta_path.open("w", encoding="utf-8") as meta:
        for dialogue_id, turns in grouped.items():
            turns.sort(key=lambda row: int(row.get("Utterance_ID", "0")))
            content = []
            for idx, row in enumerate(turns):
                message = normalize_text(row.get("Utterance", ""))
                if not message:
                    continue
                speaker = normalize_text(row.get("Speaker", "")) or f"agent_{(idx % 2) + 1}"
                content.append({"agent": speaker, "message": message})
            if not content:
                continue
            conversation_id = f"meld_{split}_{dialogue_id}"
            out.write(json.dumps([conversation_id, {"content": content}], ensure_ascii=True) + "\n")
            meta.write(
                json.dumps(
                    {
                        "conversation_id": conversation_id,
                        "source_file": str(raw_path),
                        "dialogue_id": dialogue_id,
                        "utterance_labels": meta_rows[dialogue_id],
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert MELD CSV to Cortext JSONL.")
    parser.add_argument(
        "--source-dir",
        default="data/raw/meld",
        help="Directory containing MELD raw CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/meld",
        help="Output directory for converted JSONL files.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    for split in ("train", "valid", "test"):
        raw_path = source_dir / split_source_name(split)
        if not raw_path.exists():
            raise SystemExit(f"Missing MELD file {raw_path}")
        count = convert_split(raw_path, out_dir, split)
        print(f"[OK] {split}: wrote {count} conversations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
