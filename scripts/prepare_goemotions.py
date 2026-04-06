#!/usr/bin/env python3
"""
Convert GoEmotions TSV rows into single-turn Cortext JSONL files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def load_emotions(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert_split(raw_path: Path, emotions: list[str], out_dir: Path, split: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{split}.jsonl"
    meta_path = out_dir / f"{split}.metadata.jsonl"
    count = 0
    with raw_path.open("r", encoding="utf-8") as handle, jsonl_path.open("w", encoding="utf-8") as out, meta_path.open("w", encoding="utf-8") as meta:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            text, label_ids, example_id = parts
            text = normalize_text(text)
            if not text:
                continue
            labels = []
            for raw_id in label_ids.split(","):
                raw_id = raw_id.strip()
                if not raw_id:
                    continue
                try:
                    labels.append(emotions[int(raw_id)])
                except (ValueError, IndexError):
                    continue
            conversation_id = f"goemotions_{split}_{example_id}"
            out.write(
                json.dumps(
                    [conversation_id, {"content": [{"agent": "agent_1", "message": text}]}],
                    ensure_ascii=True,
                )
                + "\n"
            )
            meta.write(
                json.dumps(
                    {
                        "conversation_id": conversation_id,
                        "source_file": str(raw_path),
                        "goemotions_id": example_id,
                        "labels": labels,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert GoEmotions TSV to Cortext JSONL.")
    parser.add_argument(
        "--source-dir",
        default="data/raw/goemotions",
        help="Directory containing GoEmotions TSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/goemotions",
        help="Output directory for converted JSONL files.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    emotions_path = source_dir / "emotions.txt"
    if not emotions_path.exists():
        raise SystemExit(f"Missing GoEmotions label file {emotions_path}")
    emotions = load_emotions(emotions_path)
    split_map = {"train": "train.tsv", "valid": "dev.tsv", "test": "test.tsv"}
    out_dir = Path(args.out_dir)
    for split, filename in split_map.items():
        raw_path = source_dir / filename
        if not raw_path.exists():
            raise SystemExit(f"Missing GoEmotions file {raw_path}")
        count = convert_split(raw_path, emotions, out_dir, split)
        print(f"[OK] {split}: wrote {count} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
