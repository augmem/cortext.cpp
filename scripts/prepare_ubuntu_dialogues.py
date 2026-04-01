#!/usr/bin/env python3
"""
Download and convert the Ubuntu Dialogue Corpus CSVs (HF mirror) to
Cortext JSONL format: [id, {"content":[{"agent":..., "message":...}, ...]}].

This keeps the raw, messy conversational text while removing __eou__/__eot__ markers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "huggingface_hub is required: pip install huggingface_hub"
    ) from exc


DEFAULT_REPO = "rojagtap/ubuntu_dialogs_corpus"
SPLIT_TO_FILE = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}


def normalize_delimiters(text: str) -> list[str]:
    text = text.replace("__eot__", "__eou__")
    parts = [p.strip() for p in text.split("__eou__")]
    return [p for p in parts if p]


def pick_column(row: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def parse_label(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def row_to_conversation(row: dict[str, str]) -> list[dict[str, str]] | None:
    context = pick_column(row, ["Context", "context"])
    if not context:
        return None
    response = pick_column(
        row,
        ["Utterance", "utterance", "Response", "response", "Ground Truth Utterance"],
    )
    label = parse_label(pick_column(row, ["Label", "label"]))
    if label is not None and label < 0.5:
        return None
    if not response:
        return None

    turns = normalize_delimiters(context) + normalize_delimiters(response)
    if not turns:
        return None

    content: list[dict[str, str]] = []
    for i, msg in enumerate(turns):
        agent = "agent_1" if (i % 2 == 0) else "agent_2"
        content.append({"agent": agent, "message": msg})
    return content


def convert_csv(csv_path: Path, out_path: Path, max_conversations: int | None) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f_in, out_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        for row_idx, row in enumerate(reader):
            content = row_to_conversation(row)
            if not content:
                continue
            conv_id = f"ubuntu_{out_path.stem}_{row_idx}"
            record = [conv_id, {"content": content}]
            f_out.write(json.dumps(record, ensure_ascii=True) + "\n")
            written += 1
            if max_conversations is not None and written >= max_conversations:
                break
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and convert Ubuntu Dialogue Corpus CSVs to Cortext JSONL."
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="HF dataset repo id (default: rojagtap/ubuntu_dialogs_corpus).",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=sorted(SPLIT_TO_FILE.keys()),
        help="Dataset split to convert.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/ubuntu_dialogue",
        help="Output directory for converted JSONL.",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    csv_name = SPLIT_TO_FILE[args.split]
    csv_path = Path(
        hf_hub_download(repo_id=args.repo, repo_type="dataset", filename=csv_name)
    )
    out_path = out_dir / f"{args.split}.jsonl"
    written = convert_csv(csv_path, out_path, args.max_conversations)
    print(f"Wrote {written} conversations to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
