#!/usr/bin/env python3
"""
Convert PersonaChat raw text files into Cortext JSONL plus metadata sidecars.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def parse_raw_dialogues(path: Path) -> list[tuple[str, dict, dict]]:
    conversations: list[tuple[str, dict, dict]] = []
    conversation_index = 0
    personas: list[str] = []
    content: list[dict[str, str]] = []

    def flush() -> None:
        nonlocal conversation_index, personas, content
        if not content:
            personas = []
            content = []
            return
        conversation_id = f"{path.stem}_{conversation_index}"
        conversations.append(
            (
                conversation_id,
                {"content": content},
                {"conversation_id": conversation_id, "personas": personas, "source_file": str(path)},
            )
        )
        conversation_index += 1
        personas = []
        content = []

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            number, _, remainder = line.partition(" ")
            try:
                turn_index = int(number)
            except ValueError:
                continue
            if turn_index == 1 and content:
                flush()
            if remainder.startswith("your persona:"):
                personas.append(normalize_text(remainder[len("your persona:") :]))
                continue
            fields = remainder.split("\t")
            utterances = [normalize_text(field) for field in fields[:2] if normalize_text(field)]
            if utterances:
                content.append({"agent": "agent_1", "message": utterances[0]})
            if len(utterances) > 1:
                content.append({"agent": "agent_2", "message": utterances[1]})
    flush()
    return conversations


def convert_split(raw_path: Path, out_dir: Path, split_name: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{split_name}.jsonl"
    meta_path = out_dir / f"{split_name}.metadata.jsonl"
    count = 0
    with jsonl_path.open("w", encoding="utf-8") as out, meta_path.open("w", encoding="utf-8") as meta:
        for conversation_id, payload, metadata in parse_raw_dialogues(raw_path):
            out.write(json.dumps([conversation_id, payload], ensure_ascii=True) + "\n")
            meta.write(json.dumps(metadata, ensure_ascii=True) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PersonaChat raw text to Cortext JSONL.")
    parser.add_argument(
        "--source-dir",
        default="data/raw/personachat/personachat",
        help="Directory containing PersonaChat raw .txt files.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/personachat",
        help="Output directory for converted JSONL.",
    )
    parser.add_argument(
        "--variant",
        default="self_original",
        choices=["self_original", "self_revised", "both_original", "both_revised"],
        help="PersonaChat split variant to convert.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise SystemExit(f"Missing PersonaChat source directory at {source_dir}")

    split_map = {
        "train": source_dir / f"train_{args.variant}.txt",
        "valid": source_dir / f"valid_{args.variant}.txt",
        "test": source_dir / f"test_{args.variant}.txt",
    }
    for split_name, raw_path in split_map.items():
        if not raw_path.exists():
            raise SystemExit(f"Missing PersonaChat raw file {raw_path}")
        count = convert_split(raw_path, Path(args.out_dir), split_name)
        print(f"[OK] {split_name}: wrote {count} conversations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
