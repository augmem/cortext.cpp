#!/usr/bin/env python3
"""
Convert Taskmaster exports into the existing Cortext JSONL conversation format.
Study-only metadata is preserved in sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def first_non_empty(record: dict, keys: list[str]) -> object | None:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def iter_records(path: Path):
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() not in {".json", ".jsonl"}:
                continue
            yield from iter_records(child)
        return
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield path, json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for row in payload:
            yield path, row
    else:
        yield path, payload


def deterministic_split(conversation_id: str) -> str:
    digest = hashlib.blake2b(conversation_id.encode("utf-8"), digest_size=4).digest()
    bucket = int.from_bytes(digest, "little") % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "valid"
    return "test"


def infer_split(source_path: Path, record: dict, fallback: str, conversation_id: str) -> str:
    split = normalize_text(first_non_empty(record, ["split", "subset", "partition"])).lower()
    if not split:
        name = source_path.name.lower()
        if "train" in name:
            split = "train"
        elif "valid" in name or "dev" in name:
            split = "valid"
        elif "test" in name:
            split = "test"
        else:
            split = deterministic_split(conversation_id) if fallback == "all" else fallback
    return "valid" if split == "dev" else split


def extract_turns(record: dict) -> list[dict[str, str]]:
    raw_turns = first_non_empty(record, ["utterances", "conversation", "dialogue", "messages"])
    if not isinstance(raw_turns, list):
        return []
    content: list[dict[str, str]] = []
    for idx, turn in enumerate(raw_turns):
        if isinstance(turn, str):
            message = normalize_text(turn)
            if message:
                content.append(
                    {
                        "agent": "agent_1" if (idx % 2 == 0) else "agent_2",
                        "message": message,
                    }
                )
            continue
        if not isinstance(turn, dict):
            continue
        message = normalize_text(first_non_empty(turn, ["text", "message", "utterance"]))
        if not message:
            segments = turn.get("segments")
            if isinstance(segments, list):
                message = normalize_text(" ".join(str(seg.get("text", "")) for seg in segments))
        if not message:
            continue
        speaker = normalize_text(first_non_empty(turn, ["speaker", "agent", "role", "participant"]))
        if not speaker:
            speaker = "agent_1" if (idx % 2 == 0) else "agent_2"
        content.append({"agent": speaker, "message": message})
    return content


def build_metadata(record: dict, conversation_id: str, source_path: Path) -> dict:
    return {
        "conversation_id": conversation_id,
        "source_file": str(source_path),
        "task_domain": first_non_empty(record, ["task_domain", "domain", "instruction_id"]),
        "goal_slots": first_non_empty(record, ["goal_slots", "frames", "apis"]),
        "api_call_markers": first_non_empty(record, ["api_call_markers", "apis"]),
    }


def convert(source: Path, out_dir: Path, split: str, limit: int | None) -> dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}
    writers = {}
    sidecars = {}
    try:
        for source_path, record in iter_records(source):
            if not isinstance(record, dict):
                continue
            conversation_id = normalize_text(
                first_non_empty(record, ["conversation_id", "id", "conv_id", "dialogue_id"])
            )
            if not conversation_id:
                conversation_id = f"taskmaster_{source_path.stem}_{sum(counts.values())}"
            resolved_split = infer_split(source_path, record, split, conversation_id)
            if split != "all" and resolved_split != split:
                continue
            if resolved_split not in counts:
                continue
            if limit is not None and counts[resolved_split] >= limit:
                continue
            content = extract_turns(record)
            if not content:
                continue
            if resolved_split not in writers:
                out_dir.mkdir(parents=True, exist_ok=True)
                writers[resolved_split] = (out_dir / f"{resolved_split}.jsonl").open(
                    "w", encoding="utf-8"
                )
                sidecars[resolved_split] = (
                    out_dir / f"{resolved_split}.metadata.jsonl"
                ).open("w", encoding="utf-8")
            writers[resolved_split].write(
                json.dumps([conversation_id, {"content": content}], ensure_ascii=True) + "\n"
            )
            sidecars[resolved_split].write(
                json.dumps(build_metadata(record, conversation_id, source_path), ensure_ascii=True)
                + "\n"
            )
            counts[resolved_split] += 1
    finally:
        for handle in writers.values():
            handle.close()
        for handle in sidecars.values():
            handle.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Taskmaster exports to Cortext JSONL + metadata sidecars."
    )
    parser.add_argument(
        "--source",
        default="data/raw/taskmaster",
        help="Raw Taskmaster export path (.json, .jsonl, or directory tree).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/taskmaster",
        help="Output directory for converted JSONL.",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "valid", "test", "all"],
        help="Split to emit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap per emitted split.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(
            f"Missing Taskmaster source at {source}. Export the raw dataset there or pass --source."
        )

    counts = convert(source, Path(args.out_dir), args.split, args.limit)
    for split_name, count in counts.items():
        if args.split != "all" and split_name != args.split:
            continue
        print(f"[OK] {split_name}: wrote {count} conversations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
