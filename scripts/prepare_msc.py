#!/usr/bin/env python3
"""
Convert Multi-Session Chat exports into the existing Cortext JSONL shape:

  [conversation_id, {"content": [{"agent": ..., "message": ...}, ...], ...}]

Extra study metadata is written to a sidecar JSONL file and is not required by
the runtime.
"""

from __future__ import annotations

import argparse
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


def iter_json_records(path: Path):
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() not in {".json", ".jsonl"}:
                continue
            yield from iter_json_records(child)
        return
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield path, json.loads(line)
        return
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for row in payload:
                yield path, row
        else:
            yield path, payload


def infer_split(source_path: Path, record: dict, fallback: str) -> str:
    split = normalize_text(first_non_empty(record, ["split", "subset", "partition"]))
    if split:
        split = split.lower()
    else:
        name = source_path.name.lower()
        if "train" in name:
            split = "train"
        elif "valid" in name or "dev" in name:
            split = "valid"
        elif "test" in name:
            split = "test"
        else:
            split = fallback
    if split == "dev":
        return "valid"
    return split


def extract_turns(record: dict) -> list[dict[str, str]]:
    raw_turns = first_non_empty(
        record,
        [
            "content",
            "dialog",
            "dialogue",
            "conversation",
            "messages",
            "utterances",
            "turns",
        ],
    )
    if not isinstance(raw_turns, list):
        return []
    content: list[dict[str, str]] = []
    for idx, turn in enumerate(raw_turns):
        if isinstance(turn, str):
            message = normalize_text(turn)
            if message:
                agent = "agent_1" if (idx % 2 == 0) else "agent_2"
                content.append({"agent": agent, "message": message})
            continue
        if not isinstance(turn, dict):
            continue
        message = normalize_text(
            first_non_empty(turn, ["message", "text", "utterance", "content", "value"])
        )
        if not message:
            continue
        agent = normalize_text(
            first_non_empty(turn, ["agent", "speaker", "role", "participant", "author"])
        )
        if not agent:
            agent = "agent_1" if (idx % 2 == 0) else "agent_2"
        content.append({"agent": agent, "message": message})
    return content


def build_metadata(record: dict, conversation_id: str, source_path: Path) -> dict:
    return {
        "conversation_id": conversation_id,
        "source_file": str(source_path),
        "session_index": first_non_empty(record, ["session_index", "session_id"]),
        "persona_summary": first_non_empty(record, ["persona_summary", "personas", "profile"]),
        "dialog_history_source": first_non_empty(
            record, ["dialog_history_source", "history_source", "source"]
        ),
        "speaker_map": first_non_empty(record, ["speaker_map", "participants"]),
    }


def convert(source: Path, out_dir: Path, split: str, limit: int | None) -> dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}
    writers = {}
    sidecars = {}
    try:
        for source_path, record in iter_json_records(source):
            if not isinstance(record, dict):
                continue
            resolved_split = infer_split(source_path, record, split if split != "all" else "valid")
            if split != "all" and resolved_split != split:
                continue
            if resolved_split not in counts:
                continue
            if limit is not None and counts[resolved_split] >= limit:
                continue
            conversation_id = normalize_text(
                first_non_empty(
                    record,
                    ["conversation_id", "conv_id", "id", "dialog_id", "session_key"],
                )
            )
            if not conversation_id:
                conversation_id = f"msc_{resolved_split}_{counts[resolved_split]}"
            content = extract_turns(record)
            if not content:
                continue
            payload = {"content": content}
            meta = build_metadata(record, conversation_id, source_path)
            if resolved_split not in writers:
                out_dir.mkdir(parents=True, exist_ok=True)
                writers[resolved_split] = (out_dir / f"{resolved_split}.jsonl").open(
                    "w", encoding="utf-8"
                )
                sidecars[resolved_split] = (
                    out_dir / f"{resolved_split}.metadata.jsonl"
                ).open("w", encoding="utf-8")
            writers[resolved_split].write(
                json.dumps([conversation_id, payload], ensure_ascii=True) + "\n"
            )
            sidecars[resolved_split].write(json.dumps(meta, ensure_ascii=True) + "\n")
            counts[resolved_split] += 1
    finally:
        for handle in writers.values():
            handle.close()
        for handle in sidecars.values():
            handle.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert MSC exports to Cortext JSONL + metadata sidecars."
    )
    parser.add_argument(
        "--source",
        default="data/raw/msc",
        help="Raw MSC export path (.json, .jsonl, or directory tree).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/msc",
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
            f"Missing MSC source at {source}. Export the raw dataset there or pass --source."
        )

    counts = convert(source, Path(args.out_dir), args.split, args.limit)
    for split_name, count in counts.items():
        if args.split != "all" and split_name != args.split:
            continue
        print(f"[OK] {split_name}: wrote {count} conversations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
