#!/usr/bin/env python3
"""
Convert LongMemEval-style exports into Cortext JSONL plus an answer-key sidecar.

The conversation JSONL remains compatible with the existing topical-chat runner.
The answer key is kept external for study-time scoring.
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


def infer_split(source_path: Path, record: dict, fallback: str) -> str:
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
            split = fallback
    return "valid" if split == "dev" else split


def extract_turns(record: dict) -> list[dict[str, str]]:
    raw_turns = first_non_empty(
        record,
        ["content", "conversation", "dialogue", "messages", "utterances", "turns", "context"],
    )
    content: list[dict[str, str]] = []
    if isinstance(raw_turns, str):
        lines = [line.strip() for line in raw_turns.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            if ":" in line:
                speaker, message = line.split(":", 1)
                agent = normalize_text(speaker) or ("agent_1" if idx % 2 == 0 else "agent_2")
                text = normalize_text(message)
            else:
                agent = "agent_1" if (idx % 2 == 0) else "agent_2"
                text = normalize_text(line)
            if text:
                content.append({"agent": agent, "message": text})
        return content
    if not isinstance(raw_turns, list):
        return []
    for idx, turn in enumerate(raw_turns):
        if isinstance(turn, str):
            text = normalize_text(turn)
            if text:
                content.append(
                    {
                        "agent": "agent_1" if (idx % 2 == 0) else "agent_2",
                        "message": text,
                    }
                )
            continue
        if not isinstance(turn, dict):
            continue
        text = normalize_text(first_non_empty(turn, ["message", "text", "utterance", "content"]))
        if not text:
            continue
        agent = normalize_text(first_non_empty(turn, ["agent", "speaker", "role", "participant"]))
        if not agent:
            agent = "agent_1" if (idx % 2 == 0) else "agent_2"
        content.append({"agent": agent, "message": text})
    return content


def answer_list(record: dict) -> list[str]:
    raw = first_non_empty(record, ["answers", "answer", "gold", "target"])
    if raw is None:
        return []
    if isinstance(raw, list):
        return [normalize_text(item) for item in raw if normalize_text(item)]
    text = normalize_text(raw)
    return [text] if text else []


def build_answer_key(record: dict, conversation_id: str) -> dict:
    question = normalize_text(first_non_empty(record, ["question", "query", "prompt"]))
    return {
        "conversation_id": conversation_id,
        "query_id": normalize_text(first_non_empty(record, ["question_id", "query_id"])) or conversation_id,
        "question": question,
        "answers": answer_list(record),
        "question_type": normalize_text(first_non_empty(record, ["question_type", "category"])) or "unknown",
        "requires_temporal_reasoning": bool(
            first_non_empty(record, ["requires_temporal_reasoning", "temporal"])
        ),
        "requires_update_reasoning": bool(
            first_non_empty(record, ["requires_update_reasoning", "update_reasoning"])
        ),
        "requires_abstention": bool(
            first_non_empty(record, ["requires_abstention", "abstain"])
        ),
    }


def convert(
    source: Path,
    out_dir: Path,
    split: str,
    limit: int | None,
    include_question: bool,
) -> dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}
    writers = {}
    answer_writers = {}
    try:
        for source_path, record in iter_records(source):
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
                first_non_empty(record, ["conversation_id", "id", "dialogue_id", "question_id"])
            )
            if not conversation_id:
                conversation_id = f"longmemeval_{resolved_split}_{counts[resolved_split]}"
            content = extract_turns(record)
            answer_key = build_answer_key(record, conversation_id)
            if include_question and answer_key["question"]:
                content.append({"agent": "agent_1", "message": answer_key["question"]})
            if not content:
                continue
            if resolved_split not in writers:
                out_dir.mkdir(parents=True, exist_ok=True)
                writers[resolved_split] = (out_dir / f"{resolved_split}.jsonl").open(
                    "w", encoding="utf-8"
                )
                answer_writers[resolved_split] = (
                    out_dir / f"{resolved_split}.answer_key.jsonl"
                ).open("w", encoding="utf-8")
            writers[resolved_split].write(
                json.dumps([conversation_id, {"content": content}], ensure_ascii=True) + "\n"
            )
            answer_writers[resolved_split].write(
                json.dumps(answer_key, ensure_ascii=True) + "\n"
            )
            counts[resolved_split] += 1
    finally:
        for handle in writers.values():
            handle.close()
        for handle in answer_writers.values():
            handle.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert LongMemEval exports to Cortext JSONL + answer key sidecars."
    )
    parser.add_argument(
        "--source",
        default="data/raw/longmemeval",
        help="Raw LongMemEval export path (.json, .jsonl, or directory tree).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/longmemeval",
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
    parser.add_argument(
        "--no-include-question",
        action="store_true",
        help="Do not append the evaluation question as the final user turn.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(
            f"Missing LongMemEval source at {source}. Export the raw dataset there or pass --source."
        )

    counts = convert(
        source,
        Path(args.out_dir),
        args.split,
        args.limit,
        include_question=not args.no_include_question,
    )
    for split_name, count in counts.items():
        if args.split != "all" and split_name != args.split:
            continue
        print(f"[OK] {split_name}: wrote {count} conversations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
