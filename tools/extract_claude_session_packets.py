#!/usr/bin/env python3
"""Extract content-private sentence/tool packets from Claude session JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def compact_json(value: Any) -> str:
    # Match JavaScript JSON.stringify used by the original replay transform:
    # preserve insertion order, omit formatting whitespace, retain Unicode.
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def javascript_slice(text: str, code_units: int) -> str:
    encoded = text.encode("utf-16-le", errors="surrogatepass")
    return encoded[: code_units * 2].decode("utf-16-le", errors="ignore")


def block_text(block: dict[str, Any]) -> str | None:
    kind = block.get("type")
    if kind == "text":
        value = block.get("text")
        return value if isinstance(value, str) else None
    if kind == "tool_use":
        return javascript_slice(compact_json(block.get("input")), 2000)
    if kind == "tool_result":
        content = block.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            if parts:
                return "\n".join(parts)
        return compact_json(content)
    return None


def message_texts(record: dict[str, Any]) -> Iterable[str]:
    if record.get("type") not in {"user", "assistant"}:
        return
    message = record.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block_text(block)
                if text:
                    yield text


def packets_from_text(text: str) -> Iterable[str]:
    for sentence in SENTENCE_BREAK.split(text):
        sentence = sentence.strip()
        while sentence:
            if len(sentence) <= 400:
                yield sentence
                break
            split = sentence.rfind(" ", 10, 401)
            if split < 10:
                split = 400
            yield sentence[:split]
            sentence = sentence[split:].lstrip()


def extract_records(paths: list[Path]) -> list[str]:
    records: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                text = "\n".join(message_texts(record)).strip()
                if text:
                    records.append(text)
    return records


def extract(paths: list[Path]) -> tuple[list[str], list[str]]:
    records = extract_records(paths)
    packets: list[str] = []
    for text in records:
        packets.extend(packets_from_text(text))
    return records, packets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--durable-out", type=Path)
    parser.add_argument("--max-packets", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records, packets = extract(args.inputs)
    if args.max_packets is not None:
        packets = packets[: args.max_packets]
    payload = ("\n".join(packets) + "\n").encode("utf-8")
    durable_payload = ("\n".join(compact_json(record) for record in records) + "\n").encode(
        "utf-8"
    )
    path_set = "\n".join(str(path.resolve()) for path in args.inputs).encode()
    summary = {
        "input_count": len(args.inputs),
        "path_set_sha256": hashlib.sha256(path_set).hexdigest(),
        "logical_record_count": len(records),
        "packet_count": len(packets),
        "corpus_sha256": hashlib.sha256(payload).hexdigest(),
        "corpus_bytes": len(payload),
        "durable_corpus_sha256": hashlib.sha256(durable_payload).hexdigest(),
        "durable_corpus_bytes": len(durable_payload),
    }
    if not args.dry_run:
        if args.out is None:
            raise ValueError("--out is required unless --dry-run is used")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(payload)
        if args.durable_out is not None:
            args.durable_out.parent.mkdir(parents=True, exist_ok=True)
            args.durable_out.write_bytes(durable_payload)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
