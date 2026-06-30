#!/usr/bin/env python3
"""Materialize Meta MSC as a deterministic chat-replay transcript.

The replay benchmark consumes timestamped text transcripts. This adapter keeps
MSC public-data handling outside the C++ runner: it fetches or reads the public
Hugging Face parquet shard, orders rows by dialogue/session, and writes one
transcript plus a manifest sidecar.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


HF_DATASET = "nayohan/multi_session_chat"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_DATASET}"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"
SPLIT_ALIASES = {
    "dev": "validation",
    "valid": "validation",
    "val": "validation",
}


def normalize_split(split: str) -> str:
    return SPLIT_ALIASES.get(split, split)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_hf_parquet(split: str) -> tuple[str, str]:
    with urllib.request.urlopen(HF_API_URL, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    prefix = f"data/{split}-"
    for sibling in payload.get("siblings", []):
        name = str(sibling.get("rfilename", ""))
        if name.startswith(prefix) and name.endswith(".parquet"):
            return name, f"{HF_RESOLVE_BASE}/{name}"
    raise RuntimeError(f"Could not find parquet shard for split {split!r}")


def download_if_needed(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=300) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(path)


def clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return list(value) if isinstance(value, tuple) else [value]


def header_for_speaker(timestamp: dt.datetime, speaker: str) -> str:
    stamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    normalized = clean_text(speaker).lower()
    if normalized.endswith("2") or normalized in {"speaker 2", "speaker2"}:
        return f"{stamp} from Contact"
    return f"{stamp} to Contact"


def write_turn(handle, timestamp: dt.datetime, speaker: str, text: str) -> None:
    handle.write("----------------------------------------------------\n")
    handle.write(header_for_speaker(timestamp, speaker) + "\n")
    handle.write(clean_text(text) + "\n")


def materialize(args: argparse.Namespace) -> dict:
    split = normalize_split(args.split)
    source_path = args.source
    source_url = ""
    hf_filename = ""
    if source_path is None:
        hf_filename, source_url = resolve_hf_parquet(split)
        source_path = args.cache_dir / hf_filename
        download_if_needed(source_path, source_url)
    df = pd.read_parquet(source_path)
    df = df.sort_values(["dialoug_id", "session_id"], kind="stable")
    if args.max_dialogs > 0:
        selected_dialogs = list(dict.fromkeys(df["dialoug_id"].tolist()))[: args.max_dialogs]
        df = df[df["dialoug_id"].isin(selected_dialogs)]
    if args.max_sessions > 0:
        df = df.head(args.max_sessions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = args.out_dir / f"msc_{split}_chat_replay.txt"
    metadata_path = args.out_dir / f"msc_{split}.metadata.jsonl"
    start = dt.datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    if start.tzinfo is not None:
        start = start.astimezone(dt.timezone.utc).replace(tzinfo=None)

    rows = 0
    turns = 0
    dialogs: set[int] = set()
    sessions_by_dialog: dict[int, int] = {}
    current_time = start
    previous_dialog: int | None = None
    previous_session: int | None = None

    with transcript_path.open("w", encoding="utf-8") as transcript, metadata_path.open(
        "w", encoding="utf-8"
    ) as metadata:
        for row in df.to_dict("records"):
            dialog_id = int(row.get("dialoug_id", rows))
            session_id = int(row.get("session_id", 0))
            if previous_dialog is None:
                pass
            elif dialog_id != previous_dialog:
                current_time += dt.timedelta(minutes=args.conversation_gap_minutes)
            elif previous_session is not None and session_id != previous_session:
                current_time += dt.timedelta(minutes=args.session_gap_minutes)

            dialogue = as_list(row.get("dialogue"))
            speakers = as_list(row.get("speaker"))
            if len(speakers) < len(dialogue):
                speakers.extend(
                    "Speaker 1" if i % 2 == 0 else "Speaker 2"
                    for i in range(len(speakers), len(dialogue))
                )
            for speaker, text in zip(speakers, dialogue):
                text = clean_text(text)
                if not text:
                    continue
                write_turn(transcript, current_time, str(speaker), text)
                turns += 1
                current_time += dt.timedelta(seconds=args.turn_gap_seconds)

            meta = {
                "dataset": row.get("dataset"),
                "dialog_id": dialog_id,
                "session_id": session_id,
                "persona1": as_list(row.get("persona1")),
                "persona2": as_list(row.get("persona2")),
            }
            metadata.write(json.dumps(meta, ensure_ascii=True, separators=(",", ":")) + "\n")
            dialogs.add(dialog_id)
            sessions_by_dialog[dialog_id] = sessions_by_dialog.get(dialog_id, 0) + 1
            previous_dialog = dialog_id
            previous_session = session_id
            rows += 1

    manifest = {
        "schema": "msc_chat_replay_materialization_v1",
        "public_benchmark": True,
        "dataset_name": "Meta Multi-Session Chat",
        "hf_dataset": HF_DATASET,
        "hf_api_url": HF_API_URL,
        "hf_source_url": source_url,
        "hf_filename": hf_filename,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "split": split,
        "rows": rows,
        "dialogs": len(dialogs),
        "turns": turns,
        "transcript_path": str(transcript_path),
        "metadata_path": str(metadata_path),
        "speaker_policy": "Speaker 1 -> User/to Contact; Speaker 2 -> Contact/from Contact",
        "timestamp_policy": {
            "start": args.start,
            "turn_gap_seconds": args.turn_gap_seconds,
            "session_gap_minutes": args.session_gap_minutes,
            "conversation_gap_minutes": args.conversation_gap_minutes,
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", help="MSC split to materialize.")
    parser.add_argument("--source", type=Path, help="Optional local parquet shard.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/msc_hf"))
    parser.add_argument("--out-dir", type=Path, default=Path("build/msc_chat_replay"))
    parser.add_argument("--max-dialogs", type=int, default=0)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--start", default="2020-01-01T00:00:00Z")
    parser.add_argument("--turn-gap-seconds", type=int, default=60)
    parser.add_argument("--session-gap-minutes", type=int, default=1440)
    parser.add_argument("--conversation-gap-minutes", type=int, default=10080)
    args = parser.parse_args()
    manifest = materialize(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
