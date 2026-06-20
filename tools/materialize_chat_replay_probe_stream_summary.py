#!/usr/bin/env python3
"""Build a summary-shaped artifact from streamed chat-replay probe rows.

This is a fail-fast helper for private local runs. It does not replace the
final release summary; it lets the existing judge consume native probe rows
before the full benchmark replay finishes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime


def load_probe_rows(path: pathlib.Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSONL row") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no}: probe row must be an object")
            rows.append(row)
            if limit >= 0 and len(rows) >= limit:
                break
    return rows


def infer_replay_timezone(
    explicit_timezone: str, probe_stream: pathlib.Path, db_path: pathlib.Path
) -> str:
    timezone = explicit_timezone.strip()
    if timezone:
        return timezone

    candidates = [
        db_path.parent / "progress.log",
        probe_stream.parent / "progress.log",
    ]
    seen: set[pathlib.Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        for line in path.read_text(errors="replace").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "replay_timezone":
                timezone = value.strip()
                if timezone:
                    return timezone
    return "process_default"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-stream", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--db", type=pathlib.Path, required=True)
    parser.add_argument("--processed-text-messages", type=int, required=True)
    parser.add_argument("--media-attempted", type=int, required=True)
    parser.add_argument("--timeline-skip-messages", type=int, default=0)
    parser.add_argument("--timeline-max-messages", type=int, default=-1)
    parser.add_argument("--timeline-media-limit", type=int, default=-1)
    parser.add_argument("--media-processed", type=int, default=0)
    parser.add_argument("--audio-processed", type=int, default=0)
    parser.add_argument("--image-processed", type=int, default=0)
    parser.add_argument("--video-processed", type=int, default=0)
    parser.add_argument("--media-failures", type=int, default=0)
    parser.add_argument("--probe-limit", type=int, default=-1)
    parser.add_argument("--warmup-events", type=int, default=0)
    parser.add_argument("--probe-stride", type=int, default=0)
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--active-history-token-budget", type=int, default=8000)
    parser.add_argument("--focus", type=float, default=0.5)
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--stability", type=float, default=0.5)
    parser.add_argument("--replay-timezone", default="")
    parser.add_argument("--daily-consolidation", action="store_true")
    args = parser.parse_args()
    if args.timeline_skip_messages < 0:
        raise RuntimeError("--timeline-skip-messages must be non-negative")

    probes = load_probe_rows(args.probe_stream, args.probe_limit)
    if not probes:
        raise RuntimeError(f"probe stream has no rows: {args.probe_stream}")
    replay_timezone = infer_replay_timezone(
        args.replay_timezone, args.probe_stream, args.db
    )

    out = {
        "schema": "chat_replay_probe_stream_partial_summary_v1",
        "partial_probe_stream_summary": True,
        "release_gate_use": "none_non_release_early_warning_only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir),
        "db_path": str(args.db),
        "replay_timezone": replay_timezone,
        "timeline_skip_messages": args.timeline_skip_messages,
        "timeline_max_messages": args.timeline_max_messages,
        "timeline_media_limit": args.timeline_media_limit,
        "skipped_transcript_messages": args.timeline_skip_messages,
        "processed_text_messages": args.processed_text_messages,
        "media_attempted": args.media_attempted,
        "media_processed": args.media_processed,
        "media_failures": args.media_failures,
        "audio_processed": args.audio_processed,
        "image_processed": args.image_processed,
        "video_processed": args.video_processed,
        "warmup_events": args.warmup_events,
        "probe_stride": args.probe_stride,
        "probe_count": len(probes),
        "probe_stream_path": str(args.probe_stream),
        "probe_stream_policy": (
            "native probe rows appended as compact JSONL immediately after "
            "each probe is constructed"
        ),
        "probes": probes,
        "rag_top_k": args.rag_top_k,
        "normal_rag_retrieval": "raw_chat_vector",
        "normal_rag_baseline_modality": "text_only",
        "normal_rag_vector_candidate_k": args.rag_top_k,
        "normal_rag_context_token_policy": (
            "text rolling chat after compaction plus unique text vector RAG "
            "hits outside the active rolling window"
        ),
        "normal_rag_compaction_summary_policy": "deterministic_extractive_prior_chat",
        "active_history_token_budget": args.active_history_token_budget,
        "knobs": {
            "focus": args.focus,
            "sensitivity": args.sensitivity,
            "stability": args.stability,
        },
        "daily_consolidation": args.daily_consolidation,
        "source_id_policy": (
            "User and Contact are opaque conversation provenance source IDs; "
            "media is not encoded into source_id"
        ),
        "timeline_policy": (
            "partial early-warning summary materialized from native live-run "
            "probe stream rows; final release claims must use the complete "
            "benchmark summary"
        ),
        "privacy_note": (
            "Private early-warning artifact; probe rows may include private "
            "conversation excerpts and source-backed memory content."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
