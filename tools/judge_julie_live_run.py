#!/usr/bin/env python3
"""Judge a Julie mixed-media live-run artifact with a local judge endpoint.

This script is intentionally an eval adapter only. It does not feed transcripts
back into Cortext. Text from the Julie export is used locally inside the judge
prompt, and output artifacts must be treated as private because local model
reasons can still contain conversation-specific details.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import pathlib
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from generate_julie_raw_speech_manifest import parse_messages, parse_timestamp


DEFAULT_NEMOTRON_MODEL = "nemotron-3-nano-omni-30b-a3b-8bit"
DEFAULT_OLLAMA_MODEL = "gemma4:12b-it-qat"
DEFAULT_MODEL = DEFAULT_NEMOTRON_MODEL
DEFAULT_NEMOTRON_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
JUDGE_CALL_ATTEMPTS = 3
LOCAL_JUDGE_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
SYSTEMS = ["cortext_native", "traditional_chat_rag", "full_history_upper_bound"]
FIELDS = [
    "relevance",
    "sufficiency",
    "noise",
    "temporal_correctness",
    "source_grounding",
    "modality_grounding",
]
QUALITY_COMPOSITE_WEIGHTS = {
    "relevance": 1.0,
    "sufficiency": 1.0,
    "noise": -0.25,
}
QUALITY_COMPOSITE_DEFINITION = "relevance + sufficiency - 0.25*noise"
PACKET_ALIASES = ["A", "B", "C"]
GABE_SOURCE_ID = "Gabe"
JULIE_SOURCE_ID = "Julie"
COMPACTED_HISTORY_SOURCE_ID = "Gabe-Julie-summary"
BLIND_FORBIDDEN_TERMS = {
    "cortext",
    "cortext_native",
    "traditional_chat_rag",
    "chat+rag",
    "normal-rag",
    "normal_rag",
    "full_history",
    "full history",
    "upper_bound",
    "upper bound",
}
SYSTEM_FAILURE_REASONS = {
    "missing_source_link",
    "temporal_drift",
    "insufficient_context",
    "unrelated_retrieval",
    "modality_blindness",
    "rag_context_advantage",
    "full_history_upper_bound_advantage",
    "cortext_wins",
    "tie_or_unclear",
}
BLIND_FAILURE_REASONS = {
    "missing_source_link",
    "temporal_drift",
    "insufficient_context",
    "unrelated_retrieval",
    "modality_blindness",
    "winner_best_context",
    "tie_or_unclear",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".gif", ".tiff"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".3gp"}
AUDIO_EXTENSIONS = {".m4a", ".wav", ".mp3"}
PACKET_SURFACE = "structurally_normalized_event_evidence_v1"


class JudgeCallTimeout(TimeoutError):
    pass


class JudgeMalformedResponse(RuntimeError):
    pass


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class TimelineDoc:
    index: int
    timestamp: int
    source_id: str
    modality: str
    text: str
    source_blob: str = ""
    media_attempt: int = 0


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def estimate_content_tokens(content: str | list[dict]) -> int:
    if isinstance(content, str):
        return estimate_tokens(content)
    total = 0
    for part in content:
        part_type = part.get("type")
        if part_type == "text":
            total += estimate_tokens(str(part.get("text", "")))
        elif part_type in {"image_url", "input_audio"}:
            total += estimate_tokens(f"[attached {part_type} evidence]")
    return max(1, total)


def sum_doc_tokens(docs: list[TimelineDoc]) -> int:
    return sum(estimate_tokens(doc.text) for doc in docs)


def source_for_message(message: dict) -> str:
    return JULIE_SOURCE_ID if message["from_contact"] else GABE_SOURCE_ID


def media_kind(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "other"


def media_path(input_dir: pathlib.Path, doc: TimelineDoc) -> pathlib.Path | None:
    if not doc.source_blob:
        return None
    path = input_dir / doc.source_blob
    if path.exists():
        return path
    matches = sorted(input_dir.rglob(path.name), key=lambda item: str(item))
    for match in matches:
        if match.is_file():
            return match
    return None


def media_mime(path: pathlib.Path, fallback: str) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    return guessed or fallback


def data_url(path: pathlib.Path, fallback_mime: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_mime(path, fallback_mime)};base64,{encoded}"


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> dict:
    summary = {"n": len(values), "mean": mean(values), "ci95": [0.0, 0.0]}
    if not values:
        return summary
    if len(values) == 1 or samples <= 0:
        summary["ci95"] = [values[0], values[0]]
        return summary
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(samples):
        boot.append(mean([values[rng.randrange(len(values))] for _ in values]))
    summary["ci95"] = [percentile(boot, 0.025), percentile(boot, 0.975)]
    return summary


def packet_assignment(
    systems: list[str],
    blind: bool,
    base_seed: int,
    event_index: int,
    repetition: int,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    ordered = list(systems)
    if blind:
        rng = random.Random(stable_seed(base_seed, "packet-order", event_index, repetition))
        rng.shuffle(ordered)
        real_to_label = {system: PACKET_ALIASES[i] for i, system in enumerate(ordered)}
    else:
        real_to_label = {system: system for system in ordered}
    label_to_real = {label: system for system, label in real_to_label.items()}
    return ordered, real_to_label, label_to_real


def normalize_winner_key(value: object, valid_keys: set[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "tie_or_unclear"
    compact = raw.lower().replace("_", " ").strip()
    if compact in {"tie", "tie unclear", "tie or unclear", "unknown", "none"}:
        return "tie_or_unclear"
    for key in valid_keys:
        key_compact = key.lower().replace("_", " ")
        if compact in {key_compact, f"packet {key_compact}", f"system {key_compact}"}:
            return key
    upper = raw.upper()
    if upper in valid_keys:
        return upper
    return raw


def require_local_base_url(base_url: str, provider: str) -> str:
    if "://" not in base_url:
        base_url = f"http://{base_url}"
    base_url = base_url.rstrip("/")
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOCAL_JUDGE_HOSTS:
        raise RuntimeError(
            "Refusing non-local judge endpoint for private Julie metadata: "
            f"{base_url!r}. Start the local {provider} judge server and set a "
            "loopback URL."
        )
    return base_url


def local_judge_base_url() -> str:
    base_url = (
        os.environ.get("CORTEXT_JUDGE_BASE_URL")
        or os.environ.get("LOCAL_JUDGE_BASE_URL")
        or DEFAULT_NEMOTRON_BASE_URL
    )
    base_url = require_local_base_url(base_url, "Nemotron/MLX")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def local_ollama_base_url(cli_base_url: str | None) -> str:
    return require_local_base_url(
        cli_base_url
        or os.environ.get("CORTEXT_OLLAMA_BASE_URL")
        or os.environ.get("OLLAMA_HOST")
        or DEFAULT_OLLAMA_BASE_URL,
        "Ollama",
    )


def ollama_model_capabilities(base_url: str, model: str) -> dict:
    out = {
        "source": "ollama_api_show",
        "model": model,
        "available": False,
        "capabilities": [],
        "text": True,
        "image": False,
        "audio": False,
        "error": "",
    }
    try:
        body = json.dumps({"model": model}).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/api/show",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
        caps = [str(capability) for capability in payload.get("capabilities", []) or []]
        out["available"] = True
        out["capabilities"] = caps
        out["image"] = "vision" in caps or "image" in caps
        out["audio"] = "audio" in caps
        details = payload.get("details")
        if isinstance(details, dict):
            out["details"] = {
                key: details.get(key)
                for key in (
                    "family",
                    "parameter_size",
                    "quantization_level",
                )
            }
        model_info = payload.get("model_info")
        if isinstance(model_info, dict):
            context_length = (
                model_info.get("general.context_length")
                or model_info.get("gemma4.context_length")
            )
            if context_length is not None:
                out.setdefault("details", {})["context_length"] = context_length
        return out
    except Exception as exc:
        out["show_error"] = exc.__class__.__name__

    out["source"] = "ollama_api_tags"
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as response:
            payload = json.load(response)
    except Exception as exc:
        out["error"] = exc.__class__.__name__
        return out
    for item in payload.get("models", []):
        if item.get("name") != model and item.get("model") != model:
            continue
        caps = [str(capability) for capability in item.get("capabilities", []) or []]
        out["available"] = True
        out["capabilities"] = caps
        out["image"] = "vision" in caps or "image" in caps
        out["audio"] = "audio" in caps
        details = item.get("details")
        if isinstance(details, dict):
            out["details"] = {
                key: details.get(key)
                for key in (
                    "family",
                    "parameter_size",
                    "quantization_level",
                    "context_length",
                )
            }
        return out
    return out


def resolve_judge_model(provider: str, model: str | None) -> str:
    if model:
        return model
    if provider == "ollama":
        return DEFAULT_OLLAMA_MODEL
    return DEFAULT_NEMOTRON_MODEL


def require_nemotron_model(model: str) -> None:
    if "nemotron" not in model.lower():
        raise RuntimeError(
            f"Refusing non-Nemotron judge model for private Julie metadata: {model!r}. "
            "Start the local Nemotron/MLX judge server and pass --model nemotron..."
        )


def convert_image_for_judge(source: pathlib.Path, work_dir: pathlib.Path, stem: str) -> pathlib.Path | None:
    out = work_dir / f"{stem}.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        "scale=768:768:force_original_aspect_ratio=decrease,pad=768:768:(ow-iw)/2:(oh-ih)/2",
        "-frames:v",
        "1",
        str(out),
    ]
    if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


def convert_audio_for_judge(source: pathlib.Path, work_dir: pathlib.Path, stem: str) -> pathlib.Path | None:
    out = work_dir / f"{stem}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-t",
        "30",
        str(out),
    ]
    if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


def media_source_id(path: pathlib.Path, kind: str) -> str:
    name = path.name.lower()
    if kind == "audio" and ("_self" in name or " self " in name):
        return GABE_SOURCE_ID
    return JULIE_SOURCE_ID


def build_timeline(
    input_dir: pathlib.Path,
    skip_messages: int,
    max_messages: int,
    media_limit: int,
) -> list[TimelineDoc]:
    messages = parse_messages(input_dir / "Messages - Julie Willen.txt")
    if skip_messages > 0:
        messages = messages[skip_messages:]
    if max_messages >= 0:
        messages = messages[:max_messages]
    message_window_start = int(messages[0]["timestamp"]) if messages else 0
    message_window_end = int(messages[-1]["timestamp"]) if messages else 0

    media: list[tuple[pathlib.Path, int, str]] = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        kind = media_kind(path)
        if kind not in {"audio", "image", "video"}:
            continue
        timestamp = parse_timestamp(path.name)
        timestamp_value = int(timestamp or 0)
        if message_window_start and (
            timestamp_value < message_window_start or timestamp_value > message_window_end
        ):
            continue
        media.append((path, timestamp_value, kind))
    media.sort(key=lambda item: (item[1], str(item[0])))

    events: list[tuple[int, int, bool, int]] = []
    for i, message in enumerate(messages):
        events.append((int(message["timestamp"]), int(message["original_index"]) * 2, False, i))
    for i, item in enumerate(media):
        events.append((item[1], len(messages) * 2 + i, True, i))
    events.sort(key=lambda row: (row[0], row[2], row[1]))

    out: list[TimelineDoc] = []
    media_attempted = 0
    for _, _, is_media, idx in events:
        if not is_media:
            message = messages[idx]
            out.append(
                TimelineDoc(
                    index=len(out),
                    timestamp=int(message["timestamp"]),
                    source_id=source_for_message(message),
                    modality="text",
                    text=message["text"],
                )
            )
            continue

        if media_limit >= 0 and media_attempted >= media_limit:
            continue
        media_attempted += 1
        path, timestamp, kind = media[idx]
        modality = "image" if kind == "video" else kind
        try:
            source_blob = str(path.relative_to(input_dir))
        except ValueError:
            source_blob = path.name
        out.append(
            TimelineDoc(
                index=len(out),
                timestamp=int(timestamp),
                source_id=media_source_id(path, kind),
                modality=modality,
                text=f"[{kind} source blob: {path.name}]",
                source_blob=source_blob,
                media_attempt=media_attempted,
            )
        )
    return out


def docs_by_index(timeline: list[TimelineDoc]) -> dict[int, TimelineDoc]:
    return {doc.index: doc for doc in timeline}


def find_query_doc(timeline: list[TimelineDoc], probe: dict) -> TimelineDoc | None:
    event_index = int(probe["event_index"])
    if 0 <= event_index < len(timeline):
        return timeline[event_index]
    query = probe.get("query", {})
    for doc in timeline:
        if (
            doc.timestamp == int(query.get("timestamp", -1))
            and doc.source_id == query.get("source_id")
            and doc.modality == query.get("modality")
        ):
            return doc
    return None


def connect_db(path: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def build_media_memory_map(
    conn: sqlite3.Connection,
    timeline: list[TimelineDoc],
) -> dict[tuple[str, str, int], list[TimelineDoc]]:
    media_docs = [
        doc
        for doc in timeline
        if doc.media_attempt > 0 and doc.modality in {"audio", "image"}
    ]
    occurrences = conn.execute(
        """
        select modality, source_id, timestamp, min(signal_id) as first_signal_id
        from signals
        where modality in ('audio', 'image')
        group by modality, source_id, timestamp, hex(blob_id)
        order by first_signal_id
        """
    ).fetchall()
    out: dict[tuple[str, str, int], list[TimelineDoc]] = {}
    used_doc_indexes: set[int] = set()
    for occurrence in occurrences:
        modality = str(occurrence["modality"])
        source_id = str(occurrence["source_id"])
        timestamp = int(occurrence["timestamp"] or 0)
        match = None
        for doc in media_docs:
            if doc.index in used_doc_indexes:
                continue
            if (
                doc.modality == modality
                and doc.source_id == source_id
                and doc.timestamp == timestamp
            ):
                match = doc
                break
        if match is None:
            continue
        used_doc_indexes.add(match.index)
        out.setdefault((modality, source_id, timestamp), []).append(match)
    return out


def load_memory_rows(conn: sqlite3.Connection, memory_ids: list[int]) -> list[sqlite3.Row]:
    if not memory_ids:
        return []
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"""
        select memory_id, kind, source_id, modality, start_ts, end_ts, label,
               n_signals, strength, retrieved_count, used_count
        from memories
        where memory_id in ({placeholders})
        """,
        memory_ids,
    ).fetchall()
    by_id = {int(row["memory_id"]): row for row in rows}
    return [by_id[mid] for mid in memory_ids if mid in by_id]


def row_get(row, key: str, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        value = row[key]
    except Exception:
        return default
    return default if value is None else value


def frozen_memory_rows(probe: dict, field: str) -> list[dict]:
    rows = []
    for row in probe.get(field, []) or []:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        try:
            item["memory_id"] = int(item.get("memory_id", 0) or 0)
        except Exception:
            item["memory_id"] = 0
        try:
            item["start_ts"] = int(item.get("start_ts", item.get("timestamp", 0)) or 0)
        except Exception:
            item["start_ts"] = 0
        try:
            item["tokens"] = int(item.get("tokens", 0) or 0)
        except Exception:
            item["tokens"] = 0
        item.setdefault("kind", "probe_time_context_memory")
        item.setdefault("source_id", "")
        item.setdefault("modality", "")
        item.setdefault("content_text", "")
        rows.append(item)
    return rows


def unique_memory_rows(rows: list) -> list:
    out = []
    seen = set()
    for row in rows:
        memory_id = int(row_get(row, "memory_id", 0) or 0)
        key = memory_id if memory_id > 0 else id(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def map_memory_to_docs(
    row,
    timeline: list[TimelineDoc],
    media_memory_map: dict[tuple[str, str, int], list[TimelineDoc]],
) -> list[TimelineDoc]:
    modality = row_get(row, "modality", "")
    source_id = row_get(row, "source_id", "")
    start_ts = int(row_get(row, "start_ts", 0) or 0)
    if modality == "text":
        for doc in timeline:
            if doc.modality == "text" and doc.source_id == source_id and doc.timestamp == start_ts:
                return [doc]
        return []
    return media_memory_map.get((str(modality), str(source_id), start_ts), [])


def map_memory_to_doc(
    row,
    timeline: list[TimelineDoc],
    media_memory_map: dict[tuple[str, str, int], list[TimelineDoc]],
) -> TimelineDoc | None:
    docs = map_memory_to_docs(row, timeline, media_memory_map)
    return docs[0] if docs else None


def select_packet_docs(docs: list[TimelineDoc], limit: int) -> list[TimelineDoc]:
    if limit < 0 or len(docs) <= limit:
        return docs

    synthetic = [doc for doc in docs if doc.index < 0]
    remaining = max(0, limit - len(synthetic))
    real_docs = [doc for doc in docs if doc.index >= 0]
    selected_real = real_docs[-remaining:] if remaining > 0 else []
    selected = {id(doc) for doc in synthetic + selected_real}
    return [doc for doc in docs if id(doc) in selected]


def evidence_text_for_doc(doc: TimelineDoc) -> str:
    text = doc.text
    if doc.modality != "text":
        text = f"{text} source_blob=true transcript_not_available_to_packet=true"
    return text


def evidence_line(
    event_index: str | int,
    modality: str,
    source_id: str,
    timestamp: int,
    text: str,
) -> str:
    return (
        f"- event_index={event_index} modality={modality} "
        f"source_id={source_id} timestamp={timestamp}: {text}"
    )


def packet_from_docs(name: str, docs: list[TimelineDoc], limit: int) -> str:
    lines = [f"{name}:"]
    selected = select_packet_docs(docs, limit)
    for doc in selected:
        lines.append(
            evidence_line(
                doc.index,
                doc.modality,
                doc.source_id,
                doc.timestamp,
                evidence_text_for_doc(doc),
            )
        )
    if len(lines) == 1:
        lines.append("- <empty>")
    return "\n".join(lines)


def compaction_doc(probe: dict, query_doc: TimelineDoc) -> TimelineDoc | None:
    events = int(probe.get("normal_rag_compaction_events", 0) or 0)
    items = int(probe.get("normal_rag_compacted_history_items", 0) or 0)
    if events <= 0 or items <= 0:
        return None
    original_tokens = int(probe.get("normal_rag_compacted_original_tokens", 0) or 0)
    summary_tokens = int(probe.get("normal_rag_compacted_summary_tokens", 0) or 0)
    summary_text = str(probe.get("normal_rag_compacted_summary", "")).strip()
    if not summary_text:
        summary_text = (
            f"[compacted_history messages={items} original_tokens={original_tokens} "
            f"summary_tokens={summary_tokens} "
            'note="older rolling chat compressed before this turn"]'
        )
    return TimelineDoc(
        index=-1,
        timestamp=query_doc.timestamp,
        source_id=COMPACTED_HISTORY_SOURCE_ID,
        modality="text",
        text=summary_text,
    )


def compaction_summary_has_content(probe: dict) -> bool:
    text = str(probe.get("normal_rag_compacted_summary", "")).strip()
    if not text:
        return False
    if text.startswith("[compacted_history "):
        return False
    return "Deterministic extractive excerpts:" in text


def packet_from_memories(
    name: str,
    rows: list,
    timeline: list[TimelineDoc],
    media_memory_map: dict[tuple[str, str, int], list[TimelineDoc]],
    limit: int,
) -> str:
    lines = [f"{name}:"]
    selected = rows if limit < 0 else rows[:limit]
    for row in selected:
        docs = map_memory_to_docs(row, timeline, media_memory_map)
        snapshot_text = str(row_get(row, "content_text", "") or "").strip()
        if docs and docs[0].modality == "text":
            content = snapshot_text or docs[0].text
            event_index = docs[0].index
            modality = docs[0].modality
            source_id = docs[0].source_id
            timestamp = docs[0].timestamp
        elif docs:
            content = " ".join(
                (
                    f"[{doc.modality} source blob: {doc.source_blob or 'unknown'}] "
                    "source_blob=true transcript_not_available_to_packet=true"
                )
                for doc in docs
            )
            event_index = ",".join(str(doc.index) for doc in docs)
            modality = str(row_get(row, "modality", docs[0].modality) or docs[0].modality)
            source_id = str(row_get(row, "source_id", docs[0].source_id) or docs[0].source_id)
            timestamp = int(row_get(row, "start_ts", docs[0].timestamp) or docs[0].timestamp)
        elif snapshot_text:
            content = snapshot_text
            event_index = "snapshot"
            modality = str(row_get(row, "modality", "") or "")
            source_id = str(row_get(row, "source_id", "") or "")
            timestamp = int(row_get(row, "start_ts", 0) or 0)
        else:
            content = "[source-backed memory; content not text-hydrated for judge]"
            event_index = "unknown"
            modality = str(row_get(row, "modality", "") or "")
            source_id = str(row_get(row, "source_id", "") or "")
            timestamp = int(row_get(row, "start_ts", 0) or 0)
        lines.append(evidence_line(event_index, modality, source_id, timestamp, content))
    if len(lines) == 1:
        lines.append("- <empty>")
    return "\n".join(lines)


def unique_media_docs(docs: list[TimelineDoc]) -> list[TimelineDoc]:
    out: list[TimelineDoc] = []
    seen: set[tuple[int, str]] = set()
    for doc in docs:
        if doc.modality == "text" or not doc.source_blob:
            continue
        key = (doc.index, doc.source_blob)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def memory_docs(
    rows: list,
    timeline: list[TimelineDoc],
    media_memory_map: dict[tuple[str, str, int], list[TimelineDoc]],
) -> list[TimelineDoc]:
    out: list[TimelineDoc] = []
    for row in rows:
        out.extend(map_memory_to_docs(row, timeline, media_memory_map))
    return out


def memory_doc_tokens(
    rows: list,
    timeline: list[TimelineDoc],
    media_memory_map: dict[tuple[str, str, int], list[TimelineDoc]],
    limit: int,
) -> int:
    selected = rows if limit < 0 else rows[:limit]
    if any(row_get(row, "tokens") is not None for row in selected):
        return sum(int(row_get(row, "tokens", 0) or 0) for row in selected)
    docs = memory_docs(selected, timeline, media_memory_map)
    return sum_doc_tokens(docs)


def text_only_docs(docs: list[TimelineDoc]) -> list[TimelineDoc]:
    return [doc for doc in docs if doc.modality == "text"]


def prior_context_docs(docs: list[TimelineDoc], event_index: int) -> list[TimelineDoc]:
    return [doc for doc in docs if doc.index < 0 or doc.index < event_index]


def exclude_current_memory_rows(
    rows: list,
    timeline: list[TimelineDoc],
    media_memory_map: dict[tuple[str, str, int], list[TimelineDoc]],
    event_index: int,
) -> tuple[list[sqlite3.Row], list[int]]:
    kept: list[sqlite3.Row] = []
    excluded: list[int] = []
    for row in rows:
        docs = map_memory_to_docs(row, timeline, media_memory_map)
        if any(doc.index == event_index for doc in docs):
            excluded.append(int(row_get(row, "memory_id", 0) or 0))
            continue
        kept.append(row)
    return kept, excluded


def build_multimodal_content(
    prompt: str,
    input_dir: pathlib.Path,
    packet_media_docs: list[tuple[str, str, list[TimelineDoc]]],
    max_media_per_system: int,
    work_dir: pathlib.Path,
) -> tuple[list[dict], dict]:
    content: list[dict] = [{ "type": "text", "text": prompt }]
    stats = {system: { "image": 0, "audio": 0, "skipped": 0 } for system, _, _ in packet_media_docs}
    stats["enabled"] = max_media_per_system != 0
    if max_media_per_system == 0:
        return content, stats

    for system_name, packet_label, docs in packet_media_docs:
        selected = unique_media_docs(docs)
        if max_media_per_system > 0:
            selected = selected[:max_media_per_system]
        for doc in selected:
            path = media_path(input_dir, doc)
            if path is None:
                stats[system_name]["skipped"] += 1
                continue
            label = (
                f"Packet {packet_label} media evidence: event_index={doc.index} "
                f"modality={doc.modality} source_id={doc.source_id} "
                f"timestamp={doc.timestamp} source_blob={doc.source_blob}"
            )
            content.append({ "type": "text", "text": label })
            if doc.modality == "audio":
                wav = convert_audio_for_judge(path, work_dir, f"{system_name}_{doc.index}")
                if wav is None:
                    stats[system_name]["skipped"] += 1
                    continue
                encoded = base64.b64encode(wav.read_bytes()).decode("ascii")
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": { "data": encoded, "format": "wav" },
                    }
                )
                stats[system_name]["audio"] += 1
                continue

            image = convert_image_for_judge(path, work_dir, f"{system_name}_{doc.index}")
            if image is None:
                stats[system_name]["skipped"] += 1
                continue
            content.append(
                {
                    "type": "image_url",
                    "image_url": { "url": data_url(image, "image/jpeg") },
                }
            )
            stats[system_name]["image"] += 1
    return content, stats


def parse_judge_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError as exc:
                raise JudgeMalformedResponse(
                    "Judge returned malformed JSON object "
                    f"chars={len(content)} object_chars={end - start + 1}"
                ) from exc
        raise JudgeMalformedResponse(
            "Judge returned non-JSON content "
            f"chars={len(content)} has_open_brace={start >= 0} has_close_brace={end >= 0}"
        )


def prompt_text(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if part.get("type") == "text":
            parts.append(str(part.get("text", "")))
    return "\n\n".join(parts)


def judge_packet_keys(content: str | list[dict]) -> list[str]:
    text = prompt_text(content)
    match = re.search(r"top-level keys\s+(.+?),\s+winner\b", text)
    if not match:
        return PACKET_ALIASES
    keys: list[str] = []
    for part in match.group(1).split(","):
        key = part.strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key):
            keys.append(key)
    return keys or PACKET_ALIASES


def judge_score_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            **{field: {"type": "number"} for field in FIELDS},
            "reason": {"type": "string"},
        },
        "required": [*FIELDS, "reason"],
        "additionalProperties": True,
    }


def judge_json_schema(content: str | list[dict]) -> dict:
    keys = judge_packet_keys(content)
    reasons = sorted(BLIND_FAILURE_REASONS | SYSTEM_FAILURE_REASONS)
    return {
        "type": "object",
        "properties": {
            **{key: judge_score_schema() for key in keys},
            "winner": {"type": "string", "enum": [*keys, "tie_or_unclear"]},
            "failure_reason": {"type": "string", "enum": reasons},
            "winner_reason": {"type": "string"},
        },
        "required": [*keys, "winner", "failure_reason", "winner_reason"],
        "additionalProperties": True,
    }


def post_json_local(url: str, body: dict, timeout_s: int) -> dict:
    payload = json.dumps(body).encode("utf-8")
    cmd = [
        "curl",
        "--silent",
        "--show-error",
        "--max-time",
        str(max(1, timeout_s)),
        "--header",
        "Content-Type: application/json",
        "--request",
        "POST",
        "--data-binary",
        "@-",
        "--write-out",
        "\n%{http_code}",
        url,
    ]
    try:
        result = subprocess.run(
            cmd,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(1, timeout_s) + 5,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise JudgeCallTimeout("local judge curl subprocess exceeded timeout") from exc
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    if "\n" not in stdout:
        raise RuntimeError(f"Local judge returned malformed HTTP payload: {stdout[:500]!r}")
    body_text, status_text = stdout.rsplit("\n", 1)
    try:
        status = int(status_text.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Local judge returned malformed HTTP status {status_text!r}: {stderr[:500]}"
        ) from exc
    if result.returncode != 0 or status >= 400:
        raise RuntimeError(
            f"Local judge failed: curl={result.returncode} http={status}: "
            f"{body_text[:500]} {stderr[:500]}"
        )
    return json.loads(body_text)


def base64_from_data_url(url: str) -> str:
    if not url.startswith("data:") or "," not in url:
        raise RuntimeError("Expected judge image data URL")
    return url.split(",", 1)[1]


def ollama_user_message(content: str | list[dict]) -> dict:
    if isinstance(content, str):
        return {"role": "user", "content": content}

    text_parts: list[str] = []
    media_payloads: list[str] = []
    for part in content:
        part_type = part.get("type")
        if part_type == "text":
            text_parts.append(str(part.get("text", "")))
            continue
        if part_type == "image_url":
            image_url = part.get("image_url", {})
            media_payloads.append(base64_from_data_url(str(image_url.get("url", ""))))
            continue
        if part_type == "input_audio":
            input_audio = part.get("input_audio", {})
            media_payloads.append(str(input_audio.get("data", "")))
            continue

    message = {"role": "user", "content": "\n\n".join(text_parts)}
    if media_payloads:
        # Ollama exposes one binary multimodal field named "images"; Gemma 4
        # accepts both image and WAV payloads through it when the model has
        # vision/audio capabilities.
        message["images"] = media_payloads
    return message


def call_nemotron_judge(
    model: str,
    content: str | list[dict],
    max_tokens: int,
    timeout_s: int,
) -> dict:
    require_nemotron_model(model)
    base_url = local_judge_base_url()
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict multimodal retrieval-quality judge. Inspect "
                    "attached image/audio evidence when present. Return only valid JSON."
                ),
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": max_tokens,
    }
    payload = post_json_local(f"{base_url}/chat/completions", body, timeout_s)
    content = payload["choices"][0]["message"]["content"]
    return parse_judge_json(content)


def call_ollama_judge(
    model: str,
    base_url: str,
    content: str | list[dict],
    max_tokens: int,
    timeout_s: int,
    context_window_tokens: int,
    keep_alive: str,
) -> dict:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict multimodal retrieval-quality judge. Inspect "
                    "attached image/audio evidence when present. Return only valid JSON."
                ),
            },
            ollama_user_message(content),
        ],
        "stream": False,
        "keep_alive": keep_alive,
        "format": judge_json_schema(content),
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
            "num_ctx": context_window_tokens,
        },
    }
    payload = post_json_local(f"{base_url}/api/chat", body, timeout_s)
    content_text = payload.get("message", {}).get("content", "")
    return parse_judge_json(str(content_text))


def call_judge_once(
    provider: str,
    model: str,
    base_url: str | None,
    content: str | list[dict],
    max_tokens: int,
    timeout_s: int,
    context_window_tokens: int,
    ollama_keep_alive: str,
) -> dict:
    if provider == "ollama":
        if base_url is None:
            raise RuntimeError("Ollama judge base URL was not resolved")
        return call_ollama_judge(
            model,
            base_url,
            content,
            max_tokens,
            timeout_s,
            context_window_tokens,
            ollama_keep_alive,
        )
    return call_nemotron_judge(model, content, max_tokens, timeout_s)


def call_judge(
    provider: str,
    model: str,
    base_url: str | None,
    content: str | list[dict],
    max_tokens: int,
    timeout_s: int,
    context_window_tokens: int,
    ollama_keep_alive: str,
) -> dict:
    for attempt in range(1, JUDGE_CALL_ATTEMPTS + 1):
        try:
            return call_judge_once(
                provider,
                model,
                base_url,
                content,
                max_tokens,
                timeout_s,
                context_window_tokens,
                ollama_keep_alive,
            )
        except (JudgeMalformedResponse, JudgeCallTimeout) as exc:
            if attempt >= JUDGE_CALL_ATTEMPTS:
                raise RuntimeError(
                    "Local judge failed after retryable errors "
                    f"attempts={JUDGE_CALL_ATTEMPTS} last_error={type(exc).__name__}: {exc}"
                ) from exc
            print(
                "judge_retry "
                f"ts={utc_now_text()} "
                f"attempt={attempt + 1}/{JUDGE_CALL_ATTEMPTS} "
                f"reason={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    raise RuntimeError("unreachable judge retry state")


def score(scores: dict, system: str, field: str) -> float:
    system_scores = scores.get(system, {})
    if not isinstance(system_scores, dict):
        return 0.0
    try:
        return max(0.0, min(5.0, float(system_scores.get(field, 0.0))))
    except (TypeError, ValueError):
        return 0.0


def safe_reason(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = " ".join(value.split())
    # Keep the artifact private-safe: reasons can describe evidence type and
    # retrieval behavior, but should not contain copied message content.
    return value[:320]


def expected_failure_reason(winner: str) -> str | None:
    if winner == "cortext_native":
        return "cortext_wins"
    if winner == "traditional_chat_rag":
        return "rag_context_advantage"
    if winner == "full_history_upper_bound":
        return "full_history_upper_bound_advantage"
    return None


def system_reason(scores: dict, system: str) -> str:
    system_scores = scores.get(system, {})
    if not isinstance(system_scores, dict):
        return ""
    for key in ("reason", "score_reason", "rationale", "evidence_reason"):
        reason = safe_reason(system_scores.get(key))
        if reason:
            return reason
    return ""


def attached_media_count(media_stats: dict, system: str) -> int:
    system_stats = media_stats.get(system, {})
    if not isinstance(system_stats, dict):
        return 0
    return int(system_stats.get("image", 0) or 0) + int(system_stats.get("audio", 0) or 0)


def normalize_scores(scores: dict) -> dict:
    if not isinstance(scores, dict):
        return {}
    normalized = dict(scores)
    for wrapper in ("scores", "systems", "results"):
        wrapped = scores.get(wrapper)
        if isinstance(wrapped, dict):
            for key, value in wrapped.items():
                normalized.setdefault(key, value)
    return normalized


def confidence_intervals(
    judged: list[dict],
    systems: list[str],
    fields: list[str],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in judged:
        groups[int(row["event_index"])].append(row)

    out = {
        "unit": "probe",
        "bootstrap_samples": bootstrap_samples,
        "probe_count": len(groups),
        "systems": {},
        "tokens": {},
    }
    for system in systems:
        system_out = {}
        win_values: list[float] = []
        field_values = {field: [] for field in fields}
        for rows in groups.values():
            win_values.append(sum(1.0 for row in rows if row["winner"] == system) / len(rows))
            for field in fields:
                field_values[field].append(
                    mean([float(row["systems"][system][field]) for row in rows])
                )
        system_out["win_rate"] = bootstrap_mean_ci(
            win_values,
            bootstrap_samples,
            stable_seed(seed, "ci", system, "win_rate"),
        )
        for field, values in field_values.items():
            system_out[field] = bootstrap_mean_ci(
                values,
                bootstrap_samples,
                stable_seed(seed, "ci", system, field),
            )
        out["systems"][system] = system_out

    token_savings_values: list[float] = []
    for rows in groups.values():
        cortext_tokens = mean([float(row["cortext_context_tokens"]) for row in rows])
        rag_tokens = mean([float(row["traditional_chat_rag_tokens"]) for row in rows])
        if rag_tokens > 0:
            token_savings_values.append(1.0 - (cortext_tokens / rag_tokens))
    out["tokens"]["cortext_savings_vs_traditional_chat_rag"] = bootstrap_mean_ci(
        token_savings_values,
        bootstrap_samples,
        stable_seed(seed, "ci", "tokens", "cortext_savings_vs_traditional_chat_rag"),
    )
    return out


def expected_judgment_count(
    summary: dict,
    timeline: list[TimelineDoc],
    judge_limit: int,
    judge_start_index: int,
    judge_repetitions: int,
) -> int:
    query_probe_index = 0
    probe_count = 0
    for probe in summary.get("probes", []):
        if not find_query_doc(timeline, probe):
            continue
        if judge_limit >= 0 and query_probe_index >= judge_limit:
            break
        if query_probe_index < judge_start_index:
            query_probe_index += 1
            continue
        query_probe_index += 1
        probe_count += 1
    return probe_count * judge_repetitions


def quality_composite_from_row(row: dict, system: str) -> float:
    scores = row.get("systems", {}).get(system, {})
    if not isinstance(scores, dict):
        return 0.0
    total = 0.0
    for field, weight in QUALITY_COMPOSITE_WEIGHTS.items():
        total += float(scores.get(field, 0.0) or 0.0) * weight
    return total


def quality_delta_from_row(row: dict) -> float:
    return quality_composite_from_row(row, "cortext_native") - quality_composite_from_row(
        row, "traditional_chat_rag"
    )


def unrecoverable_quality_stop(
    judged: list[dict],
    expected_rows: int,
    floor: float | None,
    prior_quality_delta_sum: float = 0.0,
    prior_judgment_count: int = 0,
) -> dict | None:
    if floor is None or expected_rows <= 0 or not judged:
        return None
    expected_total_rows = prior_judgment_count + expected_rows
    rows_judged = prior_judgment_count + len(judged)
    if rows_judged >= expected_total_rows:
        return None
    current_sum = prior_quality_delta_sum + sum(quality_delta_from_row(row) for row in judged)
    remaining = expected_total_rows - rows_judged
    max_remaining_delta_per_row = 11.25
    optimistic_final_delta = (
        current_sum + remaining * max_remaining_delta_per_row
    ) / expected_total_rows
    observed_delta = current_sum / rows_judged
    if optimistic_final_delta >= floor:
        return None
    return {
        "reason": "quality_delta_floor_unrecoverable",
        "segment_rows_judged": len(judged),
        "segment_expected_rows": expected_rows,
        "prior_rows": prior_judgment_count,
        "rows_judged": rows_judged,
        "expected_rows": expected_total_rows,
        "remaining_rows": remaining,
        "observed_quality_delta_vs_rag": observed_delta,
        "optimistic_final_quality_delta_vs_rag": optimistic_final_delta,
        "floor": floor,
        "max_remaining_delta_per_row": max_remaining_delta_per_row,
    }


def summarize_db(conn: sqlite3.Connection) -> dict:
    def scalar(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0] or 0)

    return {
        "memories": scalar("select count(*) from memories"),
        "long_term_memories": scalar("select count(*) from memories where kind = 'LONG_TERM'"),
        "working_memories": scalar("select count(*) from memories where kind = 'WORKING'"),
        "associations": scalar("select count(*) from associations"),
        "facts": scalar("select count(*) from fact_assertions"),
        "soft_anchor_links": scalar("select count(*) from soft_anchor_links"),
        "signals": scalar("select count(*) from signals"),
        "source_backed_memories": scalar("select count(*) from memories where blob_id is not null"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--db", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--checkpoint-rows",
        type=pathlib.Path,
        help=(
            "Private JSONL checkpoint for completed judge rows. Defaults to "
            "<out>.rows.jsonl. The final aggregate artifact is unchanged."
        ),
    )
    parser.add_argument(
        "--judge-provider",
        choices=("nemotron", "ollama"),
        default="nemotron",
        help="Local judge backend. Hosted endpoints are refused for Julie metadata.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Judge model name. Defaults to Nemotron for --judge-provider nemotron "
            "and gemma4:12b-it-qat for --judge-provider ollama."
        ),
    )
    parser.add_argument(
        "--ollama-base-url",
        help="Loopback Ollama base URL, for example http://127.0.0.1:11434.",
    )
    parser.add_argument(
        "--ollama-keep-alive",
        default="5m",
        help=(
            "Ollama model residency after each local judge call. Streamed "
            "early judging should pass 0s so the judge model releases GPU "
            "memory before Cortext consolidation resumes."
        ),
    )
    parser.add_argument("--judge-limit", type=int, default=-1)
    parser.add_argument(
        "--judge-start-index",
        type=int,
        default=0,
        help=(
            "Skip this many query-bearing probes before judging. This is used "
            "by streamed early checkpoints to avoid rejudging prior milestones."
        ),
    )
    parser.add_argument(
        "--judge-repetitions",
        type=int,
        default=1,
        help="Number of repeated judge passes per probe.",
    )
    parser.add_argument(
        "--judge-seed",
        type=int,
        default=42,
        help="Seed for deterministic packet blinding and bootstrap resampling.",
    )
    parser.add_argument(
        "--blind-packets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize packet order and expose only A/B/C labels to the judge.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Probe-level bootstrap samples for confidence intervals.",
    )
    parser.add_argument(
        "--judge-timeout-s",
        type=int,
        default=180,
        help="HTTP timeout per local judge call.",
    )
    parser.add_argument(
        "--judge-context-window-tokens",
        type=int,
        default=32768,
        help=(
            "Judge context window to request from the local backend and to use "
            "for prompt-fit checks."
        ),
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=-1,
        help=(
            "Maximum items per packet for judge prompt debugging. The release "
            "default is -1 so judged packets reflect the full benchmark context."
        ),
    )
    parser.add_argument(
        "--max-media-per-system",
        type=int,
        default=4,
        help=(
            "Maximum source blobs to attach for each scored system. Use -1 for all "
            "media in the packet or 0 to disable multimodal attachments."
        ),
    )
    parser.add_argument(
        "--early-stop-min-quality-delta-vs-rag",
        type=float,
        help=(
            "Optional confirmation-only fail-fast floor. After each completed "
            "row, stop and write a partial aggregate if even perfect remaining "
            "rows cannot bring Cortext's quality composite delta versus "
            "traditional chat+RAG up to this floor."
        ),
    )
    parser.add_argument(
        "--early-stop-prior-quality-delta-sum",
        type=float,
        default=0.0,
        help=(
            "Cumulative Cortext-minus-RAG quality-delta sum from accepted prior "
            "segments when judging only a streamed delta segment."
        ),
    )
    parser.add_argument(
        "--early-stop-prior-judgment-count",
        type=int,
        default=0,
        help=(
            "Number of accepted prior segment judgments represented by "
            "--early-stop-prior-quality-delta-sum."
        ),
    )
    args = parser.parse_args()
    if args.judge_repetitions < 1:
        raise RuntimeError("--judge-repetitions must be >= 1")
    if args.bootstrap_samples < 0:
        raise RuntimeError("--bootstrap-samples must be >= 0")
    if args.judge_start_index < 0:
        raise RuntimeError("--judge-start-index must be >= 0")
    if args.judge_limit >= 0 and args.judge_start_index > args.judge_limit:
        raise RuntimeError("--judge-start-index cannot exceed --judge-limit")
    if args.judge_timeout_s < 1:
        raise RuntimeError("--judge-timeout-s must be >= 1")
    if args.judge_context_window_tokens < 1:
        raise RuntimeError("--judge-context-window-tokens must be >= 1")
    if args.early_stop_prior_judgment_count < 0:
        raise RuntimeError("--early-stop-prior-judgment-count must be >= 0")
    judge_model = resolve_judge_model(args.judge_provider, args.model)
    judge_base_url = (
        local_ollama_base_url(args.ollama_base_url)
        if args.judge_provider == "ollama"
        else local_judge_base_url()
    )
    judge_media_capabilities = (
        ollama_model_capabilities(judge_base_url, judge_model)
        if args.judge_provider == "ollama"
        else {
            "source": "manual_nemotron_assumption",
            "model": judge_model,
            "available": True,
            "capabilities": ["text"],
            "text": True,
            "image": False,
            "audio": False,
            "error": "",
        }
    )

    summary = json.loads(args.summary.read_text())
    input_dir = pathlib.Path(summary["input_dir"])
    db_path = args.db or pathlib.Path(summary["db_path"])
    max_messages = int(
        summary.get("timeline_max_messages", summary.get("processed_text_messages", -1))
    )
    skip_messages = int(
        summary.get("timeline_skip_messages", summary.get("skipped_transcript_messages", 0))
    )
    media_limit = int(
        summary.get("timeline_media_limit", summary.get("media_attempted", -1))
    )
    timeline = build_timeline(input_dir, skip_messages, max_messages, media_limit)
    expected_judgments = expected_judgment_count(
        summary,
        timeline,
        args.judge_limit,
        args.judge_start_index,
        args.judge_repetitions,
    )
    by_index = docs_by_index(timeline)
    conn = connect_db(db_path)
    media_memory_map = build_media_memory_map(conn, timeline)

    systems = SYSTEMS
    fields = FIELDS
    totals = {system: Counter() for system in systems}
    judged: list[dict] = []
    failure_reasons = Counter()
    judge_validation = Counter()
    token_totals = Counter()
    fairness_checks = Counter()
    media_attachment_totals = {
        "enabled": args.max_media_per_system != 0,
        "max_media_per_system": args.max_media_per_system,
        "cortext_native": Counter(),
        "traditional_chat_rag": Counter(),
        "full_history_upper_bound": Counter(),
    }
    cortext_packet_source = Counter()
    early_stop: dict | None = None

    checkpoint_rows = args.checkpoint_rows or args.out.with_name(args.out.name + ".rows.jsonl")
    checkpoint_rows.parent.mkdir(parents=True, exist_ok=True)

    with checkpoint_rows.open("a") as checkpoint_file, tempfile.TemporaryDirectory(
        prefix="cortext_julie_judge_media_"
    ) as tmp:
        work_dir = pathlib.Path(tmp)
        probes_seen = 0
        query_probe_index = 0
        for probe in summary.get("probes", []):
            query_doc = find_query_doc(timeline, probe)
            if not query_doc:
                continue
            if args.judge_limit >= 0 and query_probe_index >= args.judge_limit:
                break
            if query_probe_index < args.judge_start_index:
                query_probe_index += 1
                continue
            query_probe_index += 1
            probes_seen += 1
            frozen_working_rows = frozen_memory_rows(probe, "cortext_frozen_working_memory")
            frozen_retrieved_rows = frozen_memory_rows(
                probe, "cortext_frozen_retrieved_memory"
            )
            use_frozen_cortext_packet = bool(frozen_working_rows or frozen_retrieved_rows)
            if use_frozen_cortext_packet:
                raw_working_rows = frozen_working_rows
                raw_retrieved_rows = frozen_retrieved_rows
                raw_memory_rows = unique_memory_rows(raw_working_rows + raw_retrieved_rows)
                working_memory_ids = [
                    int(row_get(row, "memory_id", 0) or 0)
                    for row in raw_working_rows
                ]
                retrieved_memory_ids = [
                    int(row_get(row, "memory_id", 0) or 0)
                    for row in raw_retrieved_rows
                ]
                memory_ids = list(dict.fromkeys(working_memory_ids + retrieved_memory_ids))
                cortext_packet_source["probe_time_summary_snapshot"] += 1
            else:
                working_memory_ids = [
                    int(mid) for mid in probe.get("cortext_working_memory_ids", [])
                ]
                retrieved_memory_ids = [
                    int(mid) for mid in probe.get("cortext_retrieved_memory_ids", [])
                ]
                memory_ids = list(dict.fromkeys(working_memory_ids + retrieved_memory_ids))
                raw_memory_rows = load_memory_rows(conn, memory_ids)
                raw_working_rows = load_memory_rows(conn, working_memory_ids)
                raw_retrieved_rows = load_memory_rows(conn, retrieved_memory_ids)
                cortext_packet_source["final_db_rehydration_fallback"] += 1
            event_index = int(probe["event_index"])
            memory_rows, excluded_current_memory_ids = exclude_current_memory_rows(
                raw_memory_rows,
                timeline,
                media_memory_map,
                event_index,
            )
            working_rows, excluded_current_working_ids = exclude_current_memory_rows(
                raw_working_rows,
                timeline,
                media_memory_map,
                event_index,
            )
            retrieved_rows, excluded_current_retrieved_ids = exclude_current_memory_rows(
                raw_retrieved_rows,
                timeline,
                media_memory_map,
                event_index,
            )
            cortext_working_tokens = memory_doc_tokens(
                working_rows,
                timeline,
                media_memory_map,
                args.context_limit,
            )
            remaining_context_limit = (
                -1
                if args.context_limit < 0
                else max(0, args.context_limit - min(len(working_rows), args.context_limit))
            )
            cortext_retrieved_tokens = memory_doc_tokens(
                retrieved_rows,
                timeline,
                media_memory_map,
                remaining_context_limit,
            )
            cortext_context_tokens = cortext_working_tokens + cortext_retrieved_tokens
            cortext_docs = memory_docs(memory_rows, timeline, media_memory_map)
            rolling_docs = [
                by_index[row["index"]]
                for row in probe.get("rolling_history", [])
                if row.get("index") in by_index
            ]
            compacted = compaction_doc(probe, query_doc)
            if compacted is not None:
                fairness_checks["traditional_chat_rag_compaction_events"] += 1
                if compaction_summary_has_content(probe):
                    fairness_checks["traditional_chat_rag_contentful_compaction"] += 1
                else:
                    fairness_checks["traditional_chat_rag_marker_only_compaction"] += 1
                rolling_docs = [compacted] + rolling_docs
            rag_docs = [
                by_index[row["index"]]
                for row in probe.get("rag_top_k", [])
                if row.get("index") in by_index
            ]
            rag_packet_docs = text_only_docs(
                prior_context_docs(rolling_docs, event_index)
                + [doc for doc in rag_docs if doc.index not in {row.index for row in rolling_docs}]
            )
            rag_packet_docs = prior_context_docs(rag_packet_docs, event_index)
            full_docs = text_only_docs(
                [doc for doc in timeline if doc.index < event_index]
            )
            rag_context_tokens = int(
                probe.get(
                    "normal_rag_context_tokens",
                    probe.get(
                        "normal_rag_active_history_tokens",
                        probe.get("rolling_history_tokens", 0),
                    ),
                )
                or 0
            )
            token_totals["cortext_working_tokens"] += cortext_working_tokens
            token_totals["cortext_retrieved_tokens"] += cortext_retrieved_tokens
            token_totals["cortext_context_tokens"] += cortext_context_tokens
            token_totals["traditional_chat_rag_tokens"] += rag_context_tokens
            token_totals["full_history_tokens"] += int(probe.get("full_history_tokens", 0) or 0)
            token_totals["judged"] += 1

            packet_docs_by_system = {
                "cortext_native": cortext_docs,
                "traditional_chat_rag": rag_packet_docs,
                "full_history_upper_bound": full_docs,
            }
            for row in memory_rows:
                docs = map_memory_to_docs(row, timeline, media_memory_map)
                if docs:
                    continue
                start_ts = int(row_get(row, "start_ts", 0) or 0)
                if start_ts > query_doc.timestamp:
                    fairness_checks["cortext_native_unmapped_future_timestamp_rows"] += 1
                elif start_ts == query_doc.timestamp:
                    fairness_checks["cortext_native_unmapped_current_timestamp_rows"] += 1
            for system, docs in packet_docs_by_system.items():
                future = [
                    doc.index
                    for doc in docs
                    if doc.index >= 0 and doc.index > event_index
                ]
                current = [
                    doc.index
                    for doc in docs
                    if doc.index >= 0 and doc.index == event_index
                ]
                if future:
                    fairness_checks[f"{system}_future_context_violations"] += len(future)
                if current:
                    fairness_checks[f"{system}_current_turn_context_inclusions"] += len(current)
            if any(doc.modality != "text" for doc in rag_packet_docs):
                fairness_checks["traditional_chat_rag_non_text_context"] += 1
            if any(doc.modality != "text" for doc in full_docs):
                fairness_checks["full_history_non_text_context"] += 1

            for repetition in range(args.judge_repetitions):
                ordered_systems, real_to_label, label_to_real = packet_assignment(
                    systems,
                    args.blind_packets,
                    args.judge_seed,
                    int(probe["event_index"]),
                    repetition,
                )
                packet_sections: list[str] = []
                for system in ordered_systems:
                    packet_label = real_to_label[system]
                    if system == "cortext_native":
                        packet_sections.append(
                            packet_from_memories(
                                f"PACKET {packet_label}",
                                memory_rows,
                                timeline,
                                media_memory_map,
                                args.context_limit,
                            )
                        )
                    else:
                        packet_sections.append(
                            packet_from_docs(
                                f"PACKET {packet_label}",
                                packet_docs_by_system[system],
                                args.context_limit,
                            )
                        )

                packet_key_list = ", ".join(real_to_label[system] for system in ordered_systems)
                if args.blind_packets:
                    prompt_rules = [
                        "Score each anonymized evidence packet for the current conversation turn.",
                        "Packet identities and generation methods are hidden. Judge only the structurally normalized event evidence shown in each packet and any media attachments explicitly labeled for that packet.",
                        "Some packets may include attached image/audio source evidence. Media attachments are judge-only evidence and were not transcribed or captioned back into the packet text.",
                        "Use the full 0-5 range for numeric scores: 0 absent, 1 weak, 2 partial, 3 useful, 4 strong, 5 excellent. noise is reverse-coded where 0 is clean and 5 is very noisy. modality_grounding rewards appropriate use of attached image/audio source evidence.",
                        "If a packet has no attached image/audio evidence, its modality_grounding must be 0. Do not award modality_grounding for text history alone.",
                        f"Return strict JSON with top-level keys {packet_key_list}, winner, failure_reason, and winner_reason. winner must be one of {packet_key_list} or tie_or_unclear. Each packet key must map to an object, never a scalar. Each packet object must contain numeric keys relevance, sufficiency, noise, temporal_correctness, source_grounding, modality_grounding, plus a short reason string explaining the score pattern.",
                        "Reason strings may mention only structural evidence: text history, source-backed evidence, source blob, image/audio evidence, source_id alignment, temporal alignment, missing source link, missing media, or noise. Do not quote, paraphrase, summarize, name, or restate private conversation content.",
                        "Allowed failure_reason values: missing_source_link, temporal_drift, insufficient_context, unrelated_retrieval, modality_blindness, winner_best_context, tie_or_unclear. Use winner_best_context only when the selected packet is simply the best context.",
                    ]
                else:
                    prompt_rules = [
                        "Score each memory/context packet for the current conversation turn.",
                        "Cortext native used production WM + STM/LTM graph retrieval and may return text, audio, or image source-backed memories. Cortext audio/image ingestion used embeddings and source blobs only; no ASR transcript or caption text was passed into Cortext.",
                        "Traditional chat+RAG uses rolling text chat history until compaction plus text RAG hits from the same prior event stream. Full history is a text-only upper bound over prior chat messages.",
                        "When the Cortext packet references attached media evidence, inspect the corresponding image/audio attachments before scoring modality_grounding and relevance. Media attachments are judge-only evidence and were not transcribed/captioned back into Cortext.",
                        "Use the full 0-5 range for numeric scores: 0 absent, 1 weak, 2 partial, 3 useful, 4 strong, 5 excellent. noise is reverse-coded where 0 is clean and 5 is very noisy. modality_grounding rewards appropriate use of text/audio/image source evidence.",
                        "For this normal-RAG benchmark, traditional_chat_rag and full_history_upper_bound are text-only systems. If a system has no attached image/audio evidence, its modality_grounding must be 0. Do not award modality_grounding for text history alone.",
                        f"Return strict JSON with top-level keys {packet_key_list}, winner, failure_reason, and winner_reason. Each system key must map to an object, never a scalar. Each system object must contain numeric keys relevance, sufficiency, noise, temporal_correctness, source_grounding, modality_grounding, plus a short reason string explaining the score pattern.",
                        "Reason strings may mention only structural evidence: text history, graph memory, source blob, image/audio evidence, source_id alignment, temporal alignment, missing source link, missing media, or noise. Do not quote, paraphrase, summarize, name, or restate private conversation content.",
                        "Allowed failure_reason values: missing_source_link, temporal_drift, insufficient_context, unrelated_retrieval, modality_blindness, rag_context_advantage, full_history_upper_bound_advantage, cortext_wins, tie_or_unclear. If winner is cortext_native use cortext_wins unless a more specific failure applies. If winner is traditional_chat_rag use rag_context_advantage. If winner is full_history_upper_bound use full_history_upper_bound_advantage.",
                    ]

                prompt = "\n\n".join(
                    [
                        *prompt_rules,
                        f"CURRENT_TURN:\nevent_index={query_doc.index} modality={query_doc.modality} source_id={query_doc.source_id} timestamp={query_doc.timestamp}: {query_doc.text}",
                        *packet_sections,
                    ]
                )
                if args.blind_packets:
                    prompt_lower = prompt.lower()
                    hidden_mentions = sorted(
                        term for term in BLIND_FORBIDDEN_TERMS if term in prompt_lower
                    )
                    if hidden_mentions:
                        fairness_checks["blind_prompt_hidden_system_label_mentions"] += 1
                packet_media_docs = [
                    (system, real_to_label[system], packet_docs_by_system[system])
                    for system in ordered_systems
                ]
                content, media_stats = build_multimodal_content(
                    prompt,
                    input_dir,
                    packet_media_docs,
                    args.max_media_per_system,
                    work_dir,
                )
                if (
                    media_stats.get("enabled")
                    and any(
                        int(media_stats.get(system, {}).get("audio", 0) or 0) > 0
                        for system in systems
                    )
                    and not judge_media_capabilities.get("audio")
                ):
                    fairness_checks["audio_attached_but_judge_audio_unsupported"] += 1
                if (
                    media_stats.get("enabled")
                    and any(
                        int(media_stats.get(system, {}).get("image", 0) or 0) > 0
                        for system in systems
                    )
                    and not judge_media_capabilities.get("image")
                ):
                    fairness_checks["image_attached_but_judge_image_unsupported"] += 1
                judge_prompt_tokens_estimate = estimate_content_tokens(content)
                token_totals["judge_prompt_tokens_estimated"] += judge_prompt_tokens_estimate
                token_totals["max_judge_prompt_tokens_estimated"] = max(
                    token_totals["max_judge_prompt_tokens_estimated"],
                    judge_prompt_tokens_estimate,
                )
                if judge_prompt_tokens_estimate > args.judge_context_window_tokens:
                    fairness_checks["judge_prompt_context_window_exceeded"] += 1
                for system in systems:
                    for key, value in media_stats.get(system, {}).items():
                        media_attachment_totals[system][key] += int(value)

                print(
                    "judge_call "
                    f"ts={utc_now_text()} "
                    f"probe_event_index={probe.get('event_index')} "
                    f"repetition={repetition + 1}/{args.judge_repetitions}",
                    flush=True,
                )
                scores = normalize_scores(
                    call_judge(
                        args.judge_provider,
                        judge_model,
                        judge_base_url,
                        content,
                        1300,
                        args.judge_timeout_s,
                        args.judge_context_window_tokens,
                        args.ollama_keep_alive,
                    )
                )
                for packet_label in label_to_real:
                    for candidate in (
                        f"Packet {packet_label}",
                        f"packet {packet_label}",
                        f"PACKET {packet_label}",
                        packet_label.lower(),
                    ):
                        if candidate in scores:
                            scores.setdefault(packet_label, scores[candidate])

                raw_winner = normalize_winner_key(
                    scores.get("winner", "unknown"),
                    set(label_to_real) | set(systems),
                )
                if raw_winner in label_to_real:
                    winner = label_to_real[raw_winner]
                elif raw_winner in systems:
                    winner = raw_winner
                else:
                    winner = "tie_or_unclear"
                raw_failure_reason = str(scores.get("failure_reason", "tie_or_unclear"))
                raw_winner_reason = safe_reason(scores.get("winner_reason") or scores.get("reason"))
                failure_reason = raw_failure_reason
                allowed = BLIND_FAILURE_REASONS if args.blind_packets else SYSTEM_FAILURE_REASONS
                if failure_reason not in allowed:
                    judge_validation["invalid_failure_reason"] += 1
                    failure_reason = expected_failure_reason(winner) or "tie_or_unclear"
                elif failure_reason == "winner_best_context":
                    failure_reason = expected_failure_reason(winner) or "tie_or_unclear"
                expected = expected_failure_reason(winner)
                if (
                    expected is not None
                    and failure_reason in {
                        "rag_context_advantage",
                        "full_history_upper_bound_advantage",
                        "cortext_wins",
                    }
                    and failure_reason != expected
                ):
                    judge_validation["winner_failure_mismatch"] += 1
                    failure_reason = expected
                failure_reasons[failure_reason] += 1
                row = {
                    "event_index": int(probe["event_index"]),
                    "repetition": repetition,
                    "query": {
                        "timestamp": query_doc.timestamp,
                        "source_id": query_doc.source_id,
                        "modality": query_doc.modality,
                        "tokens": estimate_tokens(query_doc.text),
                    },
                    "winner": winner,
                    "winner_alias": raw_winner,
                    "failure_reason": failure_reason,
                    "winner_reason": raw_winner_reason,
                    "judge_raw": {
                        "winner": raw_winner,
                        "failure_reason": raw_failure_reason,
                        "winner_reason": raw_winner_reason,
                    },
                    "packet_blinding": {
                        "enabled": args.blind_packets,
                        "real_to_label": real_to_label,
                        "label_to_real": label_to_real,
                    },
                    "systems": {},
                    "cortext_memory_ids": memory_ids,
                    "cortext_packet_source": (
                        "probe_time_summary_snapshot"
                        if use_frozen_cortext_packet
                        else "final_db_rehydration_fallback"
                    ),
                    "cortext_judged_memory_ids": [
                        int(row_get(row, "memory_id", 0) or 0)
                        for row in memory_rows
                    ],
                    "cortext_current_turn_memory_ids_excluded": excluded_current_memory_ids,
                    "cortext_current_turn_working_ids_excluded": excluded_current_working_ids,
                    "cortext_current_turn_retrieved_ids_excluded": excluded_current_retrieved_ids,
                    "rag_top_k_indices": probe.get("rag_top_k_indices", []),
                    "rolling_history_tokens": probe.get(
                        "normal_rag_active_history_tokens",
                        probe.get("rolling_history_tokens", 0),
                    ),
                    "normal_rag_context_tokens": probe.get(
                        "normal_rag_context_tokens", rag_context_tokens
                    ),
                    "full_history_tokens": probe.get("full_history_tokens", 0),
                    "cortext_working_tokens": cortext_working_tokens,
                    "cortext_retrieved_tokens": cortext_retrieved_tokens,
                    "cortext_context_tokens": cortext_context_tokens,
                    "traditional_chat_rag_tokens": rag_context_tokens,
                    "cortext_latency_ms": probe.get("cortext_latency_ms", 0.0),
                    "judge_prompt_tokens_estimate": judge_prompt_tokens_estimate,
                    "media_attachments": media_stats,
                    "judge_adjustments": [],
                }
                for system in systems:
                    judge_key = real_to_label[system]
                    totals[system]["judged"] += 1
                    if winner == system:
                        totals[system]["wins"] += 1
                    row["systems"][system] = {"judge_key": judge_key}
                    for field in fields:
                        value = score(scores, judge_key, field)
                        raw_value = value
                        if (
                            field == "modality_grounding"
                            and attached_media_count(media_stats, system) == 0
                            and value > 0.0
                        ):
                            value = 0.0
                            judge_validation["modality_score_clamped_no_media"] += 1
                            row["judge_adjustments"].append(
                                {
                                    "system": system,
                                    "judge_key": judge_key,
                                    "field": field,
                                    "raw": raw_value,
                                    "adjusted": value,
                                    "reason": "system_has_no_attached_image_or_audio_evidence",
                                }
                            )
                        totals[system][field] += value
                        row["systems"][system][field] = value
                        if value != raw_value:
                            row["systems"][system].setdefault("raw_scores", {})[field] = raw_value
                    row["systems"][system]["reason"] = system_reason(scores, judge_key)
                    if not row["systems"][system]["reason"]:
                        judge_validation["missing_system_reason"] += 1
                judged.append(row)
                checkpoint_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                checkpoint_file.flush()
                early_stop = unrecoverable_quality_stop(
                    judged,
                    expected_judgments,
                    args.early_stop_min_quality_delta_vs_rag,
                    args.early_stop_prior_quality_delta_sum,
                    args.early_stop_prior_judgment_count,
                )
                if early_stop is not None:
                    print(
                        "judge_early_stop "
                        f"ts={utc_now_text()} "
                        f"reason={early_stop['reason']} "
                        f"rows_judged={early_stop['rows_judged']} "
                        f"expected_rows={early_stop['expected_rows']} "
                        "optimistic_final_quality_delta_vs_rag="
                        f"{early_stop['optimistic_final_quality_delta_vs_rag']:.6f} "
                        f"floor={early_stop['floor']:.6f}",
                        flush=True,
                    )
                    break
            if early_stop is not None:
                break

    quality = {}
    for system in systems:
        n = max(1, int(totals[system]["judged"]))
        quality[system] = {
            "judged": int(totals[system]["judged"]),
            "wins": int(totals[system]["wins"]),
        }
        for field in fields:
            quality[system][f"mean_{field}"] = totals[system][field] / n
    token_n = max(1, int(token_totals["judged"]))
    mean_cortext_context_tokens = token_totals["cortext_context_tokens"] / token_n
    mean_traditional_chat_rag_tokens = token_totals["traditional_chat_rag_tokens"] / token_n
    token_savings_vs_traditional_chat_rag = (
        1.0 - (mean_cortext_context_tokens / mean_traditional_chat_rag_tokens)
        if mean_traditional_chat_rag_tokens > 0
        else 0.0
    )

    output = {
        "schema": "julie_live_run_multimodal_judge_v8",
        "summary_path": str(args.summary),
        "db_path": str(db_path),
        "input_dir": str(input_dir),
        "judge_model": judge_model,
        "judge_provider": (
            "local_ollama" if args.judge_provider == "ollama" else "local_nemotron_vllm_mlx"
        ),
        "judge_base_url": judge_base_url,
        "judge_media_capabilities": judge_media_capabilities,
        "remote_provider_allowed": False,
        "native_cortext_only": True,
        "multimodal_judge": args.max_media_per_system != 0,
        "protocol": {
            "packet_blinding": args.blind_packets,
            "packet_labels": PACKET_ALIASES[:len(systems)] if args.blind_packets else systems,
            "judge_repetitions": args.judge_repetitions,
            "judge_seed": args.judge_seed,
            "judge_start_index": args.judge_start_index,
            "judge_limit": args.judge_limit,
            "expected_judgments": expected_judgments,
            "judge_timeout_s": args.judge_timeout_s,
            "judge_context_window_tokens": args.judge_context_window_tokens,
            "ollama_keep_alive": args.ollama_keep_alive
            if args.judge_provider == "ollama"
            else "",
            "early_stop_prior_quality_delta_sum": (
                args.early_stop_prior_quality_delta_sum
            ),
            "early_stop_prior_judgment_count": (
                args.early_stop_prior_judgment_count
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "probe",
            "packet_surface": PACKET_SURFACE,
            "systems": systems,
            "score_fields": fields,
            "quality_composite_definition": QUALITY_COMPOSITE_DEFINITION,
            "quality_composite_weights": QUALITY_COMPOSITE_WEIGHTS,
            "cortext_packet_source": dict(cortext_packet_source),
            "normal_rag_baseline": (
                "rolling text chat history until compaction plus text RAG hits"
            ),
            "traditional_chat_rag_compaction_summary_policy": summary.get(
                "normal_rag_compaction_summary_policy"
            ),
            "full_history_upper_bound": "prior text history only",
            "daily_consolidation_required_for_release": True,
            "default_knobs_required_for_release": [0.5, 0.5, 0.5],
            "judge_media_capabilities": judge_media_capabilities,
        },
        "cortext_audio_image_transcript_shortcuts": False,
        "output_contains_private_text": True,
        "reason_privacy_policy": (
            "Judge reasons are instructed to be structural only, but the artifact "
            "is treated as private because local judges may still paraphrase content."
        ),
        "daily_consolidation": bool(summary.get("daily_consolidation")),
        "deep_consolidation": bool(summary.get("deep_consolidation")),
        "source_id_policy": summary.get("source_id_policy"),
        "timeline_policy": summary.get("timeline_policy"),
        "processed": {
            "text": summary.get("processed_text_messages", 0),
            "audio": summary.get("audio_processed", 0),
            "image": summary.get("image_processed", 0),
            "video": summary.get("video_processed", 0),
            "media_failures": summary.get("media_failures", 0),
        },
        "tokens": {
            "active_history_token_budget": summary.get("active_history_token_budget", 0),
            "mean_rolling_history_tokens": (
                sum(row["rolling_history_tokens"] for row in judged) / max(1, len(judged))
            ),
            "mean_full_history_tokens": (
                sum(row["full_history_tokens"] for row in judged) / max(1, len(judged))
            ),
            "context_limit_memories": args.context_limit,
            "mean_cortext_working_tokens": (
                token_totals["cortext_working_tokens"] / token_n
            ),
            "mean_cortext_retrieved_tokens": (
                token_totals["cortext_retrieved_tokens"] / token_n
            ),
            "mean_cortext_context_tokens": mean_cortext_context_tokens,
            "mean_traditional_chat_rag_tokens": mean_traditional_chat_rag_tokens,
            "mean_cortext_token_savings_vs_traditional_chat_rag": (
                token_savings_vs_traditional_chat_rag
            ),
            "judge_context_window_tokens": args.judge_context_window_tokens,
            "mean_judge_prompt_tokens_estimate": (
                token_totals["judge_prompt_tokens_estimated"] / max(1, len(judged))
            ),
            "max_judge_prompt_tokens_estimate": int(
                token_totals["max_judge_prompt_tokens_estimated"]
            ),
        },
        "normal_rag_compaction": summary.get("normal_rag_compaction", {}),
        "normal_rag_retrieval": summary.get("normal_rag_retrieval"),
        "normal_rag_baseline_modality": summary.get(
            "normal_rag_baseline_modality"
        ),
        "normal_rag_vector_query_encoder": summary.get(
            "normal_rag_vector_query_encoder"
        ),
        "normal_rag_vector_candidate_k": summary.get(
            "normal_rag_vector_candidate_k"
        ),
        "latency": {
            "mean_cortext_probe_latency_ms": (
                sum(float(row["cortext_latency_ms"]) for row in judged) / max(1, len(judged))
            ),
            "mean_ingest_total_ms": summary.get("mean_total_ms", 0.0),
            "wall_ms": summary.get("wall_ms", 0.0),
        },
        "db_metrics": summarize_db(conn),
        "judged": len(judged),
        "probe_count": probes_seen,
        "judge_repetitions": args.judge_repetitions,
        "early_stop": early_stop,
        "quality": quality,
        "confidence_intervals": confidence_intervals(
            judged,
            systems,
            fields,
            args.bootstrap_samples,
            args.judge_seed,
        ),
        "failure_reasons": dict(failure_reasons),
        "judge_validation": dict(judge_validation),
        "fairness_checks": {
            "cortext_audio_image_transcript_shortcuts": False,
            "traditional_chat_rag_text_only": (
                fairness_checks.get("traditional_chat_rag_non_text_context", 0) == 0
            ),
            "full_history_text_only": (
                fairness_checks.get("full_history_non_text_context", 0) == 0
            ),
            "no_future_context_violations": not any(
                key.endswith("_future_context_violations") and value > 0
                for key, value in fairness_checks.items()
            )
            and fairness_checks.get("cortext_native_unmapped_future_timestamp_rows", 0)
            == 0,
            "no_current_turn_context_inclusions": not any(
                key.endswith("_current_turn_context_inclusions") and value > 0
                for key, value in fairness_checks.items()
            )
            and fairness_checks.get("cortext_native_unmapped_current_timestamp_rows", 0)
            == 0,
            "blind_prompt_hidden_labels_absent": (
                fairness_checks.get("blind_prompt_hidden_system_label_mentions", 0) == 0
            ),
            "traditional_chat_rag_contentful_compaction": (
                fairness_checks.get("traditional_chat_rag_compaction_events", 0) == 0
                or (
                    fairness_checks.get("traditional_chat_rag_contentful_compaction", 0)
                    == fairness_checks.get("traditional_chat_rag_compaction_events", 0)
                    and fairness_checks.get("traditional_chat_rag_marker_only_compaction", 0) == 0
                )
            ),
            "judge_prompt_fits_context_window": (
                fairness_checks.get("judge_prompt_context_window_exceeded", 0) == 0
            ),
            "full_history_prompt_fits_judge_context": (
                fairness_checks.get("judge_prompt_context_window_exceeded", 0) == 0
            ),
            "judge_supports_attached_audio": bool(
                judge_media_capabilities.get("audio")
            ),
            "judge_supports_attached_images": bool(
                judge_media_capabilities.get("image")
            ),
            "attached_audio_judged_when_present": (
                fairness_checks.get("audio_attached_but_judge_audio_unsupported", 0)
                == 0
            ),
            "attached_images_judged_when_present": (
                fairness_checks.get("image_attached_but_judge_image_unsupported", 0)
                == 0
            ),
            "counters": dict(fairness_checks),
        },
        "media_attachments": {
            "enabled": media_attachment_totals["enabled"],
            "max_media_per_system": media_attachment_totals["max_media_per_system"],
            "cortext_native": dict(media_attachment_totals["cortext_native"]),
            "traditional_chat_rag": dict(media_attachment_totals["traditional_chat_rag"]),
            "full_history_upper_bound": dict(media_attachment_totals["full_history_upper_bound"]),
        },
        "judgments": judged,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
