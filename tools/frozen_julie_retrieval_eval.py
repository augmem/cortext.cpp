#!/usr/bin/env python3
"""Freeze and evaluate Julie retrieval probes against labeled target memories.

The labels produced here are deterministic, self-labeled relevance targets from
the source timeline. They are useful for regression pressure, not as independent
human ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import sqlite3
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any

from generate_julie_raw_speech_manifest import parse_messages, parse_timestamp


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9']+")
STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been",
    "but", "can", "could", "did", "for", "from", "get", "had", "has",
    "have", "her", "him", "his", "how", "just", "like", "not", "now",
    "our", "out", "she", "that", "the", "their", "them", "then", "there",
    "they", "this", "was", "what", "when", "where", "who", "why", "with",
    "would", "you", "your",
}
LOCAL_JUDGE_HOSTS = {"localhost", "127.0.0.1", "::1"}
DEFAULT_LOCAL_JUDGE_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:12b-it-qat"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".gif", ".tiff"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".3gp"}
AUDIO_EXTENSIONS = {".m4a", ".wav", ".mp3"}
GABE_SOURCE_ID = "Gabe"
JULIE_SOURCE_ID = "Julie"


@dataclass(frozen=True)
class Doc:
    index: int
    timestamp: int
    source_id: str
    modality: str
    text: str
    source_blob: str = ""


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def local_judge_base_url() -> str:
    base_url = (
        os.environ.get("CORTEXT_JUDGE_BASE_URL")
        or os.environ.get("LOCAL_JUDGE_BASE_URL")
        or DEFAULT_LOCAL_JUDGE_BASE_URL
    ).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    parsed_base = urllib.parse.urlparse(base_url)
    if parsed_base.scheme not in {"http", "https"} or parsed_base.hostname not in LOCAL_JUDGE_HOSTS:
        raise RuntimeError(
            "Refusing non-local judge endpoint for private retrieval labels: "
            f"{base_url!r}. Start the local Nemotron/MLX judge server and set "
            "CORTEXT_JUDGE_BASE_URL or LOCAL_JUDGE_BASE_URL to a loopback URL."
        )
    return base_url


def local_ollama_base_url(cli_base_url: str | None = None) -> str:
    base_url = (
        cli_base_url
        or os.environ.get("CORTEXT_OLLAMA_BASE_URL")
        or os.environ.get("OLLAMA_BASE_URL")
        or DEFAULT_OLLAMA_BASE_URL
    ).rstrip("/")
    parsed_base = urllib.parse.urlparse(base_url)
    if parsed_base.scheme not in {"http", "https"} or parsed_base.hostname not in LOCAL_JUDGE_HOSTS:
        raise RuntimeError(
            "Refusing non-local Ollama endpoint for private retrieval labels: "
            f"{base_url!r}. Set CORTEXT_OLLAMA_BASE_URL to a loopback URL."
        )
    return base_url


def require_nemotron_model(model: str) -> None:
    if "nemotron" not in model.lower():
        raise RuntimeError(
            f"Refusing non-Nemotron judge model for private retrieval labels: {model!r}. "
            "Start the local Nemotron/MLX judge server and pass --model nemotron..."
        )


def parse_json_object(content: str, source: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise RuntimeError(f"{source} returned non-JSON content: {content[:500]!r}")


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


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


def media_source_id(path: pathlib.Path, kind: str) -> str:
    name = path.name.lower()
    if kind == "audio" and ("_self" in name or " self " in name):
        return GABE_SOURCE_ID
    return JULIE_SOURCE_ID


def build_timeline(
    input_dir: pathlib.Path,
    max_messages: int,
    media_limit: int,
    skip_messages: int = 0,
) -> list[Doc]:
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
    for i, _ in enumerate(media):
        events.append((media[i][1], len(messages) * 2 + i, True, i))
    events.sort(key=lambda row: (row[0], row[2], row[1]))

    out: list[Doc] = []
    media_attempted = 0
    for _, _, is_media, idx in events:
        if not is_media:
            message = messages[idx]
            out.append(Doc(
                index=len(out),
                timestamp=int(message["timestamp"]),
                source_id=source_for_message(message),
                modality="text",
                text=message["text"],
            ))
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
        out.append(Doc(
            index=len(out),
            timestamp=int(timestamp),
            source_id=media_source_id(path, kind),
            modality=modality,
            text=f"[{kind} source blob: {path.name}]",
            source_blob=source_blob,
        ))
    return out


def timeline_args_from_summary(summary: dict) -> tuple[int, int, int]:
    skip_messages = int(
        summary.get("timeline_skip_messages", summary.get("skipped_transcript_messages", 0))
    )
    max_messages = int(
        summary.get("timeline_max_messages", summary.get("processed_text_messages", -1))
    )
    media_limit = int(
        summary.get("timeline_media_limit", summary.get("media_attempted", -1))
    )
    return skip_messages, max_messages, media_limit


def connect(path: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def load_memory_doc_map(conn: sqlite3.Connection, timeline: list[Doc]) -> dict[int, int]:
    rows = conn.execute(
        "select memory_id, source_id, modality, start_ts from memories where kind != 'LABEL'"
    ).fetchall()
    text_key_to_memory: dict[tuple[str, str, int], int] = {}
    media_by_source_modality: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for row in rows:
        memory_id = int(row["memory_id"])
        key = (str(row["source_id"]), str(row["modality"]), int(row["start_ts"] or 0))
        if row["modality"] == "text":
            text_key_to_memory.setdefault(key, memory_id)
        else:
            media_by_source_modality.setdefault((str(row["source_id"]), str(row["modality"])), []).append(
                (int(row["start_ts"] or 0), memory_id)
            )

    doc_to_memory: dict[int, int] = {}
    for doc in timeline:
        if doc.modality == "text":
            mid = text_key_to_memory.get((doc.source_id, doc.modality, doc.timestamp))
            if mid:
                doc_to_memory[doc.index] = mid
            continue
        candidates = media_by_source_modality.get((doc.source_id, doc.modality), [])
        if candidates:
            doc_to_memory[doc.index] = min(candidates, key=lambda row: abs(row[0] - doc.timestamp))[1]
    return doc_to_memory


def label_targets(
    timeline: list[Doc],
    doc_to_memory: dict[int, int],
    probe: dict,
    max_targets: int,
    min_overlap: float,
    label_sources: set[str],
) -> list[dict]:
    event_index = int(probe["event_index"])
    query_text = ""
    if 0 <= event_index < len(timeline):
        query_text = timeline[event_index].text
    query_tokens = set(tokens(query_text))
    if not query_tokens:
        return []

    scored_by_memory: dict[int, tuple[float, Doc, int, float, list[str]]] = {}

    def add_candidate(score: float, doc: Doc, memory_id: int, overlap: float, source: str) -> None:
        current = scored_by_memory.get(memory_id)
        if current is None:
            scored_by_memory[memory_id] = (score, doc, memory_id, overlap, [source])
            return
        old_score, old_doc, _, old_overlap, sources = current
        if source not in sources:
            sources.append(source)
        if score > old_score or (score == old_score and doc.timestamp > old_doc.timestamp):
            scored_by_memory[memory_id] = (score, doc, memory_id, max(overlap, old_overlap), sources)
        else:
            scored_by_memory[memory_id] = (old_score, old_doc, memory_id, max(overlap, old_overlap), sources)

    if "overlap" in label_sources:
        for doc in timeline[:event_index]:
            memory_id = doc_to_memory.get(doc.index)
            if not memory_id or doc.modality != "text":
                continue
            doc_tokens = set(tokens(doc.text))
            if not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens))
            if overlap < min_overlap:
                continue
            distance = max(1, event_index - doc.index)
            recency = 1.0 / math.sqrt(distance)
            source_bonus = 0.10 if doc.source_id == timeline[event_index].source_id else 0.0
            score = overlap + 0.25 * recency + source_bonus
            add_candidate(score, doc, memory_id, overlap, "token_overlap_recency")

    if "rag" in label_sources:
        for rank, doc_index in enumerate(probe.get("rag_top_k_indices", []), start=1):
            if not isinstance(doc_index, int):
                continue
            if doc_index < 0 or doc_index >= min(event_index, len(timeline)):
                continue
            doc = timeline[doc_index]
            memory_id = doc_to_memory.get(doc.index)
            if not memory_id:
                continue
            doc_tokens = set(tokens(doc.text))
            overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens)) if doc_tokens else 0.0
            score = 1.0 + (1.0 / max(1, rank)) + overlap
            add_candidate(score, doc, memory_id, overlap, "normal_rag_lexical_top_k")

    scored = sorted(scored_by_memory.values(), key=lambda row: (-row[0], -row[1].timestamp, row[1].index))
    out = []
    seen: set[int] = set()
    for score, doc, memory_id, overlap, sources in scored:
        if memory_id in seen:
            continue
        seen.add(memory_id)
        out.append({
            "memory_id": memory_id,
            "event_index": doc.index,
            "timestamp": doc.timestamp,
            "source_id": doc.source_id,
            "modality": doc.modality,
            "label_score": score,
            "query_token_overlap": overlap,
            "label_sources": sources,
        })
        if len(out) >= max_targets:
            break
    return out


def call_nemotron_judge(model: str, prompt: str, max_tokens: int) -> dict:
    require_nemotron_model(model)
    base_url = local_judge_base_url()
    api_key = (
        os.environ.get("CORTEXT_JUDGE_API_KEY")
        or os.environ.get("LOCAL_JUDGE_API_KEY")
        or ""
    )
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You label retrieval targets for a memory system. Return only valid JSON. "
                    "Do not quote, paraphrase, name, or restate private message content in reasons."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if "max_completion_tokens" in detail:
            body["max_tokens"] = body.pop("max_completion_tokens")
            retry = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(retry, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
        else:
            raise RuntimeError(f"judge request failed: HTTP {exc.code}: {detail}") from exc
    content = payload["choices"][0]["message"]["content"]
    return parse_json_object(content, "judge")


def call_ollama_judge(
    model: str,
    base_url: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You label retrieval targets for a memory system. Return only valid JSON. "
                    "Do not quote, paraphrase, name, or restate private message content in reasons."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ollama judge request failed: HTTP {exc.code}: {detail}") from exc
    message = payload.get("message", {})
    content = message.get("content") or payload.get("response") or ""
    return parse_json_object(str(content), "ollama judge")


def call_target_judge(
    provider: str,
    model: str,
    base_url: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    if provider == "ollama":
        return call_ollama_judge(model, base_url, prompt, max_tokens)
    if provider == "nemotron":
        return call_nemotron_judge(model, prompt, max_tokens)
    raise RuntimeError(f"unsupported target judge provider: {provider}")


def judge_candidate_targets(
    timeline: list[Doc],
    doc_to_memory: dict[int, int],
    probe: dict,
    max_targets: int,
    min_overlap: float,
    max_candidates: int,
    judge_provider: str,
    judge_model: str,
    judge_base_url: str,
) -> list[dict]:
    event_index = int(probe["event_index"])
    if event_index <= 0 or event_index >= len(timeline):
        return []
    query_doc = timeline[event_index]
    query_tokens = set(tokens(query_doc.text))
    memory_to_doc = {
        memory_id: doc_index for doc_index, memory_id in doc_to_memory.items()
    }
    candidates: dict[int, tuple[float, set[str]]] = {}

    def add_candidate(score: float, doc_index: int, source: str) -> None:
        old = candidates.get(doc_index)
        if old is None:
            candidates[doc_index] = (score, {source})
            return
        old_score, sources = old
        sources.add(source)
        candidates[doc_index] = (max(old_score, score), sources)

    active_memory_ids = []
    for field in ("cortext_working_memory_ids", "cortext_retrieved_memory_ids"):
        active_memory_ids.extend(
            int(mid) for mid in probe.get(field, []) if mid is not None
        )
    for rank, memory_id in enumerate(dict.fromkeys(active_memory_ids), start=1):
        doc_index = memory_to_doc.get(memory_id)
        if doc_index is None or doc_index >= event_index:
            continue
        add_candidate(
            3.0 + 1.0 / max(1, rank),
            doc_index,
            "active_packet_prior_memory",
        )

    for rank, doc_index in enumerate(probe.get("rag_top_k_indices", []), start=1):
        if isinstance(doc_index, int) and 0 <= doc_index < event_index and doc_index in doc_to_memory:
            add_candidate(
                2.0 + 1.0 / max(1, rank),
                doc_index,
                "normal_rag_lexical_top_k",
            )

    recent_media_added = 0
    for doc in reversed(timeline[:event_index]):
        if doc.index not in doc_to_memory or doc.modality not in {"audio", "image", "video"}:
            continue
        add_candidate(
            2.75 + 0.25 / math.sqrt(max(1, event_index - doc.index)),
            doc.index,
            "recent_media_prior",
        )
        recent_media_added += 1
        if recent_media_added >= 3:
            break

    for doc in timeline[:event_index]:
        if doc.index not in doc_to_memory or doc.modality != "text":
            continue
        doc_tokens = set(tokens(doc.text))
        if not doc_tokens:
            continue
        overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens))
        if overlap < min_overlap:
            continue
        score = overlap + 0.25 / math.sqrt(max(1, event_index - doc.index))
        add_candidate(score, doc.index, "token_overlap_recency")

    ranked = sorted(candidates.items(), key=lambda row: (-row[1][0], -timeline[row[0]].timestamp))
    selected_ranked = ranked[:max_candidates]
    if (
        max_candidates > 0
        and not any(timeline[doc_index].modality in {"audio", "image", "video"} for doc_index, _ in selected_ranked)
    ):
        for item in ranked:
            doc_index = item[0]
            if timeline[doc_index].modality in {"audio", "image", "video"}:
                if len(selected_ranked) < max_candidates:
                    selected_ranked.append(item)
                else:
                    selected_ranked[-1] = item
                break
    if not selected_ranked:
        return []

    candidate_lines = []
    for rank, (doc_index, (score, sources)) in enumerate(selected_ranked, start=1):
        doc = timeline[doc_index]
        candidate_lines.append(
            f"C{rank}: event_index={doc.index} source_id={doc.source_id} "
            f"modality={doc.modality} timestamp={doc.timestamp} "
            f"heuristic={','.join(sorted(sources))} "
            f"score={score:.3f}\nTEXT: {doc.text}"
        )

    prompt = "\n\n".join([
        "Choose prior messages that should surface as memory for the current turn.",
        "Select only candidates that would materially help answer or contextualize the current turn.",
        "Prefer specific prior context over generic topical overlap. It is acceptable to return no targets.",
        "Return JSON: {\"targets\":[{\"candidate\":\"C1\",\"event_index\":123,\"relevance\":0-3,\"reason\":\"structural reason without private content\"}],\"notes\":\"optional structural note\"}.",
        "Relevance scale: 0 unrelated/noise, 1 weak/background, 2 useful, 3 important/direct.",
        f"CURRENT_TURN event_index={query_doc.index} source_id={query_doc.source_id} modality={query_doc.modality} timestamp={query_doc.timestamp}\nTEXT: {query_doc.text}",
        "CANDIDATES:\n" + "\n\n".join(candidate_lines),
    ])
    judged = call_target_judge(
        judge_provider,
        judge_model,
        judge_base_url,
        prompt,
        900,
    )
    by_candidate = {f"C{i}": doc_index for i, (doc_index, _) in enumerate(selected_ranked, start=1)}
    targets = []
    seen: set[int] = set()
    for item in judged.get("targets", []):
        if not isinstance(item, dict):
            continue
        candidate = str(item.get("candidate", ""))
        doc_index = item.get("event_index")
        if candidate in by_candidate:
            doc_index = by_candidate[candidate]
        if not isinstance(doc_index, int) or doc_index not in doc_to_memory or doc_index in seen:
            continue
        relevance = item.get("relevance", 0)
        try:
            relevance_f = float(relevance)
        except (TypeError, ValueError):
            relevance_f = 0.0
        if relevance_f < 2.0:
            continue
        seen.add(doc_index)
        doc = timeline[doc_index]
        targets.append({
            "memory_id": doc_to_memory[doc_index],
            "event_index": doc.index,
            "timestamp": doc.timestamp,
            "source_id": doc.source_id,
            "modality": doc.modality,
            "label_score": relevance_f,
            "query_token_overlap": (
                len(set(tokens(query_doc.text)) & set(tokens(doc.text)))
                / max(1, len(set(tokens(query_doc.text))))
            ),
            "label_sources": ["judge_assisted_candidate_selection"],
            "judge_reason": " ".join(str(item.get("reason", "")).split())[:240],
        })
        if len(targets) >= max_targets:
            break
    return targets


def wilson(successes: int, n: int, z: float = 1.96) -> dict:
    if n <= 0:
        return {"successes": successes, "n": n, "rate": 0.0, "low": 0.0, "high": 0.0}
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return {"successes": successes, "n": n, "rate": p, "low": max(0.0, center - half), "high": min(1.0, center + half)}


EventKey = tuple[str, str, int]


def event_key(source_id: Any, modality: Any, start_ts: Any) -> EventKey:
    try:
        ts = int(start_ts or 0)
    except (TypeError, ValueError):
        ts = 0
    return (str(source_id), str(modality), ts)


def fallback_memory_key(memory_id: int) -> EventKey:
    return ("__memory_id__", str(memory_id), int(memory_id))


def key_for_memory(memory_id: int, memory_event_key: dict[int, EventKey]) -> EventKey:
    return memory_event_key.get(memory_id, fallback_memory_key(memory_id))


def key_to_json(key: EventKey) -> dict:
    return {"source_id": key[0], "modality": key[1], "start_ts": key[2]}


def load_memory_event_key_map(conn: sqlite3.Connection) -> dict[int, EventKey]:
    rows = conn.execute("select memory_id, source_id, modality, start_ts from memories").fetchall()
    return {
        int(row["memory_id"]): event_key(row["source_id"], row["modality"], row["start_ts"])
        for row in rows
    }


def unique_event_keys(memory_ids: list[int], memory_event_key: dict[int, EventKey]) -> list[EventKey]:
    out: list[EventKey] = []
    seen: set[EventKey] = set()
    for memory_id in memory_ids:
        key = key_for_memory(memory_id, memory_event_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def ranks_for_targets(active_ids: list[int], target_ids: set[int]) -> list[int]:
    return [i + 1 for i, mid in enumerate(active_ids) if mid in target_ids]


def ranks_for_target_keys(
    active_ids: list[int],
    target_keys: set[EventKey],
    memory_event_key: dict[int, EventKey],
) -> list[int]:
    ranks: list[int] = []
    seen: set[EventKey] = set()
    for rank, memory_id in enumerate(active_ids, start=1):
        key = key_for_memory(memory_id, memory_event_key)
        if key not in target_keys or key in seen:
            continue
        ranks.append(rank)
        seen.add(key)
    return ranks


def target_keys_for_probe(
    fp: dict,
    timeline: list[Doc],
    doc_to_memory: dict[int, int],
    frozen_target_ids: set[int],
    memory_event_key: dict[int, EventKey],
    remap_targets: bool,
) -> tuple[set[EventKey], int]:
    keys: set[EventKey] = set()
    unmapped = 0
    for target in fp.get("target_memories", []):
        mapped = False
        raw_event_index = target.get("event_index")
        if raw_event_index is not None:
            try:
                event_index = int(raw_event_index)
            except (TypeError, ValueError):
                event_index = -1
            mapped_memory_id = doc_to_memory.get(event_index)
            if mapped_memory_id is not None and mapped_memory_id in memory_event_key:
                keys.add(memory_event_key[mapped_memory_id])
                mapped = True
            elif 0 <= event_index < len(timeline):
                doc = timeline[event_index]
                keys.add(event_key(doc.source_id, doc.modality, doc.timestamp))
                mapped = True
            if not mapped:
                unmapped += 1
        if mapped:
            continue
        if (
            target.get("source_id") is not None
            and target.get("modality") is not None
            and target.get("timestamp") is not None
        ):
            keys.add(event_key(target["source_id"], target["modality"], target["timestamp"]))
            continue
        if not remap_targets:
            memory_id = target.get("memory_id")
            try:
                memory_id_int = int(memory_id) if memory_id is not None else 0
            except (TypeError, ValueError):
                memory_id_int = 0
            if memory_id_int:
                keys.add(key_for_memory(memory_id_int, memory_event_key))
    if not keys and not remap_targets:
        keys.update(key_for_memory(mid, memory_event_key) for mid in frozen_target_ids)
    return keys, unmapped


def packet_hits(
    packet_ids: list[int],
    target_keys: set[EventKey],
    memory_event_key: dict[int, EventKey],
) -> tuple[int, int]:
    selected_keys = unique_event_keys(packet_ids, memory_event_key)
    return len(set(selected_keys) & target_keys), len(selected_keys)


def packet_metrics(
    packet_ids_by_probe: list[tuple[list[int], set[EventKey]]],
    memory_event_key: dict[int, EventKey],
) -> dict:
    hit = [0, 0]
    false_surface = [0, 0]
    mrr_values: list[float] = []
    for packet_ids, target_keys in packet_ids_by_probe:
        ranks = ranks_for_target_keys(packet_ids, target_keys, memory_event_key)
        first_rank = min(ranks) if ranks else None
        packet_keys = unique_event_keys(packet_ids, memory_event_key)
        hit[1] += 1
        hit[0] += int(first_rank is not None)
        false_surface[0] += sum(1 for key in packet_keys if key not in target_keys)
        false_surface[1] += len(packet_keys)
        mrr_values.append(0.0 if first_rank is None else 1.0 / first_rank)
    return {
        "hit_rate_at_packet_size": wilson(hit[0], hit[1]),
        "false_surface_rate": wilson(false_surface[0], false_surface[1]),
        "mrr": mean(mrr_values) if mrr_values else 0.0,
    }


def load_memory_kind_map(conn: sqlite3.Connection) -> dict[int, str]:
    rows = conn.execute("select memory_id, kind from memories").fetchall()
    return {int(row["memory_id"]): str(row["kind"]) for row in rows}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[idx]


def freeze(args: argparse.Namespace) -> int:
    summary = json.loads(args.summary.read_text())
    input_dir = args.input_dir or pathlib.Path(summary["input_dir"])
    db_path = args.db or pathlib.Path(summary["db_path"])
    skip_messages, max_messages, media_limit = timeline_args_from_summary(summary)
    timeline = build_timeline(
        input_dir,
        max_messages,
        media_limit,
        skip_messages,
    )
    conn = connect(db_path)
    doc_to_memory = load_memory_doc_map(conn, timeline)
    memory_to_doc = {memory_id: doc_index for doc_index, memory_id in doc_to_memory.items()}
    memory_kind = load_memory_kind_map(conn)
    label_sources = set(args.label_source)
    probes = []
    for probe in summary.get("probes", []):
        targets = label_targets(timeline, doc_to_memory, probe, args.max_targets, args.min_overlap, label_sources)
        if not targets and not args.keep_unanswerable:
            continue
        event_index = int(probe["event_index"])
        probes.append({
            "event_index": event_index,
            "query": {
                "timestamp": timeline[event_index].timestamp if event_index < len(timeline) else probe.get("query", {}).get("timestamp"),
                "source_id": timeline[event_index].source_id if event_index < len(timeline) else probe.get("query", {}).get("source_id"),
                "modality": timeline[event_index].modality if event_index < len(timeline) else probe.get("query", {}).get("modality"),
                "tokens": estimate_tokens(timeline[event_index].text) if event_index < len(timeline) else probe.get("query", {}).get("tokens", 0),
            },
            "target_memories": targets,
            "answerability": "self_labeled_targets" if targets else "no_self_labeled_target",
        })
    body = {
        "schema": "cortext_frozen_retrieval_probe_set_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_evaluation": True,
        "source_summary": str(args.summary),
        "source_summary_sha256": sha256_file(args.summary),
        "db_path": str(db_path),
        "input_dir": str(input_dir),
        "labeling": {
            "method": "deterministic self-labeling from prior source timeline before scoring retrieval",
            "sources": sorted(label_sources),
            "source_definitions": {
                "overlap": "prior text memory with query-token overlap above threshold plus recency/source bonus",
                "rag": "normal chat+vector/text RAG top-k document indices from the frozen benchmark summary, mapped back to persisted Cortext memory IDs",
            },
            "max_targets": args.max_targets,
            "min_query_token_overlap": args.min_overlap,
            "limitations": [
                "self-labeled relevance, not independent human annotation",
                "token-overlap and vector/text RAG labels underlabel paraphrase, relationship context, and media-only relevance",
                "RAG-derived labels are useful for regression pressure but are not independent of the baseline retrieval policy",
                "single corpus, single relationship, single writing style",
            ],
        },
        "probe_count": len(probes),
        "probes": probes,
    }
    body["freeze_sha256"] = canonical_hash({k: v for k, v in body.items() if k != "freeze_sha256"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2) + "\n")
    print(args.out)
    return 0


def judge_freeze(args: argparse.Namespace) -> int:
    summary = json.loads(args.summary.read_text())
    input_dir = args.input_dir or pathlib.Path(summary["input_dir"])
    db_path = args.db or pathlib.Path(summary["db_path"])
    skip_messages, max_messages, media_limit = timeline_args_from_summary(summary)
    timeline = build_timeline(
        input_dir,
        max_messages,
        media_limit,
        skip_messages,
    )
    conn = connect(db_path)
    doc_to_memory = load_memory_doc_map(conn, timeline)
    if args.model:
        os.environ["CORTEXT_TARGET_JUDGE_MODEL"] = args.model
    judge_provider = args.judge_provider
    if judge_provider == "ollama":
        judge_model = (
            args.model
            or os.environ.get("CORTEXT_TARGET_JUDGE_MODEL")
            or os.environ.get("LOCAL_JUDGE_MODEL")
            or DEFAULT_OLLAMA_MODEL
        )
        judge_base_url = local_ollama_base_url(args.ollama_base_url)
        judge_provider_label = "local_ollama"
    else:
        judge_model = (
            args.model
            or os.environ.get("CORTEXT_TARGET_JUDGE_MODEL")
            or os.environ.get("LOCAL_JUDGE_MODEL")
            or "nemotron-3-nano-omni-30b-a3b-8bit"
        )
        require_nemotron_model(judge_model)
        judge_base_url = local_judge_base_url()
        judge_provider_label = "local_nemotron_vllm_mlx"
    probes = []
    for probe in summary.get("probes", []):
        targets = judge_candidate_targets(
            timeline,
            doc_to_memory,
            probe,
            args.max_targets,
            args.min_overlap,
            args.max_candidates,
            judge_provider,
            judge_model,
            judge_base_url,
        )
        if not targets and not args.keep_unanswerable:
            continue
        event_index = int(probe["event_index"])
        probes.append({
            "event_index": event_index,
            "query": {
                "timestamp": timeline[event_index].timestamp if event_index < len(timeline) else probe.get("query", {}).get("timestamp"),
                "source_id": timeline[event_index].source_id if event_index < len(timeline) else probe.get("query", {}).get("source_id"),
                "modality": timeline[event_index].modality if event_index < len(timeline) else probe.get("query", {}).get("modality"),
                "tokens": estimate_tokens(timeline[event_index].text) if event_index < len(timeline) else probe.get("query", {}).get("tokens", 0),
            },
            "target_memories": targets,
            "answerability": "judge_assisted_targets" if targets else "no_judge_labeled_target",
        })
    body = {
        "schema": "cortext_frozen_retrieval_probe_set_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_evaluation": True,
        "source_summary": str(args.summary),
        "source_summary_sha256": sha256_file(args.summary),
        "db_path": str(db_path),
        "input_dir": str(input_dir),
        "labeling": {
            "method": "judge-assisted frozen target selection from a prior-message candidate pool",
            "judge_provider": judge_provider_label,
            "judge_model": judge_model,
            "judge_base_url": judge_base_url,
            "remote_provider_allowed": False,
            "candidate_sources": [
                "active packet prior-memory candidates from the frozen summary",
                "normal chat+vector/text RAG top-k document indices from the frozen benchmark summary",
                "prior text memories with query-token overlap above threshold plus recency",
                "recent prior audio/image/video source-blob memories",
            ],
            "selection_rule": "judge keeps prior-message candidates with relevance >= 2 on a 0-3 scale before any scoring evaluation",
            "max_targets": args.max_targets,
            "min_query_token_overlap": args.min_overlap,
            "max_candidates_per_probe": args.max_candidates,
            "limitations": [
                "judge-assisted relevance, not independent human annotation",
                "candidate generation can underlabel paraphrase, relationship context, and media-only relevance",
                "media target freezing includes source-blob markers but does not attach raw media bytes in this helper",
                "RAG-derived candidates are useful for regression pressure but are not independent of the baseline retrieval policy",
                "single corpus, single relationship, single writing style",
                "private source text was sent only to the configured loopback judge endpoint during label freezing",
            ],
        },
        "probe_count": len(probes),
        "probes": probes,
    }
    body["freeze_sha256"] = canonical_hash({k: v for k, v in body.items() if k != "freeze_sha256"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2) + "\n")
    print(args.out)
    return 0


def evaluate(args: argparse.Namespace) -> int:
    frozen = json.loads(args.frozen.read_text())
    summary = json.loads(args.summary.read_text())
    judge = json.loads(args.judge.read_text()) if args.judge else {}
    input_dir = args.input_dir or pathlib.Path(summary.get("input_dir") or frozen.get("input_dir"))
    db_path = args.db or pathlib.Path(summary.get("db_path") or frozen.get("db_path"))
    skip_messages, max_messages, media_limit = timeline_args_from_summary(summary)
    timeline = build_timeline(
        input_dir,
        max_messages,
        media_limit,
        skip_messages,
    )
    conn = connect(db_path)
    doc_to_memory = load_memory_doc_map(conn, timeline)
    memory_to_doc = {memory_id: doc_index for doc_index, memory_id in doc_to_memory.items()}
    memory_kind = load_memory_kind_map(conn)
    memory_event_key = load_memory_event_key_map(conn)
    by_event = {int(p["event_index"]): p for p in summary.get("probes", [])}
    judge_by_event = {
        int(p["event_index"]): p for p in judge.get("judgments", [])
    }
    rows = []
    ks = [1, 3, 5]
    counters = {f"hit@{k}": [0, 0] for k in ks}
    precision_counts = {k: [0, 0] for k in ks}
    recall_counts = {k: [0, 0] for k in ks}
    active_hit = [0, 0]
    false_surface = [0, 0]
    active_precision = [0, 0]
    active_recall = [0, 0]
    mrr_values: list[float] = []
    exact_active_hit = [0, 0]
    exact_active_recall = [0, 0]
    exact_mrr_values: list[float] = []
    cortext_tokens: list[float] = []
    rag_tokens: list[float] = []
    latencies: list[float] = []
    working_only_packets: list[tuple[list[int], set[EventKey]]] = []
    retrieved_only_packets: list[tuple[list[int], set[EventKey]]] = []
    durable_ltm_only_packets: list[tuple[list[int], set[EventKey]]] = []
    association_only_packets: list[tuple[list[int], set[EventKey]]] = []
    working_only_tokens: list[float] = []
    retrieved_only_tokens: list[float] = []
    unmapped_targets = 0
    consolidation_loss_counts = {
        "missed_target_count": 0,
        "gist_preserving_proxy": 0,
        "random_loss_proxy": 0,
    }
    consolidation_loss_examples: list[dict] = []

    for fp in frozen.get("probes", []):
        probe = by_event.get(int(fp["event_index"]))
        if not probe:
            continue
        frozen_target_ids = {int(t["memory_id"]) for t in fp.get("target_memories", []) if t.get("memory_id")}
        target_event_indices = [int(t["event_index"]) for t in fp.get("target_memories", []) if t.get("event_index") is not None]
        target_ids = {doc_to_memory[idx] for idx in target_event_indices if idx in doc_to_memory}
        target_keys, target_unmapped = target_keys_for_probe(
            fp,
            timeline,
            doc_to_memory,
            frozen_target_ids,
            memory_event_key,
            args.remap_targets,
        )
        unmapped_targets += target_unmapped
        if not target_ids and not args.remap_targets:
            target_ids = frozen_target_ids
        if not target_keys:
            continue
        working_ids = [int(x) for x in probe.get("cortext_working_memory_ids", [])]
        retrieved_ids = [int(x) for x in probe.get("cortext_retrieved_memory_ids", [])]
        durable_ltm_ids = [mid for mid in retrieved_ids if memory_kind.get(mid) == "LONG_TERM"]
        association_ids = [mid for mid in retrieved_ids if memory_kind.get(mid) == "ASSOCIATION"]
        active_ids = working_ids + retrieved_ids
        active_size = len(active_ids)
        active_keys = unique_event_keys(active_ids, memory_event_key)
        ranks = ranks_for_target_keys(active_ids, target_keys, memory_event_key)
        first_rank = min(ranks) if ranks else None
        exact_ranks = ranks_for_targets(active_ids, target_ids)
        exact_first_rank = min(exact_ranks) if exact_ranks else None
        active_doc_indices = [memory_to_doc[mid] for mid in active_ids if mid in memory_to_doc]
        active_key_set = set(active_keys)
        missed_target_events = []
        for idx in sorted(set(target_event_indices)):
            key = None
            mapped_memory_id = doc_to_memory.get(idx)
            if mapped_memory_id is not None and mapped_memory_id in memory_event_key:
                key = memory_event_key[mapped_memory_id]
            elif 0 <= idx < len(timeline):
                doc = timeline[idx]
                key = event_key(doc.source_id, doc.modality, doc.timestamp)
            if key is not None and key not in active_key_set:
                missed_target_events.append(idx)
        for target_idx in missed_target_events:
            if target_idx < 0 or target_idx >= len(timeline):
                continue
            target_doc = timeline[target_idx]
            target_tokens = set(tokens(target_doc.text))
            gist_like = False
            nearest_active_distance = None
            max_overlap = 0.0
            for active_idx in active_doc_indices:
                if active_idx < 0 or active_idx >= len(timeline):
                    continue
                active_doc = timeline[active_idx]
                distance = abs(active_idx - target_idx)
                if nearest_active_distance is None or distance < nearest_active_distance:
                    nearest_active_distance = distance
                if target_tokens:
                    active_tokens = set(tokens(active_doc.text))
                    overlap = len(target_tokens & active_tokens) / max(1, len(target_tokens))
                    max_overlap = max(max_overlap, overlap)
                else:
                    overlap = 0.0
                if (
                    active_doc.source_id == target_doc.source_id
                    and active_doc.modality == target_doc.modality
                    and (distance <= 3 or overlap >= 0.20)
                ):
                    gist_like = True
            consolidation_loss_counts["missed_target_count"] += 1
            if gist_like:
                consolidation_loss_counts["gist_preserving_proxy"] += 1
            else:
                consolidation_loss_counts["random_loss_proxy"] += 1
            if len(consolidation_loss_examples) < 25:
                consolidation_loss_examples.append({
                    "probe_event_index": int(fp["event_index"]),
                    "target_event_index": target_idx,
                    "target_memory_id": doc_to_memory.get(target_idx),
                    "classification": "gist_preserving_proxy" if gist_like else "random_loss_proxy",
                    "nearest_active_event_distance": nearest_active_distance,
                    "max_query_token_overlap_with_active_packet": max_overlap,
                })
        active_hit[1] += 1
        active_hit[0] += int(first_rank is not None)
        exact_active_hit[1] += 1
        exact_active_hit[0] += int(exact_first_rank is not None)
        active_target_hits = len(active_key_set & target_keys)
        exact_active_target_hits = len(set(active_ids) & target_ids)
        active_precision[0] += active_target_hits
        active_precision[1] += max(1, len(active_keys))
        active_recall[0] += active_target_hits
        active_recall[1] += len(target_keys)
        exact_active_recall[0] += exact_active_target_hits
        exact_active_recall[1] += len(target_ids)
        false_count = sum(1 for key in active_keys if key not in target_keys)
        false_surface[0] += false_count
        false_surface[1] += len(active_keys)
        mrr_values.append(0.0 if first_rank is None else 1.0 / first_rank)
        exact_mrr_values.append(0.0 if exact_first_rank is None else 1.0 / exact_first_rank)
        for k in ks:
            selected = active_ids[:k]
            hits, selected_key_count = packet_hits(selected, target_keys, memory_event_key)
            counters[f"hit@{k}"][1] += 1
            counters[f"hit@{k}"][0] += int(hits > 0)
            precision_counts[k][0] += hits
            precision_counts[k][1] += max(1, selected_key_count)
            recall_counts[k][0] += hits
            recall_counts[k][1] += len(target_keys)
        ctoks = int(probe.get("cortext_context_tokens", 0) or 0)
        if ctoks == 0:
            ctoks = int(probe.get("cortext_working_tokens", 0) or 0) + int(probe.get("cortext_retrieved_tokens", 0) or 0)
        rtoks = int(
            probe.get(
                "normal_rag_context_tokens",
                probe.get(
                    "normal_rag_active_history_tokens",
                    probe.get("rolling_history_tokens", 0),
                ),
            )
            or 0
        )
        judged_probe = judge_by_event.get(int(fp["event_index"]), {})
        if ctoks == 0 and judged_probe:
            ctoks = int(judged_probe.get("cortext_context_tokens", 0) or 0)
        if rtoks == 0 and judged_probe:
            rtoks = int(judged_probe.get("traditional_chat_rag_tokens", 0) or 0)
        cortext_tokens.append(ctoks)
        rag_tokens.append(rtoks)
        latencies.append(float(probe.get("cortext_total_ms", probe.get("cortext_latency_ms", 0.0)) or 0.0))
        working_only_packets.append((working_ids, target_keys))
        retrieved_only_packets.append((retrieved_ids, target_keys))
        durable_ltm_only_packets.append((durable_ltm_ids, target_keys))
        association_only_packets.append((association_ids, target_keys))
        working_only_tokens.append(float(probe.get("cortext_working_tokens", 0) or 0))
        retrieved_only_tokens.append(float(probe.get("cortext_retrieved_tokens", 0) or 0))
        exact_present_ids = [mid for mid in active_ids if mid in target_ids]
        alias_hits = [
            {
                "rank": rank,
                "memory_id": mid,
                "kind": memory_kind.get(mid, ""),
                "event_key": key_to_json(key_for_memory(mid, memory_event_key)),
            }
            for rank, mid in enumerate(active_ids, start=1)
            if key_for_memory(mid, memory_event_key) in target_keys and mid not in target_ids
        ]
        rows.append({
            "event_index": int(fp["event_index"]),
            "target_memory_ids": sorted(target_ids),
            "frozen_source_target_memory_ids": sorted(frozen_target_ids),
            "target_event_indices": sorted(set(target_event_indices)),
            "target_event_keys": [key_to_json(key) for key in sorted(target_keys)],
            "working_memory_ids": working_ids,
            "retrieved_memory_ids": retrieved_ids,
            "durable_ltm_memory_ids": durable_ltm_ids,
            "association_memory_ids": association_ids,
            "active_memory_ids": active_ids,
            "active_event_keys": [key_to_json(key) for key in active_keys],
            "active_packet_size": active_size,
            "active_event_key_count": len(active_keys),
            "active_target_event_key_hits": active_target_hits,
            "first_target_rank": first_rank,
            "hit_active_packet": first_rank is not None,
            "first_exact_target_rank": exact_first_rank,
            "exact_hit_active_packet": exact_first_rank is not None,
            "exact_target_memory_ids_present": exact_present_ids,
            "active_target_memory_id_aliases": alias_hits,
            "cortext_context_tokens": ctoks,
            "traditional_chat_rag_tokens": rtoks,
        })

    working_curve = summary.get("working_set_curve", [])
    curve_tokens = [float(r.get("active_context_tokens", 0)) for r in working_curve]
    latency_all = [float(r.get("latency_ms", 0.0)) for r in working_curve] or latencies
    events = int(summary.get("processed_text_messages", 0) or 0) + int(summary.get("media_processed", 0) or 0)
    wall_ex = float(summary.get("wall_ms_excluding_consolidation", summary.get("wall_ms", 0)) or 0.0)
    consolidation_ms = float(summary.get("consolidation_ms_total", 0.0) or 0.0)
    mean_c = mean(cortext_tokens) if cortext_tokens else 0.0
    mean_r = mean(rag_tokens) if rag_tokens else 0.0
    hit_active_ci = wilson(active_hit[0], active_hit[1])
    exact_hit_active_ci = wilson(exact_active_hit[0], exact_active_hit[1])
    active_recall_ci = wilson(active_recall[0], active_recall[1])
    false_surface_ci = wilson(false_surface[0], false_surface[1])
    token_reduction = 1.0 - (mean_c / mean_r) if mean_r > 0 else 0.0
    summary_lines = [
        f"Frozen retrieval eval n={len(rows)} using freeze {frozen.get('freeze_sha256')}.",
        (
            "Labeled retrieval: "
            f"hit-rate-at-active-packet-size={hit_active_ci['rate']:.3f} "
            f"(Wilson 95% CI {hit_active_ci['low']:.3f}-{hit_active_ci['high']:.3f}); "
            f"target-recall-at-active-packet-size={active_recall_ci['rate']:.3f} "
            f"({active_recall_ci['successes']}/{active_recall_ci['n']}); "
            f"MRR={mean(mrr_values) if mrr_values else 0.0:.3f}; "
            f"false-surface-rate={false_surface_ci['rate']:.3f} "
            f"(Wilson 95% CI {false_surface_ci['low']:.3f}-{false_surface_ci['high']:.3f})."
        ),
        (
            "Mechanical/token: "
            f"mean Cortext active tokens={mean_c:.1f}, "
            f"mean traditional chat+RAG tokens={mean_r:.1f}, "
            f"paired token reduction={token_reduction:.3f}, "
            f"throughput={events / (wall_ex / 1000.0) if wall_ex > 0 else 0.0:.3f} events/s excluding consolidation."
        ),
        (
            "Limitations: "
            + "; ".join(frozen.get("labeling", {}).get("limitations", []))
        ),
    ]
    output = {
        "schema": "cortext_frozen_retrieval_eval_v1",
        "summary_path": str(args.summary),
        "db_path": str(db_path),
        "input_dir": str(input_dir),
        "judge_path": str(args.judge) if args.judge else None,
        "frozen_probe_set": str(args.frozen),
        "freeze_sha256": frozen.get("freeze_sha256"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(rows),
        "limitations": frozen.get("labeling", {}).get("limitations", []),
        "target_mapping": {
            "mode": "source_event_to_current_db_memory" if args.remap_targets else "source_event_to_current_db_memory_with_frozen_memory_id_fallback",
            "unmapped_target_count": unmapped_targets,
            "scoring_basis": "canonical_source_event_key",
            "canonical_source_event_key": ["source_id", "modality", "start_ts"],
            "note": "Frozen labels store source event targets; primary retrieval scoring treats memories with the same (source_id, modality, start_ts) as equivalent, so WORKING and LONG_TERM twins of the same source event are not false misses. Exact memory-ID scoring is retained only as a diagnostic.",
        },
        "short_summary": "\n".join(summary_lines),
        "retrieval_metrics": {
            **{name: wilson(v[0], v[1]) for name, v in counters.items()},
            **{f"precision@{k}": wilson(v[0], v[1]) for k, v in precision_counts.items()},
            **{f"recall@{k}": wilson(v[0], v[1]) for k, v in recall_counts.items()},
            "hit_rate_at_active_packet_size": hit_active_ci,
            "precision_at_active_packet_size": wilson(active_precision[0], active_precision[1]),
            "recall_at_active_packet_size": wilson(active_recall[0], active_recall[1]),
            "false_surface_rate": false_surface_ci,
            "mrr": mean(mrr_values) if mrr_values else 0.0,
        },
        "exact_memory_id_diagnostics": {
            "hit_rate_at_active_packet_size": exact_hit_active_ci,
            "recall_at_active_packet_size": wilson(exact_active_recall[0], exact_active_recall[1]),
            "mrr": mean(exact_mrr_values) if exact_mrr_values else 0.0,
            "note": "Diagnostic only. Exact memory IDs can undercount valid retrieval when the packet surfaces a WORKING or LONG_TERM twin with the same canonical source-event key.",
        },
        "token_vs_quality": {
            "paired_probe_count": len(rows),
            "mean_cortext_active_tokens": mean_c,
            "median_cortext_active_tokens": median(cortext_tokens) if cortext_tokens else 0.0,
            "mean_traditional_chat_rag_tokens": mean_r,
            "token_reduction": token_reduction,
            "paired_quality_metric": "hit_rate_at_active_packet_size",
            "paired_quality": hit_active_ci,
        },
        "working_set_curve": {
            "policy": summary.get("working_set_curve_policy", "missing_full_curve"),
            "has_full_timeline_curve": len(working_curve) == events and events > 0,
            "points": working_curve,
            "active_context_tokens": {
                "n": len(curve_tokens),
                "mean": mean(curve_tokens) if curve_tokens else 0.0,
                "median": median(curve_tokens) if curve_tokens else 0.0,
                "p95": percentile(curve_tokens, 0.95),
                "max": max(curve_tokens) if curve_tokens else 0.0,
            },
        },
        "latency_throughput": {
            "events": events,
            "latency_ms": {
                "n": len(latency_all),
                "p50": percentile(latency_all, 0.50),
                "p95": percentile(latency_all, 0.95),
                "p99": percentile(latency_all, 0.99),
            },
            "throughput_events_per_second_excluding_consolidation": events / (wall_ex / 1000.0) if wall_ex > 0 else 0.0,
            "consolidation_ms_total": consolidation_ms,
            "consolidation_ms_amortized_per_event": consolidation_ms / events if events else 0.0,
        },
        "consolidation_loss_profile": {
            "status": "proxy_measured_not_human_judged",
            "scope": "missed frozen detail targets after native daily consolidation",
            "method": "For each missed frozen source-event target, classify as gist_preserving_proxy when the active packet contains same-source/same-modality context within three timeline events or >=0.20 token overlap; otherwise classify as random_loss_proxy.",
            "limitations": [
                "This is not an independent human judgment of detail-only answerability.",
                "It cannot prove semantic gist preservation for paraphrases or media.",
                "It is a regression diagnostic for consolidation/retrieval loss shape, not a final quality claim.",
            ],
            "counts": consolidation_loss_counts,
            "gist_preserving_fraction_of_missed": (
                consolidation_loss_counts["gist_preserving_proxy"]
                / consolidation_loss_counts["missed_target_count"]
                if consolidation_loss_counts["missed_target_count"] else 0.0
            ),
            "examples": consolidation_loss_examples,
        },
        "ablations": {
            "status": "partial_posthoc_plus_native_stm_available",
            "posthoc_packet_ablation": {
                "note": "This does not rerun Cortext; it removes packet components after native retrieval. Native STM shadow drop must be measured by rerunning with CORTEXT_STM_SHADOW_DISABLE=1.",
                "drop_ltm_keep_working_memory_only": {
                    **packet_metrics(working_only_packets, memory_event_key),
                    "mean_tokens": mean(working_only_tokens) if working_only_tokens else 0.0,
                },
                "drop_wm_keep_retrieved_memory_only": {
                    **packet_metrics(retrieved_only_packets, memory_event_key),
                    "mean_tokens": mean(retrieved_only_tokens) if retrieved_only_tokens else 0.0,
                },
                "durable_ltm_only": packet_metrics(durable_ltm_only_packets, memory_event_key),
                "association_only": packet_metrics(association_only_packets, memory_event_key),
            },
            "native_stm_drop_instruction": "Run the benchmark with CORTEXT_STM_SHADOW_DISABLE=1 and evaluate against the same frozen source-event target set.",
            "missing_required_native_ablations": ["native_drop_ltm"],
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(args.out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    freeze_p = sub.add_parser("freeze")
    freeze_p.add_argument("--summary", type=pathlib.Path, required=True)
    freeze_p.add_argument("--db", type=pathlib.Path)
    freeze_p.add_argument("--input-dir", type=pathlib.Path)
    freeze_p.add_argument("--out", type=pathlib.Path, required=True)
    freeze_p.add_argument("--max-targets", type=int, default=3)
    freeze_p.add_argument("--min-overlap", type=float, default=0.34)
    freeze_p.add_argument(
        "--label-source",
        action="append",
        choices=["overlap", "rag"],
        default=["overlap"],
        help="self-label source to include; may be repeated",
    )
    freeze_p.add_argument("--keep-unanswerable", action="store_true")
    freeze_p.set_defaults(func=freeze)

    judge_freeze_p = sub.add_parser("judge-freeze")
    judge_freeze_p.add_argument("--summary", type=pathlib.Path, required=True)
    judge_freeze_p.add_argument("--db", type=pathlib.Path)
    judge_freeze_p.add_argument("--input-dir", type=pathlib.Path)
    judge_freeze_p.add_argument("--out", type=pathlib.Path, required=True)
    judge_freeze_p.add_argument("--max-targets", type=int, default=3)
    judge_freeze_p.add_argument("--min-overlap", type=float, default=0.20)
    judge_freeze_p.add_argument("--max-candidates", type=int, default=12)
    judge_freeze_p.add_argument(
        "--judge-provider",
        choices=("nemotron", "ollama"),
        default="ollama",
        help="Loopback target-label judge provider for private Julie labels.",
    )
    judge_freeze_p.add_argument("--model", default="")
    judge_freeze_p.add_argument(
        "--ollama-base-url",
        help="Loopback Ollama base URL, for example http://127.0.0.1:11434.",
    )
    judge_freeze_p.add_argument("--keep-unanswerable", action="store_true")
    judge_freeze_p.set_defaults(func=judge_freeze)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--summary", type=pathlib.Path, required=True)
    eval_p.add_argument("--frozen", type=pathlib.Path, required=True)
    eval_p.add_argument("--judge", type=pathlib.Path)
    eval_p.add_argument("--db", type=pathlib.Path)
    eval_p.add_argument("--input-dir", type=pathlib.Path)
    eval_p.add_argument(
        "--no-remap-targets",
        dest="remap_targets",
        action="store_false",
        help="fall back to frozen memory IDs when source-event remapping is unavailable",
    )
    eval_p.set_defaults(remap_targets=True)
    eval_p.add_argument("--out", type=pathlib.Path, required=True)
    eval_p.set_defaults(func=evaluate)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
