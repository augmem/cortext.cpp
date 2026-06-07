#!/usr/bin/env python3
"""Redact private chat text and prune private media with local Nemotron.

The default mode is non-destructive: it writes a redacted transcript, copies only
media classified as safe to the output tree, and records decisions in a manifest.
Use --apply-delete only when you intentionally want private source media removed.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import mimetypes
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_INPUT = pathlib.Path.home() / "Documents/Memory/Julie"
DEFAULT_MODEL = "nemotron-3-nano-omni-30b-a3b-8bit"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
LOCAL_JUDGE_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
TEXT_BATCH_SIZE = 16
DEFAULT_MEDIA_LIMIT = -1


@dataclass
class Message:
    index: int
    header: str
    timestamp: int | None
    text: str


@dataclass
class MediaItem:
    index: int
    path: pathlib.Path
    rel_path: pathlib.Path
    kind: str
    timestamp: int | None


def starts_with_date(line: str) -> bool:
    return bool(re.match(r"^\d{4}-\d{2}-\d{2} \d{2}[ :]\d{2}[ :]\d{2}", line))


def parse_timestamp(header: str) -> int | None:
    stamp = header[:19]
    if len(stamp) >= 17 and stamp[13] == " " and stamp[16] == " ":
        stamp = f"{stamp[:13]}:{stamp[14:16]}:{stamp[17:]}"
    try:
        parsed = dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


def parse_messages(path: pathlib.Path) -> list[Message]:
    messages: list[Message] = []
    pending_header = ""
    pending_text: list[str] = []

    def flush() -> None:
        nonlocal pending_header, pending_text
        if not pending_header:
            return
        text = "\n".join(pending_text).strip()
        if text:
            messages.append(
                Message(
                    index=len(messages),
                    header=pending_header,
                    timestamp=parse_timestamp(pending_header),
                    text=text,
                )
            )
        pending_header = ""
        pending_text = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("----------------------------------------------------"):
                flush()
                continue
            if starts_with_date(line):
                flush()
                pending_header = line
                continue
            if pending_header:
                pending_text.append(line)
    flush()
    return messages


def media_kind(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".heic", ".webp"}:
        return "image"
    if ext in {".mov", ".mp4", ".m4v", ".avi", ".webm"}:
        return "video"
    if ext in {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg"}:
        return "audio"
    return "other"


def discover_media(input_dir: pathlib.Path, transcript: pathlib.Path) -> list[MediaItem]:
    items: list[MediaItem] = []
    transcript_abs = transcript.resolve()
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.resolve() == transcript_abs:
                continue
        except OSError:
            pass
        kind = media_kind(path)
        if kind not in {"image", "video", "audio"}:
            continue
        rel_path = path.relative_to(input_dir)
        items.append(
            MediaItem(
                index=len(items),
                path=path,
                rel_path=rel_path,
                kind=kind,
                timestamp=parse_timestamp(path.name),
            )
        )
    return items


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_name(path: pathlib.Path) -> str:
    value = str(path)
    if len(value) <= 140:
        return value
    return "..." + value[-137:]


def load_env_file(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def require_nemotron_model(model: str) -> None:
    if "nemotron" not in model.lower():
        raise RuntimeError(
            f"Refusing non-Nemotron model for private redaction data: {model!r}. "
            "Start the local Nemotron/MLX server and pass --model nemotron..."
        )


def local_nemotron_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOCAL_JUDGE_HOSTS:
        raise RuntimeError(
            "Refusing non-local Nemotron endpoint for private redaction data: "
            f"{base_url!r}. Set --base-url or CORTEXT_JUDGE_BASE_URL to a loopback URL."
        )
    return base_url


def call_nemotron(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    require_nemotron_model(model)
    base_url = local_nemotron_base_url(base_url)
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Nemotron request failed: HTTP {exc.code}: {detail}") from exc
    content = payload["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise RuntimeError(f"Nemotron returned non-JSON content: {content[:500]!r}")


def text_prompt(batch: list[Message]) -> str:
    records = []
    for message in batch:
        records.append(
            {
                "id": message.index,
                "timestamp": message.timestamp,
                "text": message.text,
            }
        )
    return "\n".join(
        [
            "Redact private information from these chat messages.",
            "Return strict JSON with key messages, an array of objects.",
            "Each object must contain id, action, redacted_text, categories, confidence, and reason.",
            "Allowed action values: keep, redact, drop.",
            "Use drop only for messages whose remaining content would be mostly private or unsafe.",
            "Redact names of non-public people, phone numbers, emails, home/work/school addresses, precise locations, account identifiers, financial data, medical data, legal data, secrets, passwords, access tokens, intimate content, and child-related identifiers.",
            "Keep useful non-private conversational structure. Replace private spans with bracketed tags like [PERSON], [PHONE], [ADDRESS], [PRIVATE_DETAIL], [MEDICAL], [FINANCIAL], [SECRET], [CHILD].",
            "Do not quote private original text in reason.",
            json.dumps({"messages": records}, ensure_ascii=False),
        ]
    )


def normalize_text_decisions(
    batch: list[Message], response: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    raw_messages = response.get("messages", [])
    if not isinstance(raw_messages, list):
        raw_messages = []
    by_id: dict[int, dict[str, Any]] = {}
    for row in raw_messages:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        action = str(row.get("action", "redact")).lower()
        if action not in {"keep", "redact", "drop"}:
            action = "redact"
        redacted = str(row.get("redacted_text", "")).strip()
        if action != "drop" and not redacted:
            action = "drop"
        categories = row.get("categories", [])
        if not isinstance(categories, list):
            categories = [str(categories)]
        by_id[idx] = {
            "id": idx,
            "action": action,
            "redacted_text": redacted,
            "categories": [str(item) for item in categories],
            "confidence": safe_float(row.get("confidence"), 0.0),
            "reason": str(row.get("reason", ""))[:300],
        }
    for message in batch:
        by_id.setdefault(
            message.index,
            {
                "id": message.index,
                "action": "drop",
                "redacted_text": "",
                "categories": ["model_missing_decision"],
                "confidence": 0.0,
                "reason": "No model decision was returned for this message.",
            },
        )
    return by_id


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def media_policy_prompt(item: MediaItem, include_binary: bool) -> str:
    return "\n".join(
        [
            "Classify whether this personal chat media file should be kept in a sanitized research/evaluation corpus.",
            "Return strict JSON with keys action, categories, confidence, reason.",
            "Allowed action values: keep, prune.",
            "Prune private, intimate, identifying, location-revealing, child-related, medical, financial, legal, credential-bearing, document/screenshot, or face/person media unless the content is clearly non-private and non-identifying.",
            "For audio from a private conversation, prefer prune unless it is clearly synthetic or non-personal ambient audio.",
            "For video, classify from the supplied frame/metadata; prefer prune when uncertain.",
            "Do not quote any private text seen in the media.",
            json.dumps(
                {
                    "media": {
                        "id": item.index,
                        "relative_path": str(item.rel_path),
                        "kind": item.kind,
                        "timestamp": item.timestamp,
                        "binary_supplied": include_binary,
                    }
                }
            ),
        ]
    )


def media_content_part(path: pathlib.Path) -> dict[str, Any] | None:
    mime = mimetypes.guess_type(path.name)[0]
    if not mime or not mime.startswith("image/"):
        return None
    data_url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    return {"type": "image_url", "image_url": {"url": data_url}}


def extract_video_frame(path: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path | None:
    frame = out_dir / f"{path.stem}_privacy_frame.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-frames:v",
        "1",
        str(frame),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return frame if frame.exists() else None


def classify_media(
    item: MediaItem,
    *,
    base_url: str,
    model: str,
    timeout: int,
    temp_dir: pathlib.Path,
    multimodal: bool,
) -> dict[str, Any]:
    probe_path = item.path
    include_binary = False
    if item.kind == "video" and multimodal:
        frame = extract_video_frame(item.path, temp_dir)
        if frame is not None:
            probe_path = frame
    content: list[dict[str, Any]] = [{"type": "text", "text": media_policy_prompt(item, False)}]
    if multimodal and item.kind in {"image", "video"}:
        part = media_content_part(probe_path)
        if part is not None:
            include_binary = True
            content[0]["text"] = media_policy_prompt(item, True)
            content.append(part)
    if item.kind == "audio":
        # Current local Nemotron audio payload support varies by model build.
        # Use a conservative metadata policy instead of failing.
        include_binary = False
    messages = [
        {
            "role": "system",
            "content": "You are a strict privacy classifier. Return only valid JSON.",
        },
        {"role": "user", "content": content},
    ]
    response = call_nemotron(
        base_url=base_url,
        model=model,
        messages=messages,
        max_tokens=500,
        timeout=timeout,
    )
    action = str(response.get("action", "prune")).lower()
    if action not in {"keep", "prune"}:
        action = "prune"
    categories = response.get("categories", [])
    if not isinstance(categories, list):
        categories = [str(categories)]
    return {
        "id": item.index,
        "relative_path": str(item.rel_path),
        "kind": item.kind,
        "sha256": sha256_file(item.path),
        "action": action,
        "categories": [str(value) for value in categories],
        "confidence": safe_float(response.get("confidence"), 0.0),
        "reason": str(response.get("reason", ""))[:300],
        "binary_supplied": include_binary,
    }


def write_redacted_transcript(
    out_path: pathlib.Path,
    messages: list[Message],
    decisions: dict[int, dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for message in messages:
            decision = decisions[message.index]
            if decision["action"] == "drop":
                continue
            handle.write(f"{message.header}\n")
            handle.write(f"{decision['redacted_text'].strip()}\n")
            handle.write("----------------------------------------------------\n")


def copy_safe_media(
    input_dir: pathlib.Path,
    out_dir: pathlib.Path,
    decisions: list[dict[str, Any]],
) -> int:
    copied = 0
    for decision in decisions:
        if decision.get("action") != "keep":
            continue
        rel_path = pathlib.Path(str(decision["relative_path"]))
        src = input_dir / rel_path
        dst = out_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def apply_delete(input_dir: pathlib.Path, decisions: list[dict[str, Any]]) -> int:
    deleted = 0
    for decision in decisions:
        if decision.get("action") != "prune":
            continue
        path = input_dir / pathlib.Path(str(decision["relative_path"]))
        if path.exists() and path.is_file():
            path.unlink()
            deleted += 1
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--transcript-name", default="Messages - Julie Willen.txt")
    parser.add_argument("--base-url")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-file", type=pathlib.Path, default=pathlib.Path("env.sh"))
    parser.add_argument("--max-messages", type=int, default=-1)
    parser.add_argument("--media-limit", type=int, default=DEFAULT_MEDIA_LIMIT)
    parser.add_argument("--batch-size", type=int, default=TEXT_BATCH_SIZE)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--no-multimodal-media", action="store_true")
    parser.add_argument("--apply-delete", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-media", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    if not args.base_url:
        args.base_url = os.environ.get("CORTEXT_JUDGE_BASE_URL", DEFAULT_BASE_URL)
    require_nemotron_model(args.model)
    args.base_url = local_nemotron_base_url(args.base_url)
    started = time.perf_counter()
    transcript = args.input_dir / args.transcript_name
    if not transcript.exists() and not args.skip_text:
        raise FileNotFoundError(f"Transcript not found: {transcript}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    redacted_transcript = args.out_dir / args.transcript_name
    safe_media_dir = args.out_dir / "media"
    manifest_path = args.out_dir / "privacy_manifest.json"

    text_decisions: dict[int, dict[str, Any]] = {}
    messages: list[Message] = []
    if not args.skip_text:
        messages = parse_messages(transcript)
        if args.max_messages >= 0:
            messages = messages[: args.max_messages]
        for start in range(0, len(messages), max(1, args.batch_size)):
            batch = messages[start : start + max(1, args.batch_size)]
            response = call_nemotron(
                base_url=args.base_url,
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You redact private text. Return only valid JSON.",
                    },
                    {"role": "user", "content": text_prompt(batch)},
                ],
                max_tokens=max(1200, sum(estimate_tokens(m.text) for m in batch) * 2),
                timeout=args.timeout,
            )
            text_decisions.update(normalize_text_decisions(batch, response))
            print(f"redacted text batch {start // max(1, args.batch_size) + 1}")
        write_redacted_transcript(redacted_transcript, messages, text_decisions)

    media_decisions: list[dict[str, Any]] = []
    if not args.skip_media:
        media = discover_media(args.input_dir, transcript)
        if args.media_limit >= 0:
            media = media[: args.media_limit]
        with tempfile.TemporaryDirectory(prefix="cortext_privacy_media_") as tmp:
            tmp_dir = pathlib.Path(tmp)
            for item in media:
                decision = classify_media(
                    item,
                    base_url=args.base_url,
                    model=args.model,
                    timeout=args.timeout,
                    temp_dir=tmp_dir,
                    multimodal=not args.no_multimodal_media,
                )
                media_decisions.append(decision)
                print(
                    f"{decision['action']:5s} {decision['kind']:5s} "
                    f"{compact_name(pathlib.Path(decision['relative_path']))}"
                )
        copied = copy_safe_media(args.input_dir, safe_media_dir, media_decisions)
    else:
        copied = 0

    deleted = apply_delete(args.input_dir, media_decisions) if args.apply_delete else 0
    text_counts: dict[str, int] = {"keep": 0, "redact": 0, "drop": 0}
    for decision in text_decisions.values():
        text_counts[decision["action"]] = text_counts.get(decision["action"], 0) + 1
    media_counts: dict[str, int] = {"keep": 0, "prune": 0}
    for decision in media_decisions:
        action = str(decision.get("action", "prune"))
        media_counts[action] = media_counts.get(action, 0) + 1

    manifest = {
        "schema": "cortext_nemotron_privacy_prune_redact_v1",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.out_dir),
        "model": args.model,
        "base_url": args.base_url,
        "destructive_delete_applied": bool(args.apply_delete),
        "raw_private_text_in_manifest": False,
        "raw_media_copied_only_when_kept": True,
        "redacted_transcript": str(redacted_transcript) if not args.skip_text else None,
        "safe_media_dir": str(safe_media_dir) if not args.skip_media else None,
        "text_counts": text_counts,
        "media_counts": media_counts,
        "safe_media_copied": copied,
        "source_media_deleted": deleted,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "text_decisions": [
            {
                "id": idx,
                "action": row["action"],
                "categories": row["categories"],
                "confidence": row["confidence"],
                "reason": row["reason"],
            }
            for idx, row in sorted(text_decisions.items())
        ],
        "media_decisions": media_decisions,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
