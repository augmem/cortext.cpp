#!/usr/bin/env python3
"""Generate privacy-safe Julie raw-speech audio with Chatterbox Turbo."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import time
from collections import Counter


DEFAULT_INPUT = pathlib.Path.home() / "Documents/Memory/Julie"
DEFAULT_MODEL = "mlx-community/chatterbox-turbo-fp16"
DEFAULT_VLLM_MLX_SPEC = (
    "vllm-mlx[audio] @ git+https://github.com/waybarrios/vllm-mlx"
)


def trim(value: str) -> str:
    return value.strip()


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


def parse_messages(path: pathlib.Path) -> list[dict]:
    messages: list[dict] = []
    pending_header = ""
    pending_text: list[str] = []

    def flush() -> None:
        nonlocal pending_header, pending_text
        if not pending_header:
            return
        timestamp = parse_timestamp(pending_header)
        text = trim("\n".join(pending_text))
        if timestamp is not None and text:
            messages.append(
                {
                    "original_index": len(messages),
                    "timestamp": timestamp,
                    "from_contact": " from " in pending_header,
                    "text": text,
                }
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


def stratified_sample(messages: list[dict], target: int) -> list[dict]:
    if target <= 0 or len(messages) <= target:
        return messages
    n = len(messages)
    selected: set[int] = set()

    def add_window(center: int, count: int) -> None:
        if count <= 0:
            return
        start = max(0, min(center - count // 2, max(0, n - count)))
        for offset in range(count):
            if start + offset < n:
                selected.add(start + offset)

    early = target // 3
    middle = target // 3
    recent = target - early - middle
    add_window(target // 6, early)
    add_window(n // 2, middle)
    add_window(n - recent // 2, recent)
    for i in range(target):
        if len(selected) >= target:
            break
        pos = round(i * (n - 1) / max(1, target - 1))
        selected.add(max(0, min(pos, n - 1)))
    return [messages[i] for i in sorted(selected)]


def media_kind(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".heic"}:
        return "image"
    if ext in {".mov", ".mp4", ".m4v"}:
        return "video"
    if ext in {".m4a", ".mp3", ".wav", ".flac", ".aac"}:
        return "audio"
    if ext in {".url"}:
        return "link"
    return "other"


def build_media_index(input_dir: pathlib.Path) -> dict[int, list[str]]:
    index: dict[int, list[str]] = {}
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        timestamp = parse_timestamp(path.name)
        if timestamp is None:
            continue
        index.setdefault(timestamp, []).append(media_kind(path))
    return index


def macos_sdk_env() -> dict[str, str]:
    try:
        sdk = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"SDKROOT": sdk, "MACOSX_DEPLOYMENT_TARGET": "15.0"}


def run(cmd: list[str], env_prefix: dict[str, str], *, quiet: bool = False) -> float:
    env = os.environ.copy()
    env.update(env_prefix)
    started = time.perf_counter()
    if quiet:
        subprocess.run(
            cmd,
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(cmd, check=True, env=env)
    return (time.perf_counter() - started) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--sample-messages", type=int, default=120)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vllm-mlx-spec", default=DEFAULT_VLLM_MLX_SPEC)
    parser.add_argument("--male-ref-audio", type=pathlib.Path)
    parser.add_argument("--female-ref-audio", type=pathlib.Path)
    parser.add_argument("--male-ref-text", default="")
    parser.add_argument("--female-ref-text", default="")
    parser.add_argument("--max-chars", type=int, default=420)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    messages = parse_messages(args.input_dir / "Messages - Julie Willen.txt")
    media_index = build_media_index(args.input_dir)
    sample = stratified_sample(messages, args.sample_messages)
    generated_dir = args.out_dir / "generated"
    raw_dir = args.out_dir / "raw"
    generated_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    env = macos_sdk_env()
    media_counts: Counter[str] = Counter()
    manifest: dict = {
        "schema": "julie_raw_speech_manifest_v1",
        "generator": "vllm-mlx mlx_audio.tts.generate",
        "model": args.model,
        "input_dir": str(args.input_dir),
        "transcript_text_in_manifest": False,
        "sampled_messages": len(sample),
        "sampled_media_adjacent_messages": 0,
        "sampled_media_kind_counts": {},
        "records": [],
    }

    for local_index, message in enumerate(sample):
        role = "contact" if message["from_contact"] else "self"
        gender = "female" if message["from_contact"] else "male"
        ref_audio = args.female_ref_audio if message["from_contact"] else args.male_ref_audio
        ref_text = args.female_ref_text if message["from_contact"] else args.male_ref_text
        prefix = f"julie_raw_speech_{local_index:05d}_{role}"
        wav_path = generated_dir / f"{prefix}.wav"
        raw_path = raw_dir / f"{prefix}_16k_mono.f32"
        text = message["text"][: args.max_chars]

        tts_ms = 0.0
        ffmpeg_ms = 0.0
        media_kinds = sorted(set(media_index.get(message["timestamp"], [])))
        if media_kinds:
            manifest["sampled_media_adjacent_messages"] += 1
            media_counts.update(media_kinds)
        if not args.skip_existing or not raw_path.exists():
            cmd = [
                "uvx",
                "--python",
                "3.11",
                "--from",
                args.vllm_mlx_spec,
                "mlx_audio.tts.generate",
                "--model",
                args.model,
                "--text",
                text,
                "--output_path",
                str(generated_dir),
                "--file_prefix",
                prefix,
                "--audio_format",
                "wav",
                "--join_audio",
                "--gender",
                gender,
            ]
            if ref_audio:
                cmd += ["--ref_audio", str(ref_audio)]
            if ref_text:
                cmd += ["--ref_text", ref_text]
            tts_ms = run(cmd, env, quiet=True)
            ffmpeg_ms = run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(wav_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "f32le",
                    str(raw_path),
                ],
                env,
            )

        manifest["records"].append(
            {
                "local_index": local_index,
                "original_index": message["original_index"],
                "timestamp": message["timestamp"],
                "speaker_role": role,
                "voice_gender": gender,
                "voice_ref_audio": str(ref_audio) if ref_audio else "",
                "audio_path": str(raw_path),
                "wav_path": str(wav_path),
                "text_sha256": hashlib.sha256(message["text"].encode()).hexdigest(),
                "text_chars": len(message["text"]),
                "media_adjacent": bool(media_kinds),
                "media_attachment_count": len(media_index.get(message["timestamp"], [])),
                "media_kinds": media_kinds,
                "tts_ms": tts_ms,
                "ffmpeg_ms": ffmpeg_ms,
                "sample_rate": 16000,
                "num_channels": 1,
            }
        )

    manifest["sampled_media_kind_counts"] = dict(media_counts)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
