#!/usr/bin/env python3
"""Smoke-test local Ollama judge image/audio support with generated media."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import pathlib
import shutil
import struct
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import wave
import zlib


DEFAULT_MODEL = "gemma4:12b-it-qat"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def normalize_local_base_url(base_url: str) -> str:
    if "://" not in base_url:
        base_url = f"http://{base_url}"
    normalized = base_url.rstrip("/")
    parsed = urllib.parse.urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOCAL_HOSTS:
        raise RuntimeError(
            "Refusing non-local Ollama endpoint for media judge smoke: "
            f"{normalized!r}"
        )
    return normalized


def request_json(url: str, payload: dict, timeout_s: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.load(response)


def crc_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind)
    crc = binascii.crc32(payload, crc)
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc & 0xFFFFFFFF)


def write_test_png(path: pathlib.Path) -> None:
    width = 32
    height = 32
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            if 10 <= x < 22 and 10 <= y < 22:
                row.extend([255, 0, 0])
            else:
                row.extend([0, 64, 255])
        rows.append(bytes(row))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + crc_chunk(b"IHDR", ihdr)
        + crc_chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + crc_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def write_fallback_tone(path: pathlib.Path) -> None:
    sample_rate = 16000
    duration_s = 1.0
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        for i in range(int(sample_rate * duration_s)):
            sample = int(0.2 * 32767 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            stream.writeframes(struct.pack("<h", sample))


def write_test_speech(path: pathlib.Path, phrase: str) -> tuple[bool, str]:
    say = shutil.which("say")
    if say:
        cmd = [
            say,
            "-o",
            str(path),
            "--file-format=WAVE",
            "--data-format=LEI16@16000",
            phrase,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if path.exists() and path.stat().st_size > 0:
                return True, "macos_say"
        except subprocess.CalledProcessError:
            pass
    write_fallback_tone(path)
    return False, "fallback_tone_no_speech"


def b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    begin = text.find("{")
    end = text.rfind("}")
    if begin >= 0 and end > begin:
        try:
            return json.loads(text[begin : end + 1])
        except json.JSONDecodeError:
            pass
    return {"raw": text}


def ollama_capabilities(base_url: str, model: str) -> dict:
    status = {
        "source": "ollama_api_show",
        "model": model,
        "available": False,
        "capabilities": [],
        "text": False,
        "image": False,
        "audio": False,
        "error": "",
    }
    try:
        payload = request_json(
            f"{base_url}/api/show",
            {"model": model},
            timeout_s=10,
        )
        capabilities = [str(item) for item in payload.get("capabilities", []) or []]
        status.update(
            {
                "available": True,
                "capabilities": capabilities,
                "text": "completion" in capabilities,
                "image": "vision" in capabilities or "image" in capabilities,
                "audio": "audio" in capabilities,
            }
        )
        details = payload.get("details")
        model_info = payload.get("model_info")
        if isinstance(details, dict):
            status["details"] = {
                key: details.get(key)
                for key in ("family", "parameter_size", "quantization_level")
            }
        if isinstance(model_info, dict):
            context_length = model_info.get("gemma4.context_length") or model_info.get(
                "context_length"
            )
            if context_length is not None:
                status.setdefault("details", {})["context_length"] = context_length
    except Exception as exc:
        status["error"] = exc.__class__.__name__
    return status


def chat_media(
    base_url: str,
    model: str,
    prompt: str,
    media_b64: str,
    response_schema: dict,
    timeout_s: int,
    num_ctx: int,
) -> tuple[dict, float]:
    started = time.monotonic()
    payload = request_json(
        f"{base_url}/api/chat",
        {
            "model": model,
            "stream": False,
            "keep_alive": "0s",
            "think": False,
            "format": response_schema,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict local multimodal smoke-test judge. "
                        "Inspect attached image/audio evidence and return only JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": [media_b64],
                }
            ],
            "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": 256},
        },
        timeout_s=timeout_s,
    )
    elapsed = time.monotonic() - started
    content = str(payload.get("message", {}).get("content", ""))
    parsed = parse_json_object(content)
    parsed["_raw_response"] = content
    return parsed, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--num-ctx", type=int, default=32768)
    args = parser.parse_args()
    if args.num_ctx < 1:
        raise RuntimeError("--num-ctx must be positive")

    base_url = normalize_local_base_url(args.ollama_base_url)
    phrase = "blue apple seven"
    results = {}
    with tempfile.TemporaryDirectory(prefix="cortext_media_smoke_") as tmp:
        tmp_dir = pathlib.Path(tmp)
        image_path = tmp_dir / "blue_red_square.png"
        audio_path = tmp_dir / "blue_apple_seven.wav"
        write_test_png(image_path)
        speech_generated, speech_source = write_test_speech(audio_path, phrase)

        image_prompt = (
            "Look at the attached generated image. Return strict JSON only: "
            '{"image_seen": true|false, "description": "..."}'
        )
        image_parsed, image_elapsed = chat_media(
            base_url,
            args.model,
            image_prompt,
            b64(image_path),
            {
                "type": "object",
                "properties": {
                    "image_seen": {"type": "boolean"},
                    "description": {"type": "string"},
                },
                "required": ["image_seen", "description"],
            },
            args.timeout_s,
            args.num_ctx,
        )
        image_seen = bool(image_parsed.get("image_seen"))
        if not image_seen:
            raw = str(image_parsed.get("_raw_response", "")).lower()
            image_seen = "red" in raw and ("blue" in raw or "square" in raw)
        image_parsed["image_seen"] = image_seen

        audio_prompt = (
            "Listen to the attached generated WAV audio. What exact words are "
            "spoken? Return strict JSON with keys audio_seen and transcript."
        )
        audio_parsed, audio_elapsed = chat_media(
            base_url,
            args.model,
            audio_prompt,
            b64(audio_path),
            {
                "type": "object",
                "properties": {
                    "audio_seen": {"type": "boolean"},
                    "transcript": {"type": "string"},
                },
                "required": ["audio_seen", "transcript"],
            },
            args.timeout_s,
            args.num_ctx,
        )
        audio_seen = bool(audio_parsed.get("audio_seen"))
        raw_audio = str(audio_parsed.get("transcript", "")) + " " + str(
            audio_parsed.get("_raw_response", "")
        )
        raw_audio_lower = raw_audio.lower()
        if not audio_seen:
            audio_seen = all(word in raw_audio_lower for word in ("blue", "apple", "seven"))
        audio_parsed["audio_seen"] = audio_seen

        results[args.model] = {
            "image": {
                "returncode": 0,
                "elapsed_s": round(image_elapsed, 3),
                "parsed": {
                    key: value
                    for key, value in image_parsed.items()
                    if key != "_raw_response"
                },
                "error_payload": None,
                "stderr": "",
            },
            "audio": {
                "returncode": 0,
                "elapsed_s": round(audio_elapsed, 3),
                "parsed": {
                    key: value
                    for key, value in audio_parsed.items()
                    if key != "_raw_response"
                },
                "error_payload": None,
                "stderr": "",
            },
        }

    body = {
        "schema": "cortext_local_ollama_judge_media_smoke_v1",
        "private_data_used": False,
        "generated_image_description": "32x32 blue field with centered red square",
        "generated_speech_phrase": phrase,
        "generated_speech_source": speech_source,
        "generated_speech_available": speech_generated,
        "payload_field": "messages[].images base64 payloads for both image and wav audio",
        "ollama_options": {
            "temperature": 0,
            "num_ctx": args.num_ctx,
            "num_predict": 256,
            "keep_alive": "0s",
        },
        "selected_release_judge_model": args.model,
        "selection_reason": (
            "local unified Gemma 4 12B judge; this smoke records the exact "
            "Ollama model used by the release protocol"
        ),
        "ollama_model_capabilities": ollama_capabilities(base_url, args.model),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
