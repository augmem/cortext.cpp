#!/usr/bin/env python3
"""Quick local throughput benchmark: gemma4:12b-it-qat vs gemma4:e4b on Ollama.

Covers text / image / audio modalities with synthetic payloads (no private data).
Reports prompt-eval and generation tok/s from Ollama's own response stats.
Run only when the GPU is otherwise idle.
"""

import base64
import io
import json
import math
import struct
import subprocess
import sys
import wave
import zlib

BASE_URL = "http://127.0.0.1:11434"
MODELS = ["gemma4:12b-it-qat", "gemma4:e4b"]
NUM_PREDICT = 200
WARM_RUNS = 2  # first run includes model load; report the warm one


def synth_png(size: int = 512) -> str:
    """Minimal in-memory PNG (gradient) via stdlib only."""
    raw = bytearray()
    for y in range(size):
        raw.append(0)  # filter: none
        for x in range(size):
            raw += bytes((x % 256, y % 256, (x * y) % 256))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    png += chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


def synth_wav(seconds: float = 5.0, rate: int = 16000) -> str:
    """Sine sweep WAV via stdlib only."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        n = int(seconds * rate)
        for i in range(n):
            t = i / rate
            freq = 220.0 + 660.0 * (i / n)
            sample = int(20000 * math.sin(2 * math.pi * freq * t))
            frames += struct.pack("<h", sample)
        w.writeframes(bytes(frames))
    return base64.b64encode(buf.getvalue()).decode()


TEXT_PROMPT = (
    "Summarize the following project notes in five sentences.\n\n"
    + "\n".join(
        f"- On day {i}, the team reviewed module {i % 7} and recorded "
        f"{(i * 13) % 97} open items, then scheduled follow-up work for the "
        f"next sprint with priority {(i * 7) % 5}."
        for i in range(120)
    )
)

CASES = [
    ("text", TEXT_PROMPT, []),
    ("image", "Describe this image in five sentences.", [synth_png()]),
    ("audio", "Describe this audio clip in five sentences.", [synth_wav()]),
]


def call(model: str, prompt: str, media: list[str]) -> dict:
    message = {"role": "user", "content": prompt}
    if media:
        message["images"] = media
    body = {
        "model": model,
        "messages": [message],
        "stream": False,
        "keep_alive": "2m",
        "options": {"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": 8192},
    }
    out = subprocess.run(
        [
            "curl", "-s", "--max-time", "600",
            "-X", "POST", f"{BASE_URL}/api/chat",
            "-H", "Content-Type: application/json",
            "--data-binary", "@-",
        ],
        input=json.dumps(body).encode(),
        capture_output=True,
        check=True,
    )
    return json.loads(out.stdout)


def unload(model: str) -> None:
    body = {"model": model, "messages": [], "keep_alive": 0}
    subprocess.run(
        ["curl", "-s", "-X", "POST", f"{BASE_URL}/api/chat",
         "-H", "Content-Type: application/json", "--data-binary", "@-"],
        input=json.dumps(body).encode(),
        capture_output=True,
    )


def main() -> None:
    results = []
    for model in MODELS:
        for name, prompt, media in CASES:
            stats = None
            for run in range(WARM_RUNS):
                payload = call(model, prompt, media)
                if "message" not in payload:
                    print(f"{model} {name}: ERROR {json.dumps(payload)[:200]}",
                          flush=True)
                    stats = None
                    break
                stats = payload
            if not stats:
                continue
            pe_n = stats.get("prompt_eval_count", 0)
            pe_d = stats.get("prompt_eval_duration", 0) / 1e9
            ev_n = stats.get("eval_count", 0)
            ev_d = stats.get("eval_duration", 0) / 1e9
            row = {
                "model": model,
                "modality": name,
                "prompt_tokens": pe_n,
                "prompt_eval_s": round(pe_d, 3),
                "prompt_tok_s": round(pe_n / pe_d, 1) if pe_d > 0 else None,
                "gen_tokens": ev_n,
                "gen_s": round(ev_d, 3),
                "gen_tok_s": round(ev_n / ev_d, 1) if ev_d > 0 else None,
                "total_s": round(stats.get("total_duration", 0) / 1e9, 3),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
        unload(model)
    with open("/tmp/gemma_throughput_results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote /tmp/gemma_throughput_results.json", flush=True)


if __name__ == "__main__":
    main()
