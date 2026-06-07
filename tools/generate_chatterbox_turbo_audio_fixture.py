#!/usr/bin/env python3
"""Generate the conversational Bailey audio fixture with Chatterbox Turbo."""

from __future__ import annotations

import argparse
import pathlib
import subprocess


DEFAULT_TEXT = (
    "I named the golden retriever Bailey after my grandmother. "
    "Bailey loves chasing tennis balls in the backyard."
)


def run(cmd: list[str], *, env_prefix: dict[str, str] | None = None) -> None:
    env = None
    if env_prefix:
        import os

        env = os.environ.copy()
        env.update(env_prefix)
    subprocess.run(cmd, check=True, env=env)


def macos_sdk_env() -> dict[str, str]:
    try:
        sdk = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {
        "SDKROOT": sdk,
        "MACOSX_DEPLOYMENT_TARGET": "15.0",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir",
        type=pathlib.Path,
        default=pathlib.Path("build/real_multimodal_episode_assets"),
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--model", default="mlx-community/chatterbox-turbo-fp16")
    parser.add_argument("--prefix", default="bailey_sentence")
    parser.add_argument(
        "--vllm-mlx-spec",
        default="vllm-mlx[audio] @ git+https://github.com/waybarrios/vllm-mlx",
    )
    args = parser.parse_args()

    generated_dir = args.assets_dir / "generated"
    raw_dir = args.assets_dir / "raw"
    generated_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    wav_path = generated_dir / f"{args.prefix}.wav"
    raw_path = raw_dir / f"{args.prefix}_16k_mono.f32"

    env = macos_sdk_env()
    run(
        [
            "uvx",
            "--python",
            "3.11",
            "--from",
            args.vllm_mlx_spec,
            "mlx_audio.tts.generate",
            "--model",
            args.model,
            "--text",
            args.text,
            "--output_path",
            str(generated_dir),
            "--file_prefix",
            args.prefix,
            "--audio_format",
            "wav",
            "--join_audio",
        ],
        env_prefix=env,
    )

    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            str(raw_path),
        ]
    )
    print(f"wrote {wav_path}")
    print(f"wrote {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
