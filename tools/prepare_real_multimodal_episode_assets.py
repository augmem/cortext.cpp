#!/usr/bin/env python3
"""Download and normalize real media for the multimodal episode benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import urllib.request


ASSETS = [
    {
        "id": "dog_image",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Golden_Retriever.jpg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Golden_Retriever.jpg",
        "license": "Public domain",
        "source_file": "dog.jpg",
        "raw_file": "dog_384x384.rgb",
        "kind": "image",
        "width": 384,
        "height": 384,
        "channels": 3,
    },
    {
        "id": "dog_name_audio",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/En-us-bailey.ogg",
        "source_url": "https://commons.wikimedia.org/wiki/File:En-us-bailey.ogg",
        "license": "GFDL 1.2+ / CC BY-SA 3.0, 2.5, 2.0, 1.0",
        "source_file": "bailey.ogg",
        "raw_file": "bailey_16k_mono.f32",
        "kind": "audio",
        "sample_rate": 16000,
    },
    {
        "id": "car_crash_image",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Car_crash_1.jpg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Car_crash_1.jpg",
        "license": "Public domain",
        "source_file": "car_crash.jpg",
        "raw_file": "car_crash_384x384.rgb",
        "kind": "image",
        "width": 384,
        "height": 384,
        "channels": 3,
    },
    {
        "id": "crash_audio",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/En-us-crash.ogg",
        "source_url": "https://commons.wikimedia.org/wiki/File:En-us-crash.ogg",
        "license": "GFDL 1.2+ / CC BY-SA 3.0, 2.5, 2.0, 1.0",
        "source_file": "crash.ogg",
        "raw_file": "crash_16k_mono.f32",
        "kind": "audio",
        "sample_rate": 16000,
    },
]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: pathlib.Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "cortext-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="build/real_multimodal_episode_assets",
        help="Directory that will receive source/, raw/, and manifest JSON.",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.output_dir)
    source_dir = root / "source"
    raw_dir = root / "raw"
    source_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "description": "Real public media normalized for Cortext multimodal episode tests.",
        "raw_format": {
            "image": "rgb24, 384x384, row-major",
            "audio": "float32 little-endian PCM, 16 kHz mono",
        },
        "assets": [],
    }

    for asset in ASSETS:
        source_path = source_dir / asset["source_file"]
        raw_path = raw_dir / asset["raw_file"]
        download(asset["url"], source_path)
        if asset["kind"] == "image":
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source_path),
                    "-vf",
                    "scale=384:384:force_original_aspect_ratio=decrease,"
                    "pad=384:384:(ow-iw)/2:(oh-ih)/2",
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "rawvideo",
                    str(raw_path),
                ]
            )
        else:
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "f32le",
                    str(raw_path),
                ]
            )
        row = dict(asset)
        row["source_sha256"] = sha256(source_path)
        row["raw_sha256"] = sha256(raw_path)
        row["source_path"] = str(source_path)
        row["raw_path"] = str(raw_path)
        manifest["assets"].append(row)

    (root / "real_multimodal_episode_assets_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
