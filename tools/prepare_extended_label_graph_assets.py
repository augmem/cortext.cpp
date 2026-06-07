#!/usr/bin/env python3
"""Prepare extra real Wikimedia media for label-graph policy benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path


ASSETS = [
    {
        "id": "cat_image",
        "kind": "image",
        "group": "cat_entity",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Cat_August_2010-4.jpg",
        "source_file": "cat.jpg",
        "raw_file": "cat_384x384.rgb",
        "text": "Cat",
    },
    {
        "id": "cat_audio",
        "kind": "audio",
        "group": "cat_entity",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/En-us-cat.ogg",
        "source_file": "cat.ogg",
        "raw_file": "cat_16k_mono.f32",
        "text": "Cat",
    },
    {
        "id": "train_image",
        "kind": "image",
        "group": "train_entity",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/ICE_3_Oberhaider-Wald-Tunnel.jpg",
        "source_file": "train.jpg",
        "raw_file": "train_384x384.rgb",
        "text": "Train",
    },
    {
        "id": "train_audio",
        "kind": "audio",
        "group": "train_entity",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/En-us-train.ogg",
        "source_file": "train.ogg",
        "raw_file": "train_16k_mono.f32",
        "text": "Train",
    },
    {
        "id": "bell_image",
        "kind": "image",
        "group": "bell_entity",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Liberty_Bell_2008.jpg",
        "source_file": "bell.jpg",
        "raw_file": "bell_384x384.rgb",
        "text": "Bell",
    },
    {
        "id": "bell_audio",
        "kind": "audio",
        "group": "bell_entity",
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/En-us-bell.ogg",
        "source_file": "bell.ogg",
        "raw_file": "bell_16k_mono.f32",
        "text": "Bell",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "cortext-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def prepare(output_dir: Path) -> dict:
    source_dir = output_dir / "source"
    raw_dir = output_dir / "raw"
    source_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_assets = []
    for asset in ASSETS:
        source_path = source_dir / asset["source_file"]
        raw_path = raw_dir / asset["raw_file"]
        download(asset["url"], source_path)
        if asset["kind"] == "image":
            run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source_path),
                    "-vf",
                    "scale=384:384:force_original_aspect_ratio=increase,crop=384:384",
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "rawvideo",
                    str(raw_path),
                ]
            )
        else:
            run_ffmpeg(
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
        record = dict(asset)
        record["source_path"] = str(source_path)
        record["raw_path"] = str(raw_path)
        record["source_sha256"] = sha256(source_path)
        record["raw_sha256"] = sha256(raw_path)
        manifest_assets.append(record)

    manifest = {
        "description": "Extra real Wikimedia media for ES-AIST label graph policy diagnostics.",
        "raw_format": {
            "image": "rgb24, 384x384, row-major",
            "audio": "float32 little-endian PCM, 16 kHz mono",
        },
        "assets": manifest_assets,
    }
    (output_dir / "extended_label_graph_assets_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="build/extended_label_graph_assets", type=Path
    )
    args = parser.parse_args()
    manifest = prepare(args.output_dir)
    print(json.dumps({"asset_count": len(manifest["assets"]), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
