#!/usr/bin/env python3
"""
Download GoEmotions raw TSV files and label index.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


URLS = {
    "train.tsv": "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv",
    "dev.tsv": "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv",
    "test.tsv": "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv",
    "emotions.txt": "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt",
}


def download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download GoEmotions raw files.")
    parser.add_argument(
        "--out-dir",
        default="data/raw/goemotions",
        help="Output directory for GoEmotions TSV files.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        dest = out_dir / name
        download(url, dest, args.force)
        print(f"[OK] downloaded {name} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
