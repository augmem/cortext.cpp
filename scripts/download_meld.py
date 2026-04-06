#!/usr/bin/env python3
"""
Download MELD emotion-labeled dialogue CSV files.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


URLS = {
    "train_sent_emo.csv": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev_sent_emo.csv": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test_sent_emo.csv": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}


def download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MELD raw CSV files.")
    parser.add_argument(
        "--out-dir",
        default="data/raw/meld",
        help="Output directory for MELD CSV files.",
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
