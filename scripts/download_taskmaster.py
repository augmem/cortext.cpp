#!/usr/bin/env python3
"""
Download Taskmaster-1 raw JSON files used for topic/task supervision.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


URLS = {
    "self-dialogs.json": "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/self-dialogs.json",
    "woz-dialogs.json": "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/woz-dialogs.json",
}


def download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Taskmaster-1 raw JSON files.")
    parser.add_argument(
        "--out-dir",
        default="data/raw/taskmaster/TM-1-2019",
        help="Output directory for the downloaded raw JSON files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the files already exist.",
    )
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
