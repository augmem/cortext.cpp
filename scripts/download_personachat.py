#!/usr/bin/env python3
"""
Download and extract PersonaChat raw files for identity supervision.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path


DATA_URL = "http://parl.ai/downloads/personachat/personachat.tgz"


def download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def extract(tar_path: Path, out_dir: Path) -> Path:
    raw_dir = out_dir / "personachat"
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected = raw_dir / "train_self_original.txt"
    if expected.exists():
        return raw_dir
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(out_dir)
    return raw_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PersonaChat raw files.")
    parser.add_argument(
        "--out-dir",
        default="data/raw/personachat",
        help="Output directory for the archive and extracted raw text files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the archive even if it already exists.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    tar_path = out_dir / "personachat.tgz"
    download(DATA_URL, tar_path, args.force)
    raw_dir = extract(tar_path, out_dir)
    print(f"[OK] archive: {tar_path}")
    print(f"[OK] raw files: {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
