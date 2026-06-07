#!/usr/bin/env python3
"""Build a single downloadable source bundle for model training labels.

The bundle intentionally preserves the original repository-relative file paths
under one root directory so a training job can inspect provenance and licenses.
Large source files are symlinked into a temporary staging tree and archived with
dereferencing, avoiding a second uncompressed copy on disk.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


BundleFile = tuple[str, str, str]


BUNDLE_FILES: list[BundleFile] = [
    ("labels", "data/label_bank/metadata.json", "Label-bank metadata."),
    ("labels", "data/label_bank/labels.jsonl", "Embedding-bearing label bank JSONL."),
    ("labels", "data/label_bank/hf_labels.txt", "Plain label vocabulary used to build the label bank."),
    ("labels", "data/label_classifier/gold.jsonl", "Small manually curated label-classifier gold slice."),
    ("labels", "data/label_classifier/name_priors.json", "Existing lightweight name-prior labels."),
    ("wordnet", "data/english-wordnet-2025.xml", "English WordNet 2025 XML source."),
    ("salt", "data/salt.csv", "SALT multimodal text/image/audio metadata table; media paths are not bundled."),
    ("names", "data/surnames/forenames.csv", "Multilingual forename frequency table."),
    ("names", "data/surnames/surnames.csv", "Multilingual surname frequency table."),
    ("names", "data/surnames/country_codes.csv", "Country-code lookup for name tables."),
    ("names", "data/surnames/LICENSE.txt", "License for the surname/forename data."),
    ("names", "data/raw/names/census_surnames.csv", "US Census surname prior, if present."),
]


def Sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def CountLines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def CsvHeaderAndRows(path: Path) -> tuple[list[str], int | None]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            rows = 0
            for _ in reader:
                rows += 1
            return header, rows
    except UnicodeDecodeError:
        return [], None


def JsonlRows(path: Path) -> int:
    return CountLines(path)


def FileRecord(repo_root: Path, category: str, rel_path: str, description: str) -> dict[str, Any]:
    path = repo_root / rel_path
    if not path.exists():
        return {
            "category": category,
            "path": rel_path,
            "description": description,
            "present": False,
        }

    record: dict[str, Any] = {
        "category": category,
        "path": rel_path,
        "description": description,
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": Sha256(path),
    }

    suffix = path.suffix.lower()
    if suffix == ".csv":
        header, rows = CsvHeaderAndRows(path)
        record["columns"] = header
        record["row_count"] = rows
    elif suffix == ".jsonl":
        record["row_count"] = JsonlRows(path)
    elif suffix == ".txt":
        record["line_count"] = CountLines(path)
    elif suffix == ".xml":
        record["line_count"] = CountLines(path)
    return record


def WriteText(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def BuildReadme(bundle_name: str, manifest: dict[str, Any]) -> str:
    present = [item for item in manifest["files"] if item.get("present")]
    missing = [item for item in manifest["files"] if not item.get("present")]
    total_bytes = sum(int(item.get("bytes", 0)) for item in present)
    lines = [
        f"# {bundle_name}",
        "",
        "This bundle is a single-source trainer input assembled from local Cortext data.",
        "It preserves repository-relative paths under the bundle root.",
        "",
        "Included sources:",
    ]
    for item in present:
        count = item.get("row_count", item.get("line_count", ""))
        count_text = f", count={count}" if count != "" else ""
        lines.append(f"- `{item['path']}` ({item['category']}, {item['bytes']} bytes{count_text})")
    if missing:
        lines.extend(["", "Missing optional sources:"])
        for item in missing:
            lines.append(f"- `{item['path']}` ({item['category']})")
    lines.extend(
        [
            "",
            f"Total source bytes: {total_bytes}",
            "",
            "Files generated at bundle time:",
            "- `MANIFEST.json`: paths, counts, sizes, SHA-256 checksums, and descriptions.",
            "- `SHA256SUMS`: checksum list for source files.",
            "- `FILELIST.txt`: archive contents.",
            "",
            "Notes:",
            "- `data/salt.csv` contains metadata and media paths only; image/audio payload files are not included.",
            "- `data/label_bank/labels.jsonl` contains the existing embedding-bearing label bank.",
            "- The surname/forename tables include their local Apache-2.0 license file.",
        ]
    )
    return "\n".join(lines) + "\n"


def StageBundle(repo_root: Path, output_dir: Path, bundle_name: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_root = output_dir / "stage" / bundle_name
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)

    manifest = {
        "bundle_name": bundle_name,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "files": [FileRecord(repo_root, *entry) for entry in BUNDLE_FILES],
    }
    manifest["total_source_bytes"] = sum(int(item.get("bytes", 0)) for item in manifest["files"] if item.get("present"))
    manifest["present_file_count"] = sum(1 for item in manifest["files"] if item.get("present"))
    manifest["missing_file_count"] = sum(1 for item in manifest["files"] if not item.get("present"))

    for item in manifest["files"]:
        if not item.get("present"):
            continue
        source = repo_root / item["path"]
        dest = stage_root / item["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(source)

    WriteText(stage_root / "MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    WriteText(stage_root / "README.md", BuildReadme(bundle_name, manifest))
    WriteText(
        stage_root / "SHA256SUMS",
        "".join(
            f"{item['sha256']}  {item['path']}\n"
            for item in manifest["files"]
            if item.get("present") and item.get("sha256")
        ),
    )
    WriteText(
        stage_root / "FILELIST.txt",
        "".join(f"{path.relative_to(stage_root)}\n" for path in sorted(stage_root.rglob("*")) if path.is_file() or path.is_symlink()),
    )
    shutil.copyfile(stage_root / "MANIFEST.json", output_dir / f"{bundle_name}_manifest.json")
    shutil.copyfile(stage_root / "README.md", output_dir / f"{bundle_name}_README.md")
    return stage_root, output_dir / f"{bundle_name}_manifest.json"


def BuildArchive(stage_root: Path, output_dir: Path, bundle_name: str, compression_level: int) -> Path:
    archive = output_dir / f"{bundle_name}.tar.zst"
    if archive.exists():
        archive.unlink()

    parent = stage_root.parent
    zstd = shutil.which("zstd")
    if zstd is None:
        archive = output_dir / f"{bundle_name}.tar.gz"
        subprocess.run(["tar", "-C", str(parent), "-hczf", str(archive), bundle_name], check=True)
        return archive

    tar = subprocess.Popen(["tar", "-C", str(parent), "-hcf", "-", bundle_name], stdout=subprocess.PIPE)
    zstd_proc = subprocess.Popen(
        [zstd, f"-{compression_level}", "-T0", "-q", "-o", str(archive)],
        stdin=tar.stdout,
    )
    assert tar.stdout is not None
    tar.stdout.close()
    zstd_status = zstd_proc.wait()
    tar_status = tar.wait()
    if tar_status != 0 or zstd_status != 0:
        raise RuntimeError(f"archive build failed: tar={tar_status}, zstd={zstd_status}")
    return archive


def Main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Build the Cortext model-trainer source bundle.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("build/model_trainer_source_bundle"))
    parser.add_argument("--bundle-name", default="cortext_model_trainer_sources_v1")
    parser.add_argument("--compression-level", type=int, default=10)
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    stage_root, manifest_path = StageBundle(repo_root, output_dir, args.bundle_name)

    result: dict[str, Any] = {
        "bundle_name": args.bundle_name,
        "stage_root": str(stage_root),
        "manifest": str(manifest_path),
    }
    if not args.no_archive:
        archive = BuildArchive(stage_root, output_dir, args.bundle_name, args.compression_level)
        result["archive"] = str(archive)
        result["archive_bytes"] = archive.stat().st_size
        result["archive_sha256"] = Sha256(archive)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(Main(sys.argv[1:]))
