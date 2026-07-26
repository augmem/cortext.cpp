#!/usr/bin/env python3
"""Split surname CSVs into Git-friendly chunks (no LFS).

Mirrors scripts/shard_model.py: parts stay under 100 MiB (default 48 MiB) so they
can live in ordinary Git. Full CSVs stay local/build-time only and are
reassembled from chunks + checksummed via data/surnames/manifest.json.

Layout:

  data/surnames/manifest.json
  data/surnames/chunks/forenames.csv.part-000
  data/surnames/chunks/forenames.csv.part-001
  ...
  data/surnames/chunks/surnames.csv.part-000
  ...
  data/surnames/country_codes.csv   # small; tracked whole
  data/surnames/LICENSE.txt

Examples:
  python3 scripts/shard_surnames.py
  python3 scripts/shard_surnames.py --assemble
  python3 scripts/shard_surnames.py --source-dir data/surnames --output data/surnames
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "surnames"
DEFAULT_CHUNK = 48 * 1024 * 1024
SCHEMA = "augmem.cortext.surnames.v1"
# Whole files currently tracked via LFS; keep digests stable across re-shard.
SHIPPING_ASSETS = (
    {
        "filename": "forenames.csv",
        "sha256": "06be58e5cca902583623aa911218fab3a26fb88b52390606bfcc5ab5a7536eca",
        "size": 213699759,
    },
    {
        "filename": "surnames.csv",
        "sha256": "d0cb535df5954c5951698984ebdc6038b66c7fdae08cb370d58e410b5c196d54",
        "size": 360431915,
    },
)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def shard_file(
    source: Path,
    root: Path,
    chunk_size: int,
) -> tuple[list[dict[str, object]], str, int]:
    chunk_dir = root / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Drop stale parts for this asset only.
    for stale in chunk_dir.glob(f"{source.name}.part-*"):
        stale.unlink()

    whole = hashlib.sha256()
    chunks: list[dict[str, object]] = []
    index = 0
    total = 0
    with source.open("rb") as handle:
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            name = f"{source.name}.part-{index:03d}"
            rel = f"chunks/{name}"
            path = root / rel
            path.write_bytes(data)
            part_hash = hashlib.sha256(data).hexdigest()
            whole.update(data)
            chunks.append({"filename": rel, "sha256": part_hash, "size": len(data)})
            total += len(data)
            print(
                f"wrote {rel} ({len(data) / (1024 * 1024):.2f} MiB) {part_hash}",
                flush=True,
            )
            index += 1
    if index == 0:
        raise SystemExit(f"source is empty: {source}")
    return chunks, whole.hexdigest(), total


def write_manifest(
    root: Path,
    assets: list[dict[str, object]],
) -> Path:
    manifest = {
        "schema": SCHEMA,
        "note": (
            "Surname CSVs are Git-chunked under 100 MiB so they can live in "
            "ordinary Git without LFS. Reassemble with "
            "`python3 scripts/shard_surnames.py --assemble` (or the probe "
            "tool, which assembles on demand)."
        ),
        "assets": assets,
    }
    out = root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def load_manifest(root: Path) -> dict[str, object]:
    path = root / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assemble_asset(root: Path, asset: dict[str, object], *, force: bool) -> Path:
    filename = str(asset["filename"])
    dest = root / filename
    expected_sha = str(asset["sha256"])
    expected_size = int(asset["size"])
    chunks = asset.get("chunks") or []
    if not chunks:
        raise SystemExit(f"asset has no chunks: {filename}")

    if dest.is_file() and not force:
        size = dest.stat().st_size
        if size == expected_size:
            sha = digest_file(dest)
            if sha == expected_sha:
                print(f"ok {filename} (already assembled)", flush=True)
                return dest
        # Tiny stubs are almost always leftover Git LFS pointers from a skipped smudge.
        is_lfs_pointer = False
        if size < 1024:
            with dest.open("rb") as handle:
                head = handle.read(120).decode("utf-8", errors="ignore")
            is_lfs_pointer = head.startswith("version https://git-lfs.github.com/spec/v1")
        if not is_lfs_pointer:
            sha = digest_file(dest)
            raise SystemExit(
                f"existing {dest} digest/size mismatch: size={size} sha256={sha} "
                f"(expected {expected_size} / {expected_sha}); pass --force to overwrite"
            )
        print(f"replacing LFS pointer stub for {filename}", flush=True)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    whole = hashlib.sha256()
    total = 0
    with tmp.open("wb") as out:
        for chunk in chunks:
            rel = str(chunk["filename"])
            part_path = root / rel
            if not part_path.is_file():
                raise SystemExit(f"missing chunk: {part_path}")
            data = part_path.read_bytes()
            part_sha = hashlib.sha256(data).hexdigest()
            expected_part = str(chunk["sha256"])
            if part_sha != expected_part:
                raise SystemExit(
                    f"chunk digest mismatch: {rel} sha256={part_sha} "
                    f"(expected {expected_part})"
                )
            out.write(data)
            whole.update(data)
            total += len(data)

    sha = whole.hexdigest()
    if total != expected_size or sha != expected_sha:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"reassembled {filename} digest/size mismatch: size={total} sha256={sha} "
            f"(expected {expected_size} / {expected_sha})"
        )
    tmp.replace(dest)
    print(f"assembled {filename} ({total / (1024 * 1024):.2f} MiB) {sha}", flush=True)
    return dest


def assemble_all(root: Path, *, force: bool) -> list[Path]:
    manifest = load_manifest(root)
    paths: list[Path] = []
    for asset in manifest.get("assets", []):
        if not isinstance(asset, dict):
            continue
        if not asset.get("chunks"):
            continue
        paths.append(assemble_asset(root, asset, force=force))
    if not paths:
        raise SystemExit(f"no chunked assets in {root / 'manifest.json'}")
    return paths


def shard_all(
    *,
    source_dir: Path,
    output: Path,
    chunk_size: int,
    verify_expected: bool,
) -> dict[str, object]:
    source_dir = source_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "chunks").mkdir(parents=True, exist_ok=True)

    # Preserve small tracked sidecars when sharding into a fresh output tree.
    for name in ("LICENSE.txt", "country_codes.csv"):
        src = source_dir / name
        dst = output / name
        if src.is_file() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    assets: list[dict[str, object]] = []
    expected_by_name = {item["filename"]: item for item in SHIPPING_ASSETS}

    for item in SHIPPING_ASSETS:
        name = item["filename"]
        source = source_dir / name
        if not source.is_file():
            # Allow re-shard from already-assembled output tree.
            source = output / name
        if not source.is_file():
            raise SystemExit(
                f"missing {name}; place the full CSV under {source_dir} "
                f"(or assemble first from existing chunks)"
            )

        chunks, sha, size = shard_file(source, output, chunk_size)
        if verify_expected:
            exp = expected_by_name[name]
            if sha != exp["sha256"] or size != exp["size"]:
                raise SystemExit(
                    f"{name} digest/size mismatch: sha256={sha} size={size} "
                    f"(expected {exp['sha256']} / {exp['size']})"
                )
        assets.append(
            {
                "filename": name,
                "sha256": sha,
                "size": size,
                "chunks": chunks,
            }
        )

    # Keep the tiny country table in the manifest as an unchunked asset when present.
    country = output / "country_codes.csv"
    if country.is_file():
        assets.append(
            {
                "filename": "country_codes.csv",
                "sha256": digest_file(country),
                "size": country.stat().st_size,
            }
        )

    write_manifest(output, assets)
    print(f"wrote {output / 'manifest.json'}", flush=True)
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory containing full forenames.csv / surnames.csv (default: data/surnames).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT,
        help="Surname data root to write chunks + manifest into (default: data/surnames).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        help=f"Bytes per part (default {DEFAULT_CHUNK}; must be > 0 and < 100 MiB).",
    )
    parser.add_argument(
        "--assemble",
        action="store_true",
        help="Reassemble full CSVs from chunks + manifest instead of sharding.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --assemble, overwrite existing full CSVs even if present.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Do not require the shipping whole-file digests (for experiments).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise SystemExit(f"--chunk-size must be a positive integer (got {args.chunk_size})")
    if args.chunk_size >= 100 * 1024 * 1024:
        raise SystemExit(
            f"--chunk-size must stay under GitHub's 100 MiB limit (got {args.chunk_size})"
        )

    root = args.output.expanduser().resolve()
    if args.assemble:
        assemble_all(root, force=args.force)
        return 0

    shard_all(
        source_dir=args.source_dir,
        output=args.output,
        chunk_size=args.chunk_size,
        verify_expected=not args.skip_verify,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
