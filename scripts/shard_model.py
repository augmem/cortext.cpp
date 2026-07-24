#!/usr/bin/env python3
"""Split the AIST GGUF model into Git-friendly chunks for language bindings.

Ownership: this repo is the source of truth for model shards. Binding packages
(cortext.go, cortext.py, hermes plugins, …) should copy the produced tree under
`models/` (or the release-assets models/ subtree) rather than re-implementing
sharding.

Parts stay under 100 MiB (default 48 MiB) so they can live in ordinary Git
without LFS. Writes `manifest.json` with per-chunk and whole-file SHA-256.

Layout (matches cortext.go MaterializeModel):

  models/manifest.json
  models/AIST-87M-GGUF/chunks/AIST-87M_q8_0.gguf.part-000
  models/AIST-87M-GGUF/chunks/AIST-87M_q8_0.gguf.part-001
  models/AIST-87M-GGUF/chunks/AIST-87M_q8_0.gguf.part-002
  models/mdbr-leaf-ir/vocab.txt

Examples:
  python3 scripts/shard_model.py
  python3 scripts/shard_model.py --source models/AIST-87M-GGUF/AIST-87M_q8_0.gguf
  python3 scripts/shard_model.py --output dist/release-assets/models --version 1.2.3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "models"
DEFAULT_SOURCE = REPO_ROOT / "models" / "AIST-87M-GGUF" / "AIST-87M_q8_0.gguf"
DEFAULT_VOCAB = REPO_ROOT / "models" / "mdbr-leaf-ir" / "vocab.txt"
MODEL_REL = "AIST-87M-GGUF/AIST-87M_q8_0.gguf"
VOCAB_REL = "mdbr-leaf-ir/vocab.txt"
DEFAULT_CHUNK = 48 * 1024 * 1024  # Git-friendly; matches cortext.go / hermes
SCHEMA = "augmem.cortext.models.v1"
# Expected whole-file digest for the shipping q8_0 quant (download_aist_model.py).
EXPECTED_MODEL_SHA256 = "bf4c49954eccc65183f1a97e44606e86c7ee5a4fea500457124b687a3ec97898"
EXPECTED_MODEL_SIZE = 141491936
EXPECTED_VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def detect_version(explicit: str | None) -> str:
    if explicit:
        return explicit.lstrip("v")
    pyproject = REPO_ROOT / "bindings" / "python" / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
        if m:
            return m.group(1)
    cmake = REPO_ROOT / "CMakeLists.txt"
    if cmake.is_file():
        text = cmake.read_text(encoding="utf-8")
        m = re.search(r"project\s*\(\s*cortext[^)]*VERSION\s+([0-9.]+)", text, re.I | re.S)
        if m:
            return m.group(1)
    try:
        out = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return out.lstrip("v")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return "0.0.0-dev"


def ensure_source(source: Path | None, quant: str = "q8_0") -> tuple[Path, Path]:
    model = source if source is not None else DEFAULT_SOURCE
    model = model.expanduser().resolve()
    vocab = DEFAULT_VOCAB
    if model.is_file() and vocab.is_file():
        return model, vocab
    # Download into the default models tree; if --source pointed elsewhere and is
    # missing, still fall back to the standard location after download.
    print("source model or vocab missing; running download_aist_model.py", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "download_aist_model.py"),
            "--output-dir",
            str(REPO_ROOT / "models"),
            "--quant",
            quant,
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    if not model.is_file():
        model = DEFAULT_SOURCE
    if not model.is_file():
        raise SystemExit(f"model missing after download: {model}")
    if not vocab.is_file():
        raise SystemExit(f"vocab missing after download: {vocab}")
    return model, vocab


def shard_from_file(
    source: Path,
    models_root: Path,
    chunk_size: int,
) -> tuple[list[dict[str, object]], str, int]:
    chunk_dir = models_root / "AIST-87M-GGUF" / "chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    whole = hashlib.sha256()
    chunks: list[dict[str, object]] = []
    index = 0
    total = 0
    with source.open("rb") as handle:
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            name = f"AIST-87M_q8_0.gguf.part-{index:03d}"
            rel = f"AIST-87M-GGUF/chunks/{name}"
            path = models_root / rel
            path.write_bytes(data)
            part_hash = hashlib.sha256(data).hexdigest()
            whole.update(data)
            chunks.append({"filename": rel, "sha256": part_hash})
            total += len(data)
            print(
                f"wrote {rel} ({len(data) / (1024 * 1024):.2f} MiB) {part_hash}",
                flush=True,
            )
            index += 1
    if index == 0:
        raise SystemExit(f"source is empty: {source}")
    return chunks, whole.hexdigest(), total


def adopt_existing_chunks(
    chunk_dir: Path,
    models_root: Path,
) -> tuple[list[dict[str, object]], str, int]:
    parts = sorted(chunk_dir.glob("AIST-87M_q8_0.gguf.part-*"))
    if not parts:
        raise SystemExit(f"no part files in {chunk_dir}")
    dest_dir = models_root / "AIST-87M-GGUF" / "chunks"
    dest_dir.mkdir(parents=True, exist_ok=True)
    whole = hashlib.sha256()
    chunks: list[dict[str, object]] = []
    total = 0
    for part in parts:
        dest = dest_dir / part.name
        if part.resolve() != dest.resolve():
            shutil.copy2(part, dest)
        data_hash = digest_file(dest)
        with dest.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                whole.update(block)
        rel = f"AIST-87M-GGUF/chunks/{part.name}"
        size = dest.stat().st_size
        total += size
        chunks.append({"filename": rel, "sha256": data_hash})
        print(f"adopted {rel} ({size / (1024 * 1024):.2f} MiB) {data_hash}", flush=True)
    return chunks, whole.hexdigest(), total


def write_manifest(
    models_root: Path,
    version: str,
    chunks: list[dict[str, object]],
    model_sha: str,
    size: int,
    vocab_sha: str,
) -> Path:
    manifest = {
        "schema": SCHEMA,
        "cortext_version": version,
        "note": (
            "AIST model is Git-chunked under 100 MiB so it can live in ordinary "
            "Git without LFS. Language bindings reassemble and checksum on first use; "
            "do not re-shard in binding repos — copy this tree from cortext."
        ),
        "assets": [
            {
                "filename": MODEL_REL,
                "sha256": model_sha,
                "size": size,
                "chunks": chunks,
            },
            {"filename": VOCAB_REL, "sha256": vocab_sha},
        ],
    }
    out = models_root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def shard_models(
    *,
    output: Path,
    version: str,
    chunk_size: int = DEFAULT_CHUNK,
    source: Path | None = None,
    verify_expected: bool = True,
) -> dict[str, object]:
    """Shard (or adopt) the AIST model into ``output`` (a models/ root).

    Returns the manifest dict that was written.
    """
    models_root = output.expanduser().resolve()
    models_root.mkdir(parents=True, exist_ok=True)

    if source is not None and source.is_dir():
        chunks, model_sha, size = adopt_existing_chunks(source.resolve(), models_root)
        vocab_src = DEFAULT_VOCAB if DEFAULT_VOCAB.is_file() else None
        if vocab_src is None:
            # Try sibling of chunk parent: .../models/mdbr-leaf-ir/vocab.txt
            candidate = source.resolve().parents[1] / "mdbr-leaf-ir" / "vocab.txt"
            if candidate.is_file():
                vocab_src = candidate
        if vocab_src is None or not vocab_src.is_file():
            _, vocab_src = ensure_source(None)
    else:
        model_src, vocab_src = ensure_source(source)
        chunks, model_sha, size = shard_from_file(model_src, models_root, chunk_size)

    vocab_dest = models_root / VOCAB_REL
    vocab_dest.parent.mkdir(parents=True, exist_ok=True)
    if vocab_src.resolve() != vocab_dest.resolve():
        shutil.copy2(vocab_src, vocab_dest)
    vocab_sha = digest_file(vocab_dest)

    if verify_expected:
        if model_sha != EXPECTED_MODEL_SHA256 or size != EXPECTED_MODEL_SIZE:
            raise SystemExit(
                f"model digest/size mismatch: sha256={model_sha} size={size} "
                f"(expected {EXPECTED_MODEL_SHA256} / {EXPECTED_MODEL_SIZE})"
            )
        if vocab_sha != EXPECTED_VOCAB_SHA256:
            raise SystemExit(
                f"vocab digest mismatch: {vocab_sha} (expected {EXPECTED_VOCAB_SHA256})"
            )

    write_manifest(models_root, version, chunks, model_sha, size, vocab_sha)
    print(f"wrote {models_root / 'manifest.json'}", flush=True)
    print(f"model sha256={model_sha} size={size}", flush=True)
    return json.loads((models_root / "manifest.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Full .gguf file, or a directory of existing .part-* chunks to adopt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="models/ root to write (default: repo models/).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        help=f"Bytes per part (default {DEFAULT_CHUNK}).",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="cortext_version recorded in manifest (default: detect).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Do not require the shipping q8_0/vocab digests (for experiments).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = detect_version(args.version)
    shard_models(
        output=args.output,
        version=version,
        chunk_size=args.chunk_size,
        source=args.source,
        verify_expected=not args.skip_verify,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
