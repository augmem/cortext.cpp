#!/usr/bin/env python3
"""Link Git-friendly AIST *shards* into libcortext; reassemble only at load.

Source of truth for GitHub (no LFS): models/AIST-87M-GGUF/chunks/*.part-* plus
models/mdbr-leaf-ir/vocab.txt and models/manifest.json.

This script does **not** write a full .gguf into the tree. It generates:

  aist_embedded_blobs.S   — .incbin each shard + vocab (linked into libcortext)
  aist_embedded_meta.h    — sizes, per-shard and whole-file SHA-256
  aist_embedded_table.inc — C++ switch table of shard base pointers

Runtime (aist_embedded_model.cpp) concatenates the linked shards into the
process model cache on first use, verifies digests, and returns the full
.gguf path so the encoder can open a normal file.

Examples:
  python3 scripts/prepare_embedded_aist.py --out-dir build/generated/aist_embed
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
MODELS = REPO_ROOT / "models"
CHUNK_DIR = MODELS / "AIST-87M-GGUF" / "chunks"
DEFAULT_VOCAB = MODELS / "mdbr-leaf-ir" / "vocab.txt"
MANIFEST = MODELS / "manifest.json"
MODEL_REL = "AIST-87M-GGUF/AIST-87M_q8_0.gguf"
VOCAB_REL = "mdbr-leaf-ir/vocab.txt"
# Shipping q8_0 (download_aist_model.py / manifest)
EXPECTED_GGUF_SHA256 = "bf4c49954eccc65183f1a97e44606e86c7ee5a4fea500457124b687a3ec97898"
EXPECTED_GGUF_SIZE = 141491936
EXPECTED_VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest_chunks() -> tuple[list[dict[str, object]], str, int, str]:
    """Return (chunks, whole_sha, whole_size, vocab_sha) from models/manifest.json."""
    if not MANIFEST.is_file():
        raise SystemExit(
            f"missing {MANIFEST}; run: python3 scripts/shard_model.py"
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assets = manifest.get("assets") or []
    model = next((a for a in assets if a.get("filename") == MODEL_REL), None)
    vocab = next((a for a in assets if a.get("filename") == VOCAB_REL), None)
    if model is None or not model.get("chunks"):
        raise SystemExit(f"{MANIFEST} has no chunked model asset for {MODEL_REL}")
    if vocab is None or not vocab.get("sha256"):
        raise SystemExit(f"{MANIFEST} has no vocab asset for {VOCAB_REL}")
    chunks = list(model["chunks"])
    whole_sha = str(model["sha256"])
    whole_size = int(model["size"])
    vocab_sha = str(vocab["sha256"])
    return chunks, whole_sha, whole_size, vocab_sha


def resolve_shard_paths(chunks: list[dict[str, object]]) -> list[tuple[Path, str, int]]:
    """(path, expected_sha256, size) for each chunk in order."""
    resolved: list[tuple[Path, str, int]] = []
    for entry in chunks:
        rel = str(entry["filename"])
        path = MODELS / rel
        if not path.is_file():
            # Allow bare name under chunks/
            alt = CHUNK_DIR / Path(rel).name
            if alt.is_file():
                path = alt
            else:
                raise SystemExit(f"missing shard file: {path}")
        expected = str(entry["sha256"])
        actual = digest_file(path)
        if actual != expected:
            raise SystemExit(
                f"shard checksum mismatch {path}: got {actual} want {expected}"
            )
        size = path.stat().st_size
        if size >= 100 * 1024 * 1024:
            raise SystemExit(
                f"shard {path} is {size} bytes (>= 100 MiB); keep Git-friendly parts"
            )
        resolved.append((path, expected, size))
    return resolved


def ensure_shards() -> tuple[list[tuple[Path, str, int]], str, int, Path, str]:
    """Ensure shards + vocab exist (from git tree). No full .gguf required."""
    has_parts = CHUNK_DIR.is_dir() and any(CHUNK_DIR.glob("AIST-87M_q8_0.gguf.part-*"))
    if not has_parts or not MANIFEST.is_file():
        # Bootstrap once for fresh checkouts: download full model, shard, then embed shards.
        print(
            "shards/manifest missing; bootstrapping via download + shard_model.py",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "download_aist_model.py"),
                "--output-dir",
                str(MODELS),
                "--quant",
                "q8_0",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "shard_model.py"),
                "--output",
                str(MODELS),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    chunks_meta, whole_sha, whole_size, vocab_sha = load_manifest_chunks()
    if whole_sha != EXPECTED_GGUF_SHA256 or whole_size != EXPECTED_GGUF_SIZE:
        raise SystemExit(
            f"manifest whole model is sha256={whole_sha} size={whole_size}; "
            f"expected {EXPECTED_GGUF_SHA256} / {EXPECTED_GGUF_SIZE}"
        )
    shards = resolve_shard_paths(chunks_meta)

    # Verify concatenation without writing a full tree file (stream hash).
    whole = hashlib.sha256()
    total = 0
    for path, _, size in shards:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                whole.update(block)
                total += len(block)
        if total and path.stat().st_size != size:
            pass  # size already recorded
    if total != whole_size or whole.hexdigest() != whole_sha:
        raise SystemExit(
            f"concatenated shards mismatch: size={total} sha256={whole.hexdigest()}"
        )

    if not DEFAULT_VOCAB.is_file():
        raise SystemExit(f"missing vocab: {DEFAULT_VOCAB}")
    if digest_file(DEFAULT_VOCAB) != vocab_sha:
        raise SystemExit(f"vocab checksum mismatch: {DEFAULT_VOCAB}")
    if vocab_sha != EXPECTED_VOCAB_SHA256:
        raise SystemExit(f"unexpected vocab sha256: {vocab_sha}")

    return shards, whole_sha, whole_size, DEFAULT_VOCAB, vocab_sha


def _asm_symbol_block(symbol: str, path_s: str) -> list[str]:
    """Emit a .globl blob symbol with target-appropriate section visibility."""
    return [
        f"  .globl CORTEXT_SYM({symbol})",
        # Keep linkable inside the library but out of the shared-object ABI.
        "#if defined(__APPLE__)",
        f"  .private_extern CORTEXT_SYM({symbol})",
        "#elif !defined(_WIN32) && !defined(__CYGWIN__)",
        f"  .hidden CORTEXT_SYM({symbol})",
        "#endif",
        "  .p2align 4",
        f"CORTEXT_SYM({symbol}):",
        f'  .incbin "{path_s}"',
        "",
    ]


def write_asm(
    out_s: Path,
    shards: list[tuple[Path, str, int]],
    vocab: Path,
) -> None:
    lines = [
        "/* Generated by scripts/prepare_embedded_aist.py — do not edit. */",
        "/* Linked shards only; full GGUF is assembled at process load time. */",
        "#if defined(__APPLE__)",
        "#define CORTEXT_SYM(name) _##name",
        "  .section __TEXT,__const",
        "#elif defined(_WIN32) || defined(__CYGWIN__)",
        # COFF / PE (MinGW, windows-gnu Zig): ELF @progbits syntax is invalid.
        "#define CORTEXT_SYM(name) name",
        '  .section .rdata,"dr"',
        "#else",
        "#define CORTEXT_SYM(name) name",
        '  .section .rodata,"a",@progbits',
        "#endif",
        "",
    ]
    for index, (path, _, _) in enumerate(shards):
        path_s = path.resolve().as_posix().replace('"', '\\"')
        lines.extend(
            _asm_symbol_block(f"cortext_embedded_aist_shard{index}_start", path_s)
        )
    vocab_s = vocab.resolve().as_posix().replace('"', '\\"')
    lines.extend(
        _asm_symbol_block("cortext_embedded_aist_vocab_start", vocab_s)
    )
    lines.extend(
        [
            # ELF only: mark non-executable stack (not COFF/Mach-O).
            "#if !defined(__APPLE__) && !defined(_WIN32) && !defined(__CYGWIN__)",
            '  .section .note.GNU-stack,"",@progbits',
            "#endif",
            "",
        ]
    )
    out_s.parent.mkdir(parents=True, exist_ok=True)
    out_s.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_s} ({len(shards)} shards + vocab)", flush=True)


def write_meta(
    out_h: Path,
    shards: list[tuple[Path, str, int]],
    whole_sha: str,
    whole_size: int,
    vocab: Path,
    vocab_sha: str,
) -> None:
    sizes = ", ".join(str(s) for _, _, s in shards)
    shas = ",\n  ".join(f'"{sha}"' for _, sha, _ in shards)
    out_h.parent.mkdir(parents=True, exist_ok=True)
    out_h.write_text(
        f"""#pragma once
// Generated by scripts/prepare_embedded_aist.py — do not edit.
// Full GGUF is NOT stored in Git; only shards are linked into libcortext.
#include <cstddef>
#include <cstdint>

#define CORTEXT_EMBEDDED_AIST_SHARD_COUNT {len(shards)}
#define CORTEXT_EMBEDDED_AIST_GGUF_SHA256 "{whole_sha}"
#define CORTEXT_EMBEDDED_AIST_GGUF_SIZE {whole_size}ull
#define CORTEXT_EMBEDDED_AIST_VOCAB_SHA256 "{vocab_sha}"
#define CORTEXT_EMBEDDED_AIST_VOCAB_SIZE {vocab.stat().st_size}ull

static constexpr std::uint64_t kCortextEmbeddedAistShardSizes[CORTEXT_EMBEDDED_AIST_SHARD_COUNT] = {{
  {sizes}
}};

static constexpr const char *kCortextEmbeddedAistShardSha256[CORTEXT_EMBEDDED_AIST_SHARD_COUNT] = {{
  {shas}
}};
""",
        encoding="utf-8",
    )
    print(f"wrote {out_h}", flush=True)


def write_table_inc(out_inc: Path, shard_count: int) -> None:
    """C++ switch returning base pointer for each linked shard."""
    cases = "\n".join(
        f"    case {i}: return &cortext_embedded_aist_shard{i}_start;"
        for i in range(shard_count)
    )
    decls = "\n".join(
        f"extern const unsigned char cortext_embedded_aist_shard{i}_start;"
        for i in range(shard_count)
    )
    out_inc.parent.mkdir(parents=True, exist_ok=True)
    out_inc.write_text(
        f"""// Generated by scripts/prepare_embedded_aist.py — do not edit.
extern "C" {{
{decls}
extern const unsigned char cortext_embedded_aist_vocab_start;
}}

inline const unsigned char *
CortextEmbeddedAistShardData (std::size_t index)
{{
  switch (index)
    {{
{cases}
    default: return nullptr;
    }}
}}
""",
        encoding="utf-8",
    )
    print(f"wrote {out_inc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--from-shards",
        action="store_true",
        help="Accepted for compatibility; shards are always the embed source.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Do not embed; only write a marker that embed is disabled.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stub:
        (out_dir / "cortext_embed_aist.enabled").write_text("0\n", encoding="utf-8")
        return 0

    shards, whole_sha, whole_size, vocab, vocab_sha = ensure_shards()
    write_asm(out_dir / "aist_embedded_blobs.S", shards, vocab)
    write_meta(
        out_dir / "aist_embedded_meta.h",
        shards,
        whole_sha,
        whole_size,
        vocab,
        vocab_sha,
    )
    write_table_inc(out_dir / "aist_embedded_table.inc", len(shards))
    (out_dir / "cortext_embed_aist.enabled").write_text("1\n", encoding="utf-8")
    print(
        f"embed ready shards={len(shards)} whole_size={whole_size} "
        f"sha256={whole_sha} (full GGUF assembled at load, not stored in Git)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
