#!/usr/bin/env python3
"""Download the AIST GGUF runtime model used by Cortext."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys
import tempfile
import urllib.error
import urllib.request


FILES = {
    "q8_0": {
        "filename": "AIST-87M_q8_0.gguf",
        "sha256": "bf4c49954eccc65183f1a97e44606e86c7ee5a4fea500457124b687a3ec97898",
        "size": 141491936,
    },
    "q5_1": {
        "filename": "AIST-87M_q5_1.gguf",
        "sha256": "2f4edf5674c5d4096775a151a3df3494eb2a6d0acb89039819d30832389da285",
        "size": 134357664,
    },
}

TOKENIZER_FILE = {
    "repo": "bert-base-uncased",
    "filename": "vocab.txt",
    "target": pathlib.Path("mdbr-leaf-ir") / "vocab.txt",
    "sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "size": 231508,
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: pathlib.Path, expected_sha256: str, expected_size: int) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.stat().st_size != expected_size:
        return False
    return sha256_file(path) == expected_sha256


def url_for(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=1"


def download(url: str, dest: pathlib.Path, expected_sha256: str, expected_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent)
    )
    os.close(fd)
    tmp = pathlib.Path(tmp_name)

    headers = {"User-Agent": "cortext-model-bootstrap/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else 0
            seen = 0
            next_report = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                seen += len(chunk)
                if total_bytes and seen >= next_report:
                    pct = int(seen * 100 / total_bytes)
                    print(f"  {dest.name}: {pct:3d}% ({seen}/{total_bytes} bytes)")
                    next_report = seen + max(total_bytes // 20, 1)
        actual_size = tmp.stat().st_size
        actual_sha256 = sha256_file(tmp)
        if actual_size != expected_size:
            raise RuntimeError(
                f"{dest.name} size mismatch: got {actual_size}, expected {expected_size}"
            )
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{dest.name} sha256 mismatch: got {actual_sha256}, expected {expected_sha256}"
            )
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def selected_quants(value: str) -> list[str]:
    if value == "all":
        return ["q8_0", "q5_1"]
    return [value]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="models",
        type=pathlib.Path,
        help="Model asset output root. AIST files are placed under AIST-87M-GGUF/.",
    )
    parser.add_argument(
        "--quant",
        choices=["q8_0", "q5_1", "all"],
        default="q8_0",
        help="Quantization to download. q8_0 is Cortext's preferred default.",
    )
    parser.add_argument(
        "--repo",
        default="augmem/AIST-87M-GGUF",
        help="Hugging Face model repository.",
    )
    parser.add_argument("--revision", default="main", help="Repository revision.")
    parser.add_argument(
        "--tokenizer-repo",
        default=TOKENIZER_FILE["repo"],
        help="Hugging Face repository for the public BERT WordPiece vocab.",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default="main",
        help="Tokenizer repository revision.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when a valid file already exists.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    model_root = output_dir / "AIST-87M-GGUF"
    for quant in selected_quants(args.quant):
        spec = FILES[quant]
        dest = model_root / spec["filename"]
        if not args.force and verify(dest, spec["sha256"], spec["size"]):
            print(f"[OK] {dest} already present and verified")
            continue
        print(f"[download] {dest}")
        try:
            download(
                url_for(args.repo, args.revision, spec["filename"]),
                dest,
                spec["sha256"],
                spec["size"],
            )
        except urllib.error.HTTPError as exc:
            print(f"error: download failed with HTTP {exc.code}: {exc.url}", file=sys.stderr)
            return 1
        except urllib.error.URLError as exc:
            print(f"error: download failed: {exc.reason}", file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"[OK] downloaded and verified {dest}")

    tokenizer_dest = output_dir / TOKENIZER_FILE["target"]
    if not args.force and verify(
        tokenizer_dest, TOKENIZER_FILE["sha256"], TOKENIZER_FILE["size"]
    ):
        print(f"[OK] {tokenizer_dest} already present and verified")
    else:
        print(f"[download] {tokenizer_dest}")
        try:
            download(
                url_for(
                    args.tokenizer_repo,
                    args.tokenizer_revision,
                    TOKENIZER_FILE["filename"],
                ),
                tokenizer_dest,
                TOKENIZER_FILE["sha256"],
                TOKENIZER_FILE["size"],
            )
        except urllib.error.HTTPError as exc:
            print(f"error: download failed with HTTP {exc.code}: {exc.url}", file=sys.stderr)
            return 1
        except urllib.error.URLError as exc:
            print(f"error: download failed: {exc.reason}", file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"[OK] downloaded and verified {tokenizer_dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
