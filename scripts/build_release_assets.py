#!/usr/bin/env python3
"""Build the shared Cortext release asset tree for language bindings.

Ownership: multi-arch shared libraries and the AIST model shards belong to
**this repo**. Model sharding is implemented only in ``scripts/shard_model.py``
(also checked in under ``models/``). Binding repositories (cortext.go, plugins,
…) should copy the shard tree or unpack the release tarball — they must not
re-shard the GGUF themselves.

Layout written under --output (default dist/release-assets):

  native/
    manifest.json
    linux-x86_64/libcortext.so
    linux-aarch64/libcortext.so
    macos-x86_64/libcortext.dylib
    macos-aarch64/libcortext.dylib
    windows-x86_64/cortext.dll
    windows-aarch64/cortext.dll
  models/
    manifest.json
    AIST-87M-GGUF/chunks/AIST-87M_q8_0.gguf.part-*
    mdbr-leaf-ir/vocab.txt

Also writes a versioned tarball next to the tree (unless --skip-tarball):

  dist/cortext-assets-<version>.tar.gz

Examples:
  python3 scripts/build_release_assets.py
  python3 scripts/build_release_assets.py --skip-zig-build --from-python-natives
  python3 scripts/build_release_assets.py --version 1.2.3 --target macos-aarch64
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "dist" / "release-assets"
DEFAULT_BUILD_ROOT = REPO_ROOT / "build" / "release-assets" / "native"
GLOBAL_CACHE = REPO_ROOT / "build" / "release-assets" / "zig-global-cache"
DEFAULT_CHUNK = 48 * 1024 * 1024  # Git-friendly; matches cortext.go / hermes
SCHEMA_NATIVE = "augmem.cortext.native.v1"


@dataclass(frozen=True, slots=True)
class Target:
    package_tag: str
    zig_target: str
    artifact_name: str
    artifact_globs: tuple[str, ...]
    strip_tools: tuple[str, ...] = ()


TARGETS = (
    Target(
        "linux-x86_64",
        "x86_64-linux-gnu.2.17",
        "libcortext.so",
        ("lib/libcortext.so*",),
        ("x86_64-linux-gnu-strip", "strip"),
    ),
    Target(
        "linux-aarch64",
        "aarch64-linux-gnu.2.17",
        "libcortext.so",
        ("lib/libcortext.so*",),
        ("aarch64-linux-gnu-strip",),
    ),
    Target(
        "macos-x86_64",
        "x86_64-macos",
        "libcortext.dylib",
        ("lib/libcortext*.dylib",),
    ),
    Target(
        "macos-aarch64",
        "aarch64-macos",
        "libcortext.dylib",
        ("lib/libcortext*.dylib",),
    ),
    Target(
        "windows-x86_64",
        "x86_64-windows-gnu",
        "cortext.dll",
        ("bin/cortext.dll", "bin/libcortext.dll"),
    ),
    Target(
        "windows-aarch64",
        "aarch64-windows-gnu",
        "cortext.dll",
        ("bin/cortext.dll", "bin/libcortext.dll"),
    ),
)


def run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_targets(names: list[str]) -> list[Target]:
    if not names:
        return list(TARGETS)
    by_name = {t.package_tag: t for t in TARGETS}
    selected: list[Target] = []
    for name in names:
        try:
            selected.append(by_name[name])
        except KeyError as exc:
            supported = ", ".join(sorted(by_name))
            raise SystemExit(f"unknown target {name!r}; supported: {supported}") from exc
    return selected


def detect_version(explicit: str | None) -> str:
    if explicit:
        return explicit.lstrip("v")
    # Prefer bindings/python pyproject, then CMake project version, then git tag.
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


def find_artifact(prefix: Path, target: Target) -> Path:
    matches: list[Path] = []
    for pattern in target.artifact_globs:
        matches.extend(
            path for path in prefix.glob(pattern) if path.is_file() and not path.is_symlink()
        )
    if not matches:
        searched = ", ".join(str(prefix / pattern) for pattern in target.artifact_globs)
        raise FileNotFoundError(
            f"could not find {target.package_tag} shared library; searched {searched}"
        )
    return max(matches, key=lambda path: (path.stat().st_size, len(path.name), path.name))


def maybe_strip(path: Path, target: Target) -> None:
    for tool in target.strip_tools:
        resolved = shutil.which(tool)
        if resolved is None:
            continue
        try:
            run([resolved, "-s", str(path)], cwd=path.parent)
            return
        except subprocess.CalledProcessError:
            continue


def build_native(
    target: Target,
    zig: str,
    optimize: str,
    skip_zig_build: bool,
    build_root: Path,
) -> dict[str, object]:
    prefix = build_root / "prefix" / target.package_tag
    cache = build_root / "zig-cache" / target.package_tag
    if not skip_zig_build and prefix.exists():
        shutil.rmtree(prefix)
    prefix.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    GLOBAL_CACHE.mkdir(parents=True, exist_ok=True)

    if not skip_zig_build:
        run(
            [
                zig,
                "build",
                "--prefix",
                str(prefix),
                "--cache-dir",
                str(cache),
                "--global-cache-dir",
                str(GLOBAL_CACHE),
                f"-Dtarget={target.zig_target}",
                f"-Doptimize={optimize}",
                "-Dshared=true",
                "-Dcli=false",
                "-Dembed-aist-model=true",
                "-Dfetch-aist-model=false",
            ]
        )

    artifact = find_artifact(prefix, target)
    return {
        "package_tag": target.package_tag,
        "zig_target": target.zig_target,
        "artifact": target.artifact_name,
        "source": artifact,
        "size": artifact.stat().st_size,
    }


def copy_from_python_natives(targets: list[Target]) -> list[dict[str, object]]:
    src_root = REPO_ROOT / "bindings" / "python" / "augmem" / "cortext" / "native"
    entries: list[dict[str, object]] = []
    for target in targets:
        src = src_root / target.package_tag / target.artifact_name
        if not src.is_file():
            raise FileNotFoundError(f"missing prebuilt native at {src}")
        entries.append(
            {
                "package_tag": target.package_tag,
                "zig_target": target.zig_target,
                "artifact": target.artifact_name,
                "source": src,
                "size": src.stat().st_size,
            }
        )
    return entries


def install_natives(
    output: Path,
    entries: list[dict[str, object]],
    optimize: str,
) -> None:
    native_root = output / "native"
    if native_root.exists():
        shutil.rmtree(native_root)
    native_root.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, object]] = []
    for entry in entries:
        tag = str(entry["package_tag"])
        name = str(entry["artifact"])
        src = Path(str(entry["source"]))
        dest_dir = native_root / tag
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        shutil.copy2(src, dest)
        size = dest.stat().st_size
        print(f"native {tag}: {dest} ({size / (1024 * 1024):.2f} MiB)", flush=True)
        manifest_entries.append(
            {
                "package_tag": tag,
                "zig_target": entry["zig_target"],
                "artifact": name,
                "size": size,
                "sha256": digest_file(dest),
            }
        )
    (native_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA_NATIVE,
                "optimize": optimize,
                "targets": manifest_entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def shard_models(output: Path, version: str, chunk_size: int) -> None:
    """Delegate to scripts/shard_model.py (single ownership for binding layout)."""
    if chunk_size <= 0:
        raise SystemExit(f"--chunk-size must be a positive integer (got {chunk_size})")
    models_root = output / "models"
    if models_root.exists():
        shutil.rmtree(models_root)
    run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "shard_model.py"),
            "--output",
            str(models_root),
            "--version",
            version,
            "--chunk-size",
            str(chunk_size),
        ]
    )


def native_optimize_from_tree(output: Path, fallback: str) -> str:
    """Prefer optimize recorded in an existing native/manifest.json."""
    path = output / "native" / "manifest.json"
    if not path.is_file():
        return fallback
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback
    recorded = payload.get("optimize")
    if isinstance(recorded, str) and recorded:
        return recorded
    return fallback


def write_top_manifest(output: Path, version: str, optimize: str) -> None:
    native_manifest = json.loads((output / "native" / "manifest.json").read_text(encoding="utf-8"))
    models_manifest = json.loads((output / "models" / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "schema": "augmem.cortext.release_assets.v1",
        "cortext_version": version,
        "optimize": optimize,
        "layout": {
            "native": "native/",
            "models": "models/",
        },
        "native": native_manifest,
        "models": models_manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def write_tarball(output: Path, version: str, tarball: Path) -> None:
    del version  # encoded in the tarball filename by the caller
    tarball = tarball.expanduser().resolve()
    tarball.parent.mkdir(parents=True, exist_ok=True)
    if tarball.exists():
        tarball.unlink()
    # Archive contents are rooted at native/ and models/ (plus top manifest).
    # Never pack the destination tarball (or its .sha256 sibling) if either
    # path lives under --output.
    skip_paths = {tarball, Path(str(tarball) + ".sha256")}
    with tarfile.open(tarball, "w:gz") as tar:
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            if path.resolve() in skip_paths:
                continue
            tar.add(path, arcname=path.relative_to(output).as_posix())
    print(f"wrote {tarball} ({tarball.stat().st_size / (1024 * 1024):.2f} MiB)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for the unpacked asset tree (default: dist/release-assets).",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version string recorded in manifests and tarball name (default: detect).",
    )
    parser.add_argument("--zig", default=os.environ.get("ZIG", "zig"))
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Package target tag. May be repeated. Default: all six.",
    )
    parser.add_argument(
        "--optimize",
        default="ReleaseSmall",
        choices=("Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall"),
    )
    parser.add_argument(
        "--skip-zig-build",
        action="store_true",
        help="Reuse existing prefixes under build/release-assets/native.",
    )
    parser.add_argument(
        "--from-python-natives",
        action="store_true",
        help="Copy natives from bindings/python/.../native instead of zig build.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Only build/copy natives (no model shards).",
    )
    parser.add_argument(
        "--skip-natives",
        action="store_true",
        help="Only shard models (reuse existing output/native if present).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        help=f"Model part size in bytes (default {DEFAULT_CHUNK}; must be > 0).",
    )
    parser.add_argument(
        "--skip-tarball",
        action="store_true",
        help="Do not write dist/cortext-assets-<version>.tar.gz.",
    )
    parser.add_argument(
        "--tarball",
        type=Path,
        default=None,
        help="Override tarball path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = detect_version(args.version)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"version={version}", flush=True)
    print(f"output={output}", flush=True)

    targets = select_targets(args.target)
    optimize = args.optimize
    if args.chunk_size <= 0:
        raise SystemExit(f"--chunk-size must be a positive integer (got {args.chunk_size})")

    if not args.skip_natives:
        if args.from_python_natives:
            entries = copy_from_python_natives(targets)
            # Strip is already applied in package natives.
            for entry in entries:
                entry["source"] = Path(str(entry["source"]))
            optimize = "ReleaseSmall"
        else:
            if not shutil.which(args.zig) and not args.skip_zig_build:
                raise SystemExit(f"zig not found: {args.zig}")
            entries = [
                build_native(
                    target,
                    args.zig,
                    optimize,
                    args.skip_zig_build,
                    DEFAULT_BUILD_ROOT,
                )
                for target in targets
            ]
            for entry, target in zip(entries, targets):
                # Strip on a temp copy path after install — strip source if strip tools exist
                maybe_strip(Path(str(entry["source"])), target)
        install_natives(output, entries, optimize)
    elif not (output / "native" / "manifest.json").is_file():
        raise SystemExit("--skip-natives requires an existing native tree under --output")
    else:
        # Reuse existing natives; keep top-level optimize consistent with them.
        optimize = native_optimize_from_tree(output, optimize)

    if not args.skip_models:
        shard_models(output, version, args.chunk_size)
    elif not (output / "models" / "manifest.json").is_file():
        raise SystemExit("--skip-models requires an existing models tree under --output")

    write_top_manifest(output, version, optimize)

    if not args.skip_tarball:
        tarball = args.tarball
        if tarball is None:
            tarball = REPO_ROOT / "dist" / f"cortext-assets-{version}.tar.gz"
        else:
            tarball = tarball.expanduser().resolve()
        write_tarball(output, version, tarball)
        # Convenience checksum file for release notes / installers.
        # Always append ".sha256" (foo.tar.gz → foo.tar.gz.sha256).
        sums = Path(str(tarball) + ".sha256")
        sums.write_text(f"{digest_file(tarball)}  {tarball.name}\n", encoding="utf-8")
        print(f"wrote {sums}", flush=True)

    print("release assets ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
