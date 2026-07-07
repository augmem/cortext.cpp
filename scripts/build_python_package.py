#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "bindings" / "python"
PACKAGE_NATIVE = PYTHON_ROOT / "augmem" / "cortext" / "native"
BUILD_ROOT = PYTHON_ROOT / "build" / "native"
DIST_ROOT = PYTHON_ROOT / "dist"
GLOBAL_CACHE = PYTHON_ROOT / "build" / "zig-global-cache"


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


def select_targets(names: list[str]) -> list[Target]:
    if not names:
        return list(TARGETS)
    by_name = {target.package_tag: target for target in TARGETS}
    selected: list[Target] = []
    for name in names:
        try:
            selected.append(by_name[name])
        except KeyError as exc:
            supported = ", ".join(sorted(by_name))
            raise SystemExit(f"unknown target {name!r}; supported targets: {supported}") from exc
    return selected


def clean_package_native() -> None:
    if PACKAGE_NATIVE.exists():
        shutil.rmtree(PACKAGE_NATIVE)
    PACKAGE_NATIVE.mkdir(parents=True, exist_ok=True)
    (PACKAGE_NATIVE / ".gitkeep").touch()


def find_artifact(prefix: Path, target: Target) -> Path:
    matches: list[Path] = []
    for pattern in target.artifact_globs:
        matches.extend(path for path in prefix.glob(pattern) if path.is_file() and not path.is_symlink())
    if not matches:
        searched = ", ".join(str(prefix / pattern) for pattern in target.artifact_globs)
        raise FileNotFoundError(f"could not find {target.package_tag} shared library; searched {searched}")
    return max(matches, key=lambda path: (path.stat().st_size, len(path.name), path.name))


def maybe_strip(path: Path, target: Target) -> None:
    for tool in target.strip_tools:
        resolved = shutil.which(tool)
        if resolved is None:
            continue
        try:
            run([resolved, "-s", str(path)])
            return
        except subprocess.CalledProcessError:
            continue


def build_native(target: Target, zig: str, optimize: str, skip_zig_build: bool) -> dict[str, object]:
    prefix = BUILD_ROOT / "prefix" / target.package_tag
    cache = BUILD_ROOT / "zig-cache" / target.package_tag
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
                "-Dfetch-aist-model=false",
            ]
        )

    artifact = find_artifact(prefix, target)
    target_dir = PACKAGE_NATIVE / target.package_tag
    target_dir.mkdir(parents=True, exist_ok=True)
    bundled = target_dir / target.artifact_name
    shutil.copy2(artifact, bundled)
    maybe_strip(bundled, target)
    size = bundled.stat().st_size
    print(f"bundled {target.package_tag}: {bundled} ({size / (1024 * 1024):.2f} MiB)", flush=True)
    return {
        "package_tag": target.package_tag,
        "zig_target": target.zig_target,
        "artifact": target.artifact_name,
        "size": size,
    }


def write_manifest(entries: list[dict[str, object]], optimize: str) -> None:
    manifest = {
        "schema": "augmem.cortext.native.v1",
        "optimize": optimize,
        "targets": entries,
    }
    (PACKAGE_NATIVE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def has_module(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def build_wheel(build_sdist: bool, check: bool) -> None:
    if DIST_ROOT.exists():
        shutil.rmtree(DIST_ROOT)
    if has_module("build.__main__"):
        cmd = [sys.executable, "-m", "build", "--wheel", "--outdir", str(DIST_ROOT), str(PYTHON_ROOT)]
        if build_sdist:
            cmd.remove("--wheel")
        run(cmd)
    else:
        if build_sdist:
            raise RuntimeError("building an sdist requires the `build` package; install it with `python -m pip install build`")
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(DIST_ROOT), str(PYTHON_ROOT)])

    if check:
        if not has_module("twine"):
            print("warning: twine is not installed; skipping `twine check`", flush=True)
            return
        twine = [sys.executable, "-m", "twine", "check"]
        artifacts = sorted(str(path) for path in DIST_ROOT.iterdir() if path.suffix in {".whl", ".gz"})
        if artifacts:
            run([*twine, *artifacts])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the augmem.cortext PyPI package with bundled Zig libraries.")
    parser.add_argument("--zig", default=os.environ.get("ZIG", "zig"), help="Path to the Zig executable.")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Package target tag to build. May be passed more than once. Default: all targets.",
    )
    parser.add_argument(
        "--optimize",
        default="ReleaseSmall",
        choices=("Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall"),
        help="Zig optimization mode for bundled libraries. Default: ReleaseSmall.",
    )
    parser.add_argument("--skip-zig-build", action="store_true", help="Reuse existing native prefixes under bindings/python/build/native.")
    parser.add_argument("--skip-dist", action="store_true", help="Build native libraries only; do not build a Python distribution.")
    parser.add_argument("--sdist", action="store_true", help="Build both wheel and source distribution. Default: wheel only.")
    parser.add_argument("--skip-twine-check", action="store_true", help="Do not run `twine check` after building distributions.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = select_targets(args.target)
    clean_package_native()
    entries = [build_native(target, args.zig, args.optimize, args.skip_zig_build) for target in targets]
    write_manifest(entries, args.optimize)
    if not args.skip_dist:
        build_wheel(args.sdist, not args.skip_twine_check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
