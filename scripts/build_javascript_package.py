#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "bindings" / "javascript"
PACKAGE_PREBUILDS = PACKAGE_ROOT / "prebuilds"
PACKAGE_MODELS = PACKAGE_ROOT / "models"
SOURCE_MODELS = REPO_ROOT / "models"
BUILD_ROOT = PACKAGE_ROOT / "build" / "native"
DIST_ROOT = PACKAGE_ROOT / "dist"
GLOBAL_CACHE = PACKAGE_ROOT / "build" / "zig-global-cache"
NAPI_VERSION = 8

NAPI_IMPORT_SYMBOLS = (
    "napi_define_class",
    "napi_set_named_property",
    "napi_get_last_error_info",
    "napi_throw_error",
    "napi_define_properties",
    "napi_get_cb_info",
    "napi_throw_type_error",
    "napi_typeof",
    "napi_get_value_int32",
    "napi_get_undefined",
    "napi_wrap",
    "napi_create_string_utf8",
    "napi_unwrap",
    "napi_get_value_string_utf8",
    "napi_has_named_property",
    "napi_get_named_property",
    "napi_get_value_bool",
    "napi_is_typedarray",
    "napi_get_typedarray_info",
    "napi_is_buffer",
    "napi_get_buffer_info",
    "napi_get_value_double",
)


@dataclass(frozen=True, slots=True)
class Target:
    package_tag: str
    zig_target: str
    artifact_globs: tuple[str, ...]
    dlltool_machine: str | None = None


TARGETS = (
    Target("linux-x64", "x86_64-linux-gnu.2.17", ("lib/libcortext_node.so",)),
    Target("linux-arm64", "aarch64-linux-gnu.2.17", ("lib/libcortext_node.so",)),
    Target("darwin-x64", "x86_64-macos", ("lib/libcortext_node.dylib",)),
    Target("darwin-arm64", "aarch64-macos", ("lib/libcortext_node.dylib",)),
    Target("win32-x64", "x86_64-windows-gnu", ("bin/cortext_node.dll",), "i386:x86-64"),
    Target("win32-arm64", "aarch64-windows-gnu", ("bin/cortext_node.dll",), "arm64"),
)


def run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def output(cmd: list[str], *, cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def node_include_dir() -> Path:
    override = os.environ.get("NODE_INCLUDE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    script = "require('path').join(require('path').dirname(require('path').dirname(process.execPath)), 'include', 'node')"
    return Path(output(["node", "-p", script])).resolve()


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


def clean_prebuilds() -> None:
    if PACKAGE_PREBUILDS.exists():
        shutil.rmtree(PACKAGE_PREBUILDS)
    PACKAGE_PREBUILDS.mkdir(parents=True, exist_ok=True)
    (PACKAGE_PREBUILDS / ".gitkeep").touch()


def clean_package_models() -> None:
    if PACKAGE_MODELS.exists():
        shutil.rmtree(PACKAGE_MODELS)
    PACKAGE_MODELS.mkdir(parents=True, exist_ok=True)
    (PACKAGE_MODELS / ".gitkeep").touch()


def ensure_source_models(quant: str) -> None:
    run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "download_aist_model.py"),
            "--output-dir",
            str(SOURCE_MODELS),
            "--quant",
            quant,
        ]
    )


def copy_package_models(quant: str) -> None:
    ensure_source_models(quant)
    clean_package_models()

    model_src = SOURCE_MODELS / "AIST-87M-GGUF" / f"AIST-87M_{quant}.gguf"
    vocab_src = SOURCE_MODELS / "mdbr-leaf-ir" / "vocab.txt"
    if not model_src.is_file():
        raise FileNotFoundError(f"missing AIST model: {model_src}")
    if not vocab_src.is_file():
        raise FileNotFoundError(f"missing tokenizer vocab: {vocab_src}")

    model_dest = PACKAGE_MODELS / "AIST-87M-GGUF" / model_src.name
    vocab_dest = PACKAGE_MODELS / "mdbr-leaf-ir" / "vocab.txt"
    model_dest.parent.mkdir(parents=True, exist_ok=True)
    vocab_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_src, model_dest)
    shutil.copy2(vocab_src, vocab_dest)
    print(
        f"bundled model assets: {model_dest} ({model_dest.stat().st_size / (1024 * 1024):.2f} MiB)",
        flush=True,
    )


def find_artifact(prefix: Path, target: Target) -> Path:
    matches: list[Path] = []
    for pattern in target.artifact_globs:
        matches.extend(path for path in prefix.glob(pattern) if path.is_file() and not path.is_symlink())
    if not matches:
        searched = ", ".join(str(prefix / pattern) for pattern in target.artifact_globs)
        raise FileNotFoundError(f"could not find {target.package_tag} addon; searched {searched}")
    return max(matches, key=lambda path: (path.stat().st_size, len(path.name), path.name))


def build_node_import_lib(target: Target, zig: str) -> Path | None:
    if target.dlltool_machine is None:
        return None
    node_import_dir = BUILD_ROOT / "node-import-lib" / target.package_tag
    node_import_dir.mkdir(parents=True, exist_ok=True)
    def_path = node_import_dir / "node.def"
    lib_path = node_import_dir / "node.lib"
    exports = "".join(f"  {symbol}\n" for symbol in NAPI_IMPORT_SYMBOLS)
    def_path.write_text(f"LIBRARY node.exe\nEXPORTS\n{exports}", encoding="utf-8")
    run([zig, "dlltool", "-m", target.dlltool_machine, "-d", str(def_path), "-l", str(lib_path)])
    return lib_path


def build_native(target: Target, zig: str, optimize: str, include_dir: Path, skip_zig_build: bool) -> dict[str, object]:
    prefix = BUILD_ROOT / "prefix" / target.package_tag
    cache = BUILD_ROOT / "zig-cache" / target.package_tag
    if not skip_zig_build and prefix.exists():
        shutil.rmtree(prefix)
    prefix.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    GLOBAL_CACHE.mkdir(parents=True, exist_ok=True)

    node_lib = build_node_import_lib(target, zig)
    if not skip_zig_build:
        cmd = [
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
            "-Dshared=false",
            "-Dcli=false",
            "-Dnode-addon=true",
            f"-Dnode-include={include_dir}",
            "-Dfetch-aist-model=false",
        ]
        if node_lib is not None:
            cmd.append(f"-Dnode-lib={node_lib}")
        run(cmd)

    artifact = find_artifact(prefix, target)
    target_dir = PACKAGE_PREBUILDS / target.package_tag
    target_dir.mkdir(parents=True, exist_ok=True)
    bundled = target_dir / "cortext.node"
    shutil.copy2(artifact, bundled)
    size = bundled.stat().st_size
    print(f"bundled {target.package_tag}: {bundled} ({size / (1024 * 1024):.2f} MiB)", flush=True)
    return {
        "package_tag": target.package_tag,
        "zig_target": target.zig_target,
        "artifact": "cortext.node",
        "size": size,
    }


def write_manifest(entries: list[dict[str, object]], optimize: str) -> None:
    manifest = {
        "schema": "augmem.cortext.node.prebuilds.v1",
        "napi": NAPI_VERSION,
        "optimize": optimize,
        "targets": entries,
    }
    (PACKAGE_PREBUILDS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def pack_package() -> None:
    if DIST_ROOT.exists():
        shutil.rmtree(DIST_ROOT)
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    run(["npm", "pack", "--pack-destination", str(DIST_ROOT)], cwd=PACKAGE_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the @augmem/cortext npm package with bundled Zig N-API addons.")
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
        help="Zig optimization mode for bundled addons. Default: ReleaseSmall.",
    )
    parser.add_argument("--skip-zig-build", action="store_true", help="Reuse existing native prefixes under bindings/javascript/build/native.")
    parser.add_argument("--skip-models", action="store_true", help="Do not bundle model assets into the package.")
    parser.add_argument("--model-quant", default="q8_0", choices=("q8_0", "q5_1"), help="AIST model quantization to bundle. Default: q8_0.")
    parser.add_argument("--skip-pack", action="store_true", help="Build native addons only; do not run npm pack.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = select_targets(args.target)
    include_dir = node_include_dir()
    if not (include_dir / "node_api.h").is_file():
        raise FileNotFoundError(f"node_api.h not found under {include_dir}")
    clean_prebuilds()
    entries = [build_native(target, args.zig, args.optimize, include_dir, args.skip_zig_build) for target in targets]
    write_manifest(entries, args.optimize)
    if args.skip_models:
        clean_package_models()
    else:
        copy_package_models(args.model_quant)
    if not args.skip_pack:
        pack_package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
