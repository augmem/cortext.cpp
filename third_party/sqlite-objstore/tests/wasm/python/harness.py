#!/usr/bin/env python3
"""Python WASI harness for the objstore WASM matrix."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
from typing import Any, Dict, List

try:
    import wasmtime  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    print("error: wasmtime (Python) must be installed (pip install wasmtime)", file=sys.stderr)
    raise


def _load_matrix(bundle: pathlib.Path) -> List[Dict[str, Any]]:
    matrix_path = bundle / "matrix.json"
    data = json.loads(matrix_path.read_text())
    fixtures = data.get("fixtures", [])
    if not fixtures:
        raise RuntimeError(f"no fixtures defined in {matrix_path}")
    return fixtures


def _run_fixture(engine: wasmtime.Engine, module: wasmtime.Module, bundle: pathlib.Path, fixture: Dict[str, Any]) -> Dict[str, Any]:
    linker = wasmtime.Linker(engine)
    linker.define_wasi()

    store = wasmtime.Store(engine)
    wasi = wasmtime.WasiConfig()
    wasi.argv = [
        "objstore_wasm_matrix",
        "--bundle-root",
        "/bundle",
        "--fixture",
        fixture["path"],
        "--fixture-id",
        fixture["id"],
    ]
    wasi.preopen_dir(str(bundle), "/bundle")

    stdout_tmp = tempfile.NamedTemporaryFile(delete=False)
    stderr_tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        stdout_tmp.close()
        stderr_tmp.close()
        wasi.set_stdout_file(stdout_tmp.name)
        wasi.set_stderr_file(stderr_tmp.name)
        store.set_wasi(wasi)
        instance = linker.instantiate(store, module)
        start = instance.exports(store)["_start"]
        start(store)
        output = pathlib.Path(stdout_tmp.name).read_text().strip()
        if not output:
            stderr_contents = pathlib.Path(stderr_tmp.name).read_text()
            raise RuntimeError(f"fixture {fixture['id']} produced no output; stderr: {stderr_contents}")
        last_line = output.splitlines()[-1]
        result = json.loads(last_line)
        result.setdefault("stderr", pathlib.Path(stderr_tmp.name).read_text())
        return result
    finally:
        try:
            os.unlink(stdout_tmp.name)
        except FileNotFoundError:
            pass
        try:
            os.unlink(stderr_tmp.name)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the objstore WASM matrix via wasmtime-py")
    parser.add_argument("--bundle", type=pathlib.Path, default=pathlib.Path("dist/wasm/dev"), help="Path to the packaged bundle (default: dist/wasm/dev)")
    parser.add_argument("--module", type=pathlib.Path, help="Override path to the WASI matrix module", default=None)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    if not bundle.exists():
        parser.error(f"bundle path {bundle} does not exist")

    module_path = args.module.resolve() if args.module else (bundle / "wasi" / "objstore_wasm_matrix.wasm")
    if not module_path.exists():
        parser.error(f"module path {module_path} does not exist")

    engine = wasmtime.Engine()
    module = wasmtime.Module.from_file(engine, str(module_path))
    fixtures = _load_matrix(bundle)
    failures: List[Dict[str, Any]] = []

    for fixture in fixtures:
        result = _run_fixture(engine, module, bundle, fixture)
        status = result.get("status")
        label = fixture.get("id")
        print(f"[python] {label}: {status}")
        if status != "ok":
            failures.append(result)

    if failures:
        print("python harness failures:")
        print(json.dumps(failures, indent=2))
        return 1
    print("python harness passed all fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
