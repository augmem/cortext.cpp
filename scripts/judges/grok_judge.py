#!/usr/bin/env python3
"""Grok CLI adapter for memory-eval harness judging."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def read_prompt() -> str:
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        return Path(sys.argv[1]).read_text(encoding="utf-8")
    return sys.stdin.read()


def main() -> int:
    prompt = read_prompt()
    grok_bin = os.environ.get("CORTEXT_GROK_BIN", "grok")
    timeout_s = int(os.environ.get("CORTEXT_JUDGE_TIMEOUT_S", "180"))
    model = os.environ.get("CORTEXT_GROK_MODEL", "")
    extra_args = shlex.split(os.environ.get("CORTEXT_GROK_ARGS", ""))

    with tempfile.TemporaryDirectory(prefix="cortext_grok_judge_") as temp_dir:
        prompt_path = Path(temp_dir) / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        cmd = [
            grok_bin,
            "--output-format",
            "plain",
            "--no-memory",
            "--disable-web-search",
            "--max-turns",
            "1",
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.extend(extra_args)
        cmd.extend(["--prompt-file", str(prompt_path)])
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        if proc.stdout:
            sys.stderr.write(proc.stdout)
        return proc.returncode
    answer = proc.stdout.strip()
    if answer:
        print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
