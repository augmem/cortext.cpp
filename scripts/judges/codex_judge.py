#!/usr/bin/env python3
"""Codex CLI adapter for memory-eval harness judging.

Reads the judge prompt from argv[1] or stdin and writes only the final answer to
stdout. Codex's session diagnostics are forwarded to stderr so the scorer does
not accidentally score prompt text or logs as the prediction.
"""

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
    codex_bin = os.environ.get("CORTEXT_CODEX_BIN", "codex")
    timeout_s = int(os.environ.get("CORTEXT_JUDGE_TIMEOUT_S", "180"))
    effort = os.environ.get("CORTEXT_CODEX_EFFORT", "low")
    model = os.environ.get("CORTEXT_CODEX_MODEL", "")
    extra_args = shlex.split(os.environ.get("CORTEXT_CODEX_ARGS", ""))

    with tempfile.TemporaryDirectory(prefix="cortext_codex_judge_") as temp_dir:
        answer_path = Path(temp_dir) / "answer.txt"
        cmd = [
            codex_bin,
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-c",
            f'model_reasoning_effort="{effort}"',
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.extend(extra_args)
        cmd.extend(["-o", str(answer_path), "-"])
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        if proc.stdout:
            sys.stderr.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            return proc.returncode
        answer = answer_path.read_text(encoding="utf-8").strip()
        if not answer:
            answer = proc.stdout.strip()
    if answer:
        print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
