#!/usr/bin/env python3
"""Antigravity/AGY adapter for memory-eval harness judging."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def read_prompt() -> str:
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        return Path(sys.argv[1]).read_text(encoding="utf-8")
    return sys.stdin.read()


def configured_command(prompt: str) -> tuple[str | list[str], bool]:
    configured = os.environ.get("CORTEXT_AGY_COMMAND", "")
    if configured:
        return configured, True
    agy_bin = os.environ.get("CORTEXT_AGY_BIN", "")
    if not agy_bin:
        agy_bin = shutil.which("agy") or shutil.which("antigravity") or ""
    if not agy_bin:
        return "", False
    model = os.environ.get("CORTEXT_AGY_MODEL", "")
    extra_args = shlex.split(os.environ.get("CORTEXT_AGY_ARGS", ""))
    cmd = [agy_bin]
    if model:
        cmd.extend(["--model", model])
    cmd.extend(extra_args)
    cmd.extend(["--print", prompt])
    return cmd, False


def main() -> int:
    prompt = read_prompt()
    timeout_s = int(os.environ.get("CORTEXT_JUDGE_TIMEOUT_S", "180"))
    command, use_shell = configured_command(prompt)
    if not command:
        sys.stderr.write(
            "AGY is not installed and CORTEXT_AGY_COMMAND is not set.\n"
        )
        return 127

    if use_shell:
        with tempfile.TemporaryDirectory(prefix="cortext_agy_judge_") as temp_dir:
            prompt_path = Path(temp_dir) / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            expanded = str(command).replace("{prompt_file}", shlex.quote(str(prompt_path)))
            proc = subprocess.run(
                expanded,
                input=prompt,
                text=True,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
            )
    else:
        proc = subprocess.run(
            command,
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
