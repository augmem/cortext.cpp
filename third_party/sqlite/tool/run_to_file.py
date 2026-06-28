#!/usr/bin/env python3
"""Run a command and write stdout to a file."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: run_to_file.py OUT COMMAND [ARG...]", file=sys.stderr)
        return 2
    output = Path(argv[1])
    command = argv[2:]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    output.write_bytes(result.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
