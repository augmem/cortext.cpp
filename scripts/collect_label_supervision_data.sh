#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" scripts/prepare_empathetic_dialogues.py --split all
"${PYTHON_BIN}" scripts/download_meld.py
"${PYTHON_BIN}" scripts/prepare_meld.py
"${PYTHON_BIN}" scripts/download_goemotions.py
"${PYTHON_BIN}" scripts/prepare_goemotions.py
"${PYTHON_BIN}" scripts/download_taskmaster.py
"${PYTHON_BIN}" scripts/prepare_taskmaster.py --split all
"${PYTHON_BIN}" scripts/download_personachat.py
"${PYTHON_BIN}" scripts/prepare_personachat.py

echo "[OK] state data ready under data/empathetic_dialogues"
echo "[OK] dialogue emotion state data ready under data/meld"
echo "[OK] broad emotion state data ready under data/goemotions"
echo "[OK] topic data ready under data/taskmaster"
echo "[OK] identity data ready under data/personachat"
