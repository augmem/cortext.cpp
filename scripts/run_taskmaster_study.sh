#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-logs/taskmaster_study_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HARNESS="${HARNESS:-scripts/run_memory_harness.py}"
DATA="${DATA:-data/taskmaster/valid.jsonl}"

if [[ ! -f "${DATA}" ]]; then
  echo "Missing ${DATA}. Run scripts/prepare_taskmaster.py first." >&2
  exit 1
fi

mkdir -p "${ROOT}"

COMMON_ARGS=(
  --binary build/examples/topical_chat_analysis/cortext_topical_chat_analysis
  --data "${DATA}"
  --no-baseline
  --no-multi
  --cases "0.5,0.5,0.5"
)

run_case() {
  local name="$1"
  shift
  echo "[run] ${name}"
  "${PYTHON_BIN}" "${HARNESS}" "${COMMON_ARGS[@]}" --out "${ROOT}/${name}" "$@"
}

run_case procedural_default \
  --max-conversations 4 \
  --max-turns 156 \
  --max-total 156 \
  --consolidate-cycles 2 \
  --extra-args=--consolidate-every=40

run_case procedural_both_off \
  --max-conversations 4 \
  --max-turns 156 \
  --max-total 156 \
  --consolidate-cycles 2 \
  --extra-args="--consolidate-every=40 --no-procedural --no-sequential-edges"

echo "Wrote Taskmaster study runs to ${ROOT}"
