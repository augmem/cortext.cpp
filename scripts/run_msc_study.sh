#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-logs/msc_study_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HARNESS="${HARNESS:-scripts/run_memory_harness.py}"
DATA="${DATA:-data/msc/valid.jsonl}"
MODELS_DIR="${MODELS_DIR:-models}"

if [[ ! -f "${DATA}" ]]; then
  echo "Missing ${DATA}. Run scripts/prepare_msc.py first." >&2
  exit 1
fi

mkdir -p "${ROOT}"

export CORTEXT_EMBEDDINGGEMMA_BACKEND=llama.cpp
export CORTEXT_DEEP_LLM_BACKEND=auto

COMMON_ARGS=(
  --binary build/examples/topical_chat_analysis/cortext_topical_chat_analysis
  --models "${MODELS_DIR}"
  --data "${DATA}"
  --label-bank ""
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

run_case consolidation_off \
  --max-conversations 8 \
  --max-turns 360 \
  --max-total 720 \
  --consolidate-cycles 0

run_case consolidation_on \
  --max-conversations 8 \
  --max-turns 360 \
  --max-total 720 \
  --consolidate-cycles 2

echo "Wrote MSC study runs to ${ROOT}"
