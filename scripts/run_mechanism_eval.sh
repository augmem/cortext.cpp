#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-logs/mechanism_eval_$(date +%Y%m%d_%H%M%S)}"
PREDICTIONS="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HARNESS="${HARNESS:-scripts/run_memory_harness.py}"
DATA="${DATA:-data/mechanism_eval/valid.jsonl}"
ANSWER_KEY="${ANSWER_KEY:-data/mechanism_eval/valid.answer_key.jsonl}"
MODELS_DIR="${MODELS_DIR:-models}"

mkdir -p "${ROOT}"

if [[ ! -f "${DATA}" ]]; then
  "${PYTHON_BIN}" scripts/generate_mechanism_eval_pack.py --out-dir "$(dirname "${DATA}")"
fi

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
  --max-conversations 6
  --max-turns 120
  --max-total 120
  --consolidate-cycles 0
)

"${PYTHON_BIN}" "${HARNESS}" "${COMMON_ARGS[@]}" --out "${ROOT}/source_conf_on"
CORTEXT_DISABLE_SOURCE_CONF=1 \
  "${PYTHON_BIN}" "${HARNESS}" "${COMMON_ARGS[@]}" --out "${ROOT}/source_conf_off"

if [[ -n "${PREDICTIONS}" ]]; then
  "${PYTHON_BIN}" scripts/score_mechanism_eval.py \
    --answer-key "${ANSWER_KEY}" \
    --predictions "${PREDICTIONS}" \
    --out "${ROOT}/mechanism_score.json"
fi

echo "Wrote mechanism-eval artifacts to ${ROOT}"
