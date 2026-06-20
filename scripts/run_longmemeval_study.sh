#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-logs/longmemeval_study_$(date +%Y%m%d_%H%M%S)}"
PREDICTIONS="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HARNESS="${HARNESS:-scripts/run_memory_harness.py}"
DATA="${DATA:-data/longmemeval/valid.jsonl}"
ANSWER_KEY="${ANSWER_KEY:-data/longmemeval/valid.answer_key.jsonl}"
MODELS_DIR="${MODELS_DIR:-models}"

if [[ ! -f "${DATA}" ]]; then
  echo "Missing ${DATA}. Run scripts/prepare_longmemeval.py first." >&2
  exit 1
fi

mkdir -p "${ROOT}"

"${PYTHON_BIN}" "${HARNESS}" \
  --binary build/examples/topical_chat_analysis/cortext_topical_chat_analysis \
  --models "${MODELS_DIR}" \
  --data "${DATA}" \
  --out "${ROOT}/retrieval_smoke" \
  --no-multi \
  --no-sweep \
  --max-conversations 4 \
  --max-turns 120 \
  --max-total 120 \
  --consolidate-cycles 0

if [[ -n "${PREDICTIONS}" ]]; then
  "${PYTHON_BIN}" scripts/score_longmemeval.py \
    --answer-key "${ANSWER_KEY}" \
    --predictions "${PREDICTIONS}" \
    --out "${ROOT}/longmemeval_score.json"
fi

echo "Wrote LongMemEval study artifacts to ${ROOT}"
