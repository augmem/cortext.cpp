#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-logs/embeddinggemma_remaining_historical_ablations_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HARNESS="${HARNESS:-scripts/run_memory_harness.py}"
MODELS_DIR="${MODELS_DIR:-models}"
TOPICAL_DATA="${TOPICAL_DATA:-data/topical_chat/valid_freq.jsonl}"
EMPATHETIC_DATA="${EMPATHETIC_DATA:-data/empathetic_dialogues/valid.jsonl}"

mkdir -p "${ROOT}"

export CORTEXT_EMBEDDINGGEMMA_BACKEND=llama.cpp
export CORTEXT_DEEP_LLM_BACKEND=auto
unset CORTEXT_LFM2_SUMMARIZER_MODEL
unset CORTEXT_LFM2_EXTRACT_MODEL

COMMON_ARGS=(
  --binary build/examples/topical_chat_analysis/cortext_topical_chat_analysis
  --models "${MODELS_DIR}"
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
  --data "${TOPICAL_DATA}" \
  --max-conversations 10 \
  --max-turns 360 \
  --max-total 720 \
  --consolidate-cycles 0

run_case consolidation_on \
  --data "${TOPICAL_DATA}" \
  --max-conversations 10 \
  --max-turns 360 \
  --max-total 720 \
  --consolidate-cycles 2

run_case procseq_base_e40 \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 200 \
  --max-total 200 \
  --consolidate-cycles 2 \
  --extra-args=--consolidate-every=40

run_case procseq_bothoff_e40 \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 200 \
  --max-total 200 \
  --consolidate-cycles 2 \
  --extra-args="--consolidate-every=40 --no-procedural --no-sequential-edges"

run_case procseq_base_e60 \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 200 \
  --max-total 200 \
  --consolidate-cycles 2 \
  --extra-args=--consolidate-every=60

run_case procseq_bothoff_e60 \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 200 \
  --max-total 200 \
  --consolidate-cycles 2 \
  --extra-args="--consolidate-every=60 --no-procedural --no-sequential-edges"

run_case short_affect_all \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 120 \
  --max-total 120 \
  --consolidate-cycles 0 \
  --affect-mode all

run_case short_affect_interrupt \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 120 \
  --max-total 120 \
  --consolidate-cycles 0 \
  --affect-mode interrupt

run_case short_affect_retrieval \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 120 \
  --max-total 120 \
  --consolidate-cycles 0 \
  --affect-mode retrieval

run_case short_affect_off \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 120 \
  --max-total 120 \
  --consolidate-cycles 0 \
  --affect-mode off

run_case short_proc_no_procedural \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 120 \
  --max-total 120 \
  --consolidate-cycles 0 \
  --extra-args=--no-procedural

run_case short_proc_no_sequential \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 120 \
  --max-total 120 \
  --consolidate-cycles 0 \
  --extra-args=--no-sequential-edges

run_case short_proc_both_off \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 120 \
  --max-total 120 \
  --consolidate-cycles 0 \
  --extra-args="--no-procedural --no-sequential-edges"

run_case short_source_conf_on \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 120 \
  --max-total 120 \
  --consolidate-cycles 0

CORTEXT_DISABLE_SOURCE_CONF=1 run_case short_source_conf_off \
  --data "${TOPICAL_DATA}" \
  --max-conversations 1 \
  --max-turns 120 \
  --max-total 120 \
  --consolidate-cycles 0

run_case empathetic_affect_all_s05 \
  --data "${EMPATHETIC_DATA}" \
  --max-conversations 6 \
  --max-turns 360 \
  --max-total 360 \
  --consolidate-cycles 2 \
  --cases "0.5,0.5,0.5" \
  --affect-mode all

run_case empathetic_affect_off_s05 \
  --data "${EMPATHETIC_DATA}" \
  --max-conversations 6 \
  --max-turns 360 \
  --max-total 360 \
  --consolidate-cycles 2 \
  --cases "0.5,0.5,0.5" \
  --affect-mode off

run_case empathetic_affect_all_s10 \
  --data "${EMPATHETIC_DATA}" \
  --max-conversations 6 \
  --max-turns 360 \
  --max-total 360 \
  --consolidate-cycles 2 \
  --cases "0.5,1.0,0.5" \
  --affect-mode all

run_case empathetic_affect_off_s10 \
  --data "${EMPATHETIC_DATA}" \
  --max-conversations 6 \
  --max-turns 360 \
  --max-total 360 \
  --consolidate-cycles 2 \
  --cases "0.5,1.0,0.5" \
  --affect-mode off

run_case empathetic_long_affect_all \
  --data "${EMPATHETIC_DATA}" \
  --max-conversations 10 \
  --max-turns 360 \
  --max-total 720 \
  --consolidate-cycles 2 \
  --cases "0.5,0.5,0.5" \
  --affect-mode all

run_case empathetic_long_affect_off \
  --data "${EMPATHETIC_DATA}" \
  --max-conversations 10 \
  --max-turns 360 \
  --max-total 720 \
  --consolidate-cycles 2 \
  --cases "0.5,0.5,0.5" \
  --affect-mode off

echo "Wrote reruns to ${ROOT}"
