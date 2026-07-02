#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-eval_runs/msc_frontier_gpt55_$(date -u +%Y%m%dT%H%M%SZ)}"
SPLIT="${SPLIT:-validation}"
MAX_DIALOGS="${MAX_DIALOGS:-0}"
MAX_SESSIONS="${MAX_SESSIONS:-0}"
MODELS_DIR="${MODELS_DIR:-models}"
BUILD_DIR="${BUILD_DIR:-build}"
ENV_FILE="${CORTEXT_EVAL_ENV_FILE:-${ENV_FILE:-}}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.5}"
COMPACTION_MODEL="${COMPACTION_MODEL:-${JUDGE_MODEL}}"
JUDGE_REPETITIONS="${JUDGE_REPETITIONS:-3}"
JUDGE_LIMIT="${JUDGE_LIMIT:--1}"
JUDGE_CONTEXT_WINDOW_TOKENS="${JUDGE_CONTEXT_WINDOW_TOKENS:-1000000}"
JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-8192}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
WARMUP_EVENTS="${WARMUP_EVENTS:-20}"
PROBE_STRIDE="${PROBE_STRIDE:-20}"
RAG_TOP_K="${RAG_TOP_K:-5}"
ACTIVE_HISTORY_TOKEN_BUDGET="${ACTIVE_HISTORY_TOKEN_BUDGET:-49152}"
COMPACTING_SESSION_BUDGET="${COMPACTING_SESSION_BUDGET:-49152}"
COMPACTING_SESSION_TRIGGER="${COMPACTING_SESSION_TRIGGER:-0.8}"
COMPACTING_SESSION_MAX_OUTPUT_TOKENS="${COMPACTING_SESSION_MAX_OUTPUT_TOKENS:-700}"

mkdir -p "${ROOT}"

if [[ -z "${ENV_FILE}" ]]; then
  if [[ -f .env ]]; then
    ENV_FILE=.env
  fi
fi

export LIBRARY_PATH="${HOME}/.local/lib:${LIBRARY_PATH:-}"

if [[ ! -x "${BUILD_DIR}/examples/benchmark/cortext_chat_replay_live_run" ]]; then
  cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCORTEXT_BUILD_EXAMPLES=ON \
    -DCORTEXT_DISABLE_OPENTELEMETRY=ON
  cmake --build "${BUILD_DIR}" --target cortext_chat_replay_live_run -j
fi

python3 scripts/materialize_msc_chat_replay.py \
  --split "${SPLIT}" \
  --out-dir "${ROOT}/input" \
  --max-dialogs "${MAX_DIALOGS}" \
  --max-sessions "${MAX_SESSIONS}" \
  > "${ROOT}/materialize.log"

SUMMARY="${ROOT}/summary.json"
DB="${ROOT}/msc.sqlite"

"${BUILD_DIR}/examples/benchmark/cortext_chat_replay_live_run" \
  --input-dir "${ROOT}/input" \
  --db "${DB}" \
  --out "${SUMMARY}" \
  --models "${MODELS_DIR}" \
  --max-messages -1 \
  --warmup-events "${WARMUP_EVENTS}" \
  --probe-stride "${PROBE_STRIDE}" \
  --rag-top-k "${RAG_TOP_K}" \
  --active-history-token-budget "${ACTIVE_HISTORY_TOKEN_BUDGET}" \
  --focus 0.5 \
  --sensitivity 0.5 \
  --stability 0.5 \
  --daily-consolidation \
  --daily-consolidation-hour 2 \
  --replay-timezone UTC \
  > "${ROOT}/replay.stdout" \
  2> "${ROOT}/replay.stderr"

python3 - "${SUMMARY}" "${ROOT}/input/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
summary["public_benchmark"] = True
summary["benchmark_dataset"] = manifest
summary["privacy_note"] = (
    "Public Meta MSC benchmark artifact; generated transcript text comes from "
    "the public Hugging Face multi_session_chat mirror."
)
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

python3 - "${ROOT}/judge_systems.json" "${COMPACTION_MODEL}" "${COMPACTING_SESSION_BUDGET}" "${COMPACTING_SESSION_TRIGGER}" "${COMPACTING_SESSION_MAX_OUTPUT_TOKENS}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
config = {
    "compacting_session": {
        "enabled": True,
        "provider": "openai",
        "model": sys.argv[2],
        "budget_tokens": int(sys.argv[3]),
        "trigger": float(sys.argv[4]),
        "max_output_tokens": int(sys.argv[5]),
        "timeout_s": 600,
    }
}
out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

JUDGE_ARGS=(
  python3 tools/judge_chat_replay_live_run.py
  --summary "${SUMMARY}"
  --db "${DB}"
  --out "${ROOT}/judge_openai_gpt55.json"
  --judge-provider openai
  --model "${JUDGE_MODEL}"
  --judge-repetitions "${JUDGE_REPETITIONS}"
  --judge-seed 42
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
  --judge-context-window-tokens "${JUDGE_CONTEXT_WINDOW_TOKENS}"
  --judge-max-output-tokens "${JUDGE_MAX_OUTPUT_TOKENS}"
  --max-media-per-system 0
  --context-limit -1
  --require-judge-prompt-fit
  --require-full-history-complete
)

if [[ "${JUDGE_LIMIT}" != "-1" ]]; then
  JUDGE_ARGS+=(--judge-limit "${JUDGE_LIMIT}")
fi
if [[ -n "${ENV_FILE}" ]]; then
  JUDGE_ARGS+=(--env-file "${ENV_FILE}")
fi

printf '%q ' "${JUDGE_ARGS[@]}" > "${ROOT}/judge_command.txt"
printf '\n' >> "${ROOT}/judge_command.txt"
"${JUDGE_ARGS[@]}" > "${ROOT}/judge.stdout" 2> "${ROOT}/judge.stderr"

echo "Wrote MSC frontier run to ${ROOT}"
