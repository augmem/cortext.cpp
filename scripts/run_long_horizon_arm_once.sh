#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <arm>" >&2
  exit 2
fi

ARM="$1"
if [[ -z "${ROOT:-}" ]]; then
  if [[ -f /tmp/cortext_long_horizon_root.txt ]]; then
    ROOT="$(cat /tmp/cortext_long_horizon_root.txt)"
  else
    echo "ROOT is required" >&2
    exit 2
  fi
fi

BUILD_DIR="${BUILD_DIR:-build-mechanism-hooks}"
MODELS_DIR="${MODELS_DIR:-models}"
INPUT_DIR="${INPUT_DIR:-/shared/Memory/Julie}"
MAX_MESSAGES="${MAX_MESSAGES:-18000}"
MEDIA_LIMIT="${MEDIA_LIMIT:-128}"
WARMUP_EVENTS="${WARMUP_EVENTS:-0}"
PROBE_STRIDE="${PROBE_STRIDE:-600}"
RAG_TOP_K="${RAG_TOP_K:-5}"
ACTIVE_HISTORY_TOKEN_BUDGET="${ACTIVE_HISTORY_TOKEN_BUDGET:-49152}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen-omni-judge-32k}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8001/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-local-vllm}"
JUDGE_REPETITIONS="${JUDGE_REPETITIONS:-3}"
JUDGE_CONTEXT_WINDOW_TOKENS="${JUDGE_CONTEXT_WINDOW_TOKENS:-32768}"
JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-1024}"
JUDGE_TIMEOUT_S="${JUDGE_TIMEOUT_S:-300}"
JUDGE_LIMIT="${JUDGE_LIMIT:--1}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
ENV_FILE="${CORTEXT_EVAL_ENV_FILE:-${ENV_FILE:-}}"
REQUIRE_JUDGE_PROMPT_FIT="${REQUIRE_JUDGE_PROMPT_FIT:-1}"
COMPACTING_PROVIDER="${COMPACTING_PROVIDER:-openai}"
COMPACTING_BUDGET_TOKENS="${COMPACTING_BUDGET_TOKENS:-49152}"
COMPACTING_TRIGGER="${COMPACTING_TRIGGER:-0.8}"
COMPACTING_MAX_CHUNK_TOKENS="${COMPACTING_MAX_CHUNK_TOKENS:-20000}"
COMPACTING_MAX_OUTPUT_TOKENS="${COMPACTING_MAX_OUTPUT_TOKENS:-700}"
COMPACTING_TIMEOUT_S="${COMPACTING_TIMEOUT_S:-600}"

export OPENAI_BASE_URL="${JUDGE_BASE_URL}"
export OPENAI_API_KEY="${JUDGE_API_KEY}"
export LIBRARY_PATH="${HOME}/.local/lib:${LIBRARY_PATH:-}"
export INPUT_DIR MAX_MESSAGES MEDIA_LIMIT WARMUP_EVENTS PROBE_STRIDE
export ACTIVE_HISTORY_TOKEN_BUDGET JUDGE_MODEL JUDGE_CONTEXT_WINDOW_TOKENS
export COMPACTING_PROVIDER COMPACTING_BUDGET_TOKENS COMPACTING_TRIGGER
export COMPACTING_MAX_CHUNK_TOKENS COMPACTING_MAX_OUTPUT_TOKENS
export COMPACTING_TIMEOUT_S JUDGE_REPETITIONS

arm_env_name() {
  case "$1" in
    control) echo "" ;;
    emotion_mood_threshold_cascade) echo "CORTEXT_DISABLE_EMOTION_MOOD_THRESHOLD_CASCADE=1" ;;
    neuromodulator_effect_scales) echo "CORTEXT_DISABLE_NEUROMODULATOR_EFFECT_SCALES=1" ;;
    daily_consolidation) echo "" ;;
    graph_expansion) echo "CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION=1" ;;
    stm_ltm_graph_label_handoff) echo "CORTEXT_DISABLE_STM_LTM_GRAPH_LABEL_HANDOFF=1" ;;
    synaptic_tag_ttl) echo "CORTEXT_DISABLE_SYNAPTIC_TAG_TTL=1" ;;
    *)
      echo "Unknown arm: $1" >&2
      return 1
      ;;
  esac
}

mkdir -p "${ROOT}"
if [[ ! -x "${BUILD_DIR}/examples/benchmark/cortext_chat_replay_live_run" ]]; then
  cmake --build "${BUILD_DIR}" --target cortext_chat_replay_live_run -j
fi

python3 - "${ROOT}/judge_systems.json" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "base_systems": [
                "cortext_native",
                "traditional_chat_rag",
                "full_history_upper_bound",
            ],
            "compacting_session": {
                "enabled": True,
                "provider": os.environ.get("COMPACTING_PROVIDER", "openai"),
                "model": os.environ.get("JUDGE_MODEL", "qwen-omni-judge-32k"),
                "budget_tokens": int(os.environ.get("COMPACTING_BUDGET_TOKENS", "49152")),
                "trigger": float(os.environ.get("COMPACTING_TRIGGER", "0.8")),
                "max_chunk_tokens": int(os.environ.get("COMPACTING_MAX_CHUNK_TOKENS", "20000")),
                "max_output_tokens": int(os.environ.get("COMPACTING_MAX_OUTPUT_TOKENS", "700")),
                "timeout_s": int(os.environ.get("COMPACTING_TIMEOUT_S", "600")),
            },
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

env_assignment="$(arm_env_name "${ARM}")"
arm_dir="${ROOT}/${ARM}"
summary="${arm_dir}/summary.json"
db="${arm_dir}/live.sqlite"
mkdir -p "${arm_dir}"
cp "${ROOT}/judge_systems.json" "${arm_dir}/judge_systems.json"

{
  echo "arm=${ARM}"
  echo "env=${env_assignment}"
  echo "CORTEXT_WM_CAPACITY_OVERRIDE=21"
  echo "INPUT_DIR=${INPUT_DIR}"
  echo "MAX_MESSAGES=${MAX_MESSAGES}"
  echo "MEDIA_LIMIT=${MEDIA_LIMIT}"
  echo "WARMUP_EVENTS=${WARMUP_EVENTS}"
  echo "PROBE_STRIDE=${PROBE_STRIDE}"
  echo "ACTIVE_HISTORY_TOKEN_BUDGET=${ACTIVE_HISTORY_TOKEN_BUDGET}"
  echo "JUDGE_MODEL=${JUDGE_MODEL}"
  echo "JUDGE_BASE_URL=${JUDGE_BASE_URL}"
} > "${arm_dir}/arm.env"

if [[ ! -f "${summary}" ]]; then
  replay_pid="$(pgrep -f "cortext_chat_replay_live_run.*${db}" || true)"
  if [[ -n "${replay_pid}" ]]; then
    echo "waiting for existing replay pid=${replay_pid} arm=${ARM}"
    while [[ ! -f "${summary}" ]]; do
      if ! pgrep -f "cortext_chat_replay_live_run.*${db}" >/dev/null; then
        echo "existing replay exited without ${summary}" >&2
        exit 1
      fi
      read -t 30 _ || true
    done
  else
    replay_args=(
      "${BUILD_DIR}/examples/benchmark/cortext_chat_replay_live_run"
      --input-dir "${INPUT_DIR}"
      --db "${db}"
      --out "${summary}"
      --models "${MODELS_DIR}"
      --max-messages "${MAX_MESSAGES}"
      --media-limit "${MEDIA_LIMIT}"
      --warmup-events "${WARMUP_EVENTS}"
      --probe-stride "${PROBE_STRIDE}"
      --rag-top-k "${RAG_TOP_K}"
      --active-history-token-budget "${ACTIVE_HISTORY_TOKEN_BUDGET}"
      --focus 0.5
      --sensitivity 0.5
      --stability 0.5
      --daily-consolidation-hour 2
      --replay-timezone UTC
    )
    if [[ "${ARM}" != "daily_consolidation" ]]; then
      replay_args+=(--daily-consolidation)
    fi

    env_vars=(CORTEXT_WM_CAPACITY_OVERRIDE=21)
    if [[ -n "${env_assignment}" ]]; then
      env_vars+=("${env_assignment}")
    fi
    env "${env_vars[@]}" "${replay_args[@]}" \
      > "${arm_dir}/replay.stdout" \
      2> "${arm_dir}/replay.stderr"
  fi
fi

python3 - "${summary}" "${ROOT}/judge_systems.json" "${ARM}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
judge_systems_path = Path(sys.argv[2])
arm = sys.argv[3]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["public_benchmark"] = True
summary["mechanism_sweep_protocol"] = {
    "screen": "18k-message context-blowout deferred-family removal",
    "arm": arm,
    "judge_systems": json.loads(judge_systems_path.read_text(encoding="utf-8")),
}
summary_path.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

shared_cache="${ROOT}/compacting_session_snapshots.json"
arm_cache="${arm_dir}/compacting_session_snapshots.json"
if [[ -f "${shared_cache}" && ! -f "${arm_cache}" ]]; then
  cp "${shared_cache}" "${arm_cache}"
fi

judge="${arm_dir}/judge.json"
if [[ ! -f "${judge}" ]]; then
  judge_args=(
    python3 tools/judge_chat_replay_live_run.py
    --summary "${summary}"
    --db "${db}"
    --out "${judge}"
    --judge-provider "${JUDGE_PROVIDER}"
    --model "${JUDGE_MODEL}"
    --judge-repetitions "${JUDGE_REPETITIONS}"
    --judge-seed 42
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
    --judge-context-window-tokens "${JUDGE_CONTEXT_WINDOW_TOKENS}"
    --judge-max-output-tokens "${JUDGE_MAX_OUTPUT_TOKENS}"
    --judge-timeout-s "${JUDGE_TIMEOUT_S}"
    --context-limit "${MEDIA_LIMIT}"
    --max-media-per-system 0
  )
  if [[ "${REQUIRE_JUDGE_PROMPT_FIT}" == "1" ]]; then
    judge_args+=(--require-judge-prompt-fit)
  fi
  if [[ "${JUDGE_LIMIT}" != "-1" ]]; then
    judge_args+=(--judge-limit "${JUDGE_LIMIT}")
  fi
  if [[ -n "${ENV_FILE}" ]]; then
    judge_args+=(--env-file "${ENV_FILE}")
  fi
  printf '%q ' "${judge_args[@]}" > "${arm_dir}/judge_command.txt"
  printf '\n' >> "${arm_dir}/judge_command.txt"
  "${judge_args[@]}" > "${arm_dir}/judge.stdout" 2> "${arm_dir}/judge.stderr"
fi

echo "${judge}"
