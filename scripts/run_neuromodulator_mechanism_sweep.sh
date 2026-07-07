#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-eval_runs/neuromodulator_mechanism_sweep_$(date -u +%Y%m%dT%H%M%SZ)}"
BUILD_DIR="${BUILD_DIR:-build-mechanism-hooks}"
MODELS_DIR="${MODELS_DIR:-models}"
SPLIT="${SPLIT:-validation}"
MAX_DIALOGS="${MAX_DIALOGS:-25}"
MAX_SESSIONS="${MAX_SESSIONS:-0}"
WARMUP_EVENTS="${WARMUP_EVENTS:-20}"
PROBE_STRIDE="${PROBE_STRIDE:-30}"
RAG_TOP_K="${RAG_TOP_K:-5}"
ACTIVE_HISTORY_TOKEN_BUDGET="${ACTIVE_HISTORY_TOKEN_BUDGET:-4096}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen-omni-judge}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8000/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-local-vllm}"
JUDGE_REPETITIONS="${JUDGE_REPETITIONS:-1}"
JUDGE_CONTEXT_WINDOW_TOKENS="${JUDGE_CONTEXT_WINDOW_TOKENS:-7800}"
JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-384}"
JUDGE_LIMIT="${JUDGE_LIMIT:--1}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
ENV_FILE="${CORTEXT_EVAL_ENV_FILE:-${ENV_FILE:-}}"
ARMS="${ARMS:-neuromodulator_effect_scales,synaptic_tagging,encode_retrieve_oscillator,emotion_mood_threshold_cascade}"
REQUIRE_JUDGE_PROMPT_FIT="${REQUIRE_JUDGE_PROMPT_FIT:-0}"

mkdir -p "${ROOT}"
export OPENAI_BASE_URL="${JUDGE_BASE_URL}"
export OPENAI_API_KEY="${JUDGE_API_KEY}"
export LIBRARY_PATH="${HOME}/.local/lib:${LIBRARY_PATH:-}"

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORTEXT_BUILD_EXAMPLES=ON \
  -DCORTEXT_DISABLE_OPENTELEMETRY=ON \
  -DCORTEXT_EXPERIMENT_HOOKS=ON
cmake --build "${BUILD_DIR}" --target cortext_chat_replay_live_run -j

python3 scripts/materialize_msc_chat_replay.py \
  --split "${SPLIT}" \
  --out-dir "${ROOT}/input" \
  --max-dialogs "${MAX_DIALOGS}" \
  --max-sessions "${MAX_SESSIONS}" \
  > "${ROOT}/materialize.log"

python3 - "${ROOT}/judge_systems.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "base_systems": [
                "cortext_native",
                "traditional_chat_rag",
            ]
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

arm_env_name() {
  case "$1" in
    control) echo "" ;;
    neuromodulator_effect_scales) echo "CORTEXT_DISABLE_NEUROMODULATOR_EFFECT_SCALES=1" ;;
    synaptic_tagging) echo "CORTEXT_DISABLE_SYNAPTIC_TAGGING=1" ;;
    encode_retrieve_oscillator) echo "CORTEXT_DISABLE_ENCODE_RETRIEVE_OSCILLATOR=1" ;;
    emotion_mood_threshold_cascade) echo "CORTEXT_DISABLE_EMOTION_MOOD_THRESHOLD_CASCADE=1" ;;
    *)
      echo "Unknown arm: $1" >&2
      return 1
      ;;
  esac
}

run_arm() {
  local arm="$1"
  local env_assignment
  env_assignment="$(arm_env_name "${arm}")"
  local arm_dir="${ROOT}/${arm}"
  local summary="${arm_dir}/summary.json"
  local db="${arm_dir}/msc.sqlite"
  mkdir -p "${arm_dir}"
  cp "${ROOT}/judge_systems.json" "${arm_dir}/judge_systems.json"

  {
    echo "arm=${arm}"
    echo "env=${env_assignment}"
    echo "CORTEXT_WM_CAPACITY_OVERRIDE=21"
    echo "MAX_DIALOGS=${MAX_DIALOGS}"
    echo "WARMUP_EVENTS=${WARMUP_EVENTS}"
    echo "PROBE_STRIDE=${PROBE_STRIDE}"
  } > "${arm_dir}/arm.env"

  if [[ -n "${env_assignment}" ]]; then
    env CORTEXT_WM_CAPACITY_OVERRIDE=21 "${env_assignment}" \
      "${BUILD_DIR}/examples/benchmark/cortext_chat_replay_live_run" \
        --input-dir "${ROOT}/input" \
        --db "${db}" \
        --out "${summary}" \
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
        > "${arm_dir}/replay.stdout" \
        2> "${arm_dir}/replay.stderr"
  else
    env CORTEXT_WM_CAPACITY_OVERRIDE=21 \
      "${BUILD_DIR}/examples/benchmark/cortext_chat_replay_live_run" \
        --input-dir "${ROOT}/input" \
        --db "${db}" \
        --out "${summary}" \
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
        > "${arm_dir}/replay.stdout" \
        2> "${arm_dir}/replay.stderr"
  fi

  python3 - "${summary}" "${ROOT}/input/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
summary["public_benchmark"] = True
summary["benchmark_dataset"] = manifest
summary["mechanism_sweep_protocol"] = {
    "screen": "capacity-21 MSC 39-probe single-mechanism removal",
    "control_reference": {
        "wins": "22/39",
        "mean_sufficiency": 3.95,
        "mean_noise": 0.67,
        "mean_context_tokens": 503,
    },
}
summary_path.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

  local judge="${arm_dir}/judge.json"
  local judge_args=(
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
    --max-media-per-system 0
    --context-limit -1
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
}

IFS=',' read -r -a arm_list <<< "${ARMS}"
for arm in "${arm_list[@]}"; do
  run_arm "${arm}"
done

python3 - "${ROOT}" "${ARMS}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = [item for item in sys.argv[2].split(",") if item]
judge_repetitions = int(os.environ.get("JUDGE_REPETITIONS", "1"))

def mean(values):
    return sum(values) / len(values) if values else None

def sample_stdev(values):
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5

def judge_rows_path(judge_path):
    return judge_path.with_name(judge_path.name + ".rows.jsonl")

def sufficiency_by_repetition(judge_path):
    rows_path = judge_rows_path(judge_path)
    if not rows_path.exists():
        return []
    by_rep = {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rep = row.get("repetition")
        sufficiency = (
            row.get("systems", {})
            .get("cortext_native", {})
            .get("sufficiency")
        )
        if rep is None or sufficiency is None:
            continue
        by_rep.setdefault(int(rep), []).append(float(sufficiency))
    return [mean(by_rep[rep]) for rep in sorted(by_rep)]

def matched_deltas_by_repetition(arm_values, control_values):
    count = min(len(arm_values), len(control_values))
    if count == 0:
        return []
    return [
        arm_values[index] - control_values[index]
        for index in range(count)
    ]

legacy_control = {
    "wins": 22,
    "judged": 39,
    "mean_sufficiency": 3.95,
    "mean_noise": 0.67,
    "mean_context_tokens": 503,
}
control = dict(legacy_control)
control_path = root / "control" / "judge.json"
control_suff_by_repetition = sufficiency_by_repetition(control_path)
if control_path.exists():
    control_data = json.loads(control_path.read_text(encoding="utf-8"))
    control_quality = control_data.get("quality", {}).get("cortext_native", {})
    control_token_by_system = control_data.get("tokens", {}).get(
        "mean_context_tokens_by_system", {}
    )
    control = {
        "artifact": str(control_path),
        "wins": control_quality.get("wins"),
        "judged": control_data.get("judged"),
        "mean_sufficiency": control_quality.get("mean_sufficiency"),
        "mean_noise": control_quality.get("mean_noise"),
        "mean_context_tokens": control_token_by_system.get(
            "cortext_native",
            control_data.get("tokens", {}).get("mean_cortext_context_tokens"),
        ),
    }
    if control_suff_by_repetition:
        control["mean_sufficiency_by_repetition"] = control_suff_by_repetition
        control["mean_sufficiency_rep_sd"] = sample_stdev(control_suff_by_repetition)
summary = {
    "schema": "cortext_neuromodulator_mechanism_sweep_summary_v2",
    "root": str(root),
    "protocol": {
        "screen": "capacity-21 MSC 39-probe single-mechanism removal",
        "max_dialogs": 25,
        "warmup_events": 20,
        "probe_stride": 30,
        "judge_repetitions": judge_repetitions,
        "control_reference": control,
        "legacy_june_control_reference": legacy_control,
        "verdict_rules": {
            "sufficiency_delta_signal_floor": 0.15,
            "long_horizon_null_policy": "defer consolidation-style mechanisms rather than keep",
        },
    },
    "arms": {},
}
for arm in arms:
    judge_path = root / arm / "judge.json"
    if not judge_path.exists():
        summary["arms"][arm] = {"error": f"missing {judge_path}"}
        continue
    data = json.loads(judge_path.read_text(encoding="utf-8"))
    quality = data.get("quality", {}).get("cortext_native", {})
    token_by_system = data.get("tokens", {}).get("mean_context_tokens_by_system", {})
    mean_tokens = token_by_system.get(
        "cortext_native",
        data.get("tokens", {}).get("mean_cortext_context_tokens"),
    )
    suff = quality.get("mean_sufficiency")
    control_suff = control.get("mean_sufficiency")
    delta = None if suff is None or control_suff is None else suff - control_suff
    suff_by_repetition = sufficiency_by_repetition(judge_path)
    delta_by_repetition = matched_deltas_by_repetition(
        suff_by_repetition,
        control_suff_by_repetition,
    )
    if arm == "control" and delta == 0:
        reading = "measured_control"
    elif delta is None:
        reading = "missing"
    elif abs(delta) < 0.15:
        reading = "null"
    elif delta < 0:
        reading = "removal_hurts"
    else:
        reading = "removal_positive"
    summary["arms"][arm] = {
        "artifact": str(judge_path),
        "judged": data.get("judged"),
        "probe_count": data.get("probe_count"),
        "wins": quality.get("wins"),
        "mean_sufficiency": suff,
        "mean_noise": quality.get("mean_noise"),
        "mean_context_tokens": mean_tokens,
        "delta_sufficiency_vs_control": delta,
        "mean_sufficiency_by_repetition": suff_by_repetition or None,
        "mean_sufficiency_rep_sd": sample_stdev(suff_by_repetition),
        "delta_sufficiency_vs_control_by_repetition": (
            delta_by_repetition or None
        ),
        "delta_sufficiency_vs_control_rep_sd": sample_stdev(delta_by_repetition),
        "reading_by_sufficiency_rule": reading,
        "judgment_complete": data.get("judgment_complete"),
        "missing_judgments": data.get("missing_judgments"),
    }
(root / "mechanism_sweep_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(root / "mechanism_sweep_summary.json")
PY

echo "Wrote neuromodulator mechanism sweep to ${ROOT}"
