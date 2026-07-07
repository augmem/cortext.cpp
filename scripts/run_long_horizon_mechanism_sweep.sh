#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-eval_runs/long_horizon_mechanism_sweep_$(date -u +%Y%m%dT%H%M%SZ)}"
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
ARMS="${ARMS:-control,emotion_mood_threshold_cascade,neuromodulator_effect_scales,daily_consolidation,graph_expansion,stm_ltm_graph_label_handoff,synaptic_tag_ttl}"
REQUIRE_JUDGE_PROMPT_FIT="${REQUIRE_JUDGE_PROMPT_FIT:-1}"
COMPACTING_PROVIDER="${COMPACTING_PROVIDER:-openai}"
COMPACTING_BUDGET_TOKENS="${COMPACTING_BUDGET_TOKENS:-49152}"
COMPACTING_TRIGGER="${COMPACTING_TRIGGER:-0.8}"
COMPACTING_MAX_CHUNK_TOKENS="${COMPACTING_MAX_CHUNK_TOKENS:-20000}"
COMPACTING_MAX_OUTPUT_TOKENS="${COMPACTING_MAX_OUTPUT_TOKENS:-700}"
COMPACTING_TIMEOUT_S="${COMPACTING_TIMEOUT_S:-600}"

mkdir -p "${ROOT}"
export OPENAI_BASE_URL="${JUDGE_BASE_URL}"
export OPENAI_API_KEY="${JUDGE_API_KEY}"
export LIBRARY_PATH="${HOME}/.local/lib:${LIBRARY_PATH:-}"
export INPUT_DIR MAX_MESSAGES MEDIA_LIMIT WARMUP_EVENTS PROBE_STRIDE
export ACTIVE_HISTORY_TOKEN_BUDGET JUDGE_MODEL JUDGE_CONTEXT_WINDOW_TOKENS
export COMPACTING_PROVIDER COMPACTING_BUDGET_TOKENS COMPACTING_TRIGGER
export COMPACTING_MAX_CHUNK_TOKENS COMPACTING_MAX_OUTPUT_TOKENS
export COMPACTING_TIMEOUT_S JUDGE_REPETITIONS

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORTEXT_BUILD_EXAMPLES=ON \
  -DCORTEXT_DISABLE_OPENTELEMETRY=ON \
  -DCORTEXT_EXPERIMENT_HOOKS=ON
cmake --build "${BUILD_DIR}" --target cortext_chat_replay_live_run -j

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

run_arm() {
  local arm="$1"
  local env_assignment
  env_assignment="$(arm_env_name "${arm}")"
  local arm_dir="${ROOT}/${arm}"
  local summary="${arm_dir}/summary.json"
  local db="${arm_dir}/live.sqlite"
  mkdir -p "${arm_dir}"
  cp "${ROOT}/judge_systems.json" "${arm_dir}/judge_systems.json"

  {
    echo "arm=${arm}"
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

  local replay_args=(
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
  if [[ "${arm}" != "daily_consolidation" ]]; then
    replay_args+=(--daily-consolidation)
  fi

  if [[ -n "${env_assignment}" ]]; then
    env CORTEXT_WM_CAPACITY_OVERRIDE=21 "${env_assignment}" \
      "${replay_args[@]}" \
      > "${arm_dir}/replay.stdout" \
      2> "${arm_dir}/replay.stderr"
  else
    env CORTEXT_WM_CAPACITY_OVERRIDE=21 "${replay_args[@]}" \
      > "${arm_dir}/replay.stdout" \
      2> "${arm_dir}/replay.stderr"
  fi

  python3 - "${summary}" "${ROOT}/judge_systems.json" "${arm}" <<'PY'
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

  local shared_cache="${ROOT}/compacting_session_snapshots.json"
  local arm_cache="${arm_dir}/compacting_session_snapshots.json"
  if [[ -f "${shared_cache}" && ! -f "${arm_cache}" ]]; then
    cp "${shared_cache}" "${arm_cache}"
  fi

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

  if [[ -f "${arm_cache}" && ! -f "${shared_cache}" ]]; then
    cp "${arm_cache}" "${shared_cache}"
  fi
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
judge_repetitions = int(os.environ.get("JUDGE_REPETITIONS", "3"))
signal_floor = 0.15

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

def metric_by_repetition(judge_path, metric):
    rows_path = judge_rows_path(judge_path)
    if not rows_path.exists():
        return []
    by_rep = {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rep = row.get("repetition")
        value = (
            row.get("systems", {})
            .get("cortext_native", {})
            .get(metric)
        )
        if rep is None or value is None:
            continue
        by_rep.setdefault(int(rep), []).append(float(value))
    return [mean(by_rep[rep]) for rep in sorted(by_rep)]

def matched_deltas(arm_values, control_values):
    count = min(len(arm_values), len(control_values))
    return [
        arm_values[index] - control_values[index]
        for index in range(count)
    ]

def value_delta(values, first, second):
    if len(values) <= max(first, second):
        return None
    return values[first] - values[second]

def paired_delta_interpretation(arm, mean_delta, paired_deltas, floor):
    if arm == "control":
        return "measured_control"
    if mean_delta is None or not paired_deltas:
        return None
    if mean_delta > 0:
        return "removal_positive"
    if all(delta <= -floor for delta in paired_deltas):
        return "stable_removal_harm"
    after_rep1 = paired_deltas[1:]
    if after_rep1 and all(abs(delta) < floor for delta in after_rep1):
        return "mean_loss_control_rep1_sensitive"
    if after_rep1 and any(delta >= 0 for delta in after_rep1):
        return "mixed_paired_deltas_control_sensitive"
    if mean_delta < 0:
        return "negative_mean_unresolved"
    return "null"

control_path = root / "control" / "judge.json"
control_data = json.loads(control_path.read_text(encoding="utf-8"))
control_quality = control_data.get("quality", {}).get("cortext_native", {})
control_tokens = control_data.get("tokens", {})
control_token_by_system = control_tokens.get("mean_context_tokens_by_system", {})
control_suff_by_rep = metric_by_repetition(control_path, "sufficiency")
control = {
    "artifact": str(control_path),
    "wins": control_quality.get("wins"),
    "judged": control_data.get("judged"),
    "mean_sufficiency": control_quality.get("mean_sufficiency"),
    "mean_noise": control_quality.get("mean_noise"),
    "mean_context_tokens": control_token_by_system.get(
        "cortext_native",
        control_tokens.get("mean_cortext_context_tokens"),
    ),
    "mean_sufficiency_by_repetition": control_suff_by_rep,
    "mean_sufficiency_rep_sd": sample_stdev(control_suff_by_rep),
}
summary = {
    "schema": "cortext_long_horizon_mechanism_sweep_summary_v2",
    "root": str(root),
    "protocol": {
        "screen": "18k-message context-blowout deferred-family removal",
        "input_dir": os.environ.get("INPUT_DIR", "/shared/Memory/Julie"),
        "max_messages": int(os.environ.get("MAX_MESSAGES", "18000")),
        "media_limit": int(os.environ.get("MEDIA_LIMIT", "128")),
        "warmup_events": int(os.environ.get("WARMUP_EVENTS", "0")),
        "probe_stride": int(os.environ.get("PROBE_STRIDE", "600")),
        "active_history_token_budget": int(os.environ.get("ACTIVE_HISTORY_TOKEN_BUDGET", "49152")),
        "judge_repetitions": judge_repetitions,
        "judge_model": os.environ.get("JUDGE_MODEL", "qwen-omni-judge-32k"),
        "judge_context_window_tokens": int(os.environ.get("JUDGE_CONTEXT_WINDOW_TOKENS", "32768")),
        "control_reference": control,
        "stability_evidence_fields": [
            "protocol.control_reference.mean_sufficiency_by_repetition",
            "arms.*.mean_sufficiency_by_repetition",
            "arms.*.delta_sufficiency_vs_control_by_repetition",
            "arms.*.delta_sufficiency_vs_control_rep_sd",
            "arms.*.paired_delta_interpretation",
        ],
        "control_rep1_sensitivity": {
            "control_mean_sufficiency_by_repetition": control_suff_by_rep or None,
            "rep1_minus_rep2": value_delta(control_suff_by_rep, 0, 1),
            "rep1_minus_rep3": value_delta(control_suff_by_rep, 0, 2),
            "arms_with_nonnegative_paired_delta_after_rep1": [],
            "arms_with_no_signal_sized_loss_after_rep1": [],
            "interpretation": (
                "Control repetition 1 was higher than later control repetitions; "
                "mean-vs-mean negative deltas should be read next to paired "
                "per-repetition deltas before being treated as stable harm."
            ),
        },
        "verdict_rules": {
            "sufficiency_delta_signal_floor": signal_floor,
            "long_horizon_deferred_family": [
                "emotion_mood_threshold_cascade",
                "neuromodulator_effect_scales",
                "daily_consolidation",
                "graph_expansion",
                "stm_ltm_graph_label_handoff",
                "synaptic_tag_ttl",
            ],
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
    suff_by_rep = metric_by_repetition(judge_path, "sufficiency")
    delta_by_rep = matched_deltas(suff_by_rep, control_suff_by_rep)
    after_rep1 = delta_by_rep[1:]
    nonnegative_after_rep1 = bool(after_rep1) and any(
        value >= 0 for value in after_rep1
    )
    no_signal_sized_loss_after_rep1 = bool(after_rep1) and all(
        value > -signal_floor for value in after_rep1
    )
    if arm != "control" and nonnegative_after_rep1:
        summary["protocol"]["control_rep1_sensitivity"][
            "arms_with_nonnegative_paired_delta_after_rep1"
        ].append(arm)
    if arm != "control" and no_signal_sized_loss_after_rep1:
        summary["protocol"]["control_rep1_sensitivity"][
            "arms_with_no_signal_sized_loss_after_rep1"
        ].append(arm)
    if arm == "control" and delta == 0:
        reading = "measured_control"
    elif delta is None:
        reading = "missing"
    elif abs(delta) < signal_floor:
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
        "mean_sufficiency_by_repetition": suff_by_rep or None,
        "mean_sufficiency_rep_sd": sample_stdev(suff_by_rep),
        "delta_sufficiency_vs_control_by_repetition": delta_by_rep or None,
        "delta_sufficiency_vs_control_rep_sd": sample_stdev(delta_by_rep),
        "paired_delta_after_rep1_signal_count": sum(
            1 for value in after_rep1 if abs(value) >= signal_floor
        ),
        "paired_delta_after_rep1_nonnegative_count": sum(
            1 for value in after_rep1 if value >= 0
        ),
        "paired_delta_interpretation": paired_delta_interpretation(
            arm,
            delta,
            delta_by_rep,
            signal_floor,
        ),
        "reading_by_sufficiency_rule": reading,
        "judgment_complete": data.get("judgment_complete"),
        "missing_judgments": data.get("missing_judgments"),
        "fairness_checks": data.get("fairness_checks"),
    }
(root / "mechanism_sweep_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(root / "mechanism_sweep_summary.json")
PY

echo "Wrote long-horizon mechanism sweep to ${ROOT}"
