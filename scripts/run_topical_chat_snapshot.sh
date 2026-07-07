#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-build/examples/topical_chat_analysis/cortext_topical_chat_analysis}"
DATA="${DATA:-data/topical_chat/valid_freq.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-}"
MAX_TURNS="${MAX_TURNS:-120}"
MAX_TOTAL="${MAX_TOTAL:-120}"
MAX_CONVERSATIONS="${MAX_CONVERSATIONS:-1}"
FOCUS="${FOCUS:-0.5}"
SENSITIVITY="${SENSITIVITY:-0.5}"
STABILITY="${STABILITY:-0.5}"
SEED="${SEED:-1337}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SYNTHETIC_START_MS="${SYNTHETIC_START_MS:-1700000000000}"
CADENCE_ENABLED="${CADENCE_ENABLED:-0}"
CONSOLIDATE="${CONSOLIDATE:-1}"
CONSOLIDATE_CYCLES="${CONSOLIDATE_CYCLES:-2}"
CONSOLIDATE_DURING="${CONSOLIDATE_DURING:-1}"
CONSOLIDATE_IDLE="${CONSOLIDATE_IDLE:-1}"
CONSOLIDATE_EVERY="${CONSOLIDATE_EVERY:-0}"
CONSOLIDATE_EVERY_AUTO="${CONSOLIDATE_EVERY_AUTO:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUT_DIR="${OUT_DIR:-logs/topical_chat_snapshots/$(date +%Y%m%d_%H%M%S)}"
DB_PATH="$OUT_DIR/cortext.db"

mkdir -p "$OUT_DIR"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "scripts/run_topical_chat_snapshot.sh requires python3 or python" >&2
    exit 1
  fi
fi

log="$OUT_DIR/run.log"
summary="$OUT_DIR/summary.csv"
config="$OUT_DIR/config.json"
perf="$OUT_DIR/perf.json"
git_status="$OUT_DIR/git_status.txt"
git_diff="$OUT_DIR/git_diff.patch"

if [[ "$CONSOLIDATE" == "1" && "$CONSOLIDATE_EVERY" == "0" && "$CONSOLIDATE_EVERY_AUTO" == "1" ]]; then
  CONSOLIDATE_EVERY="$("$PYTHON_BIN" - <<PY
import math
stability=float("$STABILITY")
max_turns=int("$MAX_TURNS")
ratio=0.3 + 0.3*max(0.0, min(1.0, stability))
every=max(1, int(round(max_turns * ratio)))
print(every)
PY
)"
fi

git rev-parse --short HEAD > "$OUT_DIR/git_rev.txt" 2>/dev/null || true
git status -sb > "$git_status" 2>/dev/null || true
git diff > "$git_diff" 2>/dev/null || true

cat > "$config" <<EOF
{
  "focus": ${FOCUS},
  "sensitivity": ${SENSITIVITY},
  "stability": ${STABILITY},
  "data": "${DATA}",
  "model_assets": "${MODELS}",
  "db": "${DB_PATH}",
  "binary": "${BIN}",
  "max_turns": ${MAX_TURNS},
  "max_total": ${MAX_TOTAL},
  "max_conversations": ${MAX_CONVERSATIONS},
  "seed": ${SEED},
  "deterministic": ${DETERMINISTIC},
  "synthetic_start_ms": ${SYNTHETIC_START_MS},
  "consolidate": ${CONSOLIDATE},
  "consolidate_cycles": ${CONSOLIDATE_CYCLES},
  "consolidate_during": ${CONSOLIDATE_DURING},
  "consolidate_idle": ${CONSOLIDATE_IDLE},
  "consolidate_every_turns": ${CONSOLIDATE_EVERY}
}
EOF

echo "focus,sensitivity,stability,turns,writes,consolidation_runs,consolidation_failures,consolidation_every_turns,consolidation_association_created,consolidation_label_created,duration_sec,signals_per_sec,perf_encode_ms_mean,perf_process_ms_mean,perf_hydrate_ms_mean,perf_total_ms_mean,retrieval_turn_rate,retrieval_avg_candidates,retrieval_overlap_mean,retrieval_context_overlap_mean,retrieval_semantic_overlap_mean,retrieval_context_semantic_overlap_mean,retrieval_association_candidate_rate,retrieval_label_candidate_rate,retrieval_association_turn_rate,retrieval_label_turn_rate,interrupt_turn_rate,interrupt_abort_rate,interrupt_semantic_overlap_mean,interrupt_context_semantic_overlap_mean,interrupt_association_candidate_rate,interrupt_label_candidate_rate,interrupt_association_turn_rate,interrupt_label_turn_rate,boundary_at_rate,boundary_score_pass_rate,boundary_score_mean" > "$summary"

get_metric() {
  local key="$1"
  local file="$2"
  awk -v k="$key" '
    $1 ~ "^"k"=" {split($1, a, "="); print a[2]; found=1; exit}
    END {if (!found) print "0"}
  ' "$file"
}

extra_args=()
if [[ -n "$SEED" ]]; then
  extra_args+=(--seed="$SEED")
fi
if [[ "${DETERMINISTIC}" == "1" ]]; then
  extra_args+=(--deterministic)
fi
if [[ -n "$SYNTHETIC_START_MS" ]]; then
  extra_args+=(--synthetic-start-ms="$SYNTHETIC_START_MS")
fi
if [[ "$CONSOLIDATE" == "1" ]]; then
  extra_args+=(--consolidate --consolidate-cycles="$CONSOLIDATE_CYCLES")
  if [[ "$CONSOLIDATE_EVERY" != "0" ]]; then
    extra_args+=(--consolidate-every="$CONSOLIDATE_EVERY")
  fi
  if [[ "$CONSOLIDATE_DURING" == "1" ]]; then
    extra_args+=(--consolidate-during)
    if [[ "$CONSOLIDATE_IDLE" == "1" ]]; then
      extra_args+=(--consolidate-idle)
    fi
  fi
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  extra_args+=($EXTRA_ARGS)
fi

start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SECONDS=0
run_exit=0

set +e
"$BIN" \
  --data="$DATA" \
  --db="$DB_PATH" \
  --max-conversations="$MAX_CONVERSATIONS" \
  --max-turns="$MAX_TURNS" \
  --max-total="$MAX_TOTAL" \
  --focus="$FOCUS" \
  --sensitivity="$SENSITIVITY" \
  --stability="$STABILITY" \
  --reuse \
  --otel-filter=none \
  $( [[ "$CADENCE_ENABLED" == "1" ]] && echo "--cadence-speed=1" || echo "--no-cadence" ) \
  --semantic \
  "${extra_args[@]:-}" 2>&1 | tee "$log"
run_exit=${PIPESTATUS[0]}
set -e

end_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
duration_sec="$SECONDS"

turns="$(get_metric turns "$log")"
writes="$(get_metric writes "$log")"
consolidation_runs="$(get_metric consolidation_runs "$log")"
consolidation_failures="$(get_metric consolidation_failures "$log")"
consolidation_every_turns="$(get_metric consolidation_every_turns "$log")"
consolidation_association_created="$(get_metric consolidation_association_created "$log")"
consolidation_label_created="$(get_metric consolidation_label_created "$log")"
retrieval_turn_rate="$(get_metric retrieval_turn_rate "$log")"
retrieval_avg_candidates="$(get_metric retrieval_avg_candidates "$log")"
retrieval_overlap_mean="$(get_metric retrieval_overlap_mean "$log")"
retrieval_context_overlap_mean="$(get_metric retrieval_context_overlap_mean "$log")"
retrieval_semantic_overlap_mean="$(get_metric retrieval_semantic_overlap_mean "$log")"
retrieval_context_semantic_overlap_mean="$(get_metric retrieval_context_semantic_overlap_mean "$log")"
retrieval_association_candidate_rate="$(get_metric retrieval_association_candidate_rate "$log")"
retrieval_label_candidate_rate="$(get_metric retrieval_label_candidate_rate "$log")"
retrieval_association_turn_rate="$(get_metric retrieval_association_turn_rate "$log")"
retrieval_label_turn_rate="$(get_metric retrieval_label_turn_rate "$log")"
interrupt_turn_rate="$(get_metric interrupt_turn_rate "$log")"
interrupt_abort_rate="$(get_metric interrupt_abort_rate "$log")"
interrupt_semantic_overlap_mean="$(get_metric interrupt_semantic_overlap_mean "$log")"
interrupt_context_semantic_overlap_mean="$(get_metric interrupt_context_semantic_overlap_mean "$log")"
interrupt_association_candidate_rate="$(get_metric interrupt_association_candidate_rate "$log")"
interrupt_label_candidate_rate="$(get_metric interrupt_label_candidate_rate "$log")"
interrupt_association_turn_rate="$(get_metric interrupt_association_turn_rate "$log")"
interrupt_label_turn_rate="$(get_metric interrupt_label_turn_rate "$log")"
boundary_at_rate="$(get_metric boundary_at_rate "$log")"
boundary_score_pass_rate="$(get_metric boundary_score_pass_rate "$log")"
boundary_score_mean="$(get_metric boundary_score_mean "$log")"
perf_encode_ms_mean="$(get_metric perf_encode_ms_mean "$log")"
perf_process_ms_mean="$(get_metric perf_process_ms_mean "$log")"
perf_hydrate_ms_mean="$(get_metric perf_hydrate_ms_mean "$log")"
perf_total_ms_mean="$(get_metric perf_total_ms_mean "$log")"

signals_per_sec="$(python - <<PY
import math
turns=float("$turns")
dur=float("$duration_sec")
print(0.0 if dur <= 0 else (turns / dur))
PY
)"

echo "${FOCUS},${SENSITIVITY},${STABILITY},${turns},${writes},${consolidation_runs},${consolidation_failures},${consolidation_every_turns},${consolidation_association_created},${consolidation_label_created},${duration_sec},${signals_per_sec},${perf_encode_ms_mean},${perf_process_ms_mean},${perf_hydrate_ms_mean},${perf_total_ms_mean},${retrieval_turn_rate},${retrieval_avg_candidates},${retrieval_overlap_mean},${retrieval_context_overlap_mean},${retrieval_semantic_overlap_mean},${retrieval_context_semantic_overlap_mean},${retrieval_association_candidate_rate},${retrieval_label_candidate_rate},${retrieval_association_turn_rate},${retrieval_label_turn_rate},${interrupt_turn_rate},${interrupt_abort_rate},${interrupt_semantic_overlap_mean},${interrupt_context_semantic_overlap_mean},${interrupt_association_candidate_rate},${interrupt_label_candidate_rate},${interrupt_association_turn_rate},${interrupt_label_turn_rate},${boundary_at_rate},${boundary_score_pass_rate},${boundary_score_mean}" >> "$summary"

export PERF_OUT="$perf"
export PERF_START_TIME="$start_iso"
export PERF_END_TIME="$end_iso"
export PERF_DURATION_SEC="$duration_sec"
export PERF_TURNS="$turns"
export PERF_WRITES="$writes"
export PERF_SIGNALS_PER_SEC="$signals_per_sec"
export PERF_ENCODE_MS_MEAN="$perf_encode_ms_mean"
export PERF_PROCESS_MS_MEAN="$perf_process_ms_mean"
export PERF_HYDRATE_MS_MEAN="$perf_hydrate_ms_mean"
export PERF_TOTAL_MS_MEAN="$perf_total_ms_mean"
export PERF_INFER_THREADS="${CORTEXT_INFER_THREADS:-}"
export PERF_EMBED_THREADS="${CORTEXT_EMBED_THREADS:-}"

"$PYTHON_BIN" - <<'PY'
import json
import os

out_path = os.environ["PERF_OUT"]
data = {
  "start_time": os.environ.get("PERF_START_TIME", ""),
  "end_time": os.environ.get("PERF_END_TIME", ""),
  "duration_sec": float(os.environ.get("PERF_DURATION_SEC", "0") or 0),
  "turns": int(float(os.environ.get("PERF_TURNS", "0") or 0)),
  "writes": int(float(os.environ.get("PERF_WRITES", "0") or 0)),
  "signals_per_sec": float(os.environ.get("PERF_SIGNALS_PER_SEC", "0") or 0),
  "stage_ms_mean": {
    "encode": float(os.environ.get("PERF_ENCODE_MS_MEAN", "0") or 0),
    "process": float(os.environ.get("PERF_PROCESS_MS_MEAN", "0") or 0),
    "hydrate": float(os.environ.get("PERF_HYDRATE_MS_MEAN", "0") or 0),
    "total": float(os.environ.get("PERF_TOTAL_MS_MEAN", "0") or 0),
  },
  "threads": {
    "infer": os.environ.get("PERF_INFER_THREADS") or "default",
    "embed": os.environ.get("PERF_EMBED_THREADS") or "default",
  },
}

with open(out_path, "w", encoding="utf-8") as f:
  json.dump(data, f, indent=2)
  f.write("\n")
PY

echo "Wrote snapshot to $OUT_DIR"

if [[ "$run_exit" != "0" ]]; then
  echo "Snapshot run failed with exit code $run_exit" >&2
  exit "$run_exit"
fi
