#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-build/examples/topical_chat_analysis/cortext_topical_chat_analysis}"
DATA="${DATA:-data/topical_chat/valid_freq.jsonl}"
MODELS="${MODELS:-models}"
MAX_TURNS="${MAX_TURNS:-120}"
MAX_TOTAL="${MAX_TOTAL:-120}"
OUT_DIR="${OUT_DIR:-logs/topical_chat_runs/$(date +%Y%m%d_%H%M%S)}"
CONSOLIDATE="${CONSOLIDATE:-1}"
CONSOLIDATE_CYCLES="${CONSOLIDATE_CYCLES:-2}"
CONSOLIDATE_EVERY="${CONSOLIDATE_EVERY:-0}"
SEED="${SEED:-1337}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SYNTHETIC_START_MS="${SYNTHETIC_START_MS:-1700000000000}"

mkdir -p "$OUT_DIR"

git rev-parse --short HEAD > "$OUT_DIR/git_rev.txt" 2>/dev/null || true
git status -sb > "$OUT_DIR/git_status.txt" 2>/dev/null || true
git diff > "$OUT_DIR/git_diff.patch" 2>/dev/null || true

configs=(
  "0 0 0"
  "0.25 0.25 0.25"
  "0.5 0.5 0.5"
  "0.75 0.75 0.75"
  "1 1 1"
  "0.15 0.9 0.5"
)

if [[ -n "${CONFIGS:-}" ]]; then
  configs=()
  while IFS= read -r line; do
    line="${line%%#*}"
    line="${line//,/ }"
    if [[ -z "${line//[[:space:]]/}" ]]; then
      continue
    fi
    configs+=("$line")
  done < <(printf '%s\n' "${CONFIGS//;/\n}")
fi

summary_csv="$OUT_DIR/summary.csv"
echo "focus,sensitivity,stability,turns,writes,consolidation_runs,consolidation_failures,consolidation_every_turns,consolidation_association_created,consolidation_label_created,duration_sec,signals_per_sec,perf_encode_ms_mean,perf_process_ms_mean,perf_hydrate_ms_mean,perf_total_ms_mean,retrieval_turn_rate,retrieval_avg_candidates,retrieval_overlap_mean,retrieval_context_overlap_mean,retrieval_semantic_overlap_mean,retrieval_context_semantic_overlap_mean,retrieval_association_candidate_rate,retrieval_label_candidate_rate,retrieval_association_turn_rate,retrieval_label_turn_rate,interrupt_turn_rate,interrupt_abort_rate,interrupt_semantic_overlap_mean,interrupt_context_semantic_overlap_mean,interrupt_association_candidate_rate,interrupt_label_candidate_rate,interrupt_association_turn_rate,interrupt_label_turn_rate" > "$summary_csv"

get_metric() {
  local key="$1"
  local file="$2"
  awk -v k="$key" '
    $1 ~ "^"k"=" {split($1, a, "="); print a[2]; found=1; exit}
    END {if (!found) print "0"}
  ' "$file"
}

for cfg in "${configs[@]}"; do
  read -r F S T <<< "$cfg"
  label="F${F}_S${S}_T${T}"
  run_dir="$OUT_DIR/$label"
  mkdir -p "$run_dir"
  log="$run_dir/run.log"
  config="$run_dir/config.json"
  db_path="$run_dir/cortext.db"

  cat > "$config" <<EOF
{
  "focus": ${F},
  "sensitivity": ${S},
  "stability": ${T},
  "data": "${DATA}",
  "models": "${MODELS}",
  "db": "${db_path}",
  "binary": "${BIN}",
  "max_turns": ${MAX_TURNS},
  "max_total": ${MAX_TOTAL},
  "seed": ${SEED},
  "deterministic": ${DETERMINISTIC},
  "synthetic_start_ms": ${SYNTHETIC_START_MS},
  "consolidate": ${CONSOLIDATE},
  "consolidate_cycles": ${CONSOLIDATE_CYCLES},
  "consolidate_every": ${CONSOLIDATE_EVERY}
}
EOF

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
    if [[ "$CONSOLIDATE_EVERY" -gt 0 ]]; then
      extra_args+=(--consolidate-every="$CONSOLIDATE_EVERY")
    fi
  fi

  run_start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  SECONDS=0

  "$BIN" \
    --data="$DATA" \
    --models="$MODELS" \
    --db="$db_path" \
    --max-conversations=1 \
    --max-turns="$MAX_TURNS" \
    --max-total="$MAX_TOTAL" \
    --focus="$F" \
    --sensitivity="$S" \
    --stability="$T" \
    --reuse \
    --otel-filter=none \
    --no-cadence \
    --semantic \
    "${extra_args[@]}" 2>&1 | tee "$log"

  run_end_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

  echo "$F,$S,$T,$turns,$writes,$consolidation_runs,$consolidation_failures,$consolidation_every_turns,$consolidation_association_created,$consolidation_label_created,$duration_sec,$signals_per_sec,$perf_encode_ms_mean,$perf_process_ms_mean,$perf_hydrate_ms_mean,$perf_total_ms_mean,$retrieval_turn_rate,$retrieval_avg_candidates,$retrieval_overlap_mean,$retrieval_context_overlap_mean,$retrieval_semantic_overlap_mean,$retrieval_context_semantic_overlap_mean,$retrieval_association_candidate_rate,$retrieval_label_candidate_rate,$retrieval_association_turn_rate,$retrieval_label_turn_rate,$interrupt_turn_rate,$interrupt_abort_rate,$interrupt_semantic_overlap_mean,$interrupt_context_semantic_overlap_mean,$interrupt_association_candidate_rate,$interrupt_label_candidate_rate,$interrupt_association_turn_rate,$interrupt_label_turn_rate" >> "$summary_csv"

  perf="$run_dir/perf.json"
  export PERF_OUT="$perf"
  export PERF_START_TIME="$run_start_iso"
  export PERF_END_TIME="$run_end_iso"
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

  python - <<'PY'
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
done

echo "Wrote $summary_csv"
