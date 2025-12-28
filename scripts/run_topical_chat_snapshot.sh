#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-build/examples/topical_chat_analysis/cortext_topical_chat_analysis}"
DATA="${DATA:-data/topical_chat/valid_freq.jsonl}"
MODELS="${MODELS:-models}"
MAX_TURNS="${MAX_TURNS:-120}"
MAX_TOTAL="${MAX_TOTAL:-120}"
FOCUS="${FOCUS:-0.5}"
SENSITIVITY="${SENSITIVITY:-0.5}"
STABILITY="${STABILITY:-0.5}"
CONSOLIDATE="${CONSOLIDATE:-1}"
CONSOLIDATE_CYCLES="${CONSOLIDATE_CYCLES:-2}"
OUT_DIR="${OUT_DIR:-logs/topical_chat_snapshots/$(date +%Y%m%d_%H%M%S)}"
DB_PATH="$OUT_DIR/cortext.db"

mkdir -p "$OUT_DIR"

log="$OUT_DIR/run.log"
summary="$OUT_DIR/summary.csv"
config="$OUT_DIR/config.json"
git_status="$OUT_DIR/git_status.txt"
git_diff="$OUT_DIR/git_diff.patch"

git rev-parse --short HEAD > "$OUT_DIR/git_rev.txt" 2>/dev/null || true
git status -sb > "$git_status" 2>/dev/null || true
git diff > "$git_diff" 2>/dev/null || true

cat > "$config" <<EOF
{
  "focus": ${FOCUS},
  "sensitivity": ${SENSITIVITY},
  "stability": ${STABILITY},
  "data": "${DATA}",
  "models": "${MODELS}",
  "db": "${DB_PATH}",
  "binary": "${BIN}",
  "max_turns": ${MAX_TURNS},
  "max_total": ${MAX_TOTAL},
  "consolidate": ${CONSOLIDATE},
  "consolidate_cycles": ${CONSOLIDATE_CYCLES}
}
EOF

echo "focus,sensitivity,stability,turns,writes,consolidation_runs,consolidation_failures,consolidation_every_turns,consolidation_association_created,consolidation_label_created,consolidation_summary_count,consolidation_summaries_with_model,consolidation_summaries_fallback,consolidation_extraction_jobs,consolidation_extraction_results,consolidation_labels_seen,consolidation_relations_seen,retrieval_turn_rate,retrieval_avg_candidates,retrieval_overlap_mean,retrieval_context_overlap_mean,retrieval_semantic_overlap_mean,retrieval_context_semantic_overlap_mean,retrieval_association_candidate_rate,retrieval_label_candidate_rate,retrieval_association_turn_rate,retrieval_label_turn_rate,interrupt_turn_rate,interrupt_semantic_overlap_mean,interrupt_context_semantic_overlap_mean,interrupt_association_candidate_rate,interrupt_label_candidate_rate,interrupt_association_turn_rate,interrupt_label_turn_rate" > "$summary"

get_metric() {
  local key="$1"
  local file="$2"
  awk -v k="$key" '
    $1 ~ "^"k"=" {split($1, a, "="); print a[2]; found=1; exit}
    END {if (!found) print "0"}
  ' "$file"
}

extra_args=()
if [[ "$CONSOLIDATE" == "1" ]]; then
  extra_args+=(--consolidate --consolidate-cycles="$CONSOLIDATE_CYCLES")
fi

"$BIN" \
  --data="$DATA" \
  --models="$MODELS" \
  --db="$DB_PATH" \
  --max-conversations=1 \
  --max-turns="$MAX_TURNS" \
  --max-total="$MAX_TOTAL" \
  --focus="$FOCUS" \
  --sensitivity="$SENSITIVITY" \
  --stability="$STABILITY" \
  --reuse \
  --otel-filter=none \
  --no-cadence \
  --semantic \
  "${extra_args[@]}" | tee "$log"

turns="$(get_metric turns "$log")"
writes="$(get_metric writes "$log")"
consolidation_runs="$(get_metric consolidation_runs "$log")"
consolidation_failures="$(get_metric consolidation_failures "$log")"
consolidation_every_turns="$(get_metric consolidation_every_turns "$log")"
consolidation_association_created="$(get_metric consolidation_association_created "$log")"
consolidation_label_created="$(get_metric consolidation_label_created "$log")"
consolidation_summary_count="$(get_metric consolidation_summary_count "$log")"
consolidation_summaries_with_model="$(get_metric consolidation_summaries_with_model "$log")"
consolidation_summaries_fallback="$(get_metric consolidation_summaries_fallback "$log")"
consolidation_extraction_jobs="$(get_metric consolidation_extraction_jobs "$log")"
consolidation_extraction_results="$(get_metric consolidation_extraction_results "$log")"
consolidation_labels_seen="$(get_metric consolidation_labels_seen "$log")"
consolidation_relations_seen="$(get_metric consolidation_relations_seen "$log")"
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
interrupt_semantic_overlap_mean="$(get_metric interrupt_semantic_overlap_mean "$log")"
interrupt_context_semantic_overlap_mean="$(get_metric interrupt_context_semantic_overlap_mean "$log")"
interrupt_association_candidate_rate="$(get_metric interrupt_association_candidate_rate "$log")"
interrupt_label_candidate_rate="$(get_metric interrupt_label_candidate_rate "$log")"
interrupt_association_turn_rate="$(get_metric interrupt_association_turn_rate "$log")"
interrupt_label_turn_rate="$(get_metric interrupt_label_turn_rate "$log")"

echo "${FOCUS},${SENSITIVITY},${STABILITY},${turns},${writes},${consolidation_runs},${consolidation_failures},${consolidation_every_turns},${consolidation_association_created},${consolidation_label_created},${consolidation_summary_count},${consolidation_summaries_with_model},${consolidation_summaries_fallback},${consolidation_extraction_jobs},${consolidation_extraction_results},${consolidation_labels_seen},${consolidation_relations_seen},${retrieval_turn_rate},${retrieval_avg_candidates},${retrieval_overlap_mean},${retrieval_context_overlap_mean},${retrieval_semantic_overlap_mean},${retrieval_context_semantic_overlap_mean},${retrieval_association_candidate_rate},${retrieval_label_candidate_rate},${retrieval_association_turn_rate},${retrieval_label_turn_rate},${interrupt_turn_rate},${interrupt_semantic_overlap_mean},${interrupt_context_semantic_overlap_mean},${interrupt_association_candidate_rate},${interrupt_label_candidate_rate},${interrupt_association_turn_rate},${interrupt_label_turn_rate}" >> "$summary"

echo "Wrote snapshot to $OUT_DIR"
