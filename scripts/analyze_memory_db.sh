#!/bin/bash
# analyze_memory_db.sh - Analyze cortext memory database for algorithm diagnostics
#
# Usage: ./scripts/analyze_memory_db.sh [database_path]
# Default: examples/chat/chat_memory.db

set -e

DB_PATH="${1:-examples/chat/chat_memory.db}"

if [ ! -f "$DB_PATH" ]; then
    echo "Error: Database not found at $DB_PATH"
    exit 1
fi

echo "=============================================="
echo "Cortext Memory Database Analysis"
echo "Database: $DB_PATH"
echo "=============================================="
echo

# Summary counts
echo "=== SUMMARY ==="
sqlite3 "$DB_PATH" "
SELECT 'Signals Processed:' as metric, signals_processed as value FROM processor_state
UNION ALL
SELECT 'Memories Stored:', COUNT(*) FROM memories
UNION ALL
SELECT 'Active Segments:', COUNT(*) FROM segment_state WHERE n_signals > 0
UNION ALL
SELECT 'Episodes:', COUNT(*) FROM episodes
UNION ALL
SELECT 'Graph Nodes:', COUNT(*) FROM graph_nodes
UNION ALL
SELECT 'Graph Edges:', COUNT(*) FROM graph_edges;
" | column -t -s '|'
echo

# Write gate effectiveness
echo "=== WRITE GATE EFFECTIVENESS ==="
sqlite3 "$DB_PATH" "
SELECT
    COUNT(*) as total_signals,
    SUM(write_decision) as accepted,
    COUNT(*) - SUM(write_decision) as rejected,
    printf('%.1f%%', 100.0 * SUM(write_decision) / COUNT(*)) as acceptance_rate
FROM signal_metrics;
"
echo

# Threshold vs Score analysis
# Note: Actual write decision uses (T_dynamic - hysteresis) as effective threshold
echo "=== THRESHOLD vs COMPOSITE SCORE ==="
HYSTERESIS=$(sqlite3 "$DB_PATH" "SELECT hysteresis FROM processor_state LIMIT 1;" 2>/dev/null || echo "0.0")
sqlite3 -header "$DB_PATH" "
SELECT
    id,
    printf('%.4f', composite_score) as score,
    printf('%.4f', threshold_t) as T_dynamic,
    printf('%.4f', threshold_t - $HYSTERESIS) as eff_thresh,
    printf('%.4f', composite_score - (threshold_t - $HYSTERESIS)) as margin,
    CASE WHEN write_decision = 1 THEN 'ACCEPT' ELSE 'REJECT' END as decision,
    CASE
        WHEN composite_score > (threshold_t - $HYSTERESIS) AND write_decision = 1 THEN 'OK'
        WHEN composite_score <= (threshold_t - $HYSTERESIS) AND write_decision = 0 THEN 'OK'
        ELSE 'MISMATCH'
    END as consistency
FROM signal_metrics
ORDER BY id;
"
echo

# Segment state analysis
echo "=== SEGMENT STATE (per source) ==="
sqlite3 -header "$DB_PATH" "
SELECT
    source_id,
    n_signals as pending,
    printf('%.3f', s_sum) as score_sum,
    printf('%.3f', s_max) as score_max,
    printf('%.3f', drift_seg) as drift_accum,
    printf('%.3f', eta_seg) as drift_ewma,
    printf('%.3f', coherence_prev) as coherence,
    CASE WHEN last_write_ts > 0
        THEN datetime(last_write_ts/1000, 'unixepoch', 'localtime')
        ELSE 'never'
    END as last_write
FROM segment_state
ORDER BY source_id;
"
echo

# Check if segment writes occurred
echo "=== SEGMENT WRITE ANALYSIS ==="
SEGMENT_WRITES=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM segment_state WHERE last_write_ts > 0;")
if [ "$SEGMENT_WRITES" -eq 0 ]; then
    echo "WARNING: No segment writes have occurred (all last_write_ts = 0)"
    echo "This means signals are being written via per-signal gate, not segment gate."
    echo
    echo "Possible reasons:"
    echo "  1. No segment boundaries detected (continuous similar content)"
    echo "  2. Segment window scores below threshold"
    echo "  3. Signals written before segment accumulation completes"
else
    echo "Segment writes: $SEGMENT_WRITES sources have written segments"
fi
echo

# Metrics distribution
echo "=== METRIC DISTRIBUTIONS (signal_metrics) ==="
sqlite3 "$DB_PATH" "
SELECT
    'relevance' as metric,
    printf('%.3f', MIN(relevance)) as min,
    printf('%.3f', AVG(relevance)) as avg,
    printf('%.3f', MAX(relevance)) as max
FROM signal_metrics WHERE relevance IS NOT NULL
UNION ALL
SELECT 'mismatch', printf('%.3f', MIN(mismatch)), printf('%.3f', AVG(mismatch)), printf('%.3f', MAX(mismatch))
FROM signal_metrics WHERE mismatch IS NOT NULL
UNION ALL
SELECT 'surprise', printf('%.3f', MIN(surprise)), printf('%.3f', AVG(surprise)), printf('%.3f', MAX(surprise))
FROM signal_metrics WHERE surprise IS NOT NULL
UNION ALL
SELECT 'rarity', printf('%.3f', MIN(rarity)), printf('%.3f', AVG(rarity)), printf('%.3f', MAX(rarity))
FROM signal_metrics WHERE rarity IS NOT NULL
UNION ALL
SELECT 'drift', printf('%.3f', MIN(drift)), printf('%.3f', AVG(drift)), printf('%.3f', MAX(drift))
FROM signal_metrics WHERE drift IS NOT NULL
UNION ALL
SELECT 'salience', printf('%.3f', MIN(salience)), printf('%.3f', AVG(salience)), printf('%.3f', MAX(salience))
FROM signal_metrics WHERE salience IS NOT NULL
UNION ALL
SELECT 'composite', printf('%.3f', MIN(composite_score)), printf('%.3f', AVG(composite_score)), printf('%.3f', MAX(composite_score))
FROM signal_metrics WHERE composite_score IS NOT NULL
UNION ALL
SELECT 'threshold', printf('%.3f', MIN(threshold_t)), printf('%.3f', AVG(threshold_t)), printf('%.3f', MAX(threshold_t))
FROM signal_metrics WHERE threshold_t IS NOT NULL;
" | column -t -s '|'
echo

# Processor state
echo "=== PROCESSOR STATE ==="
sqlite3 -header "$DB_PATH" "
SELECT
    signals_processed,
    printf('%.4f', theta_dynamic) as theta_dynamic,
    printf('%.4f', hysteresis) as hysteresis,
    printf('%.4f', theta_dynamic - hysteresis) as effective_threshold,
    printf('%.4f', u_uncertainty) as uncertainty,
    printf('%.4f', half_life) as half_life,
    printf('%.4f', rate_target) as rate_target
FROM processor_state;
"
echo

# Memory timeline
# Note: Check if timestamps are in seconds (< 10B) or milliseconds (> 10B)
echo "=== MEMORY TIMELINE (last 10) ==="
sqlite3 -header "$DB_PATH" "
SELECT
    embedding_id as id,
    source_id,
    modality,
    CASE
        WHEN timestamp > 10000000000 THEN datetime(timestamp/1000, 'unixepoch', 'localtime')
        ELSE datetime(timestamp, 'unixepoch', 'localtime')
    END as time
FROM memories
ORDER BY timestamp DESC
LIMIT 10;
"
echo

# Recent context
echo "=== RECENT CONTEXT WINDOW ==="
RECENT_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM recent_context;")
echo "Embeddings in recent context: $RECENT_COUNT"
echo

# Working memory
echo "=== WORKING MEMORY SLOTS ==="
sqlite3 -header "$DB_PATH" "
SELECT
    slot_index,
    printf('%.4f', strength) as strength,
    CASE
        WHEN timestamp > 10000000000 THEN datetime(timestamp/1000, 'unixepoch', 'localtime')
        ELSE datetime(timestamp, 'unixepoch', 'localtime')
    END as updated
FROM working_memory_slots
ORDER BY slot_index;
"
echo

echo "=============================================="
echo "Analysis complete"
echo "=============================================="
