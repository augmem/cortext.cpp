#!/usr/bin/env python3
"""Private-safe structural audit for Julie judge losses.

The output intentionally excludes packet text and judge reason text. It keeps
only scores, counts, event indices, token counts, and coarse loss categories.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import pathlib
from statistics import mean


SYSTEMS = ["cortext_native", "traditional_chat_rag", "full_history_upper_bound"]
FIELDS = [
    "relevance",
    "sufficiency",
    "noise",
    "temporal_correctness",
    "source_grounding",
    "modality_grounding",
]


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def score(row: dict, system: str, field: str) -> float:
    return float(row.get("systems", {}).get(system, {}).get(field, 0.0) or 0.0)


def composite(row: dict, system: str) -> float:
    return (
        score(row, system, "relevance")
        + score(row, system, "sufficiency")
        - 0.25 * score(row, system, "noise")
    )


def media_total(row: dict, system: str) -> int:
    media = row.get("media_attachments", {}).get(system, {})
    if not isinstance(media, dict):
        return 0
    return int(media.get("image", 0) or 0) + int(media.get("audio", 0) or 0)


def classify_loss(row: dict, winner: str) -> list[str]:
    tags = []
    if winner not in SYSTEMS:
        return ["tie_or_unclear"]
    eps = 1e-9
    if score(row, "cortext_native", "relevance") + eps < score(row, winner, "relevance"):
        tags.append("relevance_gap")
    if score(row, "cortext_native", "sufficiency") + eps < score(row, winner, "sufficiency"):
        tags.append("sufficiency_gap")
    if score(row, "cortext_native", "noise") > score(row, winner, "noise") + eps:
        tags.append("noise_penalty")
    if score(row, "cortext_native", "temporal_correctness") + eps < score(
        row, winner, "temporal_correctness"
    ):
        tags.append("temporal_gap")
    if score(row, "cortext_native", "source_grounding") + eps < score(
        row, winner, "source_grounding"
    ):
        tags.append("source_grounding_gap")
    if score(row, "cortext_native", "modality_grounding") + eps < score(
        row, winner, "modality_grounding"
    ):
        tags.append("modality_grounding_gap")
    if media_total(row, "cortext_native") > 0 and media_total(row, winner) == 0:
        tags.append("cortext_media_only")
    if not tags:
        tags.append("winner_preference_without_score_gap")
    return tags


def aggregate_scores(rows: list[dict]) -> dict:
    out = {}
    for system in SYSTEMS:
        out[system] = {
            "wins": sum(1 for row in rows if row.get("winner") == system),
            "mean_composite": mean([composite(row, system) for row in rows]) if rows else 0.0,
        }
        for field in FIELDS:
            out[system][f"mean_{field}"] = (
                mean([score(row, system, field) for row in rows]) if rows else 0.0
            )
    return out


def mean_field_delta(rows: list[dict], left: str, right: str, field: str) -> float:
    if not rows:
        return 0.0
    return mean([score(row, left, field) - score(row, right, field) for row in rows])


def score_deltas(rows: list[dict], baseline: str) -> dict:
    return {
        field: mean_field_delta(rows, "cortext_native", baseline, field)
        for field in FIELDS
    }


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def winner_counts(rows: list[dict]) -> dict:
    return dict(Counter(str(row.get("winner", "")) for row in rows))


def numeric(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def retrieval_debug_by_event(summary: dict | None) -> dict[int, dict]:
    if not summary:
        return {}
    probes = summary.get("probes", [])
    if not isinstance(probes, list):
        return {}
    out: dict[int, dict] = {}
    for probe in probes:
        if not isinstance(probe, dict):
            continue
        debug = probe.get("cortext_retrieval_debug")
        if not isinstance(debug, dict):
            continue
        try:
            event_index = int(probe.get("event_index", 0) or 0)
        except (TypeError, ValueError):
            continue
        if event_index > 0:
            out[event_index] = debug
    return out


def probes_by_event(summary: dict | None) -> dict[int, dict]:
    if not summary:
        return {}
    probes = summary.get("probes", [])
    if not isinstance(probes, list):
        return {}
    out: dict[int, dict] = {}
    for probe in probes:
        if not isinstance(probe, dict):
            continue
        try:
            event_index = int(probe.get("event_index", 0) or 0)
        except (TypeError, ValueError):
            continue
        if event_index > 0:
            out[event_index] = probe
    return out


def candidate_signal_summary(debug: dict) -> dict:
    candidates = debug.get("ranked_candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    candidates = [row for row in candidates if isinstance(row, dict)]
    if not candidates:
        return {
            "candidate_count": 0,
            "top_score": 0.0,
            "top_relevance": 0.0,
            "positive_label_graph_count": 0,
            "positive_durable_source_count": 0,
            "positive_fact_count": 0,
            "positive_proc_count": 0,
            "positive_predictive_count": 0,
            "mean_score": 0.0,
            "mean_relevance": 0.0,
            "mean_label_graph_boost": 0.0,
            "mean_durable_source_boost": 0.0,
            "mean_fact_boost": 0.0,
            "mean_proc_score": 0.0,
            "mean_predictive_bonus": 0.0,
        }
    top = candidates[0]
    return {
        "candidate_count": len(candidates),
        "top_score": numeric(top.get("score")),
        "top_relevance": numeric(top.get("relevance")),
        "positive_label_graph_count": sum(
            1 for row in candidates if numeric(row.get("label_graph_boost")) > 0.0
        ),
        "positive_durable_source_count": sum(
            1 for row in candidates if numeric(row.get("durable_source_boost")) > 0.0
        ),
        "positive_fact_count": sum(
            1
            for row in candidates
            if numeric(row.get("fact_boost")) > 0.0
            or int(row.get("linked_fact_count", 0) or 0) > 0
        ),
        "positive_proc_count": sum(
            1 for row in candidates if numeric(row.get("proc_score")) > 0.0
        ),
        "positive_predictive_count": sum(
            1 for row in candidates if numeric(row.get("predictive_bonus")) > 0.0
        ),
        "mean_score": mean_or_zero([numeric(row.get("score")) for row in candidates]),
        "mean_relevance": mean_or_zero(
            [numeric(row.get("relevance")) for row in candidates]
        ),
        "mean_label_graph_boost": mean_or_zero(
            [numeric(row.get("label_graph_boost")) for row in candidates]
        ),
        "mean_durable_source_boost": mean_or_zero(
            [numeric(row.get("durable_source_boost")) for row in candidates]
        ),
        "mean_fact_boost": mean_or_zero(
            [numeric(row.get("fact_boost")) for row in candidates]
        ),
        "mean_proc_score": mean_or_zero(
            [numeric(row.get("proc_score")) for row in candidates]
        ),
        "mean_predictive_bonus": mean_or_zero(
            [numeric(row.get("predictive_bonus")) for row in candidates]
        ),
    }


def retrieval_row_summary(row: dict, debug_by_event: dict[int, dict]) -> dict:
    try:
        event_index = int(row.get("event_index", 0) or 0)
    except (TypeError, ValueError):
        event_index = 0
    debug = debug_by_event.get(event_index, {})
    if not debug:
        return {"event_index": event_index, "has_retrieval_debug": False}
    summary = candidate_signal_summary(debug)
    return {
        "event_index": event_index,
        "has_retrieval_debug": True,
        "fact_layer_enabled": bool(debug.get("fact_layer_enabled")),
        "fact_seed_count": int(debug.get("fact_seed_count", 0) or 0),
        "fact_text_candidate_count": int(
            debug.get("fact_text_candidate_count", 0) or 0
        ),
        "fact_text_rejected_low_score_count": int(
            debug.get("fact_text_rejected_low_score_count", 0) or 0
        ),
        "fact_text_match_count": int(debug.get("fact_text_match_count", 0) or 0),
        "fact_text_best_score": numeric(debug.get("fact_text_best_score")),
        "selected_fact_linked_count": int(
            debug.get("selected_fact_linked_count", 0) or 0
        ),
        "text_query_wm_slots": int(debug.get("text_query_wm_slots", 0) or 0),
        "text_query_token_count": int(debug.get("text_query_token_count", 0) or 0),
        **summary,
    }


def summarize_retrieval_debug(rows: list[dict], debug_by_event: dict[int, dict]) -> dict:
    per_row = [retrieval_row_summary(row, debug_by_event) for row in rows]
    present = [row for row in per_row if row.get("has_retrieval_debug")]
    numeric_fields = [
        "candidate_count",
        "top_score",
        "top_relevance",
        "positive_label_graph_count",
        "positive_durable_source_count",
        "positive_fact_count",
        "positive_proc_count",
        "positive_predictive_count",
        "mean_score",
        "mean_relevance",
        "mean_label_graph_boost",
        "mean_durable_source_boost",
        "mean_fact_boost",
        "mean_proc_score",
        "mean_predictive_bonus",
        "fact_seed_count",
        "fact_text_candidate_count",
        "fact_text_rejected_low_score_count",
        "fact_text_match_count",
        "fact_text_best_score",
        "selected_fact_linked_count",
        "text_query_wm_slots",
        "text_query_token_count",
    ]
    aggregates = {
        f"mean_{field}": mean_or_zero([numeric(row.get(field)) for row in present])
        for field in numeric_fields
    }
    return {
        "rows": len(rows),
        "rows_with_retrieval_debug": len(present),
        "coverage": len(present) / len(rows) if rows else 0.0,
        **aggregates,
        "per_loss_rows": per_row,
    }


def packet_row_summary(row: dict, probe_by_event: dict[int, dict]) -> dict:
    try:
        event_index = int(row.get("event_index", 0) or 0)
    except (TypeError, ValueError):
        event_index = 0
    probe = probe_by_event.get(event_index, {})
    if not probe:
        return {"event_index": event_index, "has_packet_summary": False}

    judged_ids = {
        int(memory_id)
        for memory_id in row.get("cortext_judged_memory_ids", []) or []
        if isinstance(memory_id, int) or str(memory_id).isdigit()
    }
    working = [
        memory
        for memory in probe.get("cortext_frozen_working_memory", []) or []
        if isinstance(memory, dict)
    ]
    retrieved = [
        memory
        for memory in probe.get("cortext_frozen_retrieved_memory", []) or []
        if isinstance(memory, dict)
    ]
    working_ids = {int(memory.get("memory_id", 0) or 0) for memory in working}
    retrieved_ids = {int(memory.get("memory_id", 0) or 0) for memory in retrieved}
    packet_by_id: dict[int, dict] = {}
    for memory in [*working, *retrieved]:
        memory_id = int(memory.get("memory_id", 0) or 0)
        if memory_id > 0:
            packet_by_id[memory_id] = memory
    selected = [
        packet_by_id[memory_id]
        for memory_id in judged_ids
        if memory_id in packet_by_id
    ]
    query = row.get("query", {})
    if not isinstance(query, dict):
        query = probe.get("query", {}) if isinstance(probe.get("query"), dict) else {}
    query_ts = numeric(query.get("timestamp"), 0.0)

    modalities = Counter(str(memory.get("modality", "")) for memory in selected)
    mimetypes = Counter(str(memory.get("mimetype", "")) for memory in selected)
    source_ids = [str(memory.get("source_id", "")) for memory in selected]
    source_id_counts = Counter(source_ids)
    query_source = str(query.get("source_id", ""))
    token_values = [numeric(memory.get("tokens")) for memory in selected]
    ages_days = []
    future_count = 0
    same_day_count = 0
    old_7d_count = 0
    old_30d_count = 0
    reaction_marker_count = 0
    text_memory_count = 0
    text_empty_count = 0
    for memory in selected:
        text = str(memory.get("content_text", ""))
        if str(memory.get("modality", "")) == "text" or str(memory.get("mimetype", "")) == "text/plain":
            text_memory_count += 1
            if not text.strip():
                text_empty_count += 1
            if "You reacted with" in text:
                reaction_marker_count += 1
        timestamp = numeric(memory.get("timestamp"), 0.0)
        if query_ts > 0.0 and timestamp > 0.0:
            age_days = (query_ts - timestamp) / 86400000.0
            ages_days.append(age_days)
            if age_days < -1e-9:
                future_count += 1
            if abs(age_days) <= 1.0:
                same_day_count += 1
            if age_days >= 7.0:
                old_7d_count += 1
            if age_days >= 30.0:
                old_30d_count += 1

    return {
        "event_index": event_index,
        "has_packet_summary": True,
        "judged_memory_count": len(judged_ids),
        "resolved_memory_count": len(selected),
        "working_memory_count": sum(
            1 for memory_id in judged_ids if memory_id in working_ids
        ),
        "retrieved_memory_count": sum(
            1 for memory_id in judged_ids if memory_id in retrieved_ids
        ),
        "text_memory_count": text_memory_count,
        "media_memory_count": len(selected) - text_memory_count,
        "image_memory_count": modalities.get("image", 0),
        "audio_memory_count": modalities.get("audio", 0),
        "video_memory_count": modalities.get("video", 0),
        "text_empty_count": text_empty_count,
        "reaction_marker_count": reaction_marker_count,
        "total_tokens": sum(token_values),
        "mean_memory_tokens": mean_or_zero(token_values),
        "unique_source_id_count": len(source_id_counts),
        "same_source_as_query_count": sum(
            1 for source_id in source_ids if source_id == query_source
        ),
        "max_same_source_repeat_count": max(source_id_counts.values(), default=0),
        "future_memory_count": future_count,
        "same_day_memory_count": same_day_count,
        "old_7d_memory_count": old_7d_count,
        "old_30d_memory_count": old_30d_count,
        "mean_age_days": mean_or_zero(ages_days),
        "max_age_days": max(ages_days, default=0.0),
        "current_turn_excluded_count": len(
            row.get("cortext_current_turn_memory_ids_excluded", []) or []
        ),
        "text_plain_count": mimetypes.get("text/plain", 0),
    }


def summarize_packet_composition(
    rows: list[dict], probe_by_event: dict[int, dict]
) -> dict:
    per_row = [packet_row_summary(row, probe_by_event) for row in rows]
    present = [row for row in per_row if row.get("has_packet_summary")]
    numeric_fields = [
        "judged_memory_count",
        "resolved_memory_count",
        "working_memory_count",
        "retrieved_memory_count",
        "text_memory_count",
        "media_memory_count",
        "image_memory_count",
        "audio_memory_count",
        "video_memory_count",
        "text_empty_count",
        "reaction_marker_count",
        "total_tokens",
        "mean_memory_tokens",
        "unique_source_id_count",
        "same_source_as_query_count",
        "max_same_source_repeat_count",
        "future_memory_count",
        "same_day_memory_count",
        "old_7d_memory_count",
        "old_30d_memory_count",
        "mean_age_days",
        "max_age_days",
        "current_turn_excluded_count",
        "text_plain_count",
    ]
    aggregates = {
        f"mean_{field}": mean_or_zero([numeric(row.get(field)) for row in present])
        for field in numeric_fields
    }
    return {
        "rows": len(rows),
        "rows_with_packet_summary": len(present),
        "coverage": len(present) / len(rows) if rows else 0.0,
        **aggregates,
        "per_rows": per_row,
    }


def rag_phase(row: dict, probe_by_event: dict[int, dict] | None = None) -> str:
    probe: dict = {}
    if probe_by_event is not None:
        try:
            event_index = int(row.get("event_index", 0) or 0)
        except (TypeError, ValueError):
            event_index = 0
        probe = probe_by_event.get(event_index, {})
    source = probe if probe else row
    rolling_tokens = int(source.get("rolling_history_tokens", 0) or 0)
    rag_tokens = int(
        source.get(
            "normal_rag_context_tokens",
            source.get("traditional_chat_rag_tokens", 0),
        )
        or 0
    )
    compaction_events = int(source.get("normal_rag_compaction_events", 0) or 0)
    compacted_items = int(source.get("normal_rag_compacted_history_items", 0) or 0)
    rag_additional = source.get("rag_top_k_additional", [])
    rag_additional_count = len(rag_additional) if isinstance(rag_additional, list) else 0
    if rolling_tokens <= 0:
        return "unknown"
    if compaction_events > 0 or compacted_items > 0:
        return "post_compaction"
    if rag_tokens > rolling_tokens or rag_additional_count > 0:
        return "pre_compaction_vector_augmented"
    return "pre_compaction_raw_history"


def packet_size_bucket(row: dict) -> str:
    count = len(row.get("cortext_judged_memory_ids", []) or [])
    if count <= 12:
        return "small_0_12"
    if count <= 20:
        return "medium_13_20"
    return "large_21_plus"


def summarize_subset(rows: list[dict]) -> dict:
    losses = [
        row
        for row in rows
        if row.get("winner") in {"traditional_chat_rag", "full_history_upper_bound"}
    ]
    return {
        "rows": len(rows),
        "winner_counts": winner_counts(rows),
        "losses_vs_cortext": len(losses),
        "mean_cortext_composite": mean_or_zero(
            [composite(row, "cortext_native") for row in rows]
        ),
        "mean_traditional_chat_rag_composite": mean_or_zero(
            [composite(row, "traditional_chat_rag") for row in rows]
        ),
        "mean_full_history_upper_bound_composite": mean_or_zero(
            [composite(row, "full_history_upper_bound") for row in rows]
        ),
        "mean_cortext_minus_traditional_chat_rag_composite": mean_or_zero(
            [
                composite(row, "cortext_native")
                - composite(row, "traditional_chat_rag")
                for row in rows
            ]
        ),
        "mean_cortext_minus_full_history_upper_bound_composite": mean_or_zero(
            [
                composite(row, "cortext_native")
                - composite(row, "full_history_upper_bound")
                for row in rows
            ]
        ),
        "mean_cortext_context_tokens": mean_or_zero(
            [float(row.get("cortext_context_tokens", 0) or 0) for row in rows]
        ),
        "mean_traditional_chat_rag_tokens": mean_or_zero(
            [float(row.get("traditional_chat_rag_tokens", 0) or 0) for row in rows]
        ),
        "mean_cortext_memory_count": mean_or_zero(
            [float(len(row.get("cortext_judged_memory_ids", []) or [])) for row in rows]
        ),
        "mean_rag_top_k_count": mean_or_zero(
            [float(len(row.get("rag_top_k_indices", []) or [])) for row in rows]
        ),
    }


def grouped_summary(rows: list[dict], classifier) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(classifier(row)), []).append(row)
    return {key: summarize_subset(value) for key, value in sorted(grouped.items())}


def query_modality(row: dict) -> str:
    query = row.get("query", {})
    if not isinstance(query, dict):
        return ""
    return str(query.get("modality", ""))


def audit(judge: dict, summary: dict | None = None) -> dict:
    rows = judge.get("judgments", [])
    if not isinstance(rows, list):
        raise RuntimeError("judge artifact does not contain judgments[]")
    rows = [row for row in rows if isinstance(row, dict)]
    debug_by_event = retrieval_debug_by_event(summary)
    probe_by_event = probes_by_event(summary)
    loss_rows = [
        row
        for row in rows
        if row.get("winner") in {"traditional_chat_rag", "full_history_upper_bound"}
    ]
    tie_rows = [row for row in rows if row.get("winner") == "tie_or_unclear"]
    cortext_win_rows = [row for row in rows if row.get("winner") == "cortext_native"]
    cortext_nonwin_rows = [
        row for row in rows if row.get("winner") != "cortext_native"
    ]

    loss_tag_counts = Counter()
    loss_by_winner = Counter()
    loss_by_query_modality = Counter()
    per_loss = []
    for row in loss_rows:
        winner = str(row.get("winner", ""))
        tags = classify_loss(row, winner)
        loss_tag_counts.update(tags)
        loss_by_winner[winner] += 1
        loss_by_query_modality[query_modality(row)] += 1
        per_loss.append(
            {
                "event_index": int(row.get("event_index", 0) or 0),
                "query_modality": query_modality(row),
                "winner": winner,
                "failure_reason": row.get("failure_reason", ""),
                "tags": tags,
                "cortext_composite": composite(row, "cortext_native"),
                "winner_composite": composite(row, winner),
                "composite_delta_vs_winner": composite(row, "cortext_native")
                - composite(row, winner),
                "cortext_context_tokens": int(row.get("cortext_context_tokens", 0) or 0),
                "traditional_chat_rag_tokens": int(
                    row.get("traditional_chat_rag_tokens", 0) or 0
                ),
                "cortext_memory_count": len(row.get("cortext_judged_memory_ids", []) or []),
                "rag_top_k_count": len(row.get("rag_top_k_indices", []) or []),
                "cortext_media_attachments": media_total(row, "cortext_native"),
            }
        )

    quality = aggregate_scores(rows)
    cortext_composite = quality["cortext_native"]["mean_composite"]
    rag_composite = quality["traditional_chat_rag"]["mean_composite"]
    full_composite = quality["full_history_upper_bound"]["mean_composite"]
    tokens = judge.get("tokens", {})
    return {
        "schema": "cortext_julie_private_judge_loss_audit_v1",
        "source_judge": judge.get("summary_path", ""),
        "judge_provider": judge.get("judge_provider", ""),
        "judge_model": judge.get("judge_model", ""),
        "probe_count": int(judge.get("probe_count", len(rows)) or len(rows)),
        "judged_rows": len(rows),
        "privacy": "private local artifact; no packet text or judge reason text",
        "quality_composite_definition": "relevance + sufficiency - 0.25*noise",
        "winner_counts": dict(Counter(str(row.get("winner", "")) for row in rows)),
        "failure_reason_counts": dict(
            Counter(str(row.get("failure_reason", "")) for row in rows)
        ),
        "losses_vs_cortext": {
            "count": len(loss_rows),
            "by_winner": dict(loss_by_winner),
            "by_query_modality": dict(loss_by_query_modality),
            "tag_counts": dict(loss_tag_counts),
            "negative_cortext_composite_count": sum(
                1 for row in loss_rows if composite(row, "cortext_native") < 0.0
            ),
        },
        "ties": {"count": len(tie_rows)},
        "mean_scores": quality,
        "mean_quality_delta": {
            "cortext_minus_traditional_chat_rag": cortext_composite - rag_composite,
            "cortext_minus_full_history_upper_bound": cortext_composite
            - full_composite,
        },
        "mean_score_deltas": {
            "cortext_minus_traditional_chat_rag": score_deltas(
                rows, "traditional_chat_rag"
            ),
            "cortext_minus_full_history_upper_bound": score_deltas(
                rows, "full_history_upper_bound"
            ),
        },
        "loss_score_deltas": {
            "cortext_minus_winning_packet": {
                field: mean(
                    [
                        score(row, "cortext_native", field)
                        - score(row, str(row.get("winner", "")), field)
                        for row in loss_rows
                    ]
                )
                if loss_rows
                else 0.0
                for field in FIELDS
            }
        },
        "structural_breakdowns": {
            "by_rag_phase": grouped_summary(
                rows, lambda row: rag_phase(row, probe_by_event)
            ),
            "losses_by_rag_phase": grouped_summary(
                loss_rows, lambda row: rag_phase(row, probe_by_event)
            ),
            "by_cortext_packet_size": grouped_summary(rows, packet_size_bucket),
            "losses_by_cortext_packet_size": grouped_summary(
                loss_rows, packet_size_bucket
            ),
            "interpretation": (
                "pre_compaction_raw_history means the traditional baseline still "
                "fits the raw rolling chat history. pre_compaction_vector_augmented "
                "means the baseline adds vector RAG before compaction. post_compaction "
                "means the baseline includes compacted prior chat. Packet size buckets "
                "count Cortext memories actually exposed to the judge after "
                "current-turn exclusion."
            ),
        },
        "retrieval_debug_summary": {
            "all_rows": summarize_retrieval_debug(rows, debug_by_event),
            "loss_rows": summarize_retrieval_debug(loss_rows, debug_by_event),
            "tie_rows": summarize_retrieval_debug(tie_rows, debug_by_event),
            "cortext_win_rows": summarize_retrieval_debug(cortext_win_rows, debug_by_event),
            "cortext_nonwin_rows": summarize_retrieval_debug(
                cortext_nonwin_rows, debug_by_event
            ),
            "interpretation": (
                "Candidate debug is emitted by the production retrieval operation "
                "and joined by probe event index. It records aggregate ranking "
                "signals only: no packet text, judge reason text, or source blobs."
            ),
        },
        "packet_composition_summary": {
            "all_rows": summarize_packet_composition(rows, probe_by_event),
            "loss_rows": summarize_packet_composition(loss_rows, probe_by_event),
            "tie_rows": summarize_packet_composition(tie_rows, probe_by_event),
            "cortext_win_rows": summarize_packet_composition(
                cortext_win_rows, probe_by_event
            ),
            "cortext_nonwin_rows": summarize_packet_composition(
                cortext_nonwin_rows, probe_by_event
            ),
            "interpretation": (
                "Packet composition is joined from frozen probe packets and "
                "judge-exposed memory ids after current-turn exclusion. It "
                "reports counts, tokens, modality mix, source repetition, and "
                "relative age only; no packet text or source blob content is "
                "included in this audit artifact."
            ),
        },
        "token_summary": {
            "mean_cortext_context_tokens": tokens.get("mean_cortext_context_tokens"),
            "mean_traditional_chat_rag_tokens": tokens.get(
                "mean_traditional_chat_rag_tokens"
            ),
            "mean_cortext_token_savings_vs_traditional_chat_rag": tokens.get(
                "mean_cortext_token_savings_vs_traditional_chat_rag"
            ),
        },
        "fairness_checks": judge.get("fairness_checks", {}),
        "judge_validation": judge.get("judge_validation", {}),
        "loss_rows": per_loss,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", type=pathlib.Path, required=True)
    parser.add_argument(
        "--summary",
        type=pathlib.Path,
        help="Optional live-run summary containing cortext_retrieval_debug probes.",
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    summary = load_json(args.summary) if args.summary else None
    body = audit(load_json(args.judge), summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
