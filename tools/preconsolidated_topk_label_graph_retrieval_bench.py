#!/usr/bin/env python3
"""Benchmark preconsolidated Top-K label graph retrieval.

This is benchmark-only. It does not read or write production Cortext databases.

The experiment models the proposed realtime path:

1. build an episode centroid from signal embeddings;
2. query a label vector store and attach Top-K provisional labels;
3. use the provisional label graph as a bounded retrieval expansion signal;
4. simulate consolidation by pruning labels that do not match offline truth.

Offline labels are used only for scoring and for the pruning simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+")

EXPECTED_TOKENS = {
    "dog_entity": {"dog", "retriever", "canine", "puppy", "animal", "bailey"},
    "car_event": {
        "car",
        "vehicle",
        "automobile",
        "crash",
        "collision",
        "accident",
        "wreck",
    },
}


@dataclass
class LabelMeta:
    label_index: int
    label_id: str
    source: str
    kind: str
    label: str


@dataclass
class LabelCandidate:
    label_index: int
    row_id: int
    score: float
    score01: float


@dataclass
class SignalRow:
    row_id: int
    scenario: str
    node_id: int
    memory_id: int
    modality: str
    offline_groups: tuple[str, ...]
    vector: np.ndarray
    labels: list[LabelCandidate] = field(default_factory=list)


@dataclass
class Episode:
    episode_id: str
    scenario: str
    memory_id: int
    signal_ids: tuple[int, ...]
    modalities: tuple[str, ...]
    offline_groups: tuple[str, ...]
    centroid: np.ndarray
    labels: list[LabelCandidate] = field(default_factory=list)
    pruned_labels: list[LabelCandidate] = field(default_factory=list)


def parse_pipe(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(x for x in value.split("|") if x)


def parse_int_pipe(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(x) for x in value.split("|") if x)


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=True)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def mean_normalized(vectors: Iterable[np.ndarray]) -> np.ndarray:
    rows = [v for v in vectors if v.size > 0]
    if not rows:
        return np.zeros((0,), dtype=np.float32)
    return normalize(np.mean(np.stack(rows), axis=0))


def label_tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "with",
        "for",
        "and",
        "or",
        "by",
        "someone",
        "something",
    }
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in stop and len(t) > 1}


def expected_tokens(groups: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for group in groups:
        out.update(EXPECTED_TOKENS.get(group, set()))
    return out


def label_matches_groups(label: LabelMeta, groups: Iterable[str]) -> bool:
    expected = expected_tokens(groups)
    return bool(expected and label_tokens(label.label) & expected)


def load_label_metadata(label_db_dir: Path) -> tuple[list[LabelMeta], np.ndarray, list[str], list[str]]:
    labels: list[LabelMeta] = []
    with (label_db_dir / "label_vector_db_labels.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            labels.append(
                LabelMeta(
                    label_index=int(row["label_index"]),
                    label_id=row["label_id"],
                    source=row["source"],
                    kind=row["kind"],
                    label=row["label"],
                )
            )

    row_label_indices: list[int] = []
    row_sources: list[str] = []
    row_kinds: list[str] = []
    with (label_db_dir / "label_vector_db_rows.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            row_label_indices.append(int(row["label_index"]))
            row_sources.append(row["source"])
            row_kinds.append(row["kind"])
    return labels, np.asarray(row_label_indices, dtype=np.int32), row_sources, row_kinds


def load_vectors(path: Path, row_count: int, dim: int) -> np.memmap:
    return np.memmap(path, dtype="<f4", mode="r", shape=(row_count, dim))


def label_row_filter(row_sources: list[str], row_kinds: list[str], name: str) -> np.ndarray:
    useful_kinds = {
        "object",
        "action",
        "event_action",
        "scene_event",
        "event",
        "entity_person",
        "entity_animal",
        "entity_plant",
        "name_or_entity",
        "place",
        "group",
        "process",
        "phenomenon",
        "communication",
        "communication_action",
        "social_action",
        "perception_action",
    }
    object_event_kinds = {
        "object",
        "action",
        "event_action",
        "scene_event",
        "event",
        "entity_animal",
        "name_or_entity",
        "place",
        "process",
        "phenomenon",
    }
    rows: list[int] = []
    for i, (source, kind) in enumerate(zip(row_sources, row_kinds)):
        if name == "all":
            rows.append(i)
        elif name == "wordnet_salt" and ("wordnet" in source or source == "salt.csv"):
            rows.append(i)
        elif name == "entities_objects_events" and kind in useful_kinds:
            rows.append(i)
        elif name == "objects_events" and kind in object_event_kinds:
            rows.append(i)
    return np.asarray(rows, dtype=np.int32)


def top_rows_for_query(
    query: np.ndarray,
    vectors: np.memmap,
    rows: np.ndarray,
    top_n: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    top_scores = np.empty((0,), dtype=np.float32)
    top_row_ids = np.empty((0,), dtype=np.int32)
    for start in range(0, len(rows), chunk_size):
        row_chunk = rows[start : start + chunk_size]
        scores = vectors[row_chunk] @ query
        if scores.size == 0:
            continue
        keep = min(top_n, scores.size)
        local = np.argpartition(scores, -keep)[-keep:]
        cand_scores = scores[local]
        cand_rows = row_chunk[local]
        if top_scores.size:
            cand_scores = np.concatenate([top_scores, cand_scores])
            cand_rows = np.concatenate([top_row_ids, cand_rows])
        keep = min(top_n, cand_scores.size)
        merged = np.argpartition(cand_scores, -keep)[-keep:]
        top_scores = cand_scores[merged]
        top_row_ids = cand_rows[merged]
    order = np.argsort(-top_scores)
    return top_row_ids[order], top_scores[order]


def candidates_for_query(
    query: np.ndarray,
    vectors: np.memmap,
    rows: np.ndarray,
    row_label_indices: np.ndarray,
    top_rows: int,
    top_labels: int,
    chunk_size: int,
) -> list[LabelCandidate]:
    row_ids, row_scores = top_rows_for_query(query, vectors, rows, top_rows, chunk_size)
    best_by_label: dict[int, LabelCandidate] = {}
    for row_id, score in zip(row_ids.tolist(), row_scores.tolist()):
        label_index = int(row_label_indices[row_id])
        score01 = max(0.0, min(1.0, (float(score) + 1.0) * 0.5))
        prev = best_by_label.get(label_index)
        if prev is None or score > prev.score:
            best_by_label[label_index] = LabelCandidate(
                label_index=label_index,
                row_id=int(row_id),
                score=float(score),
                score01=score01,
            )
    return sorted(best_by_label.values(), key=lambda c: c.score, reverse=True)[:top_labels]


def load_signal_rows(graph_dir: Path) -> list[SignalRow]:
    rows_path = graph_dir / "modality_agnostic_graph_embedding_rows.csv"
    vectors_path = graph_dir / "modality_agnostic_graph_embeddings.f32"
    metadata: list[dict[str, str]] = []
    with rows_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["node_type"] == "signal":
                metadata.append(row)

    total_rows = sum(1 for _ in rows_path.open()) - 1
    vectors = np.memmap(vectors_path, dtype="<f4", mode="r", shape=(total_rows, 256))

    out: list[SignalRow] = []
    for row in metadata:
        row_id = int(row["row_id"])
        out.append(
            SignalRow(
                row_id=row_id,
                scenario=row["scenario"],
                node_id=int(row["node_id"]),
                memory_id=int(row["memory_id"]),
                modality=row["modality"],
                offline_groups=parse_pipe(row["offline_groups"]),
                vector=normalize(np.asarray(vectors[row_id], dtype=np.float32)),
            )
        )
    return out


def build_episodes(signals: list[SignalRow], include_modalities: set[str]) -> list[Episode]:
    grouped: dict[tuple[str, int], list[SignalRow]] = defaultdict(list)
    for signal in signals:
        if include_modalities and signal.modality not in include_modalities:
            continue
        grouped[(signal.scenario, signal.memory_id)].append(signal)

    episodes: list[Episode] = []
    for (scenario, memory_id), rows in sorted(grouped.items()):
        centroid = mean_normalized(signal.vector for signal in rows)
        if centroid.size != 256:
            continue
        groups = sorted({g for signal in rows for g in signal.offline_groups})
        episodes.append(
            Episode(
                episode_id=f"{scenario}:{memory_id}",
                scenario=scenario,
                memory_id=memory_id,
                signal_ids=tuple(sorted(signal.node_id for signal in rows)),
                modalities=tuple(sorted({signal.modality for signal in rows})),
                offline_groups=tuple(groups),
                centroid=centroid,
            )
        )
    return episodes


def label_overlap_score(
    query_labels: list[LabelCandidate],
    episode_labels: list[LabelCandidate],
    k: int,
) -> float:
    left = {c.label_index: c.score01 for c in query_labels[:k]}
    right = {c.label_index: c.score01 for c in episode_labels[:k]}
    common = set(left) & set(right)
    if not common:
        return 0.0
    return float(max(min(left[i], right[i]) for i in common))


def same_group(query: SignalRow, episode: Episode) -> bool:
    return bool(set(query.offline_groups) & set(episode.offline_groups))


def candidate_pool(query: SignalRow, episodes: list[Episode], exclude_self: bool) -> list[Episode]:
    out: list[Episode] = []
    for episode in episodes:
        if exclude_self and query.scenario == episode.scenario and query.memory_id == episode.memory_id:
            continue
        if episode.offline_groups:
            out.append(episode)
    return out


def rank_episode_ids(
    query: SignalRow,
    episodes: list[Episode],
    label_k: int,
    label_weight: float,
    use_pruned: bool,
) -> list[tuple[Episode, float, float, float]]:
    ranked: list[tuple[Episode, float, float, float]] = []
    for episode in episodes:
        direct = float(np.dot(query.vector, episode.centroid))
        labels = episode.pruned_labels if use_pruned else episode.labels
        graph = label_overlap_score(query.labels, labels, label_k)
        score = direct + label_weight * graph
        ranked.append((episode, score, direct, graph))
    return sorted(ranked, key=lambda x: x[1], reverse=True)


def evaluate_retrieval(
    queries: list[SignalRow],
    episodes: list[Episode],
    label_k: int,
    label_weight: float,
    use_pruned: bool,
    exclude_self: bool,
) -> dict:
    totals = {
        "query_count": 0,
        "hit_at_1": 0,
        "hit_at_3": 0,
        "hit_at_5": 0,
        "cross_at_1": 0,
        "direct_mean_at_1": [],
        "graph_mean_at_1": [],
    }
    examples: list[dict] = []
    for query in queries:
        if not query.offline_groups:
            continue
        pool = candidate_pool(query, episodes, exclude_self)
        if not any(same_group(query, episode) for episode in pool):
            continue
        ranked = rank_episode_ids(query, pool, label_k, label_weight, use_pruned)
        if not ranked:
            continue
        totals["query_count"] += 1
        for k in (1, 3, 5):
            if any(same_group(query, episode) for episode, *_ in ranked[:k]):
                totals[f"hit_at_{k}"] += 1
        top_episode, score, direct, graph = ranked[0]
        if query.offline_groups and top_episode.offline_groups and not same_group(query, top_episode):
            totals["cross_at_1"] += 1
        totals["direct_mean_at_1"].append(direct)
        totals["graph_mean_at_1"].append(graph)
        if len(examples) < 20:
            examples.append(
                {
                    "query": f"{query.scenario}:signal_{query.node_id}",
                    "query_group": "|".join(query.offline_groups),
                    "top_episode": top_episode.episode_id,
                    "top_group": "|".join(top_episode.offline_groups),
                    "score": score,
                    "direct": direct,
                    "graph": graph,
                    "same_group": same_group(query, top_episode),
                }
            )

    n = totals["query_count"]
    return {
        "query_count": n,
        "hit_at_1": totals["hit_at_1"],
        "hit_at_3": totals["hit_at_3"],
        "hit_at_5": totals["hit_at_5"],
        "hit_at_1_rate": totals["hit_at_1"] / n if n else 0.0,
        "hit_at_3_rate": totals["hit_at_3"] / n if n else 0.0,
        "hit_at_5_rate": totals["hit_at_5"] / n if n else 0.0,
        "cross_at_1": totals["cross_at_1"],
        "cross_at_1_rate": totals["cross_at_1"] / n if n else 0.0,
        "top1_direct_mean": float(np.mean(totals["direct_mean_at_1"])) if n else 0.0,
        "top1_graph_mean": float(np.mean(totals["graph_mean_at_1"])) if n else 0.0,
        "examples": examples,
    }


def write_label_rows(
    path: Path,
    episodes: list[Episode],
    labels: list[LabelMeta],
    top_n: int,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_id",
                "scenario",
                "memory_id",
                "rank",
                "label_index",
                "label",
                "kind",
                "score",
                "score01",
                "matches_group",
                "kept_after_pruning",
                "offline_groups",
                "modalities",
            ]
        )
        for episode in episodes:
            kept = {label.label_index for label in episode.pruned_labels}
            for rank, cand in enumerate(episode.labels[:top_n], start=1):
                meta = labels[cand.label_index]
                writer.writerow(
                    [
                        episode.episode_id,
                        episode.scenario,
                        episode.memory_id,
                        rank,
                        cand.label_index,
                        meta.label,
                        meta.kind,
                        f"{cand.score:.9f}",
                        f"{cand.score01:.9f}",
                        int(label_matches_groups(meta, episode.offline_groups)),
                        int(cand.label_index in kept),
                        "|".join(episode.offline_groups),
                        "|".join(episode.modalities),
                    ]
                )


def write_retrieval_examples(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=Path("build/modality_agnostic_graph_bench_label_surface"))
    parser.add_argument("--label-db-dir", type=Path, default=Path("build/label_vector_db_full_2proto"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/preconsolidated_topk_label_graph_retrieval_bench"))
    parser.add_argument("--label-filter", choices=("all", "wordnet_salt", "entities_objects_events", "objects_events"), default="wordnet_salt")
    parser.add_argument("--include-modalities", default="all")
    parser.add_argument("--top-rows", type=int, default=768)
    parser.add_argument("--top-labels", type=int, default=50)
    parser.add_argument("--label-k", type=int, default=20)
    parser.add_argument("--label-weight", type=float, default=0.18)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--exclude-self", action="store_true", help="Exclude the query's own episode from retrieval.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    include_modalities = set()
    if args.include_modalities != "all":
        include_modalities = {m for m in args.include_modalities.split(",") if m}

    labels, row_label_indices, row_sources, row_kinds = load_label_metadata(args.label_db_dir)
    label_vectors = load_vectors(args.label_db_dir / "label_vector_db_256.f32", len(row_label_indices), 256)
    label_rows = label_row_filter(row_sources, row_kinds, args.label_filter)

    signals = load_signal_rows(args.graph_dir)
    queries = [s for s in signals if not include_modalities or s.modality in include_modalities]
    episodes = build_episodes(signals, include_modalities)

    label_start = time.perf_counter()
    label_latencies_ms: list[float] = []
    for episode in episodes:
        start = time.perf_counter()
        episode.labels = candidates_for_query(
            episode.centroid,
            label_vectors,
            label_rows,
            row_label_indices,
            args.top_rows,
            args.top_labels,
            args.chunk_size,
        )
        label_latencies_ms.append((time.perf_counter() - start) * 1000.0)
        episode.pruned_labels = [
            cand
            for cand in episode.labels
            if label_matches_groups(labels[cand.label_index], episode.offline_groups)
        ]
    for query in queries:
        query.labels = candidates_for_query(
            query.vector,
            label_vectors,
            label_rows,
            row_label_indices,
            args.top_rows,
            args.top_labels,
            args.chunk_size,
        )
    label_elapsed_s = time.perf_counter() - label_start

    baseline = evaluate_retrieval(
        queries,
        episodes,
        args.label_k,
        0.0,
        use_pruned=False,
        exclude_self=args.exclude_self,
    )
    preconsolidated = evaluate_retrieval(
        queries,
        episodes,
        args.label_k,
        args.label_weight,
        use_pruned=False,
        exclude_self=args.exclude_self,
    )
    pruned = evaluate_retrieval(
        queries,
        episodes,
        args.label_k,
        args.label_weight,
        use_pruned=True,
        exclude_self=args.exclude_self,
    )

    scored_episodes = [episode for episode in episodes if expected_tokens(episode.offline_groups)]
    episode_label_hit_topk = 0
    episode_label_any_false_topk = 0
    episode_label_count = 0
    episode_label_false_count = 0
    for episode in scored_episodes:
        top = episode.labels[: args.top_labels]
        if any(label_matches_groups(labels[c.label_index], episode.offline_groups) for c in top):
            episode_label_hit_topk += 1
        if any(not label_matches_groups(labels[c.label_index], episode.offline_groups) for c in top):
            episode_label_any_false_topk += 1
        episode_label_count += len(top)
        episode_label_false_count += sum(
            1 for c in top if not label_matches_groups(labels[c.label_index], episode.offline_groups)
        )

    result = {
        "benchmark_only": True,
        "production_behavior_changed": False,
        "experiment": "episode_centroid_topk_labels_preconsolidated_graph_retrieval",
        "graph_dir": str(args.graph_dir),
        "label_db_dir": str(args.label_db_dir),
        "label_filter": args.label_filter,
        "include_modalities": args.include_modalities,
        "top_rows": args.top_rows,
        "top_labels": args.top_labels,
        "label_k_for_retrieval": args.label_k,
        "label_weight": args.label_weight,
        "exclude_self": args.exclude_self,
        "signal_query_count": len(queries),
        "episode_count": len(episodes),
        "scored_episode_count": len(scored_episodes),
        "label_elapsed_s": label_elapsed_s,
        "episode_label_latency_ms": {
            "mean": float(np.mean(label_latencies_ms)) if label_latencies_ms else 0.0,
            "p95": float(np.percentile(label_latencies_ms, 95)) if label_latencies_ms else 0.0,
            "max": float(np.max(label_latencies_ms)) if label_latencies_ms else 0.0,
        },
        "episode_label_hit_topk": episode_label_hit_topk,
        "episode_label_hit_topk_rate": episode_label_hit_topk / len(scored_episodes) if scored_episodes else 0.0,
        "episode_any_false_label_topk_rate": episode_label_any_false_topk / len(scored_episodes) if scored_episodes else 0.0,
        "episode_false_label_fraction": episode_label_false_count / episode_label_count if episode_label_count else 0.0,
        "retrieval": {
            "baseline_direct_embedding": baseline,
            "preconsolidated_topk_label_graph": preconsolidated,
            "after_consolidation_pruning_sim": pruned,
        },
        "outputs": [
            "preconsolidated_topk_label_graph_results.json",
            "preconsolidated_episode_labels.csv",
            "retrieval_examples_preconsolidated.csv",
            "retrieval_examples_pruned.csv",
        ],
    }

    (args.output_dir / "preconsolidated_topk_label_graph_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_label_rows(args.output_dir / "preconsolidated_episode_labels.csv", episodes, labels, args.top_labels)
    write_retrieval_examples(
        args.output_dir / "retrieval_examples_preconsolidated.csv",
        preconsolidated["examples"],
    )
    write_retrieval_examples(
        args.output_dir / "retrieval_examples_pruned.csv",
        pruned["examples"],
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
