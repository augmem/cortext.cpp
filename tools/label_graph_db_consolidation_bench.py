#!/usr/bin/env python3
"""Benchmark label-vector graph accumulation over real Cortext memory rows.

This is benchmark-only. It does not write to Cortext databases and does not
change production retrieval or consolidation behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class LabelMeta:
    label_index: int
    label_id: str
    source: str
    kind: str
    label: str
    begin: int
    end: int


@dataclass
class LabelCandidate:
    label_index: int
    row_id: int
    score: float
    score01: float


@dataclass
class MemoryRecord:
    scenario: str
    node_type: str
    node_id: int
    memory_id: int
    embedding: np.ndarray
    offline_groups: tuple[str, ...]
    modalities: tuple[str, ...]
    signal_ids: tuple[int, ...]
    source_id: str
    start_ts: int
    end_ts: int
    n_signals: int
    top_by_filter: dict[str, list[LabelCandidate]] = field(default_factory=dict)


@dataclass
class EdgeRow:
    scenario: str
    variant: str
    source_node_type: str
    target_node_type: str
    source_memory_id: int
    target_memory_id: int
    edge_type: str
    weight: float
    same_offline_group: bool
    cross_offline_group: bool
    source_groups: str
    target_groups: str
    detail: str


def csv_escape_join(items: Iterable[object], sep: str = "|") -> str:
    return sep.join(str(x) for x in items)


def parse_ids(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(x) for x in value.split("|") if x)


def parse_groups(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(x for x in value.split("|") if x)


def read_truth_units(path: Path) -> dict[tuple[str, int], dict[str, tuple]]:
    truth: dict[tuple[str, int], dict[str, tuple]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["variant"] != "memory_row_units":
                continue
            memory_ids = parse_ids(row["memory_ids"])
            if len(memory_ids) != 1:
                continue
            key = (row["scenario"], memory_ids[0])
            truth[key] = {
                "offline_groups": parse_groups(row["offline_groups"]),
                "modalities": parse_groups(row["modalities"]),
                "signal_ids": parse_ids(row["signal_ids"]),
            }
    return truth


def open_sqlite_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def decode_vec(blob: bytes) -> np.ndarray:
    if blob is None:
        return np.zeros((0,), dtype=np.float32)
    if len(blob) % 4 != 0:
        return np.zeros((0,), dtype=np.float32)
    vec = np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def load_memories(graph_dir: Path, truth: dict[tuple[str, int], dict[str, tuple]]) -> list[MemoryRecord]:
    memories: list[MemoryRecord] = []
    for db_path in sorted(graph_dir.glob("*.sqlite")):
        scenario = db_path.stem
        with open_sqlite_ro(db_path) as con:
            rows = con.execute(
                """
                SELECT memory_id, context, source_id, start_ts,
                       COALESCE(end_ts, start_ts, created_at, 0) AS end_ts,
                       modality, n_signals
                FROM memories
                WHERE kind = 'LONG_TERM' AND context IS NOT NULL
                ORDER BY memory_id
                """
            ).fetchall()
        for memory_id, context, source_id, start_ts, end_ts, modality, n_signals in rows:
            emb = decode_vec(context)
            if emb.size != 256:
                continue
            t = truth.get((scenario, int(memory_id)), {})
            memories.append(
                MemoryRecord(
                    scenario=scenario,
                    node_type="memory",
                    node_id=int(memory_id),
                    memory_id=int(memory_id),
                    embedding=emb,
                    offline_groups=tuple(t.get("offline_groups", ())),
                    modalities=tuple(t.get("modalities", (modality,))),
                    signal_ids=tuple(t.get("signal_ids", ())),
                    source_id=str(source_id),
                    start_ts=int(start_ts),
                    end_ts=int(end_ts),
                    n_signals=int(n_signals),
                )
            )
    return memories


def load_embedding_surface(graph_dir: Path, surface: str) -> list[MemoryRecord]:
    rows_path = graph_dir / "modality_agnostic_graph_embedding_rows.csv"
    vectors_path = graph_dir / "modality_agnostic_graph_embeddings.f32"
    if not rows_path.exists() or not vectors_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with rows_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if surface != "all" and row["node_type"] != surface:
                continue
            rows.append(row)
    vectors = np.memmap(vectors_path, dtype="<f4", mode="r", shape=(sum(1 for _ in rows_path.open()) - 1, 256))
    out: list[MemoryRecord] = []
    for row in rows:
        row_id = int(row["row_id"])
        emb = np.asarray(vectors[row_id], dtype=np.float32).copy()
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb /= norm
        out.append(
            MemoryRecord(
                scenario=row["scenario"],
                node_type=row["node_type"],
                node_id=int(row["node_id"]),
                memory_id=int(row["memory_id"]),
                embedding=emb,
                offline_groups=parse_groups(row["offline_groups"]),
                modalities=parse_groups(row["modalities"]),
                signal_ids=parse_ids(row["signal_ids"]),
                source_id="",
                start_ts=0,
                end_ts=0,
                n_signals=len(parse_ids(row["signal_ids"])),
            )
        )
    return out


def load_label_metadata(label_db_dir: Path) -> tuple[list[LabelMeta], np.ndarray, list[str], list[str], list[str]]:
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
                    begin=int(row["prototype_begin"]),
                    end=int(row["prototype_end"]),
                )
            )

    row_label_indices: list[int] = []
    row_sources: list[str] = []
    row_kinds: list[str] = []
    row_prompts: list[str] = []
    with (label_db_dir / "label_vector_db_rows.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["label_index"])
            row_label_indices.append(idx)
            row_sources.append(row["source"])
            row_kinds.append(row["kind"])
            row_prompts.append(row["prompt"])
    return labels, np.asarray(row_label_indices, dtype=np.int32), row_sources, row_kinds, row_prompts


def load_vectors(label_db_dir: Path, row_count: int, dim: int) -> np.memmap:
    return np.memmap(
        label_db_dir / "label_vector_db_256.f32",
        dtype="<f4",
        mode="r",
        shape=(row_count, dim),
    )


def row_filter_indices(row_sources: list[str], row_kinds: list[str]) -> dict[str, np.ndarray]:
    all_rows = np.arange(len(row_sources), dtype=np.int32)
    filters: dict[str, list[int]] = {
        "all": [],
        "wordnet_only": [],
        "wordnet_salt": [],
        "entities_objects_events": [],
        "objects_events": [],
        "salt_only": [],
    }
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
    for i, (source, kind) in enumerate(zip(row_sources, row_kinds)):
        filters["all"].append(i)
        if "wordnet" in source:
            filters["wordnet_only"].append(i)
            filters["wordnet_salt"].append(i)
        if source == "salt.csv":
            filters["salt_only"].append(i)
            filters["wordnet_salt"].append(i)
        if kind in useful_kinds:
            filters["entities_objects_events"].append(i)
        if kind in object_event_kinds:
            filters["objects_events"].append(i)
    out = {name: np.asarray(rows, dtype=np.int32) for name, rows in filters.items()}
    out["all"] = all_rows
    return out


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


TOKEN_RE = re.compile(r"[a-z0-9]+")


def label_tokens(label: str) -> set[str]:
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
    return {t for t in TOKEN_RE.findall(label.lower()) if t not in stop and len(t) > 1}


EXPECTED_TOKENS = {
    "dog_entity": {"dog", "retriever", "canine", "puppy", "animal", "bailey"},
    "car_event": {"car", "vehicle", "automobile", "crash", "collision", "accident", "wreck"},
}


def group_hit(labels: list[LabelMeta], candidates: list[LabelCandidate], groups: tuple[str, ...], k: int) -> bool:
    expected: set[str] = set()
    for group in groups:
        expected.update(EXPECTED_TOKENS.get(group, set()))
    if not expected:
        return False
    for cand in candidates[:k]:
        if label_tokens(labels[cand.label_index].label) & expected:
            return True
    return False


def exact_overlap_score(a: list[LabelCandidate], b: list[LabelCandidate], k: int) -> tuple[float, str]:
    left = {c.label_index: c.score01 for c in a[:k]}
    right = {c.label_index: c.score01 for c in b[:k]}
    common = set(left) & set(right)
    if not common:
        return 0.0, ""
    score = sum(min(left[i], right[i]) for i in common) / max(1, len(common))
    detail = "|".join(str(i) for i in sorted(common)[:8])
    return float(score), detail


def token_overlap_score(labels: list[LabelMeta], a: list[LabelCandidate], b: list[LabelCandidate], k: int) -> tuple[float, str]:
    best = 0.0
    best_detail = ""
    for ca in a[:k]:
        ta = label_tokens(labels[ca.label_index].label)
        if not ta:
            continue
        for cb in b[:k]:
            tb = label_tokens(labels[cb.label_index].label)
            if not tb:
                continue
            inter = ta & tb
            if not inter:
                continue
            jaccard = len(inter) / len(ta | tb)
            score = math.sqrt(ca.score01 * cb.score01) * jaccard
            if score > best:
                best = score
                best_detail = f"{labels[ca.label_index].label}<->{labels[cb.label_index].label}"
    return float(best), best_detail


def label_vector_score(
    vectors: np.memmap,
    a: list[LabelCandidate],
    b: list[LabelCandidate],
    k: int,
) -> tuple[float, str]:
    if not a or not b:
        return 0.0, ""
    left = a[:k]
    right = b[:k]
    left_vecs = vectors[np.asarray([c.row_id for c in left], dtype=np.int32)]
    right_vecs = vectors[np.asarray([c.row_id for c in right], dtype=np.int32)]
    sims = left_vecs @ right_vecs.T
    weights = np.sqrt(
        np.outer(
            np.asarray([c.score01 for c in left], dtype=np.float32),
            np.asarray([c.score01 for c in right], dtype=np.float32),
        )
    )
    scores = weights * np.clip((sims + 1.0) * 0.5, 0.0, 1.0)
    if scores.size == 0:
        return 0.0, ""
    idx = int(np.argmax(scores))
    i, j = divmod(idx, scores.shape[1])
    return float(scores[i, j]), f"{left[i].label_index}<->{right[j].label_index}"


def same_group(a: MemoryRecord, b: MemoryRecord) -> bool:
    return bool(set(a.offline_groups) & set(b.offline_groups))


def cross_group(a: MemoryRecord, b: MemoryRecord) -> bool:
    return bool(a.offline_groups and b.offline_groups and not same_group(a, b))


def build_edges(
    memories: list[MemoryRecord],
    labels: list[LabelMeta],
    vectors: np.memmap,
    filter_name: str,
) -> list[EdgeRow]:
    specs = [
        ("label_exact_top20_t0.55", "label_exact", 20, 0.55),
        ("label_exact_top20_t0.70", "label_exact", 20, 0.70),
        ("label_exact_top50_t0.55", "label_exact", 50, 0.55),
        ("label_token_top20_t0.32", "label_token", 20, 0.32),
        ("label_token_top20_t0.45", "label_token", 20, 0.45),
        ("label_token_top20_t0.55", "label_token", 20, 0.55),
        ("label_token_top50_t0.32", "label_token", 50, 0.32),
        ("label_neighbor_top20_t0.84", "label_neighbor", 20, 0.84),
        ("label_neighbor_top20_t0.90", "label_neighbor", 20, 0.90),
        ("label_neighbor_top20_t0.94", "label_neighbor", 20, 0.94),
        ("label_neighbor_top50_t0.87", "label_neighbor", 50, 0.87),
        ("label_neighbor_top50_t0.92", "label_neighbor", 50, 0.92),
        ("label_hybrid_top20_t0.82", "label_hybrid", 20, 0.82),
        ("label_hybrid_top20_t0.88", "label_hybrid", 20, 0.88),
        ("label_hybrid_top20_t0.92", "label_hybrid", 20, 0.92),
        ("label_hybrid_top50_t0.84", "label_hybrid", 50, 0.84),
        ("label_hybrid_top50_t0.90", "label_hybrid", 50, 0.90),
    ]
    edges: list[EdgeRow] = []
    by_scenario: dict[str, list[MemoryRecord]] = defaultdict(list)
    for memory in memories:
        if filter_name in memory.top_by_filter:
            by_scenario[memory.scenario].append(memory)
    for scenario, rows in by_scenario.items():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a = rows[i]
                b = rows[j]
                cand_a = a.top_by_filter[filter_name]
                cand_b = b.top_by_filter[filter_name]
                mem_cos = float(np.dot(a.embedding, b.embedding))
                for variant, kind, k, threshold in specs:
                    if kind == "label_exact":
                        score, detail = exact_overlap_score(cand_a, cand_b, k)
                    elif kind == "label_token":
                        score, detail = token_overlap_score(labels, cand_a, cand_b, k)
                    elif kind == "label_neighbor":
                        score, detail = label_vector_score(vectors, cand_a, cand_b, k)
                    elif kind == "label_hybrid":
                        label_score, detail = label_vector_score(vectors, cand_a, cand_b, k)
                        score = 0.45 * max(0.0, min(1.0, (mem_cos + 1.0) * 0.5)) + 0.55 * label_score
                    else:
                        score, detail = 0.0, ""
                    if score >= threshold:
                        edges.append(
                            EdgeRow(
                                scenario=scenario,
                                variant=f"{filter_name}:{variant}",
                                source_node_type=a.node_type,
                                target_node_type=b.node_type,
                                source_memory_id=a.node_id,
                                target_memory_id=b.node_id,
                                edge_type=kind,
                                weight=score,
                                same_offline_group=same_group(a, b),
                                cross_offline_group=cross_group(a, b),
                                source_groups=csv_escape_join(a.offline_groups),
                                target_groups=csv_escape_join(b.offline_groups),
                                detail=detail,
                            )
                        )
    return edges


def cluster_variant(memories: list[MemoryRecord], edges: list[EdgeRow], variant: str) -> list[list[MemoryRecord]]:
    rows = [m for m in memories if any(e.scenario == m.scenario and e.variant == variant for e in edges) or True]
    by_scenario: dict[str, list[MemoryRecord]] = defaultdict(list)
    for memory in rows:
        by_scenario[memory.scenario].append(memory)
    edge_by_scenario: dict[str, list[EdgeRow]] = defaultdict(list)
    for edge in edges:
        if edge.variant == variant:
            edge_by_scenario[edge.scenario].append(edge)

    clusters: list[list[MemoryRecord]] = []
    for scenario, scenario_memories in by_scenario.items():
        index = {(m.node_type, m.node_id): i for i, m in enumerate(scenario_memories)}
        parent = list(range(len(scenario_memories)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def unite(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for edge in edge_by_scenario[scenario]:
            left = (edge.source_node_type, edge.source_memory_id)
            right = (edge.target_node_type, edge.target_memory_id)
            if left in index and right in index:
                unite(index[left], index[right])
        grouped: dict[int, list[MemoryRecord]] = defaultdict(list)
        for i, memory in enumerate(scenario_memories):
            grouped[find(i)].append(memory)
        clusters.extend(grouped.values())
    return clusters


def summarize_variant(memories: list[MemoryRecord], edges: list[EdgeRow], variant: str) -> dict:
    variant_edges = [e for e in edges if e.variant == variant]
    possible_same = 0
    possible_cross = 0
    by_scenario: dict[str, list[MemoryRecord]] = defaultdict(list)
    for memory in memories:
        by_scenario[memory.scenario].append(memory)
    for rows in by_scenario.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if not rows[i].offline_groups or not rows[j].offline_groups:
                    continue
                if same_group(rows[i], rows[j]):
                    possible_same += 1
                elif cross_group(rows[i], rows[j]):
                    possible_cross += 1
    clusters = cluster_variant(memories, edges, variant)
    covered_node_ids = set()
    mixed_units = 0
    multi_modal_units = 0
    for cluster in clusters:
        groups: set[str] = set()
        modalities: set[str] = set()
        for memory in cluster:
            covered_node_ids.add((memory.scenario, memory.node_type, memory.node_id))
            groups.update(memory.offline_groups)
            modalities.update(memory.modalities)
        if len(groups) > 1:
            mixed_units += 1
        if len(modalities) > 1:
            multi_modal_units += 1
    same_edges = sum(1 for e in variant_edges if e.same_offline_group)
    cross_edges = sum(1 for e in variant_edges if e.cross_offline_group)
    scored_edges = same_edges + cross_edges
    return {
        "variant": variant,
        "memory_count_total": len(memories),
        "edge_count": len(variant_edges),
        "same_group_edges": same_edges,
        "cross_group_edges": cross_edges,
        "scored_edge_count": scored_edges,
        "possible_same_group_pairs": possible_same,
        "possible_cross_group_pairs": possible_cross,
        "edge_precision": (
            same_edges / len(variant_edges)
            if variant_edges
            else 1.0
        ),
        "edge_cross_rate": (
            cross_edges / len(variant_edges)
            if variant_edges
            else 0.0
        ),
        "scored_edge_precision": (same_edges / scored_edges if scored_edges else 1.0),
        "same_group_pair_recall": (same_edges / possible_same if possible_same else 0.0),
        "cross_group_pair_rate": (cross_edges / possible_cross if possible_cross else 0.0),
        "unit_count": len(clusters),
        "mixed_units": mixed_units,
        "multi_modal_units": multi_modal_units,
        "unit_precision": ((len(clusters) - mixed_units) / len(clusters) if clusters else 1.0),
        "memory_coverage": (len(covered_node_ids) / len(memories) if memories else 0.0),
    }


def write_memory_labels(path: Path, memories: list[MemoryRecord], labels: list[LabelMeta], filters: list[str], top_k: int) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "node_type",
                "node_id",
                "memory_id",
                "filter",
                "rank",
                "label_index",
                "label_id",
                "kind",
                "label",
                "score",
                "score01",
                "offline_groups",
                "modalities",
                "group_hit_top10",
                "group_hit_top25",
                "group_hit_top50",
            ]
        )
        for memory in memories:
            for filter_name in filters:
                candidates = memory.top_by_filter.get(filter_name, [])
                hit10 = group_hit(labels, candidates, memory.offline_groups, 10)
                hit25 = group_hit(labels, candidates, memory.offline_groups, 25)
                hit50 = group_hit(labels, candidates, memory.offline_groups, 50)
                for rank, cand in enumerate(candidates[:top_k], start=1):
                    label = labels[cand.label_index]
                    w.writerow(
                        [
                            memory.scenario,
                            memory.node_type,
                            memory.node_id,
                            memory.memory_id,
                            filter_name,
                            rank,
                            cand.label_index,
                            label.label_id,
                            label.kind,
                            label.label,
                            f"{cand.score:.9f}",
                            f"{cand.score01:.9f}",
                            csv_escape_join(memory.offline_groups),
                            csv_escape_join(memory.modalities),
                            int(hit10),
                            int(hit25),
                            int(hit50),
                        ]
                    )


def write_edges(path: Path, edges: list[EdgeRow]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "variant",
                "source_node_type",
                "target_node_type",
                "source_memory_id",
                "target_memory_id",
                "edge_type",
                "weight",
                "same_offline_group",
                "cross_offline_group",
                "source_groups",
                "target_groups",
                "detail",
            ]
        )
        for edge in edges:
            w.writerow(
                [
                    edge.scenario,
                    edge.variant,
                    edge.source_node_type,
                    edge.target_node_type,
                    edge.source_memory_id,
                    edge.target_memory_id,
                    edge.edge_type,
                    f"{edge.weight:.9f}",
                    int(edge.same_offline_group),
                    int(edge.cross_offline_group),
                    edge.source_groups,
                    edge.target_groups,
                    edge.detail,
                ]
            )


def write_clusters(path: Path, memories: list[MemoryRecord], edges: list[EdgeRow], variants: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "cluster_id", "scenario", "nodes", "memory_ids", "offline_groups", "modalities", "mixed_offline_groups"])
        next_cluster = 0
        for variant in variants:
            for cluster in cluster_variant(memories, edges, variant):
                groups: set[str] = set()
                modalities: set[str] = set()
                scenario = cluster[0].scenario if cluster else ""
                for memory in cluster:
                    groups.update(memory.offline_groups)
                    modalities.update(memory.modalities)
                w.writerow(
                    [
                        variant,
                        next_cluster,
                        scenario,
                        csv_escape_join(f"{m.node_type}:{m.node_id}" for m in cluster),
                        csv_escape_join(sorted({m.memory_id for m in cluster})),
                        csv_escape_join(sorted(groups)),
                        csv_escape_join(sorted(modalities)),
                        int(len(groups) > 1),
                    ]
                )
                next_cluster += 1


def write_failures(path: Path, memories: list[MemoryRecord], labels: list[LabelMeta], filter_name: str) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "node_type", "node_id", "memory_id", "offline_groups", "modalities", "failure_type", "top_labels"])
        for memory in memories:
            if not memory.offline_groups:
                continue
            candidates = memory.top_by_filter.get(filter_name, [])
            if group_hit(labels, candidates, memory.offline_groups, 25):
                continue
            top = []
            for cand in candidates[:8]:
                label = labels[cand.label_index]
                top.append(f"{label.kind}:{label.label}:{cand.score:.4f}")
            w.writerow(
                [
                    memory.scenario,
                    memory.node_type,
                    memory.node_id,
                    memory.memory_id,
                    csv_escape_join(memory.offline_groups),
                    csv_escape_join(memory.modalities),
                    "expected_group_not_in_top25",
                    " | ".join(top),
                ]
            )


def write_label_summary(path: Path, memories: list[MemoryRecord], labels: list[LabelMeta], filters: list[str]) -> list[dict]:
    rows: list[dict] = []
    scored = [m for m in memories if m.offline_groups]
    for filter_name in filters:
        row = {
            "filter": filter_name,
            "scored_memories": len(scored),
            "hit_at_10": 0,
            "hit_at_25": 0,
            "hit_at_50": 0,
            "mean_top1_score": 0.0,
        }
        top1_scores: list[float] = []
        for memory in scored:
            candidates = memory.top_by_filter.get(filter_name, [])
            row["hit_at_10"] += int(group_hit(labels, candidates, memory.offline_groups, 10))
            row["hit_at_25"] += int(group_hit(labels, candidates, memory.offline_groups, 25))
            row["hit_at_50"] += int(group_hit(labels, candidates, memory.offline_groups, 50))
            if candidates:
                top1_scores.append(candidates[0].score)
        if scored:
            row["hit_at_10_rate"] = row["hit_at_10"] / len(scored)
            row["hit_at_25_rate"] = row["hit_at_25"] / len(scored)
            row["hit_at_50_rate"] = row["hit_at_50"] / len(scored)
        else:
            row["hit_at_10_rate"] = 0.0
            row["hit_at_25_rate"] = 0.0
            row["hit_at_50_rate"] = 0.0
        row["mean_top1_score"] = float(np.mean(top1_scores)) if top1_scores else 0.0
        rows.append(row)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["filter"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return rows


def group_hits_for_candidates(
    labels: list[LabelMeta],
    candidates: list[LabelCandidate],
    groups: Iterable[str],
    k: int,
) -> dict[str, bool]:
    hits: dict[str, bool] = {}
    for group in groups:
        expected = EXPECTED_TOKENS.get(group, set())
        if not expected:
            continue
        hit = False
        for cand in candidates[:k]:
            if label_tokens(labels[cand.label_index].label) & expected:
                hit = True
                break
        hits[group] = hit
    return hits


def aggregate_candidates(
    signals: list[MemoryRecord],
    filter_name: str,
    variant: str,
) -> list[LabelCandidate]:
    by_label: dict[int, tuple[LabelCandidate, set[str], int]] = {}
    for signal in signals:
        modality = signal.modalities[0] if signal.modalities else ""
        if variant.startswith("text_only") and modality != "text":
            continue
        if variant.startswith("non_text") and modality == "text":
            continue
        for cand in signal.top_by_filter.get(filter_name, []):
            prev = by_label.get(cand.label_index)
            if prev is None or cand.score > prev[0].score:
                by_label[cand.label_index] = (cand, {modality}, 1)
            else:
                prev[1].add(modality)
                by_label[cand.label_index] = (prev[0], prev[1], prev[2] + 1)

    out: list[LabelCandidate] = []
    for cand, modalities, support_count in by_label.values():
        score = cand.score
        score01 = cand.score01
        if variant.endswith("_support"):
            score01 = min(1.0, score01 + 0.04 * max(0, len(modalities) - 1) + 0.01 * max(0, support_count - 1))
            score = score01 * 2.0 - 1.0
        out.append(
            LabelCandidate(
                label_index=cand.label_index,
                row_id=cand.row_id,
                score=float(score),
                score01=float(score01),
            )
        )
    return sorted(out, key=lambda c: c.score01, reverse=True)


def write_memory_aggregate_labels(
    path: Path,
    memories: list[MemoryRecord],
    labels: list[LabelMeta],
    filters: list[str],
    top_k: int,
) -> list[dict]:
    signal_rows = [m for m in memories if m.node_type == "signal"]
    if not signal_rows:
        return []
    grouped: dict[tuple[str, int], list[MemoryRecord]] = defaultdict(list)
    for signal in signal_rows:
        grouped[(signal.scenario, signal.memory_id)].append(signal)

    variants = [
        "all_signals_max",
        "all_signals_support",
        "text_only_max",
        "text_only_support",
        "non_text_max",
        "non_text_support",
    ]
    summary_rows: list[dict] = []
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "memory_id",
                "filter",
                "aggregation_variant",
                "rank",
                "label_index",
                "label",
                "kind",
                "score01",
                "offline_groups",
                "modalities",
                "all_groups_hit_top10",
                "any_group_hit_top10",
                "mixed_offline_groups",
            ]
        )
        for filter_name in filters:
            for variant in variants:
                total = 0
                any_hit = 0
                all_hit = 0
                mixed_total = 0
                mixed_all_hit = 0
                for (scenario, memory_id), signals in grouped.items():
                    groups = sorted({g for s in signals for g in s.offline_groups})
                    if not groups:
                        continue
                    modalities = sorted({m for s in signals for m in s.modalities})
                    candidates = aggregate_candidates(signals, filter_name, variant)
                    if not candidates:
                        continue
                    hits = group_hits_for_candidates(labels, candidates, groups, 10)
                    if not hits:
                        continue
                    total += 1
                    hit_values = list(hits.values())
                    this_any = any(hit_values)
                    this_all = all(hit_values)
                    any_hit += int(this_any)
                    all_hit += int(this_all)
                    if len(groups) > 1:
                        mixed_total += 1
                        mixed_all_hit += int(this_all)
                    for rank, cand in enumerate(candidates[:top_k], start=1):
                        label = labels[cand.label_index]
                        w.writerow(
                            [
                                scenario,
                                memory_id,
                                filter_name,
                                variant,
                                rank,
                                cand.label_index,
                                label.label,
                                label.kind,
                                f"{cand.score01:.9f}",
                                csv_escape_join(groups),
                                csv_escape_join(modalities),
                                int(this_all),
                                int(this_any),
                                int(len(groups) > 1),
                            ]
                        )
                summary_rows.append(
                    {
                        "filter": filter_name,
                        "aggregation_variant": variant,
                        "scored_memory_count": total,
                        "any_group_hit_top10": any_hit,
                        "all_groups_hit_top10": all_hit,
                        "any_group_hit_top10_rate": any_hit / total if total else 0.0,
                        "all_groups_hit_top10_rate": all_hit / total if total else 0.0,
                        "mixed_memory_count": mixed_total,
                        "mixed_all_groups_hit_top10": mixed_all_hit,
                        "mixed_all_groups_hit_top10_rate": mixed_all_hit / mixed_total if mixed_total else 0.0,
                    }
                )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=Path("build/modality_agnostic_graph_bench"))
    parser.add_argument("--label-db-dir", type=Path, default=Path("build/label_vector_db_full_2proto"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/label_graph_db_consolidation_bench"))
    parser.add_argument("--top-rows", type=int, default=512)
    parser.add_argument("--top-labels", type=int, default=50)
    parser.add_argument("--write-top-labels", type=int, default=15)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--filters", default="all,wordnet_only,wordnet_salt,entities_objects_events,objects_events")
    parser.add_argument(
        "--surface",
        choices=("memory", "signal", "all"),
        default="memory",
        help="Use memory rows from sqlite contexts, or exported signal/all embedding surface rows.",
    )
    parser.add_argument(
        "--include-modalities",
        default="all",
        help="Comma-separated modality filter for surface rows, or all.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    truth = read_truth_units(args.graph_dir / "modality_agnostic_consolidation_units.csv")
    memories = load_embedding_surface(args.graph_dir, args.surface)
    if not memories:
        memories = load_memories(args.graph_dir, truth)
    if args.include_modalities != "all":
        allowed_modalities = {m for m in args.include_modalities.split(",") if m}
        memories = [
            m
            for m in memories
            if set(m.modalities) & allowed_modalities
        ]
    labels, row_label_indices, row_sources, row_kinds, _row_prompts = load_label_metadata(args.label_db_dir)
    vectors = load_vectors(args.label_db_dir, len(row_label_indices), 256)
    filters_by_name = row_filter_indices(row_sources, row_kinds)
    filters = [f for f in args.filters.split(",") if f]

    encode_start = time.perf_counter()
    per_filter_latency_ms: dict[str, list[float]] = defaultdict(list)
    for filter_name in filters:
        rows = filters_by_name[filter_name]
        for memory in memories:
            start = time.perf_counter()
            memory.top_by_filter[filter_name] = candidates_for_query(
                memory.embedding,
                vectors,
                rows,
                row_label_indices,
                args.top_rows,
                args.top_labels,
                args.chunk_size,
            )
            per_filter_latency_ms[filter_name].append((time.perf_counter() - start) * 1000.0)
    encode_elapsed = time.perf_counter() - encode_start

    all_edges: list[EdgeRow] = []
    for filter_name in filters:
        all_edges.extend(build_edges(memories, labels, vectors, filter_name))
    variants = sorted({e.variant for e in all_edges})
    results = [summarize_variant(memories, all_edges, variant) for variant in variants]
    label_summary = write_label_summary(args.output_dir / "label_graph_db_label_summary.csv", memories, labels, filters)

    write_memory_labels(args.output_dir / "label_graph_db_memory_labels.csv", memories, labels, filters, args.write_top_labels)
    write_edges(args.output_dir / "label_graph_db_edges.csv", all_edges)
    write_clusters(args.output_dir / "label_graph_db_clusters.csv", memories, all_edges, variants)
    write_failures(args.output_dir / "label_graph_db_failure_examples.csv", memories, labels, filters[0])
    memory_aggregate_summary = write_memory_aggregate_labels(
        args.output_dir / "label_graph_db_memory_aggregate_labels.csv",
        memories,
        labels,
        filters,
        args.write_top_labels,
    )

    scenario_count = len({m.scenario for m in memories})
    group_counts = Counter(g for m in memories for g in m.offline_groups)
    latency = {
        name: {
            "mean_ms": float(np.mean(values)) if values else 0.0,
            "p95_ms": float(np.percentile(values, 95)) if values else 0.0,
            "max_ms": float(np.max(values)) if values else 0.0,
        }
        for name, values in per_filter_latency_ms.items()
    }
    summary = {
        "benchmark_only": True,
        "production_behavior_changed": False,
        "runtime_label_inputs": "memory embeddings only; offline labels used for scoring only",
        "surface": args.surface,
        "include_modalities": args.include_modalities,
        "graph_dir": str(args.graph_dir),
        "label_db_dir": str(args.label_db_dir),
        "memory_count": len(memories),
        "scenario_count": scenario_count,
        "scored_memory_count": sum(1 for m in memories if m.offline_groups),
        "offline_group_counts": dict(sorted(group_counts.items())),
        "filters": filters,
        "top_rows": args.top_rows,
        "top_labels": args.top_labels,
        "elapsed_s": encode_elapsed,
        "latency_ms": latency,
        "label_summary": label_summary,
        "memory_aggregate_summary": memory_aggregate_summary,
        "graph_results": results,
        "outputs": [
            "label_graph_db_results.json",
            "label_graph_db_memory_labels.csv",
            "label_graph_db_label_summary.csv",
            "label_graph_db_edges.csv",
            "label_graph_db_clusters.csv",
            "label_graph_db_failure_examples.csv",
            "label_graph_db_memory_aggregate_labels.csv",
        ],
    }
    with (args.output_dir / "label_graph_db_results.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
