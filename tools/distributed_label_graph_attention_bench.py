#!/usr/bin/env python3
"""Benchmark distributed preconsolidated label graph traversal.

This benchmark tests the non-explicit-anchor design: memories connect to Top-K
label nodes, and reference labels such as "he" retrieve co-active labels such as
"steve" through recency-biased graph traversal. There is no entity/anchor object
and no typed label lane.

The graph is intentionally tiny and controlled, but the label vectors and Top-K
retrieval come from the existing label vector store.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Label:
    index: int
    row_id: int
    label_id: str
    label: str
    kind: str
    vector: np.ndarray


@dataclass
class Candidate:
    label: str
    label_index: int
    row_id: int
    score: float
    score01: float
    rank: int


@dataclass
class MemoryNode:
    memory_id: str
    step: int
    seed_labels: tuple[str, ...]
    modality: str = "text"
    observed_labels: tuple[str, ...] = ()
    top_labels: list[Candidate] = field(default_factory=list)


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=True)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def load_label_store(label_db_dir: Path) -> tuple[dict[int, dict], list[int], np.memmap]:
    labels_by_index: dict[int, dict] = {}
    with (label_db_dir / "label_vector_db_labels.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            labels_by_index[int(row["label_index"])] = row

    row_label_indices: list[int] = []
    with (label_db_dir / "label_vector_db_rows.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            row_label_indices.append(int(row["label_index"]))

    vectors = np.memmap(
        label_db_dir / "label_vector_db_256.f32",
        dtype="<f4",
        mode="r",
        shape=(len(row_label_indices), 256),
    )
    return labels_by_index, row_label_indices, vectors


def select_labels(
    labels_by_index: dict[int, dict],
    row_label_indices: list[int],
    vectors: np.memmap,
    names: set[str],
) -> dict[str, Label]:
    selected: dict[str, Label] = {}
    fallback: dict[str, Label] = {}
    for row_id, label_index in enumerate(row_label_indices):
        meta = labels_by_index[label_index]
        name = meta["label"].lower()
        if name not in names:
            continue
        label = Label(
            index=label_index,
            row_id=row_id,
            label_id=meta["label_id"],
            label=name,
            kind=meta["kind"],
            vector=normalize(np.asarray(vectors[row_id], dtype=np.float32)),
        )
        if name not in fallback:
            fallback[name] = label
        # Prefer the compact existing label bank rows when present.
        if meta["source"] == "data/label_bank/labels.jsonl" and name not in selected:
            selected[name] = label

    for name, label in fallback.items():
        selected.setdefault(name, label)

    missing = sorted(names - set(selected))
    if missing:
        raise RuntimeError(f"missing required label-bank labels: {missing}")
    return selected


def centroid_for(seed_labels: tuple[str, ...], selected: dict[str, Label]) -> np.ndarray:
    return normalize(np.mean(np.stack([selected[name].vector for name in seed_labels]), axis=0))


def topk_labels(
    query: np.ndarray,
    labels_by_index: dict[int, dict],
    row_label_indices: list[int],
    vectors: np.memmap,
    top_k: int,
    allowed_labels: set[str],
) -> list[Candidate]:
    scores = vectors @ query
    order = np.argsort(-scores)
    out: list[Candidate] = []
    seen: set[str] = set()
    for row_id in order.tolist():
        label_index = row_label_indices[row_id]
        meta = labels_by_index[label_index]
        name = meta["label"].lower()
        if name not in allowed_labels or name in seen:
            continue
        seen.add(name)
        score = float(scores[row_id])
        out.append(
            Candidate(
                label=name,
                label_index=label_index,
                row_id=row_id,
                score=score,
                score01=max(0.0, min(1.0, (score + 1.0) * 0.5)),
                rank=len(out) + 1,
            )
        )
        if len(out) >= top_k:
            break
    return out


def row_ids_for_labels(
    labels_by_index: dict[int, dict],
    row_label_indices: list[int],
    allowed_labels: set[str],
) -> list[int]:
    row_ids: list[int] = []
    for row_id, label_index in enumerate(row_label_indices):
        meta = labels_by_index.get(label_index, {})
        if meta.get("label", "").lower() in allowed_labels:
            row_ids.append(row_id)
    return row_ids


def label_degrees(labels_by_index: dict[int, dict], row_label_indices: list[int]) -> dict[str, int]:
    degrees: dict[str, int] = defaultdict(int)
    for label_index in row_label_indices:
        meta = labels_by_index.get(label_index, {})
        name = meta.get("label", "").lower()
        if name:
            degrees[name] += 1
    return degrees


def compute_clusters(vectors: np.ndarray, num_clusters: int, seed: int) -> dict:
    np.random.seed(seed)
    n_samples, _dim = vectors.shape
    indices = np.random.choice(n_samples, num_clusters, replace=False)
    centroids = vectors[indices].copy()

    for _ in range(50):
        dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)
        new_centroids = np.zeros_like(centroids)
        for c in range(num_clusters):
            members = vectors[assignments == c]
            if len(members) > 0:
                new_centroids[c] = members.mean(axis=0)
            else:
                new_centroids[c] = vectors[np.random.randint(n_samples)]
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    cluster_members: dict[int, list[int]] = defaultdict(list)
    for row_id, cluster_id in enumerate(assignments):
        cluster_members[int(cluster_id)].append(row_id)
    return {"centroids": centroids.astype(np.float32), "cluster_members": dict(cluster_members)}


def hierarchical_retrieve(
    query_vec: np.ndarray,
    clusters: dict,
    labels_by_index: dict[int, dict],
    row_label_indices: list[int],
    vectors: np.memmap,
    allowed_labels: set[str],
    selected: dict[str, Label],
    degrees: dict[str, int],
    M: int,
    N: int,
    selection_rule: str,
) -> list[Candidate]:
    dists = np.linalg.norm(clusters["centroids"] - query_vec[None, :], axis=1)
    top_cluster_ids = np.argsort(dists)[:M]

    candidates: list[tuple[str, float, str, int]] = []
    for cluster_id in top_cluster_ids:
        member_scores: list[tuple[str, float, str, int]] = []
        for row_id in clusters["cluster_members"].get(int(cluster_id), []):
            if row_id >= len(row_label_indices):
                continue
            label_index = row_label_indices[row_id]
            meta = labels_by_index.get(label_index, {})
            name = meta.get("label", "").lower()
            if name not in allowed_labels:
                continue
            member_scores.append(
                (
                    name,
                    float(np.dot(query_vec, vectors[row_id])),
                    meta.get("kind", ""),
                    degrees.get(name, 0),
                )
            )

        if selection_rule == "cosine":
            member_scores.sort(key=lambda x: -x[1])
        elif selection_rule == "lowest_degree":
            member_scores.sort(key=lambda x: (x[3], -x[1]))
        else:
            kind_order = {
                "name_or_entity": 0,
                "entity_person": 1,
                "entity_location": 2,
                "event": 3,
                "event_action": 4,
                "object": 5,
            }
            member_scores.sort(key=lambda x: (kind_order.get(x[2], 99), -x[1]))
        candidates.extend(member_scores[:N])

    out: list[Candidate] = []
    seen: set[str] = set()
    for name, score, _kind, _degree in candidates:
        if name in seen:
            continue
        seen.add(name)
        label = selected[name]
        out.append(
            Candidate(
                label=name,
                label_index=label.index,
                row_id=label.row_id,
                score=score,
                score01=max(0.0, min(1.0, (score + 1.0) * 0.5)),
                rank=len(out) + 1,
            )
        )
    return out


def build_graph(
    memories: list[MemoryNode],
    recency_decay: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, float]]:
    label_to_memory: dict[str, dict[str, float]] = defaultdict(dict)
    memory_to_label: dict[str, dict[str, float]] = defaultdict(dict)
    max_step = max(memory.step for memory in memories)
    memory_recency: dict[str, float] = {}
    for memory in memories:
        recency = math.exp(-(max_step - memory.step) / recency_decay)
        memory_recency[memory.memory_id] = recency
        for cand in memory.top_labels:
            weight = cand.score01
            label_to_memory[cand.label][memory.memory_id] = max(
                label_to_memory[cand.label].get(memory.memory_id, 0.0), weight
            )
            memory_to_label[memory.memory_id][cand.label] = max(
                memory_to_label[memory.memory_id].get(cand.label, 0.0), weight
            )
    return label_to_memory, memory_to_label, memory_recency


def most_recent_entries(
    query_labels: tuple[str, ...],
    memories: list[MemoryNode],
    label_to_memory: dict[str, dict[str, float]],
    memory_recency: dict[str, float],
) -> dict[str, float]:
    by_id = {memory.memory_id: memory for memory in memories}
    entries: dict[str, float] = {}
    for query in query_labels:
        candidates = label_to_memory.get(query, {})
        if not candidates:
            continue
        memory_id, edge_weight = max(
            candidates.items(),
            key=lambda item: (by_id[item[0]].step, item[1]),
        )
        entries[memory_id] = max(
            entries.get(memory_id, 0.0),
            edge_weight * memory_recency.get(memory_id, 0.0),
        )
    return entries


def attend(
    query_labels: tuple[str, ...],
    memories: list[MemoryNode],
    recency_decay: float,
    temporal_weight: float,
    active_window: int,
    degree_penalty_power: float,
) -> dict:
    label_to_memory, memory_to_label, memory_recency = build_graph(memories, recency_decay)
    by_id = {memory.memory_id: memory for memory in memories}

    # Start the pathway at the most recent memory touched by each query label.
    # This models the "left-to-right" path: a bare recurring label such as "he"
    # enters the newest compatible branch before older compatible branches.
    entries = most_recent_entries(query_labels, memories, label_to_memory, memory_recency)
    memory_scores: dict[str, float] = defaultdict(float)
    for memory_id, score in entries.items():
        memory_scores[memory_id] += score

    # Branch locally through neighboring memories and shared labels. This is
    # label-agnostic: "steve" and "john" are ordinary labels, not entity types.
    entry_ids = set(entries)
    for memory_id, entry_score in entries.items():
        source = by_id[memory_id]
        for other in memories:
            if other.memory_id == memory_id:
                continue
            distance = abs(other.step - source.step)
            if 0 < distance <= active_window:
                memory_scores[other.memory_id] += (
                    entry_score * temporal_weight / float(distance)
                )

        source_labels = set(memory_to_label[memory_id])
        for label in source_labels:
            if label in query_labels:
                continue
            for other_id, edge_weight in label_to_memory.get(label, {}).items():
                if other_id == memory_id:
                    continue
                other = by_id[other_id]
                step_gap = abs(other.step - source.step)
                hop_decay = 1.0 / (1.0 + step_gap)
                memory_scores[other_id] += (
                    entry_score
                    * edge_weight
                    * memory_recency.get(other_id, 0.0)
                    * temporal_weight
                    * 0.25
                    * hop_decay
                )

    # Keep direct, older query matches as weak side branches so content cues can
    # still contribute without overriding the newest compatible branch.
    for query in query_labels:
        for memory_id, weight in label_to_memory.get(query, {}).items():
            if memory_id in entry_ids:
                continue
            memory_scores[memory_id] += (
                weight * memory_recency.get(memory_id, 0.0) * temporal_weight * 0.25
            )

    label_scores: dict[str, float] = defaultdict(float)
    for memory_id, memory_score in memory_scores.items():
        for label, weight in memory_to_label[memory_id].items():
            if label in query_labels:
                continue
            degree = max(1, len(label_to_memory.get(label, {})))
            degree_penalty = math.pow(float(degree), degree_penalty_power)
            label_scores[label] += memory_score * weight / degree_penalty

    ranked_memories = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    top = ranked_labels[0] if ranked_labels else ("", 0.0)
    second = ranked_labels[1] if len(ranked_labels) > 1 else ("", 0.0)
    return {
        "query": "|".join(query_labels),
        "entry_memories": sorted(entries.items(), key=lambda x: x[1], reverse=True),
        "top_label": top[0],
        "top_score": top[1],
        "second_label": second[0],
        "second_score": second[1],
        "margin": top[1] - second[1],
        "ranked_labels": ranked_labels[:20],
        "ranked_labels_full": ranked_labels,
        "ranked_memories": ranked_memories[:10],
        "ranked_memories_full": ranked_memories,
    }


def run_scenario(
    name: str,
    all_memories: list[MemoryNode],
    probes: list[tuple[str, tuple[str, ...], str, str, int]],
    recency_decay: float,
    temporal_weight: float,
    active_window: int,
    degree_penalty_power: float,
) -> dict:
    rows = []
    hits = 0
    for probe_name, query_labels, expected_label, expected_memory, memory_count in probes:
        memories = all_memories[:memory_count]
        result = attend(
            query_labels,
            memories,
            recency_decay,
            temporal_weight,
            active_window,
            degree_penalty_power,
        )
        readout_labels = [label for label, _score in result["ranked_labels_full"]]
        try:
            expected_rank = readout_labels.index(expected_label) + 1
        except ValueError:
            expected_rank = 0
        ranked_memory_ids = [memory_id for memory_id, _score in result["ranked_memories_full"]]
        try:
            expected_memory_rank = ranked_memory_ids.index(expected_memory) + 1
        except ValueError:
            expected_memory_rank = 0
        # The realtime graph only needs the right label in the active branch;
        # consolidation will inspect the blob and prune false Top-K labels.
        hit = 0 < expected_rank <= 5 and 0 < expected_memory_rank <= 2
        hits += int(hit)
        rows.append(
            {
                "probe": probe_name,
                "query": result["query"],
                "expected_label": expected_label,
                "expected_memory": expected_memory,
                "top_label": result["top_label"],
                "expected_label_rank": expected_rank,
                "expected_memory_rank": expected_memory_rank,
                "hit": hit,
                "margin": result["margin"],
                "entry_memories": result["entry_memories"],
                "ranked_labels": result["ranked_labels"],
                "ranked_memories": result["ranked_memories"],
            }
        )
    return {
        "scenario": name,
        "probe_count": len(probes),
        "hit_count": hits,
        "hit_rate": hits / len(probes) if probes else 0.0,
        "probes": rows,
    }


def write_memory_labels(path: Path, memories: list[MemoryNode]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["memory_id", "step", "modality", "seed_labels", "rank", "label", "score", "score01"])
        for memory in memories:
            for cand in memory.top_labels:
                writer.writerow(
                    [
                        memory.memory_id,
                        memory.step,
                        memory.modality,
                        "|".join(memory.seed_labels),
                        cand.rank,
                        cand.label,
                        f"{cand.score:.9f}",
                        f"{cand.score01:.9f}",
                    ]
                )


def base_memories() -> list[MemoryNode]:
    return [
        MemoryNode(
            "m1_steve_intro_image",
            1,
            ("steve", "friend", "person", "he", "image", "face"),
            "image",
            ("steve", "friend", "person", "he", "image", "face"),
        ),
        MemoryNode("m2_he_likes_pizza", 2, ("he", "pizza", "eat"), "text", ("he", "pizza", "eat")),
        MemoryNode(
            "m3_we_pizza_place",
            3,
            ("we", "he", "favorite", "pizza", "place", "restaurant"),
            "text",
            ("we", "he", "favorite", "pizza", "place", "restaurant"),
        ),
        MemoryNode(
            "m4_john_intro_voice",
            4,
            ("john", "person", "he", "voice"),
            "audio",
            ("john", "person", "he", "voice"),
        ),
        MemoryNode(
            "m5_he_likes_restaurant",
            5,
            ("he", "restaurant", "favorite"),
            "text",
            ("he", "restaurant", "favorite"),
        ),
    ]


def base_scenarios(
    all_memories: list[MemoryNode],
    recency_decay: float,
    temporal_weight: float,
    active_window: int,
    degree_penalty_power: float,
) -> list[dict]:
    return [
        run_scenario(
            "steve_only_reference",
            all_memories,
            [
                ("he_after_steve_intro", ("he",), "steve", "m1_steve_intro_image", 1),
                ("he_pizza_after_preference", ("he", "pizza"), "steve", "m1_steve_intro_image", 2),
                ("we_place_after_steve_context", ("we", "place"), "steve", "m2_he_likes_pizza", 3),
            ],
            recency_decay,
            temporal_weight,
            active_window,
            degree_penalty_power,
        ),
        run_scenario(
            "competing_male_reference",
            all_memories,
            [
                ("he_after_john_intro", ("he",), "john", "m4_john_intro_voice", 4),
                ("he_pizza_after_john_intro", ("he", "pizza"), "john", "m4_john_intro_voice", 4),
                ("he_restaurant_after_john_followup", ("he", "restaurant"), "john", "m4_john_intro_voice", 5),
            ],
            recency_decay,
            temporal_weight,
            active_window,
            degree_penalty_power,
        ),
    ]


def scaled_multimodal_memories() -> list[MemoryNode]:
    return [
        MemoryNode("m01_steve_image_intro", 1, ("steve", "friend", "person", "he", "image", "face"), "image", ("steve", "friend", "person", "he", "image", "face")),
        MemoryNode("m02_he_pizza_text", 2, ("he", "pizza", "eat"), "text", ("he", "pizza", "eat")),
        MemoryNode("m03_steve_restaurant_video", 3, ("we", "he", "favorite", "pizza", "restaurant", "video"), "video", ("we", "he", "favorite", "pizza", "restaurant", "video")),
        MemoryNode("m04_maria_voice_intro", 4, ("maria", "friend", "person", "she", "voice", "audio"), "audio", ("maria", "friend", "person", "she", "voice", "audio")),
        MemoryNode("m05_she_guitar_audio", 5, ("she", "guitar", "music", "song", "audio"), "audio", ("she", "guitar", "music", "song", "audio")),
        MemoryNode("m06_maria_concert_image", 6, ("her", "guitar", "music", "image", "photo"), "image", ("her", "guitar", "music", "image", "photo")),
        MemoryNode("m07_david_video_intro", 7, ("david", "brother", "person", "he", "video", "face"), "video", ("david", "brother", "person", "he", "video", "face")),
        MemoryNode("m08_he_car_text", 8, ("he", "car", "favorite"), "text", ("he", "car", "favorite")),
        MemoryNode("m09_david_car_image", 9, ("he", "car", "image", "photo"), "image", ("he", "car", "image", "photo")),
        MemoryNode("m10_emma_image_intro", 10, ("emma", "sister", "person", "she", "image", "face"), "image", ("emma", "sister", "person", "she", "image", "face")),
        MemoryNode("m11_she_dog_video", 11, ("she", "dog", "video", "park"), "video", ("she", "dog", "video", "park")),
        MemoryNode("m12_emma_dog_audio", 12, ("her", "dog", "voice", "audio"), "audio", ("her", "dog", "voice", "audio")),
        MemoryNode("m13_john_voice_intro", 13, ("john", "person", "he", "voice", "audio"), "audio", ("john", "person", "he", "voice", "audio")),
        MemoryNode("m14_he_coffee_text", 14, ("he", "coffee", "office", "work"), "text", ("he", "coffee", "office", "work")),
        MemoryNode("m15_john_office_image", 15, ("he", "office", "laptop", "image"), "image", ("he", "office", "laptop", "image")),
        MemoryNode("m16_sarah_video_intro", 16, ("sarah", "teacher", "person", "she", "video", "face"), "video", ("sarah", "teacher", "person", "she", "video", "face")),
        MemoryNode("m17_she_book_text", 17, ("she", "book", "school", "favorite"), "text", ("she", "book", "school", "favorite")),
        MemoryNode("m18_sarah_school_audio", 18, ("her", "school", "voice", "audio"), "audio", ("her", "school", "voice", "audio")),
    ]


def scaled_multimodal_scenarios(
    all_memories: list[MemoryNode],
    recency_decay: float,
    temporal_weight: float,
    active_window: int,
    degree_penalty_power: float,
) -> list[dict]:
    return [
        run_scenario(
            "scaled_male_recency",
            all_memories,
            [
                ("steve_after_image_intro", ("he",), "steve", "m01_steve_image_intro", 1),
                ("steve_text_followup", ("he", "pizza"), "steve", "m02_he_pizza_text", 2),
                ("david_after_video_intro", ("he",), "david", "m07_david_video_intro", 7),
                ("david_image_car_followup", ("he", "car"), "david", "m09_david_car_image", 9),
                ("john_after_voice_intro", ("he",), "john", "m13_john_voice_intro", 13),
                ("john_image_office_followup", ("he", "office"), "john", "m15_john_office_image", 15),
            ],
            recency_decay,
            temporal_weight,
            active_window,
            degree_penalty_power,
        ),
        run_scenario(
            "scaled_female_recency",
            all_memories,
            [
                ("maria_after_voice_intro", ("she",), "maria", "m04_maria_voice_intro", 4),
                ("maria_audio_music_followup", ("she", "guitar"), "maria", "m05_she_guitar_audio", 5),
                ("emma_after_image_intro", ("she",), "emma", "m10_emma_image_intro", 10),
                ("emma_audio_dog_followup", ("her", "dog"), "emma", "m12_emma_dog_audio", 12),
                ("sarah_after_video_intro", ("she",), "sarah", "m16_sarah_video_intro", 16),
                ("sarah_audio_school_followup", ("her", "school"), "sarah", "m18_sarah_school_audio", 18),
            ],
            recency_decay,
            temporal_weight,
            active_window,
            degree_penalty_power,
        ),
        run_scenario(
            "scaled_cross_modal_objects",
            all_memories,
            [
                ("pizza_video_to_steve", ("pizza", "video"), "steve", "m03_steve_restaurant_video", 3),
                ("guitar_image_to_maria", ("guitar", "image"), "maria", "m06_maria_concert_image", 6),
                ("car_image_to_david", ("car", "image"), "david", "m09_david_car_image", 9),
                ("dog_audio_to_emma", ("dog", "audio"), "emma", "m12_emma_dog_audio", 12),
                ("office_image_to_john", ("office", "image"), "john", "m15_john_office_image", 15),
                ("school_audio_to_sarah", ("school", "audio"), "sarah", "m18_sarah_school_audio", 18),
            ],
            recency_decay,
            temporal_weight,
            active_window,
            degree_penalty_power,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-db-dir", type=Path, default=Path("build/label_vector_db_full_2proto"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/distributed_label_graph_attention_bench"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--label-assignment",
        choices=("topk", "hierarchical", "topk_plus_hierarchical"),
        default="topk",
        help="How to attach provisional labels to memory nodes.",
    )
    parser.add_argument("--hierarchical-k", type=int, default=50)
    parser.add_argument("--hierarchical-M", type=int, default=3)
    parser.add_argument("--hierarchical-N", type=int, default=10)
    parser.add_argument(
        "--hierarchical-rule",
        choices=("kind_priority", "lowest_degree", "cosine"),
        default="cosine",
    )
    parser.add_argument("--hierarchical-seed", type=int, default=42)
    parser.add_argument("--recency-decay", type=float, default=3.0)
    parser.add_argument("--temporal-weight", type=float, default=0.35)
    parser.add_argument("--active-window", type=int, default=1)
    parser.add_argument(
        "--degree-penalty-power",
        type=float,
        default=0.0,
        help="Divide label readout by label degree^power to dampen globally promiscuous labels.",
    )
    parser.add_argument(
        "--suite",
        choices=("base", "scaled_multimodal"),
        default="base",
        help="Benchmark scenario suite to run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_by_index, row_label_indices, vectors = load_label_store(args.label_db_dir)
    label_names = {
        "alex",
        "audio",
        "book",
        "brother",
        "car",
        "coffee",
        "david",
        "dog",
        "eat",
        "emma",
        "face",
        "favorite",
        "friend",
        "guitar",
        "he",
        "her",
        "image",
        "john",
        "laptop",
        "male",
        "man",
        "maria",
        "music",
        "office",
        "park",
        "person",
        "photo",
        "pizza",
        "place",
        "restaurant",
        "sarah",
        "school",
        "she",
        "sister",
        "song",
        "steve",
        "teacher",
        "video",
        "voice",
        "we",
        "work",
    }
    selected = select_labels(labels_by_index, row_label_indices, vectors, label_names)

    allowed_labels = label_names
    clusters = None
    degrees: dict[str, int] = {}
    if args.label_assignment in {"hierarchical", "topk_plus_hierarchical"}:
        clustered_rows = row_ids_for_labels(labels_by_index, row_label_indices, allowed_labels)
        effective_k = min(args.hierarchical_k, len(clustered_rows))
        cluster_vectors = vectors[clustered_rows]
        clusters = compute_clusters(cluster_vectors, effective_k, args.hierarchical_seed)
        clusters["cluster_members"] = {
            cluster_id: [int(clustered_rows[i]) for i in members]
            for cluster_id, members in clusters["cluster_members"].items()
        }
        degrees = label_degrees(labels_by_index, row_label_indices)
    if args.suite == "scaled_multimodal":
        all_memories = scaled_multimodal_memories()
    else:
        all_memories = base_memories()

    start = time.perf_counter()
    for memory in all_memories:
        centroid = centroid_for(memory.seed_labels, selected)
        if args.label_assignment in {"hierarchical", "topk_plus_hierarchical"}:
            assert clusters is not None
            hierarchical_labels = hierarchical_retrieve(
                centroid,
                clusters,
                labels_by_index,
                row_label_indices,
                vectors,
                allowed_labels,
                selected,
                degrees,
                args.hierarchical_M,
                args.hierarchical_N,
                args.hierarchical_rule,
            )
            if args.label_assignment == "topk_plus_hierarchical":
                topk = topk_labels(
                    centroid,
                    labels_by_index,
                    row_label_indices,
                    vectors,
                    args.top_k,
                    allowed_labels,
                )
                merged: list[Candidate] = []
                seen: set[str] = set()
                for cand in topk + hierarchical_labels:
                    if cand.label in seen:
                        continue
                    seen.add(cand.label)
                    cand.rank = len(merged) + 1
                    merged.append(cand)
                memory.top_labels = merged
            else:
                memory.top_labels = hierarchical_labels
        else:
            memory.top_labels = topk_labels(
                centroid,
                labels_by_index,
                row_label_indices,
                vectors,
                args.top_k,
                allowed_labels,
            )
    elapsed_s = time.perf_counter() - start

    if args.suite == "scaled_multimodal":
        scenarios = scaled_multimodal_scenarios(
            all_memories,
            args.recency_decay,
            args.temporal_weight,
            args.active_window,
            args.degree_penalty_power,
        )
    else:
        scenarios = base_scenarios(
            all_memories,
            args.recency_decay,
            args.temporal_weight,
            args.active_window,
            args.degree_penalty_power,
        )

    topk_contains_seed = 0
    topk_seed_checks = 0
    for memory in all_memories:
        labels = {candidate.label for candidate in memory.top_labels}
        for seed in memory.seed_labels:
            topk_seed_checks += 1
            topk_contains_seed += int(seed in labels)

    result = {
        "benchmark_only": True,
        "production_behavior_changed": False,
        "design": "distributed_label_memory_graph_no_explicit_anchor",
        "suite": args.suite,
        "label_db_dir": str(args.label_db_dir),
        "label_assignment": args.label_assignment,
        "top_k": args.top_k,
        "hierarchical": {
            "k": args.hierarchical_k,
            "M": args.hierarchical_M,
            "N": args.hierarchical_N,
            "rule": args.hierarchical_rule,
            "seed": args.hierarchical_seed,
        },
        "recency_decay": args.recency_decay,
        "temporal_weight": args.temporal_weight,
        "active_window": args.active_window,
        "degree_penalty_power": args.degree_penalty_power,
        "label_assignment_elapsed_s": elapsed_s,
        "memory_count": len(all_memories),
        "seed_label_topk_recall": topk_contains_seed / topk_seed_checks if topk_seed_checks else 0.0,
        "scenarios": scenarios,
        "overall_probe_count": sum(s["probe_count"] for s in scenarios),
        "overall_hit_count": sum(s["hit_count"] for s in scenarios),
        "overall_hit_rate": (
            sum(s["hit_count"] for s in scenarios)
            / sum(s["probe_count"] for s in scenarios)
        ),
        "interpretation": (
            "Reference binding emerges from label-memory-label spreading activation. "
            "There is no explicit Steve anchor and no typed name channel; every "
            "label is retrieved by recency-biased branch entry and graph "
            "connectivity."
        ),
        "outputs": [
            "distributed_label_graph_attention_results.json",
            "distributed_label_graph_memory_labels.csv",
        ],
    }

    (args.output_dir / "distributed_label_graph_attention_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_memory_labels(args.output_dir / "distributed_label_graph_memory_labels.csv", all_memories)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
