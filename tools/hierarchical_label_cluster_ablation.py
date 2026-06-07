#!/usr/bin/env python3
"""
Hierarchical Label Cluster Ablation Study

This script performs controlled, reproducible ablations on the offline-clustered
label retrieval approach for cross-modal reference resolution.

Pipeline under test:
    signal embedding (or accumulator state)
        → find Top-M nearest clusters (offline k-means)
        → within each cluster, select Top-N labels using a selection rule
        → feed selected labels into per-signal LabelAccumulator (optional)
        → readout for pronoun probes

All design choices are first-class experimental variables and are logged
exhaustively for reproducibility.

Design choices being ablated:
  - num_clusters (k for offline clustering)
  - clustering_method
  - num_clusters_retrieve (M)
  - labels_per_cluster (N)
  - cluster_label_selection_rule
  - query_vector_source (what we use to find best clusters)
  - use_label_accumulator_downstream (whether selected labels still go through decay/dedup)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# --- Minimal LabelAccumulator copy for self-containment ---
class LabelAccumulator:
    def __init__(self, decay=0.82, max_labels=35, generic_suppression=0.22, recency_boost=1.4):
        self.scores = defaultdict(float)
        self.last_seen_step = {}
        self.step_count = 0
        self.decay = decay
        self.max_labels = max_labels
        self.generic_suppression = generic_suppression
        self.recency_boost = recency_boost
        self.GENERIC_SUPPRESS = {"person", "friend", "man", "male", "brother", "sister", "teacher",
                                 "audio", "image", "video", "voice", "face", "photo", "we"}

    def step(self, embedding, labels_by_index, row_label_indices, vectors, allowed, top_k=8):
        self.step_count += 1
        cands = topk_labels(embedding, labels_by_index, row_label_indices, vectors, top_k, allowed)
        for lab in list(self.scores.keys()):
            self.scores[lab] *= self.decay
        for cand in cands:
            lab = cand["label"]
            sc = cand["score01"]
            if lab in self.GENERIC_SUPPRESS:
                sc *= self.generic_suppression
            if lab in self.last_seen_step and self.step_count - self.last_seen_step[lab] <= 2:
                sc *= self.recency_boost
            self.scores[lab] += sc
            self.last_seen_step[lab] = self.step_count
        if len(self.scores) > self.max_labels:
            sorted_items = sorted(self.scores.items(), key=lambda x: -x[1])[:self.max_labels]
            self.scores = defaultdict(float, dict(sorted_items))

    def get_top_labels(self, n=12):
        if not self.scores:
            return []
        return [lab for lab, sc in sorted(self.scores.items(), key=lambda x: -x[1])[:n] if sc > 0.02]

    def add_labels(self, labels, score=0.9):
        self.step_count += 1
        for lab in list(self.scores.keys()):
            self.scores[lab] *= self.decay
        for lab in labels:
            sc = score
            if lab in self.GENERIC_SUPPRESS:
                sc *= self.generic_suppression
            if lab in self.last_seen_step and self.step_count - self.last_seen_step[lab] <= 2:
                sc *= self.recency_boost
            self.scores[lab] += sc
            self.last_seen_step[lab] = self.step_count
        if len(self.scores) > self.max_labels:
            sorted_items = sorted(self.scores.items(), key=lambda x: -x[1])[:self.max_labels]
            self.scores = defaultdict(float, dict(sorted_items))


class LiveAccumulator:
    def __init__(self, win_size=12, alpha_c=0.03, drift_step=0.10):
        self.mu_acc = None
        self.c_t = None
        self.recent_window = deque(maxlen=win_size)
        self.n_signals = 0
        self.alpha_c = alpha_c

    def step(self, embedding, drift=None, modality=None):
        emb = normalize(embedding)
        if self.n_signals == 0:
            self.mu_acc = emb.copy()
            self.c_t = emb.copy()
        else:
            self.mu_acc = self.mu_acc + (emb - self.mu_acc) / (self.n_signals + 1)
            self.c_t = (1 - self.alpha_c) * self.c_t + self.alpha_c * self.mu_acc
            self.c_t = normalize(self.c_t)
        self.n_signals += 1
        self.recent_window.append(self.mu_acc.copy())

# ---------------------------------------------------------------------------
# Re-use helpers from the existing benchmark (keeps everything comparable)
# ---------------------------------------------------------------------------
# We import specific functions to keep the script self-contained in spirit
# while avoiding duplication. In production this could be a small shared lib.

# For this script we will duplicate the minimal necessary pieces so it is
# a standalone, reproducible research tool. The logic is identical to
# accumulator_label_context_bench.py.

# --- Minimal re-implementations (identical to the other bench) ---

def normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=True)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec

def load_label_store(label_db_dir: Path):
    labels_by_index = {}
    with (label_db_dir / "label_vector_db_labels.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            labels_by_index[int(row["label_index"])] = row

    row_label_indices = []
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

def select_labels(labels_by_index, row_label_indices, vectors, names):
    selected = {}
    fallback = {}
    for row_id, label_index in enumerate(row_label_indices):
        meta = labels_by_index[label_index]
        name = meta["label"].lower()
        if name not in names:
            continue
        label = {
            "index": label_index,
            "row_id": row_id,
            "label": name,
            "kind": meta["kind"],
            "vector": normalize(np.asarray(vectors[row_id], dtype=np.float32)),
        }
        if name not in fallback:
            fallback[name] = label
        if meta["source"] == "data/label_bank/labels.jsonl" and name not in selected:
            selected[name] = label
    for name, lab in fallback.items():
        selected.setdefault(name, lab)
    return selected

def topk_labels(query, labels_by_index, row_label_indices, vectors, top_k, allowed):
    scores = vectors @ query
    order = np.argsort(-scores)
    out = []
    seen = set()
    for row_id in order.tolist():
        label_index = row_label_indices[row_id]
        meta = labels_by_index[label_index]
        name = meta["label"].lower()
        if name not in allowed or name in seen:
            continue
        seen.add(name)
        out.append({
            "label": name,
            "label_index": label_index,
            "score": float(scores[row_id]),
            "score01": max(0.0, min(1.0, (float(scores[row_id]) + 1.0) * 0.5)),
        })
        if len(out) >= top_k:
            break
    return out


def centroid_for(seed_labels, selected):
    return normalize(np.mean(np.stack([selected[name]["vector"] for name in seed_labels]), axis=0))


def row_ids_for_labels(labels_by_index, row_label_indices, allowed):
    row_ids = []
    for row_id, label_index in enumerate(row_label_indices):
        meta = labels_by_index.get(label_index, {})
        name = meta.get("label", "").lower()
        if name in allowed:
            row_ids.append(row_id)
    return row_ids


def label_degrees(labels_by_index, row_label_indices):
    degrees = defaultdict(int)
    for label_index in row_label_indices:
        meta = labels_by_index.get(label_index, {})
        name = meta.get("label", "").lower()
        if name:
            degrees[name] += 1
    return degrees

# --- Scaled multimodal data (identical to other benches) ---

def scaled_multimodal_signals():
    return [
        {"id": "m01", "step": 1, "seed_labels": ("steve", "friend", "person", "he", "image", "face"), "modality": "image"},
        {"id": "m02", "step": 2, "seed_labels": ("he", "pizza", "eat"), "modality": "text"},
        {"id": "m03", "step": 3, "seed_labels": ("we", "he", "favorite", "pizza", "restaurant", "video"), "modality": "video"},
        {"id": "m04", "step": 4, "seed_labels": ("maria", "friend", "person", "she", "voice", "audio"), "modality": "audio"},
        {"id": "m05", "step": 5, "seed_labels": ("she", "guitar", "music", "song", "audio"), "modality": "audio"},
        {"id": "m06", "step": 6, "seed_labels": ("her", "guitar", "music", "image", "photo"), "modality": "image"},
        {"id": "m07", "step": 7, "seed_labels": ("david", "brother", "person", "he", "video", "face"), "modality": "video"},
        {"id": "m08", "step": 8, "seed_labels": ("he", "car", "favorite"), "modality": "text"},
        {"id": "m09", "step": 9, "seed_labels": ("he", "car", "image", "photo"), "modality": "image"},
        {"id": "m10", "step": 10, "seed_labels": ("emma", "sister", "person", "she", "image", "face"), "modality": "image"},
        {"id": "m11", "step": 11, "seed_labels": ("she", "dog", "video", "park"), "modality": "video"},
        {"id": "m12", "step": 12, "seed_labels": ("her", "dog", "voice", "audio"), "modality": "audio"},
        {"id": "m13", "step": 13, "seed_labels": ("john", "person", "he", "voice", "audio"), "modality": "audio"},
        {"id": "m14", "step": 14, "seed_labels": ("he", "coffee", "office", "work"), "modality": "text"},
        {"id": "m15", "step": 15, "seed_labels": ("he", "office", "laptop", "image"), "modality": "image"},
        {"id": "m16", "step": 16, "seed_labels": ("sarah", "teacher", "person", "she", "video", "face"), "modality": "video"},
        {"id": "m17", "step": 17, "seed_labels": ("she", "book", "school", "favorite"), "modality": "text"},
        {"id": "m18", "step": 18, "seed_labels": ("her", "school", "voice", "audio"), "modality": "audio"},
    ]

def scaled_multimodal_probes():
    return [
        {"name": "steve_after_image_intro", "query": ("he",), "expected": "steve", "prefix": 1},
        {"name": "steve_text_followup", "query": ("he", "pizza"), "expected": "steve", "prefix": 2},
        {"name": "david_after_video_intro", "query": ("he",), "expected": "david", "prefix": 7},
        {"name": "david_image_car_followup", "query": ("he", "car"), "expected": "david", "prefix": 9},
        {"name": "john_after_voice_intro", "query": ("he",), "expected": "john", "prefix": 13},
        {"name": "john_image_office_followup", "query": ("he", "office"), "expected": "john", "prefix": 15},
        {"name": "maria_after_voice_intro", "query": ("she",), "expected": "maria", "prefix": 4},
        {"name": "maria_audio_music_followup", "query": ("she", "guitar"), "expected": "maria", "prefix": 5},
        {"name": "emma_after_image_intro", "query": ("she",), "expected": "emma", "prefix": 10},
        {"name": "emma_audio_dog_followup", "query": ("her", "dog"), "expected": "emma", "prefix": 12},
        {"name": "sarah_after_video_intro", "query": ("she",), "expected": "sarah", "prefix": 16},
        {"name": "sarah_audio_school_followup", "query": ("her", "school"), "expected": "sarah", "prefix": 18},
        {"name": "pizza_video_to_steve", "query": ("pizza", "video"), "expected": "steve", "prefix": 3},
        {"name": "guitar_image_to_maria", "query": ("guitar", "image"), "expected": "maria", "prefix": 6},
        {"name": "car_image_to_david", "query": ("car", "image"), "expected": "david", "prefix": 9},
        {"name": "dog_audio_to_emma", "query": ("dog", "audio"), "expected": "emma", "prefix": 12},
        {"name": "office_image_to_john", "query": ("office", "image"), "expected": "john", "prefix": 15},
        {"name": "school_audio_to_sarah", "query": ("school", "audio"), "expected": "sarah", "prefix": 18},
    ]

# ---------------------------------------------------------------------------
# Reproducibility utilities
# ---------------------------------------------------------------------------

def get_git_info() -> dict[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip() != ""
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": True}

def make_run_id(params: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    key_parts = [
        f"k{params.get('num_clusters', 'NA')}",
        f"M{params.get('num_clusters_retrieve', 'NA')}",
        f"N{params.get('labels_per_cluster', 'NA')}",
        params.get('selection_rule', 'NA'),
        params.get('query_source', 'NA'),
    ]
    return f"hierarchical_{'_'.join(key_parts)}_{ts}"

# ---------------------------------------------------------------------------
# Offline Clustering
# ---------------------------------------------------------------------------

def compute_clusters(vectors: np.ndarray, num_clusters: int, seed: int = 42) -> dict:
    """Simple reproducible k-means using numpy (Lloyd's algorithm)."""
    np.random.seed(seed)
    n_samples, dim = vectors.shape

    # Initialize centroids by picking random points
    indices = np.random.choice(n_samples, num_clusters, replace=False)
    centroids = vectors[indices].copy()

    for _ in range(50):  # max iterations
        # Assign
        dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        # Update
        new_centroids = np.zeros_like(centroids)
        for c in range(num_clusters):
            members = vectors[labels == c]
            if len(members) > 0:
                new_centroids[c] = members.mean(axis=0)
            else:
                new_centroids[c] = vectors[np.random.randint(n_samples)]

        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    clusters = defaultdict(list)
    for idx, c in enumerate(labels):
        clusters[int(c)].append(idx)

    return {
        "num_clusters": num_clusters,
        "centroids": centroids.astype(np.float32),
        "cluster_members": {k: v for k, v in clusters.items()},
        "seed": seed,
    }

def save_clusters(path: Path, clusters: dict):
    np.savez(path, centroids=clusters["centroids"], 
             cluster_members=np.array(list(clusters["cluster_members"].items()), dtype=object))
    meta = {k: v for k, v in clusters.items() if k != "centroids"}
    (path.with_suffix(".json")).write_text(json.dumps(meta, indent=2))

def load_clusters(path: Path):
    data = np.load(path, allow_pickle=True)
    meta = json.loads((path.with_suffix(".json")).read_text())
    meta["centroids"] = data["centroids"]
    meta["cluster_members"] = dict(data["cluster_members"])
    return meta

# ---------------------------------------------------------------------------
# Hierarchical Retrieval
# ---------------------------------------------------------------------------

def hierarchical_retrieve(
    query_vec: np.ndarray,
    clusters: dict,
    labels_by_index: dict,
    row_label_indices: list[int],
    vectors: np.memmap,
    allowed: set[str],
    M: int,
    N: int,
    selection_rule: str,
    degrees: dict[str, int] | None = None,
) -> list[str]:
    """
    1. Find Top-M nearest clusters to query_vec
    2. Inside each selected cluster, pick the best N labels according to selection_rule
    """
    centroids = clusters["centroids"]
    dists = np.linalg.norm(centroids - query_vec[None, :], axis=1)
    top_cluster_ids = np.argsort(dists)[:M]

    candidates = []
    for cid in top_cluster_ids:
        member_row_ids = clusters["cluster_members"].get(int(cid), [])
        if not member_row_ids:
            continue

        member_scores = []
        for row_id in member_row_ids:
            if row_id >= len(row_label_indices):
                continue
            label_index = row_label_indices[row_id]
            meta = labels_by_index.get(label_index, {})
            name = meta.get("label", "").lower()
            if not name or name not in allowed:
                continue

            vec = vectors[row_id]
            cosine = float(np.dot(query_vec, vec))

            kind = meta.get("kind", "")
            degree = degrees.get(name, 0) if degrees is not None else 0

            member_scores.append((name, cosine, kind, degree))

        # Selection inside cluster
        if selection_rule == "cosine":
            member_scores.sort(key=lambda x: -x[1])
        elif selection_rule == "kind_priority":
            kind_order = {"name_or_entity": 0, "entity_person": 1, "entity_location": 2,
                          "event": 3, "event_action": 4, "object": 5}
            member_scores.sort(key=lambda x: (kind_order.get(x[2], 99), -x[1]))
        elif selection_rule == "lowest_degree":
            member_scores.sort(key=lambda x: (x[3], -x[1]))
        else:
            member_scores.sort(key=lambda x: -x[1])

        for name, _, _, _ in member_scores[:N]:
            candidates.append(name)

    # Deduplicate while preserving order
    seen = set()
    final = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            final.append(c)
    return final

# ---------------------------------------------------------------------------
# Main Ablation Runner (skeleton - will be expanded)
# ---------------------------------------------------------------------------

def run_single_hierarchical_experiment(
    label_db_dir: Path,
    output_dir: Path,
    num_clusters: int,
    M: int,
    N: int,
    selection_rule: str,
    query_source: str,
    use_label_accumulator: bool = True,
    seed: int = 42,
    cluster_scope: str = "allowed_labels",
) -> dict[str, Any]:
    """
    Runs one full hierarchical configuration on the scaled_multimodal suite.
    This is the core experiment the ablation harness calls.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    labels_by_index, row_label_indices, vectors = load_label_store(label_db_dir)

    label_names = {
        "alex", "audio", "book", "brother", "car", "coffee", "david", "dog", "eat",
        "emma", "face", "favorite", "friend", "guitar", "he", "her", "image", "john",
        "laptop", "male", "man", "maria", "music", "office", "park", "person", "photo",
        "pizza", "place", "restaurant", "sarah", "school", "she", "sister", "song",
        "steve", "teacher", "video", "voice", "we", "work",
    }
    selected = select_labels(labels_by_index, row_label_indices, vectors, label_names)
    allowed = label_names
    degrees = label_degrees(labels_by_index, row_label_indices)

    signals = scaled_multimodal_signals()
    probes = scaled_multimodal_probes()

    # --- Offline clustering (cached) ---
    # The original version sampled 5k rows from the full vector DB. That can
    # drop rare names entirely, making target-identity retrieval impossible.
    if cluster_scope == "allowed_labels":
        cluster_row_ids = row_ids_for_labels(labels_by_index, row_label_indices, allowed)
    elif cluster_scope == "all_rows":
        cluster_row_ids = list(range(len(row_label_indices)))
    else:
        raise ValueError(f"unknown cluster_scope: {cluster_scope}")

    if not cluster_row_ids:
        raise RuntimeError("no rows available for clustering")

    effective_num_clusters = min(num_clusters, len(cluster_row_ids))
    cluster_cache = output_dir / f"clusters_{cluster_scope}_k{effective_num_clusters}_seed{seed}.npz"
    if cluster_cache.exists():
        clusters = load_clusters(cluster_cache)
    else:
        cluster_vectors = vectors[cluster_row_ids]
        clusters = compute_clusters(cluster_vectors, effective_num_clusters, seed)
        # Map cluster members back to original row ids
        clusters["cluster_members"] = {
            k: [int(cluster_row_ids[i]) for i in v] for k, v in clusters["cluster_members"].items()
        }
        save_clusters(cluster_cache, clusters)

    # --- Run experiment ---
    results = []
    embedding_acc = None  # for mu_acc query source

    for probe in probes:
        prefix = signals[:probe["prefix"]]

        baseline_acc = LabelAccumulator(decay=0.82, max_labels=35, generic_suppression=0.22)
        hierarchical_acc = LabelAccumulator(decay=0.82, max_labels=35, generic_suppression=0.22)
        embedding_acc = LiveAccumulator(win_size=12, alpha_c=0.03, drift_step=0.10) if query_source == "mu_acc" else None
        retrieval_trace = []
        last_retrieved_labels = []

        for sig in prefix:
            emb = centroid_for(sig["seed_labels"], selected)
            baseline_acc.step(emb, labels_by_index, row_label_indices, vectors, allowed, top_k=8)

            if embedding_acc:
                embedding_acc.step(emb)

            # Choose query vector for this signal's hierarchical retrieval.
            if query_source == "per_signal":
                query_vec = emb
            elif query_source == "mu_acc" and embedding_acc and embedding_acc.mu_acc is not None:
                query_vec = embedding_acc.mu_acc
            else:
                query_vec = emb

            last_retrieved_labels = hierarchical_retrieve(
                query_vec, clusters, labels_by_index, row_label_indices, vectors,
                allowed, M, N, selection_rule, degrees
            )
            retrieval_trace.append({
                "signal": sig["id"],
                "retrieved_labels": last_retrieved_labels,
            })
            if use_label_accumulator:
                hierarchical_acc.add_labels(last_retrieved_labels)

        baseline_live_labels = baseline_acc.get_top_labels(12)
        hierarchical_live_labels = (
            hierarchical_acc.get_top_labels(12) if use_label_accumulator else last_retrieved_labels
        )
        live_labels = hierarchical_live_labels

        expected = probe["expected"]
        try:
            baseline_rank = baseline_live_labels.index(expected) + 1
        except ValueError:
            baseline_rank = 0

        try:
            hierarchical_rank = hierarchical_live_labels.index(expected) + 1
        except ValueError:
            hierarchical_rank = 0

        try:
            rank = live_labels.index(expected) + 1
        except ValueError:
            rank = 0

        results.append({
            "probe": probe["name"],
            "expected": expected,
            "rank": rank,
            "hit": 1 <= rank <= 5,
            "hierarchical_rank": hierarchical_rank,
            "hierarchical_hit": 1 <= hierarchical_rank <= 5,
            "hierarchical_top5": hierarchical_live_labels[:5],
            "retrieved_labels": last_retrieved_labels,
            "retrieval_trace": retrieval_trace,
            "baseline_rank": baseline_rank,
            "baseline_hit": 1 <= baseline_rank <= 5,
            "baseline_top5": baseline_live_labels[:5],
            "top5": live_labels[:5],
        })

    hits = sum(r["hit"] for r in results)
    baseline_hits = sum(r["baseline_hit"] for r in results)
    hierarchical_hits = sum(r["hierarchical_hit"] for r in results)
    total = len(results)

    record = {
        "config": {
            "num_clusters": num_clusters,
            "effective_num_clusters": effective_num_clusters,
            "M": M,
            "N": N,
            "selection_rule": selection_rule,
            "query_source": query_source,
            "use_label_accumulator": use_label_accumulator,
            "seed": seed,
            "cluster_scope": cluster_scope,
            "clustered_rows": len(cluster_row_ids),
            "label_db_dir": str(label_db_dir),
        },
        "hit_rate": hits / total,
        "hits": hits,
        "baseline_hit_rate": baseline_hits / total,
        "baseline_hits": baseline_hits,
        "hierarchical_hit_rate": hierarchical_hits / total,
        "hierarchical_hits": hierarchical_hits,
        "total_probes": total,
        "results": results,
        **get_git_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    (output_dir / "results.json").write_text(json.dumps(record, indent=2))
    return record


def run_ablation(args):
    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    # Focused, reproducible ablation grid. It starts coarse enough to expose
    # cluster-resolution changes, then includes the previous high-k settings.
    configs = [
        (5, 1, 1, "kind_priority", "mu_acc", True),
        (5, 5, 1, "kind_priority", "mu_acc", True),
        (10, 3, 1, "kind_priority", "mu_acc", True),
        (10, 5, 1, "kind_priority", "mu_acc", True),
        (20, 5, 1, "kind_priority", "mu_acc", True),
        (20, 5, 2, "kind_priority", "mu_acc", True),
        (20, 5, 4, "kind_priority", "mu_acc", True),
        (40, 5, 1, "kind_priority", "mu_acc", True),
        (40, 5, 8, "kind_priority", "mu_acc", True),
        (50, 3, 10, "cosine", "per_signal", True),
        (100, 5, 1, "kind_priority", "mu_acc", True),
        (100, 5, 2, "kind_priority", "mu_acc", True),
        (200, 6, 1, "kind_priority", "mu_acc", True),
        (200, 6, 2, "kind_priority", "mu_acc", True),
        (200, 6, 1, "lowest_degree", "mu_acc", True),
        (200, 6, 1, "cosine", "mu_acc", True),
        (200, 6, 1, "kind_priority", "per_signal", True),
        (400, 8, 1, "kind_priority", "mu_acc", True),
    ]

    summary_rows = []

    for k, M, N, rule, qsrc, use_acc in configs:
        run_name = f"k{k}_M{M}_N{N}_{rule}_{qsrc}"
        out_dir = base_output / run_name
        print(f"\n>>> Running: {run_name}")

        rec = run_single_hierarchical_experiment(
            label_db_dir=Path(args.label_db_dir),
            output_dir=out_dir,
            num_clusters=k,
            M=M,
            N=N,
            selection_rule=rule,
            query_source=qsrc,
            use_label_accumulator=use_acc,
            seed=args.seed,
            cluster_scope=args.cluster_scope,
        )
        rec["run_name"] = run_name
        summary_rows.append({
            "run": run_name,
            "hit_rate": rec["hit_rate"],
            "hits": rec["hits"],
            "baseline_hit_rate": rec["baseline_hit_rate"],
            "baseline_hits": rec["baseline_hits"],
            "hierarchical_hit_rate": rec["hierarchical_hit_rate"],
            "hierarchical_hits": rec["hierarchical_hits"],
            "k": k, "M": M, "N": N,
            "effective_k": rec["config"]["effective_num_clusters"],
            "cluster_scope": rec["config"]["cluster_scope"],
            "clustered_rows": rec["config"]["clustered_rows"],
            "rule": rule, "query": qsrc,
        })
        print(
            f"    Hit rate: {rec['hit_rate']:.3f} ({rec['hits']}/{rec['total_probes']}); "
            f"hierarchical-only: {rec['hierarchical_hit_rate']:.3f} "
            f"({rec['hierarchical_hits']}/{rec['total_probes']}); "
            f"baseline: {rec['baseline_hit_rate']:.3f} "
            f"({rec['baseline_hits']}/{rec['total_probes']})"
        )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "results": summary_rows,
    }
    (base_output / "ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Done. Summary: {base_output / 'ablation_summary.json'} ===")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-db-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build/hierarchical_label_ablation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cluster-scope",
        choices=("allowed_labels", "all_rows"),
        default="allowed_labels",
        help="Rows included in offline clustering. allowed_labels keeps the probe vocabulary complete.",
    )
    args = parser.parse_args()

    return run_ablation(args)


if __name__ == "__main__":
    sys.exit(main())
