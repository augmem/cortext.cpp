#!/usr/bin/env python3
"""Ablate clustered-label routing in the realtime label accumulator.

This is benchmark-only. It does not read or write production Cortext databases.

The production-shaped path under test is:

    signal embedding
      -> global Top-R labels as router hints
      -> find those labels' cluster-tree leaves
      -> take Top-K labels from each reached leaf
      -> feed the bounded label set into the realtime LabelAccumulator

The important distinction from earlier tree-only probes is that the tree is not
asked to route the signal from scratch. The global shortlist preserves identity
labels, then the cluster tree narrows/expands locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accumulator_label_context_bench import (  # noqa: E402
    LabelAccumulator,
    centroid_for,
    load_label_store,
    normalize,
    scaled_multimodal_probes,
    scaled_multimodal_signals,
    select_labels,
    topk_labels,
)


LABEL_NAMES = {
    "alex", "audio", "book", "brother", "car", "coffee", "david", "dog", "eat",
    "emma", "face", "favorite", "friend", "guitar", "he", "her", "image", "john",
    "laptop", "male", "man", "maria", "music", "office", "park", "person", "photo",
    "pizza", "place", "restaurant", "sarah", "school", "she", "sister", "song",
    "steve", "teacher", "video", "voice", "we", "work",
}


@dataclass
class ProtoRow:
    label: str
    row_id: int
    label_index: int
    kind: str
    partition: str


@dataclass
class TreeNode:
    id: int
    partition: str
    depth: int
    row_indices: list[int]
    centroid: np.ndarray
    children: list[int]
    leaf_id: int | None = None


def partition_for(label: str, kind: str) -> str:
    names = {"alex", "david", "emma", "john", "maria", "sarah", "steve"}
    pronouns = {"he", "she", "her", "we"}
    relations = {"brother", "sister", "friend"}
    generic_person = {"person", "man", "male", "teacher"}
    modalities = {"audio", "image", "video", "voice", "face", "photo"}
    food = {"pizza", "coffee", "eat", "restaurant"}
    places = {"office", "park", "place", "school"}
    objects = {"book", "car", "dog", "guitar", "laptop"}
    abstract = {"favorite", "music", "song", "work"}
    if label in names:
        return "bank:proper_or_unknown_name"
    if label in pronouns:
        return "bank:pronoun"
    if label in relations:
        return "bank:relation"
    if label in generic_person:
        return "bank:generic_person"
    if label in modalities:
        return "bank:modality"
    if label in food:
        return "bank:food"
    if label in places:
        return "bank:place"
    if label in objects:
        return "bank:object"
    if label in abstract:
        return "bank:abstract_activity"
    return f"kind:{kind or 'unknown'}"


def prototype_rows(
    labels_by_index: dict[int, dict],
    row_label_indices: list[int],
) -> list[ProtoRow]:
    fallback: dict[str, ProtoRow] = {}
    preferred: dict[str, ProtoRow] = {}
    for row_id, label_index in enumerate(row_label_indices):
        meta = labels_by_index[label_index]
        label = meta["label"].lower()
        row = ProtoRow(
            label=label,
            row_id=row_id,
            label_index=label_index,
            kind=meta.get("kind", ""),
            partition=partition_for(label, meta.get("kind", "")),
        )
        fallback.setdefault(label, row)
        if meta.get("source") == "data/label_bank/labels.jsonl":
            preferred.setdefault(label, row)
    fallback.update(preferred)
    return list(fallback.values())


def cosine_kmeans(vectors: np.ndarray, k: int, seed: int, max_iter: int = 25) -> tuple[np.ndarray, np.ndarray]:
    n = vectors.shape[0]
    if n <= k:
        return np.arange(n, dtype=np.int32), vectors.copy()
    rng = np.random.default_rng(seed)
    init = rng.choice(n, size=k, replace=False)
    centroids = vectors[init].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        new_labels = np.argmax(vectors @ centroids.T, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cid in range(k):
            members = vectors[labels == cid]
            if len(members):
                centroids[cid] = normalize(np.mean(members, axis=0))
            else:
                centroids[cid] = vectors[int(rng.integers(0, n))]
    return labels, centroids


def build_leaf_tree(
    rows: list[ProtoRow],
    vectors: np.memmap,
    branch: int,
    max_leaf: int,
    seed: int,
) -> dict[str, Any]:
    by_partition: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_partition[row.partition].append(i)

    row_vecs = np.stack([normalize(np.asarray(vectors[row.row_id], dtype=np.float32)) for row in rows])
    leaves: list[dict[str, Any]] = []
    label_to_leaf: dict[str, int] = {}
    nodes: list[TreeNode] = []
    partition_roots: dict[str, int] = {}

    def make_centroid(indices: list[int]) -> np.ndarray:
        return normalize(np.mean(row_vecs[indices], axis=0))

    def split(partition: str, indices: list[int], depth: int, local_seed: int) -> int:
        node_id = len(nodes)
        nodes.append(TreeNode(node_id, partition, depth, indices, make_centroid(indices), []))
        if len(indices) <= max_leaf:
            leaf_id = len(leaves)
            labels = [rows[i].label for i in indices]
            leaves.append({"id": leaf_id, "depth": depth, "row_indices": indices, "labels": labels})
            for label in labels:
                label_to_leaf.setdefault(label, leaf_id)
            nodes[node_id].leaf_id = leaf_id
            return node_id
        k = min(branch, max(2, int(np.ceil(len(indices) / max_leaf))))
        labels, _ = cosine_kmeans(row_vecs[indices], k, local_seed)
        for cid in range(k):
            child = [indices[i] for i in np.where(labels == cid)[0].tolist()]
            if not child:
                continue
            if len(child) == len(indices):
                mid = max_leaf
                nodes[node_id].children.append(split(partition, child[:mid], depth + 1, local_seed + cid + 1))
                nodes[node_id].children.append(split(partition, child[mid:], depth + 1, local_seed + cid + 17))
            else:
                nodes[node_id].children.append(split(partition, child, depth + 1, local_seed + cid + 1))
        return node_id

    for pnum, (partition, indices) in enumerate(sorted(by_partition.items())):
        partition_roots[partition] = split(partition, indices, 1, seed + pnum * 7919)

    sizes = [len(leaf["labels"]) for leaf in leaves]
    return {
        "branch": branch,
        "max_leaf": max_leaf,
        "row_vecs": row_vecs,
        "rows": rows,
        "leaves": leaves,
        "label_to_leaf": label_to_leaf,
        "nodes": nodes,
        "partition_roots": partition_roots,
        "summary": {
            "prototype_labels": len(rows),
            "partitions": len(by_partition),
            "leaves": len(leaves),
            "leaf_p50": float(np.percentile(sizes, 50)) if sizes else 0.0,
            "leaf_p90": float(np.percentile(sizes, 90)) if sizes else 0.0,
            "leaf_max": max(sizes) if sizes else 0,
        },
    }


def top_labels_from_leaf(
    leaf: dict[str, Any],
    query: np.ndarray,
    tree: dict[str, Any],
    allowed: set[str],
    per_leaf_k: int,
) -> list[tuple[str, float]]:
    scored: list[tuple[float, str]] = []
    row_vecs = tree["row_vecs"]
    rows = tree["rows"]
    for row_index in leaf["row_indices"]:
        label = rows[row_index].label
        if label not in allowed:
            continue
        scored.append((float(row_vecs[row_index] @ query), label))
    scored.sort(key=lambda x: -x[0])
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for score, label in scored:
        if label in seen:
            continue
        seen.add(label)
        out.append((label, max(0.0, min(1.0, (score + 1.0) * 0.5))))
        if len(out) >= per_leaf_k:
            break
    return out


def seeded_tree_candidates(
    query: np.ndarray,
    router_labels: list[str],
    tree: dict[str, Any],
    allowed: set[str],
    per_leaf_k: int,
) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    reached: list[int] = []
    for label in router_labels:
        leaf_id = tree["label_to_leaf"].get(label)
        if leaf_id is None or leaf_id in reached:
            continue
        reached.append(leaf_id)
        for cand, score in top_labels_from_leaf(tree["leaves"][leaf_id], query, tree, allowed, per_leaf_k):
            if cand not in seen:
                seen.add(cand)
                out.append((cand, score))
    return out


def drill_partition_candidates(
    query: np.ndarray,
    tree: dict[str, Any],
    allowed: set[str],
    partition_top: int,
    partition_threshold: float,
    child_top: int,
    per_leaf_k: int,
    forced_partitions: set[str] | None = None,
) -> list[tuple[str, float]]:
    nodes: list[TreeNode] = tree["nodes"]
    root_ids = list(tree["partition_roots"].values())
    root_scores = [(float(nodes[root_id].centroid @ query), root_id) for root_id in root_ids]
    root_scores.sort(key=lambda x: -x[0])

    selected_roots: list[int] = []
    forced_partitions = forced_partitions or set()
    for rank, (score, root_id) in enumerate(root_scores, start=1):
        if rank <= partition_top or score >= partition_threshold or nodes[root_id].partition in forced_partitions:
            selected_roots.append(root_id)

    reached_leaves: list[int] = []

    def descend(node_id: int) -> None:
        node = nodes[node_id]
        if node.leaf_id is not None:
            reached_leaves.append(node.leaf_id)
            return
        child_scores = [(float(nodes[child].centroid @ query), child) for child in node.children]
        child_scores.sort(key=lambda x: -x[0])
        for _, child in child_scores[:child_top]:
            descend(child)

    for root_id in selected_roots:
        descend(root_id)

    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for leaf_id in reached_leaves:
        for label, score in top_labels_from_leaf(tree["leaves"][leaf_id], query, tree, allowed, per_leaf_k):
            if label not in seen:
                seen.add(label)
                out.append((label, score))
    return out


def global_router_candidates(
    query: np.ndarray,
    labels_by_index: dict[int, dict],
    row_label_indices: list[int],
    vectors: np.memmap,
    allowed: set[str],
    router_top: int,
) -> list[tuple[str, float]]:
    return [
        (c.label, c.score01)
        for c in topk_labels(query, labels_by_index, row_label_indices, vectors, router_top, allowed)
    ]


def candidate_labels(candidates: list[tuple[str, float]]) -> list[str]:
    return [label for label, _ in candidates]


def add_candidate_labels(label_acc: LabelAccumulator, candidates: list[tuple[str, float]]) -> None:
    label_acc.step_count += 1
    for label in list(label_acc.scores.keys()):
        label_acc.scores[label] *= label_acc.decay
    for label, score in candidates:
        contribution = score
        if label in label_acc.GENERIC_SUPPRESS:
            contribution *= label_acc.generic_suppression
        if label in label_acc.last_seen_step:
            steps_ago = label_acc.step_count - label_acc.last_seen_step[label]
            if steps_ago <= 2:
                contribution *= label_acc.recency_boost
        label_acc.scores[label] += contribution
        label_acc.last_seen_step[label] = label_acc.step_count
    if len(label_acc.scores) > label_acc.max_labels:
        kept = sorted(label_acc.scores.items(), key=lambda x: -x[1])[:label_acc.max_labels]
        label_acc.scores = defaultdict(float, dict(kept))


def evaluate_config(
    name: str,
    tree: dict[str, Any],
    labels_by_index: dict[int, dict],
    row_label_indices: list[int],
    vectors: np.memmap,
    selected: dict[str, Any],
    router_top: int,
    per_leaf_k: int,
    mode: str,
    partition_top: int = 3,
    partition_threshold: float = 0.20,
    child_top: int = 1,
    force_anchor_partitions: bool = False,
) -> dict[str, Any]:
    signals = scaled_multimodal_signals()
    probes = scaled_multimodal_probes()
    rows: list[dict[str, Any]] = []
    candidate_counts: list[int] = []
    seed_hits = 0
    seed_total = 0

    for probe in probes:
        label_acc = LabelAccumulator(decay=0.82, max_labels=35, generic_suppression=0.22, recency_boost=1.4)
        traces: list[dict[str, Any]] = []
        for signal in signals[:probe["prefix"]]:
            emb = centroid_for(signal.seed_labels, selected)
            global_candidates = global_router_candidates(
                emb, labels_by_index, row_label_indices, vectors, LABEL_NAMES, router_top
            )
            global_labels = candidate_labels(global_candidates)
            if mode == "baseline":
                label_acc.step(
                    emb, labels_by_index, row_label_indices, vectors,
                    LABEL_NAMES, top_k=per_leaf_k
                )
                candidates = global_candidates[:per_leaf_k]
            elif mode == "seeded_tree":
                candidates = seeded_tree_candidates(emb, global_labels, tree, LABEL_NAMES, per_leaf_k)
            elif mode == "hybrid":
                local = seeded_tree_candidates(emb, global_labels, tree, LABEL_NAMES, per_leaf_k)
                candidates = []
                seen_candidates: set[str] = set()
                for label, score in [*global_candidates[:8], *local]:
                    if label not in seen_candidates:
                        seen_candidates.add(label)
                        candidates.append((label, score))
            elif mode == "partition_drill":
                forced = {
                    "bank:proper_or_unknown_name",
                    "bank:pronoun",
                    "bank:relation",
                    "bank:generic_person",
                } if force_anchor_partitions else set()
                candidates = drill_partition_candidates(
                    emb,
                    tree,
                    LABEL_NAMES,
                    partition_top=partition_top,
                    partition_threshold=partition_threshold,
                    child_top=child_top,
                    per_leaf_k=per_leaf_k,
                    forced_partitions=forced,
                )
            else:
                raise ValueError(mode)

            if mode != "baseline":
                add_candidate_labels(label_acc, candidates)
            candidate_counts.append(len(candidates))
            seed_set = set(signal.seed_labels)
            candidate_name_set = set(candidate_labels(candidates))
            seed_hits += len(seed_set & candidate_name_set)
            seed_total += len(seed_set)
            traces.append({
                "signal": signal.signal_id,
                "global_router": global_labels,
                "candidates": candidate_labels(candidates),
                "seed_recall": len(seed_set & candidate_name_set) / len(seed_set),
            })

        label_acc.boost_from_query(probe["query"], boost=0.55)
        top = label_acc.get_top_labels(12)
        expected = probe["expected"]
        rank = top.index(expected) + 1 if expected in top else 0
        rows.append({
            "probe": probe["name"],
            "expected": expected,
            "rank": rank,
            "hit": 1 <= rank <= 5,
            "top5": top[:5],
            "top12": top,
            "trace": traces,
        })

    hits = sum(1 for r in rows if r["hit"])
    return {
        "name": name,
        "mode": mode,
        "router_top": router_top,
        "per_leaf_k": per_leaf_k,
        "partition_top": partition_top,
        "partition_threshold": partition_threshold,
        "child_top": child_top,
        "force_anchor_partitions": force_anchor_partitions,
        "hit_count": hits,
        "hit_rate": hits / len(rows),
        "mean_candidates_per_signal": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "seed_label_recall": seed_hits / seed_total if seed_total else 0.0,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-db-dir", type=Path, default=Path("build/label_vector_db_full_2proto"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/realtime_label_cluster_tree_ablation_20260520"))
    parser.add_argument("--branch", type=int, default=8)
    parser.add_argument("--max-leaf", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_by_index, row_label_indices, vectors = load_label_store(args.label_db_dir)
    selected = select_labels(labels_by_index, row_label_indices, vectors, LABEL_NAMES)

    start = time.perf_counter()
    protos = prototype_rows(labels_by_index, row_label_indices)
    tree = build_leaf_tree(protos, vectors, args.branch, args.max_leaf, args.seed)
    build_s = time.perf_counter() - start
    tree["summary"]["build_s"] = build_s

    configs = [
        {"name": "baseline_global_top8", "router_top": 8, "per_leaf_k": 8, "mode": "baseline"},
        {"name": "seeded_tree_r8_leaf3", "router_top": 8, "per_leaf_k": 3, "mode": "seeded_tree"},
        {"name": "seeded_tree_r8_leaf5", "router_top": 8, "per_leaf_k": 5, "mode": "seeded_tree"},
        {"name": "seeded_tree_r16_leaf3", "router_top": 16, "per_leaf_k": 3, "mode": "seeded_tree"},
        {"name": "seeded_tree_r16_leaf5", "router_top": 16, "per_leaf_k": 5, "mode": "seeded_tree"},
        {"name": "seeded_tree_r32_leaf3", "router_top": 32, "per_leaf_k": 3, "mode": "seeded_tree"},
        {"name": "seeded_tree_r32_leaf5", "router_top": 32, "per_leaf_k": 5, "mode": "seeded_tree"},
        {"name": "hybrid_global8_seeded_r8_leaf3", "router_top": 8, "per_leaf_k": 3, "mode": "hybrid"},
        {"name": "hybrid_global8_seeded_r16_leaf3", "router_top": 16, "per_leaf_k": 3, "mode": "hybrid"},
        {"name": "partition_drill_p3_c1_leaf3", "router_top": 0, "per_leaf_k": 3, "mode": "partition_drill", "partition_top": 3, "child_top": 1},
        {"name": "partition_drill_p5_c1_leaf3", "router_top": 0, "per_leaf_k": 3, "mode": "partition_drill", "partition_top": 5, "child_top": 1},
        {"name": "partition_drill_p5_c2_leaf3", "router_top": 0, "per_leaf_k": 3, "mode": "partition_drill", "partition_top": 5, "child_top": 2},
        {"name": "partition_drill_p8_c2_leaf3", "router_top": 0, "per_leaf_k": 3, "mode": "partition_drill", "partition_top": 8, "child_top": 2},
        {"name": "partition_drill_forced_p3_c1_leaf3", "router_top": 0, "per_leaf_k": 3, "mode": "partition_drill", "partition_top": 3, "child_top": 1, "force_anchor_partitions": True},
        {"name": "partition_drill_forced_p5_c1_leaf3", "router_top": 0, "per_leaf_k": 3, "mode": "partition_drill", "partition_top": 5, "child_top": 1, "force_anchor_partitions": True},
        {"name": "partition_drill_forced_p5_c2_leaf3", "router_top": 0, "per_leaf_k": 3, "mode": "partition_drill", "partition_top": 5, "child_top": 2, "force_anchor_partitions": True},
    ]
    results = [
        evaluate_config(
            cfg["name"],
            tree,
            labels_by_index,
            row_label_indices,
            vectors,
            selected,
            cfg["router_top"],
            cfg["per_leaf_k"],
            cfg["mode"],
            partition_top=cfg.get("partition_top", 3),
            partition_threshold=cfg.get("partition_threshold", 0.20),
            child_top=cfg.get("child_top", 1),
            force_anchor_partitions=cfg.get("force_anchor_partitions", False),
        )
        for cfg in configs
    ]

    summary = [
        {k: r[k] for k in (
            "name", "mode", "router_top", "per_leaf_k", "partition_top",
            "partition_threshold", "child_top", "force_anchor_partitions",
            "hit_count", "hit_rate", "mean_candidates_per_signal", "seed_label_recall"
        )}
        for r in results
    ]
    record = {
        "design": "global_topk_seeded_partitioned_cluster_tree_realtime_label_ablation",
        "label_db_dir": str(args.label_db_dir),
        "tree": tree["summary"],
        "summary": summary,
        "configs": results,
    }
    out_json = args.output_dir / "realtime_label_cluster_tree_ablation.json"
    out_json.write_text(json.dumps(record, indent=2) + "\n")
    out_csv = args.output_dir / "realtime_label_cluster_tree_ablation_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(json.dumps({"tree": tree["summary"], "summary": summary}, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
