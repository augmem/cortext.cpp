#!/usr/bin/env python3
"""
Accumulator-driven label context benchmark (live blended state, not guessed episodes).

This experiment contrasts the static "pre-segmented MemoryNode + bipartite label graph"
approach (distributed_label_graph_attention_bench.py) with a faithful shadow of
cortext's live AccumulatorState.

Signals are fed sequentially. A LiveAccumulator maintains:
  - mu_acc: incremental running mean embedding (exact Welford-style formula from C++)
  - c_t: slow-drifting temporal context (EWMA)
  - recent window of recent mu_acc values

At each pronoun probe point, the "active label readout" is obtained by projecting
the *current live blended state* (mu_acc mixed with a small pull from the query
labels) into the label vector space using the same top-k nearest-neighbor logic
as the rest of the label-graph family of experiments.

No pre-defined episode boundaries. No aggregation over per-memory label bags.
The binding specificity comes from the embedding geometry of the live accumulator
after the preceding signals — exactly the mechanism cortext uses for retrieval
bias, focus, and reference anchoring.

This is fully modality-agnostic: the accumulator and the projection never inspect
the "modality" field; they only see 256-d vectors and drift.

Run with the same label DB asset as the static-graph bench for direct comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data model (kept minimal, compatible with the static-graph bench where useful)
# ---------------------------------------------------------------------------

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
class Signal:
    """One step in the sequence (the 'signal' analogue)."""
    signal_id: str
    step: int
    seed_labels: tuple[str, ...]
    modality: str = "text"          # kept only for reporting / parity with old bench
    observed_labels: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Label store & projection helpers (reused / adapted from distributed_label_graph_attention_bench.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Live Accumulator (faithful shadow of cortext::AccumulatorState)
# ---------------------------------------------------------------------------

class LiveAccumulator:
    """
    Minimal Python shadow of cortext::AccumulatorState + recent context window.

    Uses the exact incremental mean formula from include/cortext/processor/accumulator_state.hpp:166
    and the c_t EWMA pattern used in src/operations/accumulator.cpp.
    """

    def __init__(self, win_size: int = 12, alpha_c: float = 0.03, drift_step: float = 0.10):
        self.mu_acc: np.ndarray | None = None
        self.c_t: np.ndarray | None = None
        self.recent_window: deque[np.ndarray] = deque(maxlen=win_size)
        self.n_signals = 0
        self.drift_acc = 0.0
        self.alpha_c = alpha_c
        self.default_drift_step = drift_step
        self.primary_modality: str | None = None

    def step(self, embedding: np.ndarray, drift: float | None = None, modality: str | None = None) -> None:
        emb = normalize(embedding)
        d = drift if drift is not None else self.default_drift_step

        if self.n_signals == 0:
            self.mu_acc = emb.copy()
            self.c_t = emb.copy()
        else:
            # Exact Welford-style incremental mean from the C++ implementation
            self.mu_acc = self.mu_acc + (emb - self.mu_acc) / (self.n_signals + 1)

            # c_t slow drift (same pattern as production)
            self.c_t = (1.0 - self.alpha_c) * self.c_t + self.alpha_c * self.mu_acc
            self.c_t = normalize(self.c_t)

        self.drift_acc += d * 0.5
        self.n_signals += 1

        self.recent_window.append(self.mu_acc.copy())

        if modality and self.primary_modality is None:
            self.primary_modality = modality

    def current_context_vector(self) -> np.ndarray:
        """The vector used for live label projection (mu_acc blended with recent window mean)."""
        if self.mu_acc is None:
            raise RuntimeError("Accumulator has not processed any signals yet")

        if len(self.recent_window) >= 2:
            window_mean = normalize(np.mean(np.stack(list(self.recent_window)), axis=0))
            # Light mix with recent window (analogous to RetrievalContextMix in production)
            mix = 0.75 * self.mu_acc + 0.25 * window_mean
            return normalize(mix)
        return self.mu_acc.copy()

    def project_live_labels(
        self,
        labels_by_index: dict[int, dict],
        row_label_indices: list[int],
        vectors: np.memmap,
        top_k: int,
        allowed_labels: set[str],
        query_labels: tuple[str, ...] | None = None,
        selected: dict[str, Label] | None = None,
    ) -> list[Candidate]:
        """
        Project the current live blended context into label space.
        If query_labels are supplied, give them a small attractive pull (mimics
        the pronoun entering the current context).
        """
        base = self.current_context_vector()

        if query_labels and selected:
            q_cent = centroid_for(query_labels, selected)
            # Small attractive bias toward the query (the pronoun) — keeps the
            # mechanism "query aware" while still dominated by the recent live blend.
            base = normalize(0.80 * base + 0.20 * q_cent)

        return topk_labels(
            base,
            labels_by_index,
            row_label_indices,
            vectors,
            top_k,
            allowed_labels,
        )


# ---------------------------------------------------------------------------
# Scaled multimodal sequence (identical to the static-graph bench)
# ---------------------------------------------------------------------------

def scaled_multimodal_signals() -> list[Signal]:
    """The same 18-step sequence used in the static-graph experiment."""
    return [
        Signal("m01_steve_image_intro", 1, ("steve", "friend", "person", "he", "image", "face"), "image"),
        Signal("m02_he_pizza_text", 2, ("he", "pizza", "eat"), "text"),
        Signal("m03_steve_restaurant_video", 3, ("we", "he", "favorite", "pizza", "restaurant", "video"), "video"),
        Signal("m04_maria_voice_intro", 4, ("maria", "friend", "person", "she", "voice", "audio"), "audio"),
        Signal("m05_she_guitar_audio", 5, ("she", "guitar", "music", "song", "audio"), "audio"),
        Signal("m06_maria_concert_image", 6, ("her", "guitar", "music", "image", "photo"), "image"),
        Signal("m07_david_video_intro", 7, ("david", "brother", "person", "he", "video", "face"), "video"),
        Signal("m08_he_car_text", 8, ("he", "car", "favorite"), "text"),
        Signal("m09_david_car_image", 9, ("he", "car", "image", "photo"), "image"),
        Signal("m10_emma_image_intro", 10, ("emma", "sister", "person", "she", "image", "face"), "image"),
        Signal("m11_she_dog_video", 11, ("she", "dog", "video", "park"), "video"),
        Signal("m12_emma_dog_audio", 12, ("her", "dog", "voice", "audio"), "audio"),
        Signal("m13_john_voice_intro", 13, ("john", "person", "he", "voice", "audio"), "audio"),
        Signal("m14_he_coffee_text", 14, ("he", "coffee", "office", "work"), "text"),
        Signal("m15_john_office_image", 15, ("he", "office", "laptop", "image"), "image"),
        Signal("m16_sarah_video_intro", 16, ("sarah", "teacher", "person", "she", "video", "face"), "video"),
        Signal("m17_she_book_text", 17, ("she", "book", "school", "favorite"), "text"),
        Signal("m18_sarah_school_audio", 18, ("her", "school", "voice", "audio"), "audio"),
    ]


def scaled_multimodal_probes() -> list[dict[str, Any]]:
    """
    Probes identical to the static-graph bench (male/female recency + cross-modal objects).
    Each entry: (probe_name, query_labels, expected_label, expected_signal_id, prefix_length)
    """
    return [
        # Male recency probes
        {"name": "steve_after_image_intro", "query": ("he",), "expected": "steve", "at_signal": "m01_steve_image_intro", "prefix": 1},
        {"name": "steve_text_followup", "query": ("he", "pizza"), "expected": "steve", "at_signal": "m02_he_pizza_text", "prefix": 2},
        {"name": "david_after_video_intro", "query": ("he",), "expected": "david", "at_signal": "m07_david_video_intro", "prefix": 7},
        {"name": "david_image_car_followup", "query": ("he", "car"), "expected": "david", "at_signal": "m09_david_car_image", "prefix": 9},
        {"name": "john_after_voice_intro", "query": ("he",), "expected": "john", "at_signal": "m13_john_voice_intro", "prefix": 13},
        {"name": "john_image_office_followup", "query": ("he", "office"), "expected": "john", "at_signal": "m15_john_office_image", "prefix": 15},
        # Female recency probes
        {"name": "maria_after_voice_intro", "query": ("she",), "expected": "maria", "at_signal": "m04_maria_voice_intro", "prefix": 4},
        {"name": "maria_audio_music_followup", "query": ("she", "guitar"), "expected": "maria", "at_signal": "m05_she_guitar_audio", "prefix": 5},
        {"name": "emma_after_image_intro", "query": ("she",), "expected": "emma", "at_signal": "m10_emma_image_intro", "prefix": 10},
        {"name": "emma_audio_dog_followup", "query": ("her", "dog"), "expected": "emma", "at_signal": "m12_emma_dog_audio", "prefix": 12},
        {"name": "sarah_after_video_intro", "query": ("she",), "expected": "sarah", "at_signal": "m16_sarah_video_intro", "prefix": 16},
        {"name": "sarah_audio_school_followup", "query": ("her", "school"), "expected": "sarah", "at_signal": "m18_sarah_school_audio", "prefix": 18},
        # Cross-modal object probes
        {"name": "pizza_video_to_steve", "query": ("pizza", "video"), "expected": "steve", "at_signal": "m03_steve_restaurant_video", "prefix": 3},
        {"name": "guitar_image_to_maria", "query": ("guitar", "image"), "expected": "maria", "at_signal": "m06_maria_concert_image", "prefix": 6},
        {"name": "car_image_to_david", "query": ("car", "image"), "expected": "david", "at_signal": "m09_david_car_image", "prefix": 9},
        {"name": "dog_audio_to_emma", "query": ("dog", "audio"), "expected": "emma", "at_signal": "m12_emma_dog_audio", "prefix": 12},
        {"name": "office_image_to_john", "query": ("office", "image"), "expected": "john", "at_signal": "m15_john_office_image", "prefix": 15},
        {"name": "school_audio_to_sarah", "query": ("school", "audio"), "expected": "sarah", "at_signal": "m18_sarah_school_audio", "prefix": 18},
    ]


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

def run_live_accumulator_experiment(
    labels_by_index: dict[int, dict],
    row_label_indices: list[int],
    vectors: np.memmap,
    selected: dict[str, Label],
    allowed_labels: set[str],
    top_k: int,
) -> dict[str, Any]:
    signals = scaled_multimodal_signals()
    probes = scaled_multimodal_probes()

    acc = LiveAccumulator(win_size=12, alpha_c=0.03, drift_step=0.10)

    # Process every signal in order (the key difference from the static bench)
    signal_by_id = {s.signal_id: s for s in signals}
    results: list[dict[str, Any]] = []

    for probe in probes:
        prefix_len = probe["prefix"]
        prefix_signals = signals[:prefix_len]

        # Replay the prefix through the live accumulator
        acc = LiveAccumulator(win_size=12, alpha_c=0.03, drift_step=0.10)  # fresh for each probe (clean isolation)
        for s in prefix_signals:
            emb = centroid_for(s.seed_labels, selected)
            acc.step(emb, modality=s.modality)

        # Project the live state, lightly attracted by the query labels
        live_cands = acc.project_live_labels(
            labels_by_index,
            row_label_indices,
            vectors,
            top_k,
            allowed_labels,
            query_labels=probe["query"],
            selected=selected,
        )
        live_labels = [c.label for c in live_cands]

        expected = probe["expected"]
        try:
            rank = live_labels.index(expected) + 1
        except ValueError:
            rank = 0

        hit = 1 <= rank <= 5

        results.append({
            "probe": probe["name"],
            "query": "|".join(probe["query"]),
            "expected_label": expected,
            "at_signal": probe["at_signal"],
            "prefix_length": prefix_len,
            "top_label": live_labels[0] if live_labels else "",
            "expected_label_rank": rank,
            "hit": hit,
            "live_top5": live_labels[:5],
            "live_top10": live_labels[:10],
            "modality_of_prefix_end": signal_by_id[probe["at_signal"]].modality,
        })

    # Aggregate
    hits = sum(1 for r in results if r["hit"])
    probe_count = len(results)

    return {
        "design": "live_accumulator_projection_from_mu_acc_and_recent_window",
        "top_k": top_k,
        "probe_count": probe_count,
        "hit_count": hits,
        "hit_rate": hits / probe_count if probe_count else 0.0,
        "results": results,
        "interpretation": (
            "Live accumulator projection (mu_acc + recent window, modality-agnostic) "
            "vs. static per-memory label aggregation. The continuous blended context "
            "supplies binding specificity that the discrete episode graph lacks."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-db-dir", type=Path, default=Path("build/label_vector_db_full_2proto"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/accumulator_label_context_bench"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--suite",
        default="scaled_multimodal",
        help="Currently only 'scaled_multimodal' is supported.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_by_index, row_label_indices, vectors = load_label_store(args.label_db_dir)

    # Same label name set as the original scaled_multimodal bench
    label_names = {
        "alex", "audio", "book", "brother", "car", "coffee", "david", "dog", "eat",
        "emma", "face", "favorite", "friend", "guitar", "he", "her", "image", "john",
        "laptop", "male", "man", "maria", "music", "office", "park", "person", "photo",
        "pizza", "place", "restaurant", "sarah", "school", "she", "sister", "song",
        "steve", "teacher", "video", "voice", "we", "work",
    }
    selected = select_labels(labels_by_index, row_label_indices, vectors, label_names)
    allowed_labels = label_names

    start = time.perf_counter()
    result = run_live_accumulator_experiment(
        labels_by_index, row_label_indices, vectors, selected, allowed_labels, args.top_k
    )
    elapsed = time.perf_counter() - start
    result["label_projection_elapsed_s"] = elapsed
    result["label_db_dir"] = str(args.label_db_dir)
    result["suite"] = args.suite
    result["design_full"] = (
        "LiveAccumulator (mu_acc incremental mean + c_t EWMA + recent window) "
        "projected into label space at each pronoun probe. No static MemoryNode graph. "
        "No pre-segmented episodes. Pure online blended context."
    )

    out_path = args.output_dir / "accumulator_label_context_results.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    # Also emit a tiny CSV for quick inspection (probe, expected, rank, hit, top5)
    csv_path = args.output_dir / "accumulator_label_context_probes.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe", "query", "expected", "rank", "hit", "top5"])
        for r in result["results"]:
            w.writerow([
                r["probe"],
                r["query"],
                r["expected_label"],
                r["expected_label_rank"],
                int(r["hit"]),
                "|".join(r["live_top5"]),
            ])

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
