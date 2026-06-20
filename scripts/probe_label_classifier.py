#!/usr/bin/env python3
"""
Run production-style ranked span extraction against a trained offline label classifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from label_classifier_lib import (
    calibrate_type_probabilities,
    ExternalTextEmbedder,
    STOPWORDS,
    featurize_example,
    generate_candidate_spans,
    load_name_priors,
    load_wordnet_index,
    normalize_key,
    predict_softmax,
    tokenize_with_offsets,
)


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_map(vector: np.ndarray, model: dict) -> tuple[str, dict[str, float]]:
    labels, probs = predict_softmax(
        np.asarray([vector], dtype=np.float32),
        np.asarray(model["weights"], dtype=np.float32),
        np.asarray(model["bias"], dtype=np.float32),
        list(model["classes"]),
        np.asarray(model["mean"], dtype=np.float32),
        np.asarray(model["std"], dtype=np.float32),
    )
    return labels[0], {label: float(prob) for label, prob in zip(model["classes"], probs[0])}


def helper_penalty(span_text: str, label: str) -> float:
    tokens = [normalize_key(token) for token, _, _ in tokenize_with_offsets(span_text)]
    if not tokens:
        return 0.0
    token_set = set(tokens)
    if label == "topic" and token_set & {"name", "today", "yesterday", "tomorrow"}:
        return 0.45
    if label == "state" and token_set & {"feel", "feeling", "live", "next", "month"}:
        return 0.45
    if label in {"identity", "person_entity"} and token_set & {"name", "debugging", "migration"}:
        return 0.40
    return 1.0


def span_score(span_text: str, type_probs: dict[str, float], promotion_probs: dict[str, float]) -> tuple[str, float]:
    tokens = [normalize_key(token) for token, _, _ in tokenize_with_offsets(span_text)]
    stop_ratio = sum(1 for token in tokens if token in STOPWORDS) / max(1, len(tokens))
    non_none = {label: prob for label, prob in type_probs.items() if label != "none"}
    label = max(non_none, key=non_none.get)
    base = non_none[label]
    promotion = max(promotion_probs["durable"], promotion_probs["provisional"])
    length_penalty = 1.0
    if len(tokens) >= 4:
        length_penalty *= 0.55
    elif len(tokens) == 3:
        length_penalty *= 0.75
    if stop_ratio > 0.34:
        length_penalty *= 0.6
    if label in {"identity", "person_entity", "place", "org_project"} and len(tokens) > 2:
        length_penalty *= 0.7
    if label in {"state", "topic"} and len(tokens) > 2:
        length_penalty *= 0.8
    return label, base * promotion * length_penalty * helper_penalty(span_text, label)


def overlaps(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
    return not (span_a[1] <= span_b[0] or span_b[1] <= span_a[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe a trained label classifier with ranked extraction.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--wordnet-index", required=True)
    parser.add_argument("--name-priors", required=True)
    parser.add_argument("--text", action="append", default=[], help="Input sentence to probe.")
    parser.add_argument("--embedder-bin", default="")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()

    wordnet = load_wordnet_index(Path(args.wordnet_index))
    priors = load_name_priors(Path(args.name_priors))
    type_model = load_model(Path(args.model_dir) / "type_model.json")
    promotion_model = load_model(Path(args.model_dir) / "promotion_model.json")
    embedder = None
    if args.embedder_bin:
        embedder = ExternalTextEmbedder(
            binary=args.embedder_bin,
            models_dir=args.models_dir,
        )

    texts: list[str] = []
    for text in args.text:
        texts.append(text)
        for span in generate_candidate_spans(text):
            texts.append(span.text)
    if embedder:
        embedder.prefetch(texts)

    for text in args.text:
        print(f"TEXT: {text}")
        candidates = []
        for span in generate_candidate_spans(text):
            vector = featurize_example(text, span.text, wordnet, priors, embedder)
            _, type_probs = predict_map(vector, type_model)
            type_probs = calibrate_type_probabilities(text, span.text, type_probs, wordnet, priors)
            _, promotion_probs = predict_map(vector, promotion_model)
            label, score = span_score(span.text, type_probs, promotion_probs)
            if label == "none" or score < args.threshold:
                continue
            candidates.append(
                {
                    "span": span.text,
                    "start": span.start,
                    "end": span.end,
                    "label": label,
                    "score": score,
                    "type_prob": max(prob for key, prob in type_probs.items() if key != "none"),
                    "promotion_prob": max(promotion_probs["durable"], promotion_probs["provisional"]),
                }
            )
        candidates.sort(key=lambda row: (row["score"], row["type_prob"]), reverse=True)
        selected: list[dict] = []
        for candidate in candidates:
            if any(overlaps((candidate["start"], candidate["end"]), (kept["start"], kept["end"])) for kept in selected):
                continue
            selected.append(candidate)
            if len(selected) >= args.limit:
                break
        for candidate in selected:
            print(
                f"  span={candidate['span']!r:24} label={candidate['label']:14} "
                f"score={candidate['score']:.3f} type={candidate['type_prob']:.3f} "
                f"promo={candidate['promotion_prob']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
