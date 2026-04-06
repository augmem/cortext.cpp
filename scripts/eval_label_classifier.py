#!/usr/bin/env python3
"""
Evaluate offline label classifiers against a span-label evaluation JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from label_classifier_lib import (
    calibrate_type_probabilities,
    ExternalTextEmbedder,
    confusion_matrix,
    featurize_example,
    generate_candidate_spans,
    load_examples,
    load_name_priors,
    load_wordnet_index,
    macro_f1,
    normalize_key,
    predict_softmax,
    precision_recall_f1,
    write_json,
)


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_matrix(
    examples: list[dict],
    wordnet_index: dict[str, dict],
    priors: dict[str, dict[str, float]],
    embedder: ExternalTextEmbedder | None,
) -> np.ndarray:
    return np.vstack(
        [
            featurize_example(
                sentence=example["text"],
                span_text=example["span"],
                wordnet_index=wordnet_index,
                priors=priors,
                embedder=embedder,
            )
            for example in examples
        ]
    ).astype(np.float32)


def prefetch_example_embeddings(
    examples: list[dict], embedder: ExternalTextEmbedder | None
) -> None:
    if embedder is None:
        return
    texts: list[str] = []
    for example in examples:
        texts.append(example["text"])
        texts.append(example["span"])
    embedder.prefetch(texts)


def candidate_recall(examples: list[dict]) -> float:
    hits = 0
    for example in examples:
        candidates = generate_candidate_spans(example["text"])
        if any(normalize_key(candidate.text) == normalize_key(example["span"]) for candidate in candidates):
            hits += 1
    return hits / float(len(examples)) if examples else 0.0


def supported_macro_f1(per_label: dict[str, dict[str, float]], support: dict[str, int]) -> float:
    included = [metrics["f1"] for label, metrics in per_label.items() if support.get(label, 0) > 0]
    if not included:
        return 0.0
    return sum(included) / float(len(included))


def evaluate_head(
    examples: list[dict],
    label_key: str,
    model: dict,
    wordnet_index: dict[str, dict],
    priors: dict[str, dict[str, float]],
    embedder: ExternalTextEmbedder | None,
) -> dict:
    X = build_matrix(examples, wordnet_index, priors, embedder)
    gold = [example[label_key] for example in examples]
    preds, probs = predict_softmax(
        X,
        np.asarray(model["weights"], dtype=np.float32),
        np.asarray(model["bias"], dtype=np.float32),
        list(model["classes"]),
        np.asarray(model["mean"], dtype=np.float32),
        np.asarray(model["std"], dtype=np.float32),
    )
    if label_key == "type_label":
        calibrated_preds: list[str] = []
        for example, row in zip(examples, probs):
            prob_map = {label: float(prob) for label, prob in zip(model["classes"], row)}
            calibrated = calibrate_type_probabilities(
                example["text"], example["span"], prob_map, wordnet_index, priors
            )
            calibrated_preds.append(max(calibrated, key=calibrated.get))
        preds = calibrated_preds
    per_label = precision_recall_f1(gold, preds, list(model["classes"]))
    support = {label: sum(1 for value in gold if value == label) for label in model["classes"]}
    return {
        "macro_f1": macro_f1(per_label),
        "supported_macro_f1": supported_macro_f1(per_label, support),
        "per_label": per_label,
        "confusion_matrix": confusion_matrix(gold, preds, list(model["classes"])),
        "labels": list(model["classes"]),
        "support": support,
        "accuracy": sum(1 for g, p in zip(gold, preds) if g == p) / float(len(gold) or 1),
    }


def render_markdown(type_metrics: dict, promotion_metrics: dict, candidate_hit_rate: float) -> str:
    lines = [
        "| Head | Macro F1 | Supported Macro F1 | Accuracy | Candidate Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Span typing | {type_metrics['macro_f1']:.3f} | {type_metrics['supported_macro_f1']:.3f} | {type_metrics['accuracy']:.3f} | {candidate_hit_rate:.3f} |",
        f"| Promotion | {promotion_metrics['macro_f1']:.3f} | {promotion_metrics['supported_macro_f1']:.3f} | {promotion_metrics['accuracy']:.3f} | {candidate_hit_rate:.3f} |",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate offline label classifiers.")
    parser.add_argument(
        "--gold",
        default="data/label_classifier/gold.jsonl",
        help="Evaluation JSONL with text/span/type_label/promotion_label rows.",
    )
    parser.add_argument(
        "--wordnet-index",
        default="data/label_classifier/wordnet_index.json",
        help="Compact WordNet index JSON.",
    )
    parser.add_argument(
        "--name-priors",
        default="data/label_classifier/name_priors.json",
        help="Name priors JSON.",
    )
    parser.add_argument(
        "--model-dir",
        default="models/label_classifier",
        help="Directory containing type_model.json and promotion_model.json.",
    )
    parser.add_argument(
        "--embedder-bin",
        default="",
        help="Optional external embedder binary.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Models directory passed to the embedder tool.",
    )
    parser.add_argument(
        "--embedder-backend",
        default="",
        help="Optional embedder backend override.",
    )
    parser.add_argument(
        "--out-dir",
        default="logs/label_classifier_eval",
        help="Output directory for evaluation reports.",
    )
    args = parser.parse_args()

    examples = load_examples(Path(args.gold))
    wordnet_index = load_wordnet_index(Path(args.wordnet_index))
    priors = load_name_priors(Path(args.name_priors))
    embedder = None
    if args.embedder_bin:
        embedder = ExternalTextEmbedder(
            binary=args.embedder_bin,
            models_dir=args.models_dir,
            backend=args.embedder_backend or None,
        )
        prefetch_example_embeddings(examples, embedder)

    model_dir = Path(args.model_dir)
    type_model = load_model(model_dir / "type_model.json")
    promotion_model = load_model(model_dir / "promotion_model.json")
    candidate_hit_rate = candidate_recall(examples)
    type_metrics = evaluate_head(
        examples, "type_label", type_model, wordnet_index, priors, embedder
    )
    promotion_metrics = evaluate_head(
        examples, "promotion_label", promotion_model, wordnet_index, priors, embedder
    )
    summary = {
        "candidate_recall": candidate_hit_rate,
        "type_metrics": type_metrics,
        "promotion_metrics": promotion_metrics,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(
        render_markdown(type_metrics, promotion_metrics, candidate_hit_rate),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
