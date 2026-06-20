#!/usr/bin/env python3
"""
Train offline span-typing and promotion classifiers for the label prototype.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from label_classifier_lib import (
    ExternalTextEmbedder,
    featurize_example,
    load_examples,
    load_name_priors,
    load_wordnet_index,
    macro_f1,
    predict_softmax,
    precision_recall_f1,
    train_softmax_regression,
    write_json,
)


def parse_label_weights(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values:
        label, _, raw_weight = value.partition("=")
        if not label or not raw_weight:
            raise SystemExit(f"Invalid class-weight '{value}', expected label=weight")
        weights[label] = float(raw_weight)
    return weights


def build_feature_matrix(
    examples: list[dict],
    wordnet_index: dict[str, dict],
    priors: dict[str, dict[str, float]],
    embedder: ExternalTextEmbedder | None,
) -> np.ndarray:
    rows = [
        featurize_example(
            sentence=example["text"],
            span_text=example["span"],
            wordnet_index=wordnet_index,
            priors=priors,
            embedder=embedder,
        )
        for example in examples
    ]
    return np.vstack(rows).astype(np.float32)


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


def train_and_score(
    train_examples: list[dict],
    valid_examples: list[dict],
    label_key: str,
    wordnet_index: dict[str, dict],
    priors: dict[str, dict[str, float]],
    embedder: ExternalTextEmbedder | None,
    epochs: int,
    learning_rate: float,
    class_weights: dict[str, float],
) -> tuple[dict, dict]:
    X_train = build_feature_matrix(train_examples, wordnet_index, priors, embedder)
    train_labels = [example[label_key] for example in train_examples]
    W, b, classes, mean, std = train_softmax_regression(
        X=X_train,
        labels=train_labels,
        epochs=epochs,
        learning_rate=learning_rate,
        class_weights=class_weights,
    )
    model = {
        "weights": W.tolist(),
        "bias": b.tolist(),
        "classes": classes,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "label_key": label_key,
    }
    X_valid = build_feature_matrix(valid_examples, wordnet_index, priors, embedder)
    valid_labels = [example[label_key] for example in valid_examples]
    preds, _ = predict_softmax(
        X_valid, W, b, classes, mean, std
    )
    per_label = precision_recall_f1(valid_labels, preds, classes)
    metrics = {
        "macro_f1": macro_f1(per_label),
        "per_label": per_label,
        "support": {
            label: sum(1 for value in valid_labels if value == label) for label in classes
        },
        "class_weights": class_weights,
    }
    return model, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train offline label classifiers.")
    parser.add_argument(
        "--train",
        default="data/label_classifier/training/train.jsonl",
        help="Training examples JSONL.",
    )
    parser.add_argument(
        "--valid",
        default="data/label_classifier/training/valid.jsonl",
        help="Validation examples JSONL.",
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
        "--out-dir",
        default="models/label_classifier",
        help="Output directory for model artifacts.",
    )
    parser.add_argument(
        "--embedder-bin",
        default="",
        help="Optional external embedder binary (e.g. cortext_text_embedder).",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Models directory passed to the embedder tool.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=220,
        help="Training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.2,
        help="Learning rate.",
    )
    parser.add_argument(
        "--type-class-weight",
        action="append",
        default=[],
        help="Type-class weight in the form label=weight. Repeatable.",
    )
    parser.add_argument(
        "--promotion-class-weight",
        action="append",
        default=[],
        help="Promotion-class weight in the form label=weight. Repeatable.",
    )
    args = parser.parse_args()

    train_examples = load_examples(Path(args.train))
    valid_examples = load_examples(Path(args.valid))
    if not train_examples or not valid_examples:
        raise SystemExit("Training and validation examples must both be non-empty.")

    wordnet_index = load_wordnet_index(Path(args.wordnet_index))
    priors = load_name_priors(Path(args.name_priors))
    embedder = None
    if args.embedder_bin:
        embedder = ExternalTextEmbedder(
            binary=args.embedder_bin,
            models_dir=args.models_dir,
        )
        prefetch_example_embeddings(train_examples + valid_examples, embedder)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    type_class_weights = parse_label_weights(args.type_class_weight)
    promotion_class_weights = parse_label_weights(args.promotion_class_weight)

    type_model, type_metrics = train_and_score(
        train_examples=train_examples,
        valid_examples=valid_examples,
        label_key="type_label",
        wordnet_index=wordnet_index,
        priors=priors,
        embedder=embedder,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        class_weights=type_class_weights,
    )
    promotion_model, promotion_metrics = train_and_score(
        train_examples=train_examples,
        valid_examples=valid_examples,
        label_key="promotion_label",
        wordnet_index=wordnet_index,
        priors=priors,
        embedder=embedder,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        class_weights=promotion_class_weights,
    )

    write_json(out_dir / "type_model.json", type_model)
    write_json(out_dir / "promotion_model.json", promotion_model)
    summary = {
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "embedder_bin": args.embedder_bin,
        "type_class_weights": type_class_weights,
        "promotion_class_weights": promotion_class_weights,
        "type_metrics": type_metrics,
        "promotion_metrics": promotion_metrics,
    }
    write_json(out_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
