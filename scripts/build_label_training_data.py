#!/usr/bin/env python3
"""
Build weakly supervised span-typing/promotion training data for the offline
label-classifier prototype.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from label_classifier_lib import (
    generate_candidate_spans,
    iter_cortext_messages,
    load_name_priors,
    load_wordnet_index,
    normalize_text,
    split_sentences,
    weak_type_and_promotion,
)


def load_metadata_index(dataset_path: Path) -> dict[str, dict]:
    meta_path = dataset_path.with_name(f"{dataset_path.stem}.metadata.jsonl")
    if not meta_path.exists():
        return {}
    metadata: dict[str, dict] = {}
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            conversation_id = str(payload.get("conversation_id", ""))
            if conversation_id:
                metadata[conversation_id] = payload
    return metadata


def iter_messages_with_metadata(
    dataset_paths: list[Path],
) -> list[tuple[str, str, dict, dict]]:
    rows: list[tuple[str, str, dict, dict]] = []
    metadata_by_path = {path: load_metadata_index(path) for path in dataset_paths}
    for path in dataset_paths:
        if not path.exists():
            continue
        path_metadata = metadata_by_path.get(path, {})
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, list) or len(payload) != 2:
                    continue
                conversation_id = str(payload[0])
                conversation_meta = path_metadata.get(conversation_id, {})
                content = payload[1].get("content", [])
                if not isinstance(content, list):
                    continue
                utterance_labels = conversation_meta.get("utterance_labels", [])
                for index, turn in enumerate(content):
                    message = normalize_text(turn.get("message", ""))
                    if not message:
                        continue
                    turn_meta = {}
                    if index < len(utterance_labels) and isinstance(utterance_labels[index], dict):
                        turn_meta = utterance_labels[index]
                    elif conversation_meta.get("labels"):
                        turn_meta = {"labels": conversation_meta.get("labels", [])}
                    rows.append((conversation_id, message, conversation_meta, turn_meta))
                for persona in conversation_meta.get("personas", []) or []:
                    message = normalize_text(persona)
                    if message:
                        rows.append(
                            (
                                conversation_id,
                                message,
                                conversation_meta,
                                {"persona": True},
                            )
                        )
    return rows


def write_examples(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")


def load_examples(path: Path) -> list[dict]:
    examples: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def weak_chat_examples(
    dataset_paths: list[Path],
    wordnet_index: dict[str, dict],
    priors: dict[str, dict[str, float]],
    max_messages: int,
    negative_rate: float,
) -> list[dict]:
    examples: list[dict] = []
    rng = random.Random(7)
    seen = 0
    for conversation_id, message, conversation_meta, turn_meta in iter_messages_with_metadata(dataset_paths):
        if seen >= max_messages:
            break
        seen += 1
        for sentence in split_sentences(message):
            spans = generate_candidate_spans(sentence)
            if not spans:
                continue
            for span in spans:
                type_label, promotion_label = weak_type_and_promotion(
                    sentence,
                    span,
                    wordnet_index,
                    priors,
                    conversation_meta=conversation_meta,
                    turn_meta=turn_meta,
                )
                if type_label == "none" and rng.random() > negative_rate:
                    continue
                examples.append(
                    {
                        "id": f"{conversation_id}:{seen}:{span.start}:{span.end}",
                        "text": normalize_text(sentence),
                        "span": span.text,
                        "type_label": type_label,
                        "promotion_label": promotion_label,
                        "source": str(conversation_id),
                        "weak_label": True,
                    }
                )
    return examples


def split_examples(examples: list[dict], valid_fraction: float) -> tuple[list[dict], list[dict]]:
    rng = random.Random(11)
    by_label: dict[str, list[dict]] = {}
    for example in examples:
        by_label.setdefault(example["type_label"], []).append(example)
    train_examples: list[dict] = []
    valid_examples: list[dict] = []
    for bucket in by_label.values():
        rng.shuffle(bucket)
        valid_count = max(1, int(len(bucket) * valid_fraction))
        valid_examples.extend(bucket[:valid_count])
        train_examples.extend(bucket[valid_count:])
    rng.shuffle(train_examples)
    rng.shuffle(valid_examples)
    return train_examples, valid_examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Build weakly supervised label-classifier training data.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Prepared Cortext JSONL dataset path (repeatable).",
    )
    parser.add_argument(
        "--labeled-train",
        action="append",
        default=[],
        help="Existing labeled span-example JSONL to append to training data.",
    )
    parser.add_argument(
        "--labeled-valid",
        action="append",
        default=[],
        help="Existing labeled span-example JSONL to append directly to validation data.",
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
        default="data/label_classifier/training",
        help="Output directory for train/valid JSONL.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=300,
        help="Maximum prepared-chat messages to mine weak labels from.",
    )
    parser.add_argument(
        "--negative-rate",
        type=float,
        default=0.15,
        help="Sampling rate for none/ignore candidates from chat data.",
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.15,
        help="Validation split fraction.",
    )
    args = parser.parse_args()

    wordnet_index = load_wordnet_index(Path(args.wordnet_index))
    priors = load_name_priors(Path(args.name_priors))

    examples: list[dict] = []
    dataset_paths = [Path(path) for path in args.dataset]
    examples.extend(
        weak_chat_examples(
            dataset_paths=dataset_paths,
            wordnet_index=wordnet_index,
            priors=priors,
            max_messages=args.max_messages,
            negative_rate=args.negative_rate,
        )
    )

    train_examples, valid_examples = split_examples(examples, args.valid_fraction)
    for labeled_path in [Path(path) for path in args.labeled_train]:
        if labeled_path.exists():
            train_examples.extend(load_examples(labeled_path))
    for labeled_path in [Path(path) for path in args.labeled_valid]:
        if labeled_path.exists():
            valid_examples.extend(load_examples(labeled_path))
    out_dir = Path(args.out_dir)
    write_examples(out_dir / "train.jsonl", train_examples)
    write_examples(out_dir / "valid.jsonl", valid_examples)
    summary = {
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "datasets": [str(path) for path in dataset_paths],
        "labeled_train": args.labeled_train,
        "labeled_valid": args.labeled_valid,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[OK] wrote {len(train_examples)} train / {len(valid_examples)} valid examples to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
