#!/usr/bin/env python3
"""
Download WNUT17 raw CoNLL files and convert them into span-label examples for
the offline label-classifier pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from label_classifier_lib import generate_candidate_spans, normalize_key


URLS = {
    "train": "https://raw.githubusercontent.com/halolimat/NER-WNUT17/master/data/emerging.train.conll",
    "valid": "https://raw.githubusercontent.com/halolimat/NER-WNUT17/master/data/emerging.dev.conll",
    "test": "https://raw.githubusercontent.com/halolimat/NER-WNUT17/master/data/emerging.test.conll",
}

LABEL_MAP = {
    "person": ("person_entity", "durable"),
    "location": ("place", "durable"),
    "corporation": ("org_project", "durable"),
    "group": ("org_project", "durable"),
    "creative-work": ("org_project", "durable"),
    "product": ("org_project", "durable"),
}


@dataclass
class TokenTag:
    token: str
    tag: str


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response, dest.open("wb") as handle:
        handle.write(response.read())


def iter_sentences(path: Path) -> list[list[TokenTag]]:
    sentences: list[list[TokenTag]] = []
    current: list[TokenTag] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
            if len(parts) != 2:
                continue
            token, tag = parts
            current.append(TokenTag(token=token, tag=tag))
    if current:
        sentences.append(current)
    return sentences


def detokenize(tokens: list[str]) -> str:
    text = ""
    for token in tokens:
        if not text:
            text = token
            continue
        if token in {".", ",", "!", "?", ";", ":", "%", "'s", "n't", "'re", "'ve", "'ll", "'d", "'m"}:
            text += token
        elif token in {"´", "`"}:
            text += token
        elif token.startswith("'"):
            text += token
        else:
            text += " " + token
    return text


def extract_positive_spans(tokens: list[TokenTag], text: str) -> list[dict]:
    positives: list[dict] = []
    words = [token.token for token in tokens]
    cursor = 0
    offsets: list[tuple[int, int]] = []
    rebuilt = ""
    for token in words:
        if not rebuilt:
            rebuilt = token
            offsets.append((0, len(token)))
            continue
        if token in {".", ",", "!", "?", ";", ":", "%", "'s", "n't", "'re", "'ve", "'ll", "'d", "'m"} or token.startswith("'") or token in {"´", "`"}:
            start = len(rebuilt)
            rebuilt += token
            offsets.append((start, len(rebuilt)))
        else:
            start = len(rebuilt) + 1
            rebuilt += " " + token
            offsets.append((start, len(rebuilt)))
    idx = 0
    while idx < len(tokens):
        tag = tokens[idx].tag
        if not tag.startswith("B-"):
            idx += 1
            continue
        entity = tag[2:]
        end = idx + 1
        while end < len(tokens) and tokens[end].tag == f"I-{entity}":
            end += 1
        start_char = offsets[idx][0]
        end_char = offsets[end - 1][1]
        mapped = LABEL_MAP.get(entity)
        if mapped:
            positives.append(
                {
                    "text": text,
                    "span": text[start_char:end_char],
                    "type_label": mapped[0],
                    "promotion_label": mapped[1],
                    "weak_label": False,
                }
            )
        idx = end
    return positives


def add_negatives(
    positives: list[dict], text: str, negative_rate: float, max_negatives: int, rng: random.Random
) -> list[dict]:
    negatives: list[dict] = []
    for span in generate_candidate_spans(text):
        norm = normalize_key(span.text)
        if not norm:
            continue
        if any(norm == normalize_key(example["span"]) for example in positives):
            continue
        if rng.random() > negative_rate:
            continue
        negatives.append(
            {
                "text": text,
                "span": span.text,
                "type_label": "none",
                "promotion_label": "ignore",
                "weak_label": False,
            }
        )
        if len(negatives) >= max_negatives:
            break
    return negatives


def convert_split(raw_path: Path, out_path: Path, negative_rate: float, max_negatives: int) -> int:
    rng = random.Random(17)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for sentence_tokens in iter_sentences(raw_path):
            text = detokenize([token.token for token in sentence_tokens])
            positives = extract_positive_spans(sentence_tokens, text)
            negatives = add_negatives(positives, text, negative_rate, max_negatives, rng)
            for idx, example in enumerate(positives + negatives):
                payload = {
                    "id": f"{raw_path.stem}:{count}:{idx}",
                    "source": "wnut17",
                    **example,
                }
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare WNUT17 span-label data.")
    parser.add_argument(
        "--raw-dir",
        default="data/raw/wnut17",
        help="Directory for downloaded WNUT17 raw files.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/wnut17_labels",
        help="Output directory for converted JSONL examples.",
    )
    parser.add_argument(
        "--negative-rate",
        type=float,
        default=0.08,
        help="Sampling rate for none/ignore spans.",
    )
    parser.add_argument(
        "--max-negatives-per-sentence",
        type=int,
        default=4,
        help="Cap on sampled none/ignore spans per sentence.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split, url in URLS.items():
        raw_path = raw_dir / f"{split}.conll"
        download(url, raw_path)
        example_count = convert_split(
            raw_path,
            out_dir / f"{split}.jsonl",
            args.negative_rate,
            args.max_negatives_per_sentence,
        )
        summary[split] = {"raw": str(raw_path), "examples": example_count}
        print(f"[OK] {split}: wrote {example_count} sentences")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
