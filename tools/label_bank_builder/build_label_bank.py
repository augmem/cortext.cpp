#!/usr/bin/env python3
"""
Build a label bank list from Hugging Face datasets.

Sources (default):
  - FSD50K (sound labels)
  - ConceptNet 5.7 (common object/concept terms)
  - Gender-by-Name (names)

Output: a plain text file (one label per line) suitable for the label-bank
embedding generator.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def _slug_ok(text: str) -> bool:
    if not text:
        return False
    if len(text) < 2 or len(text) > 40:
        return False
    if not re.match(r"^[A-Za-z][A-Za-z\\-\\'\\s]+$", text):
        return False
    return True


def _normalize_term(term: str) -> str:
    term = term.replace("_", " ").strip()
    term = re.sub(r"\\s+", " ", term)
    return term


def _take_top(counter: Counter, limit: int) -> List[str]:
    if limit <= 0:
        return []
    return [k for k, _ in counter.most_common(limit)]


def _load_fsd50k_labels(cache_dir: Optional[Path]) -> List[str]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for FSD50K labels") from exc

    api = HfApi()
    repo_id = "Fhrozen/FSD50k"
    files = api.list_repo_files(repo_id, repo_type="dataset")
    vocab = next((f for f in files if f.endswith("vocabulary.csv")), None)
    if not vocab:
        raise RuntimeError("FSD50K vocabulary.csv not found in repo")
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=vocab,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    labels: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if not rows:
            raise RuntimeError("FSD50K vocabulary.csv empty")
        # The file is headerless with: index,label,mid
        if len(rows[0]) >= 2 and rows[0][0].isdigit():
            for row in rows:
                if len(row) < 2:
                    continue
                value = row[1].strip()
                value = _normalize_term(value)
                if _slug_ok(value):
                    labels.append(value)
        else:
            # Fallback for headered CSVs
            f.seek(0)
            reader_dict = csv.DictReader(f)
            col = None
            for candidate in ("label", "display_name", "name", "class_name", "human_label"):
                if candidate in (reader_dict.fieldnames or []):
                    col = candidate
                    break
            if not col:
                raise RuntimeError("FSD50K vocabulary.csv missing label column")
            for row in reader_dict:
                value = row.get(col, "").strip()
                value = _normalize_term(value)
                if _slug_ok(value):
                    labels.append(value)
    return sorted(set(labels))


def _load_gender_by_name(
    cache_dir: Optional[Path], max_names: int, min_count: int, titlecase: bool
) -> List[str]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for Gender-by-Name") from exc

    ds = load_dataset("erickrribeiro/gender-by-name", split="train", cache_dir=str(cache_dir) if cache_dir else None)
    name_key = "Name" if "Name" in ds.column_names else "name"
    count_key = "Count" if "Count" in ds.column_names else "count"
    counter: Counter = Counter()
    for row in ds:
        name = str(row.get(name_key, "")).strip()
        if not name:
            continue
        count = int(row.get(count_key, 0))
        if count < min_count:
            continue
        if titlecase:
            name = name.capitalize()
        if _slug_ok(name):
            counter[name] += count
    return _take_top(counter, max_names)


def _extract_conceptnet_term(uri: str) -> Optional[str]:
    if not uri or not uri.startswith("http://conceptnet.io/c/en/"):
        return None
    term = uri.split("/c/en/")[1]
    term = term.split("/")[0]
    term = _normalize_term(term)
    if not _slug_ok(term):
        return None
    # limit to up to 3 tokens to keep label bank compact
    if len(term.split()) > 3:
        return None
    return term


def _load_conceptnet_terms(
    cache_dir: Optional[Path],
    max_terms: int,
    max_rows: int,
    prune_every: int,
) -> List[str]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for ConceptNet") from exc

    ds = load_dataset(
        "CleverThis/conceptnet",
        split="data",
        streaming=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    counter: Counter = Counter()
    for i, row in enumerate(ds):
        if max_rows > 0 and i >= max_rows:
            break
        subj = _extract_conceptnet_term(row.get("subject", ""))
        obj = _extract_conceptnet_term(row.get("object", ""))
        if subj:
            counter[subj] += 1
        if obj:
            counter[obj] += 1
        if prune_every > 0 and (i + 1) % prune_every == 0 and len(counter) > max_terms * 2:
            counter = Counter(dict(counter.most_common(max_terms)))
    return _take_top(counter, max_terms)


def _write_labels(path: Path, sections: Sequence[Tuple[str, Iterable[str]]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for title, labels in sections:
            f.write(f"# {title}\n")
            for label in labels:
                f.write(f"{label}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build label bank from HF datasets")
    parser.add_argument("--out", type=Path, default=Path("data/label_bank/hf_labels.txt"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--conceptnet-max", type=int, default=5000)
    parser.add_argument("--conceptnet-max-rows", type=int, default=1000000)
    parser.add_argument("--conceptnet-prune-every", type=int, default=100000)
    parser.add_argument("--names-max", type=int, default=1500)
    parser.add_argument("--names-min-count", type=int, default=50)
    parser.add_argument("--names-titlecase", action="store_true", default=True)
    parser.add_argument("--no-names-titlecase", dest="names_titlecase", action="store_false")
    parser.add_argument("--skip-fsd50k", action="store_true")
    parser.add_argument("--skip-conceptnet", action="store_true")
    parser.add_argument("--skip-names", action="store_true")
    args = parser.parse_args()

    sections: List[Tuple[str, Iterable[str]]] = []
    if not args.skip_fsd50k:
        print("Loading FSD50K labels...")
        fsd_labels = _load_fsd50k_labels(args.cache_dir)
        sections.append(("FSD50K (sounds)", fsd_labels))
    if not args.skip_conceptnet:
        print("Streaming ConceptNet...")
        concept_terms = _load_conceptnet_terms(
            args.cache_dir,
            max_terms=args.conceptnet_max,
            max_rows=args.conceptnet_max_rows,
            prune_every=args.conceptnet_prune_every,
        )
        sections.append(("ConceptNet (common terms)", concept_terms))
    if not args.skip_names:
        print("Loading Gender-by-Name...")
        name_terms = _load_gender_by_name(
            args.cache_dir,
            max_names=args.names_max,
            min_count=args.names_min_count,
            titlecase=args.names_titlecase,
        )
        sections.append(("Gender-by-Name (names)", name_terms))

    if not sections:
        print("No sections selected; nothing to write.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_labels(args.out, sections)
    print(f"Wrote label list to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
