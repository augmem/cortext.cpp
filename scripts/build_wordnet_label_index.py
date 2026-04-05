#!/usr/bin/env python3
"""
Build a compact lemma index from Open English WordNet for label-classifier
training. The output is a JSON file keyed by normalized lemma.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def normalize_key(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def build_index(source: Path) -> dict:
    lemma_to_synsets: dict[str, list[str]] = defaultdict(list)
    lemma_to_pos: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    synset_meta: dict[str, dict[str, str]] = {}

    for _, elem in ET.iterparse(source, events=("end",)):
        tag = local_name(elem.tag)
        if tag == "LexicalEntry":
            lemma = ""
            pos = ""
            synsets: list[str] = []
            for child in elem:
                child_tag = local_name(child.tag)
                if child_tag == "Lemma":
                    lemma = normalize_key(child.attrib.get("writtenForm", ""))
                    pos = child.attrib.get("partOfSpeech", "")
                elif child_tag == "Sense":
                    synset_id = child.attrib.get("synset", "")
                    if synset_id:
                        synsets.append(synset_id)
            if lemma:
                if pos:
                    lemma_to_pos[lemma][pos] += 1
                for synset_id in synsets:
                    lemma_to_synsets[lemma].append(synset_id)
            elem.clear()
        elif tag == "Synset":
            synset_id = elem.attrib.get("id", "")
            if synset_id:
                synset_meta[synset_id] = {
                    "pos": elem.attrib.get("partOfSpeech", ""),
                    "lexfile": elem.attrib.get("lexfile", ""),
                }
            elem.clear()

    lemmas: dict[str, dict] = {}
    for lemma, synsets in lemma_to_synsets.items():
        lexfiles: dict[str, int] = defaultdict(int)
        for synset_id in synsets:
            meta = synset_meta.get(synset_id)
            if not meta:
                continue
            lexfile = meta.get("lexfile", "")
            if lexfile:
                lexfiles[lexfile] += 1
        lemmas[lemma] = {
            "synset_count": len(synsets),
            "pos": dict(sorted(lemma_to_pos[lemma].items())),
            "lexfiles": dict(sorted(lexfiles.items())),
        }
    return {
        "source": str(source),
        "lemma_count": len(lemmas),
        "lemmas": lemmas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact WordNet label index.")
    parser.add_argument(
        "--source",
        default="english-wordnet-2025.xml",
        help="Path to the Open English WordNet XML asset.",
    )
    parser.add_argument(
        "--out",
        default="data/label_classifier/wordnet_index.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Missing WordNet source at {source}")
    payload = build_index(source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[OK] wrote {payload['lemma_count']} lemmas to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
