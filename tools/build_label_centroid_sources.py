#!/usr/bin/env python3
"""Build reproducible label-centroid source records.

This script prepares the text side of a multimodal label centroid bank from
local data assets. It does not call a model. The output is a deterministic
source/prompt pack that can be fed through the current ES/AIST encoder or a
future pre-projection label-head exporter.

Default local sources:
  - data/salt.csv
  - data/english-wordnet-2025.xml
  - data/label_bank/labels.jsonl
  - data/emotion/_tmp_text_emotion/*.txt

Outputs:
  - label_centroid_source_records.jsonl
  - label_centroid_prompt_rows.jsonl
  - label_centroid_labels.csv
  - label_centroid_summary.json
  - label_centroid_repro_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_WORDNET_LEXFILES = {
    "noun.Tops": "concept",
    "noun.act": "event_action",
    "noun.animal": "entity_animal",
    "noun.artifact": "object",
    "noun.body": "body",
    "noun.communication": "communication",
    "noun.event": "event",
    "noun.food": "food",
    "noun.group": "group",
    "noun.location": "place",
    "noun.object": "object",
    "noun.person": "entity_person",
    "noun.phenomenon": "phenomenon",
    "noun.plant": "entity_plant",
    "noun.process": "process",
    "noun.substance": "substance",
    "verb.body": "action",
    "verb.change": "action",
    "verb.communication": "communication_action",
    "verb.competition": "action",
    "verb.consumption": "action",
    "verb.contact": "action",
    "verb.creation": "action",
    "verb.emotion": "affect_action",
    "verb.motion": "action",
    "verb.perception": "perception_action",
    "verb.social": "social_action",
    "verb.weather": "weather_action",
}

SALT_AUDIO_HINTS = {
    "alarm",
    "applause",
    "bark",
    "bell",
    "bird",
    "car_horn",
    "caw",
    "clap",
    "cry",
    "engine",
    "foghorn",
    "gong",
    "horn",
    "music",
    "quack",
    "ringing",
    "shout",
    "siren",
    "speech",
    "thunder",
    "typing",
    "whistle",
}


def normalize_label(value: str) -> str:
    value = value.replace("_", " ")
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"[^A-Za-z0-9' -]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def slug(value: str) -> str:
    value = normalize_label(value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def text_ok(value: str) -> bool:
    if not value:
        return False
    value = value.strip()
    if len(value) < 2 or len(value) > 160:
        return False
    if not re.search(r"[A-Za-z]", value):
        return False
    return True


def dedupe_ordered(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        clean = re.sub(r"\s+", " ", item.strip())
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def input_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
        }
    if path.is_dir():
        h = hashlib.sha256()
        file_count = 0
        total_bytes = 0
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = child.relative_to(path).as_posix()
            h.update(rel.encode("utf-8", errors="ignore"))
            h.update(b"\0")
            child_hash = sha256_file(child)
            h.update(child_hash.encode("ascii"))
            h.update(b"\0")
            file_count += 1
            total_bytes += child.stat().st_size
        return {
            "path": str(path),
            "exists": True,
            "type": "directory",
            "files": file_count,
            "bytes": total_bytes,
            "sha256": h.hexdigest(),
        }
    return {
        "path": str(path),
        "exists": True,
        "type": "file",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def salt_kind(label: str) -> str:
    label_slug = slug(label)
    tokens = set(label_slug.split("_"))
    if label_slug in SALT_AUDIO_HINTS or tokens.intersection(SALT_AUDIO_HINTS):
        return "audio_scene_event"
    if "music" in tokens:
        return "audio_scene_event"
    return "scene_event"


def salt_prompts(label: str, examples: list[str], max_examples: int) -> list[str]:
    clean = normalize_label(label)
    base = [
        clean,
        f"a memory about {clean}",
        f"a multimodal scene involving {clean}",
        f"audio or video evidence of {clean}",
    ]
    return dedupe_ordered(base + examples[:max_examples])


def load_salt(path: Path, max_examples_per_label: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    examples: dict[str, list[str]] = defaultdict(list)
    row_counts: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_id = row.get("source_id", "")
            parts = source_id.split(":")
            if len(parts) < 2 or parts[0] != "salt":
                continue
            label_key = parts[1]
            text = row.get("text", "").strip()
            row_counts[label_key] += 1
            if text_ok(text) and len(examples[label_key]) < max_examples_per_label:
                examples[label_key].append(text)

    records: list[dict[str, Any]] = []
    for label_key in sorted(row_counts):
        label = normalize_label(label_key)
        prompts = salt_prompts(label_key, examples[label_key], max_examples_per_label)
        records.append(
            {
                "label_id": f"salt:{label_key}",
                "source": "salt.csv",
                "label": label,
                "kind": salt_kind(label_key),
                "aliases": dedupe_ordered([label, label_key.replace("_", " ")]),
                "definition": "",
                "source_example_count": row_counts[label_key],
                "prompt_count": len(prompts),
                "prompts": prompts,
            }
        )
    return records


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def load_wordnet(
    path: Path,
    max_synsets: int,
    max_examples_per_synset: int,
    include_lexfiles: dict[str, str],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    root = ET.parse(path).getroot()
    lexicon = None
    for child in root:
        if local_name(child.tag) == "Lexicon":
            lexicon = child
            break
    if lexicon is None:
        return []

    lemmas_by_synset: dict[str, list[str]] = defaultdict(list)
    for elem in lexicon:
        if local_name(elem.tag) != "LexicalEntry":
            continue
        lemma = ""
        for child in elem:
            child_tag = local_name(child.tag)
            if child_tag == "Lemma":
                lemma = normalize_label(child.attrib.get("writtenForm", ""))
            elif child_tag == "Sense" and lemma:
                synset_id = child.attrib.get("synset", "")
                if synset_id:
                    lemmas_by_synset[synset_id].append(lemma)

    records: list[dict[str, Any]] = []
    for elem in lexicon:
        if local_name(elem.tag) != "Synset":
            continue
        synset_id = elem.attrib.get("id", "")
        lexfile = elem.attrib.get("lexfile", "")
        pos = elem.attrib.get("partOfSpeech", "")
        kind = include_lexfiles.get(lexfile)
        if not synset_id or not kind:
            continue
        aliases = dedupe_ordered(lemmas_by_synset.get(synset_id, []))
        if not aliases:
            continue
        label = aliases[0]
        if not text_ok(label) or len(label.split()) > 5:
            continue
        definitions = [
            child.text.strip()
            for child in elem
            if local_name(child.tag) == "Definition"
            and child.text
            and text_ok(child.text)
        ]
        examples = [
            child.text.strip()
            for child in elem
            if local_name(child.tag) == "Example"
            and child.text
            and text_ok(child.text)
        ][:max_examples_per_synset]

        prompt_seed = [
            label,
            f"a memory about {label}",
        ]
        if kind.startswith("object") or kind.startswith("entity") or kind in {
            "body",
            "food",
            "place",
            "substance",
        }:
            prompt_seed.append(f"an image or scene containing {label}")
        if "event" in kind or "action" in kind or kind in {
            "communication",
            "phenomenon",
            "process",
        }:
            prompt_seed.append(f"audio, video, or text evidence of {label}")
        if definitions:
            prompt_seed.append(f"{label}: {definitions[0]}")
        prompt_seed.extend(examples)
        prompts = dedupe_ordered(prompt_seed)

        records.append(
            {
                "label_id": f"wordnet:{synset_id}",
                "source": "english-wordnet-2025.xml",
                "label": label,
                "kind": kind,
                "wordnet_synset_id": synset_id,
                "wordnet_lexfile": lexfile,
                "wordnet_pos": pos,
                "aliases": aliases[:12],
                "definition": definitions[0] if definitions else "",
                "source_example_count": len(examples),
                "prompt_count": len(prompts),
                "prompts": prompts,
            }
        )
        if max_synsets > 0 and len(records) >= max_synsets:
            break
    return records


def load_existing_label_bank(path: Path, include_embeddings: bool) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            label = normalize_label(str(row.get("label") or row.get("key") or ""))
            key = slug(str(row.get("key") or label))
            if not key or not text_ok(label):
                continue
            prompts = dedupe_ordered([label, f"a person, place, or thing named {label}"])
            record: dict[str, Any] = {
                "label_id": f"existing_label_bank:{key}",
                "source": "data/label_bank/labels.jsonl",
                "label": label,
                "kind": "name_or_entity",
                "aliases": [label],
                "definition": "",
                "source_example_count": 1,
                "prompt_count": len(prompts),
                "prompts": prompts,
            }
            embedding = row.get("embedding")
            if isinstance(embedding, list):
                record["existing_embedding_dim"] = len(embedding)
                if include_embeddings:
                    record["existing_embedding"] = embedding
            records.append(record)
    records.sort(key=lambda r: (r["label"], r["label_id"]))
    return records


def load_text_emotion_dir(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for txt_path in sorted(path.glob("*.txt")):
        label = normalize_label(txt_path.stem)
        if not text_ok(label):
            continue
        prompts = [label, f"text expressing {label}"]
        with txt_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                text = line.strip()
                if text_ok(text):
                    prompts.append(text)
        prompts = dedupe_ordered(prompts)
        if len(prompts) <= 2:
            continue
        records.append(
            {
                "label_id": f"text_emotion:{slug(label)}",
                "source": "data/emotion/_tmp_text_emotion",
                "label": label,
                "kind": "text_emotion",
                "aliases": [label],
                "definition": "",
                "source_example_count": len(prompts) - 2,
                "prompt_count": len(prompts),
                "prompts": prompts,
            }
        )
    return records


def merge_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: (r["source"], r["kind"], r["label_id"])):
        label_id = record["label_id"]
        if label_id in seen_ids:
            continue
        seen_ids.add(label_id)
        merged.append(record)
    return merged


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def build_prompt_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for i, prompt in enumerate(record.get("prompts", [])):
            rows.append(
                {
                    "label_id": record["label_id"],
                    "prompt_id": f"{record['label_id']}#p{i:03d}",
                    "source": record["source"],
                    "kind": record["kind"],
                    "label": record["label"],
                    "prompt": prompt,
                }
            )
    return rows


def write_labels_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label_id",
        "source",
        "kind",
        "label",
        "alias_count",
        "source_example_count",
        "prompt_count",
        "definition",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "label_id": record["label_id"],
                    "source": record["source"],
                    "kind": record["kind"],
                    "label": record["label"],
                    "alias_count": len(record.get("aliases", [])),
                    "source_example_count": record.get("source_example_count", 0),
                    "prompt_count": record.get("prompt_count", 0),
                    "definition": record.get("definition", ""),
                }
            )


def write_summary(
    path: Path,
    records: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    by_source = Counter(record["source"] for record in records)
    by_kind = Counter(record["kind"] for record in records)
    prompt_by_source = Counter(row["source"] for row in prompt_rows)
    summary = {
        "record_count": len(records),
        "prompt_row_count": len(prompt_rows),
        "sources": dict(sorted(by_source.items())),
        "kinds": dict(sorted(by_kind.items())),
        "prompt_rows_by_source": dict(sorted(prompt_by_source.items())),
        "salt_csv": input_meta(args.salt_csv),
        "wordnet_xml": input_meta(args.wordnet_xml),
        "existing_label_bank": input_meta(args.existing_label_bank),
        "text_emotion_dir": input_meta(args.text_emotion_dir),
        "wordnet_max_synsets": args.wordnet_max_synsets,
        "salt_max_examples_per_label": args.salt_max_examples_per_label,
        "wordnet_max_examples_per_synset": args.wordnet_max_examples_per_synset,
        "include_existing_embeddings": args.include_existing_embeddings,
    }
    path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def write_manifest(path: Path, args: argparse.Namespace) -> None:
    manifest = {
        "script": "tools/build_label_centroid_sources.py",
        "purpose": "Build deterministic text/prompt source rows for multimodal label centroid encoding.",
        "inputs": {
            "salt_csv": input_meta(args.salt_csv),
            "wordnet_xml": input_meta(args.wordnet_xml),
            "existing_label_bank": input_meta(args.existing_label_bank),
            "text_emotion_dir": input_meta(args.text_emotion_dir),
        },
        "outputs": {
            "source_records": "label_centroid_source_records.jsonl",
            "prompt_rows": "label_centroid_prompt_rows.jsonl",
            "labels_csv": "label_centroid_labels.csv",
            "summary": "label_centroid_summary.json",
        },
        "reproduce": {
            "command": (
                "python3 tools/build_label_centroid_sources.py "
                f"--salt-csv {args.salt_csv} "
                f"--wordnet-xml {args.wordnet_xml} "
                f"--existing-label-bank {args.existing_label_bank} "
                f"--text-emotion-dir {args.text_emotion_dir} "
                f"--out-dir {args.out_dir} "
                f"--wordnet-max-synsets {args.wordnet_max_synsets} "
                f"--salt-max-examples-per-label {args.salt_max_examples_per_label} "
                f"--wordnet-max-examples-per-synset {args.wordnet_max_examples_per_synset}"
            ),
            "next_step": (
                "Encode label_centroid_prompt_rows.jsonl with ES/AIST or a "
                "pre-projection exporter, then average normalized embeddings by "
                "label_id to form semantic/entity/event centroid rows."
            ),
        },
    }
    path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salt-csv", type=Path, default=Path("data/salt.csv"))
    parser.add_argument(
        "--wordnet-xml",
        type=Path,
        default=Path("data/english-wordnet-2025.xml"),
    )
    parser.add_argument(
        "--existing-label-bank",
        type=Path,
        default=Path("data/label_bank/labels.jsonl"),
    )
    parser.add_argument(
        "--text-emotion-dir",
        type=Path,
        default=Path("data/emotion/_tmp_text_emotion"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/label_centroid_sources"),
    )
    parser.add_argument(
        "--wordnet-max-synsets",
        type=int,
        default=0,
        help="Maximum WordNet synsets to emit; 0 means all filtered synsets.",
    )
    parser.add_argument(
        "--salt-max-examples-per-label",
        type=int,
        default=50,
        help="Maximum SALT captions retained as prompts for each label.",
    )
    parser.add_argument(
        "--wordnet-max-examples-per-synset",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--skip-salt",
        action="store_true",
        help="Do not include data/salt.csv labels.",
    )
    parser.add_argument(
        "--skip-wordnet",
        action="store_true",
        help="Do not include WordNet synsets.",
    )
    parser.add_argument(
        "--skip-existing-label-bank",
        action="store_true",
        help="Do not include data/label_bank/labels.jsonl labels.",
    )
    parser.add_argument(
        "--skip-text-emotion",
        action="store_true",
        help="Do not include data/emotion/_tmp_text_emotion labels.",
    )
    parser.add_argument(
        "--include-existing-embeddings",
        action="store_true",
        help="Copy existing 256d label-bank embeddings into source records.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    if not args.skip_salt:
        records.extend(load_salt(args.salt_csv, args.salt_max_examples_per_label))
    if not args.skip_wordnet:
        records.extend(
            load_wordnet(
                args.wordnet_xml,
                args.wordnet_max_synsets,
                args.wordnet_max_examples_per_synset,
                DEFAULT_WORDNET_LEXFILES,
            )
        )
    if not args.skip_existing_label_bank:
        records.extend(
            load_existing_label_bank(
                args.existing_label_bank, args.include_existing_embeddings
            )
        )
    if not args.skip_text_emotion:
        records.extend(load_text_emotion_dir(args.text_emotion_dir))

    records = merge_records(records)
    prompt_rows = build_prompt_rows(records)

    write_jsonl(args.out_dir / "label_centroid_source_records.jsonl", records)
    write_jsonl(args.out_dir / "label_centroid_prompt_rows.jsonl", prompt_rows)
    write_labels_csv(args.out_dir / "label_centroid_labels.csv", records)
    write_summary(args.out_dir / "label_centroid_summary.json", records, prompt_rows, args)
    write_manifest(args.out_dir / "label_centroid_repro_manifest.json", args)

    print(
        f"wrote {len(records)} label records and {len(prompt_rows)} prompt rows "
        f"to {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
