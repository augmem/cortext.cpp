#!/usr/bin/env python3
"""Build a single unified trainer dataset from Cortext label sources.

The output is one row-oriented table intended for model training pipelines.
When pyarrow is installed, the primary artifact is Parquet. A JSONL fallback is
available for environments without Parquet support.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


def OptionalImportPyarrow() -> tuple[Any | None, Any | None]:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        return pa, pq
    except ModuleNotFoundError:
        return None, None


def StableId(source: str, record_type: str, source_path: str, source_row: int) -> str:
    # Keep this cheap for the 33M-row name tables. The tuple is stable within the
    # source bundle and avoids hashing every row.
    return f"{source}:{record_type}:{source_path}:{source_row}"


def JsonDump(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def BaseRow(
    *,
    source: str,
    record_type: str,
    source_path: str,
    source_row: int,
    text: str = "",
    label: str = "",
    key: str = "",
    category: str = "",
    subtype: str = "",
    language: str = "",
    gender: str = "",
    country: str = "",
    count: int | None = None,
    source_id: str = "",
    text_path: str = "",
    image_path: str = "",
    audio_path: str = "",
    wordnet_id: str = "",
    wordnet_pos: str = "",
    definition: str = "",
    embedding: list[float] | None = None,
    metadata: Any = None,
) -> dict[str, Any]:
    row = {
        "record_id": StableId(source, record_type, source_path, source_row),
        "source": source,
        "record_type": record_type,
        "source_path": source_path,
        "source_row": source_row,
        "text": text,
        "label": label,
        "key": key,
        "category": category,
        "subtype": subtype,
        "language": language,
        "gender": gender,
        "country": country,
        "count": count,
        "source_id": source_id,
        "text_path": text_path,
        "image_path": image_path,
        "audio_path": audio_path,
        "wordnet_id": wordnet_id,
        "wordnet_pos": wordnet_pos,
        "definition": definition,
        "embedding": embedding,
        "metadata_json": JsonDump(metadata),
    }
    return row


def ParseInt(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def LoadJson(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def RowsFromLabelBank(repo_root: Path) -> Iterator[dict[str, Any]]:
    path = repo_root / "data/label_bank/labels.jsonl"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                label = str(record.get("label") or record.get("key") or "")
                embedding = record.get("embedding")
                if embedding is not None:
                    embedding = [float(x) for x in embedding]
                yield BaseRow(
                    source="label_bank",
                    record_type="label_embedding",
                    source_path="data/label_bank/labels.jsonl",
                    source_row=idx,
                    text=label,
                    label=label,
                    key=str(record.get("key") or label).lower(),
                    embedding=embedding,
                    metadata={k: v for k, v in record.items() if k != "embedding"},
                )

    hf_path = repo_root / "data/label_bank/hf_labels.txt"
    if hf_path.exists():
        with hf_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                label = line.strip()
                if not label:
                    continue
                yield BaseRow(
                    source="label_bank",
                    record_type="label_text",
                    source_path="data/label_bank/hf_labels.txt",
                    source_row=idx,
                    text=label,
                    label=label,
                    key=label.lower(),
                )


def RowsFromLabelClassifier(repo_root: Path) -> Iterator[dict[str, Any]]:
    gold_path = repo_root / "data/label_classifier/gold.jsonl"
    if gold_path.exists():
        with gold_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                span = str(record.get("span") or "")
                text = str(record.get("text") or span)
                yield BaseRow(
                    source="label_classifier",
                    record_type="gold_span",
                    source_path="data/label_classifier/gold.jsonl",
                    source_row=idx,
                    text=text,
                    label=span,
                    key=span.lower(),
                    category=str(record.get("type_label") or ""),
                    subtype=str(record.get("promotion_label") or ""),
                    metadata=record,
                )

    priors_path = repo_root / "data/label_classifier/name_priors.json"
    if priors_path.exists():
        priors = LoadJson(priors_path)
        row = 0
        for prior_type in ("given", "surname"):
            values = priors.get(prior_type, {})
            if not isinstance(values, dict):
                continue
            for key, score in values.items():
                label = str(key)
                yield BaseRow(
                    source="label_classifier",
                    record_type="name_prior",
                    source_path="data/label_classifier/name_priors.json",
                    source_row=row,
                    text=label,
                    label=label,
                    key=label.lower(),
                    category=prior_type,
                    metadata={"prior_score": score},
                )
                row += 1


def RowsFromSalt(repo_root: Path) -> Iterator[dict[str, Any]]:
    path = repo_root / "data/salt.csv"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            text = str(row.get("text") or "")
            yield BaseRow(
                source="salt",
                record_type="multimodal_metadata",
                source_path="data/salt.csv",
                source_row=idx,
                text=text,
                label=text,
                key=str(row.get("id") or ""),
                source_id=str(row.get("source_id") or ""),
                text_path=str(row.get("text_path") or ""),
                image_path=str(row.get("image_path") or ""),
                audio_path=str(row.get("audio_path") or ""),
            )


def RowsFromNames(repo_root: Path) -> Iterator[dict[str, Any]]:
    for source_name, rel_path, name_field, record_type in [
        ("names", "data/surnames/forenames.csv", "forename", "forename"),
        ("names", "data/surnames/surnames.csv", "surname", "surname"),
        ("names", "data/raw/names/census_surnames.csv", "surname", "census_surname"),
    ]:
        path = repo_root / rel_path
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                label = str(row.get(name_field) or row.get("name") or "").strip()
                if not label:
                    continue
                yield BaseRow(
                    source=source_name,
                    record_type=record_type,
                    source_path=rel_path,
                    source_row=idx,
                    text=label,
                    label=label,
                    key=label.lower(),
                    gender=str(row.get("gender") or ""),
                    country=str(row.get("country") or ""),
                    count=ParseInt(row.get("count")),
                )


def TextOfChild(elem: ET.Element, tag: str) -> str:
    child = elem.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def RowsFromWordnet(repo_root: Path) -> Iterator[dict[str, Any]]:
    path = repo_root / "data/english-wordnet-2025.xml"
    if not path.exists():
        return
    lexical_row = 0
    synset_row = 0
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "LexicalEntry":
            lemma = elem.find("Lemma")
            written = ""
            pos = ""
            if lemma is not None:
                written = str(lemma.attrib.get("writtenForm") or "")
                pos = str(lemma.attrib.get("partOfSpeech") or "")
            sense_ids = []
            synsets = []
            for sense in elem.findall("Sense"):
                if sense.attrib.get("id"):
                    sense_ids.append(sense.attrib["id"])
                if sense.attrib.get("synset"):
                    synsets.append(sense.attrib["synset"])
            yield BaseRow(
                source="wordnet",
                record_type="lexical_entry",
                source_path="data/english-wordnet-2025.xml",
                source_row=lexical_row,
                text=written,
                label=written,
                key=written.lower(),
                wordnet_id=str(elem.attrib.get("id") or ""),
                wordnet_pos=pos,
                metadata={
                    "entry_id": elem.attrib.get("id"),
                    "sense_ids": sense_ids,
                    "synset_ids": synsets,
                },
            )
            lexical_row += 1
            elem.clear()
        elif elem.tag == "Synset":
            definition = TextOfChild(elem, "Definition")
            examples = [child.text.strip() for child in elem.findall("Example") if child.text]
            relations = [
                {"relType": child.attrib.get("relType"), "target": child.attrib.get("target")}
                for child in elem.findall("SynsetRelation")
            ]
            members = str(elem.attrib.get("members") or "")
            label = members.split(" ")[0] if members else str(elem.attrib.get("id") or "")
            yield BaseRow(
                source="wordnet",
                record_type="synset",
                source_path="data/english-wordnet-2025.xml",
                source_row=synset_row,
                text=definition or label,
                label=label,
                key=label.lower(),
                wordnet_id=str(elem.attrib.get("id") or ""),
                wordnet_pos=str(elem.attrib.get("partOfSpeech") or ""),
                definition=definition,
                metadata={
                    "ili": elem.attrib.get("ili"),
                    "members": members,
                    "lexfile": elem.attrib.get("lexfile"),
                    "examples": examples,
                    "relations": relations,
                },
            )
            synset_row += 1
            elem.clear()


def AllRows(repo_root: Path) -> Iterator[dict[str, Any]]:
    yield from RowsFromLabelBank(repo_root)
    yield from RowsFromLabelClassifier(repo_root)
    yield from RowsFromWordnet(repo_root)
    yield from RowsFromSalt(repo_root)
    yield from RowsFromNames(repo_root)


def ParquetSchema(pa: Any) -> Any:
    return pa.schema(
        [
            ("record_id", pa.string()),
            ("source", pa.string()),
            ("record_type", pa.string()),
            ("source_path", pa.string()),
            ("source_row", pa.int64()),
            ("text", pa.string()),
            ("label", pa.string()),
            ("key", pa.string()),
            ("category", pa.string()),
            ("subtype", pa.string()),
            ("language", pa.string()),
            ("gender", pa.string()),
            ("country", pa.string()),
            ("count", pa.int64()),
            ("source_id", pa.string()),
            ("text_path", pa.string()),
            ("image_path", pa.string()),
            ("audio_path", pa.string()),
            ("wordnet_id", pa.string()),
            ("wordnet_pos", pa.string()),
            ("definition", pa.string()),
            ("embedding", pa.list_(pa.float32())),
            ("metadata_json", pa.string()),
        ]
    )


def WriteParquet(rows: Iterable[dict[str, Any]], output_path: Path, batch_size: int) -> dict[str, Any]:
    pa, pq = OptionalImportPyarrow()
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required for Parquet output")
    schema = ParquetSchema(pa)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    writer = pq.ParquetWriter(
        output_path,
        schema,
        compression="zstd",
        compression_level=10,
        use_dictionary=True,
        write_statistics=True,
    )
    stats: Counter[str] = Counter()
    count = 0
    batch: list[dict[str, Any]] = []
    try:
        for row in rows:
            batch.append(row)
            stats[f"source:{row['source']}"] += 1
            stats[f"type:{row['record_type']}"] += 1
            count += 1
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch, schema=schema)
                writer.write_table(table)
                batch.clear()
        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            writer.write_table(table)
    finally:
        writer.close()
    return {"row_count": count, "stats": dict(stats)}


def WriteJsonl(rows: Iterable[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats[f"source:{row['source']}"] += 1
            stats[f"type:{row['record_type']}"] += 1
            count += 1
    return {"row_count": count, "stats": dict(stats)}


def FileSha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def WriteSummary(output_path: Path, dataset_path: Path, result: dict[str, Any], repo_root: Path) -> None:
    summary = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_bytes": dataset_path.stat().st_size,
        "dataset_sha256": FileSha256(dataset_path),
        "row_count": result["row_count"],
        "stats": result["stats"],
        "schema": [
            "record_id",
            "source",
            "record_type",
            "source_path",
            "source_row",
            "text",
            "label",
            "key",
            "category",
            "subtype",
            "language",
            "gender",
            "country",
            "count",
            "source_id",
            "text_path",
            "image_path",
            "audio_path",
            "wordnet_id",
            "wordnet_pos",
            "definition",
            "embedding",
            "metadata_json",
        ],
        "sources": [
            "data/label_bank/labels.jsonl",
            "data/label_bank/hf_labels.txt",
            "data/label_classifier/gold.jsonl",
            "data/label_classifier/name_priors.json",
            "data/english-wordnet-2025.xml",
            "data/salt.csv",
            "data/surnames/forenames.csv",
            "data/surnames/surnames.csv",
            "data/raw/names/census_surnames.csv",
        ],
        "repo_root": str(repo_root),
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def Main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Build one unified model-trainer dataset.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("build/model_trainer_unified_dataset"))
    parser.add_argument("--output-name", default="cortext_model_trainer_sources_v1.parquet")
    parser.add_argument("--format", choices=["parquet", "jsonl"], default="parquet")
    parser.add_argument("--batch-size", type=int, default=250_000)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / args.output_name
    if args.format == "jsonl" and dataset_path.suffix != ".jsonl":
        dataset_path = dataset_path.with_suffix(".jsonl")

    rows = AllRows(repo_root)
    if args.format == "parquet":
        result = WriteParquet(rows, dataset_path, args.batch_size)
    else:
        result = WriteJsonl(rows, dataset_path)
    summary_path = output_dir / f"{dataset_path.stem}_summary.json"
    WriteSummary(summary_path, dataset_path, result, repo_root)
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "summary": str(summary_path),
                "row_count": result["row_count"],
                "bytes": dataset_path.stat().st_size,
                "sha256": FileSha256(dataset_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(Main(sys.argv[1:]))
