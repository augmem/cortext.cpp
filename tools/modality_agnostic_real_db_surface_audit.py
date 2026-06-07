#!/usr/bin/env python3
"""Benchmark-only modality-agnostic graph/consolidation surface audit.

This script intentionally avoids vector-table reads so it can audit existing
real Cortext databases without loading sqlite-vec. It does not run production
consolidation, does not invoke labeling, and does not use synthetic data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def scalar(con: sqlite3.Connection, sql: str) -> int:
    try:
        row = con.execute(sql).fetchone()
    except sqlite3.Error:
        return 0
    if not row:
        return 0
    return int(row[0] or 0)


def rows(con: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    try:
        return list(con.execute(sql))
    except sqlite3.Error:
        return []


def join_counter(counter: Counter[str]) -> str:
    return "|".join(f"{key}:{counter[key]}" for key in sorted(counter))


def audit_db(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        memory_rows = rows(
            con,
            "SELECT memory_id, kind, modality, blob_id, label "
            "FROM memories ORDER BY memory_id",
        )
        signal_rows = rows(
            con,
            "SELECT signal_id, memory_id, source_id, modality, blob_id, "
            "       serial_position, write_decision "
            "FROM signals ORDER BY signal_id",
        )

        signals_by_memory: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for signal in signal_rows:
            if signal["memory_id"] is not None:
                signals_by_memory[int(signal["memory_id"])].append(signal)

        memory_kind = Counter(str(row["kind"] or "") for row in memory_rows)
        memory_modality = Counter(str(row["modality"] or "") for row in memory_rows)
        signal_modality = Counter(str(row["modality"] or "") for row in signal_rows)
        signal_source = Counter(str(row["source_id"] or "") for row in signal_rows)

        memory_units: list[dict[str, Any]] = []
        signal_units: list[dict[str, Any]] = []
        multi_modal_memory_units = 0
        memory_modality_split_units = 0
        source_split_units = 0

        for memory in memory_rows:
            memory_id = int(memory["memory_id"])
            linked = signals_by_memory.get(memory_id, [])
            modalities = sorted({str(row["modality"] or "") for row in linked})
            sources = sorted({str(row["source_id"] or "") for row in linked})
            multi_modal = len(modalities) > 1
            multi_modal_memory_units += int(multi_modal)
            memory_units.append(
                {
                    "db_path": str(path),
                    "unit_variant": "memory_row",
                    "unit_id": f"memory_{memory_id}",
                    "memory_id": memory_id,
                    "signal_ids": "|".join(str(row["signal_id"]) for row in linked),
                    "modalities": "|".join(modalities),
                    "sources": "|".join(sources),
                    "signal_count": len(linked),
                    "multi_modal": int(multi_modal),
                    "has_memory_blob": int(memory["blob_id"] is not None),
                }
            )

            by_modality: dict[str, list[sqlite3.Row]] = defaultdict(list)
            by_source: dict[str, list[sqlite3.Row]] = defaultdict(list)
            for signal in linked:
                by_modality[str(signal["modality"] or "")].append(signal)
                by_source[str(signal["source_id"] or "")].append(signal)
            for modality, group in sorted(by_modality.items()):
                memory_modality_split_units += 1
                memory_units.append(
                    {
                        "db_path": str(path),
                        "unit_variant": "memory_modality_split",
                        "unit_id": f"memory_{memory_id}_{modality}",
                        "memory_id": memory_id,
                        "signal_ids": "|".join(str(row["signal_id"]) for row in group),
                        "modalities": modality,
                        "sources": "|".join(sorted({str(row["source_id"] or "") for row in group})),
                        "signal_count": len(group),
                        "multi_modal": 0,
                        "has_memory_blob": int(memory["blob_id"] is not None),
                    }
                )
            for source, group in sorted(by_source.items()):
                source_split_units += 1
                memory_units.append(
                    {
                        "db_path": str(path),
                        "unit_variant": "memory_source_split",
                        "unit_id": f"memory_{memory_id}_{source}",
                        "memory_id": memory_id,
                        "signal_ids": "|".join(str(row["signal_id"]) for row in group),
                        "modalities": "|".join(sorted({str(row["modality"] or "") for row in group})),
                        "sources": source,
                        "signal_count": len(group),
                        "multi_modal": int(len({str(row["modality"] or "") for row in group}) > 1),
                        "has_memory_blob": int(memory["blob_id"] is not None),
                    }
                )

        for signal in signal_rows:
            signal_units.append(
                {
                    "db_path": str(path),
                    "unit_variant": "signal_row",
                    "unit_id": f"signal_{signal['signal_id']}",
                    "memory_id": int(signal["memory_id"] or 0),
                    "signal_ids": str(signal["signal_id"]),
                    "modalities": str(signal["modality"] or ""),
                    "sources": str(signal["source_id"] or ""),
                    "signal_count": 1,
                    "multi_modal": 0,
                    "has_memory_blob": 0,
                }
            )

        audit = {
            "db_path": str(path),
            "exists": path.exists(),
            "memory_count": len(memory_rows),
            "signal_count": len(signal_rows),
            "linked_signal_count": sum(1 for row in signal_rows if row["memory_id"] is not None),
            "memory_kind_distribution": dict(memory_kind),
            "memory_modality_distribution": dict(memory_modality),
            "signal_modality_distribution": dict(signal_modality),
            "signal_source_distribution": dict(signal_source),
            "memory_blob_rows": sum(1 for row in memory_rows if row["blob_id"] is not None),
            "signal_blob_rows": sum(1 for row in signal_rows if row["blob_id"] is not None),
            "runtime_label_rows": sum(1 for row in memory_rows if row["label"]),
            "fact_assertion_rows": scalar(con, "SELECT COUNT(*) FROM fact_assertions")
            if table_exists(con, "fact_assertions")
            else 0,
            "fact_evidence_rows": scalar(con, "SELECT COUNT(*) FROM fact_evidence")
            if table_exists(con, "fact_evidence")
            else 0,
            "soft_anchor_rows": scalar(con, "SELECT COUNT(*) FROM soft_anchors")
            if table_exists(con, "soft_anchors")
            else 0,
            "soft_anchor_link_rows": scalar(con, "SELECT COUNT(*) FROM soft_anchor_links")
            if table_exists(con, "soft_anchor_links")
            else 0,
            "multi_modal_memory_units": multi_modal_memory_units,
            "memory_row_units": len(memory_rows),
            "memory_modality_split_units": memory_modality_split_units,
            "memory_source_split_units": source_split_units,
            "signal_row_units": len(signal_rows),
            "real_db_audit_only": True,
            "production_path_changed": False,
            "production_consolidation_called": False,
            "synthetic_encoder_used": False,
        }
        return audit, memory_units, signal_units
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build/modality_agnostic_graph_bench")
    parser.add_argument("dbs", nargs="*")
    args = parser.parse_args()

    default_dbs = [
        "examples/chat/chat_memory.db",
        "build/video_media_perf/cortext_video_perf.sqlite",
        "build/video_media_perf/cortext_video_perf_10fps.sqlite",
    ]
    db_paths = [Path(p) for p in (args.dbs or default_dbs) if Path(p).exists()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audits: list[dict[str, Any]] = []
    memory_units: list[dict[str, Any]] = []
    signal_units: list[dict[str, Any]] = []
    for path in db_paths:
        audit, mem, sig = audit_db(path)
        audits.append(audit)
        memory_units.extend(mem)
        signal_units.extend(sig)

    summary = {
        "real_data_only": True,
        "synthetic_encoder_used": False,
        "production_path_changed": False,
        "production_consolidation_called": False,
        "db_count": len(audits),
        "audits": audits,
    }
    (out_dir / "modality_agnostic_real_db_surface_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    audit_fields = [
        "db_path",
        "memory_count",
        "signal_count",
        "linked_signal_count",
        "memory_kind_distribution",
        "memory_modality_distribution",
        "signal_modality_distribution",
        "signal_source_distribution",
        "memory_blob_rows",
        "signal_blob_rows",
        "runtime_label_rows",
        "fact_assertion_rows",
        "fact_evidence_rows",
        "soft_anchor_rows",
        "soft_anchor_link_rows",
        "multi_modal_memory_units",
        "memory_row_units",
        "memory_modality_split_units",
        "memory_source_split_units",
        "signal_row_units",
    ]
    with (out_dir / "modality_agnostic_real_db_surface_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=audit_fields)
        writer.writeheader()
        for audit in audits:
            row = dict(audit)
            for key in [
                "memory_kind_distribution",
                "memory_modality_distribution",
                "signal_modality_distribution",
                "signal_source_distribution",
            ]:
                row[key] = join_counter(Counter(row[key]))
            writer.writerow({key: row.get(key, "") for key in audit_fields})

    unit_fields = [
        "db_path",
        "unit_variant",
        "unit_id",
        "memory_id",
        "signal_ids",
        "modalities",
        "sources",
        "signal_count",
        "multi_modal",
        "has_memory_blob",
    ]
    with (out_dir / "modality_agnostic_real_db_memory_units.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=unit_fields)
        writer.writeheader()
        writer.writerows(memory_units)
    with (out_dir / "modality_agnostic_real_db_signal_units.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=unit_fields)
        writer.writeheader()
        writer.writerows(signal_units)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
