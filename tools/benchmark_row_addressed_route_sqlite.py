#!/usr/bin/env python3
"""Measure an incrementally writable SQLite representation of sparse routes.

The input is a private packed-route feasibility database. Output contains only
aggregate timings, counts, and content digests; it never contains corpus text,
source identifiers, local paths, or embedding values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import statistics
import struct
import time
from pathlib import Path
from typing import Iterable

import numpy as np


ROUTE_CAPACITY = 512
EDGE_CAPACITY = 128
DIMENSION = 256


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def create_schema(database: sqlite3.Connection) -> None:
    database.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE nodes(
          label INTEGER PRIMARY KEY,
          identity INTEGER NOT NULL,
          embedding BLOB NOT NULL,
          embedding_hash INTEGER NOT NULL
        );
        CREATE INDEX nodes_hash_identity
          ON nodes(embedding_hash, identity);
        CREATE TABLE edges(
          graph INTEGER NOT NULL,
          label INTEGER NOT NULL,
          neighbors BLOB NOT NULL,
          PRIMARY KEY(graph, label)
        ) WITHOUT ROWID;
        CREATE TABLE routes(
          label INTEGER PRIMARY KEY,
          neighbors BLOB NOT NULL
        );
        CREATE TABLE anchors(
          ordinal INTEGER PRIMARY KEY,
          label INTEGER NOT NULL,
          centroid BLOB NOT NULL
        );
        CREATE TABLE meta(
          key TEXT PRIMARY KEY,
          value INTEGER NOT NULL
        ) WITHOUT ROWID;
        """
    )


def route_blob(packed: bytes, label: int) -> bytes:
    stride = (ROUTE_CAPACITY + 1) * 4
    start = label * stride
    return packed[start : start + stride]


def unpack_route(blob: bytes) -> tuple[int, ...]:
    values = struct.unpack(f"<{ROUTE_CAPACITY + 1}I", blob)
    count = values[0]
    if count > ROUTE_CAPACITY:
        raise ValueError("packed route exceeds fixed capacity")
    return values[1 : count + 1]


def copy_prefix(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    packed_routes: bytes,
    count: int,
) -> None:
    target.execute("BEGIN IMMEDIATE")
    target.executemany(
        "INSERT INTO nodes VALUES(?,?,?,?)",
        source.execute(
            "SELECT label,identity,embedding,embedding_hash FROM nodes "
            "WHERE label < ? ORDER BY label",
            (count,),
        ),
    )
    target.executemany(
        "INSERT INTO edges VALUES(?,?,?)",
        source.execute(
            "SELECT graph,label,neighbors FROM edges WHERE label < ? "
            "ORDER BY graph,label",
            (count,),
        ),
    )
    target.executemany(
        "INSERT INTO routes VALUES(?,?)",
        ((label, route_blob(packed_routes, label)) for label in range(count)),
    )
    target.executemany(
        "INSERT INTO anchors VALUES(?,?,?)",
        source.execute("SELECT ordinal,label,centroid FROM anchors ORDER BY ordinal"),
    )
    target.executemany(
        "INSERT INTO meta VALUES(?,?)",
        source.execute("SELECT key,value FROM meta ORDER BY key"),
    )
    target.execute(
        "INSERT OR REPLACE INTO meta VALUES('sealed_count',?)", (count,)
    )
    target.commit()


def rows_for_range(
    source: sqlite3.Connection,
    packed_routes: bytes,
    begin: int,
    end: int,
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    nodes = list(
        source.execute(
            "SELECT label,identity,embedding,embedding_hash FROM nodes "
            "WHERE label >= ? AND label < ? ORDER BY label",
            (begin, end),
        )
    )
    edges = list(
        source.execute(
            "SELECT graph,label,neighbors FROM edges "
            "WHERE label >= ? AND label < ? ORDER BY graph,label",
            (begin, end),
        )
    )
    routes = [
        (label, route_blob(packed_routes, label)) for label in range(begin, end)
    ]
    if len(nodes) != end - begin or len(edges) != 2 * (end - begin):
        raise ValueError("source graph range is incomplete")
    return nodes, edges, routes


def reciprocal_updates(
    source: sqlite3.Connection,
    packed_routes: bytes,
    begin: int,
    end: int,
    proposals_per_embedding: int,
) -> list[tuple[bytes, int]]:
    updates: dict[int, bytes] = {}
    for label in range(begin, end):
        proposed = unpack_route(route_blob(packed_routes, label))[
            :proposals_per_embedding
        ]
        for target in proposed:
            if target >= begin:
                continue
            updates[target] = route_blob(packed_routes, target)
    return [(blob, label) for label, blob in sorted(updates.items())]


def apply_epoch(
    target: sqlite3.Connection,
    nodes: list[tuple],
    edges: list[tuple],
    routes: list[tuple],
    reciprocal: list[tuple[bytes, int]],
    sealed_count: int,
    failure_stage: str | None = None,
) -> float:
    started = time.perf_counter()
    target.execute("BEGIN IMMEDIATE")
    try:
        target.executemany("INSERT INTO nodes VALUES(?,?,?,?)", nodes)
        target.executemany("INSERT INTO edges VALUES(?,?,?)", edges)
        target.executemany("INSERT INTO routes VALUES(?,?)", routes)
        target.executemany(
            "UPDATE routes SET neighbors=? WHERE label=?", reciprocal
        )
        target.execute(
            "INSERT OR REPLACE INTO meta VALUES('sealed_count',?)",
            (sealed_count,),
        )
        if failure_stage == "before-commit":
            raise RuntimeError("injected before derived commit")
        target.commit()
    except Exception:
        target.rollback()
        raise
    if failure_stage == "after-commit":
        raise RuntimeError("injected after derived commit")
    return time.perf_counter() - started


def restart_rows(database: sqlite3.Connection) -> int:
    rows = 0
    dimension = database.execute(
        "SELECT value FROM meta WHERE key='dimension'"
    ).fetchone()
    if dimension != (DIMENSION,):
        raise ValueError("route dimension mismatch")
    rows += 1
    anchors = database.execute(
        "SELECT label,centroid FROM anchors ORDER BY ordinal"
    ).fetchall()
    if len(anchors) != 1024:
        raise ValueError("route anchor count mismatch")
    for _, centroid in anchors:
        if len(centroid) != DIMENSION * 4:
            raise ValueError("route anchor dimension mismatch")
    rows += len(anchors)
    return rows


def query_route(database: sqlite3.Connection, label: int) -> tuple[list[int], int]:
    row = database.execute(
        "SELECT n.embedding,r.neighbors FROM nodes n JOIN routes r "
        "ON r.label=n.label WHERE n.label=?",
        (label,),
    ).fetchone()
    if row is None:
        raise ValueError("missing route query label")
    query = np.frombuffer(row[0], dtype=np.float32)
    candidates = unpack_route(row[1])
    if not candidates:
        raise ValueError("empty route")
    placeholders = ",".join("?" for _ in candidates)
    rows = database.execute(
        f"SELECT label,identity,embedding FROM nodes WHERE label IN ({placeholders})",
        candidates,
    ).fetchall()
    if len(rows) != len(set(candidates)):
        raise ValueError("route refers to an absent node")
    matrix = np.vstack([np.frombuffer(row[2], dtype=np.float32) for row in rows])
    distances = 1.0 - matrix @ query
    ranked = sorted(
        zip(distances.tolist(), (row[1] for row in rows), (row[0] for row in rows))
    )
    return [label for _, _, label in ranked[:16]], len(rows)


def measure_queries(
    database: sqlite3.Connection, count: int, query_count: int
) -> dict[str, float | int | str]:
    labels = [index * (count - 1) // (query_count - 1) for index in range(query_count)]
    digest = hashlib.sha256()
    comparisons: list[int] = []
    started = time.perf_counter()
    for label in labels:
        ranked, current = query_route(database, label)
        comparisons.append(current)
        digest.update(struct.pack(f"<{len(ranked)}q", *ranked))
    elapsed = time.perf_counter() - started
    return {
        "query_count": query_count,
        "seconds": elapsed,
        "mean_ms": elapsed * 1000.0 / query_count,
        "mean_comparisons": statistics.fmean(comparisons),
        "maximum_comparisons": max(comparisons),
        "rank_digest_sha256": digest.hexdigest(),
    }


def database_counts(database: sqlite3.Connection) -> dict[str, int]:
    return {
        table: database.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("nodes", "edges", "routes", "anchors")
    }


def run(args: argparse.Namespace) -> dict:
    if args.epoch_events <= 0 or args.epoch_events > 512:
        raise ValueError("epoch events must be between 1 and 512")
    source = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    packed_row = source.execute("SELECT payload FROM packed WHERE id=3").fetchone()
    if packed_row is None:
        raise ValueError("source lacks packed routes")
    packed_routes = packed_row[0]
    count = source.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    expected_bytes = count * (ROUTE_CAPACITY + 1) * 4
    if len(packed_routes) != expected_bytes:
        raise ValueError("packed route size mismatch")
    begin = count - args.epoch_events
    late_rows = rows_for_range(source, packed_routes, begin, count)
    fresh_rows = rows_for_range(source, packed_routes, 0, args.epoch_events)
    late_reciprocal = reciprocal_updates(
        source, packed_routes, begin, count, args.reciprocal_proposals
    )
    fresh_reciprocal: list[tuple[bytes, int]] = []

    for path in (args.fresh_db, args.late_db):
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(f"{path}{suffix}")
            except FileNotFoundError:
                pass

    fresh = sqlite3.connect(args.fresh_db)
    create_schema(fresh)
    fresh.execute("BEGIN IMMEDIATE")
    fresh.executemany(
        "INSERT INTO anchors VALUES(?,?,?)",
        source.execute("SELECT ordinal,label,centroid FROM anchors ORDER BY ordinal"),
    )
    fresh.executemany(
        "INSERT INTO meta VALUES(?,?)",
        source.execute("SELECT key,value FROM meta ORDER BY key"),
    )
    fresh.commit()
    fresh_seconds = apply_epoch(
        fresh, *fresh_rows, fresh_reciprocal, args.epoch_events
    )

    late = sqlite3.connect(args.late_db)
    create_schema(late)
    copy_prefix(source, late, packed_routes, begin)
    before_counts = database_counts(late)
    try:
        apply_epoch(
            late, *late_rows, late_reciprocal, count, failure_stage="before-commit"
        )
    except RuntimeError:
        pass
    rollback_counts = database_counts(late)
    if rollback_counts != before_counts:
        raise ValueError("derived rollback did not restore row counts")
    late_seconds = apply_epoch(late, *late_rows, late_reciprocal, count)
    committed_counts = database_counts(late)
    late.close()
    late = sqlite3.connect(args.late_db)
    restarted_counts = database_counts(late)
    if restarted_counts != committed_counts:
        raise ValueError("restart changed committed route counts")
    restart_visited = restart_rows(late)
    queries = measure_queries(late, count, args.query_count)

    source.close()
    fresh.close()
    late.close()
    return {
        "schema": "cortext_row_addressed_route_sqlite_v1",
        "proof_level": "benchmark-only",
        "evidence_kind": "measured-private-representation",
        "parameters": {
            "embedding_count": count,
            "embedding_dimension": DIMENSION,
            "epoch_events": args.epoch_events,
            "route_capacity": ROUTE_CAPACITY,
            "reciprocal_proposals_per_embedding": args.reciprocal_proposals,
            "query_count": args.query_count,
        },
        "fresh_epoch": {
            "apply_commit_seconds": fresh_seconds,
            "mean_ms_per_event": fresh_seconds * 1000.0 / args.epoch_events,
        },
        "copied_late_epoch": {
            "prefix_rows": begin,
            "apply_commit_seconds": late_seconds,
            "mean_ms_per_event": late_seconds * 1000.0 / args.epoch_events,
            "fresh_ratio": late_seconds / fresh_seconds,
            "reciprocal_rows_updated": len(late_reciprocal),
        },
        "failure_and_restart": {
            "before_commit_rollback_exact": rollback_counts == before_counts,
            "committed_restart_exact": restarted_counts == committed_counts,
            "restart_rows_visited": restart_visited,
            "committed_counts": committed_counts,
        },
        "route_query": queries,
        "database_bytes": {
            "fresh": os.path.getsize(args.fresh_db),
            "copied_late": os.path.getsize(args.late_db),
        },
        "source_fingerprint": {
            "packed_route_sha256": sha256_bytes(packed_routes),
            "node_count": count,
        },
        "decision": "representation-seam-only-production-path-proof-pending",
        "production_cutover": False,
        "unresolved_limits": [
            "graph construction and public-path integration are outside this representation benchmark",
            "post-authoritative-commit publication and Durable barrier failures require engine-path proof",
            "the result does not establish a live public-consolidation sawtooth",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--fresh-db", required=True, type=Path)
    parser.add_argument("--late-db", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epoch-events", type=int, default=200)
    parser.add_argument("--reciprocal-proposals", type=int, default=64)
    parser.add_argument("--query-count", type=int, default=512)
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
