#!/usr/bin/env python3
"""Aggregate BEAM end-to-end runs: scores per protocol/arm/model, categories, latency, cost.

usage: report.py <runs dir> [--splits 100k,1m,10m] [--cache <cache.jsonl>]
Prints JSON; latency comes from each conversation's runner outputs
(ingest.jsonl per-turn wall ms, rows.jsonl query_ms, runner.err /usr/bin/time -l).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 2) if xs else None


def latency(split_dir: Path) -> dict:
    ingest, query, rss, walls, turns, stores = [], [], [], [], 0, []
    deciles = [[] for _ in range(10)]
    for conv in sorted((split_dir / "conv").iterdir()):
        rows = [json.loads(l) for l in (conv / "ingest.jsonl").open() if l.strip()]
        ms = [r["ms"] for r in rows]
        ingest += ms
        turns += len(ms)
        for i, x in enumerate(ms):
            deciles[min(9, i * 10 // max(1, len(ms)))].append(x)
        query += [json.loads(l)["query_ms"] for l in (conv / "rows.jsonl").open() if l.strip()]
        err = (conv / "runner.err").read_text()
        m = re.search(r"(\d+)\s+maximum resident set size", err)
        if m:
            rss.append(int(m.group(1)) / 2**20)
        m = re.search(r"([\d.]+) real", err)
        if m:
            walls.append(float(m.group(1)))
        st = sum(p.stat().st_size for p in conv.glob("store*.sqlite*"))
        run = conv / "run.json"
        if st == 0 and run.exists():
            st = json.loads(run.read_text()).get("store_bytes", 0)
        stores.append(st / 2**20)
    return {
        "conversations": len(walls), "turns": turns,
        "ingest_ms_per_turn": {"p50": pct(ingest, .5), "p95": pct(ingest, .95), "mean": round(sum(ingest) / max(1, len(ingest)), 2)},
        "ingest_ms_p50_by_decile": [pct(d, .5) for d in deciles],
        "query_ms": {"p50": pct(query, .5), "p95": pct(query, .95), "max": pct(query, 1.0)},
        "rss_mb_max": round(max(rss), 0) if rss else None, "rss_mb_p50": pct(rss, .5),
        "wall_s_per_conv_p50": pct(walls, .5), "store_mb_p50": pct(stores, .5), "store_mb_max": round(max(stores), 1) if stores else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--splits", default="100k,1m,10m")
    a = ap.parse_args()
    out = {}
    cache = {}
    cpath = a.runs / ".." / "cache.jsonl"
    for split in a.splits.split(","):
        d = a.runs / split
        if not (d / "conv").exists():
            continue
        cpath = d / "cache.jsonl"
        if cpath.exists():
            for l in cpath.open():
                try:
                    r = json.loads(l)
                    cache[r["key"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
        entry = {"latency": latency(d), "runs": {}}
        for e in sorted(d.glob("e2e_*")):
            if not (e / "summary.json").exists():
                continue
            s = json.loads((e / "summary.json").read_text())
            entry["runs"][e.name] = s
        out[split] = entry
    cost = {}
    for r in cache.values():
        k = (r.get("kind"), r.get("model"))
        cost[f"{k[0]}:{k[1]}"] = round(cost.get(f"{k[0]}:{k[1]}", 0) + r.get("cost", 0), 3)
    out["cost_usd_cached_calls"] = cost
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
