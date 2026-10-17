#!/usr/bin/env python3
"""Ingest each BEAM conversation into a fresh Cortext store and record the packet
for every question (one runner process per conversation, at most --slots at once).

usage: packets.py --data <split dir from build.py> --vdir <out dir> --runner <bin>
                  [--slots 2] [--top-k 100] [--keep-stores]

Per conversation: <vdir>/conv/<n>/{rows.jsonl, rows.summary.json, ingest.jsonl,
runner.err (with /usr/bin/time -l)}. Merged rows: <vdir>/packets.jsonl.
Re-running skips conversations whose rows.summary.json exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--vdir", type=Path, required=True)
    ap.add_argument("--runner", type=Path, required=True)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=100, help="runner shows min(top-k, engine packet)")
    ap.add_argument("--threads", type=int, default=4, help="CORTEXT_EMBED/INFER_THREADS per runner")
    ap.add_argument("--keep-stores", action="store_true")
    ap.add_argument("--episodes", default="", help="comma list of episode ids (default all)")
    args = ap.parse_args()

    key = [json.loads(l) for l in (args.data / "answer_key.jsonl").open() if l.strip()]
    eps = {}
    with (args.data / "episodes.jsonl").open() as fh:
        for line in fh:
            ep_id = json.loads(line)["episode_id"]
            eps[ep_id] = line
    want = [e for e in eps if not args.episodes or e in set(args.episodes.split(","))]
    sizes = {e: eps[e].count('"source_id"') for e in want}
    env = dict(os.environ)
    env["CORTEXT_EMBED_THREADS"] = env["CORTEXT_INFER_THREADS"] = str(args.threads)

    def run(ep_id: str):
        d = args.vdir / "conv" / ep_id.split(":")[-1]
        if (d / "rows.summary.json").exists():
            return ep_id, 0, 0.0, "skip"
        d.mkdir(parents=True, exist_ok=True)
        (d / "episodes.jsonl").write_text(eps[ep_id])
        with (d / "answer_key.jsonl").open("w") as fh:
            for r in key:
                if r["conversation_id"] == ep_id:
                    fh.write(json.dumps(r) + "\n")
        store = d / "store.sqlite"
        for p in d.glob("store*.sqlite*"):
            p.unlink()
        cmd = [str(args.runner), f"--episodes={d / 'episodes.jsonl'}",
               f"--answer-key={d / 'answer_key.jsonl'}", f"--out={d / 'rows.jsonl'}",
               f"--top-k={args.top_k}", "--consolidate=recommended", f"--db={store}",
               f"--ingest-log={d / 'ingest.jsonl'}"]
        t0 = time.time()
        with (d / "runner.err").open("w") as err:
            proc = subprocess.run(["/usr/bin/time", "-l", *cmd], env=env,
                                  stdout=subprocess.DEVNULL, stderr=err)
        wall = time.time() - t0
        store_bytes = sum(p.stat().st_size for p in d.glob("store*.sqlite*"))
        (d / "run.json").write_text(json.dumps({"wall_s": round(wall, 1), "exit": proc.returncode,
                                                "signals": sizes[ep_id], "store_bytes": store_bytes}) + "\n")
        if not args.keep_stores:
            for p in d.glob("store*.sqlite*"):
                p.unlink()
        if proc.returncode != 0 and (d / "rows.summary.json").exists():
            (d / "rows.summary.json").unlink()
        print(f"{ep_id}: {sizes[ep_id]} signals exit {proc.returncode} {wall:.0f}s", file=sys.stderr, flush=True)
        return ep_id, proc.returncode, wall, "ran"

    t0 = time.time()
    # Largest first so the two slots finish together.
    order = sorted(want, key=lambda e: -sizes[e])
    with ThreadPoolExecutor(max_workers=args.slots) as pool:
        results = list(pool.map(run, order))
    bad = [r for r in results if r[1] != 0]
    if bad:
        raise SystemExit(f"conversations failed: {bad}")
    with (args.vdir / "packets.jsonl").open("w") as out:
        for ep_id in want:
            d = args.vdir / "conv" / ep_id.split(":")[-1]
            for line in (d / "rows.jsonl").open():
                out.write(line)
    (args.vdir / "variant.json").write_text(json.dumps({
        "runner": str(args.runner), "top_k": args.top_k, "slots": args.slots, "threads": args.threads,
        "consolidate": "recommended", "wall_s": round(time.time() - t0, 1),
    }, indent=1) + "\n")
    print(f"packets: {args.vdir / 'packets.jsonl'} in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
