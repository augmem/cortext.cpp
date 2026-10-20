#!/usr/bin/env python3
"""Oracle packets for the MemLens eval: the packet is exactly the gold turns.

Judging these bounds what any retrieval could reach per surface:
  text      gold turn text only (images dropped)
  caption   gold turn text with the BLIP caption inline
  image     gold turn text plus the gold images as media refs

Writes <flow>/<variant>/packets.jsonl in runner row shape, so run_eval.py
judges it without running the engine (it skips ingest when packets cover
every case). Usage:
  oracle_packets.py --data DIR/<arm> --surface image --out FLOW/v12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="arm dir with cases.jsonl and episodes.jsonl")
    ap.add_argument("--surface", choices=["text", "caption", "image"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cases = [json.loads(l) for l in (args.data / "cases.jsonl").open()]
    images = {}
    for line in (args.data / "episodes.jsonl").open():
        for s in json.loads(line)["signals"]:
            if s["modality"] == "image":
                images[s["media_sha"]] = {k: s[k] for k in ("source_id", "mimetype", "judge_path", "caption")} | {
                    "media_sha": s["media_sha"], "path": s["media_path"]}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "packets.jsonl").open("w") as fh:
        for c in cases:
            items = []
            for i, g in enumerate(c["gold_turns"]):
                text = g["text"]
                if args.surface == "caption":
                    text += "".join(f" Shared image caption: {im['caption']}." for im in g["images"] if im["caption"])
                item = {"rank": i + 1, "source_id": g["source_id"], "modality": "text", "text": text}
                if args.surface == "image" and g["images"]:
                    item["media"] = [images[im["sha"]] for im in g["images"] if im["sha"] in images]
                items.append(item)
            fh.write(json.dumps({"query_id": c["query_id"], "retrieved": items, "working": [], "query_ms": 0.0}) + "\n")
    print(f"wrote {len(cases)} oracle packets ({args.surface}) to {args.out}")


if __name__ == "__main__":
    main()
