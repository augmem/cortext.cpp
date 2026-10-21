"""Collect benchmark replay streams for evaluation only (never trained on).

usage: python -I streams.py --hill .claude/hillclimb --out /Volumes/Build/cortext/directive-head/streams
Writes texts.jsonl (unique stored text over every stream, {"text"}), turns.jsonl ({stream, episode, source_id,
uid}; uid indexes texts.jsonl) and beam_gold.jsonl (BEAM instruction_following / preference_following gold
turns: {stream, episode, source_id, category, gold_text, tested}).
Streams: scorecard/data/{lme_s, lme_m100, locomo}, beam_e2e/data/{100k, 1m, 10m}, scorecard/dolphin/ingest/*.
locomo_clock / locomo_clock_ref repeat LoCoMo with other clocks and are skipped.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

STREAMS = {"lme_s": "scorecard/data/lme_s", "lme_m100": "scorecard/data/lme_m100", "locomo": "scorecard/data/locomo",
           "beam100k": "beam_e2e/data/100k", "beam1m": "beam_e2e/data/1m", "beam10m": "beam_e2e/data/10m",
           "dolphin_alex": "scorecard/dolphin/ingest/alex", "dolphin_morgan": "scorecard/dolphin/ingest/morgan",
           "dolphin_riley": "scorecard/dolphin/ingest/riley"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hill", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hill, out = Path(a.hill), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    uid: dict[str, int] = {}
    with (out / "turns.jsonl").open("w") as ft:
        for name, rel in STREAMS.items():
            n = 0
            for line in (hill / rel / "episodes.jsonl").open():
                e = json.loads(line)
                for s in e["signals"]:
                    if s.get("modality") != "text" or not s.get("text"):
                        continue
                    u = uid.setdefault(s["text"], len(uid))
                    ft.write(json.dumps({"stream": name, "episode": e["episode_id"], "source_id": s["source_id"],
                                         "uid": u}) + "\n")
                    n += 1
            print(name, n, "turns", len(uid), "unique so far", flush=True)
    with (out / "texts.jsonl").open("w") as f:
        for t in uid:  # dict order == uid order
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    with (out / "beam_gold.jsonl").open("w") as fg:
        for name in ("beam100k", "beam1m", "beam10m"):
            for line in (hill / STREAMS[name] / "questions.jsonl").open():
                q = json.loads(line)
                if q["category"] not in ("instruction_following", "preference_following"):
                    continue
                tested = q["raw"].get("instruction_being_tested") or q["raw"].get("preference_being_tested") or ""
                turns = [t for t in q.get("gold_turns") or [] if "] user:" in t["text"][:40]]
                hit = [t for t in turns if tested and tested[:40].lower() in t["text"].lower()]
                for t in (hit or turns[:1]):
                    fg.write(json.dumps({"stream": name, "episode": q["conversation_id"], "source_id": t["source_id"],
                                         "category": q["category"], "gold_text": t["text"], "tested": tested,
                                         "exact": bool(hit)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
