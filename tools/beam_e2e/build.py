#!/usr/bin/env python3
"""Build an end-to-end BEAM split for the Cortext eval runner.

BEAM (Mohammadta/BEAM 100K/500K/1M, Mohammadta/BEAM-10M; cc-by-sa-4.0) holds
long single-user conversations with 20 probing questions each (2 per ability).
Unlike tools/packet_sufficiency_eval/build_beam_cases.py this keeps EVERY
question (end-to-end accuracy is over all 20 per conversation, as in the BEAM
paper, past.dev and AMB) and carries each question's full annotation object so
the answer/judge stage can apply the published prompts.

The parquet files are untrusted input: run with `python -I` from a venv that
has pyarrow.

Writes <out>/
  episodes.jsonl     one episode per conversation (runner input)
  answer_key.jsonl   one row per question (runner input; asked at end of stream)
  questions.jsonl    one row per question: category, raw annotation object,
                     gold turn ids and gold turn texts (judge / attribution input)
  build.json         counts and source hashes
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq

CATEGORIES = [
    "abstention", "contradiction_resolution", "event_ordering", "information_extraction",
    "instruction_following", "knowledge_update", "multi_session_reasoning",
    "preference_following", "summarization", "temporal_reasoning",
]
# Same gold-answer lookup as the past.dev and AMB adapters.
GOLD_FIELDS = ["ideal_response", "answer", "expected_answer", "expected", "gold_answer", "reference"]
FALLBACK_GOLD_FIELDS = ["ideal_answer", "ideal_summary", "expected_compliance"]


def parse_obj(value):
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return ast.literal_eval(value)


def flat_ids(raw) -> list[int]:
    out: list[int] = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif x is not None and str(x).strip().lstrip("-").isdigit():
            out.append(int(x))

    walk(parse_obj(raw) if isinstance(raw, str) else raw)
    return sorted(set(out))


def plan_order(key: str) -> int:
    m = re.search(r"(\d+)$", key)
    return int(m.group(1)) if m else 0


def flatten_turns(chat) -> list[dict]:
    turns = []
    for item in chat:
        if isinstance(item, dict) and "role" in item:
            turns.append(item)
        elif isinstance(item, list):  # standard split: a session
            turns.extend(t for t in item if isinstance(t, dict) and "role" in t)
        elif isinstance(item, dict):  # 10M: {"plan-N": [batch, ...]}
            for key in sorted(item, key=plan_order):
                for batch in sorted(item[key] or [], key=lambda b: b.get("batch_number") or 0):
                    for group in batch.get("turns") or []:
                        turns.extend(t for t in (group if isinstance(group, list) else [group])
                                     if isinstance(t, dict) and "role" in t)
    return turns


def parse_anchor(value) -> datetime | None:
    text = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", str(value or "").strip())
    if not text or text == "None":
        return None
    for fmt in ("%B-%d-%Y", "%b-%d-%Y", "%B %d, %Y", "%B %d %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(hour=8, tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def gold_answer(q: dict) -> str:
    first = next((q[f] for f in GOLD_FIELDS if f in q), None)
    ans = "" if first is None else str(first)
    if not ans.strip():
        ans = next((str(q[f]) for f in FALLBACK_GOLD_FIELDS if q.get(f) is not None), "")
    return ans


def build(row: dict, split: str) -> tuple[dict, list[dict], list[dict], dict]:
    cid = str(row["conversation_id"])
    ep_id = f"beam{split}:{cid}"
    signals, by_id, dup = [], {}, 0
    cur = last = None
    for t in flatten_turns(row["chat"]):
        anchor = parse_anchor(t.get("time_anchor"))
        if anchor is not None and (cur is None or anchor > cur):
            cur = anchor
        base = cur or datetime(2024, 1, 1, 8, tzinfo=timezone.utc)
        ts = base if last is None or base > last else last + timedelta(minutes=1)
        last = ts
        text = " ".join(str(t.get("content") or "").split())
        if not text:
            continue
        tid = int(t["id"]) if t.get("id") is not None else -1 - len(signals)
        if tid in by_id:
            dup += 1
        sid = f"eval/{ep_id}/{tid}" if tid not in by_id else f"eval/{ep_id}/{tid}#{len(signals)}"
        stamp = ts.strftime("%Y/%m/%d (%a) %H:%M")
        sig = {"modality": "text", "source_id": sid,
               "text": f"[{stamp}] {t.get('role', 'speaker')}: {text}",
               "timestamp_ms": int(ts.timestamp() * 1000)}
        by_id.setdefault(tid, []).append(sig)
        signals.append(sig)
    q_ms = signals[-1]["timestamp_ms"] + 60_000
    pos = {s["source_id"]: i for i, s in enumerate(signals)}
    key_rows, q_rows = [], []
    probing = parse_obj(row["probing_questions"])
    for cat in CATEGORIES:
        for k, q in enumerate(probing.get(cat, []) or []):
            text = q.get("question", "")
            if not text:
                continue
            gold = flat_ids(q.get("source_chat_ids"))
            src = q.get("source_chat_ids")
            src = parse_obj(src) if isinstance(src, str) else src
            updated = flat_ids(src.get("updated_info")) if isinstance(src, dict) else []
            present = [g for g in gold if g in by_id]
            qid = f"{cid}:{cat}:{k}"
            ans = gold_answer(q)
            key_rows.append({
                "conversation_id": ep_id, "query_id": qid, "question": text,
                "question_type": f"beam/{cat}", "answers": [ans] if ans else [],
                "evidence": [s["text"] for g in present for s in by_id[g]],
                "requires_abstention": cat == "abstention", "query_timestamp_ms": q_ms,
            })
            q_rows.append({
                "split": split, "conversation_id": ep_id, "query_id": qid, "category": cat,
                "question": text, "gold_answer": ans, "raw": q,
                "gold_turn_ids": gold, "gold_updated_turn_ids": updated,
                "gold_missing_ids": [g for g in gold if g not in by_id],
                # BEAM 1M convs 5/26/33/34 restart turn ids mid-stream: an
                # ambiguous gold id lists every turn that carries it.
                "gold_turns": [{"id": g, "source_id": s["source_id"], "text": s["text"],
                                "depth": round(pos[s["source_id"]] / max(1, len(signals)), 4),
                                "ambiguous": len(by_id[g]) > 1}
                               for g in present for s in by_id[g]],
                "haystack_turns": len(signals), "query_timestamp_ms": q_ms,
            })
    ep = {"episode_id": ep_id, "benchmark": f"beam{split}", "signals": signals}
    return ep, key_rows, q_rows, {"turns": len(signals), "dup_ids": dup, "questions": len(q_rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet", nargs="+", type=Path)
    ap.add_argument("--split", required=True, help="label: 100k, 500k, 1m, 10m")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conversations", default="", help="comma list of conversation ids (default all)")
    args = ap.parse_args()
    want = {c for c in args.conversations.split(",") if c}
    args.out.mkdir(parents=True, exist_ok=True)
    stats, n_q = {}, 0
    with (args.out / "episodes.jsonl").open("w") as eps, \
            (args.out / "answer_key.jsonl").open("w") as keys, \
            (args.out / "questions.jsonl").open("w") as qs:
        for path in args.parquet:
            table = pq.read_table(path, columns=["conversation_id", "chat", "probing_questions"])
            for i in range(table.num_rows):
                row = table.slice(i, 1).to_pylist()[0]
                if want and str(row["conversation_id"]) not in want:
                    continue
                ep, key_rows, q_rows, st = build(row, args.split)
                eps.write(json.dumps(ep) + "\n")
                for r in key_rows:
                    keys.write(json.dumps(r) + "\n")
                for r in q_rows:
                    qs.write(json.dumps(r) + "\n")
                n_q += len(q_rows)
                stats[ep["episode_id"]] = st
                print(f"{ep['episode_id']}: {st}", flush=True)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.parquet}
    (args.out / "build.json").write_text(json.dumps(
        {"split": args.split, "sources_sha256": sources, "episodes": len(stats), "questions": n_q,
         "turns": sum(s["turns"] for s in stats.values()), "per_episode": stats}, indent=1) + "\n")
    print(json.dumps({"episodes": len(stats), "questions": n_q, "out": str(args.out)}))


if __name__ == "__main__":
    main()
