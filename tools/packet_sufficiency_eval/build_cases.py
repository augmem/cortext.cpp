#!/usr/bin/env python3
"""Build the packet-sufficiency eval case set from LongMemEval-S and LoCoMo.

Reads the normalized output of scripts/prepare_memory_evals.py plus the raw
benchmark files, draws a seeded stratified sample, and writes:

  <out>/cases.jsonl        one row per probe (question, gold, evidence, tags)
  <out>/episodes.jsonl     only the episodes the sample needs, with
                           timestamp_ms parsed from each turn's date prefix
                           and signals sorted chronologically
  <out>/answer_key.jsonl   runner-format key with query_timestamp_ms
  <out>/cases.md           human-readable review sheet

The prepare script leaves dates only inside the text prefix, so without this
step every turn would be ingested at wall-clock "now".
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

LME_ALLOC = {
    "multi-session": 35,
    "temporal-reasoning": 35,
    "knowledge-update": 25,
    "single-session-user": 20,
    "single-session-assistant": 15,
    "single-session-preference": 10,
}
LME_ABSTENTION = 10
LOCOMO_ALLOC = {"1": 20, "2": 25, "3": 10, "4": 25, "5": 20}
LOCOMO_NAMES = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial",
}

LME_PREFIX = re.compile(r"^\[(\d{4}/\d{2}/\d{2}) \(\w+\) (\d{2}:\d{2})\]")
LOCOMO_PREFIX = re.compile(r"^\[(\d{1,2}:\d{2} [ap]m) on (\d{1,2} \w+), (\d{4})\]")


def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def parse_lme_date(text: str) -> int | None:
    m = LME_PREFIX.match(text)
    if not m:
        return None
    return to_ms(datetime.strptime(f"{m[1]} {m[2]}", "%Y/%m/%d %H:%M"))


def parse_lme_question_date(value: str) -> int:
    day, _, clock = value.partition(" (")
    clock = clock.split(") ", 1)[1]
    return to_ms(datetime.strptime(f"{day} {clock}", "%Y/%m/%d %H:%M"))


def parse_locomo_date(text: str) -> int | None:
    m = LOCOMO_PREFIX.match(text)
    if not m:
        return None
    return to_ms(datetime.strptime(f"{m[1]} {m[2]} {m[3]}", "%I:%M %p %d %B %Y"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_episodes(path: Path, wanted: set[str]) -> dict[str, dict]:
    out = {}
    with path.open() as fh:
        for line in fh:
            ep = json.loads(line)
            ep_id = ep.get("episode_id") or ep.get("conversation_id")
            if ep_id in wanted:
                out[ep_id] = ep
    return out


def stamp(episode: dict, parse) -> tuple[dict, int]:
    """Attach timestamp_ms to every signal; carry the last seen date forward."""
    last = None
    missing = 0
    for sig in episode["signals"]:
        ts = parse(sig.get("text", ""))
        if ts is None:
            missing += 1
            ts = last
        if ts is not None:
            sig["timestamp_ms"] = ts
            last = ts
    episode["signals"] = [s for s in episode["signals"] if "timestamp_ms" in s]
    episode["signals"].sort(key=lambda s: s["timestamp_ms"])
    return episode, missing


def sample(rows: list[dict], alloc: dict[str, int], key, rng) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    picked = []
    for k, n in alloc.items():
        pool = sorted(by[k], key=lambda r: r["query_id"])
        if len(pool) < n:
            raise SystemExit(f"stratum {k!r} has {len(pool)} rows, need {n}")
        picked.extend(rng.sample(pool, n))
    return picked


def lme_cases(root: Path, raw_root: Path, rng) -> tuple[list[dict], list[dict]]:
    key = load_jsonl(root / "longmemeval_s" / "answer_key.jsonl")
    raw = {
        r["question_id"]: r
        for r in json.loads(
            (raw_root / "longmemeval" / "longmemeval_s_cleaned.json").read_text()
        )
    }
    abstain = [r for r in key if r["query_id"].endswith("_abs")]
    regular = [r for r in key if not r["query_id"].endswith("_abs")]
    picked = sample(regular, LME_ALLOC, lambda r: r["question_type"], rng)
    picked += rng.sample(sorted(abstain, key=lambda r: r["query_id"]), LME_ABSTENTION)

    cases = []
    for r in picked:
        src = raw[r["query_id"]]
        evidence = [
            t["content"]
            for s in src["haystack_sessions"]
            for t in s
            if t.get("has_answer")
        ]
        abstention = r["query_id"].endswith("_abs")
        cases.append(
            {
                "prompt_id": f"lme:{r['query_id']}",
                "tags": [
                    f"lme/{r['question_type']}",
                    "abstention" if abstention else "answerable",
                    "longmemeval_s",
                ],
                "benchmark": "longmemeval_s",
                "episode_id": r["conversation_id"],
                "query_id": r["query_id"],
                "question": r["question"],
                "question_date": src["question_date"],
                "query_timestamp_ms": parse_lme_question_date(src["question_date"]),
                "answers": [str(a) for a in r["answers"]],
                "evidence": evidence,
                "requires_abstention": abstention,
            }
        )
    return cases, key


def lme_question_date_text(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def locomo_cases(root: Path, rng, episodes: dict[str, dict]) -> list[dict]:
    key = load_jsonl(root / "locomo" / "answer_key.jsonl")
    picked = sample(key, LOCOMO_ALLOC, lambda r: r["question_type"], rng)
    cases = []
    for r in picked:
        cat = r["question_type"]
        ep = episodes[r["conversation_id"]]
        # LoCoMo has no question date; ask one day after the last session.
        q_ms = ep["signals"][-1]["timestamp_ms"] + int(timedelta(days=1).total_seconds() * 1000)
        cases.append(
            {
                "prompt_id": f"locomo:{r['query_id']}",
                "tags": [
                    f"locomo/{LOCOMO_NAMES[cat]}",
                    "abstention" if r["requires_abstention"] else "answerable",
                    "locomo",
                ],
                "benchmark": "locomo",
                "episode_id": r["conversation_id"],
                "query_id": r["query_id"],
                "question": r["question"],
                "question_date": lme_question_date_text(q_ms),
                "query_timestamp_ms": q_ms,
                "answers": [str(a) for a in r["answers"]],
                "evidence": r.get("evidence") or [],
                "requires_abstention": bool(r["requires_abstention"]),
            }
        )
    return cases


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def attach_dated_evidence(cases: list[dict], episodes: dict[str, dict]) -> int:
    """Swap raw evidence strings for the timestamped turns that carry them.

    Raw benchmark evidence has no date, so an oracle packet built from it
    can't answer temporal or knowledge-update questions. Match each evidence
    string to the episode signal containing its opening, keep the raw string
    when nothing matches, and return how many stayed unmatched.
    """
    unmatched = 0
    for c in cases:
        signals = [(s["text"], _norm(s["text"])) for s in episodes[c["episode_id"]]["signals"]]
        dated = []
        for ev in c["evidence"]:
            probe = _norm(ev)[:120]
            hits = [text for text, norm in signals if probe and probe in norm]
            if hits:
                dated.extend(h for h in hits if h not in dated)
            else:
                unmatched += 1
                dated.append(ev)
        c["evidence"] = dated
    return unmatched


def fence(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}\n{text}\n{ticks}"


def write_markdown(path: Path, cases: list[dict]) -> None:
    lines = ["# Packet-sufficiency eval cases", ""]
    counts = Counter(c["tags"][0] for c in cases)
    lines.append(f"{len(cases)} cases. Per stratum: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    lines += ["", "| id | tags | question | gold |", "|---|---|---|---|"]
    for c in cases:
        gold = "; ".join(c["answers"]) or "(abstain)"
        q = c["question"].replace("|", "\\|")
        g = gold.replace("|", "\\|")[:120]
        lines.append(f"| `{c['prompt_id']}` | {', '.join(c['tags'][:2])} | {q} | {g} |")
    lines.append("")
    for c in cases:
        lines += [
            f"## {c['prompt_id']}",
            "",
            f"- tags: {', '.join(c['tags'])}",
            f"- question date: {c['question_date']}",
            f"- gold: {'; '.join(c['answers']) or '(abstain)'}",
            "",
            "Question:",
            "",
            fence(c["question"]),
            "",
            "Evidence turns:",
            "",
            fence("\n---\n".join(c["evidence"]) or "(none)"),
            "",
        ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data/memory_evals")
    ap.add_argument("--raw-root", default="data/raw/memory_evals")
    ap.add_argument("--out", default="data/memory_evals/packet_sufficiency")
    ap.add_argument("--seed", type=int, default=20261012)
    args = ap.parse_args()

    root, raw_root, out = Path(args.root), Path(args.raw_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    lme, _ = lme_cases(root, raw_root, rng)
    lme_eps = load_episodes(
        root / "longmemeval_s" / "episodes.jsonl", {c["episode_id"] for c in lme}
    )
    loc_ids = {r["conversation_id"] for r in load_jsonl(root / "locomo" / "answer_key.jsonl")}
    loc_eps = load_episodes(root / "locomo" / "episodes.jsonl", loc_ids)

    missing = Counter()
    for ep in lme_eps.values():
        _, n = stamp(ep, parse_lme_date)
        missing["longmemeval_s"] += n
    for ep in loc_eps.values():
        _, n = stamp(ep, parse_locomo_date)
        missing["locomo"] += n
    loc = locomo_cases(root, rng, loc_eps)
    cases = lme + loc
    unmatched = attach_dated_evidence(cases, {**lme_eps, **loc_eps})

    with (out / "cases.jsonl").open("w") as fh:
        for c in cases:
            fh.write(json.dumps(c) + "\n")
    with (out / "episodes.jsonl").open("w") as fh:
        for ep in list(lme_eps.values()) + list(loc_eps.values()):
            fh.write(json.dumps(ep) + "\n")
    with (out / "answer_key.jsonl").open("w") as fh:
        for c in cases:
            fh.write(
                json.dumps(
                    {
                        "conversation_id": c["episode_id"],
                        "query_id": c["query_id"],
                        "question": c["question"],
                        "question_type": c["tags"][0],
                        "answers": c["answers"],
                        "evidence": c["evidence"],
                        "requires_abstention": c["requires_abstention"],
                        "query_timestamp_ms": c["query_timestamp_ms"],
                    }
                )
                + "\n"
            )
    write_markdown(out / "cases.md", cases)
    print(
        f"cases={len(cases)} episodes={len(lme_eps) + len(loc_eps)} "
        f"undated_signals={dict(missing)} unmatched_evidence={unmatched} out={out}"
    )


if __name__ == "__main__":
    main()
