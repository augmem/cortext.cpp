#!/usr/bin/env python3
"""End-to-end BEAM: answer every question from the Cortext packet only, then judge.

usage:
  e2e.py run --data <split dir> --packets <packets.jsonl> --out <dir>
             --protocol exabase|amb --arm full|r8k|r2k|none|oracle
             [--answer-model claude-haiku-5-5] [--judge-model claude-haiku-5-5]
             [--concurrency 8] [--limit N] [--queries qid,...]
  e2e.py summary <out dir> [...]
  e2e.py estimate --data <split dir> --packets <packets.jsonl> --arm full

Arms (what the answerer reads; the packet is the only memory context):
  full  engine packet as returned: retrieved (R1..) then working memory (W1..)
  r8k   retrieved items only, in rank order, cut at 8,000 tokens (past.dev recall maxTokens)
  r2k   retrieved items only, cut at 2,000 tokens (past.dev DolphinBench budget)
  oracle  only the question's gold turns (answer-side ceiling; abstention gets nothing)
  none  no memory at all (control: how much a protocol scores from its prompt alone)
Token budget uses 2.8 chars/token (measured Claude tokenizer rate on this text); the
answer call's real input tokens are recorded.

Models run through `claude -p` (no tools, no settings, no session) with thinking off
(MAX_THINKING_TOKENS=0, matching past.dev's reasoning-effort none). Every call is cached
by content hash in <out>/../cache.jsonl, so re-runs and shared judge calls are free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocols as P  # noqa: E402

ANSWER_SYSTEM = "You are a helpful assistant."
JUDGE_SYSTEM = "You are a careful, impartial evaluator. Follow the instructions exactly."
# Claude tokenizer on BEAM packet text measures 2.81 chars/token (100K fit: tokens =
# 4279 + 0.356 * chars), so a budget of N tokens is N * 2.8 chars of evidence.
CHARS_PER_TOKEN = 2.8
# r8ko: 8,000 tokens in the answerer's own (gpt-5.6-luna) tokenizer, measured 4.35 chars/token
# on BEAM packet text via CPA prompt_tokens (60k chars -> 13,807 tokens) - past.dev's budget.
LUNA_CHARS_PER_TOKEN = 4.35
BUDGET_CHARS = {"r8k": int(8000 * CHARS_PER_TOKEN), "r2k": int(2000 * CHARS_PER_TOKEN),
                "r8ko": int(8000 * LUNA_CHARS_PER_TOKEN)}
MAX_ATTEMPTS = 5
_lock = threading.Lock()
_cache: dict[str, dict] = {}
_cache_path: Path | None = None
_cost = defaultdict(float)


# --- claude -p -------------------------------------------------------------------------------

def _key(*parts) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def load_cache(path: Path) -> None:
    global _cache_path
    _cache_path = path
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
                _cache[r["key"]] = r
            except (json.JSONDecodeError, KeyError):
                continue


def backend() -> str:
    """JUDGE_BACKEND=cpa sends answer and judge calls to the OpenAI-compatible CPA proxy
    (.claude/hillclimb/cpa_judge/cpa_backend.py) instead of `claude -p`."""
    return os.environ.get("JUDGE_BACKEND", "cli").strip().lower() or "cli"


def _cpa_module():
    here = Path(__file__).resolve().parent
    for p in (here, *here.parents):
        d = p / ".claude" / "hillclimb" / "cpa_judge"
        if (d / "cpa_backend.py").is_file():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            import cpa_backend
            return cpa_backend
    raise SystemExit("JUDGE_BACKEND=cpa but .claude/hillclimb/cpa_judge/cpa_backend.py not found")


def cpa_call(prompt: str, model: str, system: str, schema: dict | None, kind: str, k: str, timeout_s: float) -> dict:
    C = _cpa_module()
    try:
        # past.dev sends the prompt as the only (user) message; E2E_CPA_SYSTEM=keep sends ours
        sys_msg = system if os.environ.get("E2E_CPA_SYSTEM", "none") == "keep" else None
        r = C.chat(sys_msg, prompt, model=model, schema=schema, schema_name=kind, timeout_s=timeout_s)
    except C.CPAError as e:
        raise RuntimeError(f"cpa {e.kind}: {e}")
    rec = {"key": k, "kind": kind, "model": model, "backend": "cpa", "served_model": r["model"],
           "text": r["text"], "json": r["json"], "usage": r["usage"], "cost": 0.0}
    with _lock:
        _cache[k] = rec
        with _cache_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return {**rec, "cached": False}


def claude(prompt: str, model: str, system: str, schema: dict | None, kind: str, timeout_s: float = 300) -> dict:
    """{'text', 'json', 'usage', 'cost', 'cached'}; raises RuntimeError after retries."""
    parts = [model, system, json.dumps(schema, sort_keys=True) if schema else "", prompt]
    if backend() != "cli":  # separate key namespace; cli keys unchanged
        parts.insert(0, backend() + ("/sys" if os.environ.get("E2E_CPA_SYSTEM", "none") == "keep" else "/nosys"))
    k = _key(*parts)
    with _lock:
        hit = _cache.get(k)
    if hit:
        return {**hit, "cached": True}
    if backend() == "cpa":
        return cpa_call(prompt, model, system, schema, kind, k, timeout_s)
    cmd = ["claude", "-p", "--model", model, "--tools", "", "--setting-sources", "",
           "--no-session-persistence", "--system-prompt", system, "--output-format", "json"]
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    # thinking off (past.dev: reasoning none); no prompt-cache writes (each prompt is unique,
    # and 1h cache creation bills ~2x input)
    env = dict(os.environ, MAX_THINKING_TOKENS="0", DISABLE_PROMPT_CACHING="1")
    last = ""
    with tempfile.TemporaryDirectory(prefix="beam_e2e_") as cwd:
        for attempt in range(MAX_ATTEMPTS):
            try:
                proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True,
                                      timeout=timeout_s, cwd=cwd, env=env)
                out = json.loads(proc.stdout)
            except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
                last = str(e)[:300]
                time.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))
                continue
            if out.get("is_error") or proc.returncode != 0:
                last = str(out.get("result"))[:300]
                time.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))
                continue
            served = set((out.get("modelUsage") or {}).keys())
            if served != {model}:
                raise RuntimeError(f"served model mismatch: asked {model}, served {sorted(served)}")
            rec = {"key": k, "kind": kind, "model": model, "text": out.get("result") or "",
                   "json": out.get("structured_output"), "usage": out.get("usage") or {},
                   "cost": float(out.get("total_cost_usd") or 0.0)}
            if schema and not isinstance(rec["json"], dict):
                last = "unparseable structured output"
                continue
            with _lock:
                _cache[k] = rec
                _cost[kind] += rec["cost"]
                with _cache_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            return {**rec, "cached": False}
    raise RuntimeError(f"claude -p failed after {MAX_ATTEMPTS} attempts: {last}")


# --- packet -> context -----------------------------------------------------------------------

def context_items(packet: dict, arm: str, q: dict | None = None) -> list[tuple[str, str]]:
    if arm == "oracle":  # answer-side ceiling: exactly the gold turns, in stream order
        turns = sorted(q["gold_turns"], key=lambda g: g["depth"]) if q else []
        return [(f"G{i + 1}", g["text"]) for i, g in enumerate(turns)]
    if arm == "none":  # memory-free control: what the protocol scores with no packet at all
        return []
    # The standing block (directives, outside the retrieved ranking) leads.
    items = [(f"S{i + 1}", m["text"]) for i, m in enumerate(packet.get("standing", []))]
    items += [(f"R{i + 1}", m["text"]) for i, m in enumerate(packet.get("retrieved", []))]
    if arm == "full":
        return items + [(f"W{i + 1}", m["text"]) for i, m in enumerate(packet.get("working", []))]
    budget, out, used = BUDGET_CHARS[arm], [], 0
    for iid, text in items:
        room = budget - used
        if room <= 0:
            break
        if len(text) > room:
            text = text[:room]
        out.append((iid, text))
        used += len(text) + len(iid) + 4
    return out


def context_text(items: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[{iid}] {text}" for iid, text in items) or "(no memories retrieved)"


_ws = re.compile(r"\s+")
_prefix = re.compile(r"^\[[^\]]*\]\s*\w+:\s*")


def norm(s: str) -> str:
    return _ws.sub(" ", s).strip().lower()


def gold_recall(q: dict, ctx: str) -> tuple[float | None, list[int]]:
    """Fraction of gold turn ids whose turn text is in the context (any occurrence of an
    ambiguous id counts). Match on the turn body's first 200 normalized chars; engine items
    may clip long turns, so fall back to the first 80."""
    ids = sorted({g["id"] for g in q["gold_turns"]})
    if not ids:
        return None, []
    c = norm(ctx)
    found = set()
    for g in q["gold_turns"]:
        body = norm(_prefix.sub("", g["text"]))
        if body[:200] in c or (len(body) > 80 and body[:80] in c):
            found.add(g["id"])
    return len(found) / len(ids), sorted(found)


# --- one question ----------------------------------------------------------------------------

def run_question(q: dict, packet: dict, a) -> dict:
    cat, raw = q["category"], q["raw"]
    meta = P.annotations(cat, raw)
    items = context_items(packet, a.arm, q)
    ctx = context_text(items)
    recall, found = gold_recall(q, ctx)
    row = {"query_id": q["query_id"], "conversation_id": q["conversation_id"], "category": cat,
           "split": q["split"], "protocol": a.protocol, "arm": a.arm, "answer_model": a.answer_model,
           "judge_model": a.judge_model, "context_chars": len(ctx), "context_items": len(items),
           "gold_ids": sorted({g["id"] for g in q["gold_turns"]}), "gold_found": found,
           "gold_recall": recall, "query_ms": packet.get("query_ms")}
    try:
        prompt = (P.exabase_answer_prompt(q["question"], ctx, cat, meta) if a.protocol == "exabase"
                  else P.amb_answer_prompt(q["question"], ctx))
        ans = claude(prompt, a.answer_model, ANSWER_SYSTEM, None, "answer")
        answer = ans["text"].strip()
        row["answer"] = answer
        row["answer_input_tokens"] = sum(int(ans["usage"].get(f) or 0) for f in
                                         ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                                          "prompt_tokens"))
        if not answer:
            raise RuntimeError("empty answer")
    except RuntimeError as e:
        row.update(score=0.0, error=f"answer: {e}")
        return row

    if a.protocol == "exabase" and cat == "event_ordering":
        reference = meta.get("ordering_tested") or []
        if not reference and q["gold_answer"]:
            reference = P.list_items(q["gold_answer"])
        sys_items = P.list_items(answer)
        if not reference or not sys_items:
            row.update(score=0.0, judge={"method": "event_ordering", "reference": reference, "answer": sys_items})
            return row
        matched, renamed = set(), []
        for item in sys_items:
            match = None
            for idx, topic in enumerate(reference):
                if idx in matched:
                    continue
                try:
                    v = claude(P.equivalence_prompt(topic, item), a.judge_model, JUDGE_SYSTEM,
                               P.EQUIVALENCE_SCHEMA, "judge")
                    same = str(v["json"].get("answer", "")).strip().upper().startswith("YES")
                except RuntimeError:
                    same = False
                if same:
                    match = idx
                    break
            if match is None:
                renamed.append(item)
            else:
                renamed.append(reference[match])
                matched.add(match)
        row.update(score=P.ordering_score(reference, renamed),
                   judge={"method": "event_ordering", "reference": reference, "answer": renamed})
        return row

    rubric = list(meta.get("rubric") or [])
    if a.protocol == "amb" and cat == "event_ordering":
        rubric = [r if "should" in str(r).lower() else f"LLM response should mention: {r}" for r in rubric]
    if not rubric and q["gold_answer"]:
        rubric = [f"LLM response should contain: {q['gold_answer']}"]
    if not rubric:
        row.update(score=0.0, judge={"method": "rubric", "note": "no rubric"})
        return row
    verdicts = []
    for item in rubric:
        prompt = (P.exabase_rubric_prompt(q["question"], answer, item) if a.protocol == "exabase"
                  else P.amb_rubric_prompt(q["question"], answer, item))
        try:
            v = claude(prompt, a.judge_model, JUDGE_SYSTEM, P.RUBRIC_SCHEMA, "judge")
            raw_s = float(v["json"].get("score", 0))
            verdicts.append({"item": item, "raw": raw_s, "score": P.round_rubric_score(raw_s),
                             "reason": v["json"].get("reason", "")})
        except (RuntimeError, TypeError, ValueError) as e:
            verdicts.append({"item": item, "score": 0.0, "error": str(e)[:300]})
    row.update(score=sum(v["score"] for v in verdicts) / len(verdicts),
               judge={"method": "rubric", "items": verdicts})
    return row


# --- run / summary ---------------------------------------------------------------------------

def load_inputs(a):
    qs = [json.loads(l) for l in (a.data / "questions.jsonl").open() if l.strip()]
    packets = {}
    for l in a.packets.open():
        r = json.loads(l)
        packets[r["query_id"]] = r
    if a.queries:
        want = set(a.queries.split(","))
        qs = [q for q in qs if q["query_id"] in want]
    if a.limit:
        qs = qs[: a.limit]
    missing = [q["query_id"] for q in qs if q["query_id"] not in packets]
    if missing:
        raise SystemExit(f"{len(missing)} questions have no packet, e.g. {missing[:3]}")
    return qs, packets


def cmd_run(a) -> None:
    a.out.mkdir(parents=True, exist_ok=True)
    load_cache(a.out.parent / ("cache.jsonl" if backend() == "cli" else f"cache_{backend()}.jsonl"))
    qs, packets = load_inputs(a)
    res_path = a.out / "results.jsonl"
    done = set()
    if res_path.exists():
        done = {json.loads(l)["query_id"] for l in res_path.open() if l.strip()}
    todo = [q for q in qs if q["query_id"] not in done]
    (a.out / "config.json").write_text(json.dumps({
        "data": str(a.data), "packets": str(a.packets), "protocol": a.protocol, "arm": a.arm,
        "answer_model": a.answer_model, "judge_model": a.judge_model, "backend": backend(),
        "thinking": "off (MAX_THINKING_TOKENS=0)" if backend() == "cli" else "off (reasoning_effort none)",
        "questions": len(qs)}, indent=1) + "\n")
    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool, res_path.open("a") as out:
        futs = {pool.submit(run_question, q, packets[q["query_id"]], a): q for q in todo}
        for f in as_completed(futs):
            row = f.result()
            with _lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
            n += 1
            if n % 25 == 0:
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s new-cost ${sum(_cost.values()):.3f}",
                      file=sys.stderr, flush=True)
    print(f"done {n} in {time.time() - t0:.0f}s, new cost ${sum(_cost.values()):.3f} {dict(_cost)}", file=sys.stderr)
    summarize(a.out)


def boot_ci(rows: list[dict], key=lambda r: r["score"], reps: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """Mean and 95% cluster-bootstrap CI (resampling conversations)."""
    by_conv = defaultdict(list)
    for r in rows:
        by_conv[r["conversation_id"]].append(key(r))
    convs = list(by_conv)
    if not convs:
        return float("nan"), float("nan"), float("nan")
    mean = sum(key(r) for r in rows) / len(rows)
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        s = n = 0
        for _ in convs:
            xs = by_conv[rng.choice(convs)]
            s += sum(xs)
            n += len(xs)
        stats.append(s / n)
    stats.sort()
    return mean, stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


def failure_class(r: dict) -> str:
    if r.get("error"):
        return "harness_error"
    if r["category"] == "abstention":
        return "abstention_error"
    rec = r.get("gold_recall")
    if rec is None:
        return "no_gold_ids"
    if rec == 0:
        return "retrieval_miss"
    if rec < 1:
        return "partial_retrieval"
    return "answer_or_judge_error"


def summarize(out: Path) -> dict:
    rows = [json.loads(l) for l in (out / "results.jsonl").open() if l.strip()]
    m, lo, hi = boot_ci(rows)
    cats = {}
    for c in sorted({r["category"] for r in rows}):
        rs = [r for r in rows if r["category"] == c]
        cm, clo, chi = boot_ci(rs)
        rc = [r["gold_recall"] for r in rs if r.get("gold_recall") is not None]
        closs = defaultdict(float)
        for r in rs:
            closs[failure_class(r)] += 1 - r["score"]
        cats[c] = {"n": len(rs), "mean": round(cm, 4), "ci95": [round(clo, 4), round(chi, 4)],
                   "gold_recall": round(sum(rc) / len(rc), 3) if rc else None,
                   "loss": {k: round(v / len(rs), 3) for k, v in sorted(closs.items())}}
    loss = defaultdict(float)
    fails = defaultdict(int)
    for r in rows:
        fc = failure_class(r)
        loss[fc] += 1 - r["score"]
        if r["score"] < 0.5:
            fails[fc] += 1
    total_loss = sum(loss.values()) or 1
    rec = [r["gold_recall"] for r in rows if r.get("gold_recall") is not None]
    toks = sorted(r.get("answer_input_tokens") or 0 for r in rows)
    s = {
        "n": len(rows), "errors": sum(1 for r in rows if r.get("error")),
        "score": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "macro": round(sum(c["mean"] for c in cats.values()) / max(1, len(cats)), 4),
        "categories": cats,
        "gold_recall_mean": round(sum(rec) / len(rec), 4) if rec else None,
        "gold_recall_full_frac": round(sum(1 for x in rec if x == 1) / len(rec), 4) if rec else None,
        "loss_share": {k: round(v / total_loss, 3) for k, v in sorted(loss.items())},
        "failed_lt_0p5": dict(sorted(fails.items())),
        "answer_input_tokens_p50": toks[len(toks) // 2] if toks else None,
        "answer_input_tokens_p95": toks[int(len(toks) * 0.95) - 1] if toks else None,
        "context_chars_p50": sorted(r["context_chars"] for r in rows)[len(rows) // 2] if rows else None,
    }
    (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps({k: s[k] for k in ("n", "errors", "score", "ci95", "gold_recall_mean", "loss_share")}))
    return s


def cmd_estimate(a) -> None:
    qs, packets = load_inputs(a)
    chars = [len(context_text(context_items(packets[q["query_id"]], a.arm, q))) for q in qs]
    judge_calls = 0
    for q in qs:
        meta = P.annotations(q["category"], q["raw"])
        if q["category"] == "event_ordering":
            n = len(meta.get("ordering_tested") or [])
            judge_calls += n * (n + 1) // 2  # greedy matching upper bound ~ n(n+1)/2
        else:
            judge_calls += max(1, len(meta.get("rubric") or []))
    print(json.dumps({"questions": len(qs), "answer_calls": len(qs), "judge_calls_upper": judge_calls,
                      "context_tokens_mean": round(sum(chars) / len(chars) / 4)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "estimate"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--packets", type=Path, required=True)
        p.add_argument("--arm", choices=["full", "r8k", "r8ko", "r2k", "none", "oracle"], default="full")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--queries", default="")
        if name == "run":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--protocol", choices=["exabase", "amb"], required=True)
            p.add_argument("--answer-model", default="claude-haiku-5-5")
            p.add_argument("--judge-model", default="claude-haiku-5-5")
            p.add_argument("--concurrency", type=int, default=8)
    p = sub.add_parser("summary")
    p.add_argument("outs", nargs="+", type=Path)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "estimate":
        cmd_estimate(a)
    else:
        for o in a.outs:
            print(o, end=" ")
            summarize(o)


if __name__ == "__main__":
    main()
