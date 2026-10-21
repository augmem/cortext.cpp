"""Offline keyword-channel study: FTS5 'porter unicode61 remove_diacritics 2' (current) vs
'unicode61 remove_diacritics 2' (no stemmer), BM25 hit@k of gold turns with the engine's query words.

usage: python -I fts5_study.py --data .claude/hillclimb/scorecard/data --out runs/fts5.json
Per conversation, every text signal is one FTS5 row (the engine indexes per LONG_TERM memory; the replay
episodes store one turn per signal). Queries follow src/store/memory_text_index_internal.cpp: QueryWords
(ASCII alnum + non-ASCII bytes form words, lowercase, English stopwords dropped, distinct, <= 16), words
with df > max(64, 0.05 * rows) dropped, the rest OR-ed as quoted strings, ranked by bm25().
Gold: LoCoMo = turns containing an evidence string (adversarial/no-evidence questions skipped);
LongMemEval-S = turns of the answer sessions (session-level hit).
Evaluation only: nothing here is trained.
"""
from __future__ import annotations

import argparse, json, random, re, sqlite3
from pathlib import Path

STOP = set("""a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of
off on once only or other our ours out over own same she should so some such than that the their theirs them then
there these they this those through to too under until up very was we were what when where which while who whom
why will with would you your yours yourself ll re ve don s t d m""".split())
TOKENIZERS = {"porter": "porter unicode61 remove_diacritics 2", "unicode61": "unicode61 remove_diacritics 2"}
KS = (1, 5, 10, 20)


def query_words(text: str, max_terms: int = 16) -> list[str]:
    out, seen = [], set()
    for w in re.findall(rb"[A-Za-z0-9\x80-\xff]+", text.encode("utf-8")):
        lw = bytes(c + 32 if 65 <= c <= 90 else c for c in w).decode("utf-8", "ignore")
        if not lw or lw in STOP or lw in seen:
            continue
        seen.add(lw); out.append(lw)
        if len(out) >= max_terms:
            break
    return out


def rank(con: sqlite3.Connection, n_rows: int, q: str, limit: int) -> list[int]:
    words = query_words(q)
    max_df = max(64, int(0.05 * n_rows))
    kept = []
    for w in words:
        df = con.execute("SELECT count(*) FROM (SELECT rowid FROM t WHERE t MATCH ? LIMIT ?)",
                         ('"' + w.replace('"', "") + '"', max_df + 1)).fetchone()[0]
        if 0 < df <= max_df:
            kept.append(w)
    if not kept:
        return []
    m = " OR ".join('"' + w.replace('"', "") + '"' for w in kept)
    return [r[0] for r in con.execute("SELECT rowid FROM t WHERE t MATCH ? ORDER BY bm25(t) LIMIT ?", (m, limit))]


def load(data: Path, bench: str):
    eps = {}
    for line in (data / bench / "episodes.jsonl").open():
        e = json.loads(line)
        eps[e["episode_id"]] = [s for s in e["signals"] if s.get("modality") == "text"]
    qs = []
    for line in (data / bench / "answer_key.jsonl").open():
        a = json.loads(line)
        sig = eps.get(a["conversation_id"])
        if not sig or a.get("requires_abstention"):
            continue
        if bench == "locomo":
            ev = [e for e in a.get("evidence") or [] if e]
            gold = {i for i, s in enumerate(sig) if any(e in s["text"] for e in ev)}
        else:
            sess = set(a.get("metadata", {}).get("answer_session_ids") or [])
            gold = {i for i, s in enumerate(sig) if s["source_id"].split("/")[-2] in sess}
        if gold:
            qs.append((a["conversation_id"], a["question"], gold, a.get("question_type")))
    return eps, qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benches", default="locomo,lme_s")
    a = ap.parse_args()
    report = {}
    for bench in a.benches.split(","):
        eps, qs = load(Path(a.data), bench)
        by_conv = {}
        for q in qs:
            by_conv.setdefault(q[0], []).append(q)
        hits = {tk: [] for tk in TOKENIZERS}
        for cid, cq in by_conv.items():
            for tk, spec in TOKENIZERS.items():
                con = sqlite3.connect(":memory:")
                con.execute(f"CREATE VIRTUAL TABLE t USING fts5(text, tokenize='{spec}')")
                con.executemany("INSERT INTO t(rowid, text) VALUES (?, ?)",
                                [(i + 1, s["text"]) for i, s in enumerate(eps[cid])])
                n = len(eps[cid])
                for _, question, gold, qt in cq:
                    r = [x - 1 for x in rank(con, n, question, max(KS))]
                    first = next((j for j, x in enumerate(r) if x in gold), None)
                    hits[tk].append(first)
                con.close()
        res = {"questions": len(qs), "conversations": len(by_conv)}
        for tk in TOKENIZERS:
            res[tk] = {f"hit@{k}": sum(1 for f in hits[tk] if f is not None and f < k) / len(qs) for k in KS}
            res[tk]["mrr@20"] = sum(1 / (f + 1) for f in hits[tk] if f is not None) / len(qs)
        rng = random.Random(0)
        for k in KS:
            d = [(int(u is not None and u < k) - int(p is not None and p < k))
                 for p, u in zip(hits["porter"], hits["unicode61"])]
            boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(1000))
            res[f"delta_hit@{k}"] = {"mean": sum(d) / len(d), "ci95": [boots[25], boots[974]],
                                     "unicode61_only": d.count(1), "porter_only": d.count(-1)}
        report[bench] = res
        print(bench, json.dumps(res, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
