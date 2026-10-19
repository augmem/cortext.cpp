"""In-engine packet comparison: gold-turn recall of the hydrated packets, two arms, plus query latency.

usage: python -I packet_recall.py --key <answer_key.jsonl> --base <vdir>/packets.jsonl --arm <vdir>/packets.jsonl
          [--splits <data dir>/queries.jsonl ...] [--json out.json]
Gold rule = tools/beam_e2e/e2e.py gold_recall (turn body's first 200 normalized chars, or first 80, in the
packet text: retrieved + standing items). Rows: all questions with evidence, and train/test when a built
dataset's queries.jsonl gives the split of a query id. Delta CI: paired bootstrap over conversations.
--image-cases (MemLens cases.jsonl): adds image gold recall = share of the gold turns' image shas present in
the packet's media (rows "img:<set>", questions with at least one gold image).
"""
import argparse, json, random, re
from collections import defaultdict

_ws = re.compile(r"\s+")
_prefix = re.compile(r"^\[[^\]]*\]\s*\w+:\s*")


def norm(s):
    return _ws.sub(" ", s).strip().lower()


def recall(evidence, ctx):
    c = norm(ctx)
    hit = 0
    for e in evidence:
        b = norm(_prefix.sub("", e))
        hit += 1 if (b[:200] in c or (len(b) > 80 and b[:80] in c)) else 0
    return hit / len(evidence)


def packet_text(r):
    return "\n".join((m.get("text") or "") for m in r.get("retrieved", []) + r.get("standing", []))


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--splits", action="append", default=[])
    ap.add_argument("--json", default="")
    ap.add_argument("--image-cases", default="")
    a = ap.parse_args()
    key = {}
    for l in open(a.key):
        k = json.loads(l)
        ev = k.get("evidence") or [g["text"] for g in k.get("gold_turns", [])]
        if ev:
            key[k["query_id"]] = (k.get("conversation_id", ""), ev)
    split = {}
    for p in a.splits:
        for l in open(p):
            q = json.loads(l)
            split[q["qid"]] = q["split"]
    load = lambda p: {r["query_id"]: r for r in map(json.loads, open(p))}
    B, A = load(a.base), load(a.arm)
    groups = defaultdict(list)
    if a.image_cases:
        for l in open(a.image_cases):
            c = json.loads(l)
            gt = c["gold_turns"] if isinstance(c["gold_turns"], list) else json.loads(c["gold_turns"])
            shas = {im["sha"] for g in gt for im in (g.get("images") or [])}
            qid = c["query_id"]
            if not shas or qid not in B or qid not in A:
                continue
            got = lambda r: len(shas & {m.get("media_sha") for it in r.get("retrieved", []) + r.get("standing", [])
                                        for m in (it.get("media") or [])}) / len(shas)
            ch = [m.get("id") for m in B[qid].get("retrieved", [])] != [m.get("id") for m in A[qid].get("retrieved", [])]
            for g in ("img:all", "img:" + split.get(qid, "")):
                if g != "img:":
                    groups[g].append((key.get(qid, (qid,))[0], got(B[qid]), got(A[qid]), ch))
    for qid, (conv, ev) in key.items():
        if qid not in B or qid not in A:
            continue
        rb, ra = recall(ev, packet_text(B[qid])), recall(ev, packet_text(A[qid]))
        ch = [m.get("id") for m in B[qid].get("retrieved", [])] != [m.get("id") for m in A[qid].get("retrieved", [])]
        for g in ("all", split.get(qid)):
            if g:
                groups[g].append((conv, rb, ra, ch))
    out = {}
    print("| set | n | base | arm | delta [95% CI] | packets changed |")
    print("|---|---|---|---|---|---|")
    for g, rows in groups.items():
        by = defaultdict(list)
        for conv, rb, ra, _ in rows:
            by[conv].append(ra - rb)
        convs = list(by)
        rng = random.Random(7)
        ds = sorted(sum(x for c in (rng.choice(convs) for _ in convs) for x in by[c]) /
                    max(1, sum(len(by[c]) for c in convs)) for _ in range(2000))
        mb = sum(r[1] for r in rows) / len(rows)
        ma = sum(r[2] for r in rows) / len(rows)
        out[g] = {"n": len(rows), "base": mb, "arm": ma, "ci": [ds[50], ds[1949]], "changed": sum(r[3] for r in rows)}
        print(f"| {g} | {len(rows)} | {mb:.3f} | {ma:.3f} | {ma - mb:+.3f} [{ds[50]:+.3f},{ds[1949]:+.3f}] | {out[g]['changed']} |")
    for name, P in (("base", B), ("arm", A)):
        ms = [r["query_ms"] for r in P.values() if r.get("query_ms") is not None]
        out[name + "_query_ms"] = {"p50": pct(ms, .5), "p95": pct(ms, .95), "n": len(ms)}
        print(f"{name} query_ms p50 {pct(ms, .5):.1f} p95 {pct(ms, .95):.1f} (n={len(ms)})")
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
