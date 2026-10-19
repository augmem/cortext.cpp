"""Paired BEAM e2e comparison (tools/beam_e2e results.jsonl rows: query_id, conversation_id, category, score,
gold_recall), overall, per category, and on held-out conversations.

usage: python -I e2e_cmp.py <base results.jsonl> <arm results.jsonl> [--heldout queries.jsonl]
--heldout: a built reranker dataset (tools/reranker/build_data.py); its test-split conversations form the
held-out row. CI: paired bootstrap over conversations (2000 reps).
"""
import argparse, json, random
from collections import defaultdict


def boot(rows, reps=2000, seed=7):
    by = defaultdict(list)
    for c, d in rows:
        by[c].append(d)
    convs = list(by)
    rng = random.Random(seed)
    ms = []
    for _ in range(reps):
        s = [x for c in (rng.choice(convs) for _ in convs) for x in by[c]]
        ms.append(sum(s) / len(s))
    ms.sort()
    return ms[int(.025 * reps)], ms[int(.975 * reps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("arm")
    ap.add_argument("--heldout", default="")
    a = ap.parse_args()
    load = lambda p: {r["query_id"]: r for r in map(json.loads, open(p)) if r.get("score") is not None}
    B, A = load(a.base), load(a.arm)
    test = set()
    if a.heldout:
        test = {json.loads(l)["conv"] for l in open(a.heldout) if json.loads(l)["split"] == "test"}
    groups = defaultdict(list)
    for q in sorted(set(B) & set(A)):
        b, r = B[q], A[q]
        row = (b["conversation_id"], b["score"], r["score"], b.get("gold_recall"), r.get("gold_recall"))
        groups["all"].append(row)
        groups[b["category"]].append(row)
        if test:
            groups["heldout" if b["conversation_id"] in test else "trained"].append(row)
    for g, rows in groups.items():
        n = len(rows)
        mb, ma = sum(x[1] for x in rows) / n, sum(x[2] for x in rows) / n
        lo, hi = boot([(x[0], x[2] - x[1]) for x in rows])
        gr = [(x[3], x[4]) for x in rows if x[3] is not None and x[4] is not None]
        grs = f" gold_recall {sum(x for x, _ in gr) / len(gr):.3f}->{sum(y for _, y in gr) / len(gr):.3f}" if gr else ""
        print(f"{g:26s} n={n:4d} base={mb:.3f} arm={ma:.3f} d={ma - mb:+.3f} [{lo:+.3f},{hi:+.3f}]{grs}")


if __name__ == "__main__":
    main()
