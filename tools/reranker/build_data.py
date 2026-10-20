"""Build reranker training/eval data from a Cortext candidate-pool export (headroom lane format).

usage: python -I build_data.py --ds eval1 --src <headroom>/runs/eval1 --out /Volumes/Build/cortext/reranker/data
Inputs (src): pool.jsonl {query_id, conversation_id, question, ids[] (Cortext pool order)}, memtext.jsonl
{conv, memory_id, members[], media[]}, gold.jsonl (turn / fact gold units, metrics.py rules).
Outputs (out/<ds>/):
  queries.jsonl  {qid, conv, split, question, text_idx, cands: [{mem, members: [text_idx], units: [..]}], n_units}
  texts.jsonl    unique texts (queries + member texts) in index order
Split: test when sha1(conv) % 10 < 3 (by conversation; Dolphin by persona: the persona listed in --test-convs).
"""
import argparse, hashlib, json, sys
from pathlib import Path

HEADROOM_SCRIPTS = Path(__file__).resolve().parents[2].parent / "hillclimb/headroom/scripts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", default=None, help="dir holding headroom metrics.py (gold matchers)")
    ap.add_argument("--test-convs", default="", help="comma list of conv ids forced to test (others train)")
    a = ap.parse_args()
    sys.path.insert(0, a.metrics or str(Path(a.src).resolve().parents[1] / "scripts"))
    from metrics import matchers, norm  # noqa: E402

    src, out = Path(a.src), Path(a.out) / a.ds
    out.mkdir(parents=True, exist_ok=True)
    conv_of = lambda r: r["conversation_id"] if a.ds != "dolphin" else r["query_id"].split("/")[0]
    pool = [json.loads(l) for l in (src / "pool.jsonl").open()]
    gold = {g["query_id"]: g for g in map(json.loads, (src / "gold.jsonl").open())}
    mem = {(r["conv"], r["memory_id"]): r for r in map(json.loads, (src / "memtext.jsonl").open())}
    forced = set(filter(None, a.test_convs.split(",")))
    texts, tix = [], {}

    def ti(t):
        t = t if t.strip() else "."
        if t not in tix:
            tix[t] = len(texts); texts.append(t)
        return tix[t]

    n = 0
    with (out / "queries.jsonl").open("w") as f:
        for p in pool:
            g = gold.get(p["query_id"])
            if g is None:
                continue
            c = conv_of(p)
            if forced:
                split = "test" if c in forced else "train"
            else:
                split = "test" if int(hashlib.sha1(c.encode()).hexdigest(), 16) % 10 < 3 else "train"
            ms = matchers(g)
            img_ts = set(g.get("image_ts", []))
            units = [u for u, _ in ms] + [f"img:{t}" for t in img_ts]
            cands = []
            for i in p["ids"]:
                r = mem.get((c, i), {"members": [], "media": []})
                nt = [norm(x) for x in r["members"]]
                held = [u for u, fn in ms if fn(r["members"], nt)]
                held += [f"img:{m[2]}" for m in r.get("media", []) if m[2] in img_ts]
                cands.append({"mem": i, "members": [ti(x) for x in r["members"]],
                              "media": r.get("media", []), "units": held})
            f.write(json.dumps({"qid": p["query_id"], "conv": c, "split": split, "question": p["question"],
                                "text_idx": ti(p["question"]), "cands": cands,
                                "standing_bytes": sum(len(t) for t in p.get("standing", [])), "n_units": len(units)}) + "\n")
            n += 1
    with (out / "texts.jsonl").open("w") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    print(a.ds, "queries", n, "texts", len(texts))


if __name__ == "__main__":
    main()
