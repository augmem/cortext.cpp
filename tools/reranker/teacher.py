"""Text cross-encoder teacher scores per (query, member text) pair of a built dataset.

usage: python -I teacher.py --data <dir> --model cross-encoder/ms-marco-MiniLM-L6-v2 --key minilm [--device mps]
Writes teacher_<key>.json: {qid: [[logit per member] per candidate]} (empty list for members-less candidates).
"""
import argparse, json, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    a = ap.parse_args()
    from sentence_transformers import CrossEncoder
    d = Path(a.data)
    texts = [json.loads(l) for l in (d / "texts.jsonl").open()]
    Q = [json.loads(l) for l in (d / "queries.jsonl").open()]
    m = CrossEncoder(a.model, device=a.device, max_length=a.max_len)
    out, t0 = {}, time.time()
    for n, q in enumerate(Q):
        pairs, shape = [], []
        for c in q["cands"]:
            shape.append(len(c["members"]))
            # image members are scored through their caption (text teacher -> cross-modal soft label)
            pairs += [(q["question"], texts[i] if isinstance(texts[i], str) else (texts[i].get("caption") or "image"))
                      for i in c["members"]]
        s = [float(x) for x in m.predict(pairs, batch_size=a.bs, show_progress_bar=False)] if pairs else []
        k, row = 0, []
        for w in shape:
            row.append(s[k:k + w]); k += w
        out[q["qid"]] = row
        if n % 50 == 0:
            print(f"{n}/{len(Q)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    (d / f"teacher_{a.key}.json").write_text(json.dumps(out))
    print(d.name, a.key, len(out), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
