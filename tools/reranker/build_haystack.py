"""Engine-free training pools from LongMemEval-format haystacks (LongMemEval-S, MIT; MemLens, CC-BY-4.0 with images).

usage: python -I build_haystack.py --name lme|memlens --src <dataset json> [--image-root DIR]
          [--exclude queries.jsonl ...] [--exclude-ids file] --out /Volumes/Build/cortext/reranker/data
Excluded: questions of any --exclude dataset (conversation ids "<bench>:<qid>") and --exclude-ids (one id per line).
Memory = 3 consecutive turns of one session; members = "[date] role: content" text signals plus one image
signal per attached image (MemLens), like the engine's image-arm ingest. Pool = top 50 memories by
max-member AIST 256-d cosine to the question (Cortext order stand-in; cross-modal for image members).
Gold units = has_answer text turns ("t:<date>|<turn>") and their images ("img:<file>").
Split: test when sha1(qid) % 10 < 2. Items in texts.jsonl: a string (text) or {"image", "caption"}.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aist_torch as A  # noqa: E402

GROUP, POOL = 3, 50


def images_of(t):
    v = t.get("images") or []
    if isinstance(v, str):
        v = json.loads(v) if v.strip().startswith("[") else []
    return [x for x in v if isinstance(x, dict) and x.get("file")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--image-root", default="")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--exclude-ids", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    excl = set()
    for p in a.exclude:
        for l in open(p):
            c = json.loads(l)["conv"]
            excl.add(c.split(":", 1)[1])
    for p in a.exclude_ids:
        excl |= {x.strip() for x in open(p) if x.strip()}
    D = [d for d in json.load(open(a.src)) if d["question_id"] not in excl]
    print("questions", len(D), "excluded ids", len(excl), file=sys.stderr)
    items, tix = [], {}

    def ti(x):
        k = json.dumps(x, sort_keys=True)
        if k not in tix:
            tix[k] = len(items); items.append(x)
        return tix[k]

    per_q = []
    for d in D:
        mems = []
        for date, sess in zip(d["haystack_dates"], d["haystack_sessions"]):
            for g in range(0, len(sess), GROUP):
                mem, units = [], []
                for k, t in enumerate(sess[g:g + GROUP]):
                    mem.append(ti(f"[{date}] {t['role']}: {t['content']}"))
                    gold = str(t.get("has_answer")) == "True"
                    if gold:
                        units.append(f"t:{date}|{g + k}")
                    for im in images_of(t):
                        mem.append(ti({"image": str(Path(a.image_root) / im["file"]), "caption": im.get("blip_caption", "")}))
                        if gold:
                            units.append(f"img:{im['file']}")
                mems.append((mem, units))
        per_q.append((d, mems))
    qidx = [ti(d["question"]) for d in D]
    sem = semantic(items, a.device)
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    keep, rows = {}, []
    for (d, mems), qi in zip(per_q, qidx):
        q = sem[qi]
        sc = [max(float(sem[t] @ q) for t in mt) for mt, _ in mems]
        top = sorted(range(len(mems)), key=lambda k: -sc[k])[:POOL]
        units = sorted({u for _, us in mems for u in us})
        cands = [{"mem": k, "members": mems[k][0], "media": [], "units": mems[k][1]} for k in top]
        split = "test" if int(hashlib.sha1(d["question_id"].encode()).hexdigest(), 16) % 10 < 2 else "train"
        rows.append({"qid": d["question_id"], "conv": f"{a.name}:{d['question_id']}", "split": split,
                     "question": d["question"], "text_idx": qi, "cands": cands, "standing_bytes": 0,
                     "n_units": len(units), "qtype": d.get("question_type", "")})
        for t in [qi] + [t for c in cands for t in c["members"]]:
            keep.setdefault(t, len(keep))
    with (out / "queries.jsonl").open("w") as f:
        for r in rows:
            r["text_idx"] = keep[r["text_idx"]]
            for c in r["cands"]:
                c["members"] = [keep[t] for t in c["members"]]
            f.write(json.dumps(r) + "\n")
    with (out / "texts.jsonl").open("w") as f:
        for t in sorted(keep, key=keep.get):
            f.write(json.dumps(items[t]) + "\n")
    print(a.name, "queries", len(rows), "items", len(keep))


def semantic(items, dev):
    """256-d retrieval slice (L2) of every item via the PyTorch AIST towers."""
    sem = np.zeros((len(items), 256), dtype=np.float32)
    t0 = time.time()
    tidx = [i for i, x in enumerate(items) if isinstance(x, str)]
    if tidx:
        tok, m = A.load_text(dev)
        ids = {i: tok.encode(items[i]) for i in tidx}
        order = sorted(tidx, key=lambda i: -len(ids[i]))
        k = 0
        while k < len(order):
            L = len(ids[order[k]])
            idx = order[k:k + max(1, 16384 // L)]
            x = torch.zeros(len(idx), L, dtype=torch.long); mk = torch.zeros_like(x)
            for j, i in enumerate(idx):
                x[j, :len(ids[i])] = torch.tensor(ids[i]); mk[j, :len(ids[i])] = 1
            _, s = m(x.to(dev), mk.to(dev))
            sem[idx] = torch.nn.functional.normalize(s[:, :256].float(), dim=-1).cpu().numpy()
            k += len(idx)
        del m
    iidx = [i for i, x in enumerate(items) if not isinstance(x, str)]
    if iidx:
        m = A.load_image(dev)
        for k in range(0, len(iidx), 32):
            idx = iidx[k:k + 32]
            x = torch.from_numpy(np.stack([A.preprocess_image(A.read_rgb(items[i]["image"])) for i in idx]))
            _, s = m(x.to(dev))
            sem[idx] = torch.nn.functional.normalize(s[:, :256].float(), dim=-1).cpu().numpy()
    print("semantic", len(items), f"{time.time() - t0:.0f}s", file=sys.stderr)
    return sem


if __name__ == "__main__":
    main()
