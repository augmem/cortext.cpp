"""Build the train/val/test dataset from gen.py output.

usage: python -I build_data.py --gen /Volumes/Build/cortext/directive-head/gen --out /Volumes/Build/cortext/directive-head/data
Writes rows.jsonl ({id, text, raw, label, lang, cat, tmpl, topic, split, hold, wrap}) and stats.json.

Steps
1. Explode generation cells into rows; strip list numbering and wrapping quotes.
2. Script check: a row in ja/zh/ko/hi/ar needs >= 30% of its letters in that language's script; a row in a
   Latin-script language needs >= 90% Latin letters. Failing rows are dropped.
3. Exact dedupe on a normalized form (NFKC, casefold, punctuation and spaces removed). A text that occurs in
   several splits is kept in the held-out split only (test > val > train).
4. Near-dup removal: char 3-5-gram TF-IDF cosine >= 0.9 between a train row and any val/test row drops the
   train row, so held-out templates and topics never leak through paraphrase copies.
5. Stream-format wrapping: 60% of rows (deterministic by hash) get a synthetic timestamp and/or speaker
   prefix in the formats memory streams use ("[2023-03-05T09:30:00-05:00] ", "[3:14 pm on 2 June, 2023] Ana: ",
   "[2024/01/05 (Fri) 10:01] user: ", "user: "). Labels never depend on the wrapper; assistant-voice rows
   get "assistant:" when a speaker is used. Dates and names are synthetic.
"""
from __future__ import annotations

import argparse, hashlib, json, re, sys, unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPTS = {"ja": [(0x3040, 0x30FF), (0x4E00, 0x9FFF)], "zh": [(0x4E00, 0x9FFF)], "ko": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
           "hi": [(0x0900, 0x097F)], "ar": [(0x0600, 0x06FF), (0x0750, 0x077F)]}
SPLIT_RANK = {"test": 0, "val": 1, "train": 2}
NAMES = ["Ana", "Ben", "Chen", "Dana", "Elif", "Farid", "Grace", "Hiro", "Iris", "Jonas", "Kemal", "Lena", "Mina", "Noah",
         "Olu", "Priya", "Rafael", "Sara", "Tomas", "Yuki"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16)


def clean(t: str) -> str:
    t = t.strip()
    t = re.sub(r"^\s*(\d+[.)]|[-*•])\s+", "", t)
    if len(t) >= 2 and t[0] in "\"'“«「『" and t[-1] in "\"'”»」』":
        t = t[1:-1].strip()
    return t


def script_ok(t: str, lang: str) -> bool:
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    if lang in SCRIPTS:
        n = sum(1 for c in letters if any(a <= ord(c) <= b for a, b in SCRIPTS[lang]))
        return n / len(letters) >= 0.3
    n = sum(1 for c in letters if ord(c) < 0x250)
    return n / len(letters) >= 0.9


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).casefold()
    return "".join(c for c in t if c.isalnum())


def wrap(row: dict) -> tuple[str, str]:
    k = h("wrap|" + row["raw"])
    if k % 100 >= 60:
        return row["raw"], "none"
    k //= 100
    speaker_role = "assistant" if row["cat"] == "neg_advice" and row["tmpl"] != "neg_advice.generic_rule" else "user"
    y, mo, d, hh, mi = 2022 + k % 4, 1 + (k >> 3) % 12, 1 + (k >> 7) % 28, (k >> 12) % 24, (k >> 17) % 60
    name = NAMES[(k >> 23) % len(NAMES)] if speaker_role == "user" else "Assistant"
    style = (k >> 29) % 5
    if style == 0:
        return f"[{y}-{mo:02d}-{d:02d}T{hh:02d}:{mi:02d}:00-05:00] {row['raw']}", "iso"
    if style == 1:
        h12 = hh % 12 or 12
        return f"[{h12}:{mi:02d} {'am' if hh < 12 else 'pm'} on {d} {MONTHS[mo - 1]}, {y}] {name}: {row['raw']}", "locomo_like"
    if style == 2:
        return f"[{y}/{mo:02d}/{d:02d} ({DAYS[k % 7]}) {hh:02d}:{mi:02d}] {speaker_role}: {row['raw']}", "dated_role"
    if style == 3:
        return f"{speaker_role}: {row['raw']}", "role"
    return f"{name}: {row['raw']}", "name"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--near-dup", type=float, default=0.9)
    a = ap.parse_args()
    rows, drop = [], Counter()
    for line in (Path(a.gen) / "calls.jsonl").open():
        c = json.loads(line)
        for t in c["items"]:
            t = clean(t)
            if not (3 <= len(t) <= 600):
                drop["length"] += 1; continue
            if not script_ok(t, c["lang"]):
                drop["script"] += 1; continue
            rows.append({k: c[k] for k in ("label", "lang", "cat", "tmpl", "topic", "split", "hold")} | {"raw": t})
    print("exploded", len(rows), dict(drop), file=sys.stderr)

    # exact dedupe, held-out split wins
    best: dict[str, dict] = {}
    for r in rows:
        k = norm(r["raw"])
        if not k:
            continue
        o = best.get(k)
        if o is None or SPLIT_RANK[r["split"]] < SPLIT_RANK[o["split"]]:
            if o is not None:
                drop["exact_dup"] += 1
            best[k] = r
        else:
            drop["exact_dup"] += 1
        if o is not None and o["label"] != r["label"]:
            drop["label_conflict"] += 1
    rows = list(best.values())

    # near-dup: train rows close to any held-out row are removed
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    X = vec.fit_transform([norm(r["raw"]) for r in rows])
    tr = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    ho = np.array([i for i, r in enumerate(rows) if r["split"] != "train"])
    kill = set()
    for s in range(0, len(tr), 2000):
        blk = tr[s:s + 2000]
        sim = (X[blk] @ X[ho].T).max(axis=1).toarray().ravel()
        kill.update(int(blk[j]) for j in np.nonzero(sim >= a.near_dup)[0])
    drop["near_dup_train"] = len(kill)
    rows = [r for i, r in enumerate(rows) if i not in kill]

    rows.sort(key=lambda r: h(r["raw"]))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.jsonl").open("w") as f:
        for i, r in enumerate(rows):
            r["id"] = i
            r["text"], r["wrap"] = wrap(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"rows": len(rows), "dropped": dict(drop),
             "by_split_label": Counter(f"{r['split']}/{r['hold']}/{r['label']}" for r in rows),
             "by_lang": Counter(r["lang"] for r in rows),
             "by_lang_split": Counter(f"{r['lang']}/{r['split']}" for r in rows),
             "by_cat": Counter(r["cat"] for r in rows)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
