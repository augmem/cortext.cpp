"""Generate the multilingual standing-directive dataset through the CPA proxy (gpt-6.1-sol only).

usage: python -I gen.py --out /Volumes/Build/cortext/directive-head/gen [--workers 12] [--per-call 12] [--limit N]
Writes calls.jsonl (one line per finished generation cell, append-only, resumable). Each cell is
(split, language, category, template, topic); the call plan is a deterministic function of spec.py and
--seed, so a rerun skips finished cells. Prompts are in spec.py (SYSTEM, USER_TMPL, LABEL_RULE).

The CPA client (.claude/hillclimb/cpa_judge/cpa_backend.py) reads its key at call time and never prints
it. Only gabe-codex-{0,1}/gpt-6.1-sol are allowed; anything else is refused here.
"""
from __future__ import annotations

import argparse, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec  # noqa: E402

CPA_DIR = Path(os.environ.get("CPA_BACKEND_DIR",
                              "/Users/gabrielwillen/VSCode/augmem/augmem/cortext.cpp/.claude/hillclimb/cpa_judge"))
ALLOWED = {"gabe-codex-0/gpt-6.1-sol", "gabe-codex-1/gpt-6.1-sol"}


def plan(seed: int) -> list[dict]:
    rng = random.Random(seed)
    train_topics = [t for t in spec.TOPICS if t not in spec.TEST_TOPICS and t not in spec.VAL_TOPICS]
    cells = []
    for lang in spec.LANGS:
        for cat, (label, _, tmpls) in spec.CATEGORIES.items():
            v2 = cat in spec.V2_CATS
            for tid, _, held in tmpls:
                r = random.Random(f"{seed}|{lang}|{tid}") if v2 else rng
                if held:
                    for t in r.sample(spec.TOPICS, 2):
                        cells.append(dict(split="test", hold="template", lang=lang, cat=cat, tmpl=tid, topic=t))
                    continue
                for t in r.sample(train_topics, spec.V2_TRAIN_CELLS if v2 else 4):
                    cells.append(dict(split="train", hold="none", lang=lang, cat=cat, tmpl=tid, topic=t))
                cells.append(dict(split="val", hold="topic", lang=lang, cat=cat, tmpl=tid,
                                  topic=r.choice(sorted(spec.VAL_TOPICS))))
                cells.append(dict(split="test", hold="topic", lang=lang, cat=cat, tmpl=tid,
                                  topic=r.choice(sorted(spec.TEST_TOPICS))))
    for c in cells:
        c["label"] = spec.CATEGORIES[c["cat"]][0]
        c["cell"] = "|".join((c["split"], c["lang"], c["tmpl"], c["topic"]))
    return cells


def prompt(c: dict, n: int) -> str:
    label, desc, tmpls = spec.CATEGORIES[c["cat"]]
    tmpl = next(t for tid, t, _ in tmpls if tid == c["tmpl"])
    return spec.USER_TMPL.format(lang_name=spec.LANGS[c["lang"]], topic=c["topic"], cat_desc=desc, tmpl=tmpl,
                                 speaker=spec.speaker_for(c["cat"], c["tmpl"]), n=n, label_rule=spec.LABEL_RULE[label])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--per-call", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="gabe-codex-0/gpt-6.1-sol")
    a = ap.parse_args()
    if a.model not in ALLOWED:
        sys.exit(f"model {a.model!r} not allowed")
    os.environ.setdefault("CPA_REASONING_EFFORT", "low")
    os.environ.setdefault("CPA_ROTATE", "1")
    sys.path.insert(0, str(CPA_DIR))
    import cpa_backend as cpa  # noqa: E402

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "calls.jsonl"
    done = set()
    if path.exists():
        for line in path.open():
            done.add(json.loads(line)["cell"])
    cells = [c for c in plan(a.seed) if c["cell"] not in done]
    if a.limit:
        cells = cells[:a.limit]
    print(f"{len(done)} done, {len(cells)} to go", flush=True)
    lock = threading.Lock()
    fh = path.open("a")
    t0, n_ok, n_err = time.time(), 0, 0

    def run(c):
        r = cpa.chat(spec.SYSTEM, prompt(c, a.per_call), model=a.model, schema=spec.SCHEMA, schema_name="items")
        items = [str(x.get("text", "")).strip() for x in r["json"].get("items", [])]
        return {**c, "items": [t for t in items if t], "model": r["model"], "prompt_version": 1}

    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run, c): c for c in cells}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except Exception as e:  # noqa: BLE001 -- keep going; the cell is retried on the next run
                n_err += 1
                print("ERR", futs[f]["cell"], type(e).__name__, cpa.redact(str(e))[:200], flush=True)
                continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"{n_ok}/{len(cells)} ok, {n_err} err, {time.time() - t0:.0f}s", flush=True)
    print(f"finished {n_ok} ok {n_err} err {time.time() - t0:.0f}s stats={cpa.STATS}", flush=True)


if __name__ == "__main__":
    main()
