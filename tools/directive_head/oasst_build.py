"""Real chat rows from OpenAssistant oasst1 (Apache-2.0), labeled by the gpt-6.1-sol judge, split by conversation.

usage: python -I oasst_build.py --src /Volumes/Build/cortext/directive-head/data/oasst1 \
           --out /Volumes/Build/cortext/directive-head/data_oasst [--merge-with data/rows.jsonl --merged data_mix]
The source directory is data only: this script lives outside it and is run with -I. It reads either the
release's message files (*messages*.jsonl[.gz]) or the Hugging Face parquet shards (*.parquet; needs pyarrow).
Steps
1. Keep messages that are not deleted and not rejected in review, with text <= --max-bytes (480: the engine's
   kMaxTextBytes; longer text is never scored).
2. Split by conversation (message_tree_id, sha1): 70% train, 10% val, 20% test, so no tree spans two splits.
3. Label. Prompter messages go to the judge (judge.SYSTEM definition, batches of --batch, cached in
   OUT/judge_cache.jsonl so reruns are free); assistant messages are label 0 (assistant-voice text is never a
   user's standing directive under the definition) and are not sent.
4. Wrap 60% of rows with a synthetic stream prefix (build_data.wrap; "assistant:" for assistant messages).
Writes OUT/rows.jsonl (build_data schema plus src/tree/role) and OUT/stats.json; with --merge-with, also
MERGED/rows.jsonl = synthetic rows + oasst rows with fresh ids (features must then be recomputed for MERGED).
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, os, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_data  # noqa: E402
import judge  # noqa: E402

LANG_MAP = {"pt-BR": "pt", "pt": "pt", "zh": "zh", "en": "en", "es": "es", "fr": "fr", "de": "de", "ja": "ja",
            "ko": "ko", "hi": "hi", "ar": "ar"}
BATCH_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["labels"],
                "properties": {"labels": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["id", "lasting"],
                    "properties": {"id": {"type": "integer"}, "lasting": {"type": "boolean"}}}}}}


def read_messages(src: Path):
    files = sorted(p for p in src.rglob("*") if p.is_file())
    jl = [p for p in files if "messages" in p.name and p.name.endswith((".jsonl", ".jsonl.gz"))]
    if jl:
        for p in jl:
            op = gzip.open if p.name.endswith(".gz") else open
            with op(p, "rt", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        return
    pq = [p for p in files if p.suffix == ".parquet"]
    if not pq:
        sys.exit(f"no *messages*.jsonl[.gz] or *.parquet under {src}")
    import pyarrow.parquet as papq
    for p in pq:
        for rec in papq.read_table(p).to_pylist():
            yield rec


def split_of(tree: str) -> str:
    k = int(hashlib.sha1(("oasst|" + tree).encode()).hexdigest()[:8], 16) % 100
    return "test" if k < 20 else "val" if k < 30 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-bytes", type=int, default=480)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--judge-limit", type=int, default=0, help="judge at most N batches (smoke tests)")
    ap.add_argument("--merge-with", default="")
    ap.add_argument("--merged", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seen, msgs, drop = set(), [], Counter()
    for m in read_messages(Path(a.src)):
        t = (m.get("text") or "").strip()
        if m.get("deleted") or m.get("review_result") is False:
            drop["deleted_or_rejected"] += 1; continue
        if not t or len(t.encode("utf-8")) > a.max_bytes:
            drop["length"] += 1; continue
        k = build_data.norm(t)
        if not k or k in seen:
            drop["dup"] += 1; continue
        seen.add(k)
        msgs.append({"id_src": m["message_id"], "tree": m.get("message_tree_id") or m["message_id"],
                     "role": m.get("role"), "lang_src": m.get("lang") or "", "raw": t})
    print("kept", len(msgs), dict(drop), file=sys.stderr)

    # judge prompter messages (cached)
    cache_p = out / "judge_cache.jsonl"
    cache = {}
    if cache_p.exists():
        for line in cache_p.open():
            r = json.loads(line); cache[r["id_src"]] = r["lasting"]
    todo = [m for m in msgs if m["role"] == "prompter" and m["id_src"] not in cache]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    if a.judge_limit:
        batches = batches[:a.judge_limit]
    os.environ.setdefault("CPA_REASONING_EFFORT", "low")
    sys.path.insert(0, str(judge.CPA_DIR))
    import cpa_backend as cpa
    system = judge.SYSTEM + (" You will receive several numbered messages; label each one independently and "
                             "return one entry per id.")

    def run(b):
        body = "\n\n".join(f"[{i}] {m['raw']}" for i, m in enumerate(b))
        res = cpa.chat(system, "Messages:\n" + body, model=judge.MODEL, schema=BATCH_SCHEMA, schema_name="labels")
        got = {int(x["id"]): bool(x["lasting"]) for x in res["json"]["labels"]}
        if set(got) != set(range(len(b))):
            raise ValueError(f"judge returned ids {sorted(got)} for {len(b)} messages")
        return [(m["id_src"], got[i]) for i, m in enumerate(b)]

    n_err = 0
    with ThreadPoolExecutor(a.workers) as ex, cache_p.open("a") as fc:
        for fut in [ex.submit(run, b) for b in batches]:
            try:
                for mid, lab in fut.result():
                    cache[mid] = lab
                    fc.write(json.dumps({"id_src": mid, "lasting": lab}) + "\n")
            except Exception as e:  # noqa: BLE001 -- unlabeled batches are skipped and retried on rerun
                n_err += 1
                print("judge batch failed:", type(e).__name__, cpa.redact(str(e))[:160], file=sys.stderr)
    rows = []
    for m in msgs:
        if m["role"] == "prompter":
            if m["id_src"] not in cache:
                drop["unjudged"] += 1; continue
            label, cat = int(cache[m["id_src"]]), "oasst_prompter"
        else:
            label, cat = 0, "oasst_assistant"
        r = {"raw": m["raw"], "label": label, "lang": LANG_MAP.get(m["lang_src"], "other"),
             "lang_src": m["lang_src"], "cat": cat,
             "tmpl": "oasst_prompter" if cat == "oasst_prompter" else "neg_advice.assistant_tip",
             "topic": "oasst", "split": split_of(m["tree"]), "hold": "conversation", "src": "oasst1",
             "tree": m["tree"], "role": m["role"]}
        wrap_in = dict(r, cat="neg_advice" if m["role"] == "assistant" else cat)
        r["text"], r["wrap"] = build_data.wrap(wrap_in)
        r["tmpl"] = "oasst"
        rows.append(r)
    with (out / "rows.jsonl").open("w") as f:
        for i, r in enumerate(rows):
            r["id"] = i
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"rows": len(rows), "dropped": dict(drop), "judge_batch_errors": n_err,
             "by_split_label": Counter(f"{r['split']}/{r['cat']}/{r['label']}" for r in rows),
             "by_lang": Counter(r["lang"] for r in rows),
             "positives_by_lang": Counter(r["lang"] for r in rows if r["label"])}
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    if a.merge_with:
        mo = Path(a.merged); mo.mkdir(parents=True, exist_ok=True)
        syn = [dict(json.loads(l), src="synthetic") for l in open(a.merge_with)]
        with (mo / "rows.jsonl").open("w") as f:
            for i, r in enumerate(syn + rows):
                r["id"] = i
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("merged", len(syn), "+", len(rows), "->", mo / "rows.jsonl", file=sys.stderr)


if __name__ == "__main__":
    main()
