"""Grade rule/head disagreements on real streams with the gpt-6.1-sol judge (evaluation only).

usage: python -I judge.py --dis RUNS/disagreements.jsonl --out RUNS/judged.jsonl [--n 50] [--seed 0]
       python -I judge.py --audit DATA/rows.jsonl --out RUNS/label_audit.jsonl [--n 200]   (generator-label agreement on test rows)
       python -I judge.py --bands STREAMS_EVAL_DIR --out RUNS/bands.jsonl [--n 25]
           (operating-point study: rule flags the head would veto / keep, and head-only flags in high score bands)
Samples n disagreements, half where only the rule fires and half where only the head fires, spread over
streams round-robin, and asks the judge whether the turn states a lasting directive or preference.
Only gabe-codex-{0,1}/gpt-6.1-sol (reasoning low) is used; the CPA key is read at call time and never printed.
"""
from __future__ import annotations

import argparse, json, os, random, sys
from collections import defaultdict
from pathlib import Path

CPA_DIR = Path(os.environ.get("CPA_BACKEND_DIR",
                              "/Users/gabrielwillen/VSCode/augmem/augmem/cortext.cpp/.claude/hillclimb/cpa_judge"))
MODEL = "gabe-codex-0/gpt-6.1-sol"
SYSTEM = ("You label single messages from a long chat history between a user and an AI assistant (or between "
          "people). Decide whether the message states a LASTING directive or preference: something the speaker wants "
          "applied to future answers, suggestions or behaviour beyond the current request (e.g. 'Always cite "
          "versions', 'From now on reply in French', 'I prefer aisle seats', 'I'm vegetarian, keep recipes "
          "meat-free'). Not lasting: one-off task instructions, idioms ('never mind'), questions, reported speech "
          "about other people, narration of habits or facts, advice given by an assistant, or preferences limited "
          "to the moment. Timestamps and speaker prefixes are metadata.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["lasting", "reason"],
          "properties": {"lasting": {"type": "boolean"}, "reason": {"type": "string"}}}


def sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["stream"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    out, keys = [], sorted(by)
    while len(out) < n and any(by.values()):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dis", default="")
    ap.add_argument("--audit", default="")
    ap.add_argument("--bands", default="", help="dir with disagreements.jsonl and rule_flags.jsonl from evaluate.py")
    ap.add_argument("--band-lo", type=float, default=0.0)
    ap.add_argument("--band-hi", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("CPA_REASONING_EFFORT", "low")
    sys.path.insert(0, str(CPA_DIR))
    import cpa_backend as cpa
    rng = random.Random(a.seed)
    from concurrent.futures import ThreadPoolExecutor
    if a.audit:
        rows = [r for r in map(json.loads, open(a.audit)) if r["split"] == "test"]
        pick = rng.sample(rows, a.n)

        def grade_a(r):
            res = cpa.chat(SYSTEM, "Message:\n" + r["text"][:4000], model=MODEL, schema=SCHEMA, schema_name="label")
            return {**r, "judge": res["json"]["lasting"], "reason": res["json"]["reason"]}
        with ThreadPoolExecutor(10) as ex:
            out = list(ex.map(grade_a, pick))
        with open(a.out, "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        agree = sum(int(r["judge"]) == r["label"] for r in out)
        by = defaultdict(lambda: [0, 0])
        for r in out:
            by[r["cat"]][0] += int(int(r["judge"]) == r["label"]); by[r["cat"]][1] += 1
        summ = {"n": len(out), "agree": agree, "agree_rate": agree / len(out),
                "pos_judged_lasting": sum(r["judge"] for r in out if r["label"] == 1),
                "pos": sum(r["label"] for r in out), "by_cat": dict(by)}
        Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
        print(json.dumps(summ, indent=1))
        return
    if a.bands:
        b = Path(a.bands)
        rf = [json.loads(l) for l in open(b / "rule_flags.jsonl")]
        dis = [json.loads(l) for l in open(b / "disagreements.jsonl")]
        groups = {f"rule_flag_head_below_{a.band_lo}": [r for r in rf if r["score"] < a.band_lo],
                  f"rule_flag_head_at_or_above_{a.band_lo}": [r for r in rf if r["score"] >= a.band_lo],
                  f"head_only_score_at_or_above_{a.band_hi}": [r for r in dis if r["head"] and r["score"] >= a.band_hi]}
        picks = [(g, r) for g, rs in groups.items() for r in sample(rs, a.n, rng)]

        def grade_b(gr):
            g, r = gr
            res = cpa.chat(SYSTEM, "Message:\n" + r["text"][:4000], model=MODEL, schema=SCHEMA, schema_name="label")
            return {**r, "group": g, "judge": res["json"]["lasting"], "reason": res["json"]["reason"]}
        with ThreadPoolExecutor(10) as ex:
            out = list(ex.map(grade_b, picks))
        with open(a.out, "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summ = {g: {"population": len(groups[g]), "n": sum(r["group"] == g for r in out),
                    "judge_lasting": sum(r["judge"] for r in out if r["group"] == g)} for g in groups}
        Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
        print(json.dumps(summ, indent=1))
        return
    rows = [json.loads(l) for l in open(a.dis)]
    pick = (sample([r for r in rows if r["rule"]], a.n // 2, rng)
            + sample([r for r in rows if r["head"]], a.n - a.n // 2, rng))

    def grade(r):
        res = cpa.chat(SYSTEM, "Message:\n" + r["text"][:4000], model=MODEL, schema=SCHEMA, schema_name="label")
        return {**r, "judge": res["json"]["lasting"], "reason": res["json"]["reason"], "judge_model": res["model"]}

    with ThreadPoolExecutor(10) as ex:
        out = list(ex.map(grade, pick))
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ro = [r for r in out if r["rule"]]; ho = [r for r in out if r["head"]]
    summ = {"rule_only": {"n": len(ro), "judge_lasting": sum(r["judge"] for r in ro)},
            "head_only": {"n": len(ho), "judge_lasting": sum(r["judge"] for r in ho)},
            "by_stream": {}}
    for r in out:
        s = summ["by_stream"].setdefault(r["stream"], {"rule_only": [0, 0], "head_only": [0, 0]})
        k = "rule_only" if r["rule"] else "head_only"
        s[k][0] += int(r["judge"]); s[k][1] += 1
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
