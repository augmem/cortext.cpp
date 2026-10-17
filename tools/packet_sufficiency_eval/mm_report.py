#!/usr/bin/env python3
"""Arms table for the MemLens multimodal packet-sufficiency eval.

Reads <flow>/<variant>/{packets.jsonl,results.jsonl,packet_inputs/*} and the
cases (with gold_turns from build_memlens_cases.py) and reports, per arm:

- tier 1: gold-turn recall in the packet (R+W), split into image-evidence and
  text-evidence gold turns. An image-evidence turn is "image recalled" when
  the packet carries that image (media ref sha) or its caption-inline text,
  and "any recalled" when it carries the image or the turn text.
- judged sufficiency (mean over reps per case, then over cases, 95% CI),
  misleading, abstain-safe, by stratum.
- realtime: ingest ms per text turn vs image turn (turn = its text signal
  plus its image signals), query ms, media bytes stored per image.

Usage: mm_report.py --flow DIR --data DIR --arm v0=text --arm v1=caption ...
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(l) for l in fh if l.strip()]


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def mean_ci(xs: list[float]) -> tuple[float, float, int]:
    n = len(xs)
    if not n:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, 1.96 * math.sqrt(var / n), n


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def tier1(cases: list[dict], packets: dict[str, dict], scope: str) -> dict:
    keys = ("retrieved", "working") if scope == "RW" else ("retrieved",)
    c = defaultdict(int)
    for case in cases:
        pkt = packets.get(case["query_id"])
        if pkt is None:
            continue
        items = [m for k in keys for m in pkt.get(k, [])]
        text = norm(" ".join(m.get("text", "") for m in items))
        shas = {r.get("media_sha") for m in items for r in m.get("media", [])}
        for g in case.get("gold_turns", []):
            text_hit = bool(g["body_probe"]) and norm(g["body_probe"]) in text
            if g["images"]:
                img_hit = any(
                    i["sha"] in shas
                    or (i["caption"] and norm(f"Shared image caption: {i['caption']}") in text)
                    for i in g["images"]
                )
                c["img_turns"] += 1
                c["img_turn_image_hit"] += img_hit
                c["img_turn_text_hit"] += text_hit
                c["img_turn_any_hit"] += img_hit or text_hit
            else:
                c["txt_turns"] += 1
                c["txt_turn_hit"] += text_hit
    r = lambda a, b: c[a] / c[b] if c[b] else float("nan")
    return {
        "image_turns": c["img_turns"],
        "image_recall_image": r("img_turn_image_hit", "img_turns"),
        "image_recall_text": r("img_turn_text_hit", "img_turns"),
        "image_recall_any": r("img_turn_any_hit", "img_turns"),
        "text_turns": c["txt_turns"],
        "text_recall": r("txt_turn_hit", "txt_turns"),
    }


def judged(vdir: Path) -> dict:
    rows = load_jsonl(vdir / "results.jsonl")
    per = defaultdict(lambda: defaultdict(list))
    stratum = {}
    for r in rows:
        stratum[r["prompt_id"]] = r["tags"][1]
        for k, v in r["grade"].items():
            per[r["prompt_id"]][k].append(v)
    def coll(metric, strata=None):
        return [sum(d[metric]) / len(d[metric]) for pid, d in per.items()
                if metric in d and (strata is None or stratum[pid] in strata)]
    out = {"rows": len(rows), "cases": len(per),
           "judge_cost_usd": sum((r["perf"].get("judge_cost_usd") or 0) for r in rows),
           "errors": len(load_jsonl(vdir / "errors.jsonl")),
           "packet_images_mean": (sum(r["perf"].get("packet_images", 0) for r in rows) / len(rows)) if rows else 0}
    for name, strata in (("all", None), ("image", {"image-evidence"}), ("control", {"text-evidence-control"})):
        for metric in ("sufficiency", "sufficient"):
            out[f"{metric}_{name}"] = mean_ci(coll(metric, strata))
    out["misleading_all"] = mean_ci(coll("misleading"))
    out["abstain_safe"] = mean_ci(coll("abstain_safe"))
    types = {r["prompt_id"]: r["tags"][0] for r in rows}
    for t in sorted(set(types.values())):
        xs = [sum(d["sufficiency"]) / len(d["sufficiency"]) for pid, d in per.items()
              if types[pid] == t and "sufficiency" in d]
        if xs:
            out[f"by_type/{t}"] = mean_ci(xs)
    return out


def realtime(vdir: Path) -> dict:
    sub = vdir / "packet_inputs"
    log = load_jsonl(sub / "ingest.jsonl")
    turns = defaultdict(lambda: {"ms": 0.0, "image": False, "img_ms": 0.0})
    for row in log:
        sid = row["source_id"]
        base = sid.rsplit("/img", 1)[0] if row["modality"] == "image" else sid
        t = turns[(row["episode_id"], base)]
        t["ms"] += row["ms"]
        if row["modality"] == "image":
            t["image"] = True
            t["img_ms"] += row["ms"]
    text_turn = [t["ms"] for t in turns.values() if not t["image"]]
    image_turn = [t["ms"] for t in turns.values() if t["image"]]
    image_sig = [r["ms"] for r in log if r["modality"] == "image"]
    image_enc = [r["encode_ms"] for r in log if r["modality"] == "image"]
    summary_path = sub / "runner.summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    n_img = sum(1 for r in log if r["modality"] == "image")
    media = (summary.get("media_bytes_ingested") or {}).get("image", 0)
    packets = load_jsonl(vdir / "packets.jsonl")
    q = [p["query_ms"] for p in packets]
    return {
        "text_turn_ms_p50": pct(text_turn, .5), "text_turn_ms_p95": pct(text_turn, .95),
        "image_turn_ms_p50": pct(image_turn, .5), "image_turn_ms_p95": pct(image_turn, .95),
        "image_signal_ms_p50": pct(image_sig, .5), "image_signal_ms_p95": pct(image_sig, .95),
        "image_encode_ms_p50": pct(image_enc, .5), "image_encode_ms_p95": pct(image_enc, .95),
        "query_ms_p50": pct(q, .5), "query_ms_p95": pct(q, .95),
        "image_signals": n_img,
        "images_filtered": (summary.get("filtered_signals") or {}).get("image", 0),
        "media_bytes_per_image": media / n_img if n_img else 0,
        "packets_with_media": sum(1 for p in packets if any(m.get("media") for k in ("retrieved", "working") for m in p.get(k, []))),
        "packets": len(packets),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flow", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="dir holding <arm>/cases.jsonl")
    ap.add_argument("--arm", action="append", required=True, help="variant=armdir, e.g. v0=text")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    report = {}
    for spec in args.arm:
        variant, arm = spec.split("=", 1)
        vdir = args.flow / variant
        cases = load_jsonl(args.data / arm / "cases.jsonl")
        packets = {p["query_id"]: p for p in load_jsonl(vdir / "packets.jsonl")}
        report[f"{variant}:{arm}"] = {
            "tier1_RW": tier1(cases, packets, "RW"),
            "tier1_R": tier1(cases, packets, "R"),
            "judged": judged(vdir),
            "realtime": realtime(vdir),
        }
    f = lambda x: "  -  " if x != x else f"{x:.3f}"
    ci = lambda t: "  -  " if t[2] == 0 else f"{t[0]:.3f}±{t[1]:.3f} (n={t[2]})"
    print("| arm | img-turn recall: image / any | text-turn recall | suff (image-ev) | suff (all ans.) | suff (controls) | misleading | abstain-safe | text turn p50/p95 ms | image turn p50/p95 ms | query p50/p95 ms | stored B/image |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in report.items():
        t, j, rt = r["tier1_RW"], r["judged"], r["realtime"]
        print(f"| {name} | {f(t['image_recall_image'])} / {f(t['image_recall_any'])} | {f(t['text_recall'])} | "
              f"{ci(j['sufficiency_image'])} | {ci(j['sufficiency_all'])} | {ci(j['sufficiency_control'])} | {ci(j['misleading_all'])} | {ci(j['abstain_safe'])} | "
              f"{rt['text_turn_ms_p50']:.1f}/{rt['text_turn_ms_p95']:.1f} | {rt['image_turn_ms_p50']:.1f}/{rt['image_turn_ms_p95']:.1f} | "
              f"{rt['query_ms_p50']:.1f}/{rt['query_ms_p95']:.1f} | {rt['media_bytes_per_image']:.0f} |")
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
