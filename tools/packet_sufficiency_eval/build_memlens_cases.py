#!/usr/bin/env python3
"""Build the MemLens-32K multimodal packet-sufficiency case set (four arms).

MemLens (CC-BY-4.0) is LongMemEval-shaped: one haystack per question, turns
with `has_answer` gold and `images: [{file, blip_caption, image_url}]`. The
dataset is untrusted input: run this with `python -I` from outside the data
directory (Pillow is needed, e.g. `uv run --no-project --with pillow python -I`).

Draws a seeded stratified sample of image-evidence questions plus refusals and
the text-only-evidence controls, then writes one data dir per arm:

  <out>/<arm>/{cases,answer_key,episodes}.jsonl   arm in ARMS
  <out>/media/<sha>.rgb     decoded RGB pixels (max side --max-side) fed to
                            ProcessImage; the engine API takes decoded pixels
  <out>/media/<sha>.jpg     judge thumbnail (max side --thumb-side)
  <out>/media_index.json    sha -> {file, bytes, width, height, caption}

Arms (same questions, same turns, same timestamps):
  text      image dropped (the `<image>` marker is stripped everywhere)
  caption   BLIP caption appended to the turn text, as LoCoMo prep does
  image     the turn text plus an image signal (pixels + original JPEG bytes)
  caption_image   caption-inline text plus the image signal

Image signals carry `media_path` (the original compressed file the runner
passes as Media bytes), `media_sha`, `caption`, and `judge_path`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ARMS = ("text", "caption", "image", "caption_image")
ALLOC = {
    "information_extraction": 28,
    "temporal_reasoning": 24,
    "multi_session_reasoning": 20,
    "knowledge_update": 18,
    "answer_refusal": 10,
}
CAPTION_FMT = " Shared image caption: {}."


def to_ms(day: str) -> int:
    d, _, rest = day.partition(" (")
    clock = rest.split(") ", 1)[1]
    dt = datetime.strptime(f"{d} {clock}", "%Y/%m/%d %H:%M")
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def norm_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def gold_kind(q: dict) -> str:
    img = txt = 0
    for s in q["haystack_sessions"]:
        for t in s:
            if t.get("has_answer"):
                if t.get("images"):
                    img += 1
                else:
                    txt += 1
    if img:
        return "image"
    return "text" if txt else "none"


def pick(data: list[dict], rng: random.Random) -> list[tuple[dict, str]]:
    picked = []
    for qtype, n in ALLOC.items():
        want = "none" if qtype == "answer_refusal" else "image"
        pool = sorted(
            (q for q in data if q["question_type"] == qtype and gold_kind(q) == want),
            key=lambda q: q["question_id"],
        )
        if len(pool) < n:
            raise SystemExit(f"stratum {qtype} has {len(pool)}, need {n}")
        picked += [(q, "refusal" if want == "none" else "image-evidence") for q in rng.sample(pool, n)]
    controls = sorted(
        (q for q in data if q["question_type"] != "answer_refusal" and gold_kind(q) == "text"),
        key=lambda q: q["question_id"],
    )
    picked += [(q, "text-evidence-control") for q in controls]
    return picked


def prepare_image(src: Path, media: Path, max_side: int, thumb_side: int, index: dict) -> dict:
    raw = src.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()[:20]
    if sha in index:
        return index[sha]
    t0 = time.perf_counter()
    im = Image.open(io.BytesIO(raw))
    im.draft("RGB", (max_side, max_side))
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side), Image.BICUBIC)
    decode_ms = (time.perf_counter() - t0) * 1000
    pix = media / f"{sha}.rgb"
    pix.write_bytes(im.tobytes())
    th = im.copy()
    th.thumbnail((thumb_side, thumb_side), Image.BICUBIC)
    jpg = media / f"{sha}.jpg"
    th.save(jpg, "JPEG", quality=85)
    index[sha] = {
        "sha": sha,
        "media_path": str(src.resolve()),
        "media_bytes": len(raw),
        "mimetype": "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg",
        "pixels_path": str(pix.resolve()),
        "width": im.width,
        "height": im.height,
        "channels": 3,
        "judge_path": str(jpg.resolve()),
        "decode_ms": round(decode_ms, 2),
    }
    return index[sha]


def build_episode(q: dict, img_root: Path, media: Path, args, index: dict):
    """Return {arm: episode}, the gold-turn list, and dated evidence."""
    eid = f"memlens:{q['question_id']}"
    sessions = sorted(
        zip(q["haystack_dates"], q["haystack_session_ids"], q["haystack_sessions"]),
        key=lambda x: to_ms(x[0]),
    )
    signals = {arm: [] for arm in ARMS}
    gold = []
    for date, sid, turns in sessions:
        ts = to_ms(date)
        for t_idx, turn in enumerate(turns):
            body = norm_ws(str(turn.get("content", "")).replace("<image>", " "))
            imgs = []
            for im in turn.get("images") or []:
                src = (img_root / im["file"]).resolve()
                if img_root.resolve() not in src.parents or not src.is_file():
                    continue
                meta = dict(prepare_image(src, media, args.max_side, args.thumb_side, index))
                meta["caption"] = norm_ws(im.get("blip_caption", ""))
                imgs.append(meta)
            if not body and not imgs:
                continue
            role = norm_ws(turn.get("role", "speaker"))
            src_id = f"eval/{eid}/{sid}/{t_idx}"
            plain = f"[{date}] {role}: {body}"
            captioned = plain + "".join(CAPTION_FMT.format(m["caption"]) for m in imgs if m["caption"])
            img_sigs = [
                {
                    "modality": "image",
                    "source_id": f"{src_id}/img{k}",
                    "timestamp_ms": ts,
                    **{f: m[f] for f in ("media_path", "mimetype", "pixels_path", "width", "height", "channels", "judge_path", "caption")},
                    "media_sha": m["sha"],
                }
                for k, m in enumerate(imgs)
            ]
            text_sig = lambda txt: {"modality": "text", "source_id": src_id, "text": txt, "timestamp_ms": ts}
            signals["text"].append(text_sig(plain))
            signals["caption"].append(text_sig(captioned))
            signals["image"] += [text_sig(plain)] + img_sigs
            signals["caption_image"] += [text_sig(captioned)] + img_sigs
            if turn.get("has_answer"):
                gold.append(
                    {
                        "source_id": src_id,
                        "text": plain,
                        "body_probe": body[:120],
                        "images": [{"sha": m["sha"], "caption": m["caption"]} for m in imgs],
                    }
                )
    eps = {arm: {"episode_id": eid, "benchmark": "memlens_32k", "signals": s} for arm, s in signals.items()}
    evidence = [
        g["text"] + "".join(f" [attached image; BLIP caption: {i['caption']}]" for i in g["images"])
        for g in gold
    ]
    return eps, gold, evidence


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=Path("data/raw/memory_evals/memlens"))
    ap.add_argument("--out", type=Path, default=Path("data/memory_evals/packet_sufficiency_memlens"))
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--max-side", type=int, default=768, help="decoded pixel max side")
    ap.add_argument("--thumb-side", type=int, default=768, help="judge thumbnail max side")
    args = ap.parse_args()

    data = json.loads((args.raw / "dataset_32k.json").read_text())
    img_root = args.raw / "release_images_extracted" / "release_images"
    media = args.out / "media"
    media.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    index: dict = {}

    rows = {arm: {"cases": [], "key": [], "eps": []} for arm in ARMS}
    stats = Counter()
    for q, stratum in pick(data, rng):
        eps, gold, evidence = build_episode(q, img_root, media, args, index)
        refusal = q["question_type"] == "answer_refusal"
        qdate = norm_ws(q["question_date"])
        case = {
            "prompt_id": f"memlens:{q['question_id']}",
            "tags": [f"memlens/{q['question_type']}", stratum, "refusal" if refusal else "answerable", "memlens_32k"],
            "benchmark": "memlens_32k",
            "episode_id": eps["text"]["episode_id"],
            "query_id": q["question_id"],
            "question": norm_ws(q["question"]),
            "question_date": qdate,
            "query_timestamp_ms": to_ms(qdate),
            "answers": [norm_ws(q["answer"])],
            "evidence": evidence,
            "gold_turns": gold,
            "requires_abstention": refusal,
        }
        stats[stratum] += 1
        stats["gold_image_turns"] += sum(1 for g in gold if g["images"])
        stats["gold_text_turns"] += sum(1 for g in gold if not g["images"])
        stats["image_signals"] += sum(1 for s in eps["image"]["signals"] if s["modality"] == "image")
        stats["text_signals"] += len(eps["text"]["signals"])
        key = {
            "conversation_id": case["episode_id"],
            "query_id": case["query_id"],
            "question": case["question"],
            "question_type": case["tags"][0],
            "answers": case["answers"],
            "evidence": [g["body_probe"] for g in gold],
            "requires_abstention": refusal,
            "query_timestamp_ms": case["query_timestamp_ms"],
        }
        for arm in ARMS:
            rows[arm]["cases"].append(case)
            rows[arm]["key"].append(key)
            rows[arm]["eps"].append(eps[arm])

    for arm in ARMS:
        d = args.out / arm
        d.mkdir(parents=True, exist_ok=True)
        for name, items in (("cases", rows[arm]["cases"]), ("answer_key", rows[arm]["key"]), ("episodes", rows[arm]["eps"])):
            with (d / f"{name}.jsonl").open("w") as fh:
                for r in items:
                    fh.write(json.dumps(r) + "\n")
    (args.out / "media_index.json").write_text(json.dumps(index, indent=1) + "\n")
    stats["unique_images"] = len(index)
    stats["media_bytes_total"] = sum(m["media_bytes"] for m in index.values())
    stats["pixel_bytes_total"] = sum(m["width"] * m["height"] * 3 for m in index.values())
    dms = sorted(m["decode_ms"] for m in index.values())
    if dms:
        stats["decode_ms_p50"] = dms[len(dms) // 2]
        stats["decode_ms_p95"] = dms[int(len(dms) * 0.95)]
    (args.out / "build_stats.json").write_text(json.dumps(dict(stats), indent=1) + "\n")
    print(json.dumps(dict(stats)))


if __name__ == "__main__":
    main()
