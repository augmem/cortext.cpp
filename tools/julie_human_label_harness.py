#!/usr/bin/env python3
"""Human labeling harness for Julie retrieval probes.

This tool builds a local labeling sample, launches a small Gradio UI, and
converts completed human labels into the same frozen/eval format used by
``frozen_julie_retrieval_eval.py``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import pathlib
import random
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from frozen_julie_retrieval_eval import (
    build_timeline,
    canonical_hash,
    connect,
    evaluate,
    estimate_tokens,
    label_targets,
    load_memory_doc_map,
    tokens,
    wilson,
)


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


LOCAL_TZ = ZoneInfo("America/Chicago")


def candidate_pool(
    timeline,
    doc_to_memory,
    memory_to_doc,
    probe,
    max_candidates: int,
    min_overlap: float,
) -> list[dict]:
    event_index = int(probe["event_index"])
    if event_index <= 0 or event_index >= len(timeline):
        return []
    query_tokens = set(tokens(timeline[event_index].text))
    candidates: dict[int, tuple[float, set[str]]] = {}

    active_memory_ids = []
    for field in ("cortext_working_memory_ids", "cortext_retrieved_memory_ids"):
        active_memory_ids.extend(int(mid) for mid in probe.get(field, []) if mid is not None)
    for rank, memory_id in enumerate(dict.fromkeys(active_memory_ids), start=1):
        doc_index = memory_to_doc.get(memory_id)
        if doc_index is None or doc_index >= event_index:
            continue
        score = 3.0 + 1.0 / max(1, rank)
        old = candidates.get(doc_index)
        if old is None:
            candidates[doc_index] = (score, {"cortext_active_packet"})
        else:
            candidates[doc_index] = (max(old[0], score), old[1] | {"cortext_active_packet"})

    for rank, doc_index in enumerate(probe.get("rag_top_k_indices", []), start=1):
        if isinstance(doc_index, int) and 0 <= doc_index < event_index and doc_index in doc_to_memory:
            score = 2.0 + 1.0 / max(1, rank)
            candidates.setdefault(doc_index, (score, set()))[1].add("normal_rag_top_k")

    recent_media_added = 0
    for doc in reversed(timeline[:event_index]):
        if doc.index not in doc_to_memory or doc.modality not in {"audio", "image", "video"}:
            continue
        score = 2.75 + 0.25 / math.sqrt(max(1, event_index - doc.index))
        old = candidates.get(doc.index)
        if old is None:
            candidates[doc.index] = (score, {"recent_media_prior"})
        else:
            candidates[doc.index] = (max(old[0], score), old[1] | {"recent_media_prior"})
        recent_media_added += 1
        if recent_media_added >= 3:
            break

    overlap_targets = label_targets(
        timeline,
        doc_to_memory,
        probe,
        max_targets=max_candidates,
        min_overlap=min_overlap,
        label_sources={"overlap"},
    )
    for t in overlap_targets:
        doc_index = int(t["event_index"])
        old = candidates.get(doc_index)
        score = float(t.get("label_score", 0.0))
        if old is None:
            candidates[doc_index] = (score, {"token_overlap_recency"})
        else:
            candidates[doc_index] = (max(old[0], score), old[1] | {"token_overlap_recency"})

    for doc in timeline[:event_index]:
        if doc.index not in doc_to_memory or doc.modality != "text":
            continue
        doc_tokens = set(tokens(doc.text))
        if not doc_tokens or not query_tokens:
            continue
        overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens))
        if overlap < min_overlap:
            continue
        score = overlap + 0.25 / math.sqrt(max(1, event_index - doc.index))
        old = candidates.get(doc.index)
        if old is None or score > old[0]:
            candidates[doc.index] = (score, {"token_overlap_scan"})

    ranked = sorted(candidates.items(), key=lambda row: (-row[1][0], -timeline[row[0]].timestamp))
    selected_ranked = ranked[:max_candidates]
    if (
        max_candidates > 0
        and not any(timeline[doc_index].modality in {"audio", "image", "video"} for doc_index, _ in selected_ranked)
    ):
        for item in ranked:
            doc_index = item[0]
            if timeline[doc_index].modality in {"audio", "image", "video"}:
                if len(selected_ranked) < max_candidates:
                    selected_ranked.append(item)
                else:
                    selected_ranked[-1] = item
                break
    out = []
    for doc_index, (score, sources) in selected_ranked:
        doc = timeline[doc_index]
        out.append({
            "candidate_id": f"e{event_index}_c{doc_index}",
            "event_index": doc.index,
            "memory_id": doc_to_memory[doc.index],
            "timestamp": doc.timestamp,
            "source_id": doc.source_id,
            "modality": doc.modality,
            "heuristic_score": score,
            "candidate_sources": sorted(sources),
            "text": doc.text,
        })
    return out


def build_sample(args: argparse.Namespace) -> int:
    summary = load_json(args.summary)
    input_dir = args.input_dir or pathlib.Path(summary["input_dir"])
    db_path = args.db or pathlib.Path(summary["db_path"])
    timeline = build_timeline(
        input_dir,
        int(summary.get("processed_text_messages", -1)),
        int(summary.get("media_attempted", -1)),
    )
    conn = connect(db_path)
    doc_to_memory = load_memory_doc_map(conn, timeline)
    memory_to_doc = {memory_id: doc_index for doc_index, memory_id in doc_to_memory.items()}

    probes = list(summary.get("probes", []))
    rng = random.Random(args.seed)
    if args.shuffle_probes:
        rng.shuffle(probes)

    candidate_tasks = []
    for probe in probes:
        event_index = int(probe["event_index"])
        if event_index >= len(timeline):
            continue
        candidates = candidate_pool(
            timeline,
            doc_to_memory,
            memory_to_doc,
            probe,
            max_candidates=args.max_candidates,
            min_overlap=args.min_overlap,
        )
        if not candidates:
            continue
        rng.shuffle(candidates)
        candidate_tasks.append({
            "probe_id": f"probe_{event_index}",
            "event_index": event_index,
            "query": {
                "timestamp": timeline[event_index].timestamp,
                "source_id": timeline[event_index].source_id,
                "modality": timeline[event_index].modality,
                "text": timeline[event_index].text,
                "tokens": estimate_tokens(timeline[event_index].text),
            },
            "candidate_order_policy": "randomized_blind_order_seeded",
            "candidates": candidates,
            "labels": {},
            "notes": "",
        })

    def has_candidate_source(task: dict, source: str) -> bool:
        return any(
            source in cand.get("candidate_sources", [])
            for cand in task.get("candidates", [])
        )

    def has_media_candidate(task: dict) -> bool:
        return any(
            cand.get("modality") in {"audio", "image", "video"}
            for cand in task.get("candidates", [])
        )

    selected_tasks = []
    selected_probe_ids = set()

    def add_task(task: dict) -> None:
        if len(selected_tasks) >= args.max_probes:
            return
        probe_id = task.get("probe_id")
        if probe_id in selected_probe_ids:
            return
        selected_tasks.append(task)
        selected_probe_ids.add(probe_id)

    def add_first_matching(predicate) -> None:
        for task in candidate_tasks:
            if predicate(task):
                add_task(task)
                return

    if any(has_candidate_source(task, "cortext_active_packet") for task in candidate_tasks):
        add_first_matching(lambda task: has_candidate_source(task, "cortext_active_packet"))
    if any(has_media_candidate(task) for task in candidate_tasks):
        add_first_matching(has_media_candidate)
    for task in candidate_tasks:
        add_task(task)

    tasks = selected_tasks

    candidate_source_counts = Counter()
    candidate_modality_counts = Counter()
    query_modality_counts = Counter()
    for task in tasks:
        query_modality_counts[str(task.get("query", {}).get("modality", "unknown"))] += 1
        for cand in task.get("candidates", []):
            candidate_modality_counts[str(cand.get("modality", "unknown"))] += 1
            for source in cand.get("candidate_sources", []):
                candidate_source_counts[str(source)] += 1

    body = {
        "schema": "cortext_human_label_sample_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary": str(args.summary),
        "db_path": str(db_path),
        "input_dir": str(input_dir),
        "seed": args.seed,
        "max_probes": args.max_probes,
        "max_candidates": args.max_candidates,
        "scale": {
            "0": "unrelated/noise",
            "1": "weak/background",
            "2": "useful should-surface memory",
            "3": "important/direct should-surface memory",
        },
        "instructions": [
            "Label each candidate independently from 0 to 3.",
            "Use 2 or 3 when the candidate should surface in the bounded active packet for the query.",
            "Candidate order is randomized; do not infer rank from display order.",
            "Use only the current turn and prior conversation context; do not use future turns.",
        ],
        "labeling_context_policy": {
            "query_context": "current_turn_plus_prior_context_only",
            "candidate_context": "candidate_neighborhood_capped_before_query",
            "future_turns_visible": False,
        },
        "sample_selection_policy": {
            "probe_order": "seeded_shuffle" if args.shuffle_probes else "frozen_summary_order",
            "required_candidate_coverage": [
                "cortext_active_packet",
                "audio_or_image_or_video",
            ],
            "coverage_applied_before_filling_remaining_probe_slots": True,
            "available_probe_count": len(candidate_tasks),
            "selected_probe_count": len(tasks),
        },
        "sample_composition": {
            "query_modality_counts": dict(query_modality_counts),
            "candidate_modality_counts": dict(candidate_modality_counts),
            "candidate_source_counts": dict(candidate_source_counts),
            "includes_cortext_active_packet_candidates": (
                candidate_source_counts.get("cortext_active_packet", 0) > 0
            ),
            "includes_media_candidates": any(
                candidate_modality_counts.get(modality, 0) > 0
                for modality in ("audio", "image", "video")
            ),
        },
        "tasks": tasks,
    }
    body["sample_sha256"] = canonical_hash({k: v for k, v in body.items() if k != "sample_sha256"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2) + "\n")
    print(args.out)
    print(f"tasks={len(tasks)} candidates={sum(len(t['candidates']) for t in tasks)} sample_sha256={body['sample_sha256']}")
    return 0


def launch(args: argparse.Namespace) -> int:
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("gradio is not installed. Install with: python3 -m pip install gradio") from exc

    sample = load_json(args.sample)
    tasks = sample.get("tasks", [])
    state = {"task_index": 0, "candidate_index": 0, "sample": sample}
    try:
        summary = load_json(pathlib.Path(sample["source_summary"]))
        input_dir = pathlib.Path(sample["input_dir"])
        timeline = build_timeline(
            input_dir,
            int(summary.get("processed_text_messages", -1)),
            int(summary.get("media_attempted", -1)),
        )
    except Exception:
        timeline = []

    def save_sample() -> None:
        args.sample.write_text(json.dumps(state["sample"], indent=2) + "\n")

    def speaker_label(source_id: str) -> str:
        if source_id == "chat/user":
            return "Gabe"
        if source_id == "chat/assistant":
            return "Julie"
        return source_id

    def display_datetime(timestamp_ms: int) -> str:
        try:
            return datetime.fromtimestamp(timestamp_ms / 1000.0, LOCAL_TZ).strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
        except Exception:
            return str(timestamp_ms)

    thumb_dir = pathlib.Path("build/julie_human_label_thumbnails")
    thumb_dir.mkdir(parents=True, exist_ok=True)

    def thumbnail_path(path: pathlib.Path, max_px: int = 720) -> str | None:
        try:
            stat = path.stat()
            key = f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{max_px}"
            out = thumb_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.jpg"
            if out.exists():
                return str(out)
            from PIL import Image, ImageOps

            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                img.thumbnail((max_px, max_px))
                if img.mode not in {"RGB", "L"}:
                    img = img.convert("RGB")
                img.save(out, format="JPEG", quality=82, optimize=True)
            return str(out)
        except Exception:
            return None

    def thumbnail_data_uri(path: pathlib.Path) -> str | None:
        thumb = thumbnail_path(path)
        if thumb is None:
            return None
        try:
            encoded = base64.b64encode(pathlib.Path(thumb).read_bytes()).decode("ascii")
        except Exception:
            return None
        return f"data:image/jpeg;base64,{encoded}"

    def context_html(center_index: int, before: int, after: int, max_index: int | None = None) -> str:
        if not timeline or center_index < 0:
            return ""
        upper = min(len(timeline) - 1, center_index + after)
        if max_index is not None:
            upper = min(upper, max_index)
        lower = max(0, center_index - before)
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        input_dir = pathlib.Path(sample.get("input_dir", ""))
        lines = []
        for doc in timeline[lower : upper + 1]:
            text = " ".join(doc.text.split())
            if len(text) > 1200:
                text = text[:1197] + "..."
            is_center = doc.index == center_index
            row_style = (
                "padding:8px 10px;margin:0 0 6px 0;border-radius:6px;"
                "border:1px solid #374151;background:#1f2937;"
                "color:#f3f4f6;"
            )
            if is_center:
                row_style += "border-color:#93c5fd;background:#263244;"
            marker = "&gt;&gt;&gt; " if is_center else ""
            header = (
                f"{marker}{doc.index} | {html.escape(display_datetime(doc.timestamp))} | "
                f"{html.escape(speaker_label(doc.source_id))} "
                f"[{html.escape(doc.modality)}]"
            )
            body = html.escape(text)
            media_html = ""
            if doc.source_blob:
                path = input_dir / doc.source_blob
                if path.suffix.lower() in image_exts and path.exists():
                    uri = thumbnail_data_uri(path)
                    if uri:
                        media_html = (
                            "<div style='margin-top:8px'>"
                            f"<img src='{uri}' style='max-width:100%;max-height:300px;"
                            "object-fit:contain;border-radius:6px;border:1px solid #ccc'/>"
                            "</div>"
                        )
            lines.append(
                f"<div style='{row_style}'>"
                f"<div style='font-family:monospace;font-size:12px;color:#cbd5e1'>{header}</div>"
                f"<div style='white-space:pre-wrap;font-size:14px;line-height:1.35;color:#f8fafc'>{body}</div>"
                f"{media_html}</div>"
            )
        return (
            "<div style='height:430px;overflow-y:auto;border:1px solid #374151;"
            "border-radius:6px;padding:8px;background:#111827'>"
            + "".join(lines)
            + "</div>"
        )
        return "\n".join(lines)

    def candidate_image_path(candidate: dict) -> tuple[str | None, str]:
        if not timeline or candidate.get("modality") != "image":
            return None, ""
        event_index = int(candidate.get("event_index", -1))
        if event_index < 0 or event_index >= len(timeline):
            return None, ""
        doc = timeline[event_index]
        if not doc.source_blob:
            return None, ""
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        input_dir = pathlib.Path(sample.get("input_dir", ""))
        path = input_dir / doc.source_blob
        if path.suffix.lower() not in image_exts or not path.exists():
            return None, ""
        thumb = thumbnail_path(path)
        if thumb is None:
            return None, ""
        return thumb, f"{doc.index} {speaker_label(doc.source_id)} {path.name}"

    def candidate_audio_path(candidate: dict) -> tuple[str | None, str]:
        if not timeline or candidate.get("modality") != "audio":
            return None, ""
        event_index = int(candidate.get("event_index", -1))
        if event_index < 0 or event_index >= len(timeline):
            return None, ""
        doc = timeline[event_index]
        if not doc.source_blob:
            return None, ""
        audio_exts = {".aac", ".aif", ".aiff", ".amr", ".caf", ".m4a", ".mp3", ".ogg", ".wav"}
        input_dir = pathlib.Path(sample.get("input_dir", ""))
        path = input_dir / doc.source_blob
        if path.suffix.lower() not in audio_exts or not path.exists():
            return None, ""
        return str(path), f"{doc.index} {speaker_label(doc.source_id)} {path.name}"

    def total_counts() -> tuple[int, int]:
        total = sum(len(t.get("candidates", [])) for t in tasks)
        labeled = sum(len(t.get("labels", {})) for t in tasks)
        return labeled, total

    def candidate_index_for_id(task: dict, candidate_id: str) -> int:
        for i, cand in enumerate(task.get("candidates", [])):
            if cand.get("candidate_id") == candidate_id:
                return i
        return 0

    def first_unlabeled_position(start_task: int = 0, start_candidate: int = 0) -> tuple[int, int]:
        if not tasks:
            return 0, 0
        for task_offset in range(len(tasks)):
            task_index = (start_task + task_offset) % len(tasks)
            task = tasks[task_index]
            candidates = task.get("candidates", [])
            labels = task.get("labels", {})
            begin = start_candidate if task_offset == 0 else 0
            for candidate_index in range(begin, len(candidates)):
                if candidates[candidate_index]["candidate_id"] not in labels:
                    return task_index, candidate_index
        return min(start_task, len(tasks) - 1), 0

    state["task_index"], state["candidate_index"] = first_unlabeled_position()

    def render(task_index: int | None = None, candidate_index: int | None = None):
        if not tasks:
            return (
                "No tasks",
                "",
                gr.update(value="", visible=True),
                "",
                gr.update(value=None, visible=False),
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(value="", visible=False),
                "",
                "",
                [],
                "",
                gr.update(choices=[], value=None),
                None,
            )
        if task_index is None:
            task_index = state["task_index"]
        if candidate_index is None:
            candidate_index = state["candidate_index"]
        task_index = max(0, min(task_index, len(tasks) - 1))
        task = tasks[task_index]
        candidates = task.get("candidates", [])
        if not candidates:
            candidate_index = 0
        else:
            candidate_index = max(0, min(candidate_index, len(candidates) - 1))
        state["task_index"] = task_index
        state["candidate_index"] = candidate_index
        labeled_total, candidate_total = total_counts()
        task_labeled = len(task.get("labels", {}))
        task_total = len(candidates)
        current = candidates[candidate_index] if candidates else None
        rows = []
        choices = []
        for cand in candidates:
            label = task.get("labels", {}).get(cand["candidate_id"], "")
            choice = f"{cand['candidate_id']} | {cand['source_id']} | {cand['text'][:120]}"
            choices.append(choice)
            rows.append([
                cand["candidate_id"],
                label,
                display_datetime(int(cand.get("timestamp", 0))),
                cand["source_id"],
                cand["modality"],
                cand["text"],
            ])
        query = task["query"]["text"]
        query_context = context_html(
            int(task["event_index"]),
            before=32,
            after=0,
            max_index=int(task["event_index"]),
        )
        status = (
            f"Probe {task_index + 1}/{len(tasks)}  "
            f"Candidate {candidate_index + 1 if candidates else 0}/{task_total}  "
            f"Labeled {labeled_total}/{candidate_total}  "
            f"Probe labels {task_labeled}/{task_total}  "
            f"event_index={task['event_index']}"
        )
        notes = task.get("notes", "")
        if current:
            candidate_text = current["text"]
            candidate_context = context_html(
                int(current["event_index"]),
                before=12,
                after=8,
                max_index=int(task["event_index"]) - 1,
            )
            image_path, image_caption = candidate_image_path(current)
            audio_path, audio_caption = candidate_audio_path(current)
            candidate_meta = (
                f"{current['candidate_id']} | {current['source_id']} | "
                f"{current['modality']} | {display_datetime(int(current.get('timestamp', 0)))} | current label: "
                f"{task.get('labels', {}).get(current['candidate_id'], 'unlabeled')}"
            )
            current_label = task.get("labels", {}).get(current["candidate_id"])
            dropdown_value = choices[candidate_index]
        else:
            candidate_text = ""
            candidate_context = ""
            image_path = None
            image_caption = ""
            audio_path = None
            audio_caption = ""
            candidate_meta = ""
            current_label = None
            dropdown_value = None
        return (
            status,
            query,
            gr.update(value=candidate_text, visible=not bool(image_path or audio_path)),
            gr.update(value=image_path, visible=bool(image_path)),
            gr.update(value=image_caption, visible=bool(image_path)),
            gr.update(value=audio_path, visible=bool(audio_path)),
            gr.update(value=audio_caption, visible=bool(audio_path)),
            query_context,
            candidate_context,
            candidate_meta,
            rows,
            notes,
            gr.update(choices=choices, value=dropdown_value),
            current_label,
        )

    def set_current_label(relevance: int, notes: str):
        if not tasks:
            return render()
        task = tasks[state["task_index"]]
        candidates = task.get("candidates", [])
        if candidates:
            cand = candidates[state["candidate_index"]]
            task.setdefault("labels", {})[cand["candidate_id"]] = int(relevance)
            task["notes"] = notes or ""
            save_sample()
            next_task_index, next_candidate_index = first_unlabeled_position(
                state["task_index"],
                state["candidate_index"] + 1,
            )
            return render(next_task_index, next_candidate_index)
        return render()

    def jump_candidate(candidate_choice: str, notes: str):
        if not tasks or not candidate_choice:
            return render()
        task = tasks[state["task_index"]]
        task["notes"] = notes or ""
        save_sample()
        candidate_id = candidate_choice.split(" | ", 1)[0]
        return render(state["task_index"], candidate_index_for_id(task, candidate_id))

    def move_candidate(delta: int, notes: str):
        if not tasks:
            return render()
        tasks[state["task_index"]]["notes"] = notes or ""
        save_sample()
        return render(state["task_index"], state["candidate_index"] + delta)

    def next_unlabeled(notes: str):
        if tasks:
            tasks[state["task_index"]]["notes"] = notes or ""
            save_sample()
        task_index, candidate_index = first_unlabeled_position(
            state["task_index"],
            state["candidate_index"] + 1,
        )
        return render(task_index, candidate_index)

    def next_task(notes: str):
        if tasks:
            tasks[state["task_index"]]["notes"] = notes or ""
            save_sample()
        return render(state["task_index"] + 1, 0)

    def prev_task(notes: str):
        if tasks:
            tasks[state["task_index"]]["notes"] = notes or ""
            save_sample()
        return render(state["task_index"] - 1, 0)

    def save_notes(notes: str):
        if tasks:
            tasks[state["task_index"]]["notes"] = notes or ""
            save_sample()
        return render()

    with gr.Blocks(title="Cortext Human Memory Labeling") as demo:
        gr.Markdown("# Cortext human memory labeling")
        status = gr.Textbox(label="Progress", interactive=False)
        with gr.Row():
            with gr.Column():
                query = gr.Textbox(label="Current turn", lines=7, interactive=False)
                gr.Markdown("Conversation context around current turn (chronological; `>>>` marks current)")
                query_context = gr.HTML()
            with gr.Column():
                candidate = gr.Textbox(label="Candidate memory", lines=7, interactive=False)
                candidate_image = gr.Image(
                    label="Candidate memory",
                    type="filepath",
                    visible=False,
                    height=360,
                    interactive=False,
                    buttons=[],
                )
                candidate_image_caption = gr.Textbox(label="Candidate image", interactive=False, visible=False)
                candidate_audio = gr.Audio(
                    label="Candidate memory",
                    type="filepath",
                    visible=False,
                    interactive=False,
                )
                candidate_audio_caption = gr.Textbox(label="Candidate audio", interactive=False, visible=False)
                gr.Markdown("Candidate conversation context (chronological; `>>>` marks candidate)")
                candidate_context = gr.HTML()
        candidate_meta = gr.Textbox(label="Candidate", interactive=False)
        with gr.Row():
            score0 = gr.Button("0 Noise")
            score1 = gr.Button("1 Weak")
            score2 = gr.Button("2 Useful")
            score3 = gr.Button("3 Important")
        with gr.Row():
            prev_candidate = gr.Button("Previous Candidate")
            next_candidate = gr.Button("Next Candidate")
            next_unlabeled_b = gr.Button("Next Unlabeled")
        candidate_id = gr.Dropdown(label="Jump to candidate")
        current_label = gr.Radio([0, 1, 2, 3], label="Current label", interactive=False)
        table = gr.Dataframe(
            headers=["candidate_id", "current_label", "datetime", "source", "modality", "candidate_text"],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )
        notes = gr.Textbox(label="Probe notes", lines=2)
        with gr.Row():
            prev_b = gr.Button("Previous")
            next_b = gr.Button("Next")
            save_b = gr.Button("Save Notes")
        outputs = [
            status,
            query,
            candidate,
            candidate_image,
            candidate_image_caption,
            candidate_audio,
            candidate_audio_caption,
            query_context,
            candidate_context,
            candidate_meta,
            table,
            notes,
            candidate_id,
            current_label,
        ]
        score0.click(lambda n: set_current_label(0, n), [notes], outputs)
        score1.click(lambda n: set_current_label(1, n), [notes], outputs)
        score2.click(lambda n: set_current_label(2, n), [notes], outputs)
        score3.click(lambda n: set_current_label(3, n), [notes], outputs)
        prev_candidate.click(lambda n: move_candidate(-1, n), [notes], outputs)
        next_candidate.click(lambda n: move_candidate(1, n), [notes], outputs)
        next_unlabeled_b.click(next_unlabeled, [notes], outputs)
        candidate_id.change(jump_candidate, [candidate_id, notes], outputs)
        prev_b.click(prev_task, [notes], outputs)
        next_b.click(next_task, [notes], outputs)
        save_b.click(save_notes, [notes], outputs)
        demo.load(lambda: render(), outputs=outputs)
    allowed_paths = []
    if sample.get("input_dir"):
        allowed_paths.append(str(pathlib.Path(sample["input_dir"])))
    demo.launch(server_name=args.host, server_port=args.port, allowed_paths=allowed_paths)
    return 0


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    n = len(labels_a)
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def score(args: argparse.Namespace) -> int:
    sample = load_json(args.sample)
    judge = load_json(args.judge_frozen) if args.judge_frozen else None
    probes = []
    human_targets_by_probe: dict[int, set[int]] = {}
    for task in sample.get("tasks", []):
        labels = task.get("labels", {})
        targets = []
        for cand in task.get("candidates", []):
            relevance = labels.get(cand["candidate_id"])
            if relevance is None:
                continue
            if int(relevance) >= args.positive_threshold:
                targets.append({
                    "memory_id": cand["memory_id"],
                    "event_index": cand["event_index"],
                    "timestamp": cand["timestamp"],
                    "source_id": cand["source_id"],
                    "modality": cand["modality"],
                    "label_score": int(relevance),
                    "label_sources": ["human_label"],
                })
        if not targets and not args.keep_unanswerable:
            continue
        event_index = int(task["event_index"])
        human_targets_by_probe[event_index] = {int(t["event_index"]) for t in targets}
        probes.append({
            "event_index": event_index,
            "query": {k: v for k, v in task["query"].items() if k != "text"},
            "target_memories": targets,
            "answerability": "human_labeled_targets" if targets else "no_human_labeled_target",
        })

    frozen = {
        "schema": "cortext_frozen_retrieval_probe_set_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_evaluation": True,
        "source_summary": sample["source_summary"],
        "db_path": sample["db_path"],
        "input_dir": sample["input_dir"],
        "labeling": {
            "method": "human labels from Gradio harness",
            "sample_sha256": sample.get("sample_sha256"),
            "positive_threshold": args.positive_threshold,
            "scale": sample.get("scale"),
            "limitations": [
                "single human annotator unless multiple labeled samples are compared",
                "candidate pool can underlabel memories not shown to the annotator",
                "single corpus, single relationship, single writing style",
            ],
        },
        "probe_count": len(probes),
        "probes": probes,
    }
    frozen["freeze_sha256"] = canonical_hash({k: v for k, v in frozen.items() if k != "freeze_sha256"})
    args.out_frozen.parent.mkdir(parents=True, exist_ok=True)
    args.out_frozen.write_text(json.dumps(frozen, indent=2) + "\n")

    agreement = {}
    if judge:
        judge_by_probe = {
            int(p["event_index"]): {int(t["event_index"]) for t in p.get("target_memories", [])}
            for p in judge.get("probes", [])
        }
        all_events = sorted(set(human_targets_by_probe) | set(judge_by_probe))
        intersections = [len(human_targets_by_probe.get(e, set()) & judge_by_probe.get(e, set())) for e in all_events]
        unions = [len(human_targets_by_probe.get(e, set()) | judge_by_probe.get(e, set())) for e in all_events]
        all_candidate_pairs = sorted({
            (int(task["event_index"]), int(c["event_index"]))
            for task in sample.get("tasks", [])
            for c in task.get("candidates", [])
        })
        human_positive_pairs = {
            (probe_event, target_event)
            for probe_event, target_events in human_targets_by_probe.items()
            for target_event in target_events
        }
        judge_positive_pairs = {
            (probe_event, target_event)
            for probe_event, target_events in judge_by_probe.items()
            for target_event in target_events
        }
        human_binary = []
        judge_binary = []
        for pair in all_candidate_pairs:
            human_binary.append(1 if pair in human_positive_pairs else 0)
            judge_binary.append(1 if pair in judge_positive_pairs else 0)
        agreement = {
            "judge_frozen": str(args.judge_frozen),
            "probe_event_jaccard": sum(intersections) / sum(unions) if sum(unions) else 0.0,
            "cohen_kappa_binary_target_membership": cohen_kappa(human_binary, judge_binary),
            "candidate_pair_count": len(all_candidate_pairs),
            "shared_probe_events": len(set(human_targets_by_probe) & set(judge_by_probe)),
            "human_probe_count": len(human_targets_by_probe),
            "judge_probe_count": len(judge_by_probe),
        }

    if args.eval_out:
        ns = argparse.Namespace(
            summary=args.summary,
            frozen=args.out_frozen,
            judge=None,
            db=args.db,
            input_dir=args.input_dir,
            remap_targets=True,
            out=args.eval_out,
        )
        evaluate(ns)

    report = {
        "schema": "cortext_human_label_score_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample": str(args.sample),
        "human_frozen": str(args.out_frozen),
        "human_freeze_sha256": frozen["freeze_sha256"],
        "probe_count": len(probes),
        "agreement": agreement,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2) + "\n")
    print(args.out_report)
    print(args.out_frozen)
    if args.eval_out:
        print(args.eval_out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build_p = sub.add_parser("build-sample")
    build_p.add_argument("--summary", type=pathlib.Path, required=True)
    build_p.add_argument("--db", type=pathlib.Path)
    build_p.add_argument("--input-dir", type=pathlib.Path)
    build_p.add_argument("--out", type=pathlib.Path, required=True)
    build_p.add_argument("--max-probes", type=int, default=40)
    build_p.add_argument("--max-candidates", type=int, default=12)
    build_p.add_argument("--min-overlap", type=float, default=0.20)
    build_p.add_argument("--seed", type=int, default=42)
    build_p.add_argument("--shuffle-probes", action="store_true")
    build_p.set_defaults(func=build_sample)

    launch_p = sub.add_parser("launch")
    launch_p.add_argument("--sample", type=pathlib.Path, required=True)
    launch_p.add_argument("--host", default="127.0.0.1")
    launch_p.add_argument("--port", type=int, default=7860)
    launch_p.set_defaults(func=launch)

    score_p = sub.add_parser("score")
    score_p.add_argument("--sample", type=pathlib.Path, required=True)
    score_p.add_argument("--summary", type=pathlib.Path, required=True)
    score_p.add_argument("--db", type=pathlib.Path)
    score_p.add_argument("--input-dir", type=pathlib.Path)
    score_p.add_argument("--judge-frozen", type=pathlib.Path)
    score_p.add_argument("--out-frozen", type=pathlib.Path, required=True)
    score_p.add_argument("--out-report", type=pathlib.Path, required=True)
    score_p.add_argument("--eval-out", type=pathlib.Path)
    score_p.add_argument("--positive-threshold", type=int, default=2)
    score_p.add_argument("--keep-unanswerable", action="store_true")
    score_p.set_defaults(func=score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
