#!/usr/bin/env python3
"""Download and normalize memory benchmarks into Cortext eval episodes.

Outputs:
  data/memory_evals/<name>/episodes.jsonl
  data/memory_evals/<name>/answer_key.jsonl
  data/memory_evals/manifest.json

The normalized episode schema is modality-aware:
  {"episode_id": "...", "benchmark": "...", "signals": [
      {"modality": "text", "source_id": "...", "text": "..."}
  ]}

The current local runner processes text signals and reports skipped non-text
signals. The schema keeps image/audio references explicit so multimodal coverage
can be added without changing the benchmark adapters.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable


RAW_ROOT_DEFAULT = Path("data/raw/memory_evals")
OUT_ROOT_DEFAULT = Path("data/memory_evals")

BENCHMARKS = (
    "longmemeval",
    "longmemeval_v2",
    "locomo",
    "locomo_plus",
    "beam",
    "memoryagentbench",
)

URLS = {
    "longmemeval_s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "lme_v2_questions": "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/resolve/main/questions.jsonl",
    "lme_v2_trajectories": "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/resolve/main/trajectories.jsonl",
    "lme_v2_small_haystack": "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/resolve/main/haystacks/lme_v2_small.json",
    "locomo": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    "locomo_plus": "https://raw.githubusercontent.com/xjtuleeyf/Locomo-Plus/main/data/locomo_plus.json",
    "locomo_plus_locomo": "https://raw.githubusercontent.com/xjtuleeyf/Locomo-Plus/main/data/locomo10.json",
    "beam_100k": "https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/data/100K-00000-of-00001.parquet",
    "beam_500k": "https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/data/500K-00000-of-00001.parquet",
    "beam_1m": "https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/data/1M-00000-of-00001.parquet",
    "mab_ar": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/data/Accurate_Retrieval-00000-of-00001.parquet",
    "mab_cr": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/data/Conflict_Resolution-00000-of-00001.parquet",
    "mab_lru": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/data/Long_Range_Understanding-00000-of-00001.parquet",
    "mab_ttl": "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/data/Test_Time_Learning-00000-of-00001.parquet",
}


def profile_limits(profile: str) -> dict[str, int]:
    if profile == "full":
        return {
            "episodes": 0,
            "queries_per_episode": 0,
            "turns_per_episode": 0,
            "context_chars": 1800,
            "lme_v2_trajectories": 0,
            "lme_v2_states": 0,
        }
    return {
        "episodes": 1,
        "queries_per_episode": 4,
        "turns_per_episode": 32,
        "context_chars": 1200,
        "lme_v2_trajectories": 1,
        "lme_v2_states": 2,
    }


def log(message: str) -> None:
    print(message, flush=True)


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    headers = {"User-Agent": "cortext-memory-eval-preparer/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    log(f"download {url} -> {dest}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def download_jsonl_subset_by_id(url: str, dest: Path, wanted_ids: set[str]) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    headers = {"User-Agent": "cortext-memory-eval-preparer/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    remaining = set(wanted_ids)
    log(f"stream subset {url} -> {dest} ids={len(remaining)}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open(
            "w", encoding="utf-8"
        ) as out:
            for raw in response:
                if not remaining:
                    break
                line = raw.decode("utf-8")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_id = normalize_text(row.get("id"))
                if row_id in remaining:
                    out.write(json.dumps(row, ensure_ascii=True) + "\n")
                    remaining.remove(row_id)
        if remaining:
            log(f"warning: missing {len(remaining)} requested trajectory ids")
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return [value]


def to_plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    try:
        import pandas as pd  # type: ignore

        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def chunk_text(text: str, max_chars: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split = text.rfind(" ", start, end)
            if split > start + max_chars // 2:
                end = split
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def limit_rows(rows: list[Any], limit: int) -> list[Any]:
    return rows if limit <= 0 else rows[:limit]


def cap_signals(signals: list[dict[str, Any]], limits: dict[str, int]) -> list[dict[str, Any]]:
    cap = limits.get("turns_per_episode", 0)
    return signals if cap <= 0 else signals[:cap]


def parse_ab_dialogue(text: str, speaker_a: str = "A", speaker_b: str = "B") -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.match(r"^([AB]):\s*(.*)$", line)
        if match:
            speaker = speaker_a if match.group(1) == "A" else speaker_b
            message = match.group(2).strip()
        else:
            speaker = speaker_a
            message = line
        if message:
            turns.append({"agent": speaker, "message": message})
    return turns


def evidence_from_locomo(conversation: dict[str, Any], evidence: Any) -> list[str]:
    evidence_ids: list[str] = []
    for item in as_list(evidence):
        for part in str(item).split(";"):
            part = part.strip()
            if part:
                evidence_ids.append(part)
    lines: list[str] = []
    for evidence_id in evidence_ids:
        try:
            session_raw, turn_raw = evidence_id.split(":", 1)
            session_idx = int(session_raw.replace("D", ""))
            turn_idx = int(turn_raw) - 1
            turn = conversation.get(f"session_{session_idx}", [])[turn_idx]
            text = normalize_text(turn.get("text", ""))
            if text:
                lines.append(text)
        except Exception:
            continue
    return lines


def locomo_signals(conversation: dict[str, Any], episode_id: str) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    session_keys = sorted(
        [
            key
            for key in conversation
            if key.startswith("session_") and not key.endswith("_date_time")
        ],
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    for session_key in session_keys:
        date = normalize_text(conversation.get(f"{session_key}_date_time", ""))
        for idx, turn in enumerate(conversation.get(session_key, [])):
            speaker = normalize_text(turn.get("speaker", "speaker"))
            text = normalize_text(turn.get("text", ""))
            caption = normalize_text(turn.get("blip_caption", ""))
            if caption:
                text = f"{text} Shared image caption: {caption}."
            if not text:
                continue
            prefix = f"[{date}] " if date else ""
            signals.append(
                {
                    "modality": "text",
                    "source_id": f"eval/{episode_id}/{session_key}/{idx}",
                    "text": f"{prefix}{speaker}: {text}",
                }
            )
    return signals


def prepare_longmemeval(raw_root: Path, out_root: Path, limits: dict[str, int]) -> dict[str, Any]:
    raw = download(URLS["longmemeval_s"], raw_root / "longmemeval" / "longmemeval_s_cleaned.json")
    records = limit_rows(read_json(raw), limits["episodes"])
    name = "longmemeval_s"
    out_dir = out_root / name
    episodes: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for record in records:
        qid = normalize_text(record.get("question_id"))
        episode_id = f"longmemeval:{qid}"
        signals: list[dict[str, Any]] = []
        sessions = record.get("haystack_sessions") or []
        dates = record.get("haystack_dates") or []
        session_ids = record.get("haystack_session_ids") or []
        for s_idx, session in enumerate(sessions):
            date = normalize_text(dates[s_idx] if s_idx < len(dates) else "")
            session_id = normalize_text(session_ids[s_idx] if s_idx < len(session_ids) else s_idx)
            for t_idx, turn in enumerate(session):
                role = normalize_text(turn.get("role", "speaker"))
                text = normalize_text(turn.get("content", ""))
                if not text:
                    continue
                signals.append(
                    {
                        "modality": "text",
                        "source_id": f"eval/{episode_id}/{session_id}/{t_idx}",
                        "text": f"[{date}] {role}: {text}" if date else f"{role}: {text}",
                    }
                )
        episodes.append({"episode_id": episode_id, "benchmark": name, "signals": cap_signals(signals, limits)})
        answer_key.append(
            {
                "benchmark": name,
                "conversation_id": episode_id,
                "query_id": qid,
                "question": normalize_text(record.get("question")),
                "answers": [normalize_text(record.get("answer"))],
                "question_type": normalize_text(record.get("question_type")) or "unknown",
                "requires_abstention": normalize_text(record.get("question_type")) == "abstention",
                "metadata": {
                    "question_date": normalize_text(record.get("question_date")),
                    "answer_session_ids": to_plain(record.get("answer_session_ids")),
                },
            }
        )
    return write_entry(name, out_dir, episodes, answer_key)


def prepare_lme_v2(raw_root: Path, out_root: Path, limits: dict[str, int]) -> dict[str, Any]:
    root = raw_root / "longmemeval_v2"
    q_path = download(URLS["lme_v2_questions"], root / "questions.jsonl")
    h_path = download(URLS["lme_v2_small_haystack"], root / "haystacks" / "lme_v2_small.json")
    questions = limit_rows(read_jsonl(q_path), limits["episodes"])
    haystack = read_json(h_path)
    wanted_trajectory_ids: set[str] = set()
    for question in questions:
        qid = normalize_text(question.get("id"))
        trajectory_ids = haystack.get(qid, [])
        trajectory_ids = limit_rows(trajectory_ids, limits["lme_v2_trajectories"])
        wanted_trajectory_ids.update(normalize_text(item) for item in trajectory_ids)
    subset_name = "trajectories_subset_smoke.jsonl" if limits["episodes"] > 0 else "trajectories_subset_full.jsonl"
    t_path = download_jsonl_subset_by_id(
        URLS["lme_v2_trajectories"], root / subset_name, wanted_trajectory_ids
    )
    trajectories = {row["id"]: row for row in read_jsonl(t_path)}
    name = "longmemeval_v2_small"
    out_dir = out_root / name
    episodes: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for question in questions:
        qid = normalize_text(question.get("id"))
        episode_id = f"longmemeval_v2:{qid}"
        signals: list[dict[str, Any]] = []
        trajectory_ids = haystack.get(qid, [])
        trajectory_ids = limit_rows(trajectory_ids, limits["lme_v2_trajectories"])
        for t_idx, trajectory_id in enumerate(trajectory_ids):
            trajectory = trajectories.get(trajectory_id)
            if not trajectory:
                continue
            goal = normalize_text(trajectory.get("goal"))
            outcome = normalize_text(trajectory.get("outcome"))
            if goal or outcome:
                signals.append(
                    {
                        "modality": "text",
                        "source_id": f"eval/{episode_id}/{trajectory_id}/goal",
                        "text": f"Trajectory goal: {goal} Outcome: {outcome}",
                    }
                )
            states = limit_rows(trajectory.get("states") or [], limits["lme_v2_states"])
            for s_idx, state in enumerate(states):
                parts = [
                    f"url={normalize_text(state.get('url'))}",
                    f"thought={normalize_text(state.get('thought'))}",
                    f"action={normalize_text(state.get('action'))}",
                    f"observation={normalize_text(state.get('accessibility_tree'))}",
                ]
                text = " ".join(part for part in parts if part.split("=", 1)[1])
                for c_idx, chunk in enumerate(chunk_text(text, limits["context_chars"])):
                    signals.append(
                        {
                            "modality": "text",
                            "source_id": f"eval/{episode_id}/{trajectory_id}/state{s_idx}/chunk{c_idx}",
                            "text": chunk,
                        }
                    )
                screenshot = normalize_text(
                    state.get("screenshot")
                    or state.get("screenshot_path")
                    or state.get("image")
                    or ""
                )
                if screenshot:
                    signals.append(
                        {
                            "modality": "image",
                            "source_id": f"eval/{episode_id}/{trajectory_id}/state{s_idx}/image",
                            "path": screenshot,
                            "mimetype": "image/png",
                        }
                    )
        episodes.append({"episode_id": episode_id, "benchmark": name, "signals": cap_signals(signals, limits)})
        answer_key.append(
            {
                "benchmark": name,
                "conversation_id": episode_id,
                "query_id": qid,
                "question": normalize_text(question.get("question")),
                "answers": [normalize_text(question.get("answer"))],
                "question_type": normalize_text(question.get("question_type")) or "unknown",
                "requires_abstention": False,
                "metadata": {
                    "domain": normalize_text(question.get("domain")),
                    "environment": normalize_text(question.get("environment")),
                    "eval_function": normalize_text(question.get("eval_function")),
                },
            }
        )
    return write_entry(name, out_dir, episodes, answer_key)


def prepare_locomo(raw_root: Path, out_root: Path, limits: dict[str, int]) -> dict[str, Any]:
    raw = download(URLS["locomo"], raw_root / "locomo" / "locomo10.json")
    records = limit_rows(read_json(raw), limits["episodes"])
    name = "locomo"
    out_dir = out_root / name
    episodes: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for item in records:
        sample_id = normalize_text(item.get("sample_id"))
        episode_id = f"locomo:{sample_id}"
        conversation = item.get("conversation") or {}
        episodes.append(
            {
                "episode_id": episode_id,
                "benchmark": name,
                "signals": cap_signals(locomo_signals(conversation, episode_id), limits),
            }
        )
        for q_idx, qa in enumerate(limit_rows(item.get("qa") or [], limits["queries_per_episode"])):
            answer = qa.get("answer")
            answer_key.append(
                {
                    "benchmark": name,
                    "conversation_id": episode_id,
                    "query_id": f"{sample_id}:{q_idx}",
                    "question": normalize_text(qa.get("question")),
                    "answers": [normalize_text(answer)] if answer is not None else [],
                    "evidence": evidence_from_locomo(conversation, qa.get("evidence")),
                    "question_type": str(qa.get("category", "unknown")),
                    "requires_abstention": qa.get("category") == 5,
                }
            )
    return write_entry(name, out_dir, episodes, answer_key)


def prepare_locomo_plus(raw_root: Path, out_root: Path, limits: dict[str, int]) -> dict[str, Any]:
    plus_path = download(URLS["locomo_plus"], raw_root / "locomo_plus" / "locomo_plus.json")
    locomo_path = download(URLS["locomo_plus_locomo"], raw_root / "locomo_plus" / "locomo10.json")
    plus_rows = limit_rows(read_json(plus_path), limits["episodes"])
    locomo_rows = read_json(locomo_path)
    name = "locomo_plus"
    out_dir = out_root / name
    episodes: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for idx, plus in enumerate(plus_rows):
        base = locomo_rows[idx % len(locomo_rows)]
        conversation = base.get("conversation") or {}
        speaker_a = conversation.get("speaker_a", "A")
        speaker_b = conversation.get("speaker_b", "B")
        episode_id = f"locomo_plus:{idx}"
        signals = locomo_signals(conversation, episode_id)
        if limits["turns_per_episode"] > 0:
            signals = signals[: limits["turns_per_episode"]]
        cue_turns = parse_ab_dialogue(plus.get("cue_dialogue", ""), speaker_a, speaker_b)
        evidence: list[str] = []
        for c_idx, turn in enumerate(cue_turns):
            text = normalize_text(turn["message"])
            evidence.append(text)
            signals.append(
                {
                    "modality": "text",
                    "source_id": f"eval/{episode_id}/cue/{c_idx}",
                    "text": f"{turn['agent']}: {text}",
                }
            )
        trigger_turns = parse_ab_dialogue(plus.get("trigger_query", ""), speaker_a, speaker_b)
        question = " ".join(turn["message"] for turn in trigger_turns) or normalize_text(
            plus.get("trigger_query")
        )
        episodes.append({"episode_id": episode_id, "benchmark": name, "signals": signals})
        answer_key.append(
            {
                "benchmark": name,
                "conversation_id": episode_id,
                "query_id": f"locomo_plus:{idx}",
                "question": question,
                "answers": [],
                "evidence": evidence,
                "question_type": "Cognitive",
                "requires_abstention": False,
                "metadata": {
                    "relation_type": normalize_text(plus.get("relation_type")),
                    "time_gap": normalize_text(plus.get("time_gap")),
                },
            }
        )
    return write_entry(name, out_dir, episodes, answer_key)


def read_parquet(path: Path):
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise SystemExit("pandas/pyarrow are required for parquet-backed evals") from exc
    return pd.read_parquet(path)


def flatten_chat(value: Any) -> list[dict[str, str]]:
    plain = to_plain(value)
    turns: list[dict[str, str]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            content = normalize_text(node.get("content") or node.get("text") or node.get("message"))
            role = normalize_text(node.get("role") or node.get("speaker") or "speaker")
            if content:
                turns.append({"agent": role, "message": content})
            return
        if isinstance(node, str):
            text = normalize_text(node)
            if text:
                turns.append({"agent": "speaker", "message": text})
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(plain)
    return turns


def parse_probe_map(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def answer_strings_from_probe(probe: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    for key in ("ideal_answer", "ideal_response", "answer", "reference_answer"):
        value = probe.get(key)
        if value:
            answers.extend(normalize_text(item) for item in as_list(value))
    for item in as_list(probe.get("rubric")):
        text = normalize_text(item)
        if text:
            answers.append(text)
    seen: set[str] = set()
    out: list[str] = []
    for answer in answers:
        if answer and answer not in seen:
            out.append(answer)
            seen.add(answer)
    return out


def prepare_beam(raw_root: Path, out_root: Path, limits: dict[str, int], include_large: bool) -> list[dict[str, Any]]:
    tiers = ["beam_100k", "beam_500k", "beam_1m"] if include_large else ["beam_100k"]
    entries: list[dict[str, Any]] = []
    for tier in tiers:
        raw = download(URLS[tier], raw_root / "beam" / f"{tier}.parquet")
        df = read_parquet(raw)
        name = tier
        out_dir = out_root / name
        episodes: list[dict[str, Any]] = []
        answer_key: list[dict[str, Any]] = []
        records = df.head(limits["episodes"]) if limits["episodes"] > 0 else df
        for _, row in records.iterrows():
            cid = normalize_text(row.get("conversation_id"))
            episode_id = f"{tier}:{cid}"
            signals: list[dict[str, Any]] = []
            turns = flatten_chat(row.get("chat"))
            for t_idx, turn in enumerate(turns):
                full_text = f"{turn['agent']}: {turn['message']}"
                for c_idx, chunk in enumerate(chunk_text(full_text, limits["context_chars"])):
                    signals.append(
                        {
                            "modality": "text",
                            "source_id": f"eval/{episode_id}/chat/{t_idx}/chunk{c_idx}",
                            "text": chunk,
                        }
                    )
                    if limits["turns_per_episode"] > 0 and len(signals) >= limits["turns_per_episode"]:
                        break
                if limits["turns_per_episode"] > 0 and len(signals) >= limits["turns_per_episode"]:
                    break
            episodes.append({"episode_id": episode_id, "benchmark": name, "signals": signals})
            emitted = 0
            for category, probes in parse_probe_map(row.get("probing_questions")).items():
                for p_idx, probe in enumerate(as_list(probes)):
                    if not isinstance(probe, dict):
                        continue
                    if limits["queries_per_episode"] > 0 and emitted >= limits["queries_per_episode"]:
                        break
                    answer_key.append(
                        {
                            "benchmark": name,
                            "conversation_id": episode_id,
                            "query_id": f"{episode_id}:{category}:{p_idx}",
                            "question": normalize_text(probe.get("question")),
                            "answers": answer_strings_from_probe(probe),
                            "question_type": normalize_text(category) or "unknown",
                            "requires_abstention": normalize_text(category) == "abstention"
                            or bool(probe.get("why_unanswerable")),
                        }
                    )
                    emitted += 1
        entries.append(write_entry(name, out_dir, episodes, answer_key))
    return entries


def flatten_answers(value: Any) -> list[str]:
    out: list[str] = []

    def walk(node: Any) -> None:
        node = to_plain(node)
        if node is None:
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        text = normalize_text(node)
        if text:
            out.append(text)

    walk(value)
    return out


def prepare_memoryagentbench(raw_root: Path, out_root: Path, limits: dict[str, int]) -> dict[str, Any]:
    files = [
        ("Accurate_Retrieval", "mab_ar"),
        ("Conflict_Resolution", "mab_cr"),
        ("Long_Range_Understanding", "mab_lru"),
        ("Test_Time_Learning", "mab_ttl"),
    ]
    name = "memoryagentbench"
    out_dir = out_root / name
    episodes: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for category, key in files:
        raw = download(URLS[key], raw_root / "memoryagentbench" / f"{category}.parquet")
        df = read_parquet(raw)
        records = df.head(limits["episodes"]) if limits["episodes"] > 0 else df
        for row_idx, row in records.iterrows():
            episode_id = f"memoryagentbench:{category}:{row_idx}"
            signals: list[dict[str, Any]] = []
            context = normalize_text(row.get("context"))
            chunks = chunk_text(context, limits["context_chars"])
            if limits["turns_per_episode"] > 0:
                chunks = chunks[: limits["turns_per_episode"]]
            for c_idx, chunk in enumerate(chunks):
                signals.append(
                    {
                        "modality": "text",
                        "source_id": f"eval/{episode_id}/context/{c_idx}",
                        "text": chunk,
                    }
                )
            episodes.append({"episode_id": episode_id, "benchmark": name, "signals": signals})
            questions = as_list(row.get("questions"))
            answers = as_list(row.get("answers"))
            metadata = to_plain(row.get("metadata")) or {}
            qa_pair_ids = as_list(metadata.get("qa_pair_ids") if isinstance(metadata, dict) else None)
            for q_idx, question in enumerate(limit_rows(questions, limits["queries_per_episode"])):
                query_id = (
                    normalize_text(qa_pair_ids[q_idx])
                    if q_idx < len(qa_pair_ids)
                    else f"{episode_id}:{q_idx}"
                )
                answer_key.append(
                    {
                        "benchmark": name,
                        "conversation_id": episode_id,
                        "query_id": query_id,
                        "question": normalize_text(question),
                        "answers": flatten_answers(answers[q_idx] if q_idx < len(answers) else []),
                        "question_type": category,
                        "requires_abstention": False,
                    }
                )
    return write_entry(name, out_dir, episodes, answer_key)


def write_entry(
    name: str,
    out_dir: Path,
    episodes: list[dict[str, Any]],
    answer_key: list[dict[str, Any]],
) -> dict[str, Any]:
    episodes_path = out_dir / "episodes.jsonl"
    answer_key_path = out_dir / "answer_key.jsonl"
    episode_count = write_jsonl(episodes_path, episodes)
    query_count = write_jsonl(answer_key_path, answer_key)
    log(f"prepared {name}: episodes={episode_count} queries={query_count}")
    return {
        "name": name,
        "episodes": str(episodes_path),
        "answer_key": str(answer_key_path),
        "episode_count": episode_count,
        "query_count": query_count,
    }


def parse_benchmarks(raw: str) -> list[str]:
    if raw == "all":
        return list(BENCHMARKS)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(names) - set(BENCHMARKS))
    if unknown:
        raise SystemExit(f"Unknown benchmark(s): {', '.join(unknown)}")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare SOTA memory evals for Cortext.")
    parser.add_argument("--benchmarks", default="all", help="Comma list or 'all'.")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--raw-root", default=str(RAW_ROOT_DEFAULT))
    parser.add_argument("--out-root", default=str(OUT_ROOT_DEFAULT))
    parser.add_argument("--include-large", action="store_true", help="Include BEAM 500K/1M tiers.")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--limit-queries-per-episode", type=int, default=None)
    parser.add_argument("--limit-turns-per-episode", type=int, default=None)
    parser.add_argument("--limit-context-chars", type=int, default=None)
    parser.add_argument("--limit-lme-v2-trajectories", type=int, default=None)
    parser.add_argument("--limit-lme-v2-states", type=int, default=None)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    limits = profile_limits(args.profile)
    overrides = {
        "episodes": args.limit_episodes,
        "queries_per_episode": args.limit_queries_per_episode,
        "turns_per_episode": args.limit_turns_per_episode,
        "context_chars": args.limit_context_chars,
        "lme_v2_trajectories": args.limit_lme_v2_trajectories,
        "lme_v2_states": args.limit_lme_v2_states,
    }
    for key, value in overrides.items():
        if value is not None:
            limits[key] = value
    selected = parse_benchmarks(args.benchmarks)
    manifest: list[dict[str, Any]] = []

    if "longmemeval" in selected:
        manifest.append(prepare_longmemeval(raw_root, out_root, limits))
    if "longmemeval_v2" in selected:
        manifest.append(prepare_lme_v2(raw_root, out_root, limits))
    if "locomo" in selected:
        manifest.append(prepare_locomo(raw_root, out_root, limits))
    if "locomo_plus" in selected:
        manifest.append(prepare_locomo_plus(raw_root, out_root, limits))
    if "beam" in selected:
        manifest.extend(prepare_beam(raw_root, out_root, limits, args.include_large))
    if "memoryagentbench" in selected:
        manifest.append(prepare_memoryagentbench(raw_root, out_root, limits))

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "profile": args.profile,
                "benchmarks": manifest,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"wrote manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
