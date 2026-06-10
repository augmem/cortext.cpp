#!/usr/bin/env python3
"""Judge a privacy-safe chat-replay raw-speech memory eval artifact.

The generation transcripts are external ground truth for judging only. Cortext
ingests raw audio blobs and embeddings in the eval; no ASR transcript or
transcript-derived text is passed back into Cortext.
"""

from __future__ import annotations

from chat_replay_corpus import discover_transcript

import argparse
import json
import os
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request

from generate_chat_replay_raw_speech_manifest import parse_messages


DEFAULT_MODEL = "nemotron-3-nano-omni-30b-a3b-8bit"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
LOCAL_JUDGE_HOSTS = {"localhost", "127.0.0.1", "::1"}


def require_nemotron_model(model: str) -> None:
    if "nemotron" not in model.lower():
        raise RuntimeError(
            f"Refusing non-Nemotron judge model for private chat-replay data: {model!r}. "
            "Start the local Nemotron/MLX judge server and pass --model nemotron..."
        )


def local_judge_base_url() -> str:
    base_url = (
        os.environ.get("CORTEXT_JUDGE_BASE_URL")
        or os.environ.get("LOCAL_JUDGE_BASE_URL")
        or DEFAULT_LOCAL_BASE_URL
    ).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOCAL_JUDGE_HOSTS:
        raise RuntimeError(
            "Refusing non-local judge endpoint for private chat-replay raw-speech eval: "
            f"{base_url!r}. Start the local Nemotron/MLX judge server and set "
            "CORTEXT_JUDGE_BASE_URL or LOCAL_JUDGE_BASE_URL to a loopback URL."
        )
    return base_url


def tokens(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "you",
        "that",
        "for",
        "with",
        "this",
        "have",
        "just",
        "but",
        "not",
        "are",
        "was",
        "what",
        "from",
        "they",
        "your",
        "our",
        "she",
        "him",
        "her",
        "his",
        "them",
        "then",
        "there",
        "were",
        "been",
        "will",
        "would",
        "could",
        "should",
        "about",
        "like",
        "yeah",
        "okay",
    }
    out: set[str] = set()
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            value = "".join(cur)
            if len(value) >= 3 and value not in stop:
                out.add(value)
            cur = []
    if cur:
        value = "".join(cur)
        if len(value) >= 3 and value not in stop:
            out.add(value)
    return out


def source_original_index(source_id: str) -> int | None:
    match = re.search(r"/raw_speech/\d+/(\d+)$", source_id)
    if not match:
        return None
    return int(match.group(1))


def cortext_indices(probe: dict, key: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for row in probe.get(key, []):
        idx = row.get("original_index")
        if idx is None:
            idx = source_original_index(row.get("source_id", ""))
        try:
            idx = int(idx) if idx is not None else None
        except (TypeError, ValueError):
            idx = None
        if idx is not None and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def lexical_top_k(prior_indices: list[int], messages: list[dict], query: str, k: int) -> list[int]:
    query_tokens = tokens(query)
    scored: list[tuple[float, int]] = []
    query_ts = messages[prior_indices[-1]]["timestamp"] if prior_indices else 0
    for idx in prior_indices:
        text_tokens = tokens(messages[idx]["text"])
        overlap = len(query_tokens & text_tokens)
        recency = 0.0
        if query_ts:
            recency = 1.0 / (1.0 + max(0, query_ts - messages[idx]["timestamp"]) / 86400000.0)
        score = float(overlap) + 0.05 * recency
        if score > 0.0:
            scored.append((score, idx))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[:k]]


def packet(name: str, indices: list[int], messages: list[dict], limit: int) -> str:
    lines = [f"{name}:"]
    selected = indices if limit < 0 else indices[:limit]
    for idx in selected:
        role = "contact" if messages[idx]["from_contact"] else "self"
        lines.append(
            f"- source_index={idx} speaker={role} timestamp={messages[idx]['timestamp']}: {messages[idx]['text']}"
        )
    if len(lines) == 1:
        lines.append("- <empty>")
    return "\n".join(lines)


def call_nemotron_judge(model: str, prompt: str, max_tokens: int) -> dict | None:
    require_nemotron_model(model)
    base_url = local_judge_base_url()
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict retrieval-quality judge. Do not explain. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("CORTEXT_JUDGE_API_KEY") or os.environ.get("LOCAL_JUDGE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Nemotron judge failed: HTTP {exc.code}: {detail}") from exc
    content = payload["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(content[start : end + 1])
        else:
            raise RuntimeError(f"Judge returned non-JSON content: {content[:500]!r}")
    return parsed if isinstance(parsed, dict) else None


def score_system(scores: dict, system: str, field: str) -> float:
    system_scores = scores.get(system, {})
    if not isinstance(system_scores, dict):
        return 0.0
    value = system_scores.get(field, 0)
    try:
        return max(0.0, min(5.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-limit", type=int, default=-1)
    parser.add_argument("--context-limit", type=int, default=-1)
    parser.add_argument("--rag-top-k", type=int, default=5)
    args = parser.parse_args()

    eval_data = json.loads(args.eval.read_text())
    manifest_path = args.manifest or pathlib.Path(eval_data["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    input_dir = pathlib.Path(manifest["input_dir"])
    messages = parse_messages(discover_transcript(input_dir))
    records = manifest["records"]
    original_by_local = {r["local_index"]: r["original_index"] for r in records}

    judged: list[dict] = []
    modality = eval_data.get("modality", "audio")
    systems = [
        "cortext_native",
        "traditional_chat_rag",
        "full_history_upper_bound",
    ]
    aggregates = {
        system: {
            "judged": 0,
            "relevance": 0.0,
            "sufficiency": 0.0,
            "noise": 0.0,
            "temporal_correctness": 0.0,
            "source_grounding": 0.0,
            "wins": 0,
        }
        for system in systems
    }

    for probe in eval_data["probes"]:
        if args.judge_limit >= 0 and len(judged) >= args.judge_limit:
            break
        query_idx = int(probe["original_index"])
        local_index = int(probe["local_index"])
        prior_indices = [
            original_by_local[r["local_index"]]
            for r in records
            if int(r["local_index"]) < local_index
        ]
        query_text = messages[query_idx]["text"]
        cortext_native = cortext_indices(probe, "cortext_native_sources")
        if not cortext_native:
            cortext_native = cortext_indices(probe, "cortext_raw_audio_only_sources")
        rag_indices = list(dict.fromkeys(prior_indices + lexical_top_k(prior_indices, messages, query_text, args.rag_top_k)))
        full_history_indices = prior_indices
        if modality == "audio":
            cortext_contract = (
                "Cortext saw only raw generated speech audio and stored source blobs; "
                "it did not receive ASR text, generation transcripts, or transcript labels. "
                "The transcript text below is the known text used to generate the speech; "
                "use it only as external ground truth for this private judge."
            )
        else:
            cortext_contract = (
                "Cortext saw the original text turn through the normal text API. "
                "The text below is the current conversation turn and ground truth."
            )

        prompt = "\n\n".join(
            [
                "Score each candidate packet for the current conversation turn.",
                cortext_contract,
                "Use 0-5 integer scores. relevance means the packet contains information useful to this turn. sufficiency means enough context to answer. noise means distracting unrelated content, where 0 is clean. temporal_correctness means ordering/date cues are preserved. source_grounding means the packet has traceable source-backed evidence.",
                "Return strict JSON. The keys cortext_native, traditional_chat_rag, and full_history_upper_bound must each map to an object with numeric keys relevance, sufficiency, noise, temporal_correctness, and source_grounding. Also return winner and failure_reason. failure_reason must be one of: speaker_confusion, temporal_drift, missing_entities, generic_audio_embeddings, media_adjacent_miss, unrelated_retrieval, insufficient_context, rag_context_advantage, full_history_upper_bound_advantage, tie_or_unclear. Do not quote private text in the JSON.",
                f"CURRENT_TURN:\nsource_index={query_idx} timestamp={messages[query_idx]['timestamp']}: {query_text}",
                packet("CORTEXT_NATIVE_WM_STM_LTM", cortext_native, messages, args.context_limit),
                packet("TRADITIONAL_CHAT_RAG", rag_indices, messages, args.context_limit),
                packet("FULL_HISTORY_UPPER_BOUND", full_history_indices, messages, args.context_limit),
            ]
        )
        scores = call_nemotron_judge(args.model, prompt, 700)
        if not scores:
            continue
        row = {
            "local_index": local_index,
            "original_index": query_idx,
            "winner": scores.get("winner", "unknown"),
            "failure_reason": scores.get("failure_reason", "unknown"),
            "systems": {},
        }
        allowed_reasons = {
            "speaker_confusion",
            "temporal_drift",
            "missing_entities",
            "generic_audio_embeddings",
            "media_adjacent_miss",
            "unrelated_retrieval",
            "insufficient_context",
            "rag_context_advantage",
            "full_history_upper_bound_advantage",
            "tie_or_unclear",
        }
        if row["failure_reason"] not in allowed_reasons:
            row["failure_reason"] = "tie_or_unclear"
        for system in systems:
            aggregates[system]["judged"] += 1
            if row["winner"] == system:
                aggregates[system]["wins"] += 1
            for field in [
                "relevance",
                "sufficiency",
                "noise",
                "temporal_correctness",
                "source_grounding",
            ]:
                value = score_system(scores, system, field)
                aggregates[system][field] += value
                row["systems"].setdefault(system, {})[field] = value
        judged.append(row)

    summary = {
        "schema": "chat_replay_native_memory_judge_v1",
        "eval_path": str(args.eval),
        "manifest_path": str(manifest_path),
        "judge_model": args.model,
        "judge_provider": "local_nemotron_vllm_mlx",
        "judge_base_url": local_judge_base_url(),
        "remote_provider_allowed": False,
        "modality": modality,
        "cortext_input_modality": eval_data.get("cortext_input_modality"),
        "judge_ground_truth": "generation_transcripts_only" if modality == "audio" else "source_text",
        "asr_transcript_used": False,
        "wer_reported": False,
        "transcript_text_passed_to_cortext": bool(eval_data.get("transcript_text_passed_to_cortext", False)),
        "transcript_text_written_to_artifact": False,
        "judged": len(judged),
        "quality": {},
        "judgments": judged,
    }
    for system, agg in aggregates.items():
        n = max(1, agg["judged"])
        summary["quality"][system] = {
            "judged": agg["judged"],
            "wins": agg["wins"],
            "mean_relevance": agg["relevance"] / n,
            "mean_sufficiency": agg["sufficiency"] / n,
            "mean_noise": agg["noise"] / n,
            "mean_temporal_correctness": agg["temporal_correctness"] / n,
            "mean_source_grounding": agg["source_grounding"] / n,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
