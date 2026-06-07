#!/usr/bin/env python3
"""Judge audio-pipeline STT output with a local Nemotron judge."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_MODEL = "nemotron-3-nano-omni-30b-a3b-8bit"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
LOCAL_JUDGE_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
SCORE_KEYS = (
    "semantic_preservation",
    "transcript_accuracy",
    "entity_preservation",
    "hallucination_control",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def clamp_score(value: object) -> float:
    try:
        return max(0.0, min(5.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def extract_json_object(content: str) -> dict:
    strict = True
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        strict = False
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError(f"Judge returned non-JSON content: {content[:500]!r}")
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise RuntimeError("Judge returned JSON that was not an object")
    parsed["_strict_json"] = strict
    return parsed


def require_nemotron_model(model: str) -> None:
    if "nemotron" not in model.lower():
        raise RuntimeError(
            f"Refusing non-Nemotron judge model for audio eval data: {model!r}. "
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
            "Refusing non-local judge endpoint for audio eval data: "
            f"{base_url!r}. Start the local Nemotron/MLX judge server and set "
            "CORTEXT_JUDGE_BASE_URL or LOCAL_JUDGE_BASE_URL to a loopback URL."
        )
    return base_url


def call_judge(model: str, prompt: str, max_tokens: int) -> dict:
    require_nemotron_model(model)
    base_url = local_judge_base_url()
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict speech-to-text quality judge. "
                    "Do not explain. Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
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
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Local judge failed: HTTP {exc.code}: {detail}") from exc
    content = payload["choices"][0]["message"]["content"]
    return extract_json_object(content)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = word_tokens(reference)
    hyp = word_tokens(hypothesis)
    return edit_distance(ref, hyp) / len(ref) if ref else (0.0 if not hyp else 1.0)


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def entity_like_tokens(text: str) -> set[str]:
    tokens = set()
    for match in re.finditer(r"\b(?:[A-Z][a-z]{2,}|[A-Z]{2,})\b", text):
        tokens.add(match.group(0).lower())
    return tokens


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        return 0 if not hyp else len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, 1):
            cost = 0 if ref_word == hyp_word else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def edit_counts(reference: str, hypothesis: str) -> dict:
    ref = word_tokens(reference)
    hyp = word_tokens(hypothesis)
    entities = entity_like_tokens(reference)
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    substitutions = deletions = insertions = 0
    number_ref_errors = 0
    entity_like_ref_errors = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            if any(ch.isdigit() for ch in ref[i - 1]):
                number_ref_errors += 1
            if ref[i - 1] in entities:
                entity_like_ref_errors += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            if any(ch.isdigit() for ch in ref[i - 1]):
                number_ref_errors += 1
            if ref[i - 1] in entities:
                entity_like_ref_errors += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    ref_count = max(1, n)
    return {
        "reference_tokens": n,
        "hypothesis_tokens": m,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "number_ref_errors": number_ref_errors,
        "entity_like_ref_errors": entity_like_ref_errors,
        "deletion_rate": deletions / ref_count,
        "insertion_rate": insertions / ref_count,
    }


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 20] + " ... [truncated]"


def build_prompt(reference: str, hypothesis: str) -> str:
    return "\n\n".join(
        [
            "Judge the STT transcript against the reference text.",
            "Return JSON with numeric 0-5 fields: semantic_preservation, transcript_accuracy, entity_preservation, hallucination_control. Higher is better for every field. Also return error_type as one of exact_or_near_exact, minor_wording, missing_entities, omitted_content, hallucinated_content, unintelligible, mixed_errors.",
            "Do not quote the private/reference text in your response.",
            "REFERENCE_TEXT:\n" + truncate(reference, 3500),
            "ASR_TRANSCRIPT:\n" + truncate(hypothesis, 3500),
        ]
    )


def wer_bucket(wer: float) -> str:
    if wer <= 0.05:
        return "0.00-0.05"
    if wer <= 0.10:
        return "0.05-0.10"
    if wer <= 0.20:
        return "0.10-0.20"
    if wer <= 0.35:
        return "0.20-0.35"
    return ">0.35"


def entity_bucket(score: float) -> str:
    if score >= 4.5:
        return "high_entity_preservation"
    if score >= 3.0:
        return "medium_entity_preservation"
    return "low_entity_preservation"


def summarize_group(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "mean_wer": mean([float(row["wer"]) for row in rows]),
        **{
            f"mean_{key}": mean([float(row[key]) for row in rows])
            for key in SCORE_KEYS
        },
    }


def grouped_summary(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {name: summarize_group(group) for name, group in sorted(groups.items())}


def failure_flags(row: dict) -> list[str]:
    counts = row["edit_counts"]
    flags = []
    if row["wer"] > 0.20:
        flags.append("high_wer")
    if row["entity_preservation"] < 3.0:
        flags.append("low_entity_preservation")
    if counts["entity_like_ref_errors"] > 0:
        flags.append("entity_like_token_error")
    if counts["number_ref_errors"] > 0:
        flags.append("number_or_citation_token_error")
    if counts["deletion_rate"] > 0.10:
        flags.append("omission_heavy")
    if counts["insertion_rate"] > 0.10:
        flags.append("insertion_heavy")
    return flags


def failure_mode_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for flag in row["failure_flags"]:
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    started_at = utc_now()

    rows = []
    totals = {
        "semantic_preservation": 0.0,
        "transcript_accuracy": 0.0,
        "entity_preservation": 0.0,
        "hallucination_control": 0.0,
        "wer": 0.0,
    }
    error_counts: dict[str, int] = {}

    with args.input.open() as infile:
        for line in infile:
            if args.limit >= 0 and len(rows) >= args.limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            reference = str(item.get("text", ""))
            hypothesis = str(item.get("asr_text", ""))
            if not reference or not hypothesis:
                continue
            judged = call_judge(args.model, build_prompt(reference, hypothesis), args.max_tokens)
            counts = edit_counts(reference, hypothesis)
            row = {
                "id": item.get("id", len(rows)),
                "input_row_index": len(rows),
                "source_cache": item.get("source_cache"),
                "source_id": item.get("source_id"),
                "wer": word_error_rate(reference, hypothesis),
                "semantic_preservation": clamp_score(judged.get("semantic_preservation")),
                "transcript_accuracy": clamp_score(judged.get("transcript_accuracy")),
                "entity_preservation": clamp_score(judged.get("entity_preservation")),
                "hallucination_control": clamp_score(judged.get("hallucination_control")),
                "error_type": str(judged.get("error_type", "mixed_errors")),
                "strict_json": bool(judged.get("_strict_json", False)),
                "edit_counts": counts,
            }
            row["wer_bucket"] = wer_bucket(float(row["wer"]))
            row["entity_bucket"] = entity_bucket(float(row["entity_preservation"]))
            row["failure_flags"] = failure_flags(row)
            rows.append(row)
            for key in totals:
                totals[key] += float(row[key])
            error_counts[row["error_type"]] = error_counts.get(row["error_type"], 0) + 1

    count = len(rows)
    strict_json_count = sum(1 for row in rows if row["strict_json"])
    summary = {
        "schema": "audio_pipeline_stt_nemotron_judge_v1",
        "input": str(args.input),
        "output": str(args.out),
        "started_at": started_at,
        "finished_at": utc_now(),
        "command": " ".join(sys.argv),
        "git": git_info(),
        "judge_model": args.model,
        "judge_provider": "local_nemotron_vllm_mlx",
        "judge_base_url": local_judge_base_url(),
        "remote_provider_allowed": False,
        "judged": count,
        "strict_json_count": strict_json_count,
        "strict_json_rate": strict_json_count / count if count else 0.0,
        "means": {key: (value / count if count else 0.0) for key, value in totals.items()},
        "error_counts": error_counts,
        "by_wer_bucket": grouped_summary(rows, "wer_bucket"),
        "by_entity_bucket": grouped_summary(rows, "entity_bucket"),
        "by_error_type": grouped_summary(rows, "error_type"),
        "failure_mode_counts": failure_mode_counts(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
