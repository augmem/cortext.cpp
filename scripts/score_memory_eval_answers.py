#!/usr/bin/env python3
"""Generate and score answers from Cortext memory-eval retrieval packets."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


ABSTAIN_MARKERS = (
    "i don't know",
    "i do not know",
    "insufficient",
    "not enough information",
    "not in the provided",
    "cannot determine",
    "unknown",
)
COMMAND_JUDGE_MODES = {"command", "harness"}


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def normalize_match_text(value: object) -> str:
    chars: list[str] = []
    last_space = True
    for ch in str(value or "").lower():
        if ch.isalnum():
            chars.append(ch)
            last_space = False
        elif not last_space:
            chars.append(" ")
            last_space = True
    if chars and chars[-1] == " ":
        chars.pop()
    return "".join(chars)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def answer_key_map(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(path):
        conversation_id = str(row.get("conversation_id") or row.get("episode_id") or "")
        query_id = str(row.get("query_id") or conversation_id)
        out[(conversation_id, query_id)] = row
    return out


def match_answer(prediction: str, answers: list[str]) -> bool:
    pred = normalize_match_text(prediction)
    if not pred:
        return False
    for answer in answers:
        gold = normalize_match_text(answer)
        if gold and (gold == pred or gold in pred or pred in gold):
            return True
    return False


def abstained(prediction: str) -> bool:
    pred = normalize_text(prediction)
    return any(marker in pred for marker in ABSTAIN_MARKERS)


def retrieved_packet(row: dict[str, Any], max_chars: int) -> str:
    parts: list[str] = []
    for item in row.get("retrieved", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("preview") or "")
        if text:
            parts.append(f"[{item.get('rank', len(parts) + 1)}] {text}")
    packet = "\n".join(parts)
    if max_chars > 0 and len(packet) > max_chars:
        return packet[:max_chars]
    return packet


def packet_answer(row: dict[str, Any], max_chars: int) -> str:
    packet = retrieved_packet(row, max_chars)
    if not packet:
        return "I don't know."
    return packet


def answer_prompt(row: dict[str, Any], max_packet_chars: int) -> str:
    packet = retrieved_packet(row, max_packet_chars)
    if not packet:
        packet = "(no retrieved memory snippets)"
    return (
        "Answer only from the provided memory snippets. "
        "If the snippets do not contain enough evidence, answer exactly: I don't know.\n\n"
        f"Question:\n{row.get('question', '')}\n\n"
        f"Memory snippets:\n{packet}\n\n"
        "Final answer:"
    )


def command_answer(
    row: dict[str, Any],
    command: str,
    timeout_s: int,
    max_packet_chars: int,
) -> str:
    if not command:
        raise RuntimeError("--judge-command or CORTEXT_EVAL_JUDGE_COMMAND is required")
    prompt = answer_prompt(row, max_packet_chars)
    input_doc = {
        "question": row.get("question", ""),
        "memory_snippets": retrieved_packet(row, max_packet_chars),
        "prompt": prompt,
        "benchmark": row.get("benchmark"),
        "conversation_id": row.get("conversation_id"),
        "query_id": row.get("query_id"),
        "question_type": row.get("question_type"),
    }
    env = os.environ.copy()
    env.update(
        {
            "CORTEXT_EVAL_BENCHMARK": str(row.get("benchmark") or ""),
            "CORTEXT_EVAL_CONVERSATION_ID": str(row.get("conversation_id") or ""),
            "CORTEXT_EVAL_QUERY_ID": str(row.get("query_id") or ""),
            "CORTEXT_EVAL_QUESTION_TYPE": str(row.get("question_type") or ""),
        }
    )
    with tempfile.TemporaryDirectory(prefix="cortext_judge_") as temp_dir:
        temp_path = Path(temp_dir)
        prompt_path = temp_path / "prompt.txt"
        input_json_path = temp_path / "input.json"
        prompt_path.write_text(prompt, encoding="utf-8")
        input_json_path.write_text(
            json.dumps(input_doc, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        expanded_command = command.replace("{prompt_file}", shlex.quote(str(prompt_path)))
        expanded_command = expanded_command.replace(
            "{input_json}", shlex.quote(str(input_json_path))
        )
        proc = subprocess.run(
            expanded_command,
            input=prompt,
            text=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            env=env,
            check=False,
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:1000]
        raise RuntimeError(f"judge command failed with exit {proc.returncode}: {detail}")
    return proc.stdout.strip()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def chat_completion_answer(
    row: dict[str, Any],
    model: str,
    base_url: str,
    api_key: str,
    timeout_s: int,
    max_packet_chars: int,
) -> str:
    packet = retrieved_packet(row, max_packet_chars)
    if not packet:
        return "I don't know."

    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer only from the provided memory snippets. "
                    "If the snippets do not contain enough evidence, answer exactly: I don't know."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{row.get('question', '')}\n\n"
                    f"Memory snippets:\n{packet}\n\n"
                    "Final answer:"
                ),
            },
        ],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"chat completion failed: HTTP {exc.code}: {detail}") from exc
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def score_rows(
    benchmark: str,
    query_rows: Path,
    answer_key: Path,
    out_dir: Path,
    mode: str,
    model: str,
    base_url: str,
    api_key: str,
    judge_command: str,
    judge_label: str,
    timeout_s: int,
    max_packet_chars: int,
) -> dict[str, Any]:
    gold = answer_key_map(answer_key)
    predictions_path = out_dir / "predictions.jsonl"
    counts = {
        "n": 0,
        "scored": 0,
        "answerable": 0,
        "correct": 0,
        "abstention_required": 0,
        "abstention_correct": 0,
    }
    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "scored": 0, "answerable": 0, "correct": 0}
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as out:
        for row in load_jsonl(query_rows):
            key = (str(row.get("conversation_id", "")), str(row.get("query_id", "")))
            gold_row = gold.get(key, {})
            if mode == "openai":
                prediction = chat_completion_answer(
                    row, model, base_url, api_key, timeout_s, max_packet_chars
                )
            elif mode in COMMAND_JUDGE_MODES:
                prediction = command_answer(row, judge_command, timeout_s, max_packet_chars)
            else:
                prediction = packet_answer(row, max_packet_chars)

            answers = [str(item) for item in gold_row.get("answers", [])]
            requires_abstention = bool(gold_row.get("requires_abstention", False))
            if requires_abstention:
                scored = True
                correct = abstained(prediction)
                counts["abstention_required"] += 1
                if correct:
                    counts["abstention_correct"] += 1
            else:
                if answers:
                    scored = True
                    correct = match_answer(prediction, answers)
                    counts["answerable"] += 1
                else:
                    scored = False
                    correct = False
            counts["n"] += 1
            if scored:
                counts["scored"] += 1
            if scored and correct:
                counts["correct"] += 1

            question_type = str(gold_row.get("question_type") or row.get("question_type") or "unknown")
            by_type[question_type]["n"] += 1
            if scored:
                by_type[question_type]["scored"] += 1
            if answers and not requires_abstention:
                by_type[question_type]["answerable"] += 1
            if scored and correct:
                by_type[question_type]["correct"] += 1

            out.write(
                json.dumps(
                    {
                        "benchmark": benchmark,
                        "conversation_id": row.get("conversation_id"),
                        "query_id": row.get("query_id"),
                        "question": row.get("question") or gold_row.get("question"),
                        "question_type": question_type,
                        "prediction": prediction,
                        "scored": scored,
                        "correct": correct,
                        "requires_abstention": requires_abstention,
                        "mode": mode,
                        "judge_label": judge_label,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    def ratio(num: int, den: int) -> float:
        return float(num) / float(den) if den else 0.0

    summary = {
        "benchmark": benchmark,
        "mode": mode,
        "model": model if mode == "openai" else "",
        "judge_label": judge_label,
        "judge_command_configured": bool(judge_command) if mode in COMMAND_JUDGE_MODES else False,
        "query_rows": str(query_rows),
        "answer_key": str(answer_key),
        "predictions": str(predictions_path),
        "n": counts["n"],
        "scored": counts["scored"],
        "correct": counts["correct"],
        "accuracy": ratio(counts["correct"], counts["scored"]),
        "answerable": counts["answerable"],
        "answer_accuracy": ratio(counts["correct"], counts["answerable"]),
        "abstention_required": counts["abstention_required"],
        "abstention_accuracy": ratio(
            counts["abstention_correct"], counts["abstention_required"]
        ),
        "by_question_type": {
            key: {
                "n": value["n"],
                "scored": value["scored"],
                "answerable": value["answerable"],
                "accuracy": ratio(value["correct"], value["scored"]),
            }
            for key, value in sorted(by_type.items())
        },
    }
    (out_dir / "answer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def safe_label(value: str) -> str:
    chars: list[str] = []
    for ch in value.lower():
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        elif ch in (" ", ".", "/"):
            chars.append("_")
    label = "".join(chars).strip("_")
    return label or "command"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score memory eval answer predictions.")
    parser.add_argument("--run", required=True, help="Run directory containing summary.json.")
    parser.add_argument(
        "--mode",
        choices=["packet", "openai", "command", "harness"],
        default="packet",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument(
        "--judge-command",
        default="",
        help=(
            "Shell command for --mode command or harness. The prompt is sent on stdin; "
            "{prompt_file} and {input_json} are also expanded to temporary files."
        ),
    )
    parser.add_argument(
        "--judge-label",
        default="",
        help="Optional label for command-judge output directories, such as codex or grok.",
    )
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--max-packet-chars", type=int, default=12000)
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    run_dir = Path(args.run)
    aggregate = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    model = args.model or os.environ.get("CORTEXT_EVAL_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5-mini"
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if args.mode == "openai" and not api_key:
        raise SystemExit("OPENAI_API_KEY is required for --mode openai")
    judge_command = args.judge_command or os.environ.get("CORTEXT_EVAL_JUDGE_COMMAND", "")
    judge_label_source = args.judge_label or os.environ.get("CORTEXT_EVAL_JUDGE_LABEL", "")
    judge_label = safe_label(
        judge_label_source or ("openai" if args.mode == "openai" else args.mode)
    )
    if args.mode in COMMAND_JUDGE_MODES and not judge_command:
        raise SystemExit(
            "--judge-command or CORTEXT_EVAL_JUDGE_COMMAND is required for command judges"
        )
    output_tag = (
        args.mode
        if args.mode not in COMMAND_JUDGE_MODES or not judge_label_source
        else f"{args.mode}_{judge_label}"
    )

    summaries = []
    for run in aggregate.get("runs", []):
        metrics = run.get("metrics") or {}
        if run.get("returncode") != 0 or not metrics:
            continue
        benchmark = str(run.get("name"))
        out_dir = run_dir / benchmark / f"answers_{output_tag}"
        summaries.append(
            score_rows(
                benchmark=benchmark,
                query_rows=Path(run["query_rows"]),
                answer_key=Path(metrics["answer_key_path"]),
                out_dir=out_dir,
                mode=args.mode,
                model=model,
                base_url=base_url,
                api_key=api_key,
                judge_command=judge_command,
                judge_label=judge_label,
                timeout_s=args.timeout_s,
                max_packet_chars=args.max_packet_chars,
            )
        )

    total_n = sum(item["n"] for item in summaries)
    total_scored = sum(item["scored"] for item in summaries)
    total_correct = sum(int(item["correct"]) for item in summaries)
    output = {
        "run": str(run_dir),
        "mode": args.mode,
        "model": model if args.mode == "openai" else "",
        "judge_label": judge_label,
        "judge_command_configured": bool(judge_command)
        if args.mode in COMMAND_JUDGE_MODES
        else False,
        "generated_at": time.time(),
        "accuracy": (float(total_correct) / float(total_scored)) if total_scored else 0.0,
        "n": total_n,
        "scored": total_scored,
        "benchmarks": summaries,
    }
    out_path = run_dir / f"answer_summary_{output_tag}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
