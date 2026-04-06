#!/usr/bin/env python3
"""
Generate OpenAI-labeled span examples for the offline semantic label-classifier study.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from build_label_training_data import iter_messages_with_metadata, write_examples
from label_classifier_lib import (
    generate_candidate_spans,
    lexical_features,
    load_name_priors,
    load_wordnet_index,
    normalize_text,
    split_sentences,
)


TYPE_LABELS = [
    "identity",
    "person_entity",
    "place",
    "org_project",
    "topic",
    "state",
    "none",
]

PROMOTION_LABELS = [
    "durable",
    "provisional",
    "ignore",
]


def load_shell_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def resolve_openai_settings(env_file: Path | None, model_override: str) -> dict[str, str]:
    values = dict(os.environ)
    if env_file is not None:
        values.update(load_shell_env(env_file))
    api_key = values.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required via environment or --env-file")
    base_url = values.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = model_override or values.get("OPENAI_LABEL_MODEL") or "gpt-5.4-mini-2026-03-17"
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def span_priority(sentence: str, span, wordnet_index: dict[str, dict], priors: dict[str, dict[str, float]]) -> float:
    features = lexical_features(sentence, span, wordnet_index, priors)
    score = 0.0
    score += features["name_given"] * 4.0
    score += features["name_surname"] * 4.0
    score += features["name_full_name"] * 5.0
    score += features["wordnet_place"] * 2.0
    score += features["wordnet_state"] * 2.0
    score += features["wordnet_topic"] * 1.5
    score += features["span_is_title"] * 1.25
    score += features["span_all_title"] * 0.75
    score += features["name_intro"] * 2.0
    score += features["state_cue"] * 1.0
    score += features["place_cue"] * 1.0
    score += features["work_cue"] * 1.25
    score += features["org_cue"] * 0.75
    score += features["work_context"] * 1.0
    score += min(features["span_token_count"], 3.0) * 0.15
    if features["span_is_helper_only"] > 0.0:
        score -= 3.0
    if features["span_is_stopword"] > 0.0:
        score -= 4.0
    return score


def select_spans(sentence: str, wordnet_index: dict[str, dict], priors: dict[str, dict[str, float]], max_candidates: int) -> list[dict]:
    ranked: list[tuple[float, int, object]] = []
    for idx, span in enumerate(generate_candidate_spans(sentence)):
        ranked.append((span_priority(sentence, span, wordnet_index, priors), idx, span))
    ranked.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    selected = []
    for score, idx, span in ranked[:max_candidates]:
        if score < -1.0:
            continue
        selected.append(
            {
                "id": idx,
                "text": span.text,
                "start": span.start,
                "end": span.end,
                "score": score,
            }
        )
    return selected


def build_prompt(sentence: str, spans: list[dict]) -> tuple[str, str]:
    system = (
        "You label candidate spans for Cortext memory formation.\n"
        "Return one label for every candidate span.\n"
        "Type labels:\n"
        "- identity: the speaker explicitly names themselves or their own stable personal identity.\n"
        "- person_entity: another person, or a named person mentioned by the speaker.\n"
        "- place: city, country, region, addressable location.\n"
        "- org_project: organization, product, tool, project, company, software, or branded system.\n"
        "- topic: subject/domain/concept worth remembering, but not a person/place/org.\n"
        "- state: emotional, physical, planning, condition, or ongoing task state.\n"
        "- none: generic word/phrase that should not be stored as a label.\n"
        "Promotion labels:\n"
        "- durable: stable identity/entity/place/org information likely useful later.\n"
        "- provisional: transient topic/state worth short-to-medium memory.\n"
        "- ignore: not worth storing.\n"
        "Important rules:\n"
        "- identity is only for the speaker's own explicit self-name/identity, not job roles or feelings.\n"
        "- project/tool names like SQLite, Redis, Cortext, OpenAI are org_project, not identity.\n"
        "- emotional adjectives like exhausted, overwhelmed, frustrated are state.\n"
        "- If unsure, choose none/ignore.\n"
        "- Use only the candidate IDs provided.\n"
    )
    candidate_lines = "\n".join(f"- id={row['id']}: {row['text']}" for row in spans)
    user = (
        f"Sentence:\n{sentence}\n\n"
        f"Candidate spans:\n{candidate_lines}\n\n"
        "Return JSON only."
    )
    return system, user


def response_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "span_labels",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "labels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "integer"},
                                "type_label": {"type": "string", "enum": TYPE_LABELS},
                                "promotion_label": {"type": "string", "enum": PROMOTION_LABELS},
                            },
                            "required": ["id", "type_label", "promotion_label"],
                        },
                    }
                },
                "required": ["labels"],
            },
        },
    }


def call_openai(settings: dict[str, str], sentence: str, spans: list[dict], max_retries: int) -> tuple[list[dict], dict]:
    system, user = build_prompt(sentence, spans)
    payload = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_completion_tokens": 1200,
        "response_format": response_schema(),
    }
    request_body = json.dumps(payload).encode("utf-8")
    url = settings["base_url"] + "/chat/completions"
    last_error = None
    for attempt in range(max_retries):
        request = urllib.request.Request(
            url,
            data=request_body,
            headers={
                "Authorization": f"Bearer {settings['api_key']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = json.loads(response.read().decode("utf-8"))
                content = raw["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return parsed.get("labels", []), raw.get("usage", {})
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:500]}"
            if exc.code in {408, 409, 429, 500, 502, 503, 504} and attempt + 1 < max_retries:
                time.sleep(min(8, 2 ** attempt))
                continue
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt + 1 < max_retries:
                time.sleep(min(8, 2 ** attempt))
                continue
            break
    raise RuntimeError(f"OpenAI labeling failed: {last_error}")


def build_examples_for_sentence(
    sentence_id: str,
    sentence: str,
    spans: list[dict],
    labels: list[dict],
    negative_rate: float,
    rng: random.Random,
    source: str,
    model: str,
) -> list[dict]:
    label_by_id = {
        int(row["id"]): {
            "type_label": row["type_label"],
            "promotion_label": row["promotion_label"],
        }
        for row in labels
    }
    examples: list[dict] = []
    for row in spans:
        assigned = label_by_id.get(int(row["id"]), {"type_label": "none", "promotion_label": "ignore"})
        if assigned["type_label"] == "none" and rng.random() > negative_rate:
            continue
        examples.append(
            {
                "id": f"{sentence_id}:{row['id']}",
                "text": sentence,
                "span": row["text"],
                "type_label": assigned["type_label"],
                "promotion_label": assigned["promotion_label"],
                "source": source,
                "weak_label": False,
                "annotator": model,
            }
        )
    return examples


def collect_sentence_pool(
    dataset_paths: list[Path],
    wordnet_index: dict[str, dict],
    priors: dict[str, dict[str, float]],
    max_candidates: int,
) -> list[dict]:
    pool: list[dict] = []
    for conversation_id, message, conversation_meta, turn_meta in iter_messages_with_metadata(dataset_paths):
        source_name = conversation_meta.get("dataset_name") or Path(str(conversation_id)).stem
        for sent_idx, sentence in enumerate(split_sentences(message)):
            sentence = normalize_text(sentence)
            if not sentence:
                continue
            spans = select_spans(sentence, wordnet_index, priors, max_candidates)
            if not spans:
                continue
            pool.append(
                {
                    "id": f"{conversation_id}:{sent_idx}",
                    "text": sentence,
                    "spans": spans,
                    "source": str(source_name),
                }
            )
    return pool


def split_rows(rows: list[dict], valid_fraction: float) -> tuple[list[dict], list[dict]]:
    rng = random.Random(23)
    rng.shuffle(rows)
    valid_count = max(1, int(len(rows) * valid_fraction))
    return rows[valid_count:], rows[:valid_count]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OpenAI-labeled span data.")
    parser.add_argument("--dataset", action="append", default=[], help="Prepared Cortext dataset JSONL.")
    parser.add_argument("--wordnet-index", required=True)
    parser.add_argument("--name-priors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--env-file", default="env.sh")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-sentences", type=int, default=120)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--negative-rate", type=float, default=0.35)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    env_file = Path(args.env_file) if args.env_file else None
    settings = resolve_openai_settings(env_file, args.model)
    dataset_paths = [Path(path) for path in args.dataset if Path(path).exists()]
    if not dataset_paths:
        raise SystemExit("At least one existing --dataset is required")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wordnet_index = load_wordnet_index(Path(args.wordnet_index))
    priors = load_name_priors(Path(args.name_priors))
    pool = collect_sentence_pool(dataset_paths, wordnet_index, priors, args.max_candidates)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    selected = pool[: min(args.max_sentences, len(pool))]

    all_examples: list[dict] = []
    usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for row in selected:
        labels, usage = call_openai(settings, row["text"], row["spans"], args.max_retries)
        for key in usage_totals:
            usage_totals[key] += int(usage.get(key, 0) or 0)
        all_examples.extend(
            build_examples_for_sentence(
                sentence_id=row["id"],
                sentence=row["text"],
                spans=row["spans"],
                labels=labels,
                negative_rate=args.negative_rate,
                rng=rng,
                source=row["source"],
                model=settings["model"],
            )
        )

    train_rows, valid_rows = split_rows(all_examples, args.valid_fraction)
    write_examples(out_dir / "train.jsonl", train_rows)
    write_examples(out_dir / "valid.jsonl", valid_rows)

    label_counts: dict[str, int] = {}
    for row in all_examples:
        label_counts[row["type_label"]] = label_counts.get(row["type_label"], 0) + 1
    summary = {
        "model": settings["model"],
        "datasets": [str(path) for path in dataset_paths],
        "selected_sentences": len(selected),
        "total_examples": len(all_examples),
        "train_examples": len(train_rows),
        "valid_examples": len(valid_rows),
        "type_counts": label_counts,
        "usage": usage_totals,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
