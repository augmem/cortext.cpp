#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}

FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i am|my|me|mine|we|our|ours)\b", re.I)
NAME_INTRO_RE = re.compile(
    r"\b(my name is|call me|this is|i go by|i am called|i'm called)\b", re.I
)
STATE_CUE_RE = re.compile(
    r"\b(i feel|i'm feeling|i am feeling|i was|i'm|i am|we are|we're)\b", re.I
)
SELF_NAME_COPULA_RE = re.compile(r"\b(i am|i'm)\b", re.I)
WORK_CUE_RE = re.compile(
    r"\b(debugging|building|working on|migrating|deploying|coding|using|for)\b",
    re.I,
)
PLACE_CUE_RE = re.compile(
    r"\b(in|to|from|at|visit|visiting|moved to|flying to|traveling to|live in)\b",
    re.I,
)
ORG_CUE_RE = re.compile(r"\b(at|for|using|project|company|team)\b", re.I)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'._-]*")
NAME_HELPER_WORDS = {"name", "call"}
PLACE_HELPER_WORDS = {
    "live",
    "living",
    "moved",
    "move",
    "visit",
    "visiting",
    "travel",
    "traveling",
    "flying",
}
STATE_HELPER_WORDS = {
    "feel",
    "feeling",
    "am",
    "are",
    "is",
    "was",
    "were",
    "today",
    "yesterday",
    "tomorrow",
    "next",
    "month",
}
WORK_HELPER_WORDS = {
    "debugging",
    "building",
    "working",
    "migrating",
    "deploying",
    "coding",
    "using",
    "project",
}
NAME_PRIOR_MIN = 0.01

WORDNET_GROUPS = {
    "person": {"noun.person"},
    "place": {"noun.location"},
    "state": {"noun.state", "adj.all", "adj.ppl"},
    "topic": {
        "noun.act",
        "noun.artifact",
        "noun.cognition",
        "noun.communication",
        "noun.event",
        "noun.group",
        "noun.process",
        "noun.substance",
        "noun.time",
        "verb.communication",
        "verb.creation",
        "verb.cognition",
    },
}


@dataclass(frozen=True)
class CandidateSpan:
    text: str
    start: int
    end: int
    token_start: int
    token_end: int


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def normalize_key(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


def generate_candidate_spans(text: str, max_tokens: int = 4) -> list[CandidateSpan]:
    tokens = tokenize_with_offsets(text)
    if not tokens:
        return []
    spans: list[CandidateSpan] = []
    seen: set[tuple[int, int]] = set()
    for start in range(len(tokens)):
        for width in range(1, max_tokens + 1):
            end_idx = start + width - 1
            if end_idx >= len(tokens):
                break
            token_slice = tokens[start : end_idx + 1]
            raw = text[token_slice[0][1] : token_slice[-1][2]]
            norm = normalize_key(raw)
            if not norm:
                continue
            normalized_tokens = [normalize_key(tok) for tok, _, _ in token_slice if normalize_key(tok)]
            if not normalized_tokens:
                continue
            non_stop_count = sum(1 for token in normalized_tokens if token not in STOPWORDS)
            if non_stop_count == 0:
                continue
            if all(tok.lower() in STOPWORDS for tok, _, _ in token_slice):
                continue
            if len(token_slice) > 1:
                first_token = normalized_tokens[0]
                last_token = normalized_tokens[-1]
                if first_token in STOPWORDS or last_token in STOPWORDS:
                    continue
                if non_stop_count <= 1:
                    continue
                if float(non_stop_count) / float(len(normalized_tokens)) < 0.5:
                    continue
            key = (token_slice[0][1], token_slice[-1][2])
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                CandidateSpan(
                    text=raw,
                    start=token_slice[0][1],
                    end=token_slice[-1][2],
                    token_start=start,
                    token_end=end_idx + 1,
                )
            )
    return spans


def stable_hash_index(value: str, dim: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def hashed_text_vector(text: str, dim: int = 128) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    lowered = normalize_key(text)
    if not lowered:
        return vector
    for token in lowered.split():
        vector[stable_hash_index("tok:" + token, dim)] += 1.0
    chars = lowered.replace(" ", "_")
    for n in (3, 4):
        for idx in range(max(0, len(chars) - n + 1)):
            gram = chars[idx : idx + n]
            vector[stable_hash_index(f"char{n}:{gram}", dim)] += 0.25
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


class ExternalTextEmbedder:
    def __init__(self, binary: str, models_dir: str = "models"):
        self.binary = binary
        self.models_dir = models_dir
        self.cache: dict[str, np.ndarray] = {}

    def prefetch(self, texts: Iterable[str]) -> None:
        self.encode_many(texts)

    def encode_many(self, texts: Iterable[str]) -> dict[str, np.ndarray]:
        missing = [text for text in dict.fromkeys(texts) if text not in self.cache and normalize_text(text)]
        if missing:
            with tempfile.TemporaryDirectory(prefix="label_embedder_") as tmpdir:
                input_path = Path(tmpdir) / "texts.txt"
                output_path = Path(tmpdir) / "embeddings.jsonl"
                input_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
                command = [
                    self.binary,
                    f"--input={input_path}",
                    f"--out={output_path}",
                    f"--models={self.models_dir}",
                ]
                subprocess.run(command, check=True, capture_output=True, text=True)
                with output_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        self.cache[row["text"]] = np.asarray(row["embedding"], dtype=np.float32)
        return {text: self.cache[text] for text in texts if text in self.cache}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_name_priors(path: Path) -> dict[str, dict[str, float]]:
    payload = load_json(path)
    return {
        "given": {normalize_key(key): float(value) for key, value in payload.get("given", {}).items()},
        "surname": {normalize_key(key): float(value) for key, value in payload.get("surname", {}).items()},
    }


def load_wordnet_index(path: Path) -> dict[str, dict[str, float]]:
    payload = load_json(path)
    return {normalize_key(key): value for key, value in payload.get("lemmas", {}).items()}


def wordnet_group_scores(span: str, wordnet_index: dict[str, dict[str, float]]) -> dict[str, float]:
    result = {group: 0.0 for group in WORDNET_GROUPS}
    entry = wordnet_index.get(normalize_key(span))
    if not entry:
        return result
    lexfiles = entry.get("lexfiles", {})
    for group, family in WORDNET_GROUPS.items():
        score = 0.0
        for lexfile, count in lexfiles.items():
            if lexfile in family:
                score += float(count)
        result[group] = score
    return result


def name_scores(span: str, priors: dict[str, dict[str, float]]) -> dict[str, float]:
    tokens = []
    for token, _, _ in tokenize_with_offsets(span):
        cleaned = normalize_key(token).strip(".,!?;:'\"()[]{}")
        if cleaned:
            tokens.append(cleaned)
    given = priors.get("given", {})
    surname = priors.get("surname", {})
    given_hits = [token for token in tokens if given.get(token, 0.0) >= NAME_PRIOR_MIN]
    surname_hits = [token for token in tokens if surname.get(token, 0.0) >= NAME_PRIOR_MIN]
    given_score = max((given.get(token, 0.0) for token in tokens if given.get(token, 0.0) >= NAME_PRIOR_MIN), default=0.0)
    surname_score = max((surname.get(token, 0.0) for token in tokens if surname.get(token, 0.0) >= NAME_PRIOR_MIN), default=0.0)
    full_name = 1.0 if len(tokens) >= 2 and given_score > 0.0 and surname_score > 0.0 else 0.0
    return {
        "given": given_score,
        "surname": surname_score,
        "full_name": full_name,
        "given_count": float(len(given_hits)),
        "surname_count": float(len(surname_hits)),
        "token_ratio": float(len(set(given_hits + surname_hits))) / float(max(1, len(tokens))),
    }


def lexical_features(sentence: str, span: CandidateSpan, wordnet_index: dict[str, dict[str, float]], priors: dict[str, dict[str, float]]) -> dict[str, float]:
    before = sentence[: span.start]
    after = sentence[span.end :]
    span_text = span.text
    span_norm = normalize_key(span_text)
    wordnet_scores = wordnet_group_scores(span_text, wordnet_index)
    person_name_scores = name_scores(span_text, priors)
    tokens = tokenize_with_offsets(span_text)
    normalized_tokens = [normalize_key(token) for token, _, _ in tokens if normalize_key(token)]
    title_tokens = sum(1 for token, _, _ in tokens if token[:1].isupper())
    alnum_tokens = max(len(tokens), 1)
    helper_only = 0.0
    helper_token_set = NAME_HELPER_WORDS | PLACE_HELPER_WORDS | STATE_HELPER_WORDS | WORK_HELPER_WORDS
    if normalized_tokens and all(token in helper_token_set for token in normalized_tokens):
        helper_only = 1.0
    features = {
        "span_token_count": float(len(tokens)),
        "span_char_count": float(len(span_text)),
        "span_is_title": 1.0 if span_text[:1].isupper() else 0.0,
        "span_all_title": 1.0 if normalized_tokens and title_tokens == len(normalized_tokens) else 0.0,
        "span_title_ratio": float(title_tokens) / float(alnum_tokens),
        "span_has_digit": 1.0 if any(char.isdigit() for char in span_text) else 0.0,
        "span_has_dot": 1.0 if "." in span_text else 0.0,
        "self_reference": 1.0 if FIRST_PERSON_RE.search(sentence) else 0.0,
        "name_intro": 1.0 if NAME_INTRO_RE.search(sentence) else 0.0,
        "state_cue": 1.0 if STATE_CUE_RE.search(sentence) else 0.0,
        "work_cue": 1.0 if WORK_CUE_RE.search(sentence) else 0.0,
        "place_cue": 1.0 if PLACE_CUE_RE.search(sentence) else 0.0,
        "org_cue": 1.0 if ORG_CUE_RE.search(sentence) else 0.0,
        "sentence_title_density": float(sum(1 for tok, _, _ in tokenize_with_offsets(sentence) if tok[:1].isupper()))
        / float(max(1, len(tokenize_with_offsets(sentence)))),
        "before_has_name_intro": 1.0 if NAME_INTRO_RE.search(before) else 0.0,
        "before_has_self_name_copula": 1.0 if SELF_NAME_COPULA_RE.search(before) else 0.0,
        "before_has_place_cue": 1.0 if PLACE_CUE_RE.search(before) else 0.0,
        "before_has_org_cue": 1.0 if ORG_CUE_RE.search(before) else 0.0,
        "before_has_state_cue": 1.0 if STATE_CUE_RE.search(before) else 0.0,
        "after_has_work_cue": 1.0 if WORK_CUE_RE.search(after) else 0.0,
        "span_is_name_helper": 1.0 if normalized_tokens and all(token in NAME_HELPER_WORDS for token in normalized_tokens) else 0.0,
        "span_is_place_helper": 1.0 if normalized_tokens and all(token in PLACE_HELPER_WORDS for token in normalized_tokens) else 0.0,
        "span_is_state_helper": 1.0 if normalized_tokens and all(token in STATE_HELPER_WORDS for token in normalized_tokens) else 0.0,
        "span_is_work_helper": 1.0 if normalized_tokens and all(token in WORK_HELPER_WORDS for token in normalized_tokens) else 0.0,
        "span_is_helper_only": helper_only,
        "work_context": 1.0 if WORK_CUE_RE.search(sentence) or ORG_CUE_RE.search(before) or WORK_CUE_RE.search(after) else 0.0,
    }
    for key, value in wordnet_scores.items():
        features[f"wordnet_{key}"] = value
    for key, value in person_name_scores.items():
        features[f"name_{key}"] = value
    features["span_is_stopword"] = 1.0 if span_norm in STOPWORDS else 0.0
    return features


def weak_type_and_promotion(
    sentence: str,
    span: CandidateSpan,
    wordnet_index: dict[str, dict[str, float]],
    priors: dict[str, dict[str, float]],
    conversation_meta: dict | None = None,
    turn_meta: dict | None = None,
) -> tuple[str, str]:
    features = lexical_features(sentence, span, wordnet_index, priors)
    span_text = span.text
    span_norm = normalize_key(span_text)
    conversation_meta = conversation_meta or {}
    turn_meta = turn_meta or {}

    emotion_labels: set[str] = set()
    emotion_value = normalize_key(turn_meta.get("emotion", ""))
    if emotion_value and emotion_value != "neutral":
        emotion_labels.add(emotion_value)
    for raw_label in turn_meta.get("labels", []) or []:
        label = normalize_key(str(raw_label))
        if label and label != "neutral":
            emotion_labels.add(label)
    emotional_supervision = 1.0 if emotion_labels else 0.0
    task_supervision = 1.0 if normalize_key(conversation_meta.get("task_domain", "")) else 0.0
    persona_supervision = 1.0 if conversation_meta.get("personas") else 0.0

    if (
        features["span_is_stopword"] > 0.0
        and features["name_given"] == 0.0
        and features["name_surname"] == 0.0
        and features["span_is_title"] == 0.0
    ):
        return ("none", "ignore")
    if features["span_is_helper_only"] > 0.0:
        return ("none", "ignore")

    if (
        features["name_given"] > 0.0
        and features["name_token_ratio"] >= 0.5
        and features["span_token_count"] <= 2.0
        and (
            features["before_has_name_intro"] > 0.0
            or features["name_intro"] > 0.0
            or features["before_has_self_name_copula"] > 0.0
        )
    ):
        return ("identity", "durable")
    if persona_supervision > 0.0 and features["self_reference"] > 0.0 and (
        features["before_has_name_intro"] > 0.0 or features["name_intro"] > 0.0
    ):
        return ("identity", "durable")
    if features["name_full_name"] > 0.0 or (
        features["name_given"] > 0.0
        and features["span_is_title"] > 0.0
        and features["name_token_ratio"] >= 0.5
        and features["span_token_count"] <= 3.0
    ):
        return ("person_entity", "durable")
    if features["wordnet_place"] > 0.0 and (
        features["before_has_place_cue"] > 0.0 or features["place_cue"] > 0.0
    ):
        return ("place", "durable")
    if (
        features["span_is_title"] > 0.0
        and features["name_token_ratio"] == 0.0
        and (
            features["before_has_org_cue"] > 0.0
            or features["org_cue"] > 0.0
            or features["work_cue"] > 0.0
            or features["work_context"] > 0.0
        )
    ):
        return ("org_project", "durable")
    if (
        features["span_is_stopword"] == 0.0
        and features["span_char_count"] >= 3.0
        and features["wordnet_state"] > 0.0
        and (
            features["before_has_state_cue"] > 0.0
            or features["state_cue"] > 0.0
            or emotional_supervision > 0.0
        )
    ):
        return ("state", "provisional")
    if (
        features["span_is_stopword"] == 0.0
        and features["span_char_count"] >= 3.0
        and (
            features["wordnet_topic"] > 0.0
            or features["work_cue"] > 0.0
            or task_supervision > 0.0
        )
    ):
        return ("topic", "provisional")
    return ("none", "ignore")


def iter_cortext_messages(paths: Iterable[Path]) -> Iterable[tuple[str, str]]:
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, list) or len(payload) != 2:
                    continue
                conversation_id = str(payload[0])
                content = payload[1].get("content", [])
                if not isinstance(content, list):
                    continue
                for turn in content:
                    message = normalize_text(turn.get("message", ""))
                    if message:
                        yield conversation_id, message


def feature_name_list(base_dim: int) -> list[str]:
    names = [f"span_hash_{idx}" for idx in range(base_dim)]
    names.extend(f"sentence_hash_{idx}" for idx in range(base_dim))
    numeric_names = [
        "span_token_count",
        "span_char_count",
        "span_is_title",
        "span_all_title",
        "span_title_ratio",
        "span_has_digit",
        "span_has_dot",
        "self_reference",
        "name_intro",
        "state_cue",
        "work_cue",
        "place_cue",
        "org_cue",
        "sentence_title_density",
        "before_has_name_intro",
        "before_has_self_name_copula",
        "before_has_place_cue",
        "before_has_org_cue",
        "before_has_state_cue",
        "after_has_work_cue",
        "span_is_name_helper",
        "span_is_place_helper",
        "span_is_state_helper",
        "span_is_work_helper",
        "span_is_helper_only",
        "work_context",
        "wordnet_person",
        "wordnet_place",
        "wordnet_state",
        "wordnet_topic",
        "name_given",
        "name_surname",
        "name_full_name",
        "name_given_count",
        "name_surname_count",
        "name_token_ratio",
        "span_is_stopword",
    ]
    names.extend(numeric_names)
    return names


def featurize_example(
    sentence: str,
    span_text: str,
    wordnet_index: dict[str, dict[str, float]],
    priors: dict[str, dict[str, float]],
    embedder: ExternalTextEmbedder | None,
    hash_dim: int = 128,
) -> np.ndarray:
    candidate = None
    for span in generate_candidate_spans(sentence):
        if normalize_key(span.text) == normalize_key(span_text):
            candidate = span
            break
    if candidate is None:
        span_start = sentence.lower().find(span_text.lower())
        if span_start < 0:
            span_start = 0
        candidate = CandidateSpan(
            text=span_text,
            start=span_start,
            end=span_start + len(span_text),
            token_start=0,
            token_end=max(1, len(span_text.split())),
        )
    numeric = lexical_features(sentence, candidate, wordnet_index, priors)
    span_vec = hashed_text_vector(span_text, dim=hash_dim)
    sentence_vec = hashed_text_vector(sentence, dim=hash_dim)
    if embedder is not None:
        embeddings = embedder.encode_many([span_text, sentence])
        if span_text in embeddings:
            span_vec = embeddings[span_text]
        if sentence in embeddings:
            sentence_vec = embeddings[sentence]
    ordered_numeric = np.asarray(
        [
            numeric["span_token_count"],
            numeric["span_char_count"],
            numeric["span_is_title"],
            numeric["span_all_title"],
            numeric["span_title_ratio"],
            numeric["span_has_digit"],
            numeric["span_has_dot"],
            numeric["self_reference"],
            numeric["name_intro"],
            numeric["state_cue"],
            numeric["work_cue"],
            numeric["place_cue"],
            numeric["org_cue"],
            numeric["sentence_title_density"],
            numeric["before_has_name_intro"],
            numeric["before_has_self_name_copula"],
            numeric["before_has_place_cue"],
            numeric["before_has_org_cue"],
            numeric["before_has_state_cue"],
            numeric["after_has_work_cue"],
            numeric["span_is_name_helper"],
            numeric["span_is_place_helper"],
            numeric["span_is_state_helper"],
            numeric["span_is_work_helper"],
            numeric["span_is_helper_only"],
            numeric["work_context"],
            numeric["wordnet_person"],
            numeric["wordnet_place"],
            numeric["wordnet_state"],
            numeric["wordnet_topic"],
            numeric["name_given"],
            numeric["name_surname"],
            numeric["name_full_name"],
            numeric["name_given_count"],
            numeric["name_surname_count"],
            numeric["name_token_ratio"],
            numeric["span_is_stopword"],
        ],
        dtype=np.float32,
    )
    return np.concatenate([span_vec.astype(np.float32), sentence_vec.astype(np.float32), ordered_numeric])


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def train_softmax_regression(
    X: np.ndarray,
    labels: list[str],
    epochs: int = 200,
    learning_rate: float = 0.25,
    l2: float = 1e-4,
    class_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    classes = sorted(set(labels))
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    y = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    sample_weights = np.asarray(
        [float((class_weights or {}).get(label, 1.0)) for label in labels],
        dtype=np.float32,
    )
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std
    rng = np.random.default_rng(7)
    W = rng.normal(scale=0.01, size=(Xn.shape[1], len(classes))).astype(np.float32)
    b = np.zeros(len(classes), dtype=np.float32)
    eye = np.eye(len(classes), dtype=np.float32)
    targets = eye[y]
    for _ in range(epochs):
        logits = Xn @ W + b
        probs = softmax(logits)
        weight_scale = sample_weights[:, np.newaxis] / float(np.sum(sample_weights) or 1.0)
        diff = (probs - targets) * weight_scale
        grad_W = Xn.T @ diff + l2 * W
        grad_b = np.sum(diff, axis=0)
        W -= learning_rate * grad_W.astype(np.float32)
        b -= learning_rate * grad_b.astype(np.float32)
    return W, b, classes, mean.astype(np.float32), std.astype(np.float32)


def predict_softmax(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    classes: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    Xn = (X - mean) / std
    probs = softmax(Xn @ W + b)
    preds = [classes[idx] for idx in np.argmax(probs, axis=1)]
    return preds, probs


def calibrate_type_probabilities(
    sentence: str,
    span_text: str,
    type_probs: dict[str, float],
    wordnet_index: dict[str, dict[str, float]],
    priors: dict[str, dict[str, float]],
) -> dict[str, float]:
    candidate = None
    for span in generate_candidate_spans(sentence):
        if normalize_key(span.text) == normalize_key(span_text):
            candidate = span
            break
    if candidate is None:
        span_start = sentence.lower().find(span_text.lower())
        if span_start < 0:
            span_start = 0
        candidate = CandidateSpan(
            text=span_text,
            start=span_start,
            end=span_start + len(span_text),
            token_start=0,
            token_end=max(1, len(span_text.split())),
        )
    features = lexical_features(sentence, candidate, wordnet_index, priors)
    calibrated = dict(type_probs)

    if (
        (features.get("before_has_name_intro", 0.0) > 0.0 or features.get("before_has_self_name_copula", 0.0) > 0.0)
        and features.get("work_context", 0.0) == 0.0
        and features.get("name_given", 0.0) >= 0.02
        and features.get("name_token_ratio", 0.0) >= 0.5
        and features.get("span_token_count", 0.0) <= 2.0
    ):
        calibrated["identity"] = max(
            calibrated.get("identity", 0.0),
            calibrated.get("person_entity", 0.0) + 0.20,
            0.80,
        )
        calibrated["person_entity"] = min(
            calibrated.get("person_entity", 0.0),
            max(0.0, calibrated["identity"] - 0.05),
        )

    if (
        features.get("work_context", 0.0) > 0.0
        and features.get("span_all_title", 0.0) > 0.0
        and features.get("name_given", 0.0) == 0.0
        and features.get("name_surname", 0.0) == 0.0
    ):
        calibrated["org_project"] = max(
            calibrated.get("org_project", 0.0),
            calibrated.get("identity", 0.0) + 0.15,
            calibrated.get("person_entity", 0.0) + 0.10,
            0.55,
        )

    if (
        features.get("before_has_place_cue", 0.0) > 0.0
        and features.get("wordnet_place", 0.0) > 0.0
    ):
        calibrated["place"] = max(calibrated.get("place", 0.0), 0.60)

    if (
        (features.get("before_has_state_cue", 0.0) > 0.0 or features.get("state_cue", 0.0) > 0.0)
        and features.get("wordnet_state", 0.0) > 0.0
        and features.get("span_is_state_helper", 0.0) == 0.0
    ):
        calibrated["state"] = max(calibrated.get("state", 0.0), 0.55)

    total = sum(max(0.0, value) for value in calibrated.values())
    if total > 0.0:
        calibrated = {label: max(0.0, value) / total for label, value in calibrated.items()}
    return calibrated


def load_examples(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def confusion_matrix(gold: list[str], pred: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for gold_label, pred_label in zip(gold, pred):
        matrix[index[gold_label]][index[pred_label]] += 1
    return matrix


def precision_recall_f1(gold: list[str], pred: list[str], labels: list[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def macro_f1(per_label: dict[str, dict[str, float]]) -> float:
    if not per_label:
        return 0.0
    return sum(metric["f1"] for metric in per_label.values()) / float(len(per_label))
