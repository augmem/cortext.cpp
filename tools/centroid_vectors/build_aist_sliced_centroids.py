#!/usr/bin/env python3
"""Regenerate production 256d centroids from the active AIST/ES-AIST encoder.

The production retrieval path stores `RetrievalEmbeddingView(encoded)`, which is
the first 256 dimensions of the encoder output, L2-normalized. This script uses
the same slice to rebuild the legacy `*_256.npy` files and then the existing
`centroid_vectors.py embed` step can rebake `src/data/embedded_centroid_vectors.cpp`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


TEXT_EMOTIONS = ("anger", "fear", "joy", "love", "sadness", "surprise")
AUDIO_EMOTIONS = ("anger", "fear", "joy", "love", "sadness", "surprise", "neutral")

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "joy",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "08": "surprise",
}

PROMPT_BANKS = {
    "goal_aligned": [
        "we completed the plan successfully",
        "the task went according to the user's goal",
        "the outcome helped the person accomplish what they wanted",
        "the result supports the stated objective",
        "the assistant found the right information and helped",
        "the action made progress toward the goal",
        "the project milestone was reached",
        "the person got what they needed",
    ],
    "goal_unaligned": [
        "the action blocked the user's goal",
        "the result was not what the person wanted",
        "the task failed and moved away from the objective",
        "the assistant gave the wrong information",
        "the plan broke down and wasted time",
        "the outcome frustrated the goal",
        "the work missed the intended target",
        "the person did not get what they needed",
    ],
    "violation_high": [
        "someone violated the rule",
        "a promise was broken",
        "the behavior was unfair and crossed a boundary",
        "private information was exposed without permission",
        "the action harmed trust",
        "the person ignored consent",
        "a safety policy was breached",
        "the decision was dishonest and wrong",
    ],
    "violation_low": [
        "no rule was broken",
        "the behavior was fair and respectful",
        "the action followed the agreement",
        "privacy and consent were preserved",
        "the interaction was calm and appropriate",
        "the decision respected the boundary",
        "the process stayed within policy",
        "the person acted honestly",
    ],
    "arousal_low": [
        "the person felt calm and settled",
        "the conversation was quiet and relaxed",
        "the situation was peaceful",
        "the mood was gentle and unhurried",
        "the moment felt still and safe",
        "the person spoke softly and calmly",
        "nothing urgent was happening",
        "the energy level was low and steady",
    ],
}


def timestamp_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_meta(path: Path) -> dict:
    h = hashlib.sha256()
    count = 0
    total = 0
    if path.exists():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = child.relative_to(path).as_posix()
            h.update(rel.encode("utf-8", errors="ignore"))
            h.update(b"\0")
            h.update(sha256_file(child).encode("ascii"))
            h.update(b"\0")
            count += 1
            total += child.stat().st_size
    return {
        "path": str(path),
        "exists": path.exists(),
        "files": count,
        "bytes": total,
        "sha256": h.hexdigest() if path.exists() else "",
    }


def model_meta_path(models: Path) -> Path:
    gguf_dir = models / "AIST-87M-GGUF"
    return gguf_dir if gguf_dir.exists() else models


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12 or not math.isfinite(norm):
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms <= 1.0e-12, 1.0, norms)
    return (mat / norms).astype(np.float32)


def retrieval_slice(vec: list[float], dim: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape[0] < dim:
        raise ValueError(f"embedding dim {arr.shape[0]} is smaller than slice {dim}")
    return l2_normalize(arr[:dim])


def load_jsonl_embeddings(path: Path, dim: int, key: str = "embedding") -> list[np.ndarray]:
    out: list[np.ndarray] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            emb = row.get(key)
            if emb:
                out.append(retrieval_slice(emb, dim))
    return out


def run_text_embedder(
    embedder: Path, models: Path, input_path: Path, output_path: Path
) -> None:
    cmd = [
        str(embedder),
        f"--input={input_path}",
        f"--out={output_path}",
        f"--models={models}",
    ]
    subprocess.check_call(cmd)


def run_audio_embedder(
    embedder: Path, models: Path, input_list: Path, output_path: Path
) -> None:
    cmd = [
        str(embedder),
        f"--input-list={input_list}",
        f"--out={output_path}",
        f"--models={models}",
    ]
    subprocess.check_call(cmd)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            clean = line.replace("\n", " ").strip()
            if clean:
                f.write(clean + "\n")


def purify_embeddings(
    embeddings: np.ndarray, keep_top: float, iters: int
) -> tuple[np.ndarray, int]:
    if embeddings.size == 0 or keep_top >= 1.0 or iters <= 0:
        return embeddings, 0
    keep_top = max(0.1, min(keep_top, 1.0))
    kept = embeddings
    removed_total = 0
    for _ in range(iters):
        if kept.shape[0] < 10:
            break
        center = l2_normalize(kept.mean(axis=0).astype(np.float32))
        sims = l2_normalize_rows(kept) @ center
        keep_n = max(10, int(round(float(len(sims)) * keep_top)))
        if keep_n >= len(sims):
            break
        idx = np.argpartition(-sims, keep_n - 1)[:keep_n]
        removed_total += len(sims) - len(idx)
        kept = kept[idx]
    return kept, removed_total


def centroid_from_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        raise ValueError("cannot build centroid from empty embeddings")
    return l2_normalize(embeddings.mean(axis=0).astype(np.float32))


def contrastive_centroid(
    label: str, embeddings_by_label: dict[str, np.ndarray]
) -> np.ndarray:
    own = embeddings_by_label[label]
    others = [
        values
        for other, values in embeddings_by_label.items()
        if other != label and values.size > 0
    ]
    if not others:
        return centroid_from_embeddings(own)
    other_mean = np.concatenate(others, axis=0).mean(axis=0)
    return l2_normalize(own.mean(axis=0).astype(np.float32) - other_mean.astype(np.float32))


def write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values.astype(np.float32))


def load_text_emotion_samples(path: Path, max_per_label: int, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    out: dict[str, list[str]] = {}
    for label in TEXT_EMOTIONS:
        txt = path / f"{label}.txt"
        if not txt.exists():
            continue
        lines = [line.strip() for line in txt.read_text(encoding="utf-8").splitlines()]
        lines = [line for line in lines if line]
        rng.shuffle(lines)
        if max_per_label > 0:
            lines = lines[:max_per_label]
        out[label] = lines
    return out


def embed_text_groups(
    groups: dict[str, list[str]],
    embedder: Path,
    models: Path,
    models_meta: dict,
    cache_dir: Path,
    dim: int,
    cache: bool,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for label, lines in sorted(groups.items()):
        input_path = cache_dir / f"{label}.txt"
        output_path = cache_dir / f"{label}.jsonl"
        meta_path = cache_dir / f"{label}.meta.json"
        text_hash = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        meta = {
            "label": label,
            "input_hash": text_hash,
            "dim": dim,
            "embedder": str(embedder),
            "models": str(models),
            "models_meta": models_meta,
        }
        use_cache = False
        if cache and output_path.exists() and meta_path.exists():
            try:
                use_cache = json.loads(meta_path.read_text(encoding="utf-8")) == meta
            except json.JSONDecodeError:
                use_cache = False
        if not use_cache:
            write_lines(input_path, lines)
            run_text_embedder(embedder, models, input_path, output_path)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        rows = load_jsonl_embeddings(output_path, dim)
        if rows:
            out[label] = np.stack(rows, axis=0)
    return out


def build_text_emotion(
    args: argparse.Namespace, summary: dict
) -> dict[str, np.ndarray]:
    samples = load_text_emotion_samples(
        args.text_emotion_dir, args.max_text_per_label, args.seed
    )
    embeddings = embed_text_groups(
        samples,
        args.text_embedder,
        args.models,
        args.models_meta,
        args.work_dir / "text_emotion_embeddings",
        args.slice_dim,
        args.cache_embeddings,
    )
    if args.center and embeddings:
        global_mean = np.concatenate(list(embeddings.values()), axis=0).mean(axis=0)
    else:
        global_mean = None
    purified: dict[str, np.ndarray] = {}
    labels_meta: dict[str, dict] = {}
    for label, values in sorted(embeddings.items()):
        centered = values - global_mean if global_mean is not None else values
        kept, removed = purify_embeddings(centered, args.purity, args.purity_iters)
        purified[label] = kept
        labels_meta[label] = {
            "file": f"{label}_256.npy",
            "source_count": int(values.shape[0]),
            "centroid_count": int(kept.shape[0]),
            "purity_removed": int(removed),
        }
    for label, kept in sorted(purified.items()):
        centroid = (
            contrastive_centroid(label, purified)
            if args.contrastive
            else centroid_from_embeddings(kept)
        )
        write_npy(args.out_root / "emotion" / f"{label}_256.npy", centroid)
    metadata = {
        "generated_by": "tools/centroid_vectors/build_aist_sliced_centroids.py",
        "timestamp": timestamp_utc(),
        "encoder": "AIST/ES-AIST via cortext_text_embedder",
        "models_meta": args.models_meta,
        "embedding_dim": args.slice_dim,
        "slice": [0, args.slice_dim],
        "source": str(args.text_emotion_dir),
        "source_meta": dir_meta(args.text_emotion_dir),
        "mean_centered": args.center,
        "purity_keep_top": args.purity,
        "purity_iters": args.purity_iters,
        "contrastive": args.contrastive,
        "labels": labels_meta,
    }
    (args.out_root / "emotion" / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["text_emotion"] = metadata
    return purified


def stack_labels(embeddings: dict[str, np.ndarray], labels: tuple[str, ...]) -> np.ndarray:
    arrays = [embeddings[label] for label in labels if label in embeddings]
    if not arrays:
        raise ValueError(f"missing embeddings for {labels}")
    return np.concatenate(arrays, axis=0)


def build_affect(
    args: argparse.Namespace, text_embeddings: dict[str, np.ndarray], summary: dict
) -> None:
    prompt_groups = {
        key: value for key, value in PROMPT_BANKS.items()
    }
    prompt_embeddings = embed_text_groups(
        prompt_groups,
        args.text_embedder,
        args.models,
        args.models_meta,
        args.work_dir / "affect_prompt_embeddings",
        args.slice_dim,
        args.cache_embeddings,
    )

    groups = {
        "val_pos": stack_labels(text_embeddings, ("joy", "love")),
        "val_neg": stack_labels(text_embeddings, ("anger", "fear", "sadness")),
        "aro_high": stack_labels(text_embeddings, ("anger", "fear", "surprise")),
        "aro_low": np.concatenate(
            [
                stack_labels(text_embeddings, ("sadness", "love")),
                prompt_embeddings["arousal_low"],
            ],
            axis=0,
        ),
        "goal_aligned": prompt_embeddings["goal_aligned"],
        "goal_unaligned": prompt_embeddings["goal_unaligned"],
        "violation_high": prompt_embeddings["violation_high"],
        "violation_low": prompt_embeddings["violation_low"],
    }
    positive_negative_pairs = {
        "val_pos": "val_neg",
        "val_neg": "val_pos",
        "aro_high": "aro_low",
        "aro_low": "aro_high",
        "goal_aligned": "goal_unaligned",
        "goal_unaligned": "goal_aligned",
        "violation_high": "violation_low",
        "violation_low": "violation_high",
    }
    labels_meta = {}
    for label, values in groups.items():
        other = groups[positive_negative_pairs[label]]
        centroid = l2_normalize(values.mean(axis=0).astype(np.float32) - other.mean(axis=0).astype(np.float32))
        write_npy(args.out_root / "affect" / f"{label}_256.npy", centroid)
        labels_meta[label] = {
            "file": f"{label}_256.npy",
            "count": int(values.shape[0]),
            "contrast_against": positive_negative_pairs[label],
        }
    metadata = {
        "generated_by": "tools/centroid_vectors/build_aist_sliced_centroids.py",
        "timestamp": timestamp_utc(),
        "encoder": "AIST/ES-AIST via cortext_text_embedder",
        "models_meta": args.models_meta,
        "embedding_dim": args.slice_dim,
        "slice": [0, args.slice_dim],
        "text_emotion_source": str(args.text_emotion_dir),
        "labels": labels_meta,
    }
    (args.out_root / "affect" / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["affect"] = metadata


def parse_ravdess_label(path: Path) -> str | None:
    parts = path.stem.split("-")
    if len(parts) < 3:
        return None
    return RAVDESS_EMOTION_MAP.get(parts[2])


def build_audio_emotion(args: argparse.Namespace, summary: dict) -> None:
    wavs = sorted(args.ravdess_dir.glob("**/*.wav"))
    source_meta = dir_meta(args.ravdess_dir)
    grouped_paths: dict[str, list[Path]] = {}
    for path in wavs:
        label = parse_ravdess_label(path)
        if label is None:
            continue
        grouped_paths.setdefault(label, []).append(path)
    all_paths = [path for paths in grouped_paths.values() for path in paths]
    input_list = args.work_dir / "ravdess_audio_paths.txt"
    write_lines(input_list, [str(path) for path in all_paths])
    output_path = args.work_dir / "ravdess_audio_embeddings.jsonl"
    meta_path = args.work_dir / "ravdess_audio_embeddings.meta.json"
    cache_meta = {
        "embedder": str(args.audio_embedder),
        "models": str(args.models),
        "models_meta": args.models_meta,
        "source_meta": source_meta,
        "slice_dim": args.slice_dim,
    }
    use_cache = False
    if args.cache_embeddings and output_path.exists() and meta_path.exists():
        try:
            use_cache = json.loads(meta_path.read_text(encoding="utf-8")) == cache_meta
        except json.JSONDecodeError:
            use_cache = False
    if not use_cache:
        run_audio_embedder(args.audio_embedder, args.models, input_list, output_path)
        meta_path.write_text(
            json.dumps(cache_meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    by_path: dict[str, np.ndarray] = {}
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_path[row["path"]] = retrieval_slice(row["embedding"], args.slice_dim)

    embeddings: dict[str, np.ndarray] = {}
    for label, paths in grouped_paths.items():
        rows = [by_path[str(path)] for path in paths if str(path) in by_path]
        if rows:
            embeddings[label] = np.stack(rows, axis=0)
    centroids: dict[str, np.ndarray] = {}
    if "neutral" in embeddings or "calm" in embeddings:
        neutral_arrays = []
        if "neutral" in embeddings:
            neutral_arrays.append(embeddings["neutral"])
        if "calm" in embeddings:
            neutral_arrays.append(embeddings["calm"])
        centroids["neutral"] = centroid_from_embeddings(np.concatenate(neutral_arrays, axis=0))
    for label in ("joy", "sadness", "anger", "fear", "surprise"):
        if label in embeddings:
            centroids[label] = centroid_from_embeddings(embeddings[label])
    if "joy" in centroids and "surprise" in centroids and "calm" in embeddings:
        calm = centroid_from_embeddings(embeddings["calm"])
        centroids["love"] = l2_normalize(
            (centroids["joy"] + calm + centroids["surprise"]) / 3.0
        )
    labels_meta = {}
    for label in AUDIO_EMOTIONS:
        if label not in centroids:
            continue
        write_npy(args.out_root / "audio_emotion" / f"{label}_256.npy", centroids[label])
        labels_meta[label] = {
            "file": f"{label}_256.npy",
            "source_count": int(embeddings.get(label, np.empty((0, args.slice_dim))).shape[0]),
        }
    metadata = {
        "generated_by": "tools/centroid_vectors/build_aist_sliced_centroids.py",
        "timestamp": timestamp_utc(),
        "encoder": "AIST/ES-AIST via cortext_aist_audio_embedder",
        "embedding_dim": args.slice_dim,
        "slice": [0, args.slice_dim],
        "dataset": "RAVDESS",
        "source": str(args.ravdess_dir),
        "source_meta": source_meta,
        "models_meta": args.models_meta,
        "labels": labels_meta,
    }
    (args.out_root / "audio_emotion" / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["audio_emotion"] = metadata


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument(
        "--text-embedder",
        type=Path,
        default=Path("build/tools/text_embedder/cortext_text_embedder"),
    )
    parser.add_argument(
        "--audio-embedder",
        type=Path,
        default=Path("build/tools/aist_audio_embedder/cortext_aist_audio_embedder"),
    )
    parser.add_argument("--out-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--work-dir", type=Path, default=Path("build/aist_sliced_centroids")
    )
    parser.add_argument(
        "--text-emotion-dir",
        type=Path,
        default=Path("data/emotion/_tmp_text_emotion"),
    )
    parser.add_argument("--ravdess-dir", type=Path, default=Path("data/ravdess"))
    parser.add_argument("--slice-dim", type=int, default=256)
    parser.add_argument("--max-text-per-label", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--purity", type=float, default=0.35)
    parser.add_argument("--purity-iters", type=int, default=3)
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contrastive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-affect", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.text_embedder.exists():
        raise FileNotFoundError(f"missing text embedder: {args.text_embedder}")
    if not args.skip_audio and not args.audio_embedder.exists():
        raise FileNotFoundError(f"missing audio embedder: {args.audio_embedder}")
    args.models_meta = dir_meta(model_meta_path(args.models))
    args.work_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_by": "tools/centroid_vectors/build_aist_sliced_centroids.py",
        "timestamp": timestamp_utc(),
        "models": str(args.models),
        "models_meta": args.models_meta,
        "slice": [0, args.slice_dim],
    }
    text_embeddings = build_text_emotion(args, summary)
    if not args.skip_affect:
        build_affect(args, text_embeddings, summary)
    if not args.skip_audio:
        build_audio_emotion(args, summary)

    summary_path = args.work_dir / "aist_sliced_centroid_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote AIST sliced centroids; summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
