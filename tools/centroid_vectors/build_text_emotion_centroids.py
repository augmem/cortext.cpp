#!/usr/bin/env python3
import argparse
import json
import random
import subprocess
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
from datasets import load_dataset


EMOTION_TARGETS = ("anger", "fear", "joy", "love", "sadness", "surprise")

EMOTION_MAP = {
    "angry": "anger",
    "annoyed": "anger",
    "furious": "anger",
    "jealous": "anger",
    "disgusted": "anger",
    "afraid": "fear",
    "terrified": "fear",
    "anxious": "fear",
    "apprehensive": "fear",
    "excited": "joy",
    "joyful": "joy",
    "proud": "joy",
    "grateful": "joy",
    "impressed": "joy",
    "hopeful": "joy",
    "confident": "joy",
    "content": "joy",
    "prepared": "joy",
    "caring": "love",
    "faithful": "love",
    "trusting": "love",
    "sentimental": "love",
    "sad": "sadness",
    "lonely": "sadness",
    "guilty": "sadness",
    "ashamed": "sadness",
    "disappointed": "sadness",
    "devastated": "sadness",
    "embarrassed": "sadness",
    "nostalgic": "sadness",
    "surprised": "surprise",
    "anticipating": "surprise",
}

GO_EMOTIONS_MAP = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "love",
    "confusion": None,
    "curiosity": None,
    "desire": "love",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "anger",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "love",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": None,
}


def parse_dataset_spec(spec: str) -> tuple[str, str, float]:
    weight = 1.0
    if "*" in spec:
        spec, weight_str = spec.rsplit("*", 1)
        try:
            weight = float(weight_str)
        except ValueError as exc:
            raise ValueError(f"Invalid dataset weight in '{spec}*{weight_str}'") from exc
    split = "train"
    if "@" in spec:
        spec, split = spec.rsplit("@", 1)
    name = spec.strip()
    if not name:
        raise ValueError("Dataset name is required")
    return name, split, weight


def timestamp_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return vec
    return vec / norm


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    return mat / norms


def load_empathetic_dialogues(
    dataset_name: str, split: str, cache_dir: Path | None
) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = {}
    ds = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    for row in ds:
        context = str(row.get("context", "")).strip()
        utterance = str(row.get("utterance", "")).strip()
        if not context or not utterance:
            continue
        target = EMOTION_MAP.get(context)
        if not target:
            continue
        samples.setdefault(target, []).append(utterance)
    return samples


def load_dair_emotion(
    dataset_name: str, split: str, cache_dir: Path | None
) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = {}
    ds = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    label_names = ds.features.get("label").names
    for row in ds:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        label_id = row.get("label")
        if label_id is None or label_id >= len(label_names):
            continue
        label = label_names[label_id]
        if label not in EMOTION_TARGETS:
            continue
        samples.setdefault(label, []).append(text)
    return samples


def load_go_emotions(
    dataset_name: str, split: str, cache_dir: Path | None
) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = {}
    ds = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    label_names = ds.features.get("labels").feature.names
    for row in ds:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        labels = row.get("labels") or []
        mapped_targets = set()
        for label_id in labels:
            if label_id >= len(label_names):
                continue
            label = label_names[label_id]
            target = GO_EMOTIONS_MAP.get(label)
            if target:
                mapped_targets.add(target)
        for target in mapped_targets:
            samples.setdefault(target, []).append(text)
    return samples


def load_dailydialog(
    dataset_name: str, split: str, cache_dir: Path | None
) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = {}
    ds = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    for row in ds:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        label = row.get("emotion")
        if label is None:
            continue
        # 0: none, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise
        if label == 0:
            continue
        if label in (1, 2):
            target = "anger"
        elif label == 3:
            target = "fear"
        elif label == 4:
            target = "joy"
        elif label == 5:
            target = "sadness"
        elif label == 6:
            target = "surprise"
        else:
            continue
        samples.setdefault(target, []).append(text)
    return samples


DATASET_LOADERS: dict[str, callable] = {
    "empathetic_dialogues": load_empathetic_dialogues,
    "dair-ai/emotion": load_dair_emotion,
    "google-research-datasets/go_emotions": load_go_emotions,
    "agentlans/li2017dailydialog": load_dailydialog,
}


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            clean = line.replace("\n", " ").strip()
            if clean:
                f.write(clean + "\n")


def run_embedder(embedder: Path, models_dir: Path, input_path: Path, out_path: Path) -> None:
    cmd = [
        str(embedder),
        f"--input={input_path}",
        f"--out={out_path}",
        f"--models={models_dir}",
    ]
    subprocess.check_call(cmd)


def load_embeddings(path: Path) -> np.ndarray:
    embeddings = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            emb = row.get("embedding")
            if not emb:
                continue
            embeddings.append(np.asarray(emb, dtype=np.float32))
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(embeddings, axis=0)


def mix_samples(
    dataset_samples: dict[str, dict[str, list[str]]],
    dataset_weights: dict[str, float],
    max_per: int,
    rng: random.Random,
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    mixed: dict[str, list[str]] = {target: [] for target in EMOTION_TARGETS}
    audit: dict[str, dict[str, int]] = {target: {} for target in EMOTION_TARGETS}
    for target in EMOTION_TARGETS:
        pools = []
        total_weight = 0.0
        for dataset_name, samples in dataset_samples.items():
            pool = samples.get(target, [])
            if not pool:
                continue
            pools.append((dataset_name, pool))
            total_weight += dataset_weights[dataset_name]
        if not pools:
            continue
        if max_per <= 0:
            max_per = sum(len(pool) for _, pool in pools)
        allocations: dict[str, int] = {}
        remaining = max_per
        for dataset_name, pool in pools:
            weight = dataset_weights[dataset_name]
            desired = int(round(max_per * (weight / total_weight)))
            take = min(desired, len(pool))
            allocations[dataset_name] = take
            remaining -= take
        while remaining > 0:
            progressed = False
            for dataset_name, pool in sorted(
                pools, key=lambda item: dataset_weights[item[0]], reverse=True
            ):
                if remaining <= 0:
                    break
                if allocations[dataset_name] >= len(pool):
                    continue
                allocations[dataset_name] += 1
                remaining -= 1
                progressed = True
            if not progressed:
                break
        for dataset_name, pool in pools:
            rng.shuffle(pool)
            take = allocations.get(dataset_name, 0)
            picked = pool[:take]
            mixed[target].extend(picked)
            audit[target][dataset_name] = len(picked)
        rng.shuffle(mixed[target])
    return mixed, audit


def purify_embeddings(
    embeddings: np.ndarray, keep_top: float, iters: int
) -> tuple[np.ndarray, int]:
    if embeddings.size == 0 or keep_top >= 1.0 or iters <= 0:
        return embeddings, 0
    keep_top = max(0.1, min(keep_top, 1.0))
    kept = embeddings
    total_removed = 0
    for _ in range(iters):
        if kept.shape[0] < 10:
            break
        mean_vec = kept.mean(axis=0)
        mean_vec = l2_normalize(mean_vec.astype(np.float32))
        sims = l2_normalize_rows(kept.astype(np.float32)) @ mean_vec
        k = max(10, int(round(len(sims) * keep_top)))
        if k >= len(sims):
            break
        idx = np.argpartition(-sims, k - 1)[:k]
        removed = len(sims) - len(idx)
        total_removed += removed
        kept = kept[idx]
    return kept, total_removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate text emotion centroids using ImageBind embeddings."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help=(
            "Dataset spec: name[/config]@split*weight "
            "(ex: dair-ai/emotion@train*1.0). "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--hf-cache",
        default="data/hf_cache",
        help="HF datasets cache directory.",
    )
    parser.add_argument(
        "--models",
        default="models",
        help="Models directory (ImageBind).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/emotion",
        help="Output directory for *_256.npy centroids.",
    )
    parser.add_argument(
        "--max-per",
        type=int,
        default=800,
        help="Max total samples per target emotion after mixing.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Mean-center embeddings across all targets before building centroids.",
    )
    parser.add_argument(
        "--cache-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached embeddings when inputs match.",
    )
    parser.add_argument(
        "--purity",
        type=float,
        default=1.0,
        help="Keep top fraction of samples by cosine similarity to class mean (0.1-1.0).",
    )
    parser.add_argument(
        "--purity-iters",
        type=int,
        default=0,
        help="Number of purification iterations.",
    )
    parser.add_argument(
        "--contrastive",
        action="store_true",
        help="Use contrastive centroids: mean(class) - mean(others).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Shuffle seed.",
    )
    parser.add_argument(
        "--embedder",
        default="build/tools/text_embedder/cortext_text_embedder",
        help="Path to cortext_text_embedder binary.",
    )
    args = parser.parse_args()

    embedder = Path(args.embedder)
    if not embedder.exists():
        print(f"Missing embedder: {embedder}", file=sys.stderr)
        return 1

    models_dir = Path(args.models)
    out_dir = Path(args.out_dir)
    tmp_dir = out_dir / "_tmp_text_emotion"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dataset_specs = args.dataset or [
        "dair-ai/emotion@train*1.0",
        "google-research-datasets/go_emotions@train*0.7",
        "agentlans/li2017dailydialog@train*0.3",
    ]
    cache_dir = Path(args.hf_cache) if args.hf_cache else None
    dataset_samples: dict[str, dict[str, list[str]]] = {}
    dataset_weights: dict[str, float] = {}
    datasets_meta: list[dict[str, str]] = []
    for spec in dataset_specs:
        dataset_name, split, weight = parse_dataset_spec(spec)
        loader = DATASET_LOADERS.get(dataset_name)
        if not loader:
            print(f"Unsupported dataset: {dataset_name}", file=sys.stderr)
            return 1
        print(f"Loading {dataset_name} ({split}) weight={weight}")
        samples = loader(dataset_name, split, cache_dir)
        dataset_samples[dataset_name] = samples
        dataset_weights[dataset_name] = weight
        datasets_meta.append({"name": dataset_name, "split": split, "weight": weight})
    rng = random.Random(args.seed)
    mixed, audit = mix_samples(dataset_samples, dataset_weights, args.max_per, rng)

    meta_labels = {}
    target_embeddings: dict[str, np.ndarray] = {}
    removed_counts: dict[str, int] = {}
    for target, texts in mixed.items():
        if not texts:
            continue
        input_path = tmp_dir / f"{target}.txt"
        out_path = tmp_dir / f"{target}.jsonl"
        cache_meta_path = tmp_dir / f"{target}.meta.json"
        hasher = hashlib.sha256()
        for line in texts:
            hasher.update(line.encode("utf-8", errors="ignore"))
            hasher.update(b"\n")
        input_hash = hasher.hexdigest()
        cache_meta = {
            "target": target,
            "dataset_specs": dataset_specs,
            "seed": args.seed,
            "max_per": args.max_per,
            "input_hash": input_hash,
        }
        cached = False
        if args.cache_embeddings and out_path.exists() and cache_meta_path.exists():
            try:
                cached_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
                if cached_meta == cache_meta:
                    embeddings = load_embeddings(out_path)
                    if embeddings.size != 0:
                        cached = True
            except json.JSONDecodeError:
                cached = False
        if not cached:
            write_lines(input_path, texts)
            run_embedder(embedder, models_dir, input_path, out_path)
            cache_meta_path.write_text(json.dumps(cache_meta, indent=2), encoding="utf-8")
            embeddings = load_embeddings(out_path)
        if embeddings.size == 0:
            print(f"No embeddings for {target}", file=sys.stderr)
            continue
        target_embeddings[target] = embeddings
        removed_counts[target] = 0

    if not target_embeddings:
        print("No embeddings generated for any target.", file=sys.stderr)
        return 1

    if args.center:
        all_embeddings = np.concatenate(list(target_embeddings.values()), axis=0)
        global_mean = all_embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)
    else:
        global_mean = None

    purified_embeddings: dict[str, np.ndarray] = {}
    for target, embeddings in target_embeddings.items():
        if global_mean is not None:
            centered = embeddings - global_mean
        else:
            centered = embeddings
        purified, removed = purify_embeddings(
            centered, args.purity, args.purity_iters
        )
        purified_embeddings[target] = purified
        removed_counts[target] = removed

    for target, embeddings in purified_embeddings.items():
        if args.contrastive:
            other_embeddings = [
                arr for other, arr in purified_embeddings.items() if other != target
            ]
            if other_embeddings:
                other_mean = np.concatenate(other_embeddings, axis=0).mean(axis=0)
                centroid = embeddings.mean(axis=0) - other_mean
            else:
                centroid = embeddings.mean(axis=0)
        else:
            centroid = embeddings.mean(axis=0)
        centroid = l2_normalize(centroid.astype(np.float32))
        out_path_npy = out_dir / f"{target}_256.npy"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path_npy, centroid.astype(np.float32))
        meta_labels[target] = {
            "file": out_path_npy.name,
            "count": int(embeddings.shape[0]),
            "count_after_purity": int(purified.shape[0]),
            "purity_removed": int(removed),
            "sources": [f"{item['name']}:{item['split']}" for item in datasets_meta],
            "mix_audit": audit.get(target, {}),
        }
        print(f"{target}: {embeddings.shape[0]} samples -> {out_path_npy}")

    meta = {
        "generated_by": "cortext/tools/centroid_vectors/build_text_emotion_centroids.py",
        "timestamp": timestamp_utc(),
        "embedding_dim": 256,
        "datasets": datasets_meta,
        "max_per": args.max_per,
        "mean_centered": bool(args.center),
        "purity_keep_top": args.purity,
        "purity_iters": args.purity_iters,
        "contrastive": bool(args.contrastive),
        "mapping": EMOTION_MAP,
        "mapping_go_emotions": GO_EMOTIONS_MAP,
        "labels": meta_labels,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
