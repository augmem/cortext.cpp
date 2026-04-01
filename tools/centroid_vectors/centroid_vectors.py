#!/usr/bin/env python3
"""
Centroid vector management for Cortext VER (Vocal Emotion Recognition).

Commands:
  generate  Generate audio emotion centroids from RAVDESS dataset
  embed     Generate C++ embedded vectors from .npy centroid files
  validate  Validate centroids against held-out RAVDESS files

Examples:
  # Generate audio emotion centroids from RAVDESS
  python tools/centroid_vectors/centroid_vectors.py generate --ravdess-dir data/ravdess --models-dir models/imagebind

  # Generate C++ embedded vectors from .npy files
  python tools/centroid_vectors/centroid_vectors.py embed

  # Validate centroids against held-out RAVDESS files
  python tools/centroid_vectors/centroid_vectors.py validate --ravdess-dir data/ravdess --models-dir models/imagebind
"""

from __future__ import annotations

import argparse
import ast
import json
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# NPY I/O (numpy-free)
# =============================================================================


def read_npy_f32_or_f64_1d(path: Path) -> List[float]:
    """Read a 1D float32/float64 .npy file without numpy."""
    data = path.read_bytes()
    if len(data) < 10:
        raise ValueError(f"{path}: file too small for .npy")
    if data[0:6] != b"\x93NUMPY":
        raise ValueError(f"{path}: invalid .npy magic")
    major = data[6]
    minor = data[7]
    if (major, minor) == (1, 0):
        header_len = struct.unpack_from("<H", data, 8)[0]
        header_start = 10
    elif (major, minor) == (2, 0):
        header_len = struct.unpack_from("<I", data, 8)[0]
        header_start = 12
    else:
        raise ValueError(f"{path}: unsupported .npy version {major}.{minor}")
    header_end = header_start + header_len
    if header_end > len(data):
        raise ValueError(f"{path}: truncated .npy header")
    header = data[header_start:header_end].decode("ascii", errors="strict").strip()
    header_obj = ast.literal_eval(header)
    if not isinstance(header_obj, dict):
        raise ValueError(f"{path}: malformed .npy header (not a dict)")
    descr = header_obj.get("descr")
    if descr not in ("<f4", "|f4", "<f8", "|f8"):
        raise ValueError(
            f"{path}: expected float32/float64 descr '<f4'/'<f8', got {descr!r}"
        )
    shape = header_obj.get("shape")
    if not (isinstance(shape, tuple) and len(shape) == 1 and isinstance(shape[0], int)):
        raise ValueError(f"{path}: expected 1D shape, got {shape!r}")
    count = shape[0]
    if count != 256:
        raise ValueError(f"{path}: expected 256-dim centroid, got {count}")
    fortran_order = header_obj.get("fortran_order")
    if fortran_order not in (False, 0):
        raise ValueError(f"{path}: expected fortran_order False, got {fortran_order!r}")
    payload = data[header_end:]
    element_size = 4 if descr.endswith("f4") else 8
    expected_bytes = count * element_size
    if len(payload) < expected_bytes:
        raise ValueError(f"{path}: truncated data payload")
    if element_size == 4:
        values_f = struct.unpack_from("<256f", payload, 0)
        return list(values_f)
    values_d = struct.unpack_from("<256d", payload, 0)
    return [float(v) for v in values_d]


def write_npy_f32_1d(path: Path, values: List[float]) -> None:
    """Write a 1D float32 .npy file without numpy."""
    count = len(values)
    header_dict = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({count},)}}"
    # Pad header to be aligned to 64 bytes
    header_bytes = header_dict.encode("ascii")
    # Header = magic(6) + version(2) + header_len(2) + header + newline
    # Total must be divisible by 64 for alignment
    prefix_len = 6 + 2 + 2  # magic + version + header_len
    total = prefix_len + len(header_bytes) + 1  # +1 for newline
    padding_needed = (64 - (total % 64)) % 64
    header_bytes = header_bytes + b" " * padding_needed + b"\n"
    header_len = len(header_bytes)

    with open(path, "wb") as f:
        f.write(b"\x93NUMPY")  # magic
        f.write(struct.pack("<BB", 1, 0))  # version 1.0
        f.write(struct.pack("<H", header_len))  # header length
        f.write(header_bytes)
        f.write(struct.pack(f"<{count}f", *values))


# =============================================================================
# EMBED command: Generate C++ from .npy files
# =============================================================================


def _format_floats(values: List[float], per_line: int = 8) -> str:
    chunks: List[str] = []
    for i in range(0, len(values), per_line):
        row = values[i : i + per_line]
        chunks.append("  " + ", ".join(f"{v:.9g}f" for v in row) + ",")
    return "\n".join(chunks)


def _load_all_centroids(repo_root: Path) -> Dict[str, List[float]]:
    """Load all centroid .npy files."""
    affect_dir = repo_root / "data" / "affect"
    emotion_dir = repo_root / "data" / "emotion"
    audio_emotion_dir = repo_root / "data" / "audio_emotion"

    mapping: Dict[str, Path] = {
        # Affect
        "valence_positive": affect_dir / "val_pos_256.npy",
        "valence_negative": affect_dir / "val_neg_256.npy",
        "arousal_high": affect_dir / "aro_high_256.npy",
        "arousal_low": affect_dir / "aro_low_256.npy",
        "goal_aligned": affect_dir / "goal_aligned_256.npy",
        "goal_unaligned": affect_dir / "goal_unaligned_256.npy",
        "violation_high": affect_dir / "violation_high_256.npy",
        "violation_low": affect_dir / "violation_low_256.npy",
        # Emotion (Algorithm 4 order is fixed)
        "anger": emotion_dir / "anger_256.npy",
        "fear": emotion_dir / "fear_256.npy",
        "joy": emotion_dir / "joy_256.npy",
        "love": emotion_dir / "love_256.npy",
        "sadness": emotion_dir / "sadness_256.npy",
        "surprise": emotion_dir / "surprise_256.npy",
    }

    # Audio emotion centroids (optional, for VER)
    audio_mapping: Dict[str, Path] = {
        "audio_anger": audio_emotion_dir / "anger_256.npy",
        "audio_fear": audio_emotion_dir / "fear_256.npy",
        "audio_joy": audio_emotion_dir / "joy_256.npy",
        "audio_love": audio_emotion_dir / "love_256.npy",
        "audio_sadness": audio_emotion_dir / "sadness_256.npy",
        "audio_surprise": audio_emotion_dir / "surprise_256.npy",
        "audio_neutral": audio_emotion_dir / "neutral_256.npy",
    }

    out: Dict[str, List[float]] = {}

    # Load required centroids
    for key, p in mapping.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing centroid asset: {p}")
        out[key] = read_npy_f32_or_f64_1d(p)

    # Load optional audio centroids
    has_audio = all(p.exists() for p in audio_mapping.values())
    if has_audio:
        for key, p in audio_mapping.items():
            out[key] = read_npy_f32_or_f64_1d(p)
        print(f"Loaded audio emotion centroids from {audio_emotion_dir}")
    else:
        print(f"No audio emotion centroids found at {audio_emotion_dir} (optional)")

    return out


def _render_cpp(values: Dict[str, List[float]]) -> str:
    """Render C++ source with embedded centroids."""
    has_audio = "audio_anger" in values

    header = """/// @file
/// @brief Baked-in centroid vectors generated from .npy assets.
///
/// This file is generated by tools/centroid_vectors/centroid_vectors.py embed.
/// Do not edit manually.

#include "cortext/data/centroids.hpp"
#include <Eigen/Dense>
#include <array>
#include <vector>

namespace cortext::data
{
namespace
{
constexpr int kEmbeddingDim = 256;

template <std::size_t N>
Eigen::VectorXf
CopyToEigen (const std::array<float, N> &a)
{
  static_assert (N == kEmbeddingDim);
  Eigen::VectorXf v (kEmbeddingDim);
  for (int i = 0; i < kEmbeddingDim; ++i)
    {
      v[i] = a[static_cast<std::size_t> (i)];
    }
  return v;
}

} // namespace

"""

    def arr(name: str, key: str) -> str:
        return (
            f"alignas(16) static const std::array<float, kEmbeddingDim> {name} = {{\n"
            + _format_floats(values[key])
            + "\n};\n\n"
        )

    body = ""
    # Affect centroids
    body += arr("kValencePositive", "valence_positive")
    body += arr("kValenceNegative", "valence_negative")
    body += arr("kArousalHigh", "arousal_high")
    body += arr("kArousalLow", "arousal_low")
    body += arr("kGoalAligned", "goal_aligned")
    body += arr("kGoalUnaligned", "goal_unaligned")
    body += arr("kViolationHigh", "violation_high")
    body += arr("kViolationLow", "violation_low")

    # Text emotion centroids
    body += arr("kAnger", "anger")
    body += arr("kFear", "fear")
    body += arr("kJoy", "joy")
    body += arr("kLove", "love")
    body += arr("kSadness", "sadness")
    body += arr("kSurprise", "surprise")

    # Audio emotion centroids (if available)
    if has_audio:
        body += arr("kAudioAnger", "audio_anger")
        body += arr("kAudioFear", "audio_fear")
        body += arr("kAudioJoy", "audio_joy")
        body += arr("kAudioLove", "audio_love")
        body += arr("kAudioSadness", "audio_sadness")
        body += arr("kAudioSurprise", "audio_surprise")
        body += arr("kAudioNeutral", "audio_neutral")

    footer = """Centroids
GetEmbeddedCentroids ()
{
  Centroids c;
  c.affect.valence_positive = CopyToEigen (kValencePositive);
  c.affect.valence_negative = CopyToEigen (kValenceNegative);
  c.affect.arousal_high = CopyToEigen (kArousalHigh);
  c.affect.arousal_low = CopyToEigen (kArousalLow);
  c.affect.goal_aligned = CopyToEigen (kGoalAligned);
  c.affect.goal_unaligned = CopyToEigen (kGoalUnaligned);
  c.affect.violation_high = CopyToEigen (kViolationHigh);
  c.affect.violation_low = CopyToEigen (kViolationLow);
  c.emotion_centroids.reserve (6);
  c.emotion_centroids.push_back (CopyToEigen (kAnger));
  c.emotion_centroids.push_back (CopyToEigen (kFear));
  c.emotion_centroids.push_back (CopyToEigen (kJoy));
  c.emotion_centroids.push_back (CopyToEigen (kLove));
  c.emotion_centroids.push_back (CopyToEigen (kSadness));
  c.emotion_centroids.push_back (CopyToEigen (kSurprise));
"""

    if has_audio:
        footer += """  c.audio_emotion_centroids.reserve (7);
  c.audio_emotion_centroids.push_back (CopyToEigen (kAudioAnger));
  c.audio_emotion_centroids.push_back (CopyToEigen (kAudioFear));
  c.audio_emotion_centroids.push_back (CopyToEigen (kAudioJoy));
  c.audio_emotion_centroids.push_back (CopyToEigen (kAudioLove));
  c.audio_emotion_centroids.push_back (CopyToEigen (kAudioSadness));
  c.audio_emotion_centroids.push_back (CopyToEigen (kAudioSurprise));
  c.audio_emotion_centroids.push_back (CopyToEigen (kAudioNeutral));
"""

    footer += """  return c;
}

} // namespace cortext::data
"""
    return header + body + footer


def cmd_embed(args: argparse.Namespace) -> int:
    """Generate C++ embedded vectors from .npy files."""
    repo_root: Path = args.repo_root.resolve()
    out_cpp: Path = (
        args.out_cpp.resolve()
        if args.out_cpp is not None
        else (repo_root / "src" / "data" / "embedded_centroid_vectors.cpp")
    )
    values = _load_all_centroids(repo_root)
    cpp = _render_cpp(values)
    out_cpp.parent.mkdir(parents=True, exist_ok=True)
    out_cpp.write_text(cpp, encoding="utf-8")
    print(f"Wrote {out_cpp}")
    return 0


# =============================================================================
# GENERATE command: Generate audio emotion centroids from RAVDESS
# =============================================================================

# RAVDESS emotion code to Cortext mapping
RAVDESS_EMOTION_MAP = {
    "01": "neutral",  # Neutral
    "02": "calm",     # Calm (also used for neutral, kept separate for love synthesis)
    "03": "joy",      # Happy -> Joy
    "04": "sadness",  # Sad -> Sadness
    "05": "anger",    # Angry -> Anger
    "06": "fear",     # Fearful -> Fear
    # "07": skip      # Disgust - not in our 6-dim map
    "08": "surprise", # Surprised -> Surprise
}

CORTEXT_EMOTIONS = ["anger", "fear", "joy", "love", "sadness", "surprise", "neutral"]


def parse_ravdess_filename(filename: str) -> Optional[str]:
    """Extract emotion code from RAVDESS filename.

    Format: MM-VV-EE-II-SS-RR-AA.wav
    EE = emotion (3rd field)
    """
    parts = filename.replace(".wav", "").split("-")
    if len(parts) >= 3:
        return parts[2]  # emotion code
    return None


def l2_normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector."""
    import math
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-9:
        return vec
    return [v / norm for v in vec]


def compute_mean(embeddings: List[List[float]]) -> List[float]:
    """Compute mean of embeddings."""
    if not embeddings:
        raise ValueError("No embeddings to average")
    dim = len(embeddings[0])
    mean = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            mean[i] += v
    n = len(embeddings)
    return [v / n for v in mean]


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate audio emotion centroids from RAVDESS dataset."""
    try:
        import librosa
    except ImportError:
        print("Error: librosa is required for audio processing.")
        print("Install with: pip install librosa")
        return 1

    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime is required for embedding generation.")
        print("Install with: pip install onnxruntime")
        return 1

    ravdess_dir: Path = args.ravdess_dir.resolve()
    models_dir: Path = args.models_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    sample_rate = 16000

    # Find audio files
    audio_dir = ravdess_dir / "Audio_Speech_Actors_01-24"
    if not audio_dir.exists():
        audio_dir = ravdess_dir  # Try root directly

    wav_files = list(audio_dir.glob("**/*.wav"))
    if not wav_files:
        print(f"Error: No .wav files found in {ravdess_dir}")
        return 1
    print(f"Found {len(wav_files)} audio files")

    # Load ONNX model
    model_path = models_dir / "audio_encoder.onnx"
    if not model_path.exists():
        model_path = models_dir / "audio_encoder_int8.onnx"
    if not model_path.exists():
        print(f"Error: No audio encoder found at {models_dir}")
        print("Expected: audio_encoder.onnx or audio_encoder_int8.onnx")
        return 1

    print(f"Loading model from {model_path}")
    session = ort.InferenceSession(str(model_path))

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Model input: {input_name}, output: {output_name}")

    # Collect embeddings per emotion
    emotion_embeddings: Dict[str, List[List[float]]] = {
        e: [] for e in ["neutral", "calm", "joy", "sadness", "anger", "fear", "surprise"]
    }

    # Process files
    processed = 0
    skipped = 0
    for wav_path in wav_files:
        emotion_code = parse_ravdess_filename(wav_path.name)
        if emotion_code is None or emotion_code == "07":  # Skip disgust
            skipped += 1
            continue

        cortext_emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
        if cortext_emotion is None:
            skipped += 1
            continue

        try:
            # Load audio at 16kHz mono
            y, sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)

            # Compute mel spectrogram (ImageBind preprocessing)
            # Using librosa's mel spectrogram with ImageBind parameters
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sample_rate,
                n_fft=512,
                hop_length=160,
                win_length=400,
                n_mels=128,
                fmin=0,
                fmax=sample_rate // 2,
            )

            # Convert to log scale
            import numpy as np
            mel_spec = np.log(mel_spec + 1e-10)

            # Normalize with ImageBind stats
            mel_spec = (mel_spec - (-4.268)) / 9.138

            # Pad/clip to 204 time frames (2 seconds at 16kHz with hop_length=160)
            target_frames = 204
            if mel_spec.shape[1] < target_frames:
                pad_width = target_frames - mel_spec.shape[1]
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_spec = mel_spec[:, :target_frames]

            # Reshape for model: [1, 1, 128, 204] or [1, 128, 204]
            mel_spec = mel_spec.astype(np.float32)

            # Try different input shapes
            embedding = None
            for shape in [(1, 1, 128, 204), (1, 128, 204)]:
                try:
                    input_data = mel_spec.reshape(shape)
                    outputs = session.run([output_name], {input_name: input_data})
                    embedding = outputs[0].flatten().tolist()
                    break
                except Exception:
                    continue

            if embedding is None:
                print(f"  Warning: Could not process {wav_path.name}")
                skipped += 1
                continue

            # Truncate/pad to 256 dimensions
            if len(embedding) > 256:
                embedding = embedding[:256]
            elif len(embedding) < 256:
                embedding = embedding + [0.0] * (256 - len(embedding))

            # L2 normalize
            embedding = l2_normalize(embedding)

            emotion_embeddings[cortext_emotion].append(embedding)
            processed += 1

            if processed % 100 == 0:
                print(f"  Processed {processed} files...")

        except Exception as e:
            print(f"  Error processing {wav_path.name}: {e}")
            skipped += 1

    print(f"\nProcessed {processed} files, skipped {skipped}")

    # Compute centroids
    out_dir.mkdir(parents=True, exist_ok=True)
    centroids: Dict[str, List[float]] = {}

    # Merge calm into neutral for final neutral centroid
    neutral_combined = emotion_embeddings["neutral"] + emotion_embeddings["calm"]
    if neutral_combined:
        neutral_mean = compute_mean(neutral_combined)
        centroids["neutral"] = l2_normalize(neutral_mean)
        print(f"  neutral: {len(neutral_combined)} samples")

    for emotion in ["joy", "sadness", "anger", "fear", "surprise"]:
        if emotion_embeddings[emotion]:
            mean = compute_mean(emotion_embeddings[emotion])
            centroids[emotion] = l2_normalize(mean)
            print(f"  {emotion}: {len(emotion_embeddings[emotion])} samples")

    # Synthesize Love centroid
    # V_love = normalize((V_joy + V_calm_raw + V_surprise) / 3)
    if emotion_embeddings["calm"] and "joy" in centroids and "surprise" in centroids:
        calm_centroid = l2_normalize(compute_mean(emotion_embeddings["calm"]))
        v_love = [
            (centroids["joy"][i] + calm_centroid[i] + centroids["surprise"][i]) / 3.0
            for i in range(256)
        ]
        centroids["love"] = l2_normalize(v_love)
        print(f"  love: synthesized from joy + calm + surprise")

    # Save .npy files
    for emotion, centroid in centroids.items():
        out_path = out_dir / f"{emotion}_256.npy"
        write_npy_f32_1d(out_path, centroid)
        print(f"Saved {out_path}")

    # Save metadata
    metadata = {
        "generated_by": "tools/centroid_vectors/centroid_vectors.py generate",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_dim": 256,
        "dataset": "RAVDESS",
        "emotions": list(centroids.keys()),
        "samples_per_emotion": {
            e: len(emotion_embeddings.get(e, []))
            for e in ["neutral", "calm", "joy", "sadness", "anger", "fear", "surprise"]
        },
    }
    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {metadata_path}")

    return 0


# =============================================================================
# VALIDATE command: Validate centroids against held-out RAVDESS files
# =============================================================================


def softmax_with_temperature(logits: List[float], beta: float) -> List[float]:
    """Apply softmax with temperature scaling."""
    import math
    max_logit = max(logits)
    exp_values = [math.exp(beta * (v - max_logit)) for v in logits]
    denom = sum(exp_values)
    if denom <= 0:
        return [1.0 / len(logits)] * len(logits)
    return [v / denom for v in exp_values]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate centroids against held-out RAVDESS files."""
    try:
        import librosa
    except ImportError:
        print("Error: librosa is required for audio processing.")
        print("Install with: pip install librosa")
        return 1

    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime is required for embedding generation.")
        print("Install with: pip install onnxruntime")
        return 1

    import random

    ravdess_dir: Path = args.ravdess_dir.resolve()
    models_dir: Path = args.models_dir.resolve()
    centroids_dir: Path = args.centroids_dir.resolve()
    holdout_ratio = args.holdout_ratio
    beta = args.beta
    sample_rate = 16000

    # Load centroids
    centroids: Dict[str, List[float]] = {}
    centroid_order = ["anger", "fear", "joy", "love", "sadness", "surprise", "neutral"]
    for emotion in centroid_order:
        path = centroids_dir / f"{emotion}_256.npy"
        if path.exists():
            centroids[emotion] = read_npy_f32_or_f64_1d(path)

    if not centroids:
        print(f"Error: No centroids found in {centroids_dir}")
        return 1
    print(f"Loaded {len(centroids)} centroids: {list(centroids.keys())}")

    # Find audio files
    audio_dir = ravdess_dir / "Audio_Speech_Actors_01-24"
    if not audio_dir.exists():
        audio_dir = ravdess_dir

    wav_files = list(audio_dir.glob("**/*.wav"))
    if not wav_files:
        print(f"Error: No .wav files found in {ravdess_dir}")
        return 1

    # Filter to only files with emotions we have centroids for
    test_files: List[Tuple[Path, str]] = []
    for wav_path in wav_files:
        emotion_code = parse_ravdess_filename(wav_path.name)
        if emotion_code is None or emotion_code == "07":
            continue
        cortext_emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
        # Map calm to neutral for testing
        if cortext_emotion == "calm":
            cortext_emotion = "neutral"
        if cortext_emotion and cortext_emotion in centroids:
            test_files.append((wav_path, cortext_emotion))

    # Random holdout split
    random.seed(42)
    random.shuffle(test_files)
    holdout_count = int(len(test_files) * holdout_ratio)
    test_set = test_files[:holdout_count]
    print(f"Using {len(test_set)} files for validation ({holdout_ratio*100:.0f}% holdout)")

    # Load ONNX model
    model_path = models_dir / "audio_encoder.onnx"
    if not model_path.exists():
        model_path = models_dir / "audio_encoder_int8.onnx"
    if not model_path.exists():
        print(f"Error: No audio encoder found at {models_dir}")
        return 1

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run validation
    import numpy as np
    correct = 0
    total = 0
    confusion: Dict[str, Dict[str, int]] = {e: {e2: 0 for e2 in centroid_order} for e in centroid_order}

    for wav_path, true_emotion in test_set:
        try:
            y, sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sample_rate, n_fft=512, hop_length=160,
                win_length=400, n_mels=128, fmin=0, fmax=sample_rate // 2,
            )
            mel_spec = np.log(mel_spec + 1e-10)
            mel_spec = (mel_spec - (-4.268)) / 9.138

            target_frames = 204
            if mel_spec.shape[1] < target_frames:
                mel_spec = np.pad(mel_spec, ((0, 0), (0, target_frames - mel_spec.shape[1])))
            else:
                mel_spec = mel_spec[:, :target_frames]

            embedding = None
            for shape in [(1, 1, 128, 204), (1, 128, 204)]:
                try:
                    input_data = mel_spec.astype(np.float32).reshape(shape)
                    outputs = session.run([output_name], {input_name: input_data})
                    embedding = outputs[0].flatten().tolist()
                    break
                except Exception:
                    continue

            if embedding is None:
                continue

            if len(embedding) > 256:
                embedding = embedding[:256]
            elif len(embedding) < 256:
                embedding = embedding + [0.0] * (256 - len(embedding))
            embedding = l2_normalize(embedding)

            # Classify using centroids
            logits = []
            for emotion in centroid_order:
                if emotion in centroids:
                    sim = cosine_similarity(embedding, centroids[emotion])
                    logits.append(max(0.0, sim))  # ReLU
                else:
                    logits.append(0.0)

            probs = softmax_with_temperature(logits, beta)
            pred_idx = probs.index(max(probs))
            pred_emotion = centroid_order[pred_idx]

            if pred_emotion == true_emotion:
                correct += 1
            total += 1

            if true_emotion in confusion and pred_emotion in confusion[true_emotion]:
                confusion[true_emotion][pred_emotion] += 1

        except Exception as e:
            print(f"  Error: {wav_path.name}: {e}")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nResults (beta={beta}):")
    print(f"  Accuracy: {accuracy*100:.1f}% ({correct}/{total})")

    if accuracy >= 0.4:
        print("  PASS: Target accuracy (>=40%) met")
    else:
        print("  FAIL: Below 40% threshold")
        print("  Consider training a lightweight classifier on 256d embeddings")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    header = "        " + "  ".join(f"{e[:3]:>5}" for e in centroid_order)
    print(header)
    for true_e in centroid_order:
        if true_e in confusion:
            row = f"{true_e[:7]:>7} " + "  ".join(
                f"{confusion[true_e].get(pred_e, 0):>5}" for pred_e in centroid_order
            )
            print(row)

    return 0 if accuracy >= 0.4 else 1


# =============================================================================
# Main CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Centroid vector management for Cortext VER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # embed command
    embed_parser = subparsers.add_parser(
        "embed", help="Generate C++ embedded vectors from .npy files"
    )
    embed_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to script parent/..)",
    )
    embed_parser.add_argument(
        "--out-cpp",
        type=Path,
        default=None,
        help="Output .cpp path (default: <repo_root>/src/data/embedded_centroid_vectors.cpp)",
    )

    # generate command
    gen_parser = subparsers.add_parser(
        "generate", help="Generate audio emotion centroids from RAVDESS dataset"
    )
    gen_parser.add_argument(
        "--ravdess-dir",
        type=Path,
        required=True,
        help="Path to RAVDESS dataset directory",
    )
    gen_parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Path to ImageBind ONNX models directory",
    )
    gen_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "audio_emotion",
        help="Output directory for .npy centroids (default: data/audio_emotion)",
    )

    # validate command
    val_parser = subparsers.add_parser(
        "validate", help="Validate centroids against held-out RAVDESS files"
    )
    val_parser.add_argument(
        "--ravdess-dir",
        type=Path,
        required=True,
        help="Path to RAVDESS dataset directory",
    )
    val_parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Path to ImageBind ONNX models directory",
    )
    val_parser.add_argument(
        "--centroids-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "audio_emotion",
        help="Directory containing centroid .npy files (default: data/audio_emotion)",
    )
    val_parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (default: 0.2)",
    )
    val_parser.add_argument(
        "--beta",
        type=float,
        default=8.0,
        help="Softmax temperature (default: 8.0, mid-sensitivity)",
    )

    args = parser.parse_args()

    if args.command == "embed":
        return cmd_embed(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "validate":
        return cmd_validate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
