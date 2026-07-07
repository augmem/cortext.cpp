#!/usr/bin/env python3
"""
Centroid vector management for Cortext VER (Vocal Emotion Recognition).

Commands:
  embed     Generate C++ embedded vectors from .npy centroid files

Examples:
  # Generate C++ embedded vectors from .npy files
  python tools/centroid_vectors/centroid_vectors.py embed
"""

from __future__ import annotations

import argparse
import ast
import struct
import sys
from pathlib import Path
from typing import Dict, List

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
        default=Path(__file__).resolve().parents[2],
        help="Repository root (defaults to script parent/../..)",
    )
    embed_parser.add_argument(
        "--out-cpp",
        type=Path,
        default=None,
        help="Output .cpp path (default: <repo_root>/src/data/embedded_centroid_vectors.cpp)",
    )

    args = parser.parse_args()

    if args.command == "embed":
        return cmd_embed(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
