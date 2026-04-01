#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTORCH_SRC = REPO_ROOT / "third_party" / "executorch" / "src"
if EXECUTORCH_SRC.exists():
    sys.path.insert(0, str(EXECUTORCH_SRC))
    # Force local ExecuTorch sources/extensions over any site-packages install.
    import types

    execu_pkg = types.ModuleType("executorch")
    execu_pkg.__path__ = [str(EXECUTORCH_SRC / "executorch")]
    sys.modules["executorch"] = execu_pkg

import torch
import torchaudio

from liquid_audio import LFM2AudioProcessor

import executorch.runtime as et_runtime
import ctypes


def _maybe_load_quantized_ops() -> None:
    aot_lib = (
        REPO_ROOT
        / "third_party"
        / "executorch"
        / "cmake-out-pybind"
        / "kernels"
        / "quantized"
        / "libquantized_ops_aot_lib.dylib"
    )
    if aot_lib.exists():
        ctypes.CDLL(str(aot_lib))


def _load_audio(audio_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    wav, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    length = torch.tensor([wav.shape[1]], dtype=torch.int64)
    return wav, length


def _prepare_inputs(
    processor: LFM2AudioProcessor,
    audio_path: Path,
    dtype: torch.dtype,
    fixed_frames: int,
) -> torch.Tensor:
    wav, length = _load_audio(audio_path)
    with torch.no_grad():
        mel, _ = processor.audio(wav, length)
    mel_feat = mel[0]
    if mel_feat.shape[1] < fixed_frames:
        pad = fixed_frames - mel_feat.shape[1]
        mel_feat = torch.nn.functional.pad(mel_feat, (0, pad))
    mel_feat = mel_feat[:, :fixed_frames].to(dtype=dtype).contiguous()
    return mel_feat


def _load_fixed_frames(pte_path: Path) -> int:
    meta_path = pte_path.with_suffix(pte_path.suffix + ".meta.json")
    legacy_meta = pte_path.with_suffix(".meta.json")
    if meta_path.exists():
        data = json.loads(meta_path.read_text())
    elif legacy_meta.exists():
        data = json.loads(legacy_meta.read_text())
    else:
        return 0
    return int(data.get("fixed_frames", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ExecuTorch LFM2-Audio embedder")
    parser.add_argument("--model-dir", default="models/lfm2-audio-1.5b")
    parser.add_argument("--audio-file", default="examples/dog.wav")
    parser.add_argument(
        "--pte",
        default="models/lfm2-audio-1.5b/executorch/lfm2_audio_embedder_8da4w.pte",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fixed-frames", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    model_dir = Path(args.model_dir).resolve()
    audio_path = Path(args.audio_file).resolve()
    pte_path = Path(args.pte).resolve()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    _maybe_load_quantized_ops()

    processor = LFM2AudioProcessor.from_pretrained(model_dir, device="cpu").eval()
    fixed_frames = args.fixed_frames if args.fixed_frames > 0 else _load_fixed_frames(pte_path)
    if fixed_frames <= 0:
        raise RuntimeError("fixed_frames is required (pass --fixed-frames or provide .meta.json)")
    audio_in = _prepare_inputs(processor, audio_path, dtype, fixed_frames)

    runtime = et_runtime.Runtime.get()
    program = runtime.load_program(str(pte_path))
    method_name = "forward" if "forward" in program.method_names else next(iter(program.method_names))
    method = program.load_method(method_name)

    # Warmup
    for _ in range(max(args.warmup, 0)):
        _ = method.execute([audio_in])

    times_ms: list[float] = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        _ = method.execute([audio_in])
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    mean_ms = sum(times_ms) / len(times_ms)
    p95_ms = sorted(times_ms)[int(0.95 * (len(times_ms) - 1))] if times_ms else 0.0

    if args.json:
        payload = {
            "model": str(model_dir),
            "pte": str(pte_path),
            "audio": str(audio_path),
            "dtype": args.dtype,
            "fixed_frames": fixed_frames,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "mean_ms": mean_ms,
            "p95_ms": p95_ms,
            "runs_ms": times_ms,
        }
        print(json.dumps(payload, indent=2))
    else:
        print("LFM2-Audio ExecuTorch Embedding Benchmark")
        print(f"  Model dir: {model_dir}")
        print(f"  PTE:       {pte_path}")
        print(f"  Audio:     {audio_path}")
        print(f"  Dtype:     {args.dtype}")
        print(f"  Frames:    {fixed_frames}")
        print(f"  Iter:      {args.iterations}")
        print(f"  Warmup:    {args.warmup}")
        print(f"  Mean:      {mean_ms:.2f} ms")
        print(f"  P95:       {p95_ms:.2f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
