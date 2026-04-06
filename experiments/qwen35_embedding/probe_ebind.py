#!/usr/bin/env python3
"""
Probe EBind for cross-modal retrieval: image→text, audio→text, text→text.
EBind projects image/video/text/audio/3D into a shared 1024d unit-norm space.
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_ebind(device):
    from ebind import EBindModel, EBindProcessor

    model_id = "encord-team/ebind-full"
    print(f"Loading {model_id} ...")
    t0 = time.time()
    model = EBindModel.from_pretrained(model_id)
    processor = EBindProcessor.from_pretrained(model_id)
    model = model.to(device).eval()
    processor = processor.to(device)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, processor


def load_flickr(n=200, seed=42):
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_flickr30k", split="test", streaming=True)
    rng = np.random.RandomState(seed)
    collected = []
    for row in ds:
        img = row["jpg"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        collected.append((img, row["txt"]))
        if len(collected) >= n * 3:
            break
    idx = rng.permutation(len(collected))[:n]
    return [collected[i][0] for i in idx], [collected[i][1] for i in idx]


def l2_norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def recall_at_k(q, g, ks=[1, 5, 10]):
    sim = l2_norm(q) @ l2_norm(g).T
    return {k: np.mean([1.0 if i in np.argsort(-sim[i])[:k] else 0.0 for i in range(len(q))]) for k in ks}


def make_test_audio(n, sr=48000, dur=3.0, seed=42):
    """Synthetic audio for pipeline testing."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
    return [0.4 * np.sin(2 * np.pi * rng.uniform(100, 1500) * t) + rng.randn(len(t)).astype(np.float32) * 0.05
            for _ in range(n)]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    n = 200

    model, processor = load_ebind(device)
    images, captions = load_flickr(n=n)

    # Save images to temp files (EBind processor expects file paths)
    tmp = Path(tempfile.mkdtemp())
    img_paths = []
    for i, img in enumerate(images):
        p = tmp / f"img_{i:04d}.png"
        img.save(p)
        img_paths.append(str(p))

    # --- Encode all modalities ---
    print("\n=== Encoding images ===")
    t0 = time.time()
    with torch.inference_mode():
        img_batch = processor({"image": img_paths}, return_tensors="pt")
        img_out = model(**img_batch)
    img_embs = img_out["image"].cpu().float().numpy()
    print(f"  {img_embs.shape} in {time.time() - t0:.1f}s")

    print("\n=== Encoding text ===")
    t0 = time.time()
    with torch.inference_mode():
        txt_batch = processor({"text": captions}, return_tensors="pt")
        txt_out = model(**txt_batch)
    txt_embs = txt_out["text"].cpu().float().numpy()
    print(f"  {txt_embs.shape} in {time.time() - t0:.1f}s")

    print("\n=== Encoding audio (synthetic) ===")
    # Save audio to temp wav files
    import wave as wave_mod
    aud_paths = []
    sr = 48000
    audios = make_test_audio(n, sr=sr)
    for i, w in enumerate(audios):
        p = tmp / f"aud_{i:04d}.wav"
        with wave_mod.open(str(p), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((w * 32767).astype(np.int16).tobytes())
        aud_paths.append(str(p))

    t0 = time.time()
    with torch.inference_mode():
        aud_batch = processor({"audio": aud_paths}, return_tensors="pt")
        aud_out = model(**aud_batch)
    aud_embs = aud_out["audio"].cpu().float().numpy()
    print(f"  {aud_embs.shape} in {time.time() - t0:.1f}s")

    # --- Retrieval evaluation ---
    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    i2t = recall_at_k(img_embs, txt_embs)
    t2i = recall_at_k(txt_embs, img_embs)
    print(f"  Image→Text: R@1={i2t[1]:.3f}  R@5={i2t[5]:.3f}  R@10={i2t[10]:.3f}")
    print(f"  Text→Image: R@1={t2i[1]:.3f}  R@5={t2i[5]:.3f}  R@10={t2i[10]:.3f}")

    a2t = recall_at_k(aud_embs, txt_embs)
    print(f"  Audio→Text (synthetic): R@1={a2t[1]:.3f}  R@5={a2t[5]:.3f}  R@10={a2t[10]:.3f}")

    a2i = recall_at_k(aud_embs, img_embs)
    print(f"  Audio→Image (synthetic): R@1={a2i[1]:.3f}  (sanity, should be ~random)")

    # --- Diagnostics ---
    print("\n--- Embedding diagnostics ---")
    img_n, txt_n, aud_n = l2_norm(img_embs), l2_norm(txt_embs), l2_norm(aud_embs)

    img_self = img_n @ img_n.T
    txt_self = txt_n @ txt_n.T
    aud_self = aud_n @ aud_n.T
    mask = ~np.eye(n, dtype=bool)

    print(f"  Intra-modal avg cosine:")
    print(f"    image: {img_self[mask].mean():.3f}")
    print(f"    text:  {txt_self[mask].mean():.3f}")
    print(f"    audio: {aud_self[mask].mean():.3f}")

    cross_it = img_n @ txt_n.T
    print(f"  Cross-modal (img↔txt): mean={cross_it.mean():.3f}  diag={np.diag(cross_it).mean():.3f}  separation={np.diag(cross_it).mean() - cross_it[mask].mean():.3f}")

    print(f"\n  Embedding dims: image={img_embs.shape[1]} text={txt_embs.shape[1]} audio={aud_embs.shape[1]}")

    # --- Comparison with Qwen3.5-0.8B ---
    print("\n" + "=" * 70)
    print("COMPARISON: EBind vs Qwen3.5-0.8B (Flickr30k n=200)")
    print("=" * 70)
    print(f"  {'Model':<30s} {'T→I R@1':>9s} {'T→I R@10':>9s} {'I→T R@1':>9s} {'I→T R@10':>9s}")
    print(f"  {'EBind-full':<30s} {t2i[1]:>9.3f} {t2i[10]:>9.3f} {i2t[1]:>9.3f} {i2t[10]:>9.3f}")
    print(f"  {'Qwen3.5 (0.5*L2+0.5*L22)':<30s} {'0.460':>9s} {'0.870':>9s} {'0.415':>9s} {'0.800':>9s}")
    print(f"  {'Qwen3.5 (all_layers_avg)':<30s} {'0.390':>9s} {'0.840':>9s} {'0.560':>9s} {'0.850':>9s}")

    # Save
    out = Path("experiments/qwen35_embedding/results")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ebind_results.json", "w") as f:
        json.dump({
            "i2t": {str(k): v for k, v in i2t.items()},
            "t2i": {str(k): v for k, v in t2i.items()},
            "a2t_synthetic": {str(k): v for k, v in a2t.items()},
            "dims": {"image": img_embs.shape[1], "text": txt_embs.shape[1], "audio": aud_embs.shape[1]},
            "diagnostics": {
                "img_self_sim": float(img_self[mask].mean()),
                "txt_self_sim": float(txt_self[mask].mean()),
                "aud_self_sim": float(aud_self[mask].mean()),
            },
        }, f, indent=2)
    print(f"\nSaved to {out / 'ebind_results.json'}")

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
