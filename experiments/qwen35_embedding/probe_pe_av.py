#!/usr/bin/env python3
"""
Probe PE-AV-Small for cross-modal retrieval: image→text, text→text, audio→text.

Uses sub-component calls since forward() requires ≥2 modalities.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_pe_av(device):
    from transformers import PeAudioVideoModel, PeAudioVideoProcessor

    model_id = "facebook/pe-av-small"
    print(f"Loading {model_id} ...")
    t0 = time.time()
    proc = PeAudioVideoProcessor.from_pretrained(model_id)
    model = PeAudioVideoModel.from_pretrained(model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s  ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
    return model, proc


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


# --- Embedding extraction via sub-components ---

def encode_images(model, proc, images, device, n_frames=16):
    """video_encoder → video_head → embedding. Replicates image to n_frames."""
    embs = []
    for i, img in enumerate(images):
        # Replicate single image to n_frames to avoid temporal encoder degeneration
        frames = np.stack([np.array(img)] * n_frames, axis=0)
        inp = proc(videos=[frames], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            vid_out = model.video_model.video_encoder(
                pixel_values_videos=inp["pixel_values_videos"],
                padding_mask_videos=inp.get("padding_mask_videos"),
            )
            emb = model.video_model.video_head(vid_out.pooler_output)
        embs.append(emb.cpu().float().numpy()[0])
        if (i + 1) % 50 == 0:
            print(f"    images: {i+1}/{len(images)}")
    return np.stack(embs)


def encode_images_vit_only(model, proc, images, device):
    """Bypass temporal encoder — use ViT backbone directly."""
    embs = []
    vit = model.video_model.video_encoder.vision_model
    for i, img in enumerate(images):
        inp = proc(videos=[np.array(img)[np.newaxis]], return_tensors="pt", padding=True).to(device)
        pv = inp["pixel_values_videos"]
        # pv shape: (batch, n_frames, C, H, W) — extract single frame
        frame = pv[:, 0]  # (1, C, H, W)
        with torch.no_grad():
            vit_out = vit(frame)
            # Use pooler_output or CLS token
            if hasattr(vit_out, 'pooler_output') and vit_out.pooler_output is not None:
                emb = vit_out.pooler_output
            else:
                emb = vit_out.last_hidden_state[:, 0]  # CLS
        embs.append(emb.cpu().float().numpy()[0])
        if (i + 1) % 50 == 0:
            print(f"    images(vit): {i+1}/{len(images)}")
    return np.stack(embs)


def encode_text_video(model, proc, texts, device):
    """text_model (CLS token) → text_video_head → embedding"""
    embs = []
    for i, t in enumerate(texts):
        inp = proc(text=[t], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            txt_out = model.video_model.text_model(
                input_ids=inp["input_ids"],
                attention_mask=inp.get("attention_mask"),
                output_hidden_states=True,
            )
            cls = txt_out.hidden_states[-1][:, 0]
            emb = model.video_model.text_video_head(cls)
        embs.append(emb.cpu().float().numpy()[0])
        if (i + 1) % 50 == 0:
            print(f"    text(video): {i+1}/{len(texts)}")
    return np.stack(embs)


def encode_text_audio(model, proc, texts, device):
    """text_model (CLS token) → text_audio_head → embedding"""
    embs = []
    for i, t in enumerate(texts):
        inp = proc(text=[t], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            txt_out = model.audio_model.text_model(
                input_ids=inp["input_ids"],
                attention_mask=inp.get("attention_mask"),
                output_hidden_states=True,
            )
            cls = txt_out.hidden_states[-1][:, 0]
            emb = model.audio_model.text_audio_head(cls)
        embs.append(emb.cpu().float().numpy()[0])
        if (i + 1) % 50 == 0:
            print(f"    text(audio): {i+1}/{len(texts)}")
    return np.stack(embs)


def encode_audio(model, proc, waveforms, device, sr=48000):
    """audio_encoder → audio_head → embedding"""
    embs = []
    for i, w in enumerate(waveforms):
        inp = proc(audio=[w], sampling_rate=sr, return_tensors="pt", padding=True)
        # Cast audio input to model dtype (float16)
        dtype = next(model.audio_model.audio_encoder.parameters()).dtype
        inp = {k: v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            aud_out = model.audio_model.audio_encoder(
                input_values=inp["input_values"],
                padding_mask=inp.get("padding_mask"),
            )
            emb = model.audio_model.audio_head(aud_out.pooler_output)
        embs.append(emb.cpu().float().numpy()[0])
        if (i + 1) % 20 == 0:
            print(f"    audio: {i+1}/{len(waveforms)}")
    return np.stack(embs)


def make_test_audio(n, sr=48000, dur=3.0, seed=42):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
    return [0.4 * np.sin(2 * np.pi * rng.uniform(100, 1500) * t) + rng.randn(len(t)).astype(np.float32) * 0.05
            for _ in range(n)]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    n = 200

    model, proc = load_pe_av(device)
    images, captions = load_flickr(n=n)

    # --- Image → Text ---
    print("\n=== Image → Text ===")
    img_embs = encode_images(model, proc, images, device)
    txt_v_embs = encode_text_video(model, proc, captions, device)

    i2t = recall_at_k(img_embs, txt_v_embs)
    t2i = recall_at_k(txt_v_embs, img_embs)
    print(f"  Image→Text: R@1={i2t[1]:.3f}  R@5={i2t[5]:.3f}  R@10={i2t[10]:.3f}")
    print(f"  Text→Image: R@1={t2i[1]:.3f}  R@5={t2i[5]:.3f}  R@10={t2i[10]:.3f}")
    print(f"  Dim: {img_embs.shape[1]}")

    # Diagnostic: check for embedding collapse
    img_n = l2_norm(img_embs)
    txt_n = l2_norm(txt_v_embs)
    img_self = img_n @ img_n.T
    txt_self = txt_n @ txt_n.T
    cross = img_n @ txt_n.T
    print(f"  Diag: img self-sim={img_self[~np.eye(n,dtype=bool)].mean():.3f}  "
          f"txt self-sim={txt_self[~np.eye(n,dtype=bool)].mean():.3f}  "
          f"cross mean={cross.mean():.3f}  cross diag={np.diag(cross).mean():.3f}")

    # Try ViT-only bypass
    print("\n  --- ViT-only (bypass temporal encoder) ---")
    vit_embs = encode_images_vit_only(model, proc, images, device)
    vit_n = l2_norm(vit_embs)
    vit_self = vit_n @ vit_n.T
    print(f"  ViT self-sim: {vit_self[~np.eye(n,dtype=bool)].mean():.3f}  dim={vit_embs.shape[1]}")
    # Can't directly compare with text (different space, no projection head)
    # but check if representations are discriminative
    i2t_vit = recall_at_k(vit_embs, txt_v_embs)
    t2i_vit = recall_at_k(txt_v_embs, vit_embs)
    print(f"  ViT Image→Text: R@1={i2t_vit[1]:.3f}  R@5={i2t_vit[5]:.3f}  R@10={i2t_vit[10]:.3f}")
    print(f"  ViT Text→Image: R@1={t2i_vit[1]:.3f}  R@5={t2i_vit[5]:.3f}  R@10={t2i_vit[10]:.3f}")

    # --- Text → Text (video-aligned vs audio-aligned) ---
    print("\n=== Text → Text (video vs audio head) ===")
    txt_a_embs = encode_text_audio(model, proc, captions, device)

    cross = l2_norm(txt_v_embs) @ l2_norm(txt_a_embs).T
    diag = np.diag(cross)
    print(f"  video-text ↔ audio-text matched cos: {diag.mean():.3f} ± {diag.std():.3f}")
    print(f"  Dims: video={txt_v_embs.shape[1]}, audio={txt_a_embs.shape[1]}")

    # Self-retrieval within each head
    self_v = recall_at_k(txt_v_embs, txt_v_embs)
    self_a = recall_at_k(txt_a_embs, txt_a_embs)
    print(f"  Text→Text (video head) self-R@5={self_v[5]:.3f}")
    print(f"  Text→Text (audio head) self-R@5={self_a[5]:.3f}")

    # --- Audio → Text ---
    print("\n=== Audio → Text ===")
    audios = make_test_audio(n)
    aud_embs = encode_audio(model, proc, audios, device)

    a2t = recall_at_k(aud_embs, txt_a_embs)
    print(f"  Audio→Text (synthetic): R@1={a2t[1]:.3f}  R@5={a2t[5]:.3f}  R@10={a2t[10]:.3f}")
    print(f"  (random tones — verifies pipeline, not retrieval quality)")
    print(f"  Dim: {aud_embs.shape[1]}")

    aud_sim = l2_norm(aud_embs) @ l2_norm(aud_embs).T
    print(f"  Intra-audio avg cos: {aud_sim[~np.eye(n, dtype=bool)].mean():.3f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PE-AV-Small SUMMARY")
    print("=" * 70)
    print(f"  Image→Text  R@1={i2t[1]:.3f}  R@5={i2t[5]:.3f}  R@10={i2t[10]:.3f}")
    print(f"  Text→Image  R@1={t2i[1]:.3f}  R@5={t2i[5]:.3f}  R@10={t2i[10]:.3f}")
    print(f"  Embedding dims: image={img_embs.shape[1]} txt_v={txt_v_embs.shape[1]} txt_a={txt_a_embs.shape[1]} audio={aud_embs.shape[1]}")

    out = Path("experiments/qwen35_embedding/results")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pe_av_results.json", "w") as f:
        json.dump({
            "i2t": {str(k): v for k, v in i2t.items()},
            "t2i": {str(k): v for k, v in t2i.items()},
            "a2t_synthetic": {str(k): v for k, v in a2t.items()},
            "dims": {"image": img_embs.shape[1], "txt_v": txt_v_embs.shape[1],
                     "txt_a": txt_a_embs.shape[1], "audio": aud_embs.shape[1]},
        }, f, indent=2)
    print(f"\nSaved to {out / 'pe_av_results.json'}")


if __name__ == "__main__":
    main()
