#!/usr/bin/env python3
"""
Test whether ASR transcripts produce useful embeddings for audio→text→image retrieval.

Uses synthetic experiment: compare embeddings of:
1. Original captions
2. Degraded captions (simulating ASR noise: word drops, substitutions)
3. Paraphrased captions (semantic equivalents)

If degraded captions still retrieve correct images, the embedding space
is robust enough for audio→ASR→embed pipeline.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_model_and_processor(model_id, device):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    return model, processor


def load_flickr_subset(n=200, seed=42):
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
    indices = rng.permutation(len(collected))[:n]
    images, captions = [], []
    for idx in indices:
        images.append(collected[idx][0])
        captions.append(collected[idx][1])
    return images, captions


def degrade_caption(caption: str, rng: random.Random, noise_level: float = 0.2) -> str:
    """Simulate ASR errors: word drops, phonetic substitutions, filler insertion."""
    words = caption.split()
    result = []
    for w in words:
        r = rng.random()
        if r < noise_level * 0.5:
            # Drop word
            continue
        elif r < noise_level:
            # Substitute with phonetically similar (simplified: truncate/mangle)
            if len(w) > 3:
                result.append(w[:len(w)-1] + rng.choice("aeions"))
            else:
                result.append(w)
        else:
            result.append(w)
        # Occasionally insert filler
        if rng.random() < noise_level * 0.1:
            result.append(rng.choice(["um", "uh", "like"]))
    return " ".join(result) if result else caption


def extract_text_embeddings(model, processor, texts, device, layer=22):
    """Extract mean-pooled hidden states at given layer."""
    embeddings = []
    for text in texts:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hs = outputs.hidden_states[layer].cpu().float().numpy()[0]
        embeddings.append(hs.mean(axis=0))

    return np.stack(embeddings)


def extract_image_embeddings(model, processor, images, device, layer=22):
    """Extract mean-pooled hidden states for images."""
    embeddings = []
    for img in images:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe this image."},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hs = outputs.hidden_states[layer].cpu().float().numpy()[0]
        embeddings.append(hs.mean(axis=0))

    return np.stack(embeddings)


def l2_normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def recall_at_k(q, g, ks=[1, 5, 10]):
    q, g = l2_normalize(q), l2_normalize(g)
    sim = q @ g.T
    results = {}
    for k in ks:
        topk = np.argsort(-sim, axis=1)[:, :k]
        results[k] = np.mean([1.0 if i in topk[i] else 0.0 for i in range(len(q))])
    return results


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    n = 200
    rng = random.Random(42)

    model, processor = load_model_and_processor("Qwen/Qwen3.5-0.8B", device)
    images, captions = load_flickr_subset(n=n)

    # Generate degraded captions at various noise levels
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    degraded_sets = {}
    for noise in noise_levels:
        degraded_sets[noise] = [degrade_caption(c, rng, noise) for c in captions]

    # Test with best two layer configs
    for layer, label in [(22, "L22"), (2, "L2")]:
        print(f"\n{'='*80}")
        print(f"LAYER {layer} — ASR ROBUSTNESS ANALYSIS")
        print(f"{'='*80}")

        # Image embeddings (reference)
        print(f"  Extracting image embeddings (layer {layer})...")
        img_embs = extract_image_embeddings(model, processor, images, device, layer=layer)

        for noise in noise_levels:
            texts = degraded_sets[noise]
            print(f"\n  Noise={noise:.1f}:")
            if noise > 0:
                # Show example
                idx = 0
                print(f"    Original: \"{captions[idx][:70]}\"")
                print(f"    Degraded: \"{texts[idx][:70]}\"")

            text_embs = extract_text_embeddings(model, processor, texts, device, layer=layer)

            # Text → Image retrieval
            t2i = recall_at_k(text_embs, img_embs)
            # Also measure semantic drift: cosine between clean and noisy text embeddings
            if noise > 0:
                clean_embs = extract_text_embeddings(model, processor, captions, device, layer=layer)
                drift = 1.0 - np.mean(np.sum(l2_normalize(text_embs) * l2_normalize(clean_embs), axis=1))
            else:
                drift = 0.0

            print(f"    T→I: R@1={t2i[1]:.3f}  R@5={t2i[5]:.3f}  R@10={t2i[10]:.3f}  drift={drift:.4f}")

    # Blended layer approach
    print(f"\n{'='*80}")
    print("BLENDED (0.5*L2 + 0.5*L22) — ASR ROBUSTNESS")
    print(f"{'='*80}")

    img_embs_2 = extract_image_embeddings(model, processor, images, device, layer=2)
    img_embs_22 = extract_image_embeddings(model, processor, images, device, layer=22)
    img_blend = 0.5 * img_embs_2 + 0.5 * img_embs_22

    for noise in noise_levels:
        texts = degraded_sets[noise]
        t_embs_2 = extract_text_embeddings(model, processor, texts, device, layer=2)
        t_embs_22 = extract_text_embeddings(model, processor, texts, device, layer=22)
        t_blend = 0.5 * t_embs_2 + 0.5 * t_embs_22

        t2i = recall_at_k(t_blend, img_blend)
        print(f"  Noise={noise:.1f}: T→I R@1={t2i[1]:.3f}  R@5={t2i[5]:.3f}  R@10={t2i[10]:.3f}")

    # Save
    out = Path("experiments/qwen35_embedding/results")
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nDone. ASR bridge analysis complete.")


if __name__ == "__main__":
    main()
