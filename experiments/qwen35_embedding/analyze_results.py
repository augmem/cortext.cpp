#!/usr/bin/env python3
"""
Deeper analysis of Qwen3.5-0.8B hidden states for embedding quality.

1. Isotropy analysis — how uniformly distributed are embeddings in the space?
2. Layer concatenation / weighted combinations
3. PCA dimensionality analysis — effective rank of the embedding space
4. Qualitative nearest-neighbor examples
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_model_and_processor(model_id: str, device: str):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    print(f"Loading model {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    return model, processor


def load_flickr_subset(n: int = 200, seed: int = 42):
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


def extract_hidden_states(model, processor, items, mode="text", device="mps"):
    """Extract all-layer hidden states. Returns {layer: list of (seq_len, dim)}."""
    num_layers = model.config.text_config.num_hidden_layers
    all_hidden = {i: [] for i in range(num_layers + 1)}

    for i, item in enumerate(items):
        if mode == "text":
            messages = [{"role": "user", "content": [{"type": "text", "text": item}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        else:
            messages = [{"role": "user", "content": [{"type": "image", "image": item}, {"type": "text", "text": "Describe this image."}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[prompt], images=[item], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        for layer_idx in range(num_layers + 1):
            hs = outputs.hidden_states[layer_idx].cpu().float().numpy()[0]
            all_hidden[layer_idx].append(hs)

        if (i + 1) % 50 == 0:
            print(f"  {mode} extraction: {i+1}/{len(items)}")

    return all_hidden


def mean_pool(hidden_list):
    return np.stack([hs.mean(axis=0) for hs in hidden_list])


def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def compute_recall(q, g, ks=[1, 5, 10]):
    q, g = l2_normalize(q), l2_normalize(g)
    sim = q @ g.T
    results = {}
    for k in ks:
        topk = np.argsort(-sim, axis=1)[:, :k]
        results[k] = np.mean([1.0 if i in topk[i] else 0.0 for i in range(len(q))])
    return results


# ---------------------------------------------------------------------------
# Analysis 1: Isotropy — measure how uniformly spread the embeddings are
# ---------------------------------------------------------------------------

def analyze_isotropy(embeddings: np.ndarray, label: str):
    """
    Measure isotropy via:
    - Average cosine similarity (lower = more isotropic)
    - Effective dimensionality via PCA eigenvalue distribution
    - Partition function I(W) from Arora et al. 2016
    """
    embs = l2_normalize(embeddings)
    n = len(embs)

    # Average pairwise cosine
    sim_matrix = embs @ embs.T
    mask = ~np.eye(n, dtype=bool)
    avg_cos = sim_matrix[mask].mean()

    # PCA eigenvalue analysis
    centered = embeddings - embeddings.mean(axis=0)
    cov = centered.T @ centered / n
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # descending
    eigenvalues = np.maximum(eigenvalues, 0)
    total = eigenvalues.sum()
    if total > 0:
        normed = eigenvalues / total
        # Effective rank (exponential of entropy)
        nonzero = normed[normed > 1e-10]
        entropy = -np.sum(nonzero * np.log(nonzero))
        effective_rank = np.exp(entropy)
        # Variance explained by top-k
        cumvar = np.cumsum(normed)
        dim_90 = np.searchsorted(cumvar, 0.90) + 1
        dim_99 = np.searchsorted(cumvar, 0.99) + 1
    else:
        effective_rank = 0
        dim_90 = 0
        dim_99 = 0

    return {
        "avg_cosine": float(avg_cos),
        "effective_rank": float(effective_rank),
        "dim_90pct_var": int(dim_90),
        "dim_99pct_var": int(dim_99),
        "top1_eigenvalue_ratio": float(normed[0]) if total > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Analysis 2: Layer concatenation / weighted combination
# ---------------------------------------------------------------------------

def try_layer_combinations(text_hidden, image_hidden, layers_to_try):
    """Try concatenating and weighting different layer combinations."""
    results = []

    # Individual layers (baseline)
    for layer in layers_to_try:
        t_emb = mean_pool(text_hidden[layer])
        i_emb = mean_pool(image_hidden[layer])
        t2i = compute_recall(t_emb, i_emb)
        i2t = compute_recall(i_emb, t_emb)
        results.append({
            "config": f"L{layer}",
            "dim": t_emb.shape[1],
            "t2i_R@1": t2i[1], "t2i_R@10": t2i[10],
            "i2t_R@1": i2t[1], "i2t_R@10": i2t[10],
        })

    # Concatenation: L6 + L22
    for l1, l2 in [(2, 22), (6, 22), (6, 24), (18, 22), (20, 24)]:
        t_emb = np.concatenate([mean_pool(text_hidden[l1]), mean_pool(text_hidden[l2])], axis=1)
        i_emb = np.concatenate([mean_pool(image_hidden[l1]), mean_pool(image_hidden[l2])], axis=1)
        t2i = compute_recall(t_emb, i_emb)
        i2t = compute_recall(i_emb, t_emb)
        results.append({
            "config": f"L{l1}+L{l2} (concat)",
            "dim": t_emb.shape[1],
            "t2i_R@1": t2i[1], "t2i_R@10": t2i[10],
            "i2t_R@1": i2t[1], "i2t_R@10": i2t[10],
        })

    # Weighted average: alpha * L_early + (1-alpha) * L_late
    for l1, l2 in [(2, 22), (6, 24), (20, 24)]:
        for alpha in [0.3, 0.5, 0.7]:
            t_emb = alpha * mean_pool(text_hidden[l1]) + (1 - alpha) * mean_pool(text_hidden[l2])
            i_emb = alpha * mean_pool(image_hidden[l1]) + (1 - alpha) * mean_pool(image_hidden[l2])
            t2i = compute_recall(t_emb, i_emb)
            i2t = compute_recall(i_emb, t_emb)
            results.append({
                "config": f"{alpha:.1f}*L{l1}+{1-alpha:.1f}*L{l2}",
                "dim": t_emb.shape[1],
                "t2i_R@1": t2i[1], "t2i_R@10": t2i[10],
                "i2t_R@1": i2t[1], "i2t_R@10": i2t[10],
            })

    # All-layer weighted average (uniform)
    all_layers = list(text_hidden.keys())
    t_emb = np.mean([mean_pool(text_hidden[l]) for l in all_layers], axis=0)
    i_emb = np.mean([mean_pool(image_hidden[l]) for l in all_layers], axis=0)
    t2i = compute_recall(t_emb, i_emb)
    i2t = compute_recall(i_emb, t_emb)
    results.append({
        "config": "all_layers_avg",
        "dim": t_emb.shape[1],
        "t2i_R@1": t2i[1], "t2i_R@10": t2i[10],
        "i2t_R@1": i2t[1], "i2t_R@10": i2t[10],
    })

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Qualitative nearest-neighbor examples
# ---------------------------------------------------------------------------

def show_nearest_neighbors(text_embs, image_embs, captions, layer, n_examples=5):
    """Show top-1 retrieved results for qualitative inspection."""
    t, i = l2_normalize(text_embs), l2_normalize(image_embs)
    sim = t @ i.T

    print(f"\n--- Nearest Neighbors (Layer {layer}, mean pool) ---")
    print("\nText → Image (top-1):")
    for idx in range(min(n_examples, len(captions))):
        best = np.argmax(sim[idx])
        print(f"  Query: \"{captions[idx][:80]}...\"")
        print(f"  Retrieved image #{best} (sim={sim[idx][best]:.3f}) {'✓' if best == idx else '✗'}")

    print("\nImage → Text (top-1):")
    for idx in range(min(n_examples, len(captions))):
        best = np.argmax(sim[:, idx])
        print(f"  Query: image #{idx}")
        print(f"  Retrieved: \"{captions[best][:80]}...\" (sim={sim[best][idx]:.3f}) {'✓' if best == idx else '✗'}")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "Qwen/Qwen3.5-0.8B"
    n_samples = 200

    model, processor = load_model_and_processor(model_id, device)
    images, captions = load_flickr_subset(n=n_samples)

    print("\n=== Extracting hidden states ===")
    print("Text...")
    text_hidden = extract_hidden_states(model, processor, captions, mode="text", device=device)
    print("Images...")
    image_hidden = extract_hidden_states(model, processor, images, mode="image", device=device)

    # --- Analysis 1: Isotropy ---
    print("\n" + "=" * 80)
    print("ISOTROPY ANALYSIS")
    print("=" * 80)
    key_layers = [0, 2, 6, 12, 18, 22, 24]
    for layer in key_layers:
        t_emb = mean_pool(text_hidden[layer])
        i_emb = mean_pool(image_hidden[layer])
        t_iso = analyze_isotropy(t_emb, f"text_L{layer}")
        i_iso = analyze_isotropy(i_emb, f"image_L{layer}")
        print(f"\nLayer {layer:2d}:")
        print(f"  Text  — avg_cos={t_iso['avg_cosine']:.3f}  eff_rank={t_iso['effective_rank']:.0f}  dim90={t_iso['dim_90pct_var']}  dim99={t_iso['dim_99pct_var']}  top1_eig={t_iso['top1_eigenvalue_ratio']:.3f}")
        print(f"  Image — avg_cos={i_iso['avg_cosine']:.3f}  eff_rank={i_iso['effective_rank']:.0f}  dim90={i_iso['dim_90pct_var']}  dim99={i_iso['dim_99pct_var']}  top1_eig={i_iso['top1_eigenvalue_ratio']:.3f}")

    # --- Analysis 2: Layer combinations ---
    print("\n" + "=" * 80)
    print("LAYER COMBINATION ANALYSIS")
    print("=" * 80)
    combo_results = try_layer_combinations(text_hidden, image_hidden, key_layers)

    # Sort by sum of R@1
    combo_results.sort(key=lambda r: r["t2i_R@1"] + r["i2t_R@1"], reverse=True)
    print(f"\n{'Config':<30s} {'Dim':>5s} {'T→I R@1':>9s} {'T→I R@10':>9s} {'I→T R@1':>9s} {'I→T R@10':>9s}")
    print("-" * 80)
    for r in combo_results:
        print(f"{r['config']:<30s} {r['dim']:>5d} {r['t2i_R@1']:>9.3f} {r['t2i_R@10']:>9.3f} {r['i2t_R@1']:>9.3f} {r['i2t_R@10']:>9.3f}")

    # --- Analysis 3: Qualitative ---
    best_layer = 22
    t_emb = mean_pool(text_hidden[best_layer])
    i_emb = mean_pool(image_hidden[best_layer])
    show_nearest_neighbors(t_emb, i_emb, captions, best_layer)

    # --- Save ---
    out_dir = Path("experiments/qwen35_embedding/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "deep_analysis.json", "w") as f:
        json.dump({
            "combo_results": combo_results,
            "model": model_id,
            "n_samples": n_samples,
        }, f, indent=2)
    print(f"\nResults saved to {out_dir / 'deep_analysis.json'}")


if __name__ == "__main__":
    main()
