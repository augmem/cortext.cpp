#!/usr/bin/env python3
"""
Probe Qwen3.5-0.8B hidden states for cross-modal embedding viability.

Experiment: extract hidden states from text and image inputs, evaluate whether
the shared 1024-dim representation space supports cross-modal retrieval via
cosine similarity.

Sweeps:
  - Layer index (0..23 + vision encoder output)
  - Pooling strategy (last_token, mean, max, eos_token)
  - Normalization (L2 vs raw)

Evaluation:
  - Text→Image recall@{1,5,10} on COCO val (1K subset)
  - Image→Text recall@{1,5,10}
  - Intra-modal vs cross-modal alignment (CKA or centered cosine)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(model_id: str, device: str):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    print(f"Loading model {model_id} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s on {device}")
    print(f"  Text hidden size: {model.config.text_config.hidden_size}")
    print(f"  Num layers: {model.config.text_config.num_hidden_layers}")
    return model, processor


# ---------------------------------------------------------------------------
# Dataset: COCO captions (val split, 1K subset)
# ---------------------------------------------------------------------------

def load_coco_subset(n: int = 1000, seed: int = 42):
    """Load n image-caption pairs from Flickr30k via HuggingFace datasets."""
    from datasets import load_dataset

    print(f"Loading Flickr30k subset (n={n}) ...")
    ds = load_dataset(
        "clip-benchmark/wds_flickr30k",
        split="test",
        streaming=True,
    )

    rng = np.random.RandomState(seed)

    images = []
    captions = []
    # Stream and collect; we'll subsample after
    collected = []
    for i, row in enumerate(ds):
        img = row["jpg"]
        cap = row["txt"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        collected.append((img, cap))
        if len(collected) >= n * 3:  # collect extra for subsampling
            break

    # Subsample
    indices = rng.permutation(len(collected))[:n]
    for idx in indices:
        img, cap = collected[idx]
        images.append(img)
        captions.append(cap)

    print(f"  Loaded {len(images)} image-caption pairs")
    return images, captions


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_text_hidden_states(
    model, processor, texts: list[str], batch_size: int = 16, device: str = "mps"
) -> dict[int, np.ndarray]:
    """
    Extract hidden states from text-only inputs across all layers.

    Returns: {layer_idx: (N, seq_len, hidden_dim)} for layers 0..num_layers
    """
    num_layers = model.config.text_config.num_hidden_layers
    all_hidden = {i: [] for i in range(num_layers + 1)}  # +1 for embedding layer

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        # Build chat-style messages (text only)
        batch_messages = []
        for text in batch_texts:
            batch_messages.append([
                {"role": "user", "content": [{"type": "text", "text": text}]}
            ])

        # Process each message individually since apply_chat_template may not batch
        for messages in batch_messages:
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = processor(
                text=[prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # outputs.hidden_states: tuple of (batch, seq_len, hidden_dim)
            # Layer 0 = embedding output, layer 1..N = transformer layer outputs
            for layer_idx in range(num_layers + 1):
                hs = outputs.hidden_states[layer_idx].cpu().float().numpy()
                all_hidden[layer_idx].append(hs[0])  # unbatch

        if (start + batch_size) % 100 == 0 or start + batch_size >= len(texts):
            print(f"  Text extraction: {min(start + batch_size, len(texts))}/{len(texts)}")

    return all_hidden


def extract_image_hidden_states(
    model, processor, images: list[Image.Image], batch_size: int = 4, device: str = "mps"
) -> dict[int, np.ndarray]:
    """
    Extract hidden states from image inputs (with minimal text prompt).

    Returns: {layer_idx: list of (seq_len, hidden_dim)} for layers 0..num_layers
    """
    num_layers = model.config.text_config.num_hidden_layers
    all_hidden = {i: [] for i in range(num_layers + 1)}

    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]

        for img in batch_images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]

            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = processor(
                text=[prompt],
                images=[img],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            for layer_idx in range(num_layers + 1):
                hs = outputs.hidden_states[layer_idx].cpu().float().numpy()
                all_hidden[layer_idx].append(hs[0])

        if (start + batch_size) % 20 == 0 or start + batch_size >= len(images):
            print(f"  Image extraction: {min(start + batch_size, len(images))}/{len(images)}")

    return all_hidden


# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------

def pool_hidden_states(
    hidden_list: list[np.ndarray], strategy: str = "mean"
) -> np.ndarray:
    """
    Pool variable-length hidden states into fixed-size embeddings.

    hidden_list: list of (seq_len, hidden_dim) arrays
    strategy: 'mean', 'last_token', 'max', 'first_token'

    Returns: (N, hidden_dim)
    """
    embeddings = []
    for hs in hidden_list:
        if strategy == "mean":
            emb = hs.mean(axis=0)
        elif strategy == "last_token":
            emb = hs[-1]
        elif strategy == "first_token":
            emb = hs[0]
        elif strategy == "max":
            emb = hs.max(axis=0)
        else:
            raise ValueError(f"Unknown pooling: {strategy}")
        embeddings.append(emb)
    return np.stack(embeddings)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return x / norms


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def compute_recall_at_k(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    ks: list[int] = [1, 5, 10],
) -> dict[int, float]:
    """
    Compute recall@k for retrieval.
    Assumes query[i] should match gallery[i] (diagonal ground truth).
    """
    # Cosine similarity matrix
    query_embs = l2_normalize(query_embs)
    gallery_embs = l2_normalize(gallery_embs)
    sim_matrix = query_embs @ gallery_embs.T  # (N_q, N_g)

    results = {}
    n = sim_matrix.shape[0]
    for k in ks:
        # For each query, check if correct gallery item is in top-k
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
        correct = sum(1 for i in range(n) if i in top_k_indices[i])
        results[k] = correct / n
    return results


def compute_alignment_stats(
    text_embs: np.ndarray, image_embs: np.ndarray
) -> dict:
    """Compute cross-modal alignment statistics."""
    text_embs = l2_normalize(text_embs)
    image_embs = l2_normalize(image_embs)

    # Matched pair similarities (diagonal)
    matched_sims = np.sum(text_embs * image_embs, axis=1)

    # All-pairs similarity
    sim_matrix = text_embs @ image_embs.T
    # Off-diagonal (unmatched) similarities
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    unmatched_sims = sim_matrix[mask]

    return {
        "matched_mean": float(matched_sims.mean()),
        "matched_std": float(matched_sims.std()),
        "unmatched_mean": float(unmatched_sims.mean()),
        "unmatched_std": float(unmatched_sims.std()),
        "separation": float(matched_sims.mean() - unmatched_sims.mean()),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model, processor = load_model_and_processor(args.model, device)
    images, captions = load_coco_subset(n=args.n_samples, seed=args.seed)

    print("\n--- Extracting text hidden states ---")
    text_hidden = extract_text_hidden_states(
        model, processor, captions, batch_size=args.text_batch_size, device=device
    )

    print("\n--- Extracting image hidden states ---")
    image_hidden = extract_image_hidden_states(
        model, processor, images, batch_size=args.image_batch_size, device=device
    )

    # Sweep across layers and pooling strategies
    num_layers = model.config.text_config.num_hidden_layers
    pooling_strategies = ["mean", "last_token", "first_token", "max"]
    layers_to_probe = list(range(0, num_layers + 1, args.layer_stride))
    # Always include last layer
    if num_layers not in layers_to_probe:
        layers_to_probe.append(num_layers)

    results = []

    print(f"\n--- Evaluating {len(layers_to_probe)} layers x {len(pooling_strategies)} pooling strategies ---\n")

    for layer_idx in layers_to_probe:
        for pool in pooling_strategies:
            text_embs = pool_hidden_states(text_hidden[layer_idx], strategy=pool)
            image_embs = pool_hidden_states(image_hidden[layer_idx], strategy=pool)

            # Text → Image retrieval
            t2i = compute_recall_at_k(text_embs, image_embs)
            # Image → Text retrieval
            i2t = compute_recall_at_k(image_embs, text_embs)
            # Alignment stats
            align = compute_alignment_stats(text_embs, image_embs)

            result = {
                "layer": layer_idx,
                "pooling": pool,
                "t2i_R@1": t2i[1],
                "t2i_R@5": t2i[5],
                "t2i_R@10": t2i[10],
                "i2t_R@1": i2t[1],
                "i2t_R@5": i2t[5],
                "i2t_R@10": i2t[10],
                "alignment": align,
            }
            results.append(result)

            print(
                f"  Layer {layer_idx:2d} | {pool:12s} | "
                f"T→I R@1={t2i[1]:.3f} R@5={t2i[5]:.3f} R@10={t2i[10]:.3f} | "
                f"I→T R@1={i2t[1]:.3f} R@5={i2t[5]:.3f} R@10={i2t[10]:.3f} | "
                f"sep={align['separation']:.3f}"
            )

    # Find best configuration
    best = max(results, key=lambda r: r["t2i_R@1"] + r["i2t_R@1"])
    print(f"\n{'='*80}")
    print(f"BEST CONFIG: Layer {best['layer']}, Pooling: {best['pooling']}")
    print(f"  T→I: R@1={best['t2i_R@1']:.3f}  R@5={best['t2i_R@5']:.3f}  R@10={best['t2i_R@10']:.3f}")
    print(f"  I→T: R@1={best['i2t_R@1']:.3f}  R@5={best['i2t_R@5']:.3f}  R@10={best['i2t_R@10']:.3f}")
    print(f"  Alignment separation: {best['alignment']['separation']:.3f}")
    print(f"{'='*80}")

    # Dimensionality report
    hidden_dim = model.config.text_config.hidden_size
    print(f"\nEmbedding dimensionality: {hidden_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_file = out_dir / "retrieval_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "model": args.model,
                "n_samples": args.n_samples,
                "device": device,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "results": results,
                "best": best,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_file}")

    return results, best


def main():
    parser = argparse.ArgumentParser(description="Probe Qwen3.5-0.8B for cross-modal embedding")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="HuggingFace model ID")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of image-caption pairs")
    parser.add_argument("--text-batch-size", type=int, default=16, help="Text batch size")
    parser.add_argument("--image-batch-size", type=int, default=2, help="Image batch size")
    parser.add_argument("--layer-stride", type=int, default=4, help="Probe every N-th layer")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="experiments/qwen35_embedding/results",
        help="Output directory",
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
