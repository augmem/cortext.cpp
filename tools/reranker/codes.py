"""Fixed (non-learned) per-item code sets: k-means of an item's AIST position features (what ingest stores).

usage: python -I codes.py --data <dataset dir> --k 16
Text item: first 256 token hidden states (incl. CLS/SEP) -> K centroids; image item: 144 patch features -> K
centroids. Spherical k-means: init at evenly spaced positions floor(linspace(0, n-1, K)), 8 Lloyd iterations,
assignment = argmax cosine (first index on ties), centroid = mean of the raw (unnormalized) member features
(empty cluster keeps its previous centroid). Items with n <= K keep their n positions.
Writes codes_k{K}_text.npy [N, K, 384] fp16, codes_k{K}_image.npy [N, K, 960] fp16 (zeros for other
modalities), codes_k{K}_cnt.npy [N] int.
The C++ ingest path (src/models/aist_reranker.cpp) implements the same procedure.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ITERS, TEXT_CAP = 8, 256


def kmeans(x: torch.Tensor, k: int) -> torch.Tensor:
    n = x.shape[0]
    if n <= k:
        return x
    xn = torch.nn.functional.normalize(x, dim=-1)
    init = torch.from_numpy(np.floor(np.linspace(0, n - 1, k)).astype(np.int64))
    c_raw = x[init].clone()
    for _ in range(ITERS):
        a = (xn @ torch.nn.functional.normalize(c_raw, dim=-1).T).argmax(1)
        oh = torch.nn.functional.one_hot(a, k).float()
        cnt = oh.sum(0)
        s = oh.T @ x
        keep = cnt > 0
        c_raw[keep] = s[keep] / cnt[keep, None]
    return c_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()
    d, K = Path(a.data), a.k
    off = np.load(d / "tok_off.npy")
    tok = np.load(d / "tok.npy", mmap_mode="r")
    N = len(off) - 1
    mod = np.load(d / "modality.npy") if (d / "modality.npy").exists() else np.zeros(N, dtype=np.int64)
    CT = np.lib.format.open_memmap(d / f"codes_k{K}_text.npy", mode="w+", dtype=np.float16, shape=(N, K, 384))
    cnt = np.zeros(N, dtype=np.int64)
    if (mod == 1).any():
        CI = np.lib.format.open_memmap(d / f"codes_k{K}_image.npy", mode="w+", dtype=np.float16, shape=(N, K, 960))
        ifeat = np.load(d / "image_feat.npy", mmap_mode="r")
        ioff = np.load(d / "image_off.npy")
    for i in range(N):
        if mod[i] == 0:
            x = torch.from_numpy(np.asarray(tok[off[i]:off[i + 1]][:TEXT_CAP], dtype=np.float32))
            c = kmeans(x, K)
            CT[i, :len(c)] = c.numpy().astype(np.float16)
        else:
            x = torch.from_numpy(np.asarray(ifeat[ioff[i]:ioff[i + 1]], dtype=np.float32))
            c = kmeans(x, K)
            CI[i, :len(c)] = c.numpy().astype(np.float16)
        cnt[i] = len(c)
        if i % 20000 == 0:
            print(f"{i}/{N}", file=sys.stderr, flush=True)
    np.save(d / f"codes_k{K}_cnt.npy", cnt)
    print(d.name, "codes", N, "K", K)


if __name__ == "__main__":
    main()
