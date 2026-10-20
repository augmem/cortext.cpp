"""Frozen AIST text features for every text of a built dataset.

usage: python -I featurize.py --data /Volumes/Build/cortext/reranker/data/eval1 [--device mps]
Writes tok.npy ([sum_T, 384] fp16 last hidden states, engine tokenization, <=512 tokens incl. CLS/SEP),
tok_off.npy ([N+1] row offsets; empty range for non-text items), sem.npy ([N, 1280] fp16 L2-normed semantic
vectors), and when image items ({"image": path}) exist: image_feat.npy ([sum_P, 960] fp16 12x12 patch
features), image_off.npy ([N+1]), modality.npy ([N] 0 text, 1 image).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aist_torch as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    a = ap.parse_args()
    d = Path(a.data)
    texts = [json.loads(l) for l in (d / "texts.jsonl").open()]
    is_img = [not isinstance(t, str) for t in texts]
    tok, m = A.load_text(a.device)
    ids = [tok.encode(t) if not im else [] for t, im in zip(texts, is_img)]
    lens = np.array([len(x) for x in ids], dtype=np.int64)
    off = np.zeros(len(ids) + 1, dtype=np.int64)
    off[1:] = np.cumsum(lens)
    np.save(d / "tok_off.npy", off)
    T = np.lib.format.open_memmap(d / "tok.npy", mode="w+", dtype=np.float16, shape=(int(off[-1]), 384))
    S = np.lib.format.open_memmap(d / "sem.npy", mode="w+", dtype=np.float16, shape=(len(ids), 1280))
    order = np.array([k for k in np.argsort(-lens) if lens[k] > 0], dtype=np.int64)
    t0, i = time.time(), 0
    while i < len(order):
        L = int(lens[order[i]])
        bs = max(1, a.tokens_per_batch // L)
        idx = order[i:i + bs]
        x = torch.zeros(len(idx), L, dtype=torch.long)
        mk = torch.zeros(len(idx), L, dtype=torch.long)
        for j, k in enumerate(idx):
            x[j, :lens[k]] = torch.tensor(ids[k]); mk[j, :lens[k]] = 1
        h, s = m(x.to(a.device), mk.to(a.device))
        h, s = h.float().cpu().numpy(), s.float().cpu().numpy()
        for j, k in enumerate(idx):
            T[off[k]:off[k + 1]] = h[j, :lens[k]].astype(np.float16)
            S[k] = s[j].astype(np.float16)
        i += len(idx)
        if (i // len(idx)) % 50 == 0:
            print(f"{i}/{len(order)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    T.flush()
    if any(is_img):
        im_idx = [k for k, f in enumerate(is_img) if f]
        ioff = np.zeros(len(texts) + 1, dtype=np.int64)
        cnt = np.zeros(len(texts), dtype=np.int64)
        cnt[im_idx] = 144
        ioff[1:] = np.cumsum(cnt)
        np.save(d / "image_off.npy", ioff)
        np.save(d / "modality.npy", np.array(is_img, dtype=np.int64))
        IF = np.lib.format.open_memmap(d / "image_feat.npy", mode="w+", dtype=np.float16, shape=(int(ioff[-1]), 960))
        im_model = A.load_image(a.device)
        for b in range(0, len(im_idx), 32):
            idx = im_idx[b:b + 32]
            x = torch.from_numpy(np.stack([A.preprocess_image(A.read_rgb(texts[k]["image"])) for k in idx]))
            p, s = im_model(x.to(a.device))
            p, s = p.float().cpu().numpy(), s.float().cpu().numpy()
            for j, k in enumerate(idx):
                IF[ioff[k]:ioff[k + 1]] = p[j].astype(np.float16)
                S[k] = s[j].astype(np.float16)
        IF.flush()
        print(d.name, "images", len(im_idx))
    S.flush()
    print(d.name, "texts", len(ids), "tokens", int(off[-1]), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
