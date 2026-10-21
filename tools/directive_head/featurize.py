"""AIST text features for a jsonl of texts, through the PyTorch mirror of the shipped GGUF (tools/reranker/aist_torch.py).

usage: python -I featurize.py --in rows.jsonl --out DIR [--tok engine|unicode] [--tokens] [--device mps] [--max-bytes 0]
       [--head runs/attn/attn.pt]   (also scores every embedded row with that attention-pool head -> head_scores.npy)
Writes DIR/sem.npy ([N, 1280] fp16, L2-normed semantic vector: exactly what the engine embeds for a text signal
with --tok engine) and, with --tokens, DIR/tok.npy ([sum_T, 384] fp16 last hidden states) + DIR/tok_off.npy.
Rows longer than --max-bytes (UTF-8, 0 = no limit) are not embedded and get a zero vector (DIR/embedded.npy marks them).

--tok engine  : the engine tokenizer (every byte >= 0x80 becomes [UNK]; src/models/aist_gguf_encoder.cpp).
--tok unicode : a counterfactual BERT-style basic tokenizer (NFD + accent strip, CJK/kana per character,
                Unicode punctuation split) over the same vocabulary. It is NOT what the engine does; it measures
                what a tokenizer fix would buy (see README).
"""
from __future__ import annotations

import argparse, json, os, sys, time, unicodedata
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reranker"))
import aist_torch as A  # noqa: E402

DEFAULT_GGUF = os.environ.get(
    "AIST_GGUF", "/Users/gabrielwillen/VSCode/augmem/augmem/cortext.cpp/.claude/worktrees/land-rr/models/AIST-87M-GGUF/AIST-87M_q8_0.gguf")


def _cjk(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF
            or 0x3040 <= cp <= 0x30FF)


class UnicodeTokenizer(A.Tokenizer):
    def basic(self, text: str) -> list[str]:  # noqa: D401
        t = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
        out, cur = [], []
        for c in t:
            if c.isspace():
                if cur: out.append("".join(cur)); cur = []
            elif _cjk(ord(c)) or unicodedata.category(c).startswith(("P", "S")) or (ord(c) < 128 and not c.isalnum()):
                if cur: out.append("".join(cur)); cur = []
                out.append(c)
            elif unicodedata.category(c).startswith("C"):
                continue
            else:
                cur.append(c)
        if cur:
            out.append("".join(cur))
        return out

    def wordpiece(self, tok: str) -> list[int]:
        if len(tok) > 100:
            return [self.unk]
        return super().wordpiece(tok)


def load(tok_mode: str, dev: str):
    A.GGUF = Path(DEFAULT_GGUF)
    _, m = A.load_text(dev)
    tok = UnicodeTokenizer() if tok_mode == "unicode" else A.Tokenizer()
    return tok, m


def load_attn(path: str, dev: str):
    """A train.py attention-pool checkpoint as (module on dev, tmax)."""
    from train import AttnPool
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    h = AttnPool(384, c.get("heads", 1), c.get("hidden", 0)); h.load_state_dict(ck["state"]); h.eval()
    return h.to(dev), int(c["tmax"])


def embed(texts: list[str], tok, m, dev: str, tokens_per_batch: int = 16384, keep_tokens: bool = False, head=None):
    ids = [tok.encode(t) for t in texts]
    lens = np.array([len(x) for x in ids], dtype=np.int64)
    sem = np.zeros((len(ids), 1280), dtype=np.float16)
    hs = np.zeros(len(ids), dtype=np.float32)
    toks = [None] * len(ids) if keep_tokens else None
    order = np.argsort(-lens)
    i = 0
    while i < len(order):
        L = int(lens[order[i]])
        idx = order[i:i + max(1, tokens_per_batch // L)]
        x = torch.zeros(len(idx), L, dtype=torch.long)
        mk = torch.zeros(len(idx), L, dtype=torch.long)
        for j, k in enumerate(idx):
            x[j, :lens[k]] = torch.tensor(ids[k]); mk[j, :lens[k]] = 1
        hdn, s = m(x.to(dev), mk.to(dev))
        s = s.float().cpu().numpy()
        sem[idx] = s.astype(np.float16)
        if head is not None:
            hm, tmax = head
            with torch.no_grad():
                hs[idx] = hm(hdn[:, :tmax].float(), mk[:, :tmax].bool().to(dev)).float().cpu().numpy()
        if keep_tokens:
            hdn = hdn.float().cpu().numpy()
            for j, k in enumerate(idx):
                toks[k] = hdn[j, :lens[k]].astype(np.float16)
        i += len(idx)
    if head is not None:
        return sem, toks, lens, hs
    return sem, toks, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok", default="engine", choices=["engine", "unicode"])
    ap.add_argument("--tokens", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-bytes", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--head", default="", help="attention-pool checkpoint: also write head_scores.npy")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    texts = [json.loads(l)["text"] for l in open(a.inp)]
    ok = np.array([a.max_bytes == 0 or len(t.encode("utf-8")) <= a.max_bytes for t in texts])
    tok, m = load(a.tok, a.device)
    sys.path.insert(0, str(HERE))
    head = load_attn(a.head, a.device) if a.head else None
    scores = np.full(len(texts), -1e9, dtype=np.float32)
    sem = np.lib.format.open_memmap(out / "sem.npy", mode="w+", dtype=np.float16, shape=(len(texts), 1280))
    tok_parts, off = [], [0]
    t0 = time.time()
    idx_all = np.nonzero(ok)[0]
    for s in range(0, len(idx_all), a.chunk):
        idx = idx_all[s:s + a.chunk]
        r = embed([texts[i] for i in idx], tok, m, a.device, keep_tokens=a.tokens, head=head)
        e, tk = r[0], r[1]
        sem[idx] = e
        if head is not None:
            scores[idx] = r[3]
        if a.tokens:
            tok_parts.extend(tk)
        print(f"{s + len(idx)}/{len(idx_all)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    sem.flush()
    np.save(out / "embedded.npy", ok)
    if head is not None:
        np.save(out / "head_scores.npy", scores)
    if a.tokens:
        assert ok.all(), "--tokens needs every row embedded"
        for t in tok_parts:
            off.append(off[-1] + len(t))
        np.save(out / "tok_off.npy", np.array(off, dtype=np.int64))
        np.save(out / "tok.npy", np.concatenate(tok_parts))
    print("texts", len(texts), "embedded", int(ok.sum()), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
