"""PyTorch mirror of the frozen AIST-87M towers, loaded from the shipped GGUF (feature source for reranker training).

Text tower: byte-level BERT basic tokenizer + greedy WordPiece identical to src/models/aist_gguf_encoder.cpp
(CLS ... SEP, truncation at context_length), 6-layer BERT (384-d, 12 heads), mean pool over all positions,
text_encoder.2.linear (384->768), text_projection residual MLP (768->1280), L2 norm.
`text_tokens` returns the last hidden states (the per-token features the reranker consumes) and the
1280-d semantic vector.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
GGUF = ROOT / "models/AIST-87M-GGUF/AIST-87M_q8_0.gguf"
VOCAB = ROOT / "models/mdbr-leaf-ir/vocab.txt"


def load_gguf(path: Path, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    import gguf
    from gguf.quants import dequantize

    r = gguf.GGUFReader(str(path))
    out = {}
    for t in r.tensors:
        if not t.name.startswith(prefixes):
            continue
        a = dequantize(t.data, t.tensor_type).astype(np.float32)
        shape = [int(x) for x in reversed(t.shape)]  # gguf stores ne0 first
        out[t.name] = torch.from_numpy(np.ascontiguousarray(a.reshape(shape)))
    return out


class Tokenizer:
    def __init__(self, vocab: Path = VOCAB, max_length: int = 512):
        toks = vocab.read_text(encoding="utf-8").split("\n")
        if toks and toks[-1] == "":
            toks.pop()
        self.v = {t: i for i, t in enumerate(toks)}
        self.max_length = max_length
        self.cls, self.sep, self.unk = self.v["[CLS]"], self.v["[SEP]"], self.v["[UNK]"]

    @staticmethod
    def basic(text: str) -> list[str]:
        out, cur = [], bytearray()
        for ch in text.encode("utf-8"):
            c = chr(ch)
            if ch < 128 and c.isalnum():
                cur.append(ord(c.lower()))
            elif ch < 128 and c in " \t\n\v\f\r":
                if cur:
                    out.append(cur.decode("latin-1")); cur = bytearray()
            else:
                if cur:
                    out.append(cur.decode("latin-1")); cur = bytearray()
                # a lone byte >= 0x80 is not valid UTF-8 and never matches the vocabulary -> [UNK]
                out.append(chr(ch) if ch < 128 else "\x00")
        if cur:
            out.append(cur.decode("latin-1"))
        return out

    def wordpiece(self, tok: str) -> list[int]:
        if len(tok) > 100 or tok == "\x00":
            return [self.unk]
        ids, start = [], 0
        while start < len(tok):
            end, best = len(tok), -1
            while start < end:
                p = tok[start:end] if start == 0 else "##" + tok[start:end]
                if p in self.v:
                    best = self.v[p]
                    break
                end -= 1
            if best < 0:
                return [self.unk]
            ids.append(best)
            start = end
        return ids or [self.unk]

    def encode(self, text: str) -> list[int]:
        ids = [self.cls]
        for t in self.basic(text):
            for i in self.wordpiece(t):
                if len(ids) >= self.max_length - 1:
                    ids.append(self.sep)
                    return ids
                ids.append(i)
        ids.append(self.sep)
        return ids


class AistText(torch.nn.Module):
    def __init__(self, w: dict[str, torch.Tensor]):
        super().__init__()
        self.w = {k: v for k, v in w.items()}
        self.heads = 12

    def to(self, dev):  # noqa: D401
        self.w = {k: v.to(dev) for k, v in self.w.items()}
        return self

    def lin(self, x, name):
        return F.linear(x, self.w[name + ".weight"], self.w[name + ".bias"])

    def ln(self, x, name, eps):
        return F.layer_norm(x, x.shape[-1:], self.w[name + ".weight"], self.w[name + ".bias"], eps)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        """ids/mask [B, T] (right padded) -> hidden [B, T, 384], semantic [B, 1280] (L2 normed)."""
        p = "text_encoder.0.model."
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = (self.w[p + "embeddings.word_embeddings.weight"][ids]
             + self.w[p + "embeddings.position_embeddings.weight"][pos][None]
             + self.w[p + "embeddings.token_type_embeddings.weight"][0][None, None])
        x = self.ln(x, p + "embeddings.LayerNorm", 1e-12)
        bias = (1.0 - mask[:, None, None, :].float()) * -1e9
        H = self.heads
        for L in range(6):
            q = f"{p}encoder.layer.{L}."
            Q = self.lin(x, q + "attention.self.query").view(B, T, H, -1).transpose(1, 2)
            K = self.lin(x, q + "attention.self.key").view(B, T, H, -1).transpose(1, 2)
            V = self.lin(x, q + "attention.self.value").view(B, T, H, -1).transpose(1, 2)
            a = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1]) + bias, -1)
            ctx = (a @ V).transpose(1, 2).reshape(B, T, -1)
            x = self.ln(x + self.lin(ctx, q + "attention.output.dense"), q + "attention.output.LayerNorm", 1e-12)
            h = F.gelu(self.lin(x, q + "intermediate.dense"))
            x = self.ln(x + self.lin(h, q + "output.dense"), q + "output.LayerNorm", 1e-12)
        m = mask.float()[..., None]
        pooled = (x * m).sum(1) / m.sum(1)
        f = self.lin(pooled, "text_encoder.2.linear")
        y = self.ln(F.gelu(self.lin(f, "text_projection.expand.0")), "text_projection.expand.2", 1e-5)
        b = 0
        while f"text_projection.residual_blocks.{b}.0.weight" in self.w:
            r = f"text_projection.residual_blocks.{b}."
            y = y + self.ln(F.gelu(self.lin(y, r + "0")), r + "2", 1e-5)
            b += 1
        s = F.normalize(self.lin(y, "text_projection.project"), dim=-1)
        return x, s


def load_text(dev: str = "cpu", max_length: int = 512):
    tok = Tokenizer(max_length=max_length)
    m = AistText(load_gguf(GGUF, ("text_encoder.", "text_projection."))).to(dev)
    return tok, m


def batch_ids(tok: Tokenizer, texts: list[str]):
    ids = [tok.encode(t) for t in texts]
    T = max(map(len, ids))
    a = torch.zeros(len(ids), T, dtype=torch.long)
    m = torch.zeros(len(ids), T, dtype=torch.long)
    for i, x in enumerate(ids):
        a[i, :len(x)] = torch.tensor(x)
        m[i, :len(x)] = 1
    return a, m


# ---------------------------------------------------------------- image tower
def _cubic(x, a=-0.5):
    x = np.abs(x)
    return np.where(x < 1, (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1,
                    np.where(x < 2, a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a, 0.0))


def _resample_matrix(n_in, n_out):
    scale = n_in / n_out
    f = (np.arange(n_out) + 0.5) * scale - 0.5
    i0 = np.floor(f).astype(np.int64)
    W = np.zeros((n_out, n_in), dtype=np.float64)
    for k in range(-1, 3):
        idx = np.clip(i0 + k, 0, n_in - 1)
        w = _cubic(f - (i0 + k))
        np.add.at(W, (np.arange(n_out), idx), w)
    return W / W.sum(1, keepdims=True)


def preprocess_image(rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWxC -> float32 3x384x384, engine PreprocessAistImageHWC (bicubic a=-0.5, edge clamp, ImageNet norm)."""
    h, w = rgb.shape[:2]
    x = rgb[..., :3].astype(np.float64)
    Wy, Wx = _resample_matrix(h, 384), _resample_matrix(w, 384)
    y = (Wy @ x.reshape(h, w * 3)).reshape(384, w, 3)                       # rows (BLAS)
    y = (Wx @ y.transpose(1, 0, 2).reshape(w, 384 * 3)).reshape(384, 384, 3).transpose(1, 0, 2)  # cols
    y = np.clip(y / 255.0, 0, 1)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    return ((y - mean) / std).transpose(2, 0, 1).astype(np.float32)


class AistImage(torch.nn.Module):
    """timm mobilenetv4_conv_medium with the GGUF weights; returns patch features [B, 144, 960] + semantic [B, 1280]."""

    def __init__(self, w: dict[str, torch.Tensor]):
        super().__init__()
        import timm
        self.net = timm.create_model("mobilenetv4_conv_medium", pretrained=False, num_classes=0)
        sd = {k[len("image_encoder."):]: v for k, v in w.items() if k.startswith("image_encoder.")}
        sd = {k: (v.reshape(()) if k.endswith("num_batches_tracked") else v) for k, v in sd.items()}
        sd = {k: v.long() if k.endswith("num_batches_tracked") else v for k, v in sd.items()}
        self.net.load_state_dict(sd)
        self.net.eval()
        self.proj = {k: v for k, v in w.items() if k.startswith("image_projection.")}

    def to(self, dev):
        self.net.to(dev)
        self.proj = {k: v.to(dev) for k, v in self.proj.items()}
        return self

    def lin(self, x, name):
        return F.linear(x, self.proj[name + ".weight"], self.proj[name + ".bias"])

    def ln(self, x, name, eps=1e-5):
        return F.layer_norm(x, x.shape[-1:], self.proj[name + ".weight"], self.proj[name + ".bias"], eps)

    @torch.no_grad()
    def forward(self, x):
        f = self.net.forward_features(x)  # [B, 960, 12, 12]
        patches = f.flatten(2).transpose(1, 2)
        pooled = self.net.forward_head(f)  # global pool, conv_head, norm_head(+act), flatten -> [B, 1280]
        y = self.ln(F.gelu(self.lin(pooled, "image_projection.expand.0")), "image_projection.expand.2")
        b = 0
        while f"image_projection.residual_blocks.{b}.0.weight" in self.proj:
            r = f"image_projection.residual_blocks.{b}."
            y = y + self.ln(F.gelu(self.lin(y, r + "0")), r + "2")
            b += 1
        s = F.normalize(self.lin(y, "image_projection.project"), dim=-1)
        return patches, s


def load_image(dev: str = "cpu"):
    return AistImage(load_gguf(GGUF, ("image_encoder.", "image_projection."))).to(dev)


def read_rgb(path: str, max_side: int = 768) -> np.ndarray:
    """Decode an image file the way the eval ingest does (PIL draft + RGB + bicubic thumbnail to max_side)."""
    from PIL import Image
    im = Image.open(path)
    im.draft("RGB", (max_side, max_side))
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


# ---------------------------------------------------------------- token / patch features in the shared space
def text_tokens_shared(m: AistText, hidden: torch.Tensor) -> torch.Tensor:
    """Per-token hidden states [.., 384] -> shared-space vectors [.., 1280] (pooling-free text head)."""
    f = m.lin(hidden, "text_encoder.2.linear")
    y = m.ln(F.gelu(m.lin(f, "text_projection.expand.0")), "text_projection.expand.2", 1e-5)
    b = 0
    while f"text_projection.residual_blocks.{b}.0.weight" in m.w:
        r = f"text_projection.residual_blocks.{b}."
        y = y + m.ln(F.gelu(m.lin(y, r + "0")), r + "2", 1e-5)
        b += 1
    return m.lin(y, "text_projection.project")


def image_patches_shared(m: AistImage, patches: torch.Tensor) -> torch.Tensor:
    """Per-patch blocks.4 features [.., 960] -> shared-space vectors [.., 1280] (conv_head + norm_head + projection per patch)."""
    net = m.net
    w = net.conv_head.weight.view(net.conv_head.weight.shape[0], -1)
    x = F.linear(patches, w, net.conv_head.bias)
    bn = net.norm_head
    x = (x - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps) * bn.weight + bn.bias
    x = bn.act(x)
    y = m.ln(F.gelu(m.lin(x, "image_projection.expand.0")), "image_projection.expand.2")
    b = 0
    while f"image_projection.residual_blocks.{b}.0.weight" in m.proj:
        r = f"image_projection.residual_blocks.{b}."
        y = y + m.ln(F.gelu(m.lin(y, r + "0")), r + "2")
        b += 1
    return m.lin(y, "image_projection.project")
