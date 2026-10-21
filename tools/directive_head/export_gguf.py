"""Export a trained train.py head to a standalone GGUF (F32 tensors), following tools/reranker/export_gguf.py.

usage: python -I export_gguf.py --ckpt RUNS/attn_h4_m0/attn.pt --out /Volumes/Build/cortext/directive-head/AIST-87M-DH.gguf
       [--fixture /Volumes/Build/cortext/directive-head/dh_parity.json --data DATA]

Two kinds (metadata dh.kind):
  attn_text   attention pool over the AIST text tower's last hidden states (384-d per token, CLS..SEP, first
              dh.tmax positions), the same position features the late-interaction reranker reads:
                g = tokens @ dh.gate.weight^T + dh.gate.bias            [T, K]
                w = softmax over T of g (per head)                     [T, K]
                p = concat_k sum_t w[t,k] * tokens[t]                  [K*384]
                z = LayerNorm(p; dh.norm.weight, dh.norm.bias, eps 1e-5)
                logit = z @ dh.fc.weight^T + dh.fc.bias   (or fc2(gelu_erf(fc1(z))) when dh.hidden > 0)
  global      MLP / logistic over the AIST signal vector (dh.view "matryoshka256" = first 256 dims of the 1280-d
              vector re-normed, i.e. the stored vector; "full1280" = the full vector):
                logit = dh.fc2(gelu_erf(dh.fc1(x)))  or  dh.fc(x)
Common metadata: dh.threshold (logit; member iff logit >= threshold), dh.max_text_bytes (text signals longer than
this are not scored; the engine's kMaxTextBytes). A signal whose tower has no input for the kind (attn_text on
image/audio) is not scored and never joins the standing set.
The fixture holds inputs and expected logits for a later C++ parity test. Nothing here is committed: the GGUF
stays under /Volumes/Build/cortext/directive-head until integration, then ships sharded like AIST-87M-LI.
"""
import argparse, json, sys
from math import sqrt
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import AttnPool, Head, view256  # noqa: E402


def _gelu(h):
    from scipy.special import erf
    return 0.5 * h * (1 + erf(h / sqrt(2)))


def read(path: str):
    import gguf
    r = gguf.GGUFReader(path)
    t = {x.name: np.asarray(x.data, dtype=np.float32).reshape([int(s) for s in reversed(x.shape)]) for x in r.tensors}

    def field(k):
        f = r.fields[k]
        v = f.parts[f.data[0]]
        return bytes(v).decode() if f.types[0] == gguf.GGUFValueType.STRING else v[0].item()
    meta = {k: field(k) for k in r.fields if k.startswith("dh.")}
    return t, meta


def reference_attn(t, meta, tokens: np.ndarray) -> float:
    """Numpy forward pass of an attn_text GGUF on one signal's token states [T, 384] (C++ must match this)."""
    x = tokens[: meta["dh.tmax"]].astype(np.float32)
    g = x @ t["dh.gate.weight"].T + t["dh.gate.bias"]
    g = np.exp(g - g.max(0)); w = g / g.sum(0)
    p = (w.T @ x).reshape(-1)
    z = (p - p.mean()) / np.sqrt(p.var() + 1e-5) * t["dh.norm.weight"] + t["dh.norm.bias"]
    if meta["dh.hidden"]:
        return float(_gelu(z @ t["dh.fc1.weight"].T + t["dh.fc1.bias"]) @ t["dh.fc2.weight"].T + t["dh.fc2.bias"][0])
    return float(z @ t["dh.fc.weight"][0] + t["dh.fc.bias"][0])


def reference_global(t, meta, X: np.ndarray) -> np.ndarray:
    Z = view256(X) if meta["dh.view"] == "matryoshka256" else X.astype(np.float32)
    if meta["dh.hidden"]:
        return (_gelu(Z @ t["dh.fc1.weight"].T + t["dh.fc1.bias"]) @ t["dh.fc2.weight"].T).ravel() + t["dh.fc2.bias"][0]
    return (Z @ t["dh.fc.weight"].T).ravel() + t["dh.fc.bias"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-text-bytes", type=int, default=480)
    ap.add_argument("--fixture", default="")
    ap.add_argument("--data", default="", help="dataset dir with feat_engine/{sem,tok,tok_off}.npy for the fixture")
    a = ap.parse_args()
    import gguf
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, st = ck["cfg"], {k: v.float().numpy() for k, v in ck["state"].items()}
    attn = cfg["arm"] == "attn"
    w = gguf.GGUFWriter(a.out, "aist-dh")
    w.add_description("AIST standing-directive head: lasting directive/preference detector over AIST features")
    w.add_string("dh.kind", "attn_text" if attn else "global")
    w.add_float32("dh.threshold", float(ck["threshold"]))
    w.add_uint32("dh.max_text_bytes", int(a.max_text_bytes))
    w.add_uint32("dh.hidden", int(cfg.get("hidden", 0)))
    if attn:
        w.add_uint32("dh.heads", int(cfg.get("heads", 1)))
        w.add_uint32("dh.tmax", int(cfg["tmax"]))
        w.add_uint32("dh.token_dim", 384)
        names = ["gate.weight", "gate.bias", "norm.weight", "norm.bias"]
    else:
        w.add_uint32("dh.input_dim", int(cfg["dim"]))
        w.add_string("dh.view", "matryoshka256" if cfg["view"] == "256" else "full1280")
        names = []
    names += ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"] if cfg.get("hidden", 0) else ["fc.weight", "fc.bias"]
    for n in names:
        w.add_tensor(f"dh.{n}", np.ascontiguousarray(st[n].astype(np.float32)))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", a.out, Path(a.out).stat().st_size, "bytes", "params", sum(v.size for v in st.values()))
    if a.fixture:
        t, meta = read(a.out)
        fd = Path(a.data) / "feat_engine"
        rows = [json.loads(l) for l in (Path(a.data) / "rows.jsonl").open()]
        idx = np.sort(np.random.default_rng(0).choice(len(rows), 24, replace=False))
        fx = {"meta": meta, "items": []}
        if attn:
            tok = np.load(fd / "tok.npy", mmap_mode="r"); off = np.load(fd / "tok_off.npy")
            m = AttnPool(384, cfg.get("heads", 1), cfg.get("hidden", 0)); m.load_state_dict(ck["state"]); m.eval()
            errs = []
            for i in idx:
                x = np.asarray(tok[off[i]:off[i + 1]], dtype=np.float32)[: cfg["tmax"]]
                with torch.no_grad():
                    tl = float(m(torch.from_numpy(x[None]), torch.ones(1, len(x), dtype=torch.bool))[0])
                rl = reference_attn(t, meta, x); errs.append(abs(tl - rl))
                fx["items"].append({"text": rows[i]["text"], "tokens_384": x.tolist(), "logit": tl})
        else:
            X = np.load(fd / "sem.npy", mmap_mode="r")
            Xs = np.asarray(X[idx], dtype=np.float32)
            m = Head(cfg["dim"], cfg["hidden"]); m.load_state_dict(ck["state"]); m.eval()
            with torch.no_grad():
                tl = m(torch.from_numpy(view256(Xs) if cfg["view"] == "256" else Xs)).numpy()
            errs = list(np.abs(reference_global(t, meta, Xs) - tl))
            fx["items"] = [{"text": rows[i]["text"], "input_1280": x.tolist(), "logit": float(l)}
                           for i, x, l in zip(idx, Xs, tl)]
        fx["max_abs_err_gguf_vs_torch"] = float(max(errs))
        Path(a.fixture).write_text(json.dumps(fx))
        print("fixture", a.fixture, "gguf-vs-torch max |err|", fx["max_abs_err_gguf_vs_torch"])


if __name__ == "__main__":
    main()
