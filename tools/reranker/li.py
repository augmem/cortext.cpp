"""Modality-agnostic late-interaction reranker (v1): train / evaluate over fixed AIST code sets.

Item (query or memory member, any modality) -> a set of vectors in one late-interaction (LI) space:
  - local positions: text = AIST token hidden states (query: all tokens; memory member: K k-means codes, codes.py),
    image = AIST 12x12 patch features (k-means codes), audio = frame features; one linear map per tower
  - global: the AIST shared-space retrieval vector (256-d slice the engine stores), one map for all modalities
Member score = weighted mean over query vectors of the max cosine to the member's vectors (MaxSim; query
weights from a learned linear gate). Memory score = soft-max over members. Final = scale*score + rank term.
usage: python -I li.py --train lme,eval1,dolphin,memlens --eval eval1,dolphin,lme,memlens --out runs/li1 [--k 16 --d 128]
"""
from __future__ import annotations

import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CAP, BUDGET, STAND_CAP = 4096, 34800, 2048
TOWER = {0: ("text", 384), 1: ("image", 960)}
Q_MAX = 128


class LI(nn.Module):
    def __init__(self, d=128, pca=None):
        super().__init__()
        self.cfg = {"d": d}
        self.maps = nn.ModuleDict({name: nn.Linear(dim, d, bias=False) for name, dim in TOWER.values()})
        self.glob = nn.Linear(256, d, bias=False)
        if pca is not None:
            with torch.no_grad():
                self.maps["text"].weight.copy_(torch.from_numpy(pca[:d]))
        with torch.no_grad():
            g = torch.zeros(d, 256)
            g[:, :min(d, 256)] = torch.eye(d, min(d, 256)) if d <= 256 else torch.eye(d, 256)
            self.glob.weight.copy_(g)
            nn.init.normal_(self.maps["image"].weight, std=0.02)
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.glob_bias = nn.Parameter(torch.zeros(()))   # log-weight of the query global vector
        self.scale = nn.Parameter(torch.tensor(20.0))
        self.member_tau = nn.Parameter(torch.tensor(-3.0))  # softmax-over-members temperature (log)
        self.rank_w = nn.Parameter(torch.tensor(0.0))

    def embed(self, x, mod):
        return F.normalize(self.maps[TOWER[mod][0]](x), dim=-1)

    def embed_glob(self, g):
        return F.normalize(self.glob(g), dim=-1)


def q8(x):
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127.0
    return x + (torch.round(x / s).clamp(-127, 127) * s - x).detach()


class DS:
    def __init__(self, d: Path, K: int, teacher="minilm"):
        self.name = d.name
        self.Q = [json.loads(l) for l in (d / "queries.jsonl").open()]
        items = [json.loads(l) for l in (d / "texts.jsonl").open()]
        self.clen = [len(t) if isinstance(t, str) else 0 for t in items]
        self.off = np.load(d / "tok_off.npy")
        self.tok = np.load(d / "tok.npy", mmap_mode="r")
        self.sem = np.load(d / "sem.npy", mmap_mode="r")
        self.mod = np.load(d / "modality.npy") if (d / "modality.npy").exists() else np.zeros(len(items), np.int64)
        self.ct = np.load(d / f"codes_k{K}_text.npy", mmap_mode="r")
        self.ci = np.load(d / f"codes_k{K}_image.npy", mmap_mode="r") if (d / f"codes_k{K}_image.npy").exists() else None
        self.cnt = np.load(d / f"codes_k{K}_cnt.npy")
        tp = d / f"teacher_{teacher}.json"
        self.teacher = json.loads(tp.read_text()) if tp.exists() else {}
        self.ifeat = np.load(d / "image_feat.npy", mmap_mode="r") if (d / "image_feat.npy").exists() else None
        self.ioff = np.load(d / "image_off.npy") if (d / "image_off.npy").exists() else None
        self.K = K

    def glob(self, i):
        g = np.asarray(self.sem[i][:256], dtype=np.float32)
        return g / max(1e-6, float(np.linalg.norm(g)))

    def query(self, i):
        """query positions (all) for item i -> (array, modality)"""
        m = int(self.mod[i])
        if m == 0:
            return np.asarray(self.tok[self.off[i]:self.off[i + 1]][:Q_MAX], dtype=np.float32), 0
        return np.asarray(self.ifeat[self.ioff[i]:self.ioff[i + 1]], dtype=np.float32), m

    def codes(self, i):
        m = int(self.mod[i])
        n = int(self.cnt[i])
        src = self.ct if m == 0 else self.ci
        return np.asarray(src[i, :n], dtype=np.float32), m


def member_vectors(model, ds_items, dev, quant=True):
    """[(ds, i)] -> padded member vector sets [U, K+1, d], mask [U, K+1] (last slot = global vector)."""
    U = len(ds_items)
    K = ds_items[0][0].K if U else 1
    vec = torch.zeros(U, K + 1, model.cfg["d"], device=dev)
    mask = torch.zeros(U, K + 1, dtype=torch.bool, device=dev)
    by_mod = {}
    G = np.zeros((U, 256), np.float32)
    for u, (ds, i) in enumerate(ds_items):
        c, m = ds.codes(i)
        by_mod.setdefault(m, []).append((u, c))
        G[u] = ds.glob(i)
    for m, lst in by_mod.items():
        X = np.zeros((len(lst), K, TOWER[m][1]), np.float32)
        Mk = np.zeros((len(lst), K), bool)
        for j, (_, c) in enumerate(lst):
            X[j, :len(c)] = c; Mk[j, :len(c)] = True
        e = model.embed(torch.from_numpy(X).to(dev), m)
        rows = torch.tensor([u for u, _ in lst], device=dev)
        vec[rows, :K] = e
        mask[rows, :K] = torch.from_numpy(Mk).to(dev)
    vec[:, K] = model.embed_glob(torch.from_numpy(G).to(dev))
    mask[:, K] = True
    if quant:
        vec = q8(vec)
    return vec, mask


def query_vectors(model, batch, dev):
    """[(ds, q)] -> [B, Lq+1, d], mask, weights (gate) ; last slot = query global vector."""
    B = len(batch)
    items = [ds.query(q["text_idx"]) for ds, q in batch]
    L = max(len(x) for x, _ in items)
    vec = torch.zeros(B, L + 1, model.cfg["d"], device=dev)
    mask = torch.zeros(B, L + 1, dtype=torch.bool, device=dev)
    for m in {m for _, m in items}:
        idx = [b for b, (_, mm) in enumerate(items) if mm == m]
        X = np.zeros((len(idx), L, TOWER[m][1]), np.float32)
        Mk = np.zeros((len(idx), L), bool)
        for j, b in enumerate(idx):
            x = items[b][0]
            X[j, :len(x)] = x; Mk[j, :len(x)] = True
        rows = torch.tensor(idx, device=dev)
        vec[rows, :L] = model.embed(torch.from_numpy(X).to(dev), m)
        mask[rows, :L] = torch.from_numpy(Mk).to(dev)
    G = np.stack([ds.glob(q["text_idx"]) for ds, q in batch])
    vec[:, L] = model.embed_glob(torch.from_numpy(G).to(dev))
    mask[:, L] = True
    logit = model.gate(vec).squeeze(-1)
    logit = torch.cat([logit[:, :L], logit[:, L:] + model.glob_bias], 1)
    w = torch.softmax(logit.masked_fill(~mask, -1e4), -1)
    return vec, w


def score_batch(model, batch, dev, n_max, quant=True):
    keys, pos = [], {}
    for ds, q in batch:
        for c in q["cands"][:n_max]:
            for i in c["members"][:8]:
                k = (id(ds), i)
                if k not in pos:
                    pos[k] = len(keys); keys.append((ds, i))
    mv, mm = member_vectors(model, keys, dev, quant) if keys else (None, None)
    qv, qw = query_vectors(model, batch, dev)
    out = []
    for b, (ds, q) in enumerate(batch):
        cands = q["cands"][:n_max]
        C = len(cands)
        idx = np.full((C, 8), -1, np.int64)
        for ci, c in enumerate(cands):
            for s, i in enumerate(c["members"][:8]):
                idx[ci, s] = pos[(id(ds), i)]
        valid = torch.from_numpy(idx >= 0).to(dev)
        if mv is None:
            out.append(torch.zeros(C, device=dev)); continue
        it = torch.from_numpy(np.maximum(idx, 0)).to(dev)
        V = mv[it.view(-1)].view(C, 8, *mv.shape[1:])            # [C, 8, K+1, d]
        Mk = mm[it.view(-1)].view(C, 8, mv.shape[1])               # [C, 8, K+1]
        sim = torch.einsum("qd,cskd->cskq", qv[b], V)              # [C, 8, K+1, Lq]
        sim = sim.masked_fill(~Mk[..., None], -2.0).amax(2)        # [C, 8, Lq]
        ms = (sim * qw[b][None, None]).sum(-1)                     # [C, 8]
        tau = model.member_tau.exp()
        ms = ms.masked_fill(~valid, -1e4)
        mem = tau * torch.logsumexp(ms / tau, 1)
        mem = torch.where(valid.any(1), mem, torch.full_like(mem, -1.0))
        rk = torch.log1p(torch.arange(C, device=dev, dtype=torch.float32)) / math.log(101.0)
        out.append(model.scale * mem + model.rank_w * rk)
    return out


def loss_fn(s, ds, q, kd_temp, w_gold, w_kd):
    loss = s.new_zeros(())
    t = ds.teacher.get(q["qid"])
    if t is not None and w_kd > 0:
        tv = torch.tensor([max(x) if x else -12.0 for x in t[:len(s)]], device=s.device)
        loss = loss + w_kd * F.kl_div(F.log_softmax(s, -1), F.softmax(tv / kd_temp, -1), reduction="sum")
    pos = [k for k, c in enumerate(q["cands"][:len(s)]) if c["units"]]
    if pos and w_gold > 0:
        loss = loss - w_gold * torch.logsumexp(F.log_softmax(s, -1)[pos], 0)
    return loss


def share(q, order, cut, kind=None):
    want = {u for c in q["cands"] for u in c["units"]} if kind is None else None
    got = set()
    for k in order[:cut]:
        got.update(q["cands"][k]["units"])
    if kind == "img":
        allu = {u for c in q["cands"] for u in c["units"] if str(u).startswith("img:")}
        return len({u for u in got if str(u).startswith("img:")}) / len(allu) if allu else float("nan")
    return len(got) / q["n_units"] if q["n_units"] else float("nan")


def budget_share(ds, q, order):
    used, got = min(q.get("standing_bytes", 0), STAND_CAP), set()
    for k in order:
        c = q["cands"][k]
        cost = min(sum(ds.clen[i] for i in c["members"]), CAP) + 8
        if used + cost > BUDGET:
            break
        used += cost
        got.update(c["units"])
    return len(got) / q["n_units"] if q["n_units"] else float("nan")


def rerank_order(scores, n_total, n):
    top = sorted(range(min(n, len(scores))), key=lambda k: (-scores[k], k))
    return top + list(range(len(top), n_total))


def evaluate(model, ds, dev, n_max, split="test"):
    rows = {}
    Q = [q for q in ds.Q if q["split"] == split and q["n_units"]]
    model.eval()
    with torch.no_grad():
        for b in range(0, len(Q), 16):
            batch = [(ds, q) for q in Q[b:b + 16]]
            for (_, q), s in zip(batch, score_batch(model, batch, dev, n_max)):
                n = len(q["cands"])
                orders = {"cortext": list(range(n)), "li": rerank_order(s.cpu().tolist(), n, n_max)}
                t = ds.teacher.get(q["qid"])
                if t is not None:
                    orders["teacher"] = rerank_order([max(x) if x else -1e9 for x in t], n, n_max)
                for k, o in orders.items():
                    rows.setdefault(k, []).append((share(q, o, 11), budget_share(ds, q, o), share(q, o, 11, "img")))
    model.train()
    out = {}
    for k, v in rows.items():
        img = [x[2] for x in v if x[2] == x[2]]
        out[k] = {"n": len(v), "top11": float(np.mean([x[0] for x in v])), "b34800": float(np.mean([x[1] for x in v])),
                  "img11": float(np.mean(img)) if img else None}
    return out


def pca_init(ds, d):
    rng = np.random.default_rng(0)
    x = np.asarray(ds.tok[np.sort(rng.integers(0, ds.tok.shape[0], 100000))], dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    return vt.astype(np.float32)


def train_items(D, names, exclude_conv_prefix):
    out = []
    for n in names:
        for q in D[n].Q:
            if q["split"] == "train" and not any(q["conv"].startswith(p) for p in exclude_conv_prefix):
                out.append((D[n], q))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/Volumes/Build/cortext/reranker/data")
    ap.add_argument("--train", default="lme,eval1,dolphin")
    ap.add_argument("--eval", default="eval1,dolphin,lme")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n-max", type=int, default=64)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--w-gold", type=float, default=1.0)
    ap.add_argument("--w-kd", type=float, default=1.0)
    ap.add_argument("--exclude-conv", default="locomo:", help="conv prefixes never trained on (license)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = sorted(set(filter(None, a.train.split(","))) | set(a.eval.split(",")))
    D = {n: DS(Path(a.data) / n, a.k) for n in names}
    model = LI(a.d, pca_init(D[names[0]], a.d)).to(a.device)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M", flush=True)
    ev = {n: evaluate(model, D[n], a.device, a.n_max) for n in a.eval.split(",")}
    print(json.dumps({"epoch": -1, "eval": ev}), flush=True)
    items = train_items(D, [n for n in a.train.split(",") if n], [p for p in a.exclude_conv.split(",") if p])
    print("train queries", len(items), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = max(1, a.epochs * math.ceil(len(items) / a.bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, steps // 20)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    log = (out / "log.jsonl").open("a")
    t0 = time.time()
    for ep in range(a.epochs):
        random.shuffle(items)
        tot = 0.0
        for b in range(0, len(items), a.bs):
            batch = items[b:b + a.bs]
            opt.zero_grad()
            ss = score_batch(model, batch, a.device, a.n_max)
            l = sum(loss_fn(s, ds, q, a.kd_temp, a.w_gold, a.w_kd) for s, (ds, q) in zip(ss, batch)) / len(batch)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(l.detach()) * len(batch)
        rec = {"epoch": ep, "loss": tot / max(1, len(items)), "t": round(time.time() - t0),
               "eval": {n: evaluate(model, D[n], a.device, a.n_max) for n in a.eval.split(",")}}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        torch.save({"cfg": model.cfg, "k": a.k, "state": model.state_dict(), "args": vars(a)}, out / "model.pt")


if __name__ == "__main__":
    main()
