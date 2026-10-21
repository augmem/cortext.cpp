"""Train standing-directive heads on frozen AIST embeddings.

usage: python -I train.py --data /Volumes/Build/cortext/directive-head/data --feat feat_engine --out runs/engine
Arms (all on the L2-normed AIST semantic vector, so every modality has an input):
  lr1280  logistic regression on the 1280-d vector
  lr256   logistic regression on the stored 256-d Matryoshka view (first 256 dims, re-normed)
  mlp1280 1280 -> H -> 1 (GELU), weight decay, early stop on validation PR-AUC
  mlp256  256 -> H -> 1
  attn    (only with --tokens) attention pool over the 384-d token states -> linear; needs per-token features
Each arm's threshold is chosen on the validation split's English rows: the smallest threshold whose precision
is >= the engine rule's precision and whose false-positive rate is <= the rule's on those rows (pick_threshold).
Writes OUT/<arm>.pt ({state, cfg, threshold, val metrics}) and OUT/summary.json.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rules  # noqa: E402


def view256(x: np.ndarray) -> np.ndarray:
    v = x[:, :256].astype(np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class Head(torch.nn.Module):
    """Logistic (hidden=0) or one-hidden-layer MLP over an L2-normed embedding."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        if hidden:
            self.fc1 = torch.nn.Linear(d, hidden)
            self.fc2 = torch.nn.Linear(hidden, 1)
        else:
            self.fc = torch.nn.Linear(d, 1)

    def forward(self, x):
        if self.hidden:
            return self.fc2(F.gelu(self.fc1(x))).squeeze(-1)
        return self.fc(x).squeeze(-1)


class AttnPool(torch.nn.Module):
    """K gate heads over token states (384) -> K softmax pools -> LayerNorm -> (MLP H | linear) -> logit.
    Cost O(T * 384 * K) + O(K * 384 * H) per signal."""

    def __init__(self, d: int = 384, heads: int = 1, hidden: int = 0):
        super().__init__()
        self.heads, self.hidden = heads, hidden
        self.gate = torch.nn.Linear(d, heads)
        self.norm = torch.nn.LayerNorm(d * heads)
        if hidden:
            self.fc1 = torch.nn.Linear(d * heads, hidden)
            self.fc2 = torch.nn.Linear(hidden, 1)
        else:
            self.fc = torch.nn.Linear(d * heads, 1)

    def forward(self, tok, mask):
        g = self.gate(tok).masked_fill(~mask[..., None], -1e9)        # [B, T, K]
        w = torch.softmax(g, 1)
        pooled = torch.einsum("btk,btd->bkd", w, tok).flatten(1)       # [B, K*d]
        z = self.norm(pooled)
        if self.hidden:
            return self.fc2(F.gelu(self.fc1(z))).squeeze(-1)
        return self.fc(z).squeeze(-1)


def fit(model, Xtr, ytr, Xva, yva, epochs, lr, wd, batch=512, seed=0, pos_weight=1.0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    Xva_t = torch.from_numpy(Xva)
    best, best_state, bad = -1.0, None, 0
    pw = torch.tensor(pos_weight)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for s in range(0, len(perm), batch):
            i = perm[s:s + batch]
            loss = F.binary_cross_entropy_with_logits(model(Xtr_t[i]), ytr_t[i], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            ap = average_precision_score(yva, model(Xva_t).numpy())
        if ap > best + 1e-4:
            best, best_state, bad = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(best_state)
    return model, best


def pick_threshold(scores: np.ndarray, y: np.ndarray, rule_precision: float, rule_fpr: float) -> float:
    """Smallest threshold (most recall) on English validation rows whose precision is >= the rule's AND whose
    false-positive rate is <= the rule's. Matching FPR as well as precision keeps the head at least as precise
    as the rule at any base rate (precision depends on the prior only through TPR/FPR), which matters because
    real streams hold far fewer directives than the balanced validation split."""
    best = float(scores.max()) + 1e-6
    for t in np.unique(scores)[::-1]:
        pred = scores >= t
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
        if tp + fp == 0:
            continue
        if tp / (tp + fp) >= rule_precision and fp / max(1, (y == 0).sum()) <= rule_fpr:
            best = float(t)
    return best


def prf(pred: np.ndarray, y: np.ndarray):
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--feat", default="feat_engine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--arms", default="lr1280,lr256,mlp1280,mlp256")
    ap.add_argument("--tokens", action="store_true")
    ap.add_argument("--attn-heads", type=int, default=1)
    ap.add_argument("--attn-hidden", type=int, default=0)
    ap.add_argument("--tmax", type=int, default=128)
    ap.add_argument("--pos-weight", type=float, default=1.0, help="BCE positive weight for the attention arm")
    a = ap.parse_args()
    d = Path(a.data)
    rows = [json.loads(l) for l in (d / "rows.jsonl").open()]
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    split = np.array([r["split"] for r in rows])
    lang = np.array([r["lang"] for r in rows])
    X = np.load(d / a.feat / "sem.npy").astype(np.float32)
    tr, va = split == "train", split == "val"
    va_en = va & (lang == "en")
    rule_va_en = np.array([rules.engine_rule(rows[i]["text"]) for i in np.nonzero(va_en)[0]])
    rule_prec = prf(rule_va_en, y[va_en])["p"]
    rule_fpr = float(rule_va_en[y[va_en] == 0].mean())
    print(f"train {tr.sum()} val {va.sum()} (en {va_en.sum()}); engine rule val-en precision {rule_prec:.3f} "
          f"fpr {rule_fpr:.4f}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {"rule_val_en_precision": rule_prec, "rule_val_en_fpr": rule_fpr, "arms": {}}
    for arm in a.arms.split(","):
        if arm == "attn":
            continue
        Z = view256(X) if arm.endswith("256") else X
        hidden = a.hidden if arm.startswith("mlp") else 0
        best = None
        for wd in ([1e-4, 1e-3, 1e-2] if hidden else [0.0, 1e-4, 1e-3]):
            for lr in ([1e-3, 3e-3] if hidden else [3e-3, 1e-2]):
                m, ap_va = fit(Head(Z.shape[1], hidden), Z[tr], y[tr], Z[va], y[va], epochs=200 if not hidden else 120,
                               lr=lr, wd=wd)
                if best is None or ap_va > best[1]:
                    best = (m, ap_va, {"wd": wd, "lr": lr})
        m, ap_va, hp = best
        with torch.no_grad():
            s_va = m(torch.from_numpy(Z[va])).numpy()
        # threshold on English validation rows: keep the rule's precision, maximize recall
        en_mask = lang[va] == "en"
        thr = pick_threshold(s_va[en_mask], y[va][en_mask], rule_prec, rule_fpr)
        m_va = prf(s_va >= thr, y[va]); m_va_en = prf(s_va[en_mask] >= thr, y[va][en_mask])
        cfg = {"arm": arm, "dim": int(Z.shape[1]), "hidden": hidden, "view": "256" if arm.endswith("256") else "1280",
               "hp": hp, "feat": a.feat}
        torch.save({"state": m.state_dict(), "cfg": cfg, "threshold": thr}, out / f"{arm}.pt")
        summary["arms"][arm] = {"cfg": cfg, "threshold": thr, "val_pr_auc": ap_va, "val": m_va, "val_en": m_va_en}
        print(arm, json.dumps(summary["arms"][arm]), flush=True)
    if a.tokens and "attn" in a.arms:
        tok = np.load(d / a.feat / "tok.npy", mmap_mode="r")
        off = np.load(d / a.feat / "tok_off.npy")
        Tmax = a.tmax

        def pack(idx):
            T = np.zeros((len(idx), Tmax, 384), dtype=np.float32); M = np.zeros((len(idx), Tmax), dtype=bool)
            for j, i in enumerate(idx):
                t = np.asarray(tok[off[i]:off[i + 1]][:Tmax], dtype=np.float32)
                T[j, :len(t)] = t; M[j, :len(t)] = True
            return torch.from_numpy(T), torch.from_numpy(M)

        tri, vai = np.nonzero(tr)[0], np.nonzero(va)[0]
        Tva, Mva = pack(vai)
        torch.manual_seed(0)
        m = AttnPool(384, a.attn_heads, a.attn_hidden); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-3)
        best, best_state, bad = -1, None, 0
        for ep in range(60):
            m.train(); perm = np.random.default_rng(ep).permutation(tri)
            for s in range(0, len(perm), 256):
                i = perm[s:s + 256]; T, M = pack(i)
                loss = F.binary_cross_entropy_with_logits(m(T, M), torch.from_numpy(y[i].astype(np.float32)),
                                                          pos_weight=torch.tensor(a.pos_weight))
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                s_va = m(Tva, Mva).numpy()
            apv = average_precision_score(y[va], s_va)
            if apv > best + 1e-4:
                best, best_state, bad = apv, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 6:
                    break
        m.load_state_dict(best_state)
        with torch.no_grad():
            s_va = m(Tva, Mva).numpy()
        en_mask = lang[va] == "en"
        thr = pick_threshold(s_va[en_mask], y[va][en_mask], rule_prec, rule_fpr)
        torch.save({"state": m.state_dict(), "cfg": {"arm": "attn", "tmax": Tmax, "feat": a.feat, "heads": a.attn_heads,
                                                     "hidden": a.attn_hidden, "pos_weight": a.pos_weight}, "threshold": thr},
                   out / "attn.pt")
        summary["arms"]["attn"] = {"threshold": thr, "val_pr_auc": best, "val": prf(s_va >= thr, y[va]),
                                   "val_en": prf(s_va[en_mask] >= thr, y[va][en_mask])}
        print("attn", json.dumps(summary["arms"]["attn"]), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
