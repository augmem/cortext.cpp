"""Offline evaluation of the standing-directive heads against the English keyword rules.

usage: python -I evaluate.py --data DATA --runs RUNS --streams STREAMS --modality MOD --out RUNS/eval.json [--arm mlp256]
Reports
  test     held-out test split (template-held and topic-held), overall / per language / per hold / per category:
           precision, recall, F1 at the arm's threshold, PR-AUC; engine_rule and brief_rule as baselines.
  streams  every turn of the replay streams (evaluation only): flags per stream for rule and head, agreement,
           Cohen's kappa; writes disagreements.jsonl for judge.py. The head sees text <= --max-bytes (default 480,
           the engine's kMaxTextBytes gate, which the rule uses too).
  beam     recall of BEAM instruction_following / preference_following gold user turns.
  modality fraction of engine image / audio / speech embeddings the head flags, and score quantiles.
"""
from __future__ import annotations

import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rules  # noqa: E402
from train import AttnPool, Head, prf, view256  # noqa: E402


def attn_scores(ckpt: Path, feat_dir: Path, rows_idx: np.ndarray):
    """Attention-pool head scores for dataset rows from stored token states (tok.npy / tok_off.npy)."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = AttnPool(384, c.get("heads", 1), c.get("hidden", 0)); m.load_state_dict(ck["state"]); m.eval()
    tok = np.load(feat_dir / "tok.npy", mmap_mode="r"); off = np.load(feat_dir / "tok_off.npy")
    T = c["tmax"]; out = []
    for s in range(0, len(rows_idx), 1024):
        idx = rows_idx[s:s + 1024]
        Xb = np.zeros((len(idx), T, 384), dtype=np.float32); Mb = np.zeros((len(idx), T), dtype=bool)
        for j, i in enumerate(idx):
            t = np.asarray(tok[off[i]:off[i + 1]][:T], dtype=np.float32); Xb[j, :len(t)] = t; Mb[j, :len(t)] = True
        with torch.no_grad():
            out.append(m(torch.from_numpy(Xb), torch.from_numpy(Mb)).numpy())
    return np.concatenate(out), ck["threshold"]


def load_head(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = Head(cfg["dim"], cfg["hidden"]); m.load_state_dict(ck["state"]); m.eval()

    def score(X: np.ndarray) -> np.ndarray:
        Z = view256(X) if cfg["view"] == "256" else X.astype(np.float32)
        out = []
        with torch.no_grad():
            for s in range(0, len(Z), 65536):
                out.append(m(torch.from_numpy(np.ascontiguousarray(Z[s:s + 65536]))).numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    return score, ck["threshold"], cfg


def unk_fraction(texts: list[str]) -> np.ndarray:
    """Share of [UNK] among the engine tokenizer's content tokens (CLS/SEP excluded). Non-Latin scripts are all
    [UNK] under the engine tokenizer, so this is the cheap 'the encoder cannot read this text' signal."""
    sys.path.insert(0, str(HERE.parent / "reranker"))
    import aist_torch as A
    tok = A.Tokenizer()
    out = np.zeros(len(texts), dtype=np.float32)
    for i, t in enumerate(texts):
        ids = tok.encode(t)[1:-1]
        out[i] = sum(1 for x in ids if x == tok.unk) / max(1, len(ids))
    return out


def block(pred, y, score=None):
    r = prf(pred, y)
    r["n"] = int(len(y)); r["pos"] = int(y.sum())
    if score is not None and 0 < y.sum() < len(y):
        r["pr_auc"] = float(average_precision_score(y, score))
    return r


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--feat", default="feat_engine")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--arms", default="lr1280,lr256,mlp1280,mlp256",
                    help="comma list of name=path/to/ckpt.pt (attention arms: name starts with attn and use --feat tokens)")
    ap.add_argument("--arm", default="mlp256", help="arm used for streams / beam / modality")
    ap.add_argument("--streams", default="")
    ap.add_argument("--stream-feat", default="feat_engine")
    ap.add_argument("--modality", default="")
    ap.add_argument("--max-bytes", type=int, default=480)
    ap.add_argument("--unk-gate", type=float, default=0.5, help="+unkgate arms skip text whose [UNK] share exceeds this")
    ap.add_argument("--stream-unkgate", action="store_true")
    ap.add_argument("--modality-arm", default="mlp256", help="pooled-vector arm scored on image/audio embeddings")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d, runs = Path(a.data), Path(a.runs)
    rows = [json.loads(l) for l in (d / "rows.jsonl").open()]
    X = np.load(d / a.feat / "sem.npy").astype(np.float32)
    te = np.array([r["split"] == "test" for r in rows])
    R = [rows[i] for i in np.nonzero(te)[0]]
    y = np.array([r["label"] for r in R])
    lang = np.array([r["lang"] for r in R]); hold = np.array([r["hold"] for r in R]); cat = np.array([r["cat"] for r in R])
    wrapd = np.array([r["wrap"] != "none" for r in R])
    src = np.array([r.get("src", "synthetic") for r in R])
    preds = {"engine_rule": np.array([rules.engine_rule(r["text"]) for r in R]),
             "brief_rule": np.array([rules.brief_rule(r["text"]) for r in R])}
    scores = {}
    arm_paths = {}
    for spec_ in a.arms.split(","):
        name, _, p = spec_.partition("=")
        arm_paths[name] = Path(p) if p else runs / f"{name}.pt"
    for arm, p in arm_paths.items():
        if not p.exists():
            print("missing", p, file=sys.stderr); continue
        if arm.startswith("attn"):
            fd = d / ("feat_unicode" if "unicode" in arm else a.feat)
            scores[arm], thr = attn_scores(p, fd, np.nonzero(te)[0])
        else:
            sc, thr, _ = load_head(p)
            scores[arm] = sc(X[te])
        preds[arm] = scores[arm] >= thr
    unk = unk_fraction([r["text"] for r in R])
    for arm in list(preds):
        if arm.startswith("attn"):
            preds[arm + "+unkgate"] = preds[arm] & (unk <= a.unk_gate)
            scores[arm + "+unkgate"] = np.where(unk <= a.unk_gate, scores[arm], -1e9)
    rep = {"test": {}, "unk_gate": a.unk_gate}
    for name, p in preds.items():
        s = scores.get(name)
        sub = lambda m: block(p[m], y[m], None if s is None else s[m])  # noqa: E731
        t = {"overall": sub(np.ones(len(y), bool)),
             "by_lang": {l: sub(lang == l) for l in sorted(set(lang))},
             "by_hold": {h: sub(hold == h) for h in sorted(set(hold))},
             "wrapped": sub(wrapd), "unwrapped": sub(~wrapd),
             "by_cat": {},
             "by_src": {sname: {"overall": sub(src == sname),
                                "by_lang": {l: sub((src == sname) & (lang == l))
                                            for l in sorted(set(lang[src == sname]))}}
                        for sname in sorted(set(src))}}
        for c in sorted(set(cat)):
            m = cat == c
            t["by_cat"][c] = {"n": int(m.sum()), "flag_rate": float(p[m].mean())}
        rep["test"][name] = t
    # ---------------------------------------------------------------- streams
    if a.streams:
        sd = Path(a.streams)
        texts = [json.loads(l)["text"] for l in (sd / "texts.jsonl").open()]
        nbytes = np.array([len(t.encode("utf-8")) for t in texts])
        rule_u = np.array([rules.engine_rule(t) for t in texts])
        emb = np.load(sd / a.stream_feat / "embedded.npy")
        gate = emb & (nbytes <= a.max_bytes)
        head_s = np.full(len(texts), -1e9, dtype=np.float32)
        arm_p = arm_paths.get(a.arm, runs / f"{a.arm}.pt")
        if a.arm.startswith("attn"):
            thr = torch.load(arm_p, map_location="cpu", weights_only=False)["threshold"]
            head_s[gate] = np.load(sd / a.stream_feat / "head_scores.npy")[gate]
        else:
            S = np.load(sd / a.stream_feat / "sem.npy", mmap_mode="r")
            sc, thr, cfg = load_head(arm_p)
            idx = np.nonzero(gate)[0]
            for s in range(0, len(idx), 200000):
                j = idx[s:s + 200000]
                head_s[j] = sc(np.asarray(S[j], dtype=np.float32))
        head_u = head_s >= thr
        if a.arm.startswith("attn") and a.stream_unkgate:
            cand = np.nonzero(head_u)[0]
            uf = unk_fraction([texts[i] for i in cand])
            head_u[cand[uf > a.unk_gate]] = False
        turns = [json.loads(l) for l in (sd / "turns.jsonl").open()]
        per = defaultdict(lambda: Counter())
        for t in turns:
            u = t["uid"]; c = per[t["stream"]]
            c["turns"] += 1; c["rule"] += int(rule_u[u]); c["head"] += int(head_u[u])
            c["both"] += int(rule_u[u] and head_u[u]); c["rule_only"] += int(rule_u[u] and not head_u[u])
            c["head_only"] += int(head_u[u] and not rule_u[u]); c["gated_in"] += int(gate[u])
        st = {}
        for k, c in per.items():
            st[k] = dict(c)
        ua = np.array([t["uid"] for t in turns])
        st["_all"] = {"turns": len(turns), "rule": int(rule_u[ua].sum()), "head": int(head_u[ua].sum()),
                      "both": int((rule_u & head_u)[ua].sum()), "kappa": kappa(rule_u[ua], head_u[ua]),
                      "unique_texts": len(texts)}
        rep["streams"] = {"arm": a.arm, "stream_feat": a.stream_feat, "threshold": thr, "max_bytes": a.max_bytes, "per_stream": st}
        stream_of = {}
        for t in turns:
            stream_of.setdefault(t["uid"], t["stream"])
        with (Path(a.out).parent / "disagreements.jsonl").open("w") as f:
            for u in np.nonzero(rule_u != head_u)[0]:
                f.write(json.dumps({"uid": int(u), "stream": stream_of.get(int(u)), "rule": bool(rule_u[u]),
                                    "head": bool(head_u[u]), "score": float(head_s[u]), "text": texts[u]},
                                   ensure_ascii=False) + "\n")
        with (Path(a.out).parent / "rule_flags.jsonl").open("w") as f:
            for u in np.nonzero(rule_u)[0]:
                f.write(json.dumps({"uid": int(u), "stream": stream_of.get(int(u)), "rule": True,
                                    "head": bool(head_u[u]), "score": float(head_s[u]), "text": texts[u]},
                                   ensure_ascii=False) + "\n")
        # BEAM gold
        uid_of = {}
        for t in turns:
            uid_of[(t["stream"], t["source_id"])] = t["uid"]
        gold = [json.loads(l) for l in (sd / "beam_gold.jsonl").open()]
        bg = {}
        for cat_name in ("instruction_following", "preference_following"):
            us = sorted({uid_of[(g["stream"], g["source_id"])] for g in gold if g["category"] == cat_name
                         and (g["stream"], g["source_id"]) in uid_of})
            us = np.array(us)
            bg[cat_name] = {"gold_turns": len(us), "rule_recall": float(rule_u[us].mean()),
                            "head_recall": float(head_u[us].mean()),
                            "either": float((rule_u | head_u)[us].mean())}
        rep["beam_gold"] = bg
        # BEAM context: what fraction of all flags land on gold instruction turns
        gi = {uid_of[(g["stream"], g["source_id"])] for g in gold if g["category"] == "instruction_following"
              and (g["stream"], g["source_id"]) in uid_of}
        beam_u = np.array(sorted({t["uid"] for t in turns if t["stream"].startswith("beam")}))
        rep["beam_gold"]["flags_in_beam"] = {"rule": int(rule_u[beam_u].sum()), "head": int(head_u[beam_u].sum()),
                                             "gold_instruction_turns": len(gi)}
    # ---------------------------------------------------------------- modality
    if a.modality:
        sc, thr, _ = load_head(arm_paths.get(a.modality_arm, runs / f"{a.modality_arm}.pt"))
        mo = {}
        for k in ("image", "audio", "speech"):
            p = Path(a.modality) / f"{k}.npy"
            if p.exists():
                s = sc(np.load(p))
                mo[k] = {"n": int(len(s)), "flag_rate": float((s >= thr).mean()),
                         "score_q": [float(np.quantile(s, q)) for q in (0.5, 0.9, 0.99, 1.0)], "threshold": thr}
        rep["modality"] = {"arm": a.modality_arm, **mo}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "test"}, indent=1))
    for name, t in rep["test"].items():
        o = t["overall"]
        print(f"{name:12s} P {o['p']:.3f} R {o['r']:.3f} F1 {o['f1']:.3f} AUC {o.get('pr_auc', float('nan')):.3f} | " +
              " ".join(f"{l}:{v['f1']:.2f}" for l, v in t["by_lang"].items()))


if __name__ == "__main__":
    main()
