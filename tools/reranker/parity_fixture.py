"""Python reference values for the C++ late-interaction parity test.

usage: python -I parity_fixture.py --ckpt runs/li/model.pt --data /Volumes/Build/cortext/reranker/data/eval1 \
          --out tests/fixtures/late_interaction_parity.json
Takes one test query and its first 4 pool memories (<= 2 members each, member positions cut to 24 tokens),
plus (--image-data) one memory holding an image member (144 patch features), writes the raw position
features (query tokens, member tokens), the 256-d retrieval vectors, and the Python memory codes (int8) and
memory scores, computed with codes.kmeans + li.py math (int8 codes renormalized, as the engine dequantizes).
Float arrays go to <out>.bin (little-endian float32, concatenated in JSON "blocks" order); the JSON holds shapes,
codes and scores.
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codes import kmeans, TEXT_CAP  # noqa: E402
from li import LI, Q_MAX  # noqa: E402


def q8_int(x):
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
    return torch.round(x / s).clamp(-127, 127)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-data", default="", help="dataset dir with image items (memlens)")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = LI(ck["cfg"]["d"])
    m.load_state_dict(ck["state"]); m.eval()
    K = ck["k"]
    d = Path(a.data)
    Q = [json.loads(l) for l in (d / "queries.jsonl").open()]
    off = np.load(d / "tok_off.npy"); tok = np.load(d / "tok.npy", mmap_mode="r"); sem = np.load(d / "sem.npy", mmap_mode="r")
    q = next(x for x in Q if x["split"] == "test" and len(x["cands"]) >= 4)
    pos = lambda i, cap: np.asarray(tok[off[i]:off[i + 1]][:cap], dtype=np.float32)
    MEM_CUT = 24
    glob = lambda i: (lambda g: g / np.linalg.norm(g))(np.asarray(sem[i][:256], dtype=np.float32))
    with torch.no_grad():
        qx = torch.from_numpy(pos(q["text_idx"], Q_MAX))
        qv = m.embed(qx, 0)
        qg = m.embed_glob(torch.from_numpy(glob(q["text_idx"]))[None])
        vec = torch.cat([qv, qg], 0)
        logit = m.gate(vec).squeeze(-1)
        logit[-1] = logit[-1] + m.glob_bias
        w = torch.softmax(logit, -1)
        mems, scores = [], []
        blocks = [pos(q["text_idx"], Q_MAX), glob(q["text_idx"])]
        tau = m.member_tau.exp()
        for rank, c in enumerate(q["cands"][:4]):
            members, ms = [], []
            for i in c["members"][:2]:
                x = torch.from_numpy(pos(i, MEM_CUT))
                codes = q8_int(F.normalize(m.maps["text"](kmeans(x, K)), dim=-1))
                g = q8_int(m.embed_glob(torch.from_numpy(glob(i))[None]))
                mv = F.normalize(torch.cat([codes, g], 0), dim=-1)
                ms.append(float((w * (vec @ mv.T).amax(1)).sum()))
                blocks.append(pos(i, MEM_CUT)); blocks.append(glob(i))
                members.append({"positions": [int(len(blocks[-2])), 384], "codes": codes.to(torch.int64).flatten().tolist()})
            msv = torch.tensor(ms)
            mem = float(tau * torch.logsumexp(msv / tau, 0))
            rk = math.log1p(rank) / math.log(101.0)
            scores.append(float(m.scale) * mem + float(m.rank_w) * rk)
            mems.append(members)
        if a.image_data:
            di = Path(a.image_data)
            mod = np.load(di / "modality.npy"); ioff = np.load(di / "image_off.npy")
            ifeat = np.load(di / "image_feat.npy", mmap_mode="r"); isem = np.load(di / "sem.npy", mmap_mode="r")
            i = int(np.nonzero(mod == 1)[0][0])
            x = torch.from_numpy(np.asarray(ifeat[ioff[i]:ioff[i + 1]], dtype=np.float32))
            g_img = np.asarray(isem[i][:256], dtype=np.float32); g_img /= np.linalg.norm(g_img)
            codes = q8_int(F.normalize(m.maps["image"](kmeans(x, K)), dim=-1))
            g = q8_int(m.embed_glob(torch.from_numpy(g_img)[None]))
            mv = F.normalize(torch.cat([codes, g], 0), dim=-1)
            msv = torch.tensor([float((w * (vec @ mv.T).amax(1)).sum())])
            mem = float(tau * torch.logsumexp(msv / tau, 0))
            rank = len(mems)
            scores.append(float(m.scale) * mem + float(m.rank_w) * math.log1p(rank) / math.log(101.0))
            blocks.append(x.numpy()); blocks.append(g_img)
            mems.append([{"tower": "image", "positions": [144, 960], "codes": codes.to(torch.int64).flatten().tolist()}])
    Path(a.out).with_suffix(".bin").write_bytes(b"".join(np.ascontiguousarray(b, dtype="<f4").tobytes() for b in blocks))
    Path(a.out).write_text(json.dumps({"query": {"positions": [int(len(blocks[0])), 384]},
                                       "memories": mems, "scores": scores, "k": K, "dim": ck["cfg"]["d"],
                                       "order": "query.positions, query.global, then per member positions, global (members default to the text tower)"},
                                      separators=(",", ":")))
    print("wrote", a.out, "scores", [round(s, 4) for s in scores])


if __name__ == "__main__":
    main()
