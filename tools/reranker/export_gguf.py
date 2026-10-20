"""Export a trained li.py checkpoint to the engine's late-interaction GGUF (F32 tensors).

usage: python -I export_gguf.py --ckpt runs/li/model.pt --out models/AIST-87M-GGUF/AIST-87M-LI.gguf [--depth 64]
Tensors: li.map.{text,image}[d x in], li.glob[d x 256], li.gate.weight[1 x d], scalars li.gate.bias,
li.glob_bias, li.scale, li.member_tau (log), li.rank_w. Metadata: li.dim, li.codes_per_member, li.text_cap,
li.query_cap, li.kmeans_iters, li.depth, li.max_members (src/models/aist_late_interaction.cpp reads them).
A tower map is exported only when it was trained on that modality (--towers).
The whole file is local only (gitignored); scripts/shard_model.py shards it into
models/AIST-87M-GGUF/chunks/AIST-87M-LI.gguf.part-* and records it in models/manifest.json.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codes import ITERS, TEXT_CAP  # noqa: E402
from li import Q_MAX  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=64)
    ap.add_argument("--towers", default="text,image")
    a = ap.parse_args()
    import gguf
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st = {k: v.float().numpy() for k, v in ck["state"].items()}
    d = ck["cfg"]["d"]
    w = gguf.GGUFWriter(a.out, "aist-li")
    w.add_description("AIST late-interaction reranker (modality-agnostic MaxSim over AIST position features)")
    for k, v in (("li.dim", d), ("li.codes_per_member", ck["k"]), ("li.text_cap", TEXT_CAP), ("li.query_cap", Q_MAX),
                 ("li.kmeans_iters", ITERS), ("li.depth", a.depth), ("li.max_members", 8)):
        w.add_uint32(k, int(v))
    for t in a.towers.split(","):
        w.add_tensor(f"li.map.{t}", np.ascontiguousarray(st[f"maps.{t}.weight"]))
    w.add_tensor("li.glob", np.ascontiguousarray(st["glob.weight"]))
    w.add_tensor("li.gate.weight", np.ascontiguousarray(st["gate.weight"]))
    for name, key in (("li.gate.bias", "gate.bias"), ("li.glob_bias", "glob_bias"), ("li.scale", "scale"),
                      ("li.member_tau", "member_tau"), ("li.rank_w", "rank_w")):
        w.add_tensor(name, np.asarray(st[key], dtype=np.float32).reshape(1))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", a.out, Path(a.out).stat().st_size, "bytes")
    print("next: python3 scripts/shard_model.py (shards it and updates models/manifest.json)")


if __name__ == "__main__":
    main()
