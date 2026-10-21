"""Engine embeddings for the modality-agnostic check and the mirror parity check (evaluation only).

usage: python -I modality_probe.py --out /Volumes/Build/cortext/directive-head/modality [--n 600]
Uses the shipped engine through the cortext Python binding (augmem_cortext 2.0.0 wheel, unpacked at
$CORTEXT_WHEEL_DIR) with CORTEXT_AIST_MODEL_PATH / CORTEXT_LI_MODEL_PATH pointing at the assembled GGUFs.
Writes image.npy, audio.npy, speech.npy ([n, 1280] float32, engine embed_image / embed_audio) and
parity_<LC_CTYPE>.json (cosine of the PyTorch mirror's text vector vs the engine's embed_text on --parity-texts).
The engine's text tokenizer calls std::isalnum / std::isspace / std::tolower on raw bytes, so its output depends
on the process locale: run with LC_ALL=C for the behaviour of the C++ runners (the mirror matches it), and with
a UTF-8 locale to see what a host that calls setlocale (Python does) gets for accented text.
Inputs: MemLens media (memlens195/media/*.jpg; images only), salt_resynth_31k audio/ and speech/ (wav, resampled
to 16 kHz mono as the engine expects for PCM input).
"""
from __future__ import annotations

import argparse, json, os, random, sys, wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAND = Path("/Users/gabrielwillen/VSCode/augmem/augmem/cortext.cpp/.claude/worktrees/land-rr/models/AIST-87M-GGUF")
os.environ.setdefault("CORTEXT_AIST_MODEL_PATH", str(LAND / "AIST-87M_q8_0.gguf"))
os.environ.setdefault("CORTEXT_LI_MODEL_PATH", str(LAND / "AIST-87M-LI.gguf"))
sys.path.insert(0, os.environ.get("CORTEXT_WHEEL_DIR", "/Volumes/Build/cortext/directive-head/wheel"))
MEDIA = Path("/Users/gabrielwillen/VSCode/augmem/augmem/cortext.cpp/.claude/hillclimb/scorecard/data/memlens195/media")
SALT = Path("/Users/gabrielwillen/VSCode/augmem/datasets/salt_resynth_31k")


def read_wav16k(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype={2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    x = x.reshape(-1, ch).mean(1) / float(2 ** (8 * sw - 1))
    if sr != 16000:
        t = np.arange(int(len(x) * 16000 / sr)) * (sr / 16000)
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--parity-texts", default="/Volumes/Build/cortext/directive-head/data/rows.jsonl")
    a = ap.parse_args()
    import cortext
    from PIL import Image
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c = cortext.Cortext(":memory:")
    rng = random.Random(0)
    if a.parity_only:
        rng = random.Random(1)
    imgs = [] if a.parity_only else sorted(p for p in MEDIA.iterdir() if p.suffix == ".jpg")
    vec = []
    for p in rng.sample(imgs, min(a.n, len(imgs))):
        im = Image.open(p).convert("RGB")
        im.thumbnail((768, 768), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.uint8)
        vec.append(c.embed_image(arr.tobytes(), arr.shape[1], arr.shape[0], 3))
    if not a.parity_only:
        np.save(out / "image.npy", np.array(vec, dtype=np.float32)); print("image", len(vec), flush=True)
    for kind in (() if a.parity_only else ("audio", "speech")):
        files = sorted((SALT / kind).iterdir())
        vec = [c.embed_audio(read_wav16k(p).tolist()) for p in rng.sample(files, a.n)]
        np.save(out / f"{kind}.npy", np.array(vec, dtype=np.float32)); print(kind, len(vec), flush=True)
    # mirror parity on dataset texts (all languages)
    sys.path.insert(0, str(HERE))
    import featurize as F
    rows = [json.loads(l) for l in open(a.parity_texts)]
    rows = rng.sample(rows, 400)
    tok, m = F.load("engine", "cpu")
    mirror, _, _ = F.embed([r["text"] for r in rows], tok, m, "cpu")
    eng = np.array([c.embed_text(r["text"]) for r in rows], dtype=np.float32)
    cos = (mirror.astype(np.float32) * eng).sum(1) / np.linalg.norm(eng, axis=1) / np.linalg.norm(mirror.astype(np.float32), axis=1)
    by = {}
    for r, v in zip(rows, cos):
        by.setdefault(r["lang"], []).append(float(v))
    import locale
    rep = {"lc_ctype": locale.setlocale(locale.LC_CTYPE), "n": len(rows), "min": float(cos.min()), "mean": float(cos.mean()),
           "by_lang_min": {k: min(v) for k, v in sorted(by.items())}}
    np.save(out / "parity_engine_text.npy", eng)
    (out / f"parity_{rep['lc_ctype'].replace('.', '_')}.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))


if __name__ == "__main__":
    main()
