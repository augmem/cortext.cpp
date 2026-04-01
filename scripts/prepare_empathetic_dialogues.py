#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path


DATA_URL = (
    "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/"
    "empatheticdialogues.tar.gz"
)


def download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def extract(tar_path: Path, out_dir: Path) -> Path:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if any(raw_dir.rglob("*.csv")):
        return raw_dir
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(raw_dir)
    return raw_dir


def resolve_csv(raw_dir: Path, split: str) -> Path:
    candidates = {p.name: p for p in raw_dir.rglob("*.csv")}
    aliases = {
        "train": ["train.csv"],
        "valid": ["valid.csv", "dev.csv"],
        "test": ["test.csv"],
    }
    for name in aliases.get(split, []):
        if name in candidates:
            return candidates[name]
    raise FileNotFoundError(f"Missing {split}.csv under {raw_dir}")


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def speaker_label(raw_value: str, fallback_idx: int) -> str:
    if raw_value is not None and raw_value != "":
        try:
            idx = int(raw_value)
            return f"agent_{idx + 1}"
        except ValueError:
            pass
    return f"agent_{(fallback_idx % 2) + 1}"


def convert_split(csv_path: Path, out_path: Path, limit: int | None) -> int:
    conv_turns: dict[str, list[tuple[int, str, str]]] = {}
    conv_order: list[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id = row.get("conv_id") or row.get("conversation_id")
            if not conv_id:
                continue
            if conv_id not in conv_turns:
                conv_turns[conv_id] = []
                conv_order.append(conv_id)
                if limit is not None and len(conv_order) > limit:
                    break
            utterance = row.get("utterance") or row.get("text") or ""
            utterance = normalize_text(utterance)
            if not utterance:
                continue
            try:
                idx = int(row.get("utterance_idx", "0"))
            except ValueError:
                idx = len(conv_turns[conv_id])
            speaker = row.get("speaker_idx") or row.get("speaker") or ""
            conv_turns[conv_id].append((idx, speaker, utterance))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for conv_id in conv_order[: limit or None]:
            turns = sorted(conv_turns[conv_id], key=lambda t: t[0])
            content = []
            for i, (_, speaker, utterance) in enumerate(turns):
                content.append(
                    {"agent": speaker_label(speaker, i), "message": utterance}
                )
            if not content:
                continue
            payload = {"content": content}
            out.write(json.dumps([conv_id, payload]) + "\n")
    return len(conv_order[: limit or None])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and convert EmpatheticDialogues to Cortext JSONL."
    )
    parser.add_argument(
        "--out-dir",
        default="data/empathetic_dialogues",
        help="Output directory for converted JSONL.",
    )
    parser.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid", "test", "all"],
        help="Dataset split to convert.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max conversations to emit (per split).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the tarball even if it exists.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    tar_path = out_dir / "empatheticdialogues.tar.gz"
    download(DATA_URL, tar_path, args.force_download)
    raw_dir = extract(tar_path, out_dir)

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    for split in splits:
        csv_path = resolve_csv(raw_dir, split)
        out_path = out_dir / f"{split}.jsonl"
        count = convert_split(csv_path, out_path, args.limit)
        print(f"[OK] {split}: wrote {count} conversations to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
