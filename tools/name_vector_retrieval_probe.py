#!/usr/bin/env python3
"""Benchmark-only open-name retrieval probe.

This script tests whether a plain embedding encoder plus a large name vector
bank can recover possible person/pet names from short memory-like text without
an autoregressive extractor. It does not change production retrieval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


NAME_RE = re.compile(r"^[A-Za-z][A-Za-z' -]{1,31}$")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class QueryCase:
    case_id: str
    text: str
    expected_names: tuple[str, ...]
    slice_name: str


def norm_name(value: str) -> str:
    value = value.strip().replace("_", " ")
    value = SPACE_RE.sub(" ", value)
    return value.title()


def key_name(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def load_names(path: Path, top_n: int) -> list[str]:
    counts: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("forename", "")
            name = norm_name(raw)
            if not NAME_RE.match(name):
                continue
            key = key_name(name)
            if len(key) < 3:
                continue
            try:
                count = int(row.get("count", "0") or "0")
            except ValueError:
                count = 0
            counts[name] = counts.get(name, 0) + count

    forced = [
        "Jared",
        "Alex",
        "Justin",
        "Bailey",
        "Maya",
        "Jordan",
        "Sarah",
        "Priya",
        "Omar",
        "Diego",
        "Aisha",
        "Chen",
        "Maria",
        "Luna",
        "Rose",
        "Summer",
        "Faith",
        "River",
        "April",
        "Mark",
        "Bill",
        "Grant",
    ]
    for name in forced:
        counts.setdefault(name, 1)

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    names = [name for name, _ in ranked[:top_n]]
    for name in forced:
        if name not in names:
            names.append(name)
    return names


def load_common_words(path: Path = Path("/usr/share/dict/words")) -> set[str]:
    if not path.exists():
        return set()
    words: set[str] = set()
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            word = line.strip().lower()
            if len(word) >= 3 and word.isalpha():
                words.add(word)
    return words


def build_cases() -> list[QueryCase]:
    return [
        QueryCase("pos_jared_house", "I went to Jared's house yesterday.", ("Jared",), "positive"),
        QueryCase("pos_bailey_dog", "That dog's name is Bailey.", ("Bailey",), "positive"),
        QueryCase("pos_alex_justin", "Alex and Justin came over after work.", ("Alex", "Justin"), "positive_multi"),
        QueryCase("pos_maya_park", "We met Maya at the park.", ("Maya",), "positive"),
        QueryCase("pos_jordan_honda", "Jordan has the red Honda.", ("Jordan",), "positive"),
        QueryCase("pos_sarah_coffee", "Please remind me Sarah likes black coffee.", ("Sarah",), "positive"),
        QueryCase("pos_priya_soup", "My neighbor Priya brought soup.", ("Priya",), "positive"),
        QueryCase("pos_omar_dinner", "Uncle Omar called about dinner.", ("Omar",), "positive"),
        QueryCase("pos_diego_contractor", "The contractor was named Diego.", ("Diego",), "positive"),
        QueryCase("pos_aisha_nurse", "A nurse named Aisha helped today.", ("Aisha",), "positive"),
        QueryCase("pos_chen_gym", "I saw Chen at the gym.", ("Chen",), "positive"),
        QueryCase("pos_maria_luna", "Maria is bringing her dog Luna.", ("Maria", "Luna"), "positive_multi"),
        QueryCase("pos_mark_name", "The repair guy is Mark.", ("Mark",), "positive_common_word"),
        QueryCase("pos_april_name", "April said the meeting moved.", ("April",), "positive_common_word"),
        QueryCase("pos_grant_name", "Grant fixed the sink.", ("Grant",), "positive_common_word"),
        QueryCase("neg_dog_yard", "The dog ran through the yard.", (), "negative"),
        QueryCase("neg_car_crash", "A car crashed into a tree.", (), "negative"),
        QueryCase("neg_pool_rain", "It started raining near the pool.", (), "negative"),
        QueryCase("neg_hung_out", "We just hung out for a while.", (), "negative"),
        QueryCase("neg_bell", "The bell rang from the church tower.", (), "negative"),
        QueryCase("neg_milk", "I need to buy milk and bread.", (), "negative"),
        QueryCase("neg_red_mug", "The red mug is on the counter.", (), "negative"),
        QueryCase("neg_rose_bush", "The rose bush needs water.", (), "negative_common_name"),
        QueryCase("neg_summer_heat", "The summer heat was rough today.", (), "negative_common_name"),
        QueryCase("neg_faith_group", "The faith group met in the hall.", (), "negative_common_name"),
        QueryCase("neg_river", "The river overflowed after the storm.", (), "negative_common_name"),
        QueryCase("neg_april_month", "The April meeting moved to Tuesday.", (), "negative_common_name"),
        QueryCase("neg_mark_wall", "The mark on the wall is still visible.", (), "negative_common_name"),
        QueryCase("neg_bill_due", "The bill is due tomorrow.", (), "negative_common_name"),
        QueryCase("neg_grant_approved", "The grant was approved last week.", (), "negative_common_name"),
    ]


def name_templates(name: str) -> list[str]:
    return [
        name,
        f"a person named {name}",
        f"someone called {name}",
        f"{name}'s house",
        f"the dog named {name}",
    ]


def write_embed_input(path: Path, names: list[str], cases: list[QueryCase]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"name_rows": [], "case_rows": []}
    with path.open("w", encoding="utf-8") as f:
        for name in names:
            for template in name_templates(name):
                index = len(manifest["name_rows"])  # type: ignore[arg-type]
                manifest["name_rows"].append({"index": index, "name": name, "text": template})  # type: ignore[union-attr]
                f.write(template + "\n")
        for case in cases:
            index = len(manifest["name_rows"]) + len(manifest["case_rows"])  # type: ignore[arg-type]
            manifest["case_rows"].append({"index": index, "case_id": case.case_id, "text": case.text})  # type: ignore[union-attr]
            f.write(case.text + "\n")
    return manifest


def run_embedder(embedder: Path, input_path: Path, output_path: Path, models: Path, force: bool) -> None:
    if output_path.exists() and not force:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(embedder),
        f"--input={input_path}",
        f"--out={output_path}",
        f"--models={models}",
    ]
    subprocess.run(cmd, check=True)


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    vectors: list[np.ndarray] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            vectors.append(np.asarray(row["embedding"], dtype=np.float32))
    matrix = np.vstack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)
    return texts, matrix


def top_names_for_query(
    query_vec: np.ndarray,
    name_vectors: np.ndarray,
    template_names: list[str],
    top_k: int,
) -> list[dict[str, object]]:
    scores = name_vectors @ query_vec
    best: dict[str, tuple[float, int]] = {}
    for i, score in enumerate(scores):
        name = template_names[i]
        current = best.get(name)
        if current is None or float(score) > current[0]:
            best[name] = (float(score), i)
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[:top_k]
    return [
        {
            "rank": rank + 1,
            "name": name,
            "score": score_index[0],
            "template": name_templates(name)[score_index[1] % len(name_templates(name))],
        }
        for rank, (name, score_index) in enumerate(ranked)
    ]


def build_rankings(
    cases: list[QueryCase],
    case_vectors: np.ndarray,
    name_vectors: np.ndarray,
    template_names: list[str],
    top_k: int,
) -> dict[str, list[dict[str, object]]]:
    rankings: dict[str, list[dict[str, object]]] = {}
    for case, query_vec in zip(cases, case_vectors):
        rankings[case.case_id] = top_names_for_query(query_vec, name_vectors, template_names, top_k)
    return rankings


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = min(len(arr) - 1, max(0, math.ceil(q * len(arr)) - 1))
    return arr[idx]


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def evaluate(cases: list[QueryCase], rankings: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    positives = [c for c in cases if c.expected_names]
    negatives = [c for c in cases if not c.expected_names]
    negative_scores = [float(rankings[c.case_id][0]["score"]) for c in negatives]
    threshold_zero_fpr = (max(negative_scores) + 1e-6) if negative_scores else 1.0
    threshold_5pct = pct(negative_scores, 0.95) if negative_scores else 1.0

    def expected_in_top(case: QueryCase, k: int) -> bool:
        top = {str(row["name"]).lower() for row in rankings[case.case_id][:k]}
        return any(name.lower() in top for name in case.expected_names)

    def all_expected_in_top(case: QueryCase, k: int) -> bool:
        top = {str(row["name"]).lower() for row in rankings[case.case_id][:k]}
        return all(name.lower() in top for name in case.expected_names)

    def expected_above(case: QueryCase, k: int, threshold: float) -> bool:
        for row in rankings[case.case_id][:k]:
            if float(row["score"]) < threshold:
                continue
            if str(row["name"]).lower() in {n.lower() for n in case.expected_names}:
                return True
        return False

    def negative_pass(case: QueryCase, threshold: float) -> bool:
        return float(rankings[case.case_id][0]["score"]) >= threshold

    return {
        "case_count": len(cases),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "top1_any_expected": safe_div(sum(expected_in_top(c, 1) for c in positives), len(positives)),
        "top3_any_expected": safe_div(sum(expected_in_top(c, 3) for c in positives), len(positives)),
        "top5_any_expected": safe_div(sum(expected_in_top(c, 5) for c in positives), len(positives)),
        "top10_any_expected": safe_div(sum(expected_in_top(c, 10) for c in positives), len(positives)),
        "top5_all_expected": safe_div(sum(all_expected_in_top(c, 5) for c in positives), len(positives)),
        "negative_best_score_mean": safe_div(sum(negative_scores), len(negative_scores)),
        "negative_best_score_p95": pct(negative_scores, 0.95),
        "threshold_zero_fpr": threshold_zero_fpr,
        "zero_fpr_top5_recovery": safe_div(
            sum(expected_above(c, 5, threshold_zero_fpr) for c in positives), len(positives)
        ),
        "zero_fpr_negative_fpr": safe_div(
            sum(negative_pass(c, threshold_zero_fpr) for c in negatives), len(negatives)
        ),
        "threshold_5pct": threshold_5pct,
        "five_pct_top5_recovery": safe_div(
            sum(expected_above(c, 5, threshold_5pct) for c in positives), len(positives)
        ),
        "five_pct_negative_fpr": safe_div(
            sum(negative_pass(c, threshold_5pct) for c in negatives), len(negatives)
        ),
    }


def write_outputs(
    output_dir: Path,
    cases: list[QueryCase],
    names: list[str],
    manifest: dict[str, object],
    rankings: dict[str, list[dict[str, object]]],
    summary: dict[str, object],
    metadata_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    summary = {
        **summary,
        "name_count": len(names),
        "template_count": len(manifest["name_rows"]),  # type: ignore[arg-type]
        "encoder_metadata": metadata,
        "outputs": {
            "summary": str(output_dir / "name_vector_retrieval_summary.json"),
            "cases": str(output_dir / "name_vector_retrieval_cases.csv"),
            "topk": str(output_dir / "name_vector_retrieval_topk.csv"),
            "false_positives": str(output_dir / "name_vector_retrieval_false_positives.csv"),
            "manifest": str(output_dir / "name_vector_bank_manifest.json"),
        },
    }
    (output_dir / "name_vector_retrieval_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "name_vector_bank_manifest.json").write_text(
        json.dumps({"names": names, **manifest}, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "name_vector_retrieval_cases.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case_id",
            "slice",
            "text",
            "expected_names",
            "top1",
            "top1_score",
            "top3",
            "top5",
            "expected_in_top1",
            "expected_in_top3",
            "expected_in_top5",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            rows = rankings[case.case_id]
            top_names = [str(row["name"]) for row in rows]
            expected = {name.lower() for name in case.expected_names}
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "slice": case.slice_name,
                    "text": case.text,
                    "expected_names": "|".join(case.expected_names),
                    "top1": top_names[0],
                    "top1_score": f"{float(rows[0]['score']):.6f}",
                    "top3": "|".join(top_names[:3]),
                    "top5": "|".join(top_names[:5]),
                    "expected_in_top1": int(bool(expected and top_names[0].lower() in expected)),
                    "expected_in_top3": int(any(name.lower() in expected for name in top_names[:3])),
                    "expected_in_top5": int(any(name.lower() in expected for name in top_names[:5])),
                }
            )

    with (output_dir / "name_vector_retrieval_topk.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["case_id", "slice", "expected_names", "rank", "name", "score", "template"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            for row in rankings[case.case_id]:
                writer.writerow(
                    {
                        "case_id": case.case_id,
                        "slice": case.slice_name,
                        "expected_names": "|".join(case.expected_names),
                        "rank": row["rank"],
                        "name": row["name"],
                        "score": f"{float(row['score']):.6f}",
                        "template": row["template"],
                    }
                )

    threshold_5pct = float(summary["threshold_5pct"])
    with (output_dir / "name_vector_retrieval_false_positives.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = ["case_id", "slice", "text", "top1", "top1_score", "top5"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            if case.expected_names:
                continue
            rows = rankings[case.case_id]
            if float(rows[0]["score"]) >= threshold_5pct:
                writer.writerow(
                    {
                        "case_id": case.case_id,
                        "slice": case.slice_name,
                        "text": case.text,
                        "top1": rows[0]["name"],
                        "top1_score": f"{float(rows[0]['score']):.6f}",
                        "top5": "|".join(str(row["name"]) for row in rows[:5]),
                    }
                )


def write_ablation_results(output_dir: Path, results: dict[str, dict[str, object]]) -> None:
    (output_dir / "name_vector_retrieval_ablation_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", type=Path, default=Path("data/surnames/forenames.csv"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--embedder", type=Path, default=Path("build/tools/text_embedder/cortext_text_embedder"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/name_vector_retrieval_probe"))
    parser.add_argument("--top-names", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    names = load_names(args.names, args.top_names)
    cases = build_cases()
    input_path = args.output_dir / "name_vector_retrieval_embed_input.txt"
    embedding_path = args.output_dir / "embeddings.jsonl"
    manifest = write_embed_input(input_path, names, cases)
    run_embedder(args.embedder, input_path, embedding_path, args.models, args.force)

    texts, matrix = load_embeddings(embedding_path)
    name_row_count = len(manifest["name_rows"])  # type: ignore[arg-type]
    if len(texts) != name_row_count + len(cases):
        raise RuntimeError(f"Embedding row mismatch: expected {name_row_count + len(cases)}, got {len(texts)}")

    name_vectors = matrix[:name_row_count]
    case_vectors = matrix[name_row_count:]
    template_names = [str(row["name"]) for row in manifest["name_rows"]]  # type: ignore[index]

    template_variants = {
        "name_only": {0},
        "person_context": {0, 1, 2},
        "name_plus_named_pet": {0, 1, 2, 4},
        "all_templates": {0, 1, 2, 3, 4},
    }
    common_words = load_common_words()
    ablation_results: dict[str, dict[str, object]] = {}
    variant_rankings: dict[str, dict[str, list[dict[str, object]]]] = {}
    template_count = len(name_templates("Example"))
    for variant_name, allowed_template_ids in template_variants.items():
        indices = [
            i
            for i in range(name_row_count)
            if (i % template_count) in allowed_template_ids
        ]
        masked_vectors = name_vectors[indices]
        masked_names = [template_names[i] for i in indices]
        rankings_for_variant = build_rankings(
            cases, case_vectors, masked_vectors, masked_names, args.top_k
        )
        variant_rankings[variant_name] = rankings_for_variant
        ablation_results[variant_name] = evaluate(cases, rankings_for_variant)

    filtered_indices = [
        i
        for i in range(name_row_count)
        if (i % template_count) in template_variants["all_templates"]
        and key_name(template_names[i]) not in common_words
    ]
    filtered_rankings = build_rankings(
        cases,
        case_vectors,
        name_vectors[filtered_indices],
        [template_names[i] for i in filtered_indices],
        args.top_k,
    )
    variant_rankings["all_templates_no_common_words"] = filtered_rankings
    filtered_summary = evaluate(cases, filtered_rankings)
    filtered_summary["filtered_name_count"] = len(
        {template_names[i] for i in filtered_indices}
    )
    filtered_summary["common_word_filter"] = "/usr/share/dict/words"
    ablation_results["all_templates_no_common_words"] = filtered_summary

    rankings = variant_rankings["all_templates"]

    summary = evaluate(cases, rankings)
    write_outputs(args.output_dir, cases, names, manifest, rankings, summary, args.output_dir / "metadata.json")
    write_ablation_results(args.output_dir, ablation_results)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
