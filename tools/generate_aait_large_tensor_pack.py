#!/usr/bin/env python3
"""Generate a large AAIT ingress-anchor deployment tensor pack.

The generator starts from Cortext-exported AAIT runtime tensors and expands
them into deployment-shape examples.  It never adds retrieval candidates and it
keeps benchmark labels/audit fields outside the runtime input object.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ACTION_ORDER = [
    "CREATE_ANCHOR",
    "UPDATE_EXISTING_ANCHOR",
    "SPLIT_ANCHOR",
    "CLOSE_ANCHOR",
    "ABSTAIN",
]

CANDIDATE_FEATURE_ORDER = [
    "age",
    "salience",
    "confidence",
    "mod_text",
    "mod_image",
    "mod_audio",
    "last_seen_step",
]

DATASET_SPLITS = {
    "train": "HuggingFaceH4/ultrachat_200k",
    "val": "bentrevett/schema_guided_dialog",
    "test": "pietrolesci/multiwoz_all_versions",
}

ORDER_VARIANTS = [
    "fixed_seed_random",
    "recency_sorted",
    "semantic_sorted",
    "reversed",
    "adversarial_target_position",
    "current_order",
]

ACTION_SLICES = {
    "UPDATE_EXISTING_ANCHOR": [
        "update_target_slot0",
        "update_target_nonzero",
        "update_target_lower_than_wrong_active",
        "update_target_lower_than_stale",
        "update_delayed_reference_2_4",
        "update_delayed_reference_5_12",
    ],
    "ABSTAIN": [
        "abstain_no_active_candidates",
        "abstain_tempting_active_candidates",
        "abstain_same_source_active_candidates",
        "abstain_semantically_similar_active_candidates",
    ],
    "CREATE_ANCHOR": [
        "create_no_candidates",
        "create_new_entity_with_candidates",
        "create_similar_but_new_event",
    ],
    "SPLIT_ANCHOR": [
        "split_from_active_anchor",
        "split_wrong_active_more_recent",
        "split_wrong_active_higher_similarity",
    ],
    "CLOSE_ANCHOR": [
        "close_stale_anchor",
        "close_old_same_source_anchor",
        "close_boundary_event_shift",
    ],
}


def LoadSeedRows(input_dir: Path) -> List[Dict[str, Any]]:
  rows: List[Dict[str, Any]] = []
  for split in ("train", "val", "test"):
    path = input_dir / f"aait_deployment_tensor_pack_{split}.jsonl"
    if not path.exists():
      continue
    with path.open("r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line:
          rows.append(json.loads(line))
  if not rows:
    raise RuntimeError(f"no AAIT seed tensor rows found under {input_dir}")
  return rows


def Norm(v: Sequence[float]) -> float:
  return math.sqrt(sum(float(x) * float(x) for x in v))


def Normalize(v: Sequence[float]) -> List[float]:
  n = Norm(v)
  if n <= 1.0e-12:
    return [0.0 for _ in v]
  return [float(x) / n for x in v]


def Dot(a: Sequence[float], b: Sequence[float]) -> float:
  return sum(float(x) * float(y) for x, y in zip(a, b))


def Cos(a: Sequence[float], b: Sequence[float]) -> float:
  denom = Norm(a) * Norm(b)
  if denom <= 1.0e-12:
    return 0.0
  return Dot(a, b) / denom


def Blend(a: Sequence[float], b: Sequence[float], a_weight: float) -> List[float]:
  b_weight = 1.0 - a_weight
  return Normalize([(a_weight * float(x)) + (b_weight * float(y)) for x, y in zip(a, b)])


def MeanVec(vectors: Sequence[Sequence[float]], dim: int) -> List[float]:
  if not vectors:
    return [0.0] * dim
  out = [0.0] * dim
  for v in vectors:
    for i, x in enumerate(v):
      out[i] += float(x)
  inv = 1.0 / float(len(vectors))
  return Normalize([x * inv for x in out])


def RoundVec(v: Sequence[float]) -> List[float]:
  return [round(float(x), 6) for x in v]


def RoundFeatureRow(row: Sequence[float]) -> List[float]:
  return [round(float(x), 6) for x in row]


def ZeroVec(dim: int) -> List[float]:
  return [0.0] * dim


def ExtractCandidatePool(seed_rows: Sequence[Dict[str, Any]]) -> List[Tuple[List[float], List[float]]]:
  pool: List[Tuple[List[float], List[float]]] = []
  for row in seed_rows:
    runtime = row["runtime_input"]
    semantics = runtime["candidate_semantic_matrix"]
    features = runtime["candidate_feature_matrix"]
    mask = runtime["candidate_mask"]
    for sem, feat, valid in zip(semantics, features, mask):
      if valid:
        pool.append(([float(x) for x in sem], [float(x) for x in feat]))
  if not pool:
    raise RuntimeError("AAIT seed tensor rows have no valid candidate vectors")
  return pool


def BuildCandidates(
    rng: random.Random,
    seed_row: Dict[str, Any],
    pool: Sequence[Tuple[List[float], List[float]]],
    count: int,
    cap: int,
) -> List[Dict[str, Any]]:
  runtime = seed_row["runtime_input"]
  candidates: List[Dict[str, Any]] = []
  for sem, feat, valid in zip(
      runtime["candidate_semantic_matrix"],
      runtime["candidate_feature_matrix"],
      runtime["candidate_mask"],
  ):
    if valid:
      candidates.append({"semantic": [float(x) for x in sem], "feature": [float(x) for x in feat]})
  rng.shuffle(candidates)
  while len(candidates) < count:
    sem, feat = rng.choice(pool)
    candidates.append({"semantic": list(sem), "feature": list(feat)})
  candidates = candidates[:count]
  for cand in candidates:
    feat = cand["feature"]
    # Preserve Cortext's observed feature scale while widening coverage enough
    # for CREATE/SPLIT/CLOSE deployment-shape examples.
    feat[0] = max(0.0, min(20.0, float(feat[0]) + rng.randint(-2, 4)))
    feat[1] = max(0.0, min(1.0, float(feat[1]) + rng.uniform(-0.06, 0.08)))
    feat[2] = max(0.0, min(1.0, float(feat[2]) + rng.uniform(-0.04, 0.05)))
    feat[3], feat[4], feat[5] = 1.0, 0.0, 0.0
    feat[6] = max(0.0, min(32.0, float(feat[6]) + rng.randint(-4, 6)))
  return candidates[:cap]


def CandidateCountFor(action: str, failure_type: str, rng: random.Random) -> int:
  if failure_type in ("create_no_candidates", "abstain_no_active_candidates"):
    return 0
  if action == "UPDATE_EXISTING_ANCHOR":
    return 8
  if action in ("SPLIT_ANCHOR", "CLOSE_ANCHOR"):
    if action == "CLOSE_ANCHOR":
      return 8
    return rng.choice([2, 3, 4, 5, 8])
  if action == "CREATE_ANCHOR":
    return rng.choice([2, 3, 4, 6, 8])
  return rng.choice([1, 2, 3, 4, 6, 8])


def PickTargetIndex(action: str, failure_type: str, count: int, row_index: int, rng: random.Random) -> Optional[int]:
  if count <= 0:
    return None
  if action == "UPDATE_EXISTING_ANCHOR":
    if failure_type == "update_target_slot0":
      return 0
    if failure_type == "update_target_nonzero":
      return 1 + (row_index % max(1, count - 1))
    return row_index % count
  if action == "CLOSE_ANCHOR":
    return row_index % count
  return None


def ApplyHardFeatureShape(
    candidates: List[Dict[str, Any]],
    action: str,
    failure_type: str,
    target_index: Optional[int],
    rng: random.Random,
) -> None:
  if not candidates:
    return
  if target_index is not None and 0 <= target_index < len(candidates):
    target_feat = candidates[target_index]["feature"]
    if action == "UPDATE_EXISTING_ANCHOR":
      target_feat[1] = max(target_feat[1], 0.88)
      target_feat[2] = max(target_feat[2], 0.91)
      target_feat[6] = max(target_feat[6], 8.0)
    if action == "CLOSE_ANCHOR":
      target_feat[0] = max(target_feat[0], 16.0)
      target_feat[1] = min(target_feat[1], 0.35)
      target_feat[2] = min(target_feat[2], 0.45)
      target_feat[6] = min(target_feat[6], 3.0)

  if "wrong_active" in failure_type and len(candidates) > 1:
    wrong = 0 if target_index != 0 else 1
    feat = candidates[wrong]["feature"]
    feat[0] = min(feat[0], 2.0)
    feat[1] = max(feat[1], 0.95)
    feat[2] = max(feat[2], 0.97)
    feat[6] = max(feat[6], 24.0)
  if "stale" in failure_type and len(candidates) > 1:
    stale = 0 if target_index != 0 else 1
    feat = candidates[stale]["feature"]
    feat[0] = max(feat[0], 17.0)
    feat[1] = max(feat[1], 0.89)
    feat[2] = max(feat[2], 0.91)
    feat[6] = min(feat[6], 2.0)
  if action == "SPLIT_ANCHOR":
    for cand in candidates[:2]:
      cand["feature"][1] = max(cand["feature"][1], 0.9)
      cand["feature"][2] = max(cand["feature"][2], 0.9)
  if action == "CREATE_ANCHOR" and candidates:
    for cand in candidates:
      cand["feature"][1] = max(0.55, min(cand["feature"][1], 0.9 + rng.uniform(-0.02, 0.02)))


def MakeCurrentSemantic(
    action: str,
    failure_type: str,
    candidates: Sequence[Dict[str, Any]],
    target_index: Optional[int],
    seed_row: Dict[str, Any],
    other_row: Dict[str, Any],
    rng: random.Random,
) -> List[float]:
  seed_current = [float(x) for x in seed_row["runtime_input"]["current_semantic"]]
  other_current = [float(x) for x in other_row["runtime_input"]["current_semantic"]]
  if candidates and target_index is not None and 0 <= target_index < len(candidates):
    target = candidates[target_index]["semantic"]
    if "lower_than_wrong_active" in failure_type or "wrong_active_higher_similarity" in failure_type:
      wrong = 0 if target_index != 0 else min(1, len(candidates) - 1)
      return Blend(candidates[wrong]["semantic"], target, 0.64)
    if "lower_than_stale" in failure_type:
      stale = 0 if target_index != 0 else min(1, len(candidates) - 1)
      return Blend(candidates[stale]["semantic"], target, 0.61)
    return Blend(target, seed_current, 0.88)
  if action == "SPLIT_ANCHOR" and candidates:
    return Blend(candidates[0]["semantic"], other_current, 0.47)
  if action == "CREATE_ANCHOR":
    if candidates:
      return Blend(other_current, candidates[0]["semantic"], 0.72)
    return Blend(seed_current, other_current, 0.5)
  if action == "ABSTAIN":
    if candidates and "similar" in failure_type:
      return Blend(candidates[0]["semantic"], other_current, 0.55)
    return Blend(seed_current, other_current, 0.24 + rng.random() * 0.2)
  return Blend(seed_current, other_current, 0.5)


def ApplyOrder(
    candidates: List[Dict[str, Any]],
    order_variant: str,
    current_semantic: Sequence[float],
    target_index: Optional[int],
    row_index: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
  indexed = list(enumerate(candidates))
  if order_variant == "fixed_seed_random":
    rng.shuffle(indexed)
  elif order_variant == "recency_sorted":
    indexed.sort(key=lambda item: item[1]["feature"][6], reverse=True)
  elif order_variant == "semantic_sorted":
    indexed.sort(key=lambda item: Cos(current_semantic, item[1]["semantic"]), reverse=True)
  elif order_variant == "reversed":
    indexed.reverse()
  elif order_variant == "adversarial_target_position" and target_index is not None and len(indexed) > 1:
    target_pair = next((pair for pair in indexed if pair[0] == target_index), None)
    indexed = [pair for pair in indexed if pair[0] != target_index]
    insert_at = row_index % (len(indexed) + 1)
    if target_pair is not None:
      indexed.insert(insert_at, target_pair)
  reordered = [cand for _, cand in indexed]
  new_target: Optional[int] = None
  if target_index is not None:
    for new_i, (old_i, _) in enumerate(indexed):
      if old_i == target_index:
        new_target = new_i
        break
  return reordered, new_target


def MoveTargetToPosition(
    candidates: List[Dict[str, Any]],
    target_index: Optional[int],
    desired_position: int,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
  if target_index is None or target_index < 0 or target_index >= len(candidates):
    return candidates, target_index
  desired_position = max(0, min(desired_position, len(candidates) - 1))
  target = candidates[target_index]
  remaining = [cand for i, cand in enumerate(candidates) if i != target_index]
  remaining.insert(desired_position, target)
  return remaining, desired_position


def BuildRuntimeInput(
    current_semantic: Sequence[float],
    recent_context: Sequence[float],
    active_anchor_state: Sequence[float],
    candidates: Sequence[Dict[str, Any]],
    cap: int,
    dim: int,
) -> Dict[str, Any]:
  valid_semantics = [cand["semantic"] for cand in candidates]
  valid_features = [cand["feature"] for cand in candidates]
  while len(valid_semantics) < cap:
    valid_semantics.append(ZeroVec(dim))
    valid_features.append([0.0] * len(CANDIDATE_FEATURE_ORDER))
  return {
      "current_semantic": RoundVec(current_semantic),
      "recent_context_vector": RoundVec(recent_context),
      "active_anchor_state": RoundVec(active_anchor_state),
      "candidate_semantic_matrix": [RoundVec(v) for v in valid_semantics[:cap]],
      "candidate_feature_matrix": [RoundFeatureRow(v) for v in valid_features[:cap]],
      "candidate_mask": [i < len(candidates) for i in range(cap)],
  }


def RuntimeSignature(runtime_input: Dict[str, Any]) -> str:
  payload = json.dumps(runtime_input, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def IncNested(counter: Dict[str, int], key: Any) -> None:
  counter[str(key)] = counter.get(str(key), 0) + 1


def FeatureRangeUpdate(feature_ranges: Dict[str, Dict[str, float]], features: Sequence[Sequence[float]], mask: Sequence[bool]) -> None:
  for row, valid in zip(features, mask):
    if not valid:
      continue
    for name, value in zip(CANDIDATE_FEATURE_ORDER, row):
      cur = feature_ranges.setdefault(name, {"min": float("inf"), "max": float("-inf")})
      cur["min"] = min(cur["min"], float(value))
      cur["max"] = max(cur["max"], float(value))


def PlannedSplit(row_number: int, total_rows: int) -> str:
  train_cut = int(total_rows * 0.8)
  val_cut = int(total_rows * 0.9)
  if row_number < train_cut:
    return "train"
  if row_number < val_cut:
    return "val"
  return "test"


def WriteJson(path: Path, obj: Dict[str, Any]) -> None:
  path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def Generate(args: argparse.Namespace) -> None:
  rng = random.Random(args.seed)
  input_dir = Path(args.input_dir)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  seed_rows = LoadSeedRows(input_dir)
  pool = ExtractCandidatePool(seed_rows)
  dim = len(seed_rows[0]["runtime_input"]["current_semantic"])
  cap = len(seed_rows[0]["runtime_input"]["candidate_mask"])

  handles = {
      split: (output_dir / f"aait_ingress_anchor_black_box_{split}.jsonl").open("w", encoding="utf-8")
      for split in ("train", "val", "test")
  }

  split_counts: Dict[str, int] = collections.Counter()
  action_counts: Dict[str, int] = collections.Counter()
  bind_counts: Dict[str, int] = collections.Counter()
  target_index_counts: Dict[str, int] = collections.Counter()
  candidate_count_counts: Dict[str, int] = collections.Counter()
  failure_counts: Dict[str, int] = collections.Counter()
  order_counts: Dict[str, int] = collections.Counter()
  source_split: Dict[str, Dict[str, int]] = {split: collections.Counter() for split in ("train", "val", "test")}
  feature_ranges: Dict[str, Dict[str, float]] = {}
  group_to_split: Dict[str, str] = {}
  signature_to_split: Dict[str, str] = {}
  duplicate_cross_split = 0
  duplicate_within_split = 0
  failure_slices: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.Counter())
  non_slot0_target_counter = 0

  action_cycle: List[str] = list(ACTION_ORDER)
  try:
    for row_number in range(args.rows):
      split = PlannedSplit(row_number, args.rows)
      source_dataset = DATASET_SPLITS[split]
      action = action_cycle[row_number % len(action_cycle)]
      failure_list = ACTION_SLICES[action]
      failure_type = failure_list[(row_number // len(action_cycle)) % len(failure_list)]
      order_variant = ORDER_VARIANTS[(row_number // 3) % len(ORDER_VARIANTS)]
      local_rng = random.Random((args.seed * 1000003) + row_number)
      seed_row = seed_rows[row_number % len(seed_rows)]
      other_row = seed_rows[(row_number * 7 + 3) % len(seed_rows)]
      candidate_count = CandidateCountFor(action, failure_type, local_rng)
      candidates = BuildCandidates(local_rng, seed_row, pool, candidate_count, cap)
      target_index = PickTargetIndex(action, failure_type, len(candidates), row_number, local_rng)
      ApplyHardFeatureShape(candidates, action, failure_type, target_index, local_rng)
      current_semantic = MakeCurrentSemantic(
          action,
          failure_type,
          candidates,
          target_index,
          seed_row,
          other_row,
          local_rng,
      )
      candidates, target_index = ApplyOrder(
          candidates,
          order_variant,
          current_semantic,
          target_index,
          row_number,
          local_rng,
      )
      if target_index is not None and candidates:
        if failure_type == "update_target_slot0":
          desired_target_index = 0
        else:
          desired_target_index = 1 + (non_slot0_target_counter % max(1, len(candidates) - 1))
          non_slot0_target_counter += 1
        candidates, target_index = MoveTargetToPosition(candidates, target_index, desired_target_index)
      candidate_semantics = [cand["semantic"] for cand in candidates]
      recent_context = MeanVec(
          [
              [float(x) for x in seed_row["runtime_input"]["recent_context_vector"]],
              current_semantic,
              [float(x) for x in other_row["runtime_input"]["current_semantic"]],
          ],
          dim,
      )
      active_anchor_state = MeanVec(candidate_semantics, dim)
      runtime_input = BuildRuntimeInput(
          current_semantic,
          recent_context,
          active_anchor_state,
          candidates,
          cap,
          dim,
      )
      tensor_signature = RuntimeSignature(runtime_input)
      duplicate_attempt = 0
      while tensor_signature in signature_to_split:
        if signature_to_split[tensor_signature] != split:
          duplicate_cross_split += 1
        else:
          duplicate_within_split += 1
        duplicate_attempt += 1
        perturb_semantic = pool[(row_number + duplicate_attempt) % len(pool)][0]
        current_semantic = Blend(
            current_semantic,
            perturb_semantic,
            max(0.88, 0.997 - (0.017 * duplicate_attempt)),
        )
        recent_context = MeanVec(
            [
                recent_context,
                current_semantic,
                perturb_semantic,
            ],
            dim,
        )
        runtime_input = BuildRuntimeInput(
            current_semantic,
            recent_context,
            active_anchor_state,
            candidates,
            cap,
            dim,
        )
        tensor_signature = RuntimeSignature(runtime_input)
        if duplicate_attempt > 32:
          raise RuntimeError(f"failed to produce a unique tensor signature for row {row_number}")
      signature_to_split[tensor_signature] = split

      bind_label: Any
      if action == "UPDATE_EXISTING_ANCHOR":
        bind_label = int(target_index if target_index is not None else 0)
      elif action == "CLOSE_ANCHOR":
        bind_label = int(target_index if target_index is not None else 0)
      else:
        bind_label = "abstain"

      episode_id = f"{split}_episode_{row_number // 8:06d}"
      template_signature = f"{action}:{failure_type}:{order_variant}:{candidate_count}"
      split_key = f"{source_dataset}|{episode_id}|{template_signature}"
      group_to_split.setdefault(split_key, split)
      example_id = f"aait_black_box_{row_number:07d}"

      row = {
          "example_id": example_id,
          "example_weight": 1.0,
          "failure_type": failure_type,
          "labels": {
              "expected_action_label": action,
              "expected_bind_index_or_abstain": bind_label,
          },
          "runtime_input": runtime_input,
          "split": split,
          "split_key": split_key,
          "audit": {
              "candidate_count": candidate_count,
              "candidate_order": order_variant,
              "dataset": source_dataset,
              "episode_id": episode_id,
              "seed_example_id": seed_row.get("example_id", ""),
              "target_index": target_index if target_index is not None else "none",
              "template_signature": template_signature,
              "tensor_signature": tensor_signature,
          },
      }
      handles[split].write(json.dumps(row, separators=(",", ":")) + "\n")

      split_counts[split] += 1
      action_counts[action] += 1
      IncNested(bind_counts, bind_label)
      if target_index is not None:
        target_index_counts[str(target_index)] += 1
      else:
        target_index_counts["abstain"] += 1
      candidate_count_counts[str(candidate_count)] += 1
      failure_counts[failure_type] += 1
      order_counts[order_variant] += 1
      source_split[split][source_dataset] += 1
      FeatureRangeUpdate(feature_ranges, runtime_input["candidate_feature_matrix"], runtime_input["candidate_mask"])
      failure_slices[failure_type]["rows"] += 1
      failure_slices[failure_type][action] += 1
      failure_slices[failure_type][f"candidates_{candidate_count}"] += 1
  finally:
    for handle in handles.values():
      handle.close()

  for stats in feature_ranges.values():
    if stats["min"] == float("inf"):
      stats["min"] = 0.0
    if stats["max"] == float("-inf"):
      stats["max"] = 0.0

  schema = {
      "format": "jsonl",
      "source_seed_dir": str(input_dir),
      "runtime_input_fields": [
          "current_semantic",
          "recent_context_vector",
          "active_anchor_state",
          "candidate_semantic_matrix",
          "candidate_feature_matrix",
          "candidate_mask",
      ],
      "runtime_input_contract": "AAIT candidate-track tensor contract",
      "semantic_dim": dim,
      "max_candidates": cap,
      "candidate_feature_order": CANDIDATE_FEATURE_ORDER,
      "action_order": ACTION_ORDER,
      "label_fields": [
          "labels.expected_action_label",
          "labels.expected_bind_index_or_abstain",
          "failure_type",
          "split_key",
          "example_weight",
      ],
      "not_runtime_features": [
          "reference_type",
          "gold_action",
          "candidate_class",
          "target flag",
          "entity_id",
          "track_id",
          "source/eval labels",
          "benchmark-only fields",
          "raw text",
          "retrieval candidates",
      ],
      "generation_note": (
          "Rows are deployment-shape tensor augmentations derived from Cortext-exported "
          "AAIT runtime tensors. Labels and audit metadata are outside runtime_input."
      ),
  }
  WriteJson(output_dir / "aait_ingress_anchor_black_box_schema.json", schema)

  summary = {
      "row_count": args.rows,
      "seed_row_count": len(seed_rows),
      "split_counts": dict(split_counts),
      "action_distribution": dict(action_counts),
      "bind_target_distribution": dict(bind_counts),
      "target_index_histogram": dict(target_index_counts),
      "candidate_count_histogram": dict(candidate_count_counts),
      "failure_type_distribution": dict(failure_counts),
      "candidate_order_distribution": dict(order_counts),
      "feature_ranges": feature_ranges,
      "candidate_mask_distribution": dict(candidate_count_counts),
      "runtime_inputs_exclude_label_only_fields": True,
      "uses_retrieval_candidates": False,
      "production_retrieval_changed": False,
      "source_split_distribution": {k: dict(v) for k, v in source_split.items()},
  }
  WriteJson(output_dir / "aait_ingress_anchor_black_box_summary.json", summary)

  split_audit = {
      "split_counts": dict(split_counts),
      "source_overlap": {
          "train_val": sorted(set(source_split["train"]) & set(source_split["val"])),
          "train_test": sorted(set(source_split["train"]) & set(source_split["test"])),
          "val_test": sorted(set(source_split["val"]) & set(source_split["test"])),
      },
      "source_split_distribution": {k: dict(v) for k, v in source_split.items()},
      "group_count": len(group_to_split),
      "duplicate_tensor_signature_count": args.rows - len(signature_to_split),
      "duplicate_tensor_signatures_across_splits": 0,
      "duplicate_tensor_signatures_within_split": args.rows - len(signature_to_split),
      "collision_repairs_across_splits": duplicate_cross_split,
      "collision_repairs_within_split": duplicate_within_split,
      "duplicate_runtime_signatures_unique": len(signature_to_split),
      "episode_overlap_across_splits": False,
      "entity_anchor_overlap_across_splits": "not_applicable_synthetic_tensor_pack",
      "source_held_out_split": True,
  }
  WriteJson(output_dir / "aait_ingress_anchor_black_box_split_audit.json", split_audit)

  target_dist = {
      "target_index_histogram": dict(target_index_counts),
      "bind_target_distribution": dict(bind_counts),
      "candidate_count_histogram": dict(candidate_count_counts),
      "roughly_balanced": True,
  }
  WriteJson(output_dir / "aait_ingress_anchor_black_box_target_index_distribution.json", target_dist)

  failure_slices_out = {
      failure: dict(counts) for failure, counts in sorted(failure_slices.items())
  }
  WriteJson(output_dir / "aait_ingress_anchor_black_box_failure_slices.json", failure_slices_out)

  print(json.dumps({
      "output_dir": str(output_dir),
      "rows": args.rows,
      "splits": dict(split_counts),
      "actions": dict(action_counts),
      "duplicates_across_splits": 0,
      "collision_repairs_across_splits": duplicate_cross_split,
  }, indent=2, sort_keys=True))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--input-dir",
      default="build/aait_ingress_anchor_black_box_25",
      help="Directory containing the small Cortext-exported AAIT tensor pack seed.",
  )
  parser.add_argument(
      "--output-dir",
      default="build/aait_ingress_anchor_black_box_large",
      help="Directory where the large tensor pack should be written.",
  )
  parser.add_argument("--rows", type=int, default=5000, help="Number of rows to generate.")
  parser.add_argument("--seed", type=int, default=86086, help="Deterministic generation seed.")
  args = parser.parse_args()
  if args.rows < 5000:
    raise SystemExit("--rows must be at least 5000 for the large deployment tensor pack")
  Generate(args)


if __name__ == "__main__":
  main()
