#!/usr/bin/env python3
"""Audit labels for the AAIT ingress-anchor black-box tensor pack.

This validates procedural label invariants locally and can optionally ask
GPT-5.4-mini to review a capped, stratified sample of compact row summaries.
The OpenAI review is benchmark/offline validation only; model labels are never
added to runtime_input.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ACTION_ORDER = {
    "CREATE_ANCHOR",
    "UPDATE_EXISTING_ANCHOR",
    "SPLIT_ANCHOR",
    "CLOSE_ANCHOR",
    "ABSTAIN",
}

RUNTIME_FIELDS = {
    "current_semantic",
    "recent_context_vector",
    "active_anchor_state",
    "candidate_semantic_matrix",
    "candidate_feature_matrix",
    "candidate_mask",
}

FORBIDDEN_RUNTIME_FIELDS = {
    "reference_type",
    "gold_action",
    "candidate_class",
    "target",
    "target_flag",
    "entity_id",
    "track_id",
    "source_id",
    "label_class",
    "benchmark_only",
}


def LoadShellEnv(path: Path) -> Dict[str, str]:
  values: Dict[str, str] = {}
  if not path.exists():
    return values
  for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
      continue
    if line.startswith("export "):
      line = line[len("export ") :].strip()
    key, sep, value = line.partition("=")
    if not sep:
      continue
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
      value = value[1:-1]
    values[key] = value
  return values


def Notify(title: str, message: str, enabled: bool) -> None:
  if not enabled:
    return
  try:
    subprocess.run(
        ["osascript", "-e", f'display notification {json.dumps(message)} with title {json.dumps(title)}'],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
  except Exception:
    pass


def Norm(v: Sequence[float]) -> float:
  return math.sqrt(sum(float(x) * float(x) for x in v))


def Dot(a: Sequence[float], b: Sequence[float]) -> float:
  return sum(float(x) * float(y) for x, y in zip(a, b))


def Cos(a: Sequence[float], b: Sequence[float]) -> float:
  denom = Norm(a) * Norm(b)
  if denom <= 1.0e-12:
    return 0.0
  return Dot(a, b) / denom


def IterRows(input_dir: Path) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
  for split in ("train", "val", "test"):
    path = input_dir / f"aait_ingress_anchor_black_box_{split}.jsonl"
    if not path.exists():
      continue
    with path.open("r", encoding="utf-8") as f:
      for line_no, line in enumerate(f, 1):
        line = line.strip()
        if line:
          yield split, line_no, json.loads(line)


def CandidateStats(row: Dict[str, Any]) -> Dict[str, Any]:
  runtime = row["runtime_input"]
  current = runtime["current_semantic"]
  semantics = runtime["candidate_semantic_matrix"]
  features = runtime["candidate_feature_matrix"]
  mask = runtime["candidate_mask"]
  valid = [i for i, flag in enumerate(mask) if flag]
  rows = []
  for i in valid:
    feat = features[i]
    rows.append(
        {
            "index": i,
            "cosine_to_current": round(Cos(current, semantics[i]), 6),
            "age": feat[0],
            "salience": feat[1],
            "confidence": feat[2],
            "mod_text": feat[3],
            "mod_image": feat[4],
            "mod_audio": feat[5],
            "last_seen_step": feat[6],
        }
    )
  rows.sort(key=lambda r: r["cosine_to_current"], reverse=True)
  return {
      "valid_indices": valid,
      "candidate_count": len(valid),
      "cosine_ranked_candidates": rows,
  }


def CheckRow(split: str, line_no: int, row: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
  failures: List[str] = []
  runtime = row.get("runtime_input", {})
  runtime_keys = set(runtime.keys())
  if runtime_keys != RUNTIME_FIELDS:
    failures.append("runtime_field_set_mismatch")
  if runtime_keys & FORBIDDEN_RUNTIME_FIELDS:
    failures.append("forbidden_runtime_field_present")
  if len(runtime.get("current_semantic", [])) != 1280:
    failures.append("current_semantic_shape")
  if len(runtime.get("recent_context_vector", [])) != 1280:
    failures.append("recent_context_shape")
  if len(runtime.get("active_anchor_state", [])) != 1280:
    failures.append("active_anchor_state_shape")
  if len(runtime.get("candidate_semantic_matrix", [])) != 8:
    failures.append("candidate_semantic_slot_count")
  if len(runtime.get("candidate_feature_matrix", [])) != 8:
    failures.append("candidate_feature_slot_count")
  if len(runtime.get("candidate_mask", [])) != 8:
    failures.append("candidate_mask_slot_count")

  labels = row.get("labels", {})
  action = labels.get("expected_action_label")
  bind = labels.get("expected_bind_index_or_abstain")
  failure_type = row.get("failure_type", "")
  audit = row.get("audit", {})
  stats = CandidateStats(row)
  valid = stats["valid_indices"]
  valid_set = set(valid)
  candidate_count = len(valid)
  if action not in ACTION_ORDER:
    failures.append("unknown_action_label")
  if audit.get("candidate_count") != candidate_count:
    failures.append("audit_candidate_count_mismatch")
  if failure_type and not failure_type.startswith(action.split("_")[0].lower()):
    failures.append("failure_type_action_prefix_mismatch")

  bind_is_int = isinstance(bind, int)
  if action == "UPDATE_EXISTING_ANCHOR":
    if not bind_is_int:
      failures.append("update_bind_not_integer")
    elif bind not in valid_set:
      failures.append("update_bind_not_valid_candidate")
  elif action == "CLOSE_ANCHOR":
    if not bind_is_int:
      failures.append("close_bind_not_integer")
    elif bind not in valid_set:
      failures.append("close_bind_not_valid_candidate")
  else:
    if bind != "abstain":
      failures.append("non_bind_action_expected_abstain")

  if "no_active_candidates" in failure_type and candidate_count != 0:
    failures.append("no_active_slice_has_candidates")
  if failure_type in {"create_no_candidates", "abstain_no_active_candidates"} and bind != "abstain":
    failures.append("no_candidate_slice_does_not_abstain")
  if "target_lower_than_wrong_active" in failure_type or "target_lower_than_stale" in failure_type:
    if not bind_is_int or bind not in valid_set:
      failures.append("hard_update_missing_target_bind")
    else:
      target_cos = next((r["cosine_to_current"] for r in stats["cosine_ranked_candidates"] if r["index"] == bind), None)
      best_other = max((r["cosine_to_current"] for r in stats["cosine_ranked_candidates"] if r["index"] != bind), default=-1.0)
      if target_cos is None or best_other <= target_cos:
        failures.append("hard_update_not_actually_lower_ranked")
  if action == "CLOSE_ANCHOR" and bind_is_int and bind in valid_set:
    feat = runtime["candidate_feature_matrix"][bind]
    if feat[0] < 12.0 or feat[6] > 6.0:
      failures.append("close_bind_not_stale_feature_shape")

  compact = {
      "example_id": row.get("example_id"),
      "split": split,
      "line_no": line_no,
      "action": action,
      "bind": bind,
      "failure_type": failure_type,
      "candidate_count": candidate_count,
      "audit_candidate_order": audit.get("candidate_order"),
      "audit_target_index": audit.get("target_index"),
      "cosine_ranked_candidates": stats["cosine_ranked_candidates"][:8],
  }
  return failures, compact


def SelectGptSample(compacts: Sequence[Dict[str, Any]], max_rows: int, seed: int) -> List[Dict[str, Any]]:
  if max_rows <= 0:
    return []
  grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
  for row in compacts:
    grouped[row["failure_type"]].append(row)
  rng = random.Random(seed)
  selected: List[Dict[str, Any]] = []
  while len(selected) < max_rows and grouped:
    progressed = False
    for key in sorted(list(grouped)):
      rows = grouped[key]
      if not rows:
        grouped.pop(key, None)
        continue
      selected.append(rows.pop(rng.randrange(len(rows))))
      progressed = True
      if len(selected) >= max_rows:
        break
    if not progressed:
      break
  return selected


def ResponseSchema() -> Dict[str, Any]:
  return {
      "type": "json_schema",
      "json_schema": {
          "name": "aait_label_audit_batch",
          "strict": True,
          "schema": {
              "type": "object",
              "additionalProperties": False,
              "properties": {
                  "rows": {
                      "type": "array",
                      "items": {
                          "type": "object",
                          "additionalProperties": False,
                          "properties": {
                              "example_id": {"type": "string"},
                              "label_valid": {"type": "boolean"},
                              "severity": {"type": "string", "enum": ["ok", "warning", "error"]},
                              "corrected_action": {"type": "string"},
                              "corrected_bind": {"type": "string"},
                              "reason": {"type": "string"},
                          },
                          "required": [
                              "example_id",
                              "label_valid",
                              "severity",
                              "corrected_action",
                              "corrected_bind",
                              "reason",
                          ],
                      },
                  }
              },
              "required": ["rows"],
          },
      },
  }


def BuildGptPrompt(rows: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
  system = (
      "You audit AAIT ingress-anchor tensor-pack labels. You only see compact "
      "candidate statistics, not raw runtime tensors or text. Judge whether the "
      "provided action and bind label are procedurally consistent with the "
      "failure_type and candidate statistics. Do not invent entity semantics. "
      "Mark error only for clear contradictions; mark warning for underdetermined "
      "cases that cannot be proven from the stats."
  )
  user = {
      "rules": {
          "UPDATE_EXISTING_ANCHOR": "must bind a valid candidate index",
          "ABSTAIN": "must bind abstain",
          "CREATE_ANCHOR": "must bind abstain",
          "SPLIT_ANCHOR": "must bind abstain",
          "CLOSE_ANCHOR": "should bind the stale/closure candidate when present",
          "hard_update_lower_ranked": "target-lower slices should have some non-target candidate with higher cosine than the bind target",
          "no_active": "no-active slices should have zero candidates and abstain",
      },
      "rows": rows,
  }
  return system, json.dumps(user, separators=(",", ":"))


def CallOpenAI(settings: Dict[str, str], rows: Sequence[Dict[str, Any]], max_retries: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
  system, user = BuildGptPrompt(rows)
  payload = {
      "model": settings["model"],
      "messages": [
          {"role": "system", "content": system},
          {"role": "user", "content": user},
      ],
      "temperature": 0,
      "max_completion_tokens": 2500,
      "response_format": ResponseSchema(),
  }
  body = json.dumps(payload).encode("utf-8")
  url = settings["base_url"].rstrip("/") + "/chat/completions"
  last_error = ""
  for attempt in range(max_retries):
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
      with urllib.request.urlopen(request, timeout=180) as response:
        raw = json.loads(response.read().decode("utf-8"))
      content = raw["choices"][0]["message"]["content"]
      parsed = json.loads(content)
      return parsed.get("rows", []), raw.get("usage", {})
    except urllib.error.HTTPError as exc:
      text = exc.read().decode("utf-8", errors="replace")
      last_error = f"HTTP {exc.code}: {text[:500]}"
      if exc.code in {408, 409, 429, 500, 502, 503, 504} and attempt + 1 < max_retries:
        time.sleep(min(20, 2 ** attempt))
        continue
      break
    except Exception as exc:  # noqa: BLE001
      last_error = str(exc)
      if attempt + 1 < max_retries:
        time.sleep(min(20, 2 ** attempt))
        continue
      break
  raise RuntimeError(f"OpenAI audit failed: {last_error}")


def RunGptAudit(
    sample: Sequence[Dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
  env = dict(os.environ)
  if args.env_file:
    env.update(LoadShellEnv(Path(args.env_file)))
  api_key = env.get("OPENAI_API_KEY", "")
  if not api_key:
    return {
        "enabled": False,
        "skipped_reason": "OPENAI_API_KEY missing; pass --env-file or set environment variable",
        "requested_rows": len(sample),
  }
  settings = {
      "api_key": api_key,
      "base_url": env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
      "model": args.model or env.get("OPENAI_LABEL_MODEL", "gpt-5.4-mini-2026-03-17"),
  }
  results_path = output_dir / "aait_ingress_anchor_black_box_gpt_label_audit.jsonl"
  progress_path = output_dir / "aait_ingress_anchor_black_box_gpt_progress.json"
  original_requested = len(sample)
  reviewed_ids = set()
  usage_totals: Dict[str, int] = collections.Counter()
  severity_counts: Dict[str, int] = collections.Counter()
  disagreements = 0
  if args.resume and results_path.exists():
    with results_path.open("r", encoding="utf-8") as existing:
      for raw in existing:
        if not raw.strip():
          continue
        try:
          row = json.loads(raw)
        except json.JSONDecodeError:
          continue
        audit = row.get("gpt_audit", {})
        example_id = audit.get("example_id") or row.get("input", {}).get("example_id")
        if example_id:
          reviewed_ids.add(example_id)
        severity_counts[audit.get("severity", "unknown")] += 1
        if audit and not audit.get("label_valid", False):
          disagreements += 1
    if progress_path.exists():
      try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        usage_totals.update({k: int(v) for k, v in progress.get("usage", {}).items()})
      except Exception:
        pass
  sample = [row for row in sample if row.get("example_id") not in reviewed_ids]
  reviewed = len(reviewed_ids)
  mode = "a" if args.resume else "w"
  with results_path.open(mode, encoding="utf-8") as out:
    for start in range(0, len(sample), args.batch_size):
      batch = sample[start:start + args.batch_size]
      labels, usage = CallOpenAI(settings, batch, args.max_retries)
      usage_totals.update({k: int(v) for k, v in usage.items() if isinstance(v, int)})
      by_id = {row["example_id"]: row for row in batch}
      for label in labels:
        reviewed += 1
        severity_counts[label.get("severity", "unknown")] += 1
        if not label.get("label_valid", False):
          disagreements += 1
        out.write(json.dumps({"input": by_id.get(label.get("example_id"), {}), "gpt_audit": label}, separators=(",", ":")) + "\n")
      progress = {
          "model": settings["model"],
          "reviewed": reviewed,
          "requested": original_requested,
          "remaining": max(0, original_requested - reviewed),
          "severity_counts": dict(severity_counts),
          "disagreements": disagreements,
          "usage": dict(usage_totals),
      }
      progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
      if args.notify and args.notify_every_batches > 0 and ((start // args.batch_size) + 1) % args.notify_every_batches == 0:
        Notify("AAIT GPT label audit", f"Reviewed {reviewed}/{len(sample)} rows; disagreements={disagreements}", True)
  return {
      "enabled": True,
      "model": settings["model"],
      "reviewed_rows": reviewed,
      "requested_rows": original_requested,
      "remaining_rows": max(0, original_requested - reviewed),
      "resumed": bool(args.resume),
      "result_path": str(results_path),
      "progress_path": str(progress_path),
      "severity_counts": dict(severity_counts),
      "disagreements": disagreements,
      "usage": dict(usage_totals),
  }


def Main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dir", default="build/aait_ingress_anchor_black_box_large")
  parser.add_argument("--output-dir", default="build/aait_ingress_anchor_black_box_label_audit")
  parser.add_argument("--gpt-rows", type=int, default=0, help="Stratified rows to review with GPT. Use 0 for deterministic-only.")
  parser.add_argument("--batch-size", type=int, default=10)
  parser.add_argument("--seed", type=int, default=54086)
  parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
  parser.add_argument("--env-file")
  parser.add_argument("--max-retries", type=int, default=4)
  parser.add_argument("--notify", action="store_true")
  parser.add_argument("--notify-every-batches", type=int, default=5)
  parser.add_argument("--resume", action="store_true", help="Append to an existing GPT audit and skip already-reviewed example ids.")
  args = parser.parse_args()

  input_dir = Path(args.input_dir)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  Notify("AAIT label audit", "Started", args.notify)

  total = 0
  deterministic_failure_counts: Dict[str, int] = collections.Counter()
  action_counts: Dict[str, int] = collections.Counter()
  failure_type_counts: Dict[str, int] = collections.Counter()
  compact_rows: List[Dict[str, Any]] = []
  failures_path = output_dir / "aait_ingress_anchor_black_box_label_disagreements.jsonl"
  with failures_path.open("w", encoding="utf-8") as failures_out:
    for split, line_no, row in IterRows(input_dir):
      total += 1
      labels = row.get("labels", {})
      action_counts[labels.get("expected_action_label", "missing")] += 1
      failure_type_counts[row.get("failure_type", "missing")] += 1
      failures, compact = CheckRow(split, line_no, row)
      compact_rows.append(compact)
      for failure in failures:
        deterministic_failure_counts[failure] += 1
      if failures:
        failures_out.write(
            json.dumps(
                {
                    "example_id": row.get("example_id"),
                    "split": split,
                    "line_no": line_no,
                    "failures": failures,
                    "compact": compact,
                },
                separators=(",", ":"),
            )
            + "\n"
        )

  sample = SelectGptSample(compact_rows, args.gpt_rows, args.seed)
  gpt_result = RunGptAudit(sample, output_dir, args) if args.gpt_rows > 0 else {
      "enabled": False,
      "skipped_reason": "--gpt-rows was 0",
      "requested_rows": 0,
  }

  summary = {
      "input_dir": str(input_dir),
      "row_count": total,
      "action_distribution": dict(action_counts),
      "failure_type_distribution": dict(failure_type_counts),
      "deterministic_failure_counts": dict(deterministic_failure_counts),
      "deterministic_failed_rows": sum(1 for _ in failures_path.open("r", encoding="utf-8")),
      "deterministic_disagreement_path": str(failures_path),
      "gpt_audit": gpt_result,
      "label_quality_status": (
          "needs_review"
          if deterministic_failure_counts or (gpt_result.get("enabled") and gpt_result.get("disagreements", 0) > 0)
          else (
              "passed_gpt_sample_audit_with_warnings"
              if gpt_result.get("enabled") and gpt_result.get("severity_counts", {}).get("warning", 0) > 0
              else (
                  "passed_gpt_sample_audit"
                  if gpt_result.get("enabled")
                  else "passed_structural_label_audit"
              )
          )
      ),
      "important_limitation": (
          "GPT review sees compact tensor-derived statistics and procedural labels, not source text/entity-track ground truth. "
          "It validates consistency of the generated label contract, not real-world referent semantics."
      ),
  }
  (output_dir / "aait_ingress_anchor_black_box_label_audit.json").write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  Notify("AAIT label audit", f"Finished {total} rows; deterministic failures={sum(deterministic_failure_counts.values())}", args.notify)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  Main()
