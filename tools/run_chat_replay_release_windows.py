#!/usr/bin/env python3
"""Plan or run the fixed multi-window Julie release protocol.

This wrapper keeps the mixed-media claim reproducible when no single Julie slice
contains all media kinds. It delegates every window to
tools/run_julie_release_protocol.py, so Cortext behavior remains the production
path used by the single-window release runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = pathlib.Path.home() / "Documents/Memory/Julie"
DEFAULT_JUDGE_MODEL = "gemma4:12b-it-qat"
DEFAULT_PROTOCOL_SPEC = REPO_ROOT / "tools/julie_release_protocol_spec.json"
CORTEXT_RELEASE_ENV_ALLOWLIST = {
    "CORTEXT_JUDGE_BASE_URL",
    "CORTEXT_OLLAMA_BASE_URL",
}
HOSTED_PROVIDER_ENV_MARKERS = (
    "OPENAI",
    "ANTHROPIC",
    "GEMINI",
    "GOOGLE_API",
    "GOOGLE_GENAI",
    "VERTEX",
    "AZURE_OPENAI",
    "TOGETHER",
    "MISTRAL",
    "COHERE",
    "GROQ",
)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def git_provenance() -> dict[str, Any]:
    status = git_output(["status", "--porcelain=v1"])
    return {
        "commit": git_output(["rev-parse", "HEAD"]) or "unknown",
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dirty_path_count": len([line for line in status.splitlines() if line.strip()]),
    }


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_ref(path: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_json_if_valid(path: pathlib.Path) -> dict[str, Any]:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_protocol_spec(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"failed to read protocol spec: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"protocol spec must be a JSON object: {path}")
    if payload.get("schema") != "cortext_julie_release_protocol_spec_v1":
        raise RuntimeError(f"unexpected protocol spec schema: {payload.get('schema')}")
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        raise RuntimeError("protocol spec must define at least one release window")
    for window in windows:
        if not isinstance(window, dict):
            raise RuntimeError("protocol spec windows must be JSON objects")
        validate_window(window)
    return payload


def normalize_window(window: dict[str, Any]) -> dict[str, Any]:
    out = dict(window)
    out["name"] = validate_window_name(str(out["name"]))
    out["skip_messages"] = int(out["skip_messages"])
    out["max_messages"] = int(out["max_messages"])
    out["media_limit"] = int(out["media_limit"])
    out["probe_stride"] = int(out.get("probe_stride", 0))
    out["warmup_events"] = int(out.get("warmup_events", 0))
    out["min_probe_rows_after_benchmark"] = int(
        out.get("min_probe_rows_after_benchmark", 0)
    )
    out["required_media_modalities"] = parse_modalities(
        ",".join(str(item) for item in out.get("required_media_modalities", []))
    )
    out["purpose"] = str(out.get("purpose", ""))
    return out


def normalize_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_window(window) for window in windows]


def is_hosted_provider_env_key(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in HOSTED_PROVIDER_ENV_MARKERS)


def redacted_env_value(key: str, value: str) -> str:
    blocked = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    if any(part in key.upper() for part in blocked):
        return "<redacted>"
    return value


def is_cortext_behavior_env_key(key: str) -> bool:
    return key.startswith("CORTEXT_") and key not in CORTEXT_RELEASE_ENV_ALLOWLIST


def cortext_behavior_env(env: dict[str, str]) -> dict[str, str]:
    return {
        key: redacted_env_value(key, value)
        for key, value in sorted(env.items())
        if is_cortext_behavior_env_key(key)
    }


def cortext_behavior_env_guard(env: dict[str, str]) -> dict[str, Any]:
    leaked = cortext_behavior_env(env)
    return {
        "mode": "fail_closed",
        "policy": (
            "Julie release windows reject ambient CORTEXT_* variables except "
            "local judge endpoint metadata before launching production Cortext."
        ),
        "allowed_cortext_env_keys": sorted(CORTEXT_RELEASE_ENV_ALLOWLIST),
        "leakage_detected": bool(leaked),
        "leaked_cortext_behavior_env": leaked,
    }


def ensure_no_ambient_cortext_behavior_env(env: dict[str, str]) -> None:
    guard = cortext_behavior_env_guard(env)
    if guard["leakage_detected"]:
        raise RuntimeError(
            "refusing to launch Julie release windows with ambient CORTEXT_* "
            "behavior variables: "
            + json.dumps(guard, sort_keys=True)
        )


def sanitized_subprocess_env() -> tuple[dict[str, str], list[str]]:
    ensure_no_ambient_cortext_behavior_env(dict(os.environ))
    env = dict(os.environ)
    stripped = sorted(key for key in env if is_hosted_provider_env_key(key))
    for key in stripped:
        env.pop(key, None)
    return env, stripped


def validate_window_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise RuntimeError(
            "window names must contain only letters, digits, '_', '-', and '.', "
            f"and cannot be empty: {name!r}"
        )
    return name


def parse_modalities(raw: str) -> list[str]:
    modalities: list[str] = []
    for part in re.split(r"[,+]", raw):
        value = part.strip().lower()
        if not value:
            continue
        if value not in {"image", "audio", "video"}:
            raise RuntimeError(
                "window required modalities must be image, audio, or video; "
                f"got {value!r}"
            )
        if value not in modalities:
            modalities.append(value)
    return modalities


def parse_window_spec(spec: str) -> dict[str, Any]:
    parts = spec.split(":", 4)
    if len(parts) != 5:
        raise RuntimeError(
            "--window must be name:skip_messages:max_messages:media_limit:modalities"
        )
    name, skip, max_messages, media_limit, modalities = parts
    try:
        window = {
            "name": validate_window_name(name),
            "skip_messages": int(skip),
            "max_messages": int(max_messages),
            "media_limit": int(media_limit),
            "required_media_modalities": parse_modalities(modalities),
            "purpose": "custom fixed release window",
        }
    except ValueError as exc:
        raise RuntimeError(f"invalid numeric value in --window {spec!r}") from exc
    # probe_stride / warmup_events / min_probe_rows_after_benchmark are filled
    # from the CLI flags in selected_windows(); hardcoding them here silently
    # discarded those flags for custom windows. Validation happens there too,
    # after the defaults are applied.
    return window


def validate_window(window: dict[str, Any]) -> None:
    validate_window_name(str(window["name"]))
    if int(window["skip_messages"]) < 0:
        raise RuntimeError(f"window {window['name']} skip_messages must be >= 0")
    if int(window["max_messages"]) <= 0:
        raise RuntimeError(f"window {window['name']} max_messages must be > 0")
    if int(window["media_limit"]) < 0:
        raise RuntimeError(f"window {window['name']} media_limit must be >= 0")
    if int(window.get("probe_stride", 0)) <= 0:
        raise RuntimeError(f"window {window['name']} probe_stride must be > 0")
    if int(window.get("warmup_events", -1)) < 0:
        raise RuntimeError(f"window {window['name']} warmup_events must be >= 0")
    if int(window.get("min_probe_rows_after_benchmark", -1)) < 0:
        raise RuntimeError(
            f"window {window['name']} min_probe_rows_after_benchmark must be >= 0"
        )
    modalities = window.get("required_media_modalities", [])
    if not isinstance(modalities, list):
        raise RuntimeError(f"window {window['name']} modalities must be a list")
    parse_modalities(",".join(str(item) for item in modalities))


def selected_windows(
    args: argparse.Namespace,
    protocol_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    windows = (
        [parse_window_spec(item) for item in args.window]
        if args.window
        else [dict(window) for window in protocol_spec.get("windows", [])]
    )
    seen: set[str] = set()
    for window in windows:
        name = str(window["name"])
        if name in seen:
            raise RuntimeError(f"duplicate release window name: {name}")
        seen.add(name)
        window.setdefault("probe_stride", args.probe_stride)
        window.setdefault("warmup_events", args.warmup_events)
        window.setdefault(
            "min_probe_rows_after_benchmark",
            args.min_probe_rows_after_benchmark,
        )
        validate_window(window)
        window["required_media_modalities"] = parse_modalities(
            ",".join(str(item) for item in window.get("required_media_modalities", []))
        )
    return normalize_windows(windows)


def add_if(flag: bool, cmd: list[str], value: str) -> None:
    if flag:
        cmd.append(value)


def build_window_command(
    args: argparse.Namespace,
    window: dict[str, Any],
    out_dir: pathlib.Path,
    *,
    preflight_only: bool,
) -> list[str]:
    cmd = [
        args.python,
        str(REPO_ROOT / "tools/run_julie_release_protocol.py"),
        "--out-dir",
        str(out_dir),
        "--benchmark",
        str(args.benchmark),
        "--input-dir",
        str(args.input_dir),
        "--skip-messages",
        str(window["skip_messages"]),
        "--max-messages",
        str(window["max_messages"]),
        "--media-limit",
        str(window["media_limit"]),
        "--require-media-modalities",
        ",".join(window.get("required_media_modalities", [])),
        "--probe-stride",
        str(window.get("probe_stride", args.probe_stride)),
        "--warmup-events",
        str(window.get("warmup_events", args.warmup_events)),
        "--rag-top-k",
        str(args.rag_top_k),
        "--active-history-token-budget",
        str(args.active_history_token_budget),
        "--judge-model",
        args.judge_model,
        "--ollama-base-url",
        args.ollama_base_url,
        "--judge-repetitions",
        str(args.judge_repetitions),
        "--judge-seed",
        str(args.judge_seed),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--judge-timeout-s",
        str(args.judge_timeout_s),
        "--judge-context-window-tokens",
        str(args.judge_context_window_tokens),
        "--judge-packet-item-limit",
        str(args.judge_packet_item_limit),
        "--judge-media-smoke",
        str(args.judge_media_smoke),
        "--early-judge",
        args.early_judge,
        "--early-judge-milestones",
        args.early_judge_milestones,
        "--early-judge-periodic-stride",
        str(args.early_judge_periodic_stride),
        "--early-judge-repetitions",
        str(args.early_judge_repetitions),
        "--early-confirm-fail-repetitions",
        str(args.early_confirm_fail_repetitions),
        "--early-judge-timeout-s",
        str(args.early_judge_timeout_s),
        "--early-judge-context-window-tokens",
        str(args.early_judge_context_window_tokens),
        "--early-judge-packet-item-limit",
        str(args.early_judge_packet_item_limit),
        "--early-min-cortext-quality-delta-vs-rag",
        str(args.early_min_cortext_quality_delta_vs_rag),
        "--min-probe-rows-after-benchmark",
        str(
            window.get(
                "min_probe_rows_after_benchmark",
                args.min_probe_rows_after_benchmark,
            )
        ),
    ]
    add_if(args.force, cmd, "--force")
    add_if(args.run_ablations, cmd, "--run-ablations")
    add_if(args.skip_final_judge, cmd, "--skip-final-judge")
    add_if(args.skip_initial_report, cmd, "--skip-initial-report")
    add_if(args.skip_human_label_sample, cmd, "--skip-human-label-sample")
    add_if(args.require_final_report_pass, cmd, "--require-final-report-pass")
    add_if(args.no_kill_on_early_fail, cmd, "--no-kill-on-early-fail")
    if not args.early_quality_gate_requires_rag_pressure:
        cmd.append("--no-early-quality-gate-requires-rag-pressure")
    if preflight_only:
        cmd.append("--preflight-only")
    return cmd


def window_artifacts(out_dir: pathlib.Path) -> dict[str, Any]:
    names = [
        "preflight_report.json",
        "command_manifest.json",
        "summary.json",
        "summary_gate.json",
        "judge_gemma4_12b_local.json",
        "release_protocol_report_initial.json",
        "release_protocol_report_final.json",
        "early_failure_report.json",
        "ablation_plan.json",
    ]
    return {name: artifact_ref(out_dir / name) for name in names}


def judge_media_smoke_needs_generation(
    judge_media_smoke: pathlib.Path,
    judge_model: str,
) -> bool:
    if not judge_media_smoke.exists():
        return True
    try:
        smoke = json.loads(judge_media_smoke.read_text())
    except Exception:
        return True
    if smoke.get("schema") != "cortext_local_ollama_judge_media_smoke_v1":
        return True
    if smoke.get("private_data_used") is not False:
        return True
    if smoke.get("selected_release_judge_model") != judge_model:
        return True
    selected = (smoke.get("results") or {}).get(judge_model, {})
    if not isinstance(selected, dict):
        return True
    image = selected.get("image", {})
    audio = selected.get("audio", {})
    image_parsed = image.get("parsed", {}) if isinstance(image, dict) else {}
    audio_parsed = audio.get("parsed", {}) if isinstance(audio, dict) else {}
    return not (
        isinstance(image_parsed, dict)
        and isinstance(audio_parsed, dict)
        and image_parsed.get("image_seen") is True
        and audio_parsed.get("audio_seen") is True
    )


def ensure_shared_judge_media_smoke(
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    if not judge_media_smoke_needs_generation(args.judge_media_smoke, args.judge_model):
        return
    cmd = [
        args.python,
        str(REPO_ROOT / "tools/smoke_ollama_judge_media.py"),
        "--out",
        str(args.judge_media_smoke),
        "--model",
        args.judge_model,
        "--ollama-base-url",
        args.ollama_base_url,
        "--timeout-s",
        str(args.judge_timeout_s),
        "--num-ctx",
        str(args.judge_context_window_tokens),
    ]
    args.judge_media_smoke.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.base_dir / "judge_media_smoke_pipeline.log"
    print(f"[windows] judge media smoke: {shlex.join(cmd)}", flush=True)
    with log_path.open("w") as log:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if judge_media_smoke_needs_generation(args.judge_media_smoke, args.judge_model):
        raise RuntimeError(f"judge media smoke did not pass: {args.judge_media_smoke}")


def collect_window_status(name: str, out_dir: pathlib.Path, exit_code: int | None) -> dict[str, Any]:
    preflight = load_json_if_valid(out_dir / "preflight_report.json")
    summary_gate = load_json_if_valid(out_dir / "summary_gate.json")
    judge = load_json_if_valid(out_dir / "judge_gemma4_12b_local.json")
    initial_report = load_json_if_valid(out_dir / "release_protocol_report_initial.json")
    return {
        "name": name,
        "out_dir": str(out_dir),
        "exit_code": exit_code,
        "preflight_status": preflight.get("status", ""),
        "summary_gate_status": summary_gate.get("overall_status", ""),
        "judge_probe_count": judge.get("probe_count"),
        "judge_repetitions": judge.get("judge_repetitions")
        or (judge.get("protocol") or {}).get("judge_repetitions"),
        "initial_release_gate_status": (
            (initial_report.get("release_gate") or {}).get("overall_status")
            if isinstance(initial_report.get("release_gate"), dict)
            else ""
        ),
        "artifacts": window_artifacts(out_dir),
    }


def write_protocol_manifest(
    args: argparse.Namespace,
    protocol_spec: dict[str, Any],
    windows: list[dict[str, Any]],
    commands: dict[str, list[str]],
    stripped_env_keys: list[str],
) -> None:
    spec_windows = normalize_windows(
        [dict(window) for window in protocol_spec.get("windows", [])]
    )
    cortext_spec = protocol_spec.get("cortext", {})
    baselines_spec = protocol_spec.get("baselines", {})
    rag_spec = baselines_spec.get("traditional_chat_rag", {})
    judge_spec = protocol_spec.get("judge", {})
    ablation_names = [
        str(item.get("name"))
        for item in protocol_spec.get("ablations", [])
        if isinstance(item, dict) and item.get("name")
    ]
    spec_ref = artifact_ref(args.protocol_spec)
    compliance = {
        "schema": "cortext_julie_release_protocol_compliance_v1",
        "protocol_spec_schema": protocol_spec.get("schema"),
        "protocol_spec_version": protocol_spec.get("version"),
        "protocol_spec_sha256": spec_ref["sha256"],
        "custom_window_override": bool(args.window),
        "windows_match_spec": normalize_windows(windows) == spec_windows,
        "knobs_match_spec": cortext_spec.get("knobs")
        == {"focus": 0.5, "sensitivity": 0.5, "stability": 0.5},
        "daily_consolidation_matches_spec": cortext_spec.get("daily_consolidation")
        is True,
        "deep_consolidation_matches_spec": cortext_spec.get("deep_consolidation")
        is True,
        "rag_top_k_matches_spec": args.rag_top_k == int(rag_spec.get("rag_top_k", 0)),
        "active_history_token_budget_matches_spec": (
            args.active_history_token_budget
            == int(rag_spec.get("active_history_token_budget", 0))
        ),
        "judge_model_matches_spec": args.judge_model == judge_spec.get("model"),
        "judge_repetitions_within_spec": (
            int(judge_spec.get("repetitions_min", 0))
            <= args.judge_repetitions
            <= int(judge_spec.get("repetitions_max", 0))
        ),
        "required_ablation_names": ablation_names,
    }
    compliance["overall_match"] = all(
        bool(value)
        for key, value in compliance.items()
        if key.endswith("_spec") or key.endswith("_match")
        or key.endswith("_matches_spec")
        or key == "judge_repetitions_within_spec"
    ) and not compliance["custom_window_override"]
    payload = {
        "schema": "cortext_julie_release_windows_manifest_v1",
        "created_at_utc": utc_now(),
        "privacy": "private local artifact; contains paths and commands, not message text",
        "git": git_provenance(),
        "protocol_spec": spec_ref,
        "protocol_spec_payload_sha256": canonical_json_sha256(protocol_spec),
        "protocol_spec_compliance": compliance,
        "fixed_protocol": {
            "window_count": len(windows),
            "windows": windows,
            "knobs": cortext_spec.get("knobs", {}),
            "daily_consolidation": bool(cortext_spec.get("daily_consolidation")),
            "deep_consolidation": bool(cortext_spec.get("deep_consolidation")),
            "no_eval_specific_cortext_behavior": bool(
                cortext_spec.get("no_eval_specific_cortext_behavior")
            ),
            "normal_rag": rag_spec.get("description", ""),
            "normal_rag_retrieval": rag_spec.get("normal_rag_retrieval", ""),
            "normal_rag_baseline_modality": rag_spec.get("modality", ""),
            "rag_top_k": args.rag_top_k,
            "active_history_token_budget": args.active_history_token_budget,
            "judge_provider": judge_spec.get("provider", ""),
            "judge_model": args.judge_model,
            "judge_repetitions": args.judge_repetitions,
            "blind_packets": bool(judge_spec.get("blind_packets")),
            "randomize_packet_order": bool(judge_spec.get("randomize_packet_order")),
            "packet_aliases": judge_spec.get("packet_aliases", []),
            "required_ablations": ablation_names,
        },
        "hosted_provider_env_policy": (
            "hosted-provider variables are stripped before launching each "
            "single-window runner"
        ),
        "ambient_cortext_behavior_env_guard": cortext_behavior_env_guard(
            dict(os.environ)
        ),
        "ambient_cortext_behavior_env_policy": (
            "ambient parent CORTEXT_* behavior variables are rejected; "
            "single-window behavior must come from the fixed command protocol"
        ),
        "stripped_hosted_provider_env_keys": stripped_env_keys,
        "commands": {name: shlex.join(command) for name, command in commands.items()},
    }
    write_json(args.base_dir / "release_windows_manifest.json", payload)


def build_aggregate_report_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        str(REPO_ROOT / "tools/julie_release_windows_report.py"),
        "--base-dir",
        str(args.base_dir),
        "--manifest",
        str(args.base_dir / "release_windows_manifest.json"),
        "--protocol-spec",
        str(args.protocol_spec),
        "--status",
        str(args.base_dir / "release_windows_status.json"),
        "--out",
        str(args.aggregate_report),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--seed",
        str(args.judge_seed),
    ]


def run_aggregate_report(
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    cmd = build_aggregate_report_command(args)
    log_path = args.base_dir / "release_windows_report_pipeline.log"
    print(f"[windows] aggregate report: {shlex.join(cmd)}", flush=True)
    with log_path.open("w") as log:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "command": shlex.join(cmd),
        "exit_code": completed.returncode,
        "log": artifact_ref(log_path),
        "report": artifact_ref(args.aggregate_report),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=pathlib.Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--aggregate-report", type=pathlib.Path)
    parser.add_argument(
        "--protocol-spec",
        type=pathlib.Path,
        default=DEFAULT_PROTOCOL_SPEC,
        help="Frozen public-safe Julie release protocol spec JSON.",
    )
    parser.add_argument(
        "--benchmark",
        type=pathlib.Path,
        default=REPO_ROOT / "build/examples/benchmark/cortext_julie_live_run",
    )
    parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help=(
            "Custom fixed window: "
            "name:skip_messages:max_messages:media_limit:modalities. "
            "Modalities can be comma- or plus-separated."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true", help="Run every fixed window sequentially.")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--probe-stride", type=int, default=25)
    parser.add_argument("--warmup-events", type=int, default=200)
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--active-history-token-budget", type=int, default=8000)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--judge-repetitions", type=int, default=3)
    parser.add_argument("--judge-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--judge-timeout-s", type=int, default=420)
    parser.add_argument("--judge-context-window-tokens", type=int, default=32768)
    parser.add_argument("--judge-packet-item-limit", type=int, default=-1)
    parser.add_argument("--judge-media-smoke", type=pathlib.Path)
    parser.add_argument("--early-judge", choices=["on", "off"], default="on")
    parser.add_argument(
        "--early-judge-milestones",
        default="1,2,3,4,5,6,7,8,9,10,11,12,16",
    )
    parser.add_argument("--early-judge-periodic-stride", type=int, default=1)
    parser.add_argument("--early-judge-repetitions", type=int, default=1)
    parser.add_argument("--early-confirm-fail-repetitions", type=int, default=3)
    parser.add_argument("--early-judge-timeout-s", type=int, default=180)
    parser.add_argument("--early-judge-context-window-tokens", type=int, default=32768)
    parser.add_argument("--early-judge-packet-item-limit", type=int, default=256)
    parser.add_argument("--early-min-cortext-quality-delta-vs-rag", type=float, default=-0.50)
    parser.add_argument(
        "--early-quality-gate-requires-rag-pressure",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-probe-rows-after-benchmark", type=int, default=30)
    parser.add_argument("--no-kill-on-early-fail", action="store_true")
    parser.add_argument("--skip-final-judge", action="store_true")
    parser.add_argument("--skip-initial-report", action="store_true")
    parser.add_argument("--skip-human-label-sample", action="store_true")
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--require-final-report-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.base_dir = args.base_dir.resolve()
    args.protocol_spec = args.protocol_spec.resolve()
    args.benchmark = args.benchmark.resolve()
    args.input_dir = args.input_dir.resolve()
    args.judge_media_smoke = (
        args.judge_media_smoke.resolve()
        if args.judge_media_smoke is not None
        else args.base_dir / "judge_media_smoke_ollama.json"
    )
    args.aggregate_report = (
        args.aggregate_report.resolve()
        if args.aggregate_report is not None
        else args.base_dir / "release_windows_report.json"
    )

    if args.judge_repetitions < 3:
        raise RuntimeError("--judge-repetitions must be at least 3")
    if args.judge_packet_item_limit == 0 or args.judge_packet_item_limit < -1:
        raise RuntimeError("--judge-packet-item-limit must be -1 or positive")
    if args.min_probe_rows_after_benchmark < 0:
        raise RuntimeError("--min-probe-rows-after-benchmark must be non-negative")

    protocol_spec = load_protocol_spec(args.protocol_spec)
    windows = selected_windows(args, protocol_spec)
    args.base_dir.mkdir(parents=True, exist_ok=True)
    env, stripped_env_keys = sanitized_subprocess_env()
    mode = "preflight" if args.preflight_only else "run" if args.run else "plan"
    commands = {
        str(window["name"]): build_window_command(
            args,
            window,
            args.base_dir / str(window["name"]),
            preflight_only=args.preflight_only,
        )
        for window in windows
    }
    write_protocol_manifest(args, protocol_spec, windows, commands, stripped_env_keys)

    statuses: list[dict[str, Any]] = []
    overall_status = "planned"
    if mode == "plan":
        for window in windows:
            name = str(window["name"])
            out_dir = args.base_dir / name
            statuses.append(collect_window_status(name, out_dir, None))
    else:
        overall_status = "pass"
        ensure_shared_judge_media_smoke(args, env)
        for window in windows:
            name = str(window["name"])
            out_dir = args.base_dir / name
            command = commands[name]
            log_path = args.base_dir / f"{name}_{mode}.log"
            print(f"[windows] {mode}: {name}: {shlex.join(command)}", flush=True)
            with log_path.open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            statuses.append(collect_window_status(name, out_dir, completed.returncode))
            if completed.returncode != 0:
                overall_status = "fail"
                if not args.keep_going:
                    break

    status_payload = {
        "schema": "cortext_julie_release_windows_status_v1",
        "created_at_utc": utc_now(),
        "privacy": "private local artifact; aggregate status and artifact hashes only",
        "mode": mode,
        "overall_status": overall_status,
        "shared_judge_media_smoke": artifact_ref(args.judge_media_smoke),
        "windows": statuses,
    }
    write_json(args.base_dir / "release_windows_status.json", status_payload)
    status_payload["aggregate_report"] = run_aggregate_report(args, env)
    if status_payload["aggregate_report"]["exit_code"] != 0:
        overall_status = "fail"
        status_payload["overall_status"] = overall_status
    write_json(args.base_dir / "release_windows_status.json", status_payload)
    print(json.dumps(status_payload, indent=2), flush=True)
    return 0 if overall_status in {"planned", "pass"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
