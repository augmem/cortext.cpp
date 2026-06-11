#!/usr/bin/env python3
"""Run the chat-replay release replay with streamed early local judging.

This is a reproducibility wrapper around the production benchmark and the
existing local judge/report tools. It does not add eval-specific Cortext
behavior. Its purpose is to avoid wasting hours on a replay that is already
failing early frozen probes.
"""

from __future__ import annotations

from chat_replay_corpus import discover_transcript

import argparse
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
import urllib.parse
import urllib.request


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gemma4:12b-it-qat"
LOCAL_JUDGE_HOSTS = {"localhost", "127.0.0.1", "::1"}
LOCAL_PROVIDER_ENV_KEYS = {
    "CORTEXT_JUDGE_BASE_URL",
    "CORTEXT_OLLAMA_BASE_URL",
    "LOCAL_JUDGE_BASE_URL",
}
CORTEXT_RELEASE_ENV_ALLOWLIST = {
    "CORTEXT_JUDGE_BASE_URL",
    "CORTEXT_OLLAMA_BASE_URL",
    # Observability only: per-summary label/fact admission counters.
    "CORTEXT_STM_LTM_AUDIT",
}
LOCAL_PROVIDER_ENV_PREFIXES = ("OLLAMA_",)
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
MEDIA_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".gif",
    ".tiff",
    ".mov",
    ".mp4",
    ".3gp",
    ".m4a",
    ".wav",
    ".mp3",
}
MEDIA_KIND_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".heic", ".gif", ".tiff"},
    "video": {".mov", ".mp4", ".3gp"},
    "audio": {".m4a", ".wav", ".mp3"},
}
DISALLOWED_RELEASE_BENCHMARK_FLAGS = [
    "--profile-probes-only",
    "--checkpoint-eval-only",
    "--checkpoint-after-timestamp",
    "--checkpoint-query-count",
    "--checkpoint-query-stride",
    "--checkpoint-query-days",
    "--checkpoint-queries-per-day",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_contains(path: pathlib.Path, needle: bytes) -> bool:
    overlap = max(0, len(needle) - 1)
    previous = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            haystack = previous + chunk
            if needle in haystack:
                return True
            previous = haystack[-overlap:] if overlap else b""
    return False


def executable_artifact(command: str) -> dict:
    parts = shlex.split(command)
    raw = parts[0] if parts else ""
    resolved = ""
    if raw:
        candidate = pathlib.Path(raw)
        if candidate.exists() or "/" in raw:
            resolved = str(candidate.resolve())
        else:
            resolved = shutil.which(raw) or ""
    path = pathlib.Path(resolved) if resolved else None
    exists = bool(path and path.exists() and path.is_file())
    return {
        "command": raw,
        "path": str(path) if path else "",
        "exists": exists,
        "sha256": file_sha256(path) if path and exists else "",
        "bytes": path.stat().st_size if path and exists else 0,
        "mtime_ns": path.stat().st_mtime_ns if path and exists else 0,
    }


def executable_supports_probe_stream(artifact: dict) -> bool:
    path = pathlib.Path(str(artifact.get("path", "")))
    return bool(
        path.exists()
        and path.is_file()
        and file_contains(path, b".probes.jsonl")
        and file_contains(path, b"native probe rows appended")
    )


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_json_if_valid(path: pathlib.Path) -> dict:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def command_sha256(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def command_has_flag(parts: list[str], flag: str) -> bool:
    return flag in parts or any(part.startswith(flag + "=") for part in parts)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def git_hash_output(args: list[str]) -> dict:
    text = git_output(args)
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "bytes": len(text.encode("utf-8")),
        "line_count": len(text.splitlines()),
    }


def git_provenance() -> dict:
    status = git_output(["status", "--porcelain=v1"])
    status_entries = sorted(
        line[3:] if len(line) > 3 else line for line in status.splitlines()
    )
    untracked = sorted(
        line
        for line in git_output(["ls-files", "--others", "--exclude-standard"]).splitlines()
        if line
    )
    staged = git_hash_output(["diff", "--cached", "--binary", "--no-ext-diff"])
    unstaged = git_hash_output(["diff", "--binary", "--no-ext-diff"])
    submodule = git_hash_output(["diff", "--submodule=short", "--no-ext-diff"])
    fingerprint_payload = {
        "commit": git_output(["rev-parse", "HEAD"]) or "unknown",
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status_entries": status_entries,
        "staged_diff_sha256": staged["sha256"],
        "unstaged_diff_sha256": unstaged["sha256"],
        "submodule_diff_sha256": submodule["sha256"],
        "untracked_paths_sha256": hashlib.sha256(
            "\n".join(untracked).encode("utf-8")
        ).hexdigest(),
    }
    return {
        "commit": fingerprint_payload["commit"],
        "dirty": bool(status),
        "status_sha256": fingerprint_payload["status_sha256"],
        "dirty_path_count": len(status_entries),
        "worktree_manifest_sha256": hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def source_input_fingerprint(input_dir: pathlib.Path) -> dict:
    body: dict = {
        "schema": "cortext_chat_replay_source_input_fingerprint_v1",
        "privacy": (
            "private local provenance: records content hashes and aggregate "
            "file metadata only, never message text or media bytes"
        ),
        "input_dir": str(input_dir),
        "exists": input_dir.exists(),
        "recursive": True,
        "file_count": 0,
        "readable_file_count": 0,
        "symlink_count": 0,
        "total_bytes": 0,
        "extension_counts": {},
        "transcript_present": False,
        "transcript_sha256": "",
        "manifest_sha256": "",
        "unreadable_files": 0,
    }
    if not input_dir.exists() or not input_dir.is_dir():
        return body

    private_entries: list[dict] = []
    extension_counts: dict[str, int] = {}
    for path in sorted(input_dir.rglob("*"), key=lambda item: str(item.relative_to(input_dir))):
        if path.is_dir():
            continue
        try:
            relative_path = path.relative_to(input_dir)
        except ValueError:
            relative_path = pathlib.Path(path.name)
        suffix = path.suffix.lower()
        extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
        entry: dict = {
            "relative_path_sha256": sha256_text(str(relative_path)),
            "suffix": suffix,
            "is_symlink": path.is_symlink(),
            "readable": False,
            "size_bytes": None,
            "sha256": "",
        }
        body["file_count"] += 1
        if path.is_symlink():
            body["symlink_count"] += 1
        try:
            stat = path.stat()
            entry["size_bytes"] = stat.st_size
            entry["sha256"] = file_sha256(path)
            entry["readable"] = True
            body["readable_file_count"] += 1
            body["total_bytes"] += stat.st_size
            if len(relative_path.parts) == 1 and relative_path.suffix == ".txt":
                body["transcript_present"] = True
                body["transcript_sha256"] = entry["sha256"]
        except OSError as exc:
            body["unreadable_files"] += 1
            entry["error"] = exc.__class__.__name__
        private_entries.append(entry)

    body["extension_counts"] = extension_counts
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema": body["schema"],
                "files": private_entries,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return body


def redacted_env_value(key: str, value: str) -> str:
    blocked = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    if any(part in key.upper() for part in blocked):
        return "<redacted>"
    return value


def is_hosted_provider_env_key(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in HOSTED_PROVIDER_ENV_MARKERS)


def is_cortext_behavior_env_key(key: str) -> bool:
    return key.startswith("CORTEXT_") and key not in CORTEXT_RELEASE_ENV_ALLOWLIST


def cortext_behavior_env(env: dict[str, str]) -> dict[str, str]:
    return {
        key: redacted_env_value(key, value)
        for key, value in sorted(env.items())
        if is_cortext_behavior_env_key(key)
    }


def cortext_behavior_env_guard(env: dict[str, str]) -> dict:
    leaked = cortext_behavior_env(env)
    return {
        "mode": "fail_closed",
        "policy": (
            "chat-replay release replay rejects CORTEXT_* variables except local "
            "judge endpoint metadata before launching production Cortext."
        ),
        "allowed_cortext_env_keys": sorted(CORTEXT_RELEASE_ENV_ALLOWLIST),
        "leakage_detected": bool(leaked),
        "leaked_cortext_behavior_env": leaked,
    }


def sanitized_subprocess_env() -> tuple[dict[str, str], list[str]]:
    env = dict(os.environ)
    stripped = sorted(key for key in env if is_hosted_provider_env_key(key))
    for key in stripped:
        env.pop(key, None)
    # Observability, not behavior: every protocol run records the per-summary
    # label admission/rejection counters (stm_ltm_relabel_audit) so label and
    # fact decisions are explainable after the fact.
    env["CORTEXT_STM_LTM_AUDIT"] = "1"
    return env, stripped


def benchmark_environment_snapshot(
    pid: int,
    benchmark_cmd: list[str],
    env: dict[str, str],
    stripped_hosted_provider_env_keys: list[str],
) -> dict:
    cortext_behavior_env = {}
    hosted_provider_behavior_env = {}
    local_provider_env = {}
    selected_env = {}
    for key, value in sorted(env.items()):
        if not (
            key.startswith("CORTEXT_")
            or key.startswith(LOCAL_PROVIDER_ENV_PREFIXES)
            or is_hosted_provider_env_key(key)
        ):
            continue
        safe_value = redacted_env_value(key, value)
        selected_env[key] = safe_value
        if is_cortext_behavior_env_key(key):
            cortext_behavior_env[key] = safe_value
        elif key in LOCAL_PROVIDER_ENV_KEYS or key.startswith(LOCAL_PROVIDER_ENV_PREFIXES):
            local_provider_env[key] = safe_value
        if is_hosted_provider_env_key(key):
            hosted_provider_behavior_env[key] = safe_value

    process_name = pathlib.Path(benchmark_cmd[0]).name if benchmark_cmd else ""
    return {
        "schema": "cortext_benchmark_environment_snapshot_v1",
        "created_at_utc": utc_now(),
        "privacy": "selected non-secret runtime variables only; secrets redacted by key name",
        "pid": pid,
        "process_name": process_name,
        "benchmark_command_sha256": command_sha256(shlex.join(benchmark_cmd)),
        "classification_policy": {
            "cortext_behavior_env": (
                "all CORTEXT_* variables except local judge endpoint metadata"
            ),
            "hosted_provider_behavior_env": (
                "provider-looking OPENAI/ANTHROPIC/GEMINI/GOOGLE/VERTEX/"
                "AZURE_OPENAI/TOGETHER/MISTRAL/COHERE/GROQ variables"
            ),
            "local_provider_env": "OLLAMA_* and local loopback judge endpoint metadata",
        },
        "cortext_behavior_env_guard": cortext_behavior_env_guard(env),
        "cortext_behavior_env": cortext_behavior_env,
        "hosted_provider_behavior_env": hosted_provider_behavior_env,
        "local_provider_env": local_provider_env,
        "selected_env": selected_env,
        "stripped_hosted_provider_env_keys": stripped_hosted_provider_env_keys,
    }


def normalize_local_base_url(base_url: str, provider: str) -> str:
    if "://" not in base_url:
        base_url = f"http://{base_url}"
    normalized = base_url.rstrip("/")
    parsed = urllib.parse.urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOCAL_JUDGE_HOSTS:
        raise RuntimeError(
            "Refusing non-local judge endpoint for private chat-replay metadata: "
            f"{normalized!r}. Start the local {provider} server and use a "
            "loopback URL."
        )
    return normalized


def ollama_model_status(base_url: str, model: str) -> dict:
    status = {
        "provider": "ollama",
        "base_url": base_url,
        "model": model,
        "local_url": False,
        "available": False,
        "image": False,
        "audio": False,
        "error": "",
    }
    try:
        local_base_url = normalize_local_base_url(base_url, "Ollama")
        status["base_url"] = local_base_url
        status["local_url"] = True
    except Exception as exc:
        status["error"] = str(exc)
        return status

    try:
        body = json.dumps({"model": model}).encode("utf-8")
        request = urllib.request.Request(
            f"{status['base_url']}/api/show",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
        capabilities = [
            str(capability)
            for capability in payload.get("capabilities", []) or []
        ]
        status["available"] = True
        status["capabilities"] = capabilities
        status["image"] = "vision" in capabilities or "image" in capabilities
        status["audio"] = "audio" in capabilities
        details = payload.get("details")
        if isinstance(details, dict):
            status["details"] = {
                key: details.get(key)
                for key in ("family", "parameter_size", "quantization_level")
            }
        return status
    except Exception as exc:
        status["show_error"] = exc.__class__.__name__

    try:
        with urllib.request.urlopen(f"{status['base_url']}/api/tags", timeout=5) as response:
            payload = json.load(response)
    except Exception as exc:
        status["error"] = exc.__class__.__name__
        return status

    for item in payload.get("models", []):
        if item.get("name") != model and item.get("model") != model:
            continue
        status["available"] = True
        details = item.get("details")
        if isinstance(details, dict):
            status["details"] = {
                key: details.get(key)
                for key in ("family", "parameter_size", "quantization_level")
            }
        capabilities = [
            str(capability)
            for capability in item.get("capabilities", []) or []
        ]
        status["capabilities"] = capabilities
        status["image"] = "vision" in capabilities or "image" in capabilities
        status["audio"] = "audio" in capabilities
        return status

    status["error"] = "model_not_listed"
    return status


def count_media_files(input_dir: pathlib.Path) -> int:
    if not input_dir.is_dir():
        return 0
    count = 0
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
            count += 1
    return count


def parse_required_modalities(value: str) -> list[str]:
    out: list[str] = []
    for part in value.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in MEDIA_KIND_EXTENSIONS:
            raise RuntimeError(
                "--require-media-modalities entries must be one of "
                f"{sorted(MEDIA_KIND_EXTENSIONS)}; got {name!r}"
            )
        if name not in out:
            out.append(name)
    return out


def parse_timestamp_prefix(text: str) -> int | None:
    if len(text) < 19:
        return None
    stamp = text[:19]
    if len(stamp) >= 19 and stamp[13] == " " and stamp[16] == " ":
        stamp = stamp[:13] + ":" + stamp[14:16] + ":" + stamp[17:]
    try:
        dt = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)


def parse_message_timestamps(transcript: pathlib.Path) -> list[int]:
    if not transcript.is_file():
        return []
    timestamps: list[int] = []
    pending_header = ""
    pending_has_body = False

    def flush() -> None:
        nonlocal pending_header, pending_has_body
        if pending_header and pending_has_body:
            ts = parse_timestamp_prefix(pending_header)
            if ts is not None:
                timestamps.append(ts)
        pending_header = ""
        pending_has_body = False

    with transcript.open(errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            if line.startswith("----------------------------------------------------"):
                flush()
                continue
            if parse_timestamp_prefix(line) is not None:
                flush()
                pending_header = line
                continue
            if pending_header and line.strip():
                pending_has_body = True
    flush()
    return timestamps


def media_kind(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    for kind, extensions in MEDIA_KIND_EXTENSIONS.items():
        if ext in extensions:
            return kind
    return ""


def selected_media_kind_counts(
    input_dir: pathlib.Path,
    transcript: pathlib.Path,
    skip_messages: int,
    max_messages: int,
    media_limit: int,
) -> dict[str, int]:
    counts = {kind: 0 for kind in MEDIA_KIND_EXTENSIONS}
    if media_limit <= 0 or not input_dir.is_dir():
        return counts

    timestamps = parse_message_timestamps(transcript)
    skipped = min(max(0, skip_messages), len(timestamps))
    window_timestamps = timestamps[skipped:]
    if max_messages >= 0:
        window_timestamps = window_timestamps[:max_messages]
    if not window_timestamps:
        return counts
    window_start = window_timestamps[0]
    window_end = window_timestamps[-1]

    media: list[tuple[int, str, str]] = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        kind = media_kind(path)
        if not kind:
            continue
        ts = parse_timestamp_prefix(path.name)
        if ts is None or ts < window_start or ts > window_end:
            continue
        media.append((ts, str(path), kind))
    media.sort(key=lambda item: (item[0], item[1]))

    for _, _, kind in media[:media_limit]:
        counts[kind] += 1
    return counts


def preflight_check(checks: list[dict], name: str, ok: bool, detail: str = "") -> None:
    checks.append(
        {
            "name": name,
            "status": "pass" if ok else "fail",
            "detail": detail,
        }
    )


def preflight_warn(checks: list[dict], name: str, ok: bool, detail: str = "") -> None:
    checks.append(
        {
            "name": name,
            "status": "pass" if ok else "warn",
            "detail": detail,
        }
    )


def run_preflight(
    args: argparse.Namespace,
    *,
    benchmark_command: str,
    benchmark_artifact: dict,
    supports_probe_stream: bool,
    early_command: list[str] | None,
    judge_media_smoke_command: list[str],
    human_label_sample_command: list[str],
    human_label_launch_command: list[str],
    finalizer_command: list[str],
    final_judge_command: list[str],
    bootstrap_report_command: list[str],
    freeze_report_command: list[str],
    release_freeze_path: pathlib.Path | None = None,
) -> dict:
    checks: list[dict] = []
    transcript = discover_transcript(args.input_dir)
    media_count = count_media_files(args.input_dir)
    required_modalities = parse_required_modalities(args.require_media_modalities)
    selected_kind_counts = selected_media_kind_counts(
        args.input_dir,
        transcript,
        args.skip_messages,
        args.max_messages,
        args.media_limit,
    )
    current_git = git_provenance()
    input_fingerprint = source_input_fingerprint(args.input_dir)
    supplied_freeze = load_json_if_valid(release_freeze_path) if release_freeze_path else {}
    ollama_status = ollama_model_status(args.ollama_base_url, args.judge_model)
    env_guard = cortext_behavior_env_guard(dict(os.environ))
    benchmark_parts = shlex.split(benchmark_command)
    disallowed_benchmark_flags = [
        flag
        for flag in DISALLOWED_RELEASE_BENCHMARK_FLAGS
        if command_has_flag(benchmark_parts, flag)
    ]
    existing_outputs = [
        path
        for path in [
            args.out_dir / "live.sqlite",
            args.out_dir / "summary.json",
            args.out_dir / "judge_gemma4_12b_local.json",
            args.out_dir / "release_protocol_report_initial.json",
            None if args.release_freeze else args.out_dir / "release_protocol_freeze.json",
        ]
        if path is not None and path.exists()
    ]

    preflight_check(
        checks,
        "input_transcript_present",
        transcript.is_file() and transcript.stat().st_size > 0,
        f"path={transcript} bytes={transcript.stat().st_size if transcript.exists() else 0}",
    )
    preflight_check(
        checks,
        "mixed_media_input_present",
        args.media_limit == 0 or media_count > 0,
        f"media_count={media_count} media_limit={args.media_limit}",
    )
    for modality in required_modalities:
        preflight_check(
            checks,
            f"required_{modality}_media_in_frozen_window",
            selected_kind_counts.get(modality, 0) > 0,
            (
                f"required_modalities={required_modalities} "
                f"selected_media_kind_counts={selected_kind_counts} "
                f"skip_messages={args.skip_messages} "
                f"max_messages={args.max_messages} "
                f"media_limit={args.media_limit}"
            ),
        )
    preflight_check(
        checks,
        "benchmark_executable_present",
        bool(benchmark_artifact.get("exists")),
        json.dumps(benchmark_artifact, sort_keys=True),
    )
    preflight_check(
        checks,
        "benchmark_streams_probe_rows",
        supports_probe_stream,
        f"benchmark={benchmark_artifact.get('path')}",
    )
    preflight_check(
        checks,
        "default_knobs_only",
        "--focus 0.5" in benchmark_command
        and "--sensitivity 0.5" in benchmark_command
        and "--stability 0.5" in benchmark_command,
        benchmark_command,
    )
    preflight_check(
        checks,
        "daily_deep_consolidation_enabled",
        "--daily-consolidation" in benchmark_command and "--deep" in benchmark_command,
        benchmark_command,
    )
    preflight_check(
        checks,
        "fixed_rag_baseline_configured",
        "--rag-top-k" in benchmark_command
        and "--active-history-token-budget" in benchmark_command,
        benchmark_command,
    )
    preflight_check(
        checks,
        "no_eval_only_benchmark_modes",
        not disallowed_benchmark_flags,
        f"disallowed_flags={disallowed_benchmark_flags}",
    )
    preflight_check(
        checks,
        "no_cortext_behavior_env_overrides",
        not env_guard["leakage_detected"],
        json.dumps(env_guard, sort_keys=True),
    )
    preflight_check(
        checks,
        "local_ollama_url",
        bool(ollama_status.get("local_url")),
        str(ollama_status.get("error") or ollama_status.get("base_url")),
    )
    preflight_check(
        checks,
        "local_gemma4_judge_available",
        bool(ollama_status.get("available")),
        json.dumps(ollama_status, sort_keys=True),
    )
    preflight_check(
        checks,
        "release_judge_repetitions",
        3 <= args.judge_repetitions <= 5,
        f"judge_repetitions={args.judge_repetitions}",
    )
    preflight_check(
        checks,
        "blind_final_judge_command",
        "--blind-packets" in shlex.join(final_judge_command),
        shlex.join(final_judge_command),
    )
    preflight_check(
        checks,
        "freeze_validated_report_command",
        "--freeze-file" in shlex.join(freeze_report_command),
        shlex.join(freeze_report_command),
    )
    preflight_check(
        checks,
        "release_freeze_supplied_when_requested",
        release_freeze_path is None or bool(supplied_freeze),
        f"path={release_freeze_path}",
    )
    if release_freeze_path is not None:
        expected_source = supplied_freeze.get("source_input_manifest_sha256")
        expected_command = supplied_freeze.get("benchmark_command_sha256")
        expected_executable = supplied_freeze.get("benchmark_executable_sha256")
        expected_commit = supplied_freeze.get("git_commit")
        expected_worktree = supplied_freeze.get("git_worktree_manifest_sha256")
        preflight_check(
            checks,
            "source_input_manifest_matches_release_freeze",
            bool(expected_source)
            and expected_source == input_fingerprint.get("manifest_sha256"),
            (
                f"expected={expected_source} "
                f"actual={input_fingerprint.get('manifest_sha256')}"
            ),
        )
        preflight_check(
            checks,
            "benchmark_command_matches_release_freeze",
            not expected_command or expected_command == command_sha256(benchmark_command),
            (
                f"expected={expected_command} "
                f"actual={command_sha256(benchmark_command)}"
            ),
        )
        preflight_check(
            checks,
            "benchmark_executable_matches_release_freeze",
            bool(expected_executable)
            and expected_executable == benchmark_artifact.get("sha256"),
            (
                f"expected={expected_executable} "
                f"actual={benchmark_artifact.get('sha256')}"
            ),
        )
        preflight_check(
            checks,
            "git_commit_matches_release_freeze",
            bool(expected_commit) and expected_commit == current_git.get("commit"),
            f"expected={expected_commit} actual={current_git.get('commit')}",
        )
        preflight_check(
            checks,
            "git_worktree_matches_release_freeze",
            bool(expected_worktree)
            and expected_worktree == current_git.get("worktree_manifest_sha256"),
            (
                f"expected={expected_worktree} "
                f"actual={current_git.get('worktree_manifest_sha256')}"
            ),
        )
    preflight_check(
        checks,
        "no_stale_outputs_without_force",
        args.force or not existing_outputs,
        ", ".join(str(path) for path in existing_outputs),
    )
    preflight_warn(
        checks,
        "early_judge_enabled",
        early_command is not None,
        "early judge is enabled; long runs can fail fast"
        if early_command is not None
        else "early judge is disabled; long runs will not fail fast",
    )
    preflight_check(
        checks,
        "strict_early_judge_failure_confirmation",
        not args.require_final_report_pass
        or (
            early_command is not None
            and args.early_confirm_fail_repetitions >= 3
        ),
        (
            f"require_final_report_pass={args.require_final_report_pass} "
            f"early_judge_enabled={early_command is not None} "
            "early_confirm_fail_repetitions="
            f"{args.early_confirm_fail_repetitions}"
        ),
    )
    preflight_warn(
        checks,
        "ablation_pipeline_requested",
        args.run_ablations,
        "ablation pipeline requested"
        if args.run_ablations
        else "release claim requires ablations; pass --run-ablations for final run",
    )
    preflight_warn(
        checks,
        "final_report_requested",
        not args.skip_initial_report,
        "release freeze/report artifacts will be produced"
        if not args.skip_initial_report
        else "release freeze/report artifacts are skipped with --skip-initial-report",
    )
    preflight_warn(
        checks,
        "final_judge_requested",
        not args.skip_final_judge,
        "final repeated local judge will run"
        if not args.skip_final_judge
        else "release quality artifacts are skipped with --skip-final-judge",
    )
    has_external_human_artifacts = bool(
        args.human_labels
        and args.human_labels.exists()
        and args.human_label_eval
        and args.human_label_eval.exists()
    )
    finalizer_handoff_configured = not args.skip_human_label_sample
    missing_strict = []
    if not has_external_human_artifacts and not finalizer_handoff_configured:
        missing_strict.extend(["human_labels", "human_label_eval"])
    preflight_check(
        checks,
        "strict_final_report_human_path_available",
        not args.require_final_report_pass or not missing_strict,
        (
            "strict final pass needs either preexisting human label artifacts "
            "or the generated human-label sample/finalizer handoff; "
            f"missing={missing_strict} "
            f"finalizer_handoff_configured={finalizer_handoff_configured}"
        ),
    )
    preflight_check(
        checks,
        "strict_finalizer_handoff_command_recorded",
        not args.require_final_report_pass
        or finalizer_handoff_configured
        or has_external_human_artifacts,
        shlex.join(finalizer_command),
    )
    preflight_check(
        checks,
        "strict_final_report_pipeline_complete",
        not args.require_final_report_pass
        or (
            args.run_ablations
            and not args.skip_initial_report
            and not args.skip_final_judge
        ),
        (
            "strict final pass requires --run-ablations, final repeated "
            "judge, and final report; "
            f"run_ablations={args.run_ablations} "
            f"skip_final_judge={args.skip_final_judge} "
            f"skip_initial_report={args.skip_initial_report}"
        ),
    )
    preflight_check(
        checks,
        "strict_final_judge_packets_uncropped",
        not args.require_final_report_pass or args.judge_packet_item_limit == -1,
        (
            "strict final pass requires uncropped final judge packets; "
            f"judge_packet_item_limit={args.judge_packet_item_limit}"
        ),
    )

    status = "fail" if any(check["status"] == "fail" for check in checks) else "pass"
    return {
        "schema": "cortext_chat_replay_release_preflight_v1",
        "created_at_utc": utc_now(),
        "privacy": "private local artifact; contains paths and commands, not message text",
        "status": status,
        "checks": checks,
        "input": {
            "input_dir": str(args.input_dir),
            "transcript": str(transcript),
            "media_file_count": media_count,
            "required_media_modalities": required_modalities,
            "selected_media_kind_counts": selected_kind_counts,
            "source_input_fingerprint": {
                "schema": input_fingerprint.get("schema"),
                "manifest_sha256": input_fingerprint.get("manifest_sha256"),
                "transcript_sha256": input_fingerprint.get("transcript_sha256"),
                "file_count": input_fingerprint.get("file_count"),
                "readable_file_count": input_fingerprint.get("readable_file_count"),
                "total_bytes": input_fingerprint.get("total_bytes"),
            },
        },
        "release_freeze": {
            "path": str(release_freeze_path) if release_freeze_path else "",
            "supplied": release_freeze_path is not None,
            "schema": supplied_freeze.get("schema", "") if supplied_freeze else "",
        },
        "judge_model": ollama_status,
        "environment_guard": env_guard,
        "commands": {
            "benchmark": benchmark_command,
            "early_judge": shlex.join(early_command) if early_command else "",
            "judge_media_smoke": shlex.join(judge_media_smoke_command),
            "human_label_sample": shlex.join(human_label_sample_command),
            "human_label_launch": shlex.join(human_label_launch_command),
            "finalizer": shlex.join(finalizer_command),
            "final_judge": shlex.join(final_judge_command),
            "bootstrap_report": shlex.join(bootstrap_report_command),
            "freeze_report": shlex.join(freeze_report_command),
        },
    }


def cleanup_runner_outputs(out_dir: pathlib.Path) -> None:
    for name in [
        "live.sqlite",
        "live.sqlite-shm",
        "live.sqlite-wal",
        "summary.json",
        "summary.json.probes.jsonl",
        "summary_gate.json",
        "judge_gemma4_12b_local.json",
        "judge_gemma4_12b_local.json.rows.jsonl",
        "judge_media_smoke_ollama.json",
        "judge_media_smoke_pipeline.log",
        "human_label_sample.json",
        "human_label_sample_pipeline.log",
        "human_label_launch_command.txt",
        "finalize_release_command.txt",
        "release_protocol_report_final.json",
        "release_protocol_report_initial.json",
        "run.log",
        "early_judge_pipeline.log",
        "judge_pipeline.log",
        "main_report_pipeline.log",
        "command_manifest.json",
        "release_protocol_freeze.json",
        "frozen_probe_manifest.json",
        "frozen_probe_schedule.json",
        "release_protocol_report_bootstrap.log",
        "benchmark_status.json",
        "benchmark_environment_snapshot.json",
        "early_failure_report.json",
        "preflight_report.json",
    ]:
        path = out_dir / name
        if path.exists():
            path.unlink()
    early_dir = out_dir / "early_judge"
    if early_dir.exists():
        shutil.rmtree(early_dir)


def ensure_report_freezable(report: dict) -> None:
    unexpected_failures = []
    for item in report.get("checks", []):
        if not isinstance(item, dict) or item.get("status") != "fail":
            continue
        name = str(item.get("name", ""))
        if name.startswith("claim_"):
            continue
        unexpected_failures.append(
            {
                "name": name,
                "detail": item.get("detail", ""),
            }
        )
    if unexpected_failures:
        raise RuntimeError(
            "refusing to freeze report with protocol/provenance failures: "
            + json.dumps(unexpected_failures, sort_keys=True)
        )


def write_probe_freeze_artifacts(report_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    report = json.loads(report_path.read_text())
    manifest = report.get("frozen_probe_manifest")
    schedule = report.get("frozen_probe_schedule")
    if not isinstance(manifest, dict) or not manifest.get("manifest_sha256"):
        raise RuntimeError("report does not contain a frozen probe manifest")
    if not isinstance(schedule, dict) or not schedule.get("schedule_sha256"):
        raise RuntimeError("report does not contain a frozen probe schedule")
    write_json(out_dir / "frozen_probe_manifest.json", manifest)
    write_json(out_dir / "frozen_probe_schedule.json", schedule)


def write_or_validate_release_freeze(
    path: pathlib.Path,
    report_path: pathlib.Path,
    benchmark_command: str,
) -> None:
    report = json.loads(report_path.read_text())
    ensure_report_freezable(report)
    source_fingerprint = report.get("source_run", {}).get(
        "source_input_fingerprint", {}
    )
    manifest = report.get("frozen_probe_manifest", {})
    schedule = report.get("frozen_probe_schedule", {})
    git = report.get("git", {})
    benchmark_executable = report.get("artifacts", {}).get(
        "benchmark_executable", {}
    )
    freeze = {
        "schema": "cortext_chat_replay_release_protocol_freeze_v1",
        "created_at_utc": utc_now(),
        "source": "main_release_report",
        "source_report_path": str(report_path),
        "source_report_sha256": file_sha256(report_path),
        "source_input_manifest_sha256": source_fingerprint.get(
            "manifest_sha256", ""
        ),
        "source_input_transcript_sha256": source_fingerprint.get(
            "transcript_sha256", ""
        ),
        "frozen_probe_manifest_sha256": manifest.get("manifest_sha256", ""),
        "frozen_probe_schedule_sha256": schedule.get("schedule_sha256", ""),
        "frozen_probe_count": schedule.get("probe_count"),
        "benchmark_command_sha256": command_sha256(benchmark_command),
        "benchmark_executable_sha256": benchmark_executable.get("sha256", ""),
        "git_commit": git.get("commit", ""),
        "git_dirty": git.get("dirty"),
        "git_status_sha256": git.get("status_sha256", ""),
        "git_worktree_manifest_sha256": git.get(
            "worktree_manifest_sha256", ""
        ),
        "privacy": "hashes only; no message text or media paths",
    }
    required = [
        "source_input_manifest_sha256",
        "frozen_probe_manifest_sha256",
        "frozen_probe_schedule_sha256",
        "benchmark_command_sha256",
        "benchmark_executable_sha256",
        "git_commit",
        "git_worktree_manifest_sha256",
    ]
    missing = [key for key in required if not freeze.get(key)]
    if missing:
        raise RuntimeError(
            "report does not contain required release freeze fields: "
            + ", ".join(missing)
        )
    if path.exists():
        existing = json.loads(path.read_text())
        for key in required:
            if existing.get(key) != freeze.get(key):
                raise RuntimeError(
                    f"existing release freeze mismatch for {key}: "
                    f"{existing.get(key)} != {freeze.get(key)}"
                )
        return
    write_json(path, freeze)


def is_json(path: pathlib.Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        json.loads(path.read_text())
        return True
    except Exception:
        return False


def probe_rows_available(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open() as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            try:
                json.loads(text)
            except json.JSONDecodeError:
                continue
            count += 1
    return count


def add_check(checks: list[dict], name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def validate_summary(
    summary_path: pathlib.Path,
    out_path: pathlib.Path,
    min_probe_count: int,
) -> dict:
    if not is_json(summary_path):
        raise RuntimeError(f"summary is missing or invalid JSON: {summary_path}")

    summary = json.loads(summary_path.read_text())
    checks: list[dict] = []
    knobs = summary.get("knobs") or {}
    add_check(
        checks,
        "default_knobs_0_5",
        all(
            abs(float(knobs.get(k, -1)) - 0.5) < 1e-9
            for k in ["focus", "sensitivity", "stability"]
        ),
        f"knobs={knobs}",
    )
    add_check(
        checks,
        "daily_and_deep",
        bool(summary.get("daily_consolidation"))
        and bool(summary.get("deep_consolidation")),
        f"daily={summary.get('daily_consolidation')} deep={summary.get('deep_consolidation')}",
    )
    add_check(
        checks,
        f"probe_count_floor_{min_probe_count}",
        int(summary.get("probe_count", 0) or 0) >= min_probe_count,
        f"probe_count={summary.get('probe_count')}",
    )
    add_check(
        checks,
        "rag_policy",
        summary.get("normal_rag_retrieval") == "raw_chat_vector"
        and summary.get("normal_rag_baseline_modality") == "text_only",
        f"rag={summary.get('normal_rag_retrieval')} modality={summary.get('normal_rag_baseline_modality')}",
    )

    missing_frozen = []
    rag_non_text = []
    future = []
    marker = []
    for probe in summary.get("probes", []) or []:
        event_index = int(probe.get("event_index", 0) or 0)
        if probe.get("cortext_frozen_packet_policy") != "probe_time_hydrated_context_snapshot":
            missing_frozen.append(event_index)
        elif not isinstance(probe.get("cortext_frozen_working_memory"), list) or not isinstance(
            probe.get("cortext_frozen_retrieved_memory"), list
        ):
            missing_frozen.append(event_index)

        compacted = int(probe.get("normal_rag_compacted_history_items", 0) or 0)
        text = str(probe.get("normal_rag_compacted_summary", "")).strip()
        if compacted > 0 and (not text or text.startswith("[compacted_history ")):
            marker.append(event_index)

        for key in ["rolling_history", "rag_top_k"]:
            for row in probe.get(key, []) or []:
                if row.get("modality") != "text":
                    rag_non_text.append(
                        {
                            "event_index": event_index,
                            "key": key,
                            "row_index": row.get("index"),
                            "modality": row.get("modality"),
                        }
                    )
                if int(row.get("index", -1) or -1) >= event_index:
                    future.append(
                        {
                            "event_index": event_index,
                            "key": key,
                            "row_index": row.get("index"),
                        }
                    )

    add_check(
        checks,
        "cortext_probe_time_packets_frozen",
        not missing_frozen,
        f"missing={missing_frozen[:20]}",
    )
    add_check(
        checks,
        "rag_compaction_contentful",
        summary.get("normal_rag_compaction_summary_policy")
        == "deterministic_extractive_prior_chat"
        and not marker,
        f"policy={summary.get('normal_rag_compaction_summary_policy')} marker={marker[:10]}",
    )
    add_check(checks, "rag_text_only", not rag_non_text, f"non_text={rag_non_text[:10]}")
    add_check(checks, "rag_prior_only", not future, f"future={future[:10]}")

    body = {
        "schema": "cortext_chat_replay_summary_gate_v3",
        "created_at_utc": utc_now(),
        "summary": str(summary_path),
        "overall_status": "pass"
        if all(check["status"] == "pass" for check in checks)
        else "fail",
        "checks": checks,
    }
    write_json(out_path, body)
    return body


def artifact_ref(path: pathlib.Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def selected_metrics(metrics: dict) -> dict:
    names = [
        "judged_rows",
        "probe_count",
        "cortext_wins",
        "traditional_chat_rag_wins",
        "full_history_upper_bound_wins",
        "cortext_win_rate",
        "cortext_quality_composite",
        "traditional_chat_rag_quality_composite",
        "full_history_upper_bound_quality_composite",
        "cortext_quality_delta_vs_traditional_chat_rag",
        "cortext_quality_delta_vs_full_history_upper_bound",
        "cortext_token_savings_vs_traditional_chat_rag",
        "mean_cortext_context_tokens",
        "mean_traditional_chat_rag_tokens",
    ]
    return {name: metrics.get(name) for name in names if name in metrics}


def write_early_failure_report(
    *,
    out_path: pathlib.Path,
    command_manifest_path: pathlib.Path,
    status_path: pathlib.Path,
    early_latest_path: pathlib.Path,
    benchmark_log: pathlib.Path,
    early_log: pathlib.Path,
) -> pathlib.Path:
    status = load_json_if_valid(status_path)
    early_latest_payload = load_json_if_valid(early_latest_path)
    command_manifest = load_json_if_valid(command_manifest_path)
    latest = early_latest_payload.get("latest", {})
    if not isinstance(latest, dict):
        latest = {}
    metrics = latest.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    pre_confirm = latest.get("pre_confirm", {})
    if not isinstance(pre_confirm, dict):
        pre_confirm = {}

    failed_checks = [
        check
        for check in latest.get("fail_fast_checks", []) or []
        if isinstance(check, dict) and check.get("status") == "fail"
    ]
    artifact_paths = {
        "benchmark_status": status_path,
        "early_judge_latest": early_latest_path,
        "benchmark_log": benchmark_log,
        "early_judge_log": early_log,
    }
    for key in [
        "summary",
        "judge",
        "loss_audit",
        "delta_judge",
        "confirm_judge",
        "confirm_delta_judge",
        "confirm_loss_audit",
    ]:
        value = latest.get(key)
        if value:
            artifact_paths[key] = pathlib.Path(str(value))

    report = {
        "schema": "cortext_chat_replay_release_early_failure_report_v1",
        "created_at_utc": utc_now(),
        "privacy": (
            "private local artifact; aggregate metrics and artifact hashes only, "
            "no message text, media bytes, or judge reason strings"
        ),
        "release_gate_use": "fail_fast_failure_evidence_not_release_claim",
        "overall_status": "fail",
        "failure": {
            "milestone": latest.get("milestone"),
            "phase": latest.get("quality_gate_phase_reason"),
            "fail_fast_status": latest.get("fail_fast_status"),
            "confirm_fail_triggered": bool(latest.get("confirm_fail_triggered")),
            "failed_checks": failed_checks,
        },
        "metrics": selected_metrics(metrics),
        "pre_confirm_metrics": selected_metrics(pre_confirm.get("metrics", {}))
        if isinstance(pre_confirm.get("metrics", {}), dict)
        else {},
        "fairness_checks": metrics.get("fairness_checks", {}),
        "judge_validation": metrics.get("judge_validation", {}),
        "rag_phase": {
            "quality_gate_active": latest.get("quality_gate_active"),
            "quality_gate_requires_rag_pressure": latest.get(
                "quality_gate_requires_rag_pressure"
            ),
            "quality_gate_min_history_budget_ratio": latest.get(
                "quality_gate_min_history_budget_ratio"
            ),
            "max_rolling_history_budget_ratio": latest.get(
                "max_rolling_history_budget_ratio"
            ),
            "compaction_probe_count": latest.get("compaction_probe_count"),
            "vector_augmented_probe_count": latest.get("vector_augmented_probe_count"),
            "rag_phase_counts": latest.get("rag_phase_counts", {}),
        },
        "runner_status": {
            "status": status.get("status"),
            "detail": status.get("detail"),
            "benchmark_exit_code": status.get("benchmark_exit_code"),
            "early_judge_exit_code": status.get("early_judge_exit_code"),
            "elapsed_s": status.get("elapsed_s"),
            "probe_stream": status.get("probe_stream", {}),
        },
        "judge": {
            "provider": early_latest_payload.get("judge_provider"),
            "model": early_latest_payload.get("judge_model"),
            "early_repetitions": early_latest_payload.get("judge_repetitions"),
            "confirm_fail_repetitions": early_latest_payload.get(
                "confirm_fail_repetitions"
            ),
        },
        "protocol": {
            "git": command_manifest.get("git", {}),
            "fixed_protocol": command_manifest.get("fixed_protocol", {}),
            "benchmark_command_sha256": command_sha256(
                str(command_manifest.get("benchmark_command", ""))
            )
            if command_manifest.get("benchmark_command")
            else "",
            "early_judge_command_sha256": command_sha256(
                str(command_manifest.get("early_judge_command_launched", ""))
            )
            if command_manifest.get("early_judge_command_launched")
            else "",
        },
        "artifacts": {
            name: artifact_ref(path) for name, path in sorted(artifact_paths.items())
        },
    }
    write_json(out_path, report)
    return out_path


def build_benchmark_command(args: argparse.Namespace, db: pathlib.Path, summary: pathlib.Path) -> list[str]:
    cmd = [
        str(args.benchmark),
        "--input-dir",
        str(args.input_dir),
        "--skip-messages",
        str(args.skip_messages),
        "--max-messages",
        str(args.max_messages),
        "--media-limit",
        str(args.media_limit),
        "--probe-stride",
        str(args.probe_stride),
        "--warmup-events",
        str(args.warmup_events),
        "--deep",
        "--daily-consolidation",
        "--rag-top-k",
        str(args.rag_top_k),
        "--active-history-token-budget",
        str(args.active_history_token_budget),
        "--focus",
        "0.5",
        "--sensitivity",
        "0.5",
        "--stability",
        "0.5",
        "--db",
        str(db),
        "--out",
        str(summary),
    ]
    # Deep-LLM provider URIs are arm-defining: they change the summarizer/
    # extractor runtime (and therefore latency comparability with on-device
    # figures), so they flow through the frozen benchmark command and are
    # recorded by the benchmark into summary.json.
    if args.summarizer_provider:
        cmd += ["--summarizer-provider", args.summarizer_provider]
    if args.extractor_provider:
        cmd += ["--extractor-provider", args.extractor_provider]
    return cmd


def build_early_judge_command(
    args: argparse.Namespace,
    db: pathlib.Path,
    summary: pathlib.Path,
    out_dir: pathlib.Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/watch_chat_replay_probe_stream_judge.py"),
        "--probe-stream",
        str(summary) + ".probes.jsonl",
        "--out-dir",
        str(out_dir / "early_judge"),
        "--input-dir",
        str(args.input_dir),
        "--db",
        str(db),
        "--timeline-skip-messages",
        str(args.skip_messages),
        "--timeline-max-messages",
        str(args.max_messages),
        "--timeline-media-limit",
        str(args.media_limit),
        "--milestones",
        args.early_judge_milestones,
        "--periodic-stride",
        str(args.early_judge_periodic_stride),
        "--completion-summary",
        str(summary),
        "--poll-seconds",
        str(args.early_judge_poll_seconds),
        "--warmup-events",
        str(args.warmup_events),
        "--probe-stride",
        str(args.probe_stride),
        "--rag-top-k",
        str(args.rag_top_k),
        "--active-history-token-budget",
        str(args.active_history_token_budget),
        "--focus",
        "0.5",
        "--sensitivity",
        "0.5",
        "--stability",
        "0.5",
        "--daily-consolidation",
        "--deep",
        "--judge-provider",
        "ollama",
        "--model",
        args.judge_model,
        "--ollama-base-url",
        args.ollama_base_url,
        "--ollama-keep-alive",
        "0s",
        "--judge-repetitions",
        str(args.early_judge_repetitions),
        "--confirm-fail-repetitions",
        str(args.early_confirm_fail_repetitions),
        "--judge-seed",
        str(args.judge_seed),
        "--bootstrap-samples",
        str(args.early_judge_bootstrap_samples),
        "--judge-timeout-s",
        str(args.early_judge_timeout_s),
        "--judge-context-window-tokens",
        str(args.early_judge_context_window_tokens),
        "--judge-max-output-tokens",
        str(args.early_judge_max_output_tokens),
        "--context-limit",
        str(args.early_judge_packet_item_limit),
        "--quality-gate-min-milestone",
        str(args.early_quality_gate_min_milestone),
        "--quality-trend-gate-min-milestone",
        str(args.early_quality_trend_gate_min_milestone),
        "--quality-trend-window",
        str(args.early_quality_trend_window),
        "--quality-gate-min-history-budget-ratio",
        str(args.early_quality_gate_min_history_budget_ratio),
        "--max-media-per-system",
        "-1",
        "--blind-packets",
        "--min-mean-cortext-context-tokens",
        str(args.early_min_mean_cortext_context_tokens),
        "--min-cortext-token-savings-vs-rag",
        str(args.early_min_cortext_token_savings_vs_rag),
        "--min-cortext-quality-delta-vs-rag",
        str(args.early_min_cortext_quality_delta_vs_rag),
        "--require-fairness-checks",
    ]
    if args.early_judge_periodic_stride == 0:
        cmd.append("--stop-after-last")
    if args.early_min_cortext_win_rate is not None:
        cmd += ["--min-cortext-win-rate", str(args.early_min_cortext_win_rate)]
    if not args.early_quality_gate_requires_rag_pressure:
        cmd.append("--no-quality-gate-requires-rag-pressure")
    return cmd


def build_final_judge_command(args: argparse.Namespace, db: pathlib.Path, summary: pathlib.Path, judge: pathlib.Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "tools/judge_chat_replay_live_run.py"),
        "--summary",
        str(summary),
        "--db",
        str(db),
        "--out",
        str(judge),
        "--judge-provider",
        "ollama",
        "--model",
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
        "--judge-max-output-tokens",
        str(args.judge_max_output_tokens),
        "--context-limit",
        str(args.judge_packet_item_limit),
        "--blind-packets",
        "--max-media-per-system",
        "-1",
    ]


def build_human_label_sample_command(
    args: argparse.Namespace,
    db: pathlib.Path,
    summary: pathlib.Path,
    sample: pathlib.Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/chat_replay_human_label_harness.py"),
        "build-sample",
        "--summary",
        str(summary),
        "--db",
        str(db),
        "--input-dir",
        str(args.input_dir),
        "--out",
        str(sample),
        "--max-probes",
        str(args.human_label_max_probes),
        "--max-candidates",
        str(args.human_label_max_candidates),
        "--min-overlap",
        str(args.human_label_min_overlap),
    ]
    if args.human_label_shuffle_probes:
        cmd.append("--shuffle-probes")
    return cmd


def build_human_label_launch_command(
    args: argparse.Namespace,
    sample: pathlib.Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "tools/chat_replay_human_label_harness.py"),
        "launch",
        "--sample",
        str(sample),
        "--host",
        args.human_label_host,
        "--port",
        str(args.human_label_port),
    ]


def build_finalizer_command(
    args: argparse.Namespace,
    db: pathlib.Path,
    summary: pathlib.Path,
    judge: pathlib.Path,
    sample: pathlib.Path,
    judge_media_smoke: pathlib.Path,
    benchmark_command_text: str,
    final_judge_cmd: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/finalize_chat_replay_release_protocol.py"),
        "--base",
        str(args.out_dir / "ablations"),
        "--summary",
        str(summary),
        "--db",
        str(db),
        "--main-judge",
        str(judge),
        "--sample",
        str(sample),
        "--ablation-plan",
        str(args.out_dir / "ablation_plan.json"),
        "--final-report",
        str(args.out_dir / "release_protocol_report_final.json"),
        "--benchmark-command",
        benchmark_command_text,
        "--judge-command",
        shlex.join(final_judge_cmd),
        "--judge-media-smoke",
        str(judge_media_smoke),
        "--ollama-base-url",
        args.ollama_base_url,
        "--target-freeze-model",
        args.judge_model,
    ]
    if args.target_freeze is not None:
        cmd += ["--target-freeze", str(args.target_freeze)]
    return cmd


def build_judge_media_smoke_command(
    args: argparse.Namespace,
    judge_media_smoke: pathlib.Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "tools/smoke_ollama_judge_media.py"),
        "--out",
        str(judge_media_smoke),
        "--model",
        args.judge_model,
        "--ollama-base-url",
        args.ollama_base_url,
        "--timeout-s",
        str(args.judge_timeout_s),
        "--num-ctx",
        str(args.judge_context_window_tokens),
    ]


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
    image = selected.get("image", {}) if isinstance(selected, dict) else {}
    audio = selected.get("audio", {}) if isinstance(selected, dict) else {}
    image_parsed = image.get("parsed", {}) if isinstance(image, dict) else {}
    audio_parsed = audio.get("parsed", {}) if isinstance(audio, dict) else {}
    return not (
        isinstance(image_parsed, dict)
        and isinstance(audio_parsed, dict)
        and image_parsed.get("image_seen") is True
        and audio_parsed.get("audio_seen") is True
    )


def ensure_default_judge_media_smoke(
    args: argparse.Namespace,
    judge_media_smoke: pathlib.Path,
    judge_media_smoke_cmd: list[str],
) -> None:
    if args.judge_media_smoke is not None:
        return
    if not judge_media_smoke_needs_generation(judge_media_smoke, args.judge_model):
        return
    run_checked(
        judge_media_smoke_cmd,
        args.out_dir / "judge_media_smoke_pipeline.log",
    )


def build_report_command(
    benchmark_command: list[str],
    judge_command: list[str],
    summary: pathlib.Path,
    judge: pathlib.Path,
    report: pathlib.Path,
    judge_media_smoke: pathlib.Path | None,
    freeze_file: pathlib.Path | None = None,
    human_labels: pathlib.Path | None = None,
    human_label_eval: pathlib.Path | None = None,
    target_freeze: pathlib.Path | None = None,
    require_pass: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/chat_replay_release_protocol_report.py"),
        "--summary",
        str(summary),
        "--judge",
        str(judge),
        "--out",
        str(report),
        "--benchmark-command",
        shlex.join(benchmark_command),
        "--judge-command",
        shlex.join(judge_command),
    ]
    if judge_media_smoke and judge_media_smoke.exists():
        cmd += ["--judge-media-smoke", str(judge_media_smoke)]
    if freeze_file is not None:
        cmd += ["--freeze-file", str(freeze_file)]
    if human_labels is not None:
        cmd += ["--human-labels", str(human_labels)]
    if human_label_eval is not None:
        cmd += ["--human-label-eval", str(human_label_eval)]
    if target_freeze is not None:
        cmd += ["--target-freeze", str(target_freeze)]
    if require_pass:
        cmd.append("--require-pass")
    return cmd


def terminate_process(process: subprocess.Popen, name: str, timeout_s: float = 30.0) -> None:
    if process.poll() is not None:
        return
    print(f"[runner] terminating {name} pid={process.pid}", flush=True)
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"[runner] killing {name} pid={process.pid}", flush=True)
        process.kill()
        process.wait()


def run_checked(cmd: list[str], log_path: pathlib.Path | None = None) -> None:
    child_env, _ = sanitized_subprocess_env()
    print("+ " + shlex.join(cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=child_env)
        return
    with log_path.open("w") as log:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env=child_env,
        )


def run_benchmark_with_early_judge(
    benchmark_cmd: list[str],
    early_cmd: list[str] | None,
    run_log: pathlib.Path,
    early_log: pathlib.Path,
    *,
    kill_on_early_fail: bool,
    status_path: pathlib.Path,
    summary_path: pathlib.Path,
    environment_snapshot_path: pathlib.Path,
    command_manifest_path: pathlib.Path,
    early_failure_report_path: pathlib.Path,
    min_probe_rows_after_benchmark: int,
) -> str | None:
    run_log.parent.mkdir(parents=True, exist_ok=True)
    probe_stream_path = pathlib.Path(str(summary_path) + ".probes.jsonl")
    early_latest_path = run_log.parent / "early_judge" / "early_judge_latest.json"
    started_at = time.monotonic()
    early_process = None
    early_log_file = None
    early_launch_command_text: str | None = None

    def write_status(
        status: str,
        *,
        benchmark_code: int | None = None,
        early_code: int | None = None,
        detail: str = "",
    ) -> None:
        write_json(
            status_path,
            {
                "schema": "cortext_chat_replay_release_benchmark_status_v1",
                "created_at_utc": utc_now(),
                "status": status,
                "detail": detail,
                "benchmark_exit_code": benchmark_code,
                "early_judge_exit_code": early_code,
                "elapsed_s": round(time.monotonic() - started_at, 3),
                "summary": {
                    "path": str(summary_path),
                    "exists": summary_path.exists(),
                    "bytes": summary_path.stat().st_size
                    if summary_path.exists()
                    else 0,
                    "valid_json": is_json(summary_path),
                },
                "probe_stream": {
                    "path": str(probe_stream_path),
                    "exists": probe_stream_path.exists(),
                    "bytes": probe_stream_path.stat().st_size
                    if probe_stream_path.exists()
                    else 0,
                    "rows": probe_rows_available(probe_stream_path),
                    "required_rows_after_benchmark": min_probe_rows_after_benchmark,
                },
                "logs": {
                    "benchmark": str(run_log),
                    "early_judge": str(early_log),
                },
                "early_failure_report": {
                    "path": str(early_failure_report_path),
                    "exists": early_failure_report_path.exists(),
                    "bytes": early_failure_report_path.stat().st_size
                    if early_failure_report_path.exists()
                    else 0,
                },
                "early_judge_launched_command": early_launch_command_text or "",
                "early_judge_latest": load_json_if_valid(early_latest_path),
            },
        )

    write_status("starting")
    failure_status_written = False
    last_running_status = started_at
    benchmark_finished_code: int | None = None
    with run_log.open("w") as run_log_file:
        print("+ " + shlex.join(benchmark_cmd), flush=True)
        child_env, stripped_hosted_provider_env_keys = sanitized_subprocess_env()
        benchmark_process = subprocess.Popen(
            benchmark_cmd,
            cwd=REPO_ROOT,
            stdout=run_log_file,
            stderr=subprocess.STDOUT,
            env=child_env,
        )
        write_json(
            environment_snapshot_path,
            benchmark_environment_snapshot(
                benchmark_process.pid,
                benchmark_cmd,
                child_env,
                stripped_hosted_provider_env_keys,
            ),
        )
        if early_cmd:
            early_log.parent.mkdir(parents=True, exist_ok=True)
            early_log_file = early_log.open("w")
            early_launch_cmd = [
                *early_cmd,
                "--pause-pid-during-judge",
                str(benchmark_process.pid),
            ]
            early_launch_command_text = shlex.join(early_launch_cmd)
            write_text(
                early_log.with_name("early_judge_command_launched.txt"),
                early_launch_command_text,
            )
            print("+ " + early_launch_command_text, flush=True)
            early_process = subprocess.Popen(
                early_launch_cmd,
                cwd=REPO_ROOT,
                stdout=early_log_file,
                stderr=subprocess.STDOUT,
                env=child_env,
            )
        write_status("running")

        try:
            while True:
                benchmark_code = (
                    benchmark_finished_code
                    if benchmark_finished_code is not None
                    else benchmark_process.poll()
                )
                early_code = early_process.poll() if early_process else None
                if early_process and early_code is not None and early_code != 0:
                    if kill_on_early_fail and benchmark_code is None:
                        terminate_process(benchmark_process, "benchmark")
                        benchmark_code = benchmark_process.poll()
                    write_status(
                        "early_judge_failed",
                        benchmark_code=benchmark_code,
                        early_code=early_code,
                        detail=f"early judge failed; see {early_log}",
                    )
                    report_path = write_early_failure_report(
                        out_path=early_failure_report_path,
                        command_manifest_path=command_manifest_path,
                        status_path=status_path,
                        early_latest_path=early_latest_path,
                        benchmark_log=run_log,
                        early_log=early_log,
                    )
                    write_status(
                        "early_judge_failed",
                        benchmark_code=benchmark_code,
                        early_code=early_code,
                        detail=(
                            f"early judge failed; see {early_log}; "
                            f"failure report: {report_path}"
                        ),
                    )
                    failure_status_written = True
                    raise RuntimeError(
                        f"early judge failed with exit code {early_code}; "
                        f"see {early_log}; failure report: {report_path}"
                    )
                if benchmark_finished_code is not None:
                    if early_process and early_code is None:
                        now = time.monotonic()
                        if now - last_running_status >= 30.0:
                            write_status(
                                "benchmark_complete_waiting_early_judge",
                                benchmark_code=benchmark_finished_code,
                                early_code=early_code,
                            )
                            last_running_status = now
                        time.sleep(5)
                        continue
                    write_status(
                        "benchmark_complete",
                        benchmark_code=benchmark_finished_code,
                        early_code=early_code,
                    )
                    break
                if benchmark_code is not None:
                    if benchmark_code != 0:
                        write_status(
                            "benchmark_failed",
                            benchmark_code=benchmark_code,
                            early_code=early_code,
                            detail=f"benchmark failed; see {run_log}",
                        )
                        failure_status_written = True
                        raise RuntimeError(
                            f"benchmark failed with exit code {benchmark_code}; see {run_log}"
                        )
                    rows = probe_rows_available(probe_stream_path)
                    if not is_json(summary_path) or rows < min_probe_rows_after_benchmark:
                        write_status(
                            "benchmark_incomplete",
                            benchmark_code=benchmark_code,
                            early_code=early_code,
                            detail=(
                                "benchmark exited successfully but did not "
                                "produce the required summary/probe stream"
                            ),
                        )
                        failure_status_written = True
                        raise RuntimeError(
                            "benchmark exited successfully but produced "
                            f"summary_json={is_json(summary_path)} "
                            f"probe_rows={rows} "
                            f"required_probe_rows={min_probe_rows_after_benchmark}; "
                            f"see {run_log}"
                        )
                    benchmark_finished_code = benchmark_code
                    if early_process and early_code is None:
                        write_status(
                            "benchmark_complete_waiting_early_judge",
                            benchmark_code=benchmark_finished_code,
                            early_code=early_code,
                        )
                        last_running_status = time.monotonic()
                        time.sleep(5)
                        continue
                    write_status(
                        "benchmark_complete",
                        benchmark_code=benchmark_finished_code,
                        early_code=early_code,
                    )
                    break
                now = time.monotonic()
                if now - last_running_status >= 30.0:
                    write_status(
                        "running",
                        benchmark_code=benchmark_code,
                        early_code=early_code,
                    )
                    last_running_status = now
                time.sleep(5)
        finally:
            if early_process is not None:
                if early_process.poll() is None:
                    terminate_process(early_process, "early judge")
                if early_log_file is not None:
                    early_log_file.close()
            if not failure_status_written:
                write_status(
                    "finished",
                    benchmark_code=benchmark_process.poll(),
                    early_code=early_process.poll() if early_process else None,
                )
    return early_launch_command_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--benchmark",
        type=pathlib.Path,
        default=REPO_ROOT / "build/examples/benchmark/cortext_chat_replay_live_run",
    )
    parser.add_argument(
        "--summarizer-provider",
        default="",
        help="Provider URI for the summarizer role (e.g. ollama://host:port/model).",
    )
    parser.add_argument(
        "--extractor-provider",
        default="",
        help="Provider URI for the extractor role (e.g. ollama://host:port/model).",
    )
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument("--skip-messages", type=int, default=0)
    parser.add_argument("--max-messages", type=int, default=1200)
    parser.add_argument("--media-limit", type=int, default=16)
    parser.add_argument(
        "--require-media-modalities",
        default="",
        help=(
            "Comma-separated media kinds that must be present in the exact "
            "frozen replay window after skip/max/media-limit selection. "
            "Supported values: image,video,audio. This is a release harness "
            "coverage guard only; it does not change Cortext behavior."
        ),
    )
    parser.add_argument("--probe-stride", type=int, default=25)
    parser.add_argument("--warmup-events", type=int, default=200)
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--active-history-token-budget", type=int, default=8000)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--judge-repetitions", type=int, default=3)
    parser.add_argument("--judge-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--judge-timeout-s", type=int, default=420)
    parser.add_argument("--judge-context-window-tokens", type=int, default=32768)
    parser.add_argument("--judge-max-output-tokens", type=int, default=1300)
    parser.add_argument(
        "--judge-packet-item-limit",
        type=int,
        default=-1,
        help=(
            "Maximum items per system packet shown to the final local judge. "
            "Use -1 for uncropped release judging. This limits judge prompt "
            "presentation only; benchmark retrieval and token accounting "
            "remain unchanged."
        ),
    )
    parser.add_argument(
        "--early-judge",
        choices=["on", "off"],
        default="on",
        help="Run local early warning judge from streamed probe rows.",
    )
    parser.add_argument(
        "--early-judge-milestones",
        default="1,2,3,4,5,6,7,8,9,10,11,12,16",
    )
    parser.add_argument(
        "--early-judge-periodic-stride",
        type=int,
        default=1,
        help=(
            "After the fixed early milestones, continue local delta judging "
            "every N probe rows until the final summary is written. Use 0 "
            "to stop after --early-judge-milestones."
        ),
    )
    parser.add_argument("--early-judge-poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--early-judge-repetitions",
        type=int,
        default=1,
        help=(
            "Judge repetitions for streamed fail-fast checkpoints. Final release "
            "judging still uses --judge-repetitions and must remain >= 3."
        ),
    )
    parser.add_argument(
        "--early-confirm-fail-repetitions",
        type=int,
        default=3,
        help=(
            "If a streamed checkpoint fails with fewer repetitions, confirm the "
            "same checkpoint with this many local judge repetitions before "
            "terminating replay."
        ),
    )
    parser.add_argument("--early-judge-bootstrap-samples", type=int, default=200)
    # Early-judge conditions mirror the formal judge (full packets, full
    # context window). The previous truncated defaults (32k window, 256-item
    # packets) clipped the fat baseline arms' packets and systematically
    # inflated Cortext's streaming verdicts relative to the formal pass —
    # the 2026-06-10 WM run looked transformative while streaming and merely
    # modest under formal conditions. A cheaper early judge is pointless if
    # its signal does not predict the formal outcome.
    parser.add_argument("--early-judge-timeout-s", type=int, default=900)
    parser.add_argument("--early-judge-context-window-tokens", type=int, default=131072)
    parser.add_argument("--early-judge-max-output-tokens", type=int, default=1300)
    parser.add_argument(
        "--early-judge-packet-item-limit",
        type=int,
        default=-1,
        help=(
            "Maximum items per system packet shown to early local judges "
            "(-1 = full packets, matching the formal judge)."
        ),
    )
    parser.add_argument("--early-quality-gate-min-milestone", type=int, default=8)
    parser.add_argument("--early-quality-trend-gate-min-milestone", type=int, default=4)
    parser.add_argument("--early-quality-trend-window", type=int, default=2)
    parser.add_argument(
        "--early-quality-gate-requires-rag-pressure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Defer quality/win-rate fail-fast gates until traditional chat+RAG "
            "has compacted, added vector RAG outside raw rolling history, or "
            "reached the full active history budget."
        ),
    )
    parser.add_argument(
        "--early-quality-gate-min-history-budget-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument("--early-min-mean-cortext-context-tokens", type=float, default=1.0)
    parser.add_argument("--early-min-cortext-token-savings-vs-rag", type=float, default=0.50)
    parser.add_argument("--early-min-cortext-quality-delta-vs-rag", type=float, default=-0.50)
    parser.add_argument("--early-min-cortext-win-rate", type=float)
    parser.add_argument(
        "--min-probe-rows-after-benchmark",
        type=int,
        default=30,
        help="Fail if the benchmark exits before writing at least this many streamed probe rows.",
    )
    parser.add_argument(
        "--no-kill-on-early-fail",
        action="store_true",
        help="Leave the benchmark running if an early judge gate fails.",
    )
    parser.add_argument("--skip-final-judge", action="store_true")
    parser.add_argument("--skip-initial-report", action="store_true")
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--human-labels", type=pathlib.Path)
    parser.add_argument("--human-label-eval", type=pathlib.Path)
    parser.add_argument("--human-label-sample", type=pathlib.Path)
    parser.add_argument("--human-label-max-probes", type=int, default=40)
    parser.add_argument("--human-label-max-candidates", type=int, default=12)
    parser.add_argument("--human-label-min-overlap", type=float, default=0.20)
    parser.add_argument("--human-label-shuffle-probes", action="store_true")
    parser.add_argument("--human-label-host", default="127.0.0.1")
    parser.add_argument("--human-label-port", type=int, default=7860)
    parser.add_argument(
        "--skip-human-label-sample",
        action="store_true",
        help=(
            "Do not build the post-run Gradio labeling sample or finalizer "
            "handoff command. Strict final pass then requires preexisting "
            "--human-labels and --human-label-eval artifacts."
        ),
    )
    parser.add_argument("--target-freeze", type=pathlib.Path)
    parser.add_argument(
        "--release-freeze",
        type=pathlib.Path,
        help=(
            "Existing release_protocol_freeze.json to validate before replay "
            "and again after report generation. When omitted, the runner "
            "creates one under --out-dir from the first report."
        ),
    )
    parser.add_argument("--judge-media-smoke", type=pathlib.Path)
    parser.add_argument(
        "--require-final-report-pass",
        action="store_true",
        help=(
            "Require the final report emitted by this runner or the ablation "
            "pipeline to pass every release gate. This needs human labels, "
            "human-label retrieval eval, frozen targets, media smoke, and "
            "all ablations."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate local prerequisites and commands without deleting or running replay.",
    )
    parser.add_argument("--force", action="store_true", help="Remove prior runner outputs in out-dir.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.benchmark = args.benchmark.resolve()
    args.input_dir = args.input_dir.resolve()
    if args.human_labels is not None:
        args.human_labels = args.human_labels.resolve()
    if args.human_label_eval is not None:
        args.human_label_eval = args.human_label_eval.resolve()
    if args.human_label_sample is not None:
        args.human_label_sample = args.human_label_sample.resolve()
    if args.target_freeze is not None:
        args.target_freeze = args.target_freeze.resolve()
    if args.release_freeze is not None:
        args.release_freeze = args.release_freeze.resolve()
    if args.judge_media_smoke is not None:
        args.judge_media_smoke = args.judge_media_smoke.resolve()

    if args.skip_messages < 0:
        raise RuntimeError("--skip-messages must be non-negative")
    if args.max_messages <= 0 or args.media_limit < 0:
        raise RuntimeError("--max-messages must be positive and --media-limit non-negative")
    parse_required_modalities(args.require_media_modalities)
    if args.judge_repetitions < 3:
        raise RuntimeError("release judge repetitions must be at least 3")
    if args.early_judge_repetitions <= 0:
        raise RuntimeError("--early-judge-repetitions must be positive")
    if args.early_confirm_fail_repetitions < 0:
        raise RuntimeError("--early-confirm-fail-repetitions must be >= 0")
    if args.early_judge_periodic_stride < 0:
        raise RuntimeError("--early-judge-periodic-stride must be >= 0")
    if args.early_judge_poll_seconds <= 0:
        raise RuntimeError("--early-judge-poll-seconds must be positive")
    if args.early_judge_timeout_s < 1:
        raise RuntimeError("--early-judge-timeout-s must be >= 1")
    if args.judge_context_window_tokens < 1:
        raise RuntimeError("--judge-context-window-tokens must be >= 1")
    if args.judge_max_output_tokens < 1:
        raise RuntimeError("--judge-max-output-tokens must be >= 1")
    if args.early_judge_max_output_tokens < 1:
        raise RuntimeError("--early-judge-max-output-tokens must be >= 1")
    if args.judge_packet_item_limit == 0 or args.judge_packet_item_limit < -1:
        raise RuntimeError("--judge-packet-item-limit must be -1 or a positive item limit")
    if args.early_judge_context_window_tokens < 1:
        raise RuntimeError("--early-judge-context-window-tokens must be >= 1")
    if (
        args.early_judge_packet_item_limit == 0
        or args.early_judge_packet_item_limit < -1
    ):
        raise RuntimeError(
            "--early-judge-packet-item-limit must be -1 or a positive item limit"
        )
    if args.early_quality_gate_min_milestone < 1:
        raise RuntimeError("--early-quality-gate-min-milestone must be >= 1")
    if args.early_quality_trend_gate_min_milestone < 1:
        raise RuntimeError("--early-quality-trend-gate-min-milestone must be >= 1")
    if args.early_quality_trend_window < 0:
        raise RuntimeError("--early-quality-trend-window must be >= 0")
    if not 0.0 <= args.early_quality_gate_min_history_budget_ratio <= 1.0:
        raise RuntimeError(
            "--early-quality-gate-min-history-budget-ratio must be in [0, 1]"
        )
    if args.min_probe_rows_after_benchmark < 0:
        raise RuntimeError("--min-probe-rows-after-benchmark must be non-negative")
    if args.human_label_max_probes < 1:
        raise RuntimeError("--human-label-max-probes must be positive")
    if args.human_label_max_candidates < 1:
        raise RuntimeError("--human-label-max-candidates must be positive")
    if args.human_label_min_overlap < 0.0:
        raise RuntimeError("--human-label-min-overlap must be non-negative")
    if args.human_label_port < 1:
        raise RuntimeError("--human-label-port must be positive")
    transcript = discover_transcript(args.input_dir)
    if not transcript.is_file():
        raise RuntimeError(
            "transcript not found under --input-dir: "
            f"{transcript}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    db = args.out_dir / "live.sqlite"
    summary = args.out_dir / "summary.json"
    summary_gate = args.out_dir / "summary_gate.json"
    judge = args.out_dir / "judge_gemma4_12b_local.json"
    initial_report = args.out_dir / "release_protocol_report_initial.json"
    release_freeze = args.release_freeze or (args.out_dir / "release_protocol_freeze.json")
    generated_release_freeze = args.release_freeze is None
    env_snapshot = args.out_dir / "benchmark_environment_snapshot.json"
    command_manifest = args.out_dir / "command_manifest.json"
    early_failure_report = args.out_dir / "early_failure_report.json"
    judge_media_smoke = args.judge_media_smoke or (
        args.out_dir / "judge_media_smoke_ollama.json"
    )
    human_label_sample = args.human_label_sample or (
        args.out_dir / "human_label_sample.json"
    )

    benchmark_cmd = build_benchmark_command(args, db, summary)
    benchmark_command_text = shlex.join(benchmark_cmd)
    artifact = executable_artifact(benchmark_command_text)
    if not artifact["exists"]:
        raise RuntimeError(f"benchmark executable not found: {args.benchmark}")
    supports_probe_stream = executable_supports_probe_stream(artifact)
    if args.early_judge == "on" and not supports_probe_stream:
        raise RuntimeError(
            "early judge is on, but benchmark executable does not emit probe streams; rebuild first"
        )

    early_cmd = (
        build_early_judge_command(args, db, summary, args.out_dir)
        if args.early_judge == "on"
        else None
    )
    final_judge_cmd = build_final_judge_command(args, db, summary, judge)
    judge_media_smoke_cmd = build_judge_media_smoke_command(args, judge_media_smoke)
    human_label_sample_cmd = build_human_label_sample_command(
        args,
        db,
        summary,
        human_label_sample,
    )
    human_label_launch_cmd = build_human_label_launch_command(
        args,
        human_label_sample,
    )
    finalizer_cmd = build_finalizer_command(
        args,
        db,
        summary,
        judge,
        human_label_sample,
        judge_media_smoke,
        benchmark_command_text,
        final_judge_cmd,
    )
    report_cmd_bootstrap = build_report_command(
        benchmark_cmd,
        final_judge_cmd,
        summary,
        judge,
        initial_report,
        judge_media_smoke,
    )
    report_cmd = build_report_command(
        benchmark_cmd,
        final_judge_cmd,
        summary,
        judge,
        initial_report,
        judge_media_smoke,
        release_freeze,
        args.human_labels,
        args.human_label_eval,
        args.target_freeze,
    )

    if args.preflight_only:
        ensure_default_judge_media_smoke(
            args,
            judge_media_smoke,
            judge_media_smoke_cmd,
        )
        preflight_report = run_preflight(
            args,
            benchmark_command=benchmark_command_text,
            benchmark_artifact=artifact,
            supports_probe_stream=supports_probe_stream,
            early_command=early_cmd,
            judge_media_smoke_command=judge_media_smoke_cmd,
            human_label_sample_command=human_label_sample_cmd,
            human_label_launch_command=human_label_launch_cmd,
            finalizer_command=finalizer_cmd,
            final_judge_command=final_judge_cmd,
            bootstrap_report_command=report_cmd_bootstrap,
            freeze_report_command=report_cmd,
            release_freeze_path=args.release_freeze,
        )
        write_json(args.out_dir / "preflight_report.json", preflight_report)
        print(json.dumps(preflight_report, indent=2), flush=True)
        return 0 if preflight_report["status"] == "pass" else 2

    if args.force and args.out_dir.exists():
        cleanup_runner_outputs(args.out_dir)

    if not args.force:
        existing = [
            path
            for path in [db, summary, judge, initial_report, release_freeze]
            if path.exists() and (generated_release_freeze or path != release_freeze)
        ]
        if existing:
            raise RuntimeError(
                "refusing to overwrite existing release outputs without --force: "
                + ", ".join(str(path) for path in existing)
            )

    ensure_default_judge_media_smoke(
        args,
        judge_media_smoke,
        judge_media_smoke_cmd,
    )

    preflight_report = run_preflight(
        args,
        benchmark_command=benchmark_command_text,
        benchmark_artifact=artifact,
        supports_probe_stream=supports_probe_stream,
        early_command=early_cmd,
        judge_media_smoke_command=judge_media_smoke_cmd,
        human_label_sample_command=human_label_sample_cmd,
        human_label_launch_command=human_label_launch_cmd,
        finalizer_command=finalizer_cmd,
        final_judge_command=final_judge_cmd,
        bootstrap_report_command=report_cmd_bootstrap,
        freeze_report_command=report_cmd,
        release_freeze_path=args.release_freeze,
    )
    write_json(args.out_dir / "preflight_report.json", preflight_report)
    if preflight_report["status"] != "pass":
        print(json.dumps(preflight_report, indent=2), flush=True)
        return 2

    write_json(
        command_manifest,
        {
            "schema": "cortext_chat_replay_release_protocol_runner_v1",
            "created_at_utc": utc_now(),
            "privacy": "private local artifact; contains paths and commands, not message text",
            "git": git_provenance(),
            "fixed_protocol": {
                "daily_consolidation": True,
                "deep_consolidation": True,
                "slice": {
                    "skip_messages": args.skip_messages,
                    "max_messages": args.max_messages,
                    "media_limit": args.media_limit,
                    "required_media_modalities": parse_required_modalities(
                        args.require_media_modalities
                    ),
                },
                "knobs": {"focus": 0.5, "sensitivity": 0.5, "stability": 0.5},
                "normal_rag": "rolling chat history until compaction plus text vector RAG",
                "judge_provider": "local_ollama",
                "judge_model": args.judge_model,
                "judge_context_window_tokens": args.judge_context_window_tokens,
                "early_judge_context_window_tokens": args.early_judge_context_window_tokens,
                "judge_packet_item_limit": args.judge_packet_item_limit,
                "early_judge_packet_item_limit": args.early_judge_packet_item_limit,
                "early_judge_ollama_keep_alive": "0s",
                "early_confirm_fail_repetitions": args.early_confirm_fail_repetitions,
                "early_judge_milestones": args.early_judge_milestones,
                "early_judge_periodic_stride": args.early_judge_periodic_stride,
                "early_quality_gate_min_milestone": args.early_quality_gate_min_milestone,
                "early_quality_trend_gate_min_milestone": (
                    args.early_quality_trend_gate_min_milestone
                ),
                "early_quality_trend_window": args.early_quality_trend_window,
                "early_quality_gate_requires_rag_pressure": (
                    args.early_quality_gate_requires_rag_pressure
                ),
                "early_quality_gate_min_history_budget_ratio": (
                    args.early_quality_gate_min_history_budget_ratio
                ),
                "early_judge_pauses_benchmark_during_local_judge": True,
                "blind_packets": True,
                "strict_final_report_requested": args.require_final_report_pass,
                "human_label_sample_generated": not args.skip_human_label_sample,
                "strict_final_pass_deferred_to_finalizer": (
                    args.require_final_report_pass
                    and not (
                        args.human_labels
                        and args.human_label_eval
                        and args.target_freeze
                    )
                ),
            },
            "benchmark_executable": artifact,
            "benchmark_supports_probe_stream": supports_probe_stream,
            "cortext_behavior_env_guard": cortext_behavior_env_guard(
                dict(os.environ)
            ),
            "benchmark_command": benchmark_command_text,
            "early_judge_command": shlex.join(early_cmd) if early_cmd else "",
            "early_judge_command_launched": "",
            "early_judge_command_launched_file": str(
                args.out_dir / "early_judge_command_launched.txt"
            ),
            "judge_media_smoke_command": shlex.join(judge_media_smoke_cmd),
            "human_label_sample_command": shlex.join(human_label_sample_cmd),
            "human_label_launch_command": shlex.join(human_label_launch_cmd),
            "finalizer_command": shlex.join(finalizer_cmd),
            "final_judge_command": shlex.join(final_judge_cmd),
            "bootstrap_report_command": shlex.join(report_cmd_bootstrap),
            "initial_report_command": shlex.join(report_cmd),
            "release_freeze": str(release_freeze),
            "early_failure_report": str(early_failure_report),
            "frozen_probe_manifest": str(args.out_dir / "frozen_probe_manifest.json"),
            "frozen_probe_schedule": str(args.out_dir / "frozen_probe_schedule.json"),
            "human_label_sample": str(human_label_sample),
            "human_label_launch_command_file": str(
                args.out_dir / "human_label_launch_command.txt"
            ),
            "finalizer_command_file": str(args.out_dir / "finalize_release_command.txt"),
            "human_labels": str(args.human_labels) if args.human_labels else "",
            "human_label_eval": (
                str(args.human_label_eval) if args.human_label_eval else ""
            ),
            "target_freeze": str(args.target_freeze) if args.target_freeze else "",
            "judge_media_smoke": str(judge_media_smoke),
            "judge_media_smoke_sha256": file_sha256(judge_media_smoke)
            if judge_media_smoke.exists()
            else "",
            "environment_snapshot": str(env_snapshot),
        },
    )
    write_text(args.out_dir / "benchmark_command.txt", benchmark_command_text)
    write_text(args.out_dir / "judge_command.txt", shlex.join(final_judge_cmd))
    write_text(
        args.out_dir / "human_label_launch_command.txt",
        shlex.join(human_label_launch_cmd),
    )
    write_text(
        args.out_dir / "finalize_release_command.txt",
        shlex.join(finalizer_cmd),
    )

    early_launch_command_text = run_benchmark_with_early_judge(
        benchmark_cmd,
        early_cmd,
        args.out_dir / "run.log",
        args.out_dir / "early_judge_pipeline.log",
        kill_on_early_fail=not args.no_kill_on_early_fail,
        status_path=args.out_dir / "benchmark_status.json",
        summary_path=summary,
        environment_snapshot_path=env_snapshot,
        command_manifest_path=command_manifest,
        early_failure_report_path=early_failure_report,
        min_probe_rows_after_benchmark=args.min_probe_rows_after_benchmark,
    )
    if early_launch_command_text:
        manifest = json.loads(command_manifest.read_text())
        manifest["early_judge_command_launched"] = early_launch_command_text
        write_json(command_manifest, manifest)

    gate = validate_summary(
        summary,
        summary_gate,
        args.min_probe_rows_after_benchmark,
    )
    print(json.dumps(gate, indent=2), flush=True)
    if gate["overall_status"] != "pass":
        return 2

    if not args.skip_human_label_sample:
        run_checked(
            human_label_sample_cmd,
            args.out_dir / "human_label_sample_pipeline.log",
        )

    if not args.skip_final_judge:
        run_checked(final_judge_cmd, args.out_dir / "judge_pipeline.log")
    if not args.skip_initial_report:
        run_checked(report_cmd_bootstrap, args.out_dir / "release_protocol_report_bootstrap.log")
        write_or_validate_release_freeze(
            release_freeze,
            initial_report,
            benchmark_command_text,
        )
        write_probe_freeze_artifacts(initial_report, args.out_dir)
        run_checked(report_cmd, args.out_dir / "main_report_pipeline.log")

    if args.run_ablations:
        external_strict_artifacts_available = bool(
            args.human_labels
            and args.human_label_eval
            and args.target_freeze
        )
        ablation_cmd = [
            sys.executable,
            str(REPO_ROOT / "tools/run_chat_replay_release_ablations.py"),
            "--base",
            str(args.out_dir / "ablations"),
            "--main-report",
            str(initial_report),
            "--main-summary",
            str(summary),
            "--main-judge",
            str(judge),
            "--final-report",
            str(args.out_dir / "release_protocol_report_with_ablations.json"),
            "--benchmark-command",
            benchmark_command_text,
            "--judge-command",
            shlex.join(final_judge_cmd),
            "--judge-media-smoke",
            str(judge_media_smoke),
            "--ablation-plan",
            str(args.out_dir / "ablation_plan.json"),
            "--early-judge",
            "on",
            "--early-judge-milestones",
            args.early_judge_milestones,
            "--early-judge-periodic-stride",
            str(args.early_judge_periodic_stride),
            "--early-judge-poll-seconds",
            str(args.early_judge_poll_seconds),
            "--early-judge-repetitions",
            str(args.early_judge_repetitions),
            "--early-confirm-fail-repetitions",
            str(args.early_confirm_fail_repetitions),
            "--early-judge-bootstrap-samples",
            str(args.early_judge_bootstrap_samples),
            "--early-judge-timeout-s",
            str(args.early_judge_timeout_s),
            "--early-judge-context-window-tokens",
            str(args.early_judge_context_window_tokens),
            "--early-judge-max-output-tokens",
            str(args.early_judge_max_output_tokens),
            "--early-judge-packet-item-limit",
            str(args.early_judge_packet_item_limit),
            "--judge-packet-item-limit",
            str(args.judge_packet_item_limit),
            "--early-quality-gate-min-milestone",
            str(args.early_quality_gate_min_milestone),
            "--early-quality-trend-gate-min-milestone",
            str(args.early_quality_trend_gate_min_milestone),
            "--early-quality-trend-window",
            str(args.early_quality_trend_window),
            "--early-quality-gate-min-history-budget-ratio",
            str(args.early_quality_gate_min_history_budget_ratio),
            "--early-min-mean-cortext-context-tokens",
            str(args.early_min_mean_cortext_context_tokens),
            "--early-min-cortext-token-savings-vs-rag",
            str(args.early_min_cortext_token_savings_vs_rag),
            "--early-min-cortext-quality-delta-vs-rag",
            str(args.early_min_cortext_quality_delta_vs_rag),
        ]
        if not args.early_quality_gate_requires_rag_pressure:
            ablation_cmd.append("--no-early-quality-gate-requires-rag-pressure")
        if args.human_labels is not None:
            ablation_cmd += ["--human-labels", str(args.human_labels)]
        if args.human_label_eval is not None:
            ablation_cmd += ["--human-label-eval", str(args.human_label_eval)]
        if args.target_freeze is not None:
            ablation_cmd += ["--target-freeze", str(args.target_freeze)]
        if args.require_final_report_pass and external_strict_artifacts_available:
            ablation_cmd.append("--require-final-report-pass")
        if args.early_min_cortext_win_rate is not None:
            ablation_cmd += [
                "--early-min-cortext-win-rate",
                str(args.early_min_cortext_win_rate),
            ]
        run_checked(ablation_cmd, args.out_dir / "ablation_pipeline.log")

    print(f"[runner] complete: {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
