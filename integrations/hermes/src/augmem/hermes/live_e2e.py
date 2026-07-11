"""Live-model end-to-end verification for silent Cortext memory.

Design goals (cold-start, no chat-history cheating):

1. **Session A** stores a fact through Hermes provider seams into SQLite.
2. Provider is fully **shutdown** (engine closed) — no in-process state left.
3. **Session B** opens a **new** provider on the same DB only (fresh process
   state / empty conversation). Prefetch is the sole source of the fact.
4. A live model answers the probe with:
   - **control**: no prior context → must *not* know allergy/appointment
   - **treatment**: only unbranded prefetch packet → must use the fact
5. Invisibility: no tools, empty system_prompt_block, no "cortext" branding.

This is intentionally stricter than stuffing the fact into ongoing chat
history. The model never sees Session A messages — only Cortext recall.

Enable with::

    export CORTEXT_HERMES_LIVE=1
    export OPENAI_API_KEY=...
    export OPENAI_MODEL=gpt-5.4-mini
    export CORTEXT_LIBRARY_PATH=/path/to/libcortext.dylib

    uv run augmem-hermes live-e2e --force
"""

from __future__ import annotations

import json
import os
import re
import ssl
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from augmem.hermes.provider import CortextMemoryProvider


def _ssl_context() -> ssl.SSLContext:
  """Prefer certifi CA bundle (macOS Python often lacks system roots)."""
  try:
    import certifi

    return ssl.create_default_context(cafile=certifi.where())
  except Exception:
    try:
      return ssl.create_default_context()
    except Exception:
      return ssl._create_unverified_context()  # noqa: S323 — last-resort live e2e


# Unique-ish marker so control answers cannot luck into our secret via common knowledge.
SECRET_CODE = "ZX-BAILEY-7741"
FACT = (
  f"Bailey (patient code {SECRET_CODE}) is allergic to bee stings and needs "
  "Benadryl within 10 minutes. The vet appointment is July 14 at 2pm with Dr. Okafor."
)
# Probe must NOT restate the secret fact — only ask for it.
PROBE = (
  "I am starting a brand new conversation with no prior messages. "
  "From durable memory only, what should the vet know about the dog Bailey? "
  "Include any allergy treatment and appointment details if you have them. "
  "If you do not know, say you do not know."
)

# Groups that should appear only if memory (or cheating) provided the fact.
# "bailey" alone is weak (probe names Bailey). Prefer secret + medical details.
MEMORY_GROUPS = (
  (SECRET_CODE.lower(),),
  ("bee", "sting", "allergic", "allergy"),
  ("benadryl",),
  ("july", "14", "okafor"),
)


@dataclass
class LiveE2EResult:
  ok: bool
  skipped: bool = False
  skip_reason: str = ""
  prefetch: str = ""
  answer: str = ""
  control_answer: str = ""
  matched_groups: list[str] = field(default_factory=list)
  missed_groups: list[str] = field(default_factory=list)
  control_matched_groups: list[str] = field(default_factory=list)
  invisible_ok: bool = False
  cold_start_ok: bool = False
  control_ok: bool = False
  latency_ms: dict[str, float] = field(default_factory=dict)
  model: str = ""
  db_path: str = ""
  details: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def live_enabled() -> bool:
  return os.environ.get("CORTEXT_HERMES_LIVE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
  }


def _env(name: str, default: str = "") -> str:
  return (os.environ.get(name) or default).strip()


def _chat_completion(
  *,
  system: str,
  user: str,
  model: str,
  base_url: str,
  api_key: str,
  timeout: float = 120.0,
) -> str:
  url = base_url.rstrip("/") + "/chat/completions"
  payload = {
    "model": model,
    "temperature": 0,
    "messages": [
      {"role": "system", "content": system},
      {"role": "user", "content": user},
    ],
  }
  req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}",
    },
    method="POST",
  )
  with urllib.request.urlopen(
    req, timeout=timeout, context=_ssl_context()
  ) as resp:
    body = json.loads(resp.read().decode("utf-8"))
  choices = body.get("choices") or []
  if not choices:
    raise RuntimeError(f"empty chat completion response: {body!r}")
  msg = choices[0].get("message") or {}
  content = msg.get("content")
  if isinstance(content, list):
    parts = []
    for part in content:
      if isinstance(part, dict) and isinstance(part.get("text"), str):
        parts.append(part["text"])
      elif isinstance(part, str):
        parts.append(part)
    return "\n".join(parts).strip()
  if not isinstance(content, str) or not content.strip():
    raise RuntimeError(f"missing message content: {body!r}")
  return content.strip()


def _match_groups(answer: str) -> tuple[list[str], list[str]]:
  lower = answer.lower()
  matched: list[str] = []
  missed: list[str] = []
  for group in MEMORY_GROUPS:
    if any(token in lower for token in group):
      matched.append("|".join(group))
    else:
      missed.append("|".join(group))
  return matched, missed


def _memory_score(matched: list[str]) -> int:
  return len(matched)


def _wait_bg(provider: CortextMemoryProvider, timeout: float = 120.0) -> None:
  provider._join_background(timeout=timeout)
  # If work is still marked unfinished after the join timeout, surface it.
  try:
    pending = int(getattr(provider._ingest_queue, "unfinished_tasks", 0))
  except Exception:
    pending = 0
  if pending > 0:
    raise TimeoutError("background Cortext ingest did not finish in time")


def _system_with_optional_prior(prefetch: str) -> str:
  base = (
    "You are a careful assistant in a brand-new conversation. "
    "Do not invent pet medical facts. "
    "If prior context is provided, use it when relevant. "
    "If you lack specifics, say you do not know."
  )
  if prefetch.strip():
    return base + "\n\nPrior context from durable memory:\n" + prefetch.strip()
  return base


def run_live_e2e(
  *,
  hermes_home: Path | None = None,
  require_enabled: bool = True,
) -> LiveE2EResult:
  """Cold-start Bailey scenario with control + treatment live model calls."""
  if require_enabled and not live_enabled():
    return LiveE2EResult(
      ok=False,
      skipped=True,
      skip_reason="set CORTEXT_HERMES_LIVE=1 to enable live model verification",
    )

  api_key = _env("OPENAI_API_KEY") or _env("CORTEXT_LIVE_API_KEY")
  if not api_key:
    return LiveE2EResult(
      ok=False,
      skipped=True,
      skip_reason="OPENAI_API_KEY (or CORTEXT_LIVE_API_KEY) is required",
    )

  base_url = (
    _env("OPENAI_BASE_URL")
    or _env("CORTEXT_LIVE_BASE_URL")
    or "https://api.openai.com/v1"
  )
  model = (
    _env("OPENAI_MODEL")
    or _env("CORTEXT_LIVE_MODEL")
    or "gpt-4o-mini"
  )

  probe = CortextMemoryProvider()
  if not probe.is_available():
    return LiveE2EResult(
      ok=False,
      skipped=True,
      skip_reason=(
        "augmem.cortext is not available (build/install native libcortext "
        "and set CORTEXT_LIBRARY_PATH if needed)"
      ),
    )

  if probe.get_tool_schemas() != []:
    return LiveE2EResult(
      ok=False,
      details=["get_tool_schemas() must be empty (agent must not see tools)"],
    )
  if probe.system_prompt_block() != "":
    return LiveE2EResult(
      ok=False,
      details=["system_prompt_block() must be empty"],
    )

  home = hermes_home or Path(tempfile.mkdtemp(prefix="cortext-hermes-live-"))
  home.mkdir(parents=True, exist_ok=True)
  latencies: dict[str, float] = {}
  details: list[str] = []
  db_path = ""
  writer: CortextMemoryProvider | None = None
  reader: CortextMemoryProvider | None = None

  try:
    # ------------------------------------------------------------------
    # Session A — write only, then full shutdown (no live context left).
    # ------------------------------------------------------------------
    writer = CortextMemoryProvider()
    t0 = time.perf_counter()
    writer.initialize(
      "live-session-a",
      hermes_home=str(home),
      platform="cli",
      agent_context="primary",
      user_id="live-user",
      agent_identity="live-agent",
    )
    latencies["initialize_write_ms"] = (time.perf_counter() - t0) * 1000.0
    db_path = str(writer._db_path or "")

    t0 = time.perf_counter()
    writer.on_turn_start(1, FACT)
    _wait_bg(writer)
    latencies["user_seam_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    writer.sync_turn(
      FACT,
      "Understood — stored for later sessions.",
    )
    _wait_bg(writer)
    writer.on_session_end([])
    writer.shutdown()
    writer = None
    latencies["post_llm_and_shutdown_ms"] = (time.perf_counter() - t0) * 1000.0
    details.append("session A wrote fact and fully shut down the engine")

    # ------------------------------------------------------------------
    # Session B — cold start: new provider instance, same SQLite file only.
    # No Session A messages exist in this "conversation".
    # ------------------------------------------------------------------
    reader = CortextMemoryProvider()
    t0 = time.perf_counter()
    reader.initialize(
      "live-session-b",
      hermes_home=str(home),
      platform="cli",
      agent_context="primary",
      user_id="live-user",
      agent_identity="live-agent",
    )
    latencies["initialize_read_ms"] = (time.perf_counter() - t0) * 1000.0

    # Sanity: reader must not share the writer object; db must still exist.
    if not Path(db_path).is_file():
      return LiveE2EResult(
        ok=False,
        db_path=db_path,
        model=model,
        latency_ms=latencies,
        details=[f"sqlite db missing after writer shutdown: {db_path}"],
      )

    t0 = time.perf_counter()
    prefetch = reader.prefetch(PROBE, session_id="live-session-b")
    latencies["prefetch_ms"] = (time.perf_counter() - t0) * 1000.0

    cold_start_ok = bool(prefetch.strip()) and (
      SECRET_CODE.lower() in prefetch.lower()
      or "benadryl" in prefetch.lower()
      or "bee" in prefetch.lower()
    )
    if not prefetch.strip():
      return LiveE2EResult(
        ok=False,
        prefetch=prefetch,
        model=model,
        db_path=db_path,
        latency_ms=latencies,
        cold_start_ok=False,
        details=[
          "cold-start prefetch empty — durable SQLite recall failed after restart",
        ],
      )
    if not cold_start_ok:
      return LiveE2EResult(
        ok=False,
        prefetch=prefetch,
        model=model,
        db_path=db_path,
        latency_ms=latencies,
        cold_start_ok=False,
        details=[
          "cold-start prefetch did not contain the stored Bailey fact",
        ],
      )
    if re.search(r"cortext", prefetch, re.I):
      return LiveE2EResult(
        ok=False,
        prefetch=prefetch,
        model=model,
        db_path=db_path,
        latency_ms=latencies,
        cold_start_ok=True,
        details=["prefetch text must not mention 'cortext' (agent-invisible)"],
      )
    details.append(
      "session B cold-started a new engine and prefetched the fact from SQLite"
    )

    # ------------------------------------------------------------------
    # Control: same probe, NO prefetch / NO session history.
    # Model should not know the secret code / medical specifics.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
      control_answer = _chat_completion(
        system=_system_with_optional_prior(""),
        user=PROBE,
        model=model,
        base_url=base_url,
        api_key=api_key,
      )
    except urllib.error.HTTPError as exc:
      err_body = exc.read().decode("utf-8", errors="replace")
      return LiveE2EResult(
        ok=False,
        prefetch=prefetch,
        model=model,
        db_path=db_path,
        latency_ms=latencies,
        cold_start_ok=True,
        details=[f"control chat HTTP {exc.code}: {err_body[:500]}"],
      )
    except Exception as exc:
      return LiveE2EResult(
        ok=False,
        prefetch=prefetch,
        model=model,
        db_path=db_path,
        latency_ms=latencies,
        cold_start_ok=True,
        details=[f"control chat failed: {exc}"],
      )
    latencies["control_llm_ms"] = (time.perf_counter() - t0) * 1000.0

    control_matched, _ = _match_groups(control_answer)
    # Control fails if it already knows secret code or ≥2 memory groups.
    control_ok = SECRET_CODE.lower() not in control_answer.lower() and (
      _memory_score(control_matched) <= 1
    )
    if control_ok:
      details.append(
        "control (no memory) did not already know the secret fact"
      )
    else:
      details.append(
        "control leak: model knew memory content without prefetch "
        f"(matched={control_matched}) — result would be inconclusive"
      )

    # ------------------------------------------------------------------
    # Treatment: probe + cold-start prefetch only (still no Session A chat).
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
      answer = _chat_completion(
        system=_system_with_optional_prior(prefetch),
        user=PROBE,
        model=model,
        base_url=base_url,
        api_key=api_key,
      )
    except urllib.error.HTTPError as exc:
      err_body = exc.read().decode("utf-8", errors="replace")
      return LiveE2EResult(
        ok=False,
        prefetch=prefetch,
        control_answer=control_answer,
        model=model,
        db_path=db_path,
        latency_ms=latencies,
        cold_start_ok=True,
        control_ok=control_ok,
        control_matched_groups=control_matched,
        details=[f"treatment chat HTTP {exc.code}: {err_body[:500]}"],
      )
    except Exception as exc:
      return LiveE2EResult(
        ok=False,
        prefetch=prefetch,
        control_answer=control_answer,
        model=model,
        db_path=db_path,
        latency_ms=latencies,
        cold_start_ok=True,
        control_ok=control_ok,
        control_matched_groups=control_matched,
        details=[f"treatment chat failed: {exc}"],
      )
    latencies["treatment_llm_ms"] = (time.perf_counter() - t0) * 1000.0

    matched, missed = _match_groups(answer)
    # Require secret code OR (allergy + appointment-ish), at least 2 groups.
    has_secret = SECRET_CODE.lower() in answer.lower()
    allergy_ok = any(
      g.startswith("bee") or g.startswith("benadryl") for g in matched
    )
    enough = (has_secret or allergy_ok) and _memory_score(matched) >= 2
    # Treatment must beat control.
    better_than_control = _memory_score(matched) > _memory_score(control_matched)

    invisible_ok = (
      reader.get_tool_schemas() == []
      and reader.system_prompt_block() == ""
      and "cortext" not in prefetch.lower()
      and "cortext" not in answer.lower()
    )

    if enough and better_than_control:
      details.append(
        "treatment answer used cold-start memory and beat no-memory control"
      )
    elif enough and not better_than_control:
      details.append(
        "treatment matched facts but did not beat control "
        f"(treatment={matched}, control={control_matched})"
      )
    else:
      details.append(
        "treatment missing memory content "
        f"(matched={matched}, missed={missed})"
      )

    if invisible_ok:
      details.append("provider remained invisible (no tools/system branding)")
    else:
      details.append("visibility contract failed")

    reader.on_session_end([])
    reader.shutdown()
    reader = None

    ok = (
      cold_start_ok
      and control_ok
      and enough
      and better_than_control
      and invisible_ok
    )
    return LiveE2EResult(
      ok=ok,
      prefetch=prefetch,
      answer=answer,
      control_answer=control_answer,
      matched_groups=matched,
      missed_groups=missed,
      control_matched_groups=control_matched,
      invisible_ok=invisible_ok,
      cold_start_ok=cold_start_ok,
      control_ok=control_ok,
      latency_ms=latencies,
      model=model,
      db_path=db_path,
      details=details,
    )
  except Exception as exc:
    for p in (writer, reader):
      if p is not None:
        try:
          p.shutdown()
        except Exception:
          pass
    return LiveE2EResult(
      ok=False,
      latency_ms=latencies,
      db_path=db_path,
      model=model,
      details=[f"live e2e crashed: {exc}"],
    )


def format_report(result: LiveE2EResult) -> str:
  lines = [
    "=== Cortext Hermes live e2e (cold-start + control) ===",
    f"ok: {result.ok}",
    f"skipped: {result.skipped}",
  ]
  if result.skip_reason:
    lines.append(f"skip_reason: {result.skip_reason}")
  if result.model:
    lines.append(f"model: {result.model}")
  if result.db_path:
    lines.append(f"db_path: {result.db_path}")
  if result.latency_ms:
    lines.append(
      "latency_ms: "
      + ", ".join(f"{k}={v:.0f}" for k, v in result.latency_ms.items())
    )
  lines.append(f"cold_start_ok: {result.cold_start_ok}")
  lines.append(f"control_ok: {result.control_ok}")
  lines.append(f"invisible_ok: {result.invisible_ok}")
  if result.control_matched_groups:
    lines.append(f"control_matched: {result.control_matched_groups}")
  if result.matched_groups:
    lines.append(f"treatment_matched: {result.matched_groups}")
  if result.missed_groups:
    lines.append(f"treatment_missed: {result.missed_groups}")
  for d in result.details:
    lines.append(f"- {d}")
  if result.prefetch:
    lines.append("--- cold-start prefetch (only memory channel) ---")
    lines.append(result.prefetch)
  if result.control_answer:
    lines.append("--- control answer (no memory / no history) ---")
    lines.append(result.control_answer)
  if result.answer:
    lines.append("--- treatment answer (cold-start prefetch only) ---")
    lines.append(result.answer)
  return "\n".join(lines) + "\n"
