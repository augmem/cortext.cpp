"""Silent Cortext MemoryProvider for Hermes Agent.

Cortext runs entirely behind the scenes — the LLM never sees tools, never
sees product branding, and never has to "use memory." Hermes lifecycle
hooks do all the work:

1. **User seam** — ingest text + screenshots + media after user input.
2. **Pre-LLM seam** — inject recalled context before every model call.
3. **Action seam** — interrupt a proposed tool action when relevant prior
   context says it should be reconsidered.
4. **Post-LLM seam** — ingest assistant text, tool calls/results, media.

No ``cortext_*`` tools are exposed. ``system_prompt_block`` is empty.
Prefetch text is unlabeled prior context only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from augmem.hermes.media import (
  MediaSignal,
  extract_from_content_parts,
  extract_from_message,
  extract_from_messages,
  media_signal_from_ref,
  process_signal_on_engine,
)

logger = logging.getLogger(__name__)

try:
  from agent.memory_provider import MemoryProvider
except ImportError:  # pragma: no cover - exercised only without Hermes

  class MemoryProvider:  # type: ignore[no-redef]
    """Minimal stand-in so the provider can be unit-tested without Hermes."""

    pass


PROVIDER_NAME = "cortext"
CONFIG_FILENAME = "cortext.json"
DEFAULT_DB_NAME = "cortext.sqlite"
DEFAULT_TOP_K = 6
DEFAULT_FOCUS = 0.55
DEFAULT_SENSITIVITY = 0.50
DEFAULT_STABILITY = 0.65
DEFAULT_USER_ID = "user"
DEFAULT_AGENT_ID = "agent"
_MAX_TOOL_RESULT_CHARS = 2000
_MAX_TOOL_INTENT_CHARS = 2000

# Provenance scheme (opaque to Cortext; used for same-source grouping):
#   hermes/{user|agent}/{actor_id}/{session_id}/...
# Examples:
#   hermes/user/alice/sess-1/turn/3
#   hermes/agent/hermes/sess-1/turn/3
#   hermes/agent/hermes/sess-1/prefetch
#   hermes/user/alice/sess-1/builtin/user


def _memory_text(item: dict[str, Any]) -> str:
  """Decode a retrieved memory item into plain text for prompt injection.

  Native Cortext packets often return ``content`` as base64 blobs rather than
  a top-level ``text`` field.
  """
  import base64

  modality = str(item.get("modality") or "").lower()
  mime = str(item.get("mime") or item.get("mimetype") or "").lower()
  if modality and modality != "text":
    return ""
  if mime.startswith("image/") or mime.startswith("audio/"):
    return ""

  text = item.get("text")
  if isinstance(text, str) and text.strip():
    return text.strip()
  content = item.get("content")
  if isinstance(content, str) and content.strip():
    return content.strip()
  if isinstance(content, list):
    parts: list[str] = []
    for blob in content:
      if isinstance(blob, str):
        parts.append(blob)
      elif isinstance(blob, (bytes, bytearray)):
        parts.append(blob.decode("utf-8", errors="replace"))
      elif isinstance(blob, dict):
        if isinstance(blob.get("text"), str) and blob["text"].strip():
          parts.append(blob["text"].strip())
          continue
        b64 = blob.get("base64")
        if isinstance(b64, str) and b64:
          try:
            raw = base64.b64decode(b64, validate=False)
            parts.append(raw.decode("utf-8", errors="replace"))
          except Exception:
            continue
        elif isinstance(blob.get("data"), (bytes, bytearray)):
          parts.append(bytes(blob["data"]).decode("utf-8", errors="replace"))
    joined = " ".join(p for p in parts if p).strip()
    if joined:
      return joined
  return ""


def _format_memories(items: list[dict[str, Any]], *, limit: int) -> str:
  lines: list[str] = []
  for item in items[:limit]:
    text = _memory_text(item)
    if not text:
      continue
    rel = item.get("rel")
    if isinstance(rel, (int, float)):
      lines.append(f"- [{rel:.3f}] {text}")
    else:
      lines.append(f"- {text}")
  return "\n".join(lines)


def _content_key(text: str) -> str:
  return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _tool_intent_text(tool_name: str, args: dict[str, Any] | None) -> str:
  """Encode a proposed Hermes action as a bounded, readable text signal."""
  name = (tool_name or "").strip()
  if not name:
    return ""
  try:
    serialized_args = json.dumps(
      args or {}, ensure_ascii=False, sort_keys=True, default=str
    )
  except (TypeError, ValueError):
    serialized_args = repr(args)
  text = f"Proposed tool action: {name}\nArguments: {serialized_args}"
  return text[:_MAX_TOOL_INTENT_CHARS]


def _default_config() -> dict[str, Any]:
  return {
    "db_path": f"$HERMES_HOME/{DEFAULT_DB_NAME}",
    "focus": DEFAULT_FOCUS,
    "sensitivity": DEFAULT_SENSITIVITY,
    "stability": DEFAULT_STABILITY,
    "top_k": DEFAULT_TOP_K,
    # Seam toggles — all on by default so Cortext runs at every boundary.
    "seam_user": True,  # after user interaction
    "seam_pre_llm": True,  # before every LLM call
    "seam_pre_tool": True,  # before each tool executes
    "seam_post_llm": True,  # after LLM / completed turn
    "auto_sync_turns": True,  # alias for seam_post_llm (Hermes-friendly name)
    "auto_consolidate": True,
    "ingest_tool_results": True,
    "ingest_media": True,  # screenshots, images, audio across all seams
    # Remote http(s) media fetch is off by default (SSRF / untrusted URLs).
    "fetch_remote_media": False,
  }


def load_config(hermes_home: str | Path | None = None) -> dict[str, Any]:
  """Load ``cortext.json`` from Hermes home, falling back to defaults."""
  cfg = _default_config()
  if hermes_home is None:
    hermes_home = os.environ.get("HERMES_HOME", "")
  if not hermes_home:
    return cfg
  path = Path(hermes_home) / CONFIG_FILENAME
  if not path.is_file():
    return cfg
  try:
    raw = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    logger.warning("Failed to read %s: %s", path, exc)
    return cfg
  if isinstance(raw, dict):
    if "auto_sync_turns" in raw and "seam_post_llm" not in raw:
      raw["seam_post_llm"] = raw["auto_sync_turns"]
    if "seam_post_llm" in raw and "auto_sync_turns" not in raw:
      raw["auto_sync_turns"] = raw["seam_post_llm"]
    cfg.update(raw)
  return cfg


def resolve_db_path(db_path: str, hermes_home: str) -> Path:
  """Expand ``$HERMES_HOME``, ``~``, and relative paths under Hermes home."""
  expanded = (
    str(db_path)
    .replace("$HERMES_HOME", hermes_home)
    .replace("${HERMES_HOME}", hermes_home)
  )
  path = Path(expanded).expanduser()
  if not path.is_absolute():
    path = Path(hermes_home) / path
  return path


def _as_bool(value: Any, default: bool) -> bool:
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    return value.strip().lower() in {"1", "true", "yes", "on"}
  if value is None:
    return default
  return bool(value)


def _sanitize_id(value: str, *, fallback: str) -> str:
  """Keep provenance path segments filesystem/path-safe and non-empty."""
  text = (value or "").strip()
  if not text:
    return fallback
  cleaned = []
  for ch in text:
    if ch.isalnum() or ch in {"-", "_", ".", "@"}:
      cleaned.append(ch)
    else:
      cleaned.append("_")
  out = "".join(cleaned).strip("._")
  return out or fallback


class CortextMemoryProvider(MemoryProvider):
  """Silent local memory provider — invisible to the Hermes agent/LLM.

  Automatic only: no tools, no system-prompt product mention. Memory is
  ingested and recalled through Hermes lifecycle hooks alone.
  """

  def __init__(self, config: dict[str, Any] | None = None) -> None:
    self._config = dict(config) if config is not None else _default_config()
    self._session_id = ""
    self._hermes_home = ""
    self._platform = ""
    self._agent_context = "primary"
    self._user_id = DEFAULT_USER_ID
    self._agent_id = DEFAULT_AGENT_ID
    self._engine: Any | None = None
    self._db_path: Path | None = None
    # Single worker + queue so hooks never block joining a prior ingest thread.
    self._ingest_queue: queue.Queue[Callable[[], None] | None] = queue.Queue()
    self._ingest_worker: threading.Thread | None = None
    self._ingest_worker_lock = threading.Lock()
    # Kept for tests/live harness that wait on background completion.
    self._bg_thread: threading.Thread | None = None
    self._prefetch_lock = threading.Lock()
    self._cached_prefetch = ""
    self._cached_prefetch_query = ""
    self._write_gate = threading.Lock()
    # Deduplicate user ingest when both on_turn_start and sync_turn fire for
    # the same known turn. Content alone is not a turn identity: users may
    # intentionally repeat an instruction in a later turn.
    self._user_key_lock = threading.Lock()
    self._last_user_turn_key = ""
    self._pending_user_keys: set[str] = set()
    self._turn_number = 0

  @property
  def name(self) -> str:
    return PROVIDER_NAME

  def is_available(self) -> bool:
    try:
      import augmem.cortext  # noqa: F401
    except Exception:
      return False
    return True

  def get_config_schema(self) -> list[dict[str, Any]]:
    return [
      {
        "key": "db_path",
        "description": "Cortext SQLite database path",
        "default": f"$HERMES_HOME/{DEFAULT_DB_NAME}",
      },
      {
        "key": "focus",
        "description": "Focus (F) knob — retrieval selectivity [0,1]",
        "default": str(DEFAULT_FOCUS),
      },
      {
        "key": "sensitivity",
        "description": "Sensitivity (S) knob — responsiveness [0,1]",
        "default": str(DEFAULT_SENSITIVITY),
      },
      {
        "key": "stability",
        "description": "Stability (T) knob — durability bias [0,1]",
        "default": str(DEFAULT_STABILITY),
      },
      {
        "key": "seam_pre_tool",
        "description": "Interrupt tool actions when recalled context requires reconsideration",
        "default": "true",
      },
    ]

  def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
    path = Path(hermes_home) / CONFIG_FILENAME
    existing: dict[str, Any] = {}
    if path.is_file():
      try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
          existing = raw
      except (OSError, json.JSONDecodeError):
        existing = {}

    merged = {**_default_config(), **existing}
    bool_keys = {
      "auto_sync_turns",
      "auto_consolidate",
      "seam_user",
      "seam_pre_llm",
      "seam_pre_tool",
      "seam_post_llm",
      "ingest_tool_results",
      "ingest_media",
      "fetch_remote_media",
    }
    for key, value in values.items():
      if key in {"focus", "sensitivity", "stability"}:
        try:
          merged[key] = float(value)
        except (TypeError, ValueError):
          continue
      elif key == "top_k":
        try:
          merged[key] = int(value)
        except (TypeError, ValueError):
          continue
      elif key in bool_keys:
        merged[key] = _as_bool(value, bool(merged.get(key, True)))
      elif key == "db_path" and value:
        merged[key] = str(value)

    # Keep alias in sync: auto_sync_turns ↔ seam_post_llm.
    if "auto_sync_turns" in values and "seam_post_llm" not in values:
      merged["seam_post_llm"] = merged["auto_sync_turns"]
    if "seam_post_llm" in values and "auto_sync_turns" not in values:
      merged["auto_sync_turns"] = merged["seam_post_llm"]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    self._config = merged

  def initialize(self, session_id: str, **kwargs: Any) -> None:
    import augmem.cortext as cortext

    hermes_home = str(
      kwargs.get("hermes_home")
      or os.environ.get("HERMES_HOME")
      or Path.home() / ".hermes"
    )
    self._hermes_home = hermes_home
    self._session_id = session_id or "session"
    self._platform = str(kwargs.get("platform") or "")
    self._agent_context = str(kwargs.get("agent_context") or "primary")
    self._config = load_config(hermes_home)
    self._user_id = self._resolve_user_id(kwargs)
    self._agent_id = self._resolve_agent_id(kwargs)
    self._last_user_turn_key = ""
    self._turn_number = 0

    db_path = resolve_db_path(
      str(self._config.get("db_path", f"$HERMES_HOME/{DEFAULT_DB_NAME}")),
      hermes_home,
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    self._db_path = db_path

    cfg = cortext.Config(
      focus=float(self._config.get("focus", DEFAULT_FOCUS)),
      sensitivity=float(self._config.get("sensitivity", DEFAULT_SENSITIVITY)),
      stability=float(self._config.get("stability", DEFAULT_STABILITY)),
    )
    self._engine = cortext.Cortext(str(db_path), config=cfg)
    logger.info(
      "Cortext ready (db=%s session=%s user=%s agent=%s seams=user/pre_llm/pre_tool/post_llm)",
      db_path,
      self._session_id,
      self._user_id,
      self._agent_id,
    )

  def system_prompt_block(self) -> str:
    # Intentionally empty: the agent must not know a memory backend exists.
    # Recall is injected only via prefetch() as unlabeled prior context.
    return ""

  # =========================================================================
  # Seam 1 — USER: after any user interaction
  # =========================================================================

  def on_turn_start(self, turn_number: int, message: Any, **kwargs: Any) -> None:
    """User seam: ingest user text + any screenshots/media as the turn begins."""
    self._turn_number = int(turn_number) if turn_number is not None else self._turn_number
    if not self._seam_user_enabled():
      return
    sid = str(kwargs.get("session_id") or self._session_id or "session")
    if kwargs.get("user_id") or kwargs.get("user_id_alt"):
      self._user_id = self._resolve_user_id(kwargs)

    base = self._source_id("user", "turn", str(self._turn_number), session_id=sid)
    signals = self._signals_from_user_payload(message, source_id_base=base, **kwargs)
    if not signals:
      return

    mark = ""
    for sig in signals:
      if sig.modality == "text" and sig.text:
        mark = self._user_turn_key(sid, self._turn_number, sig.text)
        break
    prefetch_q = next((s.text for s in signals if s.modality == "text" and s.text), "")
    self._ingest_async(
      signals,
      mark_user_key=mark or None,
      also_warm_prefetch=prefetch_q or None,
    )

  def on_memory_write(
    self,
    action: str,
    target: str,
    content: str,
    metadata: dict[str, Any] | None = None,
  ) -> None:
    """User seam: mirror Hermes built-in MEMORY.md / USER.md writes."""
    del metadata
    if not self._seam_user_enabled():
      return
    if not self._should_write() or not self._engine:
      return
    if action not in {"add", "write", "append"}:
      return
    text = (content or "").strip()
    if not text:
      return
    target_seg = _sanitize_id(str(target or "memory"), fallback="memory")
    source = self._source_id("user", "builtin", target_seg)
    self._ingest_async(
      [MediaSignal(modality="text", source_id=source, text=text)]
    )

  def on_delegation(
    self,
    task: str,
    result: str,
    **kwargs: Any,
  ) -> None:
    """Agent seam: capture subagent task + result (text and any media refs)."""
    if not self._seam_user_enabled() or not self._should_write():
      return
    del kwargs
    signals: list[MediaSignal] = []
    task_text = (task or "").strip()
    result_text = (result or "").strip()
    allow_remote = self._fetch_remote_media()
    if task_text:
      signals.extend(
        extract_from_content_parts(
          task_text[:_MAX_TOOL_RESULT_CHARS],
          source_id_base=self._source_id("agent", "delegation", "task"),
          allow_remote_url=allow_remote,
          include_media=self._media_enabled(),
        )
      )
    if result_text:
      signals.extend(
        extract_from_content_parts(
          result_text[:_MAX_TOOL_RESULT_CHARS],
          source_id_base=self._source_id("agent", "delegation", "result"),
          allow_remote_url=allow_remote,
          include_media=self._media_enabled(),
        )
      )
    if signals:
      self._ingest_async(signals)

  # =========================================================================
  # Seam 2 — PRE-LLM: before every model API call
  # =========================================================================

  def prefetch(self, query: str, *, session_id: str = "") -> str:
    """Pre-LLM seam: return recalled memory for injection before the API call.

    Hermes invokes this before each model call (including tool-loop follow-ups).
    """
    if not self._seam_pre_llm_enabled():
      return ""
    q = (query or "").strip()
    if not q or not self._engine:
      return ""

    with self._prefetch_lock:
      if self._cached_prefetch and self._cached_prefetch_query == q:
        block = self._cached_prefetch
        # Keep cache for identical back-to-back calls in the same loop;
        # still allow a fresh recall if the query changes mid-turn.
        return block

    try:
      # Prefetch is an agent-side retrieval probe, not user content.
      block = self._recall_block(
        q,
        source_id=self._source_id("agent", "prefetch", session_id=session_id or None),
      )
      with self._prefetch_lock:
        self._cached_prefetch = block
        self._cached_prefetch_query = q
      return block
    except Exception as exc:
      logger.warning("Cortext prefetch failed: %s", exc)
      return ""

  def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
    """Pre-warm recall for the next pre-LLM seam (non-blocking)."""
    if not self._seam_pre_llm_enabled():
      return
    q = (query or "").strip()
    if not q or not self._engine:
      return

    def _warm() -> None:
      try:
        block = self._recall_block(
          q,
          source_id=self._source_id(
            "agent", "prefetch", session_id=session_id or None
          ),
        )
        with self._prefetch_lock:
          self._cached_prefetch = block
          self._cached_prefetch_query = q
      except Exception as exc:
        logger.debug("Cortext queue_prefetch failed: %s", exc)

    self._ensure_ingest_worker()
    self._ingest_queue.put(_warm)

  # =========================================================================
  # Seam 3 — ACTION: before each tool executes
  # =========================================================================

  def pre_tool_call(
    self,
    tool_name: str,
    args: dict[str, Any] | None,
    task_id: str = "",
    **kwargs: Any,
  ) -> dict[str, str] | None:
    """Block an unsafe action when Cortext raises a live interrupt signal.

    Hermes calls this hook synchronously, so the proposed action is processed
    with ``Retention.NATURAL``. This lets the engine decide whether the action
    belongs to the current natural memory boundary while returning an immediate
    interrupt decision for the tool loop.
    """
    del kwargs
    if not self._seam_pre_tool_enabled() or not self._should_write():
      return None
    if not self._engine:
      return None
    intent = _tool_intent_text(tool_name, args)
    if not intent:
      return None

    try:
      import augmem.cortext as cortext

      retention = cortext.Retention.NATURAL
      source = self._source_id(
        "agent", "tool_intent", task_id or str(self._turn_number or 0)
      )
      with self._write_gate:
        packet = self._process(intent, source_id=source, retention=retention)
    except Exception as exc:
      logger.warning("Cortext tool gate failed: %s", exc)
      return None

    if not packet.get("should_interrupt"):
      return None
    memories = packet.get("retrieved_memory") or []
    if not isinstance(memories, list):
      return None
    correction = _format_memories(
      [item for item in memories if isinstance(item, dict)],
      limit=min(2, self._top_k()),
    )
    if not correction:
      return None
    # Hermes presents this as a tool error/feedback message. Keep it unbranded
    # so the agent receives only the relevant prior context.
    return {
      "action": "block",
      "message": "Reconsider this action using the following prior context:\n"
      f"{correction}",
    }

  # =========================================================================
  # Seam 4 — POST-LLM: after the model completes a turn
  # =========================================================================

  def sync_turn(
    self,
    user_content: Any,
    assistant_content: Any,
    *,
    session_id: str = "",
    messages: list[dict[str, Any]] | None = None,
  ) -> None:
    """Post-LLM seam: ingest assistant + tool calls/results + media.

    Must be non-blocking. User text already stored at ``on_turn_start`` is
    not double-written (content-hash dedupe). Screenshots, tool images, and
    audio ride the same path via Cortext's tri-modal APIs.
    """
    if not self._seam_post_llm_enabled():
      return
    if not self._should_write() or not self._engine:
      return

    sid = session_id or self._session_id or "session"
    turn = str(self._turn_number) if self._turn_number else "0"
    user_base = self._source_id("user", "turn", turn, session_id=sid)
    agent_base = self._source_id("agent", "turn", turn, session_id=sid)

    signals: list[MediaSignal] = []
    allow_remote = self._fetch_remote_media()
    include_media = self._media_enabled()

    user_signals = extract_from_content_parts(
      user_content,
      source_id_base=user_base,
      allow_remote_url=allow_remote,
      include_media=include_media,
    )
    user_text = next(
      (s.text for s in user_signals if s.modality == "text" and s.text),
      "",
    )
    user_key = self._user_turn_key(sid, self._turn_number, user_text)
    with self._user_key_lock:
      user_already_queued = (
        bool(user_key)
        and (
          user_key == self._last_user_turn_key
          or user_key in self._pending_user_keys
        )
      )
    if user_key and user_already_queued:
      # Keep media from the user payload even when text was already ingested.
      signals.extend(s for s in user_signals if s.modality != "text")
    else:
      signals.extend(user_signals)

    signals.extend(
      extract_from_content_parts(
        assistant_content,
        source_id_base=agent_base,
        allow_remote_url=allow_remote,
        include_media=include_media,
      )
    )

    if messages and _as_bool(self._config.get("ingest_tool_results"), True):
      signals.extend(
        extract_from_messages(
          messages,
          user_source=user_base,
          agent_source=agent_base,
          allow_remote_url=allow_remote,
          tool_events_only=True,
          include_media=include_media,
        )
      )

    if not signals:
      return

    self._ingest_async(signals, mark_user_key=user_key or None)

  # =========================================================================
  # Tools + session lifecycle
  # =========================================================================

  def get_tool_schemas(self) -> list[dict[str, Any]]:
    # Context-only provider: no tools. Hermes injects prefetch automatically.
    return []

  def handle_tool_call(
    self, tool_name: str, args: dict[str, Any], **kwargs: Any
  ) -> str:
    del args, kwargs
    # Should never be called when get_tool_schemas() is empty.
    return json.dumps({"error": f"no tools exposed ({tool_name})"})

  def on_session_end(self, messages: list[dict[str, Any]]) -> None:
    del messages
    if not self._engine:
      return
    try:
      self._join_background(timeout=10.0)
      with self._write_gate:
        if _as_bool(self._config.get("auto_consolidate"), True):
          self._engine.consolidate()
        self._engine.flush()
    except Exception as exc:
      logger.warning("Cortext on_session_end failed: %s", exc)

  def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
    """Safety net: snapshot recent user text/media before context is dropped."""
    if not self._seam_user_enabled() or not self._should_write():
      return ""
    if not self._engine or not messages:
      return ""
    base = self._source_id("user", "pre_compress")
    # Last few user messages only — include screenshots if still in context.
    user_msgs = [
      m for m in messages if isinstance(m, dict) and m.get("role") == "user"
    ][-3:]
    signals: list[MediaSignal] = []
    allow_remote = self._fetch_remote_media()
    include_media = self._media_enabled()
    for msg in user_msgs:
      signals.extend(
        extract_from_message(
          msg,
          role="user",
          source_id_base=base,
          allow_remote_url=allow_remote,
          include_media=include_media,
        )
      )
    if not signals:
      return ""
    self._ingest_async(signals)
    # Return empty so Hermes does not inject a product-named notice.
    return ""

  def on_session_switch(
    self,
    new_session_id: str,
    *,
    parent_session_id: str = "",
    reset: bool = False,
    rewound: bool = False,
    **kwargs: Any,
  ) -> None:
    del parent_session_id, reset, rewound
    self._session_id = new_session_id or self._session_id
    if kwargs.get("user_id") or kwargs.get("user_id_alt"):
      self._user_id = self._resolve_user_id(kwargs)
    if kwargs.get("agent_identity") or kwargs.get("agent_id"):
      self._agent_id = self._resolve_agent_id(kwargs)
    self._last_user_turn_key = ""
    with self._prefetch_lock:
      self._cached_prefetch = ""
      self._cached_prefetch_query = ""

  def shutdown(self) -> None:
    self._join_background(timeout=5.0)
    # Stop the ingest worker after draining pending jobs.
    try:
      self._ingest_queue.put_nowait(None)
    except Exception:
      pass
    worker = self._ingest_worker
    if worker is not None and worker.is_alive():
      worker.join(timeout=2.0)
    self._bg_thread = None
    engine = self._engine
    self._engine = None
    if engine is None:
      return
    try:
      with self._write_gate:
        engine.flush()
        engine.close()
    except Exception as exc:
      logger.debug("Cortext shutdown failed: %s", exc)

  # -- internals -------------------------------------------------------------

  def _resolve_user_id(self, kwargs: dict[str, Any]) -> str:
    for key in ("user_id", "user_id_alt", "peer_name", "peerName"):
      raw = kwargs.get(key)
      if raw:
        return _sanitize_id(str(raw), fallback=DEFAULT_USER_ID)
    cfg_user = self._config.get("user_id")
    if cfg_user:
      return _sanitize_id(str(cfg_user), fallback=DEFAULT_USER_ID)
    return DEFAULT_USER_ID

  def _resolve_agent_id(self, kwargs: dict[str, Any]) -> str:
    for key in ("agent_identity", "agent_id", "ai_peer", "aiPeer"):
      raw = kwargs.get(key)
      if raw:
        return _sanitize_id(str(raw), fallback=DEFAULT_AGENT_ID)
    cfg_agent = self._config.get("agent_id")
    if cfg_agent:
      return _sanitize_id(str(cfg_agent), fallback=DEFAULT_AGENT_ID)
    # Profile-ish fallback from HERMES_HOME leaf when present.
    if self._hermes_home:
      leaf = Path(self._hermes_home).name
      if leaf and leaf not in {".hermes", "hermes"}:
        return _sanitize_id(leaf, fallback=DEFAULT_AGENT_ID)
    return DEFAULT_AGENT_ID

  @staticmethod
  def _user_turn_key(session_id: str, turn_number: int, text: str) -> str:
    """Return a dedupe key only when the lifecycle supplied a turn id."""
    if turn_number <= 0 or not text:
      return ""
    return f"{session_id}:{turn_number}:{_content_key(text)}"

  def _source_id(
    self,
    role: str,
    *parts: str,
    session_id: str | None = None,
  ) -> str:
    """Build provenance: ``hermes/{user|agent}/{actor}/{session}/...``.

    Role is always ``user`` or ``agent`` so memories have clear speaker
    provenance. Actor is the Hermes user id or agent identity when known.
    """
    if role not in {"user", "agent"}:
      raise ValueError(f"role must be 'user' or 'agent', got {role!r}")
    actor = self._user_id if role == "user" else self._agent_id
    actor = _sanitize_id(actor, fallback=role)
    sid = _sanitize_id(
      session_id or self._session_id or "session",
      fallback="session",
    )
    segs = ["hermes", role, actor, sid]
    for part in parts:
      p = _sanitize_id(str(part), fallback="")
      if p:
        segs.append(p)
    return "/".join(segs)

  def _seam_user_enabled(self) -> bool:
    return _as_bool(self._config.get("seam_user"), True)

  def _seam_pre_llm_enabled(self) -> bool:
    return _as_bool(self._config.get("seam_pre_llm"), True)

  def _seam_pre_tool_enabled(self) -> bool:
    return _as_bool(self._config.get("seam_pre_tool"), True)

  def _seam_post_llm_enabled(self) -> bool:
    # Prefer seam_post_llm; fall back to auto_sync_turns for older configs.
    if "seam_post_llm" in self._config:
      return _as_bool(self._config.get("seam_post_llm"), True)
    return _as_bool(self._config.get("auto_sync_turns"), True)

  def _should_write(self) -> bool:
    # Skip cron/subagent/flush contexts so system prompts do not pollute memory.
    return self._agent_context in {"", "primary"}

  def _top_k(self) -> int:
    try:
      return max(1, int(self._config.get("top_k", DEFAULT_TOP_K)))
    except (TypeError, ValueError):
      return DEFAULT_TOP_K

  def _join_background(self, *, timeout: float) -> None:
    """Wait until queued ingest work that is ahead of this call drains."""
    if self._ingest_worker is None and self._ingest_queue.empty():
      return
    done = threading.Event()

    def _mark_done() -> None:
      done.set()

    self._ensure_ingest_worker()
    self._ingest_queue.put(_mark_done)
    done.wait(timeout=max(0.0, timeout))
    self._bg_thread = self._ingest_worker

  def _media_enabled(self) -> bool:
    return _as_bool(self._config.get("ingest_media"), True)

  def _fetch_remote_media(self) -> bool:
    return _as_bool(self._config.get("fetch_remote_media"), False)

  def _ensure_ingest_worker(self) -> None:
    with self._ingest_worker_lock:
      if self._ingest_worker is not None and self._ingest_worker.is_alive():
        self._bg_thread = self._ingest_worker
        return

      def _worker() -> None:
        while True:
          job = self._ingest_queue.get()
          if job is None:
            self._ingest_queue.task_done()
            with self._ingest_worker_lock:
              self._ingest_worker = None
              self._bg_thread = None
            return
          try:
            job()
          except Exception as exc:
            logger.warning("Cortext ingest worker job failed: %s", exc)
          finally:
            self._ingest_queue.task_done()

      self._ingest_worker = threading.Thread(
        target=_worker, name="cortext-ingest", daemon=True
      )
      self._ingest_worker.start()
      self._bg_thread = self._ingest_worker

  def _signals_from_user_payload(
    self, message: Any, *, source_id_base: str, **kwargs: Any
  ) -> list[MediaSignal]:
    """Build multimodal signals from on_turn_start message + kwargs."""
    allow_remote = self._fetch_remote_media()
    include_media = self._media_enabled()
    signals: list[MediaSignal] = []
    if isinstance(message, dict):
      signals.extend(
        extract_from_message(
          message,
          role="user",
          source_id_base=source_id_base,
          allow_remote_url=allow_remote,
          include_media=include_media,
        )
      )
    else:
      signals.extend(
        extract_from_content_parts(
          message,
          source_id_base=source_id_base,
          allow_remote_url=allow_remote,
          include_media=include_media,
        )
      )

    if not include_media:
      return signals

    # Hermes may pass images/attachments out-of-band on kwargs.
    for key in ("images", "attachments", "files", "media", "screenshots"):
      blob = kwargs.get(key)
      if isinstance(blob, list):
        for i, item in enumerate(blob):
          ref = item if isinstance(item, str) else None
          if isinstance(item, dict):
            ref = str(
              item.get("url")
              or item.get("path")
              or item.get("file_path")
              or item.get("image_url")
              or ""
            )
          if ref:
            sig = media_signal_from_ref(
              ref,
              source_id=f"{source_id_base}/{key}/{i}",
              allow_remote_url=allow_remote,
            )
            if sig:
              signals.append(sig)
      elif isinstance(blob, str) and blob.strip():
        sig = media_signal_from_ref(
          blob,
          source_id=f"{source_id_base}/{key}",
          allow_remote_url=allow_remote,
        )
        if sig:
          signals.append(sig)

    return signals

  def _ingest_async(
    self,
    items: list[MediaSignal],
    *,
    mark_user_key: str | None = None,
    also_warm_prefetch: str | None = None,
  ) -> None:
    """Non-blocking multimodal write path (text / image / audio).

    Enqueues work on a single background worker so callers never join a
    previous ingest thread (tight tool-loops must stay unblocked).
    """
    if not items or not self._engine:
      return
    if not self._should_write():
      return

    cleaned = [s for s in items if not s.is_empty()]
    if not self._media_enabled():
      cleaned = [s for s in cleaned if s.modality == "text"]
    if not cleaned:
      return

    if mark_user_key:
      with self._user_key_lock:
        self._pending_user_keys.add(mark_user_key)

    def _work() -> None:
      try:
        with self._write_gate:
          if self._engine is None:
            return
          try:
            import augmem.cortext as cortext

            store_retention = cortext.Retention.DURABLE
          except Exception:
            store_retention = None
          all_succeeded = True
          for signal in cleaned:
            try:
              process_signal_on_engine(
                self._engine, signal, retention=store_retention
              )
            except Exception as exc:
              all_succeeded = False
              logger.warning(
                "Cortext %s ingest failed (%s): %s",
                signal.modality,
                signal.source_id,
                exc,
              )
          self._engine.flush()
          # A completed write can change the answer for an identical query.
          # Drop the cache before a user-seam warmup optionally replaces it.
          with self._prefetch_lock:
            self._cached_prefetch = ""
            self._cached_prefetch_query = ""
          if mark_user_key and all_succeeded:
            with self._user_key_lock:
              self._last_user_turn_key = mark_user_key
        if also_warm_prefetch and self._seam_pre_llm_enabled():
          try:
            block = self._recall_block(
              also_warm_prefetch,
              source_id=self._source_id("agent", "prefetch"),
            )
            with self._prefetch_lock:
              self._cached_prefetch = block
              self._cached_prefetch_query = also_warm_prefetch
          except Exception as exc:
            logger.debug("Cortext user-seam prefetch warm failed: %s", exc)
      except Exception as exc:
        logger.warning("Cortext ingest failed: %s", exc)
      finally:
        if mark_user_key:
          with self._user_key_lock:
            self._pending_user_keys.discard(mark_user_key)

    self._ensure_ingest_worker()
    self._ingest_queue.put(_work)

  def _process(
    self,
    text: str,
    *,
    source_id: str,
    retention: Any = None,
  ) -> dict[str, Any]:
    if self._engine is None:
      raise RuntimeError("Cortext engine is not initialized")
    try:
      import augmem.cortext as cortext

      if retention is None:
        retention = cortext.Retention.DURABLE
    except Exception:
      pass
    return self._engine.process_text(
      text,
      source_id=source_id,
      include_embedding=False,
      retention=retention,
    )

  def _recall_block(self, query: str, *, source_id: str) -> str:
    # Prefetch/search use Ephemeral so probes do not pollute durable memory.
    try:
      import augmem.cortext as cortext

      retention = cortext.Retention.EPHEMERAL
    except Exception:
      retention = None
    with self._write_gate:
      ctx = self._process(query, source_id=source_id, retention=retention)
    items = ctx.get("retrieved_memory") or []
    if not isinstance(items, list) or not items:
      return ""
    body = _format_memories(
      [item for item in items if isinstance(item, dict)],
      limit=self._top_k(),
    )
    if not body:
      return ""
    # No product name — the model should just see prior facts, not a backend.
    return body


def register(ctx: Any) -> None:
  """Hermes plugin entry point."""
  config = load_config(os.environ.get("HERMES_HOME"))
  provider = CortextMemoryProvider(config=config)
  ctx.register_memory_provider(provider)
  ctx.register_hook("pre_tool_call", provider.pre_tool_call)
