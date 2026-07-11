from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any
import wave

import pytest

from augmem.hermes.provider import (
  PROVIDER_NAME,
  CortextMemoryProvider,
  load_config,
  resolve_db_path,
)


class _FakeEngine:
  def __init__(self) -> None:
    self.calls: list[tuple[str, str]] = []
    self.modalities: list[str] = []
    self.flushed = 0
    self.closed = 0
    self.consolidated = 0
    self._responses: list[dict[str, Any]] = []

  def push(self, ctx: dict[str, Any]) -> None:
    self._responses.append(ctx)

  def _next(self, label: str, source_id: str) -> dict[str, Any]:
    self.calls.append((label, source_id))
    if self._responses:
      return self._responses.pop(0)
    return {"retrieved_memory": [], "consolidation_recommended": False}

  def process_text(
    self,
    text: str,
    source_id: str,
    include_embedding: bool = True,
    retention: Any = None,
  ) -> dict[str, Any]:
    del include_embedding, retention
    self.modalities.append("text")
    return self._next(text, source_id)

  def process_image(
    self,
    data: bytes,
    width: int,
    height: int,
    channels: int,
    source_id: str,
    include_embedding: bool = True,
  ) -> dict[str, Any]:
    del data, width, height, channels, include_embedding
    self.modalities.append("image")
    return self._next("<image>", source_id)

  def process_image_with_media(
    self,
    data: bytes,
    width: int,
    height: int,
    channels: int,
    source_id: str,
    media: Any = None,
    include_embedding: bool = True,
  ) -> dict[str, Any]:
    del data, width, height, channels, media, include_embedding
    self.modalities.append("image")
    return self._next("<image+media>", source_id)

  def process_audio(
    self, pcm: Any, source_id: str, include_embedding: bool = True
  ) -> dict[str, Any]:
    del pcm, include_embedding
    self.modalities.append("audio")
    return self._next("<audio>", source_id)

  def process_audio_with_media(
    self,
    pcm: Any,
    source_id: str,
    media: Any = None,
    include_embedding: bool = True,
  ) -> dict[str, Any]:
    del pcm, media, include_embedding
    self.modalities.append("audio")
    return self._next("<audio+media>", source_id)

  def flush(self) -> None:
    self.flushed += 1

  def close(self) -> None:
    self.closed += 1

  def consolidate(self) -> dict[str, Any]:
    self.consolidated += 1
    return {"ok": True}


def _wait_bg(provider: CortextMemoryProvider) -> None:
  provider._join_background(timeout=2.0)


def test_invisible_to_agent_no_tools_no_system_prompt() -> None:
  provider = CortextMemoryProvider()
  assert provider.name == PROVIDER_NAME
  assert provider.get_tool_schemas() == []
  assert provider.system_prompt_block() == ""
  assert "cortext" not in provider.system_prompt_block().lower()


def test_load_and_save_config(tmp_path: Path) -> None:
  home = tmp_path / "hermes"
  home.mkdir()
  provider = CortextMemoryProvider()
  provider.save_config(
    {
      "db_path": "$HERMES_HOME/custom.sqlite",
      "focus": "0.7",
      "sensitivity": "0.4",
      "stability": "0.8",
    },
    str(home),
  )
  cfg_path = home / "cortext.json"
  assert cfg_path.is_file()
  loaded = load_config(home)
  assert loaded["db_path"] == "$HERMES_HOME/custom.sqlite"
  assert loaded["focus"] == pytest.approx(0.7)
  assert loaded["sensitivity"] == pytest.approx(0.4)
  assert loaded["stability"] == pytest.approx(0.8)


def test_resolve_db_path(tmp_path: Path) -> None:
  home = str(tmp_path)
  assert resolve_db_path("$HERMES_HOME/a.sqlite", home) == tmp_path / "a.sqlite"
  assert resolve_db_path("relative.sqlite", home) == tmp_path / "relative.sqlite"


def test_user_seam_on_turn_start() -> None:
  provider = CortextMemoryProvider()
  engine = _FakeEngine()
  provider._engine = engine
  provider._agent_context = "primary"
  provider._session_id = "s1"
  provider._user_id = "alice"
  provider._agent_id = "hermes"

  provider.on_turn_start(1, "Bailey needs Benadryl for bee stings")
  _wait_bg(provider)

  assert any("Bailey" in text for text, _ in engine.calls)
  assert any(
    source == "hermes/user/alice/s1/turn/1" for _, source in engine.calls
  )


def test_post_llm_skips_duplicate_user() -> None:
  provider = CortextMemoryProvider()
  engine = _FakeEngine()
  provider._engine = engine
  provider._agent_context = "primary"
  provider._session_id = "s1"
  provider._user_id = "alice"
  provider._agent_id = "hermes"

  user = "Remember the vet is July 14"
  provider.on_turn_start(2, user)
  _wait_bg(provider)
  n_after_user = len(engine.calls)

  provider.sync_turn(user, "Got it — July 14.")
  _wait_bg(provider)

  new_calls = engine.calls[n_after_user:]
  assert any(
    "July 14" in text and source == "hermes/agent/hermes/s1/turn/2"
    for text, source in new_calls
  )
  assert not any("/user/" in source and "/turn/" in source for _, source in new_calls)


def test_post_llm_skips_user_ingest_queued_by_turn_start() -> None:
  provider = CortextMemoryProvider()
  engine = _FakeEngine()
  provider._engine = engine
  provider._agent_context = "primary"
  provider._session_id = "s1"
  provider._user_id = "alice"
  provider._agent_id = "hermes"

  user = "remember this once"
  provider.on_turn_start(2, user)
  provider.sync_turn(user, "acknowledged")
  _wait_bg(provider)

  durable_turn_calls = [
    text for text, source in engine.calls if source.endswith("/turn/2")
  ]
  assert durable_turn_calls.count(user) == 1
  assert durable_turn_calls.count("acknowledged") == 1


def test_source_id_user_vs_agent_provenance() -> None:
  provider = CortextMemoryProvider()
  provider._session_id = "sess-9"
  provider._user_id = "gabe"
  provider._agent_id = "coder"

  assert provider._source_id("user", "turn", "3") == "hermes/user/gabe/sess-9/turn/3"
  assert provider._source_id("agent", "turn", "3") == "hermes/agent/coder/sess-9/turn/3"
  assert provider._source_id("agent", "prefetch") == "hermes/agent/coder/sess-9/prefetch"
  assert (
    provider._source_id("user", "builtin", "USER")
    == "hermes/user/gabe/sess-9/builtin/USER"
  )


def test_prefetch_is_unbranded_prior_context() -> None:
  provider = CortextMemoryProvider()
  engine = _FakeEngine()
  engine.push({"retrieved_memory": [{"text": "vet July 14", "rel": 0.9}]})
  provider._engine = engine
  block = provider.prefetch("when is the vet?")
  assert "vet July 14" in block
  assert "cortext" not in block.lower()
  assert "##" not in block  # no product/section branding


def test_memory_text_decodes_base64_content_blobs() -> None:
  import base64

  from augmem.hermes.provider import _memory_text

  payload = "Bailey needs Benadryl for bee stings"
  item = {
    "content": [
      {
        "base64": base64.b64encode(payload.encode("utf-8")).decode("ascii"),
        "size_bytes": len(payload),
      }
    ],
    "source_id": "hermes/user/u/s/turn/1",
    "relevance": 0.9,
  }
  assert _memory_text(item) == payload


def test_memory_text_skips_non_text_payloads() -> None:
  import base64

  from augmem.hermes.provider import _memory_text

  item = {
    "modality": "image",
    "content": [{"base64": base64.b64encode(b"\x89PNG").decode("ascii")}],
  }
  assert _memory_text(item) == ""


def test_skips_writes_for_non_primary_context() -> None:
  provider = CortextMemoryProvider()
  engine = _FakeEngine()
  provider._engine = engine
  provider._agent_context = "cron"

  provider.on_turn_start(1, "should not store")
  _wait_bg(provider)
  assert engine.calls == []


def test_shutdown_closes_engine() -> None:
  provider = CortextMemoryProvider()
  engine = _FakeEngine()
  provider._engine = engine
  provider.shutdown()
  assert engine.closed == 1
  assert provider._engine is None


def test_register_entry_point() -> None:
  from augmem.hermes.provider import register

  registered: list[Any] = []

  class _Ctx:
    def register_memory_provider(self, provider: Any) -> None:
      registered.append(provider)

  register(_Ctx())
  assert len(registered) == 1
  assert registered[0].name == "cortext"
  assert registered[0].get_tool_schemas() == []
  assert registered[0].system_prompt_block() == ""


def test_sync_turn_ingests_tool_calls_and_results() -> None:
  provider = CortextMemoryProvider()
  engine = _FakeEngine()
  provider._engine = engine
  provider._agent_context = "primary"
  provider._session_id = "s1"
  provider._user_id = "alice"
  provider._agent_id = "hermes"
  provider._turn_number = 4

  messages = [
    {
      "role": "assistant",
      "content": "I'll take a screenshot.",
      "tool_calls": [
        {
          "id": "1",
          "type": "function",
          "function": {
            "name": "computer_screenshot",
            "arguments": '{"region": "full"}',
          },
        }
      ],
    },
    {
      "role": "tool",
      "name": "computer_screenshot",
      "content": "Saved screenshot to /tmp/not-a-real-file.png",
    },
  ]
  provider.sync_turn(
    "capture the UI",
    "I'll take a screenshot.",
    messages=messages,
  )
  _wait_bg(provider)

  joined = " ".join(text for text, _ in engine.calls)
  assert "capture the UI" in joined or any(
    "computer_screenshot" in text for text, _ in engine.calls
  )
  assert any("tool_call" in source or "tool/" in source for _, source in engine.calls)


def test_sync_turn_does_not_reingest_regular_transcript_messages() -> None:
  provider = CortextMemoryProvider()
  engine = _FakeEngine()
  provider._engine = engine
  provider._agent_context = "primary"
  provider._session_id = "s1"
  provider._user_id = "alice"
  provider._agent_id = "hermes"
  provider._turn_number = 4

  provider.sync_turn(
    "current user message",
    "current assistant response",
    messages=[
      {"role": "user", "content": "older transcript question"},
      {"role": "assistant", "content": "older transcript response"},
      {"role": "user", "content": "current user message"},
      {"role": "assistant", "content": "current assistant response"},
    ],
  )
  _wait_bg(provider)

  ingested = [text for text, _ in engine.calls]
  assert ingested.count("current user message") == 1
  assert ingested.count("current assistant response") == 1
  assert "older transcript question" not in ingested
  assert "older transcript response" not in ingested


def test_multimodal_user_content_parts() -> None:
  from augmem.hermes.media import extract_from_content_parts

  parts = [
    {"type": "text", "text": "what is this UI?"},
    {
      "type": "image_url",
      "image_url": {
        "url": (
          "data:image/png;base64,"
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
      },
    },
  ]
  signals = extract_from_content_parts(parts, source_id_base="hermes/user/u/s/turn/1")
  modalities = {s.modality for s in signals}
  assert "text" in modalities
  try:
    import PIL  # noqa: F401

    assert "image" in modalities
  except ImportError:
    pass


def test_openai_input_audio_content_part_is_decoded() -> None:
  from augmem.hermes.media import extract_from_content_parts

  wav = BytesIO()
  with wave.open(wav, "wb") as output:
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(16000)
    output.writeframes(b"\x00\x00" * 160)

  signals = extract_from_content_parts(
    [
      {
        "type": "input_audio",
        "input_audio": {
          "data": base64.b64encode(wav.getvalue()).decode("ascii"),
          "format": "wav",
        },
      }
    ],
    source_id_base="hermes/user/u/s/turn/1",
  )

  assert len(signals) == 1
  assert signals[0].modality == "audio"
  assert signals[0].pcm


def test_text_tool_result_scans_embedded_media_path(tmp_path: Path) -> None:
  from augmem.hermes.media import extract_from_content_parts

  wav_path = tmp_path / "result.wav"
  with wave.open(str(wav_path), "wb") as output:
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(16000)
    output.writeframes(b"\x00\x00" * 160)

  signals = extract_from_content_parts(
    f"Saved audio to {wav_path}",
    source_id_base="hermes/agent/a/s/turn/1/tool/audio",
  )
  assert {signal.modality for signal in signals} == {"text", "audio"}


def test_media_disabled_skips_content_media_decode(monkeypatch: pytest.MonkeyPatch) -> None:
  import augmem.hermes.media as media

  called = False

  def _unexpected_media_load(*args: Any, **kwargs: Any) -> Any:
    nonlocal called
    del args, kwargs
    called = True
    return None

  monkeypatch.setattr(media, "media_signal_from_ref", _unexpected_media_load)
  signals = media.extract_from_content_parts(
    "data:image/png;base64,AAAA",
    source_id_base="hermes/user/u/s/turn/1",
    include_media=False,
  )

  assert signals == []
  assert not called
