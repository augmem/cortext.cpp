"""Multimodal signal extraction for Hermes → Cortext.

Cortext is tri-modal (text / image / audio). Hermes traffic arrives as
OpenAI-style message parts, data URLs, file paths, tool results, and
clipboard screenshots. This module normalizes those into processable
signals without blocking the agent loop.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

logger = logging.getLogger(__name__)

Modality = Literal["text", "image", "audio"]

_DATA_URL_RE = re.compile(
  r"^data:(?P<mime>[\w.+/-]+)?(?:;charset=[\w-]+)?;base64,(?P<data>.+)$",
  re.IGNORECASE | re.DOTALL,
)
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_AUDIO_EXTS = {".wav", ".wave"}
_MAX_TEXT_CHARS = 4000
_MAX_TOOL_JSON_CHARS = 2500
_MAX_IMAGE_SIDE = 2048  # downscale long edge for speed; Cortext stays responsive
_MAX_FILE_BYTES = 25 * 1024 * 1024


@dataclass(slots=True)
class MediaSignal:
  """One Cortext-bound signal with provenance."""

  modality: Modality
  source_id: str
  text: str = ""
  # Image: row-major RGB/RGBA + optional original container bytes
  image_rgb: bytes = b""
  width: int = 0
  height: int = 0
  channels: int = 0
  media_bytes: bytes = b""
  media_mimetype: str = ""
  # Audio: 16 kHz mono float PCM
  pcm: list[float] = field(default_factory=list)

  def is_empty(self) -> bool:
    if self.modality == "text":
      return not self.text.strip()
    if self.modality == "image":
      return not self.image_rgb or self.width <= 0 or self.height <= 0
    if self.modality == "audio":
      return not self.pcm
    return True


def _clip(text: str, limit: int = _MAX_TEXT_CHARS) -> str:
  text = (text or "").strip()
  if len(text) <= limit:
    return text
  return text[: limit - 1] + "…"


def decode_data_url(url: str) -> tuple[bytes, str] | None:
  m = _DATA_URL_RE.match((url or "").strip())
  if not m:
    return None
  mime = (m.group("mime") or "application/octet-stream").lower()
  try:
    raw = base64.b64decode(m.group("data"), validate=False)
  except Exception:
    return None
  if not raw or len(raw) > _MAX_FILE_BYTES:
    return None
  return raw, mime


def _load_local_or_url_bytes(ref: str) -> tuple[bytes, str] | None:
  ref = (ref or "").strip()
  if not ref:
    return None
  data_url = decode_data_url(ref)
  if data_url:
    return data_url

  # file:// or bare path
  path_str = ref
  if ref.startswith("file://"):
    path_str = unquote(urlparse(ref).path)
  path = Path(path_str).expanduser()
  if path.is_file():
    try:
      raw = path.read_bytes()
    except OSError as exc:
      logger.debug("Failed to read media file %s: %s", path, exc)
      return None
    if not raw or len(raw) > _MAX_FILE_BYTES:
      return None
    mime = _guess_mime(path.suffix, raw)
    return raw, mime

  if ref.startswith("http://") or ref.startswith("https://"):
    try:
      with urlopen(ref, timeout=5.0) as resp:  # noqa: S310 - intentional fetch
        raw = resp.read(_MAX_FILE_BYTES + 1)
        if len(raw) > _MAX_FILE_BYTES:
          return None
        mime = (resp.headers.get_content_type() or "").lower()
        if not mime or mime == "application/octet-stream":
          mime = _guess_mime(Path(urlparse(ref).path).suffix, raw)
        return raw, mime
    except Exception as exc:
      logger.debug("Failed to fetch media URL: %s", exc)
      return None
  return None


def _guess_mime(suffix: str, raw: bytes) -> str:
  ext = (suffix or "").lower()
  if ext in {".png"} or raw[:8] == b"\x89PNG\r\n\x1a\n":
    return "image/png"
  if ext in {".jpg", ".jpeg"} or raw[:3] == b"\xff\xd8\xff":
    return "image/jpeg"
  if ext == ".webp" or raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
    return "image/webp"
  if ext == ".gif" or raw[:6] in {b"GIF87a", b"GIF89a"}:
    return "image/gif"
  if ext in {".wav", ".wave"} or raw[:4] == b"RIFF":
    return "audio/wav"
  if ext == ".bmp":
    return "image/bmp"
  return "application/octet-stream"


def decode_image_to_rgb(
  raw: bytes, mimetype: str = ""
) -> tuple[bytes, int, int, int] | None:
  """Decode container bytes → row-major RGB/RGBA for Cortext process_image."""
  try:
    from io import BytesIO

    from PIL import Image
  except ImportError:
    logger.debug("Pillow not installed; image ingest disabled (pip install pillow)")
    return None

  try:
    img = Image.open(BytesIO(raw))
    img.load()
  except Exception as exc:
    logger.debug("Image decode failed (%s): %s", mimetype, exc)
    return None

  # Bound size for low latency while keeping screenshots usable.
  w, h = img.size
  scale = min(1.0, _MAX_IMAGE_SIDE / max(w, h, 1))
  if scale < 1.0:
    img = img.resize(
      (max(1, int(w * scale)), max(1, int(h * scale))),
      Image.Resampling.BILINEAR,
    )

  if img.mode not in {"RGB", "RGBA"}:
    img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
  channels = 4 if img.mode == "RGBA" else 3
  width, height = img.size
  return bytes(img.tobytes()), width, height, channels


def decode_wav_to_pcm16k(raw: bytes) -> list[float] | None:
  """Decode WAV bytes to 16 kHz mono float PCM for Cortext process_audio."""
  from io import BytesIO

  try:
    with wave.open(BytesIO(raw), "rb") as wf:
      nch = wf.getnchannels()
      sw = wf.getsampwidth()
      rate = wf.getframerate()
      nframes = wf.getnframes()
      frames = wf.readframes(nframes)
  except Exception as exc:
    logger.debug("WAV decode failed: %s", exc)
    return None

  if sw == 2:
    samples = array("h")
    samples.frombytes(frames)
    floats = [s / 32768.0 for s in samples]
  elif sw == 1:
    floats = [(b - 128) / 128.0 for b in frames]
  elif sw == 4:
    samples = array("i")
    samples.frombytes(frames)
    floats = [s / 2147483648.0 for s in samples]
  else:
    return None

  if nch > 1:
    mono: list[float] = []
    for i in range(0, len(floats), nch):
      chunk = floats[i : i + nch]
      mono.append(sum(chunk) / len(chunk))
    floats = mono

  if rate != 16000 and rate > 0:
    # Linear resample — good enough for memory embedding, very fast.
    target_len = max(1, int(len(floats) * 16000 / rate))
    if target_len == 1:
      floats = floats[:1]
    else:
      resampled: list[float] = []
      for i in range(target_len):
        src = i * (len(floats) - 1) / (target_len - 1)
        j = int(src)
        t = src - j
        a = floats[j]
        b = floats[min(j + 1, len(floats) - 1)]
        resampled.append(a + (b - a) * t)
      floats = resampled
  return floats


def image_signal_from_bytes(
  raw: bytes,
  *,
  source_id: str,
  mimetype: str = "",
) -> MediaSignal | None:
  decoded = decode_image_to_rgb(raw, mimetype)
  if not decoded:
    return None
  rgb, w, h, c = decoded
  mime = mimetype or _guess_mime("", raw)
  return MediaSignal(
    modality="image",
    source_id=source_id,
    image_rgb=rgb,
    width=w,
    height=h,
    channels=c,
    media_bytes=raw,
    media_mimetype=mime if mime.startswith("image/") else "image/png",
  )


def audio_signal_from_bytes(
  raw: bytes,
  *,
  source_id: str,
  mimetype: str = "",
) -> MediaSignal | None:
  pcm = decode_wav_to_pcm16k(raw)
  if not pcm:
    return None
  mime = mimetype or "audio/wav"
  return MediaSignal(
    modality="audio",
    source_id=source_id,
    pcm=pcm,
    media_bytes=raw,
    media_mimetype=mime,
  )


def media_signal_from_ref(ref: str, *, source_id: str) -> MediaSignal | None:
  loaded = _load_local_or_url_bytes(ref)
  if not loaded:
    return None
  raw, mime = loaded
  if mime.startswith("image/") or Path(ref).suffix.lower() in _IMAGE_EXTS:
    return image_signal_from_bytes(raw, source_id=source_id, mimetype=mime)
  if mime.startswith("audio/") or Path(ref).suffix.lower() in _AUDIO_EXTS:
    return audio_signal_from_bytes(raw, source_id=source_id, mimetype=mime)
  # Unknown binary: skip (don't pollute memory with opaque blobs)
  return None


def _text_signal(text: str, source_id: str) -> MediaSignal | None:
  clipped = _clip(text)
  if not clipped:
    return None
  return MediaSignal(modality="text", source_id=source_id, text=clipped)


def extract_from_content_parts(
  content: Any,
  *,
  source_id_base: str,
) -> list[MediaSignal]:
  """Parse OpenAI/Hermes content (string or multimodal part list)."""
  signals: list[MediaSignal] = []
  if content is None:
    return signals

  if isinstance(content, str):
    # Prefer media if the whole string is a data URL or media path.
    if content.startswith("data:") or _looks_like_media_path(content):
      sig = media_signal_from_ref(content, source_id=f"{source_id_base}/media")
      if sig:
        signals.append(sig)
        return signals
    t = _text_signal(content, source_id_base)
    if t:
      signals.append(t)
    return signals

  if isinstance(content, dict):
    return extract_from_content_parts([content], source_id_base=source_id_base)

  if not isinstance(content, list):
    return signals

  text_bits: list[str] = []
  media_i = 0
  for part in content:
    if isinstance(part, str):
      text_bits.append(part)
      continue
    if not isinstance(part, dict):
      continue
    ptype = str(part.get("type") or "").lower()

    if ptype in {"text", "input_text"} or "text" in part and ptype == "":
      tx = part.get("text") or part.get("content") or ""
      if isinstance(tx, str) and tx.strip():
        text_bits.append(tx)
      continue

    if ptype in {"image_url", "input_image", "image"}:
      url = ""
      image_url = part.get("image_url")
      if isinstance(image_url, dict):
        url = str(image_url.get("url") or "")
      elif isinstance(image_url, str):
        url = image_url
      url = url or str(part.get("url") or part.get("image") or "")
      media_i += 1
      sig = media_signal_from_ref(
        url, source_id=f"{source_id_base}/image/{media_i}"
      )
      if sig:
        signals.append(sig)
      continue

    if ptype in {"input_audio", "audio", "audio_url"}:
      url = ""
      audio_url = part.get("audio_url") or part.get("input_audio") or part
      if isinstance(audio_url, dict):
        url = str(
          audio_url.get("url")
          or audio_url.get("data")
          or ""
        )
        # OpenAI input_audio: {data: b64, format: wav}
        if not url and audio_url.get("data"):
          fmt = str(audio_url.get("format") or "wav")
          b64 = str(audio_url.get("data"))
          try:
            raw = base64.b64decode(b64, validate=False)
          except Exception:
            raw = b""
          if raw:
            media_i += 1
            sig = audio_signal_from_bytes(
              raw,
              source_id=f"{source_id_base}/audio/{media_i}",
              mimetype=f"audio/{fmt}",
            )
            if sig:
              signals.append(sig)
            continue
      else:
        url = str(audio_url or "")
      if url:
        media_i += 1
        sig = media_signal_from_ref(
          url if not url.startswith("data:") and "base64" not in url[:40]
          else (url if url.startswith("data:") else f"data:audio/wav;base64,{url}"),
          source_id=f"{source_id_base}/audio/{media_i}",
        )
        if sig:
          signals.append(sig)
      continue

    # Generic file / screenshot path fields Hermes sometimes uses
    for key in ("file_path", "path", "screenshot", "image_path", "url"):
      val = part.get(key)
      if isinstance(val, str) and _looks_like_media_path(val):
        media_i += 1
        sig = media_signal_from_ref(
          val, source_id=f"{source_id_base}/file/{media_i}"
        )
        if sig:
          signals.append(sig)

  if text_bits:
    t = _text_signal("\n".join(text_bits), source_id_base)
    if t:
      signals.append(t)
  return signals


def _looks_like_media_path(value: str) -> bool:
  v = (value or "").strip()
  if not v or len(v) > 4096:
    return False
  if v.startswith("data:image") or v.startswith("data:audio"):
    return True
  lower = v.lower()
  if any(lower.endswith(ext) for ext in _IMAGE_EXTS | _AUDIO_EXTS):
    return True
  if v.startswith("file://"):
    return True
  p = Path(v).expanduser()
  return p.is_file() and p.suffix.lower() in _IMAGE_EXTS | _AUDIO_EXTS


def extract_tool_calls(
  message: dict[str, Any],
  *,
  source_id_base: str,
) -> list[MediaSignal]:
  """Ingest assistant tool_calls as compact text signals (name + args)."""
  signals: list[MediaSignal] = []
  tool_calls = message.get("tool_calls")
  if not isinstance(tool_calls, list):
    # Anthropic-style content blocks
    content = message.get("content")
    if isinstance(content, list):
      for i, part in enumerate(content):
        if not isinstance(part, dict):
          continue
        if str(part.get("type") or "") not in {"tool_use", "function_call"}:
          continue
        name = str(part.get("name") or "tool")
        args = part.get("input") or part.get("arguments") or {}
        payload = {
          "tool": name,
          "arguments": args if not isinstance(args, str) else _safe_json(args),
        }
        text = _clip(json.dumps(payload, ensure_ascii=False), _MAX_TOOL_JSON_CHARS)
        t = _text_signal(text, f"{source_id_base}/tool_call/{i}/{name}")
        if t:
          signals.append(t)
        # Tool args may embed image paths / data URLs
        signals.extend(
          _scan_for_media_refs(args, source_id_base=f"{source_id_base}/tool_call/{i}")
        )
    return signals

  for i, call in enumerate(tool_calls):
    if not isinstance(call, dict):
      continue
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = str(
      (fn or {}).get("name") or call.get("name") or "tool"
    )
    raw_args = (fn or {}).get("arguments") or call.get("arguments") or {}
    args_obj = _safe_json(raw_args) if isinstance(raw_args, str) else raw_args
    payload = {"tool": name, "arguments": args_obj}
    text = _clip(json.dumps(payload, ensure_ascii=False), _MAX_TOOL_JSON_CHARS)
    t = _text_signal(text, f"{source_id_base}/tool_call/{i}/{name}")
    if t:
      signals.append(t)
    signals.extend(
      _scan_for_media_refs(args_obj, source_id_base=f"{source_id_base}/tool_call/{i}")
    )
  return signals


def _safe_json(raw: str) -> Any:
  try:
    return json.loads(raw)
  except Exception:
    return raw


def _scan_for_media_refs(obj: Any, *, source_id_base: str) -> list[MediaSignal]:
  signals: list[MediaSignal] = []
  seen: set[str] = set()

  def walk(node: Any, depth: int = 0) -> None:
    if depth > 6 or len(signals) >= 8:
      return
    if isinstance(node, str):
      if node in seen:
        return
      if node.startswith("data:image") or node.startswith("data:audio"):
        seen.add(node[:64])
        sig = media_signal_from_ref(
          node, source_id=f"{source_id_base}/media/{len(signals)+1}"
        )
        if sig:
          signals.append(sig)
      elif _looks_like_media_path(node):
        seen.add(node)
        sig = media_signal_from_ref(
          node, source_id=f"{source_id_base}/media/{len(signals)+1}"
        )
        if sig:
          signals.append(sig)
      return
    if isinstance(node, dict):
      for v in node.values():
        walk(v, depth + 1)
      return
    if isinstance(node, list):
      for v in node:
        walk(v, depth + 1)

  walk(obj)
  return signals


def extract_from_message(
  message: dict[str, Any],
  *,
  role: Literal["user", "agent"],
  source_id_base: str,
) -> list[MediaSignal]:
  """Full message → multimodal signals (text, images, audio, tool calls)."""
  signals: list[MediaSignal] = []
  content = message.get("content")
  signals.extend(
    extract_from_content_parts(content, source_id_base=source_id_base)
  )

  # Explicit attachments some Hermes gateway paths use
  for key in ("images", "attachments", "files", "media"):
    blob = message.get(key)
    if isinstance(blob, list):
      for i, item in enumerate(blob):
        if isinstance(item, str):
          sig = media_signal_from_ref(
            item, source_id=f"{source_id_base}/{key}/{i}"
          )
          if sig:
            signals.append(sig)
        elif isinstance(item, dict):
          url = str(
            item.get("url")
            or item.get("path")
            or item.get("file_path")
            or item.get("image_url")
            or ""
          )
          if url:
            sig = media_signal_from_ref(
              url, source_id=f"{source_id_base}/{key}/{i}"
            )
            if sig:
              signals.append(sig)

  if role == "agent" or str(message.get("role") or "") == "assistant":
    signals.extend(
      extract_tool_calls(message, source_id_base=source_id_base)
    )
  return _dedupe_signals(signals)


def extract_from_messages(
  messages: Iterable[dict[str, Any]],
  *,
  user_source: str,
  agent_source: str,
  max_messages: int = 24,
) -> list[MediaSignal]:
  """Scan a transcript tail for multimodal + tool signals."""
  msgs = [m for m in messages if isinstance(m, dict)]
  if len(msgs) > max_messages:
    msgs = msgs[-max_messages:]
  out: list[MediaSignal] = []
  for msg in msgs:
    role_raw = str(msg.get("role") or "")
    if role_raw == "user":
      out.extend(
        extract_from_message(msg, role="user", source_id_base=user_source)
      )
    elif role_raw == "assistant":
      out.extend(
        extract_from_message(msg, role="agent", source_id_base=agent_source)
      )
    elif role_raw == "tool":
      name = str(msg.get("name") or msg.get("tool_name") or "tool")
      out.extend(
        extract_from_message(
          msg,
          role="agent",
          source_id_base=f"{agent_source}/tool/{name}",
        )
      )
  return _dedupe_signals(out)


def _dedupe_signals(signals: list[MediaSignal]) -> list[MediaSignal]:
  seen: set[tuple[str, str, int]] = set()
  out: list[MediaSignal] = []
  for s in signals:
    if s.is_empty():
      continue
    key = (s.modality, s.source_id, len(s.text) + len(s.image_rgb) + len(s.pcm))
    if key in seen:
      continue
    seen.add(key)
    out.append(s)
  return out


def process_signal_on_engine(
  engine: Any,
  signal: MediaSignal,
  *,
  retention: Any = None,
) -> dict[str, Any]:
  """Dispatch a MediaSignal to the matching Cortext process_* API.

  Defaults to Durable for Hermes chat turns (force boundary + force write)
  when retention is omitted. Prefetch paths pass Ephemeral.
  """
  try:
    import augmem.cortext as cortext

    if retention is None:
      retention = cortext.Retention.DURABLE
  except Exception:
    retention = retention

  if signal.modality == "text":
    return engine.process_text(
      signal.text,
      source_id=signal.source_id,
      include_embedding=False,
      retention=retention,
    )
  if signal.modality == "image":
    try:
      import augmem.cortext as cortext

      media = None
      if signal.media_bytes:
        media = cortext.Media(
          data=signal.media_bytes,
          mimetype=signal.media_mimetype or "image/png",
        )
      if media is not None:
        return engine.process_image_with_media(
          signal.image_rgb,
          signal.width,
          signal.height,
          signal.channels,
          signal.source_id,
          media,
          include_embedding=False,
          retention=retention,
        )
    except Exception:
      pass
    return engine.process_image(
      signal.image_rgb,
      signal.width,
      signal.height,
      signal.channels,
      signal.source_id,
      include_embedding=False,
      retention=retention,
    )
  if signal.modality == "audio":
    try:
      import augmem.cortext as cortext

      if signal.media_bytes:
        media = cortext.Media(
          data=signal.media_bytes,
          mimetype=signal.media_mimetype or "audio/wav",
        )
        return engine.process_audio_with_media(
          signal.pcm,
          signal.source_id,
          media,
          include_embedding=False,
          retention=retention,
        )
    except Exception:
      pass
    return engine.process_audio(
      signal.pcm,
      signal.source_id,
      include_embedding=False,
      retention=retention,
    )
  raise ValueError(f"unknown modality {signal.modality}")
