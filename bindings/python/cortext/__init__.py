from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "CONSOLIDATE_BOTH",
    "CONSOLIDATE_DEEP",
    "CONSOLIDATE_SHALLOW",
    "Config",
    "Cortext",
    "CortextError",
    "last_error",
    "load_library",
    "version",
]

CONSOLIDATE_SHALLOW = 0
CONSOLIDATE_DEEP = 1
CONSOLIDATE_BOTH = 2


class CortextError(RuntimeError):
    pass


@dataclass(slots=True)
class Config:
    focus: float = 0.5
    sensitivity: float = 0.5
    stability: float = 0.5
    affect_interrupt: bool = True
    affect_retrieval: bool = True
    reinforcement_enabled: bool = True
    procedural_enabled: bool = True
    sequential_edges_enabled: bool = True
    label_bank_path: str | None = None


class _NativeConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("focus", ctypes.c_double),
        ("sensitivity", ctypes.c_double),
        ("stability", ctypes.c_double),
        ("affect_interrupt", ctypes.c_int),
        ("affect_retrieval", ctypes.c_int),
        ("reinforcement_enabled", ctypes.c_int),
        ("procedural_enabled", ctypes.c_int),
        ("sequential_edges_enabled", ctypes.c_int),
        ("label_bank_path", ctypes.c_char_p),
    ]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _candidate_library_paths() -> list[Path]:
    root = _repo_root()
    names = ("libcortext.dylib", "libcortext.so", "cortext.dll", "libcortext.dll")
    candidates: list[Path] = []

    env_path = os.environ.get("CORTEXT_LIBRARY_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    for directory in (
        root / "build" / "ffi-release",
        root / "build" / "ffi-release" / "lib",
        root / "install" / "lib",
    ):
        for name in names:
            candidates.append(directory / name)

    return candidates


def _configure_library(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.cortext_config_init.argtypes = [ctypes.POINTER(_NativeConfig)]
    lib.cortext_config_init.restype = None

    lib.cortext_create_with_config.argtypes = [
        ctypes.POINTER(_NativeConfig),
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    lib.cortext_create_with_config.restype = ctypes.c_void_p

    lib.cortext_free.argtypes = [ctypes.c_void_p]
    lib.cortext_free.restype = None

    lib.cortext_process_text_json.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    lib.cortext_process_text_json.restype = ctypes.c_void_p

    lib.cortext_process_audio_json.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_char_p,
    ]
    lib.cortext_process_audio_json.restype = ctypes.c_void_p

    lib.cortext_process_image_json.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    lib.cortext_process_image_json.restype = ctypes.c_void_p

    lib.cortext_consolidate_json.argtypes = [ctypes.c_void_p]
    lib.cortext_consolidate_json.restype = ctypes.c_void_p

    lib.cortext_consolidate_mode_json.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.cortext_consolidate_mode_json.restype = ctypes.c_void_p

    lib.cortext_flush.argtypes = [ctypes.c_void_p]
    lib.cortext_flush.restype = ctypes.c_int

    lib.cortext_version.argtypes = []
    lib.cortext_version.restype = ctypes.c_char_p

    lib.cortext_last_error.argtypes = []
    lib.cortext_last_error.restype = ctypes.c_char_p

    lib.cortext_string_free.argtypes = [ctypes.c_void_p]
    lib.cortext_string_free.restype = None
    return lib


def load_library(path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    if path is not None:
        return _configure_library(ctypes.CDLL(os.fspath(path)))

    for candidate in _candidate_library_paths():
        if candidate.exists():
            return _configure_library(ctypes.CDLL(os.fspath(candidate)))

    discovered = ctypes.util.find_library("cortext")
    if discovered:
        return _configure_library(ctypes.CDLL(discovered))

    searched = "\n".join(str(candidate) for candidate in _candidate_library_paths())
    raise CortextError(
        "Could not locate the Cortext shared library. "
        "Build it with `cmake --preset ffi-release` and "
        "`cmake --build --preset ffi-release --target cortext`, "
        "or set CORTEXT_LIBRARY_PATH.\n"
        f"Searched:\n{searched}"
    )


_LIB = load_library()


def last_error() -> str:
    raw = _LIB.cortext_last_error()
    return raw.decode("utf-8") if raw else ""


def version() -> str:
    raw = _LIB.cortext_version()
    return raw.decode("utf-8") if raw else ""


def _raise_last_error(prefix: str) -> None:
    detail = last_error()
    raise CortextError(f"{prefix}: {detail}" if detail else prefix)


def _bool_to_int(value: bool) -> int:
    return 1 if value else 0


def _encode_optional_string(value: str | None) -> ctypes.Array[ctypes.c_char] | None:
    if value is None:
        return None
    return ctypes.create_string_buffer(value.encode("utf-8"))


class Cortext:
    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        models_dir: str | None = None,
        config: Config | None = None,
        library_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._lib = _LIB if library_path is None else load_library(library_path)
        native_cfg = _NativeConfig()
        self._lib.cortext_config_init(ctypes.byref(native_cfg))

        label_bank_keepalive = None
        if config is not None:
            native_cfg.focus = config.focus
            native_cfg.sensitivity = config.sensitivity
            native_cfg.stability = config.stability
            native_cfg.affect_interrupt = _bool_to_int(config.affect_interrupt)
            native_cfg.affect_retrieval = _bool_to_int(config.affect_retrieval)
            native_cfg.reinforcement_enabled = _bool_to_int(config.reinforcement_enabled)
            native_cfg.procedural_enabled = _bool_to_int(config.procedural_enabled)
            native_cfg.sequential_edges_enabled = _bool_to_int(config.sequential_edges_enabled)
            label_bank_keepalive = _encode_optional_string(config.label_bank_path)
            native_cfg.label_bank_path = (
                ctypes.cast(label_bank_keepalive, ctypes.c_char_p)
                if label_bank_keepalive is not None
                else None
            )

        db_path_raw = db_path.encode("utf-8")
        models_dir_raw = models_dir.encode("utf-8") if models_dir is not None else None
        self._handle = self._lib.cortext_create_with_config(
            ctypes.byref(native_cfg),
            db_path_raw,
            models_dir_raw,
        )
        self._label_bank_keepalive = label_bank_keepalive
        if not self._handle:
            _raise_last_error("cortext_create_with_config failed")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.cortext_free(self._handle)
            self._handle = None

    def __enter__(self) -> "Cortext":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def flush(self) -> None:
        status = self._lib.cortext_flush(self._handle)
        if status != 0:
            _raise_last_error("cortext_flush failed")

    def process_text_json(self, text: str, source_id: str) -> str:
        return self._call_json(
            self._lib.cortext_process_text_json,
            text.encode("utf-8"),
            source_id.encode("utf-8"),
        )

    def process_text(self, text: str, source_id: str) -> dict[str, Any]:
        return json.loads(self.process_text_json(text, source_id))

    def process_audio_json(self, pcm: Iterable[float], source_id: str) -> str:
        samples = array("f", pcm)
        raw = (ctypes.c_float * len(samples)).from_buffer(samples)
        return self._call_json(
            self._lib.cortext_process_audio_json,
            raw,
            len(samples),
            source_id.encode("utf-8"),
        )

    def process_audio(self, pcm: Iterable[float], source_id: str) -> dict[str, Any]:
        return json.loads(self.process_audio_json(pcm, source_id))

    def process_image_json(
        self,
        data: bytes | bytearray | memoryview,
        width: int,
        height: int,
        channels: int,
        source_id: str,
    ) -> str:
        blob = bytes(data)
        raw = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        return self._call_json(
            self._lib.cortext_process_image_json,
            raw,
            width,
            height,
            channels,
            source_id.encode("utf-8"),
        )

    def process_image(
        self,
        data: bytes | bytearray | memoryview,
        width: int,
        height: int,
        channels: int,
        source_id: str,
    ) -> dict[str, Any]:
        return json.loads(
            self.process_image_json(data, width, height, channels, source_id)
        )

    def consolidate_json(self) -> str:
        return self._call_json(self._lib.cortext_consolidate_json)

    def consolidate(self) -> dict[str, Any]:
        return json.loads(self.consolidate_json())

    def consolidate_mode_json(self, mode: int) -> str:
        return self._call_json(self._lib.cortext_consolidate_mode_json, mode)

    def consolidate_mode(self, mode: int) -> dict[str, Any]:
        return json.loads(self.consolidate_mode_json(mode))

    def _call_json(self, fn: Any, *args: Any) -> str:
        result = fn(self._handle, *args)
        if not result:
            _raise_last_error(f"{fn.__name__} failed")
        try:
            return ctypes.string_at(result).decode("utf-8")
        finally:
            self._lib.cortext_string_free(result)
