from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import os
import platform
import tempfile
import urllib.error
import urllib.request
from array import array
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "Config",
    "Cortext",
    "CortextError",
    "DBProvider",
    "Media",
    "ObjectStoreProvider",
    "Retention",
    "last_error",
    "load_library",
    "version",
]


class CortextError(RuntimeError):
    pass


class Retention(IntEnum):
    """Ingress retention policy (matches cortext::Retention / C API).

    Natural (default when omitted): boundary and write algorithms decide.
    Durable: explicit turn commit.
    Boundary: explicit turn edge only.
    Ephemeral: never store; explicit retrieval turn edge.
    """

    DURABLE = 0
    EPHEMERAL = 1
    NATURAL = 2
    BOUNDARY = 3


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
    signal_filter_audio_enabled: bool = True
    signal_filter_image_enabled: bool = True
    signal_filter_text_enabled: bool = False


@dataclass(frozen=True, slots=True)
class Media:
    data: bytes | bytearray | memoryview
    mimetype: str


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
        ("signal_filter_audio_enabled", ctypes.c_int),
        ("signal_filter_image_enabled", ctypes.c_int),
        ("signal_filter_text_enabled", ctypes.c_int),
    ]


class _NativeMedia(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("size", ctypes.c_size_t),
        ("mimetype", ctypes.c_char_p),
    ]


class _NativeProcessJsonOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("include_embedding", ctypes.c_int),
        ("retention", ctypes.c_int),
    ]


CORTEXT_DB_VALUE_NULL = 0
CORTEXT_DB_VALUE_INT64 = 1
CORTEXT_DB_VALUE_DOUBLE = 2
CORTEXT_DB_VALUE_TEXT = 3
CORTEXT_DB_VALUE_BLOB = 4


class _DBValue(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("int64_value", ctypes.c_int64),
        ("double_value", ctypes.c_double),
        ("text_value", ctypes.c_char_p),
        ("blob_data", ctypes.POINTER(ctypes.c_uint8)),
        ("blob_size", ctypes.c_size_t),
    ]


class _DBCell(ctypes.Structure):
    pass


class _DBRow(ctypes.Structure):
    pass


class _DBResult(ctypes.Structure):
    pass


_DBCell._fields_ = [("column_name", ctypes.c_char_p), ("value", _DBValue)]
_DBRow._fields_ = [
    ("cells", ctypes.POINTER(_DBCell)),
    ("cell_count", ctypes.c_size_t),
]
_DBResult._fields_ = [
    ("rows", ctypes.POINTER(_DBRow)),
    ("row_count", ctypes.c_size_t),
]


_ExecuteCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(_DBValue),
    ctypes.c_size_t,
    ctypes.POINTER(_DBResult),
    ctypes.POINTER(ctypes.c_char_p),
)
_BeginCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_char_p),
)
_TransactionCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
)
_ReleaseTransactionCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_void_p,
)
_ReleaseResultCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.POINTER(_DBResult),
)
_ObjectBeginCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_char_p),
)
_ObjectPutCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_char_p),
)
_ObjectGetCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char_p),
)
_ObjectExistsCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_char_p),
)
_ObjectDeleteCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_char_p),
)
_ObjectTransactionCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
)
_ObjectReleaseTransactionCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_void_p,
)
_ObjectReleaseBlobCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
)


class _DBCallbacks(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("execute", _ExecuteCallback),
        ("begin", _BeginCallback),
        ("commit", _TransactionCallback),
        ("rollback", _TransactionCallback),
        ("release_transaction", _ReleaseTransactionCallback),
        ("release_result", _ReleaseResultCallback),
    ]


class _ObjectCallbacks(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("begin", _ObjectBeginCallback),
        ("put", _ObjectPutCallback),
        ("get", _ObjectGetCallback),
        ("exists", _ObjectExistsCallback),
        ("delete_object", _ObjectDeleteCallback),
        ("commit", _ObjectTransactionCallback),
        ("rollback", _ObjectTransactionCallback),
        ("release_transaction", _ObjectReleaseTransactionCallback),
        ("release_blob", _ObjectReleaseBlobCallback),
    ]


class DBProvider:
    def execute(self, query: str, params: list[Any], transaction: Any | None) -> list[dict[str, Any]]:
        raise NotImplementedError

    def begin(self, parent_transaction: Any | None) -> Any:
        raise NotImplementedError

    def commit(self, transaction: Any) -> None:
        raise NotImplementedError

    def rollback(self, transaction: Any) -> None:
        raise NotImplementedError

    def release_transaction(self, transaction: Any) -> None:
        pass


class ObjectStoreProvider:
    def begin(self, parent_transaction: Any | None) -> Any:
        raise NotImplementedError

    def put(self, object_id: bytes, data: bytes, transaction: Any) -> None:
        raise NotImplementedError

    def get(self, object_id: bytes, transaction: Any) -> bytes | None:
        raise NotImplementedError

    def exists(self, object_id: bytes, transaction: Any) -> bool:
        raise NotImplementedError

    def delete(self, object_id: bytes, transaction: Any) -> bool:
        raise NotImplementedError

    def commit(self, transaction: Any) -> None:
        raise NotImplementedError

    def rollback(self, transaction: Any) -> None:
        raise NotImplementedError

    def release_transaction(self, transaction: Any) -> None:
        pass


_SUPPORTED_NATIVE_TARGETS = {
    "linux-aarch64",
    "linux-x86_64",
    "macos-aarch64",
    "macos-x86_64",
    "windows-aarch64",
    "windows-x86_64",
}

_DLL_DIRECTORY_HANDLES: list[Any] = []

_AIST_REPO = "augmem/AIST-87M-GGUF"
_AIST_REVISION = "main"
_AIST_MODEL_FILE = {
    "filename": "AIST-87M_q8_0.gguf",
    "sha256": "bf4c49954eccc65183f1a97e44606e86c7ee5a4fea500457124b687a3ec97898",
    "size": 141491936,
}
_TOKENIZER_REPO = "bert-base-uncased"
_TOKENIZER_REVISION = "main"
_TOKENIZER_FILE = {
    "filename": "vocab.txt",
    "target": Path("mdbr-leaf-ir") / "vocab.txt",
    "sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "size": 231508,
}
_DOWNLOAD_TIMEOUT_SECONDS = 60


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _platform_tag() -> str:
    system = platform.system().lower()
    if system == "darwin":
        os_tag = "macos"
    elif system == "linux":
        os_tag = "linux"
    elif system == "windows":
        os_tag = "windows"
    else:
        os_tag = system or "unknown"

    machine = platform.machine().lower()
    if machine in {"amd64", "x64"}:
        arch = "x86_64"
    elif machine == "arm64":
        arch = "aarch64"
    else:
        arch = machine or "unknown"

    return f"{os_tag}-{arch}"


def _library_name_for_tag(tag: str) -> str:
    os_tag = tag.split("-", 1)[0]
    if os_tag == "macos":
        return "libcortext.dylib"
    if os_tag == "windows":
        return "cortext.dll"
    return "libcortext.so"


def _repo_root() -> Path | None:
    for parent in (_package_dir(), *_package_dir().parents):
        if (parent / "build.zig").is_file() and (parent / "CMakeLists.txt").is_file():
            return parent
    return None


def _aist_model_path(root: Path) -> Path | None:
    candidates = (
        root / "AIST-87M-GGUF" / "AIST-87M_q8_0.gguf",
        root / "AIST-87M-GGUF" / "AIST-87M_q5_1.gguf",
    )
    return next((path for path in candidates if path.is_file()), None)


def _hf_url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected_sha256: str, expected_size: int) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and _sha256_file(path) == expected_sha256
    )


def _model_cache_dir() -> Path:
    override = os.environ.get("CORTEXT_MODEL_CACHE_DIR")
    if override:
        return Path(override).expanduser()

    if platform.system().lower() == "windows":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base).expanduser() if base else Path.home() / "AppData" / "Local"
    elif platform.system().lower() == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()

    return root / "augmem" / "cortext" / "models"


def _download_file(url: str, dest: Path, expected_sha256: str, expected_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.", suffix=".tmp", dir=os.fspath(dest.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)

    headers = {"User-Agent": "cortext-python-model-bootstrap/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(
            request, timeout=_DOWNLOAD_TIMEOUT_SECONDS
        ) as response, tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        actual_size = tmp.stat().st_size
        actual_sha256 = _sha256_file(tmp)
        if actual_size != expected_size:
            raise CortextError(
                f"downloaded {dest.name} has size {actual_size}, expected {expected_size}"
            )
        if actual_sha256 != expected_sha256:
            raise CortextError(
                f"downloaded {dest.name} has sha256 {actual_sha256}, expected {expected_sha256}"
            )
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _ensure_verified_file(
    url: str, dest: Path, expected_sha256: str, expected_size: int
) -> None:
    if _verify_file(dest, expected_sha256, expected_size):
        return
    try:
        _download_file(url, dest, expected_sha256, expected_size)
    except urllib.error.HTTPError as exc:
        raise CortextError(
            f"could not download Cortext model asset {dest.name}: HTTP {exc.code}. "
            "Set CORTEXT_AIST_MODEL_PATH to an existing AIST GGUF file or retry later."
        ) from exc
    except urllib.error.URLError as exc:
        raise CortextError(
            f"could not download Cortext model asset {dest.name}: {exc.reason}. "
            "Set CORTEXT_AIST_MODEL_PATH to an existing AIST GGUF file or retry later."
        ) from exc


def _cached_default_aist_model_path(cache_root: Path) -> Path | None:
    model_path = cache_root / "AIST-87M-GGUF" / _AIST_MODEL_FILE["filename"]
    tokenizer_path = cache_root / _TOKENIZER_FILE["target"]
    if _verify_file(
        model_path,
        str(_AIST_MODEL_FILE["sha256"]),
        int(_AIST_MODEL_FILE["size"]),
    ) and _verify_file(
        tokenizer_path,
        str(_TOKENIZER_FILE["sha256"]),
        int(_TOKENIZER_FILE["size"]),
    ):
        return model_path
    return None


def _ensure_cached_default_aist_model_path() -> Path:
    cache_root = _model_cache_dir()
    if cached := _cached_default_aist_model_path(cache_root):
        return cached

    model_path = cache_root / "AIST-87M-GGUF" / _AIST_MODEL_FILE["filename"]
    tokenizer_path = cache_root / _TOKENIZER_FILE["target"]
    _ensure_verified_file(
        _hf_url(_AIST_REPO, _AIST_REVISION, str(_AIST_MODEL_FILE["filename"])),
        model_path,
        str(_AIST_MODEL_FILE["sha256"]),
        int(_AIST_MODEL_FILE["size"]),
    )
    _ensure_verified_file(
        _hf_url(_TOKENIZER_REPO, _TOKENIZER_REVISION, str(_TOKENIZER_FILE["filename"])),
        tokenizer_path,
        str(_TOKENIZER_FILE["sha256"]),
        int(_TOKENIZER_FILE["size"]),
    )
    return model_path


def _default_aist_model_path() -> str | None:
    package_models = _package_dir() / "models"
    if model_path := _aist_model_path(package_models):
        return os.fspath(model_path)

    root = _repo_root()
    if root is not None:
        repo_models = root / "models"
        if model_path := _aist_model_path(repo_models):
            return os.fspath(model_path)

    return os.fspath(_ensure_cached_default_aist_model_path())



def _candidate_library_paths() -> list[Path]:
    names = ("libcortext.dylib", "libcortext.so", "cortext.dll", "libcortext.dll")
    candidates: list[Path] = []

    env_path = os.environ.get("CORTEXT_LIBRARY_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    tag = _platform_tag()
    if tag in _SUPPORTED_NATIVE_TARGETS:
        candidates.append(_package_dir() / "native" / tag / _library_name_for_tag(tag))

    for name in names:
        candidates.append(_package_dir() / name)

    root = _repo_root()
    if root is not None:
        for directory in (
            root / "build" / "ffi-release",
            root / "build" / "ffi-release" / "lib",
            root / "zig-out" / "lib",
            root / "zig-out" / "bin",
            root / "install" / "lib",
            root / "install" / "bin",
        ):
            for name in names:
                candidates.append(directory / name)

    return candidates


def _load_cdll(path: Path | str) -> ctypes.CDLL:
    library_path = Path(path) if not isinstance(path, str) else Path(path)
    if os.name == "nt" and library_path.parent and hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(os.fspath(library_path.parent)))
    return ctypes.CDLL(os.fspath(path))


def _configure_library(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.cortext_config_init.argtypes = [ctypes.POINTER(_NativeConfig)]
    lib.cortext_config_init.restype = None

    lib.cortext_create_with_config.argtypes = [
        ctypes.POINTER(_NativeConfig),
        ctypes.c_char_p,
    ]
    lib.cortext_create_with_config.restype = ctypes.c_void_p

    lib.cortext_create_with_store_callbacks.argtypes = [
        ctypes.POINTER(_NativeConfig),
        ctypes.POINTER(_DBCallbacks),
        ctypes.c_void_p,
    ]
    lib.cortext_create_with_store_callbacks.restype = ctypes.c_void_p

    lib.cortext_create_with_store_and_object_callbacks.argtypes = [
        ctypes.POINTER(_NativeConfig),
        ctypes.POINTER(_DBCallbacks),
        ctypes.c_void_p,
        ctypes.POINTER(_ObjectCallbacks),
        ctypes.c_void_p,
    ]
    lib.cortext_create_with_store_and_object_callbacks.restype = ctypes.c_void_p

    lib.cortext_create_with_config_and_object_callbacks.argtypes = [
        ctypes.POINTER(_NativeConfig),
        ctypes.c_char_p,
        ctypes.POINTER(_ObjectCallbacks),
        ctypes.c_void_p,
    ]
    lib.cortext_create_with_config_and_object_callbacks.restype = ctypes.c_void_p

    lib.cortext_free.argtypes = [ctypes.c_void_p]
    lib.cortext_free.restype = None

    lib.cortext_process_text_json.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    lib.cortext_process_text_json.restype = ctypes.c_void_p

    lib.cortext_process_text_json_with_options.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_NativeProcessJsonOptions),
    ]
    lib.cortext_process_text_json_with_options.restype = ctypes.c_void_p

    lib.cortext_process_audio_json.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_char_p,
    ]
    lib.cortext_process_audio_json.restype = ctypes.c_void_p

    lib.cortext_process_audio_json_with_options.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.POINTER(_NativeProcessJsonOptions),
    ]
    lib.cortext_process_audio_json_with_options.restype = ctypes.c_void_p

    lib.cortext_process_audio_with_media_json.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.POINTER(_NativeMedia),
    ]
    lib.cortext_process_audio_with_media_json.restype = ctypes.c_void_p

    lib.cortext_process_audio_with_media_json_with_options.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.POINTER(_NativeMedia),
        ctypes.POINTER(_NativeProcessJsonOptions),
    ]
    lib.cortext_process_audio_with_media_json_with_options.restype = ctypes.c_void_p

    lib.cortext_process_image_json.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    lib.cortext_process_image_json.restype = ctypes.c_void_p

    lib.cortext_process_image_json_with_options.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(_NativeProcessJsonOptions),
    ]
    lib.cortext_process_image_json_with_options.restype = ctypes.c_void_p

    lib.cortext_process_image_with_media_json.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(_NativeMedia),
    ]
    lib.cortext_process_image_with_media_json.restype = ctypes.c_void_p

    lib.cortext_process_image_with_media_json_with_options.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(_NativeMedia),
        ctypes.POINTER(_NativeProcessJsonOptions),
    ]
    lib.cortext_process_image_with_media_json_with_options.restype = ctypes.c_void_p

    lib.cortext_embed_text_json.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.cortext_embed_text_json.restype = ctypes.c_void_p

    lib.cortext_embed_audio_json.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.cortext_embed_audio_json.restype = ctypes.c_void_p

    lib.cortext_embed_image_json.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.cortext_embed_image_json.restype = ctypes.c_void_p

    lib.cortext_consolidate_json.argtypes = [ctypes.c_void_p]
    lib.cortext_consolidate_json.restype = ctypes.c_void_p

    lib.cortext_flush.argtypes = [ctypes.c_void_p]
    lib.cortext_flush.restype = ctypes.c_int

    lib.cortext_reset.argtypes = [ctypes.c_void_p]
    lib.cortext_reset.restype = ctypes.c_int

    lib.cortext_version.argtypes = []
    lib.cortext_version.restype = ctypes.c_char_p

    lib.cortext_last_error.argtypes = []
    lib.cortext_last_error.restype = ctypes.c_char_p

    lib.cortext_string_free.argtypes = [ctypes.c_void_p]
    lib.cortext_string_free.restype = None
    return lib


def load_library(path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    if path is not None:
        return _configure_library(_load_cdll(Path(path)))

    for candidate in _candidate_library_paths():
        if candidate.exists():
            return _configure_library(_load_cdll(candidate))

    discovered = ctypes.util.find_library("cortext")
    if discovered:
        return _configure_library(ctypes.CDLL(discovered))

    searched = "\n".join(str(candidate) for candidate in _candidate_library_paths())
    supported = ", ".join(sorted(_SUPPORTED_NATIVE_TARGETS))
    raise CortextError(
        "Could not locate the Cortext shared library. "
        f"Current platform tag is {_platform_tag()}; bundled wheels support: {supported}. "
        "Build a wheel with `python scripts/build_python_package.py`, build a local shared "
        "library with `cmake --preset ffi-release && cmake --build --preset ffi-release --target cortext`, "
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


def _native_media(
    media: Media | bytes | bytearray | memoryview | None,
    mimetype: str | None,
) -> tuple[_NativeMedia, Any, bytes | None]:
    if media is None:
        return _NativeMedia(None, 0, None), None, None
    if isinstance(media, Media):
        if mimetype is None:
            mimetype = media.mimetype
        media = media.data
    blob = bytes(media)
    if not blob:
        return _NativeMedia(None, 0, None), None, None
    if not mimetype:
        raise ValueError("media_mimetype is required when media bytes are provided")
    raw = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
    mimetype_bytes = mimetype.encode("utf-8")
    return _NativeMedia(raw, len(blob), mimetype_bytes), raw, mimetype_bytes


def _coerce_retention(retention: Retention | int | None) -> int:
    if retention is None:
        return int(Retention.NATURAL)
    return int(retention)


def _native_process_json_options(
    include_embedding: bool,
    retention: Retention | int | None = None,
) -> _NativeProcessJsonOptions:
    return _NativeProcessJsonOptions(
        ctypes.sizeof(_NativeProcessJsonOptions),
        1 if include_embedding else 0,
        _coerce_retention(retention),
    )


def _bool_to_int(value: bool) -> int:
    return 1 if value else 0


def _encode_optional_string(value: str | None) -> ctypes.Array[ctypes.c_char] | None:
    if value is None:
        return None
    return ctypes.create_string_buffer(value.encode("utf-8"))


class _ProviderBridge:
    def __init__(self, provider: DBProvider) -> None:
        self.provider = provider
        self._transactions: dict[int, Any] = {}
        self._next_transaction = 1
        self._results: dict[int, Any] = {}
        self._errors: list[ctypes.Array[ctypes.c_char]] = []

        self.execute_cb = _ExecuteCallback(self._execute)
        self.begin_cb = _BeginCallback(self._begin)
        self.commit_cb = _TransactionCallback(self._commit)
        self.rollback_cb = _TransactionCallback(self._rollback)
        self.release_transaction_cb = _ReleaseTransactionCallback(
            self._release_transaction
        )
        self.release_result_cb = _ReleaseResultCallback(self._release_result)
        self.callbacks = _DBCallbacks(
            ctypes.sizeof(_DBCallbacks),
            self.execute_cb,
            self.begin_cb,
            self.commit_cb,
            self.rollback_cb,
            self.release_transaction_cb,
            self.release_result_cb,
        )

    def _set_error(self, out_error: Any, message: str) -> int:
        encoded = ctypes.create_string_buffer(message.encode("utf-8"))
        self._errors.append(encoded)
        if out_error:
            out_error[0] = ctypes.cast(encoded, ctypes.c_char_p)
        return 1

    def _transaction(self, handle: int) -> Any | None:
        return self._transactions.get(handle) if handle else None

    def _value_from_c(self, value: _DBValue) -> Any:
        if value.type == CORTEXT_DB_VALUE_INT64:
            return int(value.int64_value)
        if value.type == CORTEXT_DB_VALUE_DOUBLE:
            return float(value.double_value)
        if value.type == CORTEXT_DB_VALUE_TEXT:
            return value.text_value.decode("utf-8") if value.text_value else ""
        if value.type == CORTEXT_DB_VALUE_BLOB:
            if not value.blob_data or value.blob_size == 0:
                return b""
            return bytes(value.blob_data[: value.blob_size])
        return None

    def _db_value(self, value: Any, keepalive: list[Any]) -> _DBValue:
        out = _DBValue()
        if value is None:
            out.type = CORTEXT_DB_VALUE_NULL
        elif isinstance(value, bool):
            out.type = CORTEXT_DB_VALUE_INT64
            out.int64_value = int(value)
        elif isinstance(value, int):
            out.type = CORTEXT_DB_VALUE_INT64
            out.int64_value = value
        elif isinstance(value, float):
            out.type = CORTEXT_DB_VALUE_DOUBLE
            out.double_value = value
        elif isinstance(value, str):
            raw = ctypes.create_string_buffer(value.encode("utf-8"))
            keepalive.append(raw)
            out.type = CORTEXT_DB_VALUE_TEXT
            out.text_value = ctypes.cast(raw, ctypes.c_char_p)
        else:
            raw_bytes = bytes(value)
            out.type = CORTEXT_DB_VALUE_BLOB
            out.blob_size = len(raw_bytes)
            if raw_bytes:
                array_type = ctypes.c_uint8 * len(raw_bytes)
                raw = array_type.from_buffer_copy(raw_bytes)
                keepalive.append(raw)
                out.blob_data = ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint8))
        return out

    def _execute(
        self,
        _user_data: int,
        transaction: int,
        query: bytes,
        params: Any,
        param_count: int,
        out_result: Any,
        out_error: Any,
    ) -> int:
        try:
            py_params = [
                self._value_from_c(params[index]) for index in range(param_count)
            ]
            rows = self.provider.execute(
                query.decode("utf-8"), py_params, self._transaction(transaction)
            )

            keepalive: list[Any] = []
            row_array = (_DBRow * len(rows))()
            for row_index, row in enumerate(rows):
                items = list(row.items())
                cell_array = (_DBCell * len(items))()
                keepalive.append(cell_array)
                for cell_index, (name, value) in enumerate(items):
                    name_raw = ctypes.create_string_buffer(name.encode("utf-8"))
                    keepalive.append(name_raw)
                    cell_array[cell_index].column_name = ctypes.cast(
                        name_raw, ctypes.c_char_p
                    )
                    cell_array[cell_index].value = self._db_value(value, keepalive)
                row_array[row_index].cells = cell_array
                row_array[row_index].cell_count = len(items)

            keepalive.append(row_array)
            out_result[0].rows = row_array
            out_result[0].row_count = len(rows)
            self._results[ctypes.addressof(out_result.contents)] = keepalive
            return 0
        except Exception as exc:
            return self._set_error(out_error, str(exc))

    def _begin(
        self,
        _user_data: int,
        parent_transaction: int,
        out_transaction: Any,
        out_error: Any,
    ) -> int:
        try:
            tx = self.provider.begin(self._transaction(parent_transaction))
            handle = self._next_transaction
            self._next_transaction += 1
            self._transactions[handle] = tx
            out_transaction[0] = handle
            return 0
        except Exception as exc:
            return self._set_error(out_error, str(exc))

    def _commit(self, _user_data: int, transaction: int, out_error: Any) -> int:
        try:
            self.provider.commit(self._transactions[transaction])
            return 0
        except Exception as exc:
            return self._set_error(out_error, str(exc))

    def _rollback(self, _user_data: int, transaction: int, out_error: Any) -> int:
        try:
            self.provider.rollback(self._transactions[transaction])
            return 0
        except Exception as exc:
            return self._set_error(out_error, str(exc))

    def _release_transaction(self, _user_data: int, transaction: int) -> None:
        tx = self._transactions.pop(transaction, None)
        if tx is not None:
            self.provider.release_transaction(tx)

    def _release_result(self, _user_data: int, result: Any) -> None:
        self._results.pop(ctypes.addressof(result.contents), None)


class _ObjectProviderBridge:
    def __init__(self, provider: ObjectStoreProvider) -> None:
        self.provider = provider
        self._transactions: dict[int, Any] = {}
        self._next_transaction = 1
        self._errors: list[ctypes.Array[ctypes.c_char]] = []
        self._blobs: dict[int, Any] = {}

        self.begin_cb = _ObjectBeginCallback(self._begin)
        self.put_cb = _ObjectPutCallback(self._put)
        self.get_cb = _ObjectGetCallback(self._get)
        self.exists_cb = _ObjectExistsCallback(self._exists)
        self.delete_cb = _ObjectDeleteCallback(self._delete)
        self.commit_cb = _ObjectTransactionCallback(self._commit)
        self.rollback_cb = _ObjectTransactionCallback(self._rollback)
        self.release_transaction_cb = _ObjectReleaseTransactionCallback(
            self._release_transaction
        )
        self.release_blob_cb = _ObjectReleaseBlobCallback(self._release_blob)
        self.callbacks = _ObjectCallbacks(
            ctypes.sizeof(_ObjectCallbacks),
            self.begin_cb,
            self.put_cb,
            self.get_cb,
            self.exists_cb,
            self.delete_cb,
            self.commit_cb,
            self.rollback_cb,
            self.release_transaction_cb,
            self.release_blob_cb,
        )

    def _set_error(self, out_error: Any, message: str) -> int:
        encoded = ctypes.create_string_buffer(message.encode("utf-8"))
        self._errors.append(encoded)
        if out_error:
            out_error[0] = ctypes.cast(encoded, ctypes.c_char_p)
        return 1

    def _transaction(self, handle: int) -> Any | None:
        return self._transactions.get(handle) if handle else None

    def _bytes_from_c(self, data: Any, size: int) -> bytes:
        if not data or size == 0:
            return b""
        return bytes(data[:size])

    def _begin(
        self,
        _user_data: int,
        parent_transaction: int,
        out_transaction: Any,
        out_error: Any,
    ) -> int:
        try:
            tx = self.provider.begin(self._transaction(parent_transaction))
            handle = self._next_transaction
            self._next_transaction += 1
            self._transactions[handle] = tx
            out_transaction[0] = handle
            return 0
        except Exception as exc:
            return self._set_error(out_error, str(exc))

    def _put(
        self,
        _user_data: int,
        transaction: int,
        object_id: Any,
        object_id_size: int,
        data: Any,
        data_size: int,
        out_error: Any,
    ) -> int:
        try:
            self.provider.put(
                self._bytes_from_c(object_id, object_id_size),
                self._bytes_from_c(data, data_size),
                self._transactions[transaction],
            )
            return 0
        except Exception as exc:
            return self._set_error(out_error, str(exc))

    def _get(
        self,
        _user_data: int,
        transaction: int,
        object_id: Any,
        object_id_size: int,
        out_data: Any,
        out_data_size: Any,
        out_error: Any,
    ) -> int:
        try:
            data = self.provider.get(
                self._bytes_from_c(object_id, object_id_size),
                self._transactions[transaction],
            )
            if data is None:
                out_data[0] = ctypes.cast(None, ctypes.POINTER(ctypes.c_uint8))
                out_data_size[0] = 0
                return 0
            raw_bytes = bytes(data)
            array_type = ctypes.c_uint8 * len(raw_bytes)
            raw = array_type.from_buffer_copy(raw_bytes)
            ptr = ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint8))
            self._blobs[ctypes.addressof(raw)] = raw
            out_data[0] = ptr
            out_data_size[0] = len(raw_bytes)
            return 0
        except Exception as exc:
            return self._set_error(out_error, str(exc))

    def _exists(
        self,
        _user_data: int,
        transaction: int,
        object_id: Any,
        object_id_size: int,
        out_exists: Any,
        out_error: Any,
    ) -> int:
        try:
            out_exists[0] = int(
                self.provider.exists(
                    self._bytes_from_c(object_id, object_id_size),
                    self._transactions[transaction],
                )
            )
            return 0
        except Exception as exc:
            return self._set_error(out_error, str(exc))

    def _delete(
        self,
        _user_data: int,
        transaction: int,
        object_id: Any,
        object_id_size: int,
        out_deleted: Any,
        out_error: Any,
    ) -> int:
        try:
            out_deleted[0] = int(
                self.provider.delete(
                    self._bytes_from_c(object_id, object_id_size),
                    self._transactions[transaction],
                )
            )
            return 0
        except Exception as exc:
            return self._set_error(out_error, str(exc))

    def _commit(self, _user_data: int, transaction: int, out_error: Any) -> int:
        try:
            self.provider.commit(self._transactions[transaction])
            return 0
        except Exception as exc:
            return self._set_error(out_error, str(exc))

    def _rollback(self, _user_data: int, transaction: int, out_error: Any) -> int:
        try:
            self.provider.rollback(self._transactions[transaction])
            return 0
        except Exception as exc:
            return self._set_error(out_error, str(exc))

    def _release_transaction(self, _user_data: int, transaction: int) -> None:
        tx = self._transactions.pop(transaction, None)
        if tx is not None:
            self.provider.release_transaction(tx)

    def _release_blob(self, _user_data: int, data: Any, _data_size: int) -> None:
        if data:
            self._blobs.pop(ctypes.addressof(data.contents), None)


class Cortext:
    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        store: DBProvider | None = None,
        object_store: ObjectStoreProvider | None = None,
        config: Config | None = None,
        library_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if store is not None and db_path != ":memory:":
            raise ValueError("db_path and store are mutually exclusive")
        self._lib = _LIB if library_path is None else load_library(library_path)
        native_cfg = _NativeConfig()
        self._lib.cortext_config_init(ctypes.byref(native_cfg))

        if config is not None:
            native_cfg.focus = config.focus
            native_cfg.sensitivity = config.sensitivity
            native_cfg.stability = config.stability
            native_cfg.affect_interrupt = _bool_to_int(config.affect_interrupt)
            native_cfg.affect_retrieval = _bool_to_int(config.affect_retrieval)
            native_cfg.reinforcement_enabled = _bool_to_int(config.reinforcement_enabled)
            native_cfg.procedural_enabled = _bool_to_int(config.procedural_enabled)
            native_cfg.sequential_edges_enabled = _bool_to_int(config.sequential_edges_enabled)
            native_cfg.signal_filter_audio_enabled = _bool_to_int(
                config.signal_filter_audio_enabled
            )
            native_cfg.signal_filter_image_enabled = _bool_to_int(
                config.signal_filter_image_enabled
            )
            native_cfg.signal_filter_text_enabled = _bool_to_int(
                config.signal_filter_text_enabled
            )

        self._provider_bridge = _ProviderBridge(store) if store is not None else None
        self._object_provider_bridge = (
            _ObjectProviderBridge(object_store) if object_store is not None else None
        )
        had_model_env = "CORTEXT_AIST_MODEL_PATH" in os.environ
        previous_model_env = os.environ.get("CORTEXT_AIST_MODEL_PATH")
        default_model_path = None if had_model_env else _default_aist_model_path()
        if not had_model_env and default_model_path is not None:
            os.environ["CORTEXT_AIST_MODEL_PATH"] = default_model_path
        try:
            if self._provider_bridge is not None and self._object_provider_bridge is not None:
                self._handle = self._lib.cortext_create_with_store_and_object_callbacks(
                    ctypes.byref(native_cfg),
                    ctypes.byref(self._provider_bridge.callbacks),
                    None,
                    ctypes.byref(self._object_provider_bridge.callbacks),
                    None,
                )
            elif self._provider_bridge is not None:
                self._handle = self._lib.cortext_create_with_store_callbacks(
                    ctypes.byref(native_cfg),
                    ctypes.byref(self._provider_bridge.callbacks),
                    None,
                )
            elif self._object_provider_bridge is not None:
                db_path_raw = db_path.encode("utf-8")
                self._handle = self._lib.cortext_create_with_config_and_object_callbacks(
                    ctypes.byref(native_cfg),
                    db_path_raw,
                    ctypes.byref(self._object_provider_bridge.callbacks),
                    None,
                )
            else:
                db_path_raw = db_path.encode("utf-8")
                self._handle = self._lib.cortext_create_with_config(
                    ctypes.byref(native_cfg),
                    db_path_raw,
                )
        finally:
            if not had_model_env:
                os.environ.pop("CORTEXT_AIST_MODEL_PATH", None)
            elif previous_model_env is not None:
                os.environ["CORTEXT_AIST_MODEL_PATH"] = previous_model_env
        if not self._handle:
            _raise_last_error("cortext create failed")

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

    def reset(self) -> None:
        status = self._lib.cortext_reset(self._handle)
        if status != 0:
            _raise_last_error("cortext_reset failed")

    def process_text_json(
        self,
        text: str,
        source_id: str,
        include_embedding: bool = True,
        retention: Retention | int | None = None,
    ) -> str:
        options = _native_process_json_options(include_embedding, retention)
        return self._call_json(
            self._lib.cortext_process_text_json_with_options,
            text.encode("utf-8"),
            source_id.encode("utf-8"),
            ctypes.byref(options),
        )

    def process_text(
        self,
        text: str,
        source_id: str,
        include_embedding: bool = True,
        retention: Retention | int | None = None,
    ) -> dict[str, Any]:
        return json.loads(
            self.process_text_json(
                text, source_id, include_embedding, retention
            )
        )

    def embed_text_json(self, text: str) -> str:
        return self._call_json(
            self._lib.cortext_embed_text_json,
            text.encode("utf-8"),
        )

    def embed_text(self, text: str) -> list[float]:
        return list(json.loads(self.embed_text_json(text))["embedding"])

    def process_audio_json(
        self,
        pcm: Iterable[float],
        source_id: str,
        include_embedding: bool = True,
        retention: Retention | int | None = None,
    ) -> str:
        samples = array("f", pcm)
        raw = (ctypes.c_float * len(samples)).from_buffer(samples)
        options = _native_process_json_options(include_embedding, retention)
        return self._call_json(
            self._lib.cortext_process_audio_json_with_options,
            raw,
            len(samples),
            source_id.encode("utf-8"),
            ctypes.byref(options),
        )

    def process_audio_with_media_json(
        self,
        pcm: Iterable[float],
        source_id: str,
        media: Media | bytes | bytearray | memoryview | None = None,
        media_mimetype: str | None = None,
        include_embedding: bool = True,
        retention: Retention | int | None = None,
    ) -> str:
        samples = array("f", pcm)
        raw = (ctypes.c_float * len(samples)).from_buffer(samples)
        native_media, media_buffer, mimetype_buffer = _native_media(
            media, media_mimetype
        )
        options = _native_process_json_options(include_embedding, retention)
        _ = (media_buffer, mimetype_buffer)
        return self._call_json(
            self._lib.cortext_process_audio_with_media_json_with_options,
            raw,
            len(samples),
            source_id.encode("utf-8"),
            ctypes.byref(native_media),
            ctypes.byref(options),
        )

    def process_audio(
        self,
        pcm: Iterable[float],
        source_id: str,
        include_embedding: bool = True,
        retention: Retention | int | None = None,
    ) -> dict[str, Any]:
        return json.loads(
            self.process_audio_json(
                pcm, source_id, include_embedding, retention
            )
        )

    def process_audio_with_media(
        self,
        pcm: Iterable[float],
        source_id: str,
        media: Media | bytes | bytearray | memoryview | None = None,
        media_mimetype: str | None = None,
        include_embedding: bool = True,
        retention: Retention | int | None = None,
    ) -> dict[str, Any]:
        return json.loads(
            self.process_audio_with_media_json(
                pcm,
                source_id,
                media,
                media_mimetype,
                include_embedding,
                retention,
            )
        )

    def embed_audio_json(self, pcm: Iterable[float]) -> str:
        samples = array("f", pcm)
        raw = (ctypes.c_float * len(samples)).from_buffer(samples)
        return self._call_json(
            self._lib.cortext_embed_audio_json,
            raw,
            len(samples),
        )

    def embed_audio(self, pcm: Iterable[float]) -> list[float]:
        return list(json.loads(self.embed_audio_json(pcm))["embedding"])

    def process_image_json(
        self,
        data: bytes | bytearray | memoryview,
        width: int,
        height: int,
        channels: int,
        source_id: str,
        include_embedding: bool = True,
        retention: Retention | int | None = None,
    ) -> str:
        blob = bytes(data)
        raw = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        options = _native_process_json_options(include_embedding, retention)
        return self._call_json(
            self._lib.cortext_process_image_json_with_options,
            raw,
            width,
            height,
            channels,
            source_id.encode("utf-8"),
            ctypes.byref(options),
        )

    def process_image_with_media_json(
        self,
        data: bytes | bytearray | memoryview,
        width: int,
        height: int,
        channels: int,
        source_id: str,
        media: Media | bytes | bytearray | memoryview | None = None,
        media_mimetype: str | None = None,
        include_embedding: bool = True,
        retention: Retention | int | None = None,
    ) -> str:
        blob = bytes(data)
        raw = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        native_media, media_buffer, mimetype_buffer = _native_media(
            media, media_mimetype
        )
        options = _native_process_json_options(include_embedding, retention)
        _ = (media_buffer, mimetype_buffer)
        return self._call_json(
            self._lib.cortext_process_image_with_media_json_with_options,
            raw,
            width,
            height,
            channels,
            source_id.encode("utf-8"),
            ctypes.byref(native_media),
            ctypes.byref(options),
        )

    def process_image(
        self,
        data: bytes | bytearray | memoryview,
        width: int,
        height: int,
        channels: int,
        source_id: str,
        include_embedding: bool = True,
        retention: Retention | int | None = None,
    ) -> dict[str, Any]:
        return json.loads(
            self.process_image_json(
                data,
                width,
                height,
                channels,
                source_id,
                include_embedding,
                retention,
            )
        )

    def process_image_with_media(
        self,
        data: bytes | bytearray | memoryview,
        width: int,
        height: int,
        channels: int,
        source_id: str,
        media: Media | bytes | bytearray | memoryview | None = None,
        media_mimetype: str | None = None,
        include_embedding: bool = True,
        retention: Retention | int | None = None,
    ) -> dict[str, Any]:
        return json.loads(
            self.process_image_with_media_json(
                data,
                width,
                height,
                channels,
                source_id,
                media,
                media_mimetype,
                include_embedding,
                retention,
            )
        )

    def embed_image_json(
        self,
        data: bytes | bytearray | memoryview,
        width: int,
        height: int,
        channels: int,
    ) -> str:
        blob = bytes(data)
        raw = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        return self._call_json(
            self._lib.cortext_embed_image_json,
            raw,
            width,
            height,
            channels,
        )

    def embed_image(
        self,
        data: bytes | bytearray | memoryview,
        width: int,
        height: int,
        channels: int,
    ) -> list[float]:
        payload = self.embed_image_json(data, width, height, channels)
        return list(json.loads(payload)["embedding"])

    def consolidate_json(self) -> str:
        return self._call_json(self._lib.cortext_consolidate_json)

    def consolidate(self) -> dict[str, Any]:
        return json.loads(self.consolidate_json())

    def _call_json(self, fn: Any, *args: Any) -> str:
        result = fn(self._handle, *args)
        if not result:
            _raise_last_error(f"{fn.__name__} failed")
        try:
            return ctypes.string_at(result).decode("utf-8")
        finally:
            self._lib.cortext_string_free(result)
