#!/usr/bin/env python3
import argparse
import gzip
import html
import json
import hashlib
import os
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tokenizers import Tokenizer

try:
    import regex as re
except Exception:
    re = None

try:
    import ftfy
except Exception:
    ftfy = None


def _try_import_tflite():
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter
    except Exception:
        try:
            import tensorflow as tf  # type: ignore
            return tf.lite.Interpreter
        except Exception:
            try:
                from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
                return Interpreter
            except Exception:
                return None


def _ensure_cache_dirs():
    root = Path(__file__).resolve().parents[2]
    cache_root = root / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    hf_cache = cache_root / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
    os.environ.setdefault("MTEB_CACHE", str(cache_root / "mteb"))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _truncate_embeddings(embs: np.ndarray, target_dim: int) -> np.ndarray:
    if target_dim <= 0:
        return embs
    if embs.size == 0:
        return embs
    if embs.ndim == 1:
        embs = np.expand_dims(embs, axis=0)
    if embs.shape[1] <= target_dim:
        return embs
    truncated = embs[:, :target_dim]
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return truncated / norms


@dataclass
class LatencyStats:
    batch_ms: List[float]
    samples: List[int]

    def record(self, elapsed_ms: float, batch_size: int) -> None:
        self.batch_ms.append(elapsed_ms)
        self.samples.append(batch_size)

    def summary(self) -> Dict[str, float]:
        if not self.batch_ms:
            return {
                "batches": 0,
                "samples": 0,
                "mean_batch_ms": 0.0,
                "p50_batch_ms": 0.0,
                "p95_batch_ms": 0.0,
                "mean_sample_ms": 0.0,
            }
        batch_ms = self.batch_ms
        p50 = statistics.median(batch_ms)
        p95 = statistics.quantiles(batch_ms, n=100)[94] if len(batch_ms) >= 20 else max(batch_ms)
        total_samples = sum(self.samples)
        mean_sample_ms = sum(batch_ms) / total_samples if total_samples > 0 else 0.0
        return {
            "batches": len(batch_ms),
            "samples": total_samples,
            "mean_batch_ms": sum(batch_ms) / len(batch_ms),
            "p50_batch_ms": p50,
            "p95_batch_ms": p95,
            "mean_sample_ms": mean_sample_ms,
        }


def _summarize_samples(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {
            "samples": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
        }
    p50 = statistics.median(samples)
    p95 = statistics.quantiles(samples, n=100)[94] if len(samples) >= 20 else max(samples)
    return {
        "samples": len(samples),
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": p50,
        "p95_ms": p95,
    }


class BaseEmbedder:
    def __init__(self, name: str, model_name: str, max_length: int, batch_size: int, target_dim: int):
        self.name = name
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.target_dim = target_dim
        self.latency = LatencyStats([], [])

    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        raise NotImplementedError


class MtebEncoderAdapter:
    def __init__(self, embedder: BaseEmbedder, model_name: str):
        from mteb.models.model_meta import ModelMeta  # type: ignore

        self.embedder = embedder
        self.mteb_model_meta = ModelMeta.from_hub(model_name, compute_metadata=False)

    def _extract_texts(self, batch) -> List[str]:
        if isinstance(batch, dict):
            if "text" in batch:
                return list(batch["text"])
            return []
        if isinstance(batch, list):
            if not batch:
                return []
            if isinstance(batch[0], dict) and "text" in batch[0]:
                return [row["text"] for row in batch]
            return [str(x) for x in batch]
        if isinstance(batch, str):
            return [batch]
        return []

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs):
        try:
            from torch.utils.data import DataLoader  # type: ignore
        except Exception:
            DataLoader = ()

        batch_size = kwargs.get("batch_size", None)

        if isinstance(inputs, DataLoader):
            all_embs: List[np.ndarray] = []
            for batch in inputs:
                texts = self._extract_texts(batch)
                if not texts:
                    continue
                embs = self.embedder.encode(texts, batch_size=len(texts))
                all_embs.append(embs)
            if not all_embs:
                return np.zeros((0, 0), dtype=np.float32)
            return np.vstack(all_embs)

        if isinstance(inputs, list):
            return self.embedder.encode(inputs, batch_size=batch_size or self.embedder.batch_size)

        return self.embedder.encode([str(inputs)], batch_size=1)

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        e1 = embeddings1 / (np.linalg.norm(embeddings1, axis=-1, keepdims=True) + 1e-12)
        e2 = embeddings2 / (np.linalg.norm(embeddings2, axis=-1, keepdims=True) + 1e-12)
        return np.matmul(e1, e2.T)

    def similarity_pairwise(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        e1 = embeddings1 / (np.linalg.norm(embeddings1, axis=-1, keepdims=True) + 1e-12)
        e2 = embeddings2 / (np.linalg.norm(embeddings2, axis=-1, keepdims=True) + 1e-12)
        return np.sum(e1 * e2, axis=-1)


class TokenizerAdapter:
    def __init__(self, tokenizer_path: Path, max_length: int):
        self.max_length = max_length
        self.kind = tokenizer_path.suffix.lower()
        if self.kind == ".json":
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            pad_id = self.tokenizer.token_to_id("<pad>")
            if pad_id is None:
                pad_id = 0
            self.tokenizer.enable_truncation(max_length=max_length)
            self.tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
            self.pad_id = pad_id
            self.bos_id = self.tokenizer.token_to_id("<bos>")
            self.eos_id = self.tokenizer.token_to_id("<eos>")
        elif self.kind == ".model":
            try:
                import sentencepiece as spm  # type: ignore
            except Exception as exc:
                raise RuntimeError("sentencepiece is required for .model tokenizers") from exc
            sp = spm.SentencePieceProcessor()
            sp.Load(str(tokenizer_path))
            self.sp = sp
            self.pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0
            self.bos_id = sp.bos_id() if sp.bos_id() >= 0 else None
            self.eos_id = sp.eos_id() if sp.eos_id() >= 0 else None
        else:
            raise RuntimeError(f"Unsupported tokenizer file: {tokenizer_path}")

    def encode_batch(self, texts: List[str]):
        if self.kind == ".json":
            enc = self.tokenizer.encode_batch(texts)
            ids_list = [e.ids for e in enc]
            masks = [e.attention_mask for e in enc]
            return ids_list, masks
        # sentencepiece
        ids_list = []
        masks = []
        for text in texts:
            ids = self.sp.EncodeAsIds(text)
            if self.bos_id is not None and self.eos_id is not None and self.max_length >= 2:
                max_len = self.max_length - 2
                if len(ids) > max_len:
                    ids = ids[:max_len]
                ids = [self.bos_id] + ids + [self.eos_id]
            else:
                if len(ids) > self.max_length:
                    ids = ids[:self.max_length]
            ids_list.append(ids)
            masks.append([1] * len(ids))
        return ids_list, masks

    def set_max_length(self, max_length: int) -> None:
        self.max_length = max_length
        if self.kind == ".json":
            self.tokenizer.enable_truncation(max_length=max_length)
            self.tokenizer.enable_padding(pad_id=self.pad_id, pad_token="<pad>")

    def pad(self, ids_list: List[List[int]], masks: List[List[int]]):
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(ids_list, masks):
            if len(ids) < self.max_length:
                pad_len = self.max_length - len(ids)
                ids = ids + [self.pad_id] * pad_len
                mask = mask + [0] * pad_len
            else:
                ids = ids[: self.max_length]
                mask = mask[: self.max_length]
            padded_ids.append(ids)
            padded_masks.append(mask)
        return padded_ids, padded_masks


class OnnxEmbedder(BaseEmbedder):
    def __init__(self, name: str, model_name: str, model_path: Path, tokenizer_path: Path, max_length: int, batch_size: int, target_dim: int):
        super().__init__(name, model_name, max_length, batch_size, target_dim)
        import onnxruntime as ort  # local import to avoid hard dependency

        self.tokenizer = TokenizerAdapter(tokenizer_path, max_length)

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def _encode_batch(self, batch: List[str]) -> np.ndarray:
        ids_list, masks = self.tokenizer.encode_batch(batch)
        ids_list, masks = self.tokenizer.pad(ids_list, masks)
        input_ids = np.array(ids_list, dtype=np.int64)
        attention_mask = np.array(masks, dtype=np.int64)
        feeds = {}
        if "input_ids" in self.input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = attention_mask
        outputs = self.session.run(["sentence_embedding"], feeds)
        return outputs[0]

    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        all_embs: List[np.ndarray] = []
        for sentence in sentences:
            start = time.perf_counter()
            embs = self._encode_batch([sentence])
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.latency.record(elapsed_ms, 1)
            all_embs.append(embs)
        if not all_embs:
            return np.zeros((0, 0), dtype=np.float32)
        return _truncate_embeddings(np.vstack(all_embs), self.target_dim)


class ImageBindOnnxEmbedder(BaseEmbedder):
    def __init__(self, name: str, model_name: str, model_path: Path, tokenizer_name: str, max_length: int, batch_size: int, target_dim: int):
        super().__init__(name, model_name, max_length, batch_size, target_dim)
        import onnxruntime as ort  # local import
        from transformers import CLIPTokenizerFast

        self.tokenizer = CLIPTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = 77  # ImageBind text encoder expects 77 tokens
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_name = inputs[0].name if inputs else "text_tokens"
        self.output_name = outputs[0].name if outputs else "text_embedding"

    def _encode_single(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        outputs = self.session.run([self.output_name], {self.input_name: input_ids})
        return outputs[0]

    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        all_embs: List[np.ndarray] = []
        for sentence in sentences:
            start = time.perf_counter()
            embs = self._encode_single(sentence)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.latency.record(elapsed_ms, 1)
            all_embs.append(embs)
        if not all_embs:
            return np.zeros((0, 0), dtype=np.float32)
        return _truncate_embeddings(np.vstack(all_embs), self.target_dim)


class ImageBindBPETokenizer:
    def __init__(self, bpe_path: Path, context_length: int = 77):
        if re is None:
            raise RuntimeError("regex is required for ImageBind BPE tokenizer")
        if ftfy is None:
            raise RuntimeError("ftfy is required for ImageBind BPE tokenizer (pip install ftfy)")

        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with gzip.open(bpe_path, "rb") as fh:
            merges: List[str] = fh.read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges_pairs: List[Tuple[str, ...]] = [tuple(merge.split()) for merge in merges]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges_pairs:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges_pairs, range(len(merges_pairs))))
        self.cache: Dict[str, str] = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.context_length = context_length

    def _bytes_to_unicode(self) -> Dict[int, str]:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _basic_clean(self, text: str) -> str:
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def _whitespace_clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = self._get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        word_str = " ".join(word)
        self.cache[token] = word_str
        return word_str

    def encode(self, text: str) -> List[int]:
        text = self._whitespace_clean(self._basic_clean(text))
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token]
                for bpe_token in self._bpe(token).split(" ")
            )

        start_token = self.encoder["<|startoftext|>"]
        end_token = self.encoder["<|endoftext|>"]
        tokens = [start_token] + bpe_tokens + [end_token]
        if len(tokens) > self.context_length:
            tokens = tokens[: self.context_length]
            tokens[-1] = end_token
        while len(tokens) < self.context_length:
            tokens.append(0)
        return tokens


class ImageBindBpeOnnxEmbedder(BaseEmbedder):
    def __init__(self, name: str, model_name: str, model_path: Path, bpe_path: Path, max_length: int, batch_size: int, target_dim: int):
        super().__init__(name, model_name, max_length, batch_size, target_dim)
        import onnxruntime as ort  # local import

        self.tokenizer = ImageBindBPETokenizer(bpe_path=bpe_path, context_length=77)
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_name = inputs[0].name if inputs else "text_tokens"
        self.output_name = outputs[0].name if outputs else "text_embedding"

    def _encode_single(self, text: str) -> np.ndarray:
        tokens = np.array([self.tokenizer.encode(text)], dtype=np.int64)
        outputs = self.session.run([self.output_name], {self.input_name: tokens})
        return outputs[0]

    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        all_embs: List[np.ndarray] = []
        for sentence in sentences:
            start = time.perf_counter()
            embs = self._encode_single(sentence)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.latency.record(elapsed_ms, 1)
            all_embs.append(embs)
        if not all_embs:
            return np.zeros((0, 0), dtype=np.float32)
        return _truncate_embeddings(np.vstack(all_embs), self.target_dim)


class TfliteEmbedder(BaseEmbedder):
    def __init__(self, name: str, model_name: str, model_path: Path, tokenizer_path: Path, max_length: int, batch_size: int, target_dim: int):
        super().__init__(name, model_name, max_length, batch_size, target_dim)
        Interpreter = _try_import_tflite()
        if Interpreter is None:
            raise RuntimeError("tflite runtime not available (install tflite-runtime or tensorflow)")

        self.interpreter = Interpreter(model_path=str(model_path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input_list = self.interpreter.get_input_details()
        self.input_details = {d["name"]: d for d in self.input_list}
        self.output_details = self.interpreter.get_output_details()
        self.primary_input = self.input_list[0] if self.input_list else None
        self.expected_length = self._resolve_expected_length()

        tokenizer_max = self.expected_length or max_length
        self.tokenizer = TokenizerAdapter(tokenizer_path, tokenizer_max)
        if self.expected_length:
            self.tokenizer.set_max_length(self.expected_length)
            self.max_length = self.expected_length

    def _resolve_expected_length(self) -> Optional[int]:
        if not self.primary_input:
            return None
        shape = self.primary_input.get("shape", None)
        if shape is None:
            return None
        if len(shape) >= 2 and shape[1] > 0:
            return int(shape[1])
        if len(shape) == 1 and shape[0] > 0:
            return int(shape[0])
        return None

    def _encode_batch(self, batch: List[str]) -> np.ndarray:
        ids_list, masks = self.tokenizer.encode_batch(batch)
        ids_list, masks = self.tokenizer.pad(ids_list, masks)
        input_ids = np.array(ids_list, dtype=np.int32)
        attention_mask = np.array(masks, dtype=np.int32)

        if "input_ids" in self.input_details:
            detail = self.input_details["input_ids"]
            self.interpreter.set_tensor(detail["index"], input_ids.astype(detail["dtype"]))
        elif self.primary_input is not None:
            detail = self.primary_input
            self.interpreter.set_tensor(detail["index"], input_ids.astype(detail["dtype"]))

        for name, detail in self.input_details.items():
            if name == "attention_mask" or "mask" in name:
                self.interpreter.set_tensor(detail["index"], attention_mask.astype(detail["dtype"]))
        self.interpreter.invoke()
        # Prefer sentence_embedding if present
        outputs = []
        for od in self.output_details:
            outputs.append(self.interpreter.get_tensor(od["index"]))
        if len(outputs) == 1:
            return outputs[0]
        for od, out in zip(self.output_details, outputs):
            if "sentence_embedding" in od.get("name", ""):
                return out
        return outputs[-1]

    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        all_embs: List[np.ndarray] = []
        for sentence in sentences:
            start = time.perf_counter()
            embs = self._encode_batch([sentence])
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.latency.record(elapsed_ms, 1)
            all_embs.append(embs)
        if not all_embs:
            return np.zeros((0, 0), dtype=np.float32)
        return _truncate_embeddings(np.vstack(all_embs), self.target_dim)


class AudioPipelineEmbedder(BaseEmbedder):
    def __init__(
        self,
        name: str,
        model_name: str,
        binary_path: Path,
        config_path: Path,
        mode: str,
        max_length: int,
        batch_size: int,
        target_dim: int,
        keep_tmp: bool,
        ignore_tts: bool,
        cache_dir: Optional[Path],
    ):
        super().__init__(name, model_name, max_length, batch_size, target_dim)
        self.binary_path = binary_path
        self.config_path = config_path
        self.mode = mode
        self.keep_tmp = keep_tmp
        self.ignore_tts = ignore_tts
        self.cache_dir = cache_dir
        self.component_latency: Dict[str, List[float]] = {
            "tts_ms": [],
            "asr_ms": [],
            "embed_ms": [],
            "total_ms": [],
        }

        if not self.binary_path.exists():
            raise RuntimeError(f"Audio pipeline binary not found: {self.binary_path}")
        if not self.config_path.exists():
            raise RuntimeError(f"Audio pipeline config not found: {self.config_path}")
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, batch: List[str]) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.config_path.read_bytes())
        hasher.update(b"\0")
        for text in batch:
            hasher.update(text.encode("utf-8"))
            hasher.update(b"\0")
        return hasher.hexdigest()

    def _run_batch(self, batch: List[str], id_offset: int) -> List[np.ndarray]:
        if not batch:
            return []
        cache_path = None
        effective_mode = self.mode
        if self.cache_dir:
            effective_mode = "both"
            cache_path = self.cache_dir / f"{self._cache_key(batch)}.jsonl"
        if self.keep_tmp:
            tmp_dir = Path(tempfile.mkdtemp(prefix="cortext_audio_bench_"))
            cleanup = False
        else:
            tmp_dir = Path(tempfile.mkdtemp(prefix="cortext_audio_bench_"))
            cleanup = True

        try:
            input_path = tmp_dir / "input.jsonl"
            output_path = tmp_dir / "output.jsonl"
            ids = []
            with input_path.open("w") as fh:
                for i, text in enumerate(batch):
                    row_id = str(id_offset + i)
                    ids.append(row_id)
                    fh.write(json.dumps({"id": row_id, "text": text}) + "\n")

            cmd = [
                str(self.binary_path),
                "--input",
                str(input_path),
                "--config",
                str(self.config_path),
                "--output",
                str(output_path),
                "--mode",
                effective_mode,
            ]
            if cache_path and cache_path.exists():
                output_path = cache_path
            else:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Audio pipeline failed (code {proc.returncode}).\n"
                        f"stdout:\n{proc.stdout}\n"
                        f"stderr:\n{proc.stderr}"
                    )
                if cache_path:
                    cache_path.write_text(output_path.read_text())

            rows: Dict[str, Dict] = {}
            with output_path.open("r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rid = str(row.get("id", ""))
                    if not rid:
                        continue
                    rows[rid] = row

            embeddings: List[np.ndarray] = []
            for rid in ids:
                row = rows.get(rid)
                if row is None:
                    raise RuntimeError(f"Missing output row for id={rid}")
                tts_ms = float(row.get("tts_ms", 0.0))
                if self.mode == "gemma":
                    asr_ms = float(row.get("asr_ms", 0.0))
                    embed_ms = float(row.get("gemma_ms", 0.0))
                    emb_key = "embedding_gemma"
                else:
                    asr_ms = 0.0
                    embed_ms = float(row.get("imagebind_ms", 0.0))
                    emb_key = "embedding_imagebind"
                total_ms = asr_ms + embed_ms
                if not self.ignore_tts:
                    total_ms += tts_ms

                if not self.ignore_tts:
                    self.component_latency["tts_ms"].append(tts_ms)
                if self.mode == "gemma":
                    self.component_latency["asr_ms"].append(asr_ms)
                self.component_latency["embed_ms"].append(embed_ms)
                self.component_latency["total_ms"].append(total_ms)
                self.latency.record(total_ms, 1)

                emb = np.array(row.get(emb_key, []), dtype=np.float32)
                if emb.size == 0:
                    raise RuntimeError(f"Empty embedding for id={rid}")
                emb = _truncate_embeddings(emb, self.target_dim)
                if emb.ndim == 2:
                    emb = emb[0]
                embeddings.append(emb)

            return embeddings
        finally:
            if cleanup:
                for path in tmp_dir.glob("*"):
                    try:
                        path.unlink()
                    except Exception:
                        pass
                try:
                    tmp_dir.rmdir()
                except Exception:
                    pass

    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        if not sentences:
            return np.zeros((0, 0), dtype=np.float32)
        batch_size = kwargs.get("batch_size", None) or self.batch_size
        if batch_size <= 0:
            batch_size = len(sentences)

        all_embs: List[np.ndarray] = []
        offset = 0
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            batch_embs = self._run_batch(batch, offset)
            all_embs.extend(batch_embs)
            offset += len(batch)

        return np.vstack(all_embs)

    def component_latency_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {
            "tts_ms": _summarize_samples(self.component_latency["tts_ms"]),
            "embed_ms": _summarize_samples(self.component_latency["embed_ms"]),
            "total_ms": _summarize_samples(self.component_latency["total_ms"]),
        }
        if self.mode == "gemma":
            summary["asr_ms"] = _summarize_samples(self.component_latency["asr_ms"])
        return summary


def _resolve_default_paths() -> Dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    onnx_tokenizer = root / "models" / "embeddinggemma-300m-onnx" / "tokenizer.json"
    litert_sentencepiece = root / "models" / "embeddinggemma-300m-litert" / "sentencepiece.model"
    # sentencepiece + tf.lite model load can deadlock; default to tokenizer.json unless overridden
    tflite_tokenizer = onnx_tokenizer if onnx_tokenizer.exists() else litert_sentencepiece
    imagebind_bpe_candidates = [
        root / "models" / "imagebind" / "bpe" / "bpe_simple_vocab_16e6.txt.gz",
        root / "models" / "imagebind" / "bpe_simple_vocab_16e6.txt.gz",
        root.parent / "ImageBind" / "imagebind" / "bpe" / "bpe_simple_vocab_16e6.txt.gz",
    ]
    imagebind_bpe = None
    if ftfy is not None and re is not None:
        imagebind_bpe = next((p for p in imagebind_bpe_candidates if p.exists()), None)
    audio_bin_candidates = [
        root / "build" / "examples" / "benchmark" / "cortext_audio_pipeline_bench",
        root / "build-release" / "examples" / "benchmark" / "cortext_audio_pipeline_bench",
        root / "build-no-oga" / "examples" / "benchmark" / "cortext_audio_pipeline_bench",
        root / "build-ort" / "examples" / "benchmark" / "cortext_audio_pipeline_bench",
    ]
    audio_binary = next((p for p in audio_bin_candidates if p.exists()), audio_bin_candidates[0])
    return {
        "onnx_dir": root / "models" / "embeddinggemma-300m-onnx",
        "tflite_path": root / "models" / "embeddinggemma-300m-litert" / "embeddinggemma-300M_seq256_mixed-precision.tflite",
        "tflite_tokenizer": tflite_tokenizer,
        "imagebind_dir": root / "models" / "imagebind",
        "imagebind_bpe": imagebind_bpe,
        "audio_binary": audio_binary,
    }


def _build_embedder(kind: str, args) -> BaseEmbedder:
    defaults = _resolve_default_paths()
    if kind == "onnx-q4":
        return OnnxEmbedder(
            name="embeddinggemma-onnx-q4",
            model_name="onnx-community/embeddinggemma-300m-ONNX",
            model_path=Path(args.onnx_dir) / "onnx" / "model_q4.onnx",
            tokenizer_path=Path(args.onnx_dir) / "tokenizer.json",
            max_length=args.max_length,
            batch_size=args.batch_size,
            target_dim=args.embed_dim,
        )
    if kind == "onnx-q8":
        return OnnxEmbedder(
            name="embeddinggemma-onnx-q8",
            model_name="onnx-community/embeddinggemma-300m-ONNX",
            model_path=Path(args.onnx_dir) / "onnx" / "model_quantized.onnx",
            tokenizer_path=Path(args.onnx_dir) / "tokenizer.json",
            max_length=args.max_length,
            batch_size=args.batch_size,
            target_dim=args.embed_dim,
        )
    if kind == "litert":
        return TfliteEmbedder(
            name="embeddinggemma-litert",
            model_name="litert-community/embeddinggemma-300m",
            model_path=Path(args.tflite_path),
            tokenizer_path=Path(args.tflite_tokenizer),
            max_length=args.max_length,
            batch_size=args.batch_size,
            target_dim=args.embed_dim,
        )
    if kind in ("imagebind", "imagebind-int8", "imagebind-fp"):
        model_dir = Path(args.imagebind_dir)
        if kind == "imagebind-int8":
            model_path = model_dir / "text_encoder_int8.onnx"
        elif kind == "imagebind-fp":
            model_path = model_dir / "text_encoder.onnx"
        else:
            model_path = model_dir / "text_encoder_int8.onnx"
            if not model_path.exists():
                model_path = model_dir / "text_encoder.onnx"
        bpe_path = Path(args.imagebind_bpe) if args.imagebind_bpe else None
        if bpe_path and bpe_path.exists() and ftfy is not None and re is not None:
            return ImageBindBpeOnnxEmbedder(
                name=f"imagebind-{model_path.stem}-bpe",
                model_name="facebook/imagebind",
                model_path=model_path,
                bpe_path=bpe_path,
                max_length=77,
                batch_size=1,
                target_dim=args.embed_dim,
            )
        return ImageBindOnnxEmbedder(
            name=f"imagebind-{model_path.stem}",
            model_name="facebook/imagebind",
            model_path=model_path,
            tokenizer_name=args.imagebind_tokenizer,
            max_length=77,
            batch_size=1,
            target_dim=args.embed_dim,
        )
    if kind == "audio-gemma":
        if not args.audio_config:
            raise RuntimeError("--audio-config is required for audio-gemma")
        return AudioPipelineEmbedder(
            name="audio-gemma",
            model_name="litert-community/embeddinggemma-300m",
            binary_path=Path(args.audio_binary),
            config_path=Path(args.audio_config),
            mode="gemma",
            max_length=args.max_length,
            batch_size=args.batch_size,
            target_dim=args.embed_dim,
            keep_tmp=args.audio_keep_tmp,
            ignore_tts=args.audio_ignore_tts,
            cache_dir=Path(args.audio_cache_dir) if args.audio_cache_dir else None,
        )
    if kind == "audio-imagebind":
        if not args.audio_config:
            raise RuntimeError("--audio-config is required for audio-imagebind")
        return AudioPipelineEmbedder(
            name="audio-imagebind",
            model_name="facebook/imagebind",
            binary_path=Path(args.audio_binary),
            config_path=Path(args.audio_config),
            mode="imagebind",
            max_length=args.max_length,
            batch_size=args.batch_size,
            target_dim=args.embed_dim,
            keep_tmp=args.audio_keep_tmp,
            ignore_tts=args.audio_ignore_tts,
            cache_dir=Path(args.audio_cache_dir) if args.audio_cache_dir else None,
        )
    raise ValueError(f"Unknown embedder kind: {kind}")


def _apply_limit_to_task(task, limit: int):
    try:
        from mteb.abstasks import AbsTaskRetrieval  # type: ignore
    except Exception:
        return

    task.load_data()
    if not isinstance(task, AbsTaskRetrieval):
        return

    for subset, splits in task.dataset.items():
        for split, data in splits.items():
            queries = data["queries"]
            corpus = data["corpus"]
            relevant_docs = data["relevant_docs"]

            if len(queries) > limit:
                queries = queries.select(range(limit))
            query_ids = set(queries["id"])
            relevant_docs = {qid: rels for qid, rels in relevant_docs.items() if qid in query_ids}

            if len(corpus) > limit:
                corpus = corpus.select(range(limit))

            data["queries"] = queries
            data["corpus"] = corpus
            data["relevant_docs"] = relevant_docs


def _run_mteb(embedder: BaseEmbedder, tasks: List[str], output_dir: Path, batch_size: int, limit: Optional[int]):
    import mteb  # type: ignore
    from mteb.cache import ResultCache  # type: ignore

    adapter = MtebEncoderAdapter(embedder, embedder.model_name)

    task_objs = [mteb.get_task(t) for t in tasks]
    if limit is not None:
        for task in task_objs:
            _apply_limit_to_task(task, limit)

    cache = ResultCache(cache_path=os.environ.get("MTEB_CACHE"))
    results = mteb.evaluate(
        adapter,
        task_objs,
        encode_kwargs={"batch_size": batch_size, "show_progress_bar": False},
        cache=cache,
        overwrite_strategy="always",
        prediction_folder=output_dir,
    )
    return results


def _summarize_results(model_result) -> Dict[str, float]:
    metrics = []
    try:
        for task_result in model_result.task_results:
            scores = task_result.scores or {}
            for split, vals in scores.items():
                if isinstance(vals, list):
                    for entry in vals:
                        if "main_score" in entry:
                            metrics.append(entry["main_score"])
                        elif "ndcg_at_10" in entry:
                            metrics.append(entry["ndcg_at_10"])
                        elif "map" in entry:
                            metrics.append(entry["map"])
                elif isinstance(vals, dict):
                    if "main_score" in vals:
                        metrics.append(vals["main_score"])
                    elif "ndcg_at_10" in vals:
                        metrics.append(vals["ndcg_at_10"])
                    elif "map" in vals:
                        metrics.append(vals["map"])
    except Exception:
        return {"mean_score": 0.0}
    if not metrics:
        return {"mean_score": 0.0}
    return {"mean_score": sum(metrics) / len(metrics)}


def main() -> None:
    _ensure_cache_dirs()
    defaults = _resolve_default_paths()
    parser = argparse.ArgumentParser(description="MTEB embedding benchmark (text + audio pipeline)")
    parser.add_argument("--onnx-dir", default=str(defaults["onnx_dir"]))
    parser.add_argument("--tflite-path", default=str(defaults["tflite_path"]))
    parser.add_argument("--tflite-tokenizer", default=str(defaults["tflite_tokenizer"]))
    parser.add_argument("--imagebind-dir", default=str(defaults["imagebind_dir"]))
    parser.add_argument("--imagebind-tokenizer", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--imagebind-bpe",
        default=str(defaults["imagebind_bpe"]) if defaults["imagebind_bpe"] else "",
        help="Path to ImageBind BPE vocab (bpe_simple_vocab_16e6.txt.gz). If set, uses local BPE tokenizer.",
    )
    parser.add_argument(
        "--audio-binary",
        default=str(defaults["audio_binary"]),
        help="Path to cortext_audio_pipeline_bench binary.",
    )
    parser.add_argument(
        "--audio-config",
        default="",
        help="Path to audio pipeline config JSON (required for audio-gemma/audio-imagebind).",
    )
    parser.add_argument(
        "--audio-keep-tmp",
        action="store_true",
        help="Keep temporary JSONL files from audio pipeline runs.",
    )
    parser.add_argument(
        "--audio-ignore-tts",
        action="store_true",
        help="Exclude TTS latency from audio pipeline latency stats.",
    )
    parser.add_argument(
        "--audio-cache-dir",
        default="",
        help="Cache audio pipeline JSONL outputs to reuse across models (runs in 'both' mode).",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=0, help="Truncate embeddings to N dims (0 = no truncation).")
    parser.add_argument("--tasks", default="ArguAna", help="Comma-separated MTEB task list")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per split")
    parser.add_argument("--output", default="examples/benchmark/mteb_results.json")
    parser.add_argument(
        "--models",
        default="onnx-q4,onnx-q8,litert",
        help=(
            "Comma-separated model list: onnx-q4, onnx-q8, litert, imagebind, "
            "imagebind-int8, imagebind-fp, audio-gemma, audio-imagebind"
        ),
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    results_summary = {}
    for model_kind in models:
        print(f"\n=== Running {model_kind} ===")
        try:
            embedder = _build_embedder(model_kind, args)
        except Exception as exc:
            print(f"Skipping {model_kind}: {exc}")
            continue

        output_dir = Path(args.output).with_suffix("") / model_kind
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = _run_mteb(embedder, tasks, output_dir, args.batch_size, args.limit)
        except Exception as exc:
            print(f"MTEB run failed for {model_kind}: {exc}")
            continue

        latency = embedder.latency.summary()
        score = _summarize_results(results)
        entry = {
            "latency": latency,
            "score": score,
            "tasks": tasks,
            "output_dir": str(output_dir),
            "embed_dim": embedder.target_dim if embedder.target_dim > 0 else None,
        }
        if hasattr(embedder, "component_latency_summary"):
            entry["component_latency"] = embedder.component_latency_summary()
        results_summary[model_kind] = entry

        print(f"{model_kind} latency: {latency}")
        if "component_latency" in entry:
            print(f"{model_kind} component latency: {entry['component_latency']}")
        print(f"{model_kind} score: {score}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
