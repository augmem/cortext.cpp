#!/usr/bin/env python3
"""Benchmark-only AAIT reference runtime for native GGUF parity checks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def _tolist(t: torch.Tensor) -> list[float]:
    return [float(x) for x in t.detach().cpu().flatten().tolist()]


class AaitReferenceRuntime:
    def __init__(self, model_path: Path, max_length: int) -> None:
        self.tensors = load_file(str(model_path), device="cpu")
        self.max_length = max_length
        self.tokenizer = None
        self.tokenizer_error = ""
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("MongoDB/mdbr-leaf-ir")
        except Exception as exc:  # pragma: no cover - emitted in audit output
            self.tokenizer_error = str(exc)

    def weight(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def linear(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        return F.linear(x, self.weight(prefix + ".weight"), self.weight(prefix + ".bias"))

    def layer_norm(self, x: torch.Tensor, prefix: str, eps: float) -> torch.Tensor:
        return F.layer_norm(
            x,
            (self.weight(prefix + ".weight").numel(),),
            self.weight(prefix + ".weight"),
            self.weight(prefix + ".bias"),
            eps=eps,
        )

    def encode_text_with_hf_ids(self, text: str) -> tuple[torch.Tensor, list[int]]:
        if self.tokenizer is None:
            raise RuntimeError(f"HuggingFace tokenizer unavailable: {self.tokenizer_error}")
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        ids = [int(v) for v in encoded["input_ids"]]
        return self.encode_ids(ids), ids

    def encode_ids(self, ids: list[int]) -> torch.Tensor:
        if not ids:
            ids = [101, 102]
        ids = ids[: self.max_length]
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        token_types = torch.zeros_like(input_ids)
        hidden = (
            self.weight("text_encoder.0.model.embeddings.word_embeddings.weight")[input_ids]
            + self.weight("text_encoder.0.model.embeddings.position_embeddings.weight")[positions]
            + self.weight("text_encoder.0.model.embeddings.token_type_embeddings.weight")[
                token_types
            ]
        )
        hidden = self.layer_norm(hidden, "text_encoder.0.model.embeddings.LayerNorm", 1e-12)
        for layer in range(6):
            prefix = f"text_encoder.0.model.encoder.layer.{layer}"
            residual = hidden
            query = self.linear(hidden, prefix + ".attention.self.query")
            key = self.linear(hidden, prefix + ".attention.self.key")
            value = self.linear(hidden, prefix + ".attention.self.value")
            query = query.view(1, seq_len, 12, 32).transpose(1, 2)
            key = key.view(1, seq_len, 12, 32).transpose(1, 2)
            value = value.view(1, seq_len, 12, 32).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(32.0)
            probs = torch.softmax(scores, dim=-1)
            context = torch.matmul(probs, value).transpose(1, 2).reshape(1, seq_len, 384)
            attn_out = self.linear(context, prefix + ".attention.output.dense")
            hidden = self.layer_norm(
                residual + attn_out, prefix + ".attention.output.LayerNorm", 1e-12
            )
            residual = hidden
            intermediate = F.gelu(self.linear(hidden, prefix + ".intermediate.dense"))
            output = self.linear(intermediate, prefix + ".output.dense")
            hidden = self.layer_norm(residual + output, prefix + ".output.LayerNorm", 1e-12)
        pooled = hidden.mean(dim=1).squeeze(0)
        text_features = self.linear(pooled, "text_encoder.2.linear")
        semantic = self.projection_block(text_features, "text_projection")
        return F.normalize(semantic, p=2, dim=0)

    def projection_block(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        x = self.linear(x, prefix + ".expand.0")
        x = F.gelu(x)
        x = self.layer_norm(x, prefix + ".expand.2", 1e-5)
        for block in range(2):
            y = self.linear(x, f"{prefix}.residual_blocks.{block}.0")
            y = F.gelu(y)
            y = self.layer_norm(y, f"{prefix}.residual_blocks.{block}.2", 1e-5)
            x = x + y
        return self.linear(x, prefix + ".project")

    def encode_semantic(
        self, text: str, native_token_ids: list[int] | None, use_native_tokens: bool
    ) -> tuple[torch.Tensor, list[int], str]:
        if use_native_tokens:
            ids = native_token_ids if native_token_ids is not None else []
            return self.encode_ids(ids), ids, "native_token_reference"
        semantic, ids = self.encode_text_with_hf_ids(text)
        return semantic, ids, "hf_tokenizer_reference"

    def anchor_forward(
        self,
        semantic: torch.Tensor,
        recent: torch.Tensor,
        active: torch.Tensor,
        modality_id: int,
        source_id: int,
        time_delta: float,
    ) -> dict[str, Any]:
        time = torch.tensor([float(time_delta)], dtype=torch.float32)
        time = self.linear(time, "anchor_head.time_mlp.0")
        time = F.gelu(time)
        time = self.linear(time, "anchor_head.time_mlp.2")
        modality_table = self.weight("anchor_head.modality_embedding.weight")
        source_table = self.weight("anchor_head.source_embedding.weight")
        modality = modality_table[torch.tensor(max(0, min(int(modality_id), modality_table.shape[0] - 1)))]
        source = source_table[torch.tensor(max(0, min(int(source_id), source_table.shape[0] - 1)))]
        x = torch.cat([semantic, recent, active, modality, time, source], dim=0)
        h = self.linear(x, "anchor_head.input_mlp.0")
        h = self.layer_norm(h, "anchor_head.input_mlp.1", 1e-5)
        h = F.gelu(h)
        h = self.linear(h, "anchor_head.input_mlp.3")
        h = self.layer_norm(h, "anchor_head.input_mlp.4", 1e-5)
        fused_input = torch.cat([h, torch.zeros_like(h)], dim=0)
        fused = self.linear(fused_input, "anchor_head.fuse.0")
        fused = self.layer_norm(fused, "anchor_head.fuse.1", 1e-5)
        fused = F.gelu(fused)
        anchor_key = F.normalize(self.linear(fused, "anchor_head.anchor_proj"), p=2, dim=0)
        action_logits = self.linear(fused, "anchor_head.action_head")
        confidence = torch.sigmoid(self.linear(fused, "anchor_head.confidence_head"))[0]
        salience = torch.tanh(self.linear(fused, "anchor_head.salience_head"))[0]
        return {
            "semantic_vector": _tolist(semantic),
            "anchor_key": _tolist(anchor_key),
            "anchor_action_logits": _tolist(action_logits),
            "anchor_confidence": float(confidence.item()),
            "salience_delta": float(salience.item()),
            "has_bind_logits": False,
            "bind_logits": [],
            "reference_contract": "zero_candidate_anchor_context",
        }

    def candidate_states(
        self, candidate_semantic: list[list[float]], candidate_features: list[list[float]]
    ) -> torch.Tensor:
        count = min(len(candidate_semantic), len(candidate_features))
        if count == 0:
            return torch.empty((0, 1024), dtype=torch.float32)
        rows = []
        for i in range(count):
            semantic = torch.tensor(candidate_semantic[i], dtype=torch.float32)
            features = torch.tensor(candidate_features[i], dtype=torch.float32)
            rows.append(torch.cat([semantic, features], dim=0))
        x = torch.stack(rows, dim=0)
        x = self.linear(x, "anchor_head.candidate_state.0")
        x = self.layer_norm(x, "anchor_head.candidate_state.1", 1e-5)
        x = F.gelu(x)
        return self.linear(x, "anchor_head.candidate_state.3")

    def candidate_attention(self, query: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        if states.numel() == 0 or states.shape[0] == 0:
            return torch.zeros_like(query)
        hidden = query.numel()
        num_heads = 8
        head_dim = hidden // num_heads
        w = self.weight("anchor_head.cross_attn.in_proj_weight")
        b = self.weight("anchor_head.cross_attn.in_proj_bias")
        qkv_query = F.linear(query, w, b)
        q = qkv_query[:hidden].view(num_heads, head_dim)
        qkv_states = F.linear(states, w, b)
        k = qkv_states[:, hidden : 2 * hidden].view(states.shape[0], num_heads, head_dim)
        v = qkv_states[:, 2 * hidden : 3 * hidden].view(states.shape[0], num_heads, head_dim)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        scores = (k * q.unsqueeze(1)).sum(dim=-1) / math.sqrt(float(head_dim))
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * v).sum(dim=1).reshape(hidden)
        return self.linear(context, "anchor_head.cross_attn.out_proj")

    def bind_logits(self, fused: torch.Tensor, states: torch.Tensor) -> list[float]:
        query = self.linear(fused, "anchor_head.bind_query")
        logits = []
        if states.numel() != 0 and states.shape[0] > 0:
            scale = math.sqrt(float(query.numel()))
            logits = [float(v) for v in (states @ query / scale).detach().cpu().tolist()]
        abstain = self.linear(fused, "anchor_head.bind_abstain")
        logits.append(float(abstain.flatten()[0].item()) if abstain.numel() else 0.0)
        return logits

    def anchor_forward_candidate_contract(self, record: dict[str, Any]) -> dict[str, Any]:
        semantic = torch.tensor(record["current_semantic"], dtype=torch.float32)
        recent = torch.tensor(record["recent_context_vector"], dtype=torch.float32)
        active = torch.tensor(record["active_anchor_state"], dtype=torch.float32)
        candidate_semantic = record.get("candidate_track_semantic", [])
        candidate_features = record.get("candidate_features", [])
        time = torch.tensor([float(record.get("time_delta", 0.0))], dtype=torch.float32)
        time = self.linear(time, "anchor_head.time_mlp.0")
        time = F.gelu(time)
        time = self.linear(time, "anchor_head.time_mlp.2")
        modality_table = self.weight("anchor_head.modality_embedding.weight")
        source_table = self.weight("anchor_head.source_embedding.weight")
        modality_id = max(0, min(int(record.get("modality_id", 0)), modality_table.shape[0] - 1))
        source_id = max(0, min(int(record.get("source_id", 0)), source_table.shape[0] - 1))
        modality = modality_table[torch.tensor(modality_id)]
        source = source_table[torch.tensor(source_id)]
        x = torch.cat([semantic, recent, active, modality, time, source], dim=0)
        h = self.linear(x, "anchor_head.input_mlp.0")
        h = self.layer_norm(h, "anchor_head.input_mlp.1", 1e-5)
        h = F.gelu(h)
        h = self.linear(h, "anchor_head.input_mlp.3")
        h = self.layer_norm(h, "anchor_head.input_mlp.4", 1e-5)
        states = self.candidate_states(candidate_semantic, candidate_features)
        attended = self.candidate_attention(h, states)
        fused = self.linear(torch.cat([h, attended], dim=0), "anchor_head.fuse.0")
        fused = self.layer_norm(fused, "anchor_head.fuse.1", 1e-5)
        fused = F.gelu(fused)
        anchor_key = F.normalize(self.linear(fused, "anchor_head.anchor_proj"), p=2, dim=0)
        action_logits = self.linear(fused, "anchor_head.action_head")
        confidence = torch.sigmoid(self.linear(fused, "anchor_head.confidence_head"))[0]
        salience = torch.tanh(self.linear(fused, "anchor_head.salience_head"))[0]
        bind_logits = self.bind_logits(fused, states)
        return {
            "semantic_vector": _tolist(semantic),
            "anchor_key": _tolist(anchor_key),
            "anchor_action_logits": _tolist(action_logits),
            "anchor_confidence": float(confidence.item()),
            "salience_delta": float(salience.item()),
            "has_bind_logits": True,
            "bind_logits": bind_logits,
            "candidate_count": len(candidate_semantic),
            "reference_contract": "candidate_track_tensor_contract",
        }

    def run_record(self, record: dict[str, Any], use_native_tokens: bool) -> dict[str, Any]:
        native_tokens = record.get("native_token_ids", {})
        semantic, text_ids, mode = self.encode_semantic(
            record.get("text", ""), native_tokens.get("text"), use_native_tokens
        )
        if record.get("recent_context", ""):
            recent, recent_ids, _ = self.encode_semantic(
                record.get("recent_context", ""),
                native_tokens.get("recent_context"),
                use_native_tokens,
            )
        else:
            recent = semantic
            recent_ids = text_ids
        if record.get("active_context", ""):
            active, active_ids, _ = self.encode_semantic(
                record.get("active_context", ""),
                native_tokens.get("active_context"),
                use_native_tokens,
            )
        else:
            active = recent
            active_ids = recent_ids
        output = self.anchor_forward(
            semantic=semantic,
            recent=recent,
            active=active,
            modality_id=int(record.get("modality_id", 0)),
            source_id=int(record.get("source_id", 0)),
            time_delta=float(record.get("time_delta", 0.0)),
        )
        output.update(
            {
                "mode": mode,
                "token_ids": {
                    "text": text_ids,
                    "recent_context": recent_ids,
                    "active_context": active_ids,
                },
            }
        )
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    runtime = AaitReferenceRuntime(Path(args.model), args.max_length)
    records = []
    with torch.no_grad():
        for record in payload["records"]:
            entry: dict[str, Any] = {
                "id": record["id"],
                "kind": record.get("kind", ""),
                "expected_action": record.get("expected_action"),
                "native_token_ids": record.get("native_token_ids", {}),
            }
            if payload.get("mode") == "aait_candidate_contract_parity":
                entry["candidate_tensor_reference"] = runtime.anchor_forward_candidate_contract(
                    record
                )
                records.append(entry)
                continue
            if runtime.tokenizer is not None:
                entry["hf_tokenizer_reference"] = runtime.run_record(
                    record, use_native_tokens=False
                )
            else:
                entry["hf_tokenizer_error"] = runtime.tokenizer_error
            entry["native_token_reference"] = runtime.run_record(
                record, use_native_tokens=True
            )
            records.append(entry)
    out = {
        "status": "ok",
        "model_path": str(Path(args.model).resolve()),
        "max_length": args.max_length,
        "tokenizer_available": runtime.tokenizer is not None,
        "tokenizer_error": runtime.tokenizer_error,
        "records": records,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
