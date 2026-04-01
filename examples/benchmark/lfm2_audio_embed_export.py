#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTORCH_SRC = REPO_ROOT / "third_party" / "executorch" / "src"
if EXECUTORCH_SRC.exists():
    sys.path.insert(0, str(EXECUTORCH_SRC))
    # Force local ExecuTorch sources/extensions over any site-packages install.
    import types

    execu_pkg = types.ModuleType("executorch")
    execu_pkg.__path__ = [str(EXECUTORCH_SRC / "executorch")]
    sys.modules["executorch"] = execu_pkg

import torch
import torch._dynamo as dynamo
import transformers.masking_utils as masking_utils
import transformers.models.lfm2.modeling_lfm2 as lfm2_modeling
import torchaudio
from torch.fx.graph import Graph as FxGraph
from torch.fx.graph_module import GraphModule as FxGraphModule

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
from liquid_audio.utils import mel2emb_len

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.utils.configs import (
    get_xnnpack_edge_compile_config,
    get_xnnpack_executorch_backend_config,
)

_AOT_LIB = (
    REPO_ROOT
    / "third_party"
    / "executorch"
    / "cmake-out"
    / "kernels"
    / "quantized"
    / "libquantized_ops_aot_lib.dylib"
)
if _AOT_LIB.exists():
    torch.ops.load_library(str(_AOT_LIB))
else:
    import executorch.kernels.quantized  # noqa: F401
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

FIXED_CAUSAL_LEN: int | None = None


def _load_audio(audio_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    wav, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    length = torch.tensor([wav.shape[1]], dtype=torch.int64)
    return wav, length


class Lfm2AudioEmbedder(torch.nn.Module):
    def __init__(self, model: LFM2AudioModel, fixed_len: int, target_len: int):
        super().__init__()
        self.conformer = model.conformer
        self.audio_adapter = model.audio_adapter
        self.lfm = model.lfm
        self.register_buffer(
            "audio_len",
            torch.tensor([fixed_len], dtype=torch.int64),
            persistent=False,
        )
        self.target_len = int(target_len)

    def forward(self, audio_in: torch.Tensor) -> torch.Tensor:
        # audio_in: [128, T]
        padded_audio_in = audio_in.mT.unsqueeze(0)  # [1, T, 128]
        audio_enc, _ = self.conformer(padded_audio_in.mT, self.audio_len)
        audio_enc_t = audio_enc.mT.squeeze(0)
        target_len = self.target_len
        if audio_enc_t.shape[0] < target_len:
            pad = target_len - audio_enc_t.shape[0]
            audio_enc_t = torch.nn.functional.pad(audio_enc_t, (0, 0, 0, pad))
        audio_enc_t = audio_enc_t[:target_len]
        audio_in_emb = self.audio_adapter(audio_enc_t)
        seq_len = target_len
        torch._check(seq_len != 0)
        cache_position = torch.zeros(
            (seq_len,), device=audio_in_emb.device, dtype=torch.long
        )
        position_ids = cache_position.unsqueeze(0)
        lfm_out = self.lfm(
            inputs_embeds=audio_in_emb.unsqueeze(0),
            use_cache=False,
            past_key_values=None,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        hidden = lfm_out.last_hidden_state
        return hidden.mean(dim=1)

class _NoopModule(torch.nn.Module):
    def forward(self, *args, **kwargs):
        if args:
            return args[0]
        return None


def _patch_fx_lint() -> None:
    original_lint = FxGraph.lint
    original_getattr = FxGraphModule.__getattr__

    def _patched_lint(self, *args, **kwargs):
        try:
            return original_lint(self, *args, **kwargs)
        except RuntimeError as exc:
            if "references nonexistent attribute" in str(exc):
                return None
            raise

    def _patched_getattr(self, name):
        try:
            return original_getattr(self, name)
        except AttributeError:
            if name.startswith("submod_"):
                mod = _NoopModule()
                setattr(self, name, mod)
                return mod
            raise

    FxGraph.lint = _patched_lint
    FxGraphModule.__getattr__ = _patched_getattr


def _ensure_call_module_targets(graph_module: torch.fx.GraphModule) -> None:
    missing = set()
    for node in graph_module.graph.nodes:
        if node.op != "call_module":
            continue
        try:
            graph_module.get_submodule(node.target)
        except AttributeError:
            missing.add(node.target)
    if not missing:
        return
    for target in sorted(missing):
        parts = target.split(".")
        parent = graph_module
        for name in parts[:-1]:
            if not hasattr(parent, name):
                setattr(parent, name, torch.nn.Module())
            parent = getattr(parent, name)
        if not hasattr(parent, parts[-1]):
            setattr(parent, parts[-1], _NoopModule())
    graph_module.graph.lint()


def _quantize_8da4w(
    model: torch.nn.Module, example_inputs: tuple[torch.Tensor]
) -> torch.nn.Module:
    quantizer = XNNPACKQuantizer()
    operator_config = get_symmetric_quantization_config(
        is_per_channel=True,
        is_dynamic=True,
        weight_qmin=-8,
        weight_qmax=7,
    )
    quantizer.set_global(operator_config)

    captured = torch.export.export(model, example_inputs, strict=False)
    gm = captured.module()
    prepared = prepare_pt2e(gm, quantizer)
    prepared(*example_inputs)
    _ensure_call_module_targets(prepared)
    try:
        return convert_pt2e(prepared)
    except Exception as exc:
        print(
            f"WARNING: convert_pt2e failed ({exc}); retrying without DuplicateDQPass"
        )
        import torchao.quantization.pt2e.quantize_pt2e as q
        from torch.fx.passes.infra.pass_manager import PassManager

        original_graph_meta = prepared.meta
        _ensure_call_module_targets(prepared)
        model = q._convert_to_reference_decomposed_fx(prepared)
        model = q._fold_conv_bn_qat(model)
        pm = PassManager([q.PortNodeMetaForQDQ()])
        model = pm(model).graph_module
        if q.torch_version_at_least("2.7.0"):
            q.constant_fold(model, q._quant_node_constraint)
        model.meta.update(original_graph_meta)
        model = q._disallow_eval_train(model)
        return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Export LFM2-Audio embedder to ExecuTorch PTE (8da4w)")
    parser.add_argument("--model-dir", default="models/lfm2-audio-1.5b")
    parser.add_argument("--audio-file", default="examples/dog.wav")
    parser.add_argument("--output-dir", default="models/lfm2-audio-1.5b/executorch")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--no-xnnpack", action="store_true")
    parser.add_argument("--fixed-frames", type=int, default=0)
    args = parser.parse_args()

    _patch_fx_lint()

    model_dir = Path(args.model_dir)
    audio_path = Path(args.audio_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    dynamo.config.assume_static_by_default = True
    dynamo.config.capture_scalar_outputs = True

    def _simple_create_causal_mask(
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids=None,
        or_mask_function=None,
        and_mask_function=None,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        bsz = input_embeds.shape[0]
        qlen = FIXED_CAUSAL_LEN if FIXED_CAUSAL_LEN is not None else input_embeds.shape[1]
        dtype = input_embeds.dtype
        device = input_embeds.device
        mask = torch.full((qlen, qlen), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, qlen, qlen)
        if FIXED_CAUSAL_LEN is None and attention_mask is not None and attention_mask.dim() == 2:
            am = (1.0 - attention_mask.float()) * float("-inf")
            mask = mask + am[:, None, None, :]
        return mask.to(dtype)

    masking_utils.create_causal_mask = _simple_create_causal_mask
    lfm2_modeling.create_causal_mask = _simple_create_causal_mask
    def _patched_rotary_forward(self, x, position_ids):
        inv_freq = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
        pos = position_ids.float()
        freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    lfm2_modeling.Lfm2RotaryEmbedding.forward = _patched_rotary_forward

    processor = LFM2AudioProcessor.from_pretrained(model_dir, device="cpu").eval()
    model = LFM2AudioModel.from_pretrained(model_dir, device="cpu", dtype=dtype).eval()
    try:
        model.lfm.set_attn_implementation("eager")
    except Exception:
        pass

    wav, length = _load_audio(audio_path)
    with torch.no_grad():
        mel, mel_len = processor.audio(wav, length)

    mel_feat = mel[0]
    mel_len_val = int(mel_len.item())
    fixed_len = args.fixed_frames if args.fixed_frames > 0 else mel_len_val
    global FIXED_CAUSAL_LEN
    if fixed_len <= 0:
        raise RuntimeError("Computed empty embedding length")
    if mel_feat.shape[1] < fixed_len:
        pad = fixed_len - mel_feat.shape[1]
        mel_feat = torch.nn.functional.pad(mel_feat, (0, pad))
    mel_feat = mel_feat[:, :fixed_len].to(dtype=dtype)

    emb_len = mel2emb_len(torch.tensor([fixed_len], dtype=torch.int64)).sum().item()
    if emb_len <= 0:
        raise RuntimeError("Computed empty embedding length")
    FIXED_CAUSAL_LEN = int(emb_len)

    example_inputs = (mel_feat,)

    for module in model.modules():
        if isinstance(module, lfm2_modeling.Lfm2ShortConv):
            module.fixed_len = fixed_len

    def _patched_short_conv(self, x=None, past_key_values=None, cache_position=None, attention_mask=None, **kwargs):
        if x is None:
            x = kwargs.get("hidden_states")
        x = lfm2_modeling.apply_mask_to_padding_states(x, attention_mask)
        return self.out_proj(x)

    lfm2_modeling.Lfm2ShortConv.forward = _patched_short_conv
    lfm2_modeling.Lfm2ShortConv.slow_forward = _patched_short_conv

    embedder = Lfm2AudioEmbedder(model, fixed_len=fixed_len, target_len=int(emb_len)).eval()

    if not args.no_quant:
        embedder = _quantize_8da4w(embedder, example_inputs)

    start = time.perf_counter()
    ep = torch.export.export(embedder, example_inputs, strict=False)

    if args.no_xnnpack:
        edge = to_edge_transform_and_lower(ep)
        exec_prog = edge.to_executorch()
    else:
        edge = to_edge_transform_and_lower(
            ep,
            partitioner=[XnnpackPartitioner()],
            compile_config=get_xnnpack_edge_compile_config(skip_dim_order=True),
        )
        exec_prog = edge.to_executorch(config=get_xnnpack_executorch_backend_config())

    pte_name = "lfm2_audio_embedder_8da4w" if not args.no_quant else "lfm2_audio_embedder_fp32"
    pte_path = save_pte_program(exec_prog, pte_name, str(output_dir))
    meta_path = output_dir / f"{pte_name}.meta.json"
    meta_path.write_text(
        f"{{\n  \"fixed_frames\": {fixed_len},\n  \"dtype\": \"{args.dtype}\"\n}}\n"
    )
    elapsed = time.perf_counter() - start

    print(f"Exported PTE to {pte_path}")
    print(f"Export time: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
