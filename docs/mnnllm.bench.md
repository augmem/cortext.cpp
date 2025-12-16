# Plan: Benchmark Edge Inference Engines (MNN-LLM First)

## Objective
Benchmark MNN-LLM vs ONNX Runtime for Gemma 3n single-thread CPU inference, covering both text generation and full audio-to-text (ASR) pipeline.

## Scope
- **Single-thread CPU only** (no multi-threading)
- **Engines**: MNN-LLM first, then ExecuTorch, NCNN, llama.cpp
- **Modalities**: Text decoder + Audio encoder (full ASR pipeline)

## Phase 1: Setup MNN

### 1.1 Clone and Build MNN
```bash
git clone https://github.com/alibaba/MNN.git third_party/mnn
cd third_party/mnn && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_LLM=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_BUILD_CONVERTER=ON
make -j
```

### 1.2 Install Python Bindings
```bash
pip install MNN pymnn
```

## Phase 2: Convert Models (ONNX → MNN)

```bash
mkdir -p models/gemma-3n/mnn

# Decoder (q4)
./third_party/mnn/build/MNNConvert -f ONNX \
  --modelFile models/gemma-3n/onnx/decoder_model_merged_q4.onnx \
  --MNNModel models/gemma-3n/mnn/decoder_model_merged_q4.mnn \
  --weightQuantBits=4

# Embed tokens
./third_party/mnn/build/MNNConvert -f ONNX \
  --modelFile models/gemma-3n/onnx/embed_tokens_int8.onnx \
  --MNNModel models/gemma-3n/mnn/embed_tokens_int8.mnn

# Audio encoder (q4)
./third_party/mnn/build/MNNConvert -f ONNX \
  --modelFile models/gemma-3n/onnx/audio_encoder_q4.onnx \
  --MNNModel models/gemma-3n/mnn/audio_encoder_q4.mnn \
  --weightQuantBits=4
```

## Phase 3: Benchmark Scripts

### 3.1 Text Decoder Benchmark
**File**: `scripts/bench_mnn_text.py`

Measures:
- Model load time
- Prefill latency (ms)
- Decode throughput (TPS)
- ms/token average

### 3.2 Audio Pipeline Benchmark (Full ASR)
**File**: `scripts/bench_mnn_audio.py`

Pipeline:
```
PCM float32 (16kHz) → Mel-spectrogram (128 bins) → Audio Encoder → Features → Text Decoder → Transcription
```

Measures per phase:
- Mel-spectrogram generation (ms)
- Audio encoding (ms)
- Text decoding (ms, TPS)
- Total end-to-end latency

### 3.3 Comparison Script
**File**: `scripts/bench_engines.py`

Direct comparison table:
| Engine | Text TPS | Audio Encode (ms) | Total ASR (ms) |
|--------|----------|-------------------|----------------|
| ONNX Runtime | baseline | baseline | baseline |
| MNN-LLM | ? | ? | ? |

## Phase 4: Implementation Steps

| Step | Action | Files |
|------|--------|-------|
| 1 | Build MNN from source | `third_party/mnn/` |
| 2 | Convert ONNX models to MNN | `models/gemma-3n/mnn/` |
| 3 | Create MNN text benchmark | `scripts/bench_mnn_text.py` |
| 4 | Create MNN audio benchmark | `scripts/bench_mnn_audio.py` |
| 5 | Create comparison script | `scripts/bench_engines.py` |
| 6 | Run benchmarks, collect results | `results/` |
| 7 | Document findings | Update this file |

## Key Files

| File | Purpose |
|------|---------|
| `scripts/bench_gemma.py` | Existing ONNX benchmark pattern |
| `scripts/gemma/kv_cache.py` | KV cache logic to adapt |
| `src/generator/multimodal_encoder.cpp` | Audio mel-spectrogram constants |
| `models/gemma-3n/onnx/` | Source ONNX models |
| `models/gemma-3n/mnn/` | Converted MNN models (to create) |

## Model Constants (Gemma 3n E2B)
```
NUM_LAYERS = 30
NUM_KV_HEADS = 2
HEAD_DIM = 256
HIDDEN_DIM = 2048
SAMPLE_RATE = 16000
MEL_BINS = 128
HOP_LENGTH = 160 (10ms)
```

## Expected Results

| Metric | ONNX RT (1 thread) | MNN-LLM (1 thread) |
|--------|-------------------|-------------------|
| Text TPS | ~2.6 | ~5-8 (potential 2-3x) |
| Audio encode | ~100ms | ~50-80ms |
| Prefill | ~830ms | ~400-600ms |

## Risks
1. **Gemma 3n architecture** - May need custom MNN integration (per_layer_inputs)
2. **ONNX conversion** - Some ops may not convert cleanly
3. **KV cache format** - MNN uses different layout than ONNX

## Future Engines (After MNN)
- ExecuTorch + XNNPACK
- NCNN
- llama.cpp (text-only, hybrid approach for audio)
