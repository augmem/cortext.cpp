# Plan: Integrate gemma-3n-e2b ONNX for Consolidation

## Overview

Add gemma-3n-e2b quantized ONNX model under `models/gemma-3n/` and create `Summarizer` and `Extractor` interfaces (following the `Encoder` pattern) to power the consolidation system's semantic extraction and text summarization.

**Key asset**: Port `poc/memstream/json_decoder.py` (1757 lines) to C++ for schema-constrained JSON generation.

## Model Details

- **Source**: https://huggingface.co/onnx-community/gemma-3n-E2B-it-ONNX
- **Quantization**: q4 decoder, q8 embed_tokens/audio_encoder, fp16 vision_encoder
- **Size**: ~4-6GB total
- **Capabilities**: 2B effective params, 32K context, multimodal (text/image/audio)

## File Structure

```
include/cortext/
  summarizer/
    summarizer.hpp              # Abstract interface
    gemma_summarizer.hpp        # Gemma implementation
  extractor/
    extractor.hpp               # Abstract interface
    gemma_extractor.hpp         # Gemma implementation
  generator/
    text_generator.hpp          # Core generation interface
    gemma_text_generator.hpp    # ONNX Runtime implementation
    gemma_tokenizer.hpp         # HuggingFace tokenizer parser
    json_decoder.hpp            # Port of StreamingJSONParser
    constrained_sampler.hpp     # Port of SchemaConstrainedSampler

src/
  summarizer/gemma_summarizer.cpp
  extractor/gemma_extractor.cpp
  generator/
    gemma_text_generator.cpp
    gemma_tokenizer.cpp
    json_decoder.cpp            # JSON state machine + schema validation
    constrained_sampler.cpp     # Logit masking for schema conformance

models/
  gemma-3n/
    embed_tokens_q8.onnx
    decoder_model_merged_q4.onnx
    audio_encoder_q8.onnx
    vision_encoder_fp16.onnx
    tokenizer.json
    tokenizer_config.json

tests/
  gemma_tokenizer.test.cpp
  gemma_text_generator.test.cpp
  gemma_summarizer.test.cpp
  gemma_extractor.test.cpp
  json_decoder.test.cpp         # Parser state machine tests
  constrained_sampler.test.cpp  # Logit constraint tests
```

## Interface Definitions

### Summarizer (`include/cortext/summarizer/summarizer.hpp`)

```cpp
class Summarizer {
public:
  virtual ~Summarizer() = default;
  virtual std::string Summarize(const std::vector<std::string>& texts) = 0;
  virtual bool IsAvailable() const = 0;
};
```

### Extractor (`include/cortext/extractor/extractor.hpp`)

```cpp
class Extractor {
public:
  virtual ~Extractor() = default;
  virtual operations::ExtractionResult Extract(
      const std::string& summary,
      const std::vector<std::string>& sources) = 0;
  virtual bool IsAvailable() const = 0;
};
```

### TextGenerator (`include/cortext/generator/text_generator.hpp`)

```cpp
struct GenerationConfig {
  int max_new_tokens = 256;
  float temperature = 0.7f;
  float top_p = 0.9f;
  int top_k = 50;
  bool do_sample = true;
};

class TextGenerator {
public:
  virtual ~TextGenerator() = default;
  virtual std::string Generate(const std::string& prompt,
                               const GenerationConfig& config = {}) = 0;
  virtual bool IsAvailable() const = 0;
};
```

## Python Files to Port from `poc/memstream/`

| Python Source | C++ Target | Lines | Description |
|--------------|------------|-------|-------------|
| `json_decoder.py` | `generator/json_decoder.cpp` | 1757 | Streaming JSON parser + constrained sampler |
| `gemma/kv_cache.py` | `generator/kv_cache.cpp` | 111 | Preallocated KV cache manager |
| `gemma/runtime.py` | `generator/decoder_runner.cpp` | 70 | IOBinding-based decoder runner |
| `gemma/graph_spec.py` | `generator/graph_spec.cpp` | ~100 | Output tensor name mapping |

### Key Components to Port

1. **JSONState enum** - 12 states for JSON parsing state machine
2. **ParserStatus enum** - VALID_PARTIAL, COMPLETE, INVALID
3. **StreamingJSONParser class**:
   - Schema-aware incremental JSON parsing
   - oneOf branch handling (discriminator + heuristic)
   - Enum value constraints
   - `AddToken(text) -> ParserStatus`
   - `GetAllowedTokens() -> vector<string>`

4. **SchemaConstrainedSampler class**:
   - Token vocabulary mapping (JSON structural tokens)
   - `ConstrainLogits(logits) -> masked_logits`
   - Numeric type gating (integer vs number)
   - String enum gating

5. **KVCache class** (30 layers, 2 heads, 256 head_dim):
   - `PastInputs() -> map<string, Tensor>`
   - `AppendFromPresent(present)`
   - `InitFromPresent(present)` (prefill)
   - Dynamic capacity growth

6. **GemmaDecoderRunner class** (IOBinding):
   - Zero-copy CPU outputs
   - `Run(embeds, per_layer, pos_ids, past_kv) -> outputs`

### Key C++ Data Structures

```cpp
enum class JSONState {
  kExpectingObjectStart, kExpectingKeyOrClose, kExpectingKeyString,
  kExpectingColon, kExpectingValue, kInStringValue, kInNumberValue,
  kInBooleanValue, kInNullValue, kInArray, kInNestedObject,
  kExpectingCommaOrClose, kInvalid
};

struct OneOfBranchInfo {
  std::vector<nlohmann::json> branches;
  std::optional<std::string> discriminator_prop;
  std::unordered_map<std::string, int> value_to_branch;
  std::set<std::string> union_properties;
};

class StreamingJSONParser {
  nlohmann::json schema_;
  JSONState state_;
  std::string buffer_;
  std::vector<std::pair<std::string, nlohmann::json>> stack_;
  std::set<std::string> seen_fields_;
  std::set<std::string> required_fields_;
  OneOfBranchInfo oneof_info_;
  std::optional<int> active_branch_index_;
  // ...
};
```

## Implementation Phases

### Phase 1: Model Setup + Tokenizer

1. **Download gemma-3n-e2b ONNX files** to `models/gemma-3n/`
2. **GemmaTokenizer** - Parse HuggingFace `tokenizer.json`
   - `Encode(text) -> vector<int64_t>`
   - `Decode(vector<int64_t>) -> string`
   - Handle BOS/EOS/PAD special tokens (EOS=106)

### Phase 2: Port KV Cache + Decoder Runner

3. **KVCache class** (port `gemma/kv_cache.py`)
   - Preallocated per-layer buffers [B, H, T, D]
   - `PastInputs()`, `AppendFromPresent()`, `InitFromPresent()`
   - Dynamic capacity growth with 1.5x reallocation

4. **GemmaDecoderRunner** (port `gemma/runtime.py`)
   - IOBinding for zero-copy CPU inference
   - `Run(embeds, per_layer, pos_ids, past_kv) -> outputs`

### Phase 3: Port JSON Decoder (largest piece)

5. **StreamingJSONParser** (port `json_decoder.py:45-667`)
   - JSONState enum (12 states)
   - `_process_char()` state machine
   - oneOf branch handling (discriminator + heuristic)
   - `AddToken()`, `GetAllowedTokens()`, `IsSchemaSatisfied()`

6. **SchemaConstrainedSampler** (port `json_decoder.py:957-1391`)
   - Token vocabulary precomputation
   - `ConstrainLogits()` with -inf masking
   - Numeric/boolean/null type gating
   - String enum prefix matching

7. **ConstrainedJSONGenerator** (port `json_decoder.py:1393-1757`)
   - Generation loop with KV cache
   - Sampling: greedy, temperature, top-k, top-p
   - `Generate(prompt, schema, max_tokens) -> nlohmann::json`

### Phase 4: High-Level Interfaces

8. **GemmaSummarizer : Summarizer**
   - Format summarization prompt
   - Use unconstrained generation (no JSON schema)
   - Clean output, strip special tokens

9. **GemmaExtractor : Extractor**
   - Format extraction prompt with JSON schema
   - Use ConstrainedJSONGenerator
   - Parse into `ExtractionResult` struct

### Phase 5: CMake + Integration

10. **CMake Configuration**
    - Add `CORTEXT_ENABLE_GEMMA_ORT` option
    - Share ONNX Runtime with ImageBind
    - Add nlohmann/json dependency (for schema parsing)

11. **ProcessorContext Integration**
    - Add `Summarizer*` and `Extractor*` pointers
    - Update `ConsolidationSummarize` to use Summarizer
    - Re-enable `EnqueueExtractionJobs` with Extractor

### Phase 6: Tests + Multimodal

12. **Unit Tests**
    - JSON parser state transitions
    - Constrained sampler logit masking
    - Tokenizer roundtrip
    - KV cache append/init

13. **Vision/Audio Encoders** (optional)
    - Add multimodal generation methods
    - Preprocess image/audio inputs

## Critical Files to Modify

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Add `CORTEXT_ENABLE_GEMMA_ORT`, new sources |
| `include/cortext/processor/processor_context.hpp` | Add Summarizer*/Extractor* pointers |
| `src/operations/consolidation_summarize.cpp` | Use Summarizer when available |
| `src/operations/consolidation.cpp` (EnqueueExtractionJobs) | Re-enable with Extractor |
| `tests/CMakeLists.txt` | Add new test files |

## Code Patterns to Follow

1. **PIMPL with lazy init** (from `imagebind.cpp:907-942`)
2. **Conditional compilation** (`#if defined(CORTEXT_ENABLE_GEMMA_ORT)`)
3. **ONNX tensor creation** (from `imagebind.cpp:976-1003`)
4. **Interface style** (from `encoder.hpp` - pure virtual, throws on failure)

## CMake Changes

```cmake
option(CORTEXT_ENABLE_GEMMA_ORT "Enable ONNX Runtime for Gemma" OFF)

if(CORTEXT_ENABLE_GEMMA_ORT)
  target_compile_definitions(cortext PUBLIC CORTEXT_ENABLE_GEMMA_ORT=1)
endif()

# Share ONNX Runtime linking
if(CORTEXT_ENABLE_IMAGEBIND_ORT OR CORTEXT_ENABLE_GEMMA_ORT)
  # ... existing ONNX RT config ...
endif()
```

## Model Download

Download from HuggingFace and place in `models/gemma-3n/`:

```bash
# Using huggingface-cli (recommended)
huggingface-cli download onnx-community/gemma-3n-E2B-it-ONNX \
  --include "onnx/*" --local-dir models/gemma-3n

# Or manual: download from https://huggingface.co/onnx-community/gemma-3n-E2B-it-ONNX/tree/main/onnx
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| KV cache OOM | Implement cache pruning, limit max context |
| JSON parsing failures | Robust parsing with fallbacks, retry prompting |
| Large model size | q4 quantization reduces to ~2-3GB decoder |
| Thread safety | Mutex-protected lazy init (no std::thread for WASM) |
