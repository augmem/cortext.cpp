# Plan: Integrate Phi-4-Multimodal via OGA for Consolidation

## Overview

Use **Phi-4-multimodal** (5.6B params, INT4 quantized) via **onnxruntime-genai (OGA)** for graph consolidation, semantic extraction, and text summarization. Structured JSON output is handled by **llguidance** (built into OGA), eliminating the need for custom JSON parsers.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Cortext Stack                           │
├─────────────────────────────────────────────────────────────┤
│  Embeddings     │  ImageBind (ONNX Runtime, INT8)          │
│                 │  - Text: 151ms, Audio: 141ms              │
├─────────────────────────────────────────────────────────────┤
│  Understanding  │  Phi-4-multimodal (OGA, INT4)            │
│                 │  - Text/Audio comprehension               │
│                 │  - ~11 tps single-thread, ~30 tps 4-thread│
├─────────────────────────────────────────────────────────────┤
│  Structured     │  llguidance (built into OGA)             │
│  Output         │  - JSON Schema constraints                │
│                 │  - No custom parser needed                │
└─────────────────────────────────────────────────────────────┘
```

## Model Details

- **Model**: `lokinfey/Phi-4-Multimodal-ONNX-INT4-CPU`
- **Size**: ~3GB (INT4 quantized)
- **Capabilities**: Text, Vision, Audio understanding
- **Performance**: 11 tps (1 thread), ~30 tps (4 threads on RPi 5)
- **Context**: 131K tokens

## File Structure

```
include/cortext/
  consolidator/
    consolidator.hpp           # Abstract interface
    phi4_consolidator.hpp      # Phi-4 OGA implementation

src/
  consolidator/
    phi4_consolidator.cpp      # OGA + llguidance implementation

models/
  phi4-mm-cpu/
    phi-4-mm-text.onnx
    phi-4-mm-speech.onnx
    phi-4-mm-vision.onnx
    phi-4-mm-embedding.onnx
    genai_config.json
    tokenizer.json
    tokenizer_config.json

  imagebind/                   # Existing
    audio_encoder_int8.onnx
    text_encoder_int8.onnx
```

## Interface Definitions

### Consolidator (`include/cortext/consolidator/consolidator.hpp`)

```cpp
#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace cortext {

struct ConsolidationResult {
  std::string operation;      // "consolidate", "strengthen", "decay"
  std::string source_node;
  std::string target_node;
  std::string relation;
  double weight;
};

struct ExtractionResult {
  std::vector<std::string> entities;
  std::vector<std::string> concepts;
  std::vector<std::tuple<std::string, std::string, std::string>> relations;
};

class Consolidator {
public:
  virtual ~Consolidator() = default;

  // Summarize multiple text fragments
  virtual std::string Summarize(const std::vector<std::string>& texts) = 0;

  // Extract entities/relations with JSON schema constraint
  virtual ExtractionResult Extract(const std::string& text,
                                   const nlohmann::json& schema) = 0;

  // Generate graph operation with JSON schema constraint
  virtual ConsolidationResult Consolidate(const std::string& context,
                                          const nlohmann::json& schema) = 0;

  // Process audio input (Phi-4 native audio understanding)
  virtual std::string ProcessAudio(const float* pcm, size_t num_samples) = 0;

  virtual bool IsAvailable() const = 0;
};

} // namespace cortext
```

## llguidance Integration

OGA's built-in llguidance handles JSON Schema constraints automatically:

```cpp
// C++ usage pattern
#include <ort_genai.h>

void GenerateStructuredJSON(const std::string& prompt,
                           const nlohmann::json& schema) {
  auto model = OgaModel::Create("models/phi4-mm-cpu");
  auto params = OgaGeneratorParams::Create(*model);

  // Setup JSON schema guidance
  std::string schema_str = schema.dump();
  schema_str = R"({"x-guidance":{"whitespace_flexible":false},)" +
               schema_str.substr(1);
  std::string guidance = "start: %json " + schema_str + "\n";

  params->SetSearchOption("max_length", 200);
  params->SetSearchOption("temperature", 0.0);
  params->SetGuidance("lark_grammar", guidance.c_str(), false);

  auto generator = OgaGenerator::Create(*model, *params);
  // ... generate with automatic schema constraint
}
```

### Schema Examples

**Graph Operation Schema:**
```json
{
  "type": "object",
  "properties": {
    "operation": {"type": "string", "enum": ["consolidate", "strengthen", "decay"]},
    "source_node": {"type": "string"},
    "target_node": {"type": "string"},
    "relation": {"type": "string"},
    "weight": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": ["operation", "source_node", "target_node"]
}
```

**Extraction Schema:**
```json
{
  "type": "object",
  "properties": {
    "entities": {"type": "array", "items": {"type": "string"}},
    "concepts": {"type": "array", "items": {"type": "string"}},
    "relations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "subject": {"type": "string"},
          "predicate": {"type": "string"},
          "object": {"type": "string"}
        }
      }
    }
  }
}
```

## Implementation Phases

### Phase 1: OGA Integration

1. **Add OGA dependency to CMake**
   ```cmake
   option(CORTEXT_ENABLE_OGA "Enable onnxruntime-genai" OFF)

   if(CORTEXT_ENABLE_OGA)
     find_package(onnxruntime-genai REQUIRED)
     target_link_libraries(cortext PRIVATE onnxruntime-genai)
     target_compile_definitions(cortext PUBLIC CORTEXT_ENABLE_OGA=1)
   endif()
   ```

2. **Download Phi-4-multimodal model**
   ```bash
   huggingface-cli download lokinfey/Phi-4-Multimodal-ONNX-INT4-CPU \
     --local-dir models/phi4-mm-cpu
   ```

### Phase 2: Consolidator Implementation

3. **Phi4Consolidator class**
   - Wrap OGA Model, Generator, MultimodalProcessor
   - Implement Summarize() with unconstrained generation
   - Implement Extract() with llguidance JSON schema
   - Implement Consolidate() with llguidance JSON schema
   - Implement ProcessAudio() for native audio understanding

### Phase 3: ProcessorContext Integration

4. **Update ProcessorContext**
   ```cpp
   struct ProcessorContext {
     // Existing...
     Consolidator* consolidator = nullptr;  // Add this
   };
   ```

5. **Update consolidation operations**
   - `ConsolidationSummarize` uses `consolidator->Summarize()`
   - `ConsolidationCluster` uses `consolidator->Extract()`
   - Graph operations use `consolidator->Consolidate()`

### Phase 4: Tests

6. **Unit tests**
   - `tests/phi4_consolidator.test.cpp`
   - Schema constraint validation
   - Audio processing tests

## Performance Targets

| Operation | Tokens | Time (1 thread) | Time (4 threads) |
|-----------|--------|-----------------|------------------|
| Summarize (100 tokens) | 100 | 9s | 3s |
| Extract (50 tokens) | 50 | 4.5s | 1.5s |
| Consolidate (30 tokens) | 30 | 2.7s | <1s |

All operations run during **idle time**, so latency is not critical.

## Files to Modify

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Add `CORTEXT_ENABLE_OGA`, link onnxruntime-genai |
| `include/cortext/processor/processor_context.hpp` | Add `Consolidator*` |
| `src/operations/consolidation_summarize.cpp` | Use Consolidator |
| `src/operations/consolidation_cluster.cpp` | Use Consolidator for extraction |
| `tests/CMakeLists.txt` | Add consolidator tests |

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| onnxruntime-genai | ≥0.11.4 | LLM inference with KV cache |
| llguidance | (bundled) | JSON Schema constrained decoding |
| nlohmann/json | (existing) | Schema parsing |

## Comparison: Old vs New Approach

| Aspect | Old (Gemma-3n + Custom) | New (Phi-4 + OGA) |
|--------|------------------------|-------------------|
| Model | Gemma-3n (not supported) | Phi-4-multimodal |
| JSON Constraints | Custom StreamingJSONParser | llguidance (built-in) |
| Code Complexity | ~2000 lines custom parser | ~200 lines wrapper |
| Audio Support | Separate encoder | Native understanding |
| Maintenance | High (custom parser) | Low (OGA maintained) |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| OGA build complexity | Use pip-installed wheel for Python, prebuilt for C++ |
| llguidance not in C++ build | Rebuild OGA with `-DUSE_GUIDANCE=ON` |
| Model size (3GB) | INT4 quantization, fits in 8GB RPi 5 |
| Thread safety | OGA handles internally, one model instance per thread |
