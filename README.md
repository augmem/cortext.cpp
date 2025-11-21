# Cortext: Intelligent Context System for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Cortext** is a production-ready intelligent context system that monitors LLM token generation in real-time and dynamically injects relevant context through semantic retrieval and knowledge graph augmentation. Like the brain's hemispheres working together, the LLM handles analytical generation while Cortext provides the contextual memory and relevance processing - creating a closed-loop cognitive system that learns from every interaction.

## 🧩 C++ Core (cortext) Overview

This repository contains a modern C++17 core library named `cortext` that provides a high-level intelligent context system:

* **`Cortext`**: Main API class for multimodal context processing
* **Three-Knob Configuration**: Focus, Sensitivity, and Stability control all behavior
* **Multimodal Input**: Process text, audio, and image inputs seamlessly
* **Real-time Context Injection**: Interrupts LLM generation to inject relevant context
* **Closed-Loop Learning**: Updates context strengths based on injection outcomes
* **Automatic Consolidation**: Background merging and knowledge graph construction
* **Cross-Language Bindings**: Native APIs for Python, Go, and JavaScript via WASM

**Key Components:**

* **Context Processing**: Unified 256d embedding space for all modalities
* **Memory Consolidation**: Background merging of redundant contexts
* **Graph-Augmented Retrieval**: Vector similarity combined with knowledge graphs
* **Adaptive Thresholds**: Self-tuning relevance detection
* **Production Ready**: Comprehensive error handling and observability

Minimal example (C++):

```cpp
#include <cortext/cortext.hpp>

using namespace cortext;

int main() {
  // Configure the three-knob system
  Cortext::Config cfg;
  cfg.focus = 0.7;      // Selectivity: higher = more focused retrieval
  cfg.sensitivity = 0.5; // Plasticity: higher = more responsive to changes
  cfg.stability = 0.8;   // Persistence: higher = slower parameter drift

  // Create Cortext instance with in-memory database
  auto cortext = Cortext::Create(cfg, ":memory:", "models/");

  // Process multimodal inputs
  auto ctx_out = cortext->ProcessText("Hello, how are you today?", 1000, "user_input");
  if (ctx_out.should_interrupt) {
    for (const auto &memory : ctx_out.memories) {
      std::cout << "Memory " << memory.id << " (" << memory.modality << ")\\n";
    }
  }
  cortext->ProcessAudio(audio_samples.data(), audio_samples.size(), 1001, "microphone");

  // Trigger consolidation if needed
  cortext->Consolidate(1002);

  // Flush any pending writes
  cortext->Flush();
}
```

Build & Test (C++ core only):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
# Run only project tests (skip Eigen's own test suite)
ctest --test-dir build -R cortext_tests --output-on-failure
# Or run the binary directly
./build/tests/cortext_tests
```

### SQLite Extensions

* Default: sqlite-vec, sqlite-graph, and sqlite-objstore are statically embedded into the `cortext` library.
* Dynamic fallback (native only): If embedding is disabled, `cortext` attempts to load sqlite-vec/sqlite-graph at runtime.
  * Env overrides: `SQLITE_VEC_PATH`, `SQLITE_GRAPH_PATH`
  * Default search paths:
    * `third_party/sqlite-vec/dist/vec0.{so|dylib|dll}`
    * `third_party/sqlite-graph/build/libgraph.{so|dylib|dll}`
* WASM: Both extensions are embedded into `cortext.wasm`; runtime loading is disabled.
* sqlite-objstore is embedded-only today and provides transactional blob/object storage for multimodal payloads (video, audio, high-fidelity artifacts) with automatic backend selection (file/sqlite on native builds, OPFS/VFS on WASM).

## 🧠 Left Brain, Right Brain: Cognitive Collaboration

**Like the brain's hemispheres working in harmony**, the LLM (left brain) excels at logical reasoning and analytical generation, while Cortext (right brain) provides contextual memory, relevance processing, and creative association. Together they form a complete cognitive system that learns from every interaction.

**Closed-Loop Intelligent Context System**: Real-time monitoring, multimodal semantic retrieval, interrupt-driven injection, and continuous learning from context injection outcomes.

* **Real-time Monitoring**: Watches tokens as they're generated (right brain awareness)
* **Multimodal Semantic Relevance**: Unified embeddings across text, vision, and audio (right brain association)
* **Interrupt-Driven Injection**: Stops generation, injects relevant context, resumes seamlessly (hemispheric coordination)
* **Closed-Loop Learning**: Updates context strengths based on injection impact (right brain learning)
* **Adaptive Thresholds**: Self-tuning relevance thresholds based on retrieval quality (right brain adaptation)

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 LLM (Left Brain)                            │
│            Logical Reasoning & Generation                   │
└────────────────────┬────────────────────────────────────────┘
                     │ token stream
                     ↓
┌─────────────────────────────────────────────────────────────┐
│             Cortext (Right Brain)                           │
│             Context & Memory Processing                     │
│  ┌──────────────────────────────────────────────────────────┤  │
│  │  Token Buffer (ring buffer, size=64)                │  │
│  │  • Rolling context window                           │  │
│  │  • Semantic analysis every 16 tokens                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────────┤  │
│  │  Multimodal Relevance Engine                         │  │
│  │  • Unified embedding space (256d Gemma MRL)          │  │
│  │  • Custom projection for vision/audio                │  │
│  │  • Graph-augmented vector search                     │  │
│  │  • Adaptive threshold filtering                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────────┤  │
│  │  Interrupt Decision                                  │  │
│  │  • Relevance > threshold → INTERRUPT                 │  │
│  │  • Context assembly from multimodal memories         │  │
│  │  • Inject and resume generation                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓
                   [Resume Generation]
```

## 🛠️ Technology Stack

### Core Dependencies

* **C++17+**: Core engine implementation with WASM compilation target
* **ONNX Runtime**: C++ API for model inference
* **[EmbeddingGemma-300M ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)**: Unified 256d MRL-truncated embedding space
* **[Gemma 3n-E2B-it ONNX](https://huggingface.co/onnx-community/gemma-3n-E2B-it-ONNX)**: Frozen vision and audio encoders
* **[sqlite-vec](https://github.com/asg017/sqlite-vec)**: High-performance vector similarity search
* **[sqlite-graph](https://github.com/agentflare-ai/sqlite-graph)**: Cypher-compatible graph database
* **[sqlite-objstore](./docs/sqlite-objstore.md)**: Transactional blob/object storage for multimodal payloads backed by SQLite or sharded filesystems

### Language Bindings

* **Python**: `pip install cortext`
* **Go**: `go get github.com/agentflare/cortext/go`
* **JavaScript/TypeScript**: NPM package for browser integration

## 📦 Installation

### Prerequisites

* **C++17+ compiler** (Clang/LLVM recommended)
* **CMake 3.20+**
* **ONNX Runtime**
* **SQLite development headers**

### Build from Source

```bash
# Clone with submodules
git clone --recursive https://github.com/agentflare/cortext.git
cd cortext

# Build core engine
mkdir build && cd build
cmake ..
make -j$(nproc)

# Install
sudo make install
```

### Python Installation

```bash
pip install cortext
```

### Go Installation

```bash
go get github.com/agentflare/cortext/go
```

### C++ Library Build (cortext)

For the minimal cortext C++ library component:

```bash
# Configure build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build library and tests
cmake --build build -j

# Run tests
ctest --test-dir build --output-on-failure

# Install library (optional)
cmake --install build --prefix "$PWD/install"
```

### WebAssembly Build

To build the cortext library for WebAssembly (usable in Rust, Python, and JavaScript):

```bash
# Prerequisites: Install Emscripten
# See: https://emscripten.org/docs/getting_started/downloads.html

# Build for WebAssembly using preset
cmake --preset wasm
cmake --build --preset wasm
```

The WASM build produces:

* `cortext.wasm`: WebAssembly binary
* `cortext.js`: JavaScript glue code for ES6 module imports

## 🚀 Quick Start

### Python Usage

```python
from cortext import CortextEngine, Config

# Initialize with configuration
config = Config(
    focus=0.7,           # F: selectivity (0.0-1.0)
    sensitivity=0.5,     # S: plasticity (0.0-1.0)
    stability=0.8,       # T: persistence (0.0-1.0)
    embedding_dim=256,   # EmbeddingGemma 256d MRL-truncated space
    database_path="./memory.db"
)

engine = CortextEngine(config)

# Process token stream
for token in llm_stream:
    should_interrupt, context = engine.process_token(token)
    if should_interrupt:
        # Inject context and resume
        llm.inject_context(context)
```

### Go Usage

```go
import "github.com/agentflare/cortext/go"

config := cortext.Config{
    Focus:       0.7,
    Sensitivity: 0.5,
    Stability:   0.8,
    EmbeddingDim: 256,
    DatabasePath: "./memory.db",
}

engine, err := cortext.New(config)
if err != nil {
    log.Fatal(err)
}

// Process tokens with context
ctx := context.Background()
for token := range llmStream {
    interrupt, context, err := engine.ProcessToken(ctx, token)
    if err != nil {
        log.Printf("Error: %v", err)
        continue
    }
    if interrupt {
        llm.InjectContext(context)
    }
}
```

## 🎯 Key Features

### Closed-Loop Learning System

* **Impact Monitoring**: Tracks how memory injections affect generation quality
* **Memory Strength Evolution**: Updates relevance scores based on injection outcomes
* **Adaptive Thresholds**: Self-tuning relevance thresholds
* **Reinforcement Learning**: Learns from successful vs. unsuccessful injections

### Intelligent Context System

* **Unified Embedding Space**: Single 256d space for text, vision, and audio (right brain synthesis)
* **Cross-Modal Search**: Retrieve relevant context regardless of input modality (right brain association)
* **Graph-Augmented Retrieval**: Combine vector similarity with knowledge graph traversal (right brain connections)
* **Knowledge Graph**: Entity and relationship modeling with Cypher queries (right brain relational thinking)

### Production-Ready Features

* **WASM First-Class Citizen**: Core engine compiles to WebAssembly
* **Cross-Language Bindings**: Native APIs for Python, Go, and JavaScript
* **Comprehensive Error Handling**: Graceful degradation and recovery strategies
* **Observability**: Structured logging and performance metrics
* **Memory Optimization**: Quantization, caching, and efficient memory pooling

## 📊 Performance Targets

| Metric             | Target                                  | Description                     |
| ------------------ | --------------------------------------- | ------------------------------- |
| Embedding Latency  | <50ms (CPU), <20ms (WebGPU)             | End-to-end embedding generation |
| Search Latency     | <25ms for 100K memories                 | Vector similarity search        |
| Graph Query        | <50ms for 2-hop traversals              | Knowledge graph operations      |
| Memory Overhead    | <200MB baseline + 1.5MB per 1K memories | Working memory footprint        |
| Interrupt Overhead | <5% of total generation time            | Streaming integration cost      |

## 🏛️ Project Structure

```
cortext/
├── src/                    # Core C++ implementation
│   ├── engine/            # Cortext intelligent engine
│   ├── context/           # ContextStore implementation
│   ├── embedding/         # Multimodal embedding pipeline
│   └── bindings/          # Language binding implementations
├── third_party/           # External dependencies
│   ├── sqlite-vec/        # Vector search extension
│   └── sqlite-graph/      # Graph database extension
├── python/                # Python bindings and packaging
├── go/                    # Go bindings and packaging
├── js/                    # JavaScript/TypeScript bindings
├── docs/                  # Documentation
│   ├── pmvp.md           # Production MVP specification
│   └── algorithms.md     # Algorithm documentation
├── examples/              # Usage examples
├── tests/                 # Test suites
└── build/                 # Build configuration
```

## 🔄 Development Roadmap

### MVP (Current)

* [x] Core cortext intelligent engine
* [x] WASM compilation support
* [x] Python and Go bindings
* [x] Adaptive threshold system
* [x] Production error handling
* [x] Consolidation and graph integration system
* [x] sqlite-vec integration for vector search
* [x] sqlite-graph integration for knowledge graphs
* [x] EmbeddingGemma ONNX deployment

### Future Releases

* **v1.1**: Enhanced consolidation algorithms and graph reasoning
* **v1.2**: Additional multimodal encoders (video, documents)
* **v1.3**: Distributed memory for multi-user scenarios
* **v2.0**: Full 27-algorithm suite from research prototype

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/agentflare/cortext.git
cd cortext

# Install development dependencies
pip install -r requirements-dev.txt

# Build and test
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
ctest
```

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 📚 Documentation

* **[Production MVP Specification](docs/pmvp.md)**: Complete technical specification
* **[Algorithm Documentation](docs/algorithms.md)**: Memory algorithms and research
* **[API Reference](docs/api/)**: Language-specific API documentation
* **[Examples](examples/)**: Usage examples and tutorials

## 🙏 Acknowledgments

Built on groundbreaking research in intelligent context systems and real-time AI augmentation. Special thanks to the teams behind EmbeddingGemma, Gemma 3n, sqlite-vec, and sqlite-graph for their excellent foundational work.

***

**Cortext** - The right brain for your LLM's left brain. Intelligent context that learns and adapts in real-time.
