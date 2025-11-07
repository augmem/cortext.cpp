# Streaming Memory: Multimodal Memory System for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Streaming Memory MVP** is a production-ready multimodal memory system that monitors LLM token generation in real-time and dynamically injects relevant context through semantic retrieval and knowledge graph augmentation. Unlike traditional RAG systems, it interrupts generation mid-stream when semantic relevance exceeds configurable thresholds, while continuously consolidating memories into structured knowledge graphs for long-term understanding.

## 🚀 Key Innovation

**Closed-Loop Multimodal Memory System**: Real-time monitoring, multimodal semantic retrieval, interrupt-driven injection, and continuous learning from memory injection outcomes.

- **Real-time Monitoring**: Watches tokens as they're generated
- **Multimodal Semantic Relevance**: Unified embeddings across text, vision, and audio
- **Interrupt-Driven Injection**: Stops generation, injects relevant memories, resumes seamlessly
- **Closed-Loop Learning**: Updates memory strengths based on injection impact
- **Adaptive Thresholds**: Self-tuning relevance thresholds based on retrieval quality

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Generation Stream                     │
└────────────────────┬────────────────────────────────────────┘
                     │ token stream
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                StreamingMemory Engine                       │
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
- **C++17+**: Core engine implementation with WASM compilation target
- **ONNX Runtime**: C++ API for model inference
- **[EmbeddingGemma-300M ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)**: Unified 256d MRL-truncated embedding space
- **[Gemma 3n-E2B-it ONNX](https://huggingface.co/onnx-community/gemma-3n-E2B-it-ONNX)**: Frozen vision and audio encoders
- **[sqlite-vec](https://github.com/asg017/sqlite-vec)**: High-performance vector similarity search
- **[sqlite-graph](https://github.com/agentflare-ai/sqlite-graph)**: Cypher-compatible graph database

### Language Bindings
- **Python**: `pip install streaming-memory`
- **Go**: `go get github.com/agentflare/streaming-memory/go`
- **JavaScript/TypeScript**: NPM package for browser integration

## 📦 Installation

### Prerequisites
- **C++17+ compiler** (Clang/LLVM recommended)
- **CMake 3.20+**
- **ONNX Runtime**
- **SQLite development headers**

### Build from Source
```bash
# Clone with submodules
git clone --recursive https://github.com/agentflare/streaming-memory.git
cd streaming-memory

# Build core engine
mkdir build && cd build
cmake ..
make -j$(nproc)

# Install
sudo make install
```

### Python Installation
```bash
pip install streaming-memory
```

### Go Installation
```bash
go get github.com/agentflare/streaming-memory/go
```

## 🚀 Quick Start

### Python Usage
```python
from streaming_memory import StreamingMemory, Config

# Initialize with configuration
config = Config(
    focus=0.7,           # F: selectivity (0.0-1.0)
    sensitivity=0.5,     # S: plasticity (0.0-1.0)
    stability=0.8,       # T: persistence (0.0-1.0)
    embedding_dim=256,   # EmbeddingGemma 256d MRL-truncated space
    database_path="./memory.db"
)

memory = StreamingMemory(config)

# Process token stream
for token in llm_stream:
    should_interrupt, context = memory.process_token(token)
    if should_interrupt:
        # Inject context and resume
        llm.inject_context(context)
```

### Go Usage
```go
import "github.com/agentflare/streaming-memory/go"

config := memory.Config{
    Focus:       0.7,
    Sensitivity: 0.5,
    Stability:   0.8,
    EmbeddingDim: 256,
    DatabasePath: "./memory.db",
}

mem, err := memory.New(config)
if err != nil {
    log.Fatal(err)
}

// Process tokens with context
ctx := context.Background()
for token := range llmStream {
    interrupt, context, err := mem.ProcessToken(ctx, token)
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
- **Impact Monitoring**: Tracks how memory injections affect generation quality
- **Memory Strength Evolution**: Updates relevance scores based on injection outcomes
- **Adaptive Thresholds**: Self-tuning relevance thresholds
- **Reinforcement Learning**: Learns from successful vs. unsuccessful injections

### Multimodal Memory System
- **Unified Embedding Space**: Single 256d space for text, vision, and audio
- **Cross-Modal Search**: Retrieve relevant memories regardless of input modality
- **Graph-Augmented Retrieval**: Combine vector similarity with knowledge graph traversal
- **Knowledge Graph**: Entity and relationship modeling with Cypher queries

### Production-Ready Features
- **WASM First-Class Citizen**: Core engine compiles to WebAssembly
- **Cross-Language Bindings**: Native APIs for Python, Go, and JavaScript
- **Comprehensive Error Handling**: Graceful degradation and recovery strategies
- **Observability**: Structured logging and performance metrics
- **Memory Optimization**: Quantization, caching, and efficient memory pooling

## 📊 Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Embedding Latency | <50ms (CPU), <20ms (WebGPU) | End-to-end embedding generation |
| Search Latency | <25ms for 100K memories | Vector similarity search |
| Graph Query | <50ms for 2-hop traversals | Knowledge graph operations |
| Memory Overhead | <200MB baseline + 1.5MB per 1K memories | Working memory footprint |
| Interrupt Overhead | <5% of total generation time | Streaming integration cost |

## 🏛️ Project Structure

```
streaming-memory/
├── src/                    # Core C++ implementation
│   ├── engine/            # StreamingMemory engine
│   ├── memory/            # MemoryStore implementation
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
- [x] Core streaming memory engine
- [x] WASM compilation support
- [x] Python and Go bindings
- [x] Adaptive threshold system
- [x] Production error handling
- [x] Consolidation and graph integration system
- [x] sqlite-vec integration for vector search
- [x] sqlite-graph integration for knowledge graphs
- [x] EmbeddingGemma ONNX deployment

### Future Releases
- **v1.1**: Enhanced consolidation algorithms and graph reasoning
- **v1.2**: Additional multimodal encoders (video, documents)
- **v1.3**: Distributed memory for multi-user scenarios
- **v2.0**: Full 27-algorithm suite from research prototype

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone with submodules
git clone --recursive https://github.com/agentflare/streaming-memory.git
cd streaming-memory

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

- **[Production MVP Specification](docs/pmvp.md)**: Complete technical specification
- **[Algorithm Documentation](docs/algorithms.md)**: Memory algorithms and research
- **[API Reference](docs/api/)**: Language-specific API documentation
- **[Examples](examples/)**: Usage examples and tutorials

## 🙏 Acknowledgments

Built on groundbreaking research in multimodal memory systems and real-time AI augmentation. Special thanks to the teams behind EmbeddingGemma, Gemma 3n, sqlite-vec, and sqlite-graph for their excellent foundational work.

---

**Streaming Memory MVP** - Transforming how AI systems remember and learn in real-time.
