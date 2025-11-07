# Production MVP: Streaming Memory System

**Version:** 1.0.1\
**Status:** Production MVP Design\
**Last Updated:** November 7, 2025

***

## Executive Summary

**Streaming Memory MVP** is a production-ready multimodal memory system that monitors LLM token generation in real-time and dynamically injects relevant context through semantic retrieval and knowledge graph augmentation. Unlike traditional RAG systems, it interrupts generation mid-stream when semantic relevance exceeds configurable thresholds, while continuously consolidating memories into structured knowledge graphs for long-term understanding.

### Core Innovation: Closed-Loop Multimodal Memory System

Traditional agentic systems load entire conversation histories upfront. Streaming Memory MVP solves this with a **closed-loop multimodal system** that continuously learns from memory injection outcomes across text, vision, and audio:

1. **Real-time Monitoring**: Watches tokens as they're generated
2. **Multimodal Semantic Relevance**: Uses unified embeddings to detect context needs across modalities
3. **Interrupt-Driven Injection**: Stops generation, injects relevant multimodal memories, resumes seamlessly
4. **Closed-Loop Learning**: Monitors injection impact and updates memory strengths
5. **Adaptive Thresholds**: Self-tuning relevance thresholds based on retrieval quality

**The Closed Loop**: Retrieve → Inject → Monitor Impact → Update Strengths → Improve Future Retrieval

### Production Requirements

* **WASM First-Class Citizen**: Core engine compiles to WebAssembly
* **Cross-Language Bindings**: Native Go and Python APIs
* **Production Ready**: Comprehensive error handling, observability, performance optimization
* **Embeddable**: Can be integrated into existing LLM applications

***

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Generation Stream                     │
│                 (Claude, GPT, OpenAI, etc.)                  │
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
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                Impact Monitoring & Learning                 │
│  ┌──────────────────────────────────────────────────────────┤  │
│  │  Injection Impact Assessment                         │  │
│  │  • Generation quality metrics                        │  │
│  │  • Token efficiency analysis                         │  │
│  │  • Semantic coherence evaluation                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────────┤  │
│  │  Memory Strength Updates                             │  │
│  │  • Reinforcement learning from outcomes              │  │
│  │  • Memory consolidation triggers                     │  │
│  │  • Graph relationship updates                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────────┤  │
│  │  Parameter Adaptation                                │  │
│  │  • Adaptive threshold adjustment                     │  │
│  │  • Retrieval strategy optimization                   │  │
│  │  • Focus/Sensitivity/Stability tuning                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                   ▲          │
                   │          │
                   └──────────┘
                 Feedback Loop
```

### Data Flow (Closed Loop)

1. **Token Ingestion**: Each generated token flows through the engine
2. **Periodic Analysis**: Every 16 tokens, analyze recent context window
3. **Multimodal Embedding**: Generate unified vector representation using custom projection model
4. **Graph-Augmented Search**: Query both vector store and knowledge graph for relevant memories
5. **Threshold Check**: Compare relevance against adaptive threshold
6. **Interrupt Decision**: If relevance exceeds threshold, interrupt and inject
7. **Context Injection**: Build context from retrieved multimodal memories, resume generation
8. **Impact Monitoring**: Track how injected memories affect subsequent generation
9. **Strength Updates**: Update memory relevance scores based on observed impact
10. **Background Consolidation**: Periodically merge redundant memories and update knowledge graph
11. **Parameter Adaptation**: Adjust retrieval thresholds and algorithms based on success metrics

***

## Technology Stack

### Core Dependencies

* **C++**: Core engine implementation with WASM compilation target
  * Standard: C++17 or later (C++20 preferred for concepts and ranges)
  * Compiler: Clang/LLVM (native), Emscripten (WASM)
  * Build System: CMake 3.20+ for cross-platform builds
* **ONNX Runtime**: C++ API for model inference
  * Native: libonnxruntime.so/.dylib/.dll
  * WASM: onnxruntime-web via Emscripten bindings
* **SQLite**: Database engine (included in sqlite-vec and sqlite-graph extensions)
* **[EmbeddingGemma-300M ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)**: Unified 256d MRL-truncated embedding space (text target space)
* **[Gemma 3n-E2B-it ONNX](https://huggingface.co/onnx-community/gemma-3n-E2B-it-ONNX)**: Frozen vision and audio encoders
* **Lightweight Projection Layers**: Trained projectors mapping encoder outputs to 256d space
* **[sqlite-vec](https://github.com/asg017/sqlite-vec)**: High-performance vector similarity search extension for SQLite (C++)
* **[sqlite-graph](https://github.com/agentflare-ai/sqlite-graph)**: Knowledge graph with Cypher queries for relationship modeling
* **WebAssembly Component Model**: Cross-language bindings

### Key Technology Choices

#### sqlite-vec

**High-performance vector search for SQLite** with:

* Native C++ implementation for minimal overhead
* Support for 256-dimensional embeddings (EmbeddingGemma MRL-truncated space)
* Cosine similarity, dot product, and L2 distance metrics
* Quantization support (INT8, INT4) for reduced memory footprint
* Incremental index building for streaming workloads
* WASM compatibility for browser deployment

#### sqlite-graph

**Cypher-compatible graph extension** providing:

* Native property graph model within SQLite
* Cypher query language for complex relationship traversals
* Optimized joins between vector and graph data
* Transactional consistency across vector and graph updates
* Efficient pattern matching for knowledge graph queries

#### EmbeddingGemma-300M ONNX

**MRL-capable text embedding model** offering:

* 1024-dimensional output space with Matryoshka Representation Learning (MRL)
* Production usage: 256d truncated + renormalized embeddings for all modalities
* INT8 quantization for 4x memory reduction
* CPU-optimized inference (<50ms latency for text embeddings)
* MRL allows flexible dimensionality without retraining
* ONNX Runtime support for cross-platform deployment

#### Gemma 3n-E2B-it ONNX

**Frozen multimodal encoders** providing:

* Pre-trained vision and audio encoding capabilities
* Vision encoder: 224×224 RGB input, ImageNet normalization
* Audio encoder: 64-mel log-spectrogram (320ms window, 33 frames)
* Frozen during projection training (only projectors are trained)
* INT8 quantized for production deployment
* Target: <30ms latency per encoder on CPU
* Available from [onnx-community](https://huggingface.co/onnx-community/gemma-3n-E2B-it-ONNX)

### Language Bindings

* **Python**: `pip install streaming-memory`
* **Go**: `go get github.com/agentflare/streaming-memory/go`
* **JavaScript/TypeScript**: NPM package for browser integration

### Model Stack

* **Base Embedding**: EmbeddingGemma-300M (1024d full, 256d MRL-truncated) as unified semantic space
* **Vision Encoder**: Gemma 3n-E2B-it vision encoder (ONNX, from [onnx-community/gemma-3n-E2B-it-ONNX](https://huggingface.co/onnx-community/gemma-3n-E2B-it-ONNX))
* **Audio Encoder**: Gemma 3n-E2B-it audio encoder (ONNX, from same repository)
* **Projection Layers**: Lightweight projectors (Linear or MLP) map encoder outputs to EmbeddingGemma 256d space
* **Quantization**: INT8 for production performance (encoders + projectors)
* **Target Latency**: <40ms end-to-end (preprocessing + encoder + projector + normalization)

***

## Core Components

### StreamingMemory Engine

Main orchestrator written in C++ for performance and WASM compilation.

**Key Responsibilities:**

* Process incoming tokens and maintain rolling context window
* Generate unified multimodal embeddings at regular intervals using custom projection model
* Query memory store for semantically relevant memories across text, vision, and audio
* Apply adaptive threshold filtering
* Decide when to interrupt generation based on relevance
* Build context from retrieved multimodal memories for injection

### AdaptiveThreshold

Self-tuning relevance threshold based on retrieval quality.

**Mechanism:**

* **Threshold Increase**: Boosts threshold when finding highly relevant memories
* **Threshold Decay**: Reduces boost when no relevant memories found
* **History Tracking**: Maintains rolling window of recent retrieval quality
* **Adaptive Bounds**: Prevents runaway threshold increases/decreases

**Benefits:**

* **Precision Optimization**: Higher thresholds during periods of good retrieval
* **Responsiveness**: Lower thresholds when broader recall is needed
* **Stability**: Prevents oscillation through bounded adaptation

***

## Database Design

### Memory Store Schema

SQLite with [sqlite-vec](https://github.com/asg017/sqlite-vec) extension for persistent semantic storage.

```sql
-- Load sqlite-vec extension
.load vec0

-- Memories table with vector embeddings
CREATE VIRTUAL TABLE memories USING vec0(
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    embedding FLOAT[256],  -- EmbeddingGemma 256d MRL-truncated space
    timestamp INTEGER NOT NULL,
    source TEXT,
    metadata TEXT,  -- JSON metadata
    +distance_metric cosine
);

-- Index for timestamp-based queries
CREATE INDEX idx_memories_timestamp ON memories(timestamp DESC);

-- Metadata indices for filtering
CREATE INDEX idx_memories_source ON memories(source);
CREATE INDEX idx_memories_episode ON memories(json_extract(metadata, '$.episode_id'));
```

### Knowledge Graph Schema

Using [sqlite-graph](https://github.com/agentflare-ai/sqlite-graph) for relationship modeling.

```cypher
// Node types
CREATE (entity:Entity {name: $name, type: $type, embedding: $vec})
CREATE (memory:Memory {id: $id, summary: $text, strength: $score})
CREATE (concept:Concept {centroid: $vec, label: $name})

// Edge types with properties
CREATE (a)-[:CO_OCCURS_WITH {frequency: $freq, last_seen: $ts}]->(b)
CREATE (a)-[:IMPLIES {confidence: $conf, direction: $dir}]->(b)
CREATE (a)-[:CONTRADICTS {strength: $str}]->(b)
CREATE (a)-[:REINFORCES {weight: $w}]->(b)
CREATE (a)-[:DERIVED_FROM {consolidation_id: $id}]->(b)
```

### MemoryStore

Hybrid vector and graph database for persistent multimodal semantic storage and retrieval.

**Capabilities:**

* **Unified Vector Space**: Single 256d embedding space for text, vision, and audio memories using sqlite-vec
* **Knowledge Graph**: Entity and relationship modeling using sqlite-graph
* **Graph-Augmented Search**: Combine vector similarity with graph traversal
* **Cross-Modal Search**: Retrieve relevant memories regardless of input modality
* **Vector Similarity**: Cosine similarity search across multimodal embedding space (via sqlite-vec)
* **Persistent Storage**: Long-term memory retention across sessions
* **Metadata Association**: Link memories to source information and context
* **Temporal Queries**: Time-based memory retrieval and filtering
* **Memory Consolidation**: Background merging and summarization
* **Performance Scaling**: Handle large multimodal memory collections efficiently

***

## Consolidation and Graph Integration System

The production MVP includes a **long-term memory consolidation system** that periodically compresses redundant memories, merges related entries, and constructs a **knowledge graph** for persistent relationships and patterns. This transforms episodic short-term memories into semantic long-term structures while maintaining low-latency streaming operations.

### Consolidation Triggers

Background consolidation runs based on configurable triggers:

* **Capacity Threshold**: When memory store exceeds size limits
* **Performance Degradation**: When write rates drop below targets
* **Time-Based Cadence**: Periodic consolidation intervals

### Memory Scoring and Merging

Memories are evaluated for consolidation using adaptive metrics:

* **Strength**: Reinforcement-based persistence scores
* **Redundancy**: Similarity to nearby embeddings (via sqlite-vec cosine search)
* **Connectivity**: Shared entities or relationships (via sqlite-graph traversal)
* **Stability**: Persistence over time

Low-scoring memories are merged into summary nodes through clustering, reducing storage while preserving semantic content.

### Knowledge Graph Construction

Consolidated memories form a knowledge graph with multiple node and edge types:

**Node Types:**

* **Entity Nodes**: Named entities and topics extracted from content
* **Memory Nodes**: Summarized embeddings from merged clusters
* **Concept Nodes**: Emergent centroids representing recurrent topics

**Edge Types:**

* **co\_occurs\_with**: Shared context or temporal proximity
* **implies/causes**: High directional correlation patterns
* **contradicts**: Negative similarity or schema violations
* **reinforces**: Frequent joint retrieval patterns
* **derived\_from**: Links summaries to source memories

### Graph-Augmented Retrieval

Queries combine vector similarity with graph expansion:

1. Initial vector search finds semantically similar memories (sqlite-vec)
2. Graph traversal expands to connected entities and concepts (sqlite-graph Cypher)
3. Combined results are re-ranked for optimal relevance
4. Context assembly includes both direct matches and related concepts

**Example Query Pattern:**

```cypher
// Find memories similar to query embedding
MATCH (m:Memory)
WHERE vec_distance_cosine(m.embedding, $query_vec) < 0.3

// Expand to related entities and concepts
MATCH (m)-[:CO_OCCURS_WITH|REINFORCES*1..2]-(related)

// Return combined results with relevance scores
RETURN m, related, 
       vec_distance_cosine(m.embedding, $query_vec) as similarity,
       COUNT(related) as connectivity
ORDER BY similarity ASC, connectivity DESC
LIMIT 10
```

### Adaptive Integration

The consolidation system inherits memstream's adaptive knobs:

* **Focus (F)**: Controls merge strictness and consolidation conservativeness
* **Sensitivity (S)**: Influences graph node creation and relationship discovery
* **Stability (T)**: Controls consolidation frequency and relationship persistence

### Lifecycle Integration

The system operates in three phases:

1. **Stream**: Real-time capture, scoring, and short-term storage
2. **Consolidate**: Background merging and graph construction
3. **Retrieve**: Combined episodic and semantic memory recall

This creates a self-organizing memory system that evolves from immediate sensory capture to abstracted relational understanding.

***

## Closed-Loop Learning System

### Impact Monitoring

The closed-loop system continuously monitors how memory injections affect generation quality and updates memory strengths accordingly.

### Memory Strength Evolution

Memories evolve their relevance scores over time based on injection outcomes, with successful injections strengthening memories and unsuccessful ones weakening them.

### Learning Objectives

The closed-loop system optimizes for multiple objectives:

1. **Relevance Accuracy**: Memories retrieved should genuinely help generation
2. **Injection Efficiency**: Minimize tokens wasted on unnecessary interruptions
3. **Memory Freshness**: Prioritize recently successful memories
4. **Diversity**: Avoid over-reliance on a few "super memories"
5. **Adaptation Speed**: Quickly learn from changing user patterns

### Feedback Signals

Multiple feedback signals are combined to assess injection quality:

* **Generation Quality**: Perplexity reduction, coherence metrics
* **Token Efficiency**: Ratio of useful to total tokens after injection
* **Semantic Alignment**: Cosine similarity between injected memories and generated content
* **User Feedback**: Explicit ratings or implicit signals (continuation vs. restart)
* **Contextual Fit**: How well memories integrate with ongoing conversation

***

## Embedding Model Integration

### Multimodal Projection System

Unified embedding space using lightweight projection layers that map Gemma 3n-E2B-it encoder outputs into EmbeddingGemma's 256-dimensional MRL-truncated space.

**Architecture:**

* **Target Space**: EmbeddingGemma-300M 256d MRL-truncated space (from full 1024d output)
* **Vision Encoder**: Gemma 3n-E2B-it vision encoder (frozen, pre-trained)
  * Input: 224×224 RGB images with ImageNet normalization
  * Output: D\_vision dimensional embeddings
* **Audio Encoder**: Gemma 3n-E2B-it audio encoder (frozen, pre-trained)
  * Input: 64-mel log-spectrogram (320ms window, 33 frames)
  * Output: D\_audio dimensional embeddings
* **Projection Layers**: Trained lightweight projectors per modality
  * Architecture options: Linear (baseline) or MLP (Linear → LayerNorm → Linear)
  * Maps from D\_vision/D\_audio to 256 dimensions
  * L2 normalization applied to outputs
* **Cross-Modal Search**: All modalities searchable within unified 256d vector space

**Projection Layer Training:**

* **Loss Function**: InfoNCE (contrastive) with in-batch negatives
* **Frozen Encoders**: Gemma 3n encoders remain frozen during training
* **Training Target**: EmbeddingGemma-300M generated text embeddings (truncated + renormalized to 256d)
* **Optimization**: AdamW with cosine learning rate schedule
* **Datasets**: MS-COCO, AudioCaps, Clotho v2, Conceptual Captions (subset)

**Performance:**

* **CPU Target**: <40ms end-to-end (preprocessing + encoder + projector + norm)
* **Encoder Budget**: <30ms per encoder (INT8 quantized)
* **Projector Budget**: <5ms per projector (INT8 quantized)
* **Memory**: \~50MB working memory during inference
* **Quantization**: INT8 for both encoders and projectors

### Model Selection & ONNX Deployment

The production MVP uses [EmbeddingGemma-300M ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) with MRL (Matryoshka Representation Learning) truncation for optimal performance:

| Variant             | Dimensions  | Quantization | Size    | CPU Latency | Use Case            |
| ------------------- | ----------- | ------------ | ------- | ----------- | ------------------- |
| embeddinggemma-300m | 1024 (full) | FP32         | \~1.2GB | \~80ms      | Full-precision text |
| embeddinggemma-300m | 256 (MRL)   | FP32         | \~1.2GB | \~80ms      | **Target Space**    |
| embeddinggemma-300m | 256 (MRL)   | INT8         | \~320MB | \~45ms      | Production (Text)   |

**MRL Truncation Strategy:**

* Generate full 1024d embedding from EmbeddingGemma-300M
* Truncate to first 256 dimensions: `emb_256_raw = emb_1024[:, :256]`
* Renormalize: `emb_256 = emb_256_raw / (||emb_256_raw|| + 1e-6)`
* This 256d space serves as the unified target for all modality projectors

**Multimodal Encoders:**

| Model                 | Modality | Quantization | Latency Target | Source                              |
| --------------------- | -------- | ------------ | -------------- | ----------------------------------- |
| gemma-3n-e2b (vision) | Vision   | INT8         | <30ms          | onnx-community/gemma-3n-E2B-it-ONNX |
| gemma-3n-e2b (audio)  | Audio    | INT8         | <30ms          | onnx-community/gemma-3n-E2B-it-ONNX |

**ONNX Deployment Benefits:**

* **Cross-Platform**: Runs on CPU, CUDA, DirectML, CoreML, WebGPU
* **WebAssembly**: Direct integration with WASM engine via onnxruntime-web
* **Optimized Inference**: Graph optimizations and quantization built-in
* **No Python Runtime**: Native deployment without Python dependencies

***

## WASM Compilation

### WebAssembly Component Model

Using the WebAssembly Component Model for language-neutral interfaces.

**Component Architecture:**

* **Language Neutral**: Same interface across all target languages
* **Type Safety**: Strong typing at component boundaries
* **Versioning**: Explicit versioning for API evolution
* **Tooling**: Standard tooling for component generation and binding

**Interface Definition:**

* **Core Functions**: Process tokens, add memories, manage state
* **Streaming Types**: Async iterators for token processing
* **Error Handling**: Structured error types with component boundaries
* **Configuration**: Runtime configuration without recompilation

### Build Configuration

C++ compilation with Emscripten for WASM targets.

**Package Structure:**

* **Dual Targets**: Native (x86\_64, ARM) and WebAssembly builds from same C++ source
* **Conditional Compilation**: Platform-specific code using `#ifdef __EMSCRIPTEN__`
* **Optimization**: `-O3` for release builds with link-time optimization (`-flto`)
* **Size Optimization**: Strip debug symbols and minimize binary size with `-Oz` for WASM

**Emscripten Build Flags:**

* **Memory**: `-s ALLOW_MEMORY_GROWTH=1` for dynamic memory allocation
* **Filesystem**: `-s FORCE_FILESYSTEM=1` for SQLite database access
* **Threading**: Consider `-pthread` and `-s PTHREAD_POOL_SIZE=4` for multi-threading support
* **SIMD**: `-msimd128` for WebAssembly SIMD optimization where supported
* **Exports**: `-s EXPORTED_FUNCTIONS=[...]` and `-s EXPORTED_RUNTIME_METHODS=[...]` for API exposure

**Cross-Compilation:**

* **Native Build**: CMake or Makefile with standard C++ compiler (g++, clang++)
* **WASM Build**: `emcc` compiler with Emscripten toolchain
* **Feature Flags**: Conditional compilation for different deployment environments
* **Testing**: Unit tests run on both native and WASM targets (using Node.js for WASM)

***

## Language Bindings

The C++ core engine provides language bindings through multiple approaches:

* **Python**: Direct C++ bindings using pybind11 or SWIG
* **Go**: CGo bindings to compiled C++ library
* **JavaScript/TypeScript**: WebAssembly module compiled with Emscripten's embind
* **WebAssembly Component Model**: Future-proof interface for all languages

All bindings expose the same API surface with idiomatic error handling and conventions for each language.

### Python Binding

High-level Python API for seamless integration with existing Python applications and frameworks.

```python
from streaming_memory import StreamingMemory, Config

# Initialize with configuration
config = Config(
    focus=0.7,           # F: selectivity
    sensitivity=0.5,     # S: plasticity
    stability=0.8,       # T: persistence
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

### Go Binding

Idiomatic Go API following Go conventions for error handling and context management.

```go
import "github.com/agentflare/streaming-memory/go"

config := memory.Config{
    Focus:       0.7,
    Sensitivity: 0.5,
    Stability:   0.8,
    EmbeddingDim: 256,  // EmbeddingGemma 256d MRL-truncated space
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

***

## Production Features

### Error Handling

Comprehensive error handling with recovery strategies for production reliability.

**Error Categories:**

* **Database Errors**: Connection failures, transaction conflicts, corruption
* **Embedding Errors**: ONNX runtime failures, dimension mismatches, NaN values
* **Memory Errors**: OOM conditions, allocation failures
* **Configuration Errors**: Invalid parameters, missing dependencies

**Recovery Strategies:**

* **Graceful Degradation**: Continue operation with reduced functionality
* **Automatic Retry**: Exponential backoff for transient failures
* **Circuit Breakers**: Prevent cascading failures in embedding pipeline
* **Fallback Modes**: Use cached embeddings or simpler algorithms when ONNX unavailable

### Observability

Structured logging and metrics collection for monitoring system health and performance.

**Key Metrics:**

* **Latency**: p50, p95, p99 for embedding, search, and total interrupt overhead
* **Throughput**: Tokens/sec, interrupts/min, memory writes/min
* **Quality**: Precision/recall of retrievals, injection impact scores
* **Resource Usage**: Memory footprint, CPU utilization, database size

**Structured Logging:**

```json
{
  "event": "INTERRUPT_TRIGGERED",
  "timestamp": 1699392845,
  "latency_ms": 42,
  "relevance_score": 0.87,
  "memories_retrieved": 5,
  "context_tokens": 156,
  "knobs": {"F": 0.7, "S": 0.5, "T": 0.8}
}
```

### Performance Optimizations

1. **Embedding Caching**: Cache embeddings for recent contexts
2. **Batch Processing**: Process multiple tokens together when possible
3. **Memory Pooling**: Reuse allocated memory for tensors
4. **Async Operations**: Non-blocking I/O for database operations
5. **ANN Indexing**: Use approximate nearest neighbors (HNSW via sqlite-vec) for large memory stores
6. **Query Plan Optimization**: Leverage SQLite query planner with EXPLAIN QUERY PLAN

***

## Testing Strategy

### Unit Tests

Comprehensive testing of individual components and algorithms.

**Test Coverage:**

* **Embedding Pipeline**: ONNX model loading, inference, quantization correctness
* **Vector Search**: sqlite-vec similarity calculations, indexing, quantization
* **Graph Operations**: sqlite-graph Cypher queries, traversals, updates
* **Adaptive Algorithms**: Threshold updates, memory strength evolution
* **Error Handling**: Recovery from failures, fallback behavior

### Integration Tests

End-to-end testing of complete memory streaming workflows.

**Test Scenarios:**

* **Basic Streaming**: Token processing, interrupt triggering, context injection
* **Multimodal**: Text, vision, and audio memory integration
* **Consolidation**: Background merging, graph construction
* **Cross-Modal Retrieval**: Query with one modality, retrieve others
* **Performance**: Latency requirements, throughput targets

### Performance Benchmarks

Target performance characteristics for production deployment.

**Benchmark Suite:**

* **Embedding Latency**: <50ms (CPU), <20ms (GPU/WebGPU)
* **Vector Search**: <25ms for 100K memories (with sqlite-vec ANN)
* **Graph Traversal**: <50ms for 2-hop expansions (with sqlite-graph)
* **Memory Overhead**: <200MB baseline + 1.5MB per 1K memories
* **Interrupt Overhead**: <5% of total generation time

***

## Deployment Scenarios

### Browser Integration

Seamless integration with browser-based LLM applications.

**WebAssembly Loading:**

* **Dynamic Import**: Load WASM module on demand
* **Memory Management**: Efficient memory allocation in browser environment
* **Worker Support**: Run in Web Workers for non-blocking operation
* **Shared Memory**: Coordinate with main thread for UI updates

**Streaming Integration:**

* **Token Processing**: Intercept and analyze token streams in real-time
* **Context Injection**: Modify prompts with retrieved memories
* **UI Coordination**: Update interface to show memory injection events
* **Error Boundaries**: Graceful handling of WASM execution failures

**ONNX Runtime Web:**

* **WebGPU Acceleration**: Hardware-accelerated inference when available
* **CPU Fallback**: WASM-based CPU inference for compatibility
* **Model Loading**: Lazy load EmbeddingGemma ONNX on first use
* **Caching**: Cache compiled ONNX graph for instant subsequent loads

### Server-Side Integration

Integration with server-side LLM APIs and frameworks.

**API Integration:**

* **Streaming APIs**: Compatible with OpenAI, Anthropic, and custom streaming endpoints
* **Middleware Pattern**: Intercept and modify request/response streams
* **Context Injection**: Add system messages or modify user prompts
* **Rate Limiting**: Respect API limits while maintaining real-time responsiveness

**Framework Support:**

* **FastAPI/Flask**: Async web framework integration
* **LangChain/LlamaIndex**: Plugin architecture for memory augmentation
* **Custom Pipelines**: Drop-in replacement for existing RAG systems
* **Microservices**: Deploy as separate memory service with REST/gRPC APIs

***

## Migration Path

### From Prototype to Production

1. **Algorithm Simplification**: Reduce complexity while maintaining core functionality
2. **C++ Implementation**: Port core logic to C++ for performance and WASM support
3. **WASM Compilation**: Ensure all dependencies support WebAssembly (using Emscripten)
4. **Binding Generation**: Create language bindings using WebAssembly Interface Types
5. **Performance Optimization**: Profile and optimize bottlenecks
6. **Production Hardening**: Add comprehensive error handling and logging

### Backward Compatibility

The production MVP maintains API compatibility where possible while simplifying internal algorithms.

***

## Success Metrics

### Performance Targets

* **Embedding Latency**: <50ms (CPU), <20ms (WebGPU)
* **Search Latency**: <25ms for 100K memories (sqlite-vec ANN)
* **Graph Query**: <50ms for 2-hop traversals (sqlite-graph)
* **Memory Usage**: <200MB baseline + 1.5MB per 1K memories
* **Interrupt Overhead**: <5% of total generation time

### Quality Metrics

Target quality characteristics for memory retrieval and injection effectiveness:

* **Precision**: >80% of retrieved memories deemed relevant
* **Recall**: >70% of relevant memories successfully retrieved
* **Injection Impact**: >60% of injections improve generation quality
* **False Interrupt Rate**: <10% unnecessary interruptions

***

## Roadmap

### MVP (Current)

* [x] Core streaming memory engine
* [x] WASM compilation support
* [x] Python and Go bindings
* [x] Adaptive threshold system
* [x] Production error handling
* [x] Consolidation and graph integration system
* [x] sqlite-vec integration for vector search
* [x] sqlite-graph integration for knowledge graphs
* [x] EmbeddingGemma ONNX deployment

### Future Releases

**v1.1**: Enhanced consolidation algorithms and graph reasoning

* Advanced Cypher pattern matching
* Graph-based memory importance scoring
* Multi-hop reasoning over knowledge graph

**v1.2**: Additional multimodal encoders (video, documents)

* Video frame encoders with temporal consistency
* Document structure-aware embeddings
* Code understanding and semantic search

**v1.3**: Distributed memory for multi-user scenarios

* Sharded vector stores across nodes
* Distributed graph queries
* Collaborative memory sharing

**v2.0**: Advanced algorithms from research prototype

* Full 27-algorithm suite from memstream
* Emotional consolidation and metacognition
* Working memory gates and predictive pre-activation

***

## References & Dependencies

### Core Technologies

* **[sqlite-vec](https://github.com/asg017/sqlite-vec)**: High-performance vector similarity search extension for SQLite
  * Native C++ implementation for minimal overhead
  * Provides cosine, dot product, and L2 distance metrics
  * Supports quantization (INT8, INT4) for memory efficiency
  * WASM-compatible for browser deployment
  * Incremental index building for streaming workloads

* **[sqlite-graph](https://github.com/agentflare-ai/sqlite-graph)**: Cypher-compatible graph extension for SQLite
  * Native property graph model
  * Cypher query language support
  * Optimized joins between vector and graph data
  * Transactional consistency

* **[EmbeddingGemma-300M ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)**: MRL-capable text embedding model
  * 1024-dimensional full space with MRL truncation support
  * Production target: 256d truncated + renormalized space
  * INT8 quantized for 4x memory reduction
  * CPU-optimized inference (<50ms target for text)
  * Cross-platform ONNX deployment

* **[Gemma 3n-E2B-it ONNX](https://huggingface.co/onnx-community/gemma-3n-E2B-it-ONNX)**: Multimodal encoders
  * Frozen vision and audio encoders (pre-trained)
  * Vision: 224×224 RGB input with ImageNet normalization
  * Audio: 64-mel log-spectrogram input (320ms, 33 frames)
  * INT8 quantized for production deployment
  * Target latency: <30ms per encoder on CPU

### Additional Resources

* **ONNX Runtime**: <https://onnxruntime.ai/>
* **WebAssembly Component Model**: <https://component-model.bytecodealliance.org/>
* **Emscripten (C++ to WASM)**: <https://emscripten.org/>
* **WebAssembly C++ Guide**: <https://developer.mozilla.org/en-US/docs/WebAssembly/C_to_Wasm>

***

## License

MIT License - See LICENSE file for details.

***

**Document Version**: 1.0.1\
**Last Updated**: November 7, 2025\
**Status**: Production MVP Design
