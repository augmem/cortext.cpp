# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cortext is a C++17 intelligent context system that monitors LLM token generation in real-time and dynamically injects relevant context through semantic retrieval and knowledge graph augmentation. It implements a closed-loop cognitive system using a three-knob adaptive architecture (Focus, Sensitivity, Stability).

## Build Commands

```bash
# Configure and build (Debug)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j

# Run tests (project tests only, excludes dependency tests)
ctest --test-dir build -R cortext_tests --output-on-failure
# Or run the test binary directly
./build/tests/cortext_tests

# Run a specific test by name
./build/tests/cortext_tests "[test_name]"

# Build for WebAssembly
cmake --preset wasm
cmake --build --preset wasm

# Build with ONNX Runtime (ImageBind encoder)
cmake -S . -B build -DCORTEXT_ENABLE_IMAGEBIND_ORT=ON
cmake --build build -j

# Build examples
cmake -S . -B build -DCORTEXT_BUILD_EXAMPLES=ON
cmake --build build -j
```

## Architecture

### Three-Knob System (F, S, T)

All adaptive behavior derives from three knobs in [0,1]:
- **Focus (F)**: Perceptual selectivity and precision. Higher F narrows attention, raises relevance weighting, reduces retrieval breadth
- **Sensitivity (S)**: Plasticity and affect gain. Higher S speeds learning, raises emotional/novelty gain, increases write-rate responsiveness
- **Stability (T)**: Temporal persistence and inertia. Higher T lengthens half-lives, widens hysteresis, slows parameter updates

Knob-derived functions are defined in `include/cortext/core/knobs.hpp` and follow the formulas in `docs/algorithms.md` Section 0.

### Core Components

- **`src/cortext.cpp`**: Main API class (`Cortext`) for multimodal context processing
- **`src/signal_processor.cpp`**: Signal processing pipeline orchestrating operations
- **`src/store.cpp`**: SQLite-based memory store with vector search (sqlite-vec) and graph (sqlite-graph)
- **`src/operations/`**: Individual algorithm implementations (27+ algorithms from docs/algorithms.md)
- **`src/telemetry/`**: OpenTelemetry instrumentation (API-only, SDK providers configured by consumers)
- **`src/encoder/`**: Multimodal embedding encoders (ImageBind via ONNX Runtime)

### Key Headers

- `include/cortext/core/algorithms.hpp`: Core math primitives (Lerp, Clamp, Sigmoid, EWMA, cosine similarity)
- `include/cortext/core/knobs.hpp`: All knob-derived parameter functions
- `include/cortext/signal.hpp`: Signal types passed through the processing pipeline
- `include/cortext/store/store.hpp`: Memory store interface

### SQLite Extensions

Embedded by default (controlled via CMake options):
- **sqlite-vec**: Vector similarity search for embeddings
- **sqlite-graph**: Cypher-compatible knowledge graph
- **sqlite-objstore**: Transactional blob storage for multimodal payloads

## Code Style

Follow the Google C++ Style Guide with these specifics:
- **Naming**: PascalCase for types/functions, snake_case for variables, kPascalCase for constants
- **Files**: Lowercase with underscores (e.g., `my_class.hpp`, `my_class.cpp`)
- **Header guards**: Use `#pragma once`
- **Indentation**: 2 spaces, no tabs
- **Line length**: 80 characters maximum
- **Memory**: Use smart pointers (`std::unique_ptr`, `std::shared_ptr`), never raw `new`/`delete`
- **Concurrency**: Use C++20 coroutines only, no `std::thread`/`std::mutex` (WASM compatibility)
- **Error handling**: Prefer `std::optional`/`std::expected` over exceptions for expected failures

## Testing

Tests use Catch2 v3 with BDD-style GIVEN/WHEN/THEN sections:

```cpp
TEST_CASE("Operation behavior", "[operation_name]") {
  SECTION("GIVEN some precondition") {
    // Arrange
    auto input = CreateInput();

    // Act
    auto result = PerformOperation(input);

    // Assert
    REQUIRE(result.has_value());
    CHECK(result->value > 0.0);
  }
}
```

Test files are named `<component>.test.cpp` in the `tests/` directory.

## Algorithm Implementation Pattern

Operations follow a consistent pattern (see `src/operations/` for examples):

1. Define parameters struct with knob-derived defaults
2. Implement pure function taking state + parameters
3. Return updated state (immutable update pattern)
4. Integrate via `SignalProcessor` operation chain

Example operation header structure:
```cpp
namespace cortext::operations {
struct MyOperationParams {
  double some_param;
  static MyOperationParams FromKnobs(double F, double S, double T);
};

struct MyOperationResult { /* outputs */ };

MyOperationResult ComputeMyOperation(const InputState& state, const MyOperationParams& params);
}
```

## Key Documentation

- `docs/algorithms.md`: Complete algorithmic specification (27 algorithms, knob derivations)
- `docs/design.md`: Production MVP design, architecture, and technology stack
