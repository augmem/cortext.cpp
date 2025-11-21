# Cortext Signal Processing Design

This document outlines the design for the `SignalProcessor`, a system responsible for processing incoming signals, evaluating them based on cognitive models, and storing them efficiently as memories. This design is derived from `docs/algorithms.md` and leverages the transactional capabilities of `include/cortext/store/sqlite_store.hpp`.

## 1. Overview

The signal processing model acts as a model of human memory encoding. It processes a continuous stream of data (`signals`) and segments it into meaningful, coherent chunks called `Episodes`. Each signal is evaluated for salience, and the decision to store it as a memory is managed by a dynamic thresholding system.

The primary goals of this architecture are:

* **Conceptual Clarity**: To create a system that is a direct implementation of the cognitive concepts in `algorithms.md`.
* **Performance**: To ensure high throughput by batching database operations on episode boundaries.
* **Modularity**: To separate concerns into distinct, testable components.

## 2. Core Concepts

### The SignalProcessor

The `SignalProcessor` is the central orchestrating class for the entire memory encoding process. It is a stateful, long-lived object that hosts and manages the pipeline.

**Responsibilities:**

* **State Management**: Owns and manages the `ProcessorContext`, which holds all dynamic variables (EWMAs, thresholds, rolling windows, etc.) that evolve over time.
* **Configuration**: Initialized with the three core knobs (Focus, Sensitivity, Stability) and a `Store` instance.
* **Pipeline Orchestration**: Exposes a primary `Process(const Signal&)` method that pushes a signal through the configured pipeline of encoders.
* **Transaction Lifecycle**: Manages the `SQLiteTransaction` for the current episode, committing and starting new ones as needed.

#### Configuration Knobs

Only the three core knobs are user-configurable:

* `focus` (F)
* `sensitivity` (S)
* `stability` (T)

All other internal parameters (e.g., RLS forgetting, update interval, rate windowing) are derived from these knobs.

### The Episode

An `Episode` is a sequence of related signals that belong to a single, coherent context. The term is chosen deliberately to align with the concept of **episodic memory** in cognitive science, a core influence on the system's design.

An episode is not an arbitrary fragment of the data stream. Its beginning and end are algorithmically determined by **Algorithm 12: Trajectory Drift**. When the semantic context of incoming signals drifts significantly, the current episode ends, and a new one begins.

**Technical Importance:**

* **Performance**: The episode defines the scope of a single database transaction. All memory writes generated during an episode are buffered and committed atomically when the episode ends. This batching strategy is critical for performance, minimizing disk I/O and maximizing throughput as recommended in `docs/algorithms.md`.
* **Context Management**: An episode provides a clear boundary for context-sensitive operations like cache invalidation.

## 3. Architecture and Data Flow

The system is designed as a modular, composable pipeline hosted by the `SignalProcessor`. This allows for flexibility in defining the sequence of processing steps.

### Core Architectural Components

* **`IOperation`**: An interface representing a single, atomic step in the processing model (e.g., calculating coherence, updating a knob-derived metric). These are the "operations" of our processing model.
* **Composers (`OperationSet`)**: A special operation that arranges other operations. `OperationSet` runs its children one by one.
* **`OperationContext`**: A stateful object that is passed to every operation. It contains the input `Signal`, a reference to the global `ProcessorContext`, and a buffer for intermediate results.

### The `OperationContext`

The `OperationContext` is a transient object created for each signal that is processed. It acts as a container for all the data an operation might need to read or write, decoupling the operations from the `SignalProcessor` itself.

**Responsibilities:**

* Provide read-only access to the input `Signal`.
* Provide access to the mutable, long-lived `ProcessorContext`.
* Provide a mechanism for operations to add `BufferedWriteInstruction`s to the episode's transaction buffer.
* Provide a "scratchpad" for operations to store intermediate values (e.g., calculated metrics like `coherence` or `novelty`) that subsequent operations in the same processing run might need.

### Data Flow

1. **Operation Set Construction**: The `SignalProcessor` is constructed with a user-defined set of operations, structured as a tree of `IOperation` objects (e.g., an `OperationSet` containing other operations and composers).
2. **Signal Arrival**: The process begins when `SignalProcessor::Process(signal)` is called.
3. **Context Creation**: An `OperationContext` is created for this specific signal pass.
4. **Operation Execution**: The `SignalProcessor` invokes the root operation, passing it the `OperationContext`.
5. **Operation Processing**: Each `IOperation` performs its specific task.
6. **Boundary Check & Commit**: A dedicated operation is responsible for checking the episode boundary condition (Alg 12). If a boundary is detected, the `SignalProcessor` is notified to finalize the current episode.

This design transforms the `SignalProcessor` into a passive host for a flexible, user-defined set of operations.

## 4. Proposed C++ Interface

The following C++ headers sketch the proposed architecture, adhering to the Google C++ Style Guide.

### Operation Context

File: `include/cortext/context.hpp`

```cpp
#pragma once

#include <any>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cortext {

// Forward declarations
struct Signal;
struct ProcessorContext;
struct BufferedWriteInstruction;

/// @brief Contains all the state for a single signal processing run.
///
/// This object is created by the SignalProcessor for each signal and is passed
/// through the entire operation set.
class OperationContext {
public:
  /// @brief Constructs an OperationContext.
  /// @param signal The input signal being processed.
  /// @param context The long-lived context of the SignalProcessor.
  /// @param write_buffer A reference to the episode's database write buffer.
  OperationContext(const Signal& signal, ProcessorContext& context,
                    std::vector<BufferedWriteInstruction>& write_buffer);

  // --- Accessors ---

  /// @brief Gets the input signal.
  const Signal& GetSignal() const { return signal_; }

  /// @brief Gets the processor's long-lived context.
  ProcessorContext& GetContext() { return context_; }

  /// @brief Gets the processor's long-lived context (const version).
  const ProcessorContext& GetContext() const { return context_; }

  // --- Output ---

  /// @brief Adds a database instruction to the buffer for the current episode.
  /// @param op The buffered write instruction to add.
  void AddWriteInstruction(BufferedWriteInstruction op);

  // --- Scratchpad for Intermediate Results ---

  /// @brief Sets an intermediate result for later operations to use.
  /// @param key The name of the result (e.g., "coherence_score").
  /// @param value The value of the result.
  void SetIntermediateResult(const std::string& key, std::any value);

  /// @brief Gets an intermediate result set by a previous operation.
  /// @param key The name of the result to retrieve.
  /// @return The value of the result, or std::nullopt if not found.
  std::optional<std::any> GetIntermediateResult(const std::string& key) const;

private:
  const Signal& signal_;
  ProcessorContext& context_;
  std::vector<BufferedWriteInstruction>& write_buffer_;
  std::map<std::string, std::any> intermediate_results_;
};

} // namespace cortext
```

### Operation Interface

File: `include/cortext/pipeline/operation.hpp`

```cpp
#pragma once

#include <memory>

namespace cortext {

class OperationContext;

/// @brief Defines the interface for an operation in the signal processing model.
///
/// An operation represents a single, atomic step in the process.
class IOperation {
public:
  virtual ~IOperation() = default;

  /// @brief Executes the operation.
  /// @param context The context object for the current signal.
  virtual void Execute(OperationContext& context) const = 0;
};

} // namespace cortext
```

### SignalProcessor Class

File: `include/cortext/processor.hpp`

```cpp
#pragma once

#include "store/store.hpp"
#include "pipeline/operation.hpp" // Include the new operation interface
#include <memory>
#include <vector>

namespace cortext {

// Forward declarations
struct Signal;
struct ProcessorContext;
struct BufferedWriteInstruction;

/// @brief Orchestrates the signal processing.
///
/// The SignalProcessor hosts a composable set of operations that evaluate
/// signals and store them as memories within episodic transactions.
class SignalProcessor {
public:
  /// @brief Configuration for the Processor's behavior.
  struct Config {
    double focus = 0.5;
    double sensitivity = 0.5;
    double stability = 0.5;
  };

  /// @brief Constructs a SignalProcessor with a defined set of operations.
  /// @param config The initial knob settings.
  /// @param store A shared pointer to the underlying database store.
  /// @param root_operation The root of the operation tree to execute.
  SignalProcessor(const Config& config, std::shared_ptr<Store> store,
                  std::unique_ptr<IOperation> root_operation);

  ~SignalProcessor();

  // Disable copy and move semantics.
  SignalProcessor(const SignalProcessor&) = delete;
  SignalProcessor& operator=(const SignalProcessor&) = delete;
  SignalProcessor(SignalProcessor&&) = delete;
  SignalProcessor& operator=(SignalProcessor&&) = delete;

  /// @brief Processes a single signal by executing the instruction set.
  /// @param signal The signal to process.
  void Process(const Signal& signal);

private:
  void StartNewEpisode();
  void FinalizeEpisode();

  Config config_;
  std::shared_ptr<Store> store_;
  std::unique_ptr<Transaction> episode_transaction_;
  std::unique_ptr<IOperation> root_operation_;

  std::unique_ptr<ProcessorContext> context_;
  std::vector<BufferedWriteInstruction> write_buffer_;
};

} // namespace cortext
```

### Example Operation Set Construction

A set of operations could be constructed in C++ like this:

```cpp
// Note: OperationSet would use a variadic template constructor
// to enable this clean, vector-free syntax.

auto root_operation = std::make_unique<OperationSet>(
    std::make_unique<UpdateFocusPriorsOperation>(),
    std::make_unique<UpdateSensitivityPriorsOperation>(),
    std::make_unique<ComputeCoherenceOperation>(),
    std::make_unique<UpdateThresholdOperation>()
);

auto processor = std::make_unique<SignalProcessor>(config, store, std::move(root_operation));
```

### Design Rationale: Polymorphism, Ownership, and Move Semantics

The design relies on `std::unique_ptr<IOperation>` to manage the lifecycle of operation objects. This choice is deliberate:

1. **Polymorphism**: Pointers are required to store different concrete operation types (`UpdateFocusPriorsOperation`, `OperationSet`, etc.) in a single, uniform way. Using `std::unique_ptr<IOperation>` allows the operation set to be composed of heterogeneous components.

2. **Ownership**: `std::unique_ptr` provides exclusive, automatic ownership (RAII). The `SignalProcessor` owns the root operation, which in turn owns its children. This prevents memory leaks without manual resource management.
