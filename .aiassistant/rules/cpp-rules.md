---
apply: always
---

# C++ Programming Guidelines for WASM and Edge Devices (Google C++ Style Guide Compatible)

## Basic Principles

* ALWAYS use English for all code and documentation.
* ALWAYS declare the type of each variable and function (parameters and return value).
* ALWAYS create necessary types and classes.
* ALWAYS use Doxygen style comments to document public classes and methods.
* NEVER leave blank lines within a function.
* ALWAYS follow the one-definition rule (ODR).
* ALWAYS design for low resource consumption suitable for edge devices and WASM environments, prioritizing minimal memory usage and efficient computation.

## Nomenclature

* ALWAYS use PascalCase for classes and structures.
* ALWAYS use snake\_case for variables (Google C++ Style Guide compliant).
* ALWAYS use PascalCase for functions and methods (Google C++ Style Guide compliant).
* ALWAYS use ALL\_CAPS for macros.
* ALWAYS use snake\_case for file and directory names.
* ALWAYS use kPascalCase for constants (Google C++ Style Guide compliant).
* ALWAYS use UPPERCASE for environment variables (may not be available in WASM builds).
* NEVER use magic numbers and ALWAYS define constants.
* ALWAYS use explicit variable names that clearly describe their purpose and type.
* ALWAYS start each function with a verb.
* ALWAYS use verbs for boolean variables. Example: isLoading, hasError, canDelete, etc.
* ALWAYS use complete words instead of abbreviations and ensure correct spelling.
    * Except for standard abbreviations like API, URL, etc.
    * Except for well-known abbreviations:
        * i, j, k for loops
        * err for errors
        * ctx for contexts
        * req, res for request/response parameters

## Functions

* ALWAYS write short functions with a single purpose. Less than 20 instructions.
* ALWAYS name functions with a verb and something else.
* If it returns a boolean, use isX or hasX, canX, etc.
* If it doesn't return anything (void), use executeX or saveX, etc.
* ALWAYS avoid nesting blocks by:
    * Early checks and returns.
    * Extraction to utility functions.
* ALWAYS use standard library algorithms (std::for\_each, std::transform, std::find, etc.) to avoid function nesting.
* ALWAYS use lambda functions for simple operations.
* ALWAYS use named functions for non-simple operations.
* ALWAYS use default parameter values instead of checking for null or nullptr.
* ALWAYS reduce function parameters using structs or classes
    * Use an object to pass multiple parameters.
    * Use an object to return multiple results.
    * Declare necessary types for input arguments and output.
* ALWAYS prefer templates and variadic templates over std::vector as function arguments for better type safety and performance.
* ALWAYS use a single level of abstraction.

## Data

* NEVER abuse primitive types and ALWAYS encapsulate data in composite types.
* NEVER validate data in functions and ALWAYS use classes with internal validation.
* ALWAYS prefer immutability for data.
* ALWAYS prefer aggregates over constructors for data storage.
* ALWAYS use const for data that doesn't change.
* ALWAYS use constexpr for compile-time constants.
* ALWAYS use std::optional for possibly null values.
* ALWAYS prefer fixed sizes over allocations where possible.

## Classes

* ALWAYS follow SOLID principles.
* ALWAYS prefer composition over inheritance.
* ALWAYS declare interfaces as abstract classes or concepts.
* ALWAYS write small classes with a single purpose.
    * Less than 200 instructions.
    * Less than 10 public methods.
    * Less than 10 properties.
* ALWAYS use the Rule of Five (or Rule of Zero) for resource management.
* ALWAYS make member variables private and provide getters/setters where necessary.
* ALWAYS use const-correctness for member functions.

## Modern C++ Features

* ALWAYS use `static_cast`, `const_cast`, `reinterpret_cast`, or `std::bit_cast` instead of C-style casts
* ALWAYS use `nullptr` consistently for null pointers
* ALWAYS use `const` member functions for read-only operations
* ALWAYS use `noexcept` for functions that guarantee no exceptions
* ALWAYS use `std::optional<T>` instead of separate boolean flags for optional values
* ALWAYS use `std::variant` for type-safe unions
* ALWAYS use structured bindings for cleaner tuple/pair decomposition (C++17+)
* ALWAYS use C++20 coroutines (co\_await, co\_yield, co\_return) for async operations and cooperative concurrency
* ALWAYS use dependency injection pattern with constructor parameters
* ALWAYS decompose complex constructors into simpler initialization functions
* NEVER use C-style casts `(T)expr`
* NEVER use `NULL` or `0` as null pointers
* NEVER mark runtime values as `constexpr`
* NEVER use getters and setters unnecessarily - prefer constructor injection and dependency injection over runtime configuration
* NEVER use `auto` when it obscures type information for readers
* NEVER use std::thread, std::mutex, or other threading primitives (use coroutines instead)

## Exceptions

* Exceptions may not be fully supported or efficient in WASM and edge device builds (try-catch might require special flags or be disabled); prefer error codes, std::optional, or std::expected for all error handling to ensure portability across targets.
* Only use exceptions for truly unexpected errors in native builds where supported.
* If you catch an exception, it should be to:
    * Fix an expected problem.
    * Add context.
    * Otherwise, use a global handler.
* Use std::optional, std::expected, or error codes for expected failures.

## Error Handling

* ALWAYS use exceptions for exceptional conditions or `std::expected`/`std::optional` for expected failures.
* ALWAYS implement proper resource cleanup patterns (RAII).
* ALWAYS ensure interfaces are const-correct and thread-safe where appropriate.
* ALWAYS validate inputs and handle edge cases explicitly.

## Compile-Time Optimization

* ALWAYS mark compile-time computable values as `constexpr`
* ALWAYS use template metaprogramming for type computation and size calculations
* ALWAYS use `consteval` for functions that must be evaluated at compile time (C++20+)
* ALWAYS use template recursion or C++17 fold expressions instead of runtime loops for compile-time known values
* ALWAYS use `std::array` for compile-time sized fixed-size containers
* ALWAYS use `if constexpr` for compile-time branching
* ALWAYS prefer compile-time polymorphism (templates) over runtime polymorphism when appropriate

## Preprocessor Usage

* ALWAYS use `constexpr` variables or `enum class` constants instead of object-like macros
* ALWAYS use `constexpr` functions or templates instead of function-like macros
* ALWAYS use `if constexpr` or template specialization instead of preprocessor conditionals when possible

## Memory Management

* ALWAYS prefer stack allocation when object lifetime is clear and bounded.
* ALWAYS use smart pointers (`std::unique_ptr`, `std::shared_ptr`) for heap-allocated objects.
* ALWAYS use references instead of pointers for required dependencies in constructors.
* ALWAYS use RAII wrappers for resource management.
* ALWAYS prefer `std::make_unique` and `std::make_shared` over raw `new`.
* ALWAYS use move semantics for transferring ownership.
* ALWAYS prefer smart pointers (std::unique\_ptr, std::shared\_ptr) over raw pointers.
* ALWAYS use RAII (Resource Acquisition Is Initialization) principles.
* ALWAYS avoid memory leaks by proper resource management.
* ALWAYS use std::vector and other standard containers instead of C-style arrays.
* NEVER use raw `new` and `delete` (use smart pointers or RAII wrappers)
* NEVER use `malloc` and `free` in C++ code
* NEVER use raw pointers for ownership (use smart pointers)
* NEVER mix ownership semantics (use clear ownership through types)
* NEVER use pointers for required dependencies in constructors (use references)

## Testing and Code Quality (Catch2 Framework)

* ALWAYS follow the Arrange-Act-Assert convention for tests.
* ALWAYS name test variables clearly (e.g., `auto result = ...`, `const auto expected = ...`).
* ALWAYS use descriptive TEST\_CASE names with BDD-style GIVEN/WHEN/THEN sections where appropriate.
* ALWAYS write unit tests for each public function using REQUIRE/CHECK assertions.
* ALWAYS use separate mocking libraries (e.g., Trompeloeil, Google Mock) for test doubles, not Catch2 built-ins.
    * Except for third-party dependencies that are not expensive to execute.
* ALWAYS write integration tests for each module as separate TEST\_CASE blocks.
* ALWAYS use Catch2's SCENARIO macro for Given-When-Then style behavioral tests.
* ALWAYS organize tests with SECTION blocks for related test cases within a TEST\_CASE.
* ALWAYS ensure comprehensive testing for all components.
* ALWAYS use structured logging with appropriate severity levels.
* ALWAYS provide mock interfaces for unit testing.
* ALWAYS document non-obvious design decisions and assumptions.
* ALWAYS use modern tooling (sanitizers, static analyzers, formatters).

## Project Structure

* ALWAYS use modular architecture.
* ALWAYS organize code into logical directories:
    * include/ for header files
    * src/ for source files
    * test/ for test files
    * lib/ for libraries
    * doc/ for documentation
* ALWAYS use CMake or similar build system.
* ALWAYS separate interface (.h) from implementation (.cpp).
* ALWAYS use namespaces to organize code logically.
* ALWAYS create a core namespace for foundational components.
* ALWAYS create a utils namespace for utility functions.
* ALWAYS match namespace names to project directory structure.
* ALWAYS use namespace-qualified access for clarity and to avoid name collisions.
* ALWAYS use clean class names without namespace prefixes (e.g., `namespace foo { class Client {} }` not `namespace foo { class FooClient {} }`).
* ALWAYS rely on namespace qualification for context (e.g., `foo::Client` not `foo::FooClient`).
* NEVER use redundant namespace prefixes in class names (e.g., avoid `namespace foo { class FooClient {} }`).

## Standard Library

* ALWAYS use the C++ Standard Library whenever possible (ensure WASM compatibility).
* ALWAYS prefer std::string over C-style strings.
* ALWAYS use std::vector, std::map, std::unordered\_map, etc. for collections.
* ALWAYS use std::optional, std::variant, std::any for modern type safety.
* For file operations, ALWAYS avoid std::filesystem in WASM builds (use Emscripten-specific APIs or virtual file systems).
* ALWAYS use std::chrono for time-related operations (with awareness of WASM timing limitations).

## Concurrency

* ALWAYS use C++20 coroutines as the preferred and only concurrency primitive for cross-platform compatibility.
* Coroutines work on any platform (native, WASM, edge devices) and provide cooperative multitasking.
* For WASM builds: coroutines are driven manually (single-threaded execution only) to ensure compatibility.
* NEVER use std::thread, std::mutex, std::lock\_guard, or other threading primitives as they are not WASM-compatible.
* NEVER use Emscripten/Web Workers or any platform-specific threading mechanisms.
* ALWAYS design for cooperative concurrency where coroutines yield control explicitly rather than preemptive scheduling.
* ALWAYS use std::generator, std::lazy, or custom coroutine types for async operations.
* NEVER allow data races by ALWAYS ensuring single-threaded execution and explicit coroutine coordination.
* ALWAYS use coroutine-based task composition instead of thread-based parallelism.

## General Anti-Patterns

* NEVER use unbounded recursion without clear termination guarantees
* NEVER ignore return values from critical operations
* NEVER use global mutable state without explicit synchronization
* NEVER assume initialization order of global/static objects across translation units
* NEVER use deeply nested if-else chains (prefer pattern matching, polymorphism, or lookup tables)
* NEVER mix concerns in a single function or class

## WASM and Edge Devices

* ALWAYS target WebAssembly (WASM) for browser and serverless environments, using Emscripten for compilation.
* For edge devices, ALWAYS assume constrained resources: limited CPU, memory (< 1GB), and no persistent filesystem.
* ALWAYS minimize binary size by avoiding unnecessary dependencies and using lightweight alternatives.
* ALWAYS avoid or polyfill unsupported features: full exceptions, threads, filesystem I/O.
* ALWAYS test regularly for WASM compatibility and edge performance, using tools like wasm-opt for optimization.
* ALWAYS prioritize deterministic behavior without relying on system clocks or external resources where possible.

***

Note: This rule complements the Google C++ Style Guide and is designed to be compatible with WASM and edge device constraints. Where conflicts existed (naming conventions), Google C++ Style Guide takes precedence. WASM-specific and edge device limitations are noted throughout. This rule focuses on additional best practices, design principles, and project organization guidelines that enhance code quality and maintainability across native, WebAssembly, and edge targets. Modern C++ features, compile-time optimization, and memory management best practices are emphasized while maintaining Google Style Guide compatibility.
