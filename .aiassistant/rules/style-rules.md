---
apply: always
---

# Google C++ Style Guide

This project follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). All C++ code must adhere to these standards.

## Header Files

* **Header guards**: Prefer `#pragma once` over `#ifndef`/`#define`/`#endif` include guards
* **Include order**: Related header first, then C system headers, then C++ standard library, then other libraries
* **Forward declarations**: Use when possible instead of includes to reduce compilation time
* **Inline functions**: Define in header only if ≤10 lines and used in multiple source files

## Naming Convention

* **Files**: Lowercase with underscores: `my_class.h`, `my_class.cpp`
* **Types**: PascalCase: `class MyClass`, `struct MyStruct`, `enum class MyEnum`
* **Variables**: snake\_case: `int my_variable;`
* **Constants**: kPascalCase: `const int kMaxSize = 100;`
* **Functions**: PascalCase: `void MyFunction();`
* **Namespaces**: lowercase: `namespace my_namespace {`
* **Macros**: ALL\_CAPS\_WITH\_UNDERSCORES: `#define MY_MACRO(x) ...`

## Formatting

* **Indentation**: 2 spaces, no tabs
* **Line length**: 80 characters maximum
* **Braces**: Always on same line as statement, except functions/classes
* **Spaces**: Before/after operators, after commas, around colons in inheritance

## Best Practices

* **Smart pointers**: Use `std::unique_ptr`, `std::shared_ptr` instead of raw pointers
* **RAII**: Resource acquisition is initialization
* **Exceptions**: Use for exceptional cases, not for control flow
* **Const correctness**: Use `const` wherever possible
* **Avoid**: Global variables, `using namespace` in headers, C-style casts

## Code Structure

* **Class organization**: Public first, then protected, then private
* **Function organization**: Inputs first, then outputs in function parameters
* **Comments**: Complete sentences, Doxygen style for APIs
* **Error handling**: Return error codes or use exceptions consistently

## Modern C++ Features

* **C++11/14/17**: Use modern features when appropriate
* **auto**: Use when type is obvious from context
* **Range-based for**: Prefer over traditional loops
* **Lambda expressions**: Use for short anonymous functions

## Testing

* **Unit tests**: Required for all public APIs
* **Test naming**: PascalCase for test fixtures, snake\_case for test methods
* **Test files**: Same name as implementation with `_test` suffix

Violations of these guidelines will be flagged during code review and must be corrected before merging.
