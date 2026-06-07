# Constrained Decoding in LiteRT-LM

LiteRT-LM supports constrained decoding, allowing you to enforce specific
structures on the model's output. This is particularly useful for tasks like:

-   **Function Calling**: Ensuring the model outputs a valid function call
    matching a specific schema.
-   **Structured Data Extraction**: Forcing the model to adhere to a specific
    format (e.g., specific regex patterns).
-   **Grammar Enforcement**: Using context-free grammars (via Lark) to guide
    generation.

This document explains how to enable, configure, and use constrained decoding in
your application.

## Enabling Constrained Decoding

To use constrained decoding, you must enable it in the `ConversationConfig` when
creating your `Conversation` instance.

```cpp
#include "runtime/conversation/conversation.h"

// ...

ConversationConfig::Builder builder;
builder.SetEnableConstrainedDecoding(true);

// Set a ConstraintProviderConfig in the ConversationConfig::Builder.
// This line set the ConstraintProvider to LLGuidance with default settings.
builder.SetConstraintProviderConfig(LlGuidanceConfig());

auto config = builder.Build(*engine);
```

### Constraint Providers

LiteRT-LM supports different backends for constrained decoding, configured via
`ConstraintProviderConfig`:

1.  **LLGuidance (`LlGuidanceConfig`)**: Uses the
    [LLGuidance](https://github.com/guidance-ai/llguidance) library. Supports
    Regex, JSON Schema, and Lark grammars.
2.  **External (`ExternalConstraintConfig`)**: Allows passing a pre-constructed
    `Constraint` object per-request. Useful for custom C++ constraint
    implementations.

## Using Constraints in `SendMessage`

Once enabled, you can apply constraints to individual messages using the
`decoding_constraint` field in the `OptionalArgs` struct passed to `SendMessage`
or `SendMessageAsync`. This field is of type `std::optional<ConstraintArg>`.

### 1. LLGuidance Constraints

LLGuidance constraints can be specified as Regex, JSON Schema, or Lark grammars.

#### Regex Constraint

Constrain the output to match a regular expression.

```cpp
#include "runtime/components/constrained_decoding/llg_constraint_config.h"

// ...

LlGuidanceConstraintArg constraint_arg;
constraint_arg.constraint_type = LlgConstraintType::kRegex;
// Example: Force output to be a sequence of 'a's followed by 'b's
constraint_arg.constraint_string = "a+b+";

auto response = conversation->SendMessage(
    user_message,
    {.decoding_constraint = constraint_arg}
);
```

#### JSON Schema Constraint

Constrain the output to be a valid JSON object matching a schema.

```cpp
LlGuidanceConstraintArg constraint_arg;
constraint_arg.constraint_type = LlgConstraintType::kJsonSchema;
// Example: Simple JSON object with a "name" field
constraint_arg.constraint_string = R"({
  "type": "object",
  "properties": {
    "name": {"type": "string"}
  },
  "required": ["name"]
})";

auto response = conversation->SendMessage(
    user_message,
    {.decoding_constraint = constraint_arg}
);
```

#### Lark Grammar Constraint

Constrain the output to follow a Lark grammar.

```cpp
LlGuidanceConstraintArg constraint_arg;
constraint_arg.constraint_type = LlgConstraintType::kLark;
// Example: A simple calculator grammar
constraint_arg.constraint_string = R"(
    start: expr
    expr: atom
        | expr "+" atom
        | expr "-" atom
        | expr "*" atom
        | expr "/" atom
        | "(" expr ")"
    atom: /[0-9]+/
    WS: /[ \t\n\f]+/
    %ignore WS
)";

auto response = conversation->SendMessage(
    user_message,
    {.decoding_constraint = constraint_arg}
);
```

### 2. External Constraints

If you have a custom implementation of the `Constraint` interface (e.g., a
highly specialized C++ state machine), you can use `ExternalConstraintArg`.

Prerequisite: You must have initialized `Conversation` with
`ExternalConstraintConfig`.

```cpp
// 1. Initialize with ExternalConstraintConfig
auto config = ConversationConfig::Builder()
    .SetEnableConstrainedDecoding(true)
    .SetConstraintProviderConfig(ExternalConstraintConfig())
    .Build(*engine);
auto conversation = Conversation::Create(*engine, config);

// 2. Create your custom constraint (must implement litert::lm::Constraint)
class MyCustomConstraint : public Constraint {
    // Implement Start, ComputeNext, etc.
};
auto my_constraint = std::make_unique<MyCustomConstraint>();

// 3. Pass it to SendMessage
ExternalConstraintArg external_constraint;
external_constraint.constraint = std::move(my_constraint);

auto response = conversation->SendMessage(
    user_message,
    {.decoding_constraint = std::move(external_constraint)}
);
```

## API Reference

### `ConstraintProviderConfig`

A variant configuration passed to `ConversationConfig`.

-   `LlGuidanceConfig`: Configures LLGuidance.
    -   `eos_id`: Optional override for the End-of-Sequence token ID.
-   `ExternalConstraintConfig`: Empty struct (marker) to enable external
    constraints.

### `ConstraintArg`

A variant argument passed via `OptionalArgs` to `SendMessage`.

-   `LlGuidanceConstraintArg`:
    -   `constraint_type`: `kRegex`, `kJsonSchema`, or `kLark`.
    -   `constraint_string`: The pattern/schema/grammar string.
-   `ExternalConstraintArg`:
    -   `constraint`: `std::unique_ptr<Constraint>`. Ownership is transferred to
        the valid decoder for that request.
