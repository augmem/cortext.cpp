/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Creates idempotent schema for graph and extraction tables.
///
/// This operation only emits DDL statements into the episode write buffer.
/// It does not populate any rows.
class EnsureGraphSchema : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations

