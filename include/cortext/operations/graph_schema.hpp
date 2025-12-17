/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Creates idempotent schema for graph and extraction tables.
///
/// This operation registers schema migrations at startup.
class EnsureGraphSchema : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
  void CollectSchema (cortext::store::SchemaRegistry &registry) const override;
};

} // namespace cortext::operations
