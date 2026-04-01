#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Seed LABEL memories from a static label bank file.
/// 
/// Runs once per processor instance (guarded by context flag).
/// Labels are stored as MEMORIES(kind='LABEL') with embeddings.
class SeedLabelBank : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
