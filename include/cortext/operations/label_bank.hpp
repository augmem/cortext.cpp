#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Load LABEL memories from a static label bank asset.
///
/// Runs once per processor instance (guarded by context flag). If the active
/// database already has the current label bank imported, this operation only
/// rebuilds the in-memory label caches. Otherwise it imports labels from a
/// prebuilt SQLite label bank or falls back to the JSONL label bank manifest.
class LoadLabelBank : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
