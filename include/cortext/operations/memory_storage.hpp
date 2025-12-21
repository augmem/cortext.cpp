#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Conditional memory storage based on write gate decision.
///
/// This operation MUST run AFTER ComputeWriteGate.
///
/// If write_decision=true:
///   1. Stores per-signal payloads in objstore → SIGNALS.blob_id
///   2. Inserts per-signal embeddings → SIGNALS.embedding_id
///   3. Inserts a memory-level embedding and MEMORIES row
///   4. Sets stored_embedding_id in OperationContext
///
/// If write_decision=false:
///   - Discards entirely, no storage occurs.
///
/// If write_decision=true but no payload:
///   - Logs warning, no storage occurs.
class MemoryStorage : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
