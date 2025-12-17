#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Conditional memory storage based on write gate decision.
///
/// This operation MUST run AFTER ComputeWriteGate.
///
/// If write_decision=true and payload is present:
///   1. Stores payload in objstore → blob_id
///   2. Stores embedding in embeddings table → embedding_id
///   3. Buffers memory_index insert with blob_id reference
///   4. Buffers memory_feedback initialization
///   5. Sets stored_embedding_id in OperationContext
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
