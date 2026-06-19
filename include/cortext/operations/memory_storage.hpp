#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

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
///   4. Sets stored_embedding_id, stored_memory_id, and stored_signal_id in
///      OperationContext
///
/// If write_decision=false:
///   - Discards entirely, no storage occurs.
///
/// If write_decision=true but no payload:
///   - Logs warning, no storage occurs.
class MemoryStorage
    : public Operation<Requires<tags::AccumulatorWriteDecision, tags::BoundaryDecision, tags::EmotionProbabilities, tags::RepresentativeEmbedding>, Satisfies<tags::StoredEmbeddingId, tags::StoredMemoryId, tags::StoredSignalId> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
