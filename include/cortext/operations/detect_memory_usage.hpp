#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Marks retrieved memories as used based on interrupt gate decisions.
///
/// Usage definition (spec): used(m) = retrieved(m) AND injected into active
/// context after gate decisions. This operation:
/// 1. Reads retrieved candidate embeddings for the current step.
/// 2. Marks the selected candidate as used if the interrupt gate allowed it.
/// 3. Computes contextual_gain as cosine similarity between current signal
///    embedding and each candidate embedding.
///
/// This enables feedback algorithms (Algorithms 14–19) to use consistent
/// retrieved/used counts without cache-based heuristics.
///
/// Ordering: must run AFTER the interrupt gate, BEFORE feedback ops.
class DetectMemoryUsage
    : public Operation<Requires<tags::InterruptAllowed, tags::RetrievedMemoryEmbeddings, tags::SelectedCandidateId>, Satisfies<tags::MemoryUsageEvents> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
