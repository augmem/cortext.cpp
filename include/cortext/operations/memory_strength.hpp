/// @file
#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithms 14 + 18: Memory Strength Adjustment with Influence.
///
/// Updates per-memory use_frequency via EWMA with α_S schedule and adjusts
/// strength with reinforcement minus decay (Alg 14) and an additional
/// influence-weighted term derived from retrieval/usage counts and
/// contextual gain (Alg 18). Evicts rows below periphery cutoff.
class UpdateMemoryStrength
    : public Operation<Requires<tags::MemoryUsageEvents, tags::SerialPositionMultiplier>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
