#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Algorithm 25: Metacognitive Monitoring.
///
/// Computes metacognitive thresholds from knobs (F,S,T), evaluates
/// Feeling-Of-Knowing (FOK) and retrieval strength to detect
/// tip-of-the-tongue (TOT) and unknown states, and exposes parameters
/// for downstream strategy control.
class MetacognitiveMonitoring
    : public Operation<Requires<tags::MemoryUsageEvents>, Satisfies<tags::MetacognitiveState> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
