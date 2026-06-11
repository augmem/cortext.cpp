#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Persists signal metrics to the signal_metrics table for observability.
///
/// Records all computed metrics from Algorithm 7 composite scoring along with
/// the composite score, threshold, and write decision for each processed signal.
class PersistSignalMetrics
    : public Operation<Requires<tags::AccumulatorWriteDecision, tags::MetricValues, tags::Coherence, tags::CompositeScore, tags::EffectiveFocus, tags::ThresholdState, tags::WriteDecision>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
