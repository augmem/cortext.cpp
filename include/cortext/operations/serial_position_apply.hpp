#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Applies Algorithm 26 multipliers to be used by downstream updates.
///
/// Reads the serial position parameters (windows and bonuses) exposed by
/// ApplySerialPositionEffects and computes a per-signal multiplier that
/// represents the average primacy/recency boost across used memories in
/// this signal. The multiplier is stored in OperationContext and is
/// ephemeral (not persisted).
class ApplySerialPositionMultiplier
    : public Operation<Requires<tags::MemoryUsageEvents, tags::MetricValues, tags::SerialPositionPolicy>, Satisfies<tags::SerialPositionMultiplier> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
