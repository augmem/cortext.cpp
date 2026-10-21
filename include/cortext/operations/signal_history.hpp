#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Records the per-signal history later operations read.
///
/// Appends the composite score to the recent-score window (smoothed
/// uncertainty u(t) reads its spread) and updates the inter-signal time EWMA
/// dt_ema (boundary gap evidence and streaming pacing read it).
class UpdateSignalHistory
    : public Operation<Requires<tags::CompositeScore>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

/// @brief Updates the write-rate EWMA after the write decision.
///
/// Uses the current write decision to update the rate EWMA m_rate (which
/// drives the consolidation throughput hint) and the rate timestamp.
class UpdateRateState
    : public Operation<Requires<>, Satisfies<> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
