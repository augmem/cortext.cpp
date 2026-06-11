#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/processor/contract_tags.hpp"

namespace cortext::operations
{

/// @brief Enumeration of all metric types used in the cortext system
enum class Metric
{
  relevance,
  mismatch,
  surprise,
  rarity,
  drift,
  contradiction,
  utility,
  periphery,
  coverage,
  salience,
  valence,
  arousal,

  focus_spread,
  drift_mag,
  embedding_surprisal,  // Section 3.1.4 - replaces logprob surprise
  // Test/debugging metrics
  aw_prev,
  rate_prev,
  hys_prev
};

/// @brief Compute all 12 composite metrics (Algorithm 7 inputs) and set them
/// on the OperationContext as [-1,1] values.
///
/// Metrics:
/// - relevance, mismatch, surprise, rarity, drift, utility,
///   salience, valence, arousal, contradiction, periphery, coverage
class ComputeMetrics
    : public Operation<Requires<tags::Arousal, tags::Valence, tags::AccumulatorWindowState>, Satisfies<tags::MetricValues, tags::Violation> >
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
