/// @file
#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Algorithm 13: Logprob-derived surprise.
///
/// If the input signal provides `Signal::mean_token_nll`, computes a normalized
/// surprisal and fuses it with drift magnitude (Alg 12 output) to produce a
/// surprise value in [0,1], written to `Metric::surprise`.
///
/// When logprob data is unavailable, this operation is a no-op.
class UpdateLogprobSurprise : public IOperation
{
public:
  void Execute (OperationContext &context) const override;
};

} // namespace cortext::operations

