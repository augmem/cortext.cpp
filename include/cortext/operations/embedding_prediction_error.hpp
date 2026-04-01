#pragma once

#include "cortext/processor/operation.hpp"

namespace cortext::operations
{

/// @brief Section 3.1.4: Embedding Prediction Error
///
/// Computes surprisal using EMA-based expectation of the embedding stream.
/// The predictor state tracks an exponential moving average of embeddings,
/// and surprisal is derived from deviation between expectation and input.
///
/// This REPLACES the previous logprob-based surprise (Algorithm 13),
/// providing a pure embedding-based approach that doesn't require
/// token-level language model outputs.
///
/// Output: Sets Metric::embedding_surprisal in [0,1]
class UpdateEmbeddingPredictionError : public IOperation
{
public:
  void Execute (OperationContext &context, Transaction &tx) const override;
};

} // namespace cortext::operations
