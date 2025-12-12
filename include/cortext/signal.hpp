#pragma once

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <vector>

namespace cortext
{

/// @brief Represents a single unit of data to be processed.
///
/// A signal is a modality-agnostic representation of an event, typically
/// containing an embedding vector and relevant metadata.
struct Signal
{
  // The universal representation of the input data.
  Eigen::VectorXf embedding;

  // Optional token-level surprisal summary (Algorithm 13 input).
  // Convention: mean negative log-probability in nats/token.
  // When unset, logprob-derived surprise is unavailable and embedding/score
  // fallbacks should be used.
  std::optional<double> mean_token_nll;

  // Additional metadata can be added here.
  uint64_t timestamp;
  std::string source_id;
};

} // namespace cortext
