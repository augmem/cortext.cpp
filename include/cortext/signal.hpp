#pragma once

#include <Eigen/Dense>
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

  // Additional metadata can be added here.
  uint64_t timestamp;
  std::string source_id;
};

} // namespace cortext
