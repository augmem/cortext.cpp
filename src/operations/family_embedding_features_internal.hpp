#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cortext::operations::family_embedding_features_internal
{

constexpr std::size_t kNormBlockCount = 32;

struct Features
{
  Eigen::VectorXf normalized;
  double normalized_squared_norm = 0.0;
  std::array<double, kNormBlockCount> block_norms {};
  std::array<double, kNormBlockCount> hadamard_block_norms {};
  bool has_hadamard_features = false;
};

template <typename ValueAccessor>
void
BuildBlockNorms (Eigen::Index dimension_count, ValueAccessor value_at,
                 std::array<double, kNormBlockCount> &block_norms)
{
  for (Eigen::Index dimension = 0; dimension < dimension_count; ++dimension)
    {
      const double value = value_at (dimension);
      const auto block = std::min<std::size_t> (
          kNormBlockCount - 1,
          static_cast<std::size_t> (dimension) * kNormBlockCount
              / static_cast<std::size_t> (dimension_count));
      block_norms[block] += value * value;
    }
  for (double &block_norm : block_norms)
    block_norm = std::sqrt (block_norm);
}

inline Features
Build (const Eigen::VectorXf &embedding)
{
  Features features;
  if (embedding.size () <= 0)
    return features;

  double squared_norm = 0.0;
  for (Eigen::Index dimension = 0; dimension < embedding.size (); ++dimension)
    {
      const double value = static_cast<double> (embedding[dimension]);
      squared_norm += value * value;
    }
  if (squared_norm <= 1e-24)
    return features;

  const double inverse_norm = 1.0 / std::sqrt (squared_norm);
  features.normalized.resize (embedding.size ());
  for (Eigen::Index dimension = 0; dimension < embedding.size (); ++dimension)
    {
      features.normalized[dimension] = static_cast<float> (
          static_cast<double> (embedding[dimension]) * inverse_norm);
      const double normalized_value
          = static_cast<double> (features.normalized[dimension]);
      features.normalized_squared_norm
          += normalized_value * normalized_value;
    }
  BuildBlockNorms (
      features.normalized.size (),
      [&features] (Eigen::Index dimension) {
        return static_cast<double> (features.normalized[dimension]);
      },
      features.block_norms);

  const std::size_t dimension_count
      = static_cast<std::size_t> (features.normalized.size ());
  if (dimension_count > 0
      && (dimension_count & (dimension_count - 1)) == 0)
    {
      std::vector<double> transformed (dimension_count);
      for (std::size_t dimension = 0; dimension < dimension_count;
           ++dimension)
        transformed[dimension]
            = static_cast<double> (features.normalized[dimension]);
      for (std::size_t span = 1; span < dimension_count; span *= 2)
        for (std::size_t begin = 0; begin < dimension_count;
             begin += span * 2)
          for (std::size_t offset = 0; offset < span; ++offset)
            {
              const double left = transformed[begin + offset];
              const double right = transformed[begin + span + offset];
              transformed[begin + offset] = left + right;
              transformed[begin + span + offset] = left - right;
            }
      const double inverse_transform_norm
          = 1.0 / std::sqrt (static_cast<double> (dimension_count));
      BuildBlockNorms (
          features.normalized.size (),
          [&transformed, inverse_transform_norm] (Eigen::Index dimension) {
            return transformed[static_cast<std::size_t> (dimension)]
                   * inverse_transform_norm;
          },
          features.hadamard_block_norms);
      features.has_hadamard_features = true;
    }
  return features;
}

} // namespace cortext::operations::family_embedding_features_internal
