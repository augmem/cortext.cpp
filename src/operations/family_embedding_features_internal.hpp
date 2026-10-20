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

/// Everything Features carries except the normalized vector itself, which
/// is re-expanded from the shared embedding row on use so resident caches
/// keep only these norms beside the arena row.
struct Norms
{
  bool degenerate = true; ///< zero-norm row: Features stays empty
  double inverse_norm = 0.0;
  double normalized_squared_norm = 0.0;
  std::array<double, kNormBlockCount> block_norms {};
  std::array<double, kNormBlockCount> hadamard_block_norms {};
  bool has_hadamard_features = false;
};

inline Features
Build (const float *embedding, Eigen::Index size)
{
  Features features;
  if (size <= 0)
    return features;

  double squared_norm = 0.0;
  for (Eigen::Index dimension = 0; dimension < size; ++dimension)
    {
      const double value = static_cast<double> (embedding[dimension]);
      squared_norm += value * value;
    }
  if (squared_norm <= 1e-24)
    return features;

  const double inverse_norm = 1.0 / std::sqrt (squared_norm);
  features.normalized.resize (size);
  for (Eigen::Index dimension = 0; dimension < size; ++dimension)
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

inline Features
Build (const Eigen::VectorXf &embedding)
{
  return Build (embedding.data (), embedding.size ());
}

/// Norm side table for a row; Expand (row, BuildNorms (row)) == Build (row).
inline Norms
BuildNorms (const float *embedding, Eigen::Index size)
{
  Norms norms;
  Features features = Build (embedding, size);
  if (features.normalized.size () == 0)
    return norms;
  double squared_norm = 0.0;
  for (Eigen::Index dimension = 0; dimension < size; ++dimension)
    {
      const double value = static_cast<double> (embedding[dimension]);
      squared_norm += value * value;
    }
  norms.degenerate = false;
  norms.inverse_norm = 1.0 / std::sqrt (squared_norm);
  norms.normalized_squared_norm = features.normalized_squared_norm;
  norms.block_norms = features.block_norms;
  norms.hadamard_block_norms = features.hadamard_block_norms;
  norms.has_hadamard_features = features.has_hadamard_features;
  return norms;
}

/// Rebuilds the Features of a row from its cached norms (bit-identical to
/// Build: the normalized vector uses the same double product and rounding).
inline Features
Expand (const float *embedding, Eigen::Index size, const Norms &norms)
{
  Features features;
  if (norms.degenerate || size <= 0)
    return features;
  features.normalized.resize (size);
  for (Eigen::Index dimension = 0; dimension < size; ++dimension)
    features.normalized[dimension] = static_cast<float> (
        static_cast<double> (embedding[dimension]) * norms.inverse_norm);
  features.normalized_squared_norm = norms.normalized_squared_norm;
  features.block_norms = norms.block_norms;
  features.hadamard_block_norms = norms.hadamard_block_norms;
  features.has_hadamard_features = norms.has_hadamard_features;
  return features;
}

} // namespace cortext::operations::family_embedding_features_internal
