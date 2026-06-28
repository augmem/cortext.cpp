#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <vector>

namespace cortext::core
{

/// @brief Linear interpolation between two values.
template <typename T>
T
Lerp (T a, T b, double t)
{
  return a + (b - a) * t;
}

/// @brief Clamps a value to the a specified range [lo, hi].
template <typename T>
T
Clamp (const T &v, const T &lo, const T &hi)
{
  if (v < lo)
    return lo;
  if (v > hi)
    return hi;
  return v;
}

/// @brief The standard logistic sigmoid function.
inline double
Sigmoid (double z)
{
  return 1.0 / (1.0 + std::exp (-z));
}

/// @brief Exponentially Weighted Moving Average.
template <typename T>
T
Ewma (T previous_value, T new_value, double alpha)
{
  return (1.0 - alpha) * previous_value + alpha * new_value;
}

/// @brief Maps cosine similarity from [-1, 1] to [0, 1].
/// Reference: algorithms.md Section 2.1.2, lines 220-223
inline double
Map01 (double cosine)
{
  return Clamp ((cosine + 1.0) / 2.0, 0.0, 1.0);
}

/// @brief Calculates the cosine similarity between two vectors using Eigen.
inline double
CosineSimilarity (const Eigen::VectorXf &a, const Eigen::VectorXf &b)
{
  if (a.size () == 0 || a.size () != b.size ())
    {
      return 0.0;
    }
  double norm_a = a.norm ();
  double norm_b = b.norm ();
  if (norm_a == 0.0 || norm_b == 0.0)
    {
      return 0.0;
    }
  return a.dot (b) / (norm_a * norm_b);
}

} // namespace cortext::core
