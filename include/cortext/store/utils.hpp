#pragma once

#include <Eigen/Dense>
#include <any>
#include <optional>
#include <vector>

namespace cortext::store
{

/// @brief Convert Eigen vector to std::vector<float> for SQL storage.
inline std::vector<float>
EigenToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

/// @brief Extract blob from std::any (handles signed/unsigned char vectors).
inline std::vector<unsigned char>
BlobFromAny (const std::any &value)
{
  if (value.type () == typeid (std::vector<unsigned char>))
    {
      return std::any_cast<const std::vector<unsigned char> &> (value);
    }
  if (value.type () == typeid (std::vector<char>))
    {
      const auto &blob = std::any_cast<const std::vector<char> &> (value);
      return std::vector<unsigned char> (blob.begin (), blob.end ());
    }
  return {};
}

/// @brief Extract long long from std::any (handles int, long, long long).
inline std::optional<long long>
AnyToLongLong (const std::any &v)
{
  if (v.type () == typeid (long long))
    return std::any_cast<long long> (v);
  if (v.type () == typeid (long))
    return static_cast<long long> (std::any_cast<long> (v));
  if (v.type () == typeid (int))
    return static_cast<long long> (std::any_cast<int> (v));
  return std::nullopt;
}

/// @brief Extract double from std::any (handles double, float, int, long long).
inline std::optional<double>
AnyToDouble (const std::any &v)
{
  if (v.type () == typeid (double))
    return std::any_cast<double> (v);
  if (v.type () == typeid (float))
    return static_cast<double> (std::any_cast<float> (v));
  if (v.type () == typeid (int))
    return static_cast<double> (std::any_cast<int> (v));
  if (v.type () == typeid (long long))
    return static_cast<double> (std::any_cast<long long> (v));
  return std::nullopt;
}

/// @brief Extract double from std::any or return fallback when unsupported.
inline double
AnyToDouble (const std::any &v, double fallback)
{
  return AnyToDouble (v).value_or (fallback);
}

} // namespace cortext::store
