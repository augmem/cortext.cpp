/// @file
/// @brief Shared utility functions for cortext operations.
#pragma once

#include <Eigen/Dense>
#include <any>
#include <cstring>
#include <string>
#include <vector>

namespace cortext::core
{

/// @brief Extracts raw byte data from an std::any containing a blob.
/// @param v The std::any value (expected to be vector<char> or vector<unsigned char>).
/// @param data Output pointer to the raw data.
/// @param len Output length of the data.
/// @return True if decoding succeeded, false otherwise.
inline bool
AnyToBytes (const std::any &v, const unsigned char *&data, size_t &len)
{
  if (v.type () == typeid (std::vector<char>))
    {
      const auto &blob = std::any_cast<const std::vector<char> &> (v);
      data = reinterpret_cast<const unsigned char *> (blob.data ());
      len = blob.size ();
      return true;
    }
  if (v.type () == typeid (std::vector<unsigned char>))
    {
      const auto &blob = std::any_cast<const std::vector<unsigned char> &> (v);
      data = blob.data ();
      len = blob.size ();
      return true;
    }
  return false;
}

/// @brief Decodes a blob into an Eigen float vector.
/// @param v The std::any blob value.
/// @param dim Expected dimension of the vector.
/// @param out Output Eigen vector.
/// @return True if decoding succeeded, false otherwise.
inline bool
DecodeFloatBlob (const std::any &v, int dim, Eigen::VectorXf &out)
{
  const unsigned char *data = nullptr;
  size_t len = 0;
  if (!AnyToBytes (v, data, len))
    {
      return false;
    }
  const size_t want = static_cast<size_t> (dim) * sizeof (float);
  if (dim <= 0 || len != want)
    {
      return false;
    }
  out.resize (dim);
  for (int i = 0; i < dim; ++i)
    {
      float f = 0.0f;
      std::memcpy (&f, data + static_cast<size_t> (i) * sizeof (float),
                   sizeof (float));
      out[i] = f;
    }
  return true;
}

/// @brief Clamps a value to [0.0, 1.0].
/// @param v The value to clamp.
/// @return The clamped value.
inline double
Clamp01 (double v)
{
  if (v < 0.0)
    return 0.0;
  if (v > 1.0)
    return 1.0;
  return v;
}

/// @brief Checks if a string starts with the given prefix (zero-allocation).
/// @param s The string to check.
/// @param prefix The prefix to look for.
/// @return True if s starts with prefix.
inline bool
StartsWith (const std::string &s, const char *prefix)
{
  const std::size_t prefix_len = std::strlen (prefix);
  return s.size () >= prefix_len && s.compare (0, prefix_len, prefix, prefix_len) == 0;
}

} // namespace cortext::core
