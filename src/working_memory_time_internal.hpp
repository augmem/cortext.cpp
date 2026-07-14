#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace cortext::internal
{

/// Convert a finite, nonnegative working-memory timestamp to the nearest
/// signed-64-bit millisecond. The exclusive upper bound is exactly 2^63,
/// which remains representable even when double and long double both have
/// 53-bit precision.
inline int64_t
WorkingMemorySecondsToMillis (double seconds)
{
  if (!std::isfinite (seconds) || seconds < 0.0)
    {
      throw std::invalid_argument (
          "Working-memory timestamp must be finite and nonnegative");
    }

  const double millis = seconds * 1000.0;
  constexpr double kExclusiveUpperMillis = 0x1p63;
  if (!std::isfinite (millis) || millis >= kExclusiveUpperMillis)
    {
      throw std::out_of_range ("Working-memory timestamp exceeds int64 range");
    }
  return static_cast<int64_t> (std::llround (millis));
}

} // namespace cortext::internal
