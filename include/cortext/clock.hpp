#pragma once

#include <chrono>
#include <cstdint>

namespace cortext
{

/// @brief Time source used by Cortext for wall-clock timestamps.
class Clock
{
public:
  virtual ~Clock () = default;
  virtual std::uint64_t NowMillis () const = 0;
};

/// @brief System wall-clock implementation.
class SystemClock final : public Clock
{
public:
  std::uint64_t
  NowMillis () const override
  {
    return static_cast<std::uint64_t> (
        std::chrono::duration_cast<std::chrono::milliseconds> (
            std::chrono::system_clock::now ().time_since_epoch ())
            .count ());
  }
};

/// @brief Mutable clock for deterministic replay and tests.
class FixedClock final : public Clock
{
public:
  explicit FixedClock (std::uint64_t now_ms) : now_ms_ (now_ms) {}

  std::uint64_t
  NowMillis () const override
  {
    return now_ms_;
  }

  void
  SetNowMillis (std::uint64_t now_ms)
  {
    now_ms_ = now_ms;
  }

private:
  std::uint64_t now_ms_;
};

} // namespace cortext
