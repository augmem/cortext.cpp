#pragma once

#include "cortext/clock.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

namespace cortext::internal
{

/// Engine time: the supplied clock, but never behind the newest signal the
/// engine holds. Signals can carry their own timestamps (dated imports,
/// replays, clock skew), and retrieval treats everything at or after "now"
/// as not yet written; a clock behind the data would hide stored memories
/// from every query on the implicit clock. Explicit timestamps passed to the
/// *At APIs are used as given; only the implicit "now" is lifted.
class DataTimeClock final : public Clock
{
public:
  explicit DataTimeClock (std::shared_ptr<Clock> base)
      : base_ (std::move (base))
  {
  }

  // A clock behind the data reads just after the newest signal, so that
  // signal is already in the past; a clock at or ahead of it is unchanged.
  std::uint64_t
  NowMillis () const override
  {
    const auto base = base_->NowMillis ();
    const auto newest = newest_ms_.load (std::memory_order_relaxed);
    return base >= newest ? base : newest + 1;
  }

  /// Raise the floor to `timestamp_ms` (no-op when it is not newer).
  void
  Observe (std::uint64_t timestamp_ms)
  {
    auto current = newest_ms_.load (std::memory_order_relaxed);
    while (timestamp_ms > current
           && !newest_ms_.compare_exchange_weak (current, timestamp_ms,
                                                 std::memory_order_relaxed))
      {
      }
  }

  /// How far the supplied clock is behind the data (0 when it is not).
  std::uint64_t
  LagMillis () const
  {
    const auto base = base_->NowMillis ();
    const auto newest = newest_ms_.load (std::memory_order_relaxed);
    return newest > base ? newest - base : 0;
  }

private:
  std::shared_ptr<Clock> base_;
  std::atomic<std::uint64_t> newest_ms_ { 0 };
};

} // namespace cortext::internal
