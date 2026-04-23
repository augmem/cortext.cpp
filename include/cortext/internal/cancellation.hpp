#pragma once

#include "cortext/stop_token.hpp"

#include <stdexcept>
#include <vector>

namespace cortext::internal
{

class CancellationError : public std::runtime_error
{
public:
  CancellationError () : std::runtime_error ("operation cancelled") {}
};

namespace detail
{
extern thread_local std::vector<cortext::StopToken> stop_token_stack;
}

class ScopedStopToken
{
public:
  explicit ScopedStopToken (cortext::StopToken token) : active_ (true)
  {
    detail::stop_token_stack.push_back (std::move (token));
  }

  ~ScopedStopToken ()
  {
    if (active_ && !detail::stop_token_stack.empty ())
      {
        detail::stop_token_stack.pop_back ();
      }
  }

  ScopedStopToken (const ScopedStopToken &) = delete;
  ScopedStopToken &operator= (const ScopedStopToken &) = delete;

  ScopedStopToken (ScopedStopToken &&other) noexcept : active_ (other.active_)
  {
    other.active_ = false;
  }

  ScopedStopToken &
  operator= (ScopedStopToken &&other) noexcept
  {
    if (this != &other)
      {
        if (active_ && !detail::stop_token_stack.empty ())
          {
            detail::stop_token_stack.pop_back ();
          }
        active_ = other.active_;
        other.active_ = false;
      }
    return *this;
  }

private:
  bool active_ = false;
};

inline cortext::StopToken
CurrentStopToken ()
{
  if (detail::stop_token_stack.empty ())
    {
      return {};
    }
  return detail::stop_token_stack.back ();
}

inline bool
StopRequested ()
{
  return CurrentStopToken ().stop_requested ();
}

[[noreturn]] inline void
ThrowCancellation ()
{
  throw CancellationError ();
}

inline void
ThrowIfStopRequested ()
{
  if (StopRequested ())
    {
      ThrowCancellation ();
    }
}

} // namespace cortext::internal
