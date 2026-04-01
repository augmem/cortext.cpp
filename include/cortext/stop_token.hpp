#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace cortext
{

namespace internal
{
struct StopCallbackEntry
{
  std::function<void ()> fn;
  std::atomic<bool> active { true };
};

struct StopState
{
  std::atomic<bool> stop_requested { false };
  std::mutex mu;
  std::vector<std::weak_ptr<StopCallbackEntry> > callbacks;
};
} // namespace internal

class StopToken
{
public:
  StopToken () = default;

  bool
  stop_requested () const
  {
    return state_ && state_->stop_requested.load ();
  }

  bool
  stop_possible () const
  {
    return static_cast<bool> (state_);
  }

private:
  explicit StopToken (std::shared_ptr<internal::StopState> state)
      : state_ (std::move (state))
  {
  }

  std::shared_ptr<internal::StopState> state_;

  template <typename Callback>
  friend class StopCallback;
  friend class StopSource;
};

class StopSource
{
public:
  StopSource () : state_ (std::make_shared<internal::StopState> ()) {}

  StopToken
  get_token () const
  {
    return StopToken (state_);
  }

  bool
  request_stop () const
  {
    if (!state_)
      {
        return false;
      }
    bool expected = false;
    if (!state_->stop_requested.compare_exchange_strong (expected, true))
      {
        return false;
      }

    std::vector<std::shared_ptr<internal::StopCallbackEntry> > callbacks;
    {
      std::lock_guard<std::mutex> lock (state_->mu);
      auto &entries = state_->callbacks;
      entries.erase (
          std::remove_if (entries.begin (), entries.end (),
                          [] (const auto &weak) { return weak.expired (); }),
          entries.end ());
      callbacks.reserve (entries.size ());
      for (const auto &weak : entries)
        {
          if (auto callback = weak.lock ())
            {
              callbacks.push_back (std::move (callback));
            }
        }
    }
    for (const auto &callback : callbacks)
      {
        if (callback && callback->active.load ())
          {
            callback->fn ();
          }
      }
    return true;
  }

  bool
  stop_requested () const
  {
    return state_ && state_->stop_requested.load ();
  }

private:
  std::shared_ptr<internal::StopState> state_;
};

template <typename Callback>
class StopCallback
{
public:
  StopCallback (const StopToken &token, Callback callback)
      : state_ (token.state_),
        entry_ (std::make_shared<internal::StopCallbackEntry> ())
  {
    entry_->fn = std::move (callback);
    if (!state_)
      {
        entry_->active.store (false);
        return;
      }
    if (state_->stop_requested.load ())
      {
        entry_->fn ();
        entry_->active.store (false);
        return;
      }
    std::lock_guard<std::mutex> lock (state_->mu);
    state_->callbacks.push_back (entry_);
  }

  ~StopCallback ()
  {
    if (entry_)
      {
        entry_->active.store (false);
      }
  }

  StopCallback (const StopCallback &) = delete;
  StopCallback &operator= (const StopCallback &) = delete;
  StopCallback (StopCallback &&) = delete;
  StopCallback &operator= (StopCallback &&) = delete;

private:
  std::shared_ptr<internal::StopState> state_;
  std::shared_ptr<internal::StopCallbackEntry> entry_;
};

} // namespace cortext
