#pragma once

#include "cortext/processor/operation.hpp"
#include "cortext/store/store.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace cortext
{

class OperationContext;

namespace fork_detail
{

/// Persistent worker pool for Fork branches. Threads are created once per
/// process (lazily, on first Fork) and receive work over a queue, so a
/// Fork costs an enqueue + wakeup, never a thread spinup or teardown.
class OperationWorkerPool
{
public:
  static OperationWorkerPool &
  Instance ()
  {
    static OperationWorkerPool pool;
    return pool;
  }

  void
  Submit (std::function<void ()> task)
  {
    {
      std::lock_guard<std::mutex> lock (mutex_);
      queue_.push_back (std::move (task));
    }
    cv_.notify_one ();
  }

  int
  WorkerCount () const
  {
    return worker_count_;
  }

  OperationWorkerPool (const OperationWorkerPool &) = delete;
  OperationWorkerPool &operator= (const OperationWorkerPool &) = delete;

private:
  OperationWorkerPool ()
  {
    const unsigned hw = std::thread::hardware_concurrency ();
    worker_count_ = static_cast<int> (
        hw <= 3 ? 2u : (hw - 2u > 8u ? 8u : hw - 2u));
    workers_.reserve (static_cast<size_t> (worker_count_));
    for (int i = 0; i < worker_count_; ++i)
      {
        workers_.emplace_back ([this] { this->Run (); });
      }
  }

  ~OperationWorkerPool ()
  {
    {
      std::lock_guard<std::mutex> lock (mutex_);
      stopping_ = true;
    }
    cv_.notify_all ();
    for (auto &worker : workers_)
      {
        if (worker.joinable ())
          {
            worker.join ();
          }
      }
  }

  void
  Run ()
  {
    for (;;)
      {
        std::function<void ()> task;
        {
          std::unique_lock<std::mutex> lock (mutex_);
          cv_.wait (lock, [this] { return stopping_ || !queue_.empty (); });
          if (stopping_ && queue_.empty ())
            {
              return;
            }
          task = std::move (queue_.front ());
          queue_.pop_front ();
        }
        task ();
      }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::function<void ()> > queue_;
  std::vector<std::thread> workers_;
  bool stopping_ = false;
  int worker_count_ = 0;
};

} // namespace fork_detail

/// Per-execution state of one outstanding Fork: branch completion latch,
/// captured branch failures, and the transaction mutex shared by branch
/// work and main-thread members until the matching Join.
struct PendingFork
{
  std::mutex mutex;
  std::condition_variable cv;
  int remaining = 0;
  std::vector<std::exception_ptr> failures;

  void
  Wait ()
  {
    std::unique_lock<std::mutex> lock (mutex);
    cv.wait (lock, [this] { return remaining == 0; });
  }

  void
  BranchDone (std::exception_ptr failure)
  {
    {
      std::lock_guard<std::mutex> lock (mutex);
      if (failure)
        {
          failures.push_back (std::move (failure));
        }
      --remaining;
    }
    cv.notify_all ();
  }
};

/// Serializes Transaction access while Fork branches are outstanding: the
/// lock is held per METHOD CALL, so branch and main-thread compute runs
/// fully parallel while store access interleaves safely on the one
/// underlying connection and statement cache.
class SerializedTransaction final : public Transaction
{
public:
  SerializedTransaction (Transaction &inner, std::mutex &mutex)
      : inner_ (inner), mutex_ (mutex)
  {
  }

  std::unique_ptr<Transaction>
  Begin () override
  {
    std::lock_guard<std::mutex> lock (mutex_);
    return inner_.Begin ();
  }

  std::vector<std::map<std::string, std::any> >
  Execute (const std::string &query,
           const std::vector<std::any> &params = {}) override
  {
    std::lock_guard<std::mutex> lock (mutex_);
    return inner_.Execute (query, params);
  }

  void
  Commit () override
  {
    std::lock_guard<std::mutex> lock (mutex_);
    inner_.Commit ();
  }

  void
  Rollback () override
  {
    std::lock_guard<std::mutex> lock (mutex_);
    inner_.Rollback ();
  }

private:
  Transaction &inner_;
  std::mutex &mutex_;
};

namespace fork_detail
{

template <typename Op> using InputOf = typename Op::Input;
template <typename Op> using OutputOf = typename Op::Output;

template <typename Tag, typename List> struct ListContains;

template <typename Tag, template <typename...> class List, typename... Ts>
struct ListContains<Tag, List<Ts...> >
    : std::bool_constant<(std::is_same_v<Tag, Ts> || ...)>
{
};

template <typename A, typename B> struct ListsDisjoint;

template <template <typename...> class LA, typename... As, typename B>
struct ListsDisjoint<LA<As...>, B>
    : std::bool_constant<(!ListContains<As, B>::value && ...)>
{
};

template <typename A, typename B> struct ConcatLists;

template <template <typename...> class LA, typename... As,
          template <typename...> class LB, typename... Bs>
struct ConcatLists<LA<As...>, LB<Bs...> >
{
  using type = LA<As..., Bs...>;
};

template <typename First, typename... Rest> struct UnionInputs
{
  using type = typename ConcatLists<InputOf<First>,
                                    typename UnionInputs<Rest...>::type>::type;
};

template <typename First> struct UnionInputs<First>
{
  using type = InputOf<First>;
};

template <typename First, typename... Rest> struct UnionOutputs
{
  using type =
      typename ConcatLists<OutputOf<First>,
                           typename UnionOutputs<Rest...>::type>::type;
};

template <typename First> struct UnionOutputs<First>
{
  using type = OutputOf<First>;
};

/// True when no branch writes a value another branch writes (write/write
/// conflict) and no branch reads a value a SIBLING writes (it could not
/// observe a stable value either way; such an edge demands sequencing,
/// not forking).
template <typename A, typename B>
inline constexpr bool PairConflictFree
    = ListsDisjoint<OutputOf<A>, OutputOf<B> >::value
      && ListsDisjoint<OutputOf<A>, InputOf<B> >::value
      && ListsDisjoint<OutputOf<B>, InputOf<A> >::value;

template <typename... Ops> struct BranchesConflictFree;

template <> struct BranchesConflictFree<> : std::true_type
{
};

template <typename First, typename... Rest>
struct BranchesConflictFree<First, Rest...>
    : std::bool_constant<(PairConflictFree<First, Rest> && ...)
                         && BranchesConflictFree<Rest...>::value>
{
};

} // namespace fork_detail

/// True when the branch set can run concurrently: pairwise disjoint
/// outputs and no branch consuming a sibling's output. Exposed for tests
/// and call-site reasoning; Fork static_asserts it.
template <typename... Ops>
inline constexpr bool ForkBranchesConflictFree
    = fork_detail::BranchesConflictFree<Ops...>::value;

} // namespace cortext
