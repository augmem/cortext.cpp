// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_LITERT_LM_RUNTIME_FRAMEWORK_THREADPOOL_H_
#define THIRD_PARTY_LITERT_LM_RUNTIME_FRAMEWORK_THREADPOOL_H_

#include <cstddef>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/framework/thread_options.h"

namespace litert::lm {

// Forward declaration of WorkerThread to avoid circular dependency.
class WorkerThread;

// A thread pool consists of a set of threads that sit around waiting
// for callbacks to appear on a queue.  When that happens, one of the
// threads pulls a callback off the queue and runs it.
//
// The thread pool is shut down when the pool is destroyed.
//
// Sample usage:
//
// {
//   ThreadPool pool("testpool", max_num_workers);
//   for (int i = 0; i < N; ++i) {
//     pool.Schedule([i]() { DoWork(i); });
//   }
// }
//
class ThreadPool {
 public:
  // Creates a thread pool that creates and can use up to "max_num_threads"
  // threads.  Any standard thread options, such as stack size, should
  // be passed via "thread_options".  "name_prefix" specifies the
  // thread name prefix.
  ThreadPool(const std::string& name_prefix, size_t max_num_threads,
             ThreadOptions thread_options = ThreadOptions());

  // Waits for closures (if any) to complete. May be called without
  // having called StartWorkers().
  ~ThreadPool();

  // Adds specified callback to queue of pending callbacks.  Eventually a
  // thread will pull this callback off the queue and execute it. Note that
  // this does not guarantee that the callback is executed in the order it was
  // scheduled.
  absl::Status Schedule(absl::AnyInvocable<void() &&> callback);

  // Waits until the task queue is empty. The function will return an error if
  // the timeout is reached before the task queue is empty.
  // Note that this only indicates that there are no pending callbacks in the
  // queue, and does not guarantee that all scheduled callbacks have finished
  // executing. This is helpful for the caller to get a sense about the status
  // of the pool, but should not be used for synchronization.
  absl::Status WaitUntilIdle(absl::Duration timeout);

  // Waits until all the scheduled callbacks are executed and finished. The
  // function will return an error if the timeout is reached before all the
  // callbacks are finished.
  absl::Status WaitUntilDone(absl::Duration timeout);

  // Maximum number of threads in the pool.
  size_t max_num_threads() const { return max_num_threads_; }

  // Number of threads in the pool spawned actually.
  size_t num_threads() const {
    absl::MutexLock lock(&mutex_);
    return threads_.size();
  }

  // Standard thread options.  Use this accessor to get them.
  const ThreadOptions& thread_options() const { return thread_options_; }

 private:
  friend class WorkerThread;

  const std::string name_prefix_;
  // The number of threads in the pool.
  const size_t max_num_threads_;
  // Thread options.
  const ThreadOptions thread_options_;

  // The main function of the worker thread.
  void RunWorker();

  mutable absl::Mutex mutex_;
  std::vector<std::unique_ptr<WorkerThread>> threads_ ABSL_GUARDED_BY(mutex_);
  // Whether the pool is stopped.
  bool stopped_ ABSL_GUARDED_BY(mutex_) = false;
  // The tasks are stored in a queue using the Schedule() method and will be
  // executed by the threads.
  std::deque<absl::AnyInvocable<void() &&>> tasks_ ABSL_GUARDED_BY(mutex_);
  // Count the number of active tasks that are being executed by the threads.
  int num_active_tasks_ ABSL_GUARDED_BY(mutex_) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_LITERT_LM_RUNTIME_FRAMEWORK_THREADPOOL_H_
