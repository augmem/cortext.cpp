// Copyright 2026 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_EXECUTION_QUEUE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_EXECUTION_QUEUE_H_

#include <queue>
#include <thread>  // NOLINT

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace litert::lm {

// A thread-safe queue for executing tasks in the order they are enqueued.
// Tasks are executed by a single background thread, ensuring sequential
// execution.
class ExecutionQueue {
 public:
  ExecutionQueue();
  ~ExecutionQueue();

  ExecutionQueue(const ExecutionQueue&) = delete;
  ExecutionQueue& operator=(const ExecutionQueue&) = delete;

  // Enqueues a task and returns its unique ID.
  absl::StatusOr<int> Enqueue(absl::AnyInvocable<void() &&> task);

  // Attempts to remove a task by ID from the queue. Returns OK if successfully
  // removed, or NOT_FOUND if it the task is already running, finished, or not
  // found.
  absl::Status Remove(int id);

 private:
  // The loop run by the background thread.
  void WorkerThread();

  // Condition check for the worker thread.
  bool HasWorkOrStop() const ABSL_SHARED_LOCKS_REQUIRED(mutex_);

  absl::Mutex mutex_;
  bool stop_ ABSL_GUARDED_BY(mutex_) = false;
  int next_id_ ABSL_GUARDED_BY(mutex_) = 1;

  // Maintains the chronological order of task IDs.
  std::queue<int> task_order_ ABSL_GUARDED_BY(mutex_);

  // Holds the actual task payloads. If an ID is missing when popped
  // from `task_order_`, it means the task was cancelled.
  absl::flat_hash_map<int, absl::AnyInvocable<void() &&>> pending_tasks_
      ABSL_GUARDED_BY(mutex_);

  std::thread worker_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_EXECUTION_QUEUE_H_
