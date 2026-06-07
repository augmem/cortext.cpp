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

#include "runtime/framework/execution_queue.h"

#include <thread>  // NOLINT
#include <utility>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace litert::lm {

ExecutionQueue::ExecutionQueue() {
  worker_ = std::thread(&ExecutionQueue::WorkerThread, this);
}

ExecutionQueue::~ExecutionQueue() {
  {
    absl::MutexLock lock(mutex_);
    stop_ = true;
  }
  if (worker_.joinable()) {
    worker_.join();
  }
}

absl::StatusOr<int> ExecutionQueue::Enqueue(
    absl::AnyInvocable<void() &&> task) {
  if (!task) {
    return absl::InvalidArgumentError("Task cannot be null");
  }

  absl::MutexLock lock(mutex_);
  if (stop_) {
    return absl::FailedPreconditionError("ExecutionQueue is shutting down");
  }

  int id = next_id_++;
  pending_tasks_[id] = std::move(task);
  task_order_.push(id);

  return id;
}

absl::Status ExecutionQueue::Remove(int id) {
  absl::MutexLock lock(mutex_);

  if (pending_tasks_.erase(id)) {
    return absl::OkStatus();
  }

  return absl::NotFoundError("Task not found (may have already executed).");
}

bool ExecutionQueue::HasWorkOrStop() const {
  return !task_order_.empty() || stop_;
}

void ExecutionQueue::WorkerThread() {
  while (true) {
    absl::AnyInvocable<void() &&> current_task;
    {
      absl::MutexLock lock(mutex_);
      mutex_.Await(absl::Condition(this, &ExecutionQueue::HasWorkOrStop));

      if (stop_) {
        break;
      }

      int id = task_order_.front();
      task_order_.pop();

      auto it = pending_tasks_.find(id);
      if (it != pending_tasks_.end()) {
        current_task = std::move(it->second);
        pending_tasks_.erase(it);
      } else {
        // The task was removed. Skip it and wait for the next loop.
        continue;
      }
    }

    // Execute the task OUTSIDE the mutex lock.
    // This prevents deadlocks if a task itself calls Enqueue() or Remove().
    if (current_task) {
      std::move(current_task)();
    }
  }
}

}  // namespace litert::lm
