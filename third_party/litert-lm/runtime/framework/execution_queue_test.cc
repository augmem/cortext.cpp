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

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::Eq;
using ::testing::IsFalse;
using ::testing::status::StatusIs;

TEST(ExecutionQueueTest, ExecuteTasksInOrder) {
  ExecutionQueue queue;
  std::vector<int> results;
  absl::Notification done;

  ASSERT_OK(queue.Enqueue([&results] { results.push_back(1); }));
  ASSERT_OK(queue.Enqueue([&results] { results.push_back(2); }));
  ASSERT_OK(queue.Enqueue([&results, &done] {
    results.push_back(3);
    done.Notify();
  }));

  done.WaitForNotification();
  EXPECT_THAT(results, Eq(std::vector<int>{1, 2, 3}));
}

TEST(ExecutionQueueTest, RemoveTask) {
  ExecutionQueue queue;
  absl::Notification task1_start, task1_finish;
  absl::Notification task2_run;
  absl::Notification task3_run;

  // Task 1: Blocks until told to finish.
  ASSERT_OK(queue.Enqueue([&] {
    task1_start.Notify();
    task1_finish.WaitForNotification();
  }));

  // Task 2: Should be cancelled.
  ASSERT_OK_AND_ASSIGN(int id2, queue.Enqueue([&] { task2_run.Notify(); }));

  // Task 3: Runs after task 1 (since task 2 is removed).
  ASSERT_OK(queue.Enqueue([&] { task3_run.Notify(); }));

  // Wait for task 1 to start running.
  task1_start.WaitForNotification();

  // Remove task 2.
  ASSERT_OK(queue.Remove(id2));

  // Allow task 1 to finish.
  task1_finish.Notify();

  // Wait for task 3 to run.
  task3_run.WaitForNotification();

  EXPECT_THAT(task2_run.HasBeenNotified(), IsFalse());
}

TEST(ExecutionQueueTest, RemoveNonExistentTask) {
  ExecutionQueue queue;
  EXPECT_THAT(queue.Remove(23), StatusIs(absl::StatusCode::kNotFound));
}

TEST(ExecutionQueueTest, RemoveRunningTaskFails) {
  // Explicitly manage queue's lifetime via unique_ptr to ensure referenced
  // notifications would not go out of scope.
  auto queue = std::make_unique<ExecutionQueue>();
  absl::Notification task_running, task_continue;

  ASSERT_OK_AND_ASSIGN(int id, queue->Enqueue([&] {
    task_running.Notify();
    task_continue.WaitForNotification();
  }));
  task_running.WaitForNotification();

  EXPECT_THAT(queue->Remove(id), StatusIs(absl::StatusCode::kNotFound));

  task_continue.Notify();
  queue.reset();
}

}  // namespace
}  // namespace litert::lm
