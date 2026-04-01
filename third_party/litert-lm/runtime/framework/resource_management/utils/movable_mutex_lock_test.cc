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

#include "runtime/framework/resource_management/utils/movable_mutex_lock.h"

#include <utility>

#include <gtest/gtest.h>
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace litert::lm {

TEST(MovableMutexLock, SmokeTest) {
  absl::Mutex mutex;
  {
    MovableMutexLock lock(&mutex);
    EXPECT_FALSE(mutex.try_lock());
  }
  EXPECT_TRUE(mutex.try_lock());
}

TEST(MovableMutexLock, MoveConstruct) {
  absl::Mutex mutex;
  {
    MovableMutexLock lock(&mutex);
    {
      MovableMutexLock lock2(std::move(lock));
      EXPECT_FALSE(mutex.try_lock());
    }
    // Even thought `lock` exists, `mutex` is unlocked, because `lock` is moved.
    mutex.AssertNotHeld();
  }
  EXPECT_TRUE(mutex.try_lock());
}

TEST(MovableMutexLock, MoveAssignment) {
  absl::Mutex mutex_a, mutex_b;
  {
    MovableMutexLock lock(&mutex_a);
    {
      MovableMutexLock lock2(&mutex_b);
      lock = std::move(lock2);
    }
    // `lock2` locks `mutex_b`, and transfers to `lock`, so `mutex_a` should be
    // unlocked.
    mutex_a.AssertNotHeld();
    EXPECT_FALSE(mutex_b.try_lock());
  }
  EXPECT_TRUE(mutex_a.try_lock());
}

}  // namespace litert::lm
