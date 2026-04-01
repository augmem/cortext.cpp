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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_UTILS_MOVABLE_MUTEX_LOCK_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_UTILS_MOVABLE_MUTEX_LOCK_H_

#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace litert::lm {

// absl::MutexLock is not movable, so this is added to support transferring
// mutex lock ownership. NOTE: The passed absl::Mutex needs to outlive
// MovableMutexLock.
class ABSL_SCOPED_LOCKABLE MovableMutexLock {
 public:
  // Mutex must be non-null and the caller should not be holding the mutex lock.
  explicit MovableMutexLock(absl::Mutex* absl_nonnull mutex)
      ABSL_EXCLUSIVE_LOCK_FUNCTION(mutex)
      : mutex_(mutex) {
    mutex_->lock();
  }

  // MovableMutexLock is movable.
  MovableMutexLock(MovableMutexLock&& other) ABSL_NO_THREAD_SAFETY_ANALYSIS
      : mutex_(other.mutex_) {
    other.mutex_ = nullptr;
  }
  MovableMutexLock& operator=(MovableMutexLock&& other) {
    std::swap(this->mutex_, other.mutex_);
    return *this;
  }

  // MovableMutexLock is not copyable.
  MovableMutexLock(const MovableMutexLock&) = delete;
  MovableMutexLock& operator=(const MovableMutexLock&) = delete;

  // Destructor for mutex unlocking.
  ~MovableMutexLock() ABSL_UNLOCK_FUNCTION() {
    if (mutex_ != nullptr) {
      mutex_->unlock();
    }
  }

 private:
  // The locked mutex.
  absl::Mutex* mutex_ = nullptr;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_UTILS_MOVABLE_MUTEX_LOCK_H_
