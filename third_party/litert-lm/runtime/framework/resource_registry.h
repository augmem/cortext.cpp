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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_REGISTRY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_REGISTRY_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace litert::lm {

// A scoped lock for a resource retrieved from the ResourceRegistry.
template <typename T>
class ResourceScopedLock {
 public:
  ResourceScopedLock(T* resource, std::unique_ptr<absl::MutexLock> lock)
      : resource_(resource), lock_(std::move(lock)) {}
  ResourceScopedLock(ResourceScopedLock&& other) noexcept
      : resource_(other.resource_), lock_(std::move(other.lock_)) {
    other.resource_ = nullptr;
  }
  ~ResourceScopedLock() = default;

  // Non-copyable: Ensures exclusive lock semantics.
  ResourceScopedLock(const ResourceScopedLock&) = delete;
  ResourceScopedLock& operator=(const ResourceScopedLock&) = delete;

  ResourceScopedLock& operator=(ResourceScopedLock&& other) noexcept {
    if (this != &other) {
      resource_ = other.resource_;
      lock_ = std::move(other.lock_);
      other.resource_ = nullptr;
    }
    return *this;
  }

  // Access to the underlying resource.
  // Behavior is undefined if this ScopedLock does not hold a valid resource.
  T* operator->() const { return resource_; }
  T& operator*() const { return *resource_; }
  T* get() const { return resource_; }

 private:
  T* resource_;
  std::unique_ptr<absl::MutexLock> lock_;
};

// A registry for managing resources that require thread-safe access.
class ResourceRegistry {
 public:
  ResourceRegistry() = default;
  ~ResourceRegistry() = default;

  // Registers a resource with the given ID.
  //
  // The resource is owned by the registry after registration and will be
  // destroyed when the registry is destroyed.
  //
  // If the resource is null, an error will be returned.
  // If the resource ID already exists, an error will be returned.
  //
  // If the registration is successful, the resource is guaranteed to be
  // available for acquisition until the registry is destroyed.
  template <typename T>
  absl::Status Register(int id, std::unique_ptr<T> resource) {
    if (!resource) {
      return absl::InvalidArgumentError(
          "Cannot register a null resource with id: " + std::to_string(id));
    }

    absl::MutexLock lock(mu_);
    if (resources_.find(id) != resources_.end()) {
      return absl::AlreadyExistsError("Resource ID '" + std::to_string(id) +
                                      "' already exists.");
    }

    auto holder = std::make_unique<ResourceHolder<T>>(std::move(resource));
    resources_.emplace(id, ResourceNode(std::move(holder)));
    return absl::OkStatus();
  }

  // Acquires a resource with the given ID.
  //
  // If the resource is not found, an error will be returned.
  // If the resource type does not match the type of the resource in the
  // registry, an error will be returned.
  //
  // If the acquisition is successful, the returned scoped lock will provide
  // exclusive access to the resource. The lock will be automatically released
  // when the scoped lock is destroyed.
  template <typename T>
  absl::StatusOr<ResourceScopedLock<T>> Acquire(int id) {
    T* res_ptr = nullptr;
    std::unique_ptr<absl::MutexLock> res_lock;
    {
      absl::MutexLock lock(mu_);

      auto it = resources_.find(id);
      if (it == resources_.end()) {
        return absl::NotFoundError("Resource ID '" + std::to_string(id) +
                                   "' not found.");
      }

      auto* holder =
          dynamic_cast<ResourceHolder<T>*>(it->second.base_ptr.get());
      if (!holder) {
        return absl::InvalidArgumentError(
            "Type mismatch when acquiring resource ID '" + std::to_string(id) +
            "'.");
      }
      res_ptr = holder->resource.get();
      res_lock = std::make_unique<absl::MutexLock>(*it->second.mu);
    }
    return ResourceScopedLock<T>(res_ptr, std::move(res_lock));
  }

  // Views a resource with the given ID.
  //
  // If the resource is not found, an error will be returned.
  // If the resource type does not match the type of the resource in the
  // registry, an error will be returned.
  //
  // If the view is successful, the returned reference will provide
  // read-only access to the resource. The reference is only valid as long as
  // the resource is not modified or destroyed.
  template <typename T>
  absl::StatusOr<const T&> View(int id) {
    absl::MutexLock lock(mu_);
    auto it = resources_.find(id);
    if (it == resources_.end()) {
      return absl::NotFoundError("Resource ID '" + std::to_string(id) +
                                 "' not found.");
    }
    auto* holder = dynamic_cast<ResourceHolder<T>*>(it->second.base_ptr.get());
    if (!holder) {
      return absl::InvalidArgumentError(
          "Type mismatch when acquiring resource ID '" + std::to_string(id) +
          "'.");
    }
    return *holder->resource;
  }

  // Checks if a resource with the given ID exists in the registry.
  bool HasResource(int id) {
    absl::MutexLock lock(mu_);
    return resources_.contains(id);
  }

 private:
  struct ResourceBase {
    virtual ~ResourceBase() = default;
  };

  template <typename T>
  struct ResourceHolder : public ResourceBase {
    std::unique_ptr<T> resource;
    explicit ResourceHolder(std::unique_ptr<T> res)
        : resource(std::move(res)) {}
  };

  struct ResourceNode {
    std::unique_ptr<ResourceBase> base_ptr;
    std::unique_ptr<absl::Mutex> mu;

    explicit ResourceNode(std::unique_ptr<ResourceBase> bp)
        : base_ptr(std::move(bp)), mu(std::make_unique<absl::Mutex>()) {}

    ResourceNode(ResourceNode&& other) noexcept
        : base_ptr(std::move(other.base_ptr)), mu(std::move(other.mu)) {
      other.base_ptr = nullptr;
    }
  };

  absl::flat_hash_map<int, ResourceNode> resources_ ABSL_GUARDED_BY(mu_);
  absl::Mutex mu_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_REGISTRY_H_
