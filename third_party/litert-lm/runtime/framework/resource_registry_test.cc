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

#include "runtime/framework/resource_registry.h"

#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/framework/resource_ids.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::HasSubstr;

struct TestResource {
  int value = 0;
};

struct AnotherResource {
  std::string name;
};

TEST(ResourceRegistryTest, RegisterAndAcquire) {
  ResourceRegistry registry;
  auto resource = std::make_unique<TestResource>();
  resource->value = 42;

  ASSERT_OK(
      registry.Register(ResourceId::TOKENIZER_0_RID, std::move(resource)));

  ASSERT_OK_AND_ASSIGN(
      auto lock, registry.Acquire<TestResource>(ResourceId::TOKENIZER_0_RID));
  EXPECT_EQ(lock->value, 42);
  EXPECT_EQ((*lock).value, 42);
}

TEST(ResourceRegistryTest, RegisterNullResource) {
  ResourceRegistry registry;
  std::unique_ptr<TestResource> null_resource;
  auto status = registry.Register(/*id=*/1, std::move(null_resource));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Cannot register a null resource"));
}

TEST(ResourceRegistryTest, RegisterDuplicateId) {
  ResourceRegistry registry;
  ASSERT_OK(registry.Register(/*id=*/1, std::make_unique<TestResource>()));

  auto status = registry.Register(/*id=*/1, std::make_unique<TestResource>());
  EXPECT_EQ(status.code(), absl::StatusCode::kAlreadyExists);
  EXPECT_THAT(status.message(), HasSubstr("already exists"));
}

TEST(ResourceRegistryTest, AcquireNotFound) {
  ResourceRegistry registry;
  auto lock_or = registry.Acquire<TestResource>(/*id=*/1);
  EXPECT_EQ(lock_or.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(lock_or.status().message(), HasSubstr("not found"));
}

TEST(ResourceRegistryTest, AcquireTypeMismatch) {
  ResourceRegistry registry;
  ASSERT_OK(registry.Register(/*id=*/1, std::make_unique<TestResource>()));

  auto lock_or = registry.Acquire<AnotherResource>(1);
  EXPECT_EQ(lock_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(lock_or.status().message(), HasSubstr("Type mismatch"));
}

TEST(ResourceRegistryTest, ResourceScopedLockMove) {
  ResourceRegistry registry;
  {
    auto resource = std::make_unique<TestResource>();
    resource->value = 100;
    ASSERT_OK(registry.Register(/*id=*/1, std::move(resource)));
  }

  ASSERT_OK_AND_ASSIGN(auto lock, registry.Acquire<TestResource>(1));
  ResourceScopedLock<TestResource> lock1 = std::move(lock);
  EXPECT_EQ(lock1->value, 100);

  ResourceScopedLock<TestResource> lock2 = std::move(lock1);
  EXPECT_EQ(lock2->value, 100);
}

TEST(ResourceRegistryTest, ResourceThreadSafety) {
  struct BoolResource {
    bool is_busy = false;
  };

  ResourceRegistry registry;
  {
    auto resource = std::make_unique<BoolResource>();
    resource->is_busy = false;
    ASSERT_OK(registry.Register(/*id=*/1, std::move(resource)));
  }

  std::vector<std::thread> threads;
  threads.reserve(16);
  for (int i = 0; i < 16; ++i) {
    threads.emplace_back([&registry]() {
      ASSERT_OK_AND_ASSIGN(auto lock, registry.Acquire<BoolResource>(/*id=*/1));
      EXPECT_FALSE(lock->is_busy);
      lock->is_busy = true;
      absl::SleepFor(absl::Seconds(1));
      lock->is_busy = false;
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(ResourceRegistryTest, View) {
  ResourceRegistry registry;
  auto resource = std::make_unique<TestResource>();
  resource->value = 42;
  ASSERT_OK(registry.Register(/*id=*/1, std::move(resource)));

  ASSERT_OK_AND_ASSIGN(const TestResource& res,
                       registry.View<TestResource>(/*id=*/1));
  EXPECT_EQ(res.value, 42);
}

TEST(ResourceRegistryTest, HasResource) {
  ResourceRegistry registry;
  auto resource = std::make_unique<TestResource>();
  ASSERT_OK(registry.Register(/*id=*/1, std::move(resource)));
  EXPECT_TRUE(registry.HasResource(/*id=*/1));
  EXPECT_FALSE(registry.HasResource(/*id=*/2));
}

}  // namespace
}  // namespace litert::lm
