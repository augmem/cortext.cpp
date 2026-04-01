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

#include "runtime/components/scoring_cpu_util.h"

#include <cmath>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

TEST(ScoringCpuUtilTest, ComputeLogLikelihood_InvalidSampledId) {
  const std::vector<float> logits = {0.0, 0.0, 0.3};
  const std::vector<int> sampled_ids = {12};
  auto batchconfidence = ComputeLogLikelihood(absl::MakeConstSpan(logits),
                                              absl::MakeConstSpan(sampled_ids),
                                              /*temperature=*/1.0);
  EXPECT_FALSE(batchconfidence.ok());
}

TEST(ScoringCpuUtilTest, ComputeLogLikelihood_BatchSize1) {
  const std::vector<float> logits = {0.0, 0.0, 0.3};
  const std::vector<int> sampled_ids = {2};
  auto batchconfidence = ComputeLogLikelihood(absl::MakeConstSpan(logits),
                                              absl::MakeConstSpan(sampled_ids),
                                              /*temperature=*/1.0);
  ASSERT_OK(batchconfidence);
  EXPECT_THAT(*batchconfidence,
              testing::ElementsAre(testing::FloatNear(
                  std::log(exp(0.3f) / (2 + std::exp(0.3f))), 1e-6f)));
}

TEST(ScoringCpuUtilTest, ComputeLogLikelihood_BatchSize2) {
  std::vector<float> logits = {0.0, 0.0, 0.3, 0.0, 0.7, 0.0};
  std::vector<int> sampled_ids = {2, 1};
  auto batchconfidence = ComputeLogLikelihood(absl::MakeConstSpan(logits),
                                              absl::MakeConstSpan(sampled_ids),
                                              /*temperature=*/1.0);
  EXPECT_TRUE(batchconfidence.ok());
  EXPECT_THAT(
      *batchconfidence,
      testing::ElementsAre(
          testing::FloatNear(std::log(exp(0.3f) / (2 + std::exp(0.3f))), 1e-6f),
          testing::FloatNear(std::log(exp(0.7f) / (2 + std::exp(0.7f))),
                             1e-6f)));
}

TEST(ScoringCpuUtilTest, ComputeLogLikelihood_BatchSize2_OneStreamEnded) {
  std::vector<float> logits = {0.0, 0.0, 0.3, 0.0, 0.7, 0.0};
  std::vector<int> sampled_ids = {2, 0};
  auto batchconfidence = ComputeLogLikelihood(absl::MakeConstSpan(logits),
                                              absl::MakeConstSpan(sampled_ids),
                                              /*temperature=*/1.0);
  EXPECT_TRUE(batchconfidence.ok());
  // Ignore the second token as it has ended.
  EXPECT_THAT(
      (*batchconfidence)[0],
      testing::FloatNear(std::log(exp(0.3f) / (2 + std::exp(0.3f))), 1e-6f));
}
}  // namespace
}  // namespace litert::lm
