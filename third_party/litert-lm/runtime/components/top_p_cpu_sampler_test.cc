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

#include "runtime/components/top_p_cpu_sampler.h"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

TEST(TopPSamplerTest, Create) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/1, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
}

TEST(TopPSamplerTest, CreateWithZeroTemp) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/0.0,
                                        /*batch_size=*/1, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
}

TEST(TopPSamplerTest, CreateWithNegativeTemp) {
  auto sampler_or =
      TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/-1.0,
                          /*batch_size=*/1, /*seed=*/1);
  EXPECT_FALSE(sampler_or.ok());
  EXPECT_THAT(sampler_or.status().message(),
              testing::HasSubstr("Temperature must be >= 0"));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_IdsOnly_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
  auto sampler = std::move(sampler_or.value());

  const std::vector<float> logits = {0.0, 0.0, 10.0, 0.0, 11.0, 12.0, 1.0, 2.0};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 4});

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  /*scores_tensor=*/nullptr);
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  EXPECT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, testing::ElementsAre(2, 1));
}

TEST(TopPSamplerTest, SampleToIdAndScoreBuffer_BatchSize2) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
  auto sampler = std::move(sampler_or.value());

  const std::vector<float> logits = {
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};
  auto logits_tensor = CopyToTensorBuffer<float>(logits, {2, 4});

  std::vector<int> ids_vector(2);
  auto ids_tensor =
      CopyToTensorBuffer<int>(absl::MakeConstSpan(ids_vector), {2});
  std::vector<float> scores_vector(2);
  auto scores_tensor =
      CopyToTensorBuffer<float>(absl::MakeConstSpan(scores_vector), {2});
  auto status = sampler->SampleToIdAndScoreBuffer(*logits_tensor, *ids_tensor,
                                                  &(*scores_tensor));
  EXPECT_TRUE(status.ok());

  auto ids = CopyFromTensorBuffer<int>(*ids_tensor);
  EXPECT_TRUE(ids.HasValue());
  // The sampled id is 2 and 1.
  EXPECT_THAT(*ids, testing::ElementsAre(2, 1));

  auto scores = CopyFromTensorBuffer<float>(*scores_tensor);
  EXPECT_TRUE(scores.HasValue());
  // The scores are the log of the probability of the sampled token.
  EXPECT_THAT(*scores, testing::ElementsAre(std::log(1.0f), std::log(1.0f)));
}

}  // namespace
}  // namespace litert::lm
