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

#include "runtime/executor/llm_litert_compiled_model_cache_utils.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::litert::IsError;
using ::testing::litert::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(LlmLiteRtCompiledModelCacheUtilsTest,
     TriggerTokenDeletionFromKvcacheTest_Success) {
  EXPECT_THAT(ShouldDeleteKVCacheTokens(/*current_step=*/10,
                                        /*start_position=*/1,
                                        /*context_size=*/10),
              IsOkAndHolds(true));
  EXPECT_THAT(ShouldDeleteKVCacheTokens(/*current_step=*/10,
                                        /*start_position=*/5,
                                        /*context_size=*/10),
              IsOkAndHolds(false));
}

TEST(LlmLiteRtCompiledModelCacheUtilsTest,
     DeleteTokensFromKvcacheTest_Failure) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer1,
                              CopyToTensorBuffer<float>(data, {1, 1, 2, 5}));
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      input_kv_cache_buffers;
  input_kv_cache_buffers.try_emplace("cache_", std::move(tensor_buffer1));
  EXPECT_THAT(
      DeleteTokensFromKvCache(&input_kv_cache_buffers,
                              /*num_tokens_to_drop=*/1,
                              /*init_tokens_to_retain=*/1),
      IsError(kLiteRtStatusErrorInvalidArgument, "Unsupported input name."));
}

TEST(LlmLiteRtCompiledModelCacheUtilsTest,
     DeleteTokensFromKvcacheTest_Success) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<float>(data, {1, 1, 5, 2}));

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      input_kv_cache_buffers;
  input_kv_cache_buffers.try_emplace("cache_k_", std::move(tensor_buffer));

  LITERT_ASSERT_OK(DeleteTokensFromKvCache(&input_kv_cache_buffers,
                                           /*num_tokens_to_drop=*/2,
                                           /*init_tokens_to_retain=*/1));

  // After dropping 2 tokens after the first one, we expect data from tokens 3
  // and 4 to be moved to the positions of tokens 1 and 2. The rest of the
  // buffer is zeroed out.
  const ::litert::TensorBuffer& result_tensor =
      input_kv_cache_buffers.at("cache_k_");
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, result_tensor.TensorType());
  EXPECT_THAT(tensor_type.Layout().Dimensions(),
              testing::ElementsAre(1, 1, 5, 2));

  LITERT_ASSERT_OK_AND_ASSIGN(auto result_data,
                              CopyFromTensorBuffer<float>(result_tensor));
  EXPECT_THAT(result_data,
              testing::ElementsAreArray({1.0f, 2.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f}));
}

TEST(LlmLiteRtCompiledModelCacheUtilsTest, DeleteTokensIfNeededTest_Success) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CopyToTensorBuffer<float>(data, {1, 1, 5, 2}));
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      input_kv_cache_buffers;
  input_kv_cache_buffers.try_emplace("cache_k_", std::move(tensor_buffer));
  int start_position = 0;
  EXPECT_THAT(DeleteTokensIfNeeded(&input_kv_cache_buffers,
                                   /*num_tokens_to_drop=*/2,
                                   /*init_tokens_to_retain=*/1,
                                   /*current_step=*/4, start_position,
                                   /*context_size=*/5),
              IsOkAndHolds(true));
  EXPECT_EQ(start_position, 2);
  const ::litert::TensorBuffer& result_tensor =
      input_kv_cache_buffers.at("cache_k_");
  LITERT_ASSERT_OK_AND_ASSIGN(auto result_data,
                              CopyFromTensorBuffer<float>(result_tensor));
  // After dropping 2 tokens (4 elements) and retaining the first token,
  // the data should be:
  // Token 0: {1, 2} (Retained)
  // Token 1: {7, 8} (Original Token 3)
  // Token 2: {9, 10} (Original Token 4)
  // Tokens 3 & 4: {0, 0} (Zeroed out)
  EXPECT_THAT(result_data,
              testing::ElementsAreArray({1.0f, 2.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f}));
}

TEST(LlmLiteRtCompiledModelCacheUtilsTest, ExpandBufferMiddleDim) {
  std::vector<float> src_data = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> src_shape = {2, 2, 2};
  std::vector<int> dst_shape = {2, 4, 2};
  std::vector<float> dst_data(16);
  ASSERT_OK(ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()),
                         src_shape, reinterpret_cast<uint8_t*>(dst_data.data()),
                         dst_shape, sizeof(float)));
  EXPECT_THAT(dst_data, testing::ElementsAreArray(
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
}

TEST(LlmLiteRtCompiledModelCacheUtilsTest, ExpandBufferLastDim) {
  std::vector<int> src_data = {1, 2, 3, 4};
  std::vector<int> src_shape = {2, 2};
  std::vector<int> dst_shape = {2, 4};
  std::vector<int> dst_data(8);
  ASSERT_OK(ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()),
                         src_shape, reinterpret_cast<uint8_t*>(dst_data.data()),
                         dst_shape, sizeof(int)));
  EXPECT_THAT(dst_data, testing::ElementsAreArray({1, 2, 0, 0, 3, 4, 0, 0}));
}

TEST(LlmLiteRtCompiledModelCacheUtilsTest, ExpandBufferFirstDim) {
  std::vector<int> src_data = {1, 2, 3, 4};
  std::vector<int> src_shape = {2, 2};
  std::vector<int> dst_shape = {4, 2};
  std::vector<int> dst_data(8);
  ASSERT_OK(ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()),
                         src_shape, reinterpret_cast<uint8_t*>(dst_data.data()),
                         dst_shape, sizeof(int)));
  EXPECT_THAT(dst_data, testing::ElementsAreArray({1, 2, 3, 4, 0, 0, 0, 0}));
}

TEST(LlmLiteRtCompiledModelCacheUtilsTest, ExpandBufferDifferentRank) {
  std::vector<int> src_data = {1, 2};
  std::vector<int> src_shape = {2};
  std::vector<int> dst_shape = {2, 1};
  std::vector<int> dst_data(2);
  EXPECT_THAT(
      ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()), src_shape,
                   reinterpret_cast<uint8_t*>(dst_data.data()), dst_shape,
                   sizeof(int)),
      StatusIs(absl::StatusCode::kInternal));
}

TEST(LlmLiteRtCompiledModelCacheUtilsTest, ExpandBufferMultipleDifferentDims) {
  std::vector<int> src_data = {1, 2};
  std::vector<int> src_shape = {1, 2};
  std::vector<int> dst_shape = {2, 3};
  std::vector<int> dst_data(6);
  EXPECT_THAT(
      ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()), src_shape,
                   reinterpret_cast<uint8_t*>(dst_data.data()), dst_shape,
                   sizeof(int)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Tensors differ in more than one dimension."));
}

TEST(LlmLiteRtCompiledModelCacheUtilsTest, ExpandBufferSmallerDestDim) {
  std::vector<int> src_data = {1, 2, 3, 4};
  std::vector<int> src_shape = {2, 2};
  std::vector<int> dst_shape = {1, 2};
  std::vector<int> dst_data(2);
  EXPECT_THAT(
      ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()), src_shape,
                   reinterpret_cast<uint8_t*>(dst_data.data()), dst_shape,
                   sizeof(int)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Destination tensor dimension is smaller than source "
               "along an axis."));
}

TEST(LlmLiteRtCompiledModelCacheUtilsTest, ExpandBufferNoExpansionAxis) {
  std::vector<int> src_data = {1, 2, 3, 4};
  std::vector<int> src_shape = {2, 2};
  std::vector<int> dst_shape = {2, 2};
  std::vector<int> dst_data(4);
  EXPECT_THAT(
      ExpandBuffer(reinterpret_cast<const uint8_t*>(src_data.data()), src_shape,
                   reinterpret_cast<uint8_t*>(dst_data.data()), dst_shape,
                   sizeof(int)),
      StatusIs(absl::StatusCode::kInvalidArgument, "No expansion axis found."));
}

}  // namespace
}  // namespace litert::lm
