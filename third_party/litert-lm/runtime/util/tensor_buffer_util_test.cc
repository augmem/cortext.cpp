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

#include "runtime/util/tensor_buffer_util.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;

TEST(TensorBufferUtilTest, NumSignificantDims) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CreateTensorBuffer<int8_t>({2, 5}));
  EXPECT_THAT(NumSignificantDims(tensor_buffer), Eq(2));
  LITERT_ASSERT_OK_AND_ASSIGN(tensor_buffer,
                              CreateTensorBuffer<int8_t>({2, 1, 5}));
  EXPECT_THAT(NumSignificantDims(tensor_buffer), Eq(2));
  LITERT_ASSERT_OK_AND_ASSIGN(tensor_buffer,
                              CreateTensorBuffer<int8_t>({1, 1, 5}));
  EXPECT_THAT(NumSignificantDims(tensor_buffer), Eq(1));
}

TEST(TensorBufferUtilTest, TensorBufferDims) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CreateTensorBuffer<int8_t>({2, 5}));
  EXPECT_THAT(TensorBufferDims(tensor_buffer), ElementsAre(2, 5));
  LITERT_ASSERT_OK_AND_ASSIGN(tensor_buffer,
                              CreateTensorBuffer<int8_t>({2, 1, 5}));
  EXPECT_THAT(TensorBufferDims(tensor_buffer), ElementsAre(2, 1, 5));
  LITERT_ASSERT_OK_AND_ASSIGN(tensor_buffer,
                              CreateTensorBuffer<int8_t>({1, 1, 5}));
  EXPECT_THAT(TensorBufferDims(tensor_buffer), ElementsAre(1, 1, 5));
}

}  // namespace
}  // namespace litert::lm
