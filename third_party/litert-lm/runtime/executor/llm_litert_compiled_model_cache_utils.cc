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
#include <cstring>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {

using ::litert::Expected;
using ::litert::TensorBuffer;

::litert::Expected<bool> ShouldDeleteKVCacheTokens(int current_step,
                                                   int start_position,
                                                   size_t context_size) {
  if (current_step - start_position < 0) {
    return ::litert::Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Deleted more tokens than the number of model steps processed.");
  }
  if (current_step < 0) {
    return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                "current_step step is negative.");
  }
  if (start_position < 0) {
    return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                "start_position is negative.");
  }
  if (current_step - start_position >= context_size - 1) {
    return true;
  }
  return false;
}

// Function to dump the ring buffer.
::litert::Expected<void> DeleteTokensFromKvCache(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
        input_kv_cache_buffers,
    int num_tokens_to_drop, int init_tokens_to_retain) {
  // A k cache buffer is a 4D tensor with shape
  // [1, heads, context_size, embedding_size]
  // A v cache buffer is a 4D tensor with shape
  // [1, heads, embedding_size, context_size]
  for (auto& [input_name, input_buffer] : *input_kv_cache_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto type, input_buffer.TensorType());
    const int axis = absl::StrContains(input_name, "cache_k_")   ? 2
                     : absl::StrContains(input_name, "cache_v_") ? 3
                                                                 : -1;
    if (axis == -1) {
      return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                  "Unsupported input name.");
    }
    switch (type.ElementType()) {
      case ::litert::ElementType::Int8:
        LITERT_RETURN_IF_ERROR(DropTokensfromTensorBuffer<int8_t>(
            input_buffer, num_tokens_to_drop, axis, init_tokens_to_retain));
        break;
      case ::litert::ElementType::Int16:
        LITERT_RETURN_IF_ERROR(DropTokensfromTensorBuffer<int16_t>(
            input_buffer, num_tokens_to_drop, axis, init_tokens_to_retain));
        break;
      case ::litert::ElementType::Int32:
        LITERT_RETURN_IF_ERROR(DropTokensfromTensorBuffer<int32_t>(
            input_buffer, num_tokens_to_drop, axis, init_tokens_to_retain));
        break;
      case ::litert::ElementType::Float32:
        LITERT_RETURN_IF_ERROR(DropTokensfromTensorBuffer<float>(
            input_buffer, num_tokens_to_drop, axis, init_tokens_to_retain));
        break;
      default:
        return ::litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                    "Unsupported element type.");
    }
  }
  return {};
}

::litert::Expected<bool> DeleteTokensIfNeeded(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
        input_kv_cache_buffers,
    int num_tokens_to_drop, int init_tokens_to_retain, int current_step,
    int& start_position, size_t context_size) {
  LITERT_ASSIGN_OR_RETURN(
      bool should_delete_tokens,
      ShouldDeleteKVCacheTokens(current_step, start_position, context_size));
  if (should_delete_tokens) {
    LITERT_RETURN_IF_ERROR(DeleteTokensFromKvCache(
        input_kv_cache_buffers,
        /*num_tokens_to_drop=*/num_tokens_to_drop,
        /*init_tokens_to_retain=*/init_tokens_to_retain));
    start_position += num_tokens_to_drop;
    return true;
  }
  return false;
}

absl::Status ExpandBuffer(const uint8_t* src_data,
                          absl::Span<const int> src_shape, uint8_t* dst_data,
                          absl::Span<const int> dst_shape,
                          size_t element_size) {
  RET_CHECK_EQ(src_shape.size(), dst_shape.size());
  int expansion_axis = -1;
  for (int i = 0; i < src_shape.size(); ++i) {
    if (src_shape[i] != dst_shape[i]) {
      if (expansion_axis != -1) {
        return absl::InvalidArgumentError(
            "Tensors differ in more than one dimension.");
      }
      if (dst_shape[i] < src_shape[i]) {
        return absl::InvalidArgumentError(
            "Destination tensor dimension is smaller than source along an "
            "axis.");
      }
      expansion_axis = i;
    }
  }
  if (expansion_axis == -1) {
    return absl::InvalidArgumentError("No expansion axis found.");
  }

  int64_t dest_total_elements = 1;
  for (int dim : dst_shape) {
    dest_total_elements *= dim;
  }
  memset(dst_data, 0, dest_total_elements * element_size);

  int64_t inner_block_size_in_elements = 1;
  for (int i = expansion_axis + 1; i < src_shape.size(); ++i) {
    inner_block_size_in_elements *= src_shape[i];
  }
  const size_t inner_block_size_in_bytes =
      inner_block_size_in_elements * element_size;

  int64_t outer_block_count = 1;
  for (int i = 0; i < expansion_axis; ++i) {
    outer_block_count *= src_shape[i];
  }

  int64_t src_outer_block_stride_in_elements =
      src_shape[expansion_axis] * inner_block_size_in_elements;
  int64_t dest_outer_block_stride_in_elements =
      dst_shape[expansion_axis] * inner_block_size_in_elements;

  for (int64_t i = 0; i < outer_block_count; ++i) {
    // Calculate the starting pointer for this outer block
    const uint8_t* src_outer_block_start =
        src_data + i * src_outer_block_stride_in_elements * element_size;
    uint8_t* dest_outer_block_start =
        dst_data + i * dest_outer_block_stride_in_elements * element_size;

    // Copy each inner block from source to destination
    for (int j = 0; j < src_shape[expansion_axis]; ++j) {
      const uint8_t* src_inner_block =
          src_outer_block_start + j * inner_block_size_in_bytes;
      uint8_t* dest_inner_block =
          dest_outer_block_start + j * inner_block_size_in_bytes;
      memcpy(dest_inner_block, src_inner_block, inner_block_size_in_bytes);
    }
  }

  return absl::OkStatus();
};

}  // namespace litert::lm
