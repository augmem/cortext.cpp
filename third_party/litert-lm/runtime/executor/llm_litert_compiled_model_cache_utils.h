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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_CACHE_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_CACHE_UTILS_H_

#include <cstddef>
#include <cstdint>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Function to check if the token deletion from the KV cache should be
// triggered if the model's current step is about to exceed the model's
// context size.
// Args:
//   current_step: The current step of the model.
//   current_step_offset: The offset of the current step.
//   context_size: The context size of the model.
// Returns:
//   A boolean indicating whether the token deletion from the KV cache should
//   be triggered.
::litert::Expected<bool> ShouldDeleteKVCacheTokens(int current_step,
                                                   int start_position,
                                                   size_t context_size);

// Function to delete tokens from the KV cache.
// Args:
//   input_kv_cache_buffers: The input KV cache buffers.
//   num_tokens_to_drop: The number of tokens to drop from the KV cache.
//   init_tokens_to_retain: The number of initial tokens to retain
//   from the KV cache to implement streamingLLM behavior.
// Returns:
//   A void indicating the success of the token deletion from the KV cache.
::litert::Expected<void> DeleteTokensFromKvCache(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
        input_kv_cache_buffers,
    int num_tokens_to_drop, int init_tokens_to_retain);

// Function to delete tokens from the KV cache if needed.
// Args:
//   input_kv_cache_buffers: The input KV cache buffers.
//   num_tokens_to_drop: The number of tokens to drop from the KV cache.
//   init_tokens_to_retain: The number of initial tokens to retain
//   from the KV cache to implement streamingLLM behavior.
//   current_step: The current step of the model.
//   start_position: The start position of the model.
//   context_size: The context size of the model.
// Returns:
//   A boolean indicating whether the token deletion from the KV cache was
//   triggered.
::litert::Expected<bool> DeleteTokensIfNeeded(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
        input_kv_cache_buffers,
    int num_tokens_to_drop, int init_tokens_to_retain, int current_step,
    int& start_position, size_t context_size);

// Function to expand the buffer from src_data to dst_data. This function can
// only handle a single expansion axis. Args:
//   src_data: The source data.
//   src_shape: The source shape.
//   dst_data: The destination data.
//   dst_shape: The destination shape.
//   element_size: The element size of the data.
// Returns:
//   Status of the expansion.
absl::Status ExpandBuffer(const uint8_t* src_data,
                          absl::Span<const int> src_shape, uint8_t* dst_data,
                          absl::Span<const int> dst_shape, size_t element_size);

}  // namespace litert::lm
#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_CACHE_UTILS_H_
