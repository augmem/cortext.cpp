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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_COMMON_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_COMMON_UTILS_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

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

// Function to copy the buffer from src_buffer to dst_buffer.
// Args:
//   src_buffer: The source buffer.
//   dst_buffer: The destination buffer.
//   src_offset: The offset of the source buffer.
//   dst_offset: The offset of the destination buffer.
//   size: The size of the buffer to copy. If -1, copy the whole buffer.
// Returns:
//   Status of the copy.
absl::Status CopyBuffer(const TensorBuffer& src_buffer,
                        TensorBuffer& dst_buffer, size_t src_offset = 0,
                        size_t dst_offset = 0, int64_t size = -1);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_COMMON_UTILS_H_
