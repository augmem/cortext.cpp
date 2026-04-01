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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TENSOR_BUFFER_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TENSOR_BUFFER_UTIL_H_

#include <vector>

#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Returns the number of dimensions that are greater than 1 in the given
// tensor buffer.
int NumSignificantDims(const litert::TensorBuffer& tensor_buffer);

// Returns the dimensions of the given tensor buffer as a vector.
std::vector<int> TensorBufferDims(const litert::TensorBuffer& tensor_buffer);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TENSOR_BUFFER_UTIL_H_
