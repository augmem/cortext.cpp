// Copyright 2026 Google LLC.
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


// --- Internal Shim Start ---
#include <cstring>
#include "litert/cc/litert_tensor_buffer.h"

// Helper to replace TensorBuffer::Clear()
// which is missing in the OSS LiteRT API.
// This manually locks the buffer and memsets it to zero.
static inline absl::Status shim_clear_buffer(litert::TensorBuffer& buf) {
  void* host_mem = nullptr;
  auto status = buf.Lock(&host_mem);
  if (!status.ok()) return status;
  if (host_mem) {
    std::memset(host_mem, 0, buf.GetSize());
  }
  return buf.Unlock();
}
// --- Internal Shim End ---
