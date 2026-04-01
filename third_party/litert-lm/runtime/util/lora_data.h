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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LORA_DATA_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LORA_DATA_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/util/scoped_file.h"

namespace litert::lm {

// The class holding LoRA data for LiteRT-LM on CPU.
// It is responsible for reading data with minimum copy (e.g. with mmap from a
// file) on CPU. It will provide data as a constant view.
class LoraData {
 public:
  // Creates a LoraData instance from a file path.
  //
  // @param file_path The path to the file containing the LoRA data.
  // @return A unique_ptr to the LoraData instance, or an error status.
  static absl::StatusOr<std::unique_ptr<LoraData>> CreateFromFilePath(
      absl::string_view file_path);
  // Creates a LoraData instance from a ScopedFile object.
  //
  // @param file A shared_ptr to the ScopedFile object representing the LoRA
  // data file.
  // @return A unique_ptr to the LoraData instance, or an error status.
  static absl::StatusOr<std::unique_ptr<LoraData>> CreateFromScopedFile(
      std::shared_ptr<const ScopedFile> file);

  // Create a LoraData instance from a BufferRef object.
  //
  // @param buffer A BufferRef object as the holder of the LoRA data.
  // @return A unique_ptr to the LoraData instance, or an error status.
  static absl::StatusOr<std::unique_ptr<LoraData>> CreateFromBuffer(
      BufferRef<uint8_t> buffer);

  // Get the LoRA rank from the model.
  // @return The LoRA rank, or an error status.
  virtual absl::StatusOr<int> GetLoRARank() = 0;

  // Returns the tensor data of the tensor with `name`.
  //
  // @param name The name of the tensor to read.
  // @return A unique_ptr to the BufferRef object containing the tensor data,
  // or an error status.
  virtual absl::StatusOr<std::unique_ptr<BufferRef<uint8_t>>> ReadTensor(
      absl::string_view name) = 0;

  // Returns whether the tensor with `name` exists.
  //
  // @param name The name of the tensor to check.
  // @return True if the tensor exists, false otherwise.
  virtual bool HasTensor(absl::string_view name) const = 0;

  // Returns a list of all tensor names available in the LoRA data.
  // @return A vector of strings, each representing a tensor name.
  virtual std::vector<std::string> GetAllTensorNames() const = 0;

  // Virtual destructor to allow proper cleanup of derived classes.
  virtual ~LoraData() = default;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LORA_DATA_H_
