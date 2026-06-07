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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_FILE_DATA_STREAM_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_FILE_DATA_STREAM_H_

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/util/data_stream.h"

namespace litert::lm {

// A data stream for reading data from a file.
// This is not thread-safe.
class FileDataStream : public DataStream {
 public:
  static absl::StatusOr<std::shared_ptr<FileDataStream>> Create(
      const std::string& file_path);

  ~FileDataStream() override;

  // Reads `size` bytes starting at `offset` into `buffer`. Discarding is a
  // no-op for this file-based implementation.
  absl::Status ReadAndDiscard(void* buffer, uint64_t offset,
                              uint64_t size) override;

  // Reads `size` bytes starting at `offset` into `buffer`.
  absl::Status ReadAndPreserve(void* buffer, uint64_t offset,
                               uint64_t size) override;

  // Discards `size` bytes starting at `offset`. A no-op for this file-based
  // implementation.
  absl::Status Discard(uint64_t offset, uint64_t size) override;

 private:
  explicit FileDataStream(std::ifstream file);

  std::ifstream file_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_FILE_DATA_STREAM_H_
