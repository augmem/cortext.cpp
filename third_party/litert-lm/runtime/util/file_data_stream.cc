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

#include "runtime/util/file_data_stream.h"

#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl

namespace litert::lm {

absl::StatusOr<std::shared_ptr<FileDataStream>> FileDataStream::Create(
    const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open file: ", file_path));
  }
  // Cannot use std::make_shared because constructor is private.
  return std::shared_ptr<FileDataStream>(new FileDataStream(std::move(file)));
}

FileDataStream::FileDataStream(std::ifstream file) : file_(std::move(file)) {}

FileDataStream::~FileDataStream() = default;

absl::Status FileDataStream::ReadAndDiscard(void* buffer, uint64_t offset,
                                            uint64_t size) {
  return ReadAndPreserve(buffer, offset, size);
}

absl::Status FileDataStream::ReadAndPreserve(void* buffer, uint64_t offset,
                                             uint64_t size) {
  if (!file_.is_open()) {
    return absl::InternalError("File is not open.");
  }

  file_.clear();
  file_.seekg(offset, std::ios::beg);
  if (file_.fail()) {
    return absl::InternalError(
        absl::StrCat("Failed to seek to offset ", offset));
  }

  file_.read(static_cast<char*>(buffer), size);
  if (static_cast<uint64_t>(file_.gcount()) != size) {
    return absl::OutOfRangeError(
        absl::StrCat("Read past end of file. Requested ", size,
                     " bytes, but only ", file_.gcount(), " bytes were read."));
  }

  return absl::OkStatus();
}

absl::Status FileDataStream::Discard(uint64_t offset, uint64_t size) {
  // Discard is a no-op for a random-access file stream.
  return absl::OkStatus();
}

}  // namespace litert::lm
