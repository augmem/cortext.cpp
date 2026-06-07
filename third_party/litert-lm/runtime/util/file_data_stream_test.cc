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
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::HasSubstr;

std::string GetTestModelPath() {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  return model_path.string();
}

TEST(FileDataStreamTest, CreateSuccessfully) {
  auto stream_or = FileDataStream::Create(GetTestModelPath());
  ASSERT_OK(stream_or);
  EXPECT_NE(*stream_or, nullptr);
}

TEST(FileDataStreamTest, CreateFailsForInvalidPath) {
  auto stream_or = FileDataStream::Create("invalid/path/to/file.ext");
  EXPECT_EQ(stream_or.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(stream_or.status().message(), HasSubstr("Failed to open file"));
}

TEST(FileDataStreamTest, ReadAndPreserve) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));

  // The very beginning of the test model is a header, starting with "LITERTLM"
  // magic number
  char buffer[8];
  ASSERT_OK(stream->ReadAndPreserve(buffer, 0, 8));

  std::string magic_number(buffer, 8);
  EXPECT_EQ(magic_number, "LITERTLM");
}

TEST(FileDataStreamTest, ReadAndDiscard) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));

  char buffer[8];
  // Since it's a file stream, ReadAndDiscard just acts like ReadAndPreserve
  ASSERT_OK(stream->ReadAndDiscard(buffer, 0, 8));

  std::string magic_number(buffer, 8);
  EXPECT_EQ(magic_number, "LITERTLM");
}

TEST(FileDataStreamTest, ReadOutOfBounds) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));

  char buffer[8];
  // Attempt to read far past the end of the file.
  // 100 GB is well out of bounds.
  uint64_t huge_offset = 100ULL * 1024 * 1024 * 1024;
  absl::Status status = stream->ReadAndPreserve(buffer, huge_offset, 8);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.message(), HasSubstr("Read past end of file"));
}

TEST(FileDataStreamTest, DiscardDoesNotFail) {
  ASSERT_OK_AND_ASSIGN(auto stream, FileDataStream::Create(GetTestModelPath()));
  ASSERT_OK(stream->Discard(0, 100));
}

}  // namespace
}  // namespace litert::lm
