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

#include "runtime/util/zip_utils.h"

#include <filesystem>  // NOLINT: Required for path manipulation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {

namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(ZipUtilsTest, ExtractFilesFromZipFile) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";

  ASSERT_OK_AND_ASSIGN(auto file,
                       MemoryMappedFile::Create(model_path.string()));
  absl::string_view data(reinterpret_cast<const char*>(file->data()),
                         file->length());

  ASSERT_OK_AND_ASSIGN(auto files, ExtractFilesfromZipFile(data));

  EXPECT_THAT(files,
              UnorderedElementsAre(
                  Pair("TF_LITE_PREFILL_DECODE", OffsetAndSize{56, 27440104}),
                  Pair("TOKENIZER_MODEL", OffsetAndSize{27440208, 4689074}),
                  Pair("METADATA", OffsetAndSize{32129324, 22})));
}

}  // namespace

}  // namespace litert::lm
