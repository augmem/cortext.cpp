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

#include "schema/core/litertlm_print.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <sstream>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl

namespace litert {
namespace lm {
namespace schema {
namespace {

TEST(LiteRTLMPrintTest, ProcessLiteRTLMFileTest) {
  const auto input_filename =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/schema/testdata/test_tok_tfl_llm.litertlm";

  std::stringstream output_ss;
  absl::Status result = ProcessLiteRTLMFile(input_filename.string(), output_ss);
  ASSERT_TRUE(result.ok());
  ASSERT_GT(output_ss.str().size(), 0);
  ASSERT_NE(output_ss.str().find("AnySectionDataType_TFLiteModel"),
            std::string::npos);
  ASSERT_NE(output_ss.str().find("AnySectionDataType_SP_Tokenizer"),
            std::string::npos);
  ASSERT_NE(output_ss.str().find("AnySectionDataType_LlmMetadataProto"),
            std::string::npos);
  ASSERT_NE(output_ss.str().find("start of LlmMetadata"),
            std::string::npos);
}

}  // namespace
}  // namespace schema
}  // namespace lm
}  // namespace litert
