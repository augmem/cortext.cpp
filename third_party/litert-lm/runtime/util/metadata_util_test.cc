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

#include "runtime/util/metadata_util.h"

#include <string>

#include <gtest/gtest.h>
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/token.pb.h"

namespace litert::lm {
namespace {

TEST(MetadataUtilTest, ExtractLlmMetadataSuccess) {
  proto::LlmMetadata llm_metadata;
  llm_metadata.mutable_start_token()->mutable_token_ids()->add_ids(2);
  llm_metadata.mutable_stop_tokens()->Add()->set_token_str("<eos>");
  std::string serialized_llm_metadata = llm_metadata.SerializeAsString();

  auto result = ExtractOrConvertLlmMetadata(serialized_llm_metadata);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.value().start_token().token_ids().ids(0), 2);
  EXPECT_EQ(result.value().stop_tokens(0).token_str(), "<eos>");
}

TEST(MetadataUtilTest, InvalidInput) {
  std::string invalid_input = "invalid_input";
  auto result = ExtractOrConvertLlmMetadata(invalid_input);
  EXPECT_FALSE(result.ok());
}

}  // namespace
}  // namespace litert::lm
