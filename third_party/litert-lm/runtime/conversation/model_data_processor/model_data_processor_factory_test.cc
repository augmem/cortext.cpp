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

#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "runtime/util/test_utils.h"     // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

class ModelDataProcessorFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto tokenizer = SentencePieceTokenizer::CreateFromFile(
        (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
         "sentencepiece.model")
            .string());
    ASSERT_OK(tokenizer);
    tokenizer_ = std::move(*tokenizer);
  }

  std::unique_ptr<Tokenizer> tokenizer_;
};

TEST_F(ModelDataProcessorFactoryTest, CreateGenericModelDataProcessor) {
  proto::LlmModelType llm_model_type;
  llm_model_type.mutable_generic_model();
  ASSERT_OK_AND_ASSIGN(
      auto config, CreateDataProcessorConfigFromLlmModelType(llm_model_type));
  ASSERT_TRUE(std::holds_alternative<GenericDataProcessorConfig>(config));
  ASSERT_OK_AND_ASSIGN(auto processor, CreateModelDataProcessor(config));
  EXPECT_OK(processor->ToInputDataVector("test prompt", {},
                                         GenericDataProcessorArguments()));
  EXPECT_THAT(processor->ToInputDataVector("test prompt", {},
                                           Gemma3DataProcessorArguments()),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_OK(
      processor->ToMessage(Responses(TaskState::kProcessing, {"test response"}),
                           GenericDataProcessorArguments()));

  EXPECT_THAT(processor->ToInputDataVector("test prompt", {},
                                           Gemma3DataProcessorArguments()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ModelDataProcessorFactoryTest, CreateGemma3DataProcessor) {
  proto::LlmModelType llm_model_type;
  llm_model_type.mutable_gemma3n();
  ASSERT_OK_AND_ASSIGN(
      auto config, CreateDataProcessorConfigFromLlmModelType(llm_model_type));
  ASSERT_TRUE(std::holds_alternative<Gemma3DataProcessorConfig>(config));
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      CreateModelDataProcessor(
          config,
          JsonPreface{
              .messages = {{{"role", "system"},
                            {"content", "You are a helpful assistant."}}}}));
  EXPECT_OK(processor->ToInputDataVector("test prompt", {},
                                         Gemma3DataProcessorArguments()));
  EXPECT_THAT(processor->ToInputDataVector("test prompt", {},
                                           GenericDataProcessorArguments()),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_OK(
      processor->ToMessage(Responses(TaskState::kProcessing, {"test response"}),
                           Gemma3DataProcessorArguments()));
  EXPECT_THAT(processor->ToInputDataVector("test prompt", {},
                                           GenericDataProcessorArguments()),
              StatusIs(absl::StatusCode::kInvalidArgument));

  llm_model_type.mutable_gemma3();
  ASSERT_OK_AND_ASSIGN(
      config, CreateDataProcessorConfigFromLlmModelType(llm_model_type));
  ASSERT_TRUE(std::holds_alternative<Gemma3DataProcessorConfig>(config));
  ASSERT_OK_AND_ASSIGN(processor, CreateModelDataProcessor(config));
  EXPECT_OK(processor->ToInputDataVector("test prompt", {},
                                         Gemma3DataProcessorArguments()));
}

TEST_F(ModelDataProcessorFactoryTest,
       CreateGemma3DataProcessorWithConstrainedDecoding) {
  auto tokenizer = SentencePieceTokenizer::CreateFromFile(
      (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
       "gemma3_sentencepiece.model")
          .string());
  ASSERT_OK(tokenizer);

  proto::LlmModelType llm_model_type;
  llm_model_type.mutable_gemma3n();
  ASSERT_OK_AND_ASSIGN(
      auto config, CreateDataProcessorConfigFromLlmModelType(llm_model_type));
  ASSERT_TRUE(std::holds_alternative<Gemma3DataProcessorConfig>(config));
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      CreateModelDataProcessor(config, /*preface=*/std::nullopt,
                               (*tokenizer).get(), {},
                               /*enable_constrained_decoding=*/true));
  EXPECT_OK(processor->ToInputDataVector("test prompt", {},
                                         Gemma3DataProcessorArguments()));
}

TEST_F(ModelDataProcessorFactoryTest, CreateQwen3ModelDataProcessor) {
  proto::LlmModelType llm_model_type;
  llm_model_type.mutable_qwen3();
  ASSERT_OK_AND_ASSIGN(
      auto config, CreateDataProcessorConfigFromLlmModelType(llm_model_type));
  ASSERT_TRUE(std::holds_alternative<Qwen3DataProcessorConfig>(config));
  ASSERT_OK_AND_ASSIGN(auto processor, CreateModelDataProcessor(config));
  EXPECT_OK(processor->ToInputDataVector("test prompt", {},
                                         Qwen3DataProcessorArguments()));
  EXPECT_THAT(processor->ToInputDataVector("test prompt", {},
                                           Gemma3DataProcessorArguments()),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_OK(
      processor->ToMessage(Responses(TaskState::kProcessing, {"test response"}),
                           Qwen3DataProcessorArguments()));

  EXPECT_THAT(processor->ToInputDataVector("test prompt", {},
                                           Gemma3DataProcessorArguments()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
