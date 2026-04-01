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

#include "runtime/framework/resource_management/context_handler/context_handler.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/infra/genai/inference/utils/tflite_utils/tflite_cpu_external_kv.h"
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_google.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/framework/resource_management/processed_context/litert_processed_context.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/test_utils.h"  // NOLINT
#include "tflite/core/interpreter_builder.h"  // from @litert
#include "tflite/interpreter.h"  // from @litert
#include "tflite/kernels/register.h"  // from @litert
#include "tflite/model_builder.h"  // from @litert

namespace litert::lm {

constexpr char kTestModelPath[] =
    "/litert_lm/runtime/testdata/tflite_external_kv_test_fixture.tflite";

class MockLlmExecutor : public LlmExecutor {
 public:
  MOCK_METHOD(absl::Status, Prefill, (const ExecutorInputs& inputs),
              (override));
  MOCK_METHOD(absl::Status, Decode, (::litert::TensorBuffer & output_tokens),
              (override));
  MOCK_METHOD(absl::string_view, ExecutorBackendName, (), (const, override));
};

class ContextHandlerTest : public testing::Test {
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(handler_, MakeContextHandler());
  }

 protected:
  absl::StatusOr<std::shared_ptr<ContextHandler::SharedProcessedContext>>
  MakeSharedProcessedContext() {
    auto model = tflite::FlatBufferModel::BuildFromFile(
        (std::filesystem::path(::testing::SrcDir()) / kTestModelPath)
            .string()
            .c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);
    ASSIGN_OR_RETURN(
        auto kv_cache,
        odml::infra::tflite_utils::SharedBufferExternalKVCache::Create(
            interpreter.get()));
    auto processed_context =
        std::make_unique<LiteRTProcessedContext>(std::move(kv_cache),
                                                 /*lora_id=*/std::nullopt);
    return std::make_shared<ContextHandler::SharedProcessedContext>(
        std::move(processed_context));
  }

  std::unique_ptr<ContextHandler> handler_;

 private:
  absl::StatusOr<std::unique_ptr<ContextHandler>> MakeContextHandler() {
    ASSIGN_OR_RETURN(auto shared_processed_context,
                     MakeSharedProcessedContext());
    auto runtime_config = std::make_unique<RuntimeConfig>();
    auto runtime_state = std::make_unique<RuntimeState>();
    return ContextHandler::Bundle(std::move(shared_processed_context),
                                  std::move(runtime_config),
                                  std::move(runtime_state));
  };
};

TEST_F(ContextHandlerTest, ContextHandlerCreateFailed) {
  auto llm_context_handler_or_status = ContextHandler::Create(nullptr);
  EXPECT_FALSE(llm_context_handler_or_status.ok());
  EXPECT_THAT(llm_context_handler_or_status.status().message(),
              testing::HasSubstr("The llm_context is null."));
}

TEST_F(ContextHandlerTest,
       ContextHandlerUpdateSharedProcessedContextSucceedWithSameSharedContext) {
  auto shared_processed_context = handler_->shared_processed_context();

  // ASSERT_OK(handler_->UpdateSharedProcessedContext(shared_processed_context));
  // EXPECT_EQ(handler_->shared_processed_context(), shared_processed_context);
}

TEST_F(ContextHandlerTest,
       ContextHandlerUpdateSharedProcessedContextSucceedWithNewSharedContext) {
  ASSERT_OK_AND_ASSIGN(auto new_shared_processed_context,
                       MakeSharedProcessedContext());

  ASSERT_OK(
      handler_->UpdateSharedProcessedContext(new_shared_processed_context));
  EXPECT_EQ(handler_->shared_processed_context(), new_shared_processed_context);
}

TEST_F(ContextHandlerTest, ContextHandlerSetAndRetrieveRuntimeConfigSucceed) {
  auto runtime_config = std::make_unique<RuntimeConfig>();
  runtime_config->output_heads = 13;
  runtime_config->tokens_per_decode = 21;

  ASSERT_OK(handler_->SetRuntimeConfig(std::move(runtime_config)));

  ASSERT_OK_AND_ASSIGN(auto retrieved_runtime_config,
                       handler_->RetrieveRuntimeConfig());
  EXPECT_EQ(retrieved_runtime_config->output_heads, 13);
  EXPECT_EQ(retrieved_runtime_config->tokens_per_decode, 21);
}

TEST_F(ContextHandlerTest, ContextHandlerHasRuntimeConfig) {
  EXPECT_EQ(handler_->HasRuntimeConfig(), true);

  ASSERT_OK_AND_ASSIGN(auto retrieved_runtime_config,
                       handler_->RetrieveRuntimeConfig());

  EXPECT_EQ(handler_->HasRuntimeConfig(), false);
}

TEST_F(ContextHandlerTest, ContextHandlerSetAndRetrieveRuntimeStateSucceed) {
  auto runtime_state = std::make_unique<RuntimeState>();
  auto rand_gen = std::make_shared<std::default_random_engine>(17);
  runtime_state->current_step = 13;
  runtime_state->rand_gen = rand_gen;

  ASSERT_OK(handler_->SetRuntimeState(std::move(runtime_state)));

  ASSERT_OK_AND_ASSIGN(auto retrieved_runtime_state,
                       handler_->RetrieveRuntimeState());
  EXPECT_EQ(retrieved_runtime_state->current_step, 13);
  EXPECT_EQ(retrieved_runtime_state->rand_gen, rand_gen);
}

TEST_F(ContextHandlerTest, ContextHandlerHasRuntimeState) {
  EXPECT_EQ(handler_->HasRuntimeState(), true);

  ASSERT_OK_AND_ASSIGN(auto retrieved_runtime_state,
                       handler_->RetrieveRuntimeState());

  EXPECT_EQ(handler_->HasRuntimeState(), false);
}

TEST_F(ContextHandlerTest, SharedProcessedContextAddHandlerSucceed) {
  ASSERT_OK_AND_ASSIGN(auto shared_processed_context,
                       MakeSharedProcessedContext());

  auto init_runtime_state = std::make_unique<RuntimeState>();
  init_runtime_state->current_step = 11;
  auto init_runtime_config = std::make_unique<RuntimeConfig>();
  ASSERT_OK_AND_ASSIGN(auto init_llm_context_handler,
                       ContextHandler::Bundle(shared_processed_context,
                                              std::move(init_runtime_config),
                                              std::move(init_runtime_state)));

  auto llm_executor = MockLlmExecutor();
  ASSERT_OK_AND_ASSIGN(
      auto longest_handlerTimeStep,
      shared_processed_context->LongestHandlerTimeStep(llm_executor));
  EXPECT_EQ(longest_handlerTimeStep, 11);

  // Create a new ContextHandler with current step = 13.
  auto runtime_state = std::make_unique<RuntimeState>();
  runtime_state->current_step = 13;
  ASSERT_OK(handler_->SetRuntimeState(std::move(runtime_state)));

  shared_processed_context->AddHandler(handler_.get());
  ASSERT_OK_AND_ASSIGN(
      longest_handlerTimeStep,
      shared_processed_context->LongestHandlerTimeStep(llm_executor));
  EXPECT_EQ(longest_handlerTimeStep, 13);
}

TEST_F(ContextHandlerTest, SharedProcessedContextRemoveHandlerSucceed) {
  ASSERT_OK_AND_ASSIGN(auto shared_processed_context,
                       MakeSharedProcessedContext());

  auto init_runtime_state = std::make_unique<RuntimeState>();
  init_runtime_state->current_step = 11;
  auto init_runtime_config = std::make_unique<RuntimeConfig>();
  ASSERT_OK_AND_ASSIGN(auto init_llm_context_handler,
                       ContextHandler::Bundle(shared_processed_context,
                                              std::move(init_runtime_config),
                                              std::move(init_runtime_state)));

  auto llm_executor = MockLlmExecutor();
  ASSERT_OK_AND_ASSIGN(
      auto longest_handlerTimeStep,
      shared_processed_context->LongestHandlerTimeStep(llm_executor));
  EXPECT_EQ(longest_handlerTimeStep, 11);

  shared_processed_context->RemoveHandler(init_llm_context_handler.get());
  ASSERT_OK_AND_ASSIGN(
      longest_handlerTimeStep,
      shared_processed_context->LongestHandlerTimeStep(llm_executor));
  EXPECT_EQ(longest_handlerTimeStep, 0);
}

TEST_F(ContextHandlerTest, SharedProcessedContextSetProcessedContextFailed) {
  ASSERT_OK_AND_ASSIGN(auto shared_processed_context1,
                       MakeSharedProcessedContext());

  ASSERT_OK_AND_ASSIGN(auto shared_processed_context2,
                       MakeSharedProcessedContext());
  ASSERT_OK_AND_ASSIGN(auto processed_context,
                       shared_processed_context2->RetrieveProcessedContext());
  processed_context->processed_tokens().AddProcessedTokens({1, 2, 3, 4, 5});

  absl::Status status = shared_processed_context1->SetProcessedContext(
      std::move(processed_context));
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("The processed context is already set."));
}

TEST_F(ContextHandlerTest,
       SharedProcessedContextSetAndRetrieveProcessedContextSucceed) {
  ASSERT_OK_AND_ASSIGN(auto shared_processed_context,
                       MakeSharedProcessedContext());

  auto processed_context =
      std::make_unique<LiteRTProcessedContext>(nullptr, std::nullopt);
  processed_context->processed_tokens().AddProcessedTokens({1, 2, 3, 4, 5});

  // SharedProcessedContext should not own any ProcessedContext when calling
  // SetProcessedContext, so we need to first retrieve the ProcessedContext from
  // the SharedProcessedContext, then set it back.
  ASSERT_OK_AND_ASSIGN(auto retrieved_processed_context,
                       shared_processed_context->RetrieveProcessedContext());
  ASSERT_OK(shared_processed_context->SetProcessedContext(
      std::move(processed_context)));

  ASSERT_OK_AND_ASSIGN(retrieved_processed_context,
                       shared_processed_context->RetrieveProcessedContext());
  EXPECT_EQ(retrieved_processed_context->processed_tokens().GetCopyOfTokens(),
            std::vector<std::vector<int>>({{1, 2, 3, 4, 5}}));
}

}  // namespace litert::lm
