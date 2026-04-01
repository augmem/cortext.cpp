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

#include "runtime/framework/resource_management/execution_manager.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::Return;

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view token),
              (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (const std::vector<int>& token_ids), (override));
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
};

class ExecutionManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tokenizer_ = std::make_unique<MockTokenizer>();
    EXPECT_CALL(*tokenizer_, TokenIdsToText(ElementsAre(0)))
        .WillRepeatedly(Return("0"));
    EXPECT_CALL(*tokenizer_, TokenIdsToText(ElementsAre(4)))
        .WillRepeatedly(Return("4"));
    EXPECT_CALL(*tokenizer_, TokenIdsToText(ElementsAre(5)))
        .WillRepeatedly(Return("5"));
    EXPECT_CALL(*tokenizer_, TokenIdsToText(ElementsAre(6)))
        .WillRepeatedly(Return("6"));
  }

  absl::StatusOr<SessionConfig> CreateDefaultSessionConfig(
      bool use_external_sampler = false) {
    ASSIGN_OR_RETURN(auto model_assets,
                     ModelAssets::Create("test_model_path_1"));
    ASSIGN_OR_RETURN(auto settings,
                     EngineSettings::CreateDefault(model_assets));

    proto::LlmMetadata llm_metadata;
    llm_metadata.mutable_stop_tokens()
        ->Add()
        ->mutable_token_ids()
        ->mutable_ids()
        ->Add(0);
    llm_metadata.mutable_stop_tokens()
        ->Add()
        ->mutable_token_ids()
        ->mutable_ids()
        ->Add(6);
    llm_metadata.mutable_llm_model_type()->mutable_gemma3n();
    EXPECT_OK(settings.MaybeUpdateAndValidate(*tokenizer_, &llm_metadata));
    SessionConfig session_config = SessionConfig::CreateDefault();
    EXPECT_OK(session_config.MaybeUpdateAndValidate(settings));
    session_config.SetUseExternalSampler(use_external_sampler);

    return session_config;
  };

  void CreateExecutionManager(
      std::unique_ptr<FakeLlmExecutor> fake_llm_executor) {
    std::optional<BenchmarkInfo> benchmark_info = std::nullopt;

    // The objects are moved to execution_manager_ so we can't access them
    // after creation.
    ASSERT_OK_AND_ASSIGN(execution_manager_,
                         ExecutionManager::Create(
                             /*tokenizer=*/tokenizer_.get(),
                             /*llm_executor=*/std::move(fake_llm_executor),
                             /*vision_executor_settings=*/nullptr,
                             /*audio_executor_settings=*/nullptr,
                             /*litert_env=*/nullptr));
  }

  std::unique_ptr<FakeLlmExecutor> CreateDefaultFakeLlmExecutor() {
    auto prefill_tokens = std::vector<std::vector<int>>{{1, 2, 3}};
    auto decode_tokens = std::vector<std::vector<int>>{{4}, {5}, {6}};
    return std::make_unique<FakeLlmExecutor>(
        /*vocab_size=*/10,
        /*prefill_tokens=*/std::move(prefill_tokens),
        /*decode_tokens=*/std::move(decode_tokens));
  }

  std::unique_ptr<MockTokenizer> tokenizer_;

  std::unique_ptr<ExecutionManager> execution_manager_;
};

TEST_F(ExecutionManagerTest, AddPrefillTask) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());
  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<TaskState> task_states;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&task_states](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        task_states.push_back(responses->GetTaskState());
      };

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));

  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {}, std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));

  EXPECT_THAT(
      task_states,
      ElementsAre(TaskState::kCreated, TaskState::kQueued,
                  TaskState::kProcessing, TaskState::kDone));
}

TEST_F(ExecutionManagerTest, AddDecodeTaskWithInternalSampler) {
  // The default execution manager is using the internal sampler.
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<TaskState> task_states;
  std::vector<std::string> responses_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&task_states, &responses_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        task_states.push_back(responses->GetTaskState());
        if (!responses->GetTexts().empty()) {
          responses_texts.push_back(responses->GetTexts()[0]);
        }
      };

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs), {},
      [](absl::StatusOr<Responses> responses) {}));
  ASSERT_OK(
      execution_manager_->WaitUntilDone(prefill_task_id, absl::Seconds(3)));

  ASSERT_OK_AND_ASSIGN(const TaskId decode_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, decode_task_id,
      /*dependency_task_ids=*/{},
      /*constraint=*/nullptr, /*cancelled=*/nullptr, std::move(callback)));

  EXPECT_OK(
      execution_manager_->WaitUntilDone(decode_task_id, absl::Seconds(3)));

  EXPECT_THAT(task_states,
              ElementsAre(TaskState::kCreated, TaskState::kQueued,
                          TaskState::kProcessing, TaskState::kProcessing,
                          TaskState::kProcessing, TaskState::kDone));

  EXPECT_THAT(responses_texts, ElementsAre("4", "5"));
}

TEST_F(ExecutionManagerTest, AddDecodeTaskWithExternalSampler) {
  std::vector<std::vector<int>> prefill_tokens = {{1, 2, 3}, {6}};
  std::vector<std::vector<int>> decode_tokens = {{4}, {5}, {6}};

  CreateExecutionManager(std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/10,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens)));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig(
                                                /*use_external_sampler=*/true));
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<TaskState> task_states;
  std::vector<std::string> responses_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&task_states, &responses_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        task_states.push_back(responses->GetTaskState());
        if (!responses->GetTexts().empty()) {
          responses_texts.push_back(responses->GetTexts()[0]);
        }
      };

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId prefill_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, prefill_task_id, std::move(inputs), {},
      [](absl::StatusOr<Responses> responses) {}));
  ASSERT_OK(
      execution_manager_->WaitUntilDone(prefill_task_id, absl::Seconds(3)));

  ASSERT_OK_AND_ASSIGN(const TaskId decode_task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(session_id, decode_task_id,
                                              /*dependency_task_ids=*/{},
                                              /*constraint=*/nullptr,
                                              /*constraint_params=*/nullptr,
                                              std::move(callback)));

  EXPECT_OK(
      execution_manager_->WaitUntilDone(decode_task_id, absl::Seconds(3)));

  EXPECT_THAT(
      task_states,
      ElementsAre(TaskState::kCreated, TaskState::kQueued,
                  TaskState::kProcessing, TaskState::kProcessing,
                  TaskState::kProcessing, TaskState::kDone));

  EXPECT_THAT(responses_texts, ElementsAre("4", "5"));
}

TEST_F(ExecutionManagerTest, CreateAndRunDependentTasks) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(session_id, task_a_id,
                                               std::move(inputs),
                                               /*dependency_task_ids=*/{},
                                               /*callback=*/nullptr));

  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id},
      /*constraint=*/nullptr, /*cancelled=*/nullptr,
      /*callback=*/nullptr));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(1)));
  EXPECT_OK(execution_manager_->WaitUntilDone(task_a_id, absl::Seconds(1)));
}

TEST_F(ExecutionManagerTest, CreateTaskWithInvalidDependency) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  inputs.push_back(InputText("test"));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  auto add_task_status = execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs),
      /*dependency_task_ids=*/{12345}, /*callback=*/nullptr);
  EXPECT_FALSE(add_task_status.ok());
  EXPECT_EQ(add_task_status.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(ExecutionManagerTest, CreateTaskWithInvalidDependencyId) {
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  // Add a valid task.
  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(session_id, task_a_id,
                                               std::move(inputs),
                                               /*dependency_task_ids=*/{},
                                               /*callback=*/nullptr));
  EXPECT_OK(execution_manager_->WaitUntilDone(task_a_id, absl::Seconds(1)));

  // Try to add a task with an invalid dependency.
  std::vector<InputData> inputs_b;
  ASSERT_OK_AND_ASSIGN(auto input_text_b,
                       tokenizer_->TokenIdsToTensorBuffer({4, 5, 6}));
  inputs_b.push_back(InputText(std::move(input_text_b)));
  const TaskId invalid_task_id = 99999;
  auto task_status = execution_manager_->AddPrefillTask(
      session_id, invalid_task_id, std::move(inputs_b),
      /*dependency_task_ids=*/{invalid_task_id},
      /*callback=*/nullptr);
  EXPECT_FALSE(task_status.ok());
  EXPECT_EQ(task_status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(task_status.message(),
              testing::HasSubstr("Dependency task 99999 not found"));
}

TEST_F(ExecutionManagerTest, WaitUntilTaskDoneTimeout) {
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  decode_tokens.push_back({4});
  decode_tokens.push_back({5});
  decode_tokens.push_back({6});
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/10,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));

  // Inject a long delay to simulate a timeout.
  fake_llm_executor->SetDecodeDelay(absl::Seconds(0.5));

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(session_id, task_id,
                                              /*dependency_task_ids=*/{},

                                              /*constraint=*/nullptr,
                                              /*cancelled=*/nullptr,
                                              /*callback=*/nullptr));

  EXPECT_EQ(
      execution_manager_->WaitUntilDone(task_id, absl::Milliseconds(100)),
      absl::DeadlineExceededError(absl::StrCat(
          "Task ", task_id, " did not complete within the timeout of 100ms.")));

  // Wait for the task to actually finish to avoid use after free.
  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));
}

TEST_F(ExecutionManagerTest, WaitUntilAllDoneTimeout) {
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  decode_tokens.push_back({4});
  decode_tokens.push_back({5});
  decode_tokens.push_back({6});
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/10,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));

  // Inject a long delay to simulate a timeout.
  fake_llm_executor->SetDecodeDelay(absl::Seconds(0.5));

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_id,
      /*dependency_task_ids=*/{},
      /*constraint=*/nullptr, /*cancelled=*/nullptr, /*callback=*/nullptr));

  EXPECT_EQ(
      execution_manager_->WaitUntilAllDone(absl::Milliseconds(100)).code(),
      absl::StatusCode::kDeadlineExceeded);

  // Wait for the task to actually finish to avoid use after free.
  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(3)));
}

TEST_F(ExecutionManagerTest, TaskReturnsError) {
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  prefill_tokens.push_back({1, 2, 3});
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/10,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));

  // Inject an error.
  fake_llm_executor->SetPrefillStatus(absl::InternalError("Executor failed"));

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  absl::Status final_status = absl::OkStatus();
  ASSERT_OK_AND_ASSIGN(const TaskId task_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_id, std::move(inputs), {},
      [&](absl::StatusOr<Responses> responses) {
        if (!responses.ok()) {
          final_status = responses.status();
        }
      }));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_id, absl::Seconds(1)));
  EXPECT_EQ(final_status, absl::InternalError("Executor failed"));
}

TEST_F(ExecutionManagerTest, CreateDependentTaskOnFailedTask) {
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  prefill_tokens.push_back({1, 2, 3});
  decode_tokens.push_back({4});
  decode_tokens.push_back({5});
  decode_tokens.push_back({6});
  auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/10,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens));

  // Inject an error.
  fake_llm_executor->SetPrefillStatus(absl::InternalError("Executor failed"));

  CreateExecutionManager(std::move(fake_llm_executor));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  absl::Status task_a_status = absl::OkStatus();
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(
      session_id, task_a_id, std::move(inputs), {},
      [&](absl::StatusOr<Responses> responses) {
        task_a_status = responses.status();
      }));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_a_id, absl::Seconds(1)));
  EXPECT_EQ(task_a_status, absl::InternalError("Executor failed"));

  absl::Status task_b_status = absl::OkStatus();
  std::vector<TaskState> task_b_states;
  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id},
      /*constraint=*/nullptr, /*cancelled=*/nullptr,
      [&](absl::StatusOr<Responses> responses) {
        task_b_status = responses.status();
        if (responses.ok()) {
          task_b_states.push_back(responses->GetTaskState());
        }
      }));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(1)));
  EXPECT_EQ(task_b_status, absl::OkStatus());
  EXPECT_THAT(task_b_states, ElementsAre(TaskState::kDependentTaskFailed));
}

TEST_F(ExecutionManagerTest, AddDecodeTaskWithConstraintWithInternalSampler) {
  // The default execution manager is using the internal sampler.
  CreateExecutionManager(CreateDefaultFakeLlmExecutor());

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig());
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(session_id, task_a_id,
                                               std::move(inputs),
                                               /*dependency_task_ids=*/{},
                                               /*callback=*/nullptr));

  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  // Fake constraint that expects "45".
  std::vector<int> expected_token_ids = {4, 0};
  auto constraint = FakeConstraint(expected_token_ids, /*vocabulary_size=*/10);
  auto decode_config = DecodeConfig::CreateDefault();
  decode_config.SetConstraint(&constraint);
  std::vector<std::string> response_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&response_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (!responses->GetTexts().empty()) {
          response_texts.push_back(responses->GetTexts()[0]);
        }
      };

  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id}, decode_config.GetConstraint(),
      /*cancelled=*/nullptr, std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(3)));

  EXPECT_THAT(response_texts, ElementsAre("4"));
}

TEST_F(ExecutionManagerTest, AddDecodeTaskWithConstraintWithExternalSampler) {
  auto prefill_tokens = std::vector<std::vector<int>>{};
  auto decode_tokens = std::vector<std::vector<int>>{};
  prefill_tokens.push_back({1, 2, 3});
  prefill_tokens.push_back({0});
  decode_tokens.push_back({4});
  decode_tokens.push_back({5});
  decode_tokens.push_back({6});

  CreateExecutionManager(std::make_unique<FakeLlmExecutor>(
      /*vocab_size=*/10,
      /*prefill_tokens=*/std::move(prefill_tokens),
      /*decode_tokens=*/std::move(decode_tokens)));

  ASSERT_OK_AND_ASSIGN(auto session_config, CreateDefaultSessionConfig(
                                                /*use_external_sampler=*/true));
  ASSERT_OK_AND_ASSIGN(const SessionId session_id,
                       execution_manager_->RegisterNewSession(session_config));

  std::vector<InputData> inputs;
  ASSERT_OK_AND_ASSIGN(auto input_text,
                       tokenizer_->TokenIdsToTensorBuffer({1, 2, 3}));
  inputs.push_back(InputText(std::move(input_text)));
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  ASSERT_OK_AND_ASSIGN(const TaskId task_a_id,
                       execution_manager_->GetNewTaskId());
  ASSERT_OK(execution_manager_->AddPrefillTask(session_id, task_a_id,
                                               std::move(inputs),
                                               /*dependency_task_ids=*/{},
                                               /*callback=*/nullptr));

  ASSERT_OK_AND_ASSIGN(const TaskId task_b_id,
                       execution_manager_->GetNewTaskId());
  // Fake constraint that expects "45".
  std::vector<int> expected_token_ids = {4, 0};
  auto constraint = FakeConstraint(expected_token_ids, /*vocabulary_size=*/10);
  auto decode_config = DecodeConfig::CreateDefault();
  decode_config.SetConstraint(&constraint);
  std::vector<std::string> response_texts;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
      [&response_texts](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        if (!responses->GetTexts().empty()) {
          response_texts.push_back(responses->GetTexts()[0]);
        }
      };

  ASSERT_OK(execution_manager_->AddDecodeTask(
      session_id, task_b_id,
      /*dependency_task_ids=*/{task_a_id}, decode_config.GetConstraint(),
      /*cancelled=*/nullptr, std::move(callback)));

  EXPECT_OK(execution_manager_->WaitUntilDone(task_b_id, absl::Seconds(3)));

  EXPECT_THAT(response_texts, ElementsAre("4"));
}

}  // namespace
}  // namespace litert::lm
