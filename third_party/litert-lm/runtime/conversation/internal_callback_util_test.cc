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

#include "runtime/conversation/internal_callback_util.h"

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::status::StatusIs;

nlohmann::ordered_json TextMessage(absl::string_view text) {
  nlohmann::ordered_json message;
  message["role"] = "assistant";
  message["content"] = {{{"type", "text"}, {"text", text}}};
  return message;
}

absl::AnyInvocable<void(absl::StatusOr<Message>)> CreateUserMessageCallback(
    std::vector<nlohmann::ordered_json>& output, bool& done,
    absl::Status& status) {
  return [&](absl::StatusOr<Message> message) {
    if (!message.ok()) {
      done = true;
      status = message.status();
      return;
    }
    if (auto json_message = std::get_if<JsonMessage>(&*message)) {
      if (json_message->is_null()) {
        done = true;
      } else {
        output.push_back(*json_message);
      }
    }
  };
}

class InternalCallbackTest : public testing::Test {
 protected:
  void SetUp() override {
    Gemma3DataProcessorConfig config;

    // Need a tool in the preface to trigger tool call parsing. The actual tool
    // definition is unimportant.
    JsonPreface preface{.tools = nlohmann::ordered_json::parse(R"json([{
                  "name": "tool_name",
                  "parameters": { "properties": { "x": { "type": "integer" } } }
                }])json")};
    ASSERT_OK_AND_ASSIGN(model_data_processor_,
                         Gemma3DataProcessor::Create(config, preface));

    processor_args_ = DataProcessorArguments();
  }

  std::unique_ptr<Gemma3DataProcessor> model_data_processor_;
  std::vector<nlohmann::ordered_json> output_;
  bool done_ = false;
  absl::Status status_;
  DataProcessorArguments processor_args_;
};

TEST_F(InternalCallbackTest, OnDone) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kDone));

  EXPECT_THAT(output_, IsEmpty());
  EXPECT_TRUE(done_);
  EXPECT_OK(status_);
}

TEST_F(InternalCallbackTest, OnError) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(absl::InternalError("error"));

  EXPECT_THAT(output_, IsEmpty());
  EXPECT_TRUE(done_);
  EXPECT_THAT(status_, StatusIs(absl::StatusCode::kInternal, "error"));
}

TEST_F(InternalCallbackTest, Text) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"this "}));
  callback(Responses(TaskState::kProcessing, {"is "}));
  callback(Responses(TaskState::kProcessing, {"some "}));
  callback(Responses(TaskState::kProcessing, {"text"}));

  EXPECT_THAT(output_, ElementsAre(TextMessage("this "), TextMessage("is "),
                                   TextMessage("some "), TextMessage("text")));
}

TEST_F(InternalCallbackTest, ToolCall) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name"}));
  callback(Responses(TaskState::kProcessing, {"(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, TextAndToolCall) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"this "}));
  callback(Responses(TaskState::kProcessing, {"is "}));
  callback(Responses(TaskState::kProcessing, {"some "}));
  callback(Responses(TaskState::kProcessing, {"text\n"}));
  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name"}));
  callback(Responses(TaskState::kProcessing, {"(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(TextMessage("this "), TextMessage("is "),
                                   TextMessage("some "), TextMessage("text\n"),
                                   nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_name",
                                  "arguments": {
                                    "x": 1
                                  }
                                }
                              }
                            ]
                          })json")));
}

TEST_F(InternalCallbackTest, SplitCodeFenceStart) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_"}));
  callback(Responses(TaskState::kProcessing, {"code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name"}));
  callback(Responses(TaskState::kProcessing, {"(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, TextBeforeSplitCodeFenceStart) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"text```tool_"}));
  callback(Responses(TaskState::kProcessing, {"code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name"}));
  callback(Responses(TaskState::kProcessing, {"(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(TextMessage("text"),
                                   nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, ToolCallAfterSplitCodeFenceStart) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```"}));
  callback(Responses(TaskState::kProcessing, {"tool_code\ntool_name"}));
  callback(Responses(TaskState::kProcessing, {"(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, TextOnBothSidesOfCodeFenceStart) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"text```tool_code\ntool_name"}));
  callback(Responses(TaskState::kProcessing, {"(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(TextMessage("text"),
                                   nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, SplitCodeFenceEnd) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n`"}));
  callback(Responses(TaskState::kProcessing, {"``"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, TextBeforeSplitCodeFenceEnd) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x="}));
  callback(Responses(TaskState::kProcessing, {"1)\n``"}));
  callback(Responses(TaskState::kProcessing, {"`"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, TextAfterSplitCodeFenceEnd) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n`"}));
  callback(Responses(TaskState::kProcessing, {"``text"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_name",
                                  "arguments": {
                                    "x": 1
                                  }
                                }
                              }
                            ]
                          })json"),
                                   TextMessage("text")));
}

TEST_F(InternalCallbackTest, OnNextTextOnBothSidesOfSplitCodeFenceEnd) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x="}));
  callback(Responses(TaskState::kProcessing, {"1)\n`"}));
  callback(Responses(TaskState::kProcessing, {"``text"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_name",
                                  "arguments": {
                                    "x": 1
                                  }
                                }
                              }
                            ]
                          })json"),
                                   TextMessage("text")));
}

TEST_F(InternalCallbackTest, ParallelToolCalls) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_a(x=1)\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_b(y='z')"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json(
                {
                  "role": "assistant",
                  "tool_calls": [
                    {
                      "type": "function",
                      "function": {
                        "name": "tool_a",
                        "arguments": {
                          "x": 1
                        }
                      }
                    },
                    {
                      "type": "function",
                      "function": {
                        "name": "tool_b",
                        "arguments": {
                          "y": "z"
                        }
                      }
                    }
                  ]
                }
                )json")));
}

TEST_F(InternalCallbackTest, TwoConsecutiveToolCodeBlocks) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_a(x=1)\n"}));
  callback(Responses(TaskState::kProcessing, {"``````tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_b(y='z')\n"}));
  callback(Responses(TaskState::kProcessing, {"```"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_a",
                                  "arguments": {
                                    "x": 1
                                  }
                                }
                              }
                            ]
                          })json"),
                                   nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_b",
                                  "arguments": {
                                    "y": "z"
                                  }
                                }
                              }
                            ]
                          })json")));
}

TEST_F(InternalCallbackTest, IncompleteToolCodeBlock) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x=1)"}));
  callback(Responses(TaskState::kDone));

  // The incomplete tool code block is sent to the callback as a text message.
  EXPECT_THAT(output_,
              ElementsAre(TextMessage("```tool_code\ntool_name(x=1)")));
}

TEST_F(InternalCallbackTest, WrongCodeFenceStart) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));
  callback(Responses(TaskState::kDone));

  EXPECT_THAT(output_, ElementsAre(TextMessage("```tool\n"),
                                   TextMessage("tool_name(x=1)"),
                                   TextMessage("\n"), TextMessage("```")));
}

TEST_F(InternalCallbackTest, WrongCodeFenceEnd) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n``x"}));
  callback(Responses(TaskState::kDone));

  EXPECT_THAT(output_,
              ElementsAre(TextMessage("```tool_code\ntool_name(x=1)\n``x")));
}

TEST_F(InternalCallbackTest, InvalidFunctionCall) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"not a function call"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_TRUE(done_);
  EXPECT_THAT(status_, StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
