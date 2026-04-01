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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CONVERSATION_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CONVERSATION_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Configuration for the Conversation instance. This class is used to
// initialize the Conversation instance. The configuration is created from the
// Engine with default SessionConfig, or from a provided SessionConfig, with
// optional overwrite for the prompt template, processor config.
class ConversationConfig {
 public:
  // Creates a default ConversationConfig from the given Engine.
  // Args:
  // - `engine`: The Engine instance to be used for creating the default config.
  // - `preface`: Optional Preface for the conversation. The Preface provides
  //     the initial background for the conversation, tool uses and extra
  //     context for the conversation. If not provided, the conversation will
  //     start with an empty Preface.
  // - `overwrite_prompt_template`: Optional PromptTemplate instance to be used
  //     for the conversation. If not provided, the conversation will use the
  //     template read from the model metadata.
  // - `overwrite_processor_config`: Optional configuration for the model data
  //     processor, if not provided, the default config for the model type's
  //     data processor will be used. Most of the time, the users don't need to
  //     provide the data processor config.
  // - `enable_constrained_decoding`: Whether to enable constrained decoding. If
  //     true, constrained decoding will be used, primarily for function
  //     calling.
  // - `prefill_preface_on_init`: Whether to prefill the preface on init. If
  //     true, the preface will be prefilled on init, which will make the first
  //     response faster, but take longer to initialize.
  static absl::StatusOr<ConversationConfig> CreateDefault(
      const Engine& engine, std::optional<Preface> preface = std::nullopt,
      std::optional<PromptTemplate> overwrite_prompt_template = std::nullopt,
      std::optional<DataProcessorConfig> overwrite_processor_config =
          std::nullopt,
      bool enable_constrained_decoding = false,
      bool prefill_preface_on_init = false);

  // Creates a ConversationConfig from the given SessionConfig.
  // Args:
  // - `engine`: The Engine instance to be used to validate the SessionConfig.
  // - `session_config`: The SessionConfig to be used for creating the
  //     ConversationConfig.
  // - `preface`: Optional Preface for the conversation. The Preface provides
  //     the initial background for the conversation, tool uses and extra
  //     context for the conversation. If not provided, the conversation will
  //     start with an empty Preface.
  // - `overwrite_processor_config`: Optional configuration for the model data
  //     processor, if not provided, the default config for the model type's
  //     data processor will be used. Most of the time, the users don't need to
  //     provide the data processor config.
  // - `enable_constrained_decoding`: Whether to enable constrained decoding. If
  //     true, constrained decoding will be used, primarily for function
  //     calling.
  // - `prefill_preface_on_init`: Whether to prefill the preface on init. If
  //     true, the preface will be prefilled on init, which will make the first
  //     response faster, but take longer to initialize.
  static absl::StatusOr<ConversationConfig> CreateFromSessionConfig(
      const Engine& engine, const SessionConfig& session_config,
      std::optional<Preface> preface = std::nullopt,
      std::optional<DataProcessorConfig> overwrite_processor_config =
          std::nullopt,
      bool enable_constrained_decoding = false,
      bool prefill_preface_on_init = false);

  // Returns the SessionConfig used for creating the ConversationConfig.
  const SessionConfig& GetSessionConfig() const { return session_config_; }

  // Returns the Preface used for creating the ConversationConfig.
  const Preface& GetPreface() const { return preface_; }

  // Returns the PromptTemplate used for creating the ConversationConfig.
  const PromptTemplate& GetPromptTemplate() const { return prompt_template_; }

  // Returns the DataProcessorConfig used for creating the ConversationConfig.
  const DataProcessorConfig& GetProcessorConfig() const {
    return processor_config_;
  }

  // Returns whether constrained decoding is enabled.
  bool constrained_decoding_enabled() const {
    return constrained_decoding_enabled_;
  }

  // Returns whether the preface should be prefilled when the Conversation is
  // created. This will make the first response faster, but take longer to
  // initialize.
  bool prefill_preface_on_init() const { return prefill_preface_on_init_; }

 private:
  explicit ConversationConfig(SessionConfig session_config, Preface preface,
                              PromptTemplate prompt_template,
                              DataProcessorConfig processor_config,
                              bool constrained_decoding_enabled = false,
                              bool prefill_preface_on_init = false)
      : session_config_(std::move(session_config)),
        preface_(std::move(preface)),
        prompt_template_(std::move(prompt_template)),
        processor_config_(std::move(processor_config)),
        constrained_decoding_enabled_(constrained_decoding_enabled),
        prefill_preface_on_init_(prefill_preface_on_init) {}

  SessionConfig session_config_;
  Preface preface_;
  PromptTemplate prompt_template_;
  DataProcessorConfig processor_config_;
  bool constrained_decoding_enabled_;
  bool prefill_preface_on_init_;
};

// A multi-turn centric stateful Conversation API for high-level user
// interaction. Conversation maintains the history for users, so the users'
// messages will be used as the LLM context through the conversation.
//
// Conversation handles the complex data processing logic for Session usage,
// including:
// - Prompt template rendering.
// - Role-based messages handling.
// - Multimodal input processing.
// - History management.
// - Model-specific data processing.
//
// Example usage:
//
//   // Create an Engine instance.
//   ASSIGN_OR_RETURN(auto engine, Engine::Create(model_assets));
//
//   // Create a ConversationConfig instance from the Engine.
//   ASSIGN_OR_RETURN(auto conversation_config,
//                    ConversationConfig::CreateDefault(*engine));
//
//   // Create a Conversation instance.
//   ASSIGN_OR_RETURN(auto conversation,
//       Conversation::Create(*engine, conversation_config));
//
//   // Send a message to the LLM and returns the complete message.
//   ASSIGN_OR_RETURN(const Message message,
//                    conversation->SendMessage(JsonMessage{
//                        {"role", "user"}, {"content", "Hello world!"}}));
//
//   // Send a message to the LLM and process the asynchronous message results
//   // via the user_callback. The user_callback is a user-defined callback
//   // function that handles the message results.
//   EXPECT_OK(conversation->SendMessageAsync(
//       JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
//       [](absl::StatusOr<Message> message) {
//         // Handle the message results.
//         if (message.ok()) {
//           std::cout << "Message: " << std::endl;
//         }
//       });
//
class Conversation {
 public:
  // Creates a Conversation instance from the the Engine and ConversationConfig.
  // Args:
  // - `engine`: The Engine instance to be used for creating the Conversation.
  // - `config`: The ConversationConfig instance to be used for creating the
  // Conversation.
  static absl::StatusOr<std::unique_ptr<Conversation>> Create(
      const Engine& engine, const ConversationConfig& config);

  // Sends a message to the LLM and returns the complete message.
  // Args:
  // - `message`: The message to be sent to the LLM. If `message` is an array,
  //    each element will be treated as a separate message and be prefilled
  //    before generating the response.
  // - `args`: The optional arguments for the corresponding model data
  //    processor. Most of the time, the users don't need to provide this
  //    argument.
  // Returns :
  // - The complete message from the LLM.
  absl::StatusOr<Message> SendMessage(
      const Message& message,
      std::optional<DataProcessorArguments> args = std::nullopt);

  // Sends a message to the LLM and process the asynchronous message results via
  // the user_callback.
  // Args:
  // - `message`: The message to be sent to the LLM. If `message` is an array,
  //    each element will be treated as a separate message and be prefilled
  //    before generating the response.
  // - `user_callback`: The callback to receive the message events. The
  //    user_callback will be invoked in the following conditions:
  //    - On every new message chunk.
  //    - When the generation is complete, the user_callback will be invoked
  //      with an empty message.
  //    - When the generation is cancelled, the user_callback will be invoked
  //      with absl::CancelledError.
  //    - When an error occurs, the user_callback will be invoked with the error
  //      status.
  // - `args`: The optional arguments for the corresponding model data
  //    processor. Most of the time, the users don't need to provide this
  //    argument.
  // Returns :
  // - absl::OkStatus if the message is sent and processing successfully,
  //   otherwise the error status.
  absl::Status SendMessageAsync(
      const Message& message,
      absl::AnyInvocable<void(absl::StatusOr<Message>)> user_callback,
      std::optional<DataProcessorArguments> args = std::nullopt);

  // Returns the history of the conversation.
  // Note: the return value is a copy of the history, which may be expensive
  // for large history.
  std::vector<Message> GetHistory() const {
    absl::MutexLock lock(&history_mutex_);  // NOLINT
    return history_;
  }

  // Provides safe access to the conversation history without copying.
  // The provided visitor function is executed while the history mutex is held.
  // Args:
  // - visitor: The visitor function takes a const reference to the history
  //  vector.
  //
  // Example usage:
  //
  //   Message assistant_message;
  //   conversation->AccessHistory(
  //       [&assistant_message](const std::vector<Message>& history) {
  //         // Copy the last message to assistant_message. So we don't need to
  //         // copy the whole history, if we only need the last message.
  //         assistant_message = history.back();
  //       });
  void AccessHistory(absl::AnyInvocable<void(const std::vector<Message>&) const>
                         visitor) const {
    absl::MutexLock lock(&history_mutex_);  // NOLINT
    visitor(history_);
  }

  // Returns the configuration used for creating the Conversation.
  const ConversationConfig& GetConfig() const { return config_; }

  // Returns the benchmark info for the conversation. Underlying this method
  // triggers the benchmark info collection from the Session.
  // Returns:
  // - The benchmark info for the conversation.
  absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo();

  // Cancels the ongoing inference process, for asynchronous inference.
  // Note: the underlying Session is not rollbacked, so the message
  // from the user is actually sent to the LLM and processed for prefill.
  void CancelProcess();

 private:
  explicit Conversation(
      std::unique_ptr<Engine::Session> session,
      std::unique_ptr<ModelDataProcessor> model_data_processor, Preface preface,
      PromptTemplate prompt_template, ConversationConfig config)
      : session_(std::move(session)),
        model_data_processor_(std::move(model_data_processor)),
        preface_(preface),
        prompt_template_(std::move(prompt_template)),
        config_(config) {}

  absl::StatusOr<std::string> GetSingleTurnText(const Message& message) const;

  absl::StatusOr<DecodeConfig> CreateDecodeConfig();

  std::unique_ptr<Engine::Session> session_;
  std::unique_ptr<ModelDataProcessor> model_data_processor_;
  Preface preface_;
  PromptTemplate prompt_template_;
  // The constraint is currently created from the tools defined in the preface,
  // if any.
  std::unique_ptr<Constraint> constraint_;
  const ConversationConfig config_;
  mutable absl::Mutex history_mutex_;
  std::vector<Message> history_ ABSL_GUARDED_BY(history_mutex_);
};
}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CONVERSATION_H_
