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

#include "runtime/framework/resource_management/resource_manager.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_google.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/context_handler/context_handler.h"
#include "runtime/framework/resource_management/utils/movable_mutex_lock.h"
#include "runtime/framework/resource_management/utils/resource_manager_utils.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

// Saves the current processed context within the llm_executor to the previous
// handler's shared_processed_context, and link the current handler to an
// empty processed_context, which represents the current handler's processed
// context is already loaded in the llm_executor. Notice that the
// context_handler is assumed to be loaded in the llm_executor via the
// AcquireWithContext method, meaning the context_handler should not own any
// RuntimeState, RuntimeConfig and actual ProcessedContext within it, as they
// are all owned by the llm_executor at this point.
absl::Status SaveProcessedContextAndSeparateLoadedHandler(
    std::shared_ptr<ContextHandler> context_handler,
    std::shared_ptr<LlmExecutor> llm_executor) {
  RET_CHECK_EQ(
      context_handler->HasRuntimeConfig() ||
          context_handler->HasRuntimeState() ||
          context_handler->shared_processed_context()->HasProcessedContext(),
      false)
      << "The context_handler "
         "should not own any RuntimeState, RuntimeConfig and actual "
         "ProcessedContext within it, as they should all be owned by the "
         "llm_executor when calling "
         "SaveProcessedContextAndSeparateLoadedHandler.";
  ASSIGN_OR_RETURN(auto llm_context, llm_executor->CloneContext());
  ASSIGN_OR_RETURN(auto current_processed_context,
                   llm_context->RetrieveProcessedContext());
  RETURN_IF_ERROR(
      context_handler->shared_processed_context()->SetProcessedContext(
          std::move(current_processed_context)));

  auto new_shared_processed_context =
      std::make_shared<ContextHandler::SharedProcessedContext>(nullptr);
  RETURN_IF_ERROR(context_handler->UpdateSharedProcessedContext(
      new_shared_processed_context));
  return absl::OkStatus();
}

}  // namespace

class LockedVisionExecutor : public VisionExecutor {
 public:
  LockedVisionExecutor(std::shared_ptr<VisionExecutor> vision_executor,
                       MovableMutexLock lock)
      : vision_executor_(std::move(vision_executor)), lock_(std::move(lock)) {}

  absl::StatusOr<ExecutorVisionData> Encode(
      const TensorBuffer& input_image_tensor) override {
    return vision_executor_->Encode(input_image_tensor);
  }

  absl::StatusOr<std::vector<int>> GetExpectedInputDimension() const override {
    return vision_executor_->GetExpectedInputDimension();
  }

 private:
  std::shared_ptr<VisionExecutor> vision_executor_;
  // The mutex lock.
  MovableMutexLock lock_;
};

class LockedAudioExecutor : public AudioExecutor {
 public:
  LockedAudioExecutor(std::shared_ptr<AudioExecutor> audio_executor,
                      MovableMutexLock lock)
      : audio_executor_(std::move(audio_executor)), lock_(std::move(lock)) {}

  absl::StatusOr<ExecutorAudioData> Encode(
      const TensorBuffer& input_image_tensor) override {
    return audio_executor_->Encode(input_image_tensor);
  }

 private:
  std::shared_ptr<AudioExecutor> audio_executor_;
  // The mutex lock.
  MovableMutexLock lock_;
};

// LockedLlmExecutor's behavior should be the same as LlmExecutor, but instead
// wraps the llm executor, the corresponding mutex lock, and some additional
// optimization logic before forwarding the request to the llm executor.
// The Optimization includes:
// 1(remove matching tokens): Update the input_ids and current_step by
// removing
//    the matching tokens from the processed tokens.
// 2(Copy on write): If the current handler is not the longest handler,
// retrieve
//    the processed_context for the previous handler, and update the current
//    handler's shared_processed_context.
// For more details, please refer to go/llm_resource_manager.
class LockedLlmExecutor : public LlmExecutor {
 public:
  // LockedLlmExecutor takes ownership of the mutex lock and holds the
  // shared_ptr to the executor.
  LockedLlmExecutor(std::shared_ptr<LlmExecutor> executor,
                    MovableMutexLock lock,
                    std::shared_ptr<ContextHandler> current_handler = nullptr)
      : current_handler_(current_handler),
        llm_executor_(std::move(executor)),
        lock_(std::move(lock)) {}

  absl::string_view ExecutorBackendName() const override {
    return llm_executor_->ExecutorBackendName();
  }

  absl::Status Prefill(const ExecutorInputs& inputs) override {
    return Prefill(inputs, ExecutorPrefillParams());
  }

  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& prefill_params) override {
    // If the executor is not acquired by any handler, forward the prefill
    // request to the executor directly.
    if (current_handler_ == nullptr) {
      return llm_executor_->Prefill(inputs, prefill_params);
    }
    // Check if the input token is 1 batch. Currently only support 1 batch per
    // prefill.
    ASSIGN_OR_RETURN(auto token_ids, inputs.GetTextTokenIdsPtr());
    LITERT_ASSIGN_OR_RETURN(auto token_ids_tensor_type,
                                 token_ids->TensorType());
    RET_CHECK_EQ(token_ids_tensor_type.Layout().Dimensions()[0], 1);
    if (token_ids_tensor_type.Layout().Dimensions()[1] == 0) {
      return absl::OkStatus();
    }
    ASSIGN_OR_RETURN(int current_step, llm_executor_->GetCurrentStep());
    if (prefill_params.GetCurrentStep() != -1) {
      current_step = prefill_params.GetCurrentStep();
    }
    ASSIGN_OR_RETURN(const ProcessedTokens* processed_tokens,
                     llm_executor_->GetProcessedTokens());
    // If the current_step is pointing at the step right after the last
    // processed token, call executor directly, no optimization for the
    // input can be done.
    if (processed_tokens->TokenCount() == current_step) {
      return llm_executor_->Prefill(inputs, prefill_params);
    }

    LITERT_ASSIGN_OR_RETURN(
        auto input_ids_vec,
        CopyFromTensorBuffer<int32_t>(*(*inputs.GetTextTokenIdsPtr())));

    // If the current_step is not pointing at the step right after the last
    // processed token, update the input_ids and current_step by removing
    // the matching tokens, and then call llm_executor_->Prefill with the
    // optimized inputs and time step.

    // If the processed tokens size is larger than the current step, update
    // the input_ids and current_step by removing the matching tokens.
    RETURN_IF_ERROR(RemoveMatchingTokens(processed_tokens->GetCopyOfTokens()[0],
                                         &input_ids_vec, &current_step));
    // If the updated input_ids is empty, meaning all required prefill
    // tokens have been processed previously, just set the current step and
    // return.
    if (input_ids_vec.empty()) {
      RETURN_IF_ERROR(llm_executor_->SetCurrentStep(current_step));
      return absl::OkStatus();
    }

    LITERT_ASSIGN_OR_RETURN(
        auto new_inputs_token_ids,
        CopyToTensorBuffer(
            absl::MakeConstSpan(input_ids_vec.data(), input_ids_vec.size()),
            {1, static_cast<int>(input_ids_vec.size())}));
    auto new_inputs =
        ExecutorInputs(ExecutorTextData(std::move(new_inputs_token_ids)),
                       std::nullopt, std::nullopt);

    auto new_prefill_query_params = prefill_params;
    new_prefill_query_params.SetCurrentStep(current_step);
    // If the current_step is pointing at the step following the last
    // processed token after removing the matching tokens, call executor
    // directly.
    if (processed_tokens->TokenCount() == current_step) {
      return llm_executor_->Prefill(new_inputs, new_prefill_query_params);
    }
    // If the updated current_steps_ is still less than the processed tokens
    // size, meaning part of the processed tokens does not match the
    // input_ids.
    // Confirm if the current handler is the longest handler. If not,
    // cloning processed context is required to avoid modifying the
    // processed context of other handlers.
    ASSIGN_OR_RETURN(
        int largest_time_step,
        current_handler_->shared_processed_context()->LongestHandlerTimeStep(
            *llm_executor_));
    if (largest_time_step != current_step) {
      // If the current handler is not the longest handler, retrieve the
      // processed_context for the previous handler, and update the current
      // handler's shared_processed_context.
      RETURN_IF_ERROR(SaveProcessedContextAndSeparateLoadedHandler(
          current_handler_, llm_executor_));
    }
    // Update the current step since the new processed context (set above)
    // might not match the executor's current step, and the processed
    // context may need to be truncated.
    // TODO: b/418002952 - Consider setting the current step within Prefill
    // rather than relying on the caller.
    RETURN_IF_ERROR(llm_executor_->SetCurrentStep(current_step));
    return llm_executor_->Prefill(new_inputs, new_prefill_query_params);
  }

  absl::Status Decode(TensorBuffer& output_tokens) override {
    return Decode(output_tokens, ExecutorDecodeParams());
  }

  absl::Status Decode(TensorBuffer& output_tokens,
                      const ExecutorDecodeParams& decode_params) override {
    RETURN_IF_ERROR(MaybeTruncateProcessedTokens());
    return llm_executor_->Decode(output_tokens, decode_params);
  }

  absl::Status Decode(const ExecutorInputs& inputs,
                      TensorBuffer& output_logits) override {
    RETURN_IF_ERROR(MaybeTruncateProcessedTokens());
    return llm_executor_->Decode(output_logits);
  }

  absl::StatusOr<TensorBuffer> DecodeLogits(
      const ExecutorInputs& inputs) override {
    ASSIGN_OR_RETURN(int current_step, llm_executor_->GetCurrentStep());
    ASSIGN_OR_RETURN(const ProcessedTokens* processed_tokens,
                     llm_executor_->GetProcessedTokens());
    // If the current step is pointing at right after the pending token, set the
    // current step to the previous step, so that the current step is pointing
    // at the to be processed token.(expected by llm_executor_->DecodeLogits())
    if (current_step == processed_tokens->TokenCount() &&
        !processed_tokens->GetNextUnprocessedToken().token.empty()) {
      RETURN_IF_ERROR(llm_executor_->SetCurrentStep(current_step - 1));
    }
    RETURN_IF_ERROR(MaybeTruncateProcessedTokens());
    return llm_executor_->DecodeLogits(inputs);
  }

  absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const override {
    return llm_executor_->CloneContext();
  }

  absl::Status RestoreContext(
      std::unique_ptr<LlmContext> llm_context) override {
    return llm_executor_->RestoreContext(std::move(llm_context));
  }

  absl::Status UpdateRuntimeConfig(
      const RuntimeConfig& runtime_config) override {
    return llm_executor_->UpdateRuntimeConfig(runtime_config);
  }

  absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const override {
    return llm_executor_->GetRuntimeConfig();
  }

  absl::Status UpdateRuntimeState(const RuntimeState& runtime_state) override {
    return llm_executor_->UpdateRuntimeState(runtime_state);
  }

  absl::StatusOr<RuntimeState> GetRuntimeState() const override {
    return llm_executor_->GetRuntimeState();
  }

  absl::StatusOr<LlmExecutorSettings> GetExecutorSettings() const override {
    return llm_executor_->GetExecutorSettings();
  }

  absl::StatusOr<int> GetCurrentStep() const override {
    return llm_executor_->GetCurrentStep();
  }

  absl::Status SetCurrentStep(int new_step) override {
    return llm_executor_->SetCurrentStep(new_step);
  }

  absl::StatusOr<const ProcessedTokens*> GetProcessedTokens() const override {
    return llm_executor_->GetProcessedTokens();
  }

  absl::Status LoadLoRA(uint32_t lora_id,
                        const ModelAssets& model_assets) override {
    return llm_executor_->LoadLoRA(lora_id, model_assets);
  }

  absl::Status Reset() override { return llm_executor_->Reset(); }

  absl::StatusOr<int> GetVocabSize() override {
    return llm_executor_->GetVocabSize();
  }

 private:
  absl::Status MaybeTruncateProcessedTokens() {
    if (current_handler_ == nullptr) {
      return absl::OkStatus();
    }
    ASSIGN_OR_RETURN(int current_step, llm_executor_->GetCurrentStep());
    ASSIGN_OR_RETURN(const ProcessedTokens* processed_tokens,
                     llm_executor_->GetProcessedTokens());
    if (processed_tokens->TokenCount() == current_step) {
      return absl::OkStatus();
    }

    // Confirm if the current handler is the longest handler. If not,
    // cloning processed context is required to avoid modifying the
    // processed context of other handlers.
    ASSIGN_OR_RETURN(
        int largest_time_step,
        current_handler_->shared_processed_context()->LongestHandlerTimeStep(
            *llm_executor_));
    if (largest_time_step != current_step) {
      // If the current handler is not the longest handler, retrieve the
      // processed_context for the previous handler, and update the current
      // handler's shared_processed_context.
      RETURN_IF_ERROR(SaveProcessedContextAndSeparateLoadedHandler(
          current_handler_, llm_executor_));
    }
    // Update the current step since the new processed context (set above)
    // might not match the executor's current step, and the processed
    // context may need to be truncated.
    // TODO: b/418002952 - Consider setting the current step within Decode
    // rather than relying on the caller.
    return llm_executor_->SetCurrentStep(current_step);
  }

  // The current context handler;
  std::shared_ptr<ContextHandler> current_handler_;

  // The executor.
  std::shared_ptr<LlmExecutor> llm_executor_;

  // The mutex lock.
  MovableMutexLock lock_;
};

std::optional<uint32_t> ResourceManager::AssignLoraId(
    std::string lora_path, bool has_scoped_lora_file) {
  if (lora_path.empty() && !has_scoped_lora_file) {
    return std::nullopt;
  }
  std::optional<uint32_t> lora_id;
  // If this session is using a new lora, assign a new unique lora id. Else,
  // assign the corresponding lora id according to the provided lora path in
  // the session config.
  if (!lora_path.empty()) {
    // Lora provided by both path and scoped file will use lora path as the
    // reference key if provided.
    if (lora_hash_to_id_.find(lora_path) == lora_hash_to_id_.end()) {
      // If the lora is new, assign the id.
      lora_hash_to_id_[lora_path] = lora_hash_to_id_.size();
    }
    lora_id = lora_hash_to_id_[lora_path];
  } else if (has_scoped_lora_file) {
    // Lora provided by scoped file but without lora path will be assumed to
    // be used only once. Assign a unique id for this session only.
    // TODO: b/346421150 - Extend support to map from scoped file to hash
    // key, for multiple same scoped file use case.
    lora_id = lora_hash_to_id_.size();
    lora_hash_to_id_["scoped_lora:" + absl::StrCat(lora_hash_to_id_.size())] =
        lora_id.value();
  }
  return lora_id;
}

absl::Status ResourceManager::MaybeCreateLitertEnv() {
  if (litert_env_ != nullptr) {
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(
      auto env,
      litert::Environment::Create(std::vector<litert::Environment::Option>()));
  litert_env_ = std::make_unique<litert::Environment>(std::move(env));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ContextHandler>>
ResourceManager::CreateContextHandler(const SessionConfig& session_config) {
  // TODO: b/462499294 -
  //   1. Check if lora is loaded or not.
  //   2. Get the lora id.
  //   3. If lora is not loaded, load the lora.

  // Check if the lora is already loaded.
  // TODO: b/462499294 - Use the real lora path.
  bool lora_is_loaded =
      lora_hash_to_id_.find("fake_lora_path") != lora_hash_to_id_.end();

  // Find the lora id. If lora_id is not nullopt, it means the lora is used.
  // TODO: b/462499294 - Use the real lora path.
  std::optional<uint32_t> lora_id =
      AssignLoraId(/*lora_path=*/"", /*has_scoped_lora_file=*/false);

  // If lora is used and not loaded, load the lora.
  if (lora_id.has_value() && !lora_is_loaded) {
    // TODO: b/462499294 - Use the real lora path and scoped lora file.
    ASSIGN_OR_RETURN(ModelAssets model_assets,
                     ModelAssets::Create(/*scoped_lora_file=*/nullptr,
                                         /*lora_path=*/""));
    MovableMutexLock lock(&executor_mutex_);
    RETURN_IF_ERROR(llm_executor_->LoadLoRA(lora_id.value(), model_assets));
  }
  // TODO: b/462517405 - Remove this conversion from SamplerParams to
  // SamplerParams once the SamplerParams is cleaned up.
  odml::infra::proto::SamplerParameters sampler_params;
  switch (session_config.GetSamplerParams().type()) {
    case proto::SamplerParameters::TYPE_UNSPECIFIED: {
      sampler_params.set_type(
          odml::infra::proto::SamplerParameters::TYPE_UNSPECIFIED);
      break;
    }
    case proto::SamplerParameters::TOP_K: {
      sampler_params.set_type(odml::infra::proto::SamplerParameters::TOP_K);
      break;
    }
    case proto::SamplerParameters::TOP_P: {
      sampler_params.set_type(odml::infra::proto::SamplerParameters::TOP_P);
      break;
    }
    case proto::SamplerParameters::GREEDY: {
      sampler_params.set_type(odml::infra::proto::SamplerParameters::GREEDY);
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported sampler type: ",
                       session_config.GetSamplerParams().type()));
  }
  sampler_params.set_k(session_config.GetSamplerParams().k());
  sampler_params.set_p(session_config.GetSamplerParams().p());
  sampler_params.set_temperature(
      session_config.GetSamplerParams().temperature());

  auto runtime_config = RuntimeConfig{
      .sampler_params = sampler_params,
      .output_heads = session_config.GetNumOutputCandidates(),
      // b/368348506 - Make tokens_per_decode configurable.
      .tokens_per_decode = 1,
  };

  std::unique_ptr<litert::lm::LlmContext> llm_context;
  {
    MovableMutexLock lock(&executor_mutex_);
    ASSIGN_OR_RETURN(llm_context,
                     llm_executor_->CreateNewContext(
                         std::move(lora_id), std::move(runtime_config)));
  }
  return ContextHandler::Create(std::move(llm_context));
}

absl::StatusOr<std::unique_ptr<ContextHandler>>
ResourceManager::CloneContextHandler(
    std::shared_ptr<const ContextHandler> llm_context_handler) {
  RET_CHECK_NE(llm_context_handler, nullptr)
      << "The provided context handler should not be null.";

  RuntimeConfig runtime_config;
  RuntimeState runtime_state;

  // If the context handler has the runtime config and runtime state, use
  // them directly.
  if (llm_context_handler->HasRuntimeConfig() &&
      llm_context_handler->HasRuntimeState()) {
    ASSIGN_OR_RETURN(runtime_config, llm_context_handler->GetRuntimeConfig());
    ASSIGN_OR_RETURN(runtime_state, llm_context_handler->GetRuntimeState());
  } else {
    // Otherwise, assume the context handler is loaded by the manager to the
    // executor, and get the runtime config and runtime state from the
    // executor.
    MovableMutexLock lock(&executor_mutex_);
    RET_CHECK_EQ(current_handler_, llm_context_handler)
        << "The provided context handler does not have the runtime config "
           "and "
           "runtime state, assuming it is loaded by the manager, but the "
           "manager does not have the same handler.";
    ASSIGN_OR_RETURN(runtime_config, llm_executor_->GetRuntimeConfig());
    ASSIGN_OR_RETURN(runtime_state, llm_executor_->GetRuntimeState());
  }
  auto processed_context = llm_context_handler->shared_processed_context();
  return ContextHandler::Bundle(processed_context,
                                std::make_unique<RuntimeConfig>(runtime_config),
                                std::make_unique<RuntimeState>(runtime_state));
}

absl::StatusOr<std::unique_ptr<LlmExecutor>>
ResourceManager::AcquireExecutor() {
  MovableMutexLock lock(&executor_mutex_);

  if (llm_executor_ == nullptr) {
    return absl::InvalidArgumentError(
        "Llm executor should not be null, please do not delete the shared "
        "executor "
        "in ResourceManager at any time.");
  }

  return std::make_unique<LockedLlmExecutor>(llm_executor_, std::move(lock));
}

absl::StatusOr<std::unique_ptr<LlmExecutor>>
ResourceManager::AcquireExecutorWithContextHandler(
    std::shared_ptr<ContextHandler> new_context_handler) {
  RET_CHECK_NE(new_context_handler, nullptr)
      << "The provided context handler should not be null.";

  MovableMutexLock lock(&executor_mutex_);
  RET_CHECK_NE(llm_executor_, nullptr) << "Llm executor should not be null, "
                                          "please do not delete the shared "
                                          "executor in ResourceManager at "
                                          "any time.";

  // If the new handler is the same as the current handler, return the
  // executor directly.
  if (new_context_handler == current_handler_) {
    return std::make_unique<LockedLlmExecutor>(llm_executor_, std::move(lock),
                                               current_handler_);
  }

  // If both handler are sharing the same processed context, save the
  // runtime config and runtime state back to the current handler. Then
  // update the executor with the new handler.
  if (current_handler_ != nullptr &&
      new_context_handler->shared_processed_context() ==
          current_handler_->shared_processed_context()) {
    ASSIGN_OR_RETURN(auto current_runtime_config,
                     llm_executor_->GetRuntimeConfig());
    ASSIGN_OR_RETURN(auto current_runtime_state,
                     llm_executor_->GetRuntimeState());
    RETURN_IF_ERROR(current_handler_->SetRuntimeConfig(
        std::make_unique<RuntimeConfig>(current_runtime_config)));
    RETURN_IF_ERROR(current_handler_->SetRuntimeState(
        std::make_unique<RuntimeState>(current_runtime_state)));

    ASSIGN_OR_RETURN(auto new_runtime_config,
                     new_context_handler->RetrieveRuntimeConfig());
    ASSIGN_OR_RETURN(auto new_runtime_state,
                     new_context_handler->RetrieveRuntimeState());
    RETURN_IF_ERROR(llm_executor_->UpdateRuntimeConfig(*new_runtime_config));
    RETURN_IF_ERROR(llm_executor_->UpdateRuntimeState(*new_runtime_state));
  } else {
    // If the new handler is not sharing the same processed context with the
    // current handler, clone the processed context to the new handler. Then
    // restore the executor with the new LlmContext.
    if (current_handler_ != nullptr) {
      ASSIGN_OR_RETURN(auto current_llm_context, llm_executor_->CloneContext());
      ASSIGN_OR_RETURN(auto current_runtime_config,
                       current_llm_context->RetrieveRuntimeConfig());
      ASSIGN_OR_RETURN(auto current_runtime_state,
                       current_llm_context->RetrieveRuntimeState());
      ASSIGN_OR_RETURN(auto current_processed_context,
                       current_llm_context->RetrieveProcessedContext());

      RETURN_IF_ERROR(current_handler_->SetRuntimeConfig(
          std::move(current_runtime_config)));
      RETURN_IF_ERROR(
          current_handler_->SetRuntimeState(std::move(current_runtime_state)));
      RETURN_IF_ERROR(
          current_handler_->shared_processed_context()->SetProcessedContext(
              std::move(current_processed_context)));
    }

    ASSIGN_OR_RETURN(auto new_runtime_config,
                     new_context_handler->RetrieveRuntimeConfig());
    ASSIGN_OR_RETURN(auto new_runtime_state,
                     new_context_handler->RetrieveRuntimeState());
    ASSIGN_OR_RETURN(auto new_processed_context,
                     new_context_handler->shared_processed_context()
                         ->RetrieveProcessedContext());
    auto llm_context = std::make_unique<LlmContext>(
        std::move(new_processed_context), std::move(new_runtime_config),
        std::move(new_runtime_state));
    RETURN_IF_ERROR(llm_executor_->RestoreContext(std::move(llm_context)));
  }

  current_handler_ = new_context_handler;

  return std::make_unique<LockedLlmExecutor>(llm_executor_, std::move(lock),
                                             current_handler_);
}

absl::Status ResourceManager::TryLoadingVisionExecutor() {
  absl::MutexLock lock(vision_executor_mutex_);
  if (!vision_executor_settings_) {
    return absl::InvalidArgumentError("Vision options should not be null.");
  }
  RETURN_IF_ERROR(MaybeCreateLitertEnv());
  ASSIGN_OR_RETURN(vision_executor_,
                   VisionLiteRtCompiledModelExecutor::Create(
                       *vision_executor_settings_, *litert_env_));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<VisionExecutor>>
ResourceManager::AcquireVisionExecutor() {
  MovableMutexLock lock(&vision_executor_mutex_);
  if (vision_executor_ == nullptr) {
    return absl::InvalidArgumentError(
        "Vision executor should not be null, please TryLoadingVisionExecutor() "
        "first.");
  }
  return std::make_unique<LockedVisionExecutor>(vision_executor_,
                                                std::move(lock));
}

absl::Status ResourceManager::TryLoadingAudioExecutor() {
  absl::MutexLock lock(audio_executor_mutex_);
  if (audio_executor_ != nullptr) {
    return absl::OkStatus();
  }
  if (!audio_executor_settings_) {
    return absl::InvalidArgumentError("Audio options should not be null.");
  }
  if (audio_executor_settings_->GetBackend() ==
             litert::lm::Backend::CPU) {
    return absl::InvalidArgumentError(
        "Audio executor backend is not supported.");
  } else {
    return absl::InvalidArgumentError(
        "Audio executor backend is not supported.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<AudioExecutor>>
ResourceManager::AcquireAudioExecutor() {
  MovableMutexLock lock(&audio_executor_mutex_);
  if (audio_executor_ == nullptr) {
    return absl::InvalidArgumentError(
        "Audio executor should not be null, please TryLoadingAudioExecutor() "
        "first.");
  }
  return std::make_unique<LockedAudioExecutor>(audio_executor_,
                                               std::move(lock));
}

absl::StatusOr<std::unique_ptr<ResourceManager>> ResourceManager::Create(
    std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
    std::unique_ptr<VisionExecutorSettings> absl_nullable
    vision_executor_settings,
    std::unique_ptr<litert::lm::AudioExecutorSettings> absl_nullable
    audio_executor_settings,
    std::unique_ptr<::litert::Environment> absl_nullable litert_env) {
  if (llm_executor == nullptr) {
    return absl::InvalidArgumentError("Llm executor is null.");
  }
  auto llm_resource_manager = std::make_unique<ResourceManager>(
      std::move(llm_executor), std::move(vision_executor_settings),
      std::move(audio_executor_settings), std::move(litert_env));
  return llm_resource_manager;
}

}  // namespace litert::lm
