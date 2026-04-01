// Copyright 2024 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// GPU executor that implements the shared functionalities for all GPU backends
// (OpenCl/WebGpu/Metal/etc.). Note that this class itself is not instantiable,
// since the Create() function is not implemented.
// TODO: b/361667248 - Add test for LlmTfLiteGpuExecutor.
class LlmLiteRtCompiledModelExecutorBase : public LlmExecutor {
 public:
  using LlmExecutor::Prefill;

  // Input APIs:
  // Basic API to trigger the "prefill" or "prefix" process.
  // Input is token ids with shape `[batch, sequence_length]`
  absl::Status Prefill(const ExecutorInputs& inputs) override {
    ExecutorPrefillParams params;
    return Prefill(inputs, params);
  };

  // Output APIs:
  // Basic API to trigger the "decode" process.
  absl::Status Decode(::litert::TensorBuffer& output_tokens) override;

  // Advanced API to allow customized query parameters.
  absl::Status Decode(::litert::TensorBuffer& output_tokens,
                      const ExecutorDecodeParams& decode_params) override;

  // Basic API to trigger the "decode" process but without sampling.
  // Input is token ids with shape `[batch, sequence_length]`
  // Output is logits with shape `[batch, sequence_length, vocab_size]`
  // TODO: b/355310550 - Shall we change the function naming here to not
  // overload Decode?
  absl::Status Decode(const ExecutorInputs& inputs,
                      ::litert::TensorBuffer& output_logits) override;

  absl::StatusOr<::litert::TensorBuffer> DecodeLogits(
      const ExecutorInputs& inputs) override;

  absl::StatusOr<::litert::TensorBuffer> DecodeLogits(
      const ExecutorInputs& inputs, const ExecutorDecodeParams& decode_params);

  absl::string_view ExecutorBackendName() const override {
    return "LiteRT Compiled Model";
  }

  // Gets the executor settings.
  absl::StatusOr<LlmExecutorSettings> GetExecutorSettings() const override {
    return executor_settings_;
  }

  // Gets the current step of the executor.
  // Public API, the return value is the current step that user expects (e.g.
  // users prefill 100 tokens, then they expect the current step to be 100). It
  // is different from the internal current step.
  absl::StatusOr<int> GetCurrentStep() const override {
    return processed_tokens_.TokenCount();
  }

  // Resets all of the internal states.
  absl::Status Reset() override;

  absl::StatusOr<int> GetVocabSize() override;

  // Initializes the sampler.
  absl::Status InitializeSampler();

  using LogitsDataType = ActivationDataType;

  const ProcessedTokens& processed_tokens_for_testing() const {
    return processed_tokens_;
  }

 protected:
  LlmLiteRtCompiledModelExecutorBase(
      LlmExecutorSettings executor_settings, ::litert::Environment& env,
      const ::litert::Model* absl_nonnull model,
      ::litert::CompiledModel compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_output_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          output_kv_cache_buffers,
      std::optional<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
          decode_input_kv_cache_buffers,
      std::optional<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
          decode_output_kv_cache_buffers,
      ModelSignatures signatures, int output_batch_size,
      std::string weight_cache_path,
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup,
      std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup,
      bool use_fp16_precision, LogitsDataType logits_data_type)
      : executor_settings_(std::move(executor_settings)),
        env_(env),
        model_(*model),
        compiled_model_(std::move(compiled_model)),
        decode_input_buffers_(std::move(decode_input_buffers)),
        decode_output_buffers_(std::move(decode_output_buffers)),
        kv_cache_buffers_1_(std::move(input_kv_cache_buffers)),
        kv_cache_buffers_2_(std::move(output_kv_cache_buffers)),
        input_kv_cache_buffers_(&kv_cache_buffers_1_),
        output_kv_cache_buffers_(&kv_cache_buffers_2_),
        decode_kv_cache_buffers_1_(std::move(decode_input_kv_cache_buffers)),
        decode_kv_cache_buffers_2_(std::move(decode_output_kv_cache_buffers)),
        signatures_(signatures),
        output_batch_size_(output_batch_size),
        weight_cache_path_(std::move(weight_cache_path)),
        embedding_lookup_(std::move(embedding_lookup)),
        per_layer_embedding_lookup_(std::move(per_layer_embedding_lookup)),
        use_fp16_precision_(use_fp16_precision),
        logits_data_type_(logits_data_type) {}

 protected:
  // Samples output logits and write to ids_tensor.
  absl::Status SampleLogits(const TensorBuffer& logits,
                            TensorBuffer& ids_tensor);

  // Prefill internal implementation, for one prefill call to the Interpreter
  // with a certain length synchronously or asynchronously.
  absl::Status PrefillInternal(
      absl::string_view prefill_signature,
      absl::flat_hash_map<absl::string_view /*input_name*/,
                          ::litert::TensorBuffer>& prefill_input_buffers,
      absl::Span<const int> ids, bool async);

  // Decode internal implementation. Uses the specified 'token' as the input
  // token and uses the specified 'step' as the current time step.  The
  // logits from the decode step are stored in the 'logits' output buffer of
  // the transformer model when this function returns absl::OkStatus().
  virtual absl::Status DecodeInternal(
      int step, const std::vector<std::shared_ptr<TokenData>>& token,
      TensorBuffer& output_logits);

  // Create Prefill input buffers for a given signature.
  absl::Status CreatePrefillInputBuffers(
      absl::string_view prefill_signature, int sequence_length,
      int context_length,
      absl::flat_hash_map<absl::string_view, TensorBuffer>&
          prefill_input_buffers);

  // Fills the input buffer from the unprocessed token.
  absl::Status FillInputBufferWithToken(
      const std::vector<std::shared_ptr<TokenData>>& unprocessed_token,
      ::litert::TensorBuffer& input_buffer,
      bool is_per_layer_embedding = false);

  // Prepares the first decode step.
  // When output_batch_size_ > 1, It broadcasts KV cache buffers to
  // output_batch_size_ times for the rest of the decode steps.
  // When output_batch_size_ == 1, It doesn't do anything.
  absl::Status PrepareFirstDecode();

  // Gets the token to decode. If there is id provided in the inputs, it will be
  // returned as the token to decode. Otherwise, the next unprocessed token will
  // be returned.
  absl::StatusOr<ProcessedTokens::StepAndToken> GetTokenToDecode(
      const ExecutorInputs& inputs);

  // Mark the pending token as processed if there is one, or adds the token as a
  // processed token.
  absl::Status ConsumePendingOrAddProcessedToken(
      const std::vector<std::shared_ptr<TokenData>>& token);

  LlmExecutorSettings executor_settings_;
  ::litert::Environment& env_;
  const ::litert::Model& model_;
  ::litert::CompiledModel compiled_model_;

  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_input_buffers_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      decode_output_buffers_;
  // KV cache double buffers because some GPU backends can't allocate one buffer
  // for both read and write at the same time.
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      kv_cache_buffers_1_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      kv_cache_buffers_2_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
      input_kv_cache_buffers_;
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>*
      output_kv_cache_buffers_;
  // KV cache (double) buffers used during decode when output_batch_size_ > 1.
  std::optional<absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
      decode_kv_cache_buffers_1_;
  std::optional<absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
      decode_kv_cache_buffers_2_;

  // The signatures of the model.
  ModelSignatures signatures_;

  // The sampled ids to use for external sampling.
  // The layout is batch-major.
  // e.g. for output_batch_size=2, the layout is:
  // {batch_0_seq_0, batch_1_seq_0, batch_0_seq_1, batch_1_seq_1, ...}
  std::vector<int> sampled_ids_;
  // Output batch size for the sampled ids.
  const int output_batch_size_;
  // Whether decode has been run ever after prefill.
  // TODO: b/409401231 - Make sure this state is session dependent.
  bool ran_decode_ = false;

  // Sampler for sampling logits.
  // For now, only CPU sampler is supported.
  std::unique_ptr<Sampler> sampler_;

  // Internal timestep.
  int current_step_ = 0;

  // Keeps track of processed tokens during the LLM execution. This also keeps
  // track of the pending input token, if any.
  ProcessedTokens processed_tokens_;

  // The path to the weight cache directory. Executor will take the ownership of
  // this path to maintain the path lifecycle.
  std::string weight_cache_path_;

  // The embedding lookup for the optional embedder model.
  std::unique_ptr<EmbeddingLookupManager> embedding_lookup_;

  // The embedding lookup for the optional per layer embedder model.
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup_;

  // Whether to use FP16 precision for the calculation.
  bool use_fp16_precision_;

  // The logits data type of the model, used to determine the data type of the
  // logits tensor for gpu sampling.
  LogitsDataType logits_data_type_;
};

// The static executor for the prefill-decode compiled model.
// This variant is instantiated when the model is statically shaped.
class LlmLiteRtCompiledModelExecutorStatic
    : public LlmLiteRtCompiledModelExecutorBase {
 public:
  static absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutorStatic>>
  Create(LlmExecutorSettings executor_settings, Environment& lrt_env,
         ModelResources& resources);

  using LlmLiteRtCompiledModelExecutorBase::Prefill;

  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& params) override;

 private:
  LlmLiteRtCompiledModelExecutorStatic(
      LlmExecutorSettings executor_settings, ::litert::Environment& env,
      const ::litert::Model* absl_nonnull model,
      ::litert::CompiledModel compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_output_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          input_kv_cache_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          output_kv_cache_buffers,
      std::optional<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
          decode_input_kv_cache_buffers,
      std::optional<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>>
          decode_output_kv_cache_buffers,
      SortedPrefillSignatureMap prefill_signature_map,
      ModelSignatures signatures, int output_batch_size,
      std::string weight_cache_path,
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup = nullptr,
      std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup =
          nullptr,
      bool use_fp16_precision = true,
      LogitsDataType logits_data_type = LogitsDataType::FLOAT32)
      : LlmLiteRtCompiledModelExecutorBase(
            std::move(executor_settings), env, model, std::move(compiled_model),
            std::move(decode_input_buffers), std::move(decode_output_buffers),
            std::move(input_kv_cache_buffers),
            std::move(output_kv_cache_buffers),
            std::move(decode_input_kv_cache_buffers),
            std::move(decode_output_kv_cache_buffers), signatures,
            output_batch_size, std::move(weight_cache_path),
            std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
            use_fp16_precision, logits_data_type),
        prefill_signature_map_(std::move(prefill_signature_map)) {}

  SortedPrefillSignatureMap prefill_signature_map_;
  // Signature names are unique across all signatures in a model so it is safe
  // to refer to them by just their unique name.
  absl::flat_hash_map<std::string /*prefill_signature_name*/,
                      absl::flat_hash_map<absl::string_view /*input_name*/,
                                          ::litert::TensorBuffer>>
      prefill_input_buffers_;
};

// The dynamic executor for the prefill-decode compiled model.
// This variant is instantiated when the model is dynamically shaped, in
// particular, input sequence length and KV cache size are dynamic.
class LlmLiteRtCompiledModelExecutorDynamic
    : public LlmLiteRtCompiledModelExecutorBase {
 public:
  static absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutorDynamic>>
  Create(LlmExecutorSettings executor_settings, Environment& lrt_env,
         ModelResources& resources);

  using LlmLiteRtCompiledModelExecutorBase::Prefill;

  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& params) override;

 private:
  LlmLiteRtCompiledModelExecutorDynamic(
      LlmExecutorSettings executor_settings, ::litert::Environment& env,
      const ::litert::Model* absl_nonnull model,
      ::litert::CompiledModel compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          decode_output_buffers,
      int prefill_chunk_size, int key_dynamic_dim_index,
      int value_dynamic_dim_index, int kv_increament_size,
      std::vector<std::string> key_cache_input_names,
      std::vector<std::string> value_cache_input_names,
      ModelSignatures signatures, int output_batch_size,
      std::string weight_cache_path,
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup = nullptr,
      std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup =
          nullptr,
      bool use_fp16_precision = true,
      LogitsDataType logits_data_type = LogitsDataType::FLOAT32)
      : LlmLiteRtCompiledModelExecutorBase(
            std::move(executor_settings), env, model, std::move(compiled_model),
            std::move(decode_input_buffers), std::move(decode_output_buffers),
            /*input_kv_cache_buffers=*/{},
            /*output_kv_cache_buffers=*/{},
            /*decode_input_kv_cache_buffers=*/std::nullopt,
            /*decode_output_kv_cache_buffers=*/std::nullopt, signatures,
            output_batch_size, std::move(weight_cache_path),
            std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
            use_fp16_precision, logits_data_type),
        prefill_chunk_size_(prefill_chunk_size),
        key_dynamic_dim_index_(key_dynamic_dim_index),
        value_dynamic_dim_index_(value_dynamic_dim_index),
        kv_increament_size_(kv_increament_size),
        key_cache_input_names_(std::move(key_cache_input_names)),
        value_cache_input_names_(std::move(value_cache_input_names)) {}

  absl::Status PrefillInternal(absl::Span<int> ids,
                               const ExecutorPrefillParams& params);

  // Extends the base class DecodeInternal to handle KV cache buffers.
  absl::Status DecodeInternal(
      int step, const std::vector<std::shared_ptr<TokenData>>& token,
      TensorBuffer& output_logits) override;

  int prefill_chunk_size_;
  int key_dynamic_dim_index_;
  int value_dynamic_dim_index_;
  uint32_t kv_increament_size_;
  std::vector<std::string> key_cache_input_names_;
  std::vector<std::string> value_cache_input_names_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_H_
