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

#include "runtime/executor/vision_litert_compiled_model_executor.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#if !defined(LITERT_DISABLE_NPU)
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#endif  // !defined(LITERT_DISABLE_NPU)
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/options/litert_runtime_options.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/file_util.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {

absl::StatusOr<
    std::unique_ptr<VisionLiteRtCompiledModelExecutor::VisionEncoder>>
VisionLiteRtCompiledModelExecutor::VisionEncoder::Create(
    Environment& env, const Model* absl_nonnull model,
    const VisionExecutorSettings& vision_executor_settings) {
  auto handler = std::unique_ptr<VisionEncoder>(
      new VisionEncoder(env, model, vision_executor_settings));
  RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status VisionLiteRtCompiledModelExecutor::VisionEncoder::Initialize() {
  // TODO(b/405424188): - Add support for NPU backends.
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  std::string weight_cache_path = vision_executor_settings_.GetCacheDir();
  auto activation_data_type = ActivationDataType::FLOAT16;
  if (vision_executor_settings_.GetActivationDataType().has_value()) {
    activation_data_type =
        vision_executor_settings_.GetActivationDataType().value();
  }
  switch (backend_) {
    case Backend::CPU: {
      // TODO: b/403132820 - Add accelerator compilation options for XNNPACK.
      LITERT_ASSIGN_OR_RETURN(auto cpu_compilation_options,
                                   CpuOptions::Create());
      // Set the number of threads to 4 by default.
      cpu_compilation_options.SetNumThreads(4);

      LITERT_ASSIGN_OR_RETURN(RuntimeOptions runtime_options,
                                   RuntimeOptions::Create());
      options.AddOpaqueOptions(std::move(runtime_options));
      options.AddOpaqueOptions(std::move(cpu_compilation_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      LITERT_ASSIGN_OR_RETURN(GpuOptions gpu_compilation_options,
                              GpuOptions::Create());
      gpu_compilation_options.EnableConstantTensorSharing(true);
      if (activation_data_type == ActivationDataType::FLOAT32) {
        gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp32);
      } else {
        gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp32);
      }
      gpu_compilation_options.SetPreferTextureWeights(true);

      if (weight_cache_path != ":nocache") {
        ASSIGN_OR_RETURN(auto model_path,
                         vision_executor_settings_.GetModelAssets().GetPath());
        if (weight_cache_path.empty()) {
          weight_cache_path = Dirname(model_path);
        }
        gpu_compilation_options.SetSerializationDir(weight_cache_path.c_str());
        absl::string_view model_name = Basename(model_path);
        gpu_compilation_options.SetModelCacheKey(model_name.data());
        gpu_compilation_options.SetSerializeProgramCache(true);
        gpu_compilation_options.SetSerializeExternalTensors(true);
      }
      options.AddOpaqueOptions(std::move(gpu_compilation_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
      break;
    }
#if !defined(LITERT_DISABLE_NPU)
    case Backend::NPU: {
      LITERT_ASSIGN_OR_RETURN(auto qualcomm_options,
                                   qualcomm::QualcommOptions::Create());
      qualcomm_options.SetLogLevel(qualcomm::QualcommOptions::LogLevel::kOff);
      qualcomm_options.SetHtpPerformanceMode(
          qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
      options.AddOpaqueOptions(std::move(qualcomm_options));
      // TODO: yunandrew - Add support for other NPU backends.
      options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
#endif  // !defined(LITERT_DISABLE_NPU)
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported encoder backend: ", backend_));
  }

  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_, options));
  if (auto num_signatures = model_.GetNumSignatures(); num_signatures != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Vision Encoder model must have exactly one signature but got ",
        num_signatures));
  }
  LITERT_ASSIGN_OR_RETURN(input_buffers_, compiled_model_.CreateInputBuffers(
                                              /*signature_index=*/0));
  LITERT_ASSIGN_OR_RETURN(output_buffers_, compiled_model_.CreateOutputBuffers(
                                               /*signature_index=*/0));
  if (output_buffers_.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("The Vision Encoder model must have exactly one output "
                     "buffer but got ",
                     output_buffers_.size()));
  }

  return absl::OkStatus();
}

absl::StatusOr<
    std::unique_ptr<VisionLiteRtCompiledModelExecutor::VisionAdapter>>
VisionLiteRtCompiledModelExecutor::VisionAdapter::Create(
    Environment& env, const Model* absl_nonnull model, Backend backend) {
  auto handler =
      std::unique_ptr<VisionAdapter>(new VisionAdapter(env, model, backend));
  RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status VisionLiteRtCompiledModelExecutor::VisionAdapter::Initialize() {
  // TODO(b/405424188): - Add support for NPU backends.
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  switch (backend_) {
    case Backend::CPU: {
      // TODO: b/403132820 - Add accelerator compilation options for XNNPACK.
      LITERT_ASSIGN_OR_RETURN(auto cpu_compilation_options,
                                   CpuOptions::Create());
      // Set the number of threads to 4 by default.
      cpu_compilation_options.SetNumThreads(4);

      LITERT_ASSIGN_OR_RETURN(RuntimeOptions runtime_options,
                                   RuntimeOptions::Create());
      options.AddOpaqueOptions(std::move(runtime_options));
      options.AddOpaqueOptions(std::move(cpu_compilation_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      LITERT_ASSIGN_OR_RETURN(GpuOptions gpu_compilation_options,
                              GpuOptions::Create());
      gpu_compilation_options.EnableConstantTensorSharing(true);
      gpu_compilation_options.EnableAllowSrcQuantizedFcConvOps(true);

      gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp16);
      gpu_compilation_options.SetPreferTextureWeights(true);
      options.AddOpaqueOptions(std::move(gpu_compilation_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported adapter backend: ", backend_));
  }

  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_, options));
  if (auto num_signatures = model_.GetNumSignatures(); num_signatures != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Vision Adapter model must have exactly one signature but got ",
        num_signatures));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<VisionLiteRtCompiledModelExecutor>>
litert::lm::VisionLiteRtCompiledModelExecutor::Create(
    const VisionExecutorSettings& vision_executor_settings, Environment& env) {
  LITERT_ASSIGN_OR_RETURN(auto resources,
                          BuildLiteRtCompiledModelResources(
                              vision_executor_settings.GetModelAssets()));

  ASSIGN_OR_RETURN(auto vision_encoder_model,
                   resources->GetTFLiteModel(ModelType::kTfLiteVisionEncoder));
  if (!vision_encoder_model) {
    return absl::InternalError("Failed to build LiteRt encoder model.");
  }
  ASSIGN_OR_RETURN(auto vision_adapter_model,
                   resources->GetTFLiteModel(ModelType::kTfLiteVisionAdapter));
  if (!vision_adapter_model) {
    return absl::InternalError("Failed to build LiteRt adapter model.");
  }

  ASSIGN_OR_RETURN(auto vision_encoder,
                   VisionEncoder::Create(env, vision_encoder_model,
                                         vision_executor_settings));
  ASSIGN_OR_RETURN(
      auto vision_adapter,
      VisionAdapter::Create(env, vision_adapter_model,
                            vision_executor_settings.GetAdapterBackend()));

  LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                               vision_encoder_model->GetInputTensorType(0, 0));
  const auto& dimensions = tensor_type.Layout().Dimensions();
  if (dimensions.size() != 4) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Expected encoder input tensor to have 4 dimensions, but got ",
        dimensions.size()));
  }
  if (dimensions[3] < 3 || dimensions[3] > 4) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected encoder input tensor to have 3 or 4 channels, "
                     "but got ",
                     dimensions[3]));
  }
  auto expected_input_dimension =
      std::vector<int>(dimensions.begin(), dimensions.end());

  return absl::WrapUnique(new VisionLiteRtCompiledModelExecutor(
      vision_executor_settings, env, std::move(resources),
      std::move(vision_encoder), std::move(vision_adapter),
      expected_input_dimension));
}

absl::StatusOr<ExecutorVisionData> VisionLiteRtCompiledModelExecutor::Encode(
    const litert::TensorBuffer& input_image_tensor) {
  LITERT_ASSIGN_OR_RETURN(auto input_image_tensor_ref,
                               input_image_tensor.Duplicate());
  LITERT_ASSIGN_OR_RETURN(
      auto output_tensor_buffers,
      vision_adapter_->GetCompiledModel().CreateOutputBuffers(
          /*signature_index=*/0));
  if (output_tensor_buffers.size() != 1) {
    return absl::InternalError(
        absl::StrCat("The Vision Adapter model must have exactly one output "
                     "buffer but got ",
                     output_tensor_buffers.size()));
  }

  LITERT_ASSIGN_OR_RETURN(auto input_image_data,
                          ReferTensorBufferAsSpan<float>(input_image_tensor));
  LITERT_RETURN_IF_ERROR(
      vision_encoder_->GetMutableInputBuffers()[0].Write<float>(
          input_image_data));
  auto& encoder_outputs = vision_encoder_->GetMutableOutputBuffers();
  if (encoder_outputs[0].IsWebGpuMemory()) {
    // For WebGPU memory, we need to create a new output buffer to hold the
    // data, otherwise we will get failed to lock TensorBuffer error on the
    // second call to `Encode`. See b/457483190
    LITERT_ASSIGN_OR_RETURN(
        encoder_outputs,
        vision_encoder_->GetCompiledModel().CreateOutputBuffers(
            /*signature_index=*/0));
  }

  LITERT_RETURN_IF_ERROR(vision_encoder_->GetCompiledModel().Run(
      /*input_buffers=*/vision_encoder_->GetInputBuffers(),
      /*output_buffers=*/encoder_outputs));

  LITERT_RETURN_IF_ERROR(vision_adapter_->GetCompiledModel().Run(
      /*input_buffers=*/encoder_outputs,
      /*output_buffers=*/output_tensor_buffers));

  return ExecutorVisionData(std::move(output_tensor_buffers[0]),
                            /*per_layer_embeddings=*/std::nullopt);
}

absl::StatusOr<std::vector<int>>
VisionLiteRtCompiledModelExecutor::GetExpectedInputDimension() const {
  return expected_input_dimension_;
}

}  // namespace litert::lm
