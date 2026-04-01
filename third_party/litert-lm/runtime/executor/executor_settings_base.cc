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

#include "runtime/executor/executor_settings_base.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/file_util.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {

std::string GetBackendString(Backend backend) {
  switch (backend) {
    case Backend::CPU_ARTISAN:
      return "CPU_ARTISAN";
    case Backend::GPU_ARTISAN:
      return "GPU_ARTISAN";
    case Backend::GPU:
      return "GPU";
    case Backend::CPU:
      return "CPU";
    case Backend::GOOGLE_TENSOR_ARTISAN:
      return "GOOGLE_TENSOR_ARTISAN";
    case Backend::NPU:
      return "NPU";
    default:
      return "UNSPECIFIED";
  }
}

std::ostream& operator<<(std::ostream& os, const Backend& backend) {
  return os << GetBackendString(backend);
}

absl::StatusOr<Backend> GetBackendFromString(absl::string_view backend_str) {
  if (absl::EqualsIgnoreCase(backend_str, "cpu")) {
    return Backend::CPU;
  } else if (absl::EqualsIgnoreCase(backend_str, "gpu")) {
    return Backend::GPU;
  } else if (absl::EqualsIgnoreCase(backend_str, "npu")) {
    return Backend::NPU;
  } else if (absl::EqualsIgnoreCase(backend_str, "gpu_artisan")) {
    return Backend::GPU_ARTISAN;
  } else if (absl::EqualsIgnoreCase(backend_str, "cpu_artisan")) {
    return Backend::CPU_ARTISAN;
  } else if (absl::EqualsIgnoreCase(backend_str, "google_tensor_artisan")) {
    return Backend::GOOGLE_TENSOR_ARTISAN;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend: ", backend_str,
                     ". Supported backends are: [CPU, GPU, NPU, GPU_ARTISAN, "
                     "CPU_ARTISAN, GOOGLE_TENSOR_ARTISAN]"));
  }
}

std::ostream& operator<<(std::ostream& os,
                         const ActivationDataType& activation) {
  switch (activation) {
    case ActivationDataType::FLOAT32:
      return os << "FLOAT32";
    case ActivationDataType::FLOAT16:
      return os << "FLOAT16";
    case ActivationDataType::INT16:
      return os << "INT16";
    case ActivationDataType::INT8:
      return os << "INT8";
    default:
      return os << "UNKNOWN";
  }
}

absl::StatusOr<ActivationDataType> GetActivationDataTypeFromString(
    const std::string& activation_data_type) {
  if (absl::EqualsIgnoreCase(activation_data_type, "float32")) {
    return ActivationDataType::FLOAT32;
  } else if (absl::EqualsIgnoreCase(activation_data_type, "float16")) {
    return ActivationDataType::FLOAT16;
  } else if (absl::EqualsIgnoreCase(activation_data_type, "int16")) {
    return ActivationDataType::INT16;
  } else if (absl::EqualsIgnoreCase(activation_data_type, "int8")) {
    return ActivationDataType::INT8;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported activation data type: ", activation_data_type,
                     ". Supported activation data types are: [FLOAT32, "
                     "FLOAT16, INT16, INT8]"));
  }
}

std::ostream& operator<<(std::ostream& os,
                         const FakeWeightsMode& fake_weights_mode) {
  switch (fake_weights_mode) {
    case FakeWeightsMode::FAKE_WEIGHTS_NONE:
      return os << "FAKE_WEIGHTS_NONE";
    case FakeWeightsMode::FAKE_WEIGHTS_8BITS_ALL_LAYERS:
      return os << "FAKE_WEIGHTS_8BITS_ALL_LAYERS";
    case FakeWeightsMode::FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4:
      return os << "FAKE_WEIGHTS_ATTN_8_FFN_4_EMB_4";
    default:
      return os << "FAKE_WEIGHTS_NONE";
  }
}

std::ostream& operator<<(std::ostream& os, const FileFormat& file_format) {
  switch (file_format) {
    case FileFormat::TFLITE:
      return os << "TFLITE";
    case FileFormat::TASK:
      return os << "TASK";
    case FileFormat::LITERT_LM:
      return os << "LITERT_LM";
  }
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(absl::string_view model_path) {
  return ModelAssets(model_path);
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(
    std::shared_ptr<litert::lm::ScopedFile> model_file) {
  return ModelAssets(std::move(model_file));
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(
    std::shared_ptr<litert::lm::MemoryMappedFile> model_file) {
  return ModelAssets(std::move(model_file));
}

// static
absl::StatusOr<ModelAssets> ModelAssets::Create(
    std::shared_ptr<litert::lm::ScopedFile> model_file,
    absl::string_view model_path) {
  if (model_file) {
    return Create(std::move(model_file));
  }
  return Create(model_path);
}

ModelAssets::ModelAssets(std::shared_ptr<litert::lm::ScopedFile> model_file)
    : path_or_file_(std::move(model_file)) {}

ModelAssets::ModelAssets(absl::string_view model_path)
    : path_or_file_(std::string(model_path)) {}

ModelAssets::ModelAssets(
    std::shared_ptr<litert::lm::MemoryMappedFile> model_file)
    : path_or_file_(std::move(model_file)) {}

absl::StatusOr<absl::string_view> ModelAssets::GetPath() const {
  if (std::holds_alternative<std::string>(path_or_file_)) {
    return std::get<std::string>(path_or_file_);
  }
  return absl::InvalidArgumentError("Assets were not created with a path.");
}

absl::StatusOr<std::shared_ptr<ScopedFile>> ModelAssets::GetScopedFile() const {
  if (!HasScopedFile()) {
    return absl::InvalidArgumentError(
        "Assets were not created with a scoped file.");
  }
  return std::get<std::shared_ptr<ScopedFile>>(path_or_file_);
}

absl::StatusOr<std::shared_ptr<MemoryMappedFile>>
ModelAssets::GetMemoryMappedFile() const {
  if (!HasMemoryMappedFile()) {
    return absl::InvalidArgumentError(
        "Assets were not created with a memory mapped file.");
  }
  return std::get<std::shared_ptr<MemoryMappedFile>>(path_or_file_);
}

absl::StatusOr<std::shared_ptr<ScopedFile>> ModelAssets::GetOrCreateScopedFile()
    const {
  if (HasScopedFile()) {
    return std::get<std::shared_ptr<ScopedFile>>(path_or_file_);
  }
  if (HasMemoryMappedFile()) {
    return absl::InvalidArgumentError(
        "Cannot create ScopedFile from MemoryMappedFile.");
  }

  ASSIGN_OR_RETURN(  // NOLINT
      auto scoped_file, ScopedFile::Open(std::get<std::string>(path_or_file_)));
  return std::make_shared<ScopedFile>(std::move(scoped_file));
}

std::ostream& operator<<(std::ostream& os, const ModelAssets& model_assets) {
  if (model_assets.HasScopedFile()) {
    os << "model_file file descriptor ID: "
       << model_assets.GetScopedFile().value()->file() << "\n";
  } else if (model_assets.HasMemoryMappedFile()) {
    os << "model_file memory mapped file: "
       << model_assets.GetMemoryMappedFile().value()->data() << "\n";
  } else {
    os << "model_path: " << model_assets.GetPath().value() << "\n";
  }
  os << "fake_weights_mode: " << model_assets.fake_weights_mode() << "\n";
  return os;
}

absl::StatusOr<
    std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>
ExecutorSettingsBase::GetWeightCacheFile(absl::string_view suffix) const {
  // Cache is explicitly disabled.
  if (GetCacheDir() == ":nocache") {
    return absl::InvalidArgumentError("Cache is explicitly disabled.");
  }

  // Prefer to use the scoped cache file if it's set.
  if (GetScopedCacheFile()) {
    return GetScopedCacheFile();
  }

  auto model_path = GetModelAssets().GetPath().value_or("");

  // There is no model path to suffix.
  if (model_path.empty()) {
    return absl::InvalidArgumentError(
        "Cache path cannot be computed without knowing the model path.");
  }

  if (GetCacheDir().empty()) {
    return absl::StrCat(model_path, suffix);
  }

  return JoinPath(GetCacheDir(), absl::StrCat(Basename(model_path), suffix));
}

}  // namespace litert::lm
