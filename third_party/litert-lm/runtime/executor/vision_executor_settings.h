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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_EXECUTOR_SETTINGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_EXECUTOR_SETTINGS_H_

#include <ostream>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"

namespace litert::lm {

// The VisionExecutorSettings class is used to configure the VisionExecutor.
// It is used to configure the vision encoder and vision adapter models.
// Args:
//   - model_assets: The model assets to use for the vision encoder and vision
//   adapter models.
//   - encoder_backend: The backend to use for the vision encoder model.
//   - adapter_backend: The backend to use for the vision adapter model.
class VisionExecutorSettings : public ExecutorSettingsBase {
 public:
  static absl::StatusOr<VisionExecutorSettings> CreateDefault(
      const ModelAssets& model_assets, Backend encoder_backend,
      Backend adapter_backend);

  // Getter for encoder_backend.
  Backend GetEncoderBackend() const;
  // Setter for encoder_backend.
  absl::Status SetEncoderBackend(Backend backend);

  // Getter for adapter_backend.
  Backend GetAdapterBackend() const;
  // Setter for adapter_backend.
  absl::Status SetAdapterBackend(Backend backend);

 private:
  explicit VisionExecutorSettings(const ModelAssets& model_assets)
      : ExecutorSettingsBase(model_assets) {}

  // The backend to use for the vision encoder model.
  Backend encoder_backend_;

  // The backend to use for the vision adapter model.
  Backend adapter_backend_;
};

std::ostream& operator<<(std::ostream& os,
                         const VisionExecutorSettings& settings);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_VISION_EXECUTOR_SETTINGS_H_
