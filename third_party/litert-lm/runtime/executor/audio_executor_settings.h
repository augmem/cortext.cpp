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

#ifndef THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_AUDIO_EXECUTOR_SETTINGS_H_
#define THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_AUDIO_EXECUTOR_SETTINGS_H_

#include <ostream>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"

namespace litert::lm {

class AudioExecutorSettings : public ExecutorSettingsBase {
 public:
  static absl::StatusOr<AudioExecutorSettings> CreateDefault(
      const ModelAssets& model_assets, int max_sequence_length, Backend backend,
      bool bundled_with_main_model = true);

  // Getter for max_sequence_length.
  int GetMaxSequenceLength() const;
  // Setter for max_sequence_length.
  void SetMaxSequenceLength(int max_sequence_length);

  // Getter for bundled_with_main_model.
  bool GetBundledWithMainModel() const;
  // Setter for bundled_with_main_model.
  void SetBundledWithMainModel(bool bundled_with_main_model);

  absl::Status SetBackend(const Backend& backend) override;

 private:
  explicit AudioExecutorSettings(const ModelAssets& model_assets,
                                 int max_sequence_length)
      : ExecutorSettingsBase(model_assets),
        max_sequence_length_(max_sequence_length) {}

  int max_sequence_length_;

  bool bundled_with_main_model_;
};

std::ostream& operator<<(std::ostream& os,
                         const AudioExecutorSettings& settings);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITE_RT_LLM_EXECUTOR_AUDIO_EXECUTOR_SETTINGS_H
