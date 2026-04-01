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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_UTILS_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// The util function to get the BOS string if there is a valid BOS token id.
// Otherwise, return an empty string.
absl::StatusOr<std::string> MaybeGetBosString(
    const SessionConfig& session_config, Tokenizer& tokenizer);

// The util function to convert the string to processed input text.
absl::StatusOr<InputText> StringToProcessedInputText(
    absl::string_view text, const SessionConfig& session_config,
    Tokenizer& tokenizer, const std::optional<BenchmarkInfo>& benchmark_info);

// Util function for applying the prompt templates.
// input: The input text to apply the prompt templates.
// is_first_chunk: Whether the input is the first chunk of the turn.
// is_last_chunk: Whether the input is the last chunk of the turn.
// The output is the text input after applying the proper prompt templates.
// TODO - b/453312248: This is a temporary solution to add required templates
// to the input. Should be removed once the prompt templates are properly
// handled via the conversation layer.
absl::StatusOr<std::vector<InputData>> ApplyPromptTemplates(
    const std::vector<InputData>& contents, const SessionConfig& session_config,
    Tokenizer& tokenizer, bool& is_first_turn);

// Preprocesses the input contents. This function is used for pre-processing
// the input contents before sending them to the LLM executor.
// Text input will be preprocessed by the tokenizer.
absl::StatusOr<std::vector<InputData>> PreprocessContents(
    const std::vector<InputData>& contents, const SessionConfig& session_config,
    Tokenizer& tokenizer, const std::optional<BenchmarkInfo>& benchmark_info);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_UTILS_H_
