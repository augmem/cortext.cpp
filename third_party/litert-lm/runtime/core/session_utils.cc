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

#include "runtime/core/session_utils.h"

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/internal/litert_detail.h"  // from @litert
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {

absl::StatusOr<std::string> MaybeGetBosString(
    const SessionConfig& session_config, Tokenizer& tokenizer) {
  auto bos_token_id = session_config.GetStartTokenId();
  std::string bos_string = "";
  if (bos_token_id >= 0) {
    ASSIGN_OR_RETURN(bos_string, tokenizer.TokenIdsToText({bos_token_id}));
  }
  return bos_string;
}

absl::StatusOr<InputText> StringToProcessedInputText(
    absl::string_view text, const SessionConfig& session_config,
    Tokenizer& tokenizer, const std::optional<BenchmarkInfo>& benchmark_info) {
  auto bos_token_id = session_config.GetStartTokenId();
  std::string bos_string = "";
  if (bos_token_id >= 0) {
    ASSIGN_OR_RETURN(bos_string, tokenizer.TokenIdsToText({bos_token_id}));
  }
  bool bos_token_found = false;
  if (!bos_string.empty() && StartsWith(text, bos_string)) {
    text = text.substr(bos_string.size());
    bos_token_found = true;
  }

  int benchmark_prefill_token_count = 0;
  if (benchmark_info.has_value()) {
    benchmark_prefill_token_count =
        benchmark_info->GetBenchmarkParams().num_prefill_tokens();
  }
  ASSIGN_OR_RETURN(std::vector<int> ids, tokenizer.TextToTokenIds(text));
  if (benchmark_prefill_token_count > 0) {
    // If benchmark is enabled, we will use the benchmark prefill token
    // count to set the prefill token count.
    ids.resize(benchmark_prefill_token_count);
  } else if (bos_token_found) {
    ids.insert(ids.begin(), session_config.GetStartTokenId());
  }
  ASSIGN_OR_RETURN(auto ids_buffer, tokenizer.TokenIdsToTensorBuffer(ids));
  return InputText(std::move(ids_buffer));
}

absl::StatusOr<std::vector<InputData>> ApplyPromptTemplates(
    const std::vector<InputData>& contents, const SessionConfig& session_config,
    Tokenizer& tokenizer, bool& is_first_turn) {
  if (contents.empty()) {
    return std::vector<InputData>();
  }
  ASSIGN_OR_RETURN(std::string bos_string,
                   MaybeGetBosString(session_config, tokenizer));
  std::vector<InputData> templated_contents;
  if (!session_config.GetApplyPromptTemplateInSession()) {
    if (is_first_turn && !bos_string.empty()) {
      templated_contents.push_back(InputText(bos_string));
    }
    is_first_turn = false;
    for (int i = 0; i < contents.size(); ++i) {
      const auto& content = contents[i];
      ASSIGN_OR_RETURN(auto content_copy, CreateInputDataCopy(content));
      templated_contents.emplace_back(std::move(content_copy));
    }
    return templated_contents;
  }

  for (int i = 0; i < contents.size(); ++i) {
    const auto& content = contents[i];
    const bool is_first_chunk = i == 0;
    const bool is_last_chunk = i == contents.size() - 1;
    absl::string_view raw_text = "";
    if (const auto* input_text = std::get_if<InputText>(&content);
        input_text != nullptr && !input_text->IsTensorBuffer()) {
      ASSIGN_OR_RETURN(raw_text, input_text->GetRawTextString());
    }

    // Check if the input starts with the BOS string. If it does, return an
    // error. This is to prevent the user from including the BOS string in the
    // input. This is also needed for the current implementation as tokenizer
    // will treat the BOS string differently from other strings. If the BOS
    // string is empty, it means the BOS token id is not valid. In this case, we
    // will not check for the BOS string in the input.
    if (!bos_string.empty() && StartsWith(raw_text, bos_string)) {
      return absl::InvalidArgumentError(
          "Input contains bos control token. Control token should not be "
          "included in the input.");
    }

    std::string session_prefix = "";
    if (is_first_chunk) {
      session_prefix = is_first_turn ? bos_string : "\n";
      is_first_turn = false;
    }
    std::string turn_prefix = absl::StrCat(
        session_prefix, session_config.GetPromptTemplates().user().prefix());
    std::string turn_suffix =
        absl::StrCat(session_config.GetPromptTemplates().user().suffix(),
                     session_config.GetPromptTemplates().model().prefix());

    if (raw_text.empty()) {
      // Non-text chunk. Add templates as separate InputText objects.
      if (is_first_chunk && !turn_prefix.empty()) {
        templated_contents.push_back(InputText(std::move(turn_prefix)));
      }
      // TODO - b/445254659: Remove all actual copies.
      ASSIGN_OR_RETURN(auto content_copy, CreateInputDataCopy(content));
      templated_contents.emplace_back(std::move(content_copy));
      if (is_last_chunk && !turn_suffix.empty()) {
        templated_contents.push_back(InputText(std::move(turn_suffix)));
      }
    } else {
      // Raw text chunk. Combine templates with the raw text.
      std::string templated_text;
      if (is_first_chunk) {
        templated_text = absl::StrCat(turn_prefix, raw_text);
      } else {
        templated_text = std::string(raw_text);
      }
      if (is_last_chunk) {
        absl::StrAppend(&templated_text, turn_suffix);
      }
      templated_contents.push_back(InputText(std::move(templated_text)));
    }
  }
  return templated_contents;
}

absl::StatusOr<std::vector<InputData>> PreprocessContents(
    const std::vector<InputData>& contents, const SessionConfig& session_config,
    Tokenizer& tokenizer, const std::optional<BenchmarkInfo>& benchmark_info) {
  std::vector<InputData> preprocessed_contents;
  for (int i = 0; i < contents.size(); ++i) {
    const auto& content = contents[i];
    if (const auto* input_text = std::get_if<InputText>(&content)) {
      if (input_text->IsTensorBuffer()) {
        ASSIGN_OR_RETURN(auto input_text_copy, input_text->CreateCopy());
        preprocessed_contents.emplace_back(std::move(input_text_copy));
      } else {
        ASSIGN_OR_RETURN(auto templated_text, input_text->GetRawTextString());
        ASSIGN_OR_RETURN(
            auto processed_input_text,
            StringToProcessedInputText(templated_text, session_config,
                                       tokenizer, benchmark_info));
        preprocessed_contents.emplace_back(std::move(processed_input_text));
      }
    } else if (const auto* input_image = std::get_if<InputImage>(&content)) {
      if (input_image->IsTensorBuffer()) {
        ASSIGN_OR_RETURN(auto input_image_copy, input_image->CreateCopy());
        preprocessed_contents.emplace_back(std::move(input_image_copy));
      } else {
        return absl::InternalError(
            "Image must be preprocessed before being used in SessionAdvanced.");
      }
    } else if (const auto* input_audio = std::get_if<InputAudio>(&content)) {
      if (input_audio->IsTensorBuffer()) {
        ASSIGN_OR_RETURN(auto input_audio_copy, input_audio->CreateCopy());
        preprocessed_contents.emplace_back(std::move(input_audio_copy));
      } else {
        return absl::InternalError(
            "Audio must be preprocessed before being used in SessionAdvanced.");
      }
    } else if (const auto* input_audio_end =
                   std::get_if<InputAudioEnd>(&content)) {
      preprocessed_contents.emplace_back(InputAudioEnd());
    }
  }
  return preprocessed_contents;
}

}  // namespace litert::lm
