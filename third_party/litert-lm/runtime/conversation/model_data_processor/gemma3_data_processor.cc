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

#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/preprocessor/audio_preprocessor.h"
#include "runtime/components/preprocessor/audio_preprocessor_miniaudio.h"
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/preprocessor/stb_image_preprocessor.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/components/tool_use/python_tool_format_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2
#include "sentencepiece_model.pb.h"  // from @sentencepiece

namespace litert::lm {
namespace {

using ::nlohmann::ordered_json;

bool IsImage(absl::string_view part) {
  return part == "<start_of_image>" || part == "<image_soft_token>";
}

bool IsAudio(absl::string_view part) {
  return part == "<start_of_audio>" || part == "<audio_soft_token>";
}

bool HasToolCalls(const ordered_json& message) {
  return message.contains("tool_calls") && message["tool_calls"].is_array();
}

bool IsToolMessage(const ordered_json& message) {
  return message.contains("role") && message["role"] == "tool";
}

// Formats a tool response in Python format.
//
// The fields of the tool response may be under the key "tool_response",
// "response", or at the top-level.
//
// Example:
//
// Input:
//
// ```json
// {
//   "tool_response": {
//     "key1": "bar",
//     "key2": true
//   }
// }
// ```
//
// Output:
//
// ```
// {"key1": "bar", "key2": True}
// ```

absl::StatusOr<std::string> FormatToolResponse(
    const ordered_json& tool_response) {
  absl::string_view tool_response_key;
  if (tool_response.contains("tool_response")) {
    tool_response_key = "tool_response";
  } else if (tool_response.contains("response")) {
    tool_response_key = "response";
  } else {
    return FormatValueAsPython(tool_response);
  }

  return FormatValueAsPython(tool_response[tool_response_key]);
}

}  // namespace

absl::StatusOr<std::unique_ptr<Gemma3DataProcessor>>
Gemma3DataProcessor::Create(Gemma3DataProcessorConfig config,
                            std::optional<Preface> preface,
                            const Tokenizer* tokenizer,
                            const std::vector<std::vector<int>>& stop_token_ids,
                            bool enable_constrained_decoding) {
  ASSIGN_OR_RETURN(auto audio_preprocessor,
                   AudioPreprocessorMiniAudio::Create(
                       AudioPreprocessorConfig::CreateDefaultUsmConfig()));
  return absl::WrapUnique(new Gemma3DataProcessor(
      config, preface, std::make_unique<StbImagePreprocessor>(),
      std::move(audio_preprocessor)));
}

absl::StatusOr<ordered_json> Gemma3DataProcessor::MessageToTemplateInput(
    const ordered_json& message) const {
  // If the message doesn't contain any tool calls and isn't a tool message,
  // then the template input is the same as the message.
  if (!HasToolCalls(message) && !IsToolMessage(message)) {
    return message;
  }

  ordered_json template_input = ordered_json::object();
  if (message.contains("role")) {
    template_input["role"] = message["role"];
  }

  // Process content.
  if (message.contains("content")) {
    // If the role is "tool", convert the tool responses to Python format.
    if (IsToolMessage(message)) {
      if (message["content"].is_array()) {
        // If the content is an array, treat each item as a tool response.
        template_input["content"] = ordered_json::array();
        for (const auto& item : message["content"]) {
          ASSIGN_OR_RETURN(std::string formatted_tool_response,
                           FormatToolResponse(item));
          template_input["content"].push_back(
              {{"type", "text"}, {"text", formatted_tool_response}});
        }
      } else if (message["content"].is_object()) {
        // If the content is an object, treat it as a single tool response.
        ASSIGN_OR_RETURN(std::string formatted_tool_response,
                         FormatToolResponse(message["content"]));
        template_input["content"] = formatted_tool_response;
      } else {
        // If the content is neither an array nor an object, pass it through
        // unchanged.
        template_input["content"] = message["content"];
      }
    } else {
      // If the role is not "tool", then pass through content unchanged.
      template_input["content"] = message["content"];
    }
  }

  // If the message contains tool calls, then convert them to Python and
  // add them to the template input.
  if (message.contains("tool_calls")) {
    template_input["tool_calls"] = ordered_json::array();
    for (const auto& tool_call : message["tool_calls"]) {
      if (!tool_call.contains("function")) {
        continue;
      }
      const nlohmann::ordered_json& function = tool_call["function"];
      ordered_json tool_call_input = ordered_json::object();
      tool_call_input["type"] = "function";
      tool_call_input["function"]["name"] = function["name"];

      if (function.contains("arguments")) {
        if (function["arguments"].is_object()) {
          for (const auto& [key, value] : function["arguments"].items()) {
            ASSIGN_OR_RETURN(std::string formatted_value,
                             FormatValueAsPython(value));
            tool_call_input["function"]["arguments"][key] = formatted_value;
          }
        } else {
          tool_call_input["function"]["arguments"] = function["arguments"];
        }
      }

      template_input["tool_calls"].push_back(tool_call_input);
    }
  }

  return template_input;
}

absl::StatusOr<std::vector<InputData>>
Gemma3DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt, const ordered_json& messages,
    const Gemma3DataProcessorArguments& args) const {
  std::vector<InputData> input_data;
  std::deque<std::unique_ptr<MemoryMappedFile>> image_files;
  std::deque<std::unique_ptr<MemoryMappedFile>> audio_files;
  // Find all images and audio contained in the messages.
  for (const auto& message : messages) {
    if (message.contains("content") && message["content"].is_array()) {
      for (const auto& item : message["content"]) {
        if (item.is_string()) {
          continue;
        }
        ASSIGN_OR_RETURN(std::unique_ptr<MemoryMappedFile> mmap_file,
                         LoadItemData(item));
        if (item["type"] == "image") {
          image_files.push_back(std::move(mmap_file));
        } else if (item["type"] == "audio") {
          audio_files.push_back(std::move(mmap_file));
        }
      }
    }
  }

  RE2 re_delimiter(
      "(<start_of_image>|<image_soft_token>|<start_of_audio>|<audio_soft_token>"
      ")");
  absl::string_view prompt_view(rendered_template_prompt);
  const char* start = prompt_view.data();
  std::string part;
  ImagePreprocessParameter image_params;
  image_params.SetTargetDimensions(Dimensions(
      {1, config_.image_tensor_height, config_.image_tensor_width, 3}));
  // Replace the placeholders with the actual data. Note for Gemma3N the
  // placeholders in the prompt are <image_soft_token> and <audio_soft_token>,
  // while for Gemma3 the placeholders in the prompt are <start_of_image> and
  // <start_of_audio>.
  while (RE2::FindAndConsume(&prompt_view, re_delimiter, &part)) {
    absl::string_view text_part(start, prompt_view.data() - part.size());
    start = prompt_view.data();
    if (IsImage(part)) {
      input_data.emplace_back(
          InputText(absl::StrCat(text_part, "\n\n", config_.boi_token)));
      if (image_files.empty()) {
        return absl::InvalidArgumentError(
            "Provided less images than expected in the prompt.");
      }
      auto image_file = std::move(image_files.front());
      image_files.pop_front();
      ASSIGN_OR_RETURN(auto preprocessed_image,
                       image_preprocessor_->Preprocess(
                           InputImage(std::string(
                               static_cast<const char*>(image_file->data()),
                               image_file->length())),
                           image_params));
      input_data.emplace_back(InputImage(std::move(preprocessed_image)));
      input_data.emplace_back(InputText("\n\n"));
    } else if (IsAudio(part)) {
      input_data.emplace_back(
          InputText(absl::StrCat(text_part, "\n\n", config_.boa_token)));
      if (audio_files.empty()) {
        return absl::InvalidArgumentError(
            "Provided less audio than expected in the prompt.");
      }
      auto audio_file = std::move(audio_files.front());
      audio_files.pop_front();
      ASSIGN_OR_RETURN(auto preprocessed_audio,
                       audio_preprocessor_->Preprocess(InputAudio(std::string(
                           static_cast<const char*>(audio_file->data()),
                           audio_file->length()))));
      audio_preprocessor_->Reset();
      input_data.emplace_back(InputAudio(std::move(preprocessed_audio)));
      input_data.emplace_back(InputAudioEnd());
      input_data.emplace_back(InputText("\n\n"));
    }
  }
  if (!image_files.empty()) {
    return absl::InvalidArgumentError(
        "Provided more images than expected in the prompt.");
  }
  if (!audio_files.empty()) {
    return absl::InvalidArgumentError(
        "Provided more audio than expected in the prompt.");
  }
  // Add the remaining text in the prompt.
  if (!prompt_view.empty()) {
    input_data.push_back(InputText(std::string(prompt_view)));
  }
  return input_data;
}

absl::StatusOr<Message> Gemma3DataProcessor::ToMessageImpl(
    const Responses& responses,
    const Gemma3DataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  ordered_json message = {{"role", "assistant"}};
  if (preface_.has_value() && std::holds_alternative<JsonPreface>(*preface_) &&
      !std::get<JsonPreface>(*preface_).tools.empty()) {
    ASSIGN_OR_RETURN(
        ordered_json content_and_tool_calls,
        ParseTextAndToolCalls(
            response_text, config_.code_fence_start, config_.code_fence_end,
            GetSyntaxType(config_.syntax_type), config_.escape_fence_strings,
            config_.tool_code_regex));
    if (content_and_tool_calls.contains("content")) {
      message["content"] = content_and_tool_calls["content"];
    }
    if (content_and_tool_calls.contains("tool_calls")) {
      message["tool_calls"] = content_and_tool_calls["tool_calls"];
    }
  } else {
    message["content"] = ordered_json::array(
        {{{"type", "text"}, {"text", std::string(response_text)}}});
  }
  return message;
}

absl::StatusOr<ordered_json> Gemma3DataProcessor::FormatTools(
    const ordered_json& tools) const {
  if (!tools.is_array()) {
    return absl::InvalidArgumentError("Tools must be an array.");
  }
  ordered_json formatted_tools = ordered_json::array();
  for (const auto& tool : tools) {
    ASSIGN_OR_RETURN(std::string formatted_tool, FormatToolAsPython(tool));
    formatted_tools.push_back(formatted_tool);
  }
  return formatted_tools;
}

absl::StatusOr<std::unique_ptr<Constraint>>
Gemma3DataProcessor::CreateConstraint(
    const nlohmann::ordered_json& tools) const {
  return nullptr;
}

absl::string_view Gemma3DataProcessor::CodeFenceStart() const {
  return config_.code_fence_start;
}

absl::string_view Gemma3DataProcessor::CodeFenceEnd() const {
  return config_.code_fence_end;
}

}  // namespace litert::lm
