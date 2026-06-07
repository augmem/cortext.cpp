// Copyright 2026 The ODML Authors.
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

#include "runtime/conversation/prompt_utils.h"

#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::Status FillPrefaceForPromptTemplateInput(
    const Preface& preface, const ModelDataProcessor* model_data_processor,
    PromptTemplateInput& tmpl_input) {
  if (std::holds_alternative<JsonPreface>(preface)) {
    auto json_preface = std::get<JsonPreface>(preface);

    if (json_preface.messages.is_array()) {
      for (auto& message : json_preface.messages) {
        ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                         model_data_processor->MessageToTemplateInput(message));
        tmpl_input.messages.push_back(message_tmpl_input);
      }
    }

    if (json_preface.tools.is_null()) {
      tmpl_input.tools = nullptr;
    } else {
      ASSIGN_OR_RETURN(tmpl_input.tools,
                       model_data_processor->FormatTools(json_preface.tools));
    }
    tmpl_input.extra_context = json_preface.extra_context;
  } else {
    return absl::UnimplementedError("Preface type is not supported yet");
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
