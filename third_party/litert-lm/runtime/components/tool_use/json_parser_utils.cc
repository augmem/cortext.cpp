// Copyright 2025 The Google AI Edge Authors.
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

#include "runtime/components/tool_use/json_parser_utils.h"

#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/parser_common.h"
#include "runtime/components/tool_use/rust/parsers.rs.h"

namespace litert::lm {

absl::StatusOr<nlohmann::ordered_json> ParseJsonExpression(
    absl::string_view text) {
  auto tool_calls = parse_json_expression(text.data());
  if (!tool_calls.is_ok) {
    absl::string_view error_message =
        absl::string_view(tool_calls.error.data(), tool_calls.error.size());
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse JSON tool calls: ", error_message));
  }
  nlohmann::ordered_json tool_calls_json = nlohmann::ordered_json::array();
  for (const auto& tool_call : tool_calls.tool_calls) {
    tool_calls_json.push_back(ConvertJsonValue(tool_call));
  }
  return tool_calls_json;
}

}  // namespace litert::lm
