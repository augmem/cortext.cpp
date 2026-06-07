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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_FC_PARSER_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_FC_PARSER_UTILS_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json

namespace litert::lm {

// Parses a string containing tool calls in the FC format.
//
// Example input:
// ```
// call:tool_name{param_1:7,param_2:<escape>foo<escape>}
// ```
//
// Example output:
// ```json
// [{
//   "name": "tool_name",
//   "arguments": {
//     "param_1": 7,
//     "param_2": "foo"
//   }
// }]
// ```
absl::StatusOr<nlohmann::ordered_json> ParseFcExpression(
    absl::string_view text);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_FC_PARSER_UTILS_H_
