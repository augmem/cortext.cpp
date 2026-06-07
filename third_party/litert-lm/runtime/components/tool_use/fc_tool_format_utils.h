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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_FC_TOOL_FORMAT_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_FC_TOOL_FORMAT_UTILS_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json

namespace litert::lm {

// Formats a JSON value in the FC format.
//
// The FC format is similar to JSON, but:
// - Keys are not quoted.
// - Strings are wrapped by <escape> tags.
//
// Example input:
// ```json
// {
//   "string_value": "foo",
//   "number_value": 123,
//   "boolean_value": true,
//   "null_value": null,
//   "object": {"key": "value"},
//   "array": [4, 5, 6]
// }
// ```
//
// Example output (actual output has no whitespace outside of escaped strings):
// ```
// {
//   string_value: <escape>foo<escape>,
//   number_value: 123,
//   boolean_value": true,
//   null_value": null,
//   object": {
//     key: <escape>value<escape>
//   },
//   array: [4, 5, 6]
// }
// ```
absl::StatusOr<std::string> FormatValueAsFc(
    const nlohmann::ordered_json& value,
    absl::string_view escape_tag = "<escape>");

// Formats a JSON tool declaration in the FC format.
//
// Example input:
// ```json
// {
//   "name": "tool_name",
//   "description": "This is a test tool.",
//   "parameters": {
//     "properties": {
//       "test_param_1": {
//         "type": "string",
//         "description": "First parameter."
//       },
//       "test_param_2": {
//         "type": "array",
//         "items": {
//           "type": "integer"
//         },
//         "description": "Second parameter."
//       },
//       "test_param_3": {
//         "type": "object",
//         "properties": {
//           "field_1": {
//             "type": "string"
//           }
//         },
//         "description": "Third parameter."
//       }
//     },
//     "required": ["test_param_1", "test_param_2"]
//   }
// }
// ```
//
// Example output (actual output has no whitespace outside of escaped strings):
// ```
// declaration:tool_name{
//   description: <escape>This is a test tool.<escape>,
//   parameters: {
//     properties: {
//       test_param_1: {
//         type: <escape>string<escape>,
//         description: <escape>First parameter.<escape>
//       },
//       test_param_2: {
//         type: <escape>array<escape>,
//         items: {
//           type: <escape>integer<escape>
//         },
//         description: <escape>Second parameter.<escape>
//       },
//       test_param_3: {
//         type: <escape>object<escape>,
//         properties: {
//           field_1: {
//             type: <escape>string<escape>
//           }
//         },
//         description: <escape>Third parameter.<escape>
//       }
//     },
//     required: [<escape>test_param_1<escape>, <escape>test_param_2<escape>]
//   }
// }
// ```
absl::StatusOr<std::string> FormatToolAsFc(
    const nlohmann::ordered_json& tool,
    absl::string_view escape_tag = "<escape>");

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_FC_TOOL_FORMAT_UTILS_H_
