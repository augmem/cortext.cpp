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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PYTHON_TOOL_FORMAT_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PYTHON_TOOL_FORMAT_UTILS_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {

// Formats a JSON value as a Python value.
//
// - Numbers are formatted as Python numbers.
// - Strings are formatted as Python strings.
// - Booleans are converted to "True" and "False".
// - Arrays are converted to Python lists.
// - Objects without the "type" key are converted to Python dictionaries.
// - Objects with the "type" key are converted to Python constructor calls.
// - Null values are converted to "None".
absl::StatusOr<std::string> FormatValueAsPython(
    const nlohmann::ordered_json& value);

// Formats a JSON tool declaration as Python function signature.
//
// Example:
//
// Input - JSON tool declaration:
// ```json
// {
//   "name": "test_tool",
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
// Output - Python function signature:
// ```python
// def test_tool(
//     test_param_1: str,
//     test_param_2: list[int],
//     test_param_3: dict[str, float] | None = None,
// ) -> dict:
//   """This is a test tool.
//
//   Args:
//     test_param_1: First parameter.
//     test_param_2: Second parameter.
//     test_param_3: Third parameter.
//   """
absl::StatusOr<std::string> FormatToolAsPython(
    const nlohmann::ordered_json& tool);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PYTHON_TOOL_FORMAT_UTILS_H_
