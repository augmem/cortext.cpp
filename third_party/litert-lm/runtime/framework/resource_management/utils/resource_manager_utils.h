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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_UTILS_RESOURCE_MANAGER_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_UTILS_RESOURCE_MANAGER_UTILS_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl

namespace litert::lm {

// Removes the matching tokens from the input_ids and updates the time_step.
absl::Status RemoveMatchingTokens(const std::vector<int> &processed_tokens,
                                  std::vector<int> *input_ids, int *time_step);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_UTILS_RESOURCE_MANAGER_UTILS_H_
