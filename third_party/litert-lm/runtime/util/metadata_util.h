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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_METADATA_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_METADATA_UTIL_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"

namespace litert::lm {

// Extracts the LlmMetadata from the string_view. If the string_view does not
// contain the LlmMetadata, it will try to parse to LlmParameters and convert to
// LlmMetadata, for the legacy models.
// Returns the LlmMetadata if successful, otherwise returns the error status.
absl::StatusOr<proto::LlmMetadata> ExtractOrConvertLlmMetadata(
    absl::string_view string_view);
}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_METADATA_UTIL_H_
