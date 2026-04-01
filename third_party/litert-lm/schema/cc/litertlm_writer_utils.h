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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CC_LITERTLM_WRITER_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CC_LITERTLM_WRITER_UTILS_H_
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl

namespace litert::lm::schema {

absl::Status LitertLmWrite(const std::vector<std::string>& command_args,
                           const std::string& section_metadata_str,
                           const std::string& output_path);

}  // namespace litert::lm::schema
#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CC_LITERTLM_WRITER_UTILS_H_
