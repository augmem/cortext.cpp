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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_PRINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_PRINT_H_

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl

namespace litert {

namespace lm {

namespace schema {

// Send info about the LiteRT-LM file to the output stream.
absl::Status ProcessLiteRTLMFile(const std::string& litertlm_file,
                                 std::ostream& output_stream);

}  // end namespace schema
}  // end namespace lm
}  // end namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_PRINT_H_
