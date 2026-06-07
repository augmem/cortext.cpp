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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_METRICS_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_METRICS_UTIL_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/engine/litert_lm_lib.h"
#include "runtime/proto/litert_lm_metrics.pb.h"

namespace litert::lm {

// Translates a C++ LitertLmMetrics struct to its Protobuf equivalent.
absl::StatusOr<proto::LitertLmMetrics> ToProto(const LitertLmMetrics& metrics);

// Translates a vector of C++ LitertLmMetrics structs to a Protobuf list
// equivalent.
absl::StatusOr<proto::LitertLmMetricsList> ToProtoList(
    const std::vector<LitertLmMetrics>& metrics_list);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_METRICS_UTIL_H_
