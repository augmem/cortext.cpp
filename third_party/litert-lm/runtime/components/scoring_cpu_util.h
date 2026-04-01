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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SCORING_CPU_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SCORING_CPU_UTIL_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {
// Calculates the confidence of the batch given the logits and the sampled ids.
// Summing the confidence of all batches will give the total perplexity.
// The logits are expected to be in the shape of [batch_size, vocab_size].
// The sampled_ids are the sampled token ids for the full batch.
// The temperature is used for calculating the softmax function.
// Returns the confidence i.e. negative log probability for the entire batch.
// Ranges from [0, inf)
// In case one of the streams has ended, the user needs to still provide a valid
// sampled id for that stream and ignore the result of that element.
absl::StatusOr<std::vector<float>> ComputeLogLikelihood(
    absl::Span<const float> logits, absl::Span<const int> sampled_ids,
    float temperature);
}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SCORING_CPU_UTIL_H_
