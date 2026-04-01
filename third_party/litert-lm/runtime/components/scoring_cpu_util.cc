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

#include "runtime/components/scoring_cpu_util.h"

#include <cmath>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/sampling_cpu_util.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {

absl::StatusOr<std::vector<float>> ComputeLogLikelihood(
    absl::Span<const float> logits, absl::Span<const int> sampled_ids,
    float temperature) {
  const int batch_size = sampled_ids.size();
  const int vocab_size = logits.size() / batch_size;
  for (int i = 0; i < batch_size; ++i) {
    if (sampled_ids[i] < 0 || sampled_ids[i] >= vocab_size) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid sampled id: ", sampled_ids[i]));
    }
  }
  // Get all indices and their probabilities for calculating perplexity.
  ASSIGN_OR_RETURN(auto all_token_ids,
                   TopKTokenIds(logits, vocab_size, batch_size));
  std::vector<float> all_logit_values;
  ASSIGN_OR_RETURN(auto all_probabilities,
                   Softmax(logits, all_token_ids, temperature, batch_size,
                           all_logit_values));
  std::vector<float> batch_confidence(batch_size);
  for (int b = 0; b < batch_size; ++b) {
    if (sampled_ids[b] >= 0 && sampled_ids[b] < vocab_size) {
      const int sampled_index = b * vocab_size + sampled_ids[b];
      batch_confidence[b] = std::log(all_probabilities[sampled_index]);
    }
  }
  return batch_confidence;
}

}  // namespace litert::lm
