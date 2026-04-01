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

#include "runtime/framework/resource_management/utils/resource_manager_utils.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {

absl::Status RemoveMatchingTokens(const std::vector<int> &processed_tokens,
                                  std::vector<int> *input_ids, int *time_step) {
  RET_CHECK_NE(input_ids, nullptr) << "input_ids is null.";
  RET_CHECK_NE(time_step, nullptr) << "time_step is null.";
  RET_CHECK_GE(*time_step, 0) << "Time step is negative.";
  RET_CHECK_GE(processed_tokens.size(), *time_step)
      << "The processed tokens size is smaller than the time step.";
  // Determine the number of tokens available in processed_tokens from the
  // current time_step. *time_step is int, .size() is size_t. RET_CHECKs ensure
  // *time_step is valid.
  const size_t processed_tokens_comparable_len =
      processed_tokens.size() - static_cast<size_t>(*time_step);
  // Determine how many elements to actually compare (the minimum of the two
  // effective sequence lengths).
  const size_t comparison_len =
      std::min(input_ids->size(), processed_tokens_comparable_len);
  // Find the first mismatch.
  // The first range for mismatch is [input_ids->begin(), input_ids->begin() +
  // comparison_len). The second range for mismatch is [processed_tokens.begin()
  // + *time_step, processed_tokens.begin() + *time_step + comparison_len).
  auto mismatch_it =
      std::mismatch(input_ids->begin(), input_ids->begin() + comparison_len,
                    processed_tokens.begin() + *time_step);
  // Calculate the number of matching tokens.
  // mismatch_it.first is an iterator to the first differing element in
  // input_ids, or input_ids->begin() + comparison_len if all compared elements
  // match.
  int matching_tokens = std::distance(input_ids->begin(), mismatch_it.first);
  // Update the input_ids and time_step.
  input_ids->erase(input_ids->begin(), mismatch_it.first);
  *time_step += matching_tokens;
  return absl::OkStatus();
};

}  // namespace litert::lm
