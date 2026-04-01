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

#include "runtime/components/sampling_cpu_util.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "absl/random/random.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

absl::StatusOr<std::vector<int>> TopKTokenIds(absl::Span<const float> logits,
                                              int k, int batch_size) {
  if (logits.size() % batch_size != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Logits vector size must be a multiple of batch "
                        "size. But got %d and "
                        "%d.",
                        logits.size(), batch_size));
  }
  const int vocab_size = logits.size() / batch_size;
  std::vector<int> output_indices(batch_size * k);
  if (k == 1) {  // Greedy sampling. Use std::max_element to be more efficient.
    for (int b = 0; b < batch_size; ++b) {
      auto max_iterator =
          std::max_element(logits.begin() + b * vocab_size,
                           logits.begin() + (b + 1) * vocab_size);
      // Gets the index of the max element in the logits of each batch.
      const int max_logit_idx =
          std::distance(logits.begin() + b * vocab_size, max_iterator);
      output_indices[b] = max_logit_idx;
    }
  } else {
    std::vector<int> indices(vocab_size);
    for (int b = 0; b < batch_size; ++b) {
      // Fill with 0, 1, 2,...
      std::iota(indices.begin(), indices.end(), 0);

      // Define the comparator for descending probability
      auto desc_prob_comp = [&logits, vocab_size, b](int i1, int i2) {
        return logits[b * vocab_size + i1] > logits[b * vocab_size + i2];
      };
      // Partition Top-K.
      // O(N) average time complexity.
      // Rearranges 'indices' such that the k elements with the highest
      // probabilities are in the range [indices.begin(), indices.begin() +
      // k). The element at indices[k] is not necessarily the (k+1)th largest.
      std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                       desc_prob_comp);
      std::copy(indices.begin(), indices.begin() + k,
                output_indices.begin() + b * k);
    }
  }
  return output_indices;
}

absl::StatusOr<std::vector<float>> Softmax(
    absl::Span<const float> logits, absl::Span<const int> topk_token_ids,
    float temperature, int batch_size, std::vector<float>& max_logit_values) {
  if (logits.empty()) {
    return absl::InvalidArgumentError("Logits vector cannot be empty.");
  }
  if (logits.size() % batch_size != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Logits vector size must be a multiple of batch size. But got %d and "
        "%d.",
        logits.size(), batch_size));
  }
  if (temperature < 0.0) {
    // A very small positive temperature can mimic greedy sampling.
    // If temp = 0, we will clamp it to epsilon later.
    return absl::InvalidArgumentError(
        absl::StrCat("Temperature must be >= 0, but got ", temperature));
  }
  const int vocab_size = logits.size() / batch_size;
  const int k = topk_token_ids.size() / batch_size;
  std::vector<float> probabilities(topk_token_ids.size());
  max_logit_values.resize(batch_size);
  for (size_t b = 0; b < batch_size; ++b) {
    // Define the comparator for ascending logit to be used with
    // std::max_element. The comparator takes two indices from topk_token_ids
    // and compares the corresponding logit values.
    auto logit_comp = [&logits, vocab_size, b](int topk_idx1, int topk_idx2) {
      return logits[b * vocab_size + topk_idx1] <
             logits[b * vocab_size + topk_idx2];
    };

    // Use std::max_element to find the index in topk_token_ids that corresponds
    // to the maximum logit value.
    auto max_iterator =
        std::max_element(topk_token_ids.begin() + b * k,
                         topk_token_ids.begin() + (b + 1) * k, logit_comp);
    const int max_logit_idx =
        std::distance(topk_token_ids.begin() + b * k, max_iterator);
    max_logit_values[b] =
        logits[b * vocab_size + topk_token_ids[b * k + max_logit_idx]];

    float sum_of_exps = 0.0;
    float current_temp =
        std::max(temperature, std::numeric_limits<float>::epsilon());
    for (size_t i = b * k; i < (b + 1) * k; ++i) {
      probabilities[i] = std::exp(
          (logits[b * vocab_size + topk_token_ids[i]] - max_logit_values[b]) /
          current_temp);
      sum_of_exps += probabilities[i];
    }

    if (sum_of_exps <= std::numeric_limits<float>::epsilon()) {
      // Handle potential zero sum (uniform distribution fallback)
      float uniform_prob = 1.0 / static_cast<float>(k);
      std::fill(probabilities.begin() + b * k,
                probabilities.begin() + (b + 1) * k, uniform_prob);
    } else if (std::isinf(sum_of_exps)) {
      // Handle inf sum which is caused by very small temperature.
      // This is to avoid inf sum in the softmax calculation.
      std::fill(probabilities.begin() + b * k,
                probabilities.begin() + (b + 1) * k, 0.0f);
      probabilities[b * k + max_logit_idx] = 1.0f;
    } else {
      // Normalize probabilities
      float inv_sum =
          1.0 / sum_of_exps;  // Calculate inverse once for slight speedup
      for (size_t i = b * k; i < (b + 1) * k; ++i) {
        probabilities[i] *= inv_sum;
      }
    }
  }
  return probabilities;
};

absl::StatusOr<std::vector<int>> TopKTopPSampling(
    absl::Span<const float> logits, int k, float p, float temperature,
    absl::BitGen& rng, int batch_size, std::vector<float>& sampled_scores) {
  if (logits.empty()) {
    return absl::InvalidArgumentError("Logits vector cannot be empty.");
  }
  if (logits.size() % batch_size != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Logits vector size must be a multiple of batch "
                        "size. But got %d and "
                        "%d.",
                        logits.size(), batch_size));
  }
  if (k <= 0) {
    return absl::InvalidArgumentError("k must be greater than 0.");
  }
  if (p < 0.0 || p > 1.0) {
    return absl::InvalidArgumentError("p must be in the range [0.0, 1.0].");
  }
  const int vocab_size = logits.size() / batch_size;
  // Ensure k is not larger than the number of probabilities
  k = std::min(k, vocab_size);

  auto topk_token_ids = TopKTokenIds(logits, k, batch_size);
  if (!topk_token_ids.ok()) return topk_token_ids.status();

  std::vector<float> max_logit_values;
  auto probabilities = Softmax(logits, *topk_token_ids, temperature, batch_size,
                               max_logit_values);
  if (!probabilities.ok()) return probabilities.status();

  std::vector<int> sampled_ids;
  if (k == 1) {  // Greedy sampling. Return the topk_token_ids directly.
    sampled_ids.assign(topk_token_ids->begin(), topk_token_ids->end());
    sampled_scores = std::vector<float>(batch_size, 1.0f);
    return sampled_ids;
  }
  sampled_ids.resize(batch_size);
  sampled_scores.resize(batch_size);
  float current_temp =
      std::max(temperature, std::numeric_limits<float>::epsilon());
  for (int b = 0; b < batch_size; ++b) {
    // Define the comparator for descending probability
    auto desc_prob_comp = [&probabilities, b, k](int i1, int i2) {
      return (*probabilities)[b * k + i1] > (*probabilities)[b * k + i2];
    };

    // Sort Only the Top-K.
    // O(k log k) time complexity.
    // Sorts the indices of the top k elements based on their probabilities.
    // - index_of_topk[i] stores the offset within the current batch's top-k
    // range.
    // - probabilities[b*k + index_of_topk[i]] is the probability of the i-th
    // largest element.
    // - topk_token_ids[b*k + index_of_topk[i]] is the actual token ID of the
    // i-th largest element.
    std::vector<int> index_of_topk(k);
    std::iota(index_of_topk.begin(), index_of_topk.end(), 0);
    std::sort(index_of_topk.begin(), index_of_topk.end(), desc_prob_comp);

    // Determine Top-P Cutoff Index within Top-K.
    // O(k) time complexity.
    double cumulative_prob = 0.0;
    int final_sample_size = 0;  // Actual number of elements to sample from

    for (int i = 0; i < k; ++i) {
      // Check if adding this probability would exceed the threshold p. It
      // stops when cumulative_prob >= p.
      cumulative_prob += (*probabilities)[b * k + index_of_topk[i]];
      final_sample_size = i + 1;  // Include this element

      if (cumulative_prob >= p) {
        break;  // Found the smallest set within Top-K satisfying Top-P
      }
    }
    // final_sample_size now holds min(p_cutoff_within_top_k, k)

    // Handle Edge Case: Cumulative Probability is Zero.
    if (cumulative_prob <= std::numeric_limits<double>::epsilon()) {
      // Fallback: Return the index with the absolute highest probability
      // (indices[0] after sorting top-k).
      sampled_ids[b] = (*topk_token_ids)[b * k + index_of_topk[0]];
      sampled_scores[b] = std::exp(
          (logits[b * vocab_size + sampled_ids[b]] - max_logit_values[b]) /
          current_temp);
      continue;
    }

    // O(final_sample_size) which is O(k) time complexity.
    std::uniform_real_distribution<double> dist(0.0, cumulative_prob);
    double random_sample = dist(rng);
    double current_cumulative = 0.0;
    for (int i = 0; i < final_sample_size; ++i) {
      current_cumulative += (*probabilities)[b * k + index_of_topk[i]];
      if (random_sample <= current_cumulative) {
        sampled_ids[b] = (*topk_token_ids)[b * k + index_of_topk[i]];
        sampled_scores[b] = (*probabilities)[b * k + index_of_topk[i]];
        break;
      }
    }
  }
  return sampled_ids;
}

}  // namespace litert::lm
