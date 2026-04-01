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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// A sampler that samples token ids from logits.
class Sampler {
 public:
  virtual ~Sampler() = default;

  // Given a batch of logits, samples a batch of token ids.
  // The expected shape of the logits is [batch_size, vocab_size].
  // The output is a 1D litert::TensorBuffer of shape [batch_size].
  // The scores_tensor is optional. If it is not nullptr, the sampled scores are
  // also written to it (in the same shape as the ids_tensor). The scores are
  // the log of the probability of the sampled token.
  virtual absl::Status SampleToIdAndScoreBuffer(
      const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
      TensorBuffer* scores_tensor) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_H_
