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

#include "runtime/components/top_p_cpu_sampler.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampling_cpu_util.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {
namespace {

absl::Status ValidateTensor(const TensorBuffer& tensor, int max_num_dims,
                            int batch_size, const std::string& tensor_name) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor.TensorType());
  auto dims = tensor_type.Layout().Dimensions();
  if (NumSignificantDims(tensor) > max_num_dims) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The output ", tensor_name, " tensor must have <=", max_num_dims,
        " significant dimension, but got ", NumSignificantDims(tensor)));
  }
  if (dims[0] != batch_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("The output ", tensor_name,
                     " tensor must have the same batch size as the input "
                     "logits tensor, but got ",
                     dims[0], " vs ", batch_size));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<TopPSampler>> TopPSampler::Create(
    int k, float p, float temperature, int batch_size, int seed) {
  if (k <= 0) {
    return absl::InvalidArgumentError("k must be positive.");
  }
  if (p < 0.0f || p > 1.0f) {
    return absl::InvalidArgumentError("p must be in [0, 1].");
  }
  if (batch_size <= 0) {
    return absl::InvalidArgumentError("batch_size must be positive.");
  }
  if (temperature < 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Temperature must be >= 0, but got ", temperature));
  }
  return absl::WrapUnique(new TopPSampler(k, p, temperature, batch_size, seed));
}

absl::Status TopPSampler::SampleToIdAndScoreBuffer(
    const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
    TensorBuffer* scores_tensor) {
  auto status = ValidateTensor(logits_tensor, /*max_num_dims=*/2, batch_size_,
                               "input logits");
  if (!status.ok()) {
    return status;
  }
  status =
      ValidateTensor(ids_tensor, /*max_num_dims=*/1, batch_size_, "output ids");
  if (!status.ok()) {
    return status;
  }

  auto logits_data_or = ReferTensorBufferAsSpan<float>(logits_tensor);
  absl::Span<float> logits_data;
  if (!logits_data_or) {  // Download the data if it is not in host memory.
    LITERT_ASSIGN_OR_RETURN(auto logits_size, logits_tensor.PackedSize());
    if (logits_data_.size() != logits_size / sizeof(float)) {
      logits_data_ = std::vector<float>(logits_size / sizeof(float));
    }
    TensorBuffer& mutable_logits_tensor =
        const_cast<TensorBuffer&>(logits_tensor);
    mutable_logits_tensor.Read(absl::MakeSpan(logits_data_));
    logits_data = absl::MakeSpan(logits_data_);
  } else {
    logits_data = logits_data_or.Value();
  }
  std::vector<float> sampled_scores;
  auto sampled_ids = TopKTopPSampling(logits_data, k_, p_, temperature_,
                                      generator_, batch_size_, sampled_scores);
  if (!sampled_ids.ok()) {
    return sampled_ids.status();
  }
  ids_tensor.Write(absl::MakeConstSpan(*sampled_ids));
  if (scores_tensor != nullptr) {
    status = ValidateTensor(*scores_tensor, /*max_num_dims=*/1, batch_size_,
                            "output scores");
    if (!status.ok()) {
      return status;
    }
    std::vector<float> scores(batch_size_);
    for (int i = 0; i < batch_size_; ++i) {
      // The scores are the log of the probability of the sampled token.
      scores[i] = std::log(sampled_scores[i]);
    }
    scores_tensor->Write(absl::MakeConstSpan(scores));
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
