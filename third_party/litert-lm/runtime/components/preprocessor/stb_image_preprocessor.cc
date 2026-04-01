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

#include "runtime/components/preprocessor/stb_image_preprocessor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // from @stb
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"  // from @stb

namespace litert::lm {

absl::StatusOr<InputImage> StbImagePreprocessor::Preprocess(
    const InputImage& input_image, const ImagePreprocessParameter& parameter) {
  if (input_image.IsTensorBuffer()) {
    ASSIGN_OR_RETURN(auto processed_image_tensor,
                     input_image.GetPreprocessedImageTensor());
    LITERT_ASSIGN_OR_RETURN(auto processed_image_tensor_with_reference,
                            processed_image_tensor->Duplicate());
    InputImage processed_image(
        std::move(processed_image_tensor_with_reference));
    return processed_image;
  }

  ASSIGN_OR_RETURN(absl::string_view input_image_bytes,
                   input_image.GetRawImageBytes());

  const Dimensions& target_dimensions = parameter.GetTargetDimensions();

  if (target_dimensions.size() != 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("Target dimensions must be (batch, height, width, "
                     "channels). Got dimensions size: ",
                     target_dimensions.size()));
  }

  const int batch_size = target_dimensions.at(0);
  const int target_height = target_dimensions.at(1);
  const int target_width = target_dimensions.at(2);
  const int target_channels = target_dimensions.at(3);

  int original_width, original_height, original_channels;
  unsigned char* decoded_image = stbi_load_from_memory(
      reinterpret_cast<const stbi_uc*>(input_image_bytes.data()),
      input_image_bytes.size(), &original_width, &original_height,
      &original_channels, target_channels);
  if (decoded_image == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to decode image. Reason: ", stbi_failure_reason()));
  }
  std::unique_ptr<unsigned char[], void (*)(void*)> decoded_image_ptr(
      decoded_image, stbi_image_free);

  std::vector<uint8_t> resized_image(static_cast<size_t>(target_width) *
                                     target_height * target_channels);

  int alpha_channel = -1;
  if (target_channels == 4) {
    alpha_channel = 3;
  } else if (target_channels == 2) {
    alpha_channel = 1;
  }

  if (stbir_resize(decoded_image, original_width, original_height, 0,
                   resized_image.data(), target_width, target_height, 0,
                   static_cast<stbir_pixel_layout>(target_channels),
                   STBIR_TYPE_UINT8_SRGB, STBIR_EDGE_CLAMP,
                   STBIR_FILTER_MITCHELL) == 0) {
    return absl::InternalError("Failed to resize image.");
  }

  const int num_elements =
      batch_size * target_height * target_width * target_channels;
  const size_t buffer_size = num_elements * sizeof(float);

  LITERT_ASSIGN_OR_RETURN(
      auto processed_tensor_buffer,
      ::litert::TensorBuffer::CreateManagedHostMemory(
          MakeRankedTensorType<float>(
              {batch_size, target_height, target_width, target_channels}),
          buffer_size));

  LITERT_ASSIGN_OR_RETURN(
      auto processed_tensor_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          processed_tensor_buffer, ::litert::TensorBuffer::LockMode::kWrite));
  float* float_buffer_ptr =
      reinterpret_cast<float*>(processed_tensor_lock_and_addr.second);

  // Normalize pixel values from [0, 255] to [0.0f, 1.0f] and store them
  // in the float tensor buffer.
  for (size_t i = 0; i < resized_image.size(); ++i) {
    float_buffer_ptr[i] = static_cast<float>(resized_image[i]) / 255.0f;
  }

  InputImage processed_image(std::move(processed_tensor_buffer));

  return processed_image;
}

}  // namespace litert::lm
