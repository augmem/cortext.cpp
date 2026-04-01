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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_IMAGE_PREPROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_IMAGE_PREPROCESSOR_H_

#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {

class ImagePreprocessParameter {
 public:
  // Gets the target dimensions for preprocessing.
  const Dimensions& GetTargetDimensions() const { return dimensions_; }

  // Sets the target dimensions for preprocessing.
  void SetTargetDimensions(const Dimensions& dimensions) {
    dimensions_ = dimensions;
  }

 private:
  Dimensions dimensions_;
};

// Preprocessor for image.
// Main purpose is to process raw image bytes into a resized image TensorBuffer.
class ImagePreprocessor {
 public:
  virtual ~ImagePreprocessor() = default;

  // Preprocesses the raw image bytes into a resized image TensorBuffer.
  // Input is a string_view of the raw image bytes.
  // Output is a TensorBuffer of the resized RGB image with target dimensions.
  virtual absl::StatusOr<InputImage> Preprocess(
      const InputImage& input_image,
      const ImagePreprocessParameter& parameter) {
    if (input_image.IsTensorBuffer()) {
      ASSIGN_OR_RETURN(auto processed_image_tensor,
                       input_image.GetPreprocessedImageTensor());
      LITERT_ASSIGN_OR_RETURN(auto processed_image_tensor_with_reference,
                              processed_image_tensor->Duplicate());
      InputImage processed_image(
          std::move(processed_image_tensor_with_reference));
      return processed_image;
    }
    return absl::UnimplementedError("Image preprocessor is not implemented.");
  };
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_IMAGE_PREPROCESSOR_H_
