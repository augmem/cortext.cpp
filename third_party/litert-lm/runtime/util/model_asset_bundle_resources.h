/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MODEL_ASSET_BUNDLE_RESOURCES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MODEL_ASSET_BUNDLE_RESOURCES_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/zip_utils.h"

namespace litert::lm {

// The mediapipe task model asset bundle resources class.
// A ModelAssetBundleResources object contains model asset bundle related
// resources and the method to extract the tflite models, resource files or
// model asset bundles for the mediapipe sub-tasks. As the resources are owned
// by the ModelAssetBundleResources object, callers must keep
// ModelAssetBundleResources alive while using any of the resources.
class ModelAssetBundleResources {
 public:
  // Takes a reference to the provided model asset bundle file and creates
  // ModelAssetBundleResources from its contents. A non-empty tag
  // must be set if the ModelAssetBundleResources will be used through
  // ModelResourcesCacheService.
  static absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>> Create(
      const std::string& tag,
      std::shared_ptr<ScopedFile> model_asset_bundle_file);

  // Takes a reference to the provided model asset bundle file and creates
  // ModelAssetBundleResources from its contents. A non-empty tag
  // must be set if the ModelAssetBundleResources will be used through
  // ModelResourcesCacheService.
  static absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>> Create(
      const std::string& tag,
      std::shared_ptr<MemoryMappedFile> model_asset_bundle_file);

  // Convenience method to create from a ScopedFile directly.
  static absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>> Create(
      const std::string& tag, ScopedFile&& model_asset_bundle_file);

  // ModelResources is neither copyable nor movable.
  ModelAssetBundleResources(const ModelAssetBundleResources&) = delete;
  ModelAssetBundleResources& operator=(const ModelAssetBundleResources&) =
      delete;

  // Subclasses should override this to ensure that `files_` is cleared before
  // the memory it points to is destroyed.
  virtual ~ModelAssetBundleResources() = default;

  // Returns the model asset bundle resources tag.
  std::string GetTag() const { return tag_; }

  // Gets the contents of the model file (either tflite model file, resource
  // file or model bundle file) with the provided name. An error is returned if
  // there is no such model file.
  absl::StatusOr<absl::string_view> GetFile(absl::string_view filename) const;

  // Lists all the file names in the model asset model.
  std::vector<absl::string_view> ListFiles() const;

 private:
  ModelAssetBundleResources(
      std::string tag,
      std::shared_ptr<MemoryMappedFile> mapped_model_asset_bundle_file,
      absl::flat_hash_map<std::string, OffsetAndSize> files);

  // The model asset bundle resources tag.
  const std::string tag_;

  // This owns the memory backing `files_`.
  const std::shared_ptr<MemoryMappedFile> mapped_model_asset_bundle_file_;

  // The files bundled in model asset bundle, as a map with the filename
  // (corresponding to a basename, e.g. "hand_detector.tflite") as key and
  // a pointer to the file contents as value. Each file can be either a TFLite
  // model file, resource file or a model bundle file for sub-task.
  const absl::flat_hash_map<std::string, OffsetAndSize> files_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MODEL_ASSET_BUNDLE_RESOURCES_H_
