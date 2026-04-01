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

#include "runtime/util/litert_lm_loader.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert::lm {

namespace {

using ::testing::status::StatusIs;

TEST(LitertLmLoaderTest, GetSectionLocationNotFound) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> mapped_file,
                       MemoryMappedFile::Create(model_path.string()));
  LitertLmLoader loader(std::move(mapped_file));

  BufferKey embedder_key(schema::AnySectionDataType_TFLiteModel,
                         ModelType::kTfLiteEmbedder);
  EXPECT_THAT(loader.GetSectionLocation(embedder_key),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(LitertLmLoaderTest, InitializeWithSentencePieceFile) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  EXPECT_TRUE(model_file.ok());
  LitertLmLoader loader(std::move(model_file.value()));
  EXPECT_FALSE(loader.GetHuggingFaceTokenizer());
  EXPECT_GT(loader.GetSentencePieceTokenizer()->Size(), 0);
  EXPECT_GT(loader.GetTFLiteModel(ModelType::kTfLitePrefillDecode).Size(), 0);
  EXPECT_GT(loader.GetLlmMetadata().Size(), 0);
  // Try to get non-existent TFLite model.
  EXPECT_EQ(loader.GetTFLiteModel(ModelType::kTfLiteEmbedder).Size(), 0);
}

TEST(LitertLmLoaderTest, InitializeWithHuggingFaceFile) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_hf_tokenizer.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  LitertLmLoader loader(std::move(model_file.value()));
  ASSERT_GT(loader.GetHuggingFaceTokenizer()->Size(), 0);
  ASSERT_FALSE(loader.GetSentencePieceTokenizer());
}

TEST(LitertLmLoaderTest, InitializeWithMemoryMappedFile) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> mapped_file,
                       MemoryMappedFile::Create(model_path.string()));
  LitertLmLoader loader(std::move(mapped_file));
  EXPECT_FALSE(loader.GetHuggingFaceTokenizer());
  EXPECT_GT(loader.GetSentencePieceTokenizer()->Size(), 0);
  EXPECT_GT(loader.GetTFLiteModel(ModelType::kTfLitePrefillDecode).Size(), 0);
  EXPECT_GT(loader.GetLlmMetadata().Size(), 0);
  EXPECT_EQ(loader.GetTFLiteModel(ModelType::kTfLiteEmbedder).Size(), 0);
}

TEST(LitertLmLoaderTest, GetSectionLocationSizeMatch) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryMappedFile> mapped_file,
                       MemoryMappedFile::Create(model_path.string()));
  LitertLmLoader loader(std::move(mapped_file));

  BufferKey sp_key(schema::AnySectionDataType_SP_Tokenizer);
  ASSERT_OK_AND_ASSIGN(auto sp_location, loader.GetSectionLocation(sp_key));
  EXPECT_EQ(sp_location.second - sp_location.first,
            loader.GetSentencePieceTokenizer()->Size());

  BufferKey model_key(schema::AnySectionDataType_TFLiteModel,
                      ModelType::kTfLitePrefillDecode);
  ASSERT_OK_AND_ASSIGN(auto model_location,
                       loader.GetSectionLocation(model_key));
  EXPECT_EQ(model_location.second - model_location.first,
            loader.GetTFLiteModel(ModelType::kTfLitePrefillDecode).Size());

  BufferKey metadata_key(schema::AnySectionDataType_LlmMetadataProto);
  ASSERT_OK_AND_ASSIGN(auto metadata_location,
                       loader.GetSectionLocation(metadata_key));
  EXPECT_EQ(metadata_location.second - metadata_location.first,
            loader.GetLlmMetadata().Size());
}

}  // namespace
}  // namespace litert::lm
