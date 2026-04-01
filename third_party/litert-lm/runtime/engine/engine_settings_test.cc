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

#include "runtime/engine/engine_settings.h"

#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/optional.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::litert::lm::EngineSettings;
using ::testing::ContainsRegex;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Return;
using ::testing::status::StatusIs;

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (const std::vector<int>& token_ids), (override));
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view token),
              (override));
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
};

proto::LlmMetadata CreateLlmMetadata() {
  proto::LlmMetadata llm_metadata;
  llm_metadata.mutable_start_token()->mutable_token_ids()->add_ids(2);
  llm_metadata.mutable_stop_tokens()->Add()->set_token_str("<eos>");
  llm_metadata.mutable_stop_tokens()->Add()->set_token_str("<end_of_turn>");
  llm_metadata.mutable_stop_tokens()->Add()->set_token_str("<ctrl>");
  llm_metadata.mutable_sampler_params()->set_type(
      proto::SamplerParameters::TOP_P);
  llm_metadata.mutable_sampler_params()->set_k(1);
  llm_metadata.mutable_sampler_params()->set_p(0.95f);
  llm_metadata.mutable_sampler_params()->set_temperature(1.0f);
  llm_metadata.mutable_sampler_params()->set_seed(0);

  llm_metadata.mutable_prompt_templates()->mutable_user()->set_prefix(
      "<start>user");
  llm_metadata.mutable_prompt_templates()->mutable_user()->set_suffix("<end>");
  llm_metadata.mutable_prompt_templates()->mutable_model()->set_prefix(
      "<start>model");
  llm_metadata.mutable_prompt_templates()->mutable_model()->set_suffix("<end>");
  return llm_metadata;
}

TEST(EngineSettingsTest, MainExecutorSettingsGetModelPath) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  EXPECT_OK(settings);

  auto model_path =
      settings->GetMainExecutorSettings().GetModelAssets().GetPath();
  ASSERT_OK(model_path);
  EXPECT_EQ(*model_path, "test_model_path_1");
}

TEST(EngineSettingsTest, MainExecutorSettingsSetAndGetCacheDir) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  EXPECT_OK(settings);
  settings->GetMutableMainExecutorSettings().SetCacheDir("test_cache_dir");
  EXPECT_EQ(settings->GetMainExecutorSettings().GetCacheDir(),
            "test_cache_dir");
}

TEST(EngineSettingsTest, MainExecutorSettingsSetAndGetMaxNumTokens) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);

  auto settings = EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  EXPECT_OK(settings);
  settings->GetMutableMainExecutorSettings().SetMaxNumTokens(128);
  EXPECT_EQ(settings->GetMainExecutorSettings().GetMaxNumTokens(), 128);
}

TEST(EngineSettingsTest, MainExecutorSettingsSetAndGetExecutorBackend) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);

  auto settings = EngineSettings::CreateDefault(*model_assets, Backend::GPU);
  EXPECT_OK(settings);
  EXPECT_OK(
      settings->GetMutableMainExecutorSettings().SetBackend(Backend::GPU));
  EXPECT_THAT(settings->GetMainExecutorSettings().GetBackend(),
              Eq(Backend::GPU));
}

TEST(EngineSettingsTest, MainExecutorSettingsDefaultExecutorBackend) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);
  EXPECT_THAT(settings->GetMainExecutorSettings().GetBackend(),
              Eq(Backend::CPU));
}

TEST(EngineSettingsTest, VisionExecutorSettingsNotSet) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  EXPECT_OK(settings);
  EXPECT_FALSE(settings->GetVisionExecutorSettings().has_value());
}

TEST(EngineSettingsTest, VisionExecutorSettingsSetAndGetBackend) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU, Backend::GPU);
  EXPECT_OK(settings);
  ASSERT_TRUE(settings->GetVisionExecutorSettings().has_value());
  EXPECT_EQ(settings->GetVisionExecutorSettings()->GetBackend(), Backend::GPU);

  EXPECT_OK(
      settings->GetMutableVisionExecutorSettings()->SetBackend(Backend::NPU));
  EXPECT_EQ(settings->GetVisionExecutorSettings()->GetBackend(), Backend::NPU);
}

TEST(EngineSettingsTest, VisionExecutorSettingsSetAndGetCacheDir) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU, Backend::GPU);
  EXPECT_OK(settings);
  ASSERT_TRUE(settings->GetVisionExecutorSettings().has_value());
  settings->GetMutableVisionExecutorSettings()->SetCacheDir("vision_cache_dir");
  EXPECT_EQ(settings->GetVisionExecutorSettings()->GetCacheDir(),
            "vision_cache_dir");
}

TEST(EngineSettingsTest, VisionExecutorSettingsGetModelPath) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU, Backend::GPU);
  EXPECT_OK(settings);
  ASSERT_TRUE(settings->GetVisionExecutorSettings().has_value());
  auto model_path =
      settings->GetVisionExecutorSettings()->GetModelAssets().GetPath();
  ASSERT_OK(model_path);
  EXPECT_EQ(*model_path, "test_model_path_1");
}

TEST(EngineSettingsTest, AudioExecutorSettingsNotSet) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, EngineSettings::CreateDefault(
                                          *model_assets,
                                          /*backend=*/Backend::CPU,
                                          /*vision_backend=*/std::nullopt,
                                          /*audio_backend=*/std::nullopt));
  EXPECT_FALSE(settings.GetAudioExecutorSettings().has_value());
}

TEST(EngineSettingsTest, AudioExecutorSettingsSetAndGetBackend) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, EngineSettings::CreateDefault(
                                          *model_assets,
                                          /*backend=*/Backend::CPU,
                                          /*vision_backend=*/std::nullopt,
                                          /*audio_backend=*/Backend::CPU));
  EXPECT_EQ(settings.GetAudioExecutorSettings()->GetBackend(), Backend::CPU);
}

TEST(EngineSettingsTest, AudioExecutorSettingsSetAndGetCacheDir) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, EngineSettings::CreateDefault(
                                          *model_assets,
                                          /*backend=*/Backend::CPU,
                                          /*vision_backend=*/std::nullopt,
                                          /*audio_backend=*/Backend::CPU));
  ASSERT_TRUE(settings.GetAudioExecutorSettings().has_value());
  settings.GetMutableAudioExecutorSettings()->SetCacheDir("audio_cache_dir");
  EXPECT_EQ(settings.GetAudioExecutorSettings()->GetCacheDir(),
            "audio_cache_dir");
}

TEST(EngineSettingsTest, AudioExecutorSettingsGetModelPath) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(auto settings, EngineSettings::CreateDefault(
                                          *model_assets,
                                          /*backend=*/Backend::CPU,
                                          /*vision_backend=*/std::nullopt,
                                          /*audio_backend=*/Backend::CPU));
  ASSERT_TRUE(settings.GetAudioExecutorSettings().has_value());
  auto model_path =
      settings.GetAudioExecutorSettings()->GetModelAssets().GetPath();
  ASSERT_OK(model_path);
  EXPECT_EQ(*model_path, "test_model_path_1");
}

TEST(EngineSettingsTest, VisionAudioBackendConstraintNoConstraint) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(
      *model_assets, /*backend=*/Backend::CPU, /*vision_backend=*/Backend::GPU,
      /*audio_backend=*/Backend::GPU);
  EXPECT_OK(settings);
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  // No vision backend constraint means it's compatible with all backends, so it
  // should be OK even if the vision / audio models requires a backend.
  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, nullptr, "",
                                             /*text_backend_constraint=*/{},
                                             /*vision_backend_constraint=*/{},
                                             /*audio_backend_constraint=*/{}));
}

TEST(EngineSettingsTest, VisionBackendConstraintMatch) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(
      *model_assets, /*backend=*/Backend::CPU, /*vision_backend=*/Backend::GPU);
  EXPECT_OK(settings);
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  // The vision backend constraint matches the vision model backend, so it
  // should be OK.
  EXPECT_OK(settings->MaybeUpdateAndValidate(
      tokenizer, nullptr, "", /*text_backend_constraint=*/{},
      /*vision_backend_constraint=*/"gpu",
      /*audio_backend_constraint=*/{}));
}

TEST(EngineSettingsTest, VisionBackendConstraintMatchCaseInsensitive) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(
      *model_assets, /*backend=*/Backend::CPU, /*vision_backend=*/Backend::GPU);
  EXPECT_OK(settings);
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  // The vision backend constraint matches (case insensitive) the vision model
  // backend, so it should be OK.
  EXPECT_OK(settings->MaybeUpdateAndValidate(
      tokenizer, nullptr, "", /*text_backend_constraint=*/{},
      /*vision_backend_constraint=*/"GPU",
      /*audio_backend_constraint=*/{}));
}

TEST(EngineSettingsTest, VisionBackendConstraintMultiMatch) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(
      *model_assets, /*backend=*/Backend::CPU, /*vision_backend=*/Backend::NPU);
  EXPECT_OK(settings);
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  // The vision backend constraint matches one of the vision model backends,
  // NPU in this case, so it should be OK.
  EXPECT_OK(settings->MaybeUpdateAndValidate(
      tokenizer, nullptr, "", /*text_backend_constraint=*/{},
      /*vision_backend_constraint=*/"gpu,npu",
      /*audio_backend_constraint=*/{}));
}

TEST(EngineSettingsTest, VisionBackendConstraintNoMatch) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(
      *model_assets, /*backend=*/Backend::CPU, /*vision_backend=*/Backend::GPU);
  EXPECT_OK(settings);
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  // The vision backend constraint does not match the vision model backend, so
  // it should be an error.
  EXPECT_THAT(settings->MaybeUpdateAndValidate(
                  tokenizer, nullptr, "", /*text_backend_constraint=*/{},
                  /*vision_backend_constraint=*/"npu",
                  /*audio_backend_constraint=*/{}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EngineSettingsTest, VisionBackendConstraintMultiNoMatch) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(
      *model_assets, /*backend=*/Backend::CPU, /*vision_backend=*/Backend::GPU);
  EXPECT_OK(settings);
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  // The vision backend constraint does not match any of the vision model
  // backends, so it should be an error.
  EXPECT_THAT(settings->MaybeUpdateAndValidate(
                  tokenizer, nullptr, "", /*text_backend_constraint=*/{},
                  /*vision_backend_constraint=*/"cpu,npu",
                  /*audio_backend_constraint=*/{}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EngineSettingsTest, AudioBackendConstraintNoConstraint) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(
      auto settings,
      EngineSettings::CreateDefault(*model_assets, /*backend=*/Backend::CPU,
                                    /*vision_backend=*/Backend::GPU,
                                    /*audio_backend=*/Backend::GPU));
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  // No audio backend constraint means it's compatible with all backends, so it
  // should be OK even if the audio models requires a backend.
  EXPECT_OK(settings.MaybeUpdateAndValidate(tokenizer, nullptr, "",
                                            /*text_backend_constraint=*/{},
                                            /*vision_backend_constraint=*/{},
                                            /*audio_backend_constraint=*/{}));
}

TEST(EngineSettingsTest, AudioBackendConstraintMatch) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(
      auto settings,
      EngineSettings::CreateDefault(*model_assets, /*backend=*/Backend::CPU,
                                    /*vision_backend=*/{},
                                    /*audio_backend=*/Backend::CPU));
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));

  // The audio backend constraint matches the audio model backend, so it should
  // be OK.
  EXPECT_OK(settings.MaybeUpdateAndValidate(
      tokenizer, nullptr, "", /*text_backend_constraint=*/{},
      /*vision_backend_constraint=*/{}, /*audio_backend_constraint=*/"cpu"));
}

TEST(EngineSettingsTest, AudioBackendConstraintMatchCaseInsensitive) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(
      auto settings,
      EngineSettings::CreateDefault(*model_assets, /*backend=*/Backend::CPU,
                                    /*vision_backend=*/{},
                                    /*audio_backend=*/Backend::CPU));
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  // The audio backend constraint matches (case insensitive) the audio model
  // backend, so it should be OK.
  EXPECT_OK(settings.MaybeUpdateAndValidate(
      tokenizer, nullptr, "", /*text_backend_constraint=*/{},
      /*vision_backend_constraint=*/{}, /*audio_backend_constraint=*/"CPU"));
}

TEST(EngineSettingsTest, AudioBackendConstraintMultiMatch) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(
      auto settings,
      EngineSettings::CreateDefault(*model_assets, /*backend=*/Backend::CPU,
                                    /*vision_backend=*/{},
                                    /*audio_backend=*/Backend::CPU));
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  // The audio backend constraint matches one of the audio model backends, CPU
  // in this case, so it should be OK.
  EXPECT_OK(settings.MaybeUpdateAndValidate(
      tokenizer, nullptr, "", /*text_backend_constraint=*/{},
      /*vision_backend_constraint=*/{},
      /*audio_backend_constraint=*/"gpu,cpu"));
}

TEST(EngineSettingsTest, AudioBackendConstraintNoMatch) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(
      auto settings,
      EngineSettings::CreateDefault(*model_assets, /*backend=*/Backend::CPU,
                                    /*vision_backend=*/{},
                                    /*audio_backend=*/Backend::CPU));
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  // The audio backend constraint does not match the audio model backend, so
  // it should be an error.
  EXPECT_THAT(settings.MaybeUpdateAndValidate(
                  tokenizer, nullptr, "", /*text_backend_constraint=*/{},
                  /*vision_backend_constraint=*/{},
                  /*audio_backend_constraint=*/"npu"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EngineSettingsTest, AudioBackendConstraintMultiNoMatch) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(
      auto settings,
      EngineSettings::CreateDefault(*model_assets, /*backend=*/Backend::CPU,
                                    /*vision_backend=*/{},
                                    /*audio_backend=*/Backend::CPU));
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));

  // The audio backend constraint does not match any of the audio model
  // backends, so it should be an error.
  EXPECT_THAT(settings.MaybeUpdateAndValidate(
                  tokenizer, nullptr, "", /*text_backend_constraint=*/{},
                  /*vision_backend_constraint=*/{},
                  /*audio_backend_constraint=*/"gpu,npu"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EngineSettingsTest, TextBackendConstraintNoMatch) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  ASSERT_OK_AND_ASSIGN(
      auto settings,
      EngineSettings::CreateDefault(*model_assets, /*backend=*/Backend::CPU,
                                    /*vision_backend=*/{},
                                    /*audio_backend=*/{}));
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  // The text backend constraint does not match the text model backend, so
  // it should be an error.
  EXPECT_THAT(settings.MaybeUpdateAndValidate(
                  tokenizer, nullptr, "", /*text_backend_constraint=*/"gpu"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EngineSettingsTest, BenchmarkParams) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);
  EXPECT_FALSE(settings->IsBenchmarkEnabled());

  proto::BenchmarkParams& benchmark_params =
      settings->GetMutableBenchmarkParams();
  benchmark_params.set_num_decode_tokens(100);
  benchmark_params.set_num_prefill_tokens(100);
  EXPECT_TRUE(settings->IsBenchmarkEnabled());
  EXPECT_EQ(settings->GetBenchmarkParams()->num_decode_tokens(), 100);
  EXPECT_EQ(settings->GetBenchmarkParams()->num_prefill_tokens(), 100);
}

TEST(EngineSettingsTest, LlmMetadata) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);
  EXPECT_FALSE(settings->GetLlmMetadata().has_value());

  proto::LlmMetadata& llm_metadata = settings->GetMutableLlmMetadata();
  llm_metadata.mutable_start_token()->set_token_str("test_token_str");
  EXPECT_TRUE(settings->GetLlmMetadata().has_value());
  EXPECT_EQ(settings->GetLlmMetadata().value().start_token().token_str(),
            "test_token_str");
}

absl::Status IsExpectedLlmMetadata(const proto::LlmMetadata& llm_metadata) {
  if (!llm_metadata.has_start_token() ||
      llm_metadata.start_token().token_ids().ids_size() != 1 ||
      llm_metadata.start_token().token_ids().ids(0) != 2) {
    return absl::InvalidArgumentError("Start token is not set correctly.");
  }
  if (llm_metadata.stop_tokens_size() != 3) {
    return absl::InvalidArgumentError("Stop tokens size is not 3.");
  }
  if (llm_metadata.stop_tokens(0).token_ids().ids_size() != 1 ||
      llm_metadata.stop_tokens(0).token_ids().ids(0) != 1) {
    return absl::InvalidArgumentError("Stop tokens 0 is not set correctly.");
  }
  if (llm_metadata.stop_tokens(1).token_ids().ids_size() != 1 ||
      llm_metadata.stop_tokens(1).token_ids().ids(0) != 1) {
    return absl::InvalidArgumentError("Stop tokens 1 is not set correctly.");
  }
  if (llm_metadata.stop_tokens(2).token_ids().ids_size() != 1 ||
      llm_metadata.stop_tokens(2).token_ids().ids(0) != 1) {
    return absl::InvalidArgumentError("Stop tokens 2 is not set correctly.");
  }
  if (!llm_metadata.has_sampler_params() ||
      llm_metadata.sampler_params().type() != proto::SamplerParameters::TOP_P ||
      llm_metadata.sampler_params().k() != 1 ||
      llm_metadata.sampler_params().p() != 0.95f ||
      llm_metadata.sampler_params().temperature() != 1.0f ||
      llm_metadata.sampler_params().seed() != 0) {
    return absl::InvalidArgumentError("Sampler params is not set correctly.");
  }
  if (llm_metadata.llm_model_type().model_type_case() !=
      proto::LlmModelType::kGenericModel) {
    return absl::InvalidArgumentError("LLM model type is not set correctly.");
  }
  return absl::OkStatus();
}

TEST(EngineSettingsTest, MaybeUpdateAndValidate) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();

  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));
  EXPECT_OK(IsExpectedLlmMetadata(settings->GetLlmMetadata().value()));
}

TEST(EngineSettingsTest, MaybeUpdateAndValidateTokenToIdReturnsError) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId)
      .WillRepeatedly(Return(absl::NotFoundError("")));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();

  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));
  EXPECT_OK(IsExpectedLlmMetadata(settings->GetLlmMetadata().value()));
}

TEST(EngineSettingsTest, MaybeUpdateAndValidateNPU) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets, Backend::NPU);
  EXPECT_OK(settings);

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();

  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));
  EXPECT_EQ(settings->GetLlmMetadata().value().sampler_params().type(),
            proto::SamplerParameters::TOP_P);
}

TEST(EngineSettingsTest, PrintOperator) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);
  proto::LlmMetadata& llm_metadata = settings->GetMutableLlmMetadata();
  llm_metadata.mutable_start_token()->set_token_str("test_token_str");
  proto::BenchmarkParams& benchmark_params =
      settings->GetMutableBenchmarkParams();
  benchmark_params.set_num_decode_tokens(100);
  benchmark_params.set_num_prefill_tokens(100);
  std::stringstream oss;
  oss << *settings;
}

TEST(SessionConfigTest, CreateDefault) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_EQ(session_config.GetSamplerParams().type(),
            proto::SamplerParameters::TYPE_UNSPECIFIED);
  EXPECT_EQ(session_config.GetSamplerBackend(), Backend::UNSPECIFIED);
}

TEST(SessionConfigTest, SetAndGetAudioModalityEnabled) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_FALSE(session_config.AudioModalityEnabled());
  session_config.SetAudioModalityEnabled(true);
  EXPECT_TRUE(session_config.AudioModalityEnabled());
}

TEST(SessionConfigTest, SetAndGetVisionModalityEnabled) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_FALSE(session_config.VisionModalityEnabled());
  session_config.SetVisionModalityEnabled(true);
  EXPECT_TRUE(session_config.VisionModalityEnabled());
}

TEST(SessionConfigTest, SetAndGetSamplerParams) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  proto::SamplerParameters& sampler_params =
      session_config.GetMutableSamplerParams();
  sampler_params.set_type(proto::SamplerParameters::TOP_K);
  sampler_params.set_k(10);
  EXPECT_EQ(session_config.GetSamplerParams().type(),
            proto::SamplerParameters::TOP_K);
  EXPECT_EQ(session_config.GetSamplerParams().k(), 10);

  // Mutable sampler params.
  session_config.GetMutableSamplerParams().set_type(
      proto::SamplerParameters::TYPE_UNSPECIFIED);
  EXPECT_EQ(session_config.GetSamplerParams().type(),
            proto::SamplerParameters::TYPE_UNSPECIFIED);
}

TEST(SessionConfigTest, SetAndGetStopTokenIds) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableStopTokenIds() = {{0}, {1, 2}};
  EXPECT_EQ(session_config.GetStopTokenIds().size(), 2);
  EXPECT_THAT(session_config.GetStopTokenIds()[0], ElementsAre(0));
  EXPECT_THAT(session_config.GetStopTokenIds()[1], ElementsAre(1, 2));
}

TEST(SessionConfigTest, SetAndGetNumOutputCandidates) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_EQ(session_config.GetNumOutputCandidates(), 1);
  session_config.SetNumOutputCandidates(2);
  EXPECT_EQ(session_config.GetNumOutputCandidates(), 2);
}

TEST(SessionConfigTest, SetAndGetStartTokenId) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_EQ(session_config.GetStartTokenId(), -1);
  session_config.SetStartTokenId(1);
  EXPECT_EQ(session_config.GetStartTokenId(), 1);
}

TEST(SessionConfigTest, SetAndGetLlmModelType) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_EQ(session_config.GetLlmModelType().model_type_case(),
            proto::LlmModelType::MODEL_TYPE_NOT_SET);
  session_config.GetMutableLlmModelType().mutable_gemma3n();
  EXPECT_EQ(session_config.GetLlmModelType().model_type_case(),
            proto::LlmModelType::kGemma3N);
}

TEST(SessionConfigTest, MaybeUpdateAndValidate) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  auto session_config = SessionConfig::CreateDefault();
  EXPECT_OK(settings);
  // We didn't call MaybeUpdateAndValidate on EngineSettings, so some of the
  // required fields are not set. So the validation should fail.
  EXPECT_THAT(session_config.MaybeUpdateAndValidate(*settings),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();

  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));
  // The validation should pass now.
  EXPECT_OK(session_config.MaybeUpdateAndValidate(*settings));
  EXPECT_EQ(session_config.GetSamplerBackend(), Backend::CPU);
  EXPECT_EQ(session_config.GetLlmModelType().model_type_case(),
            proto::LlmModelType::kGenericModel);
}

TEST(SessionConfigTest, MaybeUpdateAndValidatePickGpuAsSamplerBackend) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(
      settings->GetMutableMainExecutorSettings().SetBackend(Backend::GPU));
  auto session_config = SessionConfig::CreateDefault();
  EXPECT_OK(settings);
  // We didn't call MaybeUpdateAndValidate on EngineSettings, so some of the
  // required fields are not set. So the validation should fail.
  EXPECT_THAT(session_config.MaybeUpdateAndValidate(*settings),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();

  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));
  // The validation should pass now.
  EXPECT_OK(session_config.MaybeUpdateAndValidate(*settings));
  EXPECT_EQ(session_config.GetSamplerBackend(), Backend::GPU);
}

TEST(SessionConfigTest, MaybeUpdateAndValidateMaxNumTokens) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  auto session_config = SessionConfig::CreateDefault();
  EXPECT_OK(settings);
  EXPECT_EQ(settings->GetMainExecutorSettings().GetMaxNumTokens(), 0);

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();

  llm_metadata.set_max_num_tokens(1280);
  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));
  EXPECT_EQ(settings->GetMainExecutorSettings().GetMaxNumTokens(), 1280);

  llm_metadata.set_max_num_tokens(4096);
  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));
  EXPECT_EQ(settings->GetMainExecutorSettings().GetMaxNumTokens(), 1280);
}

TEST(SessionConfigTest,
     MaybeUpdateAndValidateMaxNumTokensPrefillBatchSizeFromShortInputPrompt) {
  constexpr int kNumInputPromptTokens = 1024;
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  auto session_config = SessionConfig::CreateDefault();
  EXPECT_OK(settings);
  EXPECT_EQ(settings->GetMainExecutorSettings().GetMaxNumTokens(), 0);

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>(kNumInputPromptTokens, 1)));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();

  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata, " "));
  const auto& main_settings1 = settings->GetMainExecutorSettings();
  EXPECT_EQ(main_settings1.GetMaxNumTokens(), 4096);
  EXPECT_TRUE(main_settings1.GetAdvancedSettings().has_value());
  EXPECT_EQ(main_settings1.GetAdvancedSettings()->prefill_batch_sizes.size(),
            1);
  EXPECT_EQ(*main_settings1.GetAdvancedSettings()->prefill_batch_sizes.begin(),
            kNumInputPromptTokens + /*margin=*/2);
}

TEST(SessionConfigTest,
     MaybeUpdateAndValidateMaxNumTokensPrefillBatchSizeFromLongInputPrompt) {
  constexpr int kNumInputPromptTokens = 4096 - 100;
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  auto session_config = SessionConfig::CreateDefault();
  EXPECT_OK(settings);
  EXPECT_EQ(settings->GetMainExecutorSettings().GetMaxNumTokens(), 0);

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>(kNumInputPromptTokens, 1)));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();

  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata, " "));
  const auto& main_settings1 = settings->GetMainExecutorSettings();
  EXPECT_EQ(main_settings1.GetMaxNumTokens(), 8192);
  EXPECT_TRUE(main_settings1.GetAdvancedSettings().has_value());
  EXPECT_EQ(main_settings1.GetAdvancedSettings()->prefill_batch_sizes.size(),
            1);
  EXPECT_EQ(*main_settings1.GetAdvancedSettings()->prefill_batch_sizes.begin(),
            kNumInputPromptTokens + /*margin=*/2);
}

TEST(SessionConfigTest, MaybeUpdateAndValidateLlmGemma3N) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  auto session_config = SessionConfig::CreateDefault();
  EXPECT_OK(settings);
  // We didn't call MaybeUpdateAndValidate on EngineSettings, so some of the
  // required fields are not set. So the validation should fail.
  EXPECT_THAT(session_config.MaybeUpdateAndValidate(*settings),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TextToTokenIds("<eos>"))
      .WillRepeatedly(Return(std::vector<int>({1})));
  EXPECT_CALL(tokenizer, TextToTokenIds("<ctrl>"))
      .WillRepeatedly(Return(std::vector<int>({1})));
  EXPECT_CALL(tokenizer, TextToTokenIds("<end_of_turn>"))
      .WillRepeatedly(Return(std::vector<int>({1})));
  EXPECT_CALL(tokenizer, TokenIdsToText(std::vector<int>({105})))
      .WillRepeatedly(Return("<start_of_turn>"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(std::vector<int>({256000})));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();

  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));
  // The validation should pass now.
  EXPECT_OK(session_config.MaybeUpdateAndValidate(*settings));
  EXPECT_EQ(session_config.GetSamplerBackend(), Backend::CPU);
  EXPECT_EQ(session_config.GetLlmModelType().model_type_case(),
            proto::LlmModelType::kGemma3N);
}

TEST(SessionConfigTest, MaybeUpdateAndValidateLlmGemma3) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  auto session_config = SessionConfig::CreateDefault();
  EXPECT_OK(settings);
  // We didn't call MaybeUpdateAndValidate on EngineSettings, so some of the
  // required fields are not set. So the validation should fail.
  EXPECT_THAT(session_config.MaybeUpdateAndValidate(*settings),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TextToTokenIds("<eos>"))
      .WillRepeatedly(Return(std::vector<int>({1})));
  EXPECT_CALL(tokenizer, TextToTokenIds("<ctrl>"))
      .WillRepeatedly(Return(std::vector<int>({1})));
  EXPECT_CALL(tokenizer, TextToTokenIds("<end_of_turn>"))
      .WillRepeatedly(Return(std::vector<int>({1})));
  EXPECT_CALL(tokenizer, TokenIdsToText(std::vector<int>({105})))
      .WillRepeatedly(Return("<start_of_turn>"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(
          // The encoded ids for "<start_of_audio>" in the Gemma3 1B tokenizer.
          std::vector<int>{236820, 3041, 236779, 1340, 236779, 20156, 236813}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();

  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));
  // The validation should pass now.
  EXPECT_OK(session_config.MaybeUpdateAndValidate(*settings));
  EXPECT_EQ(session_config.GetSamplerBackend(), Backend::CPU);
  EXPECT_EQ(session_config.GetLlmModelType().model_type_case(),
            proto::LlmModelType::kGemma3);
}

// Check the Jinja prompt template is set from the deprecated prompt template
// field.
TEST(SessionConfigTest, MaybeUpdateAndValidateJinjaPromptTemplate) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  auto session_config = SessionConfig::CreateDefault();
  EXPECT_OK(settings);

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();
  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));

  EXPECT_OK(session_config.MaybeUpdateAndValidate(*settings));
  // The Jinja prompt template should not contain "<start_of_turn>" because the
  // template proto is not set.
  EXPECT_THAT(session_config.GetJinjaPromptTemplate(),
              testing::Not(testing::HasSubstr("<start_of_turn>user\n")));
  EXPECT_THAT(session_config.GetJinjaPromptTemplate(),
              testing::Not(testing::HasSubstr("<start_of_turn>model\n")));
  EXPECT_THAT(session_config.GetJinjaPromptTemplate(),
              testing::Not(testing::HasSubstr("<start_of_turn>system\n")));

  // Set the prompt template in the llm metadata. This should be used to
  // generate the Jinja prompt template.
  llm_metadata.mutable_prompt_templates()->mutable_user()->set_prefix(
      "<start_of_turn>user\n");
  llm_metadata.mutable_prompt_templates()->mutable_model()->set_prefix(
      "<start_of_turn>model\n");
  llm_metadata.mutable_prompt_templates()->mutable_system()->set_prefix(
      "<start_of_turn>system\n");
  session_config.GetMutableJinjaPromptTemplate().clear();
  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));
  EXPECT_OK(session_config.MaybeUpdateAndValidate(*settings));
  EXPECT_THAT(session_config.GetJinjaPromptTemplate(),
              testing::HasSubstr("<start_of_turn>user\n"));
  EXPECT_THAT(session_config.GetJinjaPromptTemplate(),
              testing::HasSubstr("<start_of_turn>model\n"));
  EXPECT_THAT(session_config.GetJinjaPromptTemplate(),
              testing::HasSubstr("<start_of_turn>system\n"));
}

TEST(SessionConfigTest, MaybeUpdateAndValidateNotOverwrite) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();
  llm_metadata.set_jinja_prompt_template("template_from_metadata");
  llm_metadata.mutable_llm_model_type()->mutable_gemma3n();
  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));

  auto session_config = SessionConfig::CreateDefault();
  session_config.GetMutableJinjaPromptTemplate() = "template_from_user";
  session_config.GetMutableLlmModelType().mutable_gemma3();
  EXPECT_OK(session_config.MaybeUpdateAndValidate(*settings));
  EXPECT_EQ(session_config.GetJinjaPromptTemplate(), "template_from_user");
  EXPECT_EQ(session_config.GetLlmModelType().model_type_case(),
            proto::LlmModelType::kGemma3);
}

TEST(SessionConfigTest, PrintOperator) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams().set_type(
      proto::SamplerParameters::TOP_K);
  session_config.GetMutableSamplerParams().set_k(10);
  session_config.SetStartTokenId(1);
  session_config.GetMutableStopTokenIds() = {{0}, {1, 2}};
  session_config.SetNumOutputCandidates(2);
  std::stringstream oss;
  oss << session_config;
}

TEST(SessionConfigTest, SetAndGetSamplerBackend) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  EXPECT_EQ(session_config.GetSamplerBackend(), Backend::UNSPECIFIED);
  session_config.SetSamplerBackend(Backend::CPU);
  EXPECT_EQ(session_config.GetSamplerBackend(), Backend::CPU);
  session_config.SetSamplerBackend(Backend::GPU);
  EXPECT_EQ(session_config.GetSamplerBackend(), Backend::GPU);
}

TEST(SessionConfigTest,
     MaybeUpdateAndValidatePromptTemplates_NoSessionTemplate) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();
  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));

  // SessionConfig has no promptTemplate: Use default from llm metadata.
  auto session_config = SessionConfig::CreateDefault();
  EXPECT_OK(session_config.MaybeUpdateAndValidate(*settings));
  EXPECT_EQ(session_config.GetPromptTemplates().user().prefix(), "<start>user");
  EXPECT_EQ(session_config.GetPromptTemplates().model().prefix(),
            "<start>model");
}

TEST(SessionConfigTest,
     MaybeUpdateAndValidatePromptTemplates_SessionTemplateSet) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();
  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));

  // SessionConfig has non-empty template: Use that.
  auto session_config = SessionConfig::CreateDefault();
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "session_user");
  EXPECT_OK(session_config.MaybeUpdateAndValidate(*settings));
  EXPECT_EQ(session_config.GetPromptTemplates().user().prefix(),
            "session_user");
  EXPECT_FALSE(session_config.GetPromptTemplates().has_model());
  EXPECT_FALSE(session_config.GetPromptTemplates().has_system());
}

TEST(SessionConfigTest,
     MaybeUpdateAndValidatePromptTemplates_SessionTemplateSetEmpty) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();
  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));

  // SessionConfig has non-empty template: Use that.
  auto session_config = SessionConfig::CreateDefault();
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix("");
  EXPECT_OK(session_config.MaybeUpdateAndValidate(*settings));
  EXPECT_EQ(session_config.GetPromptTemplates().user().prefix(), "");
  EXPECT_FALSE(session_config.GetPromptTemplates().has_model());
  EXPECT_FALSE(session_config.GetPromptTemplates().has_system());
}

TEST(SessionConfigTest,
     MaybeUpdateAndValidatePromptTemplates_MetadataTemplateMissing) {
  auto model_assets = ModelAssets::Create("test_model_path_1");
  ASSERT_OK(model_assets);
  auto settings = EngineSettings::CreateDefault(*model_assets);
  EXPECT_OK(settings);

  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("fake_text"));
  EXPECT_CALL(tokenizer, TokenToId).WillRepeatedly(Return(1));
  EXPECT_CALL(tokenizer, TextToTokenIds)
      .WillRepeatedly(Return(std::vector<int>{1}));
  proto::LlmMetadata llm_metadata = CreateLlmMetadata();
  llm_metadata.clear_prompt_templates();
  EXPECT_OK(settings->MaybeUpdateAndValidate(tokenizer, &llm_metadata));

  // LlmMetadata has no promptTemplate: SessionConfig template remains default.
  auto session_config = SessionConfig::CreateDefault();
  EXPECT_OK(session_config.MaybeUpdateAndValidate(*settings));
  EXPECT_FALSE(session_config.GetPromptTemplates().has_user());
  EXPECT_FALSE(session_config.GetPromptTemplates().has_model());
  EXPECT_FALSE(session_config.GetPromptTemplates().has_system());
}

}  // namespace
}  // namespace litert::lm
