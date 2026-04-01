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

#include "runtime/components/preprocessor/audio_preprocessor_miniaudio.h"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/preprocessor/audio_preprocessor.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

constexpr absl::string_view kFrontendModelPath =
    "litert_lm/runtime/components/testdata/frontend.tflite";
constexpr absl::string_view kDecodedAudioPath =
    "litert_lm/runtime/components/testdata/decoded_audio_samples.bin";
constexpr absl::string_view kAudioPath =
    "litert_lm/runtime/components/testdata/audio_sample.wav";

template <typename T>
absl::StatusOr<std::vector<T>> GetDataAsVector(
    litert::TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto elements, tensor_type.Layout().NumElements());
  std::vector<T> data(elements);
  LITERT_RETURN_IF_ERROR(tensor_buffer.Read<T>(absl::MakeSpan(data)));
  return data;
}

template <typename T>
absl::StatusOr<std::vector<T>> GetDataAsVector(
    const litert::TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto elements, tensor_type.Layout().NumElements());
  std::vector<T> data(elements);
  LITERT_RETURN_IF_ERROR(const_cast<litert::TensorBuffer&>(tensor_buffer)
                             .Read<T>(absl::MakeSpan(data)));
  return data;
}

absl::StatusOr<std::string> GetContents(const std::string& path) {
  std::ifstream input_stream(path);
  if (!input_stream.is_open()) {
    return absl::InternalError(absl::StrCat("Could not open file: ", path));
  }

  std::string content;
  content.assign((std::istreambuf_iterator<char>(input_stream)),
                 (std::istreambuf_iterator<char>()));
  return std::move(content);
}

absl::StatusOr<std::vector<float>> GetDecodedAudioData() {
  ASSIGN_OR_RETURN(
      auto decoded_audio_data,
      GetContents(absl::StrCat(::testing::SrcDir(), "/", kDecodedAudioPath)));
  std::vector<float> decoded_audio_vector(
      reinterpret_cast<const float*>(decoded_audio_data.data()),
      reinterpret_cast<const float*>(decoded_audio_data.data() +
                                     decoded_audio_data.size()));
  return decoded_audio_vector;
}

absl::StatusOr<std::string> GetRawAudioData() {
  return GetContents(absl::StrCat(::testing::SrcDir(), "/", kAudioPath));
}

class FrontendModelWrapper {
 public:
  static constexpr int kInputTensorLength = 523426;
  static absl::StatusOr<std::unique_ptr<FrontendModelWrapper>> Create(
      absl::string_view model_path) {
    LITERT_ASSIGN_OR_RETURN(auto env, litert::Environment::Create({}));

    LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);

    LITERT_ASSIGN_OR_RETURN(
        auto compiled_model,
        litert::CompiledModel::Create(
            env, absl::StrCat(::testing::SrcDir(), "/", model_path), options));

    auto wrapper = std::unique_ptr<FrontendModelWrapper>(
        new FrontendModelWrapper(std::move(env), std::move(compiled_model)));
    LITERT_RETURN_IF_ERROR(wrapper->InitializeBuffers());
    return wrapper;
  }

  absl::Status Run(const std::vector<float>& audio_data,
                   std::vector<float>* output_spectrogram,
                   std::vector<uint8_t>* output_mask) {
    if (input_buffers_.empty()) {
      return absl::FailedPreconditionError("Model not initialized.");
    }

    // Data in memory needs to be continuous, but the bool type of std library
    // vector is not guaranteed to be continuous for memory. So here we use a
    // bool* to create a continuous memory buffer. This prevent the UBSan check
    // error. See go/ubsan.
    bool* mask_data_ptr = new bool[kInputTensorLength];
    for (int i = 0; i < kInputTensorLength; ++i) {
      if (i < audio_data.size()) {
        mask_data_ptr[i] = true;
      } else {
        mask_data_ptr[i] = false;
      }
    }
    LITERT_RETURN_IF_ERROR(input_buffers_[0].Write(
        absl::MakeConstSpan(mask_data_ptr, kInputTensorLength)));
    delete[] mask_data_ptr;
    LITERT_RETURN_IF_ERROR(input_buffers_[1].Write(absl::MakeSpan(audio_data)));

    compiled_model_.Run(input_buffers_, output_buffers_);
    LITERT_ASSIGN_OR_RETURN(*output_mask,
                            GetDataAsVector<uint8_t>(output_buffers_[0]));
    LITERT_ASSIGN_OR_RETURN(*output_spectrogram,
                            GetDataAsVector<float>(output_buffers_[1]));
    return absl::OkStatus();
  }

 private:
  FrontendModelWrapper(Environment env, litert::CompiledModel compiled_model)
      : env_(std::move(env)), compiled_model_(std::move(compiled_model)) {}

  absl::Status InitializeBuffers() {
    LITERT_ASSIGN_OR_RETURN(auto signatures, compiled_model_.GetSignatures());
    if (signatures.size() != 1) {
      return absl::InvalidArgumentError(
          "Model must have exactly one signature.");
    }

    LITERT_ASSIGN_OR_RETURN(input_buffers_, compiled_model_.CreateInputBuffers(
                                                /*signature_index=*/0));

    LITERT_ASSIGN_OR_RETURN(output_buffers_,
                            compiled_model_.CreateOutputBuffers(
                                /*signature_index=*/0));
    if (output_buffers_.empty()) {
      return absl::InvalidArgumentError("Model must have at least one output.");
    }

    return absl::OkStatus();
  }

  Environment env_;
  litert::CompiledModel compiled_model_;
  std::vector<litert::TensorBuffer> input_buffers_;
  std::vector<litert::TensorBuffer> output_buffers_;
};

// TODO: b/441514829 - Enable the tests on Windows once the bug is fixed.
#if !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) && \
    !defined(__NT__) && !defined(_WIN64)
TEST(AudioPreprocessorMiniAudioTest, DecodeAudio) {
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorMiniAudio::DecodeAudio(
      raw_audio_data, /*num_channels=*/1, /*sample_rate_hz=*/16000,
      pcm_frames));
  ASSERT_OK_AND_ASSIGN(auto decoded_audio_data, GetDecodedAudioData());
  EXPECT_EQ(pcm_frames.size(), decoded_audio_data.size());
  for (int i = 0; i < pcm_frames.size(); ++i) {
    EXPECT_NEAR(pcm_frames[i], decoded_audio_data[i], 1e-6);
  }
}

TEST(AudioPreprocessorMiniAudioTest, UsmPreprocessing) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorMiniAudio::DecodeAudio(
      raw_audio_data, config.GetNumChannels(), config.GetSampleRateHz(),
      pcm_frames));

  // Ground truth from TFLite weightless USM frontend model.
  ASSERT_OK_AND_ASSIGN(auto frontend_model,
                       FrontendModelWrapper::Create(kFrontendModelPath));
  std::vector<float> frontend_mel_spectrogram;
  std::vector<uint8_t> frontend_mask;
  ASSERT_OK(frontend_model->Run(pcm_frames, &frontend_mel_spectrogram,
                                &frontend_mask));
  int true_count = 0;
  for (int i = 0; i < frontend_mask.size(); ++i) {
    if (frontend_mask[i] == 1) {
      true_count++;
    }
  }
  frontend_mel_spectrogram.resize(true_count * config.GetNumMelBins());

  // Create MiniAudio preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorMiniAudio::Create(config));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_audio,
                       preprocessor->Preprocess(InputAudio(raw_audio_data)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_mel_spectrogram_tensor,
                       preprocessed_audio.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(
      auto preprocessed_mel_spectrogram,
      GetDataAsVector<float>(*preprocessed_mel_spectrogram_tensor));

  ASSERT_EQ(preprocessed_mel_spectrogram.size(),
            frontend_mel_spectrogram.size());
  for (int i = 0; i < preprocessed_mel_spectrogram.size(); ++i) {
    EXPECT_NEAR(preprocessed_mel_spectrogram[i], frontend_mel_spectrogram[i],
                5e-4);
  }
}

TEST(AudioPreprocessorMiniAudioTest, UsmPreprocessingWithPcmFrames) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorMiniAudio::DecodeAudio(
      raw_audio_data, config.GetNumChannels(), config.GetSampleRateHz(),
      pcm_frames));

  // Ground truth from TFLite weightless USM frontend model.
  ASSERT_OK_AND_ASSIGN(auto frontend_model,
                       FrontendModelWrapper::Create(kFrontendModelPath));
  std::vector<float> frontend_mel_spectrogram;
  std::vector<uint8_t> frontend_mask;
  ASSERT_OK(frontend_model->Run(pcm_frames, &frontend_mel_spectrogram,
                                &frontend_mask));
  int true_count = 0;
  for (int i = 0; i < frontend_mask.size(); ++i) {
    if (frontend_mask[i] == 1) {
      true_count++;
    }
  }
  frontend_mel_spectrogram.resize(true_count * config.GetNumMelBins());

  // Create MiniAudio preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorMiniAudio::Create(config));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_audio,
                       preprocessor->Preprocess(InputAudio(pcm_frames)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_mel_spectrogram_tensor,
                       preprocessed_audio.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(
      auto preprocessed_mel_spectrogram,
      GetDataAsVector<float>(*preprocessed_mel_spectrogram_tensor));

  ASSERT_EQ(preprocessed_mel_spectrogram.size(),
            frontend_mel_spectrogram.size());
  for (int i = 0; i < preprocessed_mel_spectrogram.size(); ++i) {
    EXPECT_NEAR(preprocessed_mel_spectrogram[i], frontend_mel_spectrogram[i],
                5e-4);
  }
}

TEST(AudioPreprocessorMiniAudioTest, UsmPreprocessingTwice) {
  AudioPreprocessorConfig config =
      AudioPreprocessorConfig::CreateDefaultUsmConfig();
  ASSERT_OK_AND_ASSIGN(auto raw_audio_data, GetRawAudioData());
  std::vector<float> pcm_frames;
  ASSERT_OK(AudioPreprocessorMiniAudio::DecodeAudio(
      raw_audio_data, config.GetNumChannels(), config.GetSampleRateHz(),
      pcm_frames));

  // Ground truth from TFLite weightless USM frontend model.
  ASSERT_OK_AND_ASSIGN(auto frontend_model,
                       FrontendModelWrapper::Create(kFrontendModelPath));
  std::vector<float> frontend_mel_spectrogram;
  std::vector<uint8_t> frontend_mask;
  ASSERT_OK(frontend_model->Run(pcm_frames, &frontend_mel_spectrogram,
                                &frontend_mask));
  int true_count = 0;
  for (int i = 0; i < frontend_mask.size(); ++i) {
    if (frontend_mask[i] == 1) {
      true_count++;
    }
  }
  frontend_mel_spectrogram.resize(true_count * config.GetNumMelBins());

  // Create MiniAudio preprocessor.
  ASSERT_OK_AND_ASSIGN(auto preprocessor,
                       AudioPreprocessorMiniAudio::Create(config));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_audio,
                       preprocessor->Preprocess(InputAudio(raw_audio_data)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_mel_spectrogram_tensor,
                       preprocessed_audio.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(
      auto preprocessed_mel_spectrogram,
      GetDataAsVector<float>(*preprocessed_mel_spectrogram_tensor));

  ASSERT_EQ(preprocessed_mel_spectrogram.size(),
            frontend_mel_spectrogram.size());
  for (int i = 0; i < preprocessed_mel_spectrogram.size(); ++i) {
    EXPECT_NEAR(preprocessed_mel_spectrogram[i], frontend_mel_spectrogram[i],
                5e-4);
  }

  // Preprocess the same audio data again without resetting the preprocessor.
  auto result = preprocessor->Preprocess(InputAudio(raw_audio_data));
  EXPECT_THAT(result, ::testing::status::StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(
      result.status().message(),
      ::testing::HasSubstr(
          "Windowed signals size is not equal to expected number of frames"));

  // Preprocess the same audio data again after resetting the preprocessor.
  preprocessor->Reset();
  ASSERT_OK_AND_ASSIGN(preprocessed_audio,
                       preprocessor->Preprocess(InputAudio(raw_audio_data)));
  ASSERT_OK_AND_ASSIGN(preprocessed_mel_spectrogram_tensor,
                       preprocessed_audio.GetPreprocessedAudioTensor());
  ASSERT_OK_AND_ASSIGN(
      preprocessed_mel_spectrogram,
      GetDataAsVector<float>(*preprocessed_mel_spectrogram_tensor));
  ASSERT_EQ(preprocessed_mel_spectrogram.size(),
            frontend_mel_spectrogram.size());
  for (int i = 0; i < preprocessed_mel_spectrogram.size(); ++i) {
    EXPECT_NEAR(preprocessed_mel_spectrogram[i], frontend_mel_spectrogram[i],
                5e-4);
  }
}

#endif  // !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) &&
        // !defined(__NT__) && !defined(_WIN64)

}  // namespace
}  // namespace litert::lm
