#include "cortext/generator/multimodal_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#if defined(CORTEXT_ENABLE_GEMMA_ORT)
#include <onnxruntime/onnxruntime_cxx_api.h>
#endif

namespace cortext
{

namespace
{

// Model configuration
constexpr int kHiddenDim = 2048;
constexpr int64_t kImageTokenId = 128011; // gemma-3n image placeholder
constexpr int64_t kAudioTokenId = 128012; // gemma-3n audio placeholder

// Image preprocessing constants (ImageNet normalization)
constexpr float kImageMean[3] = { 0.5f, 0.5f, 0.5f };
constexpr float kImageStd[3] = { 0.5f, 0.5f, 0.5f };
constexpr int kImageSize = 512; // Target image size for gemma-3n

// Audio preprocessing constants
constexpr int kTargetSampleRate = 16000;
constexpr int kMelBins = 128;
constexpr int kHopLength = 160;     // 10ms at 16kHz
constexpr int kWinLength = 400;     // 25ms at 16kHz
constexpr int kNFft = 512;

/// @brief Simple bilinear interpolation for image resizing.
void
ResizeImage (const std::uint8_t *src, int src_w, int src_h, int channels,
             std::vector<std::uint8_t> &dst, int dst_w, int dst_h)
{
  dst.resize (static_cast<std::size_t> (dst_w * dst_h * channels));

  float x_ratio = static_cast<float> (src_w) / static_cast<float> (dst_w);
  float y_ratio = static_cast<float> (src_h) / static_cast<float> (dst_h);

  for (int y = 0; y < dst_h; ++y)
    {
      for (int x = 0; x < dst_w; ++x)
        {
          float src_x = x * x_ratio;
          float src_y = y * y_ratio;

          int x0 = static_cast<int> (src_x);
          int y0 = static_cast<int> (src_y);
          int x1 = std::min (x0 + 1, src_w - 1);
          int y1 = std::min (y0 + 1, src_h - 1);

          float x_frac = src_x - x0;
          float y_frac = src_y - y0;

          for (int c = 0; c < channels; ++c)
            {
              float v00 = src[(y0 * src_w + x0) * channels + c];
              float v01 = src[(y0 * src_w + x1) * channels + c];
              float v10 = src[(y1 * src_w + x0) * channels + c];
              float v11 = src[(y1 * src_w + x1) * channels + c];

              float v0 = v00 * (1 - x_frac) + v01 * x_frac;
              float v1 = v10 * (1 - x_frac) + v11 * x_frac;
              float v = v0 * (1 - y_frac) + v1 * y_frac;

              dst[(y * dst_w + x) * channels + c]
                  = static_cast<std::uint8_t> (std::clamp (v, 0.0f, 255.0f));
            }
        }
    }
}

/// @brief Compute Hann window.
std::vector<float>
HannWindow (int length)
{
  std::vector<float> window (length);
  for (int i = 0; i < length; ++i)
    {
      window[i] = 0.5f * (1.0f - std::cos (2.0f * M_PI * i / (length - 1)));
    }
  return window;
}

/// @brief Compute mel filterbank.
std::vector<std::vector<float> >
MelFilterbank (int n_mels, int n_fft, int sample_rate)
{
  // Convert Hz to Mel
  auto hz_to_mel = [] (float hz) {
    return 2595.0f * std::log10 (1.0f + hz / 700.0f);
  };
  auto mel_to_hz = [] (float mel) {
    return 700.0f * (std::pow (10.0f, mel / 2595.0f) - 1.0f);
  };

  float min_mel = hz_to_mel (0.0f);
  float max_mel = hz_to_mel (static_cast<float> (sample_rate) / 2.0f);

  // Create mel points
  std::vector<float> mel_points (n_mels + 2);
  for (int i = 0; i <= n_mels + 1; ++i)
    {
      mel_points[i]
          = mel_to_hz (min_mel + i * (max_mel - min_mel) / (n_mels + 1));
    }

  // Convert to FFT bin indices
  std::vector<int> bin_points (n_mels + 2);
  for (int i = 0; i <= n_mels + 1; ++i)
    {
      bin_points[i] = static_cast<int> (
          std::floor ((n_fft + 1) * mel_points[i] / sample_rate));
    }

  // Create filterbank
  int n_bins = n_fft / 2 + 1;
  std::vector<std::vector<float> > filterbank (n_mels,
                                               std::vector<float> (n_bins, 0));

  for (int m = 0; m < n_mels; ++m)
    {
      for (int k = bin_points[m]; k < bin_points[m + 1]; ++k)
        {
          if (k < n_bins)
            {
              filterbank[m][k] = static_cast<float> (k - bin_points[m])
                                 / (bin_points[m + 1] - bin_points[m]);
            }
        }
      for (int k = bin_points[m + 1]; k < bin_points[m + 2]; ++k)
        {
          if (k < n_bins)
            {
              filterbank[m][k] = static_cast<float> (bin_points[m + 2] - k)
                                 / (bin_points[m + 2] - bin_points[m + 1]);
            }
        }
    }

  return filterbank;
}

/// @brief Simple DFT for a single frame (not optimized).
void
ComputeDFT (const std::vector<float> &frame, int n_fft,
            std::vector<float> &magnitude)
{
  int n_bins = n_fft / 2 + 1;
  magnitude.resize (n_bins);

  for (int k = 0; k < n_bins; ++k)
    {
      float real = 0.0f;
      float imag = 0.0f;
      for (int n = 0; n < static_cast<int> (frame.size ()); ++n)
        {
          float angle = 2.0f * M_PI * k * n / n_fft;
          real += frame[n] * std::cos (angle);
          imag -= frame[n] * std::sin (angle);
        }
      magnitude[k] = std::sqrt (real * real + imag * imag);
    }
}

/// @brief Simple resampling (linear interpolation).
std::vector<float>
Resample (const float *pcm, std::size_t num_samples, int src_rate,
          int dst_rate)
{
  if (src_rate == dst_rate)
    {
      return std::vector<float> (pcm, pcm + num_samples);
    }

  double ratio = static_cast<double> (dst_rate) / src_rate;
  std::size_t new_length
      = static_cast<std::size_t> (num_samples * ratio);
  std::vector<float> result (new_length);

  for (std::size_t i = 0; i < new_length; ++i)
    {
      double src_pos = i / ratio;
      std::size_t idx = static_cast<std::size_t> (src_pos);
      double frac = src_pos - idx;

      if (idx + 1 < num_samples)
        {
          result[i] = static_cast<float> (pcm[idx] * (1 - frac)
                                          + pcm[idx + 1] * frac);
        }
      else if (idx < num_samples)
        {
          result[i] = pcm[idx];
        }
    }

  return result;
}

} // namespace

// ============================================================================
// GemmaVisionEncoder Implementation
// ============================================================================

struct GemmaVisionEncoder::Impl
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "cortext-gemma-vision" };
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  bool available = false;
  std::size_t hidden_dim = kHiddenDim;

  void
  LoadModel (const std::string &models_dir)
  {
    // Try quantized first, then full precision
    std::vector<std::string> model_names = { "vision_encoder_quantized.onnx",
                                             "vision_encoder_int8.onnx",
                                             "vision_encoder.onnx" };

    std::filesystem::path model_path;
    for (const auto &name : model_names)
      {
        model_path
            = std::filesystem::path (models_dir) / "onnx" / name;
        if (std::filesystem::exists (model_path))
          {
            break;
          }
      }

    if (!std::filesystem::exists (model_path))
      {
        // Vision encoder is optional
        return;
      }

    opts.SetIntraOpNumThreads (1);
    opts.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);

    session
        = std::make_unique<Ort::Session> (env, model_path.c_str (), opts);
    available = true;
  }
#else
  bool available = false;
  std::size_t hidden_dim = kHiddenDim;
#endif
};

GemmaVisionEncoder::GemmaVisionEncoder (const std::string &models_dir)
    : impl_ (std::make_unique<Impl> ())
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  impl_->LoadModel (models_dir);
#else
  (void)models_dir;
#endif
}

GemmaVisionEncoder::~GemmaVisionEncoder () = default;
GemmaVisionEncoder::GemmaVisionEncoder (GemmaVisionEncoder &&) noexcept
    = default;
GemmaVisionEncoder &
GemmaVisionEncoder::operator= (GemmaVisionEncoder &&) noexcept
    = default;

std::vector<float>
GemmaVisionEncoder::Preprocess (const std::uint8_t *data, int width,
                                int height, int channels)
{
  // Resize to target size maintaining aspect ratio
  int target_size = kImageSize;
  int new_width, new_height;

  if (width < height)
    {
      new_width = target_size;
      new_height = target_size * height / width;
    }
  else
    {
      new_height = target_size;
      new_width = target_size * width / height;
    }

  std::vector<std::uint8_t> resized;
  ResizeImage (data, width, height, channels, resized, new_width, new_height);

  // Center crop to square
  int crop_x = (new_width - target_size) / 2;
  int crop_y = (new_height - target_size) / 2;

  // Convert to normalized float tensor [1, 3, H, W]
  // CHW format with ImageNet normalization
  std::vector<float> pixel_values (3 * target_size * target_size);

  for (int c = 0; c < 3; ++c)
    {
      for (int y = 0; y < target_size; ++y)
        {
          for (int x = 0; x < target_size; ++x)
            {
              int src_y = crop_y + y;
              int src_x = crop_x + x;
              int src_idx = (src_y * new_width + src_x) * channels;

              // Get pixel value (handle RGBA by using only RGB)
              float pixel = resized[src_idx + c] / 255.0f;

              // Normalize
              float normalized = (pixel - kImageMean[c]) / kImageStd[c];

              // Store in CHW format
              pixel_values[c * target_size * target_size + y * target_size + x]
                  = normalized;
            }
        }
    }

  return pixel_values;
}

VisionEncoderOutput
GemmaVisionEncoder::Encode (const std::uint8_t *data, int width, int height,
                            int channels)
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  if (!impl_->available)
    {
      throw std::runtime_error ("Vision encoder not available");
    }

  // Preprocess image
  auto pixel_values = Preprocess (data, width, height, channels);

  // Create input tensor
  Ort::MemoryInfo mem_info
      = Ort::MemoryInfo::CreateCpu (OrtDeviceAllocator, OrtMemTypeCPU);

  std::vector<int64_t> input_shape = { 1, 3, kImageSize, kImageSize };
  Ort::Value input = Ort::Value::CreateTensor<float> (
      mem_info, pixel_values.data (), pixel_values.size (), input_shape.data (),
      input_shape.size ());

  const char *input_names[] = { "pixel_values" };
  const char *output_names[] = { "image_features" };

  auto outputs = impl_->session->Run (Ort::RunOptions{ nullptr }, input_names,
                                      &input, 1, output_names, 1);

  // Extract output
  VisionEncoderOutput result;
  const float *features_data = outputs[0].GetTensorData<float> ();
  auto shape_info = outputs[0].GetTensorTypeAndShapeInfo ();
  auto shape = shape_info.GetShape ();

  std::size_t total_size = 1;
  for (auto dim : shape)
    {
      total_size *= static_cast<std::size_t> (dim);
      result.shape.push_back (static_cast<std::size_t> (dim));
    }

  result.features.assign (features_data, features_data + total_size);

  // Calculate number of image tokens
  if (shape.size () >= 2)
    {
      result.num_image_tokens = static_cast<std::size_t> (shape[1]);
      impl_->hidden_dim = static_cast<std::size_t> (shape[2]);
    }

  return result;

#else
  (void)data;
  (void)width;
  (void)height;
  (void)channels;
  throw std::runtime_error (
      "GemmaVisionEncoder: CORTEXT_ENABLE_GEMMA_ORT not defined");
#endif
}

bool
GemmaVisionEncoder::IsAvailable () const
{
  return impl_->available;
}

std::size_t
GemmaVisionEncoder::HiddenDim () const
{
  return impl_->hidden_dim;
}

int64_t
GemmaVisionEncoder::ImageTokenId () const
{
  return kImageTokenId;
}

// ============================================================================
// GemmaAudioEncoder Implementation
// ============================================================================

struct GemmaAudioEncoder::Impl
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "cortext-gemma-audio" };
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  bool available = false;
  std::size_t hidden_dim = kHiddenDim;

  void
  LoadModel (const std::string &models_dir)
  {
    // Try quantized first
    std::vector<std::string> model_names
        = { "audio_encoder_q4.onnx", "audio_encoder_quantized.onnx",
            "audio_encoder_int8.onnx", "audio_encoder.onnx" };

    std::filesystem::path model_path;
    for (const auto &name : model_names)
      {
        model_path
            = std::filesystem::path (models_dir) / "onnx" / name;
        if (std::filesystem::exists (model_path))
          {
            break;
          }
      }

    if (!std::filesystem::exists (model_path))
      {
        // Audio encoder is optional
        return;
      }

    opts.SetIntraOpNumThreads (1);
    opts.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);

    session
        = std::make_unique<Ort::Session> (env, model_path.c_str (), opts);
    available = true;
  }
#else
  bool available = false;
  std::size_t hidden_dim = kHiddenDim;
#endif

  // Mel filterbank cache
  std::vector<std::vector<float> > mel_filterbank;

  void
  InitMelFilterbank ()
  {
    if (mel_filterbank.empty ())
      {
        mel_filterbank
            = MelFilterbank (kMelBins, kNFft, kTargetSampleRate);
      }
  }
};

GemmaAudioEncoder::GemmaAudioEncoder (const std::string &models_dir)
    : impl_ (std::make_unique<Impl> ())
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  impl_->LoadModel (models_dir);
#else
  (void)models_dir;
#endif
  impl_->InitMelFilterbank ();
}

GemmaAudioEncoder::~GemmaAudioEncoder () = default;
GemmaAudioEncoder::GemmaAudioEncoder (GemmaAudioEncoder &&) noexcept = default;
GemmaAudioEncoder &
GemmaAudioEncoder::operator= (GemmaAudioEncoder &&) noexcept
    = default;

std::pair<std::vector<float>, std::vector<float> >
GemmaAudioEncoder::ComputeMelSpectrogram (const float *pcm,
                                          std::size_t num_samples,
                                          int sample_rate)
{
  // Resample to target rate
  auto resampled = Resample (pcm, num_samples, sample_rate, kTargetSampleRate);

  // Compute mel spectrogram
  auto window = HannWindow (kWinLength);

  int num_frames
      = static_cast<int> ((resampled.size () - kWinLength) / kHopLength) + 1;
  if (num_frames < 1)
    num_frames = 1;

  // Mel spectrogram [num_frames, kMelBins]
  std::vector<float> mel_spec (num_frames * kMelBins, 0.0f);
  std::vector<float> mask (num_frames, 1.0f);

  std::vector<float> frame (kWinLength);
  std::vector<float> magnitude;

  for (int f = 0; f < num_frames; ++f)
    {
      int start = f * kHopLength;

      // Extract and window frame
      for (int i = 0; i < kWinLength; ++i)
        {
          int idx = start + i;
          if (idx < static_cast<int> (resampled.size ()))
            {
              frame[i] = resampled[idx] * window[i];
            }
          else
            {
              frame[i] = 0.0f;
            }
        }

      // Compute DFT magnitude
      ComputeDFT (frame, kNFft, magnitude);

      // Apply mel filterbank
      for (int m = 0; m < kMelBins; ++m)
        {
          float mel_energy = 0.0f;
          for (int k = 0;
               k < static_cast<int> (impl_->mel_filterbank[m].size ()); ++k)
            {
              if (k < static_cast<int> (magnitude.size ()))
                {
                  mel_energy += impl_->mel_filterbank[m][k] * magnitude[k];
                }
            }
          // Log mel (add small epsilon for stability)
          mel_spec[f * kMelBins + m]
              = std::log (std::max (mel_energy, 1e-10f));
        }
    }

  return { mel_spec, mask };
}

AudioEncoderOutput
GemmaAudioEncoder::Encode (const float *pcm, std::size_t num_samples,
                           int sample_rate)
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  if (!impl_->available)
    {
      throw std::runtime_error ("Audio encoder not available");
    }

  // Compute mel spectrogram
  auto [mel_spec, mask] = ComputeMelSpectrogram (pcm, num_samples, sample_rate);

  int num_frames = static_cast<int> (mel_spec.size () / kMelBins);

  // Create input tensors
  Ort::MemoryInfo mem_info
      = Ort::MemoryInfo::CreateCpu (OrtDeviceAllocator, OrtMemTypeCPU);

  std::vector<int64_t> features_shape = { 1, num_frames, kMelBins };
  std::vector<int64_t> mask_shape = { 1, num_frames };

  Ort::Value features_input = Ort::Value::CreateTensor<float> (
      mem_info, mel_spec.data (), mel_spec.size (), features_shape.data (),
      features_shape.size ());

  Ort::Value mask_input = Ort::Value::CreateTensor<float> (
      mem_info, mask.data (), mask.size (), mask_shape.data (),
      mask_shape.size ());

  const char *input_names[] = { "input_features", "input_features_mask" };
  const char *output_names[] = { "audio_features" };

  std::vector<Ort::Value> inputs;
  inputs.push_back (std::move (features_input));
  inputs.push_back (std::move (mask_input));

  auto outputs = impl_->session->Run (Ort::RunOptions{ nullptr }, input_names,
                                      inputs.data (), inputs.size (),
                                      output_names, 1);

  // Extract output
  AudioEncoderOutput result;
  const float *features_data = outputs[0].GetTensorData<float> ();
  auto shape_info = outputs[0].GetTensorTypeAndShapeInfo ();
  auto shape = shape_info.GetShape ();

  std::size_t total_size = 1;
  for (auto dim : shape)
    {
      total_size *= static_cast<std::size_t> (dim);
      result.shape.push_back (static_cast<std::size_t> (dim));
    }

  result.features.assign (features_data, features_data + total_size);

  // Calculate number of audio tokens (~6.25 tokens per second)
  if (shape.size () >= 2)
    {
      result.num_audio_tokens = static_cast<std::size_t> (shape[1]);
      impl_->hidden_dim = static_cast<std::size_t> (shape[2]);
    }

  return result;

#else
  (void)pcm;
  (void)num_samples;
  (void)sample_rate;
  throw std::runtime_error (
      "GemmaAudioEncoder: CORTEXT_ENABLE_GEMMA_ORT not defined");
#endif
}

bool
GemmaAudioEncoder::IsAvailable () const
{
  return impl_->available;
}

std::size_t
GemmaAudioEncoder::HiddenDim () const
{
  return impl_->hidden_dim;
}

int64_t
GemmaAudioEncoder::AudioTokenId () const
{
  return kAudioTokenId;
}

// ============================================================================
// MultimodalEncoder Implementation
// ============================================================================

struct MultimodalEncoder::Impl
{
  std::unique_ptr<GemmaVisionEncoder> vision;
  std::unique_ptr<GemmaAudioEncoder> audio;
};

MultimodalEncoder::MultimodalEncoder (const std::string &models_dir)
    : impl_ (std::make_unique<Impl> ())
{
  impl_->vision = std::make_unique<GemmaVisionEncoder> (models_dir);
  impl_->audio = std::make_unique<GemmaAudioEncoder> (models_dir);
}

MultimodalEncoder::~MultimodalEncoder () = default;
MultimodalEncoder::MultimodalEncoder (MultimodalEncoder &&) noexcept = default;
MultimodalEncoder &
MultimodalEncoder::operator= (MultimodalEncoder &&) noexcept
    = default;

std::vector<float>
MultimodalEncoder::InjectImageFeatures (
    const std::vector<float> &embeddings,
    const std::vector<std::size_t> &embed_shape,
    const std::vector<int64_t> &input_ids, const std::uint8_t *image_data,
    int width, int height, int channels)
{
  if (!impl_->vision || !impl_->vision->IsAvailable ())
    {
      throw std::runtime_error ("Vision encoder not available");
    }

  // Encode image
  auto image_output = impl_->vision->Encode (image_data, width, height,
                                             channels);

  // Create output embeddings (copy of input)
  std::vector<float> result = embeddings;

  // Find image token positions and replace with image features
  std::size_t hidden_dim = embed_shape.back ();
  std::size_t seq_len = embed_shape[1];
  int64_t image_token_id = impl_->vision->ImageTokenId ();

  std::size_t feature_idx = 0;
  std::size_t num_features = image_output.num_image_tokens;

  for (std::size_t i = 0; i < input_ids.size () && feature_idx < num_features;
       ++i)
    {
      if (input_ids[i] == image_token_id)
        {
          // Replace embedding at position i with image feature
          std::size_t embed_offset = i * hidden_dim;
          std::size_t feature_offset = feature_idx * hidden_dim;

          for (std::size_t d = 0; d < hidden_dim; ++d)
            {
              if (embed_offset + d < result.size ()
                  && feature_offset + d < image_output.features.size ())
                {
                  result[embed_offset + d]
                      = image_output.features[feature_offset + d];
                }
            }
          ++feature_idx;
        }
    }

  return result;
}

std::vector<float>
MultimodalEncoder::InjectAudioFeatures (
    const std::vector<float> &embeddings,
    const std::vector<std::size_t> &embed_shape,
    const std::vector<int64_t> &input_ids, const float *pcm,
    std::size_t num_samples, int sample_rate)
{
  if (!impl_->audio || !impl_->audio->IsAvailable ())
    {
      throw std::runtime_error ("Audio encoder not available");
    }

  // Encode audio
  auto audio_output = impl_->audio->Encode (pcm, num_samples, sample_rate);

  // Create output embeddings (copy of input)
  std::vector<float> result = embeddings;

  // Find audio token positions and replace with audio features
  std::size_t hidden_dim = embed_shape.back ();
  int64_t audio_token_id = impl_->audio->AudioTokenId ();

  std::size_t feature_idx = 0;
  std::size_t num_features = audio_output.num_audio_tokens;

  for (std::size_t i = 0; i < input_ids.size () && feature_idx < num_features;
       ++i)
    {
      if (input_ids[i] == audio_token_id)
        {
          // Replace embedding at position i with audio feature
          std::size_t embed_offset = i * hidden_dim;
          std::size_t feature_offset = feature_idx * hidden_dim;

          for (std::size_t d = 0; d < hidden_dim; ++d)
            {
              if (embed_offset + d < result.size ()
                  && feature_offset + d < audio_output.features.size ())
                {
                  result[embed_offset + d]
                      = audio_output.features[feature_offset + d];
                }
            }
          ++feature_idx;
        }
    }

  return result;
}

bool
MultimodalEncoder::HasVision () const
{
  return impl_->vision && impl_->vision->IsAvailable ();
}

bool
MultimodalEncoder::HasAudio () const
{
  return impl_->audio && impl_->audio->IsAvailable ();
}

int64_t
MultimodalEncoder::ImageTokenId () const
{
  return impl_->vision ? impl_->vision->ImageTokenId () : kImageTokenId;
}

int64_t
MultimodalEncoder::AudioTokenId () const
{
  return impl_->audio ? impl_->audio->AudioTokenId () : kAudioTokenId;
}

} // namespace cortext
