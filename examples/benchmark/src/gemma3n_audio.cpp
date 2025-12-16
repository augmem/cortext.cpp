#include "include/gemma3n_audio.hpp"

#include <algorithm>
#include <cmath>

namespace benchmark
{

namespace
{

using namespace Gemma3nAudioConstants;

// Hann window function
std::vector<float>
CreateHannWindow (int size)
{
  std::vector<float> window (size);
  for (int i = 0; i < size; ++i)
    {
      window[i] = 0.5f
                  * (1.0f
                     - std::cos (2.0f * static_cast<float> (M_PI) * i
                                 / (size - 1)));
    }
  return window;
}

// Simple DFT (for small FFT sizes; in production, use FFTW or similar)
void
ComputeDFT (const float *input, int n, std::vector<float> &magnitude)
{
  magnitude.resize (n / 2 + 1);

  for (int k = 0; k <= n / 2; ++k)
    {
      float real_sum = 0.0f;
      float imag_sum = 0.0f;

      for (int i = 0; i < n; ++i)
        {
          float angle
              = 2.0f * static_cast<float> (M_PI) * k * i / static_cast<float> (n);
          real_sum += input[i] * std::cos (angle);
          imag_sum -= input[i] * std::sin (angle);
        }

      magnitude[k]
          = std::sqrt (real_sum * real_sum + imag_sum * imag_sum) / n;
    }
}

// Create mel filterbank
std::vector<std::vector<float>>
CreateMelFilterbank (int num_fft_bins, int num_mel_bins, int sample_rate)
{
  auto hz_to_mel = [] (float hz) {
    return 2595.0f * std::log10 (1.0f + hz / 700.0f);
  };

  auto mel_to_hz = [] (float mel) {
    return 700.0f * (std::pow (10.0f, mel / 2595.0f) - 1.0f);
  };

  float mel_low = hz_to_mel (0.0f);
  float mel_high = hz_to_mel (static_cast<float> (sample_rate) / 2.0f);

  // Mel points
  std::vector<float> mel_points (num_mel_bins + 2);
  for (int i = 0; i < num_mel_bins + 2; ++i)
    {
      mel_points[i] = mel_low
                      + (mel_high - mel_low) * i
                            / static_cast<float> (num_mel_bins + 1);
    }

  // Convert to Hz and then to FFT bin indices
  std::vector<int> bin_indices (num_mel_bins + 2);
  for (int i = 0; i < num_mel_bins + 2; ++i)
    {
      float hz = mel_to_hz (mel_points[i]);
      bin_indices[i] = static_cast<int> (
          std::round ((num_fft_bins + 1) * hz / sample_rate));
    }

  // Create filterbank
  std::vector<std::vector<float>> filterbank (
      num_mel_bins, std::vector<float> (num_fft_bins, 0.0f));

  for (int m = 0; m < num_mel_bins; ++m)
    {
      int left = bin_indices[m];
      int center = bin_indices[m + 1];
      int right = bin_indices[m + 2];

      // Rising slope
      for (int k = left; k < center && k < num_fft_bins; ++k)
        {
          if (center != left)
            filterbank[m][k]
                = static_cast<float> (k - left) / (center - left);
        }

      // Falling slope
      for (int k = center; k < right && k < num_fft_bins; ++k)
        {
          if (right != center)
            filterbank[m][k]
                = static_cast<float> (right - k) / (right - center);
        }
    }

  return filterbank;
}

} // namespace

void
ComputeMelSpectrogram (const float *pcm, size_t num_samples,
                       std::vector<float> &out_features,
                       int64_t &out_num_frames)
{
  using namespace Gemma3nAudioConstants;

  // Calculate number of frames
  int64_t num_frames
      = (static_cast<int64_t> (num_samples) - kNFft) / kHopLength + 1;
  if (num_frames <= 0)
    {
      num_frames = 1;
    }
  out_num_frames = num_frames;

  // Create Hann window
  auto window = CreateHannWindow (kNFft);

  // Create mel filterbank
  auto filterbank
      = CreateMelFilterbank (kNFft / 2 + 1, kMelBins, kSampleRate);

  // Output buffer
  out_features.resize (num_frames * kMelBins);

  // Process each frame
  std::vector<float> frame (kNFft);
  std::vector<float> magnitude;

  for (int64_t f = 0; f < num_frames; ++f)
    {
      int64_t start = f * kHopLength;

      // Extract and window frame
      for (int i = 0; i < kNFft; ++i)
        {
          int64_t idx = start + i;
          if (idx < static_cast<int64_t> (num_samples))
            frame[i] = pcm[idx] * window[i];
          else
            frame[i] = 0.0f;
        }

      // Compute DFT
      ComputeDFT (frame.data (), kNFft, magnitude);

      // Apply mel filterbank
      for (int m = 0; m < kMelBins; ++m)
        {
          float mel_energy = 0.0f;
          for (size_t k = 0; k < magnitude.size () && k < filterbank[m].size ();
               ++k)
            {
              mel_energy += magnitude[k] * magnitude[k] * filterbank[m][k];
            }

          // Log mel energy with floor
          float log_mel = std::log (std::max (mel_energy, 1e-10f));

          // Normalize
          log_mel = (log_mel - kNormMean) / kNormStd;

          out_features[f * kMelBins + m] = log_mel;
        }
    }
}

Gemma3nAudioFeatures
ExtractGemma3nAudioFeatures (const float *pcm, size_t num_samples)
{
  using namespace Gemma3nAudioConstants;

  Gemma3nAudioFeatures features;

  // Clamp to max duration
  size_t samples_to_process
      = std::min (num_samples, static_cast<size_t> (kMaxSamples));

  // Compute mel spectrogram
  ComputeMelSpectrogram (pcm, samples_to_process, features.mel_features,
                         features.num_frames);
  features.mel_bins = kMelBins;

  // Create attention mask (all ones for valid frames)
  features.attention_mask.resize (features.num_frames, 1);

  return features;
}

std::vector<Gemma3nAudioFeatures>
ChunkAndExtractFeatures (const float *pcm, size_t num_samples)
{
  using namespace Gemma3nAudioConstants;

  std::vector<Gemma3nAudioFeatures> chunks;

  size_t offset = 0;
  while (offset < num_samples)
    {
      size_t chunk_samples
          = std::min (static_cast<size_t> (kMaxSamples), num_samples - offset);
      chunks.push_back (ExtractGemma3nAudioFeatures (pcm + offset, chunk_samples));
      offset += chunk_samples;
    }

  return chunks;
}

} // namespace benchmark
