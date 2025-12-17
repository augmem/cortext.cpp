#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cortext
{

/// @brief Audio features prepared for Gemma 3n USM audio encoder.
///
/// Gemma 3n uses a Universal Speech Model (USM) based audio encoder
/// that requires mel-spectrogram preprocessing with specific parameters.
struct GemmaAudioFeatures
{
  std::vector<float> mel_features; ///< Mel-spectrogram features (flattened)
  std::vector<int64_t> attention_mask; ///< Attention mask for variable length
  int64_t num_frames;                  ///< Number of audio frames
  int64_t mel_bins;                    ///< Number of mel frequency bins
};

/// @brief Constants for Gemma audio preprocessing.
namespace GemmaAudioConstants
{
/// Sample rate expected by Gemma (16kHz)
constexpr int kSampleRate = 16000;

/// Maximum audio duration in seconds (30s chunks)
constexpr int kMaxDurationSec = 30;

/// Maximum samples per chunk
constexpr int kMaxSamples = kSampleRate * kMaxDurationSec;

/// Token rate (audio frames per second)
constexpr float kTokenRate = 6.25f;

/// Number of mel frequency bins
constexpr int kMelBins = 128;

/// FFT size
constexpr int kNFft = 4096;

/// Hop length (160ms per frame = 2560 samples at 16kHz)
constexpr int kHopLength = 2560;

/// Normalization mean for USM
constexpr float kNormMean = -4.268f;

/// Normalization standard deviation for USM
constexpr float kNormStd = 9.138f;
} // namespace GemmaAudioConstants

/// @brief Compute mel-spectrogram for a chunk of audio.
///
/// @param pcm Raw PCM samples (must be 16kHz mono float32, normalized [-1, 1])
/// @param num_samples Number of samples
/// @param out_features Output mel-spectrogram features (flattened)
/// @param out_num_frames Output number of frames
void ComputeMelSpectrogram (const float *pcm, size_t num_samples,
                            std::vector<float> &out_features,
                            int64_t &out_num_frames);

/// @brief Extract audio features for Gemma USM encoder.
///
/// Converts raw PCM audio (16kHz mono float32) to mel-spectrogram
/// features expected by the USM-based audio encoder in Gemma.
///
/// @param pcm Raw PCM samples (must be 16kHz mono float32, normalized [-1, 1])
/// @param num_samples Number of samples
/// @return Audio features ready for Gemma encoder
GemmaAudioFeatures ExtractGemmaAudioFeatures (const float *pcm,
                                              size_t num_samples);

/// @brief Split long audio into 30-second chunks for processing.
///
/// @param pcm Raw PCM samples
/// @param num_samples Number of samples
/// @return Vector of feature sets for each chunk
std::vector<GemmaAudioFeatures> ChunkAndExtractFeatures (const float *pcm,
                                                         size_t num_samples);

} // namespace cortext
