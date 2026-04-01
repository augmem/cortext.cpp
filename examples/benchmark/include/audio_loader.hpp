#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace benchmark
{

/// @brief Audio data container with metadata.
struct AudioData
{
  std::vector<float> samples; ///< Normalized float32 samples [-1.0, 1.0]
  uint32_t sample_rate;       ///< Sample rate in Hz
  uint16_t num_channels;      ///< Number of channels
  double duration_seconds;    ///< Duration in seconds

  /// @brief Get number of samples per channel.
  size_t
  SamplesPerChannel () const
  {
    if (num_channels == 0)
      return 0;
    return samples.size () / num_channels;
  }
};

/// @brief Load audio from WAV file.
/// @param path Path to .wav file
/// @return AudioData on success, std::nullopt on failure
/// @note Supports 8-bit, 16-bit, 24-bit PCM and 32-bit float WAV files
std::optional<AudioData> LoadWav (const std::string &path);

/// @brief Load raw PCM data from file.
/// @param path Path to .pcm file (raw float32 samples)
/// @param sample_rate Sample rate (default: 16000 for LLM audio)
/// @param num_channels Number of channels (default: 1)
/// @return AudioData on success, std::nullopt on failure
std::optional<AudioData> LoadPcmFloat32 (const std::string &path,
                                          uint32_t sample_rate = 16000,
                                          uint16_t num_channels = 1);

/// @brief Load raw PCM data from file (16-bit signed integers).
/// @param path Path to .pcm file (raw int16 samples)
/// @param sample_rate Sample rate (default: 16000)
/// @param num_channels Number of channels (default: 1)
/// @return AudioData on success, std::nullopt on failure
std::optional<AudioData> LoadPcmInt16 (const std::string &path,
                                        uint32_t sample_rate = 16000,
                                        uint16_t num_channels = 1);

/// @brief Resample audio to target sample rate.
/// @param audio Input audio data
/// @param target_rate Target sample rate (e.g., 16000)
/// @return Resampled audio data
/// @note Uses linear interpolation for simplicity
AudioData ResampleTo (const AudioData &audio, uint32_t target_rate);

/// @brief Convert stereo to mono by averaging channels.
/// @param audio Input audio data
/// @return Mono audio data
AudioData ConvertToMono (const AudioData &audio);

/// @brief Resample and convert to 16kHz mono (standard for LLM audio).
/// @param audio Input audio data
/// @return Normalized 16kHz mono audio
AudioData NormalizeTo16kMono (const AudioData &audio);

/// @brief Detect audio format from file extension.
/// @param path File path
/// @return "wav", "pcm", or empty string if unknown
std::string DetectAudioFormat (const std::string &path);

/// @brief Load audio file, auto-detecting format.
/// @param path File path (.wav or .pcm)
/// @return AudioData on success, std::nullopt on failure
std::optional<AudioData> LoadAudio (const std::string &path);

} // namespace benchmark
