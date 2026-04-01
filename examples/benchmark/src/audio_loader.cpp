#include "include/audio_loader.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace benchmark
{

namespace
{

/// @brief WAV file header structure.
#pragma pack(push, 1)
struct WavHeader
{
  char riff[4];             // "RIFF"
  uint32_t file_size;       // File size - 8
  char wave[4];             // "WAVE"
  char fmt[4];              // "fmt "
  uint32_t fmt_size;        // Format chunk size (16 for PCM)
  uint16_t audio_format;    // 1 = PCM, 3 = IEEE float
  uint16_t num_channels;    // 1 = mono, 2 = stereo
  uint32_t sample_rate;     // e.g., 16000, 44100
  uint32_t byte_rate;       // sample_rate * num_channels * bits_per_sample/8
  uint16_t block_align;     // num_channels * bits_per_sample/8
  uint16_t bits_per_sample; // 8, 16, 24, 32
};
#pragma pack(pop)

bool
FindDataChunk (std::ifstream &file, uint32_t &out_chunk_size)
{
  char chunk_id[4];
  uint32_t chunk_size;

  while (file.read (chunk_id, 4))
    {
      file.read (reinterpret_cast<char *> (&chunk_size), 4);
      if (std::strncmp (chunk_id, "data", 4) == 0)
        {
          out_chunk_size = chunk_size;
          return true;
        }
      // Skip this chunk
      file.seekg (chunk_size, std::ios::cur);
    }
  return false;
}

} // namespace

std::optional<AudioData>
LoadWav (const std::string &path)
{
  std::ifstream file (path, std::ios::binary);
  if (!file.is_open ())
    return std::nullopt;

  WavHeader header;
  file.read (reinterpret_cast<char *> (&header), sizeof (WavHeader));

  // Validate RIFF/WAVE markers
  if (std::strncmp (header.riff, "RIFF", 4) != 0
      || std::strncmp (header.wave, "WAVE", 4) != 0)
    return std::nullopt;

  // Skip extended format chunk if present
  if (header.fmt_size > 16)
    {
      file.seekg (header.fmt_size - 16, std::ios::cur);
    }

  // Find data chunk
  uint32_t data_size = 0;
  if (!FindDataChunk (file, data_size))
    return std::nullopt;

  // Read raw audio bytes
  std::vector<char> raw_data (data_size);
  file.read (raw_data.data (), data_size);

  if (file.gcount () != static_cast<std::streamsize> (data_size))
    return std::nullopt;

  AudioData result;
  result.sample_rate = header.sample_rate;
  result.num_channels = header.num_channels;

  size_t bytes_per_sample = header.bits_per_sample / 8;
  size_t num_samples = data_size / bytes_per_sample;
  result.samples.reserve (num_samples);

  if (header.audio_format == 1)
    { // PCM
      if (header.bits_per_sample == 8)
        {
          // 8-bit unsigned PCM
          const uint8_t *samples
              = reinterpret_cast<const uint8_t *> (raw_data.data ());
          for (size_t i = 0; i < num_samples; ++i)
            result.samples.push_back ((samples[i] - 128) / 128.0f);
        }
      else if (header.bits_per_sample == 16)
        {
          // 16-bit signed PCM
          const int16_t *samples
              = reinterpret_cast<const int16_t *> (raw_data.data ());
          for (size_t i = 0; i < num_samples; ++i)
            result.samples.push_back (samples[i] / 32768.0f);
        }
      else if (header.bits_per_sample == 24)
        {
          // 24-bit signed PCM (packed)
          for (size_t i = 0; i < num_samples; ++i)
            {
              size_t offset = i * 3;
              int32_t sample = raw_data[offset] | (raw_data[offset + 1] << 8)
                               | (raw_data[offset + 2] << 16);
              // Sign extend
              if (sample & 0x800000)
                sample |= 0xFF000000;
              result.samples.push_back (sample / 8388608.0f);
            }
        }
      else if (header.bits_per_sample == 32)
        {
          // 32-bit signed PCM
          const int32_t *samples
              = reinterpret_cast<const int32_t *> (raw_data.data ());
          for (size_t i = 0; i < num_samples; ++i)
            result.samples.push_back (samples[i] / 2147483648.0f);
        }
      else
        {
          return std::nullopt; // Unsupported bit depth
        }
    }
  else if (header.audio_format == 3)
    { // IEEE float
      if (header.bits_per_sample == 32)
        {
          const float *samples
              = reinterpret_cast<const float *> (raw_data.data ());
          result.samples.assign (samples, samples + num_samples);
        }
      else if (header.bits_per_sample == 64)
        {
          const double *samples
              = reinterpret_cast<const double *> (raw_data.data ());
          for (size_t i = 0; i < num_samples; ++i)
            result.samples.push_back (static_cast<float> (samples[i]));
        }
      else
        {
          return std::nullopt;
        }
    }
  else
    {
      return std::nullopt; // Unsupported format
    }

  result.duration_seconds = static_cast<double> (result.SamplesPerChannel ())
                            / static_cast<double> (result.sample_rate);

  return result;
}

std::optional<AudioData>
LoadPcmFloat32 (const std::string &path, uint32_t sample_rate,
                uint16_t num_channels)
{
  std::ifstream file (path, std::ios::binary | std::ios::ate);
  if (!file.is_open ())
    return std::nullopt;

  auto file_size = file.tellg ();
  file.seekg (0, std::ios::beg);

  size_t num_samples = static_cast<size_t> (file_size) / sizeof (float);

  AudioData result;
  result.sample_rate = sample_rate;
  result.num_channels = num_channels;
  result.samples.resize (num_samples);

  file.read (reinterpret_cast<char *> (result.samples.data ()),
             static_cast<std::streamsize> (file_size));

  result.duration_seconds = static_cast<double> (result.SamplesPerChannel ())
                            / static_cast<double> (result.sample_rate);

  return result;
}

std::optional<AudioData>
LoadPcmInt16 (const std::string &path, uint32_t sample_rate,
              uint16_t num_channels)
{
  std::ifstream file (path, std::ios::binary | std::ios::ate);
  if (!file.is_open ())
    return std::nullopt;

  auto file_size = file.tellg ();
  file.seekg (0, std::ios::beg);

  size_t num_samples = static_cast<size_t> (file_size) / sizeof (int16_t);
  std::vector<int16_t> raw_samples (num_samples);

  file.read (reinterpret_cast<char *> (raw_samples.data ()),
             static_cast<std::streamsize> (file_size));

  AudioData result;
  result.sample_rate = sample_rate;
  result.num_channels = num_channels;
  result.samples.reserve (num_samples);

  for (int16_t sample : raw_samples)
    result.samples.push_back (sample / 32768.0f);

  result.duration_seconds = static_cast<double> (result.SamplesPerChannel ())
                            / static_cast<double> (result.sample_rate);

  return result;
}

AudioData
ResampleTo (const AudioData &audio, uint32_t target_rate)
{
  if (audio.sample_rate == target_rate)
    return audio;

  double ratio = static_cast<double> (target_rate)
                 / static_cast<double> (audio.sample_rate);
  size_t new_samples_per_channel
      = static_cast<size_t> (audio.SamplesPerChannel () * ratio);

  AudioData result;
  result.sample_rate = target_rate;
  result.num_channels = audio.num_channels;
  result.samples.resize (new_samples_per_channel * audio.num_channels);

  // Linear interpolation resampling
  for (uint16_t ch = 0; ch < audio.num_channels; ++ch)
    {
      for (size_t i = 0; i < new_samples_per_channel; ++i)
        {
          double src_idx = static_cast<double> (i) / ratio;
          size_t idx0 = static_cast<size_t> (src_idx);
          size_t idx1 = std::min (idx0 + 1, audio.SamplesPerChannel () - 1);
          double frac = src_idx - static_cast<double> (idx0);

          float sample0 = audio.samples[idx0 * audio.num_channels + ch];
          float sample1 = audio.samples[idx1 * audio.num_channels + ch];
          float interpolated
              = sample0 + static_cast<float> (frac) * (sample1 - sample0);

          result.samples[i * audio.num_channels + ch] = interpolated;
        }
    }

  result.duration_seconds = static_cast<double> (new_samples_per_channel)
                            / static_cast<double> (target_rate);

  return result;
}

AudioData
ConvertToMono (const AudioData &audio)
{
  if (audio.num_channels == 1)
    return audio;

  AudioData result;
  result.sample_rate = audio.sample_rate;
  result.num_channels = 1;
  result.samples.reserve (audio.SamplesPerChannel ());

  for (size_t i = 0; i < audio.SamplesPerChannel (); ++i)
    {
      float sum = 0.0f;
      for (uint16_t ch = 0; ch < audio.num_channels; ++ch)
        sum += audio.samples[i * audio.num_channels + ch];
      result.samples.push_back (sum / static_cast<float> (audio.num_channels));
    }

  result.duration_seconds = audio.duration_seconds;

  return result;
}

AudioData
NormalizeTo16kMono (const AudioData &audio)
{
  AudioData result = audio;

  if (result.num_channels > 1)
    result = ConvertToMono (result);

  if (result.sample_rate != 16000)
    result = ResampleTo (result, 16000);

  return result;
}

std::string
DetectAudioFormat (const std::string &path)
{
  size_t dot_pos = path.rfind ('.');
  if (dot_pos == std::string::npos)
    return "";

  std::string ext = path.substr (dot_pos + 1);
  std::transform (ext.begin (), ext.end (), ext.begin (),
                  [] (unsigned char c) { return std::tolower (c); });

  if (ext == "wav")
    return "wav";
  if (ext == "pcm")
    return "pcm";

  return "";
}

std::optional<AudioData>
LoadAudio (const std::string &path)
{
  std::string format = DetectAudioFormat (path);

  if (format == "wav")
    return LoadWav (path);

  if (format == "pcm")
    {
      // Try float32 first, fall back to int16
      auto result = LoadPcmFloat32 (path);
      if (result)
        return result;
      return LoadPcmInt16 (path);
    }

  return std::nullopt;
}

} // namespace benchmark
