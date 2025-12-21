#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cortext
{

struct SherpaOnnxAsrConfig
{
  std::string model_type;  // e.g. "whisper", "paraformer", "zipformer2_ctc"
  std::string model;
  std::string encoder;
  std::string decoder;
  std::string joiner;
  std::string tokens;
  std::string bpe_vocab;
  std::string language;
  int32_t sample_rate = 16000;
  int32_t num_threads = 1;
  std::string provider = "cpu";
};

struct SherpaOnnxTtsConfig
{
  std::string model_type;  // e.g. "vits", "matcha", "kokoro", "kitten"
  std::string model;
  std::string encoder;
  std::string decoder;
  std::string tokens;
  std::string lexicon;
  std::string data_dir;
  std::string voices;
  std::string vocoder;
  std::string language;
  int32_t num_threads = 1;
  std::string provider = "cpu";
  float speed = 1.0f;
};

struct SherpaOnnxAudio
{
  std::vector<float> samples;
  int32_t sample_rate = 0;
};

class SherpaOnnxOfflineAsr
{
public:
  explicit SherpaOnnxOfflineAsr (SherpaOnnxAsrConfig config);
  ~SherpaOnnxOfflineAsr ();

  std::string Transcribe (const float *pcm, std::size_t num_samples,
                          int32_t sample_rate);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class SherpaOnnxOfflineTts
{
public:
  explicit SherpaOnnxOfflineTts (SherpaOnnxTtsConfig config);
  ~SherpaOnnxOfflineTts ();

  SherpaOnnxAudio Synthesize (const std::string &text, int32_t speaker_id = 0);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
