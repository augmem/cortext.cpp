#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cortext::internal
{

struct SignalFilterConfig
{
  double focus = 0.5;
  double sensitivity = 0.5;
  double stability = 0.5;
  bool audio_enabled = true;
  bool image_enabled = false;
  bool text_enabled = false;
};

struct SignalFilterDecision
{
  bool evaluated = false;
  bool accepted = true;
  std::string modality;
  std::string reason;
  double score = 0.0;
  double threshold = 0.0;
  double quiet_seconds = 0.0;
};

class SignalFilter
{
public:
  explicit SignalFilter (SignalFilterConfig config);

  SignalFilterDecision EvaluateAudio (const float *pcm, std::size_t num_samples,
                                      int sample_rate);
  SignalFilterDecision EvaluateImage (const std::uint8_t *data, int width,
                                      int height, int channels);
  SignalFilterDecision EvaluateText (const std::string &text);

private:
  struct AudioState
  {
    bool initialized = false;
    bool has_previous_bins = false;
    int last_accept_index = -1;
    int index = 0;
    double stream_seconds = 0.0;
    double last_accept_seconds = 0.0;
    double ema_delta = 0.0;
    double ema_abs_dev = 0.0;
    double previous_delta = 0.0;
    double previous_bins[32] = {};
  };

  struct ImageState
  {
    bool initialized = false;
    bool has_previous_blocks = false;
    int last_accept_index = -1;
    int index = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    double last_accept_seconds = 0.0;
    double stream_seconds = 0.0;
    double ema_delta = 0.0;
    double ema_abs_dev = 0.0;
    double previous_delta = 0.0;
    std::vector<double> previous_blocks;
  };

  SignalFilterDecision DisabledDecision (const std::string &modality) const;
  SignalFilterDecision NoRegisteredFilterDecision (
      const std::string &modality) const;
  SignalFilterDecision DecideAdaptive (const std::string &modality,
                                       double mean_delta,
                                       double max_local_delta,
                                       double entropy, double velocity,
                                       double current_seconds,
                                       int current_index, bool &initialized,
                                       int &last_accept_index,
                                       double &last_accept_seconds,
                                       double &ema_delta,
                                       double &ema_abs_dev,
                                       double min_accept_seconds) const;

  SignalFilterConfig config_;
  AudioState audio_;
  ImageState image_;
};

} // namespace cortext::internal
