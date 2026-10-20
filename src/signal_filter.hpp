#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <utility>
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

  /// Change detection compares an item only with earlier items of the same
  /// modality from the same source: a camera or microphone feed is one
  /// stream, while an item sent under its own source id (a photo or clip
  /// attached to one turn) starts a new stream and is admitted as its first
  /// item.
  SignalFilterDecision EvaluateAudio (const std::string &source_id,
                                      const float *pcm, std::size_t num_samples,
                                      int sample_rate);
  SignalFilterDecision EvaluateImage (const std::string &source_id,
                                      const std::uint8_t *data, int width,
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

  /// Per-source stream state, most recently used first; bounded so unique
  /// per-item source ids cannot grow filter state. An evicted source
  /// restarts as a new stream.
  template <typename State> class SourceStreams
  {
  public:
    static constexpr std::size_t kMaxStreams = 32;

    State &
    For (const std::string &source_id)
    {
      for (auto it = streams_.begin (); it != streams_.end (); ++it)
        {
          if (it->first == source_id)
            {
              streams_.splice (streams_.begin (), streams_, it);
              return streams_.front ().second;
            }
        }
      if (streams_.size () >= kMaxStreams)
        {
          streams_.pop_back ();
        }
      streams_.emplace_front (source_id, State{});
      return streams_.front ().second;
    }

  private:
    std::list<std::pair<std::string, State>> streams_;
  };

  SignalFilterConfig config_;
  SourceStreams<AudioState> audio_streams_;
  SourceStreams<ImageState> image_streams_;
};

} // namespace cortext::internal
