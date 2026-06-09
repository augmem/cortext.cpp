#include "signal_filter.hpp"

#include "cortext/core/knobs.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace cortext::internal
{
namespace
{

double
Clamp01 (double value)
{
  return std::clamp (value, 0.0, 1.0);
}

double
Entropy01 (const double *values, std::size_t count)
{
  const double sum = std::accumulate (values, values + count, 0.0);
  if (sum <= 0.0 || count < 2)
    {
      return 0.0;
    }
  double entropy = 0.0;
  for (std::size_t i = 0; i < count; ++i)
    {
      if (values[i] > 0.0)
        {
          const double p = values[i] / sum;
          entropy -= p * std::log (p);
        }
    }
  return std::clamp (entropy / std::log (static_cast<double> (count)), 0.0,
                     1.0);
}

float
LumaAt (const std::uint8_t *data, int width, int channels, int x, int y)
{
  const std::size_t offset
      = (static_cast<std::size_t> (y) * static_cast<std::size_t> (width)
         + static_cast<std::size_t> (x))
        * static_cast<std::size_t> (channels);
  const float r = static_cast<float> (data[offset]);
  const float g = static_cast<float> (data[offset + 1]);
  const float b = static_cast<float> (data[offset + 2]);
  return (0.2126f * r + 0.7152f * g + 0.0722f * b) / 255.0f;
}

} // namespace

SignalFilter::SignalFilter (SignalFilterConfig config)
    : config_ (std::move (config))
{
  config_.focus = Clamp01 (config_.focus);
  config_.sensitivity = Clamp01 (config_.sensitivity);
  config_.stability = Clamp01 (config_.stability);
}

SignalFilterDecision
SignalFilter::DisabledDecision (const std::string &modality) const
{
  SignalFilterDecision decision;
  decision.evaluated = false;
  decision.accepted = true;
  decision.modality = modality;
  decision.reason = "disabled";
  return decision;
}

SignalFilterDecision
SignalFilter::NoRegisteredFilterDecision (const std::string &modality) const
{
  SignalFilterDecision decision;
  decision.evaluated = true;
  decision.accepted = true;
  decision.modality = modality;
  decision.reason = "no_registered_filter";
  decision.score = 1.0;
  return decision;
}

SignalFilterDecision
SignalFilter::DecideAdaptive (const std::string &modality, double mean_delta,
                              double max_local_delta, double entropy,
                              double velocity, double current_seconds,
                              int current_index, bool &initialized,
                              int &last_accept_index,
                              double &last_accept_seconds,
                              double &ema_delta,
                              double &ema_abs_dev,
                              double min_accept_seconds) const
{
  const auto policy = core::SignalFilterAdaptivePolicyForKnobs (
      config_.focus, config_.sensitivity, config_.stability);
  const double alpha = policy.alpha;
  if (!initialized)
    {
      ema_delta = mean_delta;
      ema_abs_dev = 0.0;
      initialized = true;
    }
  const double dev = std::abs (mean_delta - ema_delta);
  ema_delta = (1.0 - alpha) * ema_delta + alpha * mean_delta;
  ema_abs_dev = (1.0 - alpha) * ema_abs_dev + alpha * dev;

  SignalFilterDecision decision;
  decision.evaluated = true;
  decision.accepted = false;
  decision.modality = modality;
  decision.reason = "reject";
  decision.quiet_seconds
      = last_accept_index < 0 ? std::numeric_limits<double>::infinity ()
                              : current_seconds - last_accept_seconds;

  const bool settling
      = velocity
        < -policy.settling_velocity_scale * std::max (ema_delta, 1e-6);
  const double quiet_release = std::clamp (
      decision.quiet_seconds / policy.heartbeat_seconds, 0.0, 1.0);
  const double settling_release = settling ? policy.settling_release : 0.0;

  double threshold = policy.base_threshold;
  threshold += ema_delta * policy.focus_pressure;
  threshold += ema_abs_dev * policy.ema_abs_dev_weight;
  threshold += entropy * ema_delta * policy.entropy_weight;
  threshold *= policy.sensitivity_release * policy.stability_pressure;
  threshold *= (1.0 - policy.quiet_release_weight * quiet_release);
  threshold *= (1.0 - settling_release);
  threshold = std::clamp (threshold, policy.min_threshold,
                          policy.max_threshold);

  const double score = policy.mean_score_weight * mean_delta
                       + policy.max_score_weight * max_local_delta;
  decision.threshold = threshold;
  decision.score = score;

  if (last_accept_index < 0)
    {
      decision.accepted = true;
      decision.reason = "first_item";
    }
  else if (decision.quiet_seconds >= policy.heartbeat_seconds)
    {
      decision.accepted = true;
      decision.reason = "heartbeat";
    }
  else if (decision.quiet_seconds < min_accept_seconds)
    {
      decision.reason = "min_interval";
    }
  else if (score >= threshold)
    {
      decision.accepted = true;
      decision.reason = settling ? "settling_delta" : "adaptive_delta";
    }

  if (decision.accepted)
    {
      last_accept_index = current_index;
      last_accept_seconds = current_seconds;
    }
  return decision;
}

SignalFilterDecision
SignalFilter::EvaluateAudio (const float *pcm, std::size_t num_samples,
                             int sample_rate)
{
  if (!config_.audio_enabled)
    {
      return DisabledDecision ("audio");
    }

  constexpr std::size_t kBins = 32;
  double bins[kBins] = {};
  int counts[kBins] = {};
  if (pcm && num_samples > 0)
    {
      for (std::size_t i = 0; i < num_samples; ++i)
        {
          const std::size_t bin = std::min (
              (i * kBins) / num_samples, static_cast<std::size_t> (kBins - 1));
          bins[bin] += static_cast<double> (pcm[i]) * pcm[i];
          ++counts[bin];
        }
      for (std::size_t i = 0; i < kBins; ++i)
        {
          if (counts[i] > 0)
            {
              bins[i] = std::sqrt (bins[i] / static_cast<double> (counts[i]));
            }
        }
    }

  double deltas[kBins] = {};
  double mean_delta = 0.0;
  double max_delta = 0.0;
  if (audio_.has_previous_bins)
    {
      for (std::size_t i = 0; i < kBins; ++i)
        {
          deltas[i] = std::abs (bins[i] - audio_.previous_bins[i]);
          mean_delta += deltas[i];
          max_delta = std::max (max_delta, deltas[i]);
        }
      mean_delta /= static_cast<double> (kBins);
    }
  const double entropy = Entropy01 (deltas, kBins);
  const double velocity = mean_delta - audio_.previous_delta;
  const double chunk_seconds
      = sample_rate > 0 ? static_cast<double> (num_samples)
                              / static_cast<double> (sample_rate)
                        : 0.0;
  const double current_seconds = audio_.stream_seconds;

  auto decision = DecideAdaptive (
      "audio", mean_delta, max_delta, entropy, velocity, current_seconds,
      audio_.index, audio_.initialized, audio_.last_accept_index,
      audio_.last_accept_seconds, audio_.ema_delta, audio_.ema_abs_dev,
      core::SignalFilterAudioMinAcceptSeconds (
          config_.focus, config_.sensitivity, config_.stability));
  std::copy (bins, bins + kBins, audio_.previous_bins);
  audio_.has_previous_bins = true;
  audio_.previous_delta = mean_delta;
  audio_.stream_seconds += chunk_seconds;
  ++audio_.index;
  return decision;
}

SignalFilterDecision
SignalFilter::EvaluateImage (const std::uint8_t *data, int width, int height,
                             int channels)
{
  if (!config_.image_enabled)
    {
      return DisabledDecision ("image");
    }
  if (!data || width <= 0 || height <= 0 || channels < 3)
    {
      SignalFilterDecision decision;
      decision.evaluated = true;
      decision.accepted = false;
      decision.modality = "image";
      decision.reason = "invalid_input";
      return decision;
    }

  constexpr int kGridX = 32;
  constexpr int kGridY = 24;
  constexpr int kSamplesPerAxis = 4;
  constexpr double kAssumedVideoFps = 30.0;
  constexpr std::size_t kBlockCount
      = static_cast<std::size_t> (kGridX * kGridY);
  if (image_.width != width || image_.height != height
      || image_.channels != channels
      || image_.previous_blocks.size () != kBlockCount)
    {
      image_.width = width;
      image_.height = height;
      image_.channels = channels;
      image_.previous_blocks.assign (kBlockCount, 0.0);
      image_.has_previous_blocks = false;
      image_.initialized = false;
      image_.previous_delta = 0.0;
    }

  double blocks[kBlockCount] = {};
  for (int gy = 0; gy < kGridY; ++gy)
    {
      const int y0 = gy * height / kGridY;
      const int y1 = std::max (y0 + 1, (gy + 1) * height / kGridY);
      for (int gx = 0; gx < kGridX; ++gx)
        {
          const int x0 = gx * width / kGridX;
          const int x1 = std::max (x0 + 1, (gx + 1) * width / kGridX);
          double sum = 0.0;
          int count = 0;
          for (int sy = 0; sy < kSamplesPerAxis; ++sy)
            {
              const int y = std::clamp (
                  y0 + ((2 * sy + 1) * (y1 - y0)) / (2 * kSamplesPerAxis),
                  0, height - 1);
              for (int sx = 0; sx < kSamplesPerAxis; ++sx)
                {
                  const int x = std::clamp (
                      x0 + ((2 * sx + 1) * (x1 - x0))
                               / (2 * kSamplesPerAxis),
                      0, width - 1);
                  sum += static_cast<double> (LumaAt (data, width, channels, x, y));
                  ++count;
                }
            }
          blocks[static_cast<std::size_t> (gy * kGridX + gx)]
              = count > 0 ? sum / static_cast<double> (count) : 0.0;
        }
    }

  double deltas[kBlockCount] = {};
  double mean_delta = 0.0;
  double max_delta = 0.0;
  if (image_.has_previous_blocks)
    {
      for (std::size_t i = 0; i < kBlockCount; ++i)
        {
          deltas[i] = std::abs (blocks[i] - image_.previous_blocks[i]);
          mean_delta += deltas[i];
          max_delta = std::max (max_delta, deltas[i]);
        }
      mean_delta /= static_cast<double> (kBlockCount);
    }
  const double entropy = Entropy01 (deltas, kBlockCount);
  const double velocity = mean_delta - image_.previous_delta;
  const double current_seconds = image_.stream_seconds;
  auto decision = DecideAdaptive (
      "image", mean_delta, max_delta, entropy, velocity, current_seconds,
      image_.index, image_.initialized, image_.last_accept_index,
      image_.last_accept_seconds, image_.ema_delta, image_.ema_abs_dev,
      core::SignalFilterImageMinAcceptSeconds (
          config_.focus, config_.sensitivity, config_.stability));

  image_.previous_blocks.assign (blocks, blocks + kBlockCount);
  image_.has_previous_blocks = true;
  image_.previous_delta = mean_delta;
  image_.stream_seconds += 1.0 / kAssumedVideoFps;
  ++image_.index;
  return decision;
}

SignalFilterDecision
SignalFilter::EvaluateText (const std::string & /*text*/)
{
  if (!config_.text_enabled)
    {
      return DisabledDecision ("text");
    }
  return NoRegisteredFilterDecision ("text");
}

} // namespace cortext::internal
