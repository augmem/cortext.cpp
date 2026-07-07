#include <cortext/cortext.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

using Json = nlohmann::json;

struct Options
{
  std::string modality = "image";
  std::filesystem::path frames_path;
  std::filesystem::path audio_path;
  std::filesystem::path text_path;
  std::filesystem::path output_dir = "build/signal_filter_bench";
  int width = 640;
  int height = 480;
  int channels = 3;
  double fps = 30.0;
  int sample_rate = 16000;
  int chunk_ms = 1000;
  double text_step_seconds = 1.0;
  int max_items = 0;
  int grid_x = 32;
  int grid_y = 24;
  double fixed_threshold = 0.020;
  double base_threshold = 0.012;
  double heartbeat_seconds = 2.0;
  double min_accept_seconds = 0.10;
  double focus = 0.5;
  double sensitivity = 0.5;
  double stability = 0.5;
  bool process_accepted = false;
  std::string process_policy = "adaptive";
};

struct SignalFeatures
{
  int index = 0;
  double time_seconds = 0.0;
  double mean_delta = 0.0;
  double max_local_delta = 0.0;
  double entropy = 0.0;
  double velocity = 0.0;
  double energy = 0.0;
};

struct SignalFilterDecision
{
  bool accepted = false;
  std::string reason = "reject";
  double threshold = 0.0;
  double quiet_seconds = 0.0;
  double score = 0.0;
};

struct SignalFilterState
{
  bool initialized = false;
  int last_accept_index = -1;
  double last_accept_time = 0.0;
  double ema_delta = 0.0;
  double ema_abs_dev = 0.0;
};

struct SignalFilterStats
{
  int items = 0;
  int accepted = 0;
  double total_delta_sum = 0.0;
  double accepted_delta_sum = 0.0;
  double max_gap_seconds = 0.0;
  std::vector<double> accepted_times;
};

struct ModalityRun
{
  std::string modality;
  double media_seconds = 0.0;
  std::vector<SignalFeatures> features;
  std::vector<SignalFilterDecision> all_decisions;
  std::vector<SignalFilterDecision> fixed_decisions;
  std::vector<SignalFilterDecision> adaptive_decisions;
  Json summary;
  Json rows;
};

struct ProcessStats
{
  std::string modality;
  int processed = 0;
  double wall_seconds = 0.0;
  std::vector<double> total_ms;
  std::vector<double> encode_ms;
  std::vector<double> process_ms;
  std::vector<double> hydrate_ms;
};

struct AudioEvent
{
  int event_id = 0;
  int start_index = 0;
  int end_index = 0;
  double start_seconds = 0.0;
  double end_seconds = 0.0;
  double peak_seconds = 0.0;
  double peak_score = 0.0;
};

void
PrintUsage ()
{
  std::cout
      << "Usage: cortext_signal_filter_bench --modality image|audio|text|all [options]\n"
      << "  --frames <rgb.raw>            Image RGB/RGBA frame stream\n"
      << "  --audio <f32le.raw>           Audio float32 mono stream\n"
      << "  --text-lines <path>           UTF-8 newline-delimited text stream\n"
      << "  --width <n> --height <n> --channels <3|4> --fps <value>\n"
      << "  --sample-rate <n> --chunk-ms <n>\n"
      << "  --text-step-seconds <value>\n"
      << "  --max-items <n>\n"
      << "  --output-dir <path>\n"
      << "  --fixed-threshold <value>\n"
      << "  --base-threshold <value>\n"
      << "  --heartbeat-seconds <value>\n"
      << "  --focus <0..1> --sensitivity <0..1> --stability <0..1>\n"
      << "  --process-accepted\n"
      << "  --process-policy all|fixed|adaptive\n";
}

double
ParseDouble (const char *value)
{
  char *end = nullptr;
  const double parsed = std::strtod (value, &end);
  if (end == value || (end != nullptr && *end != '\0'))
    {
      throw std::runtime_error (std::string ("Invalid number: ") + value);
    }
  return parsed;
}

int
ParseInt (const char *value)
{
  char *end = nullptr;
  const long parsed = std::strtol (value, &end, 10);
  if (end == value || (end != nullptr && *end != '\0'))
    {
      throw std::runtime_error (std::string ("Invalid integer: ") + value);
    }
  return static_cast<int> (parsed);
}

Options
ParseArgs (int argc, char *argv[])
{
  Options opts;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if (arg == "--modality" && i + 1 < argc)
        opts.modality = argv[++i];
      else if (arg == "--frames" && i + 1 < argc)
        opts.frames_path = argv[++i];
      else if (arg == "--audio" && i + 1 < argc)
        opts.audio_path = argv[++i];
      else if (arg == "--text-lines" && i + 1 < argc)
        opts.text_path = argv[++i];
      else if (arg == "--output-dir" && i + 1 < argc)
        opts.output_dir = argv[++i];
      else if (arg == "--width" && i + 1 < argc)
        opts.width = ParseInt (argv[++i]);
      else if (arg == "--height" && i + 1 < argc)
        opts.height = ParseInt (argv[++i]);
      else if (arg == "--channels" && i + 1 < argc)
        opts.channels = ParseInt (argv[++i]);
      else if (arg == "--fps" && i + 1 < argc)
        opts.fps = ParseDouble (argv[++i]);
      else if (arg == "--sample-rate" && i + 1 < argc)
        opts.sample_rate = ParseInt (argv[++i]);
      else if (arg == "--chunk-ms" && i + 1 < argc)
        opts.chunk_ms = ParseInt (argv[++i]);
      else if (arg == "--text-step-seconds" && i + 1 < argc)
        opts.text_step_seconds = ParseDouble (argv[++i]);
      else if (arg == "--max-items" && i + 1 < argc)
        opts.max_items = ParseInt (argv[++i]);
      else if (arg == "--grid-x" && i + 1 < argc)
        opts.grid_x = ParseInt (argv[++i]);
      else if (arg == "--grid-y" && i + 1 < argc)
        opts.grid_y = ParseInt (argv[++i]);
      else if (arg == "--fixed-threshold" && i + 1 < argc)
        opts.fixed_threshold = ParseDouble (argv[++i]);
      else if (arg == "--base-threshold" && i + 1 < argc)
        opts.base_threshold = ParseDouble (argv[++i]);
      else if (arg == "--heartbeat-seconds" && i + 1 < argc)
        opts.heartbeat_seconds = ParseDouble (argv[++i]);
      else if (arg == "--min-accept-seconds" && i + 1 < argc)
        opts.min_accept_seconds = ParseDouble (argv[++i]);
      else if (arg == "--focus" && i + 1 < argc)
        opts.focus = ParseDouble (argv[++i]);
      else if (arg == "--sensitivity" && i + 1 < argc)
        opts.sensitivity = ParseDouble (argv[++i]);
      else if (arg == "--stability" && i + 1 < argc)
        opts.stability = ParseDouble (argv[++i]);
      else if (arg == "--process-accepted")
        opts.process_accepted = true;
      else if (arg == "--process-policy" && i + 1 < argc)
        opts.process_policy = argv[++i];
      else if (arg == "--help" || arg == "-h")
        {
          PrintUsage ();
          std::exit (0);
        }
      else
        throw std::runtime_error ("Unknown argument: " + arg);
    }

  if (opts.modality != "image" && opts.modality != "audio"
      && opts.modality != "text" && opts.modality != "all")
    throw std::runtime_error ("--modality must be image, audio, text, or all");
  if ((opts.modality == "image" || opts.modality == "all")
      && opts.frames_path.empty ())
    throw std::runtime_error ("--frames is required for image/all");
  if ((opts.modality == "audio" || opts.modality == "all")
      && opts.audio_path.empty ())
    throw std::runtime_error ("--audio is required for audio/all");
  if ((opts.modality == "text" || opts.modality == "all")
      && opts.text_path.empty ())
    throw std::runtime_error ("--text-lines is required for text/all");
  if (opts.width <= 0 || opts.height <= 0 || opts.channels < 3
      || opts.fps <= 0.0 || opts.sample_rate <= 0 || opts.chunk_ms <= 0
      || opts.text_step_seconds <= 0.0)
    throw std::runtime_error ("Invalid stream dimensions or timing");
  opts.grid_x = std::clamp (opts.grid_x, 1, opts.width);
  opts.grid_y = std::clamp (opts.grid_y, 1, opts.height);
  opts.focus = std::clamp (opts.focus, 0.0, 1.0);
  opts.sensitivity = std::clamp (opts.sensitivity, 0.0, 1.0);
  opts.stability = std::clamp (opts.stability, 0.0, 1.0);
  if (opts.process_policy != "all" && opts.process_policy != "fixed"
      && opts.process_policy != "adaptive")
    throw std::runtime_error ("--process-policy must be all, fixed, or adaptive");
  return opts;
}

double
ElapsedSeconds (std::chrono::steady_clock::time_point start,
                std::chrono::steady_clock::time_point end)
{
  return std::chrono::duration<double> (end - start).count ();
}

double
Percentile (std::vector<double> values, double q)
{
  if (values.empty ())
    return 0.0;
  std::sort (values.begin (), values.end ());
  const double pos = static_cast<double> (values.size () - 1) * q;
  const auto lo = static_cast<std::size_t> (std::floor (pos));
  const auto hi = static_cast<std::size_t> (std::ceil (pos));
  if (lo == hi)
    return values[lo];
  return values[lo] * (static_cast<double> (hi) - pos)
         + values[hi] * (pos - static_cast<double> (lo));
}

double
Entropy01 (const std::vector<double> &values)
{
  const double sum = std::accumulate (values.begin (), values.end (), 0.0);
  if (sum <= 0.0 || values.size () < 2)
    return 0.0;
  double entropy = 0.0;
  for (double value : values)
    {
      if (value > 0.0)
        {
          const double p = value / sum;
          entropy -= p * std::log (p);
        }
    }
  return std::clamp (entropy / std::log (static_cast<double> (values.size ())),
                     0.0, 1.0);
}

float
LumaAt (const std::vector<std::uint8_t> &frame, int width, int channels, int x,
        int y)
{
  const std::size_t offset
      = (static_cast<std::size_t> (y) * static_cast<std::size_t> (width)
         + static_cast<std::size_t> (x))
        * static_cast<std::size_t> (channels);
  const float r = static_cast<float> (frame[offset]);
  const float g = static_cast<float> (frame[offset + 1]);
  const float b = static_cast<float> (frame[offset + 2]);
  return (0.2126f * r + 0.7152f * g + 0.0722f * b) / 255.0f;
}

SignalFeatures
ImageDelta (const std::vector<std::uint8_t> &prev,
            const std::vector<std::uint8_t> &cur, const Options &opts,
            int index, double previous_delta)
{
  std::vector<double> blocks (
      static_cast<std::size_t> (opts.grid_x * opts.grid_y), 0.0);
  std::vector<int> counts (blocks.size (), 0);
  double total = 0.0;
  int count = 0;
  for (int gy = 0; gy < opts.grid_y; ++gy)
    {
      const int y0 = gy * opts.height / opts.grid_y;
      const int y1 = (gy + 1) * opts.height / opts.grid_y;
      for (int gx = 0; gx < opts.grid_x; ++gx)
        {
          const int x0 = gx * opts.width / opts.grid_x;
          const int x1 = (gx + 1) * opts.width / opts.grid_x;
          const std::size_t bi = static_cast<std::size_t> (gy * opts.grid_x + gx);
          for (int y = y0; y < y1; y += 2)
            {
              for (int x = x0; x < x1; x += 2)
                {
                  const double diff = std::abs (
                      static_cast<double> (LumaAt (cur, opts.width,
                                                   opts.channels, x, y))
                      - static_cast<double> (LumaAt (prev, opts.width,
                                                     opts.channels, x, y)));
                  blocks[bi] += diff;
                  ++counts[bi];
                  total += diff;
                  ++count;
                }
            }
        }
    }
  for (std::size_t i = 0; i < blocks.size (); ++i)
    if (counts[i] > 0)
      blocks[i] /= static_cast<double> (counts[i]);
  const double mean = count > 0 ? total / static_cast<double> (count) : 0.0;
  SignalFeatures f;
  f.index = index;
  f.time_seconds = static_cast<double> (index) / opts.fps;
  f.mean_delta = mean;
  f.max_local_delta = *std::max_element (blocks.begin (), blocks.end ());
  f.entropy = Entropy01 (blocks);
  f.velocity = mean - previous_delta;
  f.energy = mean;
  return f;
}

std::vector<SignalFeatures>
LoadImageFeatures (const Options &opts)
{
  const std::size_t frame_size
      = static_cast<std::size_t> (opts.width) * static_cast<std::size_t> (opts.height)
        * static_cast<std::size_t> (opts.channels);
  int count = static_cast<int> (std::filesystem::file_size (opts.frames_path)
                                / frame_size);
  if (opts.max_items > 0)
    count = std::min (count, opts.max_items);
  if (count <= 0)
    throw std::runtime_error ("No complete image frames");
  std::ifstream input (opts.frames_path, std::ios::binary);
  std::vector<std::uint8_t> prev (frame_size);
  std::vector<std::uint8_t> cur (frame_size);
  input.read (reinterpret_cast<char *> (prev.data ()),
              static_cast<std::streamsize> (frame_size));
  std::vector<SignalFeatures> features;
  features.reserve (static_cast<std::size_t> (count));
  features.push_back ({ 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
  double previous_delta = 0.0;
  for (int i = 1; i < count; ++i)
    {
      input.read (reinterpret_cast<char *> (cur.data ()),
                  static_cast<std::streamsize> (frame_size));
      if (input.gcount () != static_cast<std::streamsize> (frame_size))
        break;
      auto f = ImageDelta (prev, cur, opts, i, previous_delta);
      previous_delta = f.mean_delta;
      features.push_back (f);
      prev.swap (cur);
    }
  return features;
}

std::vector<float>
ReadF32File (const std::filesystem::path &path)
{
  std::ifstream input (path, std::ios::binary);
  if (!input)
    throw std::runtime_error ("Failed to open audio file");
  input.seekg (0, std::ios::end);
  const auto bytes = input.tellg ();
  input.seekg (0, std::ios::beg);
  std::vector<float> values (static_cast<std::size_t> (bytes) / sizeof (float));
  input.read (reinterpret_cast<char *> (values.data ()),
              static_cast<std::streamsize> (values.size () * sizeof (float)));
  return values;
}

std::vector<double>
AudioBins (const std::vector<float> &audio, std::size_t begin, std::size_t end)
{
  constexpr int kBins = 32;
  std::vector<double> bins (kBins, 0.0);
  std::vector<int> counts (kBins, 0);
  if (end <= begin)
    return bins;
  for (std::size_t i = begin; i < end; ++i)
    {
      const int bin = static_cast<int> ((i - begin) * kBins / (end - begin));
      bins[std::min (bin, kBins - 1)] += static_cast<double> (audio[i] * audio[i]);
      ++counts[std::min (bin, kBins - 1)];
    }
  for (int i = 0; i < kBins; ++i)
    if (counts[i] > 0)
      bins[i] = std::sqrt (bins[i] / static_cast<double> (counts[i]));
  return bins;
}

std::vector<SignalFeatures>
LoadAudioFeatures (const Options &opts)
{
  const auto audio = ReadF32File (opts.audio_path);
  const std::size_t chunk
      = static_cast<std::size_t> (opts.sample_rate * opts.chunk_ms / 1000);
  if (chunk == 0 || audio.empty ())
    throw std::runtime_error ("No complete audio chunks");
  int count = static_cast<int> ((audio.size () + chunk - 1) / chunk);
  if (opts.max_items > 0)
    count = std::min (count, opts.max_items);
  std::vector<SignalFeatures> features;
  features.reserve (static_cast<std::size_t> (count));
  std::vector<double> prev_bins;
  double previous_delta = 0.0;
  for (int i = 0; i < count; ++i)
    {
      const std::size_t begin = static_cast<std::size_t> (i) * chunk;
      const std::size_t end = std::min (audio.size (), begin + chunk);
      auto bins = AudioBins (audio, begin, end);
      double delta = 0.0;
      double max_delta = 0.0;
      std::vector<double> deltas (bins.size (), 0.0);
      if (!prev_bins.empty ())
        {
          for (std::size_t j = 0; j < bins.size (); ++j)
            {
              deltas[j] = std::abs (bins[j] - prev_bins[j]);
              delta += deltas[j];
              max_delta = std::max (max_delta, deltas[j]);
            }
          delta /= static_cast<double> (bins.size ());
        }
      const double energy = std::accumulate (bins.begin (), bins.end (), 0.0)
                            / static_cast<double> (bins.size ());
      SignalFeatures f;
      f.index = i;
      f.time_seconds = static_cast<double> (begin)
                       / static_cast<double> (opts.sample_rate);
      f.mean_delta = delta;
      f.max_local_delta = max_delta;
      f.entropy = Entropy01 (deltas);
      f.velocity = delta - previous_delta;
      f.energy = energy;
      features.push_back (f);
      previous_delta = delta;
      prev_bins = std::move (bins);
    }
  return features;
}

std::map<std::string, int>
TokenCounts (const std::string &text)
{
  std::map<std::string, int> counts;
  std::string token;
  for (unsigned char ch : text)
    {
      if (std::isalnum (ch))
        token.push_back (static_cast<char> (std::tolower (ch)));
      else if (!token.empty ())
        {
          ++counts[token];
          token.clear ();
        }
    }
  if (!token.empty ())
    ++counts[token];
  return counts;
}

double
CosineCounts (const std::map<std::string, int> &a,
              const std::map<std::string, int> &b)
{
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (const auto &[key, value] : a)
    {
      na += static_cast<double> (value * value);
      auto it = b.find (key);
      if (it != b.end ())
        dot += static_cast<double> (value * it->second);
    }
  for (const auto &[_, value] : b)
    nb += static_cast<double> (value * value);
  if (na <= 0.0 || nb <= 0.0)
    return 0.0;
  return dot / std::sqrt (na * nb);
}

std::vector<std::string>
ReadLines (const std::filesystem::path &path)
{
  std::ifstream input (path);
  if (!input)
    throw std::runtime_error ("Failed to open text lines file");
  std::vector<std::string> lines;
  std::string line;
  while (std::getline (input, line))
    if (!line.empty ())
      lines.push_back (line);
  return lines;
}

std::vector<SignalFeatures>
LoadTextFeatures (const Options &opts)
{
  auto lines = ReadLines (opts.text_path);
  if (opts.max_items > 0 && static_cast<int> (lines.size ()) > opts.max_items)
    lines.resize (static_cast<std::size_t> (opts.max_items));
  if (lines.empty ())
    throw std::runtime_error ("No text lines");
  std::vector<SignalFeatures> features;
  features.reserve (lines.size ());
  std::map<std::string, int> prev;
  double previous_delta = 0.0;
  for (std::size_t i = 0; i < lines.size (); ++i)
    {
      auto counts = TokenCounts (lines[i]);
      double delta = 0.0;
      double max_local = 0.0;
      if (!prev.empty ())
        {
          delta = 1.0 - CosineCounts (prev, counts);
          max_local = delta;
        }
      std::vector<double> token_weights;
      for (const auto &[_, value] : counts)
        token_weights.push_back (static_cast<double> (value));
      SignalFeatures f;
      f.index = static_cast<int> (i);
      f.time_seconds = static_cast<double> (i) * opts.text_step_seconds;
      f.mean_delta = std::clamp (delta, 0.0, 1.0);
      f.max_local_delta = max_local;
      f.entropy = Entropy01 (token_weights);
      f.velocity = f.mean_delta - previous_delta;
      f.energy = std::min (1.0, static_cast<double> (counts.size ()) / 16.0);
      features.push_back (f);
      previous_delta = f.mean_delta;
      prev = std::move (counts);
    }
  return features;
}

SignalFilterDecision
DecideAll (const SignalFeatures &features, SignalFilterState &state)
{
  SignalFilterDecision decision;
  decision.accepted = true;
  decision.reason = features.index == 0 ? "first_item" : "all_items";
  decision.score = 1.0;
  decision.quiet_seconds
      = state.last_accept_index < 0
            ? std::numeric_limits<double>::infinity ()
            : features.time_seconds - state.last_accept_time;
  state.last_accept_index = features.index;
  state.last_accept_time = features.time_seconds;
  return decision;
}

SignalFilterDecision
DecideFixed (const SignalFeatures &features, SignalFilterState &state,
             const Options &opts)
{
  SignalFilterDecision decision;
  decision.threshold = opts.fixed_threshold;
  decision.quiet_seconds
      = state.last_accept_index < 0
            ? std::numeric_limits<double>::infinity ()
            : features.time_seconds - state.last_accept_time;
  decision.score = std::max (features.mean_delta, 0.55 * features.max_local_delta);
  if (state.last_accept_index < 0)
    {
      decision.accepted = true;
      decision.reason = "first_item";
    }
  else if (decision.quiet_seconds >= opts.heartbeat_seconds)
    {
      decision.accepted = true;
      decision.reason = "heartbeat";
    }
  else if (decision.score >= decision.threshold)
    {
      decision.accepted = true;
      decision.reason = "fixed_delta";
    }
  if (decision.accepted)
    {
      state.last_accept_index = features.index;
      state.last_accept_time = features.time_seconds;
    }
  return decision;
}

SignalFilterDecision
DecideAdaptive (const SignalFeatures &features, SignalFilterState &state,
                const Options &opts)
{
  const double alpha = 0.08 + 0.12 * (1.0 - opts.stability);
  if (!state.initialized)
    {
      state.ema_delta = features.mean_delta;
      state.ema_abs_dev = 0.0;
      state.initialized = true;
    }
  const double dev = std::abs (features.mean_delta - state.ema_delta);
  state.ema_delta = (1.0 - alpha) * state.ema_delta + alpha * features.mean_delta;
  state.ema_abs_dev = (1.0 - alpha) * state.ema_abs_dev + alpha * dev;

  SignalFilterDecision decision;
  decision.quiet_seconds
      = state.last_accept_index < 0
            ? std::numeric_limits<double>::infinity ()
            : features.time_seconds - state.last_accept_time;
  const double focus_pressure = 0.75 + 0.70 * opts.focus;
  const double sensitivity_release = 1.10 - 0.45 * opts.sensitivity;
  const double stability_pressure = 0.85 + 0.40 * opts.stability;
  const bool settling
      = features.velocity < -0.15 * std::max (state.ema_delta, 1e-6);
  const double quiet_release = std::clamp (
      decision.quiet_seconds / std::max (opts.heartbeat_seconds, 1e-6), 0.0, 1.0);
  const double settling_release = settling ? (0.35 + 0.35 * opts.sensitivity) : 0.0;
  double threshold = opts.base_threshold;
  threshold += state.ema_delta * focus_pressure;
  threshold += state.ema_abs_dev * (0.75 + 0.40 * opts.focus);
  threshold += features.entropy * state.ema_delta * 0.40;
  threshold *= sensitivity_release * stability_pressure;
  threshold *= (1.0 - 0.35 * quiet_release);
  threshold *= (1.0 - settling_release);
  threshold = std::clamp (threshold, 0.0025, 0.50);
  const double score = 0.60 * features.mean_delta + 0.40 * features.max_local_delta;
  decision.threshold = threshold;
  decision.score = score;
  if (state.last_accept_index < 0)
    {
      decision.accepted = true;
      decision.reason = "first_item";
    }
  else if (decision.quiet_seconds >= opts.heartbeat_seconds)
    {
      decision.accepted = true;
      decision.reason = "heartbeat";
    }
  else if (decision.quiet_seconds < opts.min_accept_seconds)
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
      state.last_accept_index = features.index;
      state.last_accept_time = features.time_seconds;
    }
  return decision;
}

void
UpdateStats (SignalFilterStats &stats, const SignalFeatures &features,
             const SignalFilterDecision &decision)
{
  ++stats.items;
  stats.total_delta_sum += features.mean_delta;
  if (decision.accepted)
    {
      ++stats.accepted;
      stats.accepted_delta_sum += features.mean_delta;
      if (!stats.accepted_times.empty ())
        stats.max_gap_seconds = std::max (
            stats.max_gap_seconds, features.time_seconds - stats.accepted_times.back ());
      stats.accepted_times.push_back (features.time_seconds);
    }
}

Json
StatsJson (const SignalFilterStats &stats, double media_seconds)
{
  std::vector<double> gaps;
  for (std::size_t i = 1; i < stats.accepted_times.size (); ++i)
    gaps.push_back (stats.accepted_times[i] - stats.accepted_times[i - 1]);
  return Json{
    { "items", stats.items },
    { "accepted", stats.accepted },
    { "accept_rate", stats.items > 0
                         ? static_cast<double> (stats.accepted)
                               / static_cast<double> (stats.items)
                         : 0.0 },
    { "accepted_per_second",
      media_seconds > 0.0 ? static_cast<double> (stats.accepted) / media_seconds
                          : 0.0 },
    { "delta_coverage", stats.total_delta_sum > 0.0
                            ? stats.accepted_delta_sum / stats.total_delta_sum
                            : 0.0 },
    { "max_gap_seconds", stats.max_gap_seconds },
    { "p95_gap_seconds", Percentile (gaps, 0.95) },
  };
}

double
Median (std::vector<double> values)
{
  if (values.empty ())
    return 0.0;
  std::sort (values.begin (), values.end ());
  const std::size_t mid = values.size () / 2;
  if (values.size () % 2 == 1)
    return values[mid];
  return 0.5 * (values[mid - 1] + values[mid]);
}

std::vector<double>
AudioSalienceScores (const std::vector<SignalFeatures> &features)
{
  std::vector<double> scores;
  scores.reserve (features.size ());
  for (const auto &f : features)
    {
      const double onset = std::max (0.0, f.velocity);
      scores.push_back (0.40 * f.mean_delta + 0.35 * f.max_local_delta
                        + 0.15 * onset + 0.10 * f.energy);
    }
  return scores;
}

std::vector<AudioEvent>
DetectAudioEvents (const std::vector<SignalFeatures> &features,
                   const Options &opts)
{
  if (features.size () < 3)
    return {};
  const auto scores = AudioSalienceScores (features);
  std::vector<double> tail_scores;
  tail_scores.reserve (scores.size () - 1);
  for (std::size_t i = 1; i < scores.size (); ++i)
    tail_scores.push_back (scores[i]);
  const double med = Median (tail_scores);
  std::vector<double> abs_dev;
  abs_dev.reserve (tail_scores.size ());
  for (double score : tail_scores)
    abs_dev.push_back (std::abs (score - med));
  const double mad = Median (abs_dev);
  const double robust_sigma = std::max (1e-6, 1.4826 * mad);
  const double threshold = med + 1.25 * robust_sigma;
  const double chunk_seconds = static_cast<double> (opts.chunk_ms) / 1000.0;
  const int merge_gap = std::max (1, static_cast<int> (std::ceil (0.50 / chunk_seconds)));

  std::vector<int> peaks;
  for (std::size_t i = 1; i + 1 < scores.size (); ++i)
    {
      const bool is_peak = scores[i] >= scores[i - 1] && scores[i] >= scores[i + 1];
      if (is_peak && scores[i] >= threshold)
        peaks.push_back (static_cast<int> (i));
    }
  if (peaks.empty ())
    {
      auto best = std::max_element (scores.begin () + 1, scores.end ());
      if (best != scores.end ())
        peaks.push_back (static_cast<int> (std::distance (scores.begin (), best)));
    }

  std::vector<AudioEvent> events;
  for (int peak : peaks)
    {
      if (!events.empty () && peak - events.back ().end_index <= merge_gap)
        {
          events.back ().end_index = std::min (
              static_cast<int> (features.size ()) - 1, peak + 1);
          if (scores[static_cast<std::size_t> (peak)] > events.back ().peak_score)
            {
              events.back ().peak_score = scores[static_cast<std::size_t> (peak)];
              events.back ().peak_seconds = features[static_cast<std::size_t> (peak)].time_seconds;
            }
          events.back ().end_seconds
              = features[static_cast<std::size_t> (events.back ().end_index)].time_seconds
                + chunk_seconds;
          continue;
        }
      AudioEvent event;
      event.event_id = static_cast<int> (events.size ());
      event.start_index = std::max (0, peak - 1);
      event.end_index = std::min (static_cast<int> (features.size ()) - 1, peak + 1);
      event.start_seconds = features[static_cast<std::size_t> (event.start_index)].time_seconds;
      event.end_seconds
          = features[static_cast<std::size_t> (event.end_index)].time_seconds
            + chunk_seconds;
      event.peak_seconds = features[static_cast<std::size_t> (peak)].time_seconds;
      event.peak_score = scores[static_cast<std::size_t> (peak)];
      events.push_back (event);
    }
  return events;
}

bool
IndexNearAudioEvent (int index, const std::vector<AudioEvent> &events,
                     int tolerance_chunks)
{
  for (const auto &event : events)
    {
      if (index >= event.start_index - tolerance_chunks
          && index <= event.end_index + tolerance_chunks)
        return true;
    }
  return false;
}

Json
AudioDecisionEval (const std::string &name,
                   const std::vector<SignalFilterDecision> &decisions,
                   const std::vector<AudioEvent> &events,
                   const std::vector<SignalFeatures> &features,
                   const Options &opts)
{
  const int tolerance = std::max (
      1, static_cast<int> (std::ceil (0.25 / (static_cast<double> (opts.chunk_ms) / 1000.0))));
  int accepted = 0;
  int accepted_near_event = 0;
  int accepted_far_event = 0;
  int non_event_chunks = 0;
  std::vector<bool> event_hit (events.size (), false);
  for (std::size_t i = 0; i < decisions.size (); ++i)
    {
      const bool near = IndexNearAudioEvent (static_cast<int> (i), events, tolerance);
      if (!near)
        ++non_event_chunks;
      if (!decisions[i].accepted)
        continue;
      ++accepted;
      if (near)
        ++accepted_near_event;
      else
        ++accepted_far_event;
      for (std::size_t e = 0; e < events.size (); ++e)
        {
          if (static_cast<int> (i) >= events[e].start_index - tolerance
              && static_cast<int> (i) <= events[e].end_index + tolerance)
            event_hit[e] = true;
        }
    }
  const int hit_count = static_cast<int> (
      std::count (event_hit.begin (), event_hit.end (), true));
  const double media_seconds = features.empty () ? 0.0 : features.back ().time_seconds
      + static_cast<double> (opts.chunk_ms) / 1000.0;
  return Json{
    { "policy", name },
    { "events", events.size () },
    { "event_recall",
      events.empty () ? 0.0 : static_cast<double> (hit_count)
                            / static_cast<double> (events.size ()) },
    { "accepted", accepted },
    { "accepted_near_event", accepted_near_event },
    { "accepted_far_event", accepted_far_event },
    { "accepts_per_event",
      events.empty () ? 0.0 : static_cast<double> (accepted)
                            / static_cast<double> (events.size ()) },
    { "event_precision",
      accepted > 0 ? static_cast<double> (accepted_near_event)
                         / static_cast<double> (accepted)
                   : 0.0 },
    { "ambient_accept_rate",
      non_event_chunks > 0 ? static_cast<double> (accepted_far_event)
                                 / static_cast<double> (non_event_chunks)
                           : 0.0 },
    { "accepted_per_second",
      media_seconds > 0.0 ? static_cast<double> (accepted) / media_seconds : 0.0 },
  };
}

Json
AudioEventsJson (const std::vector<AudioEvent> &events)
{
  Json rows = Json::array ();
  for (const auto &event : events)
    {
      rows.push_back (Json{
        { "event_id", event.event_id },
        { "start_index", event.start_index },
        { "end_index", event.end_index },
        { "start_seconds", event.start_seconds },
        { "end_seconds", event.end_seconds },
        { "peak_seconds", event.peak_seconds },
        { "peak_score", event.peak_score },
      });
    }
  return rows;
}

ModalityRun
RunFilter (const std::string &modality, const Options &opts)
{
  ModalityRun run;
  run.modality = modality;
  if (modality == "image")
    {
      run.features = LoadImageFeatures (opts);
      run.media_seconds = static_cast<double> (run.features.size ()) / opts.fps;
    }
  else if (modality == "audio")
    {
      run.features = LoadAudioFeatures (opts);
      run.media_seconds = static_cast<double> (
                              std::filesystem::file_size (opts.audio_path)
                              / sizeof (float))
                          / static_cast<double> (opts.sample_rate);
    }
  else if (modality == "text")
    {
      run.features = LoadTextFeatures (opts);
      run.media_seconds = static_cast<double> (run.features.size ())
                          * opts.text_step_seconds;
    }
  else
    {
      throw std::runtime_error ("Unsupported modality in RunFilter");
    }

  SignalFilterState all_state;
  SignalFilterState fixed_state;
  SignalFilterState adaptive_state;
  SignalFilterStats all_stats;
  SignalFilterStats fixed_stats;
  SignalFilterStats adaptive_stats;
  run.rows = Json::array ();

  for (const auto &features : run.features)
    {
      auto all = DecideAll (features, all_state);
      auto fixed = DecideFixed (features, fixed_state, opts);
      auto adaptive = DecideAdaptive (features, adaptive_state, opts);
      UpdateStats (all_stats, features, all);
      UpdateStats (fixed_stats, features, fixed);
      UpdateStats (adaptive_stats, features, adaptive);
      run.all_decisions.push_back (all);
      run.fixed_decisions.push_back (fixed);
      run.adaptive_decisions.push_back (adaptive);
      run.rows.push_back (Json{
        { "index", features.index },
        { "time_seconds", features.time_seconds },
        { "mean_delta", features.mean_delta },
        { "max_local_delta", features.max_local_delta },
        { "entropy", features.entropy },
        { "velocity", features.velocity },
        { "energy", features.energy },
        { "fixed_accept", fixed.accepted },
        { "fixed_reason", fixed.reason },
        { "fixed_threshold", fixed.threshold },
        { "adaptive_accept", adaptive.accepted },
        { "adaptive_reason", adaptive.reason },
        { "adaptive_threshold", adaptive.threshold },
      });
    }

  run.summary = Json{
    { "modality", modality },
    { "media_seconds", run.media_seconds },
    { "all", StatsJson (all_stats, run.media_seconds) },
    { "fixed", StatsJson (fixed_stats, run.media_seconds) },
    { "adaptive", StatsJson (adaptive_stats, run.media_seconds) },
  };
  if (modality == "audio")
    {
      const auto events = DetectAudioEvents (run.features, opts);
      run.summary["audio_salience_events"] = AudioEventsJson (events);
      run.summary["audio_salience_eval"] = Json::array ({
        AudioDecisionEval ("all", run.all_decisions, events, run.features, opts),
        AudioDecisionEval ("fixed", run.fixed_decisions, events, run.features, opts),
        AudioDecisionEval ("adaptive", run.adaptive_decisions, events, run.features, opts),
      });
    }
  return run;
}

bool
Selected (const ModalityRun &run, const Options &opts, std::size_t index)
{
  if (opts.process_policy == "all")
    return true;
  if (opts.process_policy == "fixed")
    return index < run.fixed_decisions.size () && run.fixed_decisions[index].accepted;
  return index < run.adaptive_decisions.size () && run.adaptive_decisions[index].accepted;
}

void
AppendContextStats (ProcessStats &stats, const cortext::Cortext::Context &ctx)
{
  ++stats.processed;
  stats.total_ms.push_back (ctx.total_ms);
  stats.encode_ms.push_back (ctx.encode_ms);
  stats.process_ms.push_back (ctx.process_ms);
  stats.hydrate_ms.push_back (ctx.hydrate_ms);
}

ProcessStats
ProcessRun (const ModalityRun &run, const Options &opts)
{
  ProcessStats stats;
  stats.modality = run.modality;
  cortext::Cortext::Config cfg;
  cfg.focus = opts.focus;
  cfg.sensitivity = opts.sensitivity;
  cfg.stability = opts.stability;
  const auto db_path = opts.output_dir
                       / ("signal_filter_" + run.modality + "_"
                          + opts.process_policy + ".sqlite");
  auto ctx = cortext::Cortext::Create (cfg, db_path.string ());
  const auto start = std::chrono::steady_clock::now ();

  if (run.modality == "image")
    {
      const std::size_t frame_size
          = static_cast<std::size_t> (opts.width) * static_cast<std::size_t> (opts.height)
            * static_cast<std::size_t> (opts.channels);
      std::ifstream input (opts.frames_path, std::ios::binary);
      std::vector<std::uint8_t> frame (frame_size);
      for (std::size_t i = 0; i < run.features.size (); ++i)
        {
          input.read (reinterpret_cast<char *> (frame.data ()),
                      static_cast<std::streamsize> (frame_size));
          if (!Selected (run, opts, i))
            continue;
          auto c = ctx->ProcessImage (
              frame.data (), opts.width, opts.height, opts.channels,
              "signal-filter/image/" + opts.process_policy + "/"
                  + std::to_string (i));
          AppendContextStats (stats, c);
        }
    }
  else if (run.modality == "audio")
    {
      const auto audio = ReadF32File (opts.audio_path);
      const std::size_t chunk
          = static_cast<std::size_t> (opts.sample_rate * opts.chunk_ms / 1000);
      for (std::size_t i = 0; i < run.features.size (); ++i)
        {
          if (!Selected (run, opts, i))
            continue;
          const std::size_t begin = i * chunk;
          const std::size_t end = std::min (audio.size (), begin + chunk);
          auto c = ctx->ProcessAudio (
              audio.data () + begin, end - begin,
              "signal-filter/audio/" + opts.process_policy + "/"
                  + std::to_string (i));
          AppendContextStats (stats, c);
        }
    }
  else if (run.modality == "text")
    {
      auto lines = ReadLines (opts.text_path);
      if (opts.max_items > 0 && static_cast<int> (lines.size ()) > opts.max_items)
        lines.resize (static_cast<std::size_t> (opts.max_items));
      for (std::size_t i = 0; i < run.features.size (); ++i)
        {
          if (!Selected (run, opts, i))
            continue;
          auto c = ctx->ProcessText (
              lines[i], "signal-filter/text/" + opts.process_policy + "/"
                            + std::to_string (i));
          AppendContextStats (stats, c);
        }
    }

  ctx->Flush ();
  stats.wall_seconds = ElapsedSeconds (start, std::chrono::steady_clock::now ());
  return stats;
}

Json
ProcessJson (const ProcessStats &stats, double media_seconds)
{
  const double total = std::accumulate (stats.total_ms.begin (),
                                       stats.total_ms.end (), 0.0);
  return Json{
    { "modality", stats.modality },
    { "processed", stats.processed },
    { "wall_seconds", stats.wall_seconds },
    { "media_seconds", media_seconds },
    { "realtime_multiple",
      stats.wall_seconds > 0.0 ? media_seconds / stats.wall_seconds : 0.0 },
    { "mean_total_ms", stats.total_ms.empty ()
                           ? 0.0
                           : total / static_cast<double> (stats.total_ms.size ()) },
    { "p95_total_ms", Percentile (stats.total_ms, 0.95) },
    { "mean_encode_ms",
      stats.encode_ms.empty ()
          ? 0.0
          : std::accumulate (stats.encode_ms.begin (), stats.encode_ms.end (), 0.0)
                / static_cast<double> (stats.encode_ms.size ()) },
    { "mean_process_ms",
      stats.process_ms.empty ()
          ? 0.0
          : std::accumulate (stats.process_ms.begin (), stats.process_ms.end (), 0.0)
                / static_cast<double> (stats.process_ms.size ()) },
    { "mean_hydrate_ms",
      stats.hydrate_ms.empty ()
          ? 0.0
          : std::accumulate (stats.hydrate_ms.begin (), stats.hydrate_ms.end (), 0.0)
                / static_cast<double> (stats.hydrate_ms.size ()) },
  };
}

std::vector<std::string>
ModalitiesToRun (const Options &opts)
{
  if (opts.modality == "all")
    return { "image", "audio", "text" };
  return { opts.modality };
}

} // namespace

int
main (int argc, char *argv[])
{
  try
    {
      const Options opts = ParseArgs (argc, argv);
      std::filesystem::create_directories (opts.output_dir);
      const auto modalities = ModalitiesToRun (opts);

      std::vector<std::future<ModalityRun>> filter_futures;
      for (const auto &modality : modalities)
        {
          filter_futures.push_back (std::async (std::launch::async, [&, modality] {
            return RunFilter (modality, opts);
          }));
        }

      std::vector<ModalityRun> runs;
      for (auto &future : filter_futures)
        runs.push_back (future.get ());

      Json summary{
        { "input",
          { { "modality", opts.modality },
            { "frames_path", opts.frames_path.string () },
            { "audio_path", opts.audio_path.string () },
            { "text_path", opts.text_path.string () },
            { "width", opts.width },
            { "height", opts.height },
            { "channels", opts.channels },
            { "fps", opts.fps },
            { "sample_rate", opts.sample_rate },
            { "chunk_ms", opts.chunk_ms },
            { "text_step_seconds", opts.text_step_seconds } } },
        { "policy",
          { { "fixed_threshold", opts.fixed_threshold },
            { "base_threshold", opts.base_threshold },
            { "heartbeat_seconds", opts.heartbeat_seconds },
            { "min_accept_seconds", opts.min_accept_seconds },
            { "focus", opts.focus },
            { "sensitivity", opts.sensitivity },
            { "stability", opts.stability } } },
        { "modalities", Json::object () },
      };

      for (const auto &run : runs)
        {
          summary["modalities"][run.modality] = run.summary;
          std::ofstream (opts.output_dir
                         / ("signal_filter_" + run.modality + "_rows.json"))
              << run.rows.dump (2) << "\n";
        }

      if (opts.process_accepted)
        {
          const auto start = std::chrono::steady_clock::now ();
          std::vector<std::future<ProcessStats>> process_futures;
          for (const auto &run : runs)
            {
              process_futures.push_back (std::async (std::launch::async, [&opts, &run] {
                return ProcessRun (run, opts);
              }));
            }
          Json processing = Json::object ();
          for (std::size_t i = 0; i < runs.size (); ++i)
            {
              auto stats = process_futures[i].get ();
              processing[stats.modality] = ProcessJson (stats, runs[i].media_seconds);
            }
          const double wall = ElapsedSeconds (start, std::chrono::steady_clock::now ());
          double max_media = 0.0;
          for (const auto &run : runs)
            max_media = std::max (max_media, run.media_seconds);
          processing["parallel_wall_seconds"] = wall;
          processing["max_media_seconds"] = max_media;
          processing["parallel_realtime_multiple"]
              = wall > 0.0 ? max_media / wall : 0.0;
          summary["process_accepted_parallel"] = processing;
        }

      std::ofstream (opts.output_dir / "signal_filter_summary.json")
          << summary.dump (2) << "\n";
      std::cout << summary.dump (2) << "\n";
      return 0;
    }
  catch (const std::exception &ex)
    {
      std::cerr << "cortext_signal_filter_bench failed: " << ex.what ()
                << "\n";
      return 1;
    }
}
