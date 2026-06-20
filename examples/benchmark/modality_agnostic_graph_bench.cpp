#include <cortext/core/algorithms.hpp>
#include <cortext/cortext.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/store.hpp>
#include <cortext/store/utils.hpp>

#include <nlohmann/json.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

struct Options
{
  std::filesystem::path assets_dir = "build/real_multimodal_episode_assets";
  std::filesystem::path video_dir = "build/video_media_perf";
  std::filesystem::path output_dir
      = "build/modality_agnostic_graph_bench";
  std::string models_dir = "models";
};

struct Event
{
  std::string id;
  std::string offline_group;
  std::string modality;
  std::string text;
  std::filesystem::path path;
  int width = 0;
  int height = 0;
  int channels = 0;
  std::uint64_t byte_offset = 0;
  std::size_t byte_count = 0;
  std::size_t sample_offset = 0;
  std::size_t sample_count = 0;
  std::string source_id = "realdata/wikimedia";
};

struct Scenario
{
  std::string name;
  std::vector<Event> events;
};

struct SignalRow
{
  long long signal_id = 0;
  long long memory_id = 0;
  long long embedding_id = 0;
  long long serial_position = 0;
  std::string modality;
  std::string mime;
  std::size_t payload_bytes = 0;
  Eigen::VectorXf embedding;
  std::string offline_group;
};

struct MemoryNode
{
  long long memory_id = 0;
  long long embedding_id = 0;
  long long start_ts = 0;
  long long end_ts = 0;
  std::string modality;
  std::size_t memory_payload_bytes = 0;
  Eigen::VectorXf embedding;
  std::set<std::string> offline_groups;
  std::set<std::string> modalities;
  std::vector<long long> signal_ids;
  bool mixed_offline_groups = false;
};

struct Edge
{
  std::string scenario;
  std::string variant;
  long long source_memory_id = 0;
  long long target_memory_id = 0;
  std::string edge_type;
  double weight = 0.0;
  bool same_offline_group = false;
  bool cross_offline_group = false;
  bool touches_mixed_node = false;
  std::string source_groups;
  std::string target_groups;
};

struct ClusterRow
{
  std::string scenario;
  std::string variant;
  int cluster_id = 0;
  std::vector<long long> memory_ids;
  std::set<std::string> offline_groups;
  bool mixed_offline_groups = false;
};

struct VariantResult
{
  std::string scenario;
  std::string variant;
  int node_count = 0;
  int memory_count_total = 0;
  int signal_count_total = 0;
  int signal_evidence_edges = 0;
  int memories_without_blob = 0;
  int non_text_blob_risk = 0;
  int edge_count = 0;
  int same_group_edges = 0;
  int cross_group_edges = 0;
  int mixed_node_edges = 0;
  int cluster_count = 0;
  int mixed_cluster_count = 0;
  double edge_precision = 0.0;
  double edge_cross_rate = 0.0;
  double node_coverage = 0.0;
  double signal_coverage = 0.0;
};

struct ConsolidationUnit
{
  std::string scenario;
  std::string variant;
  std::string unit_id;
  std::vector<long long> memory_ids;
  std::vector<long long> signal_ids;
  std::set<std::string> modalities;
  std::set<std::string> offline_groups;
  bool mixed_offline_groups = false;
  bool multi_modal = false;
};

struct ConsolidationResult
{
  std::string scenario;
  std::string variant;
  int unit_count = 0;
  int memory_count_total = 0;
  int signal_count_total = 0;
  int covered_memories = 0;
  int covered_signals = 0;
  int mixed_units = 0;
  int multi_modal_units = 0;
  double memory_coverage = 0.0;
  double signal_coverage = 0.0;
  double unit_precision = 0.0;
  double mean_signals_per_unit = 0.0;
};

struct EmbeddingSurfaceRow
{
  std::string scenario;
  std::string node_type;
  long long row_id = 0;
  long long node_id = 0;
  long long memory_id = 0;
  std::string modality;
  std::string offline_groups;
  std::string modalities;
  std::string signal_ids;
  Eigen::VectorXf embedding;
};

void
PrintUsage ()
{
  std::cout << "Usage: cortext_modality_agnostic_graph_bench"
            << " [--assets-dir <path>] [--output-dir <path>]"
            << " [--video-dir <path>]"
            << " [--models <path>]\n";
}

Options
ParseArgs (int argc, char **argv)
{
  Options opts;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h")
        {
          PrintUsage ();
          std::exit (0);
        }
      if (arg == "--assets-dir" && i + 1 < argc)
        {
          opts.assets_dir = argv[++i];
        }
      else if (arg == "--output-dir" && i + 1 < argc)
        {
          opts.output_dir = argv[++i];
        }
      else if (arg == "--video-dir" && i + 1 < argc)
        {
          opts.video_dir = argv[++i];
        }
      else if (arg == "--models" && i + 1 < argc)
        {
          opts.models_dir = argv[++i];
        }
      else
        {
          throw std::runtime_error ("Unknown or incomplete argument: " + arg);
        }
    }
  return opts;
}

std::vector<unsigned char>
ReadBytes (const std::filesystem::path &path)
{
  std::ifstream in (path, std::ios::binary | std::ios::ate);
  if (!in)
    {
      throw std::runtime_error ("Failed to open " + path.string ());
    }
  const auto size = in.tellg ();
  in.seekg (0, std::ios::beg);
  std::vector<unsigned char> data (static_cast<std::size_t> (size));
  in.read (reinterpret_cast<char *> (data.data ()),
           static_cast<std::streamsize> (data.size ()));
  return data;
}

std::vector<unsigned char>
ReadBytesRange (const std::filesystem::path &path, std::uint64_t offset,
                std::size_t count)
{
  if (count == 0)
    {
      return ReadBytes (path);
    }
  std::ifstream in (path, std::ios::binary);
  if (!in)
    {
      throw std::runtime_error ("Failed to open " + path.string ());
    }
  in.seekg (static_cast<std::streamoff> (offset), std::ios::beg);
  std::vector<unsigned char> data (count);
  in.read (reinterpret_cast<char *> (data.data ()),
           static_cast<std::streamsize> (data.size ()));
  if (static_cast<std::size_t> (in.gcount ()) != count)
    {
      throw std::runtime_error ("Failed to read requested byte range from "
                                + path.string ());
    }
  return data;
}

std::vector<float>
ReadFloat32 (const std::filesystem::path &path)
{
  const auto bytes = ReadBytes (path);
  if (bytes.size () % sizeof (float) != 0)
    {
      throw std::runtime_error ("Invalid f32 audio byte length: "
                                + path.string ());
    }
  std::vector<float> out (bytes.size () / sizeof (float));
  std::memcpy (out.data (), bytes.data (), bytes.size ());
  return out;
}

std::vector<float>
ReadFloat32Range (const std::filesystem::path &path, std::size_t sample_offset,
                  std::size_t sample_count)
{
  if (sample_count == 0)
    {
      return ReadFloat32 (path);
    }
  const auto bytes = ReadBytesRange (
      path, static_cast<std::uint64_t> (sample_offset * sizeof (float)),
      sample_count * sizeof (float));
  std::vector<float> out (bytes.size () / sizeof (float));
  std::memcpy (out.data (), bytes.data (), bytes.size ());
  return out;
}

std::uint16_t
ReadU16LE (const std::vector<unsigned char> &bytes, std::size_t offset)
{
  return static_cast<std::uint16_t> (bytes[offset])
         | static_cast<std::uint16_t> (bytes[offset + 1] << 8);
}

std::uint32_t
ReadU32LE (const std::vector<unsigned char> &bytes, std::size_t offset)
{
  return static_cast<std::uint32_t> (bytes[offset])
         | (static_cast<std::uint32_t> (bytes[offset + 1]) << 8)
         | (static_cast<std::uint32_t> (bytes[offset + 2]) << 16)
         | (static_cast<std::uint32_t> (bytes[offset + 3]) << 24);
}

std::vector<float>
ResampleMonoTo16k (const std::vector<float> &mono, int sample_rate)
{
  if (sample_rate == 16000 || mono.empty ())
    {
      return mono;
    }
  const auto out_count = std::max<std::size_t> (
      1, static_cast<std::size_t> (
             std::llround (static_cast<double> (mono.size ()) * 16000.0
                           / static_cast<double> (sample_rate))));
  std::vector<float> out (out_count);
  const double step = static_cast<double> (sample_rate) / 16000.0;
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      const double pos = static_cast<double> (i) * step;
      const auto lo = static_cast<std::size_t> (std::floor (pos));
      const auto hi = std::min<std::size_t> (lo + 1, mono.size () - 1);
      const auto frac = static_cast<float> (pos - static_cast<double> (lo));
      const auto a = mono[std::min<std::size_t> (lo, mono.size () - 1)];
      const auto b = mono[hi];
      out[i] = a + (b - a) * frac;
    }
  return out;
}

std::vector<float>
ReadWavMono16k (const std::filesystem::path &path)
{
  const auto bytes = ReadBytes (path);
  if (bytes.size () < 44 || std::memcmp (bytes.data (), "RIFF", 4) != 0
      || std::memcmp (bytes.data () + 8, "WAVE", 4) != 0)
    {
      throw std::runtime_error ("Unsupported WAV header: " + path.string ());
    }

  std::uint16_t audio_format = 0;
  std::uint16_t channels = 0;
  std::uint32_t sample_rate = 0;
  std::uint16_t bits_per_sample = 0;
  std::size_t data_offset = 0;
  std::uint32_t data_size = 0;

  std::size_t offset = 12;
  while (offset + 8 <= bytes.size ())
    {
      const auto chunk_id = reinterpret_cast<const char *> (bytes.data ()
                                                            + offset);
      const auto chunk_size = ReadU32LE (bytes, offset + 4);
      const auto chunk_data = offset + 8;
      if (chunk_data + chunk_size > bytes.size ())
        {
          break;
        }
      if (std::memcmp (chunk_id, "fmt ", 4) == 0 && chunk_size >= 16)
        {
          audio_format = ReadU16LE (bytes, chunk_data);
          channels = ReadU16LE (bytes, chunk_data + 2);
          sample_rate = ReadU32LE (bytes, chunk_data + 4);
          bits_per_sample = ReadU16LE (bytes, chunk_data + 14);
        }
      else if (std::memcmp (chunk_id, "data", 4) == 0)
        {
          data_offset = chunk_data;
          data_size = chunk_size;
        }
      offset = chunk_data + chunk_size + (chunk_size % 2);
    }

  if (channels == 0 || sample_rate == 0 || data_offset == 0 || data_size == 0)
    {
      throw std::runtime_error ("Incomplete WAV chunks: " + path.string ());
    }
  const auto bytes_per_sample = bits_per_sample / 8;
  if (bytes_per_sample == 0)
    {
      throw std::runtime_error ("Invalid WAV sample width: " + path.string ());
    }
  const auto frame_bytes = static_cast<std::size_t> (channels)
                           * static_cast<std::size_t> (bytes_per_sample);
  const auto frame_count = static_cast<std::size_t> (data_size) / frame_bytes;
  std::vector<float> mono (frame_count);
  for (std::size_t frame = 0; frame < frame_count; ++frame)
    {
      double sum = 0.0;
      for (std::uint16_t ch = 0; ch < channels; ++ch)
        {
          const auto sample_offset
              = data_offset + frame * frame_bytes
                + static_cast<std::size_t> (ch) * bytes_per_sample;
          float sample = 0.0f;
          if (audio_format == 1 && bits_per_sample == 16)
            {
              sample = static_cast<float> (
                           static_cast<std::int16_t> (
                               ReadU16LE (bytes, sample_offset)))
                       / 32768.0f;
            }
          else if (audio_format == 1 && bits_per_sample == 24)
            {
              std::int32_t raw
                  = static_cast<std::int32_t> (bytes[sample_offset])
                    | (static_cast<std::int32_t> (bytes[sample_offset + 1])
                       << 8)
                    | (static_cast<std::int32_t> (bytes[sample_offset + 2])
                       << 16);
              if ((raw & 0x00800000) != 0)
                {
                  raw |= ~0x00ffffff;
                }
              sample = static_cast<float> (raw) / 8388608.0f;
            }
          else if (audio_format == 1 && bits_per_sample == 32)
            {
              sample = static_cast<float> (
                           static_cast<std::int32_t> (
                               ReadU32LE (bytes, sample_offset)))
                       / 2147483648.0f;
            }
          else if (audio_format == 3 && bits_per_sample == 32)
            {
              std::memcpy (&sample, bytes.data () + sample_offset,
                           sizeof (float));
            }
          else
            {
              throw std::runtime_error ("Unsupported WAV format: "
                                        + path.string ());
            }
          sum += sample;
        }
      mono[frame] = static_cast<float> (sum / static_cast<double> (channels));
    }
  return ResampleMonoTo16k (mono, static_cast<int> (sample_rate));
}

std::vector<float>
ReadAudioEvent (const Event &event)
{
  if (event.path.extension () == ".wav")
    {
      return ReadWavMono16k (event.path);
    }
  return ReadFloat32Range (event.path, event.sample_offset,
                           event.sample_count);
}

long long
AnyLong (const std::any &value)
{
  if (value.type () == typeid (long long))
    return std::any_cast<long long> (value);
  if (value.type () == typeid (int))
    return std::any_cast<int> (value);
  return 0;
}

double
AnyDouble (const std::any &value)
{
  if (value.type () == typeid (double))
    return std::any_cast<double> (value);
  if (value.type () == typeid (float))
    return static_cast<double> (std::any_cast<float> (value));
  if (value.type () == typeid (long long))
    return static_cast<double> (std::any_cast<long long> (value));
  if (value.type () == typeid (int))
    return static_cast<double> (std::any_cast<int> (value));
  return 0.0;
}

int
CountQuery (cortext::Store &store, const std::string &sql)
{
  const auto rows = store.Execute (sql, {});
  if (rows.empty () || rows.front ().empty ())
    {
      return 0;
    }
  return static_cast<int> (AnyLong (rows.front ().begin ()->second));
}

std::string
AnyString (const std::any &value)
{
  if (value.type () == typeid (std::string))
    return std::any_cast<std::string> (value);
  return {};
}

std::string
CsvEscape (const std::string &s)
{
  if (s.find_first_of (",\"\n\r") == std::string::npos)
    {
      return s;
    }
  std::string out = "\"";
  for (char c : s)
    {
      if (c == '"')
        {
          out += "\"\"";
        }
      else
        {
          out += c;
        }
    }
  out += '"';
  return out;
}

std::string
JoinStrings (const std::set<std::string> &items)
{
  std::ostringstream out;
  bool first = true;
  for (const auto &item : items)
    {
      if (!first)
        {
          out << "|";
        }
      first = false;
      out << item;
    }
  return out.str ();
}

std::string
JoinLongs (const std::vector<long long> &items)
{
  std::ostringstream out;
  for (std::size_t i = 0; i < items.size (); ++i)
    {
      if (i != 0)
        {
          out << "|";
        }
      out << items[i];
    }
  return out.str ();
}

std::vector<long long>
UniqueSorted (std::vector<long long> values)
{
  std::sort (values.begin (), values.end ());
  values.erase (std::unique (values.begin (), values.end ()), values.end ());
  return values;
}

std::vector<unsigned char>
FetchPayload (cortext::Store &store, const std::any &blob_value)
{
  const auto blob_id = cortext::store::BlobFromAny (blob_value);
  if (blob_id.empty ())
    {
      return {};
    }
  const auto rows = store.Execute ("SELECT objstore_get(?1) AS payload",
                                   { blob_id });
  if (rows.empty () || rows.front ().count ("payload") == 0)
    {
      return {};
    }
  return cortext::store::BlobFromAny (rows.front ().at ("payload"));
}

Eigen::VectorXf
DecodeEmbeddingAny (const std::any &value)
{
  const auto bytes = cortext::store::BlobFromAny (value);
  if (bytes.empty () || bytes.size () % sizeof (float) != 0)
    {
      return {};
    }
  Eigen::VectorXf out (
      static_cast<Eigen::Index> (bytes.size () / sizeof (float)));
  std::memcpy (out.data (), bytes.data (), bytes.size ());
  return out;
}

bool
Intersects (const std::set<std::string> &a, const std::set<std::string> &b)
{
  for (const auto &item : a)
    {
      if (b.count (item) > 0)
        {
          return true;
        }
    }
  return false;
}

std::vector<Scenario>
LoadTextCorpusScenarios ()
{
  struct Corpus
  {
    std::filesystem::path path;
    std::string name;
    std::string source_id;
    int max_turns = 24;
  };

  const std::vector<Corpus> corpora = {
    { "data/personachat/test.jsonl", "corpus_personachat_text",
      "realdata/personachat", 24 },
    { "data/taskmaster/test.jsonl", "corpus_taskmaster_text",
      "realdata/taskmaster", 24 },
    { "data/topical_chat/test_freq.jsonl", "corpus_topical_chat_text",
      "realdata/topical_chat", 24 },
    { "data/empathetic_dialogues/test.jsonl",
      "corpus_empathetic_dialogues_text", "realdata/empathetic_dialogues",
      24 },
  };

  std::vector<Scenario> scenarios;
  for (const auto &corpus : corpora)
    {
      if (!std::filesystem::exists (corpus.path))
        {
          continue;
        }
      std::ifstream in (corpus.path);
      std::string line;
      if (!std::getline (in, line))
        {
          continue;
        }
      auto json = nlohmann::json::parse (line, nullptr, false);
      if (json.is_discarded () || !json.is_array () || json.size () < 2)
        {
          continue;
        }
      const auto &payload = json.at (1);
      if (!payload.contains ("content") || !payload.at ("content").is_array ())
        {
          continue;
        }
      std::vector<Event> events;
      int turn = 0;
      for (const auto &item : payload.at ("content"))
        {
          if (!item.contains ("message") || !item.at ("message").is_string ())
            {
              continue;
            }
          Event event;
          event.id = corpus.name + "_turn_" + std::to_string (turn);
          event.modality = "text";
          event.text = item.at ("message").get<std::string> ();
          event.source_id = corpus.source_id;
          events.push_back (std::move (event));
          ++turn;
          if (turn >= corpus.max_turns)
            {
              break;
            }
        }
      if (!events.empty ())
        {
          scenarios.push_back ({ corpus.name, events });
        }
    }
  return scenarios;
}

std::vector<Scenario>
LoadRavdessScenarios ()
{
  const std::filesystem::path actor_dir = "data/ravdess/Actor_16";
  if (!std::filesystem::exists (actor_dir))
    {
      return {};
    }
  std::vector<std::filesystem::path> wavs;
  for (const auto &entry : std::filesystem::directory_iterator (actor_dir))
    {
      if (entry.is_regular_file () && entry.path ().extension () == ".wav")
        {
          wavs.push_back (entry.path ());
        }
    }
  std::sort (wavs.begin (), wavs.end ());

  std::vector<Event> events;
  for (const auto &path : wavs)
    {
      Event event;
      event.id = "ravdess_actor16_" + path.stem ().string ();
      event.modality = "audio";
      event.path = path;
      event.source_id = "realdata/ravdess_actor16";
      events.push_back (std::move (event));
    }
  if (events.empty ())
    {
      return {};
    }
  return { { "ravdess_actor16_audio_corpus", events } };
}

std::vector<Scenario>
BuildScenarios (const std::filesystem::path &assets_dir,
                const std::filesystem::path &video_dir)
{
  const auto raw = assets_dir / "raw";
  Event dog_image{ "dog_image", "dog_entity", "image", "",
                   raw / "dog_384x384.rgb", 384, 384, 3 };
  Event dog_text{ "dog_text_wikimedia_title", "dog_entity", "text",
                  "Golden Retriever", {}, 0, 0, 0 };
  Event dog_audio{ "dog_audio_bailey", "dog_entity", "audio", "",
                   raw / "bailey_16k_mono.f32", 0, 0, 0 };
  Event car_image{ "car_crash_image", "car_event", "image", "",
                   raw / "car_crash_384x384.rgb", 384, 384, 3 };
  Event car_text{ "car_text_wikimedia_title", "car_event", "text",
                  "Car crash 1", {}, 0, 0, 0 };
  Event crash_audio{ "crash_audio_word", "car_event", "audio", "",
                     raw / "crash_16k_mono.f32", 0, 0, 0 };

  std::vector<Scenario> scenarios = {
    { "wikimedia_dog_multimodal", { dog_image, dog_text, dog_audio } },
    { "wikimedia_car_crash_multimodal",
      { car_image, car_text, crash_audio } },
    { "wikimedia_dog_then_car",
      { dog_image, dog_text, dog_audio, car_image, car_text, crash_audio } },
    { "wikimedia_audio_image_interleave",
      { dog_audio, dog_image, crash_audio, car_image } },
  };

  const auto video_raw = video_dir / "video_640x480_30fps.rgb";
  const auto audio_raw = video_dir / "audio_16k_mono_f32le.raw";
  if (std::filesystem::exists (video_raw))
    {
      constexpr int width = 640;
      constexpr int height = 480;
      constexpr int channels = 3;
      constexpr std::size_t frame_bytes = width * height * channels;
      std::vector<Event> frames;
      for (int i = 0; i < 10; ++i)
        {
          Event frame;
          frame.id = "samplelib_frame_" + std::to_string (i);
          frame.modality = "image";
          frame.source_id = "realdata/samplelib";
          frame.path = video_raw;
          frame.width = width;
          frame.height = height;
          frame.channels = channels;
          frame.byte_offset = static_cast<std::uint64_t> (i * 30)
                              * static_cast<std::uint64_t> (frame_bytes);
          frame.byte_count = frame_bytes;
          frames.push_back (std::move (frame));
        }
      scenarios.push_back ({ "samplelib_video_1fps_frames", frames });

      if (std::filesystem::exists (audio_raw))
        {
          std::vector<Event> av;
          std::vector<Event> audio;
          for (int i = 0; i < 10; ++i)
            {
              av.push_back (frames[static_cast<std::size_t> (i)]);
              Event chunk;
              chunk.id = "samplelib_audio_1s_" + std::to_string (i);
              chunk.modality = "audio";
              chunk.source_id = "realdata/samplelib";
              chunk.path = audio_raw;
              chunk.sample_offset = static_cast<std::size_t> (i) * 16000;
              chunk.sample_count = 16000;
              audio.push_back (chunk);
              av.push_back (std::move (chunk));
            }
          scenarios.push_back ({ "samplelib_audio_1s_chunks", audio });
          scenarios.push_back ({ "samplelib_video_1fps_audio_1s_interleaved",
                                 av });
        }
    }
  auto text_scenarios = LoadTextCorpusScenarios ();
  scenarios.insert (scenarios.end (), text_scenarios.begin (),
                    text_scenarios.end ());
  auto ravdess_scenarios = LoadRavdessScenarios ();
  scenarios.insert (scenarios.end (), ravdess_scenarios.begin (),
                    ravdess_scenarios.end ());
  return scenarios;
}

void
ProcessScenario (const Scenario &scenario, const Options &opts,
                 const std::filesystem::path &db_path)
{
  std::filesystem::remove (db_path);
  std::filesystem::remove (db_path.string () + "-wal");
  std::filesystem::remove (db_path.string () + "-shm");
  cortext::Cortext::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.signal_filter_audio_enabled = false;
  cfg.signal_filter_image_enabled = false;
  cfg.signal_filter_text_enabled = false;

  auto engine = cortext::Cortext::Create (cfg, db_path.string (),
                                          opts.models_dir);
  for (const auto &event : scenario.events)
    {
      if (event.modality == "image")
        {
          const auto pixels = ReadBytesRange (event.path, event.byte_offset,
                                              event.byte_count);
          (void)engine->ProcessImage (pixels.data (), event.width,
                                      event.height, event.channels,
                                      event.source_id);
        }
      else if (event.modality == "audio")
        {
          const auto pcm = ReadAudioEvent (event);
          (void)engine->ProcessAudio (pcm.data (), pcm.size (),
                                      event.source_id);
        }
      else if (event.modality == "text")
        {
          (void)engine->ProcessText (event.text, event.source_id);
        }
      else
        {
          throw std::runtime_error ("Unknown modality " + event.modality);
        }
    }
  engine->Flush ();
}

std::vector<SignalRow>
LoadSignals (cortext::Store &store, const std::vector<Event> &events)
{
  std::vector<SignalRow> out;
  const auto rows = store.Execute (
      "SELECT signal_id, COALESCE(memory_id, 0) AS memory_id, "
      "       s.embedding_id AS embedding_id, serial_position, modality, "
      "       COALESCE(mime, '') AS mime, blob_id, e.embedding "
      "FROM signals s JOIN embeddings e ON e.embedding_id = s.embedding_id "
      "WHERE write_decision = 1 ORDER BY signal_id",
      {});
  std::vector<bool> assigned (events.size (), false);
  for (const auto &row : rows)
    {
      SignalRow s;
      s.signal_id = AnyLong (row.at ("signal_id"));
      s.memory_id = AnyLong (row.at ("memory_id"));
      s.embedding_id = AnyLong (row.at ("embedding_id"));
      s.serial_position = AnyLong (row.at ("serial_position"));
      s.modality = AnyString (row.at ("modality"));
      s.mime = AnyString (row.at ("mime"));
      s.embedding = DecodeEmbeddingAny (row.at ("embedding"));
      auto it_blob = row.find ("blob_id");
      if (it_blob != row.end ())
        {
          s.payload_bytes = FetchPayload (store, it_blob->second).size ();
        }
      for (std::size_t i = 0; i < events.size (); ++i)
        {
          if (!assigned[i] && events[i].modality == s.modality)
            {
              s.offline_group = events[i].offline_group;
              assigned[i] = true;
              break;
            }
        }
      out.push_back (std::move (s));
    }
  return out;
}

std::vector<MemoryNode>
LoadMemoryNodes (cortext::Store &store, const std::vector<SignalRow> &signals)
{
  std::map<long long, MemoryNode> nodes;
  const auto rows = store.Execute (
      "SELECT m.memory_id, m.embedding_id, m.modality, m.start_ts, "
      "       COALESCE(m.end_ts, m.start_ts, m.created_at, 0) AS end_ts, "
      "       m.blob_id, e.embedding "
      "FROM memories m JOIN embeddings e ON e.embedding_id = m.embedding_id "
      "WHERE m.kind = 'LONG_TERM' ORDER BY m.memory_id",
      {});
  for (const auto &row : rows)
    {
      MemoryNode node;
      node.memory_id = AnyLong (row.at ("memory_id"));
      node.embedding_id = AnyLong (row.at ("embedding_id"));
      node.modality = AnyString (row.at ("modality"));
      node.start_ts = AnyLong (row.at ("start_ts"));
      node.end_ts = AnyLong (row.at ("end_ts"));
      node.embedding = DecodeEmbeddingAny (row.at ("embedding"));
      auto it_blob = row.find ("blob_id");
      if (it_blob != row.end ())
        {
          node.memory_payload_bytes = FetchPayload (store, it_blob->second).size ();
        }
      if (node.memory_id > 0)
        {
          nodes[node.memory_id] = std::move (node);
        }
    }

  for (const auto &signal : signals)
    {
      if (signal.memory_id <= 0 || nodes.count (signal.memory_id) == 0)
        {
          continue;
        }
      auto &node = nodes[signal.memory_id];
      node.signal_ids.push_back (signal.signal_id);
      node.modalities.insert (signal.modality);
      if (!signal.offline_group.empty ())
        {
          node.offline_groups.insert (signal.offline_group);
        }
    }

  std::vector<MemoryNode> out;
  for (auto &kv : nodes)
    {
      kv.second.mixed_offline_groups = kv.second.offline_groups.size () > 1;
      out.push_back (std::move (kv.second));
    }
  return out;
}

std::vector<Edge>
SimilarityEdges (const std::string &scenario, const std::string &variant,
                 const std::vector<MemoryNode> &nodes, double threshold)
{
  std::vector<Edge> edges;
  for (std::size_t i = 0; i < nodes.size (); ++i)
    {
      for (std::size_t j = i + 1; j < nodes.size (); ++j)
        {
          if (nodes[i].embedding.size () == 0
              || nodes[i].embedding.size () != nodes[j].embedding.size ())
            {
              continue;
            }
          const double sim
              = cortext::core::CosineSimilarity (nodes[i].embedding,
                                                  nodes[j].embedding);
          if (sim < threshold)
            {
              continue;
            }
          Edge edge;
          edge.scenario = scenario;
          edge.variant = variant;
          edge.source_memory_id = nodes[i].memory_id;
          edge.target_memory_id = nodes[j].memory_id;
          edge.edge_type = "similarity";
          edge.weight = cortext::core::Map01 (sim);
          edge.same_offline_group = Intersects (nodes[i].offline_groups,
                                                nodes[j].offline_groups);
          edge.cross_offline_group
              = !edge.same_offline_group && !nodes[i].offline_groups.empty ()
                && !nodes[j].offline_groups.empty ();
          edge.touches_mixed_node = nodes[i].mixed_offline_groups
                                    || nodes[j].mixed_offline_groups;
          edge.source_groups = JoinStrings (nodes[i].offline_groups);
          edge.target_groups = JoinStrings (nodes[j].offline_groups);
          edges.push_back (std::move (edge));
        }
    }
  return edges;
}

std::vector<Edge>
TemporalEdges (const std::string &scenario, const std::string &variant,
               std::vector<MemoryNode> nodes)
{
  std::sort (nodes.begin (), nodes.end (),
             [] (const MemoryNode &a, const MemoryNode &b) {
               if (a.end_ts == b.end_ts)
                 {
                   return a.memory_id < b.memory_id;
                 }
               return a.end_ts < b.end_ts;
             });
  std::vector<Edge> edges;
  for (std::size_t i = 0; i + 1 < nodes.size (); ++i)
    {
      if (nodes[i].memory_id == nodes[i + 1].memory_id)
        {
          continue;
        }
      Edge edge;
      edge.scenario = scenario;
      edge.variant = variant;
      edge.source_memory_id = nodes[i].memory_id;
      edge.target_memory_id = nodes[i + 1].memory_id;
      edge.edge_type = "temporal_next";
      edge.weight = 1.0;
      edge.same_offline_group = Intersects (nodes[i].offline_groups,
                                            nodes[i + 1].offline_groups);
      edge.cross_offline_group
          = !edge.same_offline_group && !nodes[i].offline_groups.empty ()
            && !nodes[i + 1].offline_groups.empty ();
      edge.touches_mixed_node = nodes[i].mixed_offline_groups
                                || nodes[i + 1].mixed_offline_groups;
      edge.source_groups = JoinStrings (nodes[i].offline_groups);
      edge.target_groups = JoinStrings (nodes[i + 1].offline_groups);
      edges.push_back (std::move (edge));
    }
  return edges;
}

std::vector<Edge>
SoftAnchorEdges (const std::string &scenario, const std::string &variant,
                 cortext::Store &store,
                 const std::unordered_map<long long, MemoryNode> &node_by_id)
{
  std::vector<Edge> edges;
  const auto rows = store.Execute (
      "SELECT anchor_id, memory_id, anchor_strength, anchor_label "
      "FROM soft_anchor_links "
      "WHERE anchor_label NOT IN ('none', 'rejected') "
      "ORDER BY anchor_id, anchor_strength DESC",
      {});
  std::map<std::string, std::vector<std::pair<long long, double>>> by_anchor;
  for (const auto &row : rows)
    {
      const long long memory_id = AnyLong (row.at ("memory_id"));
      if (node_by_id.count (memory_id) == 0)
        {
          continue;
        }
      by_anchor[AnyString (row.at ("anchor_id"))].push_back (
          { memory_id, AnyDouble (row.at ("anchor_strength")) });
    }
  for (const auto &[anchor_id, links] : by_anchor)
    {
      (void)anchor_id;
      for (std::size_t i = 0; i < links.size (); ++i)
        {
          for (std::size_t j = i + 1; j < links.size (); ++j)
            {
              const auto &a = node_by_id.at (links[i].first);
              const auto &b = node_by_id.at (links[j].first);
              Edge edge;
              edge.scenario = scenario;
              edge.variant = variant;
              edge.source_memory_id = a.memory_id;
              edge.target_memory_id = b.memory_id;
              edge.edge_type = "same_soft_anchor";
              edge.weight = std::min (links[i].second, links[j].second);
              edge.same_offline_group = Intersects (a.offline_groups,
                                                    b.offline_groups);
              edge.cross_offline_group
                  = !edge.same_offline_group && !a.offline_groups.empty ()
                    && !b.offline_groups.empty ();
              edge.touches_mixed_node = a.mixed_offline_groups
                                        || b.mixed_offline_groups;
              edge.source_groups = JoinStrings (a.offline_groups);
              edge.target_groups = JoinStrings (b.offline_groups);
              edges.push_back (std::move (edge));
            }
        }
    }
  return edges;
}

std::vector<ClusterRow>
BuildClusters (const std::string &scenario, const std::string &variant,
               const std::vector<MemoryNode> &nodes,
               const std::vector<Edge> &edges)
{
  std::unordered_map<long long, int> index;
  for (std::size_t i = 0; i < nodes.size (); ++i)
    {
      index[nodes[i].memory_id] = static_cast<int> (i);
    }
  std::vector<int> parent (nodes.size ());
  std::iota (parent.begin (), parent.end (), 0);
  auto find_root = [&parent] (int x) {
    while (parent[x] != x)
      {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
    return x;
  };
  auto unite = [&] (int a, int b) {
    a = find_root (a);
    b = find_root (b);
    if (a != b)
      {
        parent[b] = a;
      }
  };
  for (const auto &edge : edges)
    {
      if (index.count (edge.source_memory_id) == 0
          || index.count (edge.target_memory_id) == 0)
        {
          continue;
        }
      unite (index[edge.source_memory_id], index[edge.target_memory_id]);
    }

  std::map<int, ClusterRow> clusters;
  int next_cluster_id = 0;
  std::map<int, int> root_to_cluster;
  for (std::size_t i = 0; i < nodes.size (); ++i)
    {
      const int root = find_root (static_cast<int> (i));
      if (root_to_cluster.count (root) == 0)
        {
          root_to_cluster[root] = next_cluster_id++;
        }
      auto &cluster = clusters[root_to_cluster[root]];
      cluster.scenario = scenario;
      cluster.variant = variant;
      cluster.cluster_id = root_to_cluster[root];
      cluster.memory_ids.push_back (nodes[i].memory_id);
      cluster.offline_groups.insert (nodes[i].offline_groups.begin (),
                                     nodes[i].offline_groups.end ());
    }
  std::vector<ClusterRow> out;
  for (auto &kv : clusters)
    {
      kv.second.mixed_offline_groups = kv.second.offline_groups.size () > 1;
      out.push_back (std::move (kv.second));
    }
  return out;
}

void
FinalizeUnit (ConsolidationUnit &unit)
{
  unit.memory_ids = UniqueSorted (std::move (unit.memory_ids));
  unit.signal_ids = UniqueSorted (std::move (unit.signal_ids));
  unit.mixed_offline_groups = unit.offline_groups.size () > 1;
  unit.multi_modal = unit.modalities.size () > 1;
}

std::vector<ConsolidationUnit>
BuildMemoryRowUnits (const std::string &scenario,
                     const std::vector<MemoryNode> &nodes)
{
  std::vector<ConsolidationUnit> units;
  for (const auto &node : nodes)
    {
      ConsolidationUnit unit;
      unit.scenario = scenario;
      unit.variant = "memory_row_units";
      unit.unit_id = "memory_" + std::to_string (node.memory_id);
      unit.memory_ids.push_back (node.memory_id);
      unit.signal_ids = node.signal_ids;
      unit.modalities = node.modalities;
      unit.offline_groups = node.offline_groups;
      FinalizeUnit (unit);
      units.push_back (std::move (unit));
    }
  return units;
}

std::vector<ConsolidationUnit>
BuildSignalEvidenceUnits (const std::string &scenario,
                          const std::vector<SignalRow> &signals)
{
  std::vector<ConsolidationUnit> units;
  for (const auto &signal : signals)
    {
      ConsolidationUnit unit;
      unit.scenario = scenario;
      unit.variant = "signal_evidence_units";
      unit.unit_id = "signal_" + std::to_string (signal.signal_id);
      if (signal.memory_id > 0)
        {
          unit.memory_ids.push_back (signal.memory_id);
        }
      unit.signal_ids.push_back (signal.signal_id);
      unit.modalities.insert (signal.modality);
      if (!signal.offline_group.empty ())
        {
          unit.offline_groups.insert (signal.offline_group);
        }
      FinalizeUnit (unit);
      units.push_back (std::move (unit));
    }
  return units;
}

std::vector<ConsolidationUnit>
BuildClusterUnits (const std::string &scenario, const std::string &variant,
                   const std::vector<MemoryNode> &nodes,
                   const std::vector<ClusterRow> &clusters)
{
  std::unordered_map<long long, const MemoryNode *> node_by_id;
  for (const auto &node : nodes)
    {
      node_by_id[node.memory_id] = &node;
    }

  std::vector<ConsolidationUnit> units;
  for (const auto &cluster : clusters)
    {
      ConsolidationUnit unit;
      unit.scenario = scenario;
      unit.variant = "cluster_" + variant;
      unit.unit_id = "cluster_" + std::to_string (cluster.cluster_id);
      for (const long long memory_id : cluster.memory_ids)
        {
          auto it = node_by_id.find (memory_id);
          if (it == node_by_id.end ())
            {
              continue;
            }
          const auto &node = *it->second;
          unit.memory_ids.push_back (node.memory_id);
          unit.signal_ids.insert (unit.signal_ids.end (),
                                  node.signal_ids.begin (),
                                  node.signal_ids.end ());
          unit.modalities.insert (node.modalities.begin (),
                                  node.modalities.end ());
          unit.offline_groups.insert (node.offline_groups.begin (),
                                      node.offline_groups.end ());
        }
      FinalizeUnit (unit);
      units.push_back (std::move (unit));
    }
  return units;
}

ConsolidationResult
EvaluateConsolidation (const std::string &scenario, const std::string &variant,
                       int memory_count_total, int signal_count_total,
                       const std::vector<ConsolidationUnit> &units)
{
  ConsolidationResult result;
  result.scenario = scenario;
  result.variant = variant;
  result.unit_count = static_cast<int> (units.size ());
  result.memory_count_total = memory_count_total;
  result.signal_count_total = signal_count_total;

  std::vector<long long> memory_ids;
  std::vector<long long> signal_ids;
  int total_unit_signals = 0;
  for (const auto &unit : units)
    {
      memory_ids.insert (memory_ids.end (), unit.memory_ids.begin (),
                         unit.memory_ids.end ());
      signal_ids.insert (signal_ids.end (), unit.signal_ids.begin (),
                         unit.signal_ids.end ());
      result.mixed_units += unit.mixed_offline_groups ? 1 : 0;
      result.multi_modal_units += unit.multi_modal ? 1 : 0;
      total_unit_signals += static_cast<int> (unit.signal_ids.size ());
    }

  result.covered_memories = static_cast<int> (UniqueSorted (memory_ids).size ());
  result.covered_signals = static_cast<int> (UniqueSorted (signal_ids).size ());
  result.memory_coverage
      = memory_count_total > 0
            ? static_cast<double> (result.covered_memories)
                  / static_cast<double> (memory_count_total)
            : 0.0;
  result.signal_coverage
      = signal_count_total > 0
            ? static_cast<double> (result.covered_signals)
                  / static_cast<double> (signal_count_total)
            : 0.0;
  result.unit_precision
      = result.unit_count > 0
            ? static_cast<double> (result.unit_count - result.mixed_units)
                  / static_cast<double> (result.unit_count)
            : 1.0;
  result.mean_signals_per_unit
      = result.unit_count > 0
            ? static_cast<double> (total_unit_signals)
                  / static_cast<double> (result.unit_count)
            : 0.0;
  return result;
}

VariantResult
EvaluateVariant (const std::string &scenario, const std::string &variant,
                 int memory_count_total, int signal_count_total,
                 int variant_signal_evidence_edges, int memories_without_blob,
                 int non_text_blob_risk,
                 const std::vector<MemoryNode> &nodes,
                 const std::vector<Edge> &edges,
                 const std::vector<ClusterRow> &clusters)
{
  VariantResult result;
  result.scenario = scenario;
  result.variant = variant;
  result.node_count = static_cast<int> (nodes.size ());
  result.memory_count_total = memory_count_total;
  result.signal_count_total = signal_count_total;
  result.signal_evidence_edges = variant_signal_evidence_edges;
  result.memories_without_blob = memories_without_blob;
  result.non_text_blob_risk = non_text_blob_risk;
  result.edge_count = static_cast<int> (edges.size ());
  result.cluster_count = static_cast<int> (clusters.size ());
  for (const auto &edge : edges)
    {
      result.same_group_edges += edge.same_offline_group ? 1 : 0;
      result.cross_group_edges += edge.cross_offline_group ? 1 : 0;
      result.mixed_node_edges += edge.touches_mixed_node ? 1 : 0;
    }
  for (const auto &cluster : clusters)
    {
      result.mixed_cluster_count += cluster.mixed_offline_groups ? 1 : 0;
    }
  result.edge_precision = result.edge_count > 0
                              ? static_cast<double> (result.same_group_edges)
                                    / static_cast<double> (result.edge_count)
                              : 1.0;
  result.edge_cross_rate = result.edge_count > 0
                               ? static_cast<double> (result.cross_group_edges)
                                     / static_cast<double> (result.edge_count)
                               : 0.0;
  result.node_coverage = memory_count_total > 0
                             ? static_cast<double> (result.node_count)
                                   / static_cast<double> (memory_count_total)
                             : 0.0;
  result.signal_coverage = signal_count_total > 0
                               ? static_cast<double> (
                                     result.signal_evidence_edges)
                                     / static_cast<double> (signal_count_total)
                               : 0.0;
  return result;
}

nlohmann::json
VariantToJson (const VariantResult &r)
{
  return {
    { "scenario", r.scenario },
    { "variant", r.variant },
    { "node_count", r.node_count },
    { "memory_count_total", r.memory_count_total },
    { "signal_count_total", r.signal_count_total },
    { "signal_evidence_edges", r.signal_evidence_edges },
    { "memories_without_blob", r.memories_without_blob },
    { "non_text_blob_risk", r.non_text_blob_risk },
    { "edge_count", r.edge_count },
    { "same_group_edges", r.same_group_edges },
    { "cross_group_edges", r.cross_group_edges },
    { "mixed_node_edges", r.mixed_node_edges },
    { "cluster_count", r.cluster_count },
    { "mixed_cluster_count", r.mixed_cluster_count },
    { "edge_precision", r.edge_precision },
    { "edge_cross_rate", r.edge_cross_rate },
    { "node_coverage", r.node_coverage },
    { "signal_coverage", r.signal_coverage },
  };
}

nlohmann::json
ConsolidationToJson (const ConsolidationResult &r)
{
  return {
    { "scenario", r.scenario },
    { "variant", r.variant },
    { "unit_count", r.unit_count },
    { "memory_count_total", r.memory_count_total },
    { "signal_count_total", r.signal_count_total },
    { "covered_memories", r.covered_memories },
    { "covered_signals", r.covered_signals },
    { "mixed_units", r.mixed_units },
    { "multi_modal_units", r.multi_modal_units },
    { "memory_coverage", r.memory_coverage },
    { "signal_coverage", r.signal_coverage },
    { "unit_precision", r.unit_precision },
    { "mean_signals_per_unit", r.mean_signals_per_unit },
  };
}

void
WriteCasesCsv (const std::filesystem::path &path,
               const std::vector<VariantResult> &rows)
{
  std::ofstream out (path);
  out << "scenario,variant,node_count,memory_count_total,signal_count_total,"
      << "signal_evidence_edges,memories_without_blob,non_text_blob_risk,"
      << "edge_count,same_group_edges,cross_group_edges,mixed_node_edges,"
      << "cluster_count,mixed_cluster_count,edge_precision,edge_cross_rate,"
      << "node_coverage,signal_coverage\n";
  for (const auto &row : rows)
    {
      out << CsvEscape (row.scenario) << ',' << CsvEscape (row.variant) << ','
          << row.node_count << ',' << row.memory_count_total << ','
          << row.signal_count_total << ',' << row.signal_evidence_edges << ','
          << row.memories_without_blob << ',' << row.non_text_blob_risk << ','
          << row.edge_count << ',' << row.same_group_edges << ','
          << row.cross_group_edges << ',' << row.mixed_node_edges << ','
          << row.cluster_count << ',' << row.mixed_cluster_count << ','
          << row.edge_precision << ',' << row.edge_cross_rate << ','
          << row.node_coverage << ',' << row.signal_coverage << '\n';
    }
}

void
WriteConsolidationCasesCsv (const std::filesystem::path &path,
                            const std::vector<ConsolidationResult> &rows)
{
  std::ofstream out (path);
  out << "scenario,variant,unit_count,memory_count_total,signal_count_total,"
      << "covered_memories,covered_signals,mixed_units,multi_modal_units,"
      << "memory_coverage,signal_coverage,unit_precision,"
      << "mean_signals_per_unit\n";
  for (const auto &row : rows)
    {
      out << CsvEscape (row.scenario) << ',' << CsvEscape (row.variant) << ','
          << row.unit_count << ',' << row.memory_count_total << ','
          << row.signal_count_total << ',' << row.covered_memories << ','
          << row.covered_signals << ',' << row.mixed_units << ','
          << row.multi_modal_units << ',' << row.memory_coverage << ','
          << row.signal_coverage << ',' << row.unit_precision << ','
          << row.mean_signals_per_unit << '\n';
    }
}

void
WriteConsolidationUnitsCsv (const std::filesystem::path &path,
                            const std::vector<ConsolidationUnit> &units)
{
  std::ofstream out (path);
  out << "scenario,variant,unit_id,memory_ids,signal_ids,modalities,"
      << "offline_groups,mixed_offline_groups,multi_modal\n";
  for (const auto &unit : units)
    {
      out << CsvEscape (unit.scenario) << ',' << CsvEscape (unit.variant)
          << ',' << CsvEscape (unit.unit_id) << ','
          << CsvEscape (JoinLongs (unit.memory_ids)) << ','
          << CsvEscape (JoinLongs (unit.signal_ids)) << ','
          << CsvEscape (JoinStrings (unit.modalities)) << ','
          << CsvEscape (JoinStrings (unit.offline_groups)) << ','
          << (unit.mixed_offline_groups ? 1 : 0) << ','
          << (unit.multi_modal ? 1 : 0) << '\n';
    }
}

void
WriteEdgesCsv (const std::filesystem::path &path, const std::vector<Edge> &edges)
{
  std::ofstream out (path);
  out << "scenario,variant,source_memory_id,target_memory_id,edge_type,weight,"
      << "same_offline_group,cross_offline_group,touches_mixed_node,"
      << "source_groups,target_groups\n";
  for (const auto &edge : edges)
    {
      out << CsvEscape (edge.scenario) << ',' << CsvEscape (edge.variant) << ','
          << edge.source_memory_id << ',' << edge.target_memory_id << ','
          << CsvEscape (edge.edge_type) << ',' << edge.weight << ','
          << (edge.same_offline_group ? 1 : 0) << ','
          << (edge.cross_offline_group ? 1 : 0) << ','
          << (edge.touches_mixed_node ? 1 : 0) << ','
          << CsvEscape (edge.source_groups) << ','
          << CsvEscape (edge.target_groups) << '\n';
    }
}

void
WriteClustersCsv (const std::filesystem::path &path,
                  const std::vector<ClusterRow> &clusters)
{
  std::ofstream out (path);
  out << "scenario,variant,cluster_id,memory_ids,offline_groups,"
      << "mixed_offline_groups\n";
  for (const auto &cluster : clusters)
    {
      out << CsvEscape (cluster.scenario) << ','
          << CsvEscape (cluster.variant) << ',' << cluster.cluster_id << ','
          << CsvEscape (JoinLongs (cluster.memory_ids)) << ','
          << CsvEscape (JoinStrings (cluster.offline_groups)) << ','
          << (cluster.mixed_offline_groups ? 1 : 0) << '\n';
  }
}

void
WriteEmbeddingSurface (const std::filesystem::path &rows_path,
                       const std::filesystem::path &vectors_path,
                       std::vector<EmbeddingSurfaceRow> rows)
{
  std::ofstream rows_out (rows_path);
  rows_out << "row_id,scenario,node_type,node_id,memory_id,modality,"
              "offline_groups,modalities,signal_ids\n";

  std::ofstream vec_out (vectors_path, std::ios::binary);
  long long row_id = 0;
  for (auto &row : rows)
    {
      if (row.embedding.size () != 256)
        {
          continue;
        }
      row.row_id = row_id++;
      rows_out << row.row_id << "," << CsvEscape (row.scenario) << ","
               << CsvEscape (row.node_type) << "," << row.node_id << ","
               << row.memory_id << "," << CsvEscape (row.modality) << ","
               << CsvEscape (row.offline_groups) << ","
               << CsvEscape (row.modalities) << ","
               << CsvEscape (row.signal_ids) << "\n";
      vec_out.write (reinterpret_cast<const char *> (row.embedding.data ()),
                     static_cast<std::streamsize> (row.embedding.size ()
                                                   * sizeof (float)));
    }
}

} // namespace

int
main (int argc, char **argv)
{
  try
    {
      const Options opts = ParseArgs (argc, argv);
      std::filesystem::create_directories (opts.output_dir);

      const auto manifest_path
          = opts.assets_dir / "real_multimodal_episode_assets_manifest.json";
      nlohmann::json asset_manifest = nlohmann::json::object ();
      if (std::filesystem::exists (manifest_path))
        {
          std::ifstream in (manifest_path);
          in >> asset_manifest;
        }

      std::vector<VariantResult> all_results;
      std::vector<Edge> all_edges;
      std::vector<ClusterRow> all_clusters;
      std::vector<ConsolidationResult> all_consolidation_results;
      std::vector<ConsolidationUnit> all_consolidation_units;
      std::vector<EmbeddingSurfaceRow> all_embedding_rows;
      nlohmann::json scenario_json = nlohmann::json::array ();
      int total_runtime_label_rows = 0;

      for (const auto &scenario : BuildScenarios (opts.assets_dir,
                                                  opts.video_dir))
        {
          const auto db_path = opts.output_dir / (scenario.name + ".sqlite");
          ProcessScenario (scenario, opts, db_path);
          auto store = cortext::SQLiteStore::Create (db_path.string ());
          const auto signals = LoadSignals (*store, scenario.events);
          const auto nodes = LoadMemoryNodes (*store, signals);

          std::unordered_map<long long, MemoryNode> node_by_id;
          for (const auto &node : nodes)
            {
              node_by_id.emplace (node.memory_id, node);
            }

          const int memory_count_total = static_cast<int> (nodes.size ());
          const int signal_count_total = static_cast<int> (signals.size ());

          for (const auto &signal : signals)
            {
              EmbeddingSurfaceRow row;
              row.scenario = scenario.name;
              row.node_type = "signal";
              row.node_id = signal.signal_id;
              row.memory_id = signal.memory_id;
              row.modality = signal.modality;
              row.offline_groups = signal.offline_group;
              row.modalities = signal.modality;
              row.signal_ids = std::to_string (signal.signal_id);
              row.embedding = signal.embedding;
              all_embedding_rows.push_back (std::move (row));
            }
          for (const auto &node : nodes)
            {
              EmbeddingSurfaceRow row;
              row.scenario = scenario.name;
              row.node_type = "memory";
              row.node_id = node.memory_id;
              row.memory_id = node.memory_id;
              row.modality = node.modality;
              row.offline_groups = JoinStrings (node.offline_groups);
              row.modalities = JoinStrings (node.modalities);
              row.signal_ids = JoinLongs (node.signal_ids);
              row.embedding = node.embedding;
              all_embedding_rows.push_back (std::move (row));
            }

          const int runtime_label_rows = CountQuery (
              *store,
              "SELECT COUNT(*) FROM memories "
              "WHERE label IS NOT NULL AND label <> ''");
          total_runtime_label_rows += runtime_label_rows;
          int signal_evidence_edges = 0;
          int memories_without_blob = 0;
          int non_text_blob_risk = 0;
          for (const auto &node : nodes)
            {
              signal_evidence_edges
                  += static_cast<int> (node.signal_ids.size ());
              memories_without_blob += node.memory_payload_bytes == 0 ? 1 : 0;
              non_text_blob_risk
                  += node.memory_payload_bytes > 0 && node.modality != "text"
                         ? 1
                         : 0;
            }

          auto add_consolidation = [&] (
              const std::string &variant,
              std::vector<ConsolidationUnit> units) {
            auto result = EvaluateConsolidation (
                scenario.name, variant, memory_count_total, signal_count_total,
                units);
            all_consolidation_results.push_back (std::move (result));
            all_consolidation_units.insert (all_consolidation_units.end (),
                                            units.begin (), units.end ());
          };

          add_consolidation ("memory_row_units",
                             BuildMemoryRowUnits (scenario.name, nodes));
          add_consolidation ("signal_evidence_units",
                             BuildSignalEvidenceUnits (scenario.name,
                                                       signals));

          const std::vector<std::string> variants = {
            "blob_only_similarity",
            "memory_embedding_similarity",
            "memory_embedding_similarity_loose",
            "memory_embedding_similarity_strict",
            "memory_similarity_temporal",
            "signal_aware_temporal",
            "soft_anchor_continuity",
          };

          for (const auto &variant : variants)
            {
              std::vector<MemoryNode> variant_nodes = nodes;
              std::vector<Edge> edges;
              if (variant == "blob_only_similarity")
                {
                  variant_nodes.erase (
                      std::remove_if (
                          variant_nodes.begin (), variant_nodes.end (),
                          [] (const MemoryNode &node) {
                            return node.memory_payload_bytes == 0;
                          }),
                      variant_nodes.end ());
                  edges = SimilarityEdges (scenario.name, variant,
                                           variant_nodes, 0.78);
                }
              else if (variant == "memory_embedding_similarity")
                {
                  edges = SimilarityEdges (scenario.name, variant,
                                           variant_nodes, 0.78);
                }
              else if (variant == "memory_embedding_similarity_loose")
                {
                  edges = SimilarityEdges (scenario.name, variant,
                                           variant_nodes, 0.65);
                }
              else if (variant == "memory_embedding_similarity_strict")
                {
                  edges = SimilarityEdges (scenario.name, variant,
                                           variant_nodes, 0.90);
                }
              else if (variant == "memory_similarity_temporal")
                {
                  edges = SimilarityEdges (scenario.name, variant,
                                           variant_nodes, 0.78);
                  auto temporal = TemporalEdges (scenario.name, variant,
                                                 variant_nodes);
                  edges.insert (edges.end (), temporal.begin (),
                                temporal.end ());
                }
              else if (variant == "signal_aware_temporal")
                {
                  edges = TemporalEdges (scenario.name, variant, variant_nodes);
                }
              else if (variant == "soft_anchor_continuity")
                {
                  edges = SoftAnchorEdges (scenario.name, variant, *store,
                                           node_by_id);
                }
              int variant_signal_evidence_edges = 0;
              for (const auto &node : variant_nodes)
                {
                  variant_signal_evidence_edges
                      += static_cast<int> (node.signal_ids.size ());
                }
              auto clusters = BuildClusters (scenario.name, variant,
                                             variant_nodes, edges);
              add_consolidation ("cluster_" + variant,
                                 BuildClusterUnits (scenario.name, variant,
                                                    variant_nodes, clusters));
              auto result = EvaluateVariant (
                  scenario.name, variant, memory_count_total,
                  signal_count_total, variant_signal_evidence_edges,
                  memories_without_blob, non_text_blob_risk, variant_nodes,
                  edges, clusters);
              all_results.push_back (result);
              all_edges.insert (all_edges.end (), edges.begin (), edges.end ());
              all_clusters.insert (all_clusters.end (), clusters.begin (),
                                   clusters.end ());
            }

          scenario_json.push_back ({
            { "scenario", scenario.name },
            { "db_path", db_path.string () },
            { "event_count", scenario.events.size () },
            { "offline_scored",
              std::any_of (scenario.events.begin (), scenario.events.end (),
                           [] (const Event &event) {
                             return !event.offline_group.empty ();
                           }) },
            { "memory_count", memory_count_total },
            { "signal_count", signal_count_total },
            { "signal_evidence_edges", signal_evidence_edges },
            { "memories_without_blob", memories_without_blob },
            { "non_text_blob_risk", non_text_blob_risk },
            { "runtime_label_rows", runtime_label_rows },
          });
        }

      nlohmann::json result_rows = nlohmann::json::array ();
      for (const auto &row : all_results)
        {
          result_rows.push_back (VariantToJson (row));
        }

      nlohmann::json consolidation_rows = nlohmann::json::array ();
      for (const auto &row : all_consolidation_results)
        {
          consolidation_rows.push_back (ConsolidationToJson (row));
        }

      std::map<std::string, nlohmann::json> aggregate;
      for (const auto &row : all_results)
        {
          auto &agg = aggregate[row.variant];
          if (agg.is_null ())
            {
              agg = {
                { "variant", row.variant },
                { "scenario_count", 0 },
                { "node_count", 0 },
                { "memory_count_total", 0 },
                { "signal_count_total", 0 },
                { "signal_evidence_edges", 0 },
                { "edge_count", 0 },
                { "same_group_edges", 0 },
                { "cross_group_edges", 0 },
                { "mixed_cluster_count", 0 },
                { "cluster_count", 0 },
              };
            }
          agg["scenario_count"] = agg["scenario_count"].get<int> () + 1;
          agg["node_count"] = agg["node_count"].get<int> () + row.node_count;
          agg["memory_count_total"] = agg["memory_count_total"].get<int> ()
                                      + row.memory_count_total;
          agg["signal_count_total"] = agg["signal_count_total"].get<int> ()
                                      + row.signal_count_total;
          agg["signal_evidence_edges"]
              = agg["signal_evidence_edges"].get<int> ()
                + row.signal_evidence_edges;
          agg["edge_count"] = agg["edge_count"].get<int> () + row.edge_count;
          agg["same_group_edges"] = agg["same_group_edges"].get<int> ()
                                    + row.same_group_edges;
          agg["cross_group_edges"] = agg["cross_group_edges"].get<int> ()
                                     + row.cross_group_edges;
          agg["mixed_cluster_count"] = agg["mixed_cluster_count"].get<int> ()
                                       + row.mixed_cluster_count;
          agg["cluster_count"] = agg["cluster_count"].get<int> ()
                                 + row.cluster_count;
        }
      nlohmann::json aggregate_rows = nlohmann::json::array ();
      for (auto &[variant, agg] : aggregate)
        {
          const int edge_count = agg["edge_count"].get<int> ();
          const int node_count = agg["node_count"].get<int> ();
          const int memory_count_total = agg["memory_count_total"].get<int> ();
          const int signal_count_total = agg["signal_count_total"].get<int> ();
          const int signal_edges = agg["signal_evidence_edges"].get<int> ();
          agg["edge_precision"]
              = edge_count > 0
                    ? static_cast<double> (agg["same_group_edges"].get<int> ())
                          / static_cast<double> (edge_count)
                    : 1.0;
          agg["edge_cross_rate"]
              = edge_count > 0
                    ? static_cast<double> (agg["cross_group_edges"].get<int> ())
                          / static_cast<double> (edge_count)
                    : 0.0;
          agg["node_coverage"]
              = memory_count_total > 0
                    ? static_cast<double> (node_count)
                          / static_cast<double> (memory_count_total)
                    : 0.0;
          agg["signal_coverage"]
              = signal_count_total > 0
                    ? static_cast<double> (signal_edges)
                          / static_cast<double> (signal_count_total)
                    : 0.0;
          aggregate_rows.push_back (agg);
        }

      std::map<std::string, nlohmann::json> consolidation_aggregate;
      for (const auto &row : all_consolidation_results)
        {
          auto &agg = consolidation_aggregate[row.variant];
          if (agg.is_null ())
            {
              agg = {
                { "variant", row.variant },
                { "scenario_count", 0 },
                { "unit_count", 0 },
                { "memory_count_total", 0 },
                { "signal_count_total", 0 },
                { "covered_memories", 0 },
                { "covered_signals", 0 },
                { "mixed_units", 0 },
                { "multi_modal_units", 0 },
              };
            }
          agg["scenario_count"] = agg["scenario_count"].get<int> () + 1;
          agg["unit_count"] = agg["unit_count"].get<int> () + row.unit_count;
          agg["memory_count_total"] = agg["memory_count_total"].get<int> ()
                                      + row.memory_count_total;
          agg["signal_count_total"] = agg["signal_count_total"].get<int> ()
                                      + row.signal_count_total;
          agg["covered_memories"] = agg["covered_memories"].get<int> ()
                                    + row.covered_memories;
          agg["covered_signals"] = agg["covered_signals"].get<int> ()
                                   + row.covered_signals;
          agg["mixed_units"] = agg["mixed_units"].get<int> ()
                               + row.mixed_units;
          agg["multi_modal_units"] = agg["multi_modal_units"].get<int> ()
                                     + row.multi_modal_units;
        }
      nlohmann::json consolidation_aggregate_rows = nlohmann::json::array ();
      for (auto &[variant, agg] : consolidation_aggregate)
        {
          const int unit_count = agg["unit_count"].get<int> ();
          const int memory_total = agg["memory_count_total"].get<int> ();
          const int signal_total = agg["signal_count_total"].get<int> ();
          agg["memory_coverage"]
              = memory_total > 0
                    ? static_cast<double> (
                          agg["covered_memories"].get<int> ())
                          / static_cast<double> (memory_total)
                    : 0.0;
          agg["signal_coverage"]
              = signal_total > 0
                    ? static_cast<double> (agg["covered_signals"].get<int> ())
                          / static_cast<double> (signal_total)
                    : 0.0;
          agg["unit_precision"]
              = unit_count > 0
                    ? static_cast<double> (
                          unit_count - agg["mixed_units"].get<int> ())
                          / static_cast<double> (unit_count)
                    : 1.0;
          agg["mean_signals_per_unit"]
              = unit_count > 0
                    ? static_cast<double> (agg["covered_signals"].get<int> ())
                          / static_cast<double> (unit_count)
                    : 0.0;
          consolidation_aggregate_rows.push_back (agg);
        }

      nlohmann::json summary = {
        { "real_media_only", true },
        { "synthetic_encoder_used", false },
        { "production_path_changed", false },
        { "production_consolidation_called", false },
        { "runtime_labeling_disabled", true },
        { "runtime_label_rows", total_runtime_label_rows },
        { "offline_group_annotations_used_for_scoring_only", true },
        { "engine_api_path",
          "Cortext::ProcessImage/ProcessAudio/ProcessText" },
        { "assets", asset_manifest },
        { "scenarios", scenario_json },
        { "aggregate", aggregate_rows },
        { "consolidation_aggregate", consolidation_aggregate_rows },
        { "consolidation_results", consolidation_rows },
        { "results", result_rows },
      };

      {
        std::ofstream out (opts.output_dir
                           / "modality_agnostic_graph_summary.json");
        out << summary.dump (2) << '\n';
      }
      {
        std::ofstream out (opts.output_dir
                           / "modality_agnostic_graph_manifest.json");
        out << nlohmann::json ({
                  { "assets_dir", opts.assets_dir.string () },
                  { "video_dir", opts.video_dir.string () },
                  { "output_dir", opts.output_dir.string () },
                  { "models_dir", opts.models_dir },
                  { "benchmark_only", true },
                  { "labeling_disabled", true },
                  { "real_media_only", true },
                  { "variants",
                    { "blob_only_similarity",
                      "memory_embedding_similarity",
                      "memory_embedding_similarity_loose",
                      "memory_embedding_similarity_strict",
                      "memory_similarity_temporal",
                      "signal_aware_temporal",
                      "soft_anchor_continuity" } },
                  { "consolidation_variants",
                    { "memory_row_units",
                      "signal_evidence_units",
                      "cluster_blob_only_similarity",
                      "cluster_memory_embedding_similarity",
                      "cluster_memory_embedding_similarity_loose",
                      "cluster_memory_embedding_similarity_strict",
                      "cluster_memory_similarity_temporal",
                      "cluster_signal_aware_temporal",
                      "cluster_soft_anchor_continuity" } },
                  { "embedding_surface_rows_file",
                    "modality_agnostic_graph_embedding_rows.csv" },
                  { "embedding_surface_vectors_file",
                    "modality_agnostic_graph_embeddings.f32" },
              }).dump (2)
            << '\n';
      }
      WriteCasesCsv (opts.output_dir / "modality_agnostic_graph_cases.csv",
                     all_results);
      WriteConsolidationCasesCsv (
          opts.output_dir / "modality_agnostic_consolidation_cases.csv",
          all_consolidation_results);
      WriteConsolidationUnitsCsv (
          opts.output_dir / "modality_agnostic_consolidation_units.csv",
          all_consolidation_units);
      WriteEdgesCsv (opts.output_dir / "modality_agnostic_graph_edges.csv",
                     all_edges);
      WriteClustersCsv (
          opts.output_dir / "modality_agnostic_graph_clusters.csv",
          all_clusters);
      WriteEmbeddingSurface (
          opts.output_dir / "modality_agnostic_graph_embedding_rows.csv",
          opts.output_dir / "modality_agnostic_graph_embeddings.f32",
          std::move (all_embedding_rows));

      std::cout << summary.dump (2) << '\n';
      return 0;
    }
  catch (const std::exception &e)
    {
      std::cerr << "error: " << e.what () << '\n';
      return 1;
    }
}
