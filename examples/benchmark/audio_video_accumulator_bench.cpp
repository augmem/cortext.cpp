#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/accumulator.hpp>
#include <cortext/operations/accumulator_reset.hpp>
#include <cortext/operations/boundary.hpp>
#include <cortext/operations/memory_storage.hpp>
#include <cortext/operations/write_gate.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{

constexpr int kSampleRate = 16000;
constexpr int kEmbeddingDim = 256;

class BenchEncoder final : public cortext::Encoder
{
public:
  void
  EncodeText (const std::string & /*text*/,
              std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeAudio (const float * /*pcm*/, std::size_t /*num_samples*/,
               std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[1] = 1.0f;
  }

  void
  EncodeImage (const std::uint8_t * /*data*/, int /*width*/, int /*height*/,
               int /*channels*/, std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[2] = 1.0f;
  }
};

struct Args
{
  std::filesystem::path output_dir = "build/audio_video_accumulator_bench";
  std::filesystem::path audit_db;
  bool help = false;
};

struct SignalRow
{
  long long memory_id = 0;
  long long signal_id = 0;
  long long timestamp = 0;
  int serial_position = 0;
  std::string modality;
  std::string mime;
  std::size_t payload_bytes = 0;
  double audio_seconds = 0.0;
};

struct MemoryRow
{
  std::string scenario;
  long long episode_id = 0;
  long long memory_id = 0;
  long long start_ts = 0;
  long long end_ts = 0;
  long long n_signals = 0;
  std::string modality;
  std::size_t memory_payload_bytes = 0;
  bool memory_payload_is_audio_only_pcm = false;
  double memory_audio_seconds_if_pcm_concat = 0.0;
  int audio_signal_count = 0;
  int image_signal_count = 0;
  int text_signal_count = 0;
  double audio_signal_seconds_sum = 0.0;
  double min_audio_signal_seconds = 0.0;
  double max_audio_signal_seconds = 0.0;
  double memory_span_seconds = 0.0;
  double audio_timestamp_span_seconds = 0.0;
  std::vector<double> per_signal_audio_seconds;
};

struct EpisodeAuditRow
{
  std::string scenario;
  long long episode_id = 0;
  int memory_count = 0;
  int signal_count = 0;
  int audio_signal_count = 0;
  int image_signal_count = 0;
  int text_signal_count = 0;
  double audio_signal_seconds_sum = 0.0;
  bool has_mixed_modalities = false;
  bool mixed_memory_blob_violation = false;
  bool audio_only_concat_violation = false;
  bool cross_episode_timestamp_overlap = false;
  std::vector<long long> memory_ids;
  std::vector<std::string> signal_sequence;
};

long long
NowMs ()
{
  return std::chrono::duration_cast<std::chrono::milliseconds> (
             std::chrono::system_clock::now ().time_since_epoch ())
      .count ();
}

std::string
AnyString (const std::map<std::string, std::any> &row, const char *key)
{
  auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    {
      return {};
    }
  if (it->second.type () == typeid (std::string))
    {
      return std::any_cast<std::string> (it->second);
    }
  return {};
}

long long
AnyLong (const std::map<std::string, std::any> &row, const char *key)
{
  auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    {
      return 0;
    }
  return cortext::store::AnyToLongLong (it->second).value_or (0);
}

std::vector<unsigned char>
AnyBlob (const std::map<std::string, std::any> &row, const char *key)
{
  auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    {
      return {};
    }
  return cortext::store::BlobFromAny (it->second);
}

double
PcmSeconds (std::size_t bytes)
{
  if (bytes == 0)
    {
      return 0.0;
    }
  return static_cast<double> (bytes)
         / static_cast<double> (sizeof (float) * kSampleRate);
}

Eigen::VectorXf
UnitEmbedding (int axis)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[axis % kEmbeddingDim] = 1.0f;
  return v;
}

std::vector<unsigned char>
PcmPayload (int samples, float value)
{
  std::vector<float> pcm (static_cast<std::size_t> (samples), value);
  std::vector<unsigned char> bytes (pcm.size () * sizeof (float));
  std::memcpy (bytes.data (), pcm.data (), bytes.size ());
  return bytes;
}

std::vector<unsigned char>
ImagePayload (int width, int height, int channels, unsigned char value)
{
  return std::vector<unsigned char> (
      static_cast<std::size_t> (width * height * channels), value);
}

cortext::Signal
MakeAudioSignal (long long timestamp, int chunk_ms, bool flush, int axis)
{
  cortext::Signal signal;
  signal.embedding = UnitEmbedding (axis);
  signal.timestamp = static_cast<uint64_t> (timestamp);
  signal.source_id = "bench/source";
  signal.force_boundary = flush;
  signal.force_write = flush;
  signal.modality = "audio";
  signal.mimetype = "audio/pcm;format=f32";
  signal.sample_rate = kSampleRate;
  signal.num_samples = static_cast<std::size_t> (
      std::llround (static_cast<double> (kSampleRate) * chunk_ms / 1000.0));
  signal.payload = PcmPayload (static_cast<int> (signal.num_samples), 0.1f);
  return signal;
}

cortext::Signal
MakeImageSignal (long long timestamp, bool flush, int axis)
{
  cortext::Signal signal;
  signal.embedding = UnitEmbedding (axis);
  signal.timestamp = static_cast<uint64_t> (timestamp);
  signal.source_id = "bench/source";
  signal.force_boundary = flush;
  signal.force_write = flush;
  signal.modality = "image";
  signal.width = 32;
  signal.height = 32;
  signal.channels = 3;
  signal.mimetype = "image/raw;width=32;height=32;channels=3";
  signal.payload = ImagePayload (signal.width, signal.height, signal.channels,
                                 static_cast<unsigned char> (axis));
  return signal;
}

std::unique_ptr<cortext::DynamicOperationSet>
MakeAccumulatorPipeline ()
{
  auto pipeline = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<cortext::operations::UpdateAccumulator> (),
      std::make_unique<cortext::operations::DetectBoundary> (),
      std::make_unique<cortext::operations::ComputeWriteGate> (),
      std::make_unique<cortext::operations::MemoryStorage> (),
      std::make_unique<cortext::operations::ResetAccumulatorAfterFlush> ());
  return pipeline;
}

std::shared_ptr<cortext::Store>
CreateStore (const std::filesystem::path &db_path)
{
  auto unique_store = cortext::SQLiteStore::Create (db_path.string ());
  return std::shared_ptr<cortext::Store> (std::move (unique_store));
}

std::unique_ptr<cortext::SignalProcessor>
CreateProcessor (std::shared_ptr<cortext::Store> store,
                 BenchEncoder &encoder)
{
  cortext::SignalProcessor::Config config;
  config.focus = 0.5;
  config.sensitivity = 0.5;
  config.stability = 0.5;
  config.encoder = &encoder;
  return std::make_unique<cortext::SignalProcessor> (
      config, std::move (store), MakeAccumulatorPipeline ());
}

std::vector<unsigned char>
FetchPayload (cortext::Store &store, const std::vector<unsigned char> &blob_id)
{
  if (blob_id.empty ())
    {
      return {};
    }
  auto rows = store.Execute ("SELECT objstore_get(?1) AS payload", { blob_id });
  if (rows.empty ())
    {
      return {};
    }
  return AnyBlob (rows.front (), "payload");
}

std::string
JoinLongs (const std::vector<long long> &values, const char *separator)
{
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size (); ++i)
    {
      if (i != 0)
        {
          out << separator;
        }
      out << values[i];
    }
  return out.str ();
}

std::string
JoinStrings (const std::vector<std::string> &values, const char *separator)
{
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size (); ++i)
    {
      if (i != 0)
        {
          out << separator;
        }
      out << values[i];
    }
  return out.str ();
}

std::vector<MemoryRow>
AuditStore (cortext::Store &store, const std::string &scenario)
{
  std::vector<MemoryRow> memories;
  auto memory_rows = store.Execute (
      "SELECT memory_id, COALESCE(episode_id, 0) AS episode_id, start_ts, "
      "end_ts, n_signals, modality, blob_id "
      "FROM memories ORDER BY memory_id",
      {});

  for (const auto &row : memory_rows)
    {
      MemoryRow out;
      out.scenario = scenario;
      out.episode_id = AnyLong (row, "episode_id");
      out.memory_id = AnyLong (row, "memory_id");
      out.start_ts = AnyLong (row, "start_ts");
      out.end_ts = AnyLong (row, "end_ts");
      out.n_signals = AnyLong (row, "n_signals");
      out.modality = AnyString (row, "modality");
      out.memory_span_seconds
          = std::max (0.0,
                      static_cast<double> (out.end_ts - out.start_ts) / 1000.0);
      const auto memory_blob_id = AnyBlob (row, "blob_id");
      const auto memory_payload = FetchPayload (store, memory_blob_id);
      out.memory_payload_bytes = memory_payload.size ();

      auto signal_rows = store.Execute (
          "SELECT signal_id, timestamp, modality, mime, blob_id, "
          "serial_position "
          "FROM signals WHERE memory_id = ? ORDER BY serial_position, "
          "signal_id",
          { out.memory_id });

      long long first_audio_ts = std::numeric_limits<long long>::max ();
      long long last_audio_ts = std::numeric_limits<long long>::min ();
      for (const auto &sig : signal_rows)
        {
          const std::string modality = AnyString (sig, "modality");
          if (modality == "audio")
            {
              ++out.audio_signal_count;
              first_audio_ts = std::min (first_audio_ts, AnyLong (sig, "timestamp"));
              last_audio_ts = std::max (last_audio_ts, AnyLong (sig, "timestamp"));
              const auto blob_id = AnyBlob (sig, "blob_id");
              const auto payload = FetchPayload (store, blob_id);
              const double seconds = PcmSeconds (payload.size ());
              out.audio_signal_seconds_sum += seconds;
              out.per_signal_audio_seconds.push_back (seconds);
            }
          else if (modality == "image")
            {
              ++out.image_signal_count;
            }
          else if (modality == "text")
            {
              ++out.text_signal_count;
            }
        }

      if (!out.per_signal_audio_seconds.empty ())
        {
          auto [min_it, max_it]
              = std::minmax_element (out.per_signal_audio_seconds.begin (),
                                     out.per_signal_audio_seconds.end ());
          out.min_audio_signal_seconds = *min_it;
          out.max_audio_signal_seconds = *max_it;
          out.audio_timestamp_span_seconds
              = std::max (0.0, static_cast<double> (last_audio_ts - first_audio_ts)
                                    / 1000.0);
        }
      out.memory_payload_is_audio_only_pcm
          = out.audio_signal_count > 0 && out.image_signal_count == 0
            && out.text_signal_count == 0
            && out.audio_signal_count == static_cast<int> (out.n_signals);
      if (out.memory_payload_is_audio_only_pcm)
        {
          out.memory_audio_seconds_if_pcm_concat
              = PcmSeconds (out.memory_payload_bytes);
        }

      memories.push_back (std::move (out));
    }
  return memories;
}

std::vector<EpisodeAuditRow>
AuditEpisodes (cortext::Store &store, const std::string &scenario)
{
  struct MemoryPayloadState
  {
    long long episode_id = 0;
    int audio_count = 0;
    int image_count = 0;
    int text_count = 0;
    std::size_t memory_payload_bytes = 0;
    std::size_t audio_payload_bytes = 0;
  };

  std::unordered_map<long long, MemoryPayloadState> memory_state;
  auto memory_rows = store.Execute (
      "SELECT memory_id, COALESCE(episode_id, 0) AS episode_id, blob_id "
      "FROM memories ORDER BY memory_id",
      {});
  for (const auto &row : memory_rows)
    {
      const long long memory_id = AnyLong (row, "memory_id");
      auto &state = memory_state[memory_id];
      state.episode_id = AnyLong (row, "episode_id");
      state.memory_payload_bytes
          = FetchPayload (store, AnyBlob (row, "blob_id")).size ();
    }

  struct EpisodeBuild
  {
    EpisodeAuditRow row;
    std::unordered_set<long long> memory_ids_seen;
    std::unordered_set<std::string> modalities;
    long long min_ts = std::numeric_limits<long long>::max ();
    long long max_ts = std::numeric_limits<long long>::min ();
  };

  std::unordered_map<long long, EpisodeBuild> episodes;
  auto signal_rows = store.Execute (
      "SELECT COALESCE(m.episode_id, 0) AS episode_id, m.memory_id, "
      "       s.signal_id, s.modality, s.timestamp, s.serial_position, "
      "       s.blob_id "
      "FROM memories m "
      "JOIN signals s ON s.memory_id = m.memory_id "
      "ORDER BY episode_id, m.memory_id, s.serial_position, s.signal_id",
      {});

  for (const auto &row : signal_rows)
    {
      const long long episode_id = AnyLong (row, "episode_id");
      const long long memory_id = AnyLong (row, "memory_id");
      const long long signal_id = AnyLong (row, "signal_id");
      const long long timestamp = AnyLong (row, "timestamp");
      const std::string modality = AnyString (row, "modality");
      const std::size_t payload_bytes
          = FetchPayload (store, AnyBlob (row, "blob_id")).size ();

      auto &build = episodes[episode_id];
      auto &out = build.row;
      out.scenario = scenario;
      out.episode_id = episode_id;
      out.signal_count += 1;
      build.modalities.insert (modality);
      build.min_ts = std::min (build.min_ts, timestamp);
      build.max_ts = std::max (build.max_ts, timestamp);
      if (build.memory_ids_seen.insert (memory_id).second)
        {
          out.memory_ids.push_back (memory_id);
        }
      std::ostringstream seq;
      seq << "m" << memory_id << ":s" << signal_id << ":" << modality;
      out.signal_sequence.push_back (seq.str ());

      auto mem_it = memory_state.find (memory_id);
      if (mem_it != memory_state.end ())
        {
          auto &state = mem_it->second;
          if (modality == "audio")
            {
              state.audio_count += 1;
              state.audio_payload_bytes += payload_bytes;
            }
          else if (modality == "image")
            {
              state.image_count += 1;
            }
          else if (modality == "text")
            {
              state.text_count += 1;
            }
        }

      if (modality == "audio")
        {
          out.audio_signal_count += 1;
          out.audio_signal_seconds_sum += PcmSeconds (payload_bytes);
        }
      else if (modality == "image")
        {
          out.image_signal_count += 1;
        }
      else if (modality == "text")
        {
          out.text_signal_count += 1;
        }
    }

  for (auto &[memory_id, state] : memory_state)
    {
      const bool mixed = (state.audio_count > 0) + (state.image_count > 0)
                         + (state.text_count > 0) > 1;
      const bool audio_only = state.audio_count > 0 && state.image_count == 0
                              && state.text_count == 0;
      auto ep_it = episodes.find (state.episode_id);
      if (ep_it == episodes.end ())
        {
          continue;
        }
      if (mixed && state.memory_payload_bytes > 0)
        {
          ep_it->second.row.mixed_memory_blob_violation = true;
        }
      if (audio_only && state.memory_payload_bytes != state.audio_payload_bytes)
        {
          ep_it->second.row.audio_only_concat_violation = true;
        }
    }

  std::vector<std::pair<long long, long long>> episode_ranges;
  for (auto &[episode_id, build] : episodes)
    {
      build.row.memory_count = static_cast<int> (build.memory_ids_seen.size ());
      build.row.has_mixed_modalities = build.modalities.size () > 1;
      if (build.min_ts != std::numeric_limits<long long>::max ())
        {
          episode_ranges.push_back ({ build.min_ts, build.max_ts });
        }
    }
  std::sort (episode_ranges.begin (), episode_ranges.end ());
  for (std::size_t i = 1; i < episode_ranges.size (); ++i)
    {
      if (episode_ranges[i].first < episode_ranges[i - 1].second)
        {
          for (auto &[episode_id, build] : episodes)
            {
              (void)episode_id;
              build.row.cross_episode_timestamp_overlap = true;
            }
          break;
        }
    }

  std::vector<EpisodeAuditRow> out;
  out.reserve (episodes.size ());
  for (auto &[episode_id, build] : episodes)
    {
      (void)episode_id;
      std::sort (build.row.memory_ids.begin (), build.row.memory_ids.end ());
      out.push_back (std::move (build.row));
    }
  std::sort (out.begin (), out.end (), [] (const auto &a, const auto &b) {
    if (a.scenario != b.scenario)
      {
        return a.scenario < b.scenario;
      }
    return a.episode_id < b.episode_id;
  });
  return out;
}

std::vector<MemoryRow>
RunControlledScenario (const std::filesystem::path &output_dir,
                       const std::string &name,
                       const std::vector<cortext::Signal> &signals,
                       std::vector<EpisodeAuditRow> *episode_rows = nullptr)
{
  const auto db_path = output_dir / (name + ".db");
  std::filesystem::remove (db_path);
  auto store = CreateStore (db_path);
  BenchEncoder encoder;
  auto processor = CreateProcessor (store, encoder);

  for (const auto &signal : signals)
    {
      processor->Process (signal);
    }
  processor->Flush ();
  if (episode_rows)
    {
      auto scenario_episode_rows = AuditEpisodes (*store, name);
      episode_rows->insert (episode_rows->end (), scenario_episode_rows.begin (),
                            scenario_episode_rows.end ());
    }
  return AuditStore (*store, name);
}

std::vector<cortext::Signal>
AudioFinalFlushSignals (int chunks, int chunk_ms)
{
  std::vector<cortext::Signal> signals;
  signals.reserve (static_cast<std::size_t> (chunks));
  const long long base = NowMs ();
  for (int i = 0; i < chunks; ++i)
    {
      const bool flush = (i == chunks - 1);
      signals.push_back (
          MakeAudioSignal (base + static_cast<long long> (i * chunk_ms),
                           chunk_ms, flush, 1));
    }
  return signals;
}

std::vector<cortext::Signal>
AudioForceEachSignals (int chunks, int chunk_ms)
{
  std::vector<cortext::Signal> signals;
  signals.reserve (static_cast<std::size_t> (chunks));
  const long long base = NowMs ();
  for (int i = 0; i < chunks; ++i)
    {
      signals.push_back (
          MakeAudioSignal (base + static_cast<long long> (i * chunk_ms),
                           chunk_ms, true, 1));
    }
  return signals;
}

std::vector<cortext::Signal>
AudioImageInterleavedSignals ()
{
  std::vector<cortext::Signal> signals;
  const long long base = NowMs ();
  for (int i = 0; i < 5; ++i)
    {
      signals.push_back (MakeAudioSignal (base + i * 1000, 1000, false, 1));
      signals.push_back (MakeImageSignal (base + i * 1000 + 500, false, 2));
    }
  signals.push_back (MakeAudioSignal (base + 5500, 1000, true, 1));
  return signals;
}

std::vector<cortext::Signal>
AudioImageTwoEpisodeSignals ()
{
  std::vector<cortext::Signal> signals;
  const long long base = NowMs ();
  signals.push_back (MakeAudioSignal (base, 1000, false, 1));
  signals.push_back (MakeImageSignal (base + 500, false, 2));
  signals.push_back (MakeAudioSignal (base + 1000, 1000, true, 1));
  signals.push_back (MakeAudioSignal (base + 4000, 1000, false, 1));
  signals.push_back (MakeImageSignal (base + 4500, false, 2));
  signals.push_back (MakeAudioSignal (base + 5000, 1000, true, 1));
  return signals;
}

nlohmann::json
ToJson (const MemoryRow &row)
{
  return {
    { "scenario", row.scenario },
    { "episode_id", row.episode_id },
    { "memory_id", row.memory_id },
    { "start_ts", row.start_ts },
    { "end_ts", row.end_ts },
    { "n_signals", row.n_signals },
    { "modality", row.modality },
    { "memory_payload_bytes", row.memory_payload_bytes },
    { "memory_payload_is_audio_only_pcm", row.memory_payload_is_audio_only_pcm },
    { "memory_audio_seconds_if_pcm_concat",
      row.memory_audio_seconds_if_pcm_concat },
    { "audio_signal_count", row.audio_signal_count },
    { "image_signal_count", row.image_signal_count },
    { "text_signal_count", row.text_signal_count },
    { "audio_signal_seconds_sum", row.audio_signal_seconds_sum },
    { "min_audio_signal_seconds", row.min_audio_signal_seconds },
    { "max_audio_signal_seconds", row.max_audio_signal_seconds },
    { "memory_span_seconds", row.memory_span_seconds },
    { "audio_timestamp_span_seconds", row.audio_timestamp_span_seconds },
    { "per_signal_audio_seconds", row.per_signal_audio_seconds }
  };
}

nlohmann::json
ToJson (const EpisodeAuditRow &row)
{
  return {
    { "scenario", row.scenario },
    { "episode_id", row.episode_id },
    { "memory_count", row.memory_count },
    { "signal_count", row.signal_count },
    { "audio_signal_count", row.audio_signal_count },
    { "image_signal_count", row.image_signal_count },
    { "text_signal_count", row.text_signal_count },
    { "audio_signal_seconds_sum", row.audio_signal_seconds_sum },
    { "has_mixed_modalities", row.has_mixed_modalities },
    { "mixed_memory_blob_violation", row.mixed_memory_blob_violation },
    { "audio_only_concat_violation", row.audio_only_concat_violation },
    { "cross_episode_timestamp_overlap", row.cross_episode_timestamp_overlap },
    { "memory_ids", row.memory_ids },
    { "signal_sequence", row.signal_sequence }
  };
}

std::string
CsvEscape (const std::string &value)
{
  if (value.find_first_of (",\"\n") == std::string::npos)
    {
      return value;
    }
  std::string escaped = "\"";
  for (char ch : value)
    {
      if (ch == '"')
        {
          escaped += "\"\"";
        }
      else
        {
          escaped += ch;
        }
    }
  escaped += '"';
  return escaped;
}

void
WriteRowsCsv (const std::filesystem::path &path,
              const std::vector<MemoryRow> &rows)
{
  std::ofstream out (path);
  out << "scenario,episode_id,memory_id,start_ts,end_ts,n_signals,modality,"
         "memory_span_seconds,audio_signal_count,image_signal_count,"
         "text_signal_count,audio_signal_seconds_sum,min_audio_signal_seconds,"
         "max_audio_signal_seconds,audio_timestamp_span_seconds,"
         "memory_payload_bytes,memory_payload_is_audio_only_pcm,"
         "memory_audio_seconds_if_pcm_concat,"
         "per_signal_audio_seconds\n";
  out << std::fixed << std::setprecision (6);
  for (const auto &row : rows)
    {
      std::ostringstream per_signal;
      for (std::size_t i = 0; i < row.per_signal_audio_seconds.size (); ++i)
        {
          if (i != 0)
            {
              per_signal << "|";
            }
          per_signal << std::fixed << std::setprecision (3)
                     << row.per_signal_audio_seconds[i];
        }
      out << CsvEscape (row.scenario) << "," << row.episode_id << ","
          << row.memory_id << ","
          << row.start_ts << "," << row.end_ts << "," << row.n_signals
          << "," << CsvEscape (row.modality) << ","
          << row.memory_span_seconds << "," << row.audio_signal_count << ","
          << row.image_signal_count << "," << row.text_signal_count << ","
          << row.audio_signal_seconds_sum << ","
          << row.min_audio_signal_seconds << ","
          << row.max_audio_signal_seconds << ","
          << row.audio_timestamp_span_seconds << ","
          << row.memory_payload_bytes << ","
          << (row.memory_payload_is_audio_only_pcm ? 1 : 0) << ","
          << row.memory_audio_seconds_if_pcm_concat << ","
          << CsvEscape (per_signal.str ()) << "\n";
    }
}

void
WriteEpisodeRowsCsv (const std::filesystem::path &path,
                     const std::vector<EpisodeAuditRow> &rows)
{
  std::ofstream out (path);
  out << "scenario,episode_id,memory_count,signal_count,audio_signal_count,"
         "image_signal_count,text_signal_count,audio_signal_seconds_sum,"
         "has_mixed_modalities,mixed_memory_blob_violation,"
         "audio_only_concat_violation,cross_episode_timestamp_overlap,"
         "memory_ids,signal_sequence\n";
  out << std::fixed << std::setprecision (6);
  for (const auto &row : rows)
    {
      out << CsvEscape (row.scenario) << "," << row.episode_id << ","
          << row.memory_count << "," << row.signal_count << ","
          << row.audio_signal_count << "," << row.image_signal_count << ","
          << row.text_signal_count << "," << row.audio_signal_seconds_sum
          << "," << (row.has_mixed_modalities ? 1 : 0) << ","
          << (row.mixed_memory_blob_violation ? 1 : 0) << ","
          << (row.audio_only_concat_violation ? 1 : 0) << ","
          << (row.cross_episode_timestamp_overlap ? 1 : 0) << ","
          << CsvEscape (JoinLongs (row.memory_ids, "|")) << ","
          << CsvEscape (JoinStrings (row.signal_sequence, "|")) << "\n";
    }
}

nlohmann::json
BuildProgrammaticChecks (const std::vector<MemoryRow> &rows,
                         const std::vector<EpisodeAuditRow> &episode_rows)
{
  nlohmann::json checks;
  checks["pass"] = true;
  checks["failures"] = nlohmann::json::array ();

  auto fail = [&checks] (const std::string &message) {
    checks["pass"] = false;
    checks["failures"].push_back (message);
  };

  for (const auto &row : rows)
    {
      if (row.scenario == "audit_db")
        {
          continue;
        }
      const bool mixed = (row.audio_signal_count > 0)
                             + (row.image_signal_count > 0)
                             + (row.text_signal_count > 0)
                         > 1;
      if (mixed && row.memory_payload_bytes != 0)
        {
          fail (row.scenario + ": memory " + std::to_string (row.memory_id)
                + " is mixed-modal but has a memory-level blob");
        }
      if (row.memory_payload_is_audio_only_pcm)
        {
          const double diff = std::abs (row.memory_audio_seconds_if_pcm_concat
                                        - row.audio_signal_seconds_sum);
          if (diff > 1e-6)
            {
              fail (row.scenario + ": memory "
                    + std::to_string (row.memory_id)
                    + " audio-only blob duration does not match signal sum");
            }
        }
    }

  for (const auto &row : episode_rows)
    {
      if (row.scenario == "audit_db")
        {
          continue;
        }
      if (row.signal_count <= 0)
        {
          fail (row.scenario + ": episode " + std::to_string (row.episode_id)
                + " has no signals");
        }
      if (row.mixed_memory_blob_violation)
        {
          fail (row.scenario + ": episode " + std::to_string (row.episode_id)
                + " has a mixed memory with a concatenated blob");
        }
      if (row.audio_only_concat_violation)
        {
          fail (row.scenario + ": episode " + std::to_string (row.episode_id)
                + " has an audio-only memory without exact audio concatenation");
        }
      if (row.cross_episode_timestamp_overlap)
        {
          fail (row.scenario + ": episode timestamp ranges overlap");
        }
    }

  return checks;
}

void
PrintUsage (const char *argv0)
{
  std::cout
      << "Usage: " << argv0
      << " [--output-dir DIR] [--audit-db PATH]\n\n"
         "Runs benchmark-only accumulator duration experiments for audio/image "
         "ingress.\n"
         "--audit-db also audits an existing Cortext DB without mutating it.\n";
}

Args
ParseArgs (int argc, char **argv)
{
  Args args;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h")
        {
          args.help = true;
        }
      else if (arg == "--output-dir" && i + 1 < argc)
        {
          args.output_dir = argv[++i];
        }
      else if (arg == "--audit-db" && i + 1 < argc)
        {
          args.audit_db = argv[++i];
        }
      else
        {
          throw std::runtime_error ("Unknown or incomplete argument: " + arg);
        }
    }
  return args;
}

} // namespace

int
main (int argc, char **argv)
{
  try
    {
      const Args args = ParseArgs (argc, argv);
      if (args.help)
        {
          PrintUsage (argv[0]);
          return 0;
        }

      std::filesystem::create_directories (args.output_dir);
      std::vector<MemoryRow> rows;
      std::vector<EpisodeAuditRow> episode_rows;

      auto append = [&rows] (std::vector<MemoryRow> scenario_rows) {
        rows.insert (rows.end (), scenario_rows.begin (), scenario_rows.end ());
      };

      append (RunControlledScenario (args.output_dir,
                                     "audio_10x_1s_final_flush",
                                     AudioFinalFlushSignals (10, 1000),
                                     &episode_rows));
      append (RunControlledScenario (args.output_dir,
                                     "audio_5x_1s_force_each",
                                     AudioForceEachSignals (5, 1000),
                                     &episode_rows));
      append (RunControlledScenario (args.output_dir,
                                     "audio_video_interleaved_final_flush",
                                     AudioImageInterleavedSignals (),
                                     &episode_rows));
      append (RunControlledScenario (args.output_dir,
                                     "audio_video_two_episode_guard",
                                     AudioImageTwoEpisodeSignals (),
                                     &episode_rows));

      for (const int chunk_ms : { 250, 500, 1000, 2000 })
        {
          append (RunControlledScenario (
              args.output_dir,
              "audio_chunk_sweep_" + std::to_string (chunk_ms) + "ms",
              AudioFinalFlushSignals (4, chunk_ms), &episode_rows));
        }

      if (!args.audit_db.empty ())
        {
          auto audit_store = CreateStore (args.audit_db);
          append (AuditStore (*audit_store, "audit_db"));
          auto audit_episode_rows = AuditEpisodes (*audit_store, "audit_db");
          episode_rows.insert (episode_rows.end (), audit_episode_rows.begin (),
                               audit_episode_rows.end ());
        }

      nlohmann::json programmatic_checks
          = BuildProgrammaticChecks (rows, episode_rows);
      nlohmann::json summary;
      summary["row_count"] = rows.size ();
      summary["episode_row_count"] = episode_rows.size ();
      summary["programmatic_checks"] = programmatic_checks;
      summary["outputs"] = {
        { "memory_rows_csv",
          (args.output_dir / "audio_video_memory_rows.csv").string () },
        { "episode_groups_csv",
          (args.output_dir / "audio_video_episode_groups.csv").string () },
        { "summary_json",
          (args.output_dir / "audio_video_accumulator_summary.json")
              .string () }
      };
      summary["interpretation"] = {
        "per-signal audio payloads represent capture chunks; memory duration "
        "is the sum/span of the signals attached to one memory",
        "audio_10x_1s_final_flush should produce one memory with ten one-second "
        "signals if the accumulator can hold multi-second audio",
        "audio_5x_1s_force_each should produce one-second memories when every "
        "chunk is explicitly flushed"
      };
      summary["rows"] = nlohmann::json::array ();
      for (const auto &row : rows)
        {
          summary["rows"].push_back (ToJson (row));
        }
      summary["episode_rows"] = nlohmann::json::array ();
      for (const auto &row : episode_rows)
        {
          summary["episode_rows"].push_back (ToJson (row));
        }

      WriteRowsCsv (args.output_dir / "audio_video_memory_rows.csv", rows);
      WriteEpisodeRowsCsv (args.output_dir / "audio_video_episode_groups.csv",
                           episode_rows);
      std::ofstream json_out (
          args.output_dir / "audio_video_accumulator_summary.json");
      json_out << std::setw (2) << summary << "\n";

      std::cout << "Wrote "
                << (args.output_dir / "audio_video_accumulator_summary.json")
                       .string ()
                << "\n";
      if (!programmatic_checks.value ("pass", false))
        {
          return 2;
        }
      return 0;
    }
  catch (const std::exception &e)
    {
      std::cerr << "audio_video_accumulator_bench failed: " << e.what ()
                << "\n";
      return 1;
    }
}
