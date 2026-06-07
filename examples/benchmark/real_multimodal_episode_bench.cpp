#include <cortext/cortext.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/store.hpp>
#include <cortext/store/utils.hpp>

#include <nlohmann/json.hpp>

#include <any>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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
  std::filesystem::path output_dir = "build/real_multimodal_episode_bench";
  std::string models_dir = "models";
};

struct Event
{
  std::string id;
  std::string episode;
  std::string modality;
  std::string text;
  std::filesystem::path path;
  int width = 0;
  int height = 0;
  int channels = 0;
};

struct Scenario
{
  std::string name;
  std::vector<Event> events;
  bool expect_dog_fusion = false;
  bool expect_event_split = false;
};

struct SignalRow
{
  long long signal_id = 0;
  long long memory_id = 0;
  long long db_episode_id = 0;
  long long serial_position = 0;
  std::string modality;
  std::string mime;
  std::size_t payload_bytes = 0;
};

struct MemoryGroup
{
  long long memory_id = 0;
  long long db_episode_id = 0;
  std::size_t memory_payload_bytes = 0;
  std::set<std::string> episodes;
  std::set<std::string> event_ids;
  std::set<std::string> modalities;
  std::vector<long long> signal_ids;
  std::vector<std::string> signal_sequence;
  int audio_signal_count = 0;
  int image_signal_count = 0;
  int text_signal_count = 0;
  std::size_t audio_payload_bytes = 0;
  bool mixed_modalities = false;
  bool mixed_episode_labels = false;
  bool mixed_memory_blob_violation = false;
  bool audio_only_concat_ok = true;
};

void
PrintUsage ()
{
  std::cout << "Usage: cortext_real_multimodal_episode_bench"
            << " [--assets-dir <path>] [--output-dir <path>]"
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

std::vector<float>
ReadFloat32 (const std::filesystem::path &path)
{
  const auto bytes = ReadBytes (path);
  if (bytes.size () % sizeof (float) != 0)
    {
      throw std::runtime_error ("Float32 audio file has invalid byte length: "
                                + path.string ());
    }
  std::vector<float> out (bytes.size () / sizeof (float));
  std::memcpy (out.data (), bytes.data (), bytes.size ());
  return out;
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
Join (const std::set<std::string> &items)
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

std::string
JoinStrings (const std::vector<std::string> &items)
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

std::vector<unsigned char>
AnyBlob (const std::any &value)
{
  if (!value.has_value ())
    {
      return {};
    }
  return cortext::store::BlobFromAny (value);
}

std::vector<unsigned char>
FetchPayload (cortext::Store &store, const std::vector<unsigned char> &blob_id)
{
  if (blob_id.empty ())
    {
      return {};
    }
  const auto rows
      = store.Execute ("SELECT objstore_get(?1) AS payload", { blob_id });
  if (rows.empty () || rows.front ().count ("payload") == 0)
    {
      return {};
    }
  return AnyBlob (rows.front ().at ("payload"));
}

std::unordered_map<long long, std::size_t>
LoadMemoryPayloadBytes (cortext::Store &store)
{
  std::unordered_map<long long, std::size_t> out;
  const auto rows = store.Execute (
      "SELECT memory_id, blob_id FROM memories ORDER BY memory_id");
  for (const auto &row : rows)
    {
      const long long memory_id = AnyLong (row.at ("memory_id"));
      const auto blob_it = row.find ("blob_id");
      if (blob_it == row.end ())
        {
          out[memory_id] = 0;
          continue;
        }
      out[memory_id] = FetchPayload (store, AnyBlob (blob_it->second)).size ();
    }
  return out;
}

std::vector<SignalRow>
LoadSignals (cortext::Store &store)
{
  std::vector<SignalRow> out;
  auto rows = store.Execute (
      "SELECT s.signal_id, COALESCE(s.memory_id, 0) AS memory_id, "
      "       COALESCE(m.episode_id, 0) AS db_episode_id, "
      "       s.modality, COALESCE(s.mime, '') AS mime, "
      "       s.serial_position, s.blob_id "
      "FROM signals s "
      "LEFT JOIN memories m ON m.memory_id = s.memory_id "
      "WHERE s.write_decision = 1 "
      "ORDER BY s.signal_id");
  for (const auto &row : rows)
    {
      SignalRow signal;
      signal.signal_id = AnyLong (row.at ("signal_id"));
      signal.memory_id = AnyLong (row.at ("memory_id"));
      signal.db_episode_id = AnyLong (row.at ("db_episode_id"));
      signal.serial_position = AnyLong (row.at ("serial_position"));
      signal.modality = AnyString (row.at ("modality"));
      signal.mime = AnyString (row.at ("mime"));
      const auto blob_it = row.find ("blob_id");
      if (blob_it != row.end ())
        {
          signal.payload_bytes
              = FetchPayload (store, AnyBlob (blob_it->second)).size ();
        }
      out.push_back (std::move (signal));
    }
  return out;
}

std::vector<MemoryGroup>
BuildGroups (const std::vector<SignalRow> &signals,
             const std::vector<Event> &events,
             const std::unordered_map<long long, std::size_t> &memory_payload_bytes)
{
  std::map<long long, MemoryGroup> by_memory;
  std::vector<bool> assigned (events.size (), false);
  for (const auto &signal : signals)
    {
      if (signal.memory_id == 0)
        {
          continue;
        }
      std::size_t event_index = events.size ();
      for (std::size_t i = 0; i < events.size (); ++i)
        {
          if (!assigned[i] && events[i].modality == signal.modality)
            {
              event_index = i;
              assigned[i] = true;
              break;
            }
        }
      if (event_index == events.size ())
        {
          continue;
        }
      const auto &event = events[event_index];
      auto &group = by_memory[signal.memory_id];
      group.memory_id = signal.memory_id;
      group.db_episode_id = signal.db_episode_id;
      auto payload_it = memory_payload_bytes.find (signal.memory_id);
      if (payload_it != memory_payload_bytes.end ())
        {
          group.memory_payload_bytes = payload_it->second;
        }
      group.episodes.insert (event.episode);
      group.event_ids.insert (event.id);
      group.modalities.insert (event.modality);
      group.signal_ids.push_back (signal.signal_id);
      std::ostringstream seq;
      seq << event.id << ":" << signal.modality << ":s" << signal.signal_id;
      group.signal_sequence.push_back (seq.str ());
      if (signal.modality == "audio")
        {
          group.audio_signal_count += 1;
          group.audio_payload_bytes += signal.payload_bytes;
        }
      else if (signal.modality == "image")
        {
          group.image_signal_count += 1;
        }
      else if (signal.modality == "text")
        {
          group.text_signal_count += 1;
        }
    }
  std::vector<MemoryGroup> out;
  for (auto &kv : by_memory)
    {
      auto &group = kv.second;
      group.mixed_modalities = group.modalities.size () > 1;
      group.mixed_episode_labels = group.episodes.size () > 1;
      group.mixed_memory_blob_violation
          = group.mixed_modalities && group.memory_payload_bytes > 0;
      if (group.audio_signal_count > 0 && group.image_signal_count == 0
          && group.text_signal_count == 0)
        {
          group.audio_only_concat_ok
              = group.memory_payload_bytes == group.audio_payload_bytes;
        }
      out.push_back (std::move (kv.second));
    }
  return out;
}

std::vector<Scenario>
BuildScenarios (const std::filesystem::path &assets_dir)
{
  const auto raw = assets_dir / "raw";
  Event dog_image{ "dog_image", "dog_naming", "image", "",
                   raw / "dog_384x384.rgb", 384, 384, 3 };
  Event dog_text{ "dog_name_text", "dog_naming", "text",
                  "That dog's name is Bailey.", {}, 0, 0, 0 };
  Event dog_audio{ "dog_name_audio", "dog_naming", "audio", "",
                   raw / "bailey_16k_mono.f32", 0, 0, 0 };
  Event car_image{ "car_crash_image", "car_crash", "image", "",
                   raw / "car_crash_384x384.rgb", 384, 384, 3 };
  Event car_text{ "car_crash_text", "car_crash", "text",
                  "A car crashed into a tree.", {}, 0, 0, 0 };
  Event crash_audio{ "crash_audio", "car_crash", "audio", "",
                     raw / "crash_16k_mono.f32", 0, 0, 0 };

  return {
    { "real_dog_name_image_text_audio",
      { dog_image, dog_text, dog_audio },
      true,
      false },
    { "real_dog_name_then_car_crash",
      { dog_image, dog_text, dog_audio, car_image, car_text, crash_audio },
      true,
      true },
    { "real_dog_name_audio_then_car_crash_audio",
      { dog_image, dog_audio, car_image, crash_audio },
      true,
      true },
  };
}

nlohmann::json
ProcessScenario (const Scenario &scenario, const Options &opts)
{
  const auto db_path = opts.output_dir / (scenario.name + ".sqlite");
  std::filesystem::remove (db_path);

  cortext::Cortext::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  std::vector<nlohmann::json> steps;
  {
    auto engine = cortext::Cortext::Create (cfg, db_path.string (),
                                            opts.models_dir);
    for (const auto &event : scenario.events)
      {
        cortext::Cortext::Context ctx;
        if (event.modality == "image")
          {
            const auto pixels = ReadBytes (event.path);
            ctx = engine->ProcessImage (pixels.data (), event.width,
                                        event.height, event.channels,
                                        "chat/user");
          }
        else if (event.modality == "audio")
          {
            const auto pcm = ReadFloat32 (event.path);
            ctx = engine->ProcessAudio (pcm.data (), pcm.size (), "chat/user");
          }
        else
          {
            ctx = engine->ProcessText (event.text, "chat/user");
          }
        steps.push_back ({
          { "event_id", event.id },
          { "episode", event.episode },
          { "modality", event.modality },
          { "encode_ms", ctx.encode_ms },
          { "process_ms", ctx.process_ms },
          { "total_ms", ctx.total_ms },
          { "stored_memory_id",
            ctx.output.stored_memory_id.value_or (static_cast<long long> (0)) },
          { "boundary_score", ctx.boundary_score.value_or (0.0) },
        });
      }
    engine->Flush ();
  }

  auto store = cortext::SQLiteStore::Create (db_path.string ());
  const auto signals = LoadSignals (*store);
  const auto memory_payload_bytes = LoadMemoryPayloadBytes (*store);
  const auto groups = BuildGroups (signals, scenario.events,
                                   memory_payload_bytes);

  bool dog_fused = false;
  bool dog_image_audio_fused = false;
  bool dog_all_modalities_fused = false;
  bool mixed_events = false;
  bool mixed_memory_blob_violation = false;
  bool audio_only_concat_violation = false;
  for (const auto &group : groups)
    {
      if (group.episodes.count ("dog_naming") > 0
          && group.modalities.count ("image") > 0
          && (group.modalities.count ("text") > 0
              || group.modalities.count ("audio") > 0))
        {
          dog_fused = true;
        }
      if (group.event_ids.count ("dog_image") > 0
          && group.event_ids.count ("dog_name_audio") > 0)
        {
          dog_image_audio_fused = true;
        }
      if (group.event_ids.count ("dog_image") > 0
          && group.event_ids.count ("dog_name_text") > 0
          && group.event_ids.count ("dog_name_audio") > 0)
        {
          dog_all_modalities_fused = true;
        }
      if (group.episodes.size () > 1)
        {
          mixed_events = true;
        }
      if (group.mixed_memory_blob_violation)
        {
          mixed_memory_blob_violation = true;
        }
      if (!group.audio_only_concat_ok)
        {
          audio_only_concat_violation = true;
        }
    }

  nlohmann::json memories = nlohmann::json::array ();
  for (const auto &group : groups)
    {
      memories.push_back ({
        { "memory_id", group.memory_id },
        { "db_episode_id", group.db_episode_id },
        { "episodes", Join (group.episodes) },
        { "event_ids", Join (group.event_ids) },
        { "modalities", Join (group.modalities) },
        { "signal_ids", group.signal_ids },
        { "signal_sequence", group.signal_sequence },
        { "audio_signal_count", group.audio_signal_count },
        { "image_signal_count", group.image_signal_count },
        { "text_signal_count", group.text_signal_count },
        { "audio_payload_bytes", group.audio_payload_bytes },
        { "memory_payload_bytes", group.memory_payload_bytes },
        { "mixed_modalities", group.mixed_modalities },
        { "mixed_episode_labels", group.mixed_episode_labels },
        { "mixed_memory_blob_violation",
          group.mixed_memory_blob_violation },
        { "audio_only_concat_ok", group.audio_only_concat_ok },
      });
    }

  return {
    { "scenario", scenario.name },
    { "event_count", scenario.events.size () },
    { "processed_event_count", scenario.events.size () },
    { "stored_signal_count", signals.size () },
    { "memory_count", groups.size () },
    { "dog_name_fused", dog_fused },
    { "dog_image_audio_fused", dog_image_audio_fused },
    { "dog_all_modalities_fused", dog_all_modalities_fused },
    { "event_split_success", scenario.expect_event_split ? !mixed_events : true },
    { "mixed_events_in_any_memory", mixed_events },
    { "mixed_memory_blob_violation", mixed_memory_blob_violation },
    { "audio_only_concat_violation", audio_only_concat_violation },
    { "programmatic_pass",
      !mixed_events && !mixed_memory_blob_violation
          && !audio_only_concat_violation },
    { "steps", std::move (steps) },
    { "memories", std::move (memories) },
  };
}

void
WriteCasesCsv (const std::filesystem::path &path, const nlohmann::json &cases)
{
  std::ofstream out (path);
  out << "scenario,event_count,processed_event_count,stored_signal_count,"
      << "memory_count,dog_name_fused,"
      << "dog_image_audio_fused,dog_all_modalities_fused,event_split_success,"
      << "mixed_events_in_any_memory,mixed_memory_blob_violation,"
      << "audio_only_concat_violation,programmatic_pass\n";
  for (const auto &row : cases)
    {
      out << CsvEscape (row.value ("scenario", "")) << ','
          << row.value ("event_count", 0) << ','
          << row.value ("processed_event_count", 0) << ','
          << row.value ("stored_signal_count", 0) << ','
          << row.value ("memory_count", 0) << ','
          << (row.value ("dog_name_fused", false) ? 1 : 0) << ','
          << (row.value ("dog_image_audio_fused", false) ? 1 : 0) << ','
          << (row.value ("dog_all_modalities_fused", false) ? 1 : 0) << ','
          << (row.value ("event_split_success", false) ? 1 : 0) << ','
          << (row.value ("mixed_events_in_any_memory", false) ? 1 : 0) << ','
          << (row.value ("mixed_memory_blob_violation", false) ? 1 : 0) << ','
          << (row.value ("audio_only_concat_violation", false) ? 1 : 0) << ','
          << (row.value ("programmatic_pass", false) ? 1 : 0)
          << '\n';
    }
}

void
WriteEpisodeGroupsCsv (const std::filesystem::path &path,
                       const nlohmann::json &cases)
{
  std::ofstream out (path);
  out << "scenario,memory_id,db_episode_id,episodes,event_ids,modalities,"
      << "signal_ids,signal_sequence,audio_signal_count,image_signal_count,"
      << "text_signal_count,audio_payload_bytes,memory_payload_bytes,"
      << "mixed_modalities,mixed_episode_labels,mixed_memory_blob_violation,"
      << "audio_only_concat_ok\n";
  for (const auto &scenario : cases)
    {
      for (const auto &memory : scenario.value ("memories",
                                                nlohmann::json::array ()))
        {
          std::vector<long long> signal_ids;
          for (const auto &id : memory.value ("signal_ids",
                                              nlohmann::json::array ()))
            {
              signal_ids.push_back (id.get<long long> ());
            }
          std::vector<std::string> sequence;
          for (const auto &entry : memory.value ("signal_sequence",
                                                 nlohmann::json::array ()))
            {
              sequence.push_back (entry.get<std::string> ());
            }
          out << CsvEscape (scenario.value ("scenario", "")) << ','
              << memory.value ("memory_id", 0LL) << ','
              << memory.value ("db_episode_id", 0LL) << ','
              << CsvEscape (memory.value ("episodes", "")) << ','
              << CsvEscape (memory.value ("event_ids", "")) << ','
              << CsvEscape (memory.value ("modalities", "")) << ','
              << CsvEscape (JoinLongs (signal_ids)) << ','
              << CsvEscape (JoinStrings (sequence)) << ','
              << memory.value ("audio_signal_count", 0) << ','
              << memory.value ("image_signal_count", 0) << ','
              << memory.value ("text_signal_count", 0) << ','
              << memory.value ("audio_payload_bytes", 0UL) << ','
              << memory.value ("memory_payload_bytes", 0UL) << ','
              << (memory.value ("mixed_modalities", false) ? 1 : 0) << ','
              << (memory.value ("mixed_episode_labels", false) ? 1 : 0)
              << ','
              << (memory.value ("mixed_memory_blob_violation", false) ? 1 : 0)
              << ','
              << (memory.value ("audio_only_concat_ok", true) ? 1 : 0)
              << '\n';
        }
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
      nlohmann::json manifest = nlohmann::json::object ();
      if (std::filesystem::exists (manifest_path))
        {
          std::ifstream in (manifest_path);
          in >> manifest;
        }

      nlohmann::json cases = nlohmann::json::array ();
      for (const auto &scenario : BuildScenarios (opts.assets_dir))
        {
          cases.push_back (ProcessScenario (scenario, opts));
        }

      int dog_fused = 0;
      int dog_image_audio_fused = 0;
      int dog_all_modalities_fused = 0;
      int split_ok = 0;
      int mixed = 0;
      int mixed_blob_violations = 0;
      int audio_concat_violations = 0;
      int programmatic_pass = 0;
      for (const auto &row : cases)
        {
          dog_fused += row.value ("dog_name_fused", false) ? 1 : 0;
          dog_image_audio_fused += row.value ("dog_image_audio_fused", false) ? 1 : 0;
          dog_all_modalities_fused += row.value ("dog_all_modalities_fused", false) ? 1 : 0;
          split_ok += row.value ("event_split_success", false) ? 1 : 0;
          mixed += row.value ("mixed_events_in_any_memory", false) ? 1 : 0;
          mixed_blob_violations
              += row.value ("mixed_memory_blob_violation", false) ? 1 : 0;
          audio_concat_violations
              += row.value ("audio_only_concat_violation", false) ? 1 : 0;
          programmatic_pass += row.value ("programmatic_pass", false) ? 1 : 0;
        }

      nlohmann::json summary = {
        { "real_media", true },
        { "engine_api_path", "Cortext::ProcessImage/ProcessAudio/ProcessText" },
        { "force_boundary_used", false },
        { "signal_filter_used", false },
        { "signal_filter_policy", "none" },
        { "case_count", cases.size () },
        { "dog_name_fused", dog_fused },
        { "dog_image_audio_fused", dog_image_audio_fused },
        { "dog_all_modalities_fused", dog_all_modalities_fused },
        { "event_split_success", split_ok },
        { "mixed_events_in_any_memory", mixed },
        { "mixed_memory_blob_violations", mixed_blob_violations },
        { "audio_only_concat_violations", audio_concat_violations },
        { "programmatic_pass_count", programmatic_pass },
        { "programmatic_all_pass",
          programmatic_pass == static_cast<int> (cases.size ()) },
        { "assets", manifest },
        { "cases", cases },
      };

      {
        std::ofstream out (opts.output_dir / "real_multimodal_episode_summary.json");
        out << summary.dump (2) << '\n';
      }
      {
        std::ofstream out (opts.output_dir / "real_multimodal_episode_cases.json");
        out << cases.dump (2) << '\n';
      }
      WriteCasesCsv (opts.output_dir / "real_multimodal_episode_cases.csv", cases);
      WriteEpisodeGroupsCsv (
          opts.output_dir / "real_multimodal_episode_groups.csv", cases);

      std::cout << summary.dump (2) << '\n';
      return summary.value ("programmatic_all_pass", false) ? 0 : 2;
    }
  catch (const std::exception &e)
    {
      std::cerr << "error: " << e.what () << '\n';
      return 1;
    }
}
