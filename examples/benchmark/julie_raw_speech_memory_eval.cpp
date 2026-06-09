#include <cortext/consolidation_mode.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/cortext.hpp>
#include <cortext/retention.hpp>

#include "include/audio_loader.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace
{

constexpr const char *kGabeSourceId = "Gabe";
constexpr const char *kJulieSourceId = "Julie";

struct Config
{
  fs::path manifest_path;
  fs::path db_path = "build/julie_raw_speech_memory_eval.sqlite";
  fs::path output_path = "build/julie_raw_speech_memory_eval.json";
  fs::path label_bank_path = "data/label_bank/metadata.json";
  std::string models_dir = "models";
  std::string modality = "audio";
  double focus = 0.35;
  double sensitivity = 0.65;
  double stability = 0.60;
  int warmup_messages = 20;
  int probe_stride = 12;
  int rag_top_k = 5;
  int active_history_token_budget = 8000;
  bool deep_consolidation = true;
  bool use_label_bank = true;
};

struct Record
{
  int local_index = 0;
  int original_index = 0;
  std::uint64_t timestamp = 0;
  std::string speaker_role;
  std::string audio_path;
  int text_chars = 0;
  double tts_ms = 0.0;
  double ffmpeg_ms = 0.0;
};

struct TextDoc
{
  int index = 0;
  std::uint64_t timestamp = 0;
  std::string source_id;
  std::string text;
  std::unordered_set<std::string> tokens;
};

struct Aggregate
{
  int probes = 0;
  long long prompt_tokens = 0;
  long long context_tokens = 0;
  long long context_items = 0;
  double latency_ms = 0.0;
  std::vector<long long> prompt_token_samples;
  std::vector<long long> context_token_samples;
};

std::string
Trim (const std::string &s)
{
  size_t first = 0;
  while (first < s.size ()
         && std::isspace (static_cast<unsigned char> (s[first])))
    ++first;
  size_t last = s.size ();
  while (last > first
         && std::isspace (static_cast<unsigned char> (s[last - 1])))
    --last;
  return s.substr (first, last - first);
}

bool
StartsWithDate (const std::string &line)
{
  if (line.size () < 19)
    return false;
  return std::isdigit (line[0]) && std::isdigit (line[1])
         && std::isdigit (line[2]) && std::isdigit (line[3])
         && line[4] == '-' && std::isdigit (line[5])
         && std::isdigit (line[6]) && line[7] == '-'
         && std::isdigit (line[8]) && std::isdigit (line[9])
         && line[10] == ' ' && std::isdigit (line[11])
         && std::isdigit (line[12]) && (line[13] == ':' || line[13] == ' ')
         && std::isdigit (line[14]) && std::isdigit (line[15])
         && (line[16] == ':' || line[16] == ' ')
         && std::isdigit (line[17]) && std::isdigit (line[18]);
}

std::optional<std::uint64_t>
ParseTimestamp (const std::string &prefix)
{
  std::tm tm{};
  std::string stamp = prefix.substr (0, 19);
  if (stamp.size () >= 19 && stamp[13] == ' ' && stamp[16] == ' ')
    {
      stamp[13] = ':';
      stamp[16] = ':';
    }
  std::istringstream in (stamp);
  in >> std::get_time (&tm, "%Y-%m-%d %H:%M:%S");
  if (in.fail ())
    return std::nullopt;
  tm.tm_isdst = -1;
  const std::time_t seconds = std::mktime (&tm);
  if (seconds < 0)
    return std::nullopt;
  return static_cast<std::uint64_t> (seconds) * 1000ULL;
}

std::vector<TextDoc>
ParseTranscriptDocs (const fs::path &path)
{
  std::ifstream in (path);
  if (!in)
    throw std::runtime_error ("failed to open transcript: " + path.string ());

  std::vector<TextDoc> docs;
  std::string line;
  std::string pending_header;
  std::string pending_text;

  auto flush = [&] {
    if (pending_header.empty ())
      return;
    auto timestamp = ParseTimestamp (pending_header);
    std::string text = Trim (pending_text);
    if (timestamp && !text.empty ())
      {
        TextDoc doc;
        doc.index = static_cast<int> (docs.size ());
        doc.timestamp = *timestamp;
        doc.source_id = pending_header.find (" from ") != std::string::npos
                            ? kJulieSourceId
                            : kGabeSourceId;
        doc.text = std::move (text);
        docs.push_back (std::move (doc));
      }
    pending_header.clear ();
    pending_text.clear ();
  };

  while (std::getline (in, line))
    {
      if (line.rfind ("----------------------------------------------------", 0) == 0)
        {
          flush ();
          continue;
        }
      if (StartsWithDate (line))
        {
          flush ();
          pending_header = line;
          continue;
        }
      if (!pending_header.empty ())
        {
          if (!pending_text.empty ())
            pending_text.push_back ('\n');
          pending_text += line;
        }
    }
  flush ();
  return docs;
}

std::unordered_set<std::string>
Tokens (const std::string &text)
{
  static const std::unordered_set<std::string> stop = {
    "the", "and", "you", "that", "for", "with", "this", "have", "just",
    "but", "not", "are", "was", "what", "from", "they", "your", "our",
    "she", "him", "her", "his", "them", "then", "there", "were", "been",
    "will", "would", "could", "should", "about", "like", "yeah", "okay"
  };
  std::unordered_set<std::string> out;
  std::string cur;
  for (unsigned char c : text)
    {
      if (std::isalnum (c))
        cur.push_back (static_cast<char> (std::tolower (c)));
      else if (!cur.empty ())
        {
          if (cur.size () >= 3 && !stop.count (cur))
            out.insert (cur);
          cur.clear ();
        }
    }
  if (cur.size () >= 3 && !stop.count (cur))
    out.insert (cur);
  return out;
}

long long
EstimateTokens (std::size_t chars)
{
  return static_cast<long long> ((chars + 3) / 4);
}

int
LocalDayBucket (std::uint64_t timestamp_ms)
{
  const std::time_t seconds = static_cast<std::time_t> (timestamp_ms / 1000ULL);
  std::tm local{};
#if defined(_WIN32)
  localtime_s (&local, &seconds);
#else
  localtime_r (&seconds, &local);
#endif
  return (local.tm_year + 1900) * 1000 + local.tm_yday;
}

std::vector<TextDoc>
RagTopK (const std::vector<TextDoc> &docs,
         const std::unordered_set<std::string> &query_tokens,
         std::uint64_t timestamp, int k)
{
  std::vector<std::pair<double, TextDoc>> scored;
  for (const auto &doc : docs)
    {
      if (doc.timestamp >= timestamp)
        continue;
      int overlap = 0;
      for (const auto &token : query_tokens)
        {
          if (doc.tokens.count (token))
            ++overlap;
        }
      const double recency = 1.0 / (1.0 + static_cast<double> (
                                      timestamp - doc.timestamp) / 86400000.0);
      const double score = static_cast<double> (overlap) + 0.05 * recency;
      if (score > 0.0)
        scored.push_back ({ score, doc });
    }
  std::sort (scored.begin (), scored.end (),
             [] (const auto &a, const auto &b) {
               return a.first > b.first;
             });
  std::vector<TextDoc> out;
  for (const auto &item : scored)
    {
      if (static_cast<int> (out.size ()) >= k)
        break;
      out.push_back (item.second);
    }
  return out;
}

std::string
BuildRagContext (const std::vector<TextDoc> &docs)
{
  std::ostringstream out;
  for (const auto &doc : docs)
    out << doc.source_id << ": " << doc.text << "\n";
  return out.str ();
}

std::string
BuildActiveHistory (const std::vector<TextDoc> &docs, int token_budget,
                    int *items)
{
  std::vector<std::string> lines;
  long long tokens = 0;
  for (auto it = docs.rbegin (); it != docs.rend (); ++it)
    {
      std::string line = it->source_id + ": " + it->text + "\n";
      const long long line_tokens = EstimateTokens (line.size ());
      if (tokens + line_tokens > token_budget)
        break;
      tokens += line_tokens;
      lines.push_back (std::move (line));
    }
  std::reverse (lines.begin (), lines.end ());
  if (items)
    *items = static_cast<int> (lines.size ());
  std::ostringstream out;
  for (const auto &line : lines)
    out << line;
  return out.str ();
}

std::vector<Record>
LoadManifestRecords (const fs::path &path)
{
  std::ifstream in (path);
  if (!in)
    throw std::runtime_error ("failed to open manifest: " + path.string ());
  nlohmann::json manifest = nlohmann::json::parse (in);
  std::vector<Record> records;
  for (const auto &row : manifest.at ("records"))
    {
      Record record;
      record.local_index = row.value ("local_index", 0);
      record.original_index = row.value ("original_index", 0);
      record.timestamp = row.value ("timestamp", 0ULL);
      record.speaker_role = row.value ("speaker_role", "");
      record.audio_path = row.value ("audio_path", "");
      record.text_chars = row.value ("text_chars", 0);
      record.tts_ms = row.value ("tts_ms", 0.0);
      record.ffmpeg_ms = row.value ("ffmpeg_ms", 0.0);
      records.push_back (std::move (record));
    }
  return records;
}

std::string
StreamSourceId (const Record &record)
{
  return record.speaker_role == "contact" ? kJulieSourceId : kGabeSourceId;
}

template <typename Callback>
void
VisitPacketMemories (const cortext::Cortext::Context &ctx, int max_items,
                     Callback callback)
{
  int count = 0;
  std::set<long long> seen;
  auto visit = [&] (const auto &memories) {
    for (const auto &memory : memories)
      {
        if (max_items >= 0 && count >= max_items)
          return false;
        if (memory.id != 0 && !seen.insert (memory.id).second)
          continue;
        callback (memory);
        ++count;
      }
    return true;
  };
  if (visit (ctx.working_memory))
    visit (ctx.retrieved_memory);
}

std::string
MemoryPacketShape (const cortext::Cortext::Context &ctx, int max_items,
                   long long *tokens, int *items, int *text_like_items)
{
  std::ostringstream out;
  int count = 0;
  int text_count = 0;
  VisitPacketMemories (ctx, max_items, [&] (const auto &memory) {
      out << "[memory id=" << memory.id << " modality=" << memory.modality
          << " mimetype=" << memory.mimetype
          << " source=" << memory.source_id << "]\n";
      if (memory.mimetype.rfind ("text/", 0) == 0)
        {
          ++text_count;
          for (const auto &blob : memory.content)
            out << std::string (blob.begin (), blob.end ()) << "\n";
        }
      ++count;
    });
  if (tokens)
    *tokens = EstimateTokens (out.str ().size ());
  if (items)
    *items = count;
  if (text_like_items)
    *text_like_items = text_count;
  return out.str ();
}

nlohmann::json
MemorySourcesJson (
    const cortext::Cortext::Context &ctx, int max_items,
    const std::unordered_map<long long, Record> &memory_records)
{
  nlohmann::json out = nlohmann::json::array ();
  int count = 0;
  VisitPacketMemories (ctx, max_items, [&] (const auto &memory) {
      out.push_back ({
        { "memory_id", memory.id },
        { "source_id", memory.source_id },
        { "modality", memory.modality },
        { "mimetype", memory.mimetype },
        { "timestamp", memory.timestamp },
      });
      auto it = memory_records.find (memory.id);
      if (it != memory_records.end ())
        {
          out.back ()["local_index"] = it->second.local_index;
          out.back ()["original_index"] = it->second.original_index;
          out.back ()["source_timestamp"] = it->second.timestamp;
          out.back ()["speaker_role"] = it->second.speaker_role;
        }
      ++count;
    });
  return out;
}

int
SourceBackedMemoryCount (const cortext::Cortext::Context &ctx)
{
  int count = 0;
  VisitPacketMemories (ctx, -1, [&] (const auto &memory) {
      if (!memory.content.empty ())
        ++count;
    });
  return count;
}

double
Percentile (std::vector<long long> values, double q)
{
  if (values.empty ())
    return 0.0;
  std::sort (values.begin (), values.end ());
  const double pos = q * static_cast<double> (values.size () - 1);
  const auto lo = static_cast<std::size_t> (std::floor (pos));
  const auto hi = static_cast<std::size_t> (std::ceil (pos));
  if (lo == hi)
    return static_cast<double> (values[lo]);
  const double frac = pos - static_cast<double> (lo);
  return static_cast<double> (values[lo]) * (1.0 - frac)
         + static_cast<double> (values[hi]) * frac;
}

nlohmann::json
AggregateJson (const Aggregate &agg)
{
  nlohmann::json out;
  out["probes"] = agg.probes;
  out["cumulative_prompt_tokens"] = agg.prompt_tokens;
  out["cumulative_context_tokens"] = agg.context_tokens;
  out["mean_prompt_tokens"] = agg.probes > 0
                                  ? static_cast<double> (agg.prompt_tokens)
                                        / agg.probes
                                  : 0.0;
  out["mean_context_tokens"] = agg.probes > 0
                                   ? static_cast<double> (agg.context_tokens)
                                         / agg.probes
                                   : 0.0;
  out["mean_context_items"] = agg.probes > 0
                                  ? static_cast<double> (agg.context_items)
                                        / agg.probes
                                  : 0.0;
  out["mean_latency_ms"] = agg.probes > 0 ? agg.latency_ms / agg.probes : 0.0;
  out["p50_prompt_tokens"] = Percentile (agg.prompt_token_samples, 0.50);
  out["p90_prompt_tokens"] = Percentile (agg.prompt_token_samples, 0.90);
  out["p99_prompt_tokens"] = Percentile (agg.prompt_token_samples, 0.99);
  out["p50_context_tokens"] = Percentile (agg.context_token_samples, 0.50);
  out["p90_context_tokens"] = Percentile (agg.context_token_samples, 0.90);
  out["p99_context_tokens"] = Percentile (agg.context_token_samples, 0.99);
  return out;
}

void
AddAggregate (Aggregate &agg, long long prompt_tokens,
              long long context_tokens, long long context_items,
              double latency_ms)
{
  ++agg.probes;
  agg.prompt_tokens += prompt_tokens;
  agg.context_tokens += context_tokens;
  agg.context_items += context_items;
  agg.latency_ms += latency_ms;
  agg.prompt_token_samples.push_back (prompt_tokens);
  agg.context_token_samples.push_back (context_tokens);
}

Config
ParseArgs (int argc, char **argv)
{
  Config cfg;
  for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
      auto require_value = [&] {
        if (i + 1 >= argc)
          throw std::runtime_error ("missing value for " + arg);
        return std::string (argv[++i]);
      };
      if (arg == "--manifest")
        cfg.manifest_path = require_value ();
      else if (arg == "--db")
        cfg.db_path = require_value ();
      else if (arg == "--out")
        cfg.output_path = require_value ();
      else if (arg == "--models")
        cfg.models_dir = require_value ();
      else if (arg == "--modality")
        cfg.modality = require_value ();
      else if (arg == "--label-bank")
        cfg.label_bank_path = require_value ();
      else if (arg == "--warmup-messages")
        cfg.warmup_messages = std::stoi (require_value ());
      else if (arg == "--probe-stride")
        cfg.probe_stride = std::stoi (require_value ());
      else if (arg == "--rag-top-k")
        cfg.rag_top_k = std::stoi (require_value ());
      else if (arg == "--active-history-token-budget")
        cfg.active_history_token_budget = std::stoi (require_value ());
      else if (arg == "--shallow")
        cfg.deep_consolidation = false;
      else if (arg == "--no-label-bank")
        cfg.use_label_bank = false;
      else
        throw std::runtime_error ("unknown argument: " + arg);
    }
  if (cfg.manifest_path.empty ())
    throw std::runtime_error ("--manifest is required");
  if (cfg.modality != "audio" && cfg.modality != "text")
    throw std::runtime_error ("--modality must be audio or text");
  return cfg;
}

} // namespace

int
main (int argc, char **argv)
{
  try
    {
      Config cfg = ParseArgs (argc, argv);
      auto records = LoadManifestRecords (cfg.manifest_path);
      fs::path input_dir = fs::path (std::getenv ("HOME"))
                           / "Documents/Memory/Julie";
      std::ifstream manifest_in (cfg.manifest_path);
      nlohmann::json manifest = nlohmann::json::parse (manifest_in);
      if (manifest.contains ("input_dir"))
        input_dir = manifest["input_dir"].get<std::string> ();
      auto transcript_docs = ParseTranscriptDocs (
          input_dir / "Messages - Julie Willen.txt");
      for (auto &doc : transcript_docs)
        doc.tokens = Tokens (doc.text);

      if (!cfg.db_path.parent_path ().empty ())
        fs::create_directories (cfg.db_path.parent_path ());
      if (!cfg.output_path.parent_path ().empty ())
        fs::create_directories (cfg.output_path.parent_path ());
      fs::remove (cfg.db_path);
      fs::remove (cfg.db_path.string () + "-wal");
      fs::remove (cfg.db_path.string () + "-shm");

      cortext::Cortext::Config cortext_cfg;
      cortext_cfg.focus = cfg.focus;
      cortext_cfg.sensitivity = cfg.sensitivity;
      cortext_cfg.stability = cfg.stability;
      if (cfg.use_label_bank)
        cortext_cfg.label_bank_path = cfg.label_bank_path.string ();

      auto engine = cortext::Cortext::Create (cortext_cfg, cfg.db_path.string (),
                                              cfg.models_dir);
      std::vector<TextDoc> prior_docs;
      nlohmann::json probes = nlohmann::json::array ();
      Aggregate cortext_agg;
      Aggregate text_rag_agg;
      Aggregate full_history_agg;
      std::unordered_map<long long, Record> memory_records;
      int processed = 0;
      int load_failures = 0;
      int consolidation_runs = 0;
      int daily_consolidation_runs = 0;
      int current_day_bucket = -1;
      double audio_load_ms = 0.0;
      double audio_ingest_ms = 0.0;
      double audio_probe_ms = 0.0;
      double probe_context_ms = 0.0;
      const auto run_started = std::chrono::steady_clock::now ();

      for (int i = 0; i < static_cast<int> (records.size ()); ++i)
        {
          const auto &record = records[static_cast<size_t> (i)];
          const int day_bucket = LocalDayBucket (record.timestamp);
          if (processed > 0 && current_day_bucket >= 0
              && day_bucket != current_day_bucket)
            {
              engine->Consolidate (
                  cfg.deep_consolidation ? cortext::ConsolidationMode::Both
                                         : cortext::ConsolidationMode::Shallow);
              ++consolidation_runs;
              ++daily_consolidation_runs;
            }
          current_day_bucket = day_bucket;

          std::optional<benchmark::AudioData> audio;
          if (cfg.modality == "audio")
            {
              const auto load_started = std::chrono::steady_clock::now ();
              audio = benchmark::LoadPcmFloat32 (record.audio_path, 16000, 1);
              const auto load_ended = std::chrono::steady_clock::now ();
              audio_load_ms += std::chrono::duration<double, std::milli> (
                                   load_ended - load_started)
                                   .count ();
              if (!audio || audio->samples.empty ())
                {
                  ++load_failures;
                  continue;
                }
            }

          const auto &query_doc = transcript_docs.at (
              static_cast<size_t> (record.original_index));
          const bool should_probe = i >= cfg.warmup_messages
                                    && cfg.probe_stride > 0
                                    && i % cfg.probe_stride == 0;
          const auto ingest_started = std::chrono::steady_clock::now ();
          cortext::Cortext::Context ingest_ctx
              = cfg.modality == "audio"
                    ? engine->ProcessAudio (audio->samples.data (),
                                            audio->samples.size (),
                                            StreamSourceId (record))
                    : engine->ProcessTextAt (query_doc.text,
                                             StreamSourceId (record),
                                             record.timestamp);
          const auto ingest_ended = std::chrono::steady_clock::now ();
          const double ingest_ms
              = std::chrono::duration<double, std::milli> (ingest_ended
                                                           - ingest_started)
                    .count ();
          audio_ingest_ms += ingest_ms;
          if (ingest_ctx.output.stored_memory_id)
            {
              memory_records[*ingest_ctx.output.stored_memory_id] = record;
            }

          if (should_probe)
            {
              cortext::Cortext::Context &ctx = ingest_ctx;
              const double latency_ms = ingest_ms;
              probe_context_ms += latency_ms;

              long long cortext_tokens = 0;
              int cortext_items = 0;
              int cortext_text_items = 0;
              const std::string cortext_packet = MemoryPacketShape (
                  ctx, -1, &cortext_tokens,
                  &cortext_items, &cortext_text_items);

              const auto rag_docs = RagTopK (prior_docs, query_doc.tokens,
                                             record.timestamp, cfg.rag_top_k);
              const std::string rag_context = BuildRagContext (rag_docs);
              int active_history_items = 0;
              const std::string active_history = BuildActiveHistory (
                  prior_docs, cfg.active_history_token_budget,
                  &active_history_items);
              std::ostringstream full_history;
              for (const auto &doc : prior_docs)
                full_history << doc.source_id << ": " << doc.text << "\n";

              const long long user_tokens = EstimateTokens (
                  static_cast<std::size_t> (std::max (0, record.text_chars)));
              const long long normal_rag_tokens
                  = EstimateTokens (active_history.size () + rag_context.size ());
              const long long full_history_tokens
                  = EstimateTokens (full_history.str ().size ());

              AddAggregate (cortext_agg, cortext_tokens + user_tokens,
                            cortext_tokens, cortext_items, latency_ms);
              AddAggregate (text_rag_agg,
                            normal_rag_tokens + user_tokens,
                            normal_rag_tokens,
                            active_history_items
                                + static_cast<int> (rag_docs.size ()),
                            0.0);
              AddAggregate (full_history_agg,
                            full_history_tokens + user_tokens,
                            full_history_tokens,
                            static_cast<long long> (prior_docs.size ()),
                            0.0);

              nlohmann::json probe;
              probe["local_index"] = record.local_index;
              probe["original_index"] = record.original_index;
              probe["timestamp"] = record.timestamp;
              probe["speaker_role"] = record.speaker_role;
              probe["query_text_chars"] = record.text_chars;
              probe["audio_samples"] = audio ? audio->samples.size () : 0;
              probe["cortext_native_context_tokens"] = cortext_tokens;
              probe["cortext_native_retrieved_items"] = cortext_items;
              probe["cortext_native_working_memory_items"]
                  = ctx.working_memory.size ();
              probe["cortext_native_source_backed_items"]
                  = SourceBackedMemoryCount (ctx);
              probe["cortext_native_text_like_items"] = cortext_text_items;
              probe["cortext_native_sources"]
                  = MemorySourcesJson (ctx, -1, memory_records);
              // Backward-compatible aliases for the current judge script.
              probe["cortext_raw_audio_only_context_tokens"] = cortext_tokens;
              probe["cortext_raw_audio_only_items"] = cortext_items;
              probe["cortext_raw_audio_only_text_like_items"]
                  = cortext_text_items;
              probe["cortext_raw_audio_only_sources"]
                  = probe["cortext_native_sources"];
              probe["text_rag_oracle_context_tokens"] = normal_rag_tokens;
              probe["text_rag_oracle_items"] = active_history_items
                                               + static_cast<int> (rag_docs.size ());
              probe["text_full_history_oracle_context_tokens"]
                  = full_history_tokens;
              probe["text_full_history_oracle_items"] = prior_docs.size ();
              probe["cortext_probe_latency_ms"] = latency_ms;
              probe["cortext_probe_policy"]
                  = "single_durable_chat_turn_reused_for_probe_and_ingest";
              probe["quality_status"] = "not_judged_in_cpp_artifact";
              probes.push_back (std::move (probe));

              (void)cortext_packet;
            }

          ++processed;
          prior_docs.push_back (query_doc);
        }

      engine->Consolidate (cfg.deep_consolidation
                               ? cortext::ConsolidationMode::Both
                               : cortext::ConsolidationMode::Shallow);
      ++consolidation_runs;
      if (processed > 0)
        ++daily_consolidation_runs;
      engine->Flush ();
      const auto run_ended = std::chrono::steady_clock::now ();

      nlohmann::json out;
      out["schema"] = "julie_native_memory_eval_v1";
      out["manifest_path"] = cfg.manifest_path.string ();
      out["db_path"] = cfg.db_path.string ();
      out["modality"] = cfg.modality;
      out["processed_messages"] = processed;
      out["processed_audio_messages"] = cfg.modality == "audio" ? processed : 0;
      out["processed_text_messages"] = cfg.modality == "text" ? processed : 0;
      out["audio_load_failures"] = load_failures;
      out["probe_count"] = probes.size ();
      out["warmup_messages"] = cfg.warmup_messages;
      out["probe_stride"] = cfg.probe_stride;
      out["knobs"]["focus"] = cfg.focus;
      out["knobs"]["sensitivity"] = cfg.sensitivity;
      out["knobs"]["stability"] = cfg.stability;
      out["cortext_runtime_behavior"] =
          "native API output is recorded without benchmark-side memory caps";
      out["daily_consolidation_enabled"] = true;
      out["consolidation_policy"]
          = "manifest_local_day_boundary_plus_final_day";
      out["consolidation_runs"] = consolidation_runs;
      out["daily_consolidation_runs"] = daily_consolidation_runs;
      out["deep_consolidation"] = cfg.deep_consolidation;
      out["audio_signal_filter_enabled"]
          = cortext::Cortext::Config{}.signal_filter_audio_enabled;
      out["cortext_input_modality"]
          = cfg.modality == "audio" ? "raw_audio_only" : "text_only";
      out["stt_enabled"] = false;
      out["transcript_text_passed_to_cortext"] = cfg.modality == "text";
      out["transcript_text_used_for"] =
          cfg.modality == "audio"
              ? "tts_generation_before_ingest_and_text_rag_oracle_metrics_only"
              : "text_cortext_ingest_and_text_rag_oracle_metrics";
      out["source_id_policy"] =
          "Cortext receives the same Gabe or Julie opaque stream "
          "source_id for text and audio; eval-only original_index metadata is "
          "keyed by stored memory_id and never passed as source_id";
      out["timestamp_handling"]
          = cfg.modality == "audio"
                ? "manifest timestamps drive replay-day consolidation and are "
                  "stored only as eval metadata; production ProcessAudio stores "
                  "normal ingestion time"
                : "manifest timestamps are passed through ProcessTextAt for "
                  "event-time replay";
      out["privacy_note"]
          = "No raw Julie transcript text is written to this artifact.";
      out["audio_pipeline_cost_ms"]["manifest_tts_sum"] = 0.0;
      out["audio_pipeline_cost_ms"]["manifest_ffmpeg_sum"] = 0.0;
      for (const auto &record : records)
        {
          out["audio_pipeline_cost_ms"]["manifest_tts_sum"]
              = out["audio_pipeline_cost_ms"]["manifest_tts_sum"].get<double> ()
                + record.tts_ms;
          out["audio_pipeline_cost_ms"]["manifest_ffmpeg_sum"]
              = out["audio_pipeline_cost_ms"]["manifest_ffmpeg_sum"].get<double> ()
                + record.ffmpeg_ms;
        }
      out["audio_pipeline_cost_ms"]["load_sum"] = audio_load_ms;
      out["audio_pipeline_cost_ms"]["durable_ingest_sum"] = audio_ingest_ms;
      out["audio_pipeline_cost_ms"]["probe_context_sum"] = probe_context_ms;
      out["audio_pipeline_cost_ms"]["ephemeral_probe_sum"] = audio_probe_ms;
      out["wall_ms"]
          = std::chrono::duration_cast<std::chrono::milliseconds> (run_ended
                                                                   - run_started)
                .count ();
      out["cortext_native_wm_stm_ltm"] = AggregateJson (cortext_agg);
      out["text_rag_with_history_oracle"] = AggregateJson (text_rag_agg);
      out["text_full_history_oracle"] = AggregateJson (full_history_agg);
      if (text_rag_agg.prompt_tokens > 0)
        out["token_ratio_cortext_vs_text_rag_oracle"]
            = static_cast<double> (cortext_agg.prompt_tokens)
              / static_cast<double> (text_rag_agg.prompt_tokens);
      if (full_history_agg.prompt_tokens > 0)
        out["token_ratio_cortext_vs_text_full_history_oracle"]
            = static_cast<double> (cortext_agg.prompt_tokens)
              / static_cast<double> (full_history_agg.prompt_tokens);
      out["probes"] = std::move (probes);

      std::ofstream summary (cfg.output_path);
      summary << out.dump (2) << "\n";
      std::cout << out.dump (2) << "\n";
      return 0;
    }
  catch (const std::exception &e)
    {
      std::cerr << "julie_raw_speech_memory_eval failed: " << e.what ()
                << "\n";
      return 1;
    }
}
