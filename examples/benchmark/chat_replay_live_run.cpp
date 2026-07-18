#include <cortext/clock.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/cortext.hpp>
#include <cortext/internal/replay_ingress.hpp>
#include <cortext/store/object_store.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>

#include "../../src/encoder/text_encoder_factory.hpp"
#include "../../src/operations/bounded_activation_shadow_internal.hpp"
#include "../../src/operations/retrieval_trace_state.hpp"

#include <nlohmann/json.hpp>

#include <sqlite3.h>

#include <algorithm>
#include <any>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif

#include "transcript_discovery.hpp"

namespace fs = std::filesystem;

namespace
{

constexpr const char *kUserSourceId = "User";
constexpr const char *kContactSourceId = "Contact";
constexpr long long kNormalRagVectorSearchMultiplier = 8;

const char *
ConsolidationStateName (cortext::ConsolidationState state)
{
  switch (state)
    {
    case cortext::ConsolidationState::Recommended:
      return "recommended";
    case cortext::ConsolidationState::Required:
      return "required";
    case cortext::ConsolidationState::None:
      return "none";
    }
  return "none";
}

struct Config
{
  fs::path input_dir;
  fs::path transcript;
  fs::path packet_corpus;
  fs::path message_corpus;
  fs::path shadow_embedding_corpus;
  fs::path db_path = "build/chat_replay_live_run.sqlite";
  fs::path output_path = "build/chat_replay_live_run_summary.json";
  int skip_messages = 0;
  int max_messages = 1000;
  int max_packets = -1;
  int shadow_queries = 512;
  int shadow_result_k = 16;
  int skip_packets = 0;
  int media_limit = 0;
  int consolidate_every = 0;
  int warmup_events = 0;
  int probe_stride = 0;
  int rag_top_k = 5;
  int active_history_token_budget = 8000;
  double focus = 0.5;
  double sensitivity = 0.5;
  double stability = 0.5;
  bool daily_consolidation = false;
  bool skip_final_consolidation = false;
  int daily_consolidation_hour = 2;
  std::string replay_timezone;
  bool append = false;
  bool resume_from_existing = false;
  bool profile_probes_only = false;
  bool checkpoint_eval_only = false;
  int progress_stride = 1000;
  std::uint64_t fixed_timestamp_base_ms = 0;
  std::uint64_t fixed_timestamp_step_ms = 0;
  std::uint64_t checkpoint_after_timestamp = 0;
  int checkpoint_query_count = 8;
  int checkpoint_query_stride = 1;
  int checkpoint_query_days = 0;
  int checkpoint_queries_per_day = 0;
  bool full_operation_ms = false;
  bool behavior_oracle = false;
  bool honor_required_consolidation = false;
  bool no_profile_rows = false;
  bool reuse_db = false;
  cortext::Retention retention = cortext::Retention::Durable;
  std::string sqlite_profile = "realtime";
};

struct ResumeCheckpoint
{
  bool enabled = false;
  int event_count_offset = 0;
  int text_count_offset = 0;
  int media_count_offset = 0;
  int probe_count_offset = 0;
  int rebuilt_event_count = 0;
  int rebuilt_text_count = 0;
  int prior_timeline_events = 0;
  int db_signal_count = 0;
  int db_text_signal_count = 0;
  std::uint64_t after_timestamp = 0;
};

struct Message
{
  int index = 0;
  std::uint64_t timestamp = 0;
  bool from_contact = false;
  std::string text;
};

struct MediaItem
{
  fs::path path;
  std::uint64_t timestamp = 0;
  std::string kind;
};

struct TimelineEvent
{
  std::uint64_t timestamp = 0;
  int order = 0;
  bool is_media = false;
  std::size_t index = 0;
};

struct EventDoc
{
  int index = 0;
  std::uint64_t timestamp = 0;
  std::string source_id;
  std::string modality;
  std::string text;
};

struct CompactedHistory
{
  std::vector<EventDoc> raw_docs;
  std::string summary_text;
  int compaction_events = 0;
  int compacted_items = 0;
  int compacted_summary_items = 0;
  int raw_items = 0;
  int raw_tokens = 0;
  int compacted_original_tokens = 0;
  int compacted_summary_tokens = 0;
};

struct VectorRagPacket
{
  std::vector<EventDoc> docs;
  long long query_rows = 0;
  long long query_embedding_bytes = 0;
  long long vector_search_k = 0;
  long long candidate_rows = 0;
  long long prior_chat_rows = 0;
  double best_distance = 0.0;
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

std::string
Lower (std::string value)
{
  std::transform (value.begin (), value.end (), value.begin (),
                  [] (unsigned char c) {
                    return static_cast<char> (std::tolower (c));
                  });
  return value;
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
         && std::isdigit (line[12]) && line[13] == ':'
         && std::isdigit (line[14]) && std::isdigit (line[15])
         && line[16] == ':' && std::isdigit (line[17])
         && std::isdigit (line[18]);
}

std::optional<std::uint64_t>
ParseTimestamp (const std::string &text)
{
  if (text.size () < 19)
    return std::nullopt;
  std::string stamp = text.substr (0, 19);
  if (stamp.size () >= 19 && stamp[13] == ' ' && stamp[16] == ' ')
    {
      stamp[13] = ':';
      stamp[16] = ':';
    }
  std::tm tm{};
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

void
ApplyReplayTimezone (const std::string &timezone)
{
  if (timezone.empty ())
    return;
#if defined(_WIN32)
  if (_putenv_s ("TZ", timezone.c_str ()) != 0)
    throw std::runtime_error ("failed to set replay timezone: " + timezone);
  _tzset ();
#else
  if (setenv ("TZ", timezone.c_str (), 1) != 0)
    throw std::runtime_error ("failed to set replay timezone: " + timezone);
  tzset ();
#endif
}

bool
HasEnvValue (const char *name)
{
  const char *value = std::getenv (name);
  return value && value[0] != '\0';
}

void
SetEnvValue (const char *name, const char *value, bool overwrite)
{
#if defined(_WIN32)
  if (!overwrite && HasEnvValue (name))
    return;
  if (_putenv_s (name, value) != 0)
    throw std::runtime_error (std::string ("failed to set environment: ")
                              + name);
#else
  if (setenv (name, value, overwrite ? 1 : 0) != 0)
    throw std::runtime_error (std::string ("failed to set environment: ")
                              + name);
#endif
}

void
ApplyEnvDefault (const char *name, const char *value,
                 nlohmann::json &applied)
{
  if (HasEnvValue (name))
    return;
  SetEnvValue (name, value, false);
  applied.push_back (name);
}

nlohmann::json
SQLiteProfileJson (const std::string &profile,
                   const nlohmann::json &applied)
{
  nlohmann::json env = nlohmann::json::object ();
  for (const char *name :
       { "CORTEXT_FOREGROUND_WAL_CHECKPOINT",
         "CORTEXT_SQLITE_JOURNAL_MODE",
         "CORTEXT_SQLITE_SYNCHRONOUS",
         "CORTEXT_SQLITE_CACHE_SIZE_KB",
         "CORTEXT_SQLITE_TEMP_STORE",
         "CORTEXT_SQLITE_MMAP_SIZE" })
    {
      const char *value = std::getenv (name);
      if (value && value[0] != '\0')
        env[name] = value;
    }
  return {
    { "profile", profile },
    { "applied_defaults", applied },
    { "effective_env", env },
    { "policy",
      "benchmark-only SQLite profile; core library defaults remain unchanged" },
  };
}

nlohmann::json
ApplySQLiteProfile (const std::string &profile)
{
  nlohmann::json applied = nlohmann::json::array ();
  if (profile == "inherit")
    return SQLiteProfileJson (profile, applied);
  if (profile == "durable")
    {
      SetEnvValue ("CORTEXT_FOREGROUND_WAL_CHECKPOINT", "0", true);
      SetEnvValue ("CORTEXT_SQLITE_JOURNAL_MODE", "wal", true);
      SetEnvValue ("CORTEXT_SQLITE_SYNCHRONOUS", "normal", true);
      applied = {
        "CORTEXT_FOREGROUND_WAL_CHECKPOINT",
        "CORTEXT_SQLITE_JOURNAL_MODE",
        "CORTEXT_SQLITE_SYNCHRONOUS",
      };
      return SQLiteProfileJson (profile, applied);
    }
  if (profile != "realtime")
    throw std::runtime_error (
        "--sqlite-profile must be realtime, durable, or inherit");

  ApplyEnvDefault ("CORTEXT_FOREGROUND_WAL_CHECKPOINT", "0", applied);
  ApplyEnvDefault ("CORTEXT_SQLITE_JOURNAL_MODE", "memory", applied);
  ApplyEnvDefault ("CORTEXT_SQLITE_SYNCHRONOUS", "off", applied);
  return SQLiteProfileJson (profile, applied);
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

int
LocalSleepBucket (std::uint64_t timestamp_ms, int sleep_hour)
{
  const std::time_t seconds = static_cast<std::time_t> (timestamp_ms / 1000ULL);
  std::tm local{};
#if defined(_WIN32)
  localtime_s (&local, &seconds);
#else
  localtime_r (&seconds, &local);
#endif
  local.tm_hour -= sleep_hour;
  local.tm_isdst = -1;
  if (std::mktime (&local) < 0)
    return LocalDayBucket (timestamp_ms);
  return (local.tm_year + 1900) * 1000 + local.tm_yday;
}

std::uint64_t
LocalSleepCheckpointTimestamp (std::uint64_t timestamp_ms, int sleep_hour)
{
  const std::time_t seconds = static_cast<std::time_t> (timestamp_ms / 1000ULL);
  std::tm local{};
#if defined(_WIN32)
  localtime_s (&local, &seconds);
#else
  localtime_r (&seconds, &local);
#endif
  if (local.tm_hour < sleep_hour)
    local.tm_mday -= 1;
  local.tm_hour = sleep_hour;
  local.tm_min = 0;
  local.tm_sec = 0;
  local.tm_isdst = -1;
  const std::time_t checkpoint_seconds = std::mktime (&local);
  if (checkpoint_seconds < 0)
    return timestamp_ms;
  return static_cast<std::uint64_t> (checkpoint_seconds) * 1000ULL;
}

std::vector<Message>
ParseMessages (const fs::path &path)
{
  std::ifstream in (path);
  if (!in)
    throw std::runtime_error ("failed to open transcript: " + path.string ());

  std::vector<Message> messages;
  std::string line;
  std::string pending_header;
  std::string pending_text;
  bool in_body = false;

  auto flush = [&] {
    if (pending_header.empty ())
      return;
    auto ts = ParseTimestamp (pending_header);
    std::string text = Trim (pending_text);
    if (ts && !text.empty ())
      {
        Message msg;
        msg.index = static_cast<int> (messages.size ());
        msg.timestamp = *ts;
        msg.from_contact = pending_header.find (" from ") != std::string::npos;
        msg.text = std::move (text);
        messages.push_back (std::move (msg));
      }
    pending_header.clear ();
    pending_text.clear ();
    in_body = false;
  };

  while (std::getline (in, line))
    {
      if (line.rfind ("----------------------------------------------------", 0)
          == 0)
        {
          flush ();
          continue;
        }
      if (StartsWithDate (line))
        {
          flush ();
          pending_header = line;
          in_body = true;
          continue;
        }
      if (in_body)
        {
          if (!pending_text.empty ())
            pending_text.push_back ('\n');
          pending_text += line;
        }
    }
  flush ();
  return messages;
}

std::string
ShellQuote (const fs::path &path)
{
  std::string s = path.string ();
  std::string out = "'";
  for (char c : s)
    out += c == '\'' ? "'\\''" : std::string (1, c);
  out += "'";
  return out;
}

bool
RunCommand (const std::string &cmd)
{
  return std::system (cmd.c_str ()) == 0;
}

bool
LoadFile (const fs::path &path, std::vector<unsigned char> &bytes)
{
  std::ifstream in (path, std::ios::binary);
  if (!in)
    return false;
  bytes.assign (std::istreambuf_iterator<char> (in),
                std::istreambuf_iterator<char> ());
  return true;
}

std::vector<float>
BytesToFloats (const std::vector<unsigned char> &bytes)
{
  std::vector<float> out (bytes.size () / sizeof (float));
  if (!out.empty ())
    std::memcpy (out.data (), bytes.data (), out.size () * sizeof (float));
  return out;
}

std::vector<std::string>
Tokens (const std::string &text)
{
  std::vector<std::string> tokens;
  std::string current;
  for (unsigned char c : text)
    {
      if (std::isalnum (c))
        current.push_back (static_cast<char> (std::tolower (c)));
      else if (!current.empty ())
        {
          tokens.push_back (current);
          current.clear ();
        }
    }
  if (!current.empty ())
    tokens.push_back (current);
  return tokens;
}

int
EstimateTokens (const std::string &text)
{
  return std::max (1, static_cast<int> ((text.size () + 3) / 4));
}

double
Overlap (const std::string &lhs, const std::string &rhs)
{
  const auto lhs_tokens = Tokens (lhs);
  const auto rhs_tokens = Tokens (rhs);
  if (lhs_tokens.empty () || rhs_tokens.empty ())
    return 0.0;
  std::unordered_set<std::string> left (lhs_tokens.begin (), lhs_tokens.end ());
  int overlap = 0;
  for (const auto &token : rhs_tokens)
    {
      if (left.count (token) != 0)
        ++overlap;
    }
  return static_cast<double> (overlap)
         / static_cast<double> (std::max<std::size_t> (1, rhs_tokens.size ()));
}

nlohmann::json
ContextMemoryIdsJson (
    const std::vector<cortext::Cortext::Context::Memory> &memories)
{
  nlohmann::json out = nlohmann::json::array ();
  for (const auto &memory : memories)
    {
      if (memory.id > 0)
        out.push_back (memory.id);
    }
  return out;
}

std::string
PrivateValueDigest (const std::vector<unsigned char> &bytes)
{
  const auto digest = cortext::ComputeObjectId (bytes);
  std::ostringstream out;
  out << std::hex << std::setfill ('0');
  for (const unsigned char byte : digest)
    {
      out << std::setw (2) << static_cast<unsigned int> (byte);
    }
  return out.str ();
}

std::string
PrivateValueDigest (const std::string &value)
{
  return PrivateValueDigest (
      std::vector<unsigned char> (value.begin (), value.end ()));
}

nlohmann::json
PublicMemoryBehaviorJson (
    const std::vector<cortext::Cortext::Context::Memory> &memories)
{
  nlohmann::json out = nlohmann::json::array ();
  for (const auto &memory : memories)
    {
      nlohmann::json content = nlohmann::json::array ();
      for (const auto &blob : memory.content)
        {
          content.push_back (
              { { "bytes", blob.size () },
                { "blake3", PrivateValueDigest (blob) } });
        }
      nlohmann::json soft_anchors = nlohmann::json::array ();
      for (const auto &anchor : memory.soft_anchors)
        {
          soft_anchors.push_back (
              { { "id_blake3", PrivateValueDigest (anchor.id) },
                { "strength", anchor.strength },
                { "likelihood", anchor.likelihood },
                { "label_blake3", PrivateValueDigest (anchor.label) },
                { "tier_blake3", PrivateValueDigest (anchor.tier) },
                { "score", anchor.score },
                { "margin", anchor.margin },
                { "entropy", anchor.entropy } });
        }
      out.push_back (
          { { "source_id_blake3", PrivateValueDigest (memory.source_id) },
            { "id", memory.id },
            { "timestamp", memory.timestamp },
            { "retrieved_count", memory.retrieved_count },
            { "used_count", memory.used_count },
            { "content", std::move (content) },
            { "modality", memory.modality },
            { "mimetype", memory.mimetype },
            { "relevance", memory.relevance },
            { "mismatch", memory.mismatch },
            { "surprise", memory.surprise },
            { "rarity", memory.rarity },
            { "drift", memory.drift },
            { "contradiction", memory.contradiction },
            { "utility", memory.utility },
            { "periphery", memory.periphery },
            { "coverage", memory.coverage },
            { "salience", memory.salience },
            { "valence", memory.valence },
            { "arousal", memory.arousal },
            { "composite_score", memory.composite_score },
            { "threshold_t", memory.threshold_t },
            { "soft_anchors", std::move (soft_anchors) } });
    }
  return out;
}

nlohmann::json
PublicBehaviorJson (const cortext::Cortext::Context &ctx,
                    bool bounded_working_memory = false)
{
  nlohmann::json metrics = nlohmann::json::object ();
  for (const auto &[metric, value] : ctx.output.metrics)
    {
      metrics[std::to_string (metric)] = value;
    }
  std::vector<unsigned char> embedding_bytes (
      ctx.embedding.size () * sizeof (float));
  if (!embedding_bytes.empty ())
    {
      std::memcpy (embedding_bytes.data (), ctx.embedding.data (),
                   embedding_bytes.size ());
    }
  nlohmann::json working_memory;
  if (bounded_working_memory)
    {
      working_memory = {
        { "count", ctx.working_memory.size () },
        { "first_id",
          ctx.working_memory.empty () ? 0 : ctx.working_memory.front ().id },
        { "last_id",
          ctx.working_memory.empty () ? 0 : ctx.working_memory.back ().id },
      };
    }
  else
    {
      working_memory = PublicMemoryBehaviorJson (ctx.working_memory);
    }
  return {
    { "working_memory", std::move (working_memory) },
    { "retrieved_memory", PublicMemoryBehaviorJson (ctx.retrieved_memory) },
    { "embedding_size", ctx.embedding.size () },
    { "embedding_blake3", PrivateValueDigest (embedding_bytes) },
    { "should_interrupt", ctx.should_interrupt },
    { "interrupt_aborted", ctx.interrupt_aborted },
    { "at_boundary", ctx.at_boundary },
    { "consolidation_state",
      ConsolidationStateName (ctx.consolidation_state) },
    { "interrupt_gate_has_candidates", ctx.interrupt_gate_has_candidates },
    { "interrupt_gate_blocked_no_store",
      ctx.interrupt_gate_blocked_no_store },
    { "interrupt_gate_rel_pass", ctx.interrupt_gate_rel_pass },
    { "interrupt_gate_novelty_pass", ctx.interrupt_gate_novelty_pass },
    { "interrupt_gate_mu_pass", ctx.interrupt_gate_mu_pass },
    { "interrupt_gate_novelty_mu_pass",
      ctx.interrupt_gate_novelty_mu_pass },
    { "interrupt_gate_dup_pass", ctx.interrupt_gate_dup_pass },
    { "interrupt_gate_boundary_mu_pass",
      ctx.interrupt_gate_boundary_mu_pass },
    { "interrupt_gate_rel_star", ctx.interrupt_gate_rel_star },
    { "interrupt_gate_retrieval_thresh",
      ctx.interrupt_gate_retrieval_thresh },
    { "interrupt_gate_boundary_mult_eff",
      ctx.interrupt_gate_boundary_mult_eff },
    { "interrupt_gate_affect_drive", ctx.interrupt_gate_affect_drive },
    { "boundary_score",
      ctx.boundary_score.has_value () ? nlohmann::json (*ctx.boundary_score)
                                       : nlohmann::json (nullptr) },
    { "boundary_type",
      ctx.boundary_type.has_value () ? nlohmann::json (*ctx.boundary_type)
                                      : nlohmann::json (nullptr) },
    { "output",
      {
        { "composite_score",
          ctx.output.composite_score.has_value ()
              ? nlohmann::json (*ctx.output.composite_score)
              : nlohmann::json (nullptr) },
        { "threshold",
          ctx.output.threshold.has_value ()
              ? nlohmann::json (*ctx.output.threshold)
              : nlohmann::json (nullptr) },
        { "decision",
          ctx.output.decision.has_value ()
              ? nlohmann::json (*ctx.output.decision)
              : nlohmann::json (nullptr) },
        { "effective_focus", ctx.output.effective_focus },
        { "coherence", ctx.output.coherence },
        { "emotion_intensity", ctx.output.emotion_intensity },
        { "valence", ctx.output.valence },
        { "arousal", ctx.output.arousal },
        { "metrics", std::move (metrics) },
        { "stored_embedding_id",
          ctx.output.stored_embedding_id.has_value ()
              ? nlohmann::json (*ctx.output.stored_embedding_id)
              : nlohmann::json (nullptr) },
        { "stored_memory_id",
          ctx.output.stored_memory_id.has_value ()
              ? nlohmann::json (*ctx.output.stored_memory_id)
              : nlohmann::json (nullptr) },
        { "stored_signal_id",
          ctx.output.stored_signal_id.has_value ()
              ? nlohmann::json (*ctx.output.stored_signal_id)
              : nlohmann::json (nullptr) },
        { "signal_filter_evaluated", ctx.output.signal_filter_evaluated },
        { "signal_filter_accepted", ctx.output.signal_filter_accepted },
        { "signal_filter_modality", ctx.output.signal_filter_modality },
        { "signal_filter_reason", ctx.output.signal_filter_reason },
        { "signal_filter_score", ctx.output.signal_filter_score },
        { "signal_filter_threshold", ctx.output.signal_filter_threshold },
        { "signal_filter_quiet_seconds",
          ctx.output.signal_filter_quiet_seconds },
        { "soft_anchor_enabled", ctx.output.soft_anchor_enabled },
        { "soft_anchor_state_count", ctx.output.soft_anchor_state_count },
        { "soft_anchor_link_count", ctx.output.soft_anchor_link_count },
        { "soft_anchor_create_count", ctx.output.soft_anchor_create_count },
        { "soft_anchor_update_count", ctx.output.soft_anchor_update_count },
        { "soft_anchor_none_count", ctx.output.soft_anchor_none_count },
      } },
  };
}

int
EstimateMemoryTokens (const cortext::Cortext::Context::Memory &memory)
{
  int tokens = EstimateTokens (memory.source_id) + EstimateTokens (memory.modality)
               + EstimateTokens (memory.mimetype);
  if (memory.content.empty ())
    {
      return tokens;
    }
  for (const auto &blob : memory.content)
    {
      if (memory.modality == "text" || memory.mimetype == "text/plain")
        {
          tokens += EstimateTokens (
              std::string (blob.begin (), blob.end ()));
        }
      else
        {
          tokens += EstimateTokens ("[" + memory.modality + " source blob]");
        }
    }
  return tokens;
}

int
EstimateMemoryPacketTokens (
    const std::vector<cortext::Cortext::Context::Memory> &memories)
{
  int tokens = 0;
  for (const auto &memory : memories)
    {
      tokens += EstimateMemoryTokens (memory);
    }
  return tokens;
}

nlohmann::json
MemoryPacketJson (
    const std::vector<cortext::Cortext::Context::Memory> &memories)
{
  nlohmann::json out = nlohmann::json::array ();
  int rank = 0;
  for (const auto &memory : memories)
    {
      std::string text;
      if (memory.modality == "text" || memory.mimetype == "text/plain")
        {
          bool first = true;
          for (const auto &blob : memory.content)
            {
              if (!first)
                text += "\n";
              text.append (blob.begin (), blob.end ());
              first = false;
            }
        }

      out.push_back ({
        { "rank", rank++ },
        { "memory_id", memory.id },
        { "kind", "probe_time_context_memory" },
        { "source_id", memory.source_id },
        { "modality", memory.modality },
        { "mimetype", memory.mimetype },
        { "start_ts", memory.timestamp },
        { "timestamp", memory.timestamp },
        { "tokens", EstimateMemoryTokens (memory) },
        { "content_text", text },
        { "content_blob_count", memory.content.size () },
      });
    }
  return out;
}

nlohmann::json
RetrievalTraceJson ()
{
  const auto summary
      = cortext::operations::retrieval_trace::GetLastRetrievalSummary ();
  nlohmann::json ranked = nlohmann::json::array ();
  int rank = 0;
  for (const auto &candidate :
       cortext::operations::retrieval_trace::GetLastRankedCandidates ())
    {
      ranked.push_back ({
        { "rank", rank++ },
        { "embedding_id", candidate.embedding_id },
        { "memory_id", candidate.memory_id },
        { "score", candidate.score },
        { "relevance", candidate.relevance },
        { "proc_score", candidate.proc_score },
        { "predictive_bonus", candidate.predictive_bonus },
        { "pre_activation", candidate.pre_activation },
        { "label_match_count", candidate.label_match_count },
        { "durable_source_boost", candidate.durable_source_boost },
        { "durable_source_count", candidate.durable_source_count },
        { "activation",
          {
              { "base_level", candidate.activation.base_level },
              { "spreading_activation",
                candidate.activation.spreading_activation },
              { "partial_match_penalty",
                candidate.activation.partial_match_penalty },
              { "recent_inhibition", candidate.activation.recent_inhibition },
              { "utility", candidate.activation.utility },
              { "exploration_noise", candidate.activation.exploration_noise },
              { "activation_total", candidate.activation.activation_total },
          } },
      });
    }
  nlohmann::json rejected = nlohmann::json::array ();
  for (const auto &entry :
       cortext::operations::retrieval_trace::GetLastRejectedCandidates ())
    {
      const auto &candidate = entry.candidate;
      rejected.push_back ({
        { "reason", entry.reason },
        { "stage", entry.stage },
        { "observed", entry.observed },
        { "threshold", entry.threshold },
        { "embedding_id", candidate.embedding_id },
        { "memory_id", candidate.memory_id },
        { "score", candidate.score },
        { "relevance", candidate.relevance },
        { "proc_score", candidate.proc_score },
        { "predictive_bonus", candidate.predictive_bonus },
        { "pre_activation", candidate.pre_activation },
        { "label_match_count", candidate.label_match_count },
        { "durable_source_boost", candidate.durable_source_boost },
        { "durable_source_count", candidate.durable_source_count },
        { "activation",
          {
              { "base_level", candidate.activation.base_level },
              { "spreading_activation",
                candidate.activation.spreading_activation },
              { "partial_match_penalty",
                candidate.activation.partial_match_penalty },
              { "recent_inhibition", candidate.activation.recent_inhibition },
              { "utility", candidate.activation.utility },
              { "exploration_noise", candidate.activation.exploration_noise },
              { "activation_total", candidate.activation.activation_total },
          } },
      });
    }
  nlohmann::json evidence_packets = nlohmann::json::array ();
  for (const auto &packet :
       cortext::operations::retrieval_trace::GetLastEvidencePackets ())
    {
      nlohmann::json members = nlohmann::json::array ();
      for (const auto &member : packet.members)
        {
          members.push_back ({
            { "rank", member.rank },
            { "embedding_id", member.embedding_id },
	            { "memory_id", member.memory_id },
	            { "weight", member.weight },
	            { "score", member.score },
	            { "evidence_confidence", member.evidence_confidence },
	            { "evidence_weight", member.evidence_weight },
	            { "evidence_source_diversity",
	              member.evidence_source_diversity },
	            { "evidence_contradiction_mass",
	              member.evidence_contradiction_mass },
	            { "activation",
              {
                  { "base_level", member.activation.base_level },
                  { "spreading_activation",
                    member.activation.spreading_activation },
                  { "partial_match_penalty",
                    member.activation.partial_match_penalty },
                  { "recent_inhibition",
                    member.activation.recent_inhibition },
                  { "utility", member.activation.utility },
                  { "exploration_noise",
                    member.activation.exploration_noise },
                  { "activation_total",
                    member.activation.activation_total },
              } },
          });
        }
      evidence_packets.push_back ({
        { "packet_id", packet.packet_id },
        { "consumer", packet.consumer },
        { "reason", packet.reason },
        { "tie_margin", packet.tie_margin },
        { "temperature", packet.temperature },
	        { "score_span", packet.score_span },
	        { "activation_total", packet.activation_total },
	        { "evidence_confidence", packet.evidence_confidence },
	        { "evidence_weight", packet.evidence_weight },
	        { "evidence_source_diversity", packet.evidence_source_diversity },
	        { "evidence_contradiction_mass",
	          packet.evidence_contradiction_mass },
	        { "members", std::move (members) },
      });
    }

  return {
    { "text_query_token_count", summary.text_query_token_count },
    { "text_query_wm_slots", summary.text_query_wm_slots },
    { "text_query_wm_chars", summary.text_query_wm_chars },
    { "rejected_candidate_count", summary.rejected_candidate_count },
    { "rejected_filter_count", summary.rejected_filter_count },
    { "rejected_selection_count", summary.rejected_selection_count },
	    { "evidence_packet_count", summary.evidence_packet_count },
	    { "evidence_packet_member_count", summary.evidence_packet_member_count },
	    { "evidence_packet_confidence_mean",
	      summary.evidence_packet_confidence_mean },
	    { "ranked_candidates", std::move (ranked) },
    { "rejected_candidates", std::move (rejected) },
    { "evidence_packets", std::move (evidence_packets) },
  };
}

nlohmann::json
WorkCountersJson (const cortext::Cortext::Context &ctx)
{
  const auto value = [&ctx] (const char *name) {
    const auto it = ctx.output.operation_ms.find (name);
    return it == ctx.output.operation_ms.end () ? 0.0 : it->second;
  };
  const double supersession_current
      = value ("MemoryStorage.supersession_current_candidate_count");
  const double supersession_historical
      = value ("MemoryStorage.supersession_historical_candidate_count");
  const double supersession_current_rows
      = value ("MemoryStorage.supersession_current_rows_visited");
  const double supersession_historical_rows
      = value ("MemoryStorage.supersession_historical_rows_visited");
  return {
    { "retrieval_suppression_id_count",
      value ("SignalProcessor.retrieval_suppression_id_count") },
    { "predictive_id_count", value ("SignalProcessor.predictive_id_count") },
    { "rollback_full_cache_copy_count",
      value ("SignalProcessor.snapshot_full_cache_copy_count") },
    { "rollback_cache_entry_copy_count",
      value ("SignalProcessor.snapshot_cache_entry_copy_count") },
    { "graph_candidate_count", value ("GraphRetrieve.candidate_count") },
    { "graph_exact_comparison_count",
      value ("GraphRetrieve.family_exact_comparison_count") },
    { "graph_cache_rebuild_count",
      value ("GraphRetrieve.cache_rebuild_count") },
    { "graph_rows_visited", value ("GraphRetrieve.rows_visited") },
    { "competition_candidate_count",
      value ("Competition.candidate_count") },
    { "competition_rows_visited", value ("Competition.rows_visited") },
    { "competition_rows_touched", value ("Competition.rows_touched") },
    { "supersession_current_candidate_count", supersession_current },
    { "supersession_historical_candidate_count", supersession_historical },
    { "supersession_rows_visited",
      supersession_current_rows + supersession_historical_rows },
    { "emotional_source_count", value ("EmotionalCascade.source_count") },
    { "emotional_neighbor_count", value ("EmotionalCascade.neighbor_count") },
    { "emotional_update_count", value ("EmotionalCascade.update_count") },
  };
}

std::uint64_t
RequiredOperationCounter (const cortext::Cortext::Context &ctx,
                          const std::string &name)
{
  const auto value = ctx.output.operation_ms.find (name);
  if (value == ctx.output.operation_ms.end ()
      || !std::isfinite (value->second) || value->second < 0.0
      || std::floor (value->second) != value->second)
    throw std::runtime_error ("missing or invalid operation counter: " + name);
  return static_cast<std::uint64_t> (value->second);
}

nlohmann::json
ConsolidationEpochCountersJson (const cortext::Cortext::Context &ctx,
                                const std::string &prefix)
{
  const auto counter = [&ctx, &prefix] (const char *name) {
    return RequiredOperationCounter (ctx, prefix + "." + name);
  };
  return {
    { "accumulator_signal_count", counter ("accumulator_signal_count") },
    { "working_memory_pending_signal_count",
      counter ("working_memory_pending_signal_count") },
    { "consolidation_dirty_memory_count",
      counter ("consolidation_dirty_memory_count") },
    { "consolidation_dirty_association_count",
      counter ("consolidation_dirty_association_count") },
    { "consolidation_dirty_index_count",
      counter ("consolidation_dirty_index_count") },
  };
}

nlohmann::json
ActiveEpochCountersJson (const cortext::Cortext::Context &ctx)
{
  return {
    { "event_count",
      RequiredOperationCounter (
          ctx, "SignalProcessor.rif_active_epoch_event_count") },
    { "mutation_count",
      RequiredOperationCounter (
          ctx, "SignalProcessor.rif_active_epoch_mutation_count") },
    { "allocated_bytes",
      RequiredOperationCounter (
          ctx, "SignalProcessor.rif_active_epoch_allocated_bytes") },
    { "required",
      RequiredOperationCounter (
          ctx, "SignalProcessor.rif_active_epoch_required") == 1.0 },
  };
}

nlohmann::json
WorkingSetCurveRow (int event_index,
                    const EventDoc &doc,
                    const cortext::Cortext::Context &ctx,
                    bool full_operation_ms = false,
                    bool bounded_behavior = false)
{
  const int retrieved_tokens = EstimateMemoryPacketTokens (ctx.retrieved_memory);
  const int working_tokens = bounded_behavior
                                 ? 0
                                 : EstimateMemoryPacketTokens (
                                       ctx.working_memory);
  std::vector<std::pair<std::string, double> > sorted_ops (
      ctx.output.operation_ms.begin (), ctx.output.operation_ms.end ());
  std::sort (sorted_ops.begin (), sorted_ops.end (),
             [] (const auto &lhs, const auto &rhs) {
               if (lhs.second != rhs.second)
                 return lhs.second > rhs.second;
               return lhs.first < rhs.first;
             });
  nlohmann::json top_ops = nlohmann::json::array ();
  for (const auto &row : sorted_ops)
    {
      if (static_cast<int> (top_ops.size ()) >= 8)
        break;
      top_ops.push_back ({ { "operation", row.first }, { "ms", row.second } });
    }
  nlohmann::json row = {
    { "event_index", event_index },
    { "cumulative_events", event_index + 1 },
    { "timestamp", doc.timestamp },
    { "source_id", doc.source_id },
    { "modality", doc.modality },
    { "working_items", ctx.working_memory.size () },
    { "retrieved_items", ctx.retrieved_memory.size () },
    { "working_tokens",
      bounded_behavior ? nlohmann::json (nullptr)
                       : nlohmann::json (working_tokens) },
    { "retrieved_tokens", retrieved_tokens },
    { "active_context_tokens",
      bounded_behavior ? nlohmann::json (nullptr)
                       : nlohmann::json (working_tokens + retrieved_tokens) },
    { "working_tokens_observed", !bounded_behavior },
    { "latency_ms", ctx.total_ms },
    { "encode_ms", ctx.encode_ms },
    { "process_ms", ctx.process_ms },
    { "hydrate_ms", ctx.hydrate_ms },
    { "top_operation_ms", std::move (top_ops) },
    { "behavior", PublicBehaviorJson (ctx, bounded_behavior) },
  };
  if (HasEnvValue ("CORTEXT_PROFILE_WORK_COUNTERS"))
    row["work_counters"] = WorkCountersJson (ctx);
  if (full_operation_ms)
    {
      nlohmann::json operation_ms = nlohmann::json::object ();
      for (const auto &[name, ms] : ctx.output.operation_ms)
        {
          operation_ms[name] = ms;
        }
      row["operation_ms"] = std::move (operation_ms);
    }
  return row;
}

nlohmann::json
DocIndicesJson (const std::vector<EventDoc> &docs)
{
  nlohmann::json out = nlohmann::json::array ();
  for (const auto &doc : docs)
    out.push_back (doc.index);
  return out;
}

std::vector<EventDoc>
RollingHistoryDocs (const std::vector<EventDoc> &prior_docs, int token_budget)
{
  std::vector<EventDoc> selected;
  int used = 0;
  for (auto it = prior_docs.rbegin (); it != prior_docs.rend (); ++it)
    {
      const int cost = EstimateTokens (it->source_id) + EstimateTokens (it->text);
      if (!selected.empty () && token_budget > 0 && used + cost > token_budget)
        break;
      selected.push_back (*it);
      used += cost;
    }
  std::reverse (selected.begin (), selected.end ());
  return selected;
}

int
DocTokenCost (const EventDoc &doc)
{
  return EstimateTokens (doc.source_id) + EstimateTokens (doc.text);
}

bool
IsTextChatDoc (const EventDoc &doc)
{
  return doc.modality == "text"
         && (doc.source_id == kUserSourceId
             || doc.source_id == kContactSourceId);
}

std::vector<EventDoc>
TextChatDocs (const std::vector<EventDoc> &docs)
{
  std::vector<EventDoc> out;
  for (const auto &doc : docs)
    {
      if (IsTextChatDoc (doc))
        out.push_back (doc);
    }
  return out;
}

int
DocPacketTokens (const std::vector<EventDoc> &docs)
{
  int tokens = 0;
  for (const auto &doc : docs)
    tokens += DocTokenCost (doc);
  return tokens;
}

int
RagContextTokens (const CompactedHistory &history,
                  const std::vector<EventDoc> &rag_docs)
{
  int tokens = history.raw_tokens + history.compacted_summary_tokens;
  std::unordered_set<int> seen;
  for (const auto &doc : history.raw_docs)
    seen.insert (doc.index);
  for (const auto &doc : rag_docs)
    {
      if (!IsTextChatDoc (doc) || seen.count (doc.index) > 0)
        continue;
      tokens += DocTokenCost (doc);
      seen.insert (doc.index);
    }
  return tokens;
}

std::string
SpeakerLabel (const std::string &source_id)
{
  if (source_id == kUserSourceId)
    return "User";
  if (source_id == kContactSourceId)
    return "Contact";
  return source_id;
}

std::string
CompactOneLine (std::string text)
{
  for (char &c : text)
    {
      if (c == '\n' || c == '\r' || c == '\t')
        c = ' ';
    }
  return Trim (text);
}

std::string
TruncateForTokenBudget (const std::string &text, int token_budget)
{
  if (token_budget <= 0 || EstimateTokens (text) <= token_budget)
    return text;
  const std::size_t char_budget = static_cast<std::size_t> (
      std::max (16, token_budget * 4 - 3));
  if (text.size () <= char_budget)
    return text;
  return text.substr (0, char_budget) + "...";
}

int
CompactionSummaryTokenBudget (int active_history_token_budget,
                              int compacted_original_tokens)
{
  if (active_history_token_budget <= 0)
    return std::max (256, compacted_original_tokens / 8);
  const int budget_cap = std::max (1, active_history_token_budget);
  return std::max (
      1,
      std::min ({ 1024, budget_cap,
                  std::max (32, active_history_token_budget / 8),
                  std::max (32, compacted_original_tokens / 6) }));
}

std::string
BuildExtractiveCompactionSummary (const std::vector<EventDoc> &prior_docs,
                                  int compacted_items,
                                  int compacted_original_tokens,
                                  int active_history_token_budget,
                                  int &selected_items)
{
  selected_items = 0;
  if (compacted_items <= 0)
    return {};

  const int summary_budget = CompactionSummaryTokenBudget (
      active_history_token_budget, compacted_original_tokens);
  const int prefix_items = std::min<int> (compacted_items, prior_docs.size ());
  std::vector<std::string> selected;
  int used = EstimateTokens ("Compacted prior chat summary:");

  for (int i = prefix_items - 1; i >= 0; --i)
    {
      const auto &doc = prior_docs[static_cast<std::size_t> (i)];
      std::ostringstream line;
      line << "- timestamp=" << doc.timestamp << " "
           << SpeakerLabel (doc.source_id) << ": "
           << CompactOneLine (doc.text);
      std::string text = line.str ();
      int cost = EstimateTokens (text);
      const int remaining = summary_budget - used;
      if (remaining <= 8)
        break;
      if (cost > remaining)
        {
          text = TruncateForTokenBudget (text, remaining);
          cost = EstimateTokens (text);
        }
      if (cost > remaining)
        break;
      selected.push_back (std::move (text));
      used += cost;
      ++selected_items;
    }

  std::reverse (selected.begin (), selected.end ());
  std::ostringstream out;
  out << "Compacted prior chat summary: " << compacted_items
      << " older messages, " << compacted_original_tokens
      << " original estimated tokens. Deterministic extractive excerpts:\n";
  for (const auto &line : selected)
    out << line << "\n";
  if (selected.empty ())
    out << "- <no excerpt fit within summary budget>\n";
  return Trim (out.str ());
}

CompactedHistory
CompactRollingHistoryDocs (const std::vector<EventDoc> &prior_docs,
                           int token_budget)
{
  CompactedHistory result;
  int used = 0;
  for (auto it = prior_docs.rbegin (); it != prior_docs.rend (); ++it)
    {
      const int cost = DocTokenCost (*it);
      if (!result.raw_docs.empty () && token_budget > 0
          && used + cost > token_budget)
        break;
      result.raw_docs.push_back (*it);
      used += cost;
    }
  std::reverse (result.raw_docs.begin (), result.raw_docs.end ());
  result.raw_items = static_cast<int> (result.raw_docs.size ());
  result.raw_tokens = used;
  result.compacted_items
      = static_cast<int> (prior_docs.size ()) - result.raw_items;
  if (result.compacted_items <= 0)
    return result;

  auto update_summary = [&] {
    result.compacted_original_tokens = 0;
    for (int i = 0; i < result.compacted_items; ++i)
      result.compacted_original_tokens += DocTokenCost (prior_docs[i]);
    result.summary_text = BuildExtractiveCompactionSummary (
        prior_docs, result.compacted_items, result.compacted_original_tokens,
        token_budget, result.compacted_summary_items);
    result.compacted_summary_tokens = result.summary_text.empty ()
                                          ? 0
                                          : EstimateTokens (result.summary_text);
  };
  update_summary ();
  result.compaction_events = 1;

  while (!result.raw_docs.empty () && token_budget > 0
         && result.raw_tokens + result.compacted_summary_tokens
                > token_budget)
    {
      result.raw_tokens -= DocTokenCost (result.raw_docs.front ());
      result.raw_docs.erase (result.raw_docs.begin ());
      ++result.compacted_items;
      result.raw_items = static_cast<int> (result.raw_docs.size ());
      update_summary ();
    }
  return result;
}

std::vector<EventDoc>
AdditionalRagDocs (const std::vector<EventDoc> &rag_docs,
                   const std::vector<EventDoc> &rolling_docs)
{
  std::unordered_set<int> rolling_indices;
  for (const auto &doc : rolling_docs)
    rolling_indices.insert (doc.index);

  std::vector<EventDoc> out;
  for (const auto &doc : rag_docs)
    {
      if (IsTextChatDoc (doc) && rolling_indices.count (doc.index) == 0)
        out.push_back (doc);
    }
  return out;
}

nlohmann::json
DocsPacketJson (const std::vector<EventDoc> &docs)
{
  nlohmann::json out = nlohmann::json::array ();
  for (const auto &doc : docs)
    {
      out.push_back ({
        { "index", doc.index },
        { "timestamp", doc.timestamp },
        { "source_id", doc.source_id },
        { "modality", doc.modality },
        { "tokens", EstimateTokens (doc.source_id) + EstimateTokens (doc.text) },
      });
    }
  return out;
}

long long
AnyLongLong (const std::map<std::string, std::any> &row, const char *key)
{
  auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    return 0;
  auto value = cortext::store::AnyToLongLong (it->second);
  return value.value_or (0);
}

double
AnyDouble (const std::map<std::string, std::any> &row, const char *key)
{
  auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    return 0.0;
  if (it->second.type () == typeid (double))
    return std::any_cast<double> (it->second);
  if (it->second.type () == typeid (float))
    return static_cast<double> (std::any_cast<float> (it->second));
  if (auto value = cortext::store::AnyToLongLong (it->second))
    return static_cast<double> (*value);
  return 0.0;
}

std::string
AnyString (const std::map<std::string, std::any> &row, const char *key)
{
  auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    return {};
  if (it->second.type () == typeid (std::string))
    return std::any_cast<std::string> (it->second);
  return {};
}

long long
CountRows (cortext::Store &store, const std::string &table)
{
  const auto rows = store.Execute ("SELECT COUNT(*) AS n FROM " + table);
  if (rows.empty ())
    return 0;
  return AnyLongLong (rows[0], "n");
}

std::string
RagDocKey (const std::string &source_id, std::uint64_t timestamp)
{
  return source_id + "\n" + std::to_string (timestamp);
}

void
NormalizeFloatVector (std::vector<float> &values)
{
  double sum = 0.0;
  for (float value : values)
    sum += static_cast<double> (value) * value;
  const double norm = std::sqrt (sum);
  if (norm <= 1e-12 || !std::isfinite (norm))
    return;
  const float inv_norm = static_cast<float> (1.0 / norm);
  for (float &value : values)
    value *= inv_norm;
}

std::vector<float>
RetrievalEmbeddingViewForBenchmark (const std::vector<float> &encoded)
{
  std::vector<float> out = encoded;
  if (out.size () > 256)
    {
      out.resize (256);
      NormalizeFloatVector (out);
    }
  return out;
}

VectorRagPacket
BuildVectorRagPacket (cortext::Store &store,
                      const std::vector<EventDoc> &prior_text_docs,
                      const std::vector<float> &query_embedding,
                      std::uint64_t query_ts, int requested_top_k)
{
  VectorRagPacket packet;
  std::unordered_map<std::string, EventDoc> docs_by_key;
  docs_by_key.reserve (prior_text_docs.size ());
  for (const auto &doc : prior_text_docs)
    docs_by_key[RagDocKey (doc.source_id, doc.timestamp)] = doc;

  packet.query_rows = query_embedding.empty () ? 0 : 1;
  packet.query_embedding_bytes = static_cast<long long> (
      query_embedding.size () * sizeof (float));
  if (query_embedding.empty ())
    return packet;

  store.Execute ("DROP TABLE IF EXISTS temp.rag_signal_embeddings");
  store.Execute (
      "CREATE VIRTUAL TABLE temp.rag_signal_embeddings USING vec0("
      "signal_id INTEGER PRIMARY KEY, embedding float[256], "
      "+source_id TEXT, +timestamp INTEGER)");
  auto prior_rows = store.Execute (
      "SELECT s.signal_id, s.source_id, s.timestamp, e.embedding "
      "FROM signals s "
      "JOIN embeddings e ON e.embedding_id = s.embedding_id "
      "WHERE s.timestamp < ? "
      "  AND s.modality = 'text' "
      "  AND s.source_id IN (?, ?) "
      "ORDER BY s.timestamp ASC, s.signal_id ASC",
      { static_cast<long long> (query_ts), std::string (kUserSourceId),
        std::string (kContactSourceId) });
  for (const auto &row : prior_rows)
    {
      auto it_embedding = row.find ("embedding");
      if (it_embedding == row.end () || !it_embedding->second.has_value ())
        continue;
      const auto embedding = cortext::store::BlobFromAny (it_embedding->second);
      if (embedding.empty ())
        continue;
      store.Execute (
          "INSERT INTO temp.rag_signal_embeddings("
          "signal_id, embedding, source_id, timestamp) VALUES (?, ?, ?, ?)",
          { AnyLongLong (row, "signal_id"), embedding,
            AnyString (row, "source_id"), AnyLongLong (row, "timestamp") });
    }
  packet.prior_chat_rows = CountRows (store, "temp.rag_signal_embeddings");
  if (packet.prior_chat_rows <= 0)
    return packet;

  const long long requested = std::max<long long> (1, requested_top_k);
  packet.vector_search_k = std::min<long long> (
      std::max<long long> (
          requested, requested * kNormalRagVectorSearchMultiplier),
      packet.prior_chat_rows);

  auto rows = store.Execute (
      "SELECT s.source_id, s.timestamp, distance "
      "FROM temp.rag_signal_embeddings s "
      "WHERE embedding MATCH ? AND k = ? "
      "ORDER BY distance ASC",
      { query_embedding, packet.vector_search_k });
  packet.candidate_rows = static_cast<long long> (rows.size ());

  std::unordered_set<std::string> seen;
  for (const auto &row : rows)
    {
      const std::string source_id = AnyString (row, "source_id");
      const long long ts = AnyLongLong (row, "timestamp");
      if (source_id.empty () || ts <= 0)
        continue;
      const std::string key = RagDocKey (
          source_id, static_cast<std::uint64_t> (ts));
      if (!seen.insert (key).second)
        continue;
      auto doc_it = docs_by_key.find (key);
      if (doc_it == docs_by_key.end ())
        continue;
      if (packet.docs.empty ())
        packet.best_distance = AnyDouble (row, "distance");
      packet.docs.push_back (doc_it->second);
      if (static_cast<int> (packet.docs.size ()) >= requested_top_k)
        break;
    }
  return packet;
}

nlohmann::json
OperationTimingsJson (const std::unordered_map<std::string, double> &timings)
{
  nlohmann::json out = nlohmann::json::object ();
  std::vector<std::pair<std::string, double> > sorted (timings.begin (),
                                                       timings.end ());
  std::sort (sorted.begin (), sorted.end (),
             [] (const auto &lhs, const auto &rhs) {
               if (lhs.second != rhs.second)
                 return lhs.second > rhs.second;
               return lhs.first < rhs.first;
             });
  for (const auto &row : sorted)
    out[row.first] = row.second;
  return out;
}

nlohmann::json
TopOperationTimingsJson (const std::unordered_map<std::string, double> &timings,
                         int limit)
{
  std::vector<std::pair<std::string, double> > sorted (timings.begin (),
                                                       timings.end ());
  std::sort (sorted.begin (), sorted.end (),
             [] (const auto &lhs, const auto &rhs) {
               if (lhs.second != rhs.second)
                 return lhs.second > rhs.second;
               return lhs.first < rhs.first;
             });
  nlohmann::json out = nlohmann::json::array ();
  for (const auto &row : sorted)
    {
      if (static_cast<int> (out.size ()) >= limit)
        break;
      out.push_back ({ { "operation", row.first }, { "ms", row.second } });
    }
  return out;
}

std::string
DumpJsonArtifact (const nlohmann::json &out)
{
  return out.dump (2, ' ', false, nlohmann::json::error_handler_t::replace);
}

void
WriteJsonArtifact (const fs::path &path, const nlohmann::json &out)
{
  const std::string payload = DumpJsonArtifact (out) + "\n";
  fs::path tmp = path;
  tmp += ".tmp";
  {
    std::ofstream stream (tmp, std::ios::binary | std::ios::trunc);
    if (!stream)
      {
        throw std::runtime_error ("failed to open summary artifact: "
                                  + tmp.string ());
      }
    stream << payload;
    if (!stream)
      {
        throw std::runtime_error ("failed to write summary artifact: "
                                  + tmp.string ());
      }
  }
  fs::rename (tmp, path);
}

fs::path
ProbeStreamPath (const fs::path &summary_path)
{
  fs::path path = summary_path;
  path += ".probes.jsonl";
  return path;
}

void
ResetProbeStream (const fs::path &path)
{
  if (!path.parent_path ().empty ())
    fs::create_directories (path.parent_path ());
  fs::remove (path);
}

void
AppendProbeStream (const fs::path &path, const nlohmann::json &probe)
{
  std::ofstream stream (path, std::ios::binary | std::ios::app);
  if (!stream)
    {
      throw std::runtime_error ("failed to open probe stream artifact: "
                                + path.string ());
    }
  stream << probe.dump (-1, ' ', false,
                        nlohmann::json::error_handler_t::replace)
         << "\n";
  if (!stream)
    {
      throw std::runtime_error ("failed to write probe stream artifact: "
                                + path.string ());
    }
}

std::string
SourceIdForMessage (const Message &message)
{
  return message.from_contact ? kContactSourceId : kUserSourceId;
}

std::string
MediaSourceId (const MediaItem &item)
{
  const std::string name = Lower (item.path.filename ().string ());
  if (item.kind == "audio"
      && (name.find ("_self") != std::string::npos
          || name.find (" self ") != std::string::npos))
    return kUserSourceId;
  return kContactSourceId;
}

std::vector<MediaItem>
FindMedia (const fs::path &dir)
{
  std::vector<MediaItem> items;
  for (const auto &entry : fs::recursive_directory_iterator (dir))
    {
      if (!entry.is_regular_file ())
        continue;
      const auto ext = Lower (entry.path ().extension ().string ());
      std::string kind;
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".heic"
          || ext == ".gif" || ext == ".tiff")
        kind = "image";
      else if (ext == ".mov" || ext == ".mp4" || ext == ".3gp")
        kind = "video";
      else if (ext == ".m4a" || ext == ".wav" || ext == ".mp3")
        kind = "audio";
      else
        continue;

      std::uint64_t ts = 0;
      if (auto parsed = ParseTimestamp (entry.path ().filename ().string ()))
        ts = *parsed;
      items.push_back ({ entry.path (), ts, kind });
    }
  std::sort (items.begin (), items.end (),
             [] (const auto &a, const auto &b) {
               if (a.timestamp != b.timestamp)
                 return a.timestamp < b.timestamp;
               return a.path.string () < b.path.string ();
             });
  return items;
}

long long
SqlCount (const fs::path &db_path, const std::string &sql)
{
  sqlite3 *db = nullptr;
  if (sqlite3_open_v2 (db_path.string ().c_str (), &db, SQLITE_OPEN_READONLY,
                       nullptr)
      != SQLITE_OK)
    {
      if (db)
        sqlite3_close (db);
      return -1;
    }

  sqlite3_stmt *stmt = nullptr;
  long long value = -1;
  if (sqlite3_prepare_v2 (db, sql.c_str (), -1, &stmt, nullptr) == SQLITE_OK
      && sqlite3_step (stmt) == SQLITE_ROW)
    {
      value = sqlite3_column_int64 (stmt, 0);
    }
  if (stmt)
    sqlite3_finalize (stmt);
  sqlite3_close (db);
  return value;
}

nlohmann::json
StoreCheckpoint (const fs::path &db_path, std::size_t event_end)
{
  return {
    { "event_end", event_end },
    { "counts",
      {
        { "memories", SqlCount (db_path, "SELECT COUNT(*) FROM memories") },
        { "signals", SqlCount (db_path, "SELECT COUNT(*) FROM signals") },
        { "associations",
          SqlCount (db_path, "SELECT COUNT(*) FROM associations") },
        // Read vec0's ordinary shadow tables so this separate read-only
        // connection does not need to load the extension outside the timed
        // engine connection.
        { "embeddings",
          SqlCount (db_path, "SELECT COUNT(*) FROM embeddings_rowids") },
        { "current_memory_embeddings",
          SqlCount (db_path,
                    "SELECT COUNT(*) FROM "
                    "current_memory_embeddings_rowids") },
        { "memory_reconstructions",
          SqlCount (db_path, "SELECT COUNT(*) FROM memory_reconstructions") },
        { "rif_recovery_clock",
          SqlCount (db_path, "SELECT COUNT(*) FROM rif_recovery_clock") },
        { "rif_generation_resets",
          SqlCount (db_path, "SELECT COUNT(*) FROM rif_generation_resets") },
        { "rif_active_state",
          SqlCount (db_path, "SELECT COUNT(*) FROM rif_active_state") },
      } },
  };
}

int
ProbeStreamRows (const fs::path &path)
{
  std::ifstream in (path);
  if (!in)
    return 0;
  int rows = 0;
  std::string line;
  while (std::getline (in, line))
    {
      if (!Trim (line).empty ())
        ++rows;
    }
  return rows;
}

double
PeakRssMb ()
{
#if defined(__APPLE__) || defined(__linux__)
  struct rusage usage {};
  if (getrusage (RUSAGE_SELF, &usage) != 0)
    return 0.0;
#if defined(__APPLE__)
  return static_cast<double> (usage.ru_maxrss) / (1024.0 * 1024.0);
#else
  return static_cast<double> (usage.ru_maxrss) / 1024.0;
#endif
#else
  return 0.0;
#endif
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
      if (arg == "--input-dir")
        cfg.input_dir = require_value ();
      else if (arg == "--transcript")
        cfg.transcript = require_value ();
      else if (arg == "--packet-corpus")
        cfg.packet_corpus = require_value ();
      else if (arg == "--message-corpus")
        cfg.message_corpus = require_value ();
      else if (arg == "--shadow-embedding-corpus")
        cfg.shadow_embedding_corpus = require_value ();
      else if (arg == "--db")
        cfg.db_path = require_value ();
      else if (arg == "--out")
        cfg.output_path = require_value ();
      else if (arg == "--max-messages")
        cfg.max_messages = std::stoi (require_value ());
      else if (arg == "--max-packets")
        cfg.max_packets = std::stoi (require_value ());
      else if (arg == "--shadow-queries")
        cfg.shadow_queries = std::stoi (require_value ());
      else if (arg == "--shadow-result-k")
        cfg.shadow_result_k = std::stoi (require_value ());
      else if (arg == "--skip-packets")
        cfg.skip_packets = std::stoi (require_value ());
      else if (arg == "--skip-messages")
        cfg.skip_messages = std::stoi (require_value ());
      else if (arg == "--media-limit")
        cfg.media_limit = std::stoi (require_value ());
      else if (arg == "--consolidate-every")
        cfg.consolidate_every = std::stoi (require_value ());
      else if (arg == "--warmup-events")
        cfg.warmup_events = std::stoi (require_value ());
      else if (arg == "--probe-stride")
        cfg.probe_stride = std::stoi (require_value ());
      else if (arg == "--rag-top-k")
        cfg.rag_top_k = std::stoi (require_value ());
      else if (arg == "--active-history-token-budget")
        cfg.active_history_token_budget = std::stoi (require_value ());
      else if (arg == "--focus")
        cfg.focus = std::stod (require_value ());
      else if (arg == "--sensitivity")
        cfg.sensitivity = std::stod (require_value ());
      else if (arg == "--stability")
        cfg.stability = std::stod (require_value ());
      else if (arg == "--daily-consolidation")
        {
          cfg.daily_consolidation = true;
          cfg.consolidate_every = 0;
        }
      else if (arg == "--skip-final-consolidation")
        cfg.skip_final_consolidation = true;
      else if (arg == "--daily-consolidation-hour")
        {
          cfg.daily_consolidation_hour = std::stoi (require_value ());
          if (cfg.daily_consolidation_hour < 0
              || cfg.daily_consolidation_hour > 23)
            throw std::runtime_error (
                "--daily-consolidation-hour must be in [0, 23]");
        }
      else if (arg == "--replay-timezone")
        cfg.replay_timezone = require_value ();
      else if (arg == "--append")
        cfg.append = true;
      else if (arg == "--resume-from-existing")
        {
          cfg.resume_from_existing = true;
          cfg.append = true;
        }
      else if (arg == "--profile-probes-only")
        {
          cfg.profile_probes_only = true;
          cfg.append = true;
        }
      else if (arg == "--checkpoint-eval-only")
        {
          cfg.checkpoint_eval_only = true;
          cfg.append = true;
        }
      else if (arg == "--checkpoint-after-timestamp")
        cfg.checkpoint_after_timestamp
            = static_cast<std::uint64_t> (std::stoull (require_value ()));
      else if (arg == "--checkpoint-query-count")
        cfg.checkpoint_query_count = std::stoi (require_value ());
      else if (arg == "--checkpoint-query-stride")
        cfg.checkpoint_query_stride = std::stoi (require_value ());
      else if (arg == "--checkpoint-query-days")
        cfg.checkpoint_query_days = std::stoi (require_value ());
      else if (arg == "--checkpoint-queries-per-day")
        cfg.checkpoint_queries_per_day = std::stoi (require_value ());
      else if (arg == "--progress-stride")
        cfg.progress_stride = std::stoi (require_value ());
      else if (arg == "--fixed-timestamp-base-ms")
        cfg.fixed_timestamp_base_ms
            = static_cast<std::uint64_t> (std::stoull (require_value ()));
      else if (arg == "--fixed-timestamp-step-ms")
        cfg.fixed_timestamp_step_ms
            = static_cast<std::uint64_t> (std::stoull (require_value ()));
      else if (arg == "--full-operation-ms")
        cfg.full_operation_ms = true;
      else if (arg == "--behavior-oracle")
        cfg.behavior_oracle = true;
      else if (arg == "--honor-required-consolidation")
        cfg.honor_required_consolidation = true;
      else if (arg == "--no-profile-rows")
        cfg.no_profile_rows = true;
      else if (arg == "--reuse-db")
        cfg.reuse_db = true;
      else if (arg == "--retention")
        {
          const std::string value = Lower (require_value ());
          if (value == "natural")
            cfg.retention = cortext::Retention::Natural;
          else if (value == "durable")
            cfg.retention = cortext::Retention::Durable;
          else
            throw std::runtime_error (
                "--retention must be natural or durable");
        }
      else if (arg == "--sqlite-profile")
        cfg.sqlite_profile = require_value ();
      else
        throw std::runtime_error ("unknown argument: " + arg);
    }
  return cfg;
}

std::vector<std::string>
ReadPacketCorpus (const fs::path &path, int skip_packets, int max_packets)
{
  std::ifstream stream (path);
  if (!stream)
    throw std::runtime_error ("failed to open packet corpus: "
                              + path.string ());
  std::vector<std::string> packets;
  std::string line;
  int observed = 0;
  while (std::getline (stream, line))
    {
      line = Trim (line);
      if (line.empty ())
        continue;
      if (observed++ < skip_packets)
        continue;
      packets.push_back (std::move (line));
      if (max_packets >= 0
          && static_cast<int> (packets.size ()) >= max_packets)
        break;
    }
  return packets;
}

std::vector<std::string>
ReadMessageCorpus (const fs::path &path, int skip_messages,
                   int max_messages)
{
  std::ifstream stream (path);
  if (!stream)
    throw std::runtime_error ("failed to open message corpus: "
                              + path.string ());
  std::vector<std::string> messages;
  std::string line;
  int observed = 0;
  while (std::getline (stream, line))
    {
      line = Trim (line);
      if (line.empty ())
        continue;
      const auto value = nlohmann::json::parse (line);
      if (!value.is_string ())
        throw std::runtime_error (
            "message corpus rows must be JSON strings");
      if (observed++ < skip_messages)
        continue;
      messages.push_back (value.get<std::string> ());
      if (max_messages >= 0
          && static_cast<int> (messages.size ()) >= max_messages)
        break;
    }
  return messages;
}

struct ShadowEmbeddingCorpus
{
  std::vector<long long> ids;
  std::vector<cortext::operations::bounded_activation_shadow_internal::
                  FixedEmbedding>
      embeddings;
};

ShadowEmbeddingCorpus
ReadShadowEmbeddingCorpus (const fs::path &path)
{
  namespace shadow
      = cortext::operations::bounded_activation_shadow_internal;
  std::ifstream stream (path, std::ios::binary);
  if (!stream)
    throw std::runtime_error ("failed to open shadow embedding corpus: "
                              + path.string ());
  std::uint64_t count = 0;
  std::uint32_t dimension = 0;
  stream.read (reinterpret_cast<char *> (&count), sizeof (count));
  stream.read (reinterpret_cast<char *> (&dimension), sizeof (dimension));
  if (!stream || count == 0 || dimension == 0
      || dimension > shadow::kMaximumEmbeddingDimension)
    throw std::runtime_error ("invalid shadow embedding corpus header");
  ShadowEmbeddingCorpus corpus;
  corpus.ids.resize (static_cast<std::size_t> (count));
  corpus.embeddings.resize (static_cast<std::size_t> (count));
  std::vector<float> values (dimension, 0.0F);
  for (std::size_t index = 0; index < corpus.embeddings.size (); ++index)
    {
      stream.read (reinterpret_cast<char *> (&corpus.ids[index]),
                   sizeof (long long));
      stream.read (reinterpret_cast<char *> (values.data ()),
                   static_cast<std::streamsize> (values.size ()
                                                 * sizeof (float)));
      if (!stream)
        throw std::runtime_error (
            "truncated shadow embedding corpus at record "
            + std::to_string (index));
      Eigen::Map<const Eigen::VectorXf> mapped (
          values.data (), static_cast<Eigen::Index> (values.size ()));
      if (!shadow::Normalize (mapped, corpus.embeddings[index]))
        throw std::runtime_error (
            "invalid embedding in shadow corpus at record "
            + std::to_string (index));
    }
  char trailing = 0;
  if (stream.read (&trailing, 1))
    throw std::runtime_error ("shadow embedding corpus has trailing bytes");
  return corpus;
}

struct ShadowQualityAccumulator
{
  std::uint64_t count = 0;
  double exact_id_recall = 0.0;
  double duplicate_equivalent_recall = 0.0;
  double top1_exact_id = 0.0;
  double top1_duplicate_equivalent = 0.0;
  double best_cosine_similarity = 0.0;
  double best_cosine_regret = 0.0;
  double semantic_coverage = 0.0;
  double hit_similarity_0_99 = 0.0;
  double hit_similarity_0_95 = 0.0;
  double comparisons = 0.0;
  double candidate_count = 0.0;
  std::uint64_t max_comparisons = 0;
  std::array<std::uint64_t, 4> retrieved_age_counts {};
};

nlohmann::json
FinalShadowQuality (const ShadowQualityAccumulator &value)
{
  const double denominator = std::max<std::uint64_t> (1, value.count);
  const auto age_total = std::accumulate (
      value.retrieved_age_counts.begin (), value.retrieved_age_counts.end (),
      std::uint64_t { 0 });
  nlohmann::json age = nlohmann::json::array ();
  for (const auto count : value.retrieved_age_counts)
    age.push_back (static_cast<double> (count)
                   / std::max<std::uint64_t> (1, age_total));
  return {
    { "count", value.count },
    { "mean_exact_id_recall_at_k", value.exact_id_recall / denominator },
    { "mean_duplicate_equivalent_recall_at_k",
      value.duplicate_equivalent_recall / denominator },
    { "mean_top1_exact_id", value.top1_exact_id / denominator },
    { "mean_top1_duplicate_equivalent",
      value.top1_duplicate_equivalent / denominator },
    { "mean_best_cosine_similarity",
      value.best_cosine_similarity / denominator },
    { "mean_best_cosine_regret", value.best_cosine_regret / denominator },
    { "mean_exact_neighbor_semantic_coverage",
      value.semantic_coverage / denominator },
    { "mean_hit_similarity_0_99",
      value.hit_similarity_0_99 / denominator },
    { "mean_hit_similarity_0_95",
      value.hit_similarity_0_95 / denominator },
    { "mean_comparisons", value.comparisons / denominator },
    { "max_comparisons", value.max_comparisons },
    { "mean_candidate_count", value.candidate_count / denominator },
    { "retrieved_age_quartile_fraction", std::move (age) },
  };
}

nlohmann::json
ShadowWorkQuartiles (const std::vector<std::uint64_t> &values)
{
  nlohmann::json result = nlohmann::json::array ();
  for (std::size_t quartile = 0; quartile < 4; ++quartile)
    {
      const std::size_t begin = values.size () * quartile / 4;
      const std::size_t end = values.size () * (quartile + 1) / 4;
      std::uint64_t maximum = 0;
      long double sum = 0.0;
      for (std::size_t index = begin; index < end; ++index)
        {
          maximum = std::max (maximum, values[index]);
          sum += values[index];
        }
      result.push_back (
          { { "count", end - begin },
            { "mean", end > begin
                          ? static_cast<double> (sum / (end - begin))
                          : 0.0 },
            { "max", maximum } });
    }
  return result;
}

int
RunShadowEmbeddingQuality (const Config &cfg)
{
  namespace shadow
      = cortext::operations::bounded_activation_shadow_internal;
  if (cfg.shadow_queries <= 0 || cfg.shadow_result_k <= 0)
    throw std::runtime_error (
        "shadow query count and result k must be positive");
  const auto corpus = ReadShadowEmbeddingCorpus (
      cfg.shadow_embedding_corpus);
  const auto parameters
      = shadow::DeriveParameters (cfg.focus, cfg.sensitivity, cfg.stability);
  shadow::State state (parameters);
  state.available = true;
  std::vector<std::uint64_t> normal_work;
  std::vector<std::uint64_t> consolidation_work;
  normal_work.reserve (corpus.embeddings.size ());
  consolidation_work.reserve (
      corpus.embeddings.size () / parameters.consolidation_interval + 1);
  const auto build_started = std::chrono::steady_clock::now ();
  for (std::size_t index = 0; index < corpus.embeddings.size (); ++index)
    {
      const auto comparisons = shadow::Ingest (
          state, corpus.embeddings[index], index, index + 1);
      normal_work.push_back (comparisons);
      if ((index + 1) % parameters.consolidation_interval == 0)
        consolidation_work.push_back (shadow::Consolidate (state));
      if (state.disabled)
        throw std::runtime_error (
            std::string ("shadow disabled while building corpus: ")
            + shadow::FailureName (state.failure));
    }
  consolidation_work.push_back (shadow::Consolidate (state));
  const double build_seconds
      = std::chrono::duration<double> (
            std::chrono::steady_clock::now () - build_started)
            .count ();

  struct RankedCandidate
  {
    float distance = 0.0F;
    std::size_t index = 0;
  };
  const auto rank_less = [&] (const RankedCandidate &left,
                              const RankedCandidate &right) {
    return left.distance < right.distance
           || (left.distance == right.distance
               && corpus.ids[left.index] < corpus.ids[right.index]);
  };
  const auto take_best = [&] (std::vector<RankedCandidate> &ranked,
                              std::size_t count) {
    count = std::min (count, ranked.size ());
    if (count < ranked.size ())
      std::nth_element (ranked.begin (), ranked.begin () + count,
                        ranked.end (), rank_less);
    ranked.resize (count);
    std::sort (ranked.begin (), ranked.end (), rank_less);
  };

  ShadowQualityAccumulator all;
  ShadowQualityAccumulator duplicates;
  ShadowQualityAccumulator nonduplicates;
  std::array<ShadowQualityAccumulator, 4> query_age;
  const auto evaluation_started = std::chrono::steady_clock::now ();
  for (int query_number = 0; query_number < cfg.shadow_queries;
       ++query_number)
    {
      const std::size_t query_index
          = cfg.shadow_queries == 1
                ? 0
                : static_cast<std::size_t> (
                      static_cast<long double> (query_number)
                      * (corpus.embeddings.size () - 1)
                      / (cfg.shadow_queries - 1));
      std::vector<RankedCandidate> exact;
      exact.reserve (corpus.embeddings.size () - 1);
      for (std::size_t index = 0; index < corpus.embeddings.size (); ++index)
        if (index != query_index)
          exact.push_back (
              { shadow::SquaredDistance (corpus.embeddings[query_index],
                                         corpus.embeddings[index]),
                index });
      take_best (exact, static_cast<std::size_t> (cfg.shadow_result_k));

      shadow::PreparedObservation prepared;
      const auto route_comparisons
          = shadow::Route (state, corpus.embeddings[query_index], prepared);
      std::vector<RankedCandidate> candidates;
      candidates.reserve (prepared.candidate_count);
      for (std::size_t candidate = 0;
           candidate < prepared.candidate_count; ++candidate)
        {
          const auto index = static_cast<std::size_t> (
              prepared.candidate_identities[candidate]);
          if (index >= corpus.embeddings.size () || index == query_index)
            continue;
          candidates.push_back (
              { shadow::SquaredDistance (corpus.embeddings[query_index],
                                         corpus.embeddings[index]),
                index });
        }
      const std::size_t candidate_count = candidates.size ();
      take_best (candidates,
                 static_cast<std::size_t> (cfg.shadow_result_k));
      if (exact.empty () || candidates.empty ())
        throw std::runtime_error ("shadow quality query returned no result");

      double exact_hits = 0.0;
      for (const auto &expected : exact)
        if (std::any_of (
                candidates.begin (), candidates.end (),
                [&] (const RankedCandidate &candidate) {
                  return candidate.index == expected.index;
                }))
          exact_hits += 1.0;
      double equivalent_hits = 0.0;
      double semantic_coverage = 0.0;
      for (const auto &expected : exact)
        {
          float nearest = std::numeric_limits<float>::infinity ();
          for (const auto &candidate : candidates)
            nearest = std::min (
                nearest,
                shadow::SquaredDistance (corpus.embeddings[expected.index],
                                         corpus.embeddings[candidate.index]));
          equivalent_hits += nearest <= 1.0e-6F ? 1.0 : 0.0;
          semantic_coverage += 1.0 - nearest / 2.0;
        }
      const double exact_best_similarity = 1.0 - exact.front ().distance / 2.0;
      const double best_similarity = 1.0 - candidates.front ().distance / 2.0;
      const bool duplicate_query = exact.front ().distance <= 1.0e-6F;

      auto add = [&] (ShadowQualityAccumulator &accumulator) {
        ++accumulator.count;
        accumulator.exact_id_recall
            += exact_hits / static_cast<double> (cfg.shadow_result_k);
        accumulator.duplicate_equivalent_recall
            += equivalent_hits / exact.size ();
        accumulator.top1_exact_id
            += candidates.front ().index == exact.front ().index ? 1.0 : 0.0;
        accumulator.top1_duplicate_equivalent
            += candidates.front ().distance
                       <= exact.front ().distance + 1.0e-6F
                   ? 1.0
                   : 0.0;
        accumulator.best_cosine_similarity += best_similarity;
        accumulator.best_cosine_regret
            += std::max (0.0, exact_best_similarity - best_similarity);
        accumulator.semantic_coverage
            += semantic_coverage / exact.size ();
        accumulator.hit_similarity_0_99
            += best_similarity >= 0.99 ? 1.0 : 0.0;
        accumulator.hit_similarity_0_95
            += best_similarity >= 0.95 ? 1.0 : 0.0;
        accumulator.comparisons += route_comparisons;
        accumulator.max_comparisons
            = std::max (accumulator.max_comparisons, route_comparisons);
        accumulator.candidate_count += candidate_count;
        for (const auto &candidate : candidates)
          {
            const auto quartile = std::min<std::size_t> (
                3, candidate.index * 4 / corpus.embeddings.size ());
            ++accumulator.retrieved_age_counts[quartile];
          }
      };
      add (all);
      add (duplicate_query ? duplicates : nonduplicates);
      add (query_age[std::min<std::size_t> (
          3, query_index * 4 / corpus.embeddings.size ())]);
    }
  const double evaluation_seconds
      = std::chrono::duration<double> (
            std::chrono::steady_clock::now () - evaluation_started)
            .count ();

  nlohmann::json age_quality = nlohmann::json::array ();
  for (const auto &quartile : query_age)
    age_quality.push_back (FinalShadowQuality (quartile));
  const auto normal_max
      = *std::max_element (normal_work.begin (), normal_work.end ());
  const auto consolidation_max = *std::max_element (
      consolidation_work.begin (), consolidation_work.end ());
  const auto normal_sum = std::accumulate (
      normal_work.begin (), normal_work.end (), static_cast<long double> (0));
  const auto consolidation_sum = std::accumulate (
      consolidation_work.begin (), consolidation_work.end (),
      static_cast<long double> (0));
  nlohmann::json restart_measurements = nlohmann::json::array ();
  for (const double fraction : { 0.25, 0.5, 1.0 })
    {
      const std::size_t row_count = std::max<std::size_t> (
          1, static_cast<std::size_t> (
                 corpus.embeddings.size () * fraction));
      auto restart_store = std::shared_ptr<cortext::Store> (
          cortext::SQLiteStore::Create (":memory:"));
      restart_store->Execute (
          "CREATE TABLE embeddings(embedding_id INTEGER PRIMARY KEY, "
          "embedding BLOB NOT NULL)");
      restart_store->Execute (
          "CREATE TABLE signals(signal_id INTEGER PRIMARY KEY, "
          "embedding_id INTEGER NOT NULL, timestamp INTEGER NOT NULL)");
      auto seed = restart_store->Begin ();
      std::vector<float> embedding (
          corpus.embeddings.front ().dimension, 0.0F);
      for (std::size_t index = 0; index < row_count; ++index)
        {
          std::copy_n (corpus.embeddings[index].values.begin (),
                       embedding.size (), embedding.begin ());
          const auto id = static_cast<long long> (index + 1);
          seed->Execute (
              "INSERT INTO embeddings(embedding_id, embedding) VALUES(?, ?)",
              { id, embedding });
          seed->Execute (
              "INSERT INTO signals(signal_id, embedding_id, timestamp) "
              "VALUES(?, ?, ?)",
              { id, id, id });
        }
      seed->Commit ();
      shadow::State rebuilt (parameters);
      shadow::RebuildFromPersistentAuthority (rebuilt, *restart_store);
      if (rebuilt.disabled || rebuilt.metrics.restart_rows_visited != row_count)
        throw std::runtime_error (
            "shadow SQLite restart measurement failed at fraction "
            + std::to_string (fraction));
      restart_measurements.push_back (
          { { "fraction", fraction },
            { "retained_rows", row_count },
            { "rows_visited", rebuilt.metrics.restart_rows_visited },
            { "rebuild_ms", rebuilt.metrics.restart_rebuild_ms },
            { "linear_history",
              rebuilt.metrics.restart_rebuild_linear_history },
            { "restart_production_gate",
              rebuilt.metrics.restart_production_gate } });
    }
  nlohmann::json out {
    { "schema", "cortext_bounded_activation_shadow_quality_v1" },
    { "observation_only", true },
    { "corpus", cfg.shadow_embedding_corpus.string () },
    { "embedding_count", corpus.embeddings.size () },
    { "embedding_dimension", corpus.embeddings.front ().dimension },
    { "query_count", cfg.shadow_queries },
    { "result_k", cfg.shadow_result_k },
    { "focus", cfg.focus },
    { "sensitivity", cfg.sensitivity },
    { "stability", cfg.stability },
    { "build_seconds", build_seconds },
    { "evaluation_seconds", evaluation_seconds },
    { "root_count", state.root_count },
    { "leaf_count", state.leaf_count },
    { "allocated_bytes", state.allocated_bytes },
    { "state_digest", shadow::StateDigest (state) },
    { "fixed_embedding_slots", parameters.fixed_embedding_slots },
    { "fixed_neighbor_slots", parameters.fixed_neighbor_slots },
    { "normal_comparison_bound", parameters.normal_comparison_bound },
    { "normal_comparison_mean",
      static_cast<double> (normal_sum / normal_work.size ()) },
    { "normal_comparison_max", normal_max },
    { "normal_comparison_quartiles", ShadowWorkQuartiles (normal_work) },
    { "recall_comparison_bound", parameters.recall_comparison_bound },
    { "consolidation_comparison_bound",
      parameters.consolidation_comparison_bound },
    { "consolidation_comparison_mean",
      static_cast<double> (consolidation_sum / consolidation_work.size ()) },
    { "consolidation_comparison_max", consolidation_max },
    { "consolidation_comparison_quartiles",
      ShadowWorkQuartiles (consolidation_work) },
    { "amortized_consolidation_comparisons_per_observation",
      static_cast<double> (consolidation_sum / normal_work.size ()) },
    { "quality",
      { { "all", FinalShadowQuality (all) },
        { "duplicate_queries", FinalShadowQuality (duplicates) },
        { "nonduplicate_queries", FinalShadowQuality (nonduplicates) } } },
    { "query_age_quartiles", std::move (age_quality) },
    { "sqlite_restart_measurements", std::move (restart_measurements) },
  };
  WriteJsonArtifact (cfg.output_path, out);
  std::cout << DumpJsonArtifact (out) << "\n";
  return 0;
}

int
RunPacketCorpusProfile (const Config &cfg,
                        const nlohmann::json &sqlite_profile)
{
  if (!cfg.packet_corpus.empty () && !cfg.message_corpus.empty ())
    throw std::runtime_error (
        "--packet-corpus and --message-corpus are mutually exclusive");
  if ((cfg.fixed_timestamp_base_ms == 0)
      != (cfg.fixed_timestamp_step_ms == 0))
    throw std::runtime_error (
        "fixed timestamp base and step must be provided together");
  if (!cfg.db_path.parent_path ().empty ())
    fs::create_directories (cfg.db_path.parent_path ());
  if (!cfg.output_path.parent_path ().empty ())
    fs::create_directories (cfg.output_path.parent_path ());
  if (!cfg.reuse_db)
    {
      fs::remove (cfg.db_path);
      fs::remove (cfg.db_path.string () + "-wal");
      fs::remove (cfg.db_path.string () + "-shm");
    }

  const bool message_mode = !cfg.message_corpus.empty ();
  const fs::path corpus_path
      = message_mode ? cfg.message_corpus : cfg.packet_corpus;
  const auto packets = message_mode
                           ? ReadMessageCorpus (
                                 corpus_path, cfg.skip_messages,
                                 cfg.max_messages)
                           : ReadPacketCorpus (
                                 corpus_path, cfg.skip_packets,
                                 cfg.max_packets);
  if (packets.empty ())
    throw std::runtime_error ("corpus contains no events");

  SetEnvValue ("CORTEXT_PROFILE_WORK_COUNTERS", "1", true);
  SetEnvValue ("CORTEXT_PROFILE_CONSOLIDATION_EPOCH", "1", true);
  SetEnvValue ("CORTEXT_PROFILE_EMOTIONAL_CASCADE", "1", true);
  SetEnvValue ("CORTEXT_PROFILE_GRAPH_RETRIEVAL", "1", false);

  cortext::Cortext::Config cortext_cfg;
  cortext_cfg.focus = cfg.focus;
  cortext_cfg.sensitivity = cfg.sensitivity;
  cortext_cfg.stability = cfg.stability;
  std::shared_ptr<cortext::Clock> replay_clock;
  if (cfg.fixed_timestamp_base_ms > 0)
    replay_clock = std::make_shared<cortext::FixedClock> (
        cfg.fixed_timestamp_base_ms);
  auto engine = cortext::Cortext::Create (
      cortext_cfg, cfg.db_path.string (), std::move (replay_clock));

  constexpr std::array<const char *, 49> kSelectedOperations {
    "cortext::operations::GraphAugmentedRetrieveCandidates",
    "GraphRetrieve.total",
    "GraphRetrieve.family_index",
    "GraphRetrieve.seed_cache_family_compare",
    "GraphRetrieve.seed_cache_distance",
    "GraphRetrieve.seed_knn_cache",
    "GraphRetrieve.seed_processor_surface",
    "GraphRetrieve.seed_knn_cache_rebuild",
    "GraphRetrieve.seed_knn_sql",
    "GraphRetrieve.seed_rank",
    "GraphRetrieve.association_expansion",
    "GraphRetrieve.rank_candidates",
    "GraphRetrieve.reconstruction_versions",
    "cortext::operations::ApplyRetrievalCompetition",
    "Competition.rif_recovery_active_sql",
    "Competition.score_candidates",
    "Predictive.decay_active_sql",
    "cortext::operations::MemoryStorage",
    "MemoryStorage.supersession_edges",
    "MemoryStorage.supersession_candidate_load",
    "MemoryStorage.supersession_current_candidate_execution_count",
    "MemoryStorage.supersession_historical_candidate_execution_count",
    "MemoryStorage.supersession_current_rows_visited",
    "MemoryStorage.supersession_historical_rows_visited",
    "MemoryStorage.supersession_sql_fallback_count",
    "cortext::operations::PropagateEmotionalCascade",
    "EmotionalCascade.source_execution_count",
    "EmotionalCascade.neighbor_execution_count",
    "EmotionalCascade.update_execution_count",
    "SignalProcessor.snapshot_context",
    "SignalProcessor.snapshot_full_cache_copy",
    "SignalProcessor.snapshot_cache_entry_copy",
    "SignalProcessor.commit_transaction",
    "SignalProcessor.retrieval_suppression_id_count",
    "SignalProcessor.predictive_id_count",
    "SignalProcessor.snapshot_full_cache_copy_count",
    "SignalProcessor.snapshot_cache_entry_copy_count",
    "GraphRetrieve.candidate_count",
    "GraphRetrieve.family_exact_comparison_count",
    "GraphRetrieve.cache_rebuild_count",
    "GraphRetrieve.rows_visited",
    "Competition.candidate_count",
    "Competition.rows_visited",
    "Competition.rows_touched",
    "MemoryStorage.supersession_current_candidate_count",
    "MemoryStorage.supersession_historical_candidate_count",
    "EmotionalCascade.source_count",
    "EmotionalCascade.neighbor_count",
    "EmotionalCascade.update_count",
  };

  nlohmann::json rows = nlohmann::json::array ();
  nlohmann::json consolidation_events = nlohmann::json::array ();
  nlohmann::json store_checkpoints = nlohmann::json::array ();
  const std::size_t checkpoint_stride = message_mode ? 100U : 500U;
  const std::size_t leading_checkpoint
      = packets.size () % checkpoint_stride;
  if (leading_checkpoint == 0)
    store_checkpoints.push_back (StoreCheckpoint (cfg.db_path, 0));
  double process_ms_total = 0.0;
  double total_ms_total = 0.0;
  double consolidation_ms_total = 0.0;
  int consolidation_runs = 0;
  const auto wall_start = std::chrono::steady_clock::now ();
  for (std::size_t index = 0; index < packets.size (); ++index)
    {
      const auto event_started = std::chrono::steady_clock::now ();
      const std::uint64_t timestamp
          = cfg.fixed_timestamp_base_ms > 0
                ? cfg.fixed_timestamp_base_ms
                      + static_cast<std::uint64_t> (cfg.skip_packets)
                            * cfg.fixed_timestamp_step_ms
                      + index * cfg.fixed_timestamp_step_ms
                : static_cast<std::uint64_t> (
                      std::chrono::duration_cast<std::chrono::milliseconds> (
                          std::chrono::system_clock::now ().time_since_epoch ())
                          .count ());
      EventDoc doc {
        static_cast<int> (index), timestamp, "profile/corpus", "text",
        packets[index],
      };
      auto ctx = cfg.fixed_timestamp_base_ms > 0
                     ? engine->ProcessTextAt (packets[index], doc.source_id,
                                              timestamp, cfg.retention)
                     : engine->ProcessText (packets[index], doc.source_id,
                                            cfg.retention);
      process_ms_total += ctx.process_ms;
      total_ms_total += ctx.total_ms;
      double event_consolidation_ms = 0.0;
      std::optional<nlohmann::json> consolidation_event;

      if (cfg.honor_required_consolidation
          && ctx.consolidation_state == cortext::ConsolidationState::Required)
        {
          const auto consolidation_start = std::chrono::steady_clock::now ();
          auto consolidation_ctx = engine->Consolidate ();
          const double consolidation_ms = std::chrono::duration<
              double, std::milli> (
              std::chrono::steady_clock::now () - consolidation_start)
                                               .count ();
          consolidation_ms_total += consolidation_ms;
          event_consolidation_ms = consolidation_ms;
          ++consolidation_runs;
          consolidation_event = {
            { "event_index", index },
            { "closing_event_index", index },
            { "consolidation_epoch_id",
              RequiredOperationCounter (
                  consolidation_ctx,
                  "ConsolidationEpoch.closed_epoch_id") },
            { "sealed_epoch_event_count",
              RequiredOperationCounter (
                  consolidation_ctx,
                  "ConsolidationEpoch.sealed_epoch_event_count") },
            { "sealed_epoch_mutation_count",
              RequiredOperationCounter (
                  consolidation_ctx,
                  "ConsolidationEpoch.sealed_epoch_mutation_count") },
            { "sealed_mutation_identity_verified",
              RequiredOperationCounter (
                  consolidation_ctx,
                  "ConsolidationEpoch.sealed_mutation_identity_verified")
                  == 1 },
            { "sealed_mutation_identity_count",
              RequiredOperationCounter (
                  consolidation_ctx,
                  "ConsolidationEpoch.sealed_mutation_identity_count") },
            { "duration_ms", consolidation_ms },
            { "pre_reset_counters",
              ConsolidationEpochCountersJson (
                  consolidation_ctx, "ConsolidationEpoch.pre") },
            { "post_reset_counters",
              ConsolidationEpochCountersJson (
                  consolidation_ctx, "ConsolidationEpoch.post") },
            { "post_reset_active_epoch",
              ActiveEpochCountersJson (consolidation_ctx) },
            { "state_after",
              ConsolidationStateName (
                  consolidation_ctx.consolidation_state) },
          };
          consolidation_events.push_back (*consolidation_event);
        }
      const auto event_ended = std::chrono::steady_clock::now ();
      if (!cfg.no_profile_rows)
        {
          auto row = WorkingSetCurveRow (
              static_cast<int> (index), doc, ctx, true, true);
          row["retention"] = cfg.retention == cortext::Retention::Natural
                                 ? "natural"
                                 : "durable";
          row["consolidation_ms"] = event_consolidation_ms;
          row["consolidation_epoch_id"] = RequiredOperationCounter (
              ctx, "ConsolidationEpoch.epoch_id");
          row["events_since_epoch_start"] = RequiredOperationCounter (
              ctx, "ConsolidationEpoch.events_since_epoch_start");
          row["consolidation_after_event"]
              = consolidation_event.has_value ();
          row["consolidation_epoch_counters"]
              = ConsolidationEpochCountersJson (
                  ctx, "ConsolidationEpoch");
          row["active_epoch"] = ActiveEpochCountersJson (ctx);
          const auto shadow_work
              = cortext::operations::bounded_activation_shadow_internal::
                  SingleWorkSnapshot ();
          if (shadow_work)
            row["bounded_activation_shadow_work"] = {
              { "available", shadow_work->available },
              { "disabled", shadow_work->disabled },
              { "failure",
                cortext::operations::bounded_activation_shadow_internal::
                    FailureName (shadow_work->failure) },
              { "generation", shadow_work->generation },
              { "event_index", shadow_work->event_index },
              { "root_count", shadow_work->root_count },
              { "leaf_count", shadow_work->leaf_count },
              { "normal_comparisons",
                shadow_work->metrics.normal_comparisons },
              { "recall_comparisons",
                shadow_work->metrics.recall_comparisons },
              { "consolidation_comparisons",
                shadow_work->metrics.consolidation_comparisons },
              { "consolidation_count",
                shadow_work->metrics.consolidation_count },
              { "allocation_after_initialization_count",
                shadow_work->metrics.allocation_after_initialization_count },
            };
          row["durable_barrier_ms"]
              = ctx.output.operation_ms.count (
                    "SignalProcessor.sqlite_wal_checkpoint")
                    ? ctx.output.operation_ms.at (
                          "SignalProcessor.sqlite_wal_checkpoint")
                    : 0.0;
          row["end_to_end_ms"] = std::chrono::duration<double, std::milli> (
                                     event_ended - event_started)
                                     .count ();
          for (const char *operation : kSelectedOperations)
            {
              if (!row["operation_ms"].contains (operation))
                row["operation_ms"][operation] = 0.0;
            }
          rows.push_back (std::move (row));
          const std::size_t event_end = index + 1;
          if (event_end >= leading_checkpoint
              && (event_end - leading_checkpoint) % checkpoint_stride == 0)
            store_checkpoints.push_back (
                StoreCheckpoint (cfg.db_path, event_end));
        }
      if (cfg.progress_stride > 0
          && (index + 1) % static_cast<std::size_t> (cfg.progress_stride) == 0)
        std::cerr << "packet profile " << (index + 1) << "/"
                  << packets.size () << "\n";
    }
  const auto wall_end = std::chrono::steady_clock::now ();
  const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds> (
      wall_end - wall_start).count ();

  nlohmann::json out;
  out["schema"] = "cortext_packet_storage_profile_v2";
  out["consolidation_epoch_schema"]
      = "post-process-pre-consolidation rows with adjacent successful close records";
  out["consolidation_epoch_relational_contract_sha256"]
      = "13806d20db179c8cf2dd0c7990f6355b6513c255bdf53df3b2c1082c636f0e4b";
  out["corpus"] = corpus_path.string ();
  out["corpus_unit"] = message_mode ? "whole_message" : "sentence_packet";
  out["retention"] = cfg.retention == cortext::Retention::Natural
                         ? "natural"
                         : "durable";
  out["processed_events"] = packets.size ();
  out["focus"] = cfg.focus;
  out["sensitivity"] = cfg.sensitivity;
  out["stability"] = cfg.stability;
  out["sqlite_profile"] = sqlite_profile;
  out["full_operation_ms"] = true;
  out["behavior_oracle"] = true;
  out["profile_rows_enabled"] = !cfg.no_profile_rows;
  out["honor_required_consolidation"]
      = cfg.honor_required_consolidation;
  out["active_epoch_limits"] = {
    { "event_count", 512 },
    { "mutation_count", 32768 },
    { "allocated_bytes", 64ULL * 1024ULL * 1024ULL },
  };
  out["fixed_timestamp_base_ms"] = cfg.fixed_timestamp_base_ms;
  out["fixed_timestamp_step_ms"] = cfg.fixed_timestamp_step_ms;
  out["mean_process_ms"] = process_ms_total / packets.size ();
  out["mean_total_ms"] = total_ms_total / packets.size ();
  out["wall_ms"] = wall_ms;
  out["consolidation_runs"] = consolidation_runs;
  out["consolidation_ms"] = consolidation_ms_total;
  out["consolidation_events"] = std::move (consolidation_events);
  out["profile_window_size"] = checkpoint_stride;
  out["store_checkpoint_policy"]
      = "end-anchored complete-window boundaries captured outside per-event timing";
  out["store_checkpoints"] = std::move (store_checkpoints);
  out["peak_rss_mb"] = PeakRssMb ();
  const auto shadow_snapshots
      = cortext::operations::bounded_activation_shadow_internal::Snapshots ();
  if (!shadow_snapshots.empty ())
    {
      const auto &shadow = shadow_snapshots.back ();
      const auto &parameters = shadow.parameters;
      const auto &metrics = shadow.metrics;
      out["bounded_activation_shadow"] = {
        { "schema", "cortext_bounded_activation_shadow_runtime_v1" },
        { "observation_only", true },
        { "canonical_sqlite_authoritative", true },
        { "available", shadow.available },
        { "disabled", shadow.disabled },
        { "failure",
          cortext::operations::bounded_activation_shadow_internal::FailureName (
              shadow.failure) },
        { "restart_rebuild_required", shadow.restart_rebuild_required },
        { "generation", shadow.generation },
        { "event_index", shadow.event_index },
        { "root_count", shadow.root_count },
        { "leaf_count", shadow.leaf_count },
        { "allocated_bytes", shadow.allocated_bytes },
        { "state_digest", shadow.state_digest },
        { "parameters",
          { { "roots", parameters.roots },
            { "leaf_capacity", parameters.leaf_capacity },
            { "children_per_root", parameters.children_per_root },
            { "root_beam", parameters.root_beam },
            { "leaf_beam", parameters.leaf_beam },
            { "representatives_per_leaf",
              parameters.representatives_per_leaf },
            { "neighbor_degree", parameters.neighbor_degree },
            { "activated_leaves_per_consolidation",
              parameters.activated_leaves_per_consolidation },
            { "consolidation_interval",
              parameters.consolidation_interval },
            { "centroid_alpha", parameters.centroid_alpha },
            { "root_creation_squared_distance",
              parameters.root_creation_squared_distance },
            { "leaf_creation_squared_distance",
              parameters.leaf_creation_squared_distance },
            { "fixed_embedding_slots", parameters.fixed_embedding_slots },
            { "fixed_neighbor_slots", parameters.fixed_neighbor_slots },
            { "normal_comparison_bound",
              parameters.normal_comparison_bound },
            { "recall_comparison_bound",
              parameters.recall_comparison_bound },
            { "consolidation_comparison_bound",
              parameters.consolidation_comparison_bound } } },
        { "metrics",
          { { "normal_comparisons", metrics.normal_comparisons },
            { "normal_comparisons_max", metrics.normal_comparisons_max },
            { "recall_comparisons", metrics.recall_comparisons },
            { "recall_comparisons_max", metrics.recall_comparisons_max },
            { "consolidation_comparisons",
              metrics.consolidation_comparisons },
            { "consolidation_comparisons_max",
              metrics.consolidation_comparisons_max },
            { "consolidation_count", metrics.consolidation_count },
            { "representative_replacements",
              metrics.representative_replacements },
            { "allocation_after_initialization_count",
              metrics.allocation_after_initialization_count },
            { "restart_rows_visited", metrics.restart_rows_visited },
            { "restart_rebuild_ms", metrics.restart_rebuild_ms },
            { "restart_rebuild_linear_history",
              metrics.restart_rebuild_linear_history },
            { "restart_production_gate",
              metrics.restart_production_gate } } },
      };
    }
  out["working_set_curve"] = std::move (rows);
  WriteJsonArtifact (cfg.output_path, out);
  std::cout << DumpJsonArtifact (
      { { "processed_events", packets.size () },
        { "retention", out["retention"] },
        { "wall_ms", wall_ms },
        { "consolidation_runs", consolidation_runs },
        { "output", cfg.output_path.string () },
        { "db", cfg.db_path.string () } })
            << "\n";
  return 0;
}

} // namespace

int
main (int argc, char **argv)
{
  try
    {
      Config cfg = ParseArgs (argc, argv);
      ApplyReplayTimezone (cfg.replay_timezone);
      if (!cfg.shadow_embedding_corpus.empty ())
        return RunShadowEmbeddingQuality (cfg);
      const nlohmann::json sqlite_profile
          = ApplySQLiteProfile (cfg.sqlite_profile);
      if (!cfg.packet_corpus.empty () || !cfg.message_corpus.empty ())
        return RunPacketCorpusProfile (cfg, sqlite_profile);
      const fs::path transcript
          = chat_replay::DiscoverTranscript (cfg.input_dir, cfg.transcript);
      auto messages = ParseMessages (transcript);
      const int parsed_messages = static_cast<int> (messages.size ());
      const int skipped_messages
          = std::min (std::max (0, cfg.skip_messages), parsed_messages);
      if (skipped_messages > 0)
        {
          messages.erase (messages.begin (),
                          messages.begin () + skipped_messages);
        }
      if (cfg.max_messages >= 0
          && static_cast<int> (messages.size ()) > cfg.max_messages)
        messages.resize (static_cast<size_t> (cfg.max_messages));
      const std::uint64_t message_window_start
          = messages.empty () ? 0 : messages.front ().timestamp;
      const std::uint64_t message_window_end
          = messages.empty () ? 0 : messages.back ().timestamp;

      if (!cfg.db_path.parent_path ().empty ())
        fs::create_directories (cfg.db_path.parent_path ());
      if (!cfg.output_path.parent_path ().empty ())
        fs::create_directories (cfg.output_path.parent_path ());
      const fs::path probe_stream_path = ProbeStreamPath (cfg.output_path);
      if (!cfg.resume_from_existing)
        ResetProbeStream (probe_stream_path);
      if (!cfg.append)
        {
          fs::remove (cfg.db_path);
          fs::remove (cfg.db_path.string () + "-wal");
          fs::remove (cfg.db_path.string () + "-shm");
        }

      ResumeCheckpoint resume;
      resume.enabled = cfg.resume_from_existing;
      if (resume.enabled)
        {
          if (!fs::exists (cfg.db_path))
            {
              throw std::runtime_error (
                  "--resume-from-existing requires an existing DB: "
                  + cfg.db_path.string ());
            }
          resume.db_signal_count = static_cast<int> (std::max<long long> (
              0, SqlCount (cfg.db_path, "SELECT COUNT(*) FROM signals")));
          resume.db_text_signal_count = static_cast<int> (std::max<long long> (
              0,
              SqlCount (cfg.db_path,
                        "SELECT COUNT(*) FROM signals WHERE modality = 'text'")));
          resume.after_timestamp = static_cast<std::uint64_t> (
              std::max<long long> (
                  0,
                  SqlCount (cfg.db_path,
                            "SELECT COALESCE(MAX(timestamp), 0) FROM signals "
                            "WHERE source_id IN ('User', 'Contact')")));
          if (resume.db_signal_count <= 0 || resume.after_timestamp == 0)
            {
              throw std::runtime_error (
                  "--resume-from-existing found no durable signal checkpoint in "
                  + cfg.db_path.string ());
            }
          resume.probe_count_offset = ProbeStreamRows (probe_stream_path);
        }

      cortext::Cortext::Config cortext_cfg;
      cortext_cfg.focus = cfg.focus;
      cortext_cfg.sensitivity = cfg.sensitivity;
      cortext_cfg.stability = cfg.stability;

      if (cfg.checkpoint_eval_only && cfg.checkpoint_after_timestamp == 0)
        {
          throw std::runtime_error (
              "--checkpoint-after-timestamp is required for checkpoint eval");
        }
      std::shared_ptr<cortext::Clock> clock;
      std::shared_ptr<cortext::FixedClock> fixed_clock;
      if (cfg.checkpoint_eval_only)
        {
          fixed_clock = std::make_shared<cortext::FixedClock> (
              cfg.checkpoint_after_timestamp);
          clock = fixed_clock;
        }

      auto engine = cortext::Cortext::Create (cortext_cfg, cfg.db_path.string (),
                                              clock);
      auto rag_encoder_selection
          = cortext::internal::CreatePreferredTextEncoder ();
      auto vector_rag_store
          = cortext::SQLiteStore::Create (cfg.db_path.string ());

      int processed_text = 0;
      int text_write_events = 0;
      int retrieval_events = 0;
      int retrieval_items = 0;
      int consolidation_runs = 0;
      double consolidation_ms_total = 0.0;
      double encode_ms_total = 0.0;
      double process_ms_total = 0.0;
      double hydrate_ms_total = 0.0;
      double total_ms_total = 0.0;
      int normal_rag_compaction_events = 0;
      int normal_rag_compacted_items = 0;
      int normal_rag_raw_items = 0;
      int normal_rag_raw_tokens = 0;
      int normal_rag_compacted_original_tokens = 0;
      int normal_rag_compacted_summary_tokens = 0;
      double normal_rag_compaction_ms_total = 0.0;
      double normal_rag_retrieval_ms_total = 0.0;
      int normal_rag_probe_count = 0;

      int media_attempted = 0;
      int media_processed = 0;
      int media_failures = 0;
      int image_processed = 0;
      int audio_processed = 0;
      int video_processed = 0;
      auto media = FindMedia (cfg.input_dir);
      if (message_window_start > 0 && message_window_end >= message_window_start)
        {
          media.erase (
              std::remove_if (
                  media.begin (), media.end (),
                  [&] (const MediaItem &item) {
                    return item.timestamp < message_window_start
                           || item.timestamp > message_window_end;
                  }),
              media.end ());
        }
      const fs::path tmp_dir = cfg.db_path.parent_path () / "chat_replay_live_media_tmp";
      fs::create_directories (tmp_dir);
      std::vector<EventDoc> prior_docs;
      std::vector<EventDoc> prior_text_docs;
      nlohmann::json probes = nlohmann::json::array ();
      nlohmann::json working_set_curve = nlohmann::json::array ();
      nlohmann::json consolidation_events = nlohmann::json::array ();
      std::vector<TimelineEvent> timeline;
      timeline.reserve (messages.size () + media.size ());
      for (std::size_t i = 0; i < messages.size (); ++i)
        timeline.push_back (
            { messages[i].timestamp, messages[i].index * 2, false, i });
      for (std::size_t i = 0; i < media.size (); ++i)
        timeline.push_back ({ media[i].timestamp,
                              static_cast<int> (messages.size () * 2 + i),
                              true,
                              i });
      std::sort (timeline.begin (), timeline.end (),
                 [] (const auto &a, const auto &b) {
                   if (a.timestamp != b.timestamp)
                     return a.timestamp < b.timestamp;
                   if (a.is_media != b.is_media)
                     return !a.is_media;
                   return a.order < b.order;
                 });

      if (cfg.resume_from_existing
          && (cfg.checkpoint_eval_only || cfg.profile_probes_only))
        {
          throw std::runtime_error (
              "--resume-from-existing is only supported for normal replay mode");
        }

      if (cfg.checkpoint_eval_only)
        {
          const auto checkpoint_started = std::chrono::steady_clock::now ();
          nlohmann::json checkpoint_probes = nlohmann::json::array ();
          std::vector<EventDoc> checkpoint_prior_docs;
          std::vector<EventDoc> checkpoint_prior_text_docs;
          int checkpoint_event_count = 0;
          int checkpoint_media_attempted = 0;
          int skipped_future_queries = 0;
          int checkpoint_future_turns_replayed = 0;
          int checkpoint_future_turns_probed = 0;
          int max_query_message_index = 0;
          const bool checkpoint_day_spread
              = cfg.checkpoint_query_days > 0
                || cfg.checkpoint_queries_per_day > 0;
          int effective_checkpoint_queries_per_day
              = cfg.checkpoint_queries_per_day;
          if (checkpoint_day_spread && cfg.checkpoint_query_days > 0
              && effective_checkpoint_queries_per_day <= 0)
            {
              effective_checkpoint_queries_per_day = std::max (
                  1,
                  (std::max (0, cfg.checkpoint_query_count)
                   + cfg.checkpoint_query_days - 1)
                      / cfg.checkpoint_query_days);
            }
          std::unordered_map<int, int> checkpoint_probes_by_day;

          for (const auto &event : timeline)
            {
              if (event.is_media)
                {
                  if (cfg.media_limit >= 0
                      && checkpoint_media_attempted >= cfg.media_limit)
                    continue;
                  const auto &item = media[event.index];
                  ++checkpoint_media_attempted;
                  EventDoc media_doc {
                    checkpoint_event_count,
                    item.timestamp,
                    MediaSourceId (item),
                    item.kind == "video" ? "image" : item.kind,
                    "[" + item.kind + " source blob: "
                        + item.path.filename ().string () + "]",
                  };
                  checkpoint_prior_docs.push_back (std::move (media_doc));
                  ++checkpoint_event_count;
                  continue;
                }

              const auto &msg = messages[event.index];
              const std::string source = SourceIdForMessage (msg);
              const EventDoc doc {
                checkpoint_event_count,
                msg.timestamp,
                source,
                "text",
                msg.text,
              };

              if (msg.timestamp > cfg.checkpoint_after_timestamp)
                {
                  const int query_day = LocalDayBucket (msg.timestamp);
                  bool stop_checkpoint_scan = false;
                  bool should_probe = checkpoint_probes.size ()
                                      < static_cast<std::size_t> (
                                          std::max (
                                              0,
                                              cfg.checkpoint_query_count));
                  if (should_probe && cfg.checkpoint_query_stride > 1
                      && skipped_future_queries % cfg.checkpoint_query_stride
                             != 0)
                    {
                      should_probe = false;
                    }
                  if (should_probe && checkpoint_day_spread)
                    {
                      const auto day_it
                          = checkpoint_probes_by_day.find (query_day);
                      const bool day_has_probe
                          = day_it != checkpoint_probes_by_day.end ();
                      if (!day_has_probe && cfg.checkpoint_query_days > 0
                          && static_cast<int> (checkpoint_probes_by_day.size ())
                                 >= cfg.checkpoint_query_days)
                        {
                          should_probe = false;
                          stop_checkpoint_scan = true;
                        }
                      else if (effective_checkpoint_queries_per_day > 0
                               && day_has_probe
                               && day_it->second
                                      >= effective_checkpoint_queries_per_day)
                        {
                          should_probe = false;
                        }
                    }

                  const bool should_replay_future_turn
                      = should_probe || checkpoint_day_spread;
                  cortext::Cortext::Context probe_ctx;
                  double cortext_latency_ms = 0.0;
                  if (should_replay_future_turn)
                    {
                      if (fixed_clock)
                        {
                          fixed_clock->SetNowMillis (msg.timestamp);
                        }
                      const auto cortext_started
                          = std::chrono::steady_clock::now ();
                      probe_ctx = engine->ProcessTextAt (
                          msg.text, source, msg.timestamp,
                          cortext::Retention::Ephemeral);
                      const auto cortext_ended
                          = std::chrono::steady_clock::now ();
                      cortext_latency_ms
                          = std::chrono::duration<double, std::milli> (
                                cortext_ended - cortext_started)
                                .count ();
                      ++checkpoint_future_turns_replayed;
                    }

                  if (should_probe)
                    {
                      const auto compaction_started
                          = std::chrono::steady_clock::now ();
                      const auto compacted_history
                          = CompactRollingHistoryDocs (
                              checkpoint_prior_text_docs,
                              cfg.active_history_token_budget);
                      const auto compaction_ended
                          = std::chrono::steady_clock::now ();
                      const double compaction_ms
                          = std::chrono::duration<double, std::milli> (
                                compaction_ended - compaction_started)
                                .count ();
                      const auto rag_started
                          = std::chrono::steady_clock::now ();
                      std::vector<float> rag_query_embedding;
                      rag_encoder_selection.encoder->EncodeText (
                          msg.text, rag_query_embedding);
                      rag_query_embedding
                          = RetrievalEmbeddingViewForBenchmark (
                              rag_query_embedding);
                      const auto vector_rag = BuildVectorRagPacket (
                          *vector_rag_store, checkpoint_prior_text_docs,
                          rag_query_embedding, msg.timestamp, cfg.rag_top_k);
                      const auto rag_ended = std::chrono::steady_clock::now ();
                      const double rag_retrieval_ms
                          = std::chrono::duration<double, std::milli> (
                                rag_ended - rag_started)
                                .count ();

                      normal_rag_compaction_events
                          += compacted_history.compaction_events;
                      normal_rag_compacted_items
                          += compacted_history.compacted_items;
                      normal_rag_raw_items += compacted_history.raw_items;
                      normal_rag_raw_tokens += compacted_history.raw_tokens;
                      normal_rag_compacted_original_tokens
                          += compacted_history.compacted_original_tokens;
                      normal_rag_compacted_summary_tokens
                          += compacted_history.compacted_summary_tokens;
                      normal_rag_compaction_ms_total += compaction_ms;
                      normal_rag_retrieval_ms_total += rag_retrieval_ms;
                      ++normal_rag_probe_count;

                      nlohmann::json probe;
                      probe["event_index"] = checkpoint_event_count;
                      probe["query"] = {
                        { "timestamp", msg.timestamp },
                        { "source_id", source },
                        { "modality", "text" },
                        { "tokens", EstimateTokens (msg.text) },
                      };
                      probe["cortext_retrieved_memory_ids"]
                          = ContextMemoryIdsJson (
                              probe_ctx.retrieved_memory);
                      probe["cortext_working_memory_ids"]
                          = ContextMemoryIdsJson (probe_ctx.working_memory);
                      probe["cortext_frozen_retrieved_memory"]
                          = MemoryPacketJson (probe_ctx.retrieved_memory);
                      probe["cortext_frozen_working_memory"]
                          = MemoryPacketJson (probe_ctx.working_memory);
                      probe["cortext_frozen_packet_policy"]
                          = "probe_time_hydrated_context_snapshot";
                      probe["cortext_retrieved_items"]
                          = probe_ctx.retrieved_memory.size ();
                      probe["cortext_working_items"]
                          = probe_ctx.working_memory.size ();
                      probe["cortext_retrieved_tokens"]
                          = EstimateMemoryPacketTokens (
                              probe_ctx.retrieved_memory);
                      probe["cortext_working_tokens"]
                          = EstimateMemoryPacketTokens (
                              probe_ctx.working_memory);
                      probe["cortext_context_tokens"]
                          = probe["cortext_retrieved_tokens"].get<int> ()
                            + probe["cortext_working_tokens"].get<int> ();
                      probe["cortext_latency_ms"] = cortext_latency_ms;
                      probe["cortext_probe_policy"]
                          = "checkpoint_ephemeral_future_turn";
                      probe["cortext_encode_ms"] = probe_ctx.encode_ms;
                      probe["cortext_process_ms"] = probe_ctx.process_ms;
                      probe["cortext_hydrate_ms"] = probe_ctx.hydrate_ms;
                      probe["cortext_total_ms"] = probe_ctx.total_ms;
                      probe["cortext_operation_ms"]
                          = OperationTimingsJson (
                              probe_ctx.output.operation_ms);
                      probe["cortext_top_operation_ms"]
                          = TopOperationTimingsJson (
                              probe_ctx.output.operation_ms, 12);
                      probe["cortext_retrieval_trace"]
                          = RetrievalTraceJson ();
                      probe["normal_rag_compaction_latency_ms"]
                          = compaction_ms;
                      probe["normal_rag_retrieval_latency_ms"]
                          = rag_retrieval_ms;
                      probe["normal_rag_total_latency_ms"]
                          = compaction_ms + rag_retrieval_ms;
                      probe["rolling_history"]
                          = DocsPacketJson (compacted_history.raw_docs);
                      probe["rolling_history_tokens"]
                          = compacted_history.raw_tokens;
                      probe["normal_rag_context_tokens"] = RagContextTokens (
                          compacted_history, vector_rag.docs);
                      probe["normal_rag_compaction_events"]
                          = compacted_history.compaction_events;
                      probe["normal_rag_compacted_history_items"]
                          = compacted_history.compacted_items;
                      probe["normal_rag_compacted_summary_items"]
                          = compacted_history.compacted_summary_items;
                      probe["normal_rag_raw_history_items"]
                          = compacted_history.raw_items;
                      probe["normal_rag_raw_history_tokens"]
                          = compacted_history.raw_tokens;
                      probe["normal_rag_compacted_original_tokens"]
                          = compacted_history.compacted_original_tokens;
                      probe["normal_rag_compacted_summary_tokens"]
                          = compacted_history.compacted_summary_tokens;
                      probe["normal_rag_compacted_summary"]
                          = compacted_history.summary_text;
                      probe["normal_rag_compaction_summary_policy"]
                          = "deterministic_extractive_prior_chat";
                      probe["normal_rag_active_history_tokens"]
                          = compacted_history.raw_tokens
                            + compacted_history.compacted_summary_tokens;
                      probe["rag_top_k"] = DocsPacketJson (vector_rag.docs);
                      probe["rag_top_k_indices"]
                          = DocIndicesJson (vector_rag.docs);
                      probe["rag_top_k_additional"]
                          = DocsPacketJson (AdditionalRagDocs (
                              vector_rag.docs, compacted_history.raw_docs));
                      probe["normal_rag_vector_query_rows"]
                          = vector_rag.query_rows;
                      probe["normal_rag_vector_query_embedding_bytes"]
                          = vector_rag.query_embedding_bytes;
                      probe["normal_rag_vector_search_k"]
                          = vector_rag.vector_search_k;
                      probe["normal_rag_vector_candidate_rows"]
                          = vector_rag.candidate_rows;
                      probe["normal_rag_vector_prior_chat_rows"]
                          = vector_rag.prior_chat_rows;
                      probe["normal_rag_vector_best_distance"]
                          = vector_rag.best_distance;
                      probe["full_history_items"]
                          = checkpoint_prior_text_docs.size ();
                      probe["full_history_tokens"]
                          = DocPacketTokens (checkpoint_prior_text_docs);

                      AppendProbeStream (probe_stream_path, probe);
                      checkpoint_probes.push_back (std::move (probe));
                      ++checkpoint_future_turns_probed;
                      ++checkpoint_probes_by_day[query_day];
                      max_query_message_index
                          = std::max (max_query_message_index, msg.index);
                    }
                  ++skipped_future_queries;
                  if (stop_checkpoint_scan)
                    {
                      break;
                    }
                }

              checkpoint_prior_docs.push_back (doc);
              checkpoint_prior_text_docs.push_back (doc);
              ++checkpoint_event_count;
              if (checkpoint_probes.size ()
                  >= static_cast<std::size_t> (
                      std::max (0, cfg.checkpoint_query_count)))
                break;
            }

          engine->Flush ();
          const auto checkpoint_ended = std::chrono::steady_clock::now ();
          const auto wall_ms
              = std::chrono::duration_cast<std::chrono::milliseconds> (
                    checkpoint_ended - checkpoint_started)
                    .count ();
          const int summary_message_count
              = checkpoint_probes.empty ()
                    ? static_cast<int> (messages.size ())
                    : std::min (parsed_messages, max_query_message_index + 1);

          nlohmann::json out;
          out["input_dir"] = cfg.input_dir.string ();
          out["db_path"] = cfg.db_path.string ();
          out["sqlite_profile"] = sqlite_profile;
          out["append"] = cfg.append;
          out["checkpoint_eval_only"] = true;
          out["checkpoint_after_timestamp"]
              = cfg.checkpoint_after_timestamp;
          out["checkpoint_query_count"] = cfg.checkpoint_query_count;
          out["checkpoint_query_stride"] = cfg.checkpoint_query_stride;
          out["checkpoint_query_days"] = cfg.checkpoint_query_days;
          out["checkpoint_queries_per_day"] = cfg.checkpoint_queries_per_day;
          out["effective_checkpoint_queries_per_day"]
              = effective_checkpoint_queries_per_day;
          out["checkpoint_future_turns_replayed"]
              = checkpoint_future_turns_replayed;
          out["checkpoint_future_turns_probed"]
              = checkpoint_future_turns_probed;
          out["checkpoint_probe_day_count"]
              = checkpoint_probes_by_day.size ();
          nlohmann::json checkpoint_probe_days = nlohmann::json::object ();
          for (const auto &row : checkpoint_probes_by_day)
            {
              checkpoint_probe_days[std::to_string (row.first)] = row.second;
            }
          out["checkpoint_probes_by_day"] = checkpoint_probe_days;
          out["parsed_transcript_messages"] = parsed_messages;
          out["skipped_transcript_messages"] = skipped_messages;
          out["processed_text_messages"] = summary_message_count;
          out["normal_rag_compaction"] = {
            { "probe_events", checkpoint_probes.size () },
            { "compaction_events", normal_rag_compaction_events },
            { "compacted_items", normal_rag_compacted_items },
            { "raw_items", normal_rag_raw_items },
            { "raw_tokens", normal_rag_raw_tokens },
            { "compacted_original_tokens",
              normal_rag_compacted_original_tokens },
            { "compacted_summary_tokens",
              normal_rag_compacted_summary_tokens },
            { "probe_count", normal_rag_probe_count },
            { "mean_compaction_latency_ms",
              normal_rag_probe_count > 0
                  ? normal_rag_compaction_ms_total
                        / static_cast<double> (normal_rag_probe_count)
                  : 0.0 },
            { "mean_retrieval_latency_ms",
              normal_rag_probe_count > 0
                  ? normal_rag_retrieval_ms_total
                        / static_cast<double> (normal_rag_probe_count)
                  : 0.0 },
            { "mean_total_latency_ms",
              normal_rag_probe_count > 0
                  ? (normal_rag_compaction_ms_total
                     + normal_rag_retrieval_ms_total)
                        / static_cast<double> (normal_rag_probe_count)
                  : 0.0 },
          };
          out["warmup_events"] = cfg.warmup_events;
          out["probe_stride"] = cfg.probe_stride;
          out["probe_count"] = checkpoint_probes.size ();
          out["probes"] = checkpoint_probes;
          out["probe_stream_path"] = probe_stream_path.string ();
          out["probe_stream_policy"]
              = "native probe rows appended as compact JSONL immediately "
                "after each probe is constructed";
          out["rag_top_k"] = cfg.rag_top_k;
          out["normal_rag_retrieval"] = "raw_chat_vector";
          out["normal_rag_baseline_modality"] = "text_only";
          out["normal_rag_vector_query_encoder"]
              = rag_encoder_selection.backend_name;
          out["normal_rag_vector_query_encoder_path"]
              = rag_encoder_selection.resolved_path.string ();
          out["normal_rag_vector_candidate_k"] = cfg.rag_top_k;
          out["normal_rag_vector_final_k"] = cfg.rag_top_k;
          out["normal_rag_vector_search_multiplier"]
              = kNormalRagVectorSearchMultiplier;
          out["normal_rag_vector_search_k_policy"]
              = "min(prior_text_rows, max(rag_top_k, rag_top_k * "
                "normal_rag_vector_search_multiplier)) before dedupe";
          out["normal_rag_vector_candidate_k_policy"]
              = "final unique text RAG packet cap after vector-search fanout";
          out["normal_rag_context_token_policy"]
              = "text rolling chat after compaction plus unique text vector "
                "RAG hits outside the active rolling window";
          out["normal_rag_compaction_summary_policy"]
              = "deterministic_extractive_prior_chat";
          out["active_history_token_budget"]
              = cfg.active_history_token_budget;
          out["knobs"] = {
            { "focus", cfg.focus },
            { "sensitivity", cfg.sensitivity },
            { "stability", cfg.stability },
          };
          out["media_candidates_found"] = media.size ();
          out["media_attempted"] = checkpoint_media_attempted;
          out["media_processed"]
              = SqlCount (cfg.db_path,
                          "SELECT COUNT(*) FROM signals "
                          "WHERE modality IN ('audio', 'image')");
          out["media_failures"] = 0;
          out["image_processed"]
              = SqlCount (cfg.db_path,
                          "SELECT COUNT(*) FROM signals "
                          "WHERE modality = 'image'");
          out["video_processed"] = 0;
          out["audio_processed"]
              = SqlCount (cfg.db_path,
                          "SELECT COUNT(*) FROM signals "
                          "WHERE modality = 'audio'");
          out["consolidation_runs"]
              = SqlCount (cfg.db_path,
                          "SELECT COUNT(*) FROM signals "
                          "WHERE source_id = 'cortext/maintenance'");
          out["consolidation_ms_total"] = 0.0;
          out["wall_ms_excluding_consolidation"] = wall_ms;
          out["daily_consolidation"] = cfg.daily_consolidation;
          out["daily_consolidation_hour_local"]
              = cfg.daily_consolidation_hour;
          out["daily_consolidation_policy"]
              = "source_time_sleep_checkpoint";
          out["replay_timezone"] = cfg.replay_timezone.empty ()
                                       ? "process_default"
                                       : cfg.replay_timezone;
          out["source_id_policy"]
              = "User and Contact are opaque conversation provenance source "
                "IDs; media is not encoded into source_id";
          out["timeline_policy"]
              = "checkpoint eval opens an existing Cortext replay database "
                "and scores future text turns with public timestamped "
                "Retention::Ephemeral ingress; day-spread checkpoint eval "
                "also replays intervening future text turns ephemerally so "
                "Cortext WM advances like a chat session; no custom retrieval "
                "or scoring is used";
          out["media_timestamp_policy"]
              = "checkpoint database was produced by timestamped replay "
                "media ingress";
          out["consolidation_timestamp_policy"]
              = "checkpoint database consolidation rows use source event "
                "timestamps; daily replay consolidation uses the configured "
                "local sleep checkpoint";
          out["wall_ms"] = wall_ms;
          out["peak_rss_mb"] = PeakRssMb ();
          out["mean_encode_ms"] = 0.0;
          out["mean_process_ms"] = 0.0;
          out["mean_hydrate_ms"] = 0.0;
          out["mean_total_ms"] = 0.0;
          out["db_memory_count"] = SqlCount (cfg.db_path,
                                             "SELECT COUNT(*) FROM memories");
          out["db_label_memory_count"] = SqlCount (
              cfg.db_path, "SELECT COUNT(*) FROM memories WHERE kind = 'LABEL'");
          out["db_long_term_memory_count"]
              = SqlCount (cfg.db_path,
                          "SELECT COUNT(*) FROM memories "
                          "WHERE kind = 'LONG_TERM'");
          out["db_working_memory_count"]
              = SqlCount (cfg.db_path,
                          "SELECT COUNT(*) FROM memories "
                          "WHERE kind = 'WORKING'");
          out["db_memory_rows_with_embeddings"]
              = SqlCount (cfg.db_path,
                          "SELECT COUNT(*) FROM memories "
                          "WHERE embedding_id IS NOT NULL");
          out["db_association_count"]
              = SqlCount (cfg.db_path, "SELECT COUNT(*) FROM associations");
          out["behavior_note"]
              = "Checkpoint eval uses native Cortext retrieval from an "
                "existing replay DB plus ephemeral future-turn probes. It is "
                "not a corpus replay and does not ingest future media.";
          out["privacy_note"]
              = "Private benchmark artifact: probe rows include deterministic "
                "extractive compacted-history summaries for the RAG baseline; "
                "the public release report excludes message text and media "
                "content.";

          WriteJsonArtifact (cfg.output_path, out);
          std::cout << DumpJsonArtifact (out) << "\n";
          return 0;
        }

      if (cfg.profile_probes_only)
        {
          nlohmann::json profile_probes = nlohmann::json::array ();
          std::vector<EventDoc> profile_prior_docs;
          const auto profile_started = std::chrono::steady_clock::now ();
          int profile_event_count = 0;
          for (const auto &event : timeline)
            {
              if (event.is_media)
                {
                  ++profile_event_count;
                  continue;
                }
              const auto &msg = messages[event.index];
              const std::string source = SourceIdForMessage (msg);
              const EventDoc doc {
                profile_event_count,
                msg.timestamp,
                source,
                "text",
                msg.text,
              };
              if (cfg.probe_stride > 0
                  && profile_event_count >= cfg.warmup_events
                  && profile_event_count % cfg.probe_stride == 0)
                {
                  const auto probe_started = std::chrono::steady_clock::now ();
                  auto probe_ctx = engine->ProcessTextAt (
                      msg.text, source, msg.timestamp);
                  const auto probe_ended = std::chrono::steady_clock::now ();
                  nlohmann::json probe;
                  probe["event_index"] = profile_event_count;
                  probe["query"] = {
                    { "source_id", source },
                    { "modality", "text" },
                    { "tokens", EstimateTokens (msg.text) },
                  };
                  probe["cortext_latency_ms"]
                      = std::chrono::duration<double, std::milli> (
                            probe_ended - probe_started)
                            .count ();
                  probe["cortext_encode_ms"] = probe_ctx.encode_ms;
                  probe["cortext_process_ms"] = probe_ctx.process_ms;
                  probe["cortext_hydrate_ms"] = probe_ctx.hydrate_ms;
                  probe["cortext_total_ms"] = probe_ctx.total_ms;
                  probe["cortext_retrieved_items"]
                      = probe_ctx.retrieved_memory.size ();
                  probe["cortext_working_items"]
                      = probe_ctx.working_memory.size ();
                  probe["cortext_retrieved_memory_ids"]
                      = ContextMemoryIdsJson (probe_ctx.retrieved_memory);
                  probe["cortext_frozen_retrieved_memory"]
                      = MemoryPacketJson (probe_ctx.retrieved_memory);
                  probe["cortext_frozen_working_memory"]
                      = MemoryPacketJson (probe_ctx.working_memory);
                  probe["cortext_frozen_packet_policy"]
                      = "probe_time_hydrated_context_snapshot";
                  probe["cortext_probe_policy"]
                      = "single_durable_chat_turn_profile_probe";
                  probe["cortext_working_memory_ids"]
                      = ContextMemoryIdsJson (probe_ctx.working_memory);
                  probe["cortext_retrieved_tokens"]
                      = EstimateMemoryPacketTokens (
                          probe_ctx.retrieved_memory);
                  probe["cortext_working_tokens"]
                      = EstimateMemoryPacketTokens (
                          probe_ctx.working_memory);
                  probe["cortext_context_tokens"]
                      = probe["cortext_retrieved_tokens"].get<int> ()
                        + probe["cortext_working_tokens"].get<int> ();
                  probe["cortext_operation_ms"]
                      = OperationTimingsJson (probe_ctx.output.operation_ms);
                  probe["cortext_top_operation_ms"]
                      = TopOperationTimingsJson (probe_ctx.output.operation_ms,
                                                 12);
                  probe["cortext_retrieval_trace"] = RetrievalTraceJson ();
                  std::vector<float> rag_query_embedding;
                  rag_encoder_selection.encoder->EncodeText (
                      msg.text, rag_query_embedding);
                  rag_query_embedding = RetrievalEmbeddingViewForBenchmark (
                      rag_query_embedding);
                  const auto vector_rag = BuildVectorRagPacket (
                      *vector_rag_store, profile_prior_docs,
                      rag_query_embedding, msg.timestamp, cfg.rag_top_k);
                  probe["rag_top_k"] = DocsPacketJson (vector_rag.docs);
                  probe["rag_top_k_indices"]
                      = DocIndicesJson (vector_rag.docs);
                  const auto compacted_history = CompactRollingHistoryDocs (
                      profile_prior_docs, cfg.active_history_token_budget);
                  probe["rolling_history"]
                      = DocsPacketJson (compacted_history.raw_docs);
                  probe["rolling_history_tokens"]
                      = compacted_history.raw_tokens;
                  probe["normal_rag_active_history_tokens"]
                      = compacted_history.raw_tokens
                        + compacted_history.compacted_summary_tokens;
                  probe["normal_rag_context_tokens"] = RagContextTokens (
                      compacted_history, vector_rag.docs);
                  probe["normal_rag_vector_query_rows"]
                      = vector_rag.query_rows;
                  probe["normal_rag_vector_query_embedding_bytes"]
                      = vector_rag.query_embedding_bytes;
                  probe["normal_rag_vector_search_k"]
                      = vector_rag.vector_search_k;
                  probe["normal_rag_vector_candidate_rows"]
                      = vector_rag.candidate_rows;
                  probe["normal_rag_vector_prior_chat_rows"]
                      = vector_rag.prior_chat_rows;
                  probe["normal_rag_vector_best_distance"]
                      = vector_rag.best_distance;
                  probe["full_history_items"] = profile_prior_docs.size ();
                  probe["full_history_tokens"]
                      = DocPacketTokens (profile_prior_docs);
                  AppendProbeStream (probe_stream_path, probe);
                  profile_probes.push_back (std::move (probe));
                }
              profile_prior_docs.push_back (doc);
              ++profile_event_count;
            }

          const auto profile_ended = std::chrono::steady_clock::now ();
          nlohmann::json out;
          out["input_dir"] = cfg.input_dir.string ();
          out["db_path"] = cfg.db_path.string ();
          out["sqlite_profile"] = sqlite_profile;
          out["profile_probes_only"] = true;
          out["append"] = cfg.append;
          out["warmup_events"] = cfg.warmup_events;
          out["probe_stride"] = cfg.probe_stride;
          out["probe_count"] = profile_probes.size ();
          out["probes"] = profile_probes;
          out["probe_stream_path"] = probe_stream_path.string ();
          out["probe_stream_policy"]
              = "native probe rows appended as compact JSONL immediately "
                "after each probe is constructed";
          out["rag_top_k"] = cfg.rag_top_k;
          out["normal_rag_retrieval"] = "raw_chat_vector";
          out["normal_rag_baseline_modality"] = "text_only";
          out["normal_rag_vector_query_encoder"]
              = rag_encoder_selection.backend_name;
          out["normal_rag_vector_query_encoder_path"]
              = rag_encoder_selection.resolved_path.string ();
          out["normal_rag_vector_candidate_k"] = cfg.rag_top_k;
          out["normal_rag_vector_final_k"] = cfg.rag_top_k;
          out["normal_rag_vector_search_multiplier"]
              = kNormalRagVectorSearchMultiplier;
          out["normal_rag_vector_search_k_policy"]
              = "min(prior_text_rows, max(rag_top_k, rag_top_k * "
                "normal_rag_vector_search_multiplier)) before dedupe";
          out["normal_rag_vector_candidate_k_policy"]
              = "final unique text RAG packet cap after vector-search fanout";
          out["normal_rag_context_token_policy"]
              = "text rolling chat after compaction plus unique text vector "
                "RAG hits outside the active rolling window";
          out["active_history_token_budget"]
              = cfg.active_history_token_budget;
          out["processed_text_messages"] = messages.size ();
          out["media_processed"] = 0;
          out["media_attempted"] = 0;
          out["knobs"] = {
            { "focus", cfg.focus },
            { "sensitivity", cfg.sensitivity },
            { "stability", cfg.stability },
          };
          out["wall_ms"]
              = std::chrono::duration_cast<std::chrono::milliseconds> (
                    profile_ended - profile_started)
                    .count ();
          out["peak_rss_mb"] = PeakRssMb ();
          WriteJsonArtifact (cfg.output_path, out);
          std::cout << DumpJsonArtifact (out) << "\n";
          return 0;
        }

      int current_sleep_bucket = -1;
      auto started = std::chrono::steady_clock::now ();
      int event_count = 0;
      std::uint64_t last_processed_timestamp = 0;
      bool final_window_consolidated = false;
      std::size_t replay_start_index = 0;
      if (resume.enabled)
        {
          while (replay_start_index < timeline.size ())
            {
              const auto &event = timeline[replay_start_index];
              const bool prior_event = event.timestamp <= resume.after_timestamp;
              if (!prior_event)
                break;

              if (!event.is_media)
                {
                  const auto &msg = messages[event.index];
                  EventDoc doc {
                    resume.rebuilt_event_count,
                    msg.timestamp,
                    SourceIdForMessage (msg),
                    "text",
                    msg.text,
                  };
                  prior_docs.push_back (doc);
                  prior_text_docs.push_back (std::move (doc));
                  ++resume.rebuilt_text_count;
                }
              else
                {
                  const auto &item = media[event.index];
                  EventDoc media_doc {
                    resume.rebuilt_event_count,
                    item.timestamp,
                    MediaSourceId (item),
                    item.kind == "video" ? "image" : item.kind,
                    "[" + item.kind
                        + " source blob: " + item.path.filename ().string ()
                        + "]",
                  };
                  prior_docs.push_back (std::move (media_doc));
                }
              ++resume.rebuilt_event_count;
              ++resume.prior_timeline_events;
              ++replay_start_index;
            }

          resume.event_count_offset = resume.rebuilt_event_count;
          resume.text_count_offset = resume.rebuilt_text_count;
          resume.media_count_offset
              = std::max (0,
                          resume.event_count_offset - resume.text_count_offset);

          processed_text = resume.text_count_offset;
          media_processed = resume.media_count_offset;
          media_attempted = resume.media_count_offset;
          event_count = resume.event_count_offset;
          current_sleep_bucket = LocalSleepBucket (
              resume.after_timestamp, cfg.daily_consolidation_hour);
          last_processed_timestamp = resume.after_timestamp;

          nlohmann::json resume_log {
            { "event_count_offset", resume.event_count_offset },
            { "text_count_offset", resume.text_count_offset },
            { "media_count_offset", resume.media_count_offset },
            { "probe_count_offset", resume.probe_count_offset },
            { "db_signal_count", resume.db_signal_count },
            { "db_text_signal_count", resume.db_text_signal_count },
            { "after_timestamp", resume.after_timestamp },
            { "rebuilt_event_count", resume.rebuilt_event_count },
            { "rebuilt_text_count", resume.rebuilt_text_count },
            { "prior_timeline_events", resume.prior_timeline_events },
            { "replay_start_timeline_index", replay_start_index },
          };
          std::cerr << "CHAT_REPLAY_RESUME "
                    << resume_log.dump (-1, ' ', false,
                                        nlohmann::json::error_handler_t::replace)
                    << "\n";
        }
      auto record_consolidation_event = [&] (const char *trigger,
                                             std::uint64_t timestamp,
                                             double elapsed_ms,
                                             const cortext::Cortext::Context
                                                 &ctx) {
        consolidation_events.push_back ({
          { "trigger", trigger },
          { "timestamp", timestamp },
          { "local_day_bucket", LocalDayBucket (timestamp) },
          { "local_sleep_bucket",
            LocalSleepBucket (timestamp, cfg.daily_consolidation_hour) },
          { "sleep_hour_local", cfg.daily_consolidation_hour },
          { "elapsed_ms", elapsed_ms },
          { "processed_text_messages", processed_text },
          { "media_processed", media_processed },
          { "encode_ms", ctx.encode_ms },
          { "process_ms", ctx.process_ms },
          { "hydrate_ms", ctx.hydrate_ms },
          { "total_ms", ctx.total_ms },
          { "operation_ms", OperationTimingsJson (ctx.output.operation_ms) },
          { "top_operation_ms",
            TopOperationTimingsJson (ctx.output.operation_ms, 16) },
        });
      };
      auto maybe_run_daily_consolidation = [&] (std::uint64_t timestamp) {
        const int sleep_bucket
            = LocalSleepBucket (timestamp, cfg.daily_consolidation_hour);
        if (cfg.daily_consolidation && current_sleep_bucket >= 0
            && sleep_bucket != current_sleep_bucket
            && (processed_text > 0 || media_processed > 0))
          {
            const std::uint64_t checkpoint_timestamp
                = LocalSleepCheckpointTimestamp (
                    timestamp, cfg.daily_consolidation_hour);
            const auto consolidation_started
                = std::chrono::steady_clock::now ();
            auto consolidation_ctx
                = cortext::internal::ReplayIngress::ConsolidateAt (
                *engine, checkpoint_timestamp);
            const auto consolidation_ended = std::chrono::steady_clock::now ();
            consolidation_ms_total
                += std::chrono::duration<double, std::milli> (
                       consolidation_ended - consolidation_started)
                       .count ();
            record_consolidation_event (
                "daily_sleep_checkpoint", checkpoint_timestamp,
                std::chrono::duration<double, std::milli> (
                    consolidation_ended - consolidation_started)
                    .count (),
                consolidation_ctx);
            ++consolidation_runs;
          }
        current_sleep_bucket = sleep_bucket;
      };
      auto maybe_log_progress = [&] (const EventDoc &doc,
                                     const cortext::Cortext::Context &ctx) {
        if (cfg.progress_stride <= 0)
          return;
        const int cumulative_events = event_count + 1;
        if (cumulative_events <= 0
            || cumulative_events % cfg.progress_stride != 0)
          return;
        const int processed_text_this_run
            = std::max (0, processed_text - resume.text_count_offset);
        nlohmann::json progress {
          { "event_index", event_count },
          { "cumulative_events", cumulative_events },
          { "timestamp", doc.timestamp },
          { "modality", doc.modality },
          { "resume_from_existing", resume.enabled },
          { "processed_text_messages", processed_text },
          { "processed_text_messages_this_run", processed_text_this_run },
          { "media_processed", media_processed },
          { "media_processed_this_run",
            std::max (0, media_processed - resume.media_count_offset) },
          { "last_total_ms", ctx.total_ms },
          { "last_process_ms", ctx.process_ms },
          { "last_hydrate_ms", ctx.hydrate_ms },
          { "last_top_operation_ms",
            TopOperationTimingsJson (ctx.output.operation_ms, 8) },
          { "mean_total_ms_this_run",
            processed_text_this_run > 0
                ? total_ms_total / static_cast<double> (processed_text_this_run)
                : 0.0 },
          { "mean_process_ms_this_run",
            processed_text_this_run > 0
                ? process_ms_total
                      / static_cast<double> (processed_text_this_run)
                : 0.0 },
          { "mean_hydrate_ms_this_run",
            processed_text_this_run > 0
                ? hydrate_ms_total
                      / static_cast<double> (processed_text_this_run)
                : 0.0 },
          { "probe_stream_rows_before_resume", resume.probe_count_offset },
          { "probe_rows_this_run", probes.size () },
        };
        std::cerr << "CHAT_REPLAY_PROGRESS "
                  << progress.dump (-1, ' ', false,
                                    nlohmann::json::error_handler_t::replace)
                  << "\n";
      };

      for (std::size_t timeline_index = replay_start_index;
           timeline_index < timeline.size (); ++timeline_index)
        {
          const auto &event = timeline[timeline_index];
          if (!event.is_media)
            {
              const auto &msg = messages[event.index];
              maybe_run_daily_consolidation (msg.timestamp);
              const std::string source = SourceIdForMessage (msg);
              const EventDoc doc {
                event_count,
                msg.timestamp,
                source,
                "text",
                msg.text,
              };
              const bool run_probe
                  = cfg.probe_stride > 0 && event_count >= cfg.warmup_events
                    && event_count % cfg.probe_stride == 0;
              const auto cortext_started = std::chrono::steady_clock::now ();
              auto ctx = cortext::internal::ReplayIngress::ProcessTextAt (
                  *engine, msg.text, source, msg.timestamp,
                  cortext::Retention::Durable,
                  run_probe || cfg.behavior_oracle);
              const auto cortext_ended = std::chrono::steady_clock::now ();
              if (run_probe)
                {
                  const auto &probe_ctx = ctx;
                  const auto compaction_started
                      = std::chrono::steady_clock::now ();
                  const auto compacted_history = CompactRollingHistoryDocs (
                      prior_text_docs, cfg.active_history_token_budget);
                  const auto compaction_ended
                      = std::chrono::steady_clock::now ();
                  const double compaction_ms
                      = std::chrono::duration<double, std::milli> (
                            compaction_ended - compaction_started)
                            .count ();
                  const auto &rolling = compacted_history.raw_docs;
                  normal_rag_compaction_events
                      += compacted_history.compaction_events;
                  normal_rag_compacted_items
                      += compacted_history.compacted_items;
                  normal_rag_raw_items += compacted_history.raw_items;
                  normal_rag_raw_tokens += compacted_history.raw_tokens;
                  normal_rag_compacted_original_tokens
                      += compacted_history.compacted_original_tokens;
                  normal_rag_compacted_summary_tokens
                      += compacted_history.compacted_summary_tokens;
                  const auto rag_started = std::chrono::steady_clock::now ();
                  std::vector<float> rag_query_embedding;
                  rag_encoder_selection.encoder->EncodeText (
                      msg.text, rag_query_embedding);
                  rag_query_embedding = RetrievalEmbeddingViewForBenchmark (
                      rag_query_embedding);
                  const auto vector_rag = BuildVectorRagPacket (
                      *vector_rag_store, prior_text_docs, rag_query_embedding,
                      msg.timestamp, cfg.rag_top_k);
                  const auto rag_ended = std::chrono::steady_clock::now ();
                  const double rag_retrieval_ms
                      = std::chrono::duration<double, std::milli> (
                            rag_ended - rag_started)
                            .count ();
                  normal_rag_compaction_ms_total += compaction_ms;
                  normal_rag_retrieval_ms_total += rag_retrieval_ms;
                  ++normal_rag_probe_count;
                  nlohmann::json probe;
                  probe["event_index"] = event_count;
                  probe["query"] = {
                    { "timestamp", msg.timestamp },
                    { "source_id", source },
                    { "modality", "text" },
                    { "tokens", EstimateTokens (msg.text) },
                  };
                  probe["cortext_retrieved_memory_ids"]
                      = ContextMemoryIdsJson (probe_ctx.retrieved_memory);
                  probe["cortext_working_memory_ids"]
                      = ContextMemoryIdsJson (probe_ctx.working_memory);
                  probe["cortext_frozen_retrieved_memory"]
                      = MemoryPacketJson (probe_ctx.retrieved_memory);
                  probe["cortext_frozen_working_memory"]
                      = MemoryPacketJson (probe_ctx.working_memory);
                  probe["cortext_frozen_packet_policy"]
                      = "probe_time_hydrated_context_snapshot";
                  probe["cortext_retrieved_items"]
                      = probe_ctx.retrieved_memory.size ();
                  probe["cortext_working_items"]
                      = probe_ctx.working_memory.size ();
                  probe["cortext_retrieved_tokens"]
                      = EstimateMemoryPacketTokens (
                          probe_ctx.retrieved_memory);
                  probe["cortext_working_tokens"]
                      = EstimateMemoryPacketTokens (
                          probe_ctx.working_memory);
                  probe["cortext_context_tokens"]
                      = probe["cortext_retrieved_tokens"].get<int> ()
                        + probe["cortext_working_tokens"].get<int> ();
                  probe["cortext_latency_ms"]
                      = std::chrono::duration<double, std::milli> (
                            cortext_ended - cortext_started)
                            .count ();
                  probe["cortext_probe_policy"]
                      = "single_durable_chat_turn_reused_for_probe_and_ingest";
                  probe["cortext_encode_ms"] = probe_ctx.encode_ms;
                  probe["cortext_process_ms"] = probe_ctx.process_ms;
                  probe["cortext_hydrate_ms"] = probe_ctx.hydrate_ms;
                  probe["cortext_total_ms"] = probe_ctx.total_ms;
                  probe["cortext_operation_ms"]
                      = OperationTimingsJson (probe_ctx.output.operation_ms);
                  probe["cortext_top_operation_ms"]
                      = TopOperationTimingsJson (probe_ctx.output.operation_ms,
                                                 12);
                  probe["cortext_retrieval_trace"] = RetrievalTraceJson ();
                  probe["normal_rag_compaction_latency_ms"] = compaction_ms;
                  probe["normal_rag_retrieval_latency_ms"] = rag_retrieval_ms;
                  probe["normal_rag_total_latency_ms"]
                      = compaction_ms + rag_retrieval_ms;
                  probe["rolling_history"] = DocsPacketJson (rolling);
                  probe["rolling_history_tokens"] = compacted_history.raw_tokens;
                  probe["normal_rag_context_tokens"] = RagContextTokens (
                      compacted_history, vector_rag.docs);
                  probe["normal_rag_compaction_events"]
                      = compacted_history.compaction_events;
                  probe["normal_rag_compacted_history_items"]
                      = compacted_history.compacted_items;
                  probe["normal_rag_compacted_summary_items"]
                      = compacted_history.compacted_summary_items;
                  probe["normal_rag_raw_history_items"]
                      = compacted_history.raw_items;
                  probe["normal_rag_raw_history_tokens"]
                      = compacted_history.raw_tokens;
                  probe["normal_rag_compacted_original_tokens"]
                      = compacted_history.compacted_original_tokens;
                  probe["normal_rag_compacted_summary_tokens"]
                      = compacted_history.compacted_summary_tokens;
                  probe["normal_rag_compacted_summary"]
                      = compacted_history.summary_text;
                  probe["normal_rag_compaction_summary_policy"]
                      = "deterministic_extractive_prior_chat";
                  probe["normal_rag_active_history_tokens"]
                      = compacted_history.raw_tokens
                        + compacted_history.compacted_summary_tokens;
                  probe["rag_top_k"] = DocsPacketJson (vector_rag.docs);
                  probe["rag_top_k_indices"]
                      = DocIndicesJson (vector_rag.docs);
                  probe["rag_top_k_additional"]
                      = DocsPacketJson (AdditionalRagDocs (
                          vector_rag.docs, compacted_history.raw_docs));
                  probe["normal_rag_vector_query_rows"]
                      = vector_rag.query_rows;
                  probe["normal_rag_vector_query_embedding_bytes"]
                      = vector_rag.query_embedding_bytes;
                  probe["normal_rag_vector_search_k"]
                      = vector_rag.vector_search_k;
                  probe["normal_rag_vector_candidate_rows"]
                      = vector_rag.candidate_rows;
                  probe["normal_rag_vector_prior_chat_rows"]
                      = vector_rag.prior_chat_rows;
                  probe["normal_rag_vector_best_distance"]
                      = vector_rag.best_distance;
                  probe["full_history_items"] = prior_text_docs.size ();
                  probe["full_history_tokens"]
                      = DocPacketTokens (prior_text_docs);
                  AppendProbeStream (probe_stream_path, probe);
                  probes.push_back (std::move (probe));
                }
              working_set_curve.push_back (
                  WorkingSetCurveRow (event_count, doc, ctx,
                                      cfg.full_operation_ms));
              ++processed_text;
              encode_ms_total += ctx.encode_ms;
              process_ms_total += ctx.process_ms;
              hydrate_ms_total += ctx.hydrate_ms;
              total_ms_total += ctx.total_ms;
              if (ctx.output.stored_memory_id || ctx.output.stored_embedding_id)
                ++text_write_events;
              if (!ctx.retrieved_memory.empty ())
                ++retrieval_events;
              retrieval_items += static_cast<int> (ctx.retrieved_memory.size ());
              maybe_log_progress (doc, ctx);

              if (cfg.consolidate_every > 0
                  && processed_text % cfg.consolidate_every == 0)
                {
                  const auto consolidation_started
                      = std::chrono::steady_clock::now ();
                  auto consolidation_ctx
                      = cortext::internal::ReplayIngress::ConsolidateAt (
                      *engine, msg.timestamp);
                  const auto consolidation_ended
                      = std::chrono::steady_clock::now ();
                  consolidation_ms_total
                      += std::chrono::duration<double, std::milli> (
                             consolidation_ended - consolidation_started)
                             .count ();
                  record_consolidation_event (
                      "periodic_interval", msg.timestamp,
                      std::chrono::duration<double, std::milli> (
                          consolidation_ended - consolidation_started)
                          .count (),
                      consolidation_ctx);
                  ++consolidation_runs;
                }
              last_processed_timestamp = msg.timestamp;
              prior_docs.push_back (doc);
              prior_text_docs.push_back (doc);
              ++event_count;
              continue;
            }

          if (cfg.media_limit >= 0 && media_attempted >= cfg.media_limit)
            continue;
          const auto &item = media[event.index];
          ++media_attempted;
          const std::string media_source = MediaSourceId (item);
          EventDoc media_doc {
            event_count,
            item.timestamp,
            media_source,
            item.kind == "video" ? "image" : item.kind,
            "[" + item.kind + " source blob: " + item.path.filename ().string ()
                + "]",
          };
          if (item.kind == "image" || item.kind == "video")
            {
              const fs::path raw = tmp_dir
                                   / ("frame_" + std::to_string (media_attempted)
                                      + ".rgb");
              std::string cmd
                  = "ffmpeg -y -v error -i " + ShellQuote (item.path)
                    + " -vf "
                    + ShellQuote (
                        "scale=224:224:force_original_aspect_ratio=decrease,"
                        "pad=224:224:(ow-iw)/2:(oh-ih)/2")
                    + " -frames:v 1 -f rawvideo"
                      " -pix_fmt rgb24 "
                    + ShellQuote (raw);
              std::vector<unsigned char> bytes;
              if (RunCommand (cmd) && LoadFile (raw, bytes)
                  && bytes.size () == 224ULL * 224ULL * 3ULL)
                {
                  maybe_run_daily_consolidation (item.timestamp);
                  auto ctx = cortext::internal::ReplayIngress::ProcessImageAt (
                      *engine, bytes.data (), 224, 224, 3, media_source,
                      item.timestamp);
                  working_set_curve.push_back (
                      WorkingSetCurveRow (event_count, media_doc, ctx,
                                          cfg.full_operation_ms));
                  ++media_processed;
                  item.kind == "image" ? ++image_processed : ++video_processed;
                  last_processed_timestamp = item.timestamp;
                  prior_docs.push_back (media_doc);
                  maybe_log_progress (media_doc, ctx);
                  ++event_count;
                }
              else
                {
                  ++media_failures;
                }
            }
          else if (item.kind == "audio")
            {
              const fs::path raw = tmp_dir
                                   / ("audio_" + std::to_string (media_attempted)
                                      + ".f32");
              std::string cmd = "ffmpeg -y -v error -i " + ShellQuote (item.path)
                                + " -ac 1 -ar 16000 -f f32le "
                                + ShellQuote (raw);
              std::vector<unsigned char> bytes;
              if (RunCommand (cmd) && LoadFile (raw, bytes))
                {
                  auto pcm = BytesToFloats (bytes);
                  if (!pcm.empty ())
                    {
                      maybe_run_daily_consolidation (item.timestamp);
                      auto ctx
                          = cortext::internal::ReplayIngress::ProcessAudioAt (
                              *engine, pcm.data (), pcm.size (), media_source,
                              item.timestamp);
                      working_set_curve.push_back (
                          WorkingSetCurveRow (event_count, media_doc, ctx,
                                              cfg.full_operation_ms));
                      ++media_processed;
                      ++audio_processed;
                      last_processed_timestamp = item.timestamp;
                      prior_docs.push_back (media_doc);
                      maybe_log_progress (media_doc, ctx);
                      ++event_count;
                    }
                  else
                    {
                      ++media_failures;
                    }
                }
              else
                {
                  ++media_failures;
                }
            }
        }

      if (!cfg.skip_final_consolidation
          && (processed_text > 0 || media_processed > 0)
          && last_processed_timestamp > 0)
        {
          const auto consolidation_started = std::chrono::steady_clock::now ();
          auto consolidation_ctx
              = cortext::internal::ReplayIngress::ConsolidateAt (
              *engine, last_processed_timestamp);
          const auto consolidation_ended = std::chrono::steady_clock::now ();
          consolidation_ms_total
              += std::chrono::duration<double, std::milli> (
                     consolidation_ended - consolidation_started)
                     .count ();
          record_consolidation_event (
              "final_window", last_processed_timestamp,
              std::chrono::duration<double, std::milli> (
                  consolidation_ended - consolidation_started)
                  .count (),
              consolidation_ctx);
          ++consolidation_runs;
          final_window_consolidated = true;
        }
      engine->Flush ();
      const auto ended = std::chrono::steady_clock::now ();

      const auto wall_ms
          = std::chrono::duration_cast<std::chrono::milliseconds> (ended
                                                                   - started)
                .count ();
      const int processed_text_this_run
          = std::max (0, processed_text - resume.text_count_offset);
      const int media_processed_this_run
          = std::max (0, media_processed - resume.media_count_offset);
      const int probe_count_this_run = static_cast<int> (probes.size ());

      nlohmann::json out;
      out["input_dir"] = cfg.input_dir.string ();
      out["db_path"] = cfg.db_path.string ();
      out["sqlite_profile"] = sqlite_profile;
      out["append"] = cfg.append;
      out["resume_from_existing"] = resume.enabled;
      out["resume"] = {
        { "event_count_offset", resume.event_count_offset },
        { "text_count_offset", resume.text_count_offset },
        { "media_count_offset", resume.media_count_offset },
        { "probe_count_offset", resume.probe_count_offset },
        { "db_signal_count", resume.db_signal_count },
        { "db_text_signal_count", resume.db_text_signal_count },
        { "after_timestamp", resume.after_timestamp },
        { "rebuilt_event_count", resume.rebuilt_event_count },
        { "rebuilt_text_count", resume.rebuilt_text_count },
        { "prior_timeline_events", resume.prior_timeline_events },
        { "replay_start_timeline_index", replay_start_index },
      };
      out["parsed_transcript_messages"] = parsed_messages;
      out["skipped_transcript_messages"] = skipped_messages;
      out["message_window_start_timestamp"] = message_window_start;
      out["message_window_end_timestamp"] = message_window_end;
      out["processed_text_messages"] = processed_text;
      out["processed_text_messages_this_run"] = processed_text_this_run;
      out["text_write_events"] = text_write_events;
      out["retrieval_events_during_ingest"] = retrieval_events;
      out["retrieval_items_during_ingest"] = retrieval_items;
      out["normal_rag_compaction"] = {
        { "probe_events", probes.size () },
        { "compaction_events", normal_rag_compaction_events },
        { "compacted_items", normal_rag_compacted_items },
        { "raw_items", normal_rag_raw_items },
        { "raw_tokens", normal_rag_raw_tokens },
        { "compacted_original_tokens",
          normal_rag_compacted_original_tokens },
        { "compacted_summary_tokens", normal_rag_compacted_summary_tokens },
        { "probe_count", normal_rag_probe_count },
        { "mean_compaction_latency_ms",
          normal_rag_probe_count > 0
              ? normal_rag_compaction_ms_total
                    / static_cast<double> (normal_rag_probe_count)
              : 0.0 },
        { "mean_retrieval_latency_ms",
          normal_rag_probe_count > 0
              ? normal_rag_retrieval_ms_total
                    / static_cast<double> (normal_rag_probe_count)
              : 0.0 },
        { "mean_total_latency_ms",
          normal_rag_probe_count > 0
              ? (normal_rag_compaction_ms_total
                 + normal_rag_retrieval_ms_total)
                    / static_cast<double> (normal_rag_probe_count)
              : 0.0 },
      };
      out["warmup_events"] = cfg.warmup_events;
      out["probe_stride"] = cfg.probe_stride;
      out["full_operation_ms"] = cfg.full_operation_ms;
      out["behavior_oracle"] = cfg.behavior_oracle;
      out["probe_count"] = resume.probe_count_offset + probe_count_this_run;
      out["probe_count_this_run"] = probe_count_this_run;
      out["probe_count_before_resume"] = resume.probe_count_offset;
      out["probes"] = probes;
      out["probe_stream_path"] = probe_stream_path.string ();
      out["probe_stream_policy"]
          = "native probe rows appended as compact JSONL immediately after "
            "each probe is constructed";
      out["working_set_curve_policy"]
          = "one row per successfully ingested timeline event after the "
            "durable replay ingress call; non-probe text rows may omit "
            "hydrated context packets";
      out["behavior_oracle_policy"]
          = "when enabled, hydrate the normal durable-ingest result for every "
            "text event and record all public non-timing outputs in the "
            "working-set curve; private strings, content blobs, and embeddings "
            "are represented only by size and BLAKE3 digest; do not run the "
            "comparison RAG probe";
      out["working_set_curve"] = working_set_curve;
      out["rag_top_k"] = cfg.rag_top_k;
      out["normal_rag_retrieval"] = "raw_chat_vector";
      out["normal_rag_baseline_modality"] = "text_only";
      out["normal_rag_vector_query_encoder"]
          = rag_encoder_selection.backend_name;
      out["normal_rag_vector_query_encoder_path"]
          = rag_encoder_selection.resolved_path.string ();
      out["normal_rag_vector_candidate_k"] = cfg.rag_top_k;
      out["normal_rag_vector_final_k"] = cfg.rag_top_k;
      out["normal_rag_vector_search_multiplier"]
          = kNormalRagVectorSearchMultiplier;
      out["normal_rag_vector_search_k_policy"]
          = "min(prior_text_rows, max(rag_top_k, rag_top_k * "
            "normal_rag_vector_search_multiplier)) before dedupe";
      out["normal_rag_vector_candidate_k_policy"]
          = "final unique text RAG packet cap after vector-search fanout";
      out["normal_rag_context_token_policy"]
          = "text rolling chat after compaction plus unique text vector RAG "
            "hits outside the active rolling window";
      out["normal_rag_compaction_summary_policy"]
          = "deterministic_extractive_prior_chat";
      out["active_history_token_budget"] = cfg.active_history_token_budget;
      out["knobs"] = {
        { "focus", cfg.focus },
        { "sensitivity", cfg.sensitivity },
        { "stability", cfg.stability },
      };
      out["media_candidates_found"] = media.size ();
      out["media_attempted"] = media_attempted;
      out["media_processed"] = media_processed;
      out["media_processed_this_run"] = media_processed_this_run;
      out["media_failures"] = media_failures;
      out["image_processed"] = image_processed;
      out["video_processed"] = video_processed;
      out["audio_processed"] = audio_processed;
      out["consolidation_runs"] = consolidation_runs;
      out["consolidation_ms_total"] = consolidation_ms_total;
      out["consolidation_events"] = consolidation_events;
      out["skip_final_consolidation"] = cfg.skip_final_consolidation;
      out["final_window_consolidated"] = final_window_consolidated;
      out["daily_final_window_consolidated"]
          = cfg.daily_consolidation && final_window_consolidated;
      out["wall_ms_excluding_consolidation"]
          = static_cast<double> (wall_ms) - consolidation_ms_total;
      out["daily_consolidation"] = cfg.daily_consolidation;
      out["daily_consolidation_hour_local"] = cfg.daily_consolidation_hour;
      out["daily_consolidation_policy"] = "source_time_sleep_checkpoint";
      out["replay_timezone"] = cfg.replay_timezone.empty () ? "process_default"
                                                            : cfg.replay_timezone;
      out["source_id_policy"]
          = "User and Contact are opaque conversation provenance source IDs; "
            "media is not encoded into source_id";
      out["timeline_policy"]
          = "transcript messages and media files are processed in timestamp "
            "order; text uses internal replay timestamped ingress so "
            "non-probe turns can skip context hydration, media uses internal "
            "replay timestamped ingress, and benchmark consolidation uses "
            "internal replay timestamped consolidation";
      out["media_timestamp_policy"]
          = "media replay uses internal timestamped ingress so signal "
            "timestamps match source event timestamps";
      out["consolidation_timestamp_policy"]
          = "benchmark replay consolidation uses source event timestamps at "
            "the configured local sleep checkpoint, not wall-clock time";
      out["wall_ms"] = wall_ms;
      out["peak_rss_mb"] = PeakRssMb ();
      out["mean_encode_ms"] = processed_text_this_run > 0
                                  ? encode_ms_total / processed_text_this_run
                                  : 0.0;
      out["mean_process_ms"] = processed_text_this_run > 0
                                   ? process_ms_total / processed_text_this_run
                                   : 0.0;
      out["mean_hydrate_ms"] = processed_text_this_run > 0
                                   ? hydrate_ms_total / processed_text_this_run
                                   : 0.0;
      out["mean_total_ms"] = processed_text_this_run > 0
                                 ? total_ms_total / processed_text_this_run
                                 : 0.0;
      out["db_memory_count"] = SqlCount (cfg.db_path,
                                         "SELECT COUNT(*) FROM memories");
      out["db_label_memory_count"]
          = SqlCount (cfg.db_path,
                      "SELECT COUNT(*) FROM memories WHERE kind = 'LABEL'");
      out["db_long_term_memory_count"]
          = SqlCount (cfg.db_path,
                      "SELECT COUNT(*) FROM memories WHERE kind = 'LONG_TERM'");
      out["db_working_memory_count"]
          = SqlCount (cfg.db_path,
                      "SELECT COUNT(*) FROM memories WHERE kind = 'WORKING'");
      out["db_memory_rows_with_embeddings"]
          = SqlCount (cfg.db_path,
                      "SELECT COUNT(*) FROM memories "
                      "WHERE embedding_id IS NOT NULL");
      out["db_association_count"]
          = SqlCount (cfg.db_path, "SELECT COUNT(*) FROM associations");
      out["behavior_note"]
          = "Live Cortext-only run: text/media are fed chronologically "
            "through Cortext ingress, with timestamped internal replay ingress "
            "for media and consolidation; no custom retrieval or scoring is "
            "used.";
      out["privacy_note"]
          = "Private benchmark artifact: probe rows include deterministic "
            "extractive compacted-history summaries for the RAG baseline; "
            "the public release report excludes message text and media "
            "content.";

      WriteJsonArtifact (cfg.output_path, out);
      std::cout << DumpJsonArtifact (out) << "\n";
      return 0;
    }
  catch (const std::exception &e)
    {
      std::cerr << "chat_replay_live_run failed: " << e.what () << "\n";
      return 1;
    }
}
