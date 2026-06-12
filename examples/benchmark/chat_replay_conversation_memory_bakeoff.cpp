#include <cortext/consolidation_mode.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/cortext.hpp>
#include <cortext/retention.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>

#include "../../src/encoder/text_encoder_factory.hpp"
#include "../../src/operations/retrieval_debug_state.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <any>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#include "transcript_discovery.hpp"

namespace fs = std::filesystem;

namespace
{

constexpr const char *kUserSourceId = "User";
constexpr const char *kContactSourceId = "Contact";

double
PeakResidentSetMb ()
{
#if defined(__APPLE__)
  struct rusage usage
  {
  };
  if (getrusage (RUSAGE_SELF, &usage) != 0)
    return 0.0;
  return static_cast<double> (usage.ru_maxrss) / (1024.0 * 1024.0);
#elif defined(__unix__)
  struct rusage usage
  {
  };
  if (getrusage (RUSAGE_SELF, &usage) != 0)
    return 0.0;
  return static_cast<double> (usage.ru_maxrss) / 1024.0;
#else
  return 0.0;
#endif
}

struct Config
{
  fs::path input_dir;
  fs::path db_path = "build/chat_replay_conversation_memory_bakeoff.sqlite";
  fs::path output_path = "build/chat_replay_conversation_memory_bakeoff.json";
  fs::path label_bank_path = "data/label_bank/metadata.json";
  std::string models_dir = "models";
  int skip_messages = 0;
  int max_messages = 80;
  int warmup_messages = 20;
  int probe_stride = 10;
  int min_probe_query_tokens = 2;
  int rag_top_k = 5;
  int max_injected_memories = 8;
  int active_history_token_budget = 8000;
  double focus = 0.5;
  double sensitivity = 0.5;
  double stability = 0.5;
  int consolidate_every = 20;
  int judge_limit = 6;
  std::string judge_model = "nemotron-3-nano-omni-30b-a3b-8bit";
  int fact_prompt_k = 1;
  int source_stm_recent_k = 4;
  int source_ltm_lexical_k = 5;
  int stratified_sample_messages = 0;
  int media_adjacent_min = 0;
  int rolling_probe_target = 0;
  bool deep_consolidation = false;
  bool daily_consolidation = false;
  bool use_label_bank = true;
  bool judge_enabled = true;
  bool fact_prompt_bakeoff = false;
  bool source_tagged_bakeoff = false;
  bool prompt_policy_bakeoff = false;
  bool compact_policy_bakeoff = false;
  bool stm_graph_bakeoff = false;
  bool graph_expanded_rag_bakeoff = false;
  bool rolling_eval = false;
};

struct Message
{
  int index = 0;
  std::uint64_t timestamp = 0;
  bool from_contact = false;
  bool media_adjacent = false;
  int media_attachment_count = 0;
  std::vector<std::string> media_kinds;
  std::string text;
};

struct RagDoc
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
  long long memory_context_tokens = 0;
  long long active_history_tokens = 0;
  long long context_chars = 0;
  long long context_items = 0;
  double overlap_sum = 0.0;
  double latency_ms_sum = 0.0;
  std::vector<long long> prompt_token_samples;
  std::vector<long long> context_token_samples;
  std::vector<double> latency_ms_samples;
};

struct QualityAggregate
{
  int judged = 0;
  double relevance_sum = 0.0;
  double sufficiency_sum = 0.0;
  double noise_sum = 0.0;
  double source_grounding_sum = 0.0;
  double temporal_correctness_sum = 0.0;
  double media_usefulness_sum = 0.0;
  int wins = 0;
};

struct CompactionAggregate
{
  int probe_events = 0;
  int compaction_events = 0;
  long long compacted_items = 0;
  long long raw_items = 0;
  long long raw_tokens = 0;
  long long compacted_original_tokens = 0;
  long long compacted_summary_tokens = 0;
};

struct FactPromptProbe
{
  std::vector<long long> fact_ids;
  std::vector<long long> evidence_memory_ids;
  double top_fact_distance = 0.0;
  long long source_fact_cache_rows = 0;
  long long fact_only_rows = 0;
  long long query_rows = 0;
  long long query_embedding_bytes = 0;
};

struct PendingFactPromptProbe
{
  std::size_t probe_json_index = 0;
  int message_index = 0;
  std::uint64_t timestamp = 0;
  std::string query;
  std::unordered_set<std::string> query_tokens;
  std::string current_context;
  std::string normal_rag_context;
  std::string full_history_context;
  std::vector<long long> current_memory_ids;
};

struct SourceTaggedProbe
{
  std::size_t probe_json_index = 0;
  int message_index = 0;
  std::string query;
  std::string current_context;
  std::string wm_history_context;
  std::string stm_recent_context;
  std::string ltm_lexical_context;
  std::string stm_ltm_union_context;
  std::string normal_rag_context;
  std::string full_history_context;
};

struct StmGraphPacket
{
  std::vector<long long> raw_memory_ids;
  std::vector<long long> relabel_memory_ids;
  int raw_cycle_count = 0;
  int relabel_cycle_count = 0;
  int raw_positive_label_cycles = 0;
  int relabel_positive_label_cycles = 0;
  int raw_positive_source_cycles = 0;
  int relabel_positive_source_cycles = 0;
  int relabel_durable_candidate_count = 0;
  int relabel_durable_source_count = 0;
  int relabel_fact_linked_candidate_count = 0;
  int raw_label_count = 0;
  int relabel_label_count = 0;
  double raw_best_label_overlap = 0.0;
  double relabel_best_label_overlap = 0.0;
  double raw_best_source_overlap = 0.0;
  double relabel_best_source_overlap = 0.0;
  std::string raw_context;
  std::string relabel_context;
};

struct GraphExpandedRagPacket
{
  std::vector<long long> seed_memory_ids;
  std::vector<long long> expanded_memory_ids;
  std::vector<long long> ranked_memory_ids;
  int temporal_neighbor_count = 0;
  int graph_candidate_count = 0;
  int fact_candidate_count = 0;
  int compact_label_count = 0;
  int compact_fact_count = 0;
  int missing_seed_count = 0;
  double best_graph_score = 0.0;
  double best_fact_score = 0.0;
  std::string source_context;
  std::string compact_context;
  std::string chat_context;
};

struct VectorRagPacket
{
  std::vector<RagDoc> docs;
  long long query_rows = 0;
  long long query_embedding_bytes = 0;
  long long vector_search_k = 0;
  long long candidate_rows = 0;
  long long prior_chat_rows = 0;
  double best_distance = 0.0;
};

std::optional<nlohmann::json>
JudgeCompactPolicyContexts (const Config &cfg, int probe_index,
                            const std::string &query,
                            const std::string &cortext_ltm_context,
                            const std::string &stm_recent_context,
                            const std::string &ltm_lexical_context,
                            const std::string &stm_ltm_union_context,
                            const std::string &normal_rag_context,
                            const std::string &full_history_context);

std::optional<nlohmann::json>
JudgeStmGraphContexts (const Config &cfg, int probe_index,
                       const std::string &query,
                       const std::string &raw_stm_graph_context,
                       const std::string &relabel_prune_context,
                       const std::string &normal_rag_context,
                       const std::string &full_history_context);

std::optional<nlohmann::json>
JudgeGraphExpandedRagContexts (const Config &cfg, int probe_index,
                               const std::string &query,
                               const std::string &normal_rag_context,
                               const std::string &cortext_ltm_context,
                               const std::string &graph_expanded_rag_context,
                               const std::string &full_history_context);

std::unordered_set<std::string>
Tokens (const std::string &text);

bool
TableExists (cortext::Store &store, const std::string &table);

std::string
EscapeXml (const std::string &value)
{
  std::string escaped;
  escaped.reserve (value.size ());
  for (char ch : value)
    {
      switch (ch)
        {
        case '&':
          escaped += "&amp;";
          break;
        case '<':
          escaped += "&lt;";
          break;
        case '>':
          escaped += "&gt;";
          break;
        case '"':
          escaped += "&quot;";
          break;
        case '\'':
          escaped += "&apos;";
          break;
        default:
          escaped.push_back (ch);
          break;
        }
    }
  return escaped;
}

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
         && std::isdigit (line[12]) && line[13] == ':'
         && std::isdigit (line[14]) && std::isdigit (line[15])
         && line[16] == ':' && std::isdigit (line[17])
         && std::isdigit (line[18]);
}

std::optional<std::uint64_t>
ParseTimestamp (const std::string &text)
{
  std::tm tm{};
  std::istringstream in (text.substr (0, 19));
  in >> std::get_time (&tm, "%Y-%m-%d %H:%M:%S");
  if (in.fail ())
    return std::nullopt;
  tm.tm_isdst = -1;
  const std::time_t seconds = std::mktime (&tm);
  if (seconds < 0)
    return std::nullopt;
  return static_cast<std::uint64_t> (seconds) * 1000ULL;
}

std::optional<std::uint64_t>
ParseMediaTimestamp (const std::string &name)
{
  if (name.size () < 19)
    return std::nullopt;
  std::string stamp = name.substr (0, 19);
  if (stamp.size () < 19 || stamp[4] != '-' || stamp[7] != '-'
      || stamp[10] != ' ' || stamp[13] != ' ' || stamp[16] != ' ')
    return std::nullopt;
  stamp[13] = ':';
  stamp[16] = ':';
  return ParseTimestamp (stamp);
}

int
LocalDayBucket (std::uint64_t timestamp_ms)
{
  const std::time_t seconds
      = static_cast<std::time_t> (timestamp_ms / 1000ULL);
  std::tm local{};
  localtime_r (&seconds, &local);
  return (local.tm_year + 1900) * 1000 + local.tm_yday;
}

std::string
LowerExt (const fs::path &path)
{
  std::string ext = path.extension ().string ();
  std::transform (ext.begin (), ext.end (), ext.begin (),
                  [] (unsigned char c) {
                    return static_cast<char> (std::tolower (c));
                  });
  return ext;
}

std::string
MediaKindFromExt (const std::string &ext)
{
  if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".heic"
      || ext == ".gif")
    return "image";
  if (ext == ".mov" || ext == ".mp4" || ext == ".m4v")
    return "video";
  if (ext == ".wav" || ext == ".m4a" || ext == ".mp3" || ext == ".aac")
    return "audio";
  if (ext == ".url")
    return "link";
  return "other";
}

struct MediaIndex
{
  std::unordered_map<std::uint64_t, std::vector<std::string>> by_timestamp;
  std::map<std::string, int> kind_counts;
  std::map<std::string, int> skipped_reason_counts;
  int file_count = 0;
};

MediaIndex
BuildMediaIndex (const fs::path &dir)
{
  MediaIndex index;
  if (!fs::exists (dir))
    {
      index.skipped_reason_counts["input_dir_missing"]++;
      return index;
    }
  for (const auto &entry : fs::directory_iterator (dir))
    {
      if (!entry.is_regular_file ())
        continue;
      const auto ext = LowerExt (entry.path ());
      const auto kind = MediaKindFromExt (ext);
      if (kind == "other")
        {
          index.skipped_reason_counts["unsupported_extension"]++;
          continue;
        }
      auto ts = ParseMediaTimestamp (entry.path ().filename ().string ());
      if (!ts)
        {
          index.skipped_reason_counts["timestamp_parse_failed"]++;
          continue;
        }
      index.by_timestamp[*ts].push_back (kind);
      index.kind_counts[kind]++;
      index.file_count++;
    }
  return index;
}

void
AnnotateMediaAdjacency (std::vector<Message> &messages,
                        const MediaIndex &media_index)
{
  for (auto &message : messages)
    {
      auto it = media_index.by_timestamp.find (message.timestamp);
      if (it == media_index.by_timestamp.end ())
        continue;
      message.media_adjacent = true;
      message.media_attachment_count = static_cast<int> (it->second.size ());
      std::set<std::string> kinds (it->second.begin (), it->second.end ());
      message.media_kinds.assign (kinds.begin (), kinds.end ());
    }
}

std::string
EvalText (const Message &message)
{
  if (!message.media_adjacent)
    return message.text;
  std::ostringstream out;
  out << message.text;
  out << "\n[media attachments count=" << message.media_attachment_count
      << " kinds=";
  for (size_t i = 0; i < message.media_kinds.size (); ++i)
    {
      if (i > 0)
        out << ",";
      out << message.media_kinds[i];
    }
  out << "]";
  return out.str ();
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

std::vector<Message>
StratifiedSampleMessages (const std::vector<Message> &messages, int target_count,
                          int media_adjacent_min)
{
  if (target_count <= 0
      || static_cast<int> (messages.size ()) <= target_count)
    return messages;

  std::set<int> selected;
  const int n = static_cast<int> (messages.size ());

  auto add_window = [&] (int center, int count) {
    if (count <= 0)
      return;
    int start = std::clamp (center - count / 2, 0, std::max (0, n - count));
    for (int offset = 0; offset < count && start + offset < n; ++offset)
      selected.insert (start + offset);
  };

  const int early_count = target_count / 3;
  const int middle_count = target_count / 3;
  const int recent_count = target_count - early_count - middle_count;
  add_window (target_count / 6, early_count);
  add_window (n / 2, middle_count);
  add_window (n - recent_count / 2, recent_count);

  for (int i = 0;
       static_cast<int> (selected.size ()) < target_count && i < target_count;
       ++i)
    {
      const int idx = static_cast<int> (
          std::llround (static_cast<double> (i) * (n - 1)
                        / static_cast<double> (std::max (1, target_count - 1))));
      selected.insert (std::clamp (idx, 0, n - 1));
    }

  int selected_media = 0;
  for (int idx : selected)
    {
      if (messages[static_cast<size_t> (idx)].media_adjacent)
        ++selected_media;
    }
  if (media_adjacent_min > selected_media)
    {
      for (int i = 0; i < n && selected_media < media_adjacent_min; ++i)
        {
          if (!messages[static_cast<size_t> (i)].media_adjacent)
            continue;
          if (selected.insert (i).second)
            ++selected_media;
        }
    }

  std::vector<Message> out;
  out.reserve (selected.size ());
  int out_index = 0;
  for (int idx : selected)
    {
      Message copy = messages[static_cast<size_t> (idx)];
      copy.index = out_index++;
      out.push_back (std::move (copy));
    }
  return out;
}

std::set<int>
BuildProbeIndices (const std::vector<Message> &messages, int warmup,
                   int probe_target, int min_query_tokens)
{
  std::vector<int> eligible;
  for (int i = std::max (0, warmup); i < static_cast<int> (messages.size ());
       ++i)
    {
      if (static_cast<int> (Tokens (EvalText (messages[static_cast<size_t> (i)]))
                                .size ())
          >= min_query_tokens)
        eligible.push_back (i);
    }
  if (probe_target <= 0 || static_cast<int> (eligible.size ()) <= probe_target)
    return std::set<int> (eligible.begin (), eligible.end ());

  std::set<int> out;
  for (int i = 0; i < probe_target; ++i)
    {
      const int pos = static_cast<int> (
          std::llround (static_cast<double> (i) * (eligible.size () - 1)
                        / static_cast<double> (std::max (1, probe_target - 1))));
      out.insert (eligible[static_cast<size_t> (
          std::clamp (pos, 0, static_cast<int> (eligible.size ()) - 1))]);
    }
  return out;
}

std::unordered_set<std::string>
Tokens (const std::string &text)
{
  static const std::unordered_set<std::string> stop = {
    "the", "and", "you", "that", "for", "with", "this", "have", "just",
    "but", "not", "are", "was", "what", "from", "they", "your", "our",
    "can", "all", "will", "there", "about", "would", "could", "should",
    "then", "when", "where", "were", "been", "into", "like", "okay",
    "yeah", "yes", "no", "ok", "lol", "i", "im", "it", "is", "to" };
  std::unordered_set<std::string> out;
  std::string cur;
  for (char ch : text)
    {
      unsigned char c = static_cast<unsigned char> (ch);
      if (std::isalnum (c))
        cur.push_back (static_cast<char> (std::tolower (c)));
      else if (!cur.empty ())
        {
          if (cur.size () >= 4 && stop.find (cur) == stop.end ())
            out.insert (cur);
          cur.clear ();
        }
    }
  if (cur.size () >= 4 && stop.find (cur) == stop.end ())
    out.insert (cur);
  return out;
}

long long
EstimateTokens (std::size_t chars)
{
  if (chars == 0)
    return 0;
  return std::max<long long> (
      1, static_cast<long long> (std::ceil (static_cast<double> (chars) / 4.0)));
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

std::string
MemoryText (const cortext::Cortext::Context::Memory &memory)
{
  std::string text;
  for (const auto &blob : memory.content)
    {
      text.append (blob.begin (), blob.end ());
      text.push_back ('\n');
    }
  return Trim (text);
}

std::vector<cortext::Cortext::Context::Memory>
HydrateMemories (cortext::Cortext &engine, const std::vector<long long> &ids);

double
Overlap (const std::unordered_set<std::string> &query_tokens,
         const std::string &context)
{
  if (query_tokens.empty ())
    return 0.0;
  const auto context_tokens = Tokens (context);
  int hits = 0;
  for (const auto &token : query_tokens)
    {
      if (context_tokens.find (token) != context_tokens.end ())
        ++hits;
    }
  return static_cast<double> (hits)
         / static_cast<double> (query_tokens.size ());
}

std::vector<RagDoc>
RagTopK (const std::vector<RagDoc> &docs,
         const std::unordered_set<std::string> &query_tokens,
         std::uint64_t query_ts, int top_k)
{
  struct Scored
  {
    double score = 0.0;
    const RagDoc *doc = nullptr;
  };
  std::vector<Scored> scored;
  scored.reserve (docs.size ());
  for (const auto &doc : docs)
    {
      int overlap = 0;
      for (const auto &token : query_tokens)
        {
          if (doc.tokens.find (token) != doc.tokens.end ())
            ++overlap;
        }
      const double age_hours
          = query_ts > doc.timestamp
                ? static_cast<double> (query_ts - doc.timestamp) / 3600000.0
                : 0.0;
      const double recency = 1.0 / (1.0 + age_hours / 24.0);
      scored.push_back ({ static_cast<double> (overlap) + 0.05 * recency,
                          &doc });
    }
  std::sort (scored.begin (), scored.end (),
             [] (const auto &a, const auto &b) {
               if (a.score == b.score)
                 return a.doc->timestamp > b.doc->timestamp;
               return a.score > b.score;
             });
  std::vector<RagDoc> out;
  for (const auto &item : scored)
    {
      if (static_cast<int> (out.size ()) >= top_k)
        break;
      if (item.score <= 0.0)
        break;
      out.push_back (*item.doc);
    }
  return out;
}

std::vector<RagDoc>
RecentDocs (const std::vector<RagDoc> &docs, int count)
{
  std::vector<RagDoc> out;
  if (count <= 0)
    return out;
  for (auto it = docs.rbegin (); it != docs.rend (); ++it)
    {
      out.push_back (*it);
      if (static_cast<int> (out.size ()) >= count)
        break;
    }
  std::reverse (out.begin (), out.end ());
  return out;
}

std::vector<RagDoc>
UnionDocs (const std::vector<RagDoc> &first,
           const std::vector<RagDoc> &second, int max_count)
{
  std::vector<RagDoc> out;
  std::unordered_set<int> seen;
  auto append = [&] (const std::vector<RagDoc> &docs) {
    for (const auto &doc : docs)
      {
        if (!seen.insert (doc.index).second)
          continue;
        out.push_back (doc);
        if (max_count > 0 && static_cast<int> (out.size ()) >= max_count)
          return;
      }
  };
  append (first);
  if (max_count <= 0 || static_cast<int> (out.size ()) < max_count)
    append (second);
  return out;
}

void
AppendMemoryContext (std::ostringstream &out, std::set<long long> &seen,
                     const cortext::Cortext::Context::Memory &mem,
                     const char *kind, int *items)
{
  if (mem.id != 0 && seen.find (mem.id) != seen.end ())
    return;
  const std::string text = MemoryText (mem);
  if (text.empty ())
    return;
  seen.insert (mem.id);
  out << "[" << kind << " source=" << mem.source_id << " ts="
      << mem.timestamp << "] " << text << "\n";
  ++(*items);
}

std::string
BuildCortextRetrievedContext (const cortext::Cortext::Context &ctx, int *items)
{
  std::ostringstream out;
  std::set<long long> seen;
  for (const auto &mem : ctx.retrieved_memory)
    AppendMemoryContext (out, seen, mem, "retrieved", items);
  return out.str ();
}

std::string
BuildCortextWorkingContext (const cortext::Cortext::Context &ctx, int *items)
{
  std::ostringstream out;
  std::set<long long> seen;
  for (const auto &mem : ctx.working_memory)
    AppendMemoryContext (out, seen, mem, "working", items);
  return out.str ();
}

std::string
BuildRagContext (const std::vector<RagDoc> &docs)
{
  std::ostringstream out;
  for (const auto &doc : docs)
    {
      out << "[rag source=" << doc.source_id << " ts=" << doc.timestamp
          << "] " << doc.text << "\n";
    }
  return out.str ();
}

std::vector<cortext::Cortext::Context::Memory>
FilterInjectedMemories (
    const std::vector<cortext::Cortext::Context::Memory> &retrieved_memories,
    const std::vector<cortext::Cortext::Context::Memory> &working_memory,
    int max_injected_memories)
{
  std::unordered_set<std::string> working_keys;
  std::unordered_set<std::string> injected_keys;
  std::vector<cortext::Cortext::Context::Memory> injected;
  for (const auto &mem : working_memory)
    {
      const std::string text = MemoryText (mem);
      if (!text.empty ())
        working_keys.insert (mem.source_id + "\n" + text);
    }
  for (const auto &mem : retrieved_memories)
    {
      const std::string text = MemoryText (mem);
      if (text.empty ())
        continue;
      const std::string key = mem.source_id + "\n" + text;
      if (working_keys.find (key) != working_keys.end ()
          || injected_keys.find (key) != injected_keys.end ())
        continue;
      if (max_injected_memories > 0
          && static_cast<int> (injected.size ()) >= max_injected_memories)
        continue;
      injected_keys.insert (key);
      injected.push_back (mem);
    }
  return injected;
}

std::string
BuildInjectedSystemPrompt (
    const std::vector<cortext::Cortext::Context::Memory> &injected_memories)
{
  std::ostringstream oss;
  oss << "<snapshot>\n";
  oss << "<clock date=\"benchmark\" time=\"benchmark\" timezone=\"local\"/>\n";
  oss << "The XML snapshot below represents the current chat turn. Treat the "
         "<clock> element as the current local time for temporal references "
         "like now, today, tomorrow, and deadlines. The snapshot may contain "
         "retrieved memories from earlier interaction with this same user. "
         "Treat any memories in it as facts about the current user unless a "
         "memory clearly refers to someone else or quotes someone else. Use "
         "them as supporting context when they are relevant to the user's "
         "current message. Prefer these memories over guesses, but do not "
         "mention memory IDs, soft_anchor IDs, the XML format, or that you "
         "were given retrieved memories. Treat soft_anchor entries as optional "
         "continuity likelihoods for their memory, not as resolved facts.\n\n";
  oss << "<memories>\n";
  for (const auto &mem : injected_memories)
    {
      const std::string text = MemoryText (mem);
      if (text.empty ())
        continue;
      oss << "  <memory source_id=\"" << EscapeXml (mem.source_id)
          << "\" datetime=\"" << mem.timestamp << "\" memory_id=\""
          << mem.id << "\">\n"
          << "    <text>" << EscapeXml (text) << "</text>\n"
          << "  </memory>\n";
    }
  oss << "</memories>\n</snapshot>";
  return oss.str ();
}

std::string
BuildSimpleRagSystemPrompt (
    const std::vector<cortext::Cortext::Context::Memory> &injected_memories,
    std::size_t max_memories)
{
  std::ostringstream oss;
  oss << "Relevant notes from prior conversation. Use them when helpful, but "
         "answer the current user directly.\n";
  std::size_t emitted = 0;
  for (const auto &mem : injected_memories)
    {
      if (emitted >= max_memories)
        break;
      const std::string text = MemoryText (mem);
      if (text.empty ())
        continue;
      oss << "- " << text << "\n";
      ++emitted;
    }
  return oss.str ();
}

std::string
BuildSimpleRagSystemPrompt (const std::vector<RagDoc> &docs,
                            std::size_t max_memories)
{
  std::ostringstream oss;
  oss << "Relevant notes from prior conversation. Use them when helpful, but "
         "answer the current user directly.\n";
  std::size_t emitted = 0;
  for (const auto &doc : docs)
    {
      if (emitted >= max_memories)
        break;
      if (doc.text.empty ())
        continue;
      oss << "- " << doc.text << "\n";
      ++emitted;
    }
  return oss.str ();
}

std::string
BuildMemoryListContext (
    const std::vector<cortext::Cortext::Context::Memory> &memories,
    std::size_t max_memories, const std::string &prefix)
{
  std::ostringstream out;
  std::size_t emitted = 0;
  for (const auto &mem : memories)
    {
      if (emitted >= max_memories)
        break;
      const std::string text = MemoryText (mem);
      if (text.empty ())
        continue;
      out << "[" << prefix << " source=" << mem.source_id << " ts="
          << mem.timestamp << "] " << text << "\n";
      ++emitted;
    }
  return out.str ();
}

std::vector<std::string>
SplitAuditList (const std::string &value)
{
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&] {
    std::string item = Trim (cur);
    cur.clear ();
    if (!item.empty ())
      out.push_back (std::move (item));
  };
  for (char ch : value)
    {
      if (ch == ',')
        flush ();
      else
        cur.push_back (ch);
    }
  flush ();
  return out;
}

std::vector<long long>
SplitAuditIds (const std::string &value)
{
  std::vector<long long> out;
  for (const auto &item : SplitAuditList (value))
    {
      try
        {
          const long long id = std::stoll (item);
          if (id > 0)
            out.push_back (id);
        }
      catch (...)
        {
        }
    }
  return out;
}

double
LabelOverlap (const std::unordered_set<std::string> &query_tokens,
              const std::vector<std::string> &labels)
{
  if (query_tokens.empty () || labels.empty ())
    return 0.0;
  std::unordered_set<std::string> label_tokens;
  for (const auto &label : labels)
    {
      auto tokens = Tokens (label);
      label_tokens.insert (tokens.begin (), tokens.end ());
    }
  int hits = 0;
  for (const auto &token : query_tokens)
    {
      if (label_tokens.find (token) != label_tokens.end ())
        ++hits;
    }
  return static_cast<double> (hits)
         / static_cast<double> (query_tokens.size ());
}

std::string
RoleFromSourceId (const std::string &source_id)
{
  if (source_id == kUserSourceId)
    return "user";
  if (source_id == kContactSourceId)
    return "assistant";
  return {};
}

std::size_t
ChatDemoCortextPromptChars (
    const std::vector<cortext::Cortext::Context::Memory> &working_memory,
    const std::string &injected_system, const std::string &latest_user_input)
{
  std::size_t chars = injected_system.empty ()
                          ? 0
                          : std::string ("system").size ()
                                + injected_system.size ();
  std::vector<const cortext::Cortext::Context::Memory *> ordered;
  for (const auto &mem : working_memory)
    ordered.push_back (&mem);
  std::sort (ordered.begin (), ordered.end (),
             [] (const auto *a, const auto *b) {
               if (a->timestamp == b->timestamp)
                 return a->id < b->id;
               return a->timestamp < b->timestamp;
             });
  for (const auto *mem : ordered)
    {
      const std::string role = RoleFromSourceId (mem->source_id);
      const std::string text = MemoryText (*mem);
      if (role.empty () || text.empty ())
        continue;
      if (role == "user" && !latest_user_input.empty ()
          && text == latest_user_input)
        continue;
      chars += role.size () + text.size ();
    }
  if (!latest_user_input.empty ())
    chars += std::string ("user").size () + latest_user_input.size ();
  return chars;
}

std::size_t
ChatDemoRagPromptChars (
    const std::vector<cortext::Cortext::Context::Memory> &injected_memories,
    const std::string &latest_user_input, std::size_t max_memories)
{
  const std::string system_prompt
      = BuildSimpleRagSystemPrompt (injected_memories, max_memories);
  std::size_t chars = system_prompt.empty ()
                          ? 0
                          : std::string ("system").size ()
                                + system_prompt.size ();
  if (!latest_user_input.empty ())
    chars += std::string ("user").size () + latest_user_input.size ();
  return chars;
}

std::size_t
ChatDemoRagPromptChars (const std::vector<RagDoc> &docs,
                        const std::string &latest_user_input,
                        std::size_t max_memories)
{
  const std::string system_prompt
      = BuildSimpleRagSystemPrompt (docs, max_memories);
  std::size_t chars = system_prompt.empty ()
                          ? 0
                          : std::string ("system").size ()
                                + system_prompt.size ();
  if (!latest_user_input.empty ())
    chars += std::string ("user").size () + latest_user_input.size ();
  return chars;
}

std::string
BuildActiveHistoryContext (const std::vector<RagDoc> &docs,
                           int token_budget, int *items)
{
  std::vector<const RagDoc *> selected;
  long long used_tokens = 0;
  for (auto it = docs.rbegin (); it != docs.rend (); ++it)
    {
      const long long tokens
          = EstimateTokens (it->source_id.size () + it->text.size () + 3);
      if (token_budget >= 0 && used_tokens + tokens > token_budget
          && !selected.empty ())
        {
          break;
        }
      selected.push_back (&*it);
      used_tokens += tokens;
      if (token_budget >= 0 && used_tokens >= token_budget)
        {
          break;
        }
    }
  std::reverse (selected.begin (), selected.end ());

  std::ostringstream out;
  for (const auto *doc : selected)
    {
      out << doc->source_id << ": " << doc->text << "\n";
      ++(*items);
    }
  return out.str ();
}

struct CompactedHistory
{
  std::string context;
  int raw_items = 0;
  int compacted_items = 0;
  long long raw_tokens = 0;
  long long compacted_original_tokens = 0;
  long long compacted_summary_tokens = 0;
  int compaction_events = 0;
};

CompactedHistory
BuildCompactedActiveHistoryContext (const std::vector<RagDoc> &docs,
                                    int token_budget)
{
  CompactedHistory result;
  if (docs.empty ())
    return result;

  std::vector<const RagDoc *> raw_selected;
  long long raw_tokens = 0;
  for (auto it = docs.rbegin (); it != docs.rend (); ++it)
    {
      const long long tokens
          = EstimateTokens (it->source_id.size () + it->text.size () + 3);
      if (token_budget >= 0 && raw_tokens + tokens > token_budget
          && !raw_selected.empty ())
        break;
      raw_selected.push_back (&*it);
      raw_tokens += tokens;
      if (token_budget >= 0 && raw_tokens >= token_budget)
        break;
    }
  std::reverse (raw_selected.begin (), raw_selected.end ());

  const int raw_begin_index
      = raw_selected.empty () ? static_cast<int> (docs.size ())
                              : raw_selected.front ()->index;
  for (const auto &doc : docs)
    {
      const long long tokens
          = EstimateTokens (doc.source_id.size () + doc.text.size () + 3);
      if (doc.index < raw_begin_index)
        {
          result.compacted_items++;
          result.compacted_original_tokens += tokens;
        }
    }

  std::ostringstream out;
  if (result.compacted_items > 0)
    {
      result.compaction_events = 1;
      out << "[compacted_history messages=" << result.compacted_items
          << " original_tokens=" << result.compacted_original_tokens
          << " note=\"older rolling chat compressed before this turn\"]\n";
      result.compacted_summary_tokens = EstimateTokens (out.str ().size ());
    }
  for (const auto *doc : raw_selected)
    {
      out << doc->source_id << ": " << doc->text << "\n";
    }
  result.context = out.str ();
  result.raw_items = static_cast<int> (raw_selected.size ());
  result.raw_tokens = raw_tokens;
  return result;
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

std::string
ShellQuoteString (const std::string &s)
{
  std::string out = "'";
  for (char c : s)
    out += c == '\'' ? "'\\''" : std::string (1, c);
  out += "'";
  return out;
}

std::string
CurlConfigQuote (const std::string &s)
{
  std::string out = "\"";
  for (char c : s)
    {
      if (c == '\\' || c == '"')
        out += '\\';
      out += c;
    }
  out += "\"";
  return out;
}

std::string
GetEnvString (const char *name, const std::string &fallback = "")
{
  const char *value = std::getenv (name);
  return value ? std::string (value) : fallback;
}

std::string
RunCommandCapture (const std::string &cmd)
{
  std::array<char, 4096> buffer{};
  std::string output;
  FILE *pipe = popen (cmd.c_str (), "r");
  if (!pipe)
    return output;
  while (fgets (buffer.data (), static_cast<int> (buffer.size ()), pipe))
    output += buffer.data ();
  pclose (pipe);
  return output;
}

std::string
LowerAscii (std::string value)
{
  std::transform (value.begin (), value.end (), value.begin (),
                  [] (unsigned char c) {
                    return static_cast<char> (std::tolower (c));
                  });
  return value;
}

bool
IsNemotronJudgeModel (const std::string &model)
{
  return LowerAscii (model).find ("nemotron") != std::string::npos;
}

std::string
JudgeUrlHost (const std::string &url)
{
  std::string rest = url;
  const auto scheme = rest.find ("://");
  if (scheme != std::string::npos)
    rest = rest.substr (scheme + 3);
  if (!rest.empty () && rest.front () == '[')
    {
      const auto end = rest.find (']');
      return end == std::string::npos ? rest : rest.substr (1, end - 1);
    }
  const auto end = rest.find_first_of (":/");
  return end == std::string::npos ? rest : rest.substr (0, end);
}

bool
IsLoopbackJudgeHost (const std::string &host)
{
  const std::string lower = LowerAscii (host);
  return lower == "localhost" || lower == "127.0.0.1" || lower == "::1";
}

std::string
LocalNemotronJudgeBaseUrl ()
{
  std::string base_url = GetEnvString (
      "CORTEXT_JUDGE_BASE_URL",
      GetEnvString ("LOCAL_JUDGE_BASE_URL", "http://127.0.0.1:8000/v1"));
  while (!base_url.empty () && base_url.back () == '/')
    base_url.pop_back ();
  if (base_url.size () < 3 || base_url.substr (base_url.size () - 3) != "/v1")
    base_url += "/v1";

  const std::string host = JudgeUrlHost (base_url);
  if (!IsLoopbackJudgeHost (host))
    {
      throw std::runtime_error (
          "Refusing non-local judge endpoint for private chat-replay bakeoff: "
          + base_url
          + ". Start the local Nemotron/MLX judge server and set "
            "CORTEXT_JUDGE_BASE_URL or LOCAL_JUDGE_BASE_URL to a loopback "
            "URL.");
    }
  return base_url;
}

std::optional<nlohmann::json>
CallJudgeModel (const Config &cfg, const fs::path &tmp_stem,
                const std::string &prompt, int max_tokens)
{
  if (!cfg.judge_enabled || cfg.judge_model.empty ())
    return std::nullopt;

  const fs::path tmp_path = tmp_stem.string () + ".json";
  if (!IsNemotronJudgeModel (cfg.judge_model))
    throw std::runtime_error (
        "Refusing non-Nemotron judge model for private chat-replay bakeoff: "
        + cfg.judge_model);

  const std::string api_key = GetEnvString (
      "CORTEXT_JUDGE_API_KEY", GetEnvString ("LOCAL_JUDGE_API_KEY"));
  const std::string base_url = LocalNemotronJudgeBaseUrl ();
  nlohmann::json request;
  request["model"] = cfg.judge_model;
  request["messages"] = nlohmann::json::array (
      { { { "role", "system" },
          { "content",
            "You are a strict JSON-only evaluator for memory context "
            "quality. Return only the requested JSON object." } },
        { { "role", "user" }, { "content", prompt } } });
  request["temperature"] = 0;
  request["max_tokens"] = max_tokens;
  request["response_format"] = { { "type", "json_object" } };
  request["enable_thinking"] = false;
  request["chat_template_kwargs"] = { { "enable_thinking", false } };

  {
    std::ofstream tmp (tmp_path);
    tmp << request.dump ();
  }
  const fs::path curl_cfg_path = tmp_stem.string () + ".curl";
  {
    std::ofstream curl_cfg (curl_cfg_path);
    curl_cfg << "url = "
             << CurlConfigQuote (base_url + "/chat/completions") << "\n";
    curl_cfg << "request = POST\n";
    curl_cfg << "max-time = 180\n";
    curl_cfg << "silent\n";
    curl_cfg << "show-error\n";
    curl_cfg << "header = "
             << CurlConfigQuote ("Content-Type: application/json") << "\n";
    if (!api_key.empty ())
      {
        curl_cfg << "header = "
                 << CurlConfigQuote ("Authorization: Bearer " + api_key)
                 << "\n";
      }
    curl_cfg << "data-binary = " << CurlConfigQuote ("@" + tmp_path.string ())
             << "\n";
  }
  const std::string raw
      = RunCommandCapture ("curl --config " + ShellQuote (curl_cfg_path));
  fs::remove (tmp_path);
  fs::remove (curl_cfg_path);
  auto parsed = nlohmann::json::parse (raw, nullptr, false);
  if (parsed.is_discarded () || !parsed.contains ("choices")
      || !parsed["choices"].is_array () || parsed["choices"].empty ())
    return std::nullopt;
  const auto &choice = parsed["choices"][0];
  if (!choice.contains ("message") || !choice["message"].is_object ()
      || !choice["message"].contains ("content")
      || !choice["message"]["content"].is_string ())
    return std::nullopt;
  auto judged = nlohmann::json::parse (
      choice["message"]["content"].get<std::string> (), nullptr, false);
  if (judged.is_discarded () || !judged.is_object ())
    return std::nullopt;
  return judged;
}

std::optional<nlohmann::json>
JudgeContexts (const Config &cfg, int probe_index,
               const std::string &query,
               const std::string &cortext_context,
               const std::string &cortext_ltm_context,
               const std::string &rag_context,
               const std::string &lexical_rag_context,
               const std::string &normal_rag_context,
               const std::string &full_history_context)
{
  if (!cfg.judge_enabled || cfg.judge_model.empty ())
    return std::nullopt;

  std::ostringstream prompt;
  prompt
      << "You are judging memory context quality for a chat assistant.\n"
      << "Given the current user message and six candidate context packets, "
         "score only how useful each packet is for answering or continuing the "
         "conversation.\n"
      << "Use 0-5 integer scores. relevance means contains information likely "
         "useful to this turn. sufficiency means enough context to answer. "
         "noise means distracting or excessive unrelated content, where 0 is "
         "clean and 5 is very noisy. source_grounding means the packet gives "
         "traceable source-backed evidence. temporal_correctness means the "
         "packet preserves ordering/date cues. media_usefulness means any media "
         "markers or media-derived context help the turn; use 0 when no useful "
         "media context appears.\n"
      << "Return strict JSON only with this shape:\n"
      << "{\"cortext\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0,"
         "\"source_grounding\":0,\"temporal_correctness\":0,\"media_usefulness\":0},"
         "\"cortext_ltm\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0,"
         "\"source_grounding\":0,\"temporal_correctness\":0,\"media_usefulness\":0},"
         "\"rag\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0,"
         "\"source_grounding\":0,\"temporal_correctness\":0,\"media_usefulness\":0},"
         "\"lexical_rag\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0,"
         "\"source_grounding\":0,\"temporal_correctness\":0,\"media_usefulness\":0},"
         "\"normal_rag\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0,"
         "\"source_grounding\":0,\"temporal_correctness\":0,\"media_usefulness\":0},"
         "\"full_history\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0,"
         "\"source_grounding\":0,\"temporal_correctness\":0,\"media_usefulness\":0},"
         "\"winner\":\"cortext|cortext_ltm|rag|lexical_rag|normal_rag|full_history|tie\"}\n\n"
      << "CURRENT_USER_MESSAGE:\n" << query << "\n\n"
      << "CORTEXT_CONTEXT:\n" << cortext_context << "\n\n"
      << "CORTEXT_LTM_CONTEXT:\n" << cortext_ltm_context << "\n\n"
      << "RAG_CONTEXT:\n" << rag_context << "\n\n"
      << "LEXICAL_RAG_CONTEXT:\n" << lexical_rag_context << "\n\n"
      << "NORMAL_RAG_CONTEXT:\n" << normal_rag_context << "\n\n"
      << "FULL_HISTORY_CONTEXT:\n" << full_history_context << "\n";

  const fs::path tmp_stem
      = cfg.output_path.parent_path ()
        / ("chat_replay_bakeoff_judge_" + std::to_string (probe_index));
  return CallJudgeModel (cfg, tmp_stem, prompt.str (), 512);
}

std::optional<nlohmann::json>
JudgeFactPromptContexts (const Config &cfg, int probe_index,
                         const std::string &query,
                         const std::string &current_context,
                         const std::string &fact_replace_context,
                         const std::string &fact_union_context,
                         const std::string &normal_rag_context,
                         const std::string &full_history_context)
{
  if (!cfg.judge_enabled || cfg.judge_model.empty ())
    return std::nullopt;

  std::ostringstream prompt;
  prompt
      << "You are judging memory context quality for a chat assistant.\n"
      << "Given the current user message and five candidate context packets, "
         "score only how useful each packet is for answering or continuing the "
         "conversation.\n"
      << "Use 0-5 integer scores. relevance means contains information likely "
         "useful to this turn. sufficiency means enough context to answer. "
         "noise means distracting or excessive unrelated content, where 0 is "
         "clean and 5 is very noisy.\n"
      << "Return strict JSON only with this shape:\n"
      << "{\"current\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"fact_replace\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"fact_union\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"normal_rag\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"full_history\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"winner\":\"current|fact_replace|fact_union|normal_rag|full_history|tie\"}\n\n"
      << "CURRENT_USER_MESSAGE:\n" << query << "\n\n"
      << "CURRENT_CORTEXT:\n" << current_context << "\n\n"
      << "FACT_REPLACE:\n" << fact_replace_context << "\n\n"
      << "FACT_UNION:\n" << fact_union_context << "\n\n"
      << "NORMAL_RAG:\n" << normal_rag_context << "\n\n"
      << "FULL_HISTORY:\n" << full_history_context << "\n";

  const fs::path tmp_stem
      = cfg.output_path.parent_path ()
        / ("chat_replay_fact_prompt_judge_" + std::to_string (probe_index));
  return CallJudgeModel (cfg, tmp_stem, prompt.str (), 256);
}

std::optional<nlohmann::json>
JudgeSourceTaggedContexts (const Config &cfg, int probe_index,
                           const std::string &query,
                           const std::string &current_context,
                           const std::string &wm_history_context,
                           const std::string &stm_recent_context,
                           const std::string &ltm_lexical_context,
                           const std::string &stm_ltm_union_context,
                           const std::string &normal_rag_context,
                           const std::string &full_history_context)
{
  if (!cfg.judge_enabled || cfg.judge_model.empty ())
    return std::nullopt;

  std::ostringstream prompt;
  prompt
      << "You are judging memory context quality for a chat assistant.\n"
      << "Given the current user message and seven candidate context packets, "
         "score only how useful each packet is for answering or continuing the "
         "conversation.\n"
      << "Use 0-5 integer scores. relevance means contains information likely "
         "useful to this turn. sufficiency means enough context to answer. "
         "noise means distracting or excessive unrelated content, where 0 is "
         "clean and 5 is very noisy.\n"
      << "Return strict JSON only with this shape:\n"
      << "{\"current\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"wm_history\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"stm_recent\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"ltm_lexical\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"stm_ltm_union\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"normal_rag\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"full_history\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"winner\":\"current|wm_history|stm_recent|ltm_lexical|stm_ltm_union|normal_rag|full_history|tie\"}\n\n"
      << "CURRENT_USER_MESSAGE:\n" << query << "\n\n"
      << "CURRENT_CORTEXT:\n" << current_context << "\n\n"
      << "WM_HISTORY:\n" << wm_history_context << "\n\n"
      << "STM_RECENT:\n" << stm_recent_context << "\n\n"
      << "LTM_LEXICAL:\n" << ltm_lexical_context << "\n\n"
      << "STM_LTM_UNION:\n" << stm_ltm_union_context << "\n\n"
      << "NORMAL_RAG:\n" << normal_rag_context << "\n\n"
      << "FULL_HISTORY:\n" << full_history_context << "\n";

  const fs::path tmp_stem
      = cfg.output_path.parent_path ()
        / ("chat_replay_source_tagged_judge_" + std::to_string (probe_index)
           );
  return CallJudgeModel (cfg, tmp_stem, prompt.str (), 384);
}

std::optional<nlohmann::json>
JudgePromptPolicyContexts (const Config &cfg, int probe_index,
                           const std::string &query,
                           const std::string &current_context,
                           const std::string &stm_recent_context,
                           const std::string &current_stm_context,
                           const std::string &current_stm_ltm_context,
                           const std::string &normal_rag_context,
                           const std::string &full_history_context)
{
  if (!cfg.judge_enabled || cfg.judge_model.empty ())
    return std::nullopt;

  std::ostringstream prompt;
  prompt
      << "You are judging memory prompt policies for a chat assistant.\n"
      << "Given the current user message and six candidate context packets, "
         "score only how useful each packet is for answering or continuing the "
         "conversation.\n"
      << "Use 0-5 integer scores. relevance means contains information likely "
         "useful to this turn. sufficiency means enough context to answer. "
         "noise means distracting or excessive unrelated content, where 0 is "
         "clean and 5 is very noisy.\n"
      << "Return strict JSON only with this shape:\n"
      << "{\"current\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"stm_recent\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"current_stm\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"current_stm_ltm\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"normal_rag\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"full_history\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"winner\":\"current|stm_recent|current_stm|current_stm_ltm|normal_rag|full_history|tie\"}\n\n"
      << "CURRENT_USER_MESSAGE:\n" << query << "\n\n"
      << "CURRENT_CORTEXT:\n" << current_context << "\n\n"
      << "STM_RECENT_ONLY:\n" << stm_recent_context << "\n\n"
      << "CURRENT_PLUS_STM:\n" << current_stm_context << "\n\n"
      << "CURRENT_PLUS_STM_LTM:\n" << current_stm_ltm_context << "\n\n"
      << "NORMAL_RAG:\n" << normal_rag_context << "\n\n"
      << "FULL_HISTORY:\n" << full_history_context << "\n";

  const fs::path tmp_stem
      = cfg.output_path.parent_path ()
        / ("chat_replay_prompt_policy_judge_" + std::to_string (probe_index)
           );
  return CallJudgeModel (cfg, tmp_stem, prompt.str (), 384);
}

std::optional<nlohmann::json>
JudgeCompactPolicyContexts (const Config &cfg, int probe_index,
                            const std::string &query,
                            const std::string &cortext_ltm_context,
                            const std::string &stm_recent_context,
                            const std::string &ltm_lexical_context,
                            const std::string &stm_ltm_union_context,
                            const std::string &normal_rag_context,
                            const std::string &full_history_context)
{
  if (!cfg.judge_enabled || cfg.judge_model.empty ())
    return std::nullopt;

  std::ostringstream prompt;
  prompt
      << "You are judging compact memory prompt policies for a chat assistant.\n"
      << "Given the current user message and six candidate replacement context "
         "packets, score only how useful each packet is for answering or "
         "continuing the conversation.\n"
      << "Use 0-5 integer scores. relevance means contains information likely "
         "useful to this turn. sufficiency means enough context to answer. "
         "noise means distracting or excessive unrelated content, where 0 is "
         "clean and 5 is very noisy.\n"
      << "Return strict JSON only with this shape:\n"
      << "{\"cortext_ltm\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"stm_recent\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"ltm_lexical\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"stm_ltm_union\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"normal_rag\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"full_history\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"winner\":\"cortext_ltm|stm_recent|ltm_lexical|stm_ltm_union|normal_rag|full_history|tie\"}\n\n"
      << "CURRENT_USER_MESSAGE:\n" << query << "\n\n"
      << "CORTEXT_LTM_ONLY:\n" << cortext_ltm_context << "\n\n"
      << "STM_RECENT_REPLACEMENT:\n" << stm_recent_context << "\n\n"
      << "LTM_LEXICAL_REPLACEMENT:\n" << ltm_lexical_context << "\n\n"
      << "STM_LTM_UNION_REPLACEMENT:\n" << stm_ltm_union_context << "\n\n"
      << "NORMAL_RAG:\n" << normal_rag_context << "\n\n"
      << "FULL_HISTORY:\n" << full_history_context << "\n";

  const fs::path tmp_stem
      = cfg.output_path.parent_path ()
        / ("chat_replay_compact_policy_judge_" + std::to_string (probe_index)
           );
  return CallJudgeModel (cfg, tmp_stem, prompt.str (), 384);
}

std::optional<nlohmann::json>
JudgeStmGraphContexts (const Config &cfg, int probe_index,
                       const std::string &query,
                       const std::string &raw_stm_graph_context,
                       const std::string &relabel_prune_context,
                       const std::string &normal_rag_context,
                       const std::string &full_history_context)
{
  if (!cfg.judge_enabled || cfg.judge_model.empty ())
    return std::nullopt;

  std::ostringstream prompt;
  prompt
      << "You are judging memory context quality for a chat assistant.\n"
      << "Given the current user message and four candidate context packets, "
         "score only how useful each packet is for answering or continuing the "
         "conversation.\n"
      << "RAW_STM_GRAPH is bounded working memory plus source memories selected "
         "by provisional STM graph labels before relabel/prune. "
         "RELABEL_PRUNE_LTM is bounded working memory plus source memories "
         "selected by the durable labels after relabel/prune consolidation. "
         "NORMAL_RAG is rolling chat history plus lexical notes.\n"
      << "Use 0-5 integer scores. relevance means contains information likely "
         "useful to this turn. sufficiency means enough context to answer. "
         "noise means distracting or excessive unrelated content, where 0 is "
         "clean and 5 is very noisy.\n"
      << "Return strict JSON only with this shape:\n"
      << "{\"raw_stm_graph\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"relabel_prune_ltm\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"normal_rag\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"full_history\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0},"
         "\"winner\":\"raw_stm_graph|relabel_prune_ltm|normal_rag|full_history|tie\"}\n\n"
      << "CURRENT_USER_MESSAGE:\n" << query << "\n\n"
      << "RAW_STM_GRAPH:\n" << raw_stm_graph_context << "\n\n"
      << "RELABEL_PRUNE_LTM:\n" << relabel_prune_context << "\n\n"
      << "NORMAL_RAG:\n" << normal_rag_context << "\n\n"
      << "FULL_HISTORY:\n" << full_history_context << "\n";

  const fs::path tmp_stem
      = cfg.output_path.parent_path ()
        / ("chat_replay_stm_graph_judge_" + std::to_string (probe_index));
  return CallJudgeModel (cfg, tmp_stem, prompt.str (), 384);
}

std::optional<nlohmann::json>
JudgeGraphExpandedRagContexts (const Config &cfg, int probe_index,
                               const std::string &query,
                               const std::string &normal_rag_context,
                               const std::string &cortext_ltm_context,
                               const std::string &graph_expanded_rag_context,
                               const std::string &full_history_context)
{
  if (!cfg.judge_enabled || cfg.judge_model.empty ())
    return std::nullopt;

  std::ostringstream prompt;
  prompt
      << "You are judging memory context quality for a chat assistant.\n"
      << "Given the current user message and four candidate context packets, "
         "score only how useful each packet is for answering or continuing the "
         "conversation.\n"
      << "NORMAL_RAG is rolling chat history plus raw-chat vector notes. CORTEXT_LTM "
         "is the current Cortext long-term-memory packet. "
         "GRAPH_EXPANDED_RAG starts with source-memory hits, then adds "
         "source memories related by Cortext graph labels, relations, facts, "
         "and temporal neighbors.\n"
      << "Use 0-5 integer scores. relevance means contains information likely "
         "useful to this turn. sufficiency means enough context to answer. "
         "noise means distracting or excessive unrelated content, where 0 is "
         "clean and 5 is very noisy. source_grounding means the packet gives "
         "traceable source-backed evidence. temporal_correctness means the "
         "packet preserves ordering/date cues. media_usefulness means any media "
         "markers or media-derived context help the turn; use 0 when no useful "
         "media context appears.\n"
      << "Return strict JSON only with this shape:\n"
      << "{\"normal_rag\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0,"
         "\"source_grounding\":0,\"temporal_correctness\":0,\"media_usefulness\":0},"
         "\"cortext_ltm\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0,"
         "\"source_grounding\":0,\"temporal_correctness\":0,\"media_usefulness\":0},"
         "\"graph_expanded_rag\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0,"
         "\"source_grounding\":0,\"temporal_correctness\":0,\"media_usefulness\":0},"
         "\"full_history\":{\"relevance\":0,\"sufficiency\":0,\"noise\":0,"
         "\"source_grounding\":0,\"temporal_correctness\":0,\"media_usefulness\":0},"
         "\"winner\":\"normal_rag|cortext_ltm|graph_expanded_rag|full_history|tie\"}\n\n"
      << "CURRENT_USER_MESSAGE:\n" << query << "\n\n"
      << "NORMAL_RAG:\n" << normal_rag_context << "\n\n"
      << "CORTEXT_LTM:\n" << cortext_ltm_context << "\n\n"
      << "GRAPH_EXPANDED_RAG:\n" << graph_expanded_rag_context << "\n\n"
      << "FULL_HISTORY:\n" << full_history_context << "\n";

  const fs::path tmp_stem
      = cfg.output_path.parent_path ()
        / ("chat_replay_graph_expanded_rag_judge_" + std::to_string (probe_index));
  return CallJudgeModel (cfg, tmp_stem, prompt.str (), 384);
}

double
ScoreValue (const nlohmann::json &quality, const std::string &system,
            const std::string &field)
{
  if (!quality.contains (system) || !quality[system].is_object ()
      || !quality[system].contains (field))
    return 0.0;
  const auto &value = quality[system][field];
  if (value.is_number ())
    return std::clamp (value.get<double> (), 0.0, 5.0);
  if (value.is_string ())
    {
      try
        {
          return std::clamp (std::stod (value.get<std::string> ()), 0.0, 5.0);
        }
      catch (...)
        {
        }
    }
  return 0.0;
}

void
AddQuality (QualityAggregate &agg, const nlohmann::json &quality,
            const std::string &system)
{
  ++agg.judged;
  agg.relevance_sum += ScoreValue (quality, system, "relevance");
  agg.sufficiency_sum += ScoreValue (quality, system, "sufficiency");
  agg.noise_sum += ScoreValue (quality, system, "noise");
  agg.source_grounding_sum += ScoreValue (quality, system, "source_grounding");
  agg.temporal_correctness_sum += ScoreValue (quality, system,
                                              "temporal_correctness");
  agg.media_usefulness_sum += ScoreValue (quality, system, "media_usefulness");
  if (quality.contains ("winner") && quality["winner"].is_string ()
      && quality["winner"].get<std::string> () == system)
    ++agg.wins;
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

nlohmann::json
BuildAddressabilityProbe (cortext::Store &store, cortext::Cortext &engine,
                          const std::unordered_set<std::string> &query_tokens)
{
  auto summarize = [&query_tokens] (
                       const std::vector<std::map<std::string, std::any>> &rows) {
    nlohmann::json out;
    long long text_count = 0;
    long long text_with_query_overlap = 0;
    long long best_hit_count = 0;
    long long total_hit_count = 0;
    for (const auto &row : rows)
      {
        const auto text = AnyString (row, "text");
        if (text.empty ())
          continue;
        ++text_count;
        const auto text_tokens = Tokens (text);
        long long hits = 0;
        for (const auto &token : query_tokens)
          {
            if (text_tokens.find (token) != text_tokens.end ())
              ++hits;
          }
        if (hits > 0)
          ++text_with_query_overlap;
        best_hit_count = std::max (best_hit_count, hits);
        total_hit_count += hits;
      }
    out["text_count"] = text_count;
    out["texts_with_query_overlap"] = text_with_query_overlap;
    out["best_hit_count"] = best_hit_count;
    out["total_hit_count"] = total_hit_count;
    out["best_overlap"] = query_tokens.empty ()
                              ? 0.0
                              : static_cast<double> (best_hit_count)
                                    / static_cast<double> (query_tokens.size ());
    return out;
  };
  auto summarize_memory_ids = [&engine, &summarize] (
                                  const std::vector<long long> &ids) {
    std::vector<std::map<std::string, std::any>> rows;
    std::unordered_set<long long> seen;
    for (long long id : ids)
      {
        if (id <= 0 || !seen.insert (id).second)
          continue;
        const auto memories = HydrateMemories (engine, { id });
        for (const auto &memory : memories)
          {
            const std::string text = MemoryText (memory);
            if (!text.empty ())
              rows.push_back ({ { "text", text } });
          }
      }
    auto out = summarize (rows);
    out["source_memory_count"] = static_cast<long long> (seen.size ());
    return out;
  };
  auto collect_ids = [] (
                         const std::vector<std::map<std::string, std::any>> &rows,
                         const char *field) {
    std::vector<long long> ids;
    std::unordered_set<long long> seen;
    for (const auto &row : rows)
      {
        const long long id = AnyLongLong (row, field);
        if (id > 0 && seen.insert (id).second)
          ids.push_back (id);
      }
    return ids;
  };

  nlohmann::json out;
  try
    {
      const auto durable_label_rows = store.Execute (
          "SELECT DISTINCT COALESCE(lm.label, lm.source_id, '') AS text "
          "FROM memories lm "
          "JOIN associations hl ON hl.target_memory_id = lm.memory_id "
          "  AND hl.edge_type = 'has_label' "
          "JOIN memories cue ON cue.memory_id = hl.source_memory_id "
          "  AND cue.kind = 'ASSOCIATION' "
          "JOIN associations df ON df.source_memory_id = cue.memory_id "
          "  AND df.edge_type = 'derived_from' "
          "WHERE lm.kind = 'LABEL'",
          {});
      out["durable_labels"] = summarize (durable_label_rows);
    }
  catch (...)
    {
      out["durable_labels_error"] = true;
    }

  try
    {
      const auto source_rows = store.Execute (
          "SELECT DISTINCT df.target_memory_id AS source_memory_id "
          "FROM memories lm "
          "JOIN associations hl ON hl.target_memory_id = lm.memory_id "
          "  AND hl.edge_type = 'has_label' "
          "JOIN memories cue ON cue.memory_id = hl.source_memory_id "
          "  AND cue.kind = 'ASSOCIATION' "
          "JOIN associations df ON df.source_memory_id = cue.memory_id "
          "  AND df.edge_type = 'derived_from' "
          "WHERE lm.kind = 'LABEL'",
          {});
      out["durable_label_sources"] = summarize_memory_ids (
          collect_ids (source_rows, "source_memory_id"));
    }
  catch (...)
    {
      out["durable_label_sources_error"] = true;
    }

  try
    {
      const auto fact_rows = store.Execute (
          "SELECT subject || ' ' || predicate || ' ' || object AS text "
          "FROM fact_assertions",
          {});
      out["facts_all"] = summarize (fact_rows);
    }
  catch (...)
    {
      out["facts_all_error"] = true;
    }

  try
    {
      const auto fact_rows = store.Execute (
          "SELECT subject || ' ' || predicate || ' ' || object AS text "
          "FROM fact_assertions "
          "WHERE COALESCE(lifecycle_state, 'active') = 'active'",
          {});
      out["facts_active"] = summarize (fact_rows);
    }
  catch (...)
    {
      out["facts_active_error"] = true;
    }

  try
    {
      const auto source_rows = store.Execute (
          "SELECT DISTINCT fe.source_memory_id AS source_memory_id "
          "FROM fact_evidence fe "
          "JOIN fact_assertions fa ON fa.fact_id = fe.fact_id",
          {});
      out["fact_sources_all"] = summarize_memory_ids (
          collect_ids (source_rows, "source_memory_id"));
    }
  catch (...)
    {
      out["fact_sources_all_error"] = true;
    }

  try
    {
      const auto source_rows = store.Execute (
          "SELECT DISTINCT fe.source_memory_id AS source_memory_id "
          "FROM fact_evidence fe "
          "JOIN fact_assertions fa ON fa.fact_id = fe.fact_id "
          "WHERE COALESCE(fa.lifecycle_state, 'active') = 'active'",
          {});
      out["fact_sources_active"] = summarize_memory_ids (
          collect_ids (source_rows, "source_memory_id"));
    }
  catch (...)
    {
      out["fact_sources_active_error"] = true;
    }

  return out;
}

void
RefreshFactOnlyTable (cortext::Store &store)
{
  store.Execute ("DROP TABLE IF EXISTS temp.fact_only_embeddings");
  store.Execute (
      "CREATE VIRTUAL TABLE temp.fact_only_embeddings USING vec0("
      "fact_id INTEGER PRIMARY KEY, embedding float[256], "
      "+embedding_id INTEGER, +lifecycle_state TEXT, "
      "+recorded_at_ts INTEGER, +valid_start_ts INTEGER, "
      "+valid_end_ts INTEGER, +superseded_at_ts INTEGER, "
      "+evidence_count INTEGER)");
  auto rows = store.Execute (
      "SELECT fc.fact_id, fc.embedding_id, e.embedding, "
      "COALESCE(fa.lifecycle_state, 'active') AS lifecycle_state, "
      "fa.recorded_at_ts, fa.valid_start_ts, fa.valid_end_ts, "
      "fa.superseded_at_ts, COALESCE(ev.evidence_count, 0) AS evidence_count "
      "FROM fact_cache fc "
      "JOIN embeddings e ON e.embedding_id = fc.embedding_id "
      "JOIN fact_assertions fa ON fa.fact_id = fc.fact_id "
      "LEFT JOIN (SELECT fact_id, COUNT(*) AS evidence_count "
      "           FROM fact_evidence GROUP BY fact_id) ev "
      "ON ev.fact_id = fc.fact_id");
  for (const auto &row : rows)
    {
      auto it_embedding = row.find ("embedding");
      if (it_embedding == row.end () || !it_embedding->second.has_value ())
        continue;
      const auto embedding = cortext::store::BlobFromAny (it_embedding->second);
      if (embedding.empty ())
        continue;
      store.Execute (
          "INSERT INTO temp.fact_only_embeddings("
          "fact_id, embedding, embedding_id, lifecycle_state, recorded_at_ts, "
          "valid_start_ts, valid_end_ts, superseded_at_ts, evidence_count) "
          "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
          { AnyLongLong (row, "fact_id"), embedding,
            AnyLongLong (row, "embedding_id"),
            AnyString (row, "lifecycle_state"),
            AnyLongLong (row, "recorded_at_ts"),
            AnyLongLong (row, "valid_start_ts"),
            AnyLongLong (row, "valid_end_ts"),
            AnyLongLong (row, "superseded_at_ts"),
            AnyLongLong (row, "evidence_count") });
    }
}

long long
CountRows (cortext::Store &store, const std::string &table)
{
  const auto rows = store.Execute ("SELECT COUNT(*) AS n FROM " + table);
  if (rows.empty ())
    return 0;
  return AnyLongLong (rows[0], "n");
}

bool
TableExists (cortext::Store &store, const std::string &table)
{
  const auto rows = store.Execute (
      "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
      { table });
  return !rows.empty ();
}

nlohmann::json
BuildStmLtmAuditSummary (cortext::Store &store)
{
  nlohmann::json out;
  out["enabled"] = TableExists (store, "stm_ltm_relabel_audit");
  if (!out["enabled"].get<bool> ())
    {
      return out;
    }

  auto totals = store.Execute (
      "SELECT COUNT(*) AS cycles, "
      "COALESCE(SUM(source_memory_count), 0) AS source_memory_count, "
      "COALESCE(SUM(source_text_count), 0) AS source_text_count, "
      "COALESCE(SUM(source_blob_count), 0) AS source_blob_count, "
      "COALESCE(SUM(stm_item_count), 0) AS stm_item_count, "
      "COALESCE(SUM(stm_label_edge_count), 0) AS stm_label_edge_count, "
      "COALESCE(SUM(current_label_count), 0) AS current_label_count, "
      "COALESCE(SUM(refined_label_count), 0) AS refined_label_count, "
      "COALESCE(SUM(kept_label_count), 0) AS kept_label_count, "
      "COALESCE(SUM(added_label_count), 0) AS added_label_count, "
      "COALESCE(SUM(removed_label_count), 0) AS removed_label_count, "
      "COALESCE(SUM(current_labels_in_selected_evidence), 0) AS "
      "current_labels_in_selected_evidence, "
      "COALESCE(SUM(current_labels_in_full_source), 0) AS "
      "current_labels_in_full_source, "
      "COALESCE(SUM(removed_labels_in_selected_evidence), 0) AS "
      "removed_labels_in_selected_evidence, "
      "COALESCE(SUM(removed_labels_in_full_source), 0) AS "
      "removed_labels_in_full_source, "
      "COALESCE(SUM(refined_labels_in_selected_evidence), 0) AS "
      "refined_labels_in_selected_evidence, "
      "COALESCE(SUM(refined_labels_in_full_source), 0) AS "
      "refined_labels_in_full_source, "
      "COALESCE(SUM(extraction_label_candidate_count), 0) AS "
      "extraction_label_candidate_count, "
      "COALESCE(SUM(extraction_relation_candidate_count), 0) AS "
      "extraction_relation_candidate_count, "
      "COALESCE(SUM(source_span_candidate_count), 0) AS "
      "source_span_candidate_count, "
      "COALESCE(SUM(label_candidates_rejected_non_durable), 0) AS "
      "label_candidates_rejected_non_durable, "
      "COALESCE(SUM(label_candidates_rejected_ungrounded), 0) AS "
      "label_candidates_rejected_ungrounded, "
      "COALESCE(SUM(label_candidates_rejected_duplicate), 0) AS "
      "label_candidates_rejected_duplicate, "
      "COALESCE(SUM(label_candidates_rejected_legacy_gate), 0) AS "
      "label_candidates_rejected_legacy_gate, "
      "COALESCE(SUM(labels_inserted_from_extractor), 0) AS "
      "labels_inserted_from_extractor, "
      "COALESCE(SUM(labels_inserted_from_current_floor), 0) AS "
      "labels_inserted_from_current_floor, "
      "COALESCE(SUM(labels_inserted_from_source_span_floor), 0) AS "
      "labels_inserted_from_source_span_floor, "
      "COALESCE(SUM(labels_inserted_from_relation_endpoint), 0) AS "
      "labels_inserted_from_relation_endpoint, "
      "COALESCE(SUM(has_label_edges_after), 0) AS has_label_edges_after, "
      "COALESCE(SUM(derived_from_edges), 0) AS derived_from_edges, "
      "COALESCE(SUM(durable_ltm_nodes_with_source), 0) AS "
      "durable_ltm_nodes_with_source, "
      "COALESCE(SUM(durable_ltm_nodes_missing_source), 0) AS "
      "durable_ltm_nodes_missing_source, "
      "COALESCE(SUM(durable_ltm_source_link_pairs), 0) AS "
      "durable_ltm_source_link_pairs, "
      "COALESCE(SUM(relation_count), 0) AS relation_count, "
      "COALESCE(SUM(relation_edges_created), 0) AS relation_edges_created, "
      "COALESCE(SUM(label_cooccurrence_edges_created), 0) AS "
      "label_cooccurrence_edges_created, "
      "COALESCE(SUM(relation_edges_skipped_non_durable_endpoint), 0) AS "
      "relation_edges_skipped_non_durable_endpoint, "
      "COALESCE(SUM(relation_edges_skipped_missing_endpoint), 0) AS "
      "relation_edges_skipped_missing_endpoint, "
      "COALESCE(SUM(relation_edges_skipped_unsupported_predicate), 0) AS "
      "relation_edges_skipped_unsupported_predicate, "
      "COALESCE(SUM(relation_endpoint_direct_hits), 0) AS "
      "relation_endpoint_direct_hits, "
      "COALESCE(SUM(relation_endpoint_repair_hits), 0) AS "
      "relation_endpoint_repair_hits, "
      "COALESCE(SUM(relation_endpoint_created_labels), 0) AS "
      "relation_endpoint_created_labels, "
      "COALESCE(SUM(relation_endpoint_relation_backed_labels), 0) AS "
      "relation_endpoint_relation_backed_labels, "
      "COALESCE(SUM(relation_endpoint_rejected_count), 0) AS "
      "relation_endpoint_rejected_count, "
      "COALESCE(SUM(relation_endpoint_rejected_non_durable), 0) AS "
      "relation_endpoint_rejected_non_durable, "
      "COALESCE(SUM(relation_endpoint_rejected_ungrounded), 0) AS "
      "relation_endpoint_rejected_ungrounded, "
      "COALESCE(SUM(fact_assertions_touched), 0) AS fact_assertions_touched, "
      "COALESCE(SUM(source_memories_with_content), 0) AS source_memories_with_content "
      "FROM stm_ltm_relabel_audit");
  if (!totals.empty ())
    {
      const auto &row = totals[0];
      out["cycles"] = AnyLongLong (row, "cycles");
      out["source_memory_count"] = AnyLongLong (row, "source_memory_count");
      out["source_text_count"] = AnyLongLong (row, "source_text_count");
      out["source_blob_count"] = AnyLongLong (row, "source_blob_count");
      out["stm_item_count"] = AnyLongLong (row, "stm_item_count");
      out["stm_label_edge_count"] = AnyLongLong (row, "stm_label_edge_count");
      out["current_label_count"] = AnyLongLong (row, "current_label_count");
      out["refined_label_count"] = AnyLongLong (row, "refined_label_count");
      out["kept_label_count"] = AnyLongLong (row, "kept_label_count");
      out["added_label_count"] = AnyLongLong (row, "added_label_count");
      out["removed_label_count"] = AnyLongLong (row, "removed_label_count");
      out["current_labels_in_selected_evidence"]
          = AnyLongLong (row, "current_labels_in_selected_evidence");
      out["current_labels_in_full_source"]
          = AnyLongLong (row, "current_labels_in_full_source");
      out["removed_labels_in_selected_evidence"]
          = AnyLongLong (row, "removed_labels_in_selected_evidence");
      out["removed_labels_in_full_source"]
          = AnyLongLong (row, "removed_labels_in_full_source");
      out["refined_labels_in_selected_evidence"]
          = AnyLongLong (row, "refined_labels_in_selected_evidence");
      out["refined_labels_in_full_source"]
          = AnyLongLong (row, "refined_labels_in_full_source");
      out["extraction_label_candidate_count"]
          = AnyLongLong (row, "extraction_label_candidate_count");
      out["extraction_relation_candidate_count"]
          = AnyLongLong (row, "extraction_relation_candidate_count");
      out["source_span_candidate_count"]
          = AnyLongLong (row, "source_span_candidate_count");
      out["label_candidates_rejected_non_durable"]
          = AnyLongLong (row, "label_candidates_rejected_non_durable");
      out["label_candidates_rejected_ungrounded"]
          = AnyLongLong (row, "label_candidates_rejected_ungrounded");
      out["label_candidates_rejected_duplicate"]
          = AnyLongLong (row, "label_candidates_rejected_duplicate");
      out["label_candidates_rejected_legacy_gate"]
          = AnyLongLong (row, "label_candidates_rejected_legacy_gate");
      out["labels_inserted_from_extractor"]
          = AnyLongLong (row, "labels_inserted_from_extractor");
      out["labels_inserted_from_current_floor"]
          = AnyLongLong (row, "labels_inserted_from_current_floor");
      out["labels_inserted_from_source_span_floor"]
          = AnyLongLong (row, "labels_inserted_from_source_span_floor");
      out["labels_inserted_from_relation_endpoint"]
          = AnyLongLong (row, "labels_inserted_from_relation_endpoint");
      out["has_label_edges_after"] = AnyLongLong (row, "has_label_edges_after");
      out["derived_from_edges"] = AnyLongLong (row, "derived_from_edges");
      out["durable_ltm_nodes_with_source"]
          = AnyLongLong (row, "durable_ltm_nodes_with_source");
      out["durable_ltm_nodes_missing_source"]
          = AnyLongLong (row, "durable_ltm_nodes_missing_source");
      out["durable_ltm_source_link_pairs"]
          = AnyLongLong (row, "durable_ltm_source_link_pairs");
      out["relation_count"] = AnyLongLong (row, "relation_count");
      out["relation_edges_created"] = AnyLongLong (row, "relation_edges_created");
      out["label_cooccurrence_edges_created"]
          = AnyLongLong (row, "label_cooccurrence_edges_created");
      out["relation_edges_skipped_non_durable_endpoint"]
          = AnyLongLong (row,
                         "relation_edges_skipped_non_durable_endpoint");
      out["relation_edges_skipped_missing_endpoint"]
          = AnyLongLong (row, "relation_edges_skipped_missing_endpoint");
      out["relation_edges_skipped_unsupported_predicate"]
          = AnyLongLong (row, "relation_edges_skipped_unsupported_predicate");
      out["relation_endpoint_direct_hits"]
          = AnyLongLong (row, "relation_endpoint_direct_hits");
      out["relation_endpoint_repair_hits"]
          = AnyLongLong (row, "relation_endpoint_repair_hits");
      out["relation_endpoint_created_labels"]
          = AnyLongLong (row, "relation_endpoint_created_labels");
      out["relation_endpoint_relation_backed_labels"]
          = AnyLongLong (row, "relation_endpoint_relation_backed_labels");
      out["relation_endpoint_rejected_count"]
          = AnyLongLong (row, "relation_endpoint_rejected_count");
      out["relation_endpoint_rejected_non_durable"]
          = AnyLongLong (row, "relation_endpoint_rejected_non_durable");
      out["relation_endpoint_rejected_ungrounded"]
          = AnyLongLong (row, "relation_endpoint_rejected_ungrounded");
      out["fact_assertions_touched"] = AnyLongLong (row, "fact_assertions_touched");
      out["source_memories_with_content"]
          = AnyLongLong (row, "source_memories_with_content");
    }

  nlohmann::json cycles = nlohmann::json::array ();
  auto cycle_rows = store.Execute (
      "SELECT summary_id, cluster_size, source_memory_count, "
      "stm_graph_count, stm_item_count, stm_label_edge_count, "
      "current_label_count, refined_label_count, kept_label_count, "
      "added_label_count, removed_label_count, has_label_edges_after, "
      "current_labels_in_selected_evidence, "
      "current_labels_in_full_source, "
      "removed_labels_in_selected_evidence, "
      "removed_labels_in_full_source, "
      "refined_labels_in_selected_evidence, "
      "refined_labels_in_full_source, "
      "extraction_label_candidate_count, "
      "extraction_relation_candidate_count, "
      "source_span_candidate_count, "
      "label_candidates_rejected_non_durable, "
      "label_candidates_rejected_ungrounded, "
      "label_candidates_rejected_duplicate, "
      "label_candidates_rejected_legacy_gate, "
      "labels_inserted_from_extractor, "
      "labels_inserted_from_current_floor, "
      "labels_inserted_from_source_span_floor, "
      "labels_inserted_from_relation_endpoint, "
      "derived_from_edges, durable_ltm_nodes_with_source, "
      "durable_ltm_nodes_missing_source, durable_ltm_source_link_pairs, "
      "relation_count, relation_edges_created, "
      "label_cooccurrence_edges_created, "
      "relation_edges_skipped_non_durable_endpoint, "
      "relation_edges_skipped_missing_endpoint, "
      "relation_edges_skipped_unsupported_predicate, "
      "relation_endpoint_direct_hits, relation_endpoint_repair_hits, "
      "relation_endpoint_created_labels, "
      "relation_endpoint_relation_backed_labels, "
      "relation_endpoint_rejected_count, "
      "relation_endpoint_rejected_non_durable, "
      "relation_endpoint_rejected_ungrounded, "
      "fact_assertions_touched, "
      "source_memories_with_content "
      "FROM stm_ltm_relabel_audit ORDER BY created_at, summary_id");
  for (const auto &row : cycle_rows)
    {
      nlohmann::json cycle;
      cycle["summary_id"] = AnyString (row, "summary_id");
      cycle["cluster_size"] = AnyLongLong (row, "cluster_size");
      cycle["source_memory_count"] = AnyLongLong (row, "source_memory_count");
      cycle["stm_graph_count"] = AnyLongLong (row, "stm_graph_count");
      cycle["stm_item_count"] = AnyLongLong (row, "stm_item_count");
      cycle["stm_label_edge_count"] = AnyLongLong (row, "stm_label_edge_count");
      cycle["current_label_count"] = AnyLongLong (row, "current_label_count");
      cycle["refined_label_count"] = AnyLongLong (row, "refined_label_count");
      cycle["kept_label_count"] = AnyLongLong (row, "kept_label_count");
      cycle["added_label_count"] = AnyLongLong (row, "added_label_count");
      cycle["removed_label_count"] = AnyLongLong (row, "removed_label_count");
      cycle["current_labels_in_selected_evidence"]
          = AnyLongLong (row, "current_labels_in_selected_evidence");
      cycle["current_labels_in_full_source"]
          = AnyLongLong (row, "current_labels_in_full_source");
      cycle["removed_labels_in_selected_evidence"]
          = AnyLongLong (row, "removed_labels_in_selected_evidence");
      cycle["removed_labels_in_full_source"]
          = AnyLongLong (row, "removed_labels_in_full_source");
      cycle["refined_labels_in_selected_evidence"]
          = AnyLongLong (row, "refined_labels_in_selected_evidence");
      cycle["refined_labels_in_full_source"]
          = AnyLongLong (row, "refined_labels_in_full_source");
      cycle["extraction_label_candidate_count"]
          = AnyLongLong (row, "extraction_label_candidate_count");
      cycle["extraction_relation_candidate_count"]
          = AnyLongLong (row, "extraction_relation_candidate_count");
      cycle["source_span_candidate_count"]
          = AnyLongLong (row, "source_span_candidate_count");
      cycle["label_candidates_rejected_non_durable"]
          = AnyLongLong (row, "label_candidates_rejected_non_durable");
      cycle["label_candidates_rejected_ungrounded"]
          = AnyLongLong (row, "label_candidates_rejected_ungrounded");
      cycle["label_candidates_rejected_duplicate"]
          = AnyLongLong (row, "label_candidates_rejected_duplicate");
      cycle["label_candidates_rejected_legacy_gate"]
          = AnyLongLong (row, "label_candidates_rejected_legacy_gate");
      cycle["labels_inserted_from_extractor"]
          = AnyLongLong (row, "labels_inserted_from_extractor");
      cycle["labels_inserted_from_current_floor"]
          = AnyLongLong (row, "labels_inserted_from_current_floor");
      cycle["labels_inserted_from_source_span_floor"]
          = AnyLongLong (row, "labels_inserted_from_source_span_floor");
      cycle["labels_inserted_from_relation_endpoint"]
          = AnyLongLong (row, "labels_inserted_from_relation_endpoint");
      cycle["has_label_edges_after"] = AnyLongLong (row, "has_label_edges_after");
      cycle["derived_from_edges"] = AnyLongLong (row, "derived_from_edges");
      cycle["durable_ltm_nodes_with_source"]
          = AnyLongLong (row, "durable_ltm_nodes_with_source");
      cycle["durable_ltm_nodes_missing_source"]
          = AnyLongLong (row, "durable_ltm_nodes_missing_source");
      cycle["durable_ltm_source_link_pairs"]
          = AnyLongLong (row, "durable_ltm_source_link_pairs");
      cycle["relation_count"] = AnyLongLong (row, "relation_count");
      cycle["relation_edges_created"] = AnyLongLong (row, "relation_edges_created");
      cycle["label_cooccurrence_edges_created"]
          = AnyLongLong (row, "label_cooccurrence_edges_created");
      cycle["relation_edges_skipped_non_durable_endpoint"]
          = AnyLongLong (row,
                         "relation_edges_skipped_non_durable_endpoint");
      cycle["relation_edges_skipped_missing_endpoint"]
          = AnyLongLong (row, "relation_edges_skipped_missing_endpoint");
      cycle["relation_edges_skipped_unsupported_predicate"]
          = AnyLongLong (row, "relation_edges_skipped_unsupported_predicate");
      cycle["relation_endpoint_direct_hits"]
          = AnyLongLong (row, "relation_endpoint_direct_hits");
      cycle["relation_endpoint_repair_hits"]
          = AnyLongLong (row, "relation_endpoint_repair_hits");
      cycle["relation_endpoint_created_labels"]
          = AnyLongLong (row, "relation_endpoint_created_labels");
      cycle["relation_endpoint_relation_backed_labels"]
          = AnyLongLong (row, "relation_endpoint_relation_backed_labels");
      cycle["relation_endpoint_rejected_count"]
          = AnyLongLong (row, "relation_endpoint_rejected_count");
      cycle["relation_endpoint_rejected_non_durable"]
          = AnyLongLong (row, "relation_endpoint_rejected_non_durable");
      cycle["relation_endpoint_rejected_ungrounded"]
          = AnyLongLong (row, "relation_endpoint_rejected_ungrounded");
      cycle["fact_assertions_touched"] = AnyLongLong (row, "fact_assertions_touched");
      cycle["source_memories_with_content"]
          = AnyLongLong (row, "source_memories_with_content");
      cycles.push_back (std::move (cycle));
    }
  out["cycle_summaries"] = std::move (cycles);
  return out;
}

FactPromptProbe
BuildFactPromptProbe (cortext::Store &store, std::uint64_t timestamp,
                      int fact_k)
{
  FactPromptProbe probe;
  RefreshFactOnlyTable (store);
  probe.source_fact_cache_rows = CountRows (store, "fact_cache");
  probe.fact_only_rows = CountRows (store, "temp.fact_only_embeddings");
  auto query_rows = store.Execute (
      "SELECT e.embedding FROM signals s "
      "JOIN embeddings e ON e.embedding_id = s.embedding_id "
      "WHERE s.timestamp = ? ORDER BY s.signal_id DESC LIMIT 1",
      { static_cast<long long> (timestamp) });
  probe.query_rows = static_cast<long long> (query_rows.size ());
  if (query_rows.empty ())
    return probe;
  const auto it_embedding = query_rows[0].find ("embedding");
  if (it_embedding == query_rows[0].end () || !it_embedding->second.has_value ())
    return probe;
  const auto query_embedding = cortext::store::BlobFromAny (it_embedding->second);
  probe.query_embedding_bytes = static_cast<long long> (query_embedding.size ());
  if (query_embedding.empty ())
    return probe;

  const long long fact_search_k = std::max<long long> (
      std::max (1, fact_k), std::max<long long> (1, probe.fact_only_rows));
  auto fact_rows = store.Execute (
      "SELECT fact_id, distance, evidence_count, lifecycle_state, "
      "recorded_at_ts "
      "FROM temp.fact_only_embeddings "
      "WHERE embedding MATCH ? AND k = ? ORDER BY distance ASC",
      { query_embedding, fact_search_k });
  for (const auto &row : fact_rows)
    {
      const long long fact_id = AnyLongLong (row, "fact_id");
      if (fact_id <= 0)
        continue;
      const long long recorded_at_ts = AnyLongLong (row, "recorded_at_ts");
      if (recorded_at_ts > 0
          && recorded_at_ts > static_cast<long long> (timestamp))
        continue;
      if (AnyString (row, "lifecycle_state") == "archived")
        continue;
      if (probe.fact_ids.empty ())
        probe.top_fact_distance = AnyDouble (row, "distance");
      probe.fact_ids.push_back (fact_id);
      if (static_cast<int> (probe.fact_ids.size ()) >= std::max (1, fact_k))
        break;
    }
  if (probe.fact_ids.empty ())
    return probe;

  std::ostringstream sql;
  sql << "SELECT fe.fact_id, fe.source_memory_id, fe.support_weight "
         "FROM fact_evidence fe WHERE fe.fact_id IN (";
  std::vector<std::any> params;
  for (std::size_t i = 0; i < probe.fact_ids.size (); ++i)
    {
      if (i > 0)
        sql << ",";
      sql << "?";
      params.push_back (probe.fact_ids[i]);
    }
  sql << ") ORDER BY fe.fact_id, fe.support_weight DESC, fe.source_memory_id";
  auto evidence_rows = store.Execute (sql.str (), params);
  std::unordered_set<long long> seen;
  for (const auto &row : evidence_rows)
    {
      const long long memory_id = AnyLongLong (row, "source_memory_id");
      if (memory_id <= 0 || !seen.insert (memory_id).second)
        continue;
      probe.evidence_memory_ids.push_back (memory_id);
    }
  return probe;
}

std::vector<cortext::Cortext::Context::Memory>
HydrateMemories (cortext::Cortext &engine, const std::vector<long long> &ids)
{
#if defined(CORTEXT_TESTING)
  return engine.DebugHydrateForTest (ids, ids).retrieved_memory;
#else
  (void)engine;
  (void)ids;
  return {};
#endif
}

std::vector<long long>
TopMemoryIdsFromDebug (std::size_t max_count)
{
  std::vector<long long> ids;
  std::unordered_set<long long> seen;
  for (const auto &candidate :
       cortext::operations::retrieval_debug::GetLastRankedCandidates ())
    {
      if (candidate.memory_id <= 0 || !seen.insert (candidate.memory_id).second)
        continue;
      ids.push_back (candidate.memory_id);
      if (ids.size () >= max_count)
        break;
    }
  return ids;
}

std::vector<long long>
TakeUnique (const std::vector<long long> &ids, std::size_t max_count)
{
  std::vector<long long> out;
  std::unordered_set<long long> seen;
  for (long long id : ids)
    {
      if (id <= 0 || !seen.insert (id).second)
        continue;
      out.push_back (id);
      if (out.size () >= max_count)
        break;
    }
  return out;
}

std::vector<long long>
UnionThenExistingRank (const std::vector<long long> &fact_ids,
                       const std::vector<long long> &current_ids,
                       std::size_t max_count)
{
  std::vector<long long> out;
  std::unordered_set<long long> seen;
  for (long long id : fact_ids)
    {
      if (id > 0 && seen.insert (id).second)
        out.push_back (id);
      if (out.size () >= max_count)
        return out;
    }
  for (long long id : current_ids)
    {
      if (id > 0 && seen.insert (id).second)
        out.push_back (id);
      if (out.size () >= max_count)
        return out;
    }
  return out;
}

nlohmann::json
MemoryIdsJson (const std::vector<long long> &ids)
{
  nlohmann::json out = nlohmann::json::array ();
  for (long long id : ids)
    out.push_back (id);
  return out;
}

std::vector<long long>
MemoryIdsFromContext (
    const std::vector<cortext::Cortext::Context::Memory> &memories)
{
  std::vector<long long> ids;
  ids.reserve (memories.size ());
  for (const auto &memory : memories)
    {
      if (memory.id > 0)
        ids.push_back (memory.id);
    }
  return ids;
}

std::vector<long long>
LtmSourceIdsFromRankedCandidates (cortext::Store &store,
                                  std::size_t max_count,
                                  int *durable_candidate_count,
                                  int *fact_linked_candidate_count)
{
  std::vector<long long> out;
  if (durable_candidate_count)
    *durable_candidate_count = 0;
  if (fact_linked_candidate_count)
    *fact_linked_candidate_count = 0;
  if (max_count == 0)
    return out;

  std::unordered_set<long long> seen;
  for (const auto &candidate :
       cortext::operations::retrieval_debug::GetLastRankedCandidates ())
    {
      if ((candidate.linked_fact_count > 0 || candidate.fact_boost > 0.0)
          && candidate.memory_id > 0)
        {
          if (fact_linked_candidate_count)
            ++(*fact_linked_candidate_count);
          if (seen.insert (candidate.memory_id).second)
            {
              out.push_back (candidate.memory_id);
              if (out.size () >= max_count)
                break;
            }
        }
    }
  if (out.size () >= max_count)
    return out;

  for (const auto &candidate :
       cortext::operations::retrieval_debug::GetLastRankedCandidates ())
    {
      if (candidate.memory_id <= 0 || candidate.durable_source_boost <= 0.0)
        continue;
      if (durable_candidate_count)
        ++(*durable_candidate_count);

      auto rows = store.Execute (
          "WITH candidate(id) AS (VALUES (?)), source_links AS ("
          "  SELECT c.id AS candidate_memory_id, "
          "         df.target_memory_id AS source_memory_id, "
          "         MAX(df.weight) AS link_weight "
          "  FROM candidate c "
          "  JOIN memories cm ON cm.memory_id = c.id "
          "  JOIN associations df ON df.source_memory_id = c.id "
          "    AND df.edge_type = 'derived_from' "
          "  WHERE cm.kind = 'ASSOCIATION' "
          "  GROUP BY c.id, df.target_memory_id "
          "  UNION ALL "
          "  SELECT c.id AS candidate_memory_id, "
          "         df.target_memory_id AS source_memory_id, "
          "         MAX(hl.weight * df.weight) AS link_weight "
          "  FROM candidate c "
          "  JOIN memories cm ON cm.memory_id = c.id "
          "  JOIN associations hl ON hl.target_memory_id = c.id "
          "    AND hl.edge_type = 'has_label' "
          "  JOIN associations df ON df.source_memory_id = hl.source_memory_id "
          "    AND df.edge_type = 'derived_from' "
          "  WHERE cm.kind = 'LABEL' "
          "  GROUP BY c.id, df.target_memory_id "
          ") "
          "SELECT source_memory_id FROM source_links "
          "ORDER BY link_weight DESC, source_memory_id DESC",
          { candidate.memory_id });
      for (const auto &row : rows)
        {
          const long long id = AnyLongLong (row, "source_memory_id");
          if (id <= 0 || !seen.insert (id).second)
            continue;
          out.push_back (id);
          if (out.size () >= max_count)
            return out;
        }
    }
  return out;
}

long long
MemoryIdForRagDoc (cortext::Store &store, const RagDoc &doc)
{
  auto rows = store.Execute (
      "SELECT memory_id FROM memories "
      "WHERE source_id = ? AND start_ts = ? "
      "  AND kind NOT IN ('WORKING', 'LABEL', 'ASSOCIATION') "
      "ORDER BY memory_id DESC LIMIT 1",
      { doc.source_id, static_cast<long long> (doc.timestamp) });
  if (rows.empty ())
    return 0;
  return AnyLongLong (rows[0], "memory_id");
}

std::string
RagDocKey (const std::string &source_id, std::uint64_t timestamp)
{
  return source_id + "\n" + std::to_string (timestamp);
}

VectorRagPacket
BuildVectorRagPacket (cortext::Store &store, const std::vector<RagDoc> &all_docs,
                      const std::vector<float> &query_embedding,
                      std::uint64_t query_ts, double F, double T,
                      std::size_t requested_max_items)
{
  VectorRagPacket packet;
  std::unordered_map<std::string, RagDoc> docs_by_key;
  docs_by_key.reserve (all_docs.size ());
  for (const auto &doc : all_docs)
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

  const long long vector_search_k = std::max<long long> (
      static_cast<long long> (std::max (1, cortext::core::MaxResults (F))),
      static_cast<long long> (std::max<std::size_t> (1, requested_max_items)));
  packet.vector_search_k = std::min (vector_search_k, packet.prior_chat_rows);

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
      if (packet.docs.size () >= requested_max_items)
        break;
    }
  return packet;
}

void
AddScoredMemory (std::unordered_map<long long, double> &scores,
                 long long memory_id, double score)
{
  if (memory_id <= 0 || score <= 0.0)
    return;
  auto it = scores.find (memory_id);
  if (it == scores.end ())
    scores.emplace (memory_id, score);
  else
    it->second += score;
}

std::vector<long long>
RankScoredMemoryIds (const std::unordered_map<long long, double> &scores,
                     std::size_t max_count)
{
  std::vector<std::pair<long long, double>> ranked (scores.begin (),
                                                    scores.end ());
  std::sort (ranked.begin (), ranked.end (),
             [] (const auto &lhs, const auto &rhs) {
               if (lhs.second == rhs.second)
                 return lhs.first > rhs.first;
               return lhs.second > rhs.second;
             });
  std::vector<long long> ids;
  ids.reserve (std::min (ranked.size (), max_count));
  for (const auto &[id, score] : ranked)
    {
      (void)score;
      if (ids.size () >= max_count)
        break;
      ids.push_back (id);
    }
  return ids;
}

std::string
BuildGraphExpandedRagContext (
    const std::vector<RagDoc> &docs,
    const std::string &compact_context, std::size_t max_memories)
{
  std::ostringstream out;
  if (!compact_context.empty ())
    out << compact_context;
  std::size_t emitted = 0;
  for (const auto &doc : docs)
    {
      if (emitted >= max_memories)
        break;
      if (doc.text.empty ())
        continue;
      out << "[graph_expanded_rag source=" << doc.source_id << " ts="
          << doc.timestamp << "] " << doc.text << "\n";
      ++emitted;
    }
  return out.str ();
}

GraphExpandedRagPacket
BuildGraphExpandedRagPacket (cortext::Store &store, cortext::Cortext &engine,
                             const std::vector<RagDoc> &all_docs,
                             const std::vector<RagDoc> &lexical_top,
                             const std::unordered_set<std::string> &query_tokens,
                             double F, double S, double T,
                             std::size_t requested_max_memories)
{
  GraphExpandedRagPacket packet;
  const int knob_max = std::max (
      1, cortext::core::RetrievalGraphExpandedRagMaxItems (F, T));
  const std::size_t max_memories = std::max<std::size_t> (
      1, requested_max_memories > 0
             ? std::min<std::size_t> (requested_max_memories,
                                      static_cast<std::size_t> (knob_max))
             : static_cast<std::size_t> (knob_max));
  const int temporal_window = std::max (
      0, cortext::core::RetrievalGraphExpandedRagTemporalWindow (F, T));
  const int compact_limit = std::max (
      0, cortext::core::RetrievalGraphExpandedRagCompactItemLimit (F, T));
  const double seed_weight
      = cortext::core::RetrievalGraphExpandedRagSeedWeight (F, S, T);
  const double graph_weight
      = cortext::core::RetrievalGraphExpandedRagGraphWeight (F, S, T);
  const double relation_weight
      = cortext::core::RetrievalGraphExpandedRagRelationWeight (F, S, T);
  const double temporal_weight
      = cortext::core::RetrievalGraphExpandedRagTemporalWeight (F, S, T);
  const double fact_weight
      = cortext::core::RetrievalGraphExpandedRagFactWeight (F, S, T);

  std::unordered_map<int, long long> index_to_memory_id;
  std::unordered_map<long long, RagDoc> memory_id_to_doc;
  for (const auto &doc : all_docs)
    {
      const long long memory_id = MemoryIdForRagDoc (store, doc);
      if (memory_id > 0)
        {
          index_to_memory_id[doc.index] = memory_id;
          memory_id_to_doc[memory_id] = doc;
        }
    }
  std::unordered_map<long long, double> scores;
  for (std::size_t rank = 0; rank < lexical_top.size (); ++rank)
    {
      const auto &doc = lexical_top[rank];
      long long memory_id = 0;
      auto mapped_seed = index_to_memory_id.find (doc.index);
      if (mapped_seed != index_to_memory_id.end ())
        memory_id = mapped_seed->second;
      if (memory_id <= 0)
        {
          ++packet.missing_seed_count;
          continue;
        }
      packet.seed_memory_ids.push_back (memory_id);
      index_to_memory_id[doc.index] = memory_id;
      AddScoredMemory (scores, memory_id,
                       seed_weight
                           * (1.0 / (1.0 + static_cast<double> (rank))));

      if (temporal_window > 0)
        {
          for (const auto &candidate : all_docs)
            {
              const int distance = std::abs (candidate.index - doc.index);
              if (distance == 0 || distance > temporal_window)
                continue;
              long long neighbor_id = 0;
              auto it = index_to_memory_id.find (candidate.index);
              if (it != index_to_memory_id.end ())
                neighbor_id = it->second;
              else
                {
                  neighbor_id = MemoryIdForRagDoc (store, candidate);
                  if (neighbor_id > 0)
                    {
                      index_to_memory_id[candidate.index] = neighbor_id;
                      memory_id_to_doc[neighbor_id] = candidate;
                    }
                }
              if (neighbor_id <= 0)
                continue;
              ++packet.temporal_neighbor_count;
              AddScoredMemory (scores, neighbor_id,
                               temporal_weight
                                   / static_cast<double> (distance + 1));
            }
        }
    }

  if (!packet.seed_memory_ids.empty ())
    {
      std::ostringstream values;
      std::vector<std::any> params;
      for (std::size_t i = 0; i < packet.seed_memory_ids.size (); ++i)
        {
          if (i > 0)
            values << ",";
          values << "(?,?)";
          params.push_back (packet.seed_memory_ids[i]);
          params.push_back (1.0 / (1.0 + static_cast<double> (i)));
        }

      {
        std::ostringstream sql;
        sql
            << "WITH seed(id, seed_score) AS (VALUES " << values.str ()
            << "), source_cue AS ("
               "  SELECT df.source_memory_id AS cue_id, "
               "         MAX(seed.seed_score * df.weight) AS score "
               "  FROM seed "
               "  JOIN associations df ON df.target_memory_id = seed.id "
               "    AND df.edge_type = 'derived_from' "
               "  JOIN memories cue ON cue.memory_id = df.source_memory_id "
               "    AND cue.kind = 'ASSOCIATION' "
               "  GROUP BY df.source_memory_id "
               "), seed_label AS ("
               "  SELECT hl.target_memory_id AS label_id, "
               "         MAX(source_cue.score * hl.weight) AS score "
               "  FROM source_cue "
               "  JOIN associations hl ON hl.source_memory_id = source_cue.cue_id "
               "    AND hl.edge_type = 'has_label' "
               "  JOIN memories label ON label.memory_id = hl.target_memory_id "
               "    AND label.kind = 'LABEL' "
               "  GROUP BY hl.target_memory_id "
               "), related_label AS ("
               "  SELECT label_id, score FROM seed_label "
               "  UNION ALL "
               "  SELECT a.target_memory_id AS label_id, "
               "         seed_label.score * a.weight * ? AS score "
               "  FROM seed_label "
               "  JOIN associations a ON a.source_memory_id = seed_label.label_id "
               "  JOIN memories target_label ON target_label.memory_id = a.target_memory_id "
               "    AND target_label.kind = 'LABEL' "
               "  WHERE a.edge_type NOT IN ('has_label', 'derived_from') "
               "  UNION ALL "
               "  SELECT a.source_memory_id AS label_id, "
               "         seed_label.score * a.weight * ? AS score "
               "  FROM seed_label "
               "  JOIN associations a ON a.target_memory_id = seed_label.label_id "
               "  JOIN memories source_label ON source_label.memory_id = a.source_memory_id "
               "    AND source_label.kind = 'LABEL' "
               "  WHERE a.edge_type NOT IN ('has_label', 'derived_from') "
               "), related_cue AS ("
               "  SELECT hl.source_memory_id AS cue_id, "
               "         MAX(related_label.score * hl.weight) AS score "
               "  FROM related_label "
               "  JOIN associations hl ON hl.target_memory_id = related_label.label_id "
               "    AND hl.edge_type = 'has_label' "
               "  JOIN memories cue ON cue.memory_id = hl.source_memory_id "
               "    AND cue.kind = 'ASSOCIATION' "
               "  GROUP BY hl.source_memory_id "
               ") "
               "SELECT df.target_memory_id AS source_id, "
               "       MAX(related_cue.score * df.weight) AS graph_score "
               "FROM related_cue "
               "JOIN associations df ON df.source_memory_id = related_cue.cue_id "
               "  AND df.edge_type = 'derived_from' "
               "JOIN memories source_memory ON source_memory.memory_id = df.target_memory_id "
               "  AND source_memory.kind NOT IN ('WORKING', 'LABEL', 'ASSOCIATION') "
               "GROUP BY df.target_memory_id "
               "ORDER BY graph_score DESC, source_id DESC";
        std::vector<std::any> graph_params = params;
        graph_params.push_back (relation_weight);
        graph_params.push_back (relation_weight);
        auto rows = store.Execute (sql.str (), graph_params);
        for (const auto &row : rows)
          {
            const long long memory_id = AnyLongLong (row, "source_id");
            const double graph_score = AnyDouble (row, "graph_score");
            if (memory_id <= 0 || graph_score <= 0.0)
              continue;
            ++packet.graph_candidate_count;
            packet.best_graph_score = std::max (packet.best_graph_score,
                                                graph_score);
            packet.expanded_memory_ids.push_back (memory_id);
            AddScoredMemory (scores, memory_id, graph_weight * graph_score);
          }
      }

      {
        std::ostringstream sql;
        sql
            << "WITH seed(id, seed_score) AS (VALUES " << values.str ()
            << "), seed_fact AS ("
               "  SELECT fe.fact_id, MAX(seed.seed_score * fe.support_weight) AS score "
               "  FROM seed "
               "  JOIN fact_evidence fe ON fe.source_memory_id = seed.id "
               "  JOIN fact_assertions fa ON fa.fact_id = fe.fact_id "
               "  WHERE COALESCE(fa.lifecycle_state, 'active') = 'active' "
               "  GROUP BY fe.fact_id "
               ") "
               "SELECT fe.source_memory_id AS source_id, "
               "       MAX(seed_fact.score * fe.support_weight) AS fact_score "
               "FROM seed_fact "
               "JOIN fact_evidence fe ON fe.fact_id = seed_fact.fact_id "
               "JOIN memories source_memory ON source_memory.memory_id = fe.source_memory_id "
               "  AND source_memory.kind NOT IN ('WORKING', 'LABEL', 'ASSOCIATION') "
               "GROUP BY fe.source_memory_id "
               "ORDER BY fact_score DESC, source_id DESC";
        auto rows = store.Execute (sql.str (), params);
        for (const auto &row : rows)
          {
            const long long memory_id = AnyLongLong (row, "source_id");
            const double candidate_score = AnyDouble (row, "fact_score");
            if (memory_id <= 0 || candidate_score <= 0.0)
              continue;
            ++packet.fact_candidate_count;
            packet.best_fact_score = std::max (packet.best_fact_score,
                                               candidate_score);
            packet.expanded_memory_ids.push_back (memory_id);
            AddScoredMemory (scores, memory_id, fact_weight * candidate_score);
          }
      }

      if (compact_limit > 0)
        {
          std::ostringstream compact;
          {
            std::ostringstream sql;
            sql
                << "WITH seed(id, seed_score) AS (VALUES " << values.str ()
                << "), labels AS ("
                   "  SELECT label.memory_id, "
                   "         COALESCE(label.label, label.source_id, '') AS label_text, "
                   "         MAX(seed.seed_score * df.weight * hl.weight) AS score "
                   "  FROM seed "
                   "  JOIN associations df ON df.target_memory_id = seed.id "
                   "    AND df.edge_type = 'derived_from' "
                   "  JOIN associations hl ON hl.source_memory_id = df.source_memory_id "
                   "    AND hl.edge_type = 'has_label' "
                   "  JOIN memories label ON label.memory_id = hl.target_memory_id "
                   "    AND label.kind = 'LABEL' "
                   "  GROUP BY label.memory_id "
                   ") SELECT label_text FROM labels "
                   "ORDER BY score DESC, memory_id DESC LIMIT ?";
            std::vector<std::any> compact_params = params;
            compact_params.push_back (static_cast<long long> (compact_limit));
            auto rows = store.Execute (sql.str (), compact_params);
            for (const auto &row : rows)
              {
                const std::string label = Trim (AnyString (row, "label_text"));
                if (label.empty () || Overlap (query_tokens, label) <= 0.0)
                  continue;
                compact << "[graph_label] " << label << "\n";
                ++packet.compact_label_count;
              }
          }
          {
            std::ostringstream sql;
            sql
                << "WITH seed(id, seed_score) AS (VALUES " << values.str ()
                << "), seed_fact AS ("
                   "  SELECT fe.fact_id, MAX(seed.seed_score * fe.support_weight) AS score "
                   "  FROM seed "
                   "  JOIN fact_evidence fe ON fe.source_memory_id = seed.id "
                   "  JOIN fact_assertions fa ON fa.fact_id = fe.fact_id "
                   "  WHERE COALESCE(fa.lifecycle_state, 'active') = 'active' "
                   "  GROUP BY fe.fact_id "
                   ") SELECT fc.fact_text FROM seed_fact "
                   "JOIN fact_cache fc ON fc.fact_id = seed_fact.fact_id "
                   "ORDER BY seed_fact.score DESC, fc.fact_id DESC LIMIT ?";
            std::vector<std::any> compact_params = params;
            compact_params.push_back (static_cast<long long> (compact_limit));
            auto rows = store.Execute (sql.str (), compact_params);
            for (const auto &row : rows)
              {
                const std::string fact = Trim (AnyString (row, "fact_text"));
                if (fact.empty () || Overlap (query_tokens, fact) <= 0.0)
                  continue;
                compact << "[graph_fact] " << fact << "\n";
                ++packet.compact_fact_count;
              }
          }
          packet.compact_context = compact.str ();
        }
    }

  packet.ranked_memory_ids = RankScoredMemoryIds (scores, max_memories);
  (void)engine;
  std::vector<RagDoc> ranked_docs;
  ranked_docs.reserve (packet.ranked_memory_ids.size ());
  for (long long id : packet.ranked_memory_ids)
    {
      auto it = memory_id_to_doc.find (id);
      if (it != memory_id_to_doc.end ())
        ranked_docs.push_back (it->second);
    }
  packet.source_context = BuildGraphExpandedRagContext (
      ranked_docs, packet.compact_context, max_memories);
  packet.chat_context = packet.source_context;
  return packet;
}

StmGraphPacket
BuildStmGraphPacket (cortext::Store &store, cortext::Cortext &engine,
                     const std::unordered_set<std::string> &query_tokens,
                     std::size_t max_memories)
{
  StmGraphPacket packet;
  if (!TableExists (store, "stm_ltm_relabel_audit") || max_memories == 0)
    return packet;

  struct Candidate
  {
    double label_overlap = 0.0;
    double source_overlap = 0.0;
    long long created_at = 0;
    std::vector<long long> source_ids;
    int label_count = 0;
  };

  std::vector<Candidate> raw_candidates;
  std::vector<Candidate> relabel_candidates;
  auto rows = store.Execute (
      "SELECT created_at, source_memory_ids, current_labels, refined_labels "
      "FROM stm_ltm_relabel_audit ORDER BY created_at DESC, summary_id DESC");
  for (const auto &row : rows)
    {
      const auto source_ids = SplitAuditIds (AnyString (row,
                                                       "source_memory_ids"));
      if (source_ids.empty ())
        continue;
      const auto source_memories = HydrateMemories (engine, source_ids);
      std::ostringstream source_text;
      for (const auto &memory : source_memories)
        {
          const std::string text = MemoryText (memory);
          if (!text.empty ())
            source_text << text << "\n";
        }
      const double source_overlap = Overlap (query_tokens, source_text.str ());

      const auto current_labels = SplitAuditList (AnyString (row,
                                                            "current_labels"));
      Candidate raw;
      raw.label_overlap = LabelOverlap (query_tokens, current_labels);
      raw.source_overlap = source_overlap;
      raw.created_at = AnyLongLong (row, "created_at");
      raw.source_ids = source_ids;
      raw.label_count = static_cast<int> (current_labels.size ());
      packet.raw_label_count += raw.label_count;
      packet.raw_best_label_overlap = std::max (packet.raw_best_label_overlap,
                                                raw.label_overlap);
      packet.raw_best_source_overlap = std::max (packet.raw_best_source_overlap,
                                                 raw.source_overlap);
      if (raw.label_overlap > 0.0)
        ++packet.raw_positive_label_cycles;
      if (raw.source_overlap > 0.0)
        ++packet.raw_positive_source_cycles;
      raw_candidates.push_back (std::move (raw));

      const auto refined_labels = SplitAuditList (AnyString (row,
                                                            "refined_labels"));
      Candidate relabel;
      relabel.label_overlap = LabelOverlap (query_tokens, refined_labels);
      relabel.source_overlap = source_overlap;
      relabel.created_at = AnyLongLong (row, "created_at");
      relabel.source_ids = source_ids;
      relabel.label_count = static_cast<int> (refined_labels.size ());
      packet.relabel_label_count += relabel.label_count;
      packet.relabel_best_label_overlap = std::max (
          packet.relabel_best_label_overlap, relabel.label_overlap);
      packet.relabel_best_source_overlap = std::max (
          packet.relabel_best_source_overlap, relabel.source_overlap);
      if (relabel.label_overlap > 0.0)
        ++packet.relabel_positive_label_cycles;
      if (relabel.source_overlap > 0.0)
        ++packet.relabel_positive_source_cycles;
      relabel_candidates.push_back (std::move (relabel));
    }

  auto select_ids = [max_memories] (std::vector<Candidate> candidates,
                                    int &cycle_count) {
    std::stable_sort (
        candidates.begin (), candidates.end (),
        [] (const Candidate &lhs, const Candidate &rhs) {
          const double lhs_score = lhs.label_overlap + lhs.source_overlap;
          const double rhs_score = rhs.label_overlap + rhs.source_overlap;
          if (lhs_score != rhs_score)
            return lhs_score > rhs_score;
          if (lhs.label_overlap != rhs.label_overlap)
            return lhs.label_overlap > rhs.label_overlap;
          if (lhs.source_overlap != rhs.source_overlap)
            return lhs.source_overlap > rhs.source_overlap;
          return lhs.created_at > rhs.created_at;
        });
    std::vector<long long> ids;
    std::unordered_set<long long> seen;
    for (const auto &candidate : candidates)
      {
        bool used_cycle = false;
        for (long long id : candidate.source_ids)
          {
            if (id <= 0 || !seen.insert (id).second)
              continue;
            ids.push_back (id);
            used_cycle = true;
            if (ids.size () >= max_memories)
              break;
          }
        if (used_cycle)
          ++cycle_count;
        if (ids.size () >= max_memories)
          break;
      }
    return ids;
  };

  packet.raw_memory_ids = select_ids (std::move (raw_candidates),
                                      packet.raw_cycle_count);
  packet.relabel_memory_ids = select_ids (std::move (relabel_candidates),
                                          packet.relabel_cycle_count);
  int durable_candidate_count = 0;
  int fact_linked_candidate_count = 0;
  auto durable_source_ids = LtmSourceIdsFromRankedCandidates (
      store, max_memories, &durable_candidate_count,
      &fact_linked_candidate_count);
  packet.relabel_durable_candidate_count = durable_candidate_count;
  packet.relabel_fact_linked_candidate_count = fact_linked_candidate_count;
  packet.relabel_durable_source_count
      = static_cast<int> (durable_source_ids.size ());
  if (!durable_source_ids.empty ())
    packet.relabel_memory_ids = std::move (durable_source_ids);
  packet.raw_context = BuildMemoryListContext (
      HydrateMemories (engine, packet.raw_memory_ids), max_memories,
      "raw_stm_graph");
  packet.relabel_context = BuildMemoryListContext (
      HydrateMemories (engine, packet.relabel_memory_ids), max_memories,
      "relabel_prune_ltm");
  return packet;
}

int
IntersectionCount (const std::vector<long long> &lhs,
                   const std::vector<long long> &rhs)
{
  std::unordered_set<long long> left (lhs.begin (), lhs.end ());
  int count = 0;
  for (long long id : rhs)
    {
      if (left.count (id) != 0)
        ++count;
    }
  return count;
}

nlohmann::json
QualityJson (const QualityAggregate &agg)
{
  nlohmann::json out;
  out["judged"] = agg.judged;
  out["mean_relevance"] = agg.judged > 0 ? agg.relevance_sum / agg.judged : 0.0;
  out["mean_sufficiency"] = agg.judged > 0
                                ? agg.sufficiency_sum / agg.judged
                                : 0.0;
	  out["mean_noise"] = agg.judged > 0 ? agg.noise_sum / agg.judged : 0.0;
  out["mean_source_grounding"] = agg.judged > 0
                                     ? agg.source_grounding_sum / agg.judged
                                     : 0.0;
  out["mean_temporal_correctness"]
      = agg.judged > 0 ? agg.temporal_correctness_sum / agg.judged : 0.0;
  out["mean_media_usefulness"] = agg.judged > 0
                                     ? agg.media_usefulness_sum / agg.judged
                                     : 0.0;
	  out["wins"] = agg.wins;
  return out;
}

nlohmann::json
RetrievalDebugJson ()
{
  nlohmann::json out = nlohmann::json::array ();
  for (const auto &candidate :
       cortext::operations::retrieval_debug::GetLastRankedCandidates ())
    {
      nlohmann::json row;
      row["embedding_id"] = candidate.embedding_id;
      row["memory_id"] = candidate.memory_id;
      row["score"] = candidate.score;
      row["relevance"] = candidate.relevance;
      row["proc_score"] = candidate.proc_score;
      row["predictive_bonus"] = candidate.predictive_bonus;
      row["pre_activation"] = candidate.pre_activation;
      row["fact_boost"] = candidate.fact_boost;
      row["fact_stale_penalty"] = candidate.fact_stale_penalty;
      row["linked_fact_count"] = candidate.linked_fact_count;
      row["label_graph_boost"] = candidate.label_graph_boost;
      row["label_match_count"] = candidate.label_match_count;
      row["durable_source_boost"] = candidate.durable_source_boost;
      row["durable_source_count"] = candidate.durable_source_count;
      row["activation"] = {
        { "base_level", candidate.activation.base_level },
        { "spreading_activation",
          candidate.activation.spreading_activation },
        { "partial_match_penalty",
          candidate.activation.partial_match_penalty },
        { "recent_inhibition", candidate.activation.recent_inhibition },
        { "utility", candidate.activation.utility },
        { "exploration_noise", candidate.activation.exploration_noise },
        { "activation_total", candidate.activation.activation_total },
      };
      out.push_back (std::move (row));
    }
  return out;
}

nlohmann::json
RetrievalRejectionJson ()
{
  nlohmann::json out = nlohmann::json::array ();
  for (const auto &entry :
       cortext::operations::retrieval_debug::GetLastRejectedCandidates ())
    {
      const auto &candidate = entry.candidate;
      nlohmann::json row;
      row["reason"] = entry.reason;
      row["stage"] = entry.stage;
      row["observed"] = entry.observed;
      row["threshold"] = entry.threshold;
      row["embedding_id"] = candidate.embedding_id;
      row["memory_id"] = candidate.memory_id;
      row["score"] = candidate.score;
      row["relevance"] = candidate.relevance;
      row["proc_score"] = candidate.proc_score;
      row["predictive_bonus"] = candidate.predictive_bonus;
      row["pre_activation"] = candidate.pre_activation;
      row["fact_boost"] = candidate.fact_boost;
      row["fact_stale_penalty"] = candidate.fact_stale_penalty;
      row["linked_fact_count"] = candidate.linked_fact_count;
      row["label_graph_boost"] = candidate.label_graph_boost;
      row["label_match_count"] = candidate.label_match_count;
      row["durable_source_boost"] = candidate.durable_source_boost;
      row["durable_source_count"] = candidate.durable_source_count;
      row["activation"] = {
        { "base_level", candidate.activation.base_level },
        { "spreading_activation",
          candidate.activation.spreading_activation },
        { "partial_match_penalty",
          candidate.activation.partial_match_penalty },
        { "recent_inhibition", candidate.activation.recent_inhibition },
        { "utility", candidate.activation.utility },
        { "exploration_noise", candidate.activation.exploration_noise },
        { "activation_total", candidate.activation.activation_total },
      };
      out.push_back (std::move (row));
    }
  return out;
}

nlohmann::json
RetrievalEvidencePacketJson ()
{
  nlohmann::json out = nlohmann::json::array ();
  for (const auto &packet :
       cortext::operations::retrieval_debug::GetLastEvidencePackets ())
    {
      nlohmann::json members = nlohmann::json::array ();
      for (const auto &member : packet.members)
        {
          nlohmann::json row;
          row["rank"] = member.rank;
          row["embedding_id"] = member.embedding_id;
	          row["memory_id"] = member.memory_id;
	          row["weight"] = member.weight;
	          row["score"] = member.score;
	          row["evidence_confidence"] = member.evidence_confidence;
	          row["evidence_weight"] = member.evidence_weight;
	          row["evidence_source_diversity"]
	              = member.evidence_source_diversity;
	          row["evidence_contradiction_mass"]
	              = member.evidence_contradiction_mass;
	          row["activation"] = {
            { "base_level", member.activation.base_level },
            { "spreading_activation",
              member.activation.spreading_activation },
            { "partial_match_penalty",
              member.activation.partial_match_penalty },
            { "recent_inhibition", member.activation.recent_inhibition },
            { "utility", member.activation.utility },
            { "exploration_noise", member.activation.exploration_noise },
            { "activation_total", member.activation.activation_total },
          };
          members.push_back (std::move (row));
        }

      nlohmann::json row;
      row["packet_id"] = packet.packet_id;
      row["consumer"] = packet.consumer;
      row["reason"] = packet.reason;
      row["tie_margin"] = packet.tie_margin;
      row["temperature"] = packet.temperature;
	      row["score_span"] = packet.score_span;
	      row["activation_total"] = packet.activation_total;
	      row["evidence_confidence"] = packet.evidence_confidence;
	      row["evidence_weight"] = packet.evidence_weight;
	      row["evidence_source_diversity"] = packet.evidence_source_diversity;
	      row["evidence_contradiction_mass"]
	          = packet.evidence_contradiction_mass;
	      row["members"] = std::move (members);
      out.push_back (std::move (row));
    }
  return out;
}

nlohmann::json
RetrievalSummaryJson ()
{
  const auto summary
      = cortext::operations::retrieval_debug::GetLastRetrievalSummary ();
  nlohmann::json out;
  out["fact_layer_enabled"] = summary.fact_layer_enabled;
  out["fact_seed_count"] = summary.fact_seed_count;
  out["candidate_fact_link_memory_count"]
      = summary.candidate_fact_link_memory_count;
  out["candidate_fact_link_row_count"] = summary.candidate_fact_link_row_count;
  out["selected_fact_linked_count"] = summary.selected_fact_linked_count;
  out["text_query_token_count"] = summary.text_query_token_count;
  out["text_query_wm_slots"] = summary.text_query_wm_slots;
  out["text_query_wm_chars"] = summary.text_query_wm_chars;
  out["fact_text_candidate_count"] = summary.fact_text_candidate_count;
  out["fact_text_rejected_low_score_count"]
      = summary.fact_text_rejected_low_score_count;
  out["fact_text_match_count"] = summary.fact_text_match_count;
  out["fact_text_best_score"] = summary.fact_text_best_score;
  out["rejected_candidate_count"] = summary.rejected_candidate_count;
  out["rejected_filter_count"] = summary.rejected_filter_count;
  out["rejected_selection_count"] = summary.rejected_selection_count;
	  out["evidence_packet_count"] = summary.evidence_packet_count;
	  out["evidence_packet_member_count"] = summary.evidence_packet_member_count;
	  out["evidence_packet_confidence_mean"]
	      = summary.evidence_packet_confidence_mean;
	  return out;
}

void
AddAggregate (Aggregate &agg, long long prompt_tokens, long long context_tokens,
              long long memory_context_tokens,
              long long active_history_tokens, long long context_chars,
              long long context_items, double overlap,
              double latency_ms)
{
  ++agg.probes;
  agg.prompt_tokens += prompt_tokens;
  agg.context_tokens += context_tokens;
  agg.memory_context_tokens += memory_context_tokens;
  agg.active_history_tokens += active_history_tokens;
  agg.context_chars += context_chars;
  agg.context_items += context_items;
  agg.overlap_sum += overlap;
  agg.latency_ms_sum += latency_ms;
  agg.prompt_token_samples.push_back (prompt_tokens);
  agg.context_token_samples.push_back (context_tokens);
  agg.latency_ms_samples.push_back (latency_ms);
}

void
AddCompaction (CompactionAggregate &agg, const CompactedHistory &history)
{
  ++agg.probe_events;
  agg.compaction_events += history.compaction_events;
  agg.compacted_items += history.compacted_items;
  agg.raw_items += history.raw_items;
  agg.raw_tokens += history.raw_tokens;
  agg.compacted_original_tokens += history.compacted_original_tokens;
  agg.compacted_summary_tokens += history.compacted_summary_tokens;
}

double
Percentile (std::vector<long long> values, double q)
{
  if (values.empty ())
    return 0.0;
  std::sort (values.begin (), values.end ());
  const double pos = cortext::core::Clamp (q, 0.0, 1.0)
                     * static_cast<double> (values.size () - 1);
  const auto lo = static_cast<size_t> (std::floor (pos));
  const auto hi = static_cast<size_t> (std::ceil (pos));
  if (lo == hi)
    return static_cast<double> (values[lo]);
  const double frac = pos - static_cast<double> (lo);
  return static_cast<double> (values[lo]) * (1.0 - frac)
         + static_cast<double> (values[hi]) * frac;
}

double
Percentile (std::vector<double> values, double q)
{
  if (values.empty ())
    return 0.0;
  std::sort (values.begin (), values.end ());
  const double pos = cortext::core::Clamp (q, 0.0, 1.0)
                     * static_cast<double> (values.size () - 1);
  const auto lo = static_cast<size_t> (std::floor (pos));
  const auto hi = static_cast<size_t> (std::ceil (pos));
  if (lo == hi)
    return values[lo];
  const double frac = pos - static_cast<double> (lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

nlohmann::json
AggregateJson (const Aggregate &agg)
{
  nlohmann::json out;
  out["probes"] = agg.probes;
  out["mean_prompt_tokens"] = agg.probes > 0
                                  ? static_cast<double> (agg.prompt_tokens)
                                        / agg.probes
                                  : 0.0;
  out["mean_context_tokens"] = agg.probes > 0
                                   ? static_cast<double> (agg.context_tokens)
                                         / agg.probes
                                   : 0.0;
  out["mean_memory_context_tokens"]
      = agg.probes > 0 ? static_cast<double> (agg.memory_context_tokens)
                             / agg.probes
                       : 0.0;
  out["mean_active_history_tokens"]
      = agg.probes > 0 ? static_cast<double> (agg.active_history_tokens)
                             / agg.probes
                       : 0.0;
  out["mean_context_chars"] = agg.probes > 0
                                  ? static_cast<double> (agg.context_chars)
                                        / agg.probes
                                  : 0.0;
  out["mean_context_items"] = agg.probes > 0
                                  ? static_cast<double> (agg.context_items)
                                        / agg.probes
                                  : 0.0;
  out["mean_query_context_overlap"] = agg.probes > 0
                                          ? agg.overlap_sum / agg.probes
                                          : 0.0;
	  out["mean_probe_latency_ms"] = agg.probes > 0
	                                     ? agg.latency_ms_sum / agg.probes
	                                     : 0.0;
  out["cumulative_prompt_tokens"] = agg.prompt_tokens;
  out["prompt_tokens_p50"] = Percentile (agg.prompt_token_samples, 0.50);
  out["prompt_tokens_p90"] = Percentile (agg.prompt_token_samples, 0.90);
  out["prompt_tokens_p99"] = Percentile (agg.prompt_token_samples, 0.99);
  out["context_tokens_p50"] = Percentile (agg.context_token_samples, 0.50);
  out["context_tokens_p90"] = Percentile (agg.context_token_samples, 0.90);
  out["context_tokens_p99"] = Percentile (agg.context_token_samples, 0.99);
  out["probe_latency_ms_p50"] = Percentile (agg.latency_ms_samples, 0.50);
  out["probe_latency_ms_p95"] = Percentile (agg.latency_ms_samples, 0.95);
  out["probe_latency_ms_p99"] = Percentile (agg.latency_ms_samples, 0.99);
  return out;
}

nlohmann::json
CompactionJson (const CompactionAggregate &agg)
{
  nlohmann::json out;
  out["probe_events"] = agg.probe_events;
  out["compaction_events"] = agg.compaction_events;
  out["compacted_items"] = agg.compacted_items;
  out["raw_items"] = agg.raw_items;
  out["raw_tokens"] = agg.raw_tokens;
  out["compacted_original_tokens"] = agg.compacted_original_tokens;
  out["compacted_summary_tokens"] = agg.compacted_summary_tokens;
  out["mean_compacted_original_tokens"] =
      agg.probe_events > 0
          ? static_cast<double> (agg.compacted_original_tokens)
                / agg.probe_events
          : 0.0;
  out["mean_compacted_summary_tokens"] =
      agg.probe_events > 0
          ? static_cast<double> (agg.compacted_summary_tokens)
                / agg.probe_events
          : 0.0;
  return out;
}

double
JsonDouble (const nlohmann::json &object, const char *key,
            double fallback = 0.0)
{
  if (!object.is_object () || !object.contains (key))
    return fallback;
  const auto &value = object[key];
  if (value.is_number ())
    return value.get<double> ();
  if (value.is_string ())
    {
      try
        {
          return std::stod (value.get<std::string> ());
        }
      catch (...)
        {
        }
    }
  return fallback;
}

long long
JsonInt64 (const nlohmann::json &object, const char *key,
           long long fallback = 0)
{
  if (!object.is_object () || !object.contains (key))
    return fallback;
  const auto &value = object[key];
  if (value.is_number_integer () || value.is_number_unsigned ())
    return value.get<long long> ();
  if (value.is_number_float ())
    return static_cast<long long> (value.get<double> ());
  if (value.is_string ())
    {
      try
        {
          return std::stoll (value.get<std::string> ());
        }
      catch (...)
        {
        }
    }
  return fallback;
}

bool
ProbeHasLabelBoost (const nlohmann::json &probe)
{
  if (!probe.contains ("retrieval_debug")
      || !probe["retrieval_debug"].is_array ())
    return false;
  for (const auto &candidate : probe["retrieval_debug"])
    {
      if (JsonDouble (candidate, "label_graph_boost") > 0.0
          || JsonInt64 (candidate, "label_match_count") > 0)
        return true;
    }
  return false;
}

nlohmann::json
BuildFailureTaxonomy (const nlohmann::json &probes,
                      const nlohmann::json &stm_ltm_audit)
{
  nlohmann::json out;
  long long probe_count = 0;
  long long judged_compact_count = 0;
  long long ltm_under_rag = 0;
  long long ltm_under_stm = 0;
  long long stm_ltm_under_rag = 0;
  long long zero_ltm_query_overlap = 0;
  long long zero_stm_ltm_query_overlap = 0;
  long long stm_graph_probe_count = 0;
  long long raw_graph_zero_overlap = 0;
  long long relabel_graph_zero_overlap = 0;
  long long raw_graph_positive_label_cycles = 0;
  long long relabel_graph_positive_label_cycles = 0;
  long long raw_graph_positive_source_cycles = 0;
  long long relabel_graph_positive_source_cycles = 0;
  long long relabel_graph_durable_candidates = 0;
  long long relabel_graph_durable_sources = 0;
  long long raw_graph_under_rag = 0;
  long long relabel_graph_under_rag = 0;
  long long label_boost_no_ltm_overlap = 0;
  long long many_ranked_few_injected = 0;
  long long low_ltm_relevance = 0;
  long long ltm_no_wins = 0;

  if (probes.is_array ())
    {
      for (const auto &probe : probes)
        {
          ++probe_count;
          const double ltm_overlap = JsonDouble (probe,
                                                 "cortext_ltm_overlap");
          if (ltm_overlap <= 0.0)
            ++zero_ltm_query_overlap;

          const long long raw_retrieved = JsonInt64 (
              probe, "cortext_raw_retrieved_memory_items");
          const long long max_injected = JsonInt64 (probe,
                                                    "max_injected_memories");
          if (max_injected > 0 && raw_retrieved > max_injected)
            ++many_ranked_few_injected;

          if (ProbeHasLabelBoost (probe) && ltm_overlap <= 0.0)
            ++label_boost_no_ltm_overlap;

          if (probe.contains ("compact_policy_bakeoff")
              && probe["compact_policy_bakeoff"].is_object ())
            {
              const auto &compact = probe["compact_policy_bakeoff"];
              const double stm_ltm_overlap = JsonDouble (
                  compact, "stm_ltm_union_overlap");
              if (stm_ltm_overlap <= 0.0)
                ++zero_stm_ltm_query_overlap;

              if (compact.contains ("quality")
                  && compact["quality"].is_object ())
                {
                  const auto &quality = compact["quality"];
                  ++judged_compact_count;
                  const double ltm_rel = JsonDouble (
                      quality, "cortext_ltm_relevance");
                  const double ltm_suff = JsonDouble (
                      quality, "cortext_ltm_sufficiency");
                  const double stm_rel = JsonDouble (
                      quality, "stm_recent_relevance");
                  const double stm_suff = JsonDouble (
                      quality, "stm_recent_sufficiency");
                  const double stm_ltm_rel = JsonDouble (
                      quality, "stm_ltm_union_relevance");
                  const double stm_ltm_suff = JsonDouble (
                      quality, "stm_ltm_union_sufficiency");
                  const double rag_rel = JsonDouble (
                      quality, "normal_rag_relevance");
                  const double rag_suff = JsonDouble (
                      quality, "normal_rag_sufficiency");
                  if (ltm_rel < rag_rel || ltm_suff < rag_suff)
                    ++ltm_under_rag;
                  if (ltm_rel < stm_rel || ltm_suff < stm_suff)
                    ++ltm_under_stm;
                  if (stm_ltm_rel < rag_rel || stm_ltm_suff < rag_suff)
                    ++stm_ltm_under_rag;
                  if (ltm_rel <= 2.0 || ltm_suff <= 1.0)
                    ++low_ltm_relevance;
                  if (quality.value ("winner", "") != "cortext_ltm")
                    ++ltm_no_wins;
                }
            }
          if (probe.contains ("stm_graph_bakeoff")
              && probe["stm_graph_bakeoff"].is_object ())
            {
              const auto &graph = probe["stm_graph_bakeoff"];
              ++stm_graph_probe_count;
              if (JsonDouble (graph, "raw_overlap") <= 0.0)
                ++raw_graph_zero_overlap;
              if (JsonDouble (graph, "relabel_prune_overlap") <= 0.0)
                ++relabel_graph_zero_overlap;
              raw_graph_positive_label_cycles += JsonInt64 (
                  graph, "raw_positive_label_cycles");
              relabel_graph_positive_label_cycles += JsonInt64 (
                  graph, "relabel_positive_label_cycles");
              raw_graph_positive_source_cycles += JsonInt64 (
                  graph, "raw_positive_source_cycles");
              relabel_graph_positive_source_cycles += JsonInt64 (
                  graph, "relabel_positive_source_cycles");
              relabel_graph_durable_candidates += JsonInt64 (
                  graph, "relabel_durable_candidate_count");
              relabel_graph_durable_sources += JsonInt64 (
                  graph, "relabel_durable_source_count");
              if (graph.contains ("quality") && graph["quality"].is_object ())
                {
                  const auto &quality = graph["quality"];
                  const double raw_rel = JsonDouble (
                      quality, "raw_stm_graph_relevance");
                  const double raw_suff = JsonDouble (
                      quality, "raw_stm_graph_sufficiency");
                  const double relabel_rel = JsonDouble (
                      quality, "relabel_prune_ltm_relevance");
                  const double relabel_suff = JsonDouble (
                      quality, "relabel_prune_ltm_sufficiency");
                  const double rag_rel = JsonDouble (
                      quality, "normal_rag_relevance");
                  const double rag_suff = JsonDouble (
                      quality, "normal_rag_sufficiency");
                  if (raw_rel < rag_rel || raw_suff < rag_suff)
                    ++raw_graph_under_rag;
                  if (relabel_rel < rag_rel || relabel_suff < rag_suff)
                    ++relabel_graph_under_rag;
                }
            }
        }
    }

  out["probe_count"] = probe_count;
  out["judged_compact_count"] = judged_compact_count;
  out["missed_identity_or_topic_evidence"]["zero_ltm_query_overlap"]
      = zero_ltm_query_overlap;
  out["missed_identity_or_topic_evidence"]["ltm_under_stm_when_judged"]
      = ltm_under_stm;
  out["generic_label_collapse_or_weak_label_boost"]
      ["label_boost_without_ltm_query_overlap"]
      = label_boost_no_ltm_overlap;
  out["under_admission"]["many_ranked_few_injected"]
      = many_ranked_few_injected;
  out["stale_or_irrelevant_retrieval"]["low_ltm_relevance_when_judged"]
      = low_ltm_relevance;
  out["stale_or_irrelevant_retrieval"]["ltm_under_rag_when_judged"]
      = ltm_under_rag;
  out["stale_or_irrelevant_retrieval"]["ltm_no_wins_when_judged"]
      = ltm_no_wins;
  out["raw_stm_vs_relabel_prune"]["stm_ltm_union_under_rag_when_judged"]
      = stm_ltm_under_rag;
  out["raw_stm_vs_relabel_prune"]["zero_stm_ltm_union_query_overlap"]
      = zero_stm_ltm_query_overlap;
  out["raw_stm_vs_relabel_prune"]["stm_graph_probe_count"]
      = stm_graph_probe_count;
  out["raw_stm_vs_relabel_prune"]["raw_graph_zero_query_overlap"]
      = raw_graph_zero_overlap;
  out["raw_stm_vs_relabel_prune"]["relabel_graph_zero_query_overlap"]
      = relabel_graph_zero_overlap;
  out["raw_stm_vs_relabel_prune"]["raw_graph_positive_label_cycles"]
      = raw_graph_positive_label_cycles;
  out["raw_stm_vs_relabel_prune"]["relabel_graph_positive_label_cycles"]
      = relabel_graph_positive_label_cycles;
  out["raw_stm_vs_relabel_prune"]["raw_graph_positive_source_cycles"]
      = raw_graph_positive_source_cycles;
  out["raw_stm_vs_relabel_prune"]["relabel_graph_positive_source_cycles"]
      = relabel_graph_positive_source_cycles;
  out["raw_stm_vs_relabel_prune"]["relabel_graph_durable_candidates"]
      = relabel_graph_durable_candidates;
  out["raw_stm_vs_relabel_prune"]["relabel_graph_durable_sources"]
      = relabel_graph_durable_sources;
  out["raw_stm_vs_relabel_prune"]["raw_graph_under_rag_when_judged"]
      = raw_graph_under_rag;
  out["raw_stm_vs_relabel_prune"]["relabel_graph_under_rag_when_judged"]
      = relabel_graph_under_rag;

  const long long source_count = JsonInt64 (stm_ltm_audit,
                                            "source_memory_count");
  const long long source_with_content = JsonInt64 (
      stm_ltm_audit, "source_memories_with_content");
  const long long has_label_edges = JsonInt64 (stm_ltm_audit,
                                               "has_label_edges_after");
  const long long durable_ltm_nodes_with_source = JsonInt64 (
      stm_ltm_audit, "durable_ltm_nodes_with_source");
  const long long durable_ltm_nodes_missing_source = JsonInt64 (
      stm_ltm_audit, "durable_ltm_nodes_missing_source");
  const long long durable_ltm_source_link_pairs = JsonInt64 (
      stm_ltm_audit, "durable_ltm_source_link_pairs");
  const long long relation_edges = JsonInt64 (stm_ltm_audit,
                                              "relation_edges_created");
  const long long label_cooccurrence_edges = JsonInt64 (
      stm_ltm_audit, "label_cooccurrence_edges_created");
  const long long relation_count = JsonInt64 (stm_ltm_audit,
                                             "relation_count");
  const long long relation_edges_skipped_non_durable_endpoint = JsonInt64 (
      stm_ltm_audit, "relation_edges_skipped_non_durable_endpoint");
  const long long relation_edges_skipped_missing_endpoint = JsonInt64 (
      stm_ltm_audit, "relation_edges_skipped_missing_endpoint");
  const long long relation_edges_skipped_unsupported_predicate = JsonInt64 (
      stm_ltm_audit, "relation_edges_skipped_unsupported_predicate");
  const long long relation_endpoint_direct_hits = JsonInt64 (
      stm_ltm_audit, "relation_endpoint_direct_hits");
  const long long relation_endpoint_repair_hits = JsonInt64 (
      stm_ltm_audit, "relation_endpoint_repair_hits");
  const long long relation_endpoint_created_labels = JsonInt64 (
      stm_ltm_audit, "relation_endpoint_created_labels");
  const long long relation_endpoint_relation_backed_labels = JsonInt64 (
      stm_ltm_audit, "relation_endpoint_relation_backed_labels");
  const long long relation_endpoint_rejected_count = JsonInt64 (
      stm_ltm_audit, "relation_endpoint_rejected_count");
  const long long relation_endpoint_rejected_non_durable = JsonInt64 (
      stm_ltm_audit, "relation_endpoint_rejected_non_durable");
  const long long relation_endpoint_rejected_ungrounded = JsonInt64 (
      stm_ltm_audit, "relation_endpoint_rejected_ungrounded");
  const long long refined_labels = JsonInt64 (stm_ltm_audit,
                                              "refined_label_count");
  const long long removed_labels = JsonInt64 (stm_ltm_audit,
                                              "removed_label_count");
  const long long current_labels = JsonInt64 (stm_ltm_audit,
                                              "current_label_count");
  const long long current_labels_in_selected_evidence = JsonInt64 (
      stm_ltm_audit, "current_labels_in_selected_evidence");
  const long long current_labels_in_full_source = JsonInt64 (
      stm_ltm_audit, "current_labels_in_full_source");
  const long long removed_labels_in_selected_evidence = JsonInt64 (
      stm_ltm_audit, "removed_labels_in_selected_evidence");
  const long long removed_labels_in_full_source = JsonInt64 (
      stm_ltm_audit, "removed_labels_in_full_source");
  const long long refined_labels_in_selected_evidence = JsonInt64 (
      stm_ltm_audit, "refined_labels_in_selected_evidence");
  const long long refined_labels_in_full_source = JsonInt64 (
      stm_ltm_audit, "refined_labels_in_full_source");
  const long long extraction_label_candidate_count = JsonInt64 (
      stm_ltm_audit, "extraction_label_candidate_count");
  const long long extraction_relation_candidate_count = JsonInt64 (
      stm_ltm_audit, "extraction_relation_candidate_count");
  const long long source_span_candidate_count = JsonInt64 (
      stm_ltm_audit, "source_span_candidate_count");
  const long long label_candidates_rejected_non_durable = JsonInt64 (
      stm_ltm_audit, "label_candidates_rejected_non_durable");
  const long long label_candidates_rejected_ungrounded = JsonInt64 (
      stm_ltm_audit, "label_candidates_rejected_ungrounded");
  const long long label_candidates_rejected_duplicate = JsonInt64 (
      stm_ltm_audit, "label_candidates_rejected_duplicate");
  const long long label_candidates_rejected_legacy_gate = JsonInt64 (
      stm_ltm_audit, "label_candidates_rejected_legacy_gate");
  const long long labels_inserted_from_extractor = JsonInt64 (
      stm_ltm_audit, "labels_inserted_from_extractor");
  const long long labels_inserted_from_current_floor = JsonInt64 (
      stm_ltm_audit, "labels_inserted_from_current_floor");
  const long long labels_inserted_from_source_span_floor = JsonInt64 (
      stm_ltm_audit, "labels_inserted_from_source_span_floor");
  const long long labels_inserted_from_relation_endpoint = JsonInt64 (
      stm_ltm_audit, "labels_inserted_from_relation_endpoint");
  long long cycles_without_durable_labels = 0;
  long long cycles_without_source_links = 0;
  long long cycles_with_current_labels_pruned_to_zero = 0;
  if (stm_ltm_audit.contains ("cycle_summaries")
      && stm_ltm_audit["cycle_summaries"].is_array ())
    {
      for (const auto &cycle : stm_ltm_audit["cycle_summaries"])
        {
          if (JsonInt64 (cycle, "has_label_edges_after") == 0)
            ++cycles_without_durable_labels;
          if (JsonInt64 (cycle, "derived_from_edges") == 0)
            ++cycles_without_source_links;
          if (JsonInt64 (cycle, "current_label_count") > 0
              && JsonInt64 (cycle, "refined_label_count") == 0)
            ++cycles_with_current_labels_pruned_to_zero;
        }
    }
  out["lost_source_mapping"]["source_memories_missing_content"]
      = std::max<long long> (0, source_count - source_with_content);
  out["lost_source_mapping"]["source_memory_count"] = source_count;
  out["lost_source_mapping"]["source_memories_with_content"]
      = source_with_content;
  out["lost_source_mapping"]["cycles_without_source_links"]
      = cycles_without_source_links;
  out["lost_source_mapping"]["durable_ltm_nodes_with_source"]
      = durable_ltm_nodes_with_source;
  out["lost_source_mapping"]["durable_ltm_nodes_missing_source"]
      = durable_ltm_nodes_missing_source;
  out["lost_source_mapping"]["durable_ltm_source_link_pairs"]
      = durable_ltm_source_link_pairs;
  out["over_pruning"]["pruned_label_count"] = removed_labels;
  out["over_pruning"]["refined_label_count"] = refined_labels;
  out["over_pruning"]["cycles_with_current_labels_pruned_to_zero"]
      = cycles_with_current_labels_pruned_to_zero;
  out["evidence_window"]["current_label_count"] = current_labels;
  out["evidence_window"]["current_labels_in_selected_evidence"]
      = current_labels_in_selected_evidence;
  out["evidence_window"]["current_labels_in_full_source"]
      = current_labels_in_full_source;
  out["evidence_window"]["current_labels_absent_from_selected_evidence"]
      = std::max<long long> (0, current_labels
                                    - current_labels_in_selected_evidence);
  out["evidence_window"]["current_labels_absent_from_full_source"]
      = std::max<long long> (0, current_labels
                                    - current_labels_in_full_source);
  out["evidence_window"]["removed_labels_in_selected_evidence"]
      = removed_labels_in_selected_evidence;
  out["evidence_window"]["removed_labels_in_full_source"]
      = removed_labels_in_full_source;
  out["evidence_window"]["refined_labels_in_selected_evidence"]
      = refined_labels_in_selected_evidence;
  out["evidence_window"]["refined_labels_in_full_source"]
      = refined_labels_in_full_source;
  out["under_admission"]["extraction_label_candidate_count"]
      = extraction_label_candidate_count;
  out["under_admission"]["extraction_relation_candidate_count"]
      = extraction_relation_candidate_count;
  out["under_admission"]["source_span_candidate_count"]
      = source_span_candidate_count;
  out["under_admission"]["label_candidates_rejected_non_durable"]
      = label_candidates_rejected_non_durable;
  out["under_admission"]["label_candidates_rejected_ungrounded"]
      = label_candidates_rejected_ungrounded;
  out["under_admission"]["label_candidates_rejected_duplicate"]
      = label_candidates_rejected_duplicate;
  out["under_admission"]["label_candidates_rejected_legacy_gate"]
      = label_candidates_rejected_legacy_gate;
  out["under_admission"]["labels_inserted_from_extractor"]
      = labels_inserted_from_extractor;
  out["under_admission"]["labels_inserted_from_current_floor"]
      = labels_inserted_from_current_floor;
  out["under_admission"]["labels_inserted_from_source_span_floor"]
      = labels_inserted_from_source_span_floor;
  out["under_admission"]["labels_inserted_from_relation_endpoint"]
      = labels_inserted_from_relation_endpoint;
  out["under_linking"]["cycles_without_durable_labels"]
      = cycles_without_durable_labels;
  out["under_linking"]["relation_count"] = relation_count;
  out["under_linking"]["relation_edges_created"] = relation_edges;
  out["under_linking"]["label_cooccurrence_edges_created"]
      = label_cooccurrence_edges;
  out["under_linking"]["relation_edges_skipped_non_durable_endpoint"]
      = relation_edges_skipped_non_durable_endpoint;
  out["under_linking"]["relation_edges_skipped_missing_endpoint"]
      = relation_edges_skipped_missing_endpoint;
  out["under_linking"]["relation_edges_skipped_unsupported_predicate"]
      = relation_edges_skipped_unsupported_predicate;
  out["under_linking"]["relation_endpoint_direct_hits"]
      = relation_endpoint_direct_hits;
  out["under_linking"]["relation_endpoint_repair_hits"]
      = relation_endpoint_repair_hits;
  out["under_linking"]["relation_endpoint_created_labels"]
      = relation_endpoint_created_labels;
  out["under_linking"]["relation_endpoint_relation_backed_labels"]
      = relation_endpoint_relation_backed_labels;
  out["under_linking"]["relation_endpoint_rejected_count"]
      = relation_endpoint_rejected_count;
  out["under_linking"]["relation_endpoint_rejected_non_durable"]
      = relation_endpoint_rejected_non_durable;
  out["under_linking"]["relation_endpoint_rejected_ungrounded"]
      = relation_endpoint_rejected_ungrounded;
  out["under_linking"]["has_label_edges_after"] = has_label_edges;
  out["definitions"]
      = "All counters are privacy-safe aggregates. Query-overlap counters use "
        "token overlap between the current user message and the candidate "
        "prompt packet. Label-boost counters mean at least one ranked Cortext "
        "candidate had label_graph_boost or label_match_count. Under-rag/stm "
        "counters use local Nemotron compact-policy judge "
        "relevance/sufficiency when "
        "present. Source-mapping counters compare durable derived_from sources "
        "with hydratable source content and count durable label nodes with "
        "source-backed experiences behind them. Addressability source-oracle "
        "counters measure whether any source memory behind durable labels or "
        "facts overlaps the live query without recording that source text. "
        "Relation-endpoint counters measure "
        "whether relabeler relation endpoints aligned with durable labels "
        "without recording endpoint text. Evidence-window counters record only "
        "aggregate label/evidence coverage for selected relabeler evidence and "
        "full source text. Under-admission counters record aggregate extractor "
        "candidate counts, post-processing reject reasons, and label insertion "
        "sources. None of these counters include label strings or message text.";
  return out;
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
      else if (arg == "--db")
        cfg.db_path = require_value ();
      else if (arg == "--out")
        cfg.output_path = require_value ();
      else if (arg == "--models")
        cfg.models_dir = require_value ();
      else if (arg == "--label-bank")
        cfg.label_bank_path = require_value ();
      else if (arg == "--skip-messages")
        cfg.skip_messages = std::stoi (require_value ());
      else if (arg == "--max-messages")
        cfg.max_messages = std::stoi (require_value ());
      else if (arg == "--warmup-messages")
        cfg.warmup_messages = std::stoi (require_value ());
      else if (arg == "--probe-stride")
        cfg.probe_stride = std::stoi (require_value ());
      else if (arg == "--min-probe-query-tokens")
        cfg.min_probe_query_tokens = std::stoi (require_value ());
      else if (arg == "--rag-top-k")
        cfg.rag_top_k = std::stoi (require_value ());
      else if (arg == "--max-injected-memories")
        cfg.max_injected_memories = std::stoi (require_value ());
      else if (arg == "--active-history-token-budget")
        cfg.active_history_token_budget = std::stoi (require_value ());
      else if (arg == "--focus")
        cfg.focus = std::stod (require_value ());
      else if (arg == "--sensitivity")
        cfg.sensitivity = std::stod (require_value ());
      else if (arg == "--stability")
        cfg.stability = std::stod (require_value ());
      else if (arg == "--consolidate-every")
        cfg.consolidate_every = std::stoi (require_value ());
      else if (arg == "--judge-model")
        cfg.judge_model = require_value ();
      else if (arg == "--judge-limit")
        cfg.judge_limit = std::stoi (require_value ());
      else if (arg == "--fact-prompt-k")
        cfg.fact_prompt_k = std::stoi (require_value ());
      else if (arg == "--source-stm-recent-k")
        cfg.source_stm_recent_k = std::stoi (require_value ());
	      else if (arg == "--source-ltm-lexical-k")
	        cfg.source_ltm_lexical_k = std::stoi (require_value ());
	      else if (arg == "--stratified-sample-messages")
	        cfg.stratified_sample_messages = std::stoi (require_value ());
	      else if (arg == "--media-adjacent-min")
	        cfg.media_adjacent_min = std::stoi (require_value ());
	      else if (arg == "--rolling-probe-target")
	        cfg.rolling_probe_target = std::stoi (require_value ());
	      else if (arg == "--deep")
	        cfg.deep_consolidation = true;
      else if (arg == "--rolling-eval")
        {
          cfg.rolling_eval = true;
          cfg.max_messages = -1;
          cfg.consolidate_every = 0;
          cfg.daily_consolidation = true;
          cfg.deep_consolidation = true;
          cfg.graph_expanded_rag_bakeoff = true;
        }
      else if (arg == "--daily-consolidation")
        {
          cfg.daily_consolidation = true;
          cfg.consolidate_every = 0;
        }
      else if (arg == "--message-count-consolidation")
        cfg.daily_consolidation = false;
      else if (arg == "--no-label-bank")
        cfg.use_label_bank = false;
      else if (arg == "--no-judge")
        cfg.judge_enabled = false;
      else if (arg == "--fact-prompt-bakeoff")
        cfg.fact_prompt_bakeoff = true;
      else if (arg == "--source-tagged-bakeoff")
        cfg.source_tagged_bakeoff = true;
      else if (arg == "--prompt-policy-bakeoff")
        cfg.prompt_policy_bakeoff = true;
      else if (arg == "--compact-policy-bakeoff")
        cfg.compact_policy_bakeoff = true;
      else if (arg == "--stm-graph-bakeoff")
        cfg.stm_graph_bakeoff = true;
      else if (arg == "--graph-expanded-rag-bakeoff")
        cfg.graph_expanded_rag_bakeoff = true;
      else
        throw std::runtime_error ("unknown argument: " + arg);
    }
  return cfg;
}

} // namespace

int
main (int argc, char **argv)
{
  try
    {
	      Config cfg = ParseArgs (argc, argv);
	      auto messages = ParseMessages (
		  chat_replay::DiscoverTranscript (cfg.input_dir));
	      const int parsed_messages = static_cast<int> (messages.size ());
	      MediaIndex media_index = BuildMediaIndex (cfg.input_dir);
	      AnnotateMediaAdjacency (messages, media_index);
	      const int parsed_media_adjacent_messages = static_cast<int> (
	          std::count_if (messages.begin (), messages.end (),
	                         [] (const Message &message) {
	                           return message.media_adjacent;
	                         }));
	      if (cfg.skip_messages > 0)
	        {
	          const auto skip = std::min (static_cast<size_t> (cfg.skip_messages),
	                                      messages.size ());
	          messages.erase (messages.begin (), messages.begin () + skip);
	        }
	      if (cfg.rolling_eval && cfg.stratified_sample_messages > 0)
	        {
	          messages = StratifiedSampleMessages (
	              messages, cfg.stratified_sample_messages,
	              cfg.media_adjacent_min);
	        }
	      if (cfg.max_messages >= 0
	          && static_cast<int> (messages.size ()) > cfg.max_messages)
	        messages.resize (static_cast<size_t> (cfg.max_messages));
	      const int sampled_media_adjacent_messages = static_cast<int> (
	          std::count_if (messages.begin (), messages.end (),
	                         [] (const Message &message) {
	                           return message.media_adjacent;
	                         }));
	      const auto probe_indices = cfg.rolling_eval
	                                     ? BuildProbeIndices (
	                                         messages, cfg.warmup_messages,
	                                         cfg.rolling_probe_target,
	                                         cfg.min_probe_query_tokens)
	                                     : std::set<int> ();

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
      auto rag_encoder_selection
          = cortext::internal::CreatePreferredTextEncoder (cfg.models_dir);
      auto vector_rag_store
          = cortext::SQLiteStore::Create (cfg.db_path.string ());
      std::unique_ptr<cortext::SQLiteStore> fact_prompt_store;
      if (cfg.fact_prompt_bakeoff)
        fact_prompt_store = cortext::SQLiteStore::Create (cfg.db_path.string ());
      std::unique_ptr<cortext::SQLiteStore> stm_graph_store;
      if (cfg.stm_graph_bakeoff)
        stm_graph_store = cortext::SQLiteStore::Create (cfg.db_path.string ());
      std::unique_ptr<cortext::SQLiteStore> graph_expanded_rag_store;
      if (cfg.graph_expanded_rag_bakeoff)
        graph_expanded_rag_store
            = cortext::SQLiteStore::Create (cfg.db_path.string ());
      std::vector<RagDoc> rag_docs;
      Aggregate cortext_agg;
      Aggregate cortext_ltm_agg;
      Aggregate rag_agg;
      Aggregate lexical_rag_agg;
      Aggregate normal_rag_agg;
      Aggregate graph_expanded_rag_agg;
      Aggregate full_history_agg;
      QualityAggregate cortext_quality;
      QualityAggregate cortext_ltm_quality;
      QualityAggregate rag_quality;
      QualityAggregate lexical_rag_quality;
      QualityAggregate normal_rag_quality;
      QualityAggregate full_history_quality;
      QualityAggregate fact_replace_quality;
      QualityAggregate fact_union_quality;
      QualityAggregate posthoc_fact_replace_quality;
      QualityAggregate posthoc_fact_union_quality;
      QualityAggregate source_current_quality;
      QualityAggregate source_wm_history_quality;
      QualityAggregate source_stm_recent_quality;
      QualityAggregate source_ltm_lexical_quality;
      QualityAggregate source_stm_ltm_union_quality;
      QualityAggregate source_normal_rag_quality;
      QualityAggregate source_full_history_quality;
      QualityAggregate policy_current_quality;
      QualityAggregate policy_stm_recent_quality;
      QualityAggregate policy_current_stm_quality;
      QualityAggregate policy_current_stm_ltm_quality;
      QualityAggregate policy_normal_rag_quality;
      QualityAggregate policy_full_history_quality;
      QualityAggregate compact_cortext_ltm_quality;
      QualityAggregate compact_stm_recent_quality;
      QualityAggregate compact_ltm_lexical_quality;
      QualityAggregate compact_stm_ltm_union_quality;
      QualityAggregate compact_normal_rag_quality;
      QualityAggregate compact_full_history_quality;
      QualityAggregate stm_graph_raw_quality;
      QualityAggregate stm_graph_relabel_quality;
      QualityAggregate stm_graph_normal_rag_quality;
      QualityAggregate stm_graph_full_history_quality;
      QualityAggregate graph_expanded_normal_rag_quality;
      QualityAggregate graph_expanded_cortext_ltm_quality;
      QualityAggregate graph_expanded_rag_quality;
      QualityAggregate graph_expanded_full_history_quality;
      CompactionAggregate normal_rag_compaction;
      nlohmann::json probes = nlohmann::json::array ();
      std::vector<PendingFactPromptProbe> pending_fact_prompts;
      std::vector<SourceTaggedProbe> source_tagged_probes;

      int durable_processed = 0;
      int consolidation_runs = 0;
      int daily_consolidation_runs = 0;
      int current_day_bucket = -1;
      auto run_started = std::chrono::steady_clock::now ();

      for (int i = 0; i < static_cast<int> (messages.size ()); ++i)
        {
	          const auto &msg = messages[static_cast<size_t> (i)];
	          const std::string source
	              = msg.from_contact ? kContactSourceId : kUserSourceId;
	          const std::string message_text = EvalText (msg);
	          const auto query_tokens = Tokens (message_text);
          const int message_day_bucket = LocalDayBucket (msg.timestamp);
          if (cfg.daily_consolidation && durable_processed > 0
              && current_day_bucket >= 0
              && message_day_bucket != current_day_bucket)
            {
              engine->Consolidate (
                  cfg.deep_consolidation ? cortext::ConsolidationMode::Both
                                         : cortext::ConsolidationMode::Shallow);
              ++consolidation_runs;
              ++daily_consolidation_runs;
            }
          current_day_bucket = message_day_bucket;

	          const bool should_probe
	              = cfg.rolling_eval
	                    ? probe_indices.count (i) > 0
	                    : (i >= cfg.warmup_messages && cfg.probe_stride > 0
	                       && i % cfg.probe_stride == 0
	                       && static_cast<int> (query_tokens.size ())
	                              >= cfg.min_probe_query_tokens);

          auto ingest_started = std::chrono::steady_clock::now ();
          auto ctx = engine->ProcessTextAt (message_text, source,
                                            msg.timestamp);
          auto ingest_ended = std::chrono::steady_clock::now ();
          const double cortext_latency_ms
              = std::chrono::duration<double, std::milli> (ingest_ended
                                                           - ingest_started)
                    .count ();

	          if (should_probe)
	            {

              int cortext_memory_items = 0;
              int cortext_working_items = 0;
              const int raw_retrieved_memory_items
                  = static_cast<int> (ctx.retrieved_memory.size ());
              const auto injected_memories = FilterInjectedMemories (
                  ctx.retrieved_memory, ctx.working_memory,
                  cfg.max_injected_memories);
              cortext_memory_items = static_cast<int> (injected_memories.size ());
              const std::string injected_system
                  = BuildInjectedSystemPrompt (injected_memories);
              const std::string cortext_memory_context = injected_system;
              const std::string cortext_working_context
                  = BuildCortextWorkingContext (ctx, &cortext_working_items);
              const std::string cortext_ltm_context = BuildMemoryListContext (
                  injected_memories, static_cast<std::size_t> (cfg.rag_top_k),
                  "cortext_ltm");
              const auto lexical_top
                  = RagTopK (rag_docs, query_tokens, msg.timestamp,
                             cfg.rag_top_k);
              const int vector_rag_item_limit = std::max (
                  1, cortext::core::RetrievalGraphExpandedRagMaxItems (
                         cortext_cfg.focus, cortext_cfg.stability));
              const auto normal_rag_started
                  = std::chrono::steady_clock::now ();
              std::vector<float> rag_query_embedding;
              rag_encoder_selection.encoder->EncodeText (message_text,
                                                         rag_query_embedding);
              rag_query_embedding = RetrievalEmbeddingViewForBenchmark (
                  rag_query_embedding);
              const VectorRagPacket vector_rag_packet = BuildVectorRagPacket (
                  *vector_rag_store, rag_docs, rag_query_embedding,
                  msg.timestamp, cortext_cfg.focus, cortext_cfg.stability,
                  static_cast<std::size_t> (vector_rag_item_limit));
              const auto normal_rag_retrieval_ended
                  = std::chrono::steady_clock::now ();
              const double normal_rag_retrieval_latency_ms
                  = std::chrono::duration<double, std::milli> (
                        normal_rag_retrieval_ended - normal_rag_started)
                        .count ();
              const std::string rag_context = BuildSimpleRagSystemPrompt (
                  vector_rag_packet.docs,
                  static_cast<std::size_t> (vector_rag_item_limit));
              const std::string lexical_rag_system_context = BuildSimpleRagSystemPrompt (
                  lexical_top, static_cast<std::size_t> (cfg.rag_top_k));
              const std::string lexical_rag_context
                  = BuildRagContext (lexical_top);
	              const int cortext_ltm_context_items = std::min (
	                  static_cast<int> (injected_memories.size ()), cfg.rag_top_k);
	              int active_history_items = 0;
	              const CompactedHistory compacted_history
	                  = BuildCompactedActiveHistoryContext (
	                      rag_docs, cfg.active_history_token_budget);
              AddCompaction (normal_rag_compaction, compacted_history);
	              active_history_items
	                  = compacted_history.raw_items
	                    + (compacted_history.compacted_items > 0 ? 1 : 0);
	              const std::string active_history_context
	                  = compacted_history.context;
              std::ostringstream full_history;
              for (const auto &doc : rag_docs)
                full_history << doc.source_id << ": " << doc.text << "\n";

	              const long long user_tokens = EstimateTokens (message_text.size ());
              const long long cortext_memory_tokens
                  = EstimateTokens (cortext_memory_context.size ());
              const long long cortext_working_tokens
                  = EstimateTokens (cortext_working_context.size ());
	              const std::size_t cortext_prompt_chars
	                  = ChatDemoCortextPromptChars (
	                      ctx.working_memory, injected_system, message_text);
	              const std::size_t rag_prompt_chars = ChatDemoRagPromptChars (
	                  vector_rag_packet.docs, message_text,
	                  static_cast<std::size_t> (vector_rag_item_limit));
              const std::size_t lexical_rag_prompt_chars
                  = lexical_rag_system_context.size () + message_text.size ();
              const long long rag_context_tokens
                  = EstimateTokens (rag_prompt_chars) - user_tokens;
              const long long cortext_ltm_tokens
                  = EstimateTokens (cortext_ltm_context.size ());
              const long long lexical_rag_tokens
                  = EstimateTokens (lexical_rag_prompt_chars) - user_tokens;
              const long long active_history_tokens
                  = EstimateTokens (active_history_context.size ());
              const long long full_history_context_tokens
                  = EstimateTokens (full_history.str ().size ());
              const long long cortext_total_context_tokens
                  = EstimateTokens (cortext_prompt_chars) - user_tokens;
              const long long rag_total_context_tokens = rag_context_tokens;
              const std::string normal_rag_context
                  = active_history_context + rag_context;
              const long long normal_rag_total_context_tokens
                  = active_history_tokens + rag_context_tokens;
              const auto normal_rag_ended = std::chrono::steady_clock::now ();
              const double normal_rag_total_latency_ms
                  = std::chrono::duration<double, std::milli> (
                        normal_rag_ended - normal_rag_started)
                        .count ();
              const auto stm_recent_docs
                  = RecentDocs (rag_docs, cfg.source_stm_recent_k);
              const auto ltm_lexical_docs
                  = RagTopK (rag_docs, query_tokens, msg.timestamp,
                             cfg.source_ltm_lexical_k);
              const auto stm_ltm_union_docs = UnionDocs (
                  stm_recent_docs, ltm_lexical_docs,
                  cfg.max_injected_memories > 0 ? cfg.max_injected_memories
                                                : cfg.source_stm_recent_k
                                                      + cfg.source_ltm_lexical_k);
              const std::string stm_recent_context
                  = BuildRagContext (stm_recent_docs);
              const std::string ltm_lexical_context
                  = BuildRagContext (ltm_lexical_docs);
              const std::string stm_ltm_union_context
                  = BuildRagContext (stm_ltm_union_docs);
              const std::string current_context
                  = cortext_working_context + cortext_memory_context;
              const std::string current_stm_context
                  = current_context + "\n[stm_recent]\n"
                    + stm_recent_context;
              const std::string current_stm_ltm_context
                  = current_context + "\n[stm_ltm_union]\n"
                    + stm_ltm_union_context;

              AddAggregate (cortext_agg,
                            cortext_total_context_tokens + user_tokens,
                            cortext_total_context_tokens,
                            cortext_memory_tokens, cortext_working_tokens,
                            cortext_working_context.size ()
                                + cortext_memory_context.size (),
                            cortext_memory_items + cortext_working_items,
                            Overlap (query_tokens,
                                     cortext_working_context
                                         + cortext_memory_context),
                            cortext_latency_ms);
              AddAggregate (rag_agg, rag_total_context_tokens + user_tokens,
                            rag_total_context_tokens, rag_context_tokens,
	                            0, rag_prompt_chars - message_text.size (),
                            lexical_top.size (),
                            Overlap (query_tokens, rag_context),
                            0.0);
              AddAggregate (cortext_ltm_agg, cortext_ltm_tokens + user_tokens,
                            cortext_ltm_tokens, cortext_ltm_tokens, 0,
                            cortext_ltm_context.size (),
                            cortext_ltm_context_items,
                            Overlap (query_tokens, cortext_ltm_context),
                            0.0);
              AddAggregate (lexical_rag_agg,
                            lexical_rag_tokens + user_tokens,
                            lexical_rag_tokens, lexical_rag_tokens, 0,
                            lexical_rag_system_context.size (),
                            lexical_top.size (),
                            Overlap (query_tokens, lexical_rag_context),
                            0.0);
              AddAggregate (normal_rag_agg,
                            normal_rag_total_context_tokens + user_tokens,
                            normal_rag_total_context_tokens,
                            rag_context_tokens, active_history_tokens,
                            active_history_context.size ()
	                                + rag_prompt_chars - message_text.size (),
                            active_history_items
                                + vector_rag_packet.docs.size (),
                            Overlap (query_tokens, normal_rag_context),
                            normal_rag_total_latency_ms);
              GraphExpandedRagPacket graph_expanded_packet;
              std::string graph_expanded_rag_context;
              long long graph_expanded_rag_source_tokens = 0;
              long long graph_expanded_rag_total_context_tokens = 0;
              if (cfg.graph_expanded_rag_bakeoff
                  && graph_expanded_rag_store)
                {
                  graph_expanded_packet = BuildGraphExpandedRagPacket (
                      *graph_expanded_rag_store, *engine, rag_docs,
                      vector_rag_packet.docs, query_tokens, cortext_cfg.focus,
                      cortext_cfg.sensitivity, cortext_cfg.stability,
                      static_cast<std::size_t> (std::max (
                          1, cfg.max_injected_memories)));
                  graph_expanded_rag_context
                      = cortext_working_context
                        + graph_expanded_packet.chat_context;
                  graph_expanded_rag_source_tokens = EstimateTokens (
                      graph_expanded_packet.chat_context.size ());
                  graph_expanded_rag_total_context_tokens
                      = cortext_working_tokens
                        + graph_expanded_rag_source_tokens;
                  AddAggregate (
                      graph_expanded_rag_agg,
                      graph_expanded_rag_total_context_tokens + user_tokens,
                      graph_expanded_rag_total_context_tokens,
                      graph_expanded_rag_source_tokens, 0,
                      cortext_working_context.size ()
                          + graph_expanded_packet.chat_context.size (),
                      cortext_working_items
                          + graph_expanded_packet.ranked_memory_ids.size (),
                      Overlap (query_tokens, graph_expanded_rag_context),
                      0.0);
                }
              AddAggregate (full_history_agg,
                            full_history_context_tokens + user_tokens,
                            full_history_context_tokens, 0,
                            full_history_context_tokens,
                            full_history.str ().size (), rag_docs.size (),
                            Overlap (query_tokens, full_history.str ()), 0.0);

              nlohmann::json probe;
              probe["message_index"] = i;
              probe["timestamp"] = msg.timestamp;
              probe["query_tokens"] = query_tokens.size ();
              probe["cortext_memory_context_tokens"] = cortext_memory_tokens;
              probe["cortext_working_memory_tokens"] = cortext_working_tokens;
              probe["simple_rag_active_history_tokens"] = 0;
              probe["active_history_items"] = active_history_items;
              probe["cortext_total_context_tokens"]
                  = cortext_total_context_tokens;
              probe["cortext_memory_context_items"] = cortext_memory_items;
              probe["cortext_working_memory_items"] = cortext_working_items;
	              probe["cortext_overlap"]
                  = Overlap (query_tokens,
                             cortext_working_context + cortext_memory_context);
	              probe["cortext_probe_latency_ms"] = cortext_latency_ms;
	              probe["cortext_latency_ms"] = cortext_latency_ms;
              probe["cortext_probe_policy"]
                  = "single_durable_chat_turn_reused_for_probe_and_ingest";
	              probe["retrieval_debug"] = RetrievalDebugJson ();
	              probe["retrieval_rejections"] = RetrievalRejectionJson ();
	              probe["retrieval_evidence_packets"]
	                  = RetrievalEvidencePacketJson ();
	              probe["retrieval_summary"] = RetrievalSummaryJson ();
	              probe["cortext_raw_retrieved_memory_items"]
	                  = raw_retrieved_memory_items;
              probe["max_injected_memories"] = cfg.max_injected_memories;
              probe["rag_context_tokens"] = rag_context_tokens;
              probe["rag_total_context_tokens"] = rag_total_context_tokens;
              probe["rag_context_items"] = vector_rag_packet.docs.size ();
              probe["rag_overlap"] = Overlap (query_tokens, rag_context);
              probe["rag_retrieval"] = "vector";
              probe["rag_vector_query_rows"] = vector_rag_packet.query_rows;
              probe["rag_vector_query_embedding_bytes"]
                  = vector_rag_packet.query_embedding_bytes;
              probe["rag_vector_search_k"] = vector_rag_packet.vector_search_k;
              probe["rag_vector_candidate_rows"]
                  = vector_rag_packet.candidate_rows;
              probe["rag_vector_prior_chat_rows"]
                  = vector_rag_packet.prior_chat_rows;
              probe["rag_vector_best_distance"]
                  = vector_rag_packet.best_distance;
              probe["cortext_ltm_tokens"] = cortext_ltm_tokens;
              probe["cortext_ltm_items"] = cortext_ltm_context_items;
              probe["cortext_ltm_overlap"]
                  = Overlap (query_tokens, cortext_ltm_context);
              probe["lexical_rag_tokens"] = lexical_rag_tokens;
              probe["lexical_rag_items"] = lexical_top.size ();
              probe["lexical_rag_overlap"]
                  = Overlap (query_tokens, lexical_rag_context);
              probe["normal_rag_total_context_tokens"]
                  = normal_rag_total_context_tokens;
              probe["normal_rag_active_history_tokens"] = active_history_tokens;
              probe["normal_rag_raw_history_tokens"]
                  = compacted_history.raw_tokens;
              probe["normal_rag_compaction_events"]
                  = compacted_history.compaction_events;
              probe["normal_rag_compacted_history_items"]
                  = compacted_history.compacted_items;
              probe["normal_rag_raw_history_items"]
                  = compacted_history.raw_items;
              probe["normal_rag_compacted_original_tokens"]
                  = compacted_history.compacted_original_tokens;
              probe["normal_rag_compacted_summary_tokens"]
                  = compacted_history.compacted_summary_tokens;
              probe["normal_rag_context_tokens"] = rag_context_tokens;
              probe["normal_rag_context_items"]
                  = active_history_items + vector_rag_packet.docs.size ();
              probe["normal_rag_overlap"]
                  = Overlap (query_tokens, normal_rag_context);
              probe["normal_rag_retrieval_latency_ms"]
                  = normal_rag_retrieval_latency_ms;
              probe["normal_rag_total_latency_ms"]
                  = normal_rag_total_latency_ms;
              if (cfg.graph_expanded_rag_bakeoff)
                {
                  nlohmann::json graph_rag_json;
                  graph_rag_json["enabled"] = true;
                  graph_rag_json["total_context_tokens"]
                      = graph_expanded_rag_total_context_tokens;
                  graph_rag_json["source_context_tokens"]
                      = graph_expanded_rag_source_tokens;
                  graph_rag_json["active_history_tokens"]
                      = 0;
                  graph_rag_json["working_memory_tokens"]
                      = cortext_working_tokens;
                  graph_rag_json["working_memory_items"]
                      = cortext_working_items;
                  graph_rag_json["seed_memory_ids"]
                      = MemoryIdsJson (graph_expanded_packet.seed_memory_ids);
                  graph_rag_json["expanded_memory_ids"]
                      = MemoryIdsJson (
                          graph_expanded_packet.expanded_memory_ids);
                  graph_rag_json["ranked_memory_ids"]
                      = MemoryIdsJson (
                          graph_expanded_packet.ranked_memory_ids);
                  graph_rag_json["seed_items"]
                      = graph_expanded_packet.seed_memory_ids.size ();
                  graph_rag_json["ranked_items"]
                      = graph_expanded_packet.ranked_memory_ids.size ();
                  graph_rag_json["missing_seed_count"]
                      = graph_expanded_packet.missing_seed_count;
                  graph_rag_json["temporal_neighbor_count"]
                      = graph_expanded_packet.temporal_neighbor_count;
                  graph_rag_json["graph_candidate_count"]
                      = graph_expanded_packet.graph_candidate_count;
                  graph_rag_json["fact_candidate_count"]
                      = graph_expanded_packet.fact_candidate_count;
                  graph_rag_json["compact_label_count"]
                      = graph_expanded_packet.compact_label_count;
                  graph_rag_json["compact_fact_count"]
                      = graph_expanded_packet.compact_fact_count;
                  graph_rag_json["best_graph_score"]
                      = graph_expanded_packet.best_graph_score;
                  graph_rag_json["best_fact_score"]
                      = graph_expanded_packet.best_fact_score;
                  graph_rag_json["overlap"]
                      = Overlap (query_tokens, graph_expanded_rag_context);
                  graph_rag_json["source_overlap"]
                      = Overlap (query_tokens,
                                 graph_expanded_packet.chat_context);

                  const bool should_judge_graph_rag
                      = cfg.judge_enabled
                        && (cfg.judge_limit < 0
                            || graph_expanded_rag_quality.judged
                                   < cfg.judge_limit);
                  if (should_judge_graph_rag)
                    {
                      auto graph_rag_quality = JudgeGraphExpandedRagContexts (
	                          cfg, i, message_text, normal_rag_context,
                          cortext_ltm_context, graph_expanded_rag_context,
                          full_history.str ());
                      if (graph_rag_quality)
                        {
                          AddQuality (graph_expanded_normal_rag_quality,
                                      *graph_rag_quality, "normal_rag");
                          AddQuality (graph_expanded_cortext_ltm_quality,
                                      *graph_rag_quality, "cortext_ltm");
                          AddQuality (graph_expanded_rag_quality,
                                      *graph_rag_quality,
                                      "graph_expanded_rag");
                          AddQuality (graph_expanded_full_history_quality,
                                      *graph_rag_quality, "full_history");
                          nlohmann::json quality_probe;
                          const std::vector<std::string> systems = {
                            "normal_rag", "cortext_ltm",
                            "graph_expanded_rag", "full_history"
                          };
	                          for (const auto &system : systems)
	                            {
	                              const std::vector<std::string> fields = {
	                                "relevance", "sufficiency", "noise",
	                                "source_grounding",
	                                "temporal_correctness",
	                                "media_usefulness"
	                              };
	                              for (const auto &field : fields)
	                                {
	                                  quality_probe[system + "_" + field]
	                                      = ScoreValue (*graph_rag_quality,
	                                                    system, field);
	                                }
	                            }
                          quality_probe["winner"]
                              = graph_rag_quality->value ("winner",
                                                          "unknown");
                          graph_rag_json["quality"]
                              = std::move (quality_probe);
                        }
                      else
                        {
                          graph_rag_json["quality_error"] = "judge_failed";
                        }
                    }
                  probe["graph_expanded_rag_bakeoff"]
                      = std::move (graph_rag_json);
                }
              const bool should_judge
                  = cfg.judge_enabled
                    && (cfg.judge_limit < 0
                        || cortext_quality.judged < cfg.judge_limit);
              if (cfg.source_tagged_bakeoff)
                {
                  nlohmann::json source_json;
                  source_json["wm_history_tokens"] = active_history_tokens;
                  source_json["wm_history_items"] = active_history_items;
                  source_json["stm_recent_tokens"]
                      = EstimateTokens (stm_recent_context.size ());
                  source_json["stm_recent_items"] = stm_recent_docs.size ();
                  source_json["ltm_lexical_tokens"]
                      = EstimateTokens (ltm_lexical_context.size ());
                  source_json["ltm_lexical_items"] = ltm_lexical_docs.size ();
                  source_json["stm_ltm_union_tokens"]
                      = EstimateTokens (stm_ltm_union_context.size ());
                  source_json["stm_ltm_union_items"]
                      = stm_ltm_union_docs.size ();
                  source_json["normal_rag_tokens"]
                      = normal_rag_total_context_tokens;
                  source_json["normal_rag_items"]
                      = active_history_items + lexical_top.size ();
                  source_json["wm_history_overlap"]
                      = Overlap (query_tokens, active_history_context);
                  source_json["stm_recent_overlap"]
                      = Overlap (query_tokens, stm_recent_context);
                  source_json["ltm_lexical_overlap"]
                      = Overlap (query_tokens, ltm_lexical_context);
                  source_json["stm_ltm_union_overlap"]
                      = Overlap (query_tokens, stm_ltm_union_context);
                  probe["source_tagged_bakeoff"] = std::move (source_json);

                  SourceTaggedProbe source_probe;
                  source_probe.probe_json_index = probes.size ();
                  source_probe.message_index = i;
	                  source_probe.query = message_text;
                  source_probe.current_context
                      = cortext_working_context + cortext_memory_context;
                  source_probe.wm_history_context = active_history_context;
                  source_probe.stm_recent_context = stm_recent_context;
                  source_probe.ltm_lexical_context = ltm_lexical_context;
                  source_probe.stm_ltm_union_context = stm_ltm_union_context;
                  source_probe.normal_rag_context = normal_rag_context;
                  source_probe.full_history_context = full_history.str ();
                  source_tagged_probes.push_back (std::move (source_probe));
                }
              if (cfg.prompt_policy_bakeoff)
                {
                  nlohmann::json policy_json;
                  policy_json["stm_recent_k"] = cfg.source_stm_recent_k;
                  policy_json["ltm_lexical_k"] = cfg.source_ltm_lexical_k;
                  policy_json["current_tokens"]
                      = EstimateTokens (current_context.size ());
                  policy_json["stm_recent_tokens"]
                      = EstimateTokens (stm_recent_context.size ());
                  policy_json["current_stm_tokens"]
                      = EstimateTokens (current_stm_context.size ());
                  policy_json["current_stm_ltm_tokens"]
                      = EstimateTokens (current_stm_ltm_context.size ());
                  policy_json["normal_rag_tokens"]
                      = normal_rag_total_context_tokens;
                  policy_json["full_history_tokens"]
                      = full_history_context_tokens;
                  policy_json["stm_recent_items"] = stm_recent_docs.size ();
                  policy_json["stm_ltm_union_items"]
                      = stm_ltm_union_docs.size ();
                  policy_json["current_overlap"]
                      = Overlap (query_tokens, current_context);
                  policy_json["stm_recent_overlap"]
                      = Overlap (query_tokens, stm_recent_context);
                  policy_json["current_stm_overlap"]
                      = Overlap (query_tokens, current_stm_context);
                  policy_json["current_stm_ltm_overlap"]
                      = Overlap (query_tokens, current_stm_ltm_context);

                  if (should_judge)
                    {
                      auto policy_quality = JudgePromptPolicyContexts (
	                          cfg, i, message_text, current_context,
                          stm_recent_context, current_stm_context,
                          current_stm_ltm_context, normal_rag_context,
                          full_history.str ());
                      if (policy_quality)
                        {
                          AddQuality (policy_current_quality, *policy_quality,
                                      "current");
                          AddQuality (policy_stm_recent_quality,
                                      *policy_quality, "stm_recent");
                          AddQuality (policy_current_stm_quality,
                                      *policy_quality, "current_stm");
                          AddQuality (policy_current_stm_ltm_quality,
                                      *policy_quality, "current_stm_ltm");
                          AddQuality (policy_normal_rag_quality,
                                      *policy_quality, "normal_rag");
                          AddQuality (policy_full_history_quality,
                                      *policy_quality, "full_history");
                          nlohmann::json quality_probe;
                          const std::vector<std::string> systems = {
                            "current", "stm_recent", "current_stm",
                            "current_stm_ltm", "normal_rag", "full_history"
                          };
                          for (const auto &system : systems)
                            {
                              quality_probe[system + "_relevance"]
                                  = ScoreValue (*policy_quality, system,
                                                "relevance");
                              quality_probe[system + "_sufficiency"]
                                  = ScoreValue (*policy_quality, system,
                                                "sufficiency");
                              quality_probe[system + "_noise"]
                                  = ScoreValue (*policy_quality, system,
                                                "noise");
                            }
                          quality_probe["winner"]
                              = policy_quality->value ("winner", "unknown");
                          policy_json["quality"] = std::move (quality_probe);
                        }
                      else
                        {
                          policy_json["quality_error"] = "judge_failed";
                        }
                    }
                  probe["prompt_policy_bakeoff"] = std::move (policy_json);
                }
              if (cfg.compact_policy_bakeoff)
                {
                  nlohmann::json compact_json;
                  compact_json["stm_recent_k"] = cfg.source_stm_recent_k;
                  compact_json["ltm_lexical_k"] = cfg.source_ltm_lexical_k;
                  compact_json["cortext_ltm_tokens"]
                      = EstimateTokens (cortext_ltm_context.size ());
                  compact_json["stm_recent_tokens"]
                      = EstimateTokens (stm_recent_context.size ());
                  compact_json["ltm_lexical_tokens"]
                      = EstimateTokens (ltm_lexical_context.size ());
                  compact_json["stm_ltm_union_tokens"]
                      = EstimateTokens (stm_ltm_union_context.size ());
                  compact_json["normal_rag_tokens"]
                      = normal_rag_total_context_tokens;
                  compact_json["full_history_tokens"]
                      = full_history_context_tokens;
                  compact_json["stm_recent_items"] = stm_recent_docs.size ();
                  compact_json["ltm_lexical_items"] = ltm_lexical_docs.size ();
                  compact_json["stm_ltm_union_items"]
                      = stm_ltm_union_docs.size ();
                  compact_json["cortext_ltm_overlap"]
                      = Overlap (query_tokens, cortext_ltm_context);
                  compact_json["stm_recent_overlap"]
                      = Overlap (query_tokens, stm_recent_context);
                  compact_json["ltm_lexical_overlap"]
                      = Overlap (query_tokens, ltm_lexical_context);
                  compact_json["stm_ltm_union_overlap"]
                      = Overlap (query_tokens, stm_ltm_union_context);
                  compact_json["adaptive_choice"]
                      = EstimateTokens (stm_ltm_union_context.size ())
                                <= normal_rag_total_context_tokens
                            ? "stm_ltm_union"
                            : "stm_recent";

                  const bool should_judge_compact
                      = cfg.judge_enabled
                        && (cfg.judge_limit < 0
                            || compact_stm_recent_quality.judged
                                   < cfg.judge_limit);
                  if (should_judge_compact)
                    {
                      auto compact_quality = JudgeCompactPolicyContexts (
	                          cfg, i, message_text, cortext_ltm_context,
                          stm_recent_context, ltm_lexical_context,
                          stm_ltm_union_context, normal_rag_context,
                          full_history.str ());
                      if (compact_quality)
                        {
                          AddQuality (compact_cortext_ltm_quality,
                                      *compact_quality, "cortext_ltm");
                          AddQuality (compact_stm_recent_quality,
                                      *compact_quality, "stm_recent");
                          AddQuality (compact_ltm_lexical_quality,
                                      *compact_quality, "ltm_lexical");
                          AddQuality (compact_stm_ltm_union_quality,
                                      *compact_quality, "stm_ltm_union");
                          AddQuality (compact_normal_rag_quality,
                                      *compact_quality, "normal_rag");
                          AddQuality (compact_full_history_quality,
                                      *compact_quality, "full_history");
                          nlohmann::json quality_probe;
                          const std::vector<std::string> systems = {
                            "cortext_ltm", "stm_recent", "ltm_lexical",
                            "stm_ltm_union", "normal_rag", "full_history"
                          };
                          for (const auto &system : systems)
                            {
                              quality_probe[system + "_relevance"]
                                  = ScoreValue (*compact_quality, system,
                                                "relevance");
                              quality_probe[system + "_sufficiency"]
                                  = ScoreValue (*compact_quality, system,
                                                "sufficiency");
                              quality_probe[system + "_noise"]
                                  = ScoreValue (*compact_quality, system,
                                                "noise");
                            }
                          quality_probe["winner"]
                              = compact_quality->value ("winner", "unknown");
                          compact_json["quality"] = std::move (quality_probe);
                        }
                      else
                        {
                          compact_json["quality_error"] = "judge_failed";
                        }
                    }
                  probe["compact_policy_bakeoff"] = std::move (compact_json);
                }
              if (cfg.stm_graph_bakeoff && stm_graph_store)
                {
                  const auto stm_packet = BuildStmGraphPacket (
                      *stm_graph_store, *engine, query_tokens,
                      static_cast<std::size_t> (std::max (
                          1, cfg.max_injected_memories)));
                  nlohmann::json graph_json;
                  graph_json["raw_tokens"]
                      = EstimateTokens (stm_packet.raw_context.size ());
                  graph_json["relabel_prune_tokens"]
                      = EstimateTokens (stm_packet.relabel_context.size ());
                  const std::string raw_graph_chat_context
                      = cortext_working_context + "\n[raw_stm_graph]\n"
                        + stm_packet.raw_context;
                  const std::string relabel_graph_chat_context
                      = cortext_working_context + "\n[relabel_prune_ltm]\n"
                        + stm_packet.relabel_context;
                  graph_json["raw_chat_tokens"]
                      = EstimateTokens (raw_graph_chat_context.size ());
                  graph_json["relabel_prune_chat_tokens"]
                      = EstimateTokens (relabel_graph_chat_context.size ());
                  graph_json["normal_rag_tokens"]
                      = normal_rag_total_context_tokens;
                  graph_json["full_history_tokens"]
                      = full_history_context_tokens;
                  graph_json["raw_items"] = stm_packet.raw_memory_ids.size ();
                  graph_json["relabel_prune_items"]
                      = stm_packet.relabel_memory_ids.size ();
                  graph_json["raw_cycles"] = stm_packet.raw_cycle_count;
                  graph_json["relabel_prune_cycles"]
                      = stm_packet.relabel_cycle_count;
                  graph_json["raw_positive_label_cycles"]
                      = stm_packet.raw_positive_label_cycles;
                  graph_json["relabel_positive_label_cycles"]
                      = stm_packet.relabel_positive_label_cycles;
                  graph_json["raw_positive_source_cycles"]
                      = stm_packet.raw_positive_source_cycles;
                  graph_json["relabel_positive_source_cycles"]
                      = stm_packet.relabel_positive_source_cycles;
                  graph_json["relabel_durable_candidate_count"]
                      = stm_packet.relabel_durable_candidate_count;
                  graph_json["relabel_fact_linked_candidate_count"]
                      = stm_packet.relabel_fact_linked_candidate_count;
                  graph_json["relabel_durable_source_count"]
                      = stm_packet.relabel_durable_source_count;
                  graph_json["raw_label_count"] = stm_packet.raw_label_count;
                  graph_json["relabel_label_count"]
                      = stm_packet.relabel_label_count;
                  graph_json["raw_best_label_overlap"]
                      = stm_packet.raw_best_label_overlap;
                  graph_json["relabel_best_label_overlap"]
                      = stm_packet.relabel_best_label_overlap;
                  graph_json["raw_best_source_overlap"]
                      = stm_packet.raw_best_source_overlap;
                  graph_json["relabel_best_source_overlap"]
                      = stm_packet.relabel_best_source_overlap;
                  graph_json["raw_overlap"]
                      = Overlap (query_tokens, stm_packet.raw_context);
                  graph_json["relabel_prune_overlap"]
                      = Overlap (query_tokens, stm_packet.relabel_context);
                  graph_json["raw_chat_overlap"]
                      = Overlap (query_tokens, raw_graph_chat_context);
                  graph_json["relabel_prune_chat_overlap"]
                      = Overlap (query_tokens, relabel_graph_chat_context);
                  graph_json["addressability"]
                      = BuildAddressabilityProbe (*stm_graph_store, *engine,
                                                  query_tokens);
                  const auto working_query_tokens = Tokens (
	                      message_text + "\n" + cortext_working_context);
                  graph_json["addressability_with_working_memory"]
                      = BuildAddressabilityProbe (*stm_graph_store, *engine,
                                                  working_query_tokens);

                  const bool should_judge_graph
                      = cfg.judge_enabled
                        && (cfg.judge_limit < 0
                            || stm_graph_raw_quality.judged
                                   < cfg.judge_limit);
                  if (should_judge_graph)
                    {
                      auto graph_quality = JudgeStmGraphContexts (
	                          cfg, i, message_text, raw_graph_chat_context,
                          relabel_graph_chat_context, normal_rag_context,
                          full_history.str ());
                      if (graph_quality)
                        {
                          AddQuality (stm_graph_raw_quality, *graph_quality,
                                      "raw_stm_graph");
                          AddQuality (stm_graph_relabel_quality,
                                      *graph_quality, "relabel_prune_ltm");
                          AddQuality (stm_graph_normal_rag_quality,
                                      *graph_quality, "normal_rag");
                          AddQuality (stm_graph_full_history_quality,
                                      *graph_quality, "full_history");
                          nlohmann::json quality_probe;
                          const std::vector<std::string> systems = {
                            "raw_stm_graph", "relabel_prune_ltm",
                            "normal_rag", "full_history"
                          };
                          for (const auto &system : systems)
                            {
                              quality_probe[system + "_relevance"]
                                  = ScoreValue (*graph_quality, system,
                                                "relevance");
                              quality_probe[system + "_sufficiency"]
                                  = ScoreValue (*graph_quality, system,
                                                "sufficiency");
                              quality_probe[system + "_noise"]
                                  = ScoreValue (*graph_quality, system,
                                                "noise");
                            }
                          quality_probe["winner"]
                              = graph_quality->value ("winner", "unknown");
                          graph_json["quality"] = std::move (quality_probe);
                        }
                      else
                        {
                          graph_json["quality_error"] = "judge_failed";
                        }
                    }
                  probe["stm_graph_bakeoff"] = std::move (graph_json);
                }
              probe["full_history_context_tokens"] = full_history_context_tokens;
              const std::vector<long long> current_memory_ids
                  = MemoryIdsFromContext (injected_memories);
              if (cfg.fact_prompt_bakeoff && fact_prompt_store)
                {
                  FactPromptProbe fact_probe = BuildFactPromptProbe (
                      *fact_prompt_store, msg.timestamp, cfg.fact_prompt_k);
                  const std::vector<long long> fact_replace_ids
                      = TakeUnique (fact_probe.evidence_memory_ids,
                                    static_cast<std::size_t> (
                                        cfg.max_injected_memories));
                  const std::vector<long long> fact_union_ids
                      = UnionThenExistingRank (
                          fact_replace_ids, current_memory_ids,
                          static_cast<std::size_t> (cfg.max_injected_memories));
                  const auto fact_replace_memories
                      = HydrateMemories (*engine, fact_replace_ids);
                  const auto fact_union_memories
                      = HydrateMemories (*engine, fact_union_ids);
                  const std::string fact_replace_context
                      = BuildInjectedSystemPrompt (fact_replace_memories);
                  const std::string fact_union_context
                      = BuildInjectedSystemPrompt (fact_union_memories);
                  const long long fact_replace_tokens
                      = EstimateTokens (fact_replace_context.size ());
                  const long long fact_union_tokens
                      = EstimateTokens (fact_union_context.size ());

                  nlohmann::json fact_json;
                  fact_json["fact_prompt_k"] = cfg.fact_prompt_k;
                  fact_json["fact_ids"] = MemoryIdsJson (fact_probe.fact_ids);
                  fact_json["evidence_memory_ids"]
                      = MemoryIdsJson (fact_probe.evidence_memory_ids);
                  fact_json["current_memory_ids"]
                      = MemoryIdsJson (current_memory_ids);
                  fact_json["fact_replace_memory_ids"]
                      = MemoryIdsJson (fact_replace_ids);
                  fact_json["fact_union_memory_ids"]
                      = MemoryIdsJson (fact_union_ids);
                  fact_json["top_fact_distance"]
                      = fact_probe.top_fact_distance;
                  fact_json["source_fact_cache_rows"]
                      = fact_probe.source_fact_cache_rows;
                  fact_json["fact_only_rows"] = fact_probe.fact_only_rows;
                  fact_json["query_rows"] = fact_probe.query_rows;
                  fact_json["query_embedding_bytes"]
                      = fact_probe.query_embedding_bytes;
                  fact_json["evidence_memory_count"]
                      = fact_probe.evidence_memory_ids.size ();
                  fact_json["fact_replace_items"]
                      = fact_replace_memories.size ();
                  fact_json["fact_union_items"] = fact_union_memories.size ();
                  fact_json["fact_replace_context_tokens"]
                      = fact_replace_tokens;
                  fact_json["fact_union_context_tokens"] = fact_union_tokens;
                  fact_json["fact_replace_overlap_with_current"]
                      = IntersectionCount (fact_replace_ids,
                                           current_memory_ids);
                  fact_json["fact_union_overlap_with_current"]
                      = IntersectionCount (fact_union_ids,
                                           current_memory_ids);
                  fact_json["fact_replace_query_overlap"]
                      = Overlap (query_tokens, fact_replace_context);
                  fact_json["fact_union_query_overlap"]
                      = Overlap (query_tokens, fact_union_context);

                  if (should_judge && fact_probe.query_rows > 0)
                    {
                      auto fact_quality = JudgeFactPromptContexts (
	                          cfg, i, message_text,
                          cortext_working_context + cortext_memory_context,
                          fact_replace_context, fact_union_context,
                          normal_rag_context, full_history.str ());
                      if (fact_quality)
                        {
                          AddQuality (fact_replace_quality, *fact_quality,
                                      "fact_replace");
                          AddQuality (fact_union_quality, *fact_quality,
                                      "fact_union");
                          nlohmann::json quality_probe;
                          quality_probe["current_relevance"] = ScoreValue (
                              *fact_quality, "current", "relevance");
                          quality_probe["current_sufficiency"] = ScoreValue (
                              *fact_quality, "current", "sufficiency");
                          quality_probe["current_noise"] = ScoreValue (
                              *fact_quality, "current", "noise");
                          quality_probe["fact_replace_relevance"]
                              = ScoreValue (*fact_quality, "fact_replace",
                                            "relevance");
                          quality_probe["fact_replace_sufficiency"]
                              = ScoreValue (*fact_quality, "fact_replace",
                                            "sufficiency");
                          quality_probe["fact_replace_noise"] = ScoreValue (
                              *fact_quality, "fact_replace", "noise");
                          quality_probe["fact_union_relevance"] = ScoreValue (
                              *fact_quality, "fact_union", "relevance");
                          quality_probe["fact_union_sufficiency"]
                              = ScoreValue (*fact_quality, "fact_union",
                                            "sufficiency");
                          quality_probe["fact_union_noise"]
                              = ScoreValue (*fact_quality, "fact_union",
                                            "noise");
                          quality_probe["normal_rag_relevance"]
                              = ScoreValue (*fact_quality, "normal_rag",
                                            "relevance");
                          quality_probe["normal_rag_sufficiency"]
                              = ScoreValue (*fact_quality, "normal_rag",
                                            "sufficiency");
                          quality_probe["normal_rag_noise"] = ScoreValue (
                              *fact_quality, "normal_rag", "noise");
                          quality_probe["full_history_relevance"]
                              = ScoreValue (*fact_quality, "full_history",
                                            "relevance");
                          quality_probe["full_history_sufficiency"]
                              = ScoreValue (*fact_quality, "full_history",
                                            "sufficiency");
                          quality_probe["full_history_noise"] = ScoreValue (
                              *fact_quality, "full_history", "noise");
                          quality_probe["winner"]
                              = fact_quality->value ("winner", "unknown");
                          fact_json["quality"] = std::move (quality_probe);
                        }
                      else
                        {
                          fact_json["quality_error"] = "judge_failed";
                        }
                    }
                  probe["fact_prompt_bakeoff"] = std::move (fact_json);
                }
              if (cfg.fact_prompt_bakeoff)
                {
                  PendingFactPromptProbe pending;
                  pending.probe_json_index = probes.size ();
                  pending.message_index = i;
                  pending.timestamp = msg.timestamp;
	                  pending.query = message_text;
                  pending.query_tokens = query_tokens;
                  pending.current_context
                      = cortext_working_context + cortext_memory_context;
                  pending.normal_rag_context = normal_rag_context;
                  pending.full_history_context = full_history.str ();
                  pending.current_memory_ids = current_memory_ids;
                  pending_fact_prompts.push_back (std::move (pending));
                }
              if (should_judge)
                {
                  auto quality = JudgeContexts (
	                      cfg, i, message_text,
                      current_context,
                      cortext_ltm_context, rag_context, lexical_rag_context,
                      normal_rag_context, full_history.str ());
                  if (quality)
                    {
                      AddQuality (cortext_quality, *quality, "cortext");
                      AddQuality (cortext_ltm_quality, *quality,
                                  "cortext_ltm");
                      AddQuality (rag_quality, *quality, "rag");
                      AddQuality (lexical_rag_quality, *quality,
                                  "lexical_rag");
                      AddQuality (normal_rag_quality, *quality, "normal_rag");
                      AddQuality (full_history_quality, *quality,
                                  "full_history");
                      nlohmann::json quality_probe;
                      quality_probe["cortext_relevance"]
                          = ScoreValue (*quality, "cortext", "relevance");
                      quality_probe["cortext_sufficiency"]
                          = ScoreValue (*quality, "cortext", "sufficiency");
                      quality_probe["cortext_noise"]
                          = ScoreValue (*quality, "cortext", "noise");
                      quality_probe["cortext_source_grounding"] = ScoreValue (
                          *quality, "cortext", "source_grounding");
                      quality_probe["cortext_temporal_correctness"]
                          = ScoreValue (*quality, "cortext",
                                        "temporal_correctness");
                      quality_probe["cortext_media_usefulness"] = ScoreValue (
                          *quality, "cortext", "media_usefulness");
                      quality_probe["cortext_ltm_relevance"]
                          = ScoreValue (*quality, "cortext_ltm", "relevance");
                      quality_probe["cortext_ltm_sufficiency"] = ScoreValue (
                          *quality, "cortext_ltm", "sufficiency");
                      quality_probe["cortext_ltm_noise"]
                          = ScoreValue (*quality, "cortext_ltm", "noise");
                      quality_probe["cortext_ltm_source_grounding"]
                          = ScoreValue (*quality, "cortext_ltm",
                                        "source_grounding");
                      quality_probe["cortext_ltm_temporal_correctness"]
                          = ScoreValue (*quality, "cortext_ltm",
                                        "temporal_correctness");
                      quality_probe["cortext_ltm_media_usefulness"]
                          = ScoreValue (*quality, "cortext_ltm",
                                        "media_usefulness");
                      quality_probe["rag_relevance"]
                          = ScoreValue (*quality, "rag", "relevance");
                      quality_probe["rag_sufficiency"]
                          = ScoreValue (*quality, "rag", "sufficiency");
                      quality_probe["rag_noise"]
                          = ScoreValue (*quality, "rag", "noise");
                      quality_probe["rag_source_grounding"] = ScoreValue (
                          *quality, "rag", "source_grounding");
                      quality_probe["rag_temporal_correctness"] = ScoreValue (
                          *quality, "rag", "temporal_correctness");
                      quality_probe["rag_media_usefulness"] = ScoreValue (
                          *quality, "rag", "media_usefulness");
                      quality_probe["lexical_rag_relevance"] = ScoreValue (
                          *quality, "lexical_rag", "relevance");
                      quality_probe["lexical_rag_sufficiency"] = ScoreValue (
                          *quality, "lexical_rag", "sufficiency");
                      quality_probe["lexical_rag_noise"]
                          = ScoreValue (*quality, "lexical_rag", "noise");
                      quality_probe["lexical_rag_source_grounding"]
                          = ScoreValue (*quality, "lexical_rag",
                                        "source_grounding");
                      quality_probe["lexical_rag_temporal_correctness"]
                          = ScoreValue (*quality, "lexical_rag",
                                        "temporal_correctness");
                      quality_probe["lexical_rag_media_usefulness"]
                          = ScoreValue (*quality, "lexical_rag",
                                        "media_usefulness");
                      quality_probe["normal_rag_relevance"]
                          = ScoreValue (*quality, "normal_rag", "relevance");
                      quality_probe["normal_rag_sufficiency"]
                          = ScoreValue (*quality, "normal_rag", "sufficiency");
                      quality_probe["normal_rag_noise"]
                          = ScoreValue (*quality, "normal_rag", "noise");
                      quality_probe["normal_rag_source_grounding"]
                          = ScoreValue (*quality, "normal_rag",
                                        "source_grounding");
                      quality_probe["normal_rag_temporal_correctness"]
                          = ScoreValue (*quality, "normal_rag",
                                        "temporal_correctness");
                      quality_probe["normal_rag_media_usefulness"]
                          = ScoreValue (*quality, "normal_rag",
                                        "media_usefulness");
                      quality_probe["full_history_relevance"] = ScoreValue (
                          *quality, "full_history", "relevance");
                      quality_probe["full_history_sufficiency"] = ScoreValue (
                          *quality, "full_history", "sufficiency");
                      quality_probe["full_history_noise"]
                          = ScoreValue (*quality, "full_history", "noise");
                      quality_probe["full_history_source_grounding"]
                          = ScoreValue (*quality, "full_history",
                                        "source_grounding");
                      quality_probe["full_history_temporal_correctness"]
                          = ScoreValue (*quality, "full_history",
                                        "temporal_correctness");
                      quality_probe["full_history_media_usefulness"]
                          = ScoreValue (*quality, "full_history",
                                        "media_usefulness");
                      quality_probe["winner"]
                          = quality->value ("winner", "unknown");
                      probe["quality"] = std::move (quality_probe);
                    }
                  else
                    {
                      probe["quality_error"] = "judge_failed";
                    }
                }
              probes.push_back (std::move (probe));
            }

          ++durable_processed;
	          rag_docs.push_back ({ i, msg.timestamp, source, message_text,
	                                Tokens (message_text) });
          if (!cfg.daily_consolidation && cfg.consolidate_every > 0
              && durable_processed % cfg.consolidate_every == 0)
            {
              engine->Consolidate (
                  cfg.deep_consolidation ? cortext::ConsolidationMode::Both
                                         : cortext::ConsolidationMode::Shallow);
              ++consolidation_runs;
            }
        }

      engine->Consolidate (cfg.deep_consolidation
                               ? cortext::ConsolidationMode::Both
                               : cortext::ConsolidationMode::Shallow);
      ++consolidation_runs;
      if (cfg.daily_consolidation && durable_processed > 0)
        ++daily_consolidation_runs;
      engine->Flush ();
      std::unique_ptr<cortext::SQLiteStore> audit_store;
      try
        {
          audit_store = cortext::SQLiteStore::Create (cfg.db_path.string ());
        }
      catch (...)
        {
          audit_store.reset ();
        }
      auto run_ended = std::chrono::steady_clock::now ();

      if (cfg.fact_prompt_bakeoff && fact_prompt_store)
        {
          int judged_posthoc = 0;
          for (const auto &pending : pending_fact_prompts)
            {
              if (pending.probe_json_index >= probes.size ())
                continue;
              FactPromptProbe fact_probe = BuildFactPromptProbe (
                  *fact_prompt_store, pending.timestamp, cfg.fact_prompt_k);
              const std::vector<long long> fact_replace_ids
                  = TakeUnique (fact_probe.evidence_memory_ids,
                                static_cast<std::size_t> (
                                    cfg.max_injected_memories));
              const std::vector<long long> fact_union_ids
                  = UnionThenExistingRank (
                      fact_replace_ids, pending.current_memory_ids,
                      static_cast<std::size_t> (cfg.max_injected_memories));
              const auto fact_replace_memories
                  = HydrateMemories (*engine, fact_replace_ids);
              const auto fact_union_memories
                  = HydrateMemories (*engine, fact_union_ids);
              const std::string fact_replace_context
                  = BuildInjectedSystemPrompt (fact_replace_memories);
              const std::string fact_union_context
                  = BuildInjectedSystemPrompt (fact_union_memories);

              nlohmann::json fact_json;
              fact_json["fact_prompt_k"] = cfg.fact_prompt_k;
              fact_json["fact_ids"] = MemoryIdsJson (fact_probe.fact_ids);
              fact_json["evidence_memory_ids"]
                  = MemoryIdsJson (fact_probe.evidence_memory_ids);
              fact_json["current_memory_ids"]
                  = MemoryIdsJson (pending.current_memory_ids);
              fact_json["fact_replace_memory_ids"]
                  = MemoryIdsJson (fact_replace_ids);
              fact_json["fact_union_memory_ids"] = MemoryIdsJson (fact_union_ids);
              fact_json["top_fact_distance"] = fact_probe.top_fact_distance;
              fact_json["source_fact_cache_rows"]
                  = fact_probe.source_fact_cache_rows;
              fact_json["fact_only_rows"] = fact_probe.fact_only_rows;
              fact_json["query_rows"] = fact_probe.query_rows;
              fact_json["query_embedding_bytes"]
                  = fact_probe.query_embedding_bytes;
              fact_json["evidence_memory_count"]
                  = fact_probe.evidence_memory_ids.size ();
              fact_json["fact_replace_items"] = fact_replace_memories.size ();
              fact_json["fact_union_items"] = fact_union_memories.size ();
              fact_json["fact_replace_context_tokens"]
                  = EstimateTokens (fact_replace_context.size ());
              fact_json["fact_union_context_tokens"]
                  = EstimateTokens (fact_union_context.size ());
              fact_json["fact_replace_overlap_with_current"]
                  = IntersectionCount (fact_replace_ids,
                                       pending.current_memory_ids);
              fact_json["fact_union_overlap_with_current"]
                  = IntersectionCount (fact_union_ids,
                                       pending.current_memory_ids);
              fact_json["fact_replace_query_overlap"]
                  = Overlap (pending.query_tokens, fact_replace_context);
              fact_json["fact_union_query_overlap"]
                  = Overlap (pending.query_tokens, fact_union_context);

              const bool should_judge_posthoc
                  = cfg.judge_enabled
                    && !fact_probe.fact_ids.empty ()
                    && (cfg.judge_limit < 0 || judged_posthoc < cfg.judge_limit);
              if (should_judge_posthoc)
                {
                  auto fact_quality = JudgeFactPromptContexts (
                      cfg, pending.message_index, pending.query,
                      pending.current_context, fact_replace_context,
                      fact_union_context, pending.normal_rag_context,
                      pending.full_history_context);
                  if (fact_quality)
                    {
                      ++judged_posthoc;
                      AddQuality (posthoc_fact_replace_quality, *fact_quality,
                                  "fact_replace");
                      AddQuality (posthoc_fact_union_quality, *fact_quality,
                                  "fact_union");
                      nlohmann::json quality_probe;
                      quality_probe["current_relevance"] = ScoreValue (
                          *fact_quality, "current", "relevance");
                      quality_probe["current_sufficiency"] = ScoreValue (
                          *fact_quality, "current", "sufficiency");
                      quality_probe["current_noise"] = ScoreValue (
                          *fact_quality, "current", "noise");
                      quality_probe["fact_replace_relevance"] = ScoreValue (
                          *fact_quality, "fact_replace", "relevance");
                      quality_probe["fact_replace_sufficiency"] = ScoreValue (
                          *fact_quality, "fact_replace", "sufficiency");
                      quality_probe["fact_replace_noise"] = ScoreValue (
                          *fact_quality, "fact_replace", "noise");
                      quality_probe["fact_union_relevance"] = ScoreValue (
                          *fact_quality, "fact_union", "relevance");
                      quality_probe["fact_union_sufficiency"] = ScoreValue (
                          *fact_quality, "fact_union", "sufficiency");
                      quality_probe["fact_union_noise"] = ScoreValue (
                          *fact_quality, "fact_union", "noise");
                      quality_probe["normal_rag_relevance"] = ScoreValue (
                          *fact_quality, "normal_rag", "relevance");
                      quality_probe["normal_rag_sufficiency"] = ScoreValue (
                          *fact_quality, "normal_rag", "sufficiency");
                      quality_probe["normal_rag_noise"] = ScoreValue (
                          *fact_quality, "normal_rag", "noise");
                      quality_probe["full_history_relevance"] = ScoreValue (
                          *fact_quality, "full_history", "relevance");
                      quality_probe["full_history_sufficiency"] = ScoreValue (
                          *fact_quality, "full_history", "sufficiency");
                      quality_probe["full_history_noise"] = ScoreValue (
                          *fact_quality, "full_history", "noise");
                      quality_probe["winner"]
                          = fact_quality->value ("winner", "unknown");
                      fact_json["quality"] = std::move (quality_probe);
                    }
                  else
                    {
                      fact_json["quality_error"] = "judge_failed";
                    }
                }
              probes[pending.probe_json_index]["posthoc_fact_prompt_bakeoff"]
                  = std::move (fact_json);
            }
        }

      if (cfg.source_tagged_bakeoff)
        {
          int judged_source = 0;
          for (const auto &source_probe : source_tagged_probes)
            {
              if (source_probe.probe_json_index >= probes.size ())
                continue;
              const bool should_judge_source
                  = cfg.judge_enabled
                    && (cfg.judge_limit < 0
                        || judged_source < cfg.judge_limit);
              if (!should_judge_source)
                continue;
              auto quality = JudgeSourceTaggedContexts (
                  cfg, source_probe.message_index, source_probe.query,
                  source_probe.current_context,
                  source_probe.wm_history_context,
                  source_probe.stm_recent_context,
                  source_probe.ltm_lexical_context,
                  source_probe.stm_ltm_union_context,
                  source_probe.normal_rag_context,
                  source_probe.full_history_context);
              if (quality)
                {
                  ++judged_source;
                  AddQuality (source_current_quality, *quality, "current");
                  AddQuality (source_wm_history_quality, *quality,
                              "wm_history");
                  AddQuality (source_stm_recent_quality, *quality,
                              "stm_recent");
                  AddQuality (source_ltm_lexical_quality, *quality,
                              "ltm_lexical");
                  AddQuality (source_stm_ltm_union_quality, *quality,
                              "stm_ltm_union");
                  AddQuality (source_normal_rag_quality, *quality,
                              "normal_rag");
                  AddQuality (source_full_history_quality, *quality,
                              "full_history");
                  nlohmann::json quality_probe;
                  const std::vector<std::string> systems = {
                    "current", "wm_history", "stm_recent", "ltm_lexical",
                    "stm_ltm_union", "normal_rag", "full_history"
                  };
                  for (const auto &system : systems)
                    {
                      quality_probe[system + "_relevance"]
                          = ScoreValue (*quality, system, "relevance");
                      quality_probe[system + "_sufficiency"]
                          = ScoreValue (*quality, system, "sufficiency");
                      quality_probe[system + "_noise"]
                          = ScoreValue (*quality, system, "noise");
                    }
                  quality_probe["winner"] = quality->value ("winner",
                                                            "unknown");
                  probes[source_probe.probe_json_index]["source_tagged_bakeoff"]
                        ["quality"]
                      = std::move (quality_probe);
                }
              else
                {
                  probes[source_probe.probe_json_index]["source_tagged_bakeoff"]
                        ["quality_error"]
                      = "judge_failed";
                }
            }
        }

      nlohmann::json out;
      out["input_dir"] = cfg.input_dir.string ();
      out["db_path"] = cfg.db_path.string ();
      out["parsed_transcript_messages"] = parsed_messages;
      out["skip_messages"] = cfg.skip_messages;
      out["processed_messages"] = durable_processed;
      out["rolling_eval_enabled"] = cfg.rolling_eval;
      out["sampling_policy"]
          = "contiguous early/middle/recent windows, with media-adjacent "
            "messages added when needed";
      out["stratified_sample_messages"] = cfg.stratified_sample_messages;
      out["media_adjacent_min"] = cfg.media_adjacent_min;
      out["rolling_probe_target"] = cfg.rolling_probe_target;
      out["sampled_messages"] = static_cast<int> (messages.size ());
      out["selected_probe_count"] = probe_indices.size ();
      out["parsed_media_adjacent_messages"] = parsed_media_adjacent_messages;
      out["sampled_media_adjacent_messages"] = sampled_media_adjacent_messages;
      out["media_coverage"]["indexed_media_files"] = media_index.file_count;
      out["media_coverage"]["indexed_media_kinds"] = media_index.kind_counts;
      out["media_coverage"]["skipped_reasons"]
          = media_index.skipped_reason_counts;
      if (!messages.empty ())
        {
          out["sample_window"]["first_timestamp"] = messages.front ().timestamp;
          out["sample_window"]["last_timestamp"] = messages.back ().timestamp;
        }
      out["warmup_messages"] = cfg.warmup_messages;
      out["probe_stride"] = cfg.probe_stride;
      out["min_probe_query_tokens"] = cfg.min_probe_query_tokens;
      out["rag_top_k"] = cfg.rag_top_k;
      out["max_injected_memories"] = cfg.max_injected_memories;
      out["active_history_token_budget"] = cfg.active_history_token_budget;
      out["knobs"] = {
        { "focus", cfg.focus },
        { "sensitivity", cfg.sensitivity },
        { "stability", cfg.stability },
      };
      out["consolidate_every"] = cfg.consolidate_every;
      out["daily_consolidation_enabled"] = cfg.daily_consolidation;
      out["consolidation_policy"] =
          cfg.daily_consolidation
              ? "local_calendar_day_boundary_plus_final_day"
              : "message_count_interval_plus_final";
      out["consolidation_runs"] = consolidation_runs;
      out["daily_consolidation_runs"] = daily_consolidation_runs;
      out["judge_enabled"] = cfg.judge_enabled;
      out["judge_model"] = cfg.judge_model;
      out["judge_provider"] = cfg.judge_enabled
                                  ? "local_nemotron_vllm_mlx"
                                  : "disabled";
      out["judge_base_url"] =
          cfg.judge_enabled ? nlohmann::json (LocalNemotronJudgeBaseUrl ())
                            : nlohmann::json (nullptr);
      out["remote_provider_allowed"] = false;
      out["judge_limit"] = cfg.judge_limit;
      out["fact_prompt_bakeoff_enabled"] = cfg.fact_prompt_bakeoff;
      out["fact_prompt_k"] = cfg.fact_prompt_k;
      out["source_tagged_bakeoff_enabled"] = cfg.source_tagged_bakeoff;
      out["prompt_policy_bakeoff_enabled"] = cfg.prompt_policy_bakeoff;
      out["compact_policy_bakeoff_enabled"] = cfg.compact_policy_bakeoff;
      out["stm_graph_bakeoff_enabled"] = cfg.stm_graph_bakeoff;
      out["graph_expanded_rag_bakeoff_enabled"]
          = cfg.graph_expanded_rag_bakeoff;
      out["source_stm_recent_k"] = cfg.source_stm_recent_k;
      out["source_ltm_lexical_k"] = cfg.source_ltm_lexical_k;
      out["normal_rag_retrieval"] = "raw_chat_vector";
      out["normal_rag_vector_query_encoder"]
          = rag_encoder_selection.backend_name;
      out["normal_rag_vector_query_encoder_path"]
          = rag_encoder_selection.resolved_path.string ();
      out["normal_rag_vector_candidate_k"]
          = std::max (1, cortext::core::MaxResults (cortext_cfg.focus));
      out["normal_rag_prompt_item_limit"] = std::max (
          1, cortext::core::RetrievalGraphExpandedRagMaxItems (
                 cortext_cfg.focus, cortext_cfg.stability));
      out["deep_consolidation"] = cfg.deep_consolidation;
      out["label_bank_enabled"] = cfg.use_label_bank;
      out["label_bank_path"] = cfg.use_label_bank ? cfg.label_bank_path.string ()
                                                  : "";
      if (audit_store)
        {
          out["stm_ltm_relabel_audit"] = BuildStmLtmAuditSummary (*audit_store);
        }
      out["wall_ms"]
          = std::chrono::duration_cast<std::chrono::milliseconds> (run_ended
                                                                   - run_started)
                .count ();
      out["peak_rss_mb"] = PeakResidentSetMb ();
      out["cortext"] = AggregateJson (cortext_agg);
      out["cortext_ltm_only"] = AggregateJson (cortext_ltm_agg);
      out["simple_rag"] = AggregateJson (rag_agg);
      out["lexical_rag_only"] = AggregateJson (lexical_rag_agg);
      out["normal_rag_with_history"] = AggregateJson (normal_rag_agg);
      out["normal_rag_compaction"] = CompactionJson (normal_rag_compaction);
	      if (cfg.graph_expanded_rag_bakeoff)
	        out["graph_expanded_rag_cortext_policy"]
	            = AggregateJson (graph_expanded_rag_agg);
      out["full_history"] = AggregateJson (full_history_agg);
      out["quality"]["cortext"] = QualityJson (cortext_quality);
      out["quality"]["cortext_ltm_only"] = QualityJson (cortext_ltm_quality);
      out["quality"]["simple_rag"] = QualityJson (rag_quality);
      out["quality"]["lexical_rag_only"] = QualityJson (lexical_rag_quality);
      out["quality"]["normal_rag_with_history"]
          = QualityJson (normal_rag_quality);
      out["quality"]["full_history"] = QualityJson (full_history_quality);
      if (cfg.fact_prompt_bakeoff)
        {
          out["quality"]["fact_replace"] = QualityJson (fact_replace_quality);
          out["quality"]["fact_union"] = QualityJson (fact_union_quality);
          out["quality"]["posthoc_fact_replace"]
              = QualityJson (posthoc_fact_replace_quality);
          out["quality"]["posthoc_fact_union"]
              = QualityJson (posthoc_fact_union_quality);
          if (fact_prompt_store)
            {
              nlohmann::json db_counts;
              db_counts["memories"] = CountRows (*fact_prompt_store, "memories");
              db_counts["signals"] = CountRows (*fact_prompt_store, "signals");
              db_counts["embeddings"] = CountRows (*fact_prompt_store,
                                                   "embeddings");
              db_counts["fact_assertions"] = CountRows (
                  *fact_prompt_store, "fact_assertions");
              db_counts["fact_cache"] = CountRows (*fact_prompt_store,
                                                   "fact_cache");
              db_counts["fact_evidence"] = CountRows (*fact_prompt_store,
                                                       "fact_evidence");
              out["db_counts"] = std::move (db_counts);
            }
        }
      if (cfg.source_tagged_bakeoff)
        {
          out["quality"]["source_tagged"]["current"]
              = QualityJson (source_current_quality);
          out["quality"]["source_tagged"]["wm_history"]
              = QualityJson (source_wm_history_quality);
          out["quality"]["source_tagged"]["stm_recent"]
              = QualityJson (source_stm_recent_quality);
          out["quality"]["source_tagged"]["ltm_lexical"]
              = QualityJson (source_ltm_lexical_quality);
          out["quality"]["source_tagged"]["stm_ltm_union"]
              = QualityJson (source_stm_ltm_union_quality);
          out["quality"]["source_tagged"]["normal_rag"]
              = QualityJson (source_normal_rag_quality);
          out["quality"]["source_tagged"]["full_history"]
              = QualityJson (source_full_history_quality);
        }
      if (cfg.prompt_policy_bakeoff)
        {
          out["quality"]["prompt_policy"]["current"]
              = QualityJson (policy_current_quality);
          out["quality"]["prompt_policy"]["stm_recent"]
              = QualityJson (policy_stm_recent_quality);
          out["quality"]["prompt_policy"]["current_stm"]
              = QualityJson (policy_current_stm_quality);
          out["quality"]["prompt_policy"]["current_stm_ltm"]
              = QualityJson (policy_current_stm_ltm_quality);
          out["quality"]["prompt_policy"]["normal_rag"]
              = QualityJson (policy_normal_rag_quality);
          out["quality"]["prompt_policy"]["full_history"]
              = QualityJson (policy_full_history_quality);
        }
      if (cfg.compact_policy_bakeoff)
        {
          out["quality"]["compact_policy"]["cortext_ltm"]
              = QualityJson (compact_cortext_ltm_quality);
          out["quality"]["compact_policy"]["stm_recent"]
              = QualityJson (compact_stm_recent_quality);
          out["quality"]["compact_policy"]["ltm_lexical"]
              = QualityJson (compact_ltm_lexical_quality);
          out["quality"]["compact_policy"]["stm_ltm_union"]
              = QualityJson (compact_stm_ltm_union_quality);
          out["quality"]["compact_policy"]["normal_rag"]
              = QualityJson (compact_normal_rag_quality);
          out["quality"]["compact_policy"]["full_history"]
              = QualityJson (compact_full_history_quality);
        }
      if (cfg.stm_graph_bakeoff)
        {
          out["quality"]["stm_graph_bakeoff"]["raw_stm_graph"]
              = QualityJson (stm_graph_raw_quality);
          out["quality"]["stm_graph_bakeoff"]["relabel_prune_ltm"]
              = QualityJson (stm_graph_relabel_quality);
          out["quality"]["stm_graph_bakeoff"]["normal_rag"]
              = QualityJson (stm_graph_normal_rag_quality);
          out["quality"]["stm_graph_bakeoff"]["full_history"]
              = QualityJson (stm_graph_full_history_quality);
        }
      if (cfg.graph_expanded_rag_bakeoff)
        {
          out["quality"]["graph_expanded_rag_bakeoff"]["normal_rag"]
              = QualityJson (graph_expanded_normal_rag_quality);
          out["quality"]["graph_expanded_rag_bakeoff"]["cortext_ltm"]
              = QualityJson (graph_expanded_cortext_ltm_quality);
          out["quality"]["graph_expanded_rag_bakeoff"]["graph_expanded_rag"]
              = QualityJson (graph_expanded_rag_quality);
          out["quality"]["graph_expanded_rag_bakeoff"]["full_history"]
              = QualityJson (graph_expanded_full_history_quality);
        }
      if (cortext_agg.probes > 0)
        {
          const double cortext_tokens
              = static_cast<double> (cortext_agg.prompt_tokens)
                / cortext_agg.probes;
          const double rag_tokens
              = static_cast<double> (rag_agg.prompt_tokens) / rag_agg.probes;
          const double normal_rag_tokens
              = static_cast<double> (normal_rag_agg.prompt_tokens)
                / normal_rag_agg.probes;
          const double full_tokens
              = static_cast<double> (full_history_agg.prompt_tokens)
                / full_history_agg.probes;
          out["token_ratio_cortext_vs_simple_rag"]
              = rag_tokens > 0.0 ? cortext_tokens / rag_tokens : 0.0;
          out["token_ratio_cortext_vs_normal_rag_with_history"]
              = normal_rag_tokens > 0.0 ? cortext_tokens / normal_rag_tokens
                                        : 0.0;
          out["token_ratio_cortext_vs_full_history"]
              = full_tokens > 0.0 ? cortext_tokens / full_tokens : 0.0;
          const double cortext_ltm_tokens
              = cortext_ltm_agg.probes > 0
                    ? static_cast<double> (cortext_ltm_agg.prompt_tokens)
                          / cortext_ltm_agg.probes
                    : 0.0;
          const double lexical_rag_tokens
              = lexical_rag_agg.probes > 0
                    ? static_cast<double> (lexical_rag_agg.prompt_tokens)
                          / lexical_rag_agg.probes
                    : 0.0;
          out["token_ratio_cortext_ltm_vs_lexical_rag"]
              = lexical_rag_tokens > 0.0 ? cortext_ltm_tokens
                                                / lexical_rag_tokens
                                          : 0.0;
          if (cfg.graph_expanded_rag_bakeoff
              && graph_expanded_rag_agg.probes > 0)
            {
              const double graph_rag_tokens
                  = static_cast<double> (
                        graph_expanded_rag_agg.prompt_tokens)
                    / graph_expanded_rag_agg.probes;
	              out["token_ratio_graph_expanded_rag_cortext_policy_vs_normal_rag_with_history"]
	                  = normal_rag_tokens > 0.0
	                        ? graph_rag_tokens / normal_rag_tokens
	                        : 0.0;
            }
        }
      out["failure_taxonomy"] = BuildFailureTaxonomy (
          probes, out.value ("stm_ltm_relabel_audit", nlohmann::json::object ()));
      out["probes"] = probes;
      out["method_note"]
          = "Cortext side uses only public Cortext processing and consolidation. "
            "Cortext prompt cost mirrors the chat demo: XML memory snapshot "
            "system prompt plus Cortext working-memory messages plus the "
            "current user message. Simple RAG is an independent lexical top-k "
            "baseline over prior chat documents, formatted as a notes system "
            "prompt plus the current user message, without working-memory chat "
         "history. normal_rag_with_history is the production-style RAG "
         "baseline requested here: rolling active chat history until the "
            "configured compaction budget plus the same independent raw-chat "
            "vector retrieval notes prompt. "
            "cortext_ltm_only and lexical_rag_only isolate long-term memory "
            "retrieval quality without working-memory or rolling-history "
            "context. "
            "When fact_prompt_bakeoff is enabled, fact_replace and fact_union "
            "are diagnostic prompt-selection variants only; they build a "
            "temporary fact-only vec table from the existing database and do "
            "not change production retrieval or consolidation. "
            "When source_tagged_bakeoff is enabled, wm_history, stm_recent, "
            "ltm_lexical, and stm_ltm_union are benchmark-only source-isolation "
            "prompt packets over prior chat docs; they do not change Cortext "
            "ranking, storage, retrieval, or consolidation. "
            "When prompt_policy_bakeoff is enabled, current_stm and "
            "current_stm_ltm are benchmark-only prompt-composition packets "
            "that append bounded recent-chat continuity and lexical sidecar "
            "context to the current Cortext context; they do not change "
            "Cortext ranking, storage, retrieval, or consolidation. "
            "When compact_policy_bakeoff is enabled, compact-policy packets "
            "are benchmark-only replacement packets comparing Cortext LTM, "
            "STM recent continuity, lexical LTM, their bounded union, normal "
            "RAG, and full history; they do not change Cortext ranking, "
            "storage, retrieval, or consolidation. "
            "When stm_graph_bakeoff is enabled, raw_stm_graph and "
            "relabel_prune_ltm are benchmark-only chat packets that combine "
            "the same bounded Cortext working-memory context with source "
            "packets built from the STM/LTM audit table: raw_stm_graph ranks "
            "consolidation cues by provisional STM current-label/query overlap "
            "plus preserved source/query overlap, while relabel_prune_ltm uses "
            "the production retrieval debug path to hydrate original source "
            "memories behind source-backed durable LTM candidates after "
            "relabel/prune, falling back to durable refined-label/query and "
            "source/query overlap when no durable candidate is selected. "
            "When graph_expanded_rag_bakeoff is enabled, graph_expanded_rag "
            "uses the Cortext chat policy: bounded Cortext working memory plus "
            "a benchmark-only memory packet that starts from the same "
            "independent source-memory top-k as normal RAG, maps those "
	            "hits back to Cortext source memories, expands through "
	            "derived_from, has_label, label relation/co-occurrence, fact "
	            "evidence, and temporal-neighbor links, then ranks source memories "
	            "with only F/S/T-derived weights. It does not include the rolling "
	            "normal chat history. It keeps compact label/fact text only when "
	            "that compact text overlaps the live query. "
            "Full history is reported as the separate chat-demo baseline. Token "
            "counts use the chat demo's chars/4 estimator.";
      out["privacy_note"]
          = "No message text is written to this summary; per-probe rows contain "
            "only counters and scores.";

      std::ofstream summary (cfg.output_path);
      summary << out.dump (2) << "\n";
      std::cout << out.dump (2) << "\n";
      return 0;
    }
  catch (const std::exception &e)
    {
      std::cerr << "chat_replay_conversation_memory_bakeoff failed: " << e.what ()
                << "\n";
      return 1;
    }
}
