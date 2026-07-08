// cortext_memory_eval_runner: benchmark adapter for normalized memory evals.
//
// The runner uses only the public Cortext C++ API. Histories are ingested with
// durable retention, then benchmark questions are issued with ephemeral
// retention so the query path does not pollute long-term memory.

#include <cortext/cortext.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

struct Options
{
  std::filesystem::path episodes_path;
  std::filesystem::path answer_key_path;
  std::filesystem::path out_path;
  std::string db_path = ":memory:";
  std::string benchmark = "unknown";
  double focus = 0.5;
  double sensitivity = 0.5;
  double stability = 0.5;
  std::size_t top_k = 8;
  int max_episodes = 0;
  int max_queries = 0;
  bool reuse_engine = false;
};

struct Signal
{
  std::string modality = "text";
  std::string source_id;
  std::string text;
  std::string mimetype;
  std::string data_base64;
  std::vector<float> samples;
  int width = 0;
  int height = 0;
  int channels = 0;
  std::uint64_t timestamp_ms = 0;
};

struct Episode
{
  std::string id;
  std::string benchmark;
  std::vector<Signal> signals;
};

struct Query
{
  std::string benchmark;
  std::string conversation_id;
  std::string query_id;
  std::string question;
  std::string question_type;
  std::vector<std::string> answers;
  std::vector<std::string> evidence;
  bool requires_abstention = false;
};

struct Counts
{
  int n = 0;
  int answerable = 0;
  int answer_hit = 0;
  int evidence_queries = 0;
  int evidence_hit = 0;
  int abstention_required = 0;
  int abstention_empty = 0;
};

void PrintUsage (std::ostream &out)
{
  out << "Usage: cortext_memory_eval_runner --episodes PATH "
      << "--answer-key PATH [options]\n\n"
      << "Options:\n"
      << "  --out PATH          Write JSONL query rows and summary JSON next to PATH\n"
      << "  --db PATH           SQLite DB path, default :memory:\n"
      << "  --benchmark NAME    Override benchmark label\n"
      << "  --focus X           F knob in [0,1], default 0.5\n"
      << "  --sensitivity X     S knob in [0,1], default 0.5\n"
      << "  --stability X       T knob in [0,1], default 0.5\n"
      << "  --top-k N           Retrieval packet depth to score, default 8\n"
      << "  --max-episodes N    Limit episodes, 0 means all\n"
      << "  --max-queries N     Limit queries, 0 means all\n"
      << "  --reuse-engine      Reuse one engine/DB across episodes\n"
      << "  -h, --help          Show this help\n";
}

std::optional<std::string> TakeArg (const std::string &arg,
                                    const std::string &prefix)
{
  if (arg.rfind (prefix, 0) == 0)
    {
      return arg.substr (prefix.size ());
    }
  return std::nullopt;
}

Options ParseArgs (int argc, char **argv)
{
  Options opts;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if (arg == "-h" || arg == "--help")
        {
          PrintUsage (std::cout);
          std::exit (0);
        }
      if (auto v = TakeArg (arg, "--episodes="))
        {
          opts.episodes_path = *v;
        }
      else if (auto v = TakeArg (arg, "--answer-key="))
        {
          opts.answer_key_path = *v;
        }
      else if (auto v = TakeArg (arg, "--out="))
        {
          opts.out_path = *v;
        }
      else if (auto v = TakeArg (arg, "--db="))
        {
          opts.db_path = *v;
        }
      else if (auto v = TakeArg (arg, "--benchmark="))
        {
          opts.benchmark = *v;
        }
      else if (auto v = TakeArg (arg, "--focus="))
        {
          opts.focus = std::stod (*v);
        }
      else if (auto v = TakeArg (arg, "--sensitivity="))
        {
          opts.sensitivity = std::stod (*v);
        }
      else if (auto v = TakeArg (arg, "--stability="))
        {
          opts.stability = std::stod (*v);
        }
      else if (auto v = TakeArg (arg, "--top-k="))
        {
          opts.top_k = static_cast<std::size_t> (std::stoul (*v));
        }
      else if (auto v = TakeArg (arg, "--max-episodes="))
        {
          opts.max_episodes = std::stoi (*v);
        }
      else if (auto v = TakeArg (arg, "--max-queries="))
        {
          opts.max_queries = std::stoi (*v);
        }
      else if (arg == "--reuse-engine")
        {
          opts.reuse_engine = true;
        }
      else
        {
          throw std::runtime_error ("Unknown argument: " + arg);
        }
    }
  if (opts.episodes_path.empty ())
    {
      throw std::runtime_error ("Missing --episodes");
    }
  if (opts.answer_key_path.empty ())
    {
      throw std::runtime_error ("Missing --answer-key");
    }
  if (opts.out_path.empty ())
    {
      opts.out_path = opts.episodes_path;
      opts.out_path += ".results.jsonl";
    }
  return opts;
}

std::string JsonString (const nlohmann::json &j, const std::string &key,
                        const std::string &fallback = "")
{
  auto it = j.find (key);
  if (it == j.end () || it->is_null ())
    {
      return fallback;
    }
  if (it->is_string ())
    {
      return it->get<std::string> ();
    }
  if (it->is_number_integer ())
    {
      return std::to_string (it->get<long long> ());
    }
  if (it->is_number_unsigned ())
    {
      return std::to_string (it->get<unsigned long long> ());
    }
  if (it->is_number_float ())
    {
      std::ostringstream out;
      out << it->get<double> ();
      return out.str ();
    }
  if (it->is_boolean ())
    {
      return it->get<bool> () ? "true" : "false";
    }
  return it->dump ();
}

std::vector<std::string> JsonStringList (const nlohmann::json &j,
                                         const std::string &key)
{
  std::vector<std::string> out;
  auto it = j.find (key);
  if (it == j.end () || it->is_null ())
    {
      return out;
    }
  if (it->is_array ())
    {
      for (const auto &item : *it)
        {
          if (item.is_array ())
            {
              for (const auto &nested : item)
                {
                  const std::string text = JsonString (
                      nlohmann::json{{"v", nested}}, "v");
                  if (!text.empty ())
                    {
                      out.push_back (text);
                    }
                }
            }
          else
            {
              const std::string text = JsonString (
                  nlohmann::json{{"v", item}}, "v");
              if (!text.empty ())
                {
                  out.push_back (text);
                }
            }
        }
      return out;
    }
  const std::string text = JsonString (j, key);
  if (!text.empty ())
    {
      out.push_back (text);
    }
  return out;
}

std::uint64_t JsonU64 (const nlohmann::json &j, const std::string &key)
{
  auto it = j.find (key);
  if (it == j.end () || it->is_null ())
    {
      return 0;
    }
  if (it->is_number_unsigned ())
    {
      return it->get<std::uint64_t> ();
    }
  if (it->is_number_integer ())
    {
      const auto value = it->get<long long> ();
      return value > 0 ? static_cast<std::uint64_t> (value) : 0;
    }
  if (it->is_string ())
    {
      try
        {
          return static_cast<std::uint64_t> (std::stoull (it->get<std::string> ()));
        }
      catch (const std::exception &)
        {
          return 0;
        }
    }
  return 0;
}

std::string MemoryText (const cortext::Cortext::Context::Memory &memory)
{
  std::string text;
  for (const auto &blob : memory.content)
    {
      if (!text.empty ())
        {
          text.push_back (' ');
        }
      text.append (blob.begin (), blob.end ());
    }
  for (auto &ch : text)
    {
      if (ch == '\n' || ch == '\r' || ch == '\t')
        {
          ch = ' ';
        }
    }
  return text;
}

std::string Normalize (const std::string &text)
{
  std::string out;
  out.reserve (text.size ());
  bool last_space = true;
  for (unsigned char c : text)
    {
      if (std::isalnum (c))
        {
          out.push_back (static_cast<char> (std::tolower (c)));
          last_space = false;
        }
      else if (!last_space)
        {
          out.push_back (' ');
          last_space = true;
        }
    }
  if (!out.empty () && out.back () == ' ')
    {
      out.pop_back ();
    }
  return out;
}

bool ContainsAnyNormalized (const std::vector<std::string> &needles,
                            const std::string &haystack)
{
  const std::string norm_haystack = Normalize (haystack);
  for (const auto &needle : needles)
    {
      const std::string norm_needle = Normalize (needle);
      if (!norm_needle.empty ()
          && norm_haystack.find (norm_needle) != std::string::npos)
        {
          return true;
        }
    }
  return false;
}

std::string Clip (const std::string &text, std::size_t max_chars)
{
  if (text.size () <= max_chars)
    {
      return text;
    }
  return text.substr (0, max_chars);
}

int Base64Value (char c)
{
  if (c >= 'A' && c <= 'Z')
    {
      return c - 'A';
    }
  if (c >= 'a' && c <= 'z')
    {
      return c - 'a' + 26;
    }
  if (c >= '0' && c <= '9')
    {
      return c - '0' + 52;
    }
  if (c == '+')
    {
      return 62;
    }
  if (c == '/')
    {
      return 63;
    }
  return -1;
}

std::vector<std::uint8_t> DecodeBase64 (const std::string &encoded)
{
  std::vector<std::uint8_t> out;
  int val = 0;
  int bits = -8;
  for (char c : encoded)
    {
      if (c == '=')
        {
          break;
        }
      const int decoded = Base64Value (c);
      if (decoded < 0)
        {
          if (std::isspace (static_cast<unsigned char> (c)))
            {
              continue;
            }
          throw std::runtime_error ("Invalid base64 payload");
        }
      val = (val << 6) + decoded;
      bits += 6;
      if (bits >= 0)
        {
          out.push_back (static_cast<std::uint8_t> ((val >> bits) & 0xFF));
          bits -= 8;
        }
    }
  return out;
}

std::vector<float> DecodeFloat32Base64 (const std::string &encoded)
{
  const auto bytes = DecodeBase64 (encoded);
  if (bytes.size () % sizeof (float) != 0)
    {
      throw std::runtime_error ("audio_f32_base64 size is not a float32 multiple");
    }
  std::vector<float> samples (bytes.size () / sizeof (float));
  if (!bytes.empty ())
    {
      std::memcpy (samples.data (), bytes.data (), bytes.size ());
    }
  return samples;
}

int JsonInt (const nlohmann::json &j, const std::string &key)
{
  auto it = j.find (key);
  if (it == j.end () || it->is_null ())
    {
      return 0;
    }
  if (it->is_number_integer () || it->is_number_unsigned ())
    {
      return it->get<int> ();
    }
  if (it->is_string ())
    {
      try
        {
          return std::stoi (it->get<std::string> ());
        }
      catch (const std::exception &)
        {
          return 0;
        }
    }
  return 0;
}

Signal ParseSignal (const nlohmann::json &j, const std::string &episode_id,
                    std::size_t index)
{
  Signal signal;
  signal.modality = JsonString (j, "modality", "text");
  signal.source_id = JsonString (j, "source_id");
  if (signal.source_id.empty ())
    {
      signal.source_id = "eval/" + episode_id + "/" + std::to_string (index);
    }
  signal.text = JsonString (j, "text");
  if (signal.text.empty ())
    {
      signal.text = JsonString (j, "message");
    }
  if (signal.text.empty ())
    {
      signal.text = JsonString (j, "content");
    }
  signal.mimetype = JsonString (j, "mimetype");
  signal.data_base64 = JsonString (j, "data_base64");
  if (signal.data_base64.empty ())
    {
      signal.data_base64 = JsonString (j, "pixels_base64");
    }
  signal.width = JsonInt (j, "width");
  signal.height = JsonInt (j, "height");
  signal.channels = JsonInt (j, "channels");
  if (auto samples = j.find ("samples");
      samples != j.end () && samples->is_array ())
    {
      for (const auto &sample : *samples)
        {
          if (sample.is_number ())
            {
              signal.samples.push_back (sample.get<float> ());
            }
        }
    }
  const std::string audio_f32 = JsonString (j, "audio_f32_base64");
  if (!audio_f32.empty ())
    {
      signal.samples = DecodeFloat32Base64 (audio_f32);
    }
  signal.timestamp_ms = JsonU64 (j, "timestamp_ms");
  return signal;
}

Episode ParseEpisode (const nlohmann::json &j)
{
  Episode episode;
  if (j.is_array () && j.size () >= 2)
    {
      episode.id = JsonString (nlohmann::json{{"id", j.at (0)}}, "id");
      const auto &payload = j.at (1);
      const auto content = payload.find ("content");
      if (content != payload.end () && content->is_array ())
        {
          std::size_t index = 0;
          for (const auto &turn : *content)
            {
              nlohmann::json signal = {
                  {"modality", "text"},
                  {"source_id",
                   "eval/" + episode.id + "/" + JsonString (turn, "agent",
                                                             "agent")},
                  {"text", JsonString (turn, "message")},
              };
              episode.signals.push_back (
                  ParseSignal (signal, episode.id, index++));
            }
        }
      return episode;
    }

  episode.id = JsonString (j, "episode_id");
  if (episode.id.empty ())
    {
      episode.id = JsonString (j, "conversation_id");
    }
  episode.benchmark = JsonString (j, "benchmark");
  auto signals = j.find ("signals");
  if (signals != j.end () && signals->is_array ())
    {
      std::size_t index = 0;
      for (const auto &signal : *signals)
        {
          episode.signals.push_back (
              ParseSignal (signal, episode.id, index++));
        }
    }
  else if (auto content = j.find ("content");
           content != j.end () && content->is_array ())
    {
      std::size_t index = 0;
      for (const auto &turn : *content)
        {
          episode.signals.push_back (
              ParseSignal (turn, episode.id, index++));
        }
    }
  return episode;
}

std::vector<Episode> LoadEpisodes (const std::filesystem::path &path,
                                   int max_episodes)
{
  std::ifstream in (path);
  if (!in)
    {
      throw std::runtime_error ("Failed to open episodes: " + path.string ());
    }
  std::vector<Episode> episodes;
  std::string line;
  while (std::getline (in, line))
    {
      if (line.empty ())
        {
          continue;
        }
      const auto parsed = nlohmann::json::parse (line);
      Episode episode = ParseEpisode (parsed);
      if (!episode.id.empty ())
        {
          episodes.push_back (std::move (episode));
        }
      if (max_episodes > 0
          && static_cast<int> (episodes.size ()) >= max_episodes)
        {
          break;
        }
    }
  return episodes;
}

std::unordered_map<std::string, std::vector<Query>>
LoadQueries (const std::filesystem::path &path, int max_queries)
{
  std::ifstream in (path);
  if (!in)
    {
      throw std::runtime_error ("Failed to open answer key: " + path.string ());
    }
  std::unordered_map<std::string, std::vector<Query>> out;
  std::string line;
  int count = 0;
  while (std::getline (in, line))
    {
      if (line.empty ())
        {
          continue;
        }
      const auto j = nlohmann::json::parse (line);
      Query q;
      q.benchmark = JsonString (j, "benchmark");
      q.conversation_id = JsonString (j, "conversation_id");
      if (q.conversation_id.empty ())
        {
          q.conversation_id = JsonString (j, "episode_id");
        }
      q.query_id = JsonString (j, "query_id", q.conversation_id);
      q.question = JsonString (j, "question");
      q.question_type = JsonString (j, "question_type", "unknown");
      q.answers = JsonStringList (j, "answers");
      q.evidence = JsonStringList (j, "evidence");
      if (q.evidence.empty ())
        {
          q.evidence = JsonStringList (j, "evidence_phrases");
        }
      auto abstain_it = j.find ("requires_abstention");
      if (abstain_it != j.end () && abstain_it->is_boolean ())
        {
          q.requires_abstention = abstain_it->get<bool> ();
        }
      if (!q.conversation_id.empty () && !q.question.empty ())
        {
          out[q.conversation_id].push_back (std::move (q));
          ++count;
        }
      if (max_queries > 0 && count >= max_queries)
        {
          break;
        }
    }
  return out;
}

std::unique_ptr<cortext::Cortext> CreateEngine (const Options &opts,
                                                const std::string &db_path)
{
  cortext::Cortext::Config cfg;
  cfg.focus = opts.focus;
  cfg.sensitivity = opts.sensitivity;
  cfg.stability = opts.stability;
  return cortext::Cortext::Create (cfg, db_path);
}

std::string DbPathForEpisode (const Options &opts, std::size_t episode_index)
{
  if (opts.db_path == ":memory:" || opts.reuse_engine)
    {
      return opts.db_path;
    }
  std::filesystem::path path = opts.db_path;
  const auto stem = path.stem ().string ();
  const auto ext = path.extension ().string ();
  path.replace_filename (stem + "_" + std::to_string (episode_index) + ext);
  return path.string ();
}

void AddCounts (Counts &counts, const Query &query, bool answer_hit,
                bool evidence_hit, bool retrieval_empty)
{
  counts.n += 1;
  if (!query.answers.empty () && !query.requires_abstention)
    {
      counts.answerable += 1;
      if (answer_hit)
        {
          counts.answer_hit += 1;
        }
    }
  if (!query.evidence.empty ())
    {
      counts.evidence_queries += 1;
      if (evidence_hit)
        {
          counts.evidence_hit += 1;
        }
    }
  if (query.requires_abstention)
    {
      counts.abstention_required += 1;
      if (retrieval_empty)
        {
          counts.abstention_empty += 1;
        }
    }
}

nlohmann::json CountsJson (const Counts &counts)
{
  auto ratio = [] (int num, int den) {
    return den == 0 ? 0.0 : static_cast<double> (num) / static_cast<double> (den);
  };
  return {
      {"n", counts.n},
      {"answerable", counts.answerable},
      {"answer_hit_at_k", counts.answer_hit},
      {"answer_hit_rate", ratio (counts.answer_hit, counts.answerable)},
      {"evidence_queries", counts.evidence_queries},
      {"evidence_hit_at_k", counts.evidence_hit},
      {"evidence_hit_rate", ratio (counts.evidence_hit,
                                    counts.evidence_queries)},
      {"abstention_required", counts.abstention_required},
      {"abstention_retrieval_empty", counts.abstention_empty},
      {"abstention_empty_rate",
       ratio (counts.abstention_empty, counts.abstention_required)},
  };
}

int Run (const Options &opts)
{
  const auto episodes = LoadEpisodes (opts.episodes_path, opts.max_episodes);
  const auto queries = LoadQueries (opts.answer_key_path, opts.max_queries);

  std::filesystem::create_directories (opts.out_path.parent_path ());
  std::ofstream rows_out (opts.out_path);
  if (!rows_out)
    {
      throw std::runtime_error ("Failed to open output: " + opts.out_path.string ());
    }

  Counts totals;
  std::map<std::string, Counts> by_type;
  std::map<std::string, int> signal_modality_counts;
  int processed_episodes = 0;
  int processed_signals = 0;
  int skipped_unsupported_signals = 0;
  int episode_index = 0;

  std::unique_ptr<cortext::Cortext> shared_engine;
  if (opts.reuse_engine)
    {
      shared_engine = CreateEngine (opts, opts.db_path);
    }

  for (const auto &episode : episodes)
    {
      auto q_it = queries.find (episode.id);
      if (q_it == queries.end () || q_it->second.empty ())
        {
          continue;
        }

      std::unique_ptr<cortext::Cortext> local_engine;
      cortext::Cortext *engine = shared_engine.get ();
      if (!engine)
        {
          local_engine = CreateEngine (opts,
                                       DbPathForEpisode (opts, episode_index));
          engine = local_engine.get ();
        }

      int text_signals = 0;
      int image_signals = 0;
      int audio_signals = 0;
      int skipped_signals = 0;
      for (const auto &signal : episode.signals)
        {
          signal_modality_counts[signal.modality] += 1;
          if (signal.modality == "text")
            {
              if (signal.text.empty ())
                {
                  ++skipped_signals;
                  ++skipped_unsupported_signals;
                  continue;
                }
              if (signal.timestamp_ms > 0)
                {
                  (void)engine->ProcessTextAt (signal.text, signal.source_id,
                                               signal.timestamp_ms,
                                               cortext::Retention::Durable);
                }
              else
                {
                  (void)engine->ProcessText (signal.text, signal.source_id,
                                             cortext::Retention::Durable);
                }
              ++text_signals;
              ++processed_signals;
              continue;
            }
          if (signal.modality == "image")
            {
              if (signal.data_base64.empty () || signal.width <= 0
                  || signal.height <= 0 || signal.channels <= 0)
                {
                  ++skipped_signals;
                  ++skipped_unsupported_signals;
                  continue;
                }
              const auto pixels = DecodeBase64 (signal.data_base64);
              cortext::Cortext::Media media;
              media.mimetype = signal.mimetype;
              if (signal.timestamp_ms > 0)
                {
                  (void)engine->ProcessImage (pixels.data (), signal.width,
                                              signal.height, signal.channels,
                                              signal.source_id, media,
                                              cortext::Retention::Durable);
                }
              else
                {
                  (void)engine->ProcessImage (pixels.data (), signal.width,
                                              signal.height, signal.channels,
                                              signal.source_id, media,
                                              cortext::Retention::Durable);
                }
              ++image_signals;
              ++processed_signals;
              continue;
            }
          if (signal.modality == "audio")
            {
              if (signal.samples.empty ())
                {
                  ++skipped_signals;
                  ++skipped_unsupported_signals;
                  continue;
                }
              cortext::Cortext::Media media;
              media.mimetype = signal.mimetype;
              (void)engine->ProcessAudio (signal.samples.data (),
                                          signal.samples.size (),
                                          signal.source_id, media,
                                          cortext::Retention::Durable);
              ++audio_signals;
              ++processed_signals;
              continue;
            }
          ++skipped_signals;
          ++skipped_unsupported_signals;
        }
      engine->Flush ();

      for (const auto &query : q_it->second)
        {
          const auto start = std::chrono::steady_clock::now ();
          auto context = engine->ProcessText (query.question, "eval/query",
                                              cortext::Retention::Ephemeral);
          const auto end = std::chrono::steady_clock::now ();
          const double query_ms =
              std::chrono::duration<double, std::milli> (end - start).count ();

          std::string retrieved_text;
          nlohmann::json retrieved = nlohmann::json::array ();
          const std::size_t shown =
              std::min (opts.top_k, context.retrieved_memory.size ());
          for (std::size_t i = 0; i < shown; ++i)
            {
              const auto &memory = context.retrieved_memory[i];
              const std::string text = MemoryText (memory);
              retrieved_text += " ";
              retrieved_text += text;
              retrieved.push_back ({
                  {"rank", i + 1},
                  {"id", memory.id},
                  {"source_id", memory.source_id},
                  {"modality", memory.modality},
                  {"mimetype", memory.mimetype},
                  {"relevance", memory.relevance},
                  {"salience", memory.salience},
                  {"text", text},
                  {"preview", Clip (text, 240)},
              });
            }

          const bool answer_hit =
              ContainsAnyNormalized (query.answers, retrieved_text);
          const bool evidence_hit =
              ContainsAnyNormalized (query.evidence, retrieved_text);
          const bool retrieval_empty = context.retrieved_memory.empty ();
          AddCounts (totals, query, answer_hit, evidence_hit,
                     retrieval_empty);
          AddCounts (by_type[query.question_type], query, answer_hit,
                     evidence_hit, retrieval_empty);

          rows_out
              << nlohmann::json{
                     {"benchmark",
                      query.benchmark.empty () ? opts.benchmark
                                               : query.benchmark},
                     {"conversation_id", query.conversation_id},
                     {"query_id", query.query_id},
                     {"question", query.question},
                     {"question_type", query.question_type},
                     {"retrieved_count", context.retrieved_memory.size ()},
                     {"top_k", opts.top_k},
                     {"answer_hit_at_k", answer_hit},
                     {"evidence_hit_at_k", evidence_hit},
                     {"requires_abstention", query.requires_abstention},
                     {"retrieval_empty", retrieval_empty},
                     {"query_ms", query_ms},
                     {"ingested_signals",
                      text_signals + image_signals + audio_signals},
                     {"ingested_text_signals", text_signals},
                     {"ingested_image_signals", image_signals},
                     {"ingested_audio_signals", audio_signals},
                     {"skipped_unsupported_signals", skipped_signals},
                     {"retrieved", retrieved},
                 }.dump ()
              << "\n";
        }

      ++processed_episodes;
      ++episode_index;
      if (opts.max_episodes > 0 && processed_episodes >= opts.max_episodes)
        {
          break;
        }
    }

  nlohmann::json by_type_json = nlohmann::json::object ();
  for (const auto &kv : by_type)
    {
      by_type_json[kv.first] = CountsJson (kv.second);
    }
  nlohmann::json summary = {
      {"benchmark", opts.benchmark},
      {"episodes_path", opts.episodes_path.string ()},
      {"answer_key_path", opts.answer_key_path.string ()},
      {"processed_episodes", processed_episodes},
      {"processed_signals", processed_signals},
      {"skipped_unsupported_signals", skipped_unsupported_signals},
      {"signal_modality_counts", signal_modality_counts},
      {"top_k", opts.top_k},
      {"focus", opts.focus},
      {"sensitivity", opts.sensitivity},
      {"stability", opts.stability},
      {"totals", CountsJson (totals)},
      {"by_question_type", by_type_json},
      {"query_rows", opts.out_path.string ()},
  };

  auto summary_path = opts.out_path;
  summary_path.replace_extension (".summary.json");
  std::ofstream summary_out (summary_path);
  summary_out << summary.dump (2) << "\n";

  std::cout << summary.dump (2) << "\n";
  return 0;
}

} // namespace

int main (int argc, char **argv)
{
  try
    {
      return Run (ParseArgs (argc, argv));
    }
  catch (const std::exception &e)
    {
      std::cerr << "cortext_memory_eval_runner: " << e.what () << "\n";
      return 1;
    }
}
