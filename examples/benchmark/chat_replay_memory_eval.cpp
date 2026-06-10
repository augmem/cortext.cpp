#include <cortext/cortext.hpp>

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
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "transcript_discovery.hpp"

namespace fs = std::filesystem;

namespace
{

constexpr const char *kUserSourceId = "User";
constexpr const char *kContactSourceId = "Contact";

struct Config
{
  fs::path input_dir;
  fs::path db_path = "build/chat_replay_memory_eval.sqlite";
  fs::path output_path = "build/chat_replay_memory_eval_summary.json";
  std::string models_dir = "models";
  int max_messages = 2000;
  int holdout_stride = 17;
  int query_count = 120;
  int media_limit = 12;
  int consolidate_every = 0;
  double focus = 0.5;
  double sensitivity = 0.5;
  double stability = 0.5;
  bool deep_consolidation = false;
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

std::vector<std::string>
Tokens (const std::string &text)
{
  static const std::unordered_set<std::string> stop = {
    "the",  "and",  "you",  "that", "for", "with", "this", "have",
    "just", "but",  "not",  "are",  "was", "what", "from", "they",
    "your", "our",  "can",  "all",  "will", "there", "about", "would",
    "could", "should", "then", "when", "where", "were", "been", "into",
    "like", "okay", "yeah", "yes", "no", "ok", "lol", "im", "i"};
  std::vector<std::string> out;
  std::string cur;
  for (char ch : text)
    {
      unsigned char c = static_cast<unsigned char> (ch);
      if (std::isalnum (c))
        cur.push_back (static_cast<char> (std::tolower (c)));
      else if (!cur.empty ())
        {
          if (cur.size () >= 4 && stop.find (cur) == stop.end ())
            out.push_back (cur);
          cur.clear ();
        }
    }
  if (cur.size () >= 4 && stop.find (cur) == stop.end ())
    out.push_back (cur);
  return out;
}

double
OverlapScore (const std::vector<std::string> &query_tokens,
              const std::string &memory_text)
{
  if (query_tokens.empty ())
    return 0.0;
  const auto memory_tokens = Tokens (memory_text);
  if (memory_tokens.empty ())
    return 0.0;
  std::unordered_set<std::string> memory_set (memory_tokens.begin (),
                                              memory_tokens.end ());
  int hits = 0;
  for (const auto &token : query_tokens)
    {
      if (memory_set.find (token) != memory_set.end ())
        ++hits;
    }
  return static_cast<double> (hits)
         / static_cast<double> (std::max<size_t> (1, query_tokens.size ()));
}

std::string
MemoryText (const cortext::Cortext::Context::Memory &memory)
{
  std::string text;
  for (const auto &blob : memory.content)
    {
      for (unsigned char c : blob)
        {
          if (c == 0)
            break;
          if (c == '\n' || c == '\r' || c == '\t'
              || (c >= 32 && c < 127))
            text.push_back (static_cast<char> (c));
        }
      text.push_back (' ');
    }
  return text;
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

std::vector<MediaItem>
FindMedia (const fs::path &dir)
{
  std::vector<MediaItem> items;
  for (const auto &entry : fs::directory_iterator (dir))
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
      else if (ext == ".m4a")
        kind = "audio";
      else
        continue;

      auto ts = ParseTimestamp (entry.path ().filename ().string ());
      if (!ts)
        continue;
      items.push_back ({ entry.path (), *ts, kind });
    }
  std::sort (items.begin (), items.end (),
             [] (const auto &a, const auto &b) {
               return a.timestamp < b.timestamp;
             });
  return items;
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
      else if (arg == "--max-messages")
        cfg.max_messages = std::stoi (require_value ());
      else if (arg == "--holdout-stride")
        cfg.holdout_stride = std::stoi (require_value ());
      else if (arg == "--query-count")
        cfg.query_count = std::stoi (require_value ());
      else if (arg == "--media-limit")
        cfg.media_limit = std::stoi (require_value ());
      else if (arg == "--consolidate-every")
        cfg.consolidate_every = std::stoi (require_value ());
      else if (arg == "--focus")
        cfg.focus = std::stod (require_value ());
      else if (arg == "--sensitivity")
        cfg.sensitivity = std::stod (require_value ());
      else if (arg == "--stability")
        cfg.stability = std::stod (require_value ());
      else if (arg == "--deep")
        cfg.deep_consolidation = true;
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
      const fs::path transcript
          = chat_replay::DiscoverTranscript (cfg.input_dir);
      auto messages = ParseMessages (transcript);
      if (cfg.max_messages > 0
          && static_cast<int> (messages.size ()) > cfg.max_messages)
        messages.resize (static_cast<size_t> (cfg.max_messages));

      fs::create_directories (cfg.db_path.parent_path ());
      fs::remove (cfg.db_path);
      fs::remove (cfg.db_path.string () + "-wal");
      fs::remove (cfg.db_path.string () + "-shm");

      cortext::Cortext::Config cortext_cfg;
      cortext_cfg.focus = cfg.focus;
      cortext_cfg.sensitivity = cfg.sensitivity;
      cortext_cfg.stability = cfg.stability;
      auto engine = cortext::Cortext::Create (cortext_cfg, cfg.db_path.string (),
                                              cfg.models_dir);

      std::vector<Message> holdouts;
      int processed = 0;
      int writes = 0;
      int retrieval_events = 0;
      int retrieval_items = 0;
      int consolidation_runs = 0;
      double process_ms_total = 0.0;

      for (const auto &msg : messages)
        {
          const bool holdout = cfg.holdout_stride > 0
                               && msg.index % cfg.holdout_stride == 0
                               && Tokens (msg.text).size () >= 3;
          if (holdout)
            {
              holdouts.push_back (msg);
              continue;
            }
          const std::string source = msg.from_contact ? kContactSourceId
                                                      : kUserSourceId;
          auto ctx = engine->ProcessTextAt (msg.text, source, msg.timestamp);
          ++processed;
          process_ms_total += ctx.process_ms;
          if (ctx.output.stored_memory_id || ctx.output.stored_embedding_id)
            ++writes;
          if (!ctx.retrieved_memory.empty ())
            ++retrieval_events;
          retrieval_items += static_cast<int> (ctx.retrieved_memory.size ());
          if (cfg.consolidate_every > 0
              && processed % cfg.consolidate_every == 0)
            {
              engine->Consolidate (
                  cfg.deep_consolidation ? cortext::ConsolidationMode::Both
                                         : cortext::ConsolidationMode::Shallow);
              ++consolidation_runs;
            }
        }
      std::uint64_t query_ts
          = messages.empty () ? 0 : messages.back ().timestamp + 1000;

      engine->Consolidate (cfg.deep_consolidation
                               ? cortext::ConsolidationMode::Both
                               : cortext::ConsolidationMode::Shallow);
      ++consolidation_runs;

      std::mt19937 rng (42);
      std::shuffle (holdouts.begin (), holdouts.end (), rng);
      if (cfg.query_count > 0
          && static_cast<int> (holdouts.size ()) > cfg.query_count)
        holdouts.resize (static_cast<size_t> (cfg.query_count));

      int query_count = 0;
      int query_with_retrieval = 0;
      int temporal_day_hit = 0;
      int lexical_hit = 0;
      double max_overlap_total = 0.0;
      int retrieved_for_queries = 0;

      for (const auto &query : holdouts)
        {
          auto ctx = engine->ProcessTextAt (
              query.text, query.from_contact ? kContactSourceId : kUserSourceId,
              query_ts, cortext::Retention::Ephemeral);
          query_ts += 1000;
          ++query_count;
          if (!ctx.retrieved_memory.empty ())
            ++query_with_retrieval;
          retrieved_for_queries += static_cast<int> (ctx.retrieved_memory.size ());

          const auto q_tokens = Tokens (query.text);
          double best_overlap = 0.0;
          bool day_hit = false;
          for (const auto &memory : ctx.retrieved_memory)
            {
              const std::uint64_t diff
                  = memory.timestamp > query.timestamp
                        ? memory.timestamp - query.timestamp
                        : query.timestamp - memory.timestamp;
              if (diff <= 24ULL * 60ULL * 60ULL * 1000ULL)
                day_hit = true;
              best_overlap
                  = std::max (best_overlap, OverlapScore (q_tokens,
                                                          MemoryText (memory)));
            }
          if (day_hit)
            ++temporal_day_hit;
          if (best_overlap >= 0.25)
            ++lexical_hit;
          max_overlap_total += best_overlap;
        }

      int media_attempted = 0;
      int media_processed = 0;
      int media_failures = 0;
      int image_processed = 0;
      int audio_processed = 0;
      int video_processed = 0;
      const auto media = FindMedia (cfg.input_dir);
      const fs::path tmp_dir = cfg.db_path.parent_path () / "chat_replay_media_tmp";
      fs::create_directories (tmp_dir);
      for (const auto &item : media)
        {
          if (cfg.media_limit >= 0 && media_attempted >= cfg.media_limit)
            break;
          ++media_attempted;
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
                  engine->ProcessImage (bytes.data (), 224, 224, 3,
                                        item.kind == "image" ? "chat_replay/image"
                                                            : "chat_replay/video");
                  ++media_processed;
                  item.kind == "image" ? ++image_processed : ++video_processed;
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
                      engine->ProcessAudio (pcm.data (), pcm.size (),
                                            "chat_replay/audio");
                      ++media_processed;
                      ++audio_processed;
                    }
                }
              else
                {
                  ++media_failures;
                }
            }
        }

      nlohmann::json out;
      out["input_dir"] = cfg.input_dir.string ();
      out["transcript_messages_considered"] = messages.size ();
      out["processed_text_messages"] = processed;
      out["heldout_queries_available"] = holdouts.size ();
      out["stored_write_events"] = writes;
      out["retrieval_events_during_ingest"] = retrieval_events;
      out["retrieval_items_during_ingest"] = retrieval_items;
      out["consolidation_runs"] = consolidation_runs;
      out["mean_process_ms"] = processed > 0 ? process_ms_total / processed : 0.0;
      out["query_count"] = query_count;
      out["query_with_retrieval"] = query_with_retrieval;
      out["query_retrieval_rate"]
          = query_count > 0 ? static_cast<double> (query_with_retrieval)
                                  / query_count
                            : 0.0;
      out["temporal_day_hit_rate"]
          = query_count > 0 ? static_cast<double> (temporal_day_hit)
                                  / query_count
                            : 0.0;
      out["lexical_overlap_hit_rate"]
          = query_count > 0 ? static_cast<double> (lexical_hit) / query_count
                            : 0.0;
      out["mean_best_lexical_overlap"]
          = query_count > 0 ? max_overlap_total / query_count : 0.0;
      out["retrieved_items_for_queries"] = retrieved_for_queries;
      out["media_candidates_found"] = media.size ();
      out["media_attempted"] = media_attempted;
      out["media_processed"] = media_processed;
      out["media_failures"] = media_failures;
      out["image_processed"] = image_processed;
      out["video_processed"] = video_processed;
      out["audio_processed"] = audio_processed;
      out["deep_consolidation"] = cfg.deep_consolidation;
      out["knobs"] = {
        { "focus", cfg.focus },
        { "sensitivity", cfg.sensitivity },
        { "stability", cfg.stability },
      };
      out["privacy_note"]
          = "Aggregate metrics only; message text and media content are not "
            "written to this summary.";

      std::ofstream summary (cfg.output_path);
      summary << out.dump (2) << "\n";
      std::cout << out.dump (2) << "\n";
      return 0;
    }
  catch (const std::exception &e)
    {
      std::cerr << "chat_replay_memory_eval failed: " << e.what () << "\n";
      return 1;
    }
}
