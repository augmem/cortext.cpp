// cortext_memory_eval_runner: benchmark adapter for normalized memory evals.
//
// The runner uses only the public Cortext C++ API. Histories are ingested with
// durable retention, then benchmark questions are issued with ephemeral
// retention so the query path does not pollute long-term memory.

#include <cortext/clock.hpp>
#include <cortext/cortext.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
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
  // Per-query evidence budget in bytes (Cortext::EvidenceBudget); unset
  // returns whole memories.
  cortext::Cortext::EvidenceBudget evidence_budget;
  int max_episodes = 0;
  int max_queries = 0;
  bool reuse_engine = false;
  bool consolidate_on_recommend = false;
  // Drive the engine clock from the stream (turn timestamps). Needed for
  // image/audio signals, whose public API has no explicit-timestamp form.
  bool stream_clock = false;
  // Engine default is on; "off" measures images the change detector drops.
  bool image_filter = true;
  // "on": a turn is one multi-signal unit. Image signals whose source_id is
  // "<text source>/<suffix>" and that directly follow that text signal are
  // sent under the text's source id; the text goes in as Natural and the
  // turn's last signal as Durable, so the engine stores them as one memory.
  bool turn_signals = false;
  // Ingest source id. "conversation": one id per conversation/session
  // (<dataset id prefix>/conv), "speaker": one per speaker in it
  // (<prefix>/<speaker>). The dataset's per-turn ids stay in rows/logs.
  bool speaker_sources = false;
  // Durable forces a boundary and a write per turn; Natural leaves
  // boundary detection, grouping and the write gate to the engine (a real
  // host's default). Both flush at episode end, before queries.
  cortext::Retention retention = cortext::Retention::Durable;
  // Query-time-only variants: copy <dir>/store_<i>.sqlite (kept by an
  // earlier run with the same --db stem) and skip ingest.
  std::string query_from;
  // Natural only: send each conversation's last signal as Boundary, as a
  // host that knows a conversation ended (closes the open unit; the write
  // gate still decides).
  bool close_conversations = false;
  std::filesystem::path ingest_log_path;
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
  // Multimodal file inputs: decoded pixels for ProcessImage, and the original
  // compressed bytes stored as Media. media_ref is echoed into packet rows.
  std::string pixels_path;
  std::string media_path;
  nlohmann::json media_ref;
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
  std::uint64_t timestamp_ms = 0;
  // Optional query reference time (answer-key "reference_ms", "as_of"),
  // passed to the engine as Cortext::RetrievalTime.
  std::optional<std::uint64_t> reference_ms;
  bool as_of = false;
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
      << "  --evidence-budget-bytes=N  Per-query evidence budget in bytes\n"
      << "                      (standing + retrieved packet); default unset\n"
      << "                      Answer-key rows may carry \"reference_ms\" (query\n"
      << "                      reference time, ms) and \"as_of\" (bool), passed\n"
      << "                      to the engine as Cortext::RetrievalTime\n"
      << "  --max-episodes N    Limit episodes, 0 means all\n"
      << "  --max-queries N     Limit queries, 0 means all\n"
      << "  --reuse-engine      Reuse one engine/DB across episodes\n"
      << "  --consolidate=recommended  Run Consolidate() whenever an ingest\n"
      << "                      context reports Recommended or Required\n"
      << "  --stream-clock      Engine clock follows turn timestamps (needed\n"
      << "                      for image/audio, which have no *At API)\n"
      << "  --image-filter=on|off  Engine image signal filter, default on\n"
      << "  --source-ids=conversation|speaker  Ingest source id per\n"
      << "                      conversation (default) or per speaker\n"
      << "  --retention=durable|natural  Ingest retention, default durable\n"
      << "                      (one memory per turn); natural lets the\n"
      << "                      engine group turns into memories\n"
      << "  --conversation-end=none|boundary  Natural only: boundary sends\n"
      << "                      each conversation's last signal as Boundary\n"
      << "  --query-from=DIR    Query DIR/store_<i>.sqlite (copied) without\n"
      << "                      ingesting; needs a per-episode --db file\n"
      << "  --ingest-log PATH   Write one JSONL row per ingested signal\n"
      << "                      (modality, wall/encode/process ms, filter)\n"
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
      else if (auto v = TakeArg (arg, "--evidence-budget-bytes="))
        {
          const auto bytes = std::stoull (*v);
          if (bytes == 0)
            throw std::runtime_error ("--evidence-budget-bytes must be > 0");
          opts.evidence_budget = static_cast<std::size_t> (bytes);
        }
      else if (auto v = TakeArg (arg, "--max-episodes="))
        {
          opts.max_episodes = std::stoi (*v);
        }
      else if (auto v = TakeArg (arg, "--max-queries="))
        {
          opts.max_queries = std::stoi (*v);
        }
      else if (arg == "--stream-clock")
        {
          opts.stream_clock = true;
        }
      else if (auto v = TakeArg (arg, "--image-filter="))
        {
          if (*v != "on" && *v != "off")
            throw std::runtime_error ("--image-filter must be on or off");
          opts.image_filter = *v == "on";
        }
      else if (auto v = TakeArg (arg, "--turn-signals="))
        {
          if (*v != "on" && *v != "off")
            throw std::runtime_error ("--turn-signals must be on or off");
          opts.turn_signals = *v == "on";
        }
      else if (auto v = TakeArg (arg, "--source-ids="))
        {
          if (*v != "conversation" && *v != "speaker")
            throw std::runtime_error (
                "--source-ids must be conversation or speaker");
          opts.speaker_sources = *v == "speaker";
        }
      else if (auto v = TakeArg (arg, "--retention="))
        {
          if (*v != "durable" && *v != "natural")
            throw std::runtime_error ("--retention must be durable or natural");
          opts.retention = *v == "natural" ? cortext::Retention::Natural
                                           : cortext::Retention::Durable;
        }
      else if (auto v = TakeArg (arg, "--conversation-end="))
        {
          if (*v != "none" && *v != "boundary")
            throw std::runtime_error (
                "--conversation-end must be none or boundary");
          opts.close_conversations = *v == "boundary";
        }
      else if (auto v = TakeArg (arg, "--query-from="))
        {
          opts.query_from = *v;
        }
      else if (auto v = TakeArg (arg, "--ingest-log="))
        {
          opts.ingest_log_path = *v;
        }
      else if (arg == "--reuse-engine")
        {
          opts.reuse_engine = true;
        }
      else if (auto v = TakeArg (arg, "--consolidate="))
        {
          if (*v != "recommended" && *v != "off")
            throw std::runtime_error ("--consolidate must be recommended or off");
          opts.consolidate_on_recommend = *v == "recommended";
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
  if (opts.close_conversations
      && opts.retention != cortext::Retention::Natural)
    {
      throw std::runtime_error (
          "--conversation-end=boundary needs --retention=natural");
    }
  if (!opts.query_from.empty ()
      && (opts.db_path == ":memory:" || opts.reuse_engine))
    {
      throw std::runtime_error (
          "--query-from needs a per-episode --db file");
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

/// Engine clock for --stream-clock: the stream's latest timestamp, or the
/// wall clock until the first timestamped signal arrives.
class StreamClock final : public cortext::Clock
{
public:
  std::uint64_t
  NowMillis () const override
  {
    const auto now = now_ms_.load ();
    return now != 0 ? now : system_.NowMillis ();
  }
  void
  Set (std::uint64_t ms)
  {
    if (ms > now_ms_.load ())
      {
        now_ms_.store (ms);
      }
  }

private:
  std::atomic<std::uint64_t> now_ms_{0};
  cortext::SystemClock system_;
};

std::uint64_t Fnv1a64 (const unsigned char *data, std::size_t size)
{
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < size; ++i)
    {
      h ^= data[i];
      h *= 1099511628211ULL;
    }
  return h ^ static_cast<std::uint64_t> (size);
}

/// Payload bytes the runner ingested as media -> reference echoed in packets.
using MediaRegistry = std::unordered_map<std::uint64_t, nlohmann::json>;

bool LooksLikeEncodedImage (const std::vector<unsigned char> &blob)
{
  static const unsigned char kJpeg[] = {0xFF, 0xD8, 0xFF};
  static const unsigned char kPng[] = {0x89, 'P', 'N', 'G'};
  return (blob.size () >= 3 && std::memcmp (blob.data (), kJpeg, 3) == 0)
         || (blob.size () >= 4 && std::memcmp (blob.data (), kPng, 4) == 0);
}

/// Split a memory's blobs into text (joined exactly as MemoryText) and media
/// references. Media blobs are never flattened into text.
std::pair<std::string, nlohmann::json>
MemoryTextAndMedia (const cortext::Cortext::Context::Memory &memory,
                    const MediaRegistry &registry)
{
  cortext::Cortext::Context::Memory text_only;
  nlohmann::json media = nlohmann::json::array ();
  for (std::size_t i = 0; i < memory.content.size (); ++i)
    {
      const auto &blob = memory.content[i];
      const std::string &mime = i < memory.content_mimetypes.size ()
                                    ? memory.content_mimetypes[i]
                                    : memory.mimetype;
      const auto it = registry.find (Fnv1a64 (blob.data (), blob.size ()));
      if (it != registry.end ())
        {
          media.push_back (it->second);
        }
      else if (LooksLikeEncodedImage (blob)
               || (!mime.empty () && mime.rfind ("text/", 0) != 0))
        {
          media.push_back ({{"mimetype", mime},
                            {"bytes", blob.size ()},
                            {"unregistered", true}});
        }
      else
        {
          text_only.content.push_back (blob);
        }
    }
  return {MemoryText (text_only), media};
}

std::vector<unsigned char> ReadFileBytes (const std::string &path)
{
  std::ifstream in (path, std::ios::binary);
  if (!in)
    {
      throw std::runtime_error ("Failed to open media file: " + path);
    }
  return std::vector<unsigned char> (std::istreambuf_iterator<char> (in),
                                     std::istreambuf_iterator<char> ());
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
  // Back off to a UTF-8 code point boundary so the JSON dump stays valid.
  std::size_t cut = max_chars;
  while (cut > 0 && (static_cast<unsigned char> (text[cut]) & 0xC0u) == 0x80u)
    {
      --cut;
    }
  return text.substr (0, cut);
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

// Dataset ids are <prefix>/<turn index>[/<attachment>...], prefix being
// eval/<episode>[/<session>]. Returns the prefix (at least two segments).
std::string ConversationPrefix (const std::string &dataset_source_id)
{
  std::vector<std::string> parts;
  std::stringstream ss (dataset_source_id);
  std::string part;
  while (std::getline (ss, part, '/'))
    parts.push_back (part);
  auto numeric = [] (const std::string &x) {
    return !x.empty ()
           && std::all_of (x.begin (), x.end (), [] (unsigned char c) {
                return std::isdigit (c) != 0;
              });
  };
  while (parts.size () > 2 && !numeric (parts.back ()))
    parts.pop_back ();
  if (parts.size () > 2)
    parts.pop_back ();
  std::string out;
  for (const auto &p : parts)
    out += (out.empty () ? "" : "/") + p;
  return out;
}

// "[<timestamp>] <speaker>: ..." -> lowercase speaker, else empty.
std::string TurnSpeaker (const std::string &text)
{
  if (text.empty () || text[0] != '[')
    return "";
  const auto close = text.find (']');
  if (close == std::string::npos)
    return "";
  std::size_t b = close + 1;
  while (b < text.size () && text[b] == ' ')
    ++b;
  std::size_t e = b;
  while (e < text.size ()
         && (std::isalnum (static_cast<unsigned char> (text[e])) != 0
             || text[e] == '_'))
    ++e;
  if (e == b || e >= text.size () || text[e] != ':')
    return "";
  std::string speaker = text.substr (b, e - b);
  for (auto &c : speaker)
    c = static_cast<char> (std::tolower (static_cast<unsigned char> (c)));
  return speaker;
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
  signal.pixels_path = JsonString (j, "pixels_path");
  signal.media_path = JsonString (j, "media_path");
  if (!signal.media_path.empty ())
    {
      signal.media_ref = {{"source_id", signal.source_id},
                          {"mimetype", signal.mimetype},
                          {"path", signal.media_path}};
      for (const char *key : {"caption", "judge_path", "media_sha"})
        {
          const std::string value = JsonString (j, key);
          if (!value.empty ())
            {
              signal.media_ref[key] = value;
            }
        }
    }
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
      q.timestamp_ms = JsonU64 (j, "query_timestamp_ms");
      if (q.evidence.empty ())
        {
          q.evidence = JsonStringList (j, "evidence_phrases");
        }
      const auto reference_it = j.find ("reference_ms");
      if (reference_it != j.end () && reference_it->is_number_integer ()
          && reference_it->get<long long> () >= 0)
        {
          q.reference_ms = reference_it->get<std::uint64_t> ();
        }
      const auto as_of_it = j.find ("as_of");
      if (as_of_it != j.end () && as_of_it->is_boolean ())
        {
          q.as_of = as_of_it->get<bool> ();
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

bool ImageSignalUsable (const Signal &signal)
{
  return !(signal.data_base64.empty () && signal.pixels_path.empty ())
         && signal.width > 0 && signal.height > 0 && signal.channels > 0;
}

// Decoded pixels of a usable image signal; throws on a size mismatch.
std::vector<unsigned char> ImagePixels (const Signal &signal)
{
  auto pixels = signal.pixels_path.empty () ? DecodeBase64 (signal.data_base64)
                                            : ReadFileBytes (signal.pixels_path);
  const std::size_t want = static_cast<std::size_t> (signal.width)
                           * static_cast<std::size_t> (signal.height)
                           * static_cast<std::size_t> (signal.channels);
  if (pixels.size () != want)
    {
      throw std::runtime_error ("pixel buffer size mismatch for "
                                + signal.source_id);
    }
  return pixels;
}

// Packet rows echo a media ref for stored image bytes, keyed by content hash.
void RegisterImageMedia (const Signal &signal,
                         const std::vector<unsigned char> &stored,
                         MediaRegistry &registry)
{
  nlohmann::json ref = signal.media_ref.is_null ()
                           ? nlohmann::json{{"source_id", signal.source_id},
                                            {"mimetype", signal.mimetype}}
                           : signal.media_ref;
  ref["bytes"] = stored.size ();
  registry[Fnv1a64 (stored.data (), stored.size ())] = ref;
}

std::unique_ptr<cortext::Cortext>
CreateEngine (const Options &opts, const std::string &db_path,
              const std::shared_ptr<StreamClock> &clock)
{
  cortext::Cortext::Config cfg;
  cfg.focus = opts.focus;
  cfg.sensitivity = opts.sensitivity;
  cfg.stability = opts.stability;
  cfg.signal_filter_image_enabled = opts.image_filter;
  if (clock)
    {
      return cortext::Cortext::Create (cfg, db_path, clock);
    }
  return cortext::Cortext::Create (cfg, db_path);
}

nlohmann::json Percentiles (std::vector<double> xs)
{
  if (xs.empty ())
    {
      return {{"n", 0}};
    }
  std::sort (xs.begin (), xs.end ());
  auto at = [&] (double q) {
    return xs[std::min (xs.size () - 1,
                        static_cast<std::size_t> (q * static_cast<double> (xs.size ())))];
  };
  double sum = 0.0;
  for (double x : xs)
    {
      sum += x;
    }
  return {{"n", xs.size ()},
          {"mean", sum / static_cast<double> (xs.size ())},
          {"p50", at (0.50)},
          {"p95", at (0.95)},
          {"max", xs.back ()}};
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
  std::shared_ptr<StreamClock> shared_clock;
  if (opts.reuse_engine)
    {
      if (opts.stream_clock)
        {
          shared_clock = std::make_shared<StreamClock> ();
        }
      shared_engine = CreateEngine (opts, opts.db_path, shared_clock);
    }
  MediaRegistry media_registry;
  std::map<std::string, std::vector<double>> ingest_ms_by_modality;
  std::map<std::string, std::vector<double>> encode_ms_by_modality;
  std::map<std::string, int> filtered_by_modality;
  std::map<std::string, std::uint64_t> media_bytes_by_modality;
  std::vector<double> query_ms_all;
  std::ofstream ingest_log;
  if (!opts.ingest_log_path.empty ())
    {
      ingest_log.open (opts.ingest_log_path);
      if (!ingest_log)
        {
          throw std::runtime_error ("Failed to open ingest log: "
                                    + opts.ingest_log_path.string ());
        }
    }

  for (const auto &episode : episodes)
    {
      auto q_it = queries.find (episode.id);
      if (q_it == queries.end () || q_it->second.empty ())
        {
          continue;
        }

      std::unique_ptr<cortext::Cortext> local_engine;
      std::shared_ptr<StreamClock> clock = shared_clock;
      cortext::Cortext *engine = shared_engine.get ();
      const bool query_only = !opts.query_from.empty ();
      if (query_only)
        {
          const auto name
              = std::filesystem::path (DbPathForEpisode (opts, episode_index))
                    .filename ()
                    .string ();
          const auto src = std::filesystem::path (opts.query_from) / name;
          if (!std::filesystem::exists (src))
            throw std::runtime_error ("--query-from store missing: "
                                      + src.string ());
          for (const char *suffix : {"", "-wal"})
            {
              const auto from = src.string () + suffix;
              const auto to = DbPathForEpisode (opts, episode_index) + suffix;
              std::filesystem::remove (to);
              if (std::filesystem::exists (from))
                std::filesystem::copy_file (from, to);
            }
          std::filesystem::remove (DbPathForEpisode (opts, episode_index)
                                   + "-shm");
        }
      if (!engine)
        {
          if (opts.stream_clock)
            {
              clock = std::make_shared<StreamClock> ();
            }
          local_engine = CreateEngine (
              opts, DbPathForEpisode (opts, episode_index), clock);
          engine = local_engine.get ();
        }

      int text_signals = 0;
      int image_signals = 0;
      int audio_signals = 0;
      int skipped_signals = 0;
      int consolidations = 0;
      // Production cadence: the host consolidates when the engine says so.
      auto settle = [&] (const auto &ingest_context) {
        if (opts.consolidate_on_recommend
            && ingest_context.consolidation_state
                   != cortext::ConsolidationState::None)
          {
            (void)engine->Consolidate ();
            engine->Flush ();
            ++consolidations;
          }
      };
      std::uint64_t last_timestamp_ms = 0;
      // Per-signal ingest timing; does not touch engine state.
      auto record = [&] (const Signal &signal, const auto &ctx,
                         std::chrono::steady_clock::time_point t0) {
        const double ms = std::chrono::duration<double, std::milli> (
                              std::chrono::steady_clock::now () - t0)
                              .count ();
        ingest_ms_by_modality[signal.modality].push_back (ms);
        encode_ms_by_modality[signal.modality].push_back (ctx.encode_ms);
        const bool accepted = !ctx.output.signal_filter_evaluated
                              || ctx.output.signal_filter_accepted;
        if (!accepted)
          {
            filtered_by_modality[signal.modality] += 1;
          }
        if (ingest_log.is_open ())
          {
            ingest_log << nlohmann::json{
                {"episode_id", episode.id},
                {"source_id", signal.source_id},
                {"modality", signal.modality},
                {"ms", ms},
                {"encode_ms", ctx.encode_ms},
                {"process_ms", ctx.process_ms},
                {"hydrate_ms", ctx.hydrate_ms},
                {"accepted", accepted},
                {"filter_reason", ctx.output.signal_filter_reason},
                {"consolidation", ctx.consolidation_state
                                      != cortext::ConsolidationState::None},
            }.dump () << "\n";
          }
      };
      // Per-signal source id and retention (multi-signal turns, see
      // Options::turn_signals).
      std::vector<std::string> turn_source;
      turn_source.reserve (episode.signals.size ());
      for (const auto &s : episode.signals)
        turn_source.push_back (s.source_id);
      std::vector<cortext::Retention> turn_retention (
          episode.signals.size (), cortext::Retention::Durable);
      for (std::size_t i = 0; i < episode.signals.size (); ++i)
        {
          if (!opts.turn_signals || episode.signals[i].modality != "text")
            continue;
          const std::string prefix = episode.signals[i].source_id + "/";
          std::size_t last = i;
          for (std::size_t j = i + 1; j < episode.signals.size (); ++j)
            {
              const auto &next = episode.signals[j];
              if (next.modality == "text"
                  || next.source_id.rfind (prefix, 0) != 0)
                break;
              last = j;
            }
          for (std::size_t j = i + 1; j <= last; ++j)
            {
              turn_source[j] = episode.signals[i].source_id;
              turn_retention[j - 1] = cortext::Retention::Natural;
            }
        }
      // Shared conversation (or speaker) ids instead of per-turn ids.
      std::string speaker = "unknown";
      for (std::size_t i = 0; i < episode.signals.size (); ++i)
        {
          if (episode.signals[i].modality == "text")
            {
              const auto s = TurnSpeaker (episode.signals[i].text);
              speaker = s.empty () ? "unknown" : s;
            }
          turn_source[i] = ConversationPrefix (turn_source[i]) + "/"
                           + (opts.speaker_sources ? speaker : "conv");
          if (opts.retention == cortext::Retention::Natural)
            turn_retention[i] = cortext::Retention::Natural;
        }
      if (opts.close_conversations)
        {
          std::map<std::string, std::size_t> last_of;
          for (std::size_t i = 0; i < turn_source.size (); ++i)
            last_of[turn_source[i]] = i;
          for (const auto &[source, i] : last_of)
            turn_retention[i] = cortext::Retention::Boundary;
        }
      // Query-only replays what ingest leaves outside the store: the media
      // registry (packet media refs) and the clamp time, which ingest takes
      // from timestamped non-empty text signals only.
      if (query_only)
        for (const auto &signal : episode.signals)
          {
            if (clock && signal.timestamp_ms > 0)
              clock->Set (signal.timestamp_ms);
            if (signal.modality == "text" && !signal.text.empty ()
                && signal.timestamp_ms > 0)
              last_timestamp_ms
                  = std::max (last_timestamp_ms, signal.timestamp_ms);
            if (signal.modality == "image" && ImageSignalUsable (signal))
              {
                auto stored = signal.media_path.empty ()
                                  ? ImagePixels (signal)
                                  : ReadFileBytes (signal.media_path);
                RegisterImageMedia (signal, stored, media_registry);
              }
          }
      for (std::size_t signal_index = 0;
           !query_only && signal_index < episode.signals.size ();
           ++signal_index)
        {
          const auto &signal = episode.signals[signal_index];
          const auto &source_id = turn_source[signal_index];
          const auto retention = turn_retention[signal_index];
          signal_modality_counts[signal.modality] += 1;
          if (clock && signal.timestamp_ms > 0)
            {
              clock->Set (signal.timestamp_ms);
            }
          const auto signal_start = std::chrono::steady_clock::now ();
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
                  last_timestamp_ms
                      = std::max (last_timestamp_ms, signal.timestamp_ms);
                  const auto ctx = engine->ProcessTextAt (
                      signal.text, source_id, last_timestamp_ms, retention);
                  record (signal, ctx, signal_start);
                  settle (ctx);
                }
              else
                {
                  const auto ctx = engine->ProcessText (
                      signal.text, source_id, retention);
                  record (signal, ctx, signal_start);
                  settle (ctx);
                }
              ++text_signals;
              ++processed_signals;
              continue;
            }
          if (signal.modality == "image")
            {
              if (!ImageSignalUsable (signal))
                {
                  ++skipped_signals;
                  ++skipped_unsupported_signals;
                  continue;
                }
              // Pixel decode happens upstream (build step); file reads are
              // excluded from the ingest timing below.
              const auto pixels = ImagePixels (signal);
              std::vector<unsigned char> media_bytes;
              if (!signal.media_path.empty ())
                {
                  media_bytes = ReadFileBytes (signal.media_path);
                }
              cortext::Cortext::Media media;
              media.mimetype = signal.mimetype;
              media.data = media_bytes.empty () ? nullptr : media_bytes.data ();
              media.size = media_bytes.size ();
              // The engine stores Media bytes when given, else raw pixels.
              const auto &stored = media_bytes.empty () ? pixels : media_bytes;
              RegisterImageMedia (signal, stored, media_registry);
              media_bytes_by_modality["image"] += stored.size ();
              const auto image_start = std::chrono::steady_clock::now ();
              const auto ctx = engine->ProcessImage (
                  pixels.data (), signal.width, signal.height, signal.channels,
                  source_id, media, retention);
              record (signal, ctx, image_start);
              settle (ctx);
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
              const auto ctx = engine->ProcessAudio (
                  signal.samples.data (), signal.samples.size (),
                  source_id, media, retention);
              record (signal, ctx, signal_start);
              settle (ctx);
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
          if (clock)
            {
              clock->Set (std::max (query.timestamp_ms, last_timestamp_ms));
            }
          const auto start = std::chrono::steady_clock::now ();
          // Ask at the benchmark's question date when the key carries one,
          // clamped so engine time never runs backwards.
          const cortext::RetrievalTime retrieval_time{ query.reference_ms,
                                                       query.as_of };
          auto context
              = query.timestamp_ms > 0
                    ? engine->ProcessTextAt (
                          query.question, "eval/query",
                          std::max (query.timestamp_ms, last_timestamp_ms),
                          cortext::Retention::Ephemeral, opts.evidence_budget,
                          retrieval_time)
                    : engine->ProcessText (query.question, "eval/query",
                                           cortext::Retention::Ephemeral,
                                           opts.evidence_budget,
                                           retrieval_time);
          const auto end = std::chrono::steady_clock::now ();
          const double query_ms =
              std::chrono::duration<double, std::milli> (end - start).count ();
          query_ms_all.push_back (query_ms);

          std::string retrieved_text;
          nlohmann::json retrieved = nlohmann::json::array ();
          const std::size_t shown =
              std::min (opts.top_k, context.retrieved_memory.size ());
          for (std::size_t i = 0; i < shown; ++i)
            {
              const auto &memory = context.retrieved_memory[i];
              const auto [text, media]
                  = MemoryTextAndMedia (memory, media_registry);
              retrieved_text += " ";
              retrieved_text += text;
              nlohmann::json item = {
                  {"rank", i + 1},
                  {"id", memory.id},
                  {"source_id", memory.source_id},
                  {"modality", memory.modality},
                  {"mimetype", memory.mimetype},
                  {"relevance", memory.relevance},
                  {"salience", memory.salience},
                  {"text", text},
                  {"preview", Clip (text, 240)},
              };
              // Text-only packets keep the exact prior row shape.
              if (!media.empty ())
                {
                  item["media"] = media;
                }
              retrieved.push_back (std::move (item));
            }

          // The standing block is outside the top-k cut (retrieved).
          nlohmann::json standing = nlohmann::json::array ();
          for (std::size_t i = 0; i < context.standing_memory.size (); ++i)
            {
              const auto &memory = context.standing_memory[i];
              const auto [text, media]
                  = MemoryTextAndMedia (memory, media_registry);
              nlohmann::json item = {
                  {"rank", i + 1},
                  {"id", memory.id},
                  {"source_id", memory.source_id},
                  {"modality", memory.modality},
                  {"text", text},
              };
              if (!media.empty ())
                {
                  item["media"] = media;
                }
              standing.push_back (std::move (item));
            }

          nlohmann::json working = nlohmann::json::array ();
          for (std::size_t i = 0; i < context.working_memory.size (); ++i)
            {
              const auto &memory = context.working_memory[i];
              const auto [text, media]
                  = MemoryTextAndMedia (memory, media_registry);
              nlohmann::json item = {
                  {"slot", i},
                  {"id", memory.id},
                  {"source_id", memory.source_id},
                  {"modality", memory.modality},
                  {"text", text},
              };
              if (!media.empty ())
                {
                  item["media"] = media;
                }
              working.push_back (std::move (item));
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

          nlohmann::json row = nlohmann::json{
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
                     {"consolidations", consolidations},
                     {"query_timestamp_ms", query.timestamp_ms},
                     {"retrieved", retrieved},
                     {"standing", standing},
                     {"working", working},
                 };
          // Rows of queries without a reference time keep their prior shape.
          if (query.reference_ms)
            {
              row["reference_ms"] = *query.reference_ms;
              row["as_of"] = query.as_of;
            }
          rows_out << row.dump ()
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
      {"evidence_budget_bytes",
       opts.evidence_budget ? nlohmann::json (*opts.evidence_budget)
                            : nlohmann::json ()},
      {"stream_clock", opts.stream_clock},
      {"image_filter", opts.image_filter},
      {"turn_signals", opts.turn_signals},
      {"source_ids", opts.speaker_sources ? "speaker" : "conversation"},
      {"conversation_end", opts.close_conversations ? "boundary" : "none"},
      {"retention", opts.retention == cortext::Retention::Natural
                        ? "natural"
                        : "durable"},
      {"query_ms", Percentiles (query_ms_all)},
      {"media_bytes_ingested", media_bytes_by_modality},
      {"filtered_signals", filtered_by_modality},
  };
  nlohmann::json ingest_ms = nlohmann::json::object ();
  for (const auto &kv : ingest_ms_by_modality)
    {
      ingest_ms[kv.first] = {{"wall", Percentiles (kv.second)},
                             {"encode", Percentiles (encode_ms_by_modality[kv.first])}};
    }
  summary["ingest_ms"] = ingest_ms;

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
