#include <cortext/cortext.hpp>
#include <cortext/core/knobs.hpp>

#include "encoder/text_encoder_factory.hpp"
#include "streaming_text_probe.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

struct Turn
{
  std::string role;
  std::string text;
};

struct Checkpoint
{
  std::size_t call_index = 0;
  std::size_t accumulated_chars = 0;
  double wall_ms = 0.0;
  double encode_ms = 0.0;
  double process_ms = 0.0;
  double hydrate_ms = 0.0;
  double total_ms = 0.0;
};

struct PhaseStats
{
  int calls = 0;
  int stored = 0;
  int should_interrupt = 0;
  int at_boundary = 0;
  int interrupt_aborted = 0;
  std::size_t retrieved_memories = 0;
  std::size_t working_memory_slots = 0;
  double wall_ms = 0.0;
  double encode_ms = 0.0;
  double process_ms = 0.0;
  double hydrate_ms = 0.0;
  double total_ms = 0.0;
  std::unordered_map<std::string, double> operation_ms;

  void
  AddSample (const cortext::Cortext::Context &ctx, double wall_elapsed_ms)
  {
    ++calls;
    if (ctx.output.stored_embedding_id.has_value ())
      {
        ++stored;
      }
    if (ctx.should_interrupt)
      {
        ++should_interrupt;
      }
    if (ctx.at_boundary)
      {
        ++at_boundary;
      }
    if (ctx.interrupt_aborted)
      {
        ++interrupt_aborted;
      }
    retrieved_memories += ctx.retrieved_memory.size ();
    working_memory_slots += ctx.working_memory.size ();
    wall_ms += wall_elapsed_ms;
    encode_ms += ctx.encode_ms;
    process_ms += ctx.process_ms;
    hydrate_ms += ctx.hydrate_ms;
    total_ms += ctx.total_ms;
    for (const auto &[name, value] : ctx.output.operation_ms)
      {
        operation_ms[name] += value;
      }
  }
};

struct Options
{
  std::filesystem::path db_path;
  std::filesystem::path models_dir = "models";
  std::size_t chunk_words = 1;
  int top_ops = 12;
  bool keep_db = false;
  double focus = 0.5;
  double sensitivity = 0.5;
  double stability = 0.5;
};

constexpr std::size_t kStreamingProbeMinChars = 32;
constexpr std::size_t kStreamingProbeMaxChars = 128;

std::string
GetEnv (const char *name, const std::string &fallback = {})
{
  const char *value = std::getenv (name);
  if (!value)
    {
      return fallback;
    }
  return value;
}

double
GetEnvDouble (const char *name, double fallback)
{
  const std::string value = GetEnv (name);
  if (value.empty ())
    {
      return fallback;
    }
  try
    {
      return std::stod (value);
    }
  catch (...)
    {
      return fallback;
    }
}

std::optional<std::filesystem::path>
FindRepoRootFromExe (const char *argv0)
{
  namespace fs = std::filesystem;

  if (!argv0 || std::string (argv0).empty ())
    {
      return std::nullopt;
    }

  std::error_code ec;
  fs::path exe_path = fs::path (argv0);
  if (exe_path.is_relative ())
    {
      exe_path = fs::absolute (exe_path, ec);
    }
  if (ec)
    {
      return std::nullopt;
    }

  fs::path dir = exe_path.parent_path ();
  for (int i = 0; i < 12; ++i)
    {
      if (dir.empty ())
        {
          break;
        }
      if (fs::exists (dir / "CMakeLists.txt", ec)
          && fs::exists (dir / "models", ec))
        {
          return dir;
        }
      auto parent = dir.parent_path ();
      if (parent == dir)
        {
          break;
        }
      dir = parent;
    }

  return std::nullopt;
}

std::uint64_t
NowUnixMillis ()
{
  using namespace std::chrono;
  return static_cast<std::uint64_t> (
      duration_cast<milliseconds> (system_clock::now ().time_since_epoch ())
          .count ());
}

void
RemoveDbArtifacts (const std::filesystem::path &db_path)
{
  std::error_code ec;
  std::filesystem::remove (db_path, ec);
  std::filesystem::remove (db_path.string () + "-wal", ec);
  std::filesystem::remove (db_path.string () + "-shm", ec);
}

std::vector<Turn>
SeedTurns ()
{
  return {
      {
          "user",
          "I'm taking a cheap weekend trip to Chicago next month. I like "
          "walkable neighborhoods, independent bookstores, good coffee, "
          "and one nice dinner, but I want to keep most of the day low-cost."
      },
      {
          "assistant",
          "Logan Square and Wicker Park fit that well. You can use the Blue "
          "Line to stay mobile, spend the morning in cafes and bookstores, "
          "walk the boulevard, and save your budget for one dinner "
          "reservation instead of paying for taxis all day."
      },
      {
          "user",
          "That sounds right. I'm staying near Logan Square, I don't want a "
          "museum that eats the whole day, and I want an indoor backup in "
          "case it rains."
      },
      {
          "assistant",
          "For architecture without a full museum day, use the riverwalk or "
          "the Chicago Architecture Center for a short stop. If rain starts, "
          "switch to a bookstore crawl, a long coffee break, and a compact "
          "indoor site you can leave quickly."
      },
      {
          "user",
          "I'm also trying to keep transit simple. Ideally I do one main "
          "train ride, lots of walking, and a dinner spot that feels special "
          "but not jacket-and-tie expensive."
      },
      {
          "assistant",
          "Then build the day around a Blue Line corridor. Start in Logan "
          "Square, ride once toward downtown for architecture and the "
          "riverwalk, then work your way back northwest. For dinner, pick a "
          "place where the food feels like the splurge rather than the whole "
          "experience being formal."
      }};
}

std::string
ProfileUserPrompt ()
{
  return "Can you make me a Saturday plan that starts near Logan Square, "
         "includes coffee, architecture, a bookstore, and one splurge "
         "dinner, keeps transit simple, and still has a rainy-day backup?";
}

std::string
ProfileAssistantReply ()
{
  return "Start with coffee in Logan Square around 8:30 so you can keep the "
         "morning cheap and local. Walk the boulevard for a bit, then take "
         "the Blue Line once toward downtown and get off near the river for "
         "an architecture-focused walk instead of committing to a long museum "
         "visit. Spend late morning on the riverwalk, pause for a quick lunch "
         "you can grab casually, and keep the middle of the day flexible.\n\n"
         "In the afternoon, head back northwest and give yourself time for an "
         "independent bookstore stop plus a second coffee or pastry break. "
         "That keeps the day walkable, layered, and inexpensive before "
         "dinner. For the splurge, book one neighborhood restaurant where the "
         "food is the treat but the atmosphere is still relaxed rather than "
         "formal.\n\n"
         "If rain starts early, flip the order: do the bookstore and cafe "
         "time first, use a shorter indoor architecture stop later, and keep "
         "the same dinner plan. That way the structure of the day stays the "
         "same, but you are not burning time or money fighting the weather.";
}

std::vector<std::string>
BuildStreamingChunks (const std::string &text, std::size_t chunk_words)
{
  std::vector<std::string> chunks;
  if (text.empty ())
    {
      return chunks;
    }

  if (chunk_words == 0)
    {
      chunk_words = 1;
    }

  std::istringstream input (text);
  std::string word;
  std::string chunk;
  std::size_t words_in_chunk = 0;

  while (input >> word)
    {
      if (!chunk.empty ())
        {
          chunk.push_back (' ');
        }
      chunk += word;
      ++words_in_chunk;
      if (words_in_chunk >= chunk_words)
        {
          chunks.push_back (chunk);
          chunk.clear ();
          words_in_chunk = 0;
        }
    }

  if (!chunk.empty ())
    {
      chunks.push_back (chunk);
    }

  if (chunks.empty ())
    {
      chunks.push_back (text);
    }

  return chunks;
}

std::size_t
TrimmedProbeLength (const std::string &text)
{
  std::size_t end = text.size ();
  while (end > 0
         && std::isspace (static_cast<unsigned char> (text[end - 1]))
         && text[end - 1] != '\n')
    {
      --end;
    }
  return end;
}

bool
HasProbeBoundary (const std::string &text)
{
  if (text.empty ())
    {
      return false;
    }
  if (text.back () == '\n')
    {
      return true;
    }

  std::size_t idx = text.size ();
  while (idx > 0
         && std::isspace (static_cast<unsigned char> (text[idx - 1])))
    {
      --idx;
    }
  if (idx == 0)
    {
      return false;
    }

  switch (text[idx - 1])
    {
    case '.':
    case '!':
    case '?':
    case ':':
    case ';':
      return true;
    default:
      return false;
    }
}

bool
ShouldRunStreamingProbe (const std::string &text, bool force_flush = false)
{
  const std::size_t trimmed = TrimmedProbeLength (text);
  if (trimmed == 0)
    {
      return false;
    }
  if (force_flush)
    {
      return true;
    }
  if (trimmed >= kStreamingProbeMaxChars)
    {
      return true;
    }
  if (trimmed < kStreamingProbeMinChars)
    {
      return false;
    }
  return HasProbeBoundary (text);
}

std::vector<std::size_t>
BuildCheckpointIndices (std::size_t total_calls)
{
  std::vector<std::size_t> indices;
  if (total_calls == 0)
    {
      return indices;
    }

  indices.push_back (1);
  indices.push_back (std::max<std::size_t> (1, total_calls / 4));
  indices.push_back (std::max<std::size_t> (1, total_calls / 2));
  indices.push_back (std::max<std::size_t> (1, (3 * total_calls) / 4));
  indices.push_back (total_calls);

  std::sort (indices.begin (), indices.end ());
  indices.erase (std::unique (indices.begin (), indices.end ()),
                 indices.end ());
  return indices;
}

double
ElapsedMillis (const std::chrono::steady_clock::time_point &start,
               const std::chrono::steady_clock::time_point &end)
{
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>> (
             end - start)
      .count ();
}

void
PrintTopOperations (const PhaseStats &stats, int top_ops)
{
  std::vector<std::pair<std::string, double>> entries (stats.operation_ms.begin (),
                                                       stats.operation_ms.end ());
  std::sort (entries.begin (), entries.end (),
             [] (const auto &a, const auto &b) {
               if (a.second == b.second)
                 {
                   return a.first < b.first;
                 }
               return a.second > b.second;
             });

  const int limit = std::min<int> (top_ops, static_cast<int> (entries.size ()));
  for (int i = 0; i < limit; ++i)
    {
      std::cout << "    " << std::setw (2) << (i + 1) << ". "
                << entries[static_cast<std::size_t> (i)].first << ": "
                << std::fixed << std::setprecision (2)
                << entries[static_cast<std::size_t> (i)].second << " ms\n";
    }
}

void
PrintPhase (const std::string &name, const PhaseStats &stats, int top_ops)
{
  const double avg_wall = stats.calls == 0 ? 0.0 : stats.wall_ms / stats.calls;
  const double avg_total = stats.calls == 0 ? 0.0 : stats.total_ms / stats.calls;
  const double avg_encode = stats.calls == 0 ? 0.0 : stats.encode_ms / stats.calls;
  const double avg_process = stats.calls == 0 ? 0.0 : stats.process_ms / stats.calls;
  const double avg_hydrate = stats.calls == 0 ? 0.0 : stats.hydrate_ms / stats.calls;
  const double component_total
      = stats.total_ms > 0.0 ? stats.total_ms : 1.0;

  std::cout << "\n[" << name << "]\n";
  std::cout << "  calls=" << stats.calls << ", wall_ms=" << std::fixed
            << std::setprecision (2) << stats.wall_ms << ", avg_wall_ms="
            << avg_wall << "\n";
  std::cout << "  ctx_total_ms=" << stats.total_ms << ", avg_ctx_total_ms="
            << avg_total << "\n";
  std::cout << "  encode_ms=" << stats.encode_ms << " ("
            << (100.0 * stats.encode_ms / component_total)
            << "%), process_ms=" << stats.process_ms << " ("
            << (100.0 * stats.process_ms / component_total)
            << "%), hydrate_ms=" << stats.hydrate_ms << " ("
            << (100.0 * stats.hydrate_ms / component_total) << "%)\n";
  std::cout << "  avg_encode_ms=" << avg_encode << ", avg_process_ms="
            << avg_process << ", avg_hydrate_ms=" << avg_hydrate << "\n";
  std::cout << "  stored=" << stats.stored
            << ", should_interrupt=" << stats.should_interrupt
            << ", at_boundary=" << stats.at_boundary
            << ", interrupt_aborted=" << stats.interrupt_aborted << "\n";
  std::cout << "  retrieved_memories=" << stats.retrieved_memories
            << ", working_memory_slots=" << stats.working_memory_slots << "\n";
  if (!stats.operation_ms.empty ())
    {
      std::cout << "  top_operations:\n";
      PrintTopOperations (stats, top_ops);
    }
}

void
PrintStreamingCheckpoints (const std::vector<Checkpoint> &checkpoints)
{
  if (checkpoints.empty ())
    {
      return;
    }

  std::cout << "\n[streaming_checkpoints]\n";
  for (const auto &checkpoint : checkpoints)
    {
      std::cout << "  call=" << checkpoint.call_index
                << ", chars=" << checkpoint.accumulated_chars
                << ", wall_ms=" << std::fixed << std::setprecision (2)
                << checkpoint.wall_ms << ", total_ms=" << checkpoint.total_ms
                << ", encode_ms=" << checkpoint.encode_ms
                << ", process_ms=" << checkpoint.process_ms
                << ", hydrate_ms=" << checkpoint.hydrate_ms << "\n";
    }
}

void
PrintUsage ()
{
  std::cout
      << "Usage: cortext_chat_profile [--db=PATH] [--models-dir=PATH]\n"
      << "                            [--chunk-words=N] [--top-ops=N]\n"
      << "                            [--keep-db]\n";
}

Options
ParseArgs (int argc, char **argv)
{
  Options options;
  options.focus = GetEnvDouble ("CORTEXT_FOCUS", 0.5);
  options.sensitivity = GetEnvDouble ("CORTEXT_SENSITIVITY", 0.5);
  options.stability = GetEnvDouble ("CORTEXT_STABILITY", 0.5);

  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h")
        {
          PrintUsage ();
          std::exit (0);
        }
      if (arg == "--keep-db")
        {
          options.keep_db = true;
          continue;
        }
      auto take_value = [&arg] (const char *prefix)
          -> std::optional<std::string> {
        const std::string prefix_string = prefix;
        if (arg.rfind (prefix_string, 0) == 0)
          {
            return arg.substr (prefix_string.size ());
          }
        return std::nullopt;
      };

      if (auto value = take_value ("--db="))
        {
          options.db_path = *value;
          continue;
        }
      if (auto value = take_value ("--models-dir="))
        {
          options.models_dir = *value;
          continue;
        }
      if (auto value = take_value ("--chunk-words="))
        {
          options.chunk_words = static_cast<std::size_t> (std::stoul (*value));
          continue;
        }
      if (auto value = take_value ("--top-ops="))
        {
          options.top_ops = std::stoi (*value);
          continue;
        }

      throw std::runtime_error ("unknown argument: " + arg);
    }

  return options;
}

} // namespace

int
main (int argc, char **argv)
{
  try
    {
      Options options = ParseArgs (argc, argv);
      const auto repo_root
          = FindRepoRootFromExe ((argc > 0) ? argv[0] : nullptr);

      if (options.models_dir.is_relative () && repo_root.has_value ())
        {
          options.models_dir = *repo_root / options.models_dir;
        }
      if (options.db_path.empty ())
        {
          options.db_path = std::filesystem::temp_directory_path ()
                            / ("cortext_chat_profile_"
                               + std::to_string (NowUnixMillis ()) + ".db");
        }
      else if (options.db_path.is_relative () && repo_root.has_value ())
        {
          options.db_path = *repo_root / options.db_path;
        }

      std::error_code ec;
      const auto db_parent = options.db_path.parent_path ();
      if (!db_parent.empty ())
        {
          std::filesystem::create_directories (db_parent, ec);
        }
      RemoveDbArtifacts (options.db_path);

      cortext::Cortext::Config cfg;
      cfg.focus = options.focus;
      cfg.sensitivity = options.sensitivity;
      cfg.stability = options.stability;

      const auto create_start = std::chrono::steady_clock::now ();
      std::unique_ptr<cortext::Cortext> cortext_ctx
          = cortext::Cortext::Create (cfg, options.db_path.string (),
                                      options.models_dir.string ());
      const auto create_end = std::chrono::steady_clock::now ();
      if (!cortext_ctx)
        {
          throw std::runtime_error ("failed to create cortext instance");
        }

      const auto seed_turns = SeedTurns ();
      double seed_wall_ms = 0.0;
      for (const auto &turn : seed_turns)
        {
          const std::string source
              = turn.role == "assistant" ? "chat/assistant" : "chat/user";
          const auto start = std::chrono::steady_clock::now ();
          (void) cortext_ctx->ProcessText (turn.text, source);
          const auto end = std::chrono::steady_clock::now ();
          seed_wall_ms += ElapsedMillis (start, end);
        }
      const std::string user_prompt = ProfileUserPrompt ();
      const std::string assistant_reply = ProfileAssistantReply ();
      const auto chunks
          = BuildStreamingChunks (assistant_reply, options.chunk_words);
      const auto checkpoint_indices = BuildCheckpointIndices (chunks.size ());

      PhaseStats user_phase;
      PhaseStats stream_phase;
      PhaseStats final_phase;
      std::vector<Checkpoint> checkpoints;

      const auto user_start = std::chrono::steady_clock::now ();
      const auto user_ctx = cortext_ctx->ProcessText (user_prompt, "chat/user");
      const auto user_end = std::chrono::steady_clock::now ();
      user_phase.AddSample (user_ctx, ElapsedMillis (user_start, user_end));

      cortext::internal::StreamingTextProbeSession probe_session(
          *cortext_ctx, "chat/assistant");
      std::string accumulated;
      std::string probe_buffer;
      for (std::size_t i = 0; i < chunks.size (); ++i)
        {
          if (!accumulated.empty ())
            {
              accumulated.push_back (' ');
            }
          accumulated += chunks[i];
          if (!probe_buffer.empty ())
            {
              probe_buffer.push_back (' ');
            }
          probe_buffer += chunks[i];

          if (!ShouldRunStreamingProbe (probe_buffer))
            {
              continue;
            }

          const auto call_start = std::chrono::steady_clock::now ();
          const auto ctx = probe_session.AppendTextChunk (probe_buffer);
          const auto call_end = std::chrono::steady_clock::now ();
          const double wall_ms = ElapsedMillis (call_start, call_end);
          stream_phase.AddSample (ctx, wall_ms);
          probe_buffer.clear ();

          if (std::binary_search (checkpoint_indices.begin (),
                                  checkpoint_indices.end (), i + 1))
            {
              checkpoints.push_back (
                  {i + 1, accumulated.size (), wall_ms, ctx.encode_ms,
                   ctx.process_ms, ctx.hydrate_ms, ctx.total_ms});
            }
        }

      if (!probe_buffer.empty ())
        {
          const auto call_start = std::chrono::steady_clock::now ();
          const auto ctx = probe_session.CacheTextChunk (probe_buffer);
          const auto call_end = std::chrono::steady_clock::now ();
          const double wall_ms = ElapsedMillis (call_start, call_end);
          stream_phase.AddSample (ctx, wall_ms);
        }

      const auto final_start = std::chrono::steady_clock::now ();
      const auto final_ctx
          = probe_session.FinalizeText (assistant_reply);
      const auto final_end = std::chrono::steady_clock::now ();
      final_phase.AddSample (final_ctx, ElapsedMillis (final_start, final_end));

      const double create_ms = ElapsedMillis (create_start, create_end);
      const double local_chat_wall_ms
          = user_phase.wall_ms + stream_phase.wall_ms + final_phase.wall_ms;
      std::string resolved_backend = "unknown";
      try
        {
          auto resolved
              = cortext::internal::CreatePreferredTextEncoder (
                  options.models_dir.string ());
          resolved_backend = resolved.backend_name;
        }
      catch (...)
        {
        }

      std::cout << std::fixed << std::setprecision (2);
      std::cout << "cortext_chat_profile\n";
      std::cout << "  db_path=" << options.db_path << "\n";
      std::cout << "  models_dir=" << options.models_dir << "\n";
      std::cout << "  resolved_text_encoder=" << resolved_backend << "\n";
      std::cout << "  focus=" << options.focus
                << ", sensitivity=" << options.sensitivity
                << ", stability=" << options.stability << "\n";
      std::cout << "  startup_create_ms=" << create_ms << "\n";
      std::cout << "  seed_turns=" << seed_turns.size ()
                << ", seed_wall_ms=" << seed_wall_ms << "\n";
      std::cout << "  profile_user_chars=" << user_prompt.size ()
                << ", profile_assistant_chars=" << assistant_reply.size ()
                << ", streaming_checks=" << chunks.size ()
                << ", chunk_words=" << options.chunk_words << "\n";
      std::cout << "  local_chat_wall_ms=" << local_chat_wall_ms << "\n";

      PrintPhase ("phase1_user", user_phase, options.top_ops);
      PrintPhase ("phase2_streaming_interrupt_checks", stream_phase,
                  options.top_ops);
      PrintStreamingCheckpoints (checkpoints);
      PrintPhase ("phase3_final_assistant", final_phase, options.top_ops);

      const double total_phase_ctx_ms
          = user_phase.total_ms + stream_phase.total_ms + final_phase.total_ms;
      const double safe_total = total_phase_ctx_ms > 0.0 ? total_phase_ctx_ms : 1.0;
      std::cout << "\n[phase_share]\n";
      std::cout << "  phase1_user_ctx_ms=" << user_phase.total_ms << " ("
                << (100.0 * user_phase.total_ms / safe_total) << "%)\n";
      std::cout << "  phase2_stream_ctx_ms=" << stream_phase.total_ms << " ("
                << (100.0 * stream_phase.total_ms / safe_total) << "%)\n";
      std::cout << "  phase3_final_ctx_ms=" << final_phase.total_ms << " ("
                << (100.0 * final_phase.total_ms / safe_total) << "%)\n";
      if (!options.keep_db)
        {
          cortext_ctx.reset ();
          RemoveDbArtifacts (options.db_path);
        }

      return 0;
    }
  catch (const std::exception &ex)
    {
      std::cerr << "cortext_chat_profile failed: " << ex.what () << "\n";
      return 1;
    }
}
