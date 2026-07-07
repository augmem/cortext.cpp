// cortext_cli: command-line surface over the public Cortext API.
//
// Subcommands:
//   remember <text...>   store one durable memory (or one per stdin line with "-")
//   recall <query...>    retrieve context without storing (Retention::Ephemeral)
//   consolidate          run explicit shallow consolidation
//   repl                 interactive session (default when no subcommand)
//
// Uses only public headers so it doubles as an API smoke test.

#include <cortext/cortext.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{

struct CliOptions
{
  std::string db_path = "cortext_memory.db";
  std::string models_dir = "models";
  std::string source_id = "cli/main";
  bool source_id_overridden = false;
  double focus = 0.7;
  double sensitivity = 0.5;
  double stability = 0.8;
  std::size_t top = 5;
  bool quiet = false;
  bool durable_recall = false;
  std::string command;
  std::vector<std::string> command_args;
};

void PrintUsage (std::ostream &out)
{
  out << "cortext_cli - memory you can talk to\n"
      << "\n"
      << "Usage:\n"
      << "  cortext_cli [options] remember <text...>   store one durable memory\n"
      << "  cortext_cli [options] remember -           store one memory per stdin line\n"
      << "  cortext_cli [options] recall <query...>    retrieve context without storing\n"
      << "  cortext_cli [options] consolidate          run explicit shallow consolidation\n"
      << "  cortext_cli [options] repl                 interactive session (default)\n"
      << "\n"
      << "Options:\n"
      << "  --db PATH          SQLite database path (default: cortext_memory.db,\n"
      << "                     or CORTEXT_CLI_DB)\n"
      << "  --models DIR       models directory (default: models, or CORTEXT_MODELS_DIR)\n"
      << "  --source ID        provenance stream id (default: cli/main)\n"
      << "  --focus X          F knob in [0,1] (default 0.7)\n"
      << "  --sensitivity X    S knob in [0,1] (default 0.5)\n"
      << "  --stability X      T knob in [0,1] (default 0.8)\n"
      << "  --top N            max retrieved memories to print (default 5)\n"
      << "  --quiet            suppress per-call gauges\n"
      << "  --durable          store recall queries as durable memories under\n"
      << "                     cli/recall (default: recall is ephemeral and\n"
      << "                     stores nothing)\n"
      << "  -h, --help         show this help\n"
      << "\n"
      << "Repl commands: /recall <query>, /consolidate, /stats, /quit\n";
}

std::string JoinArgs (const std::vector<std::string> &args)
{
  std::ostringstream out;
  for (std::size_t i = 0; i < args.size (); ++i)
    {
      if (i > 0)
        {
          out << ' ';
        }
      out << args[i];
    }
  return out.str ();
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

std::string ClipForDisplay (const std::string &text, std::size_t max_chars)
{
  if (text.size () <= max_chars)
    {
      return text;
    }
  return text.substr (0, max_chars) + "...";
}

std::string FormatTimestamp (std::uint64_t timestamp_ms)
{
  if (timestamp_ms == 0)
    {
      return "unknown-time";
    }
  const std::time_t seconds
      = static_cast<std::time_t> (timestamp_ms / 1000ULL);
  std::tm parts{};
#if defined(_WIN32)
  localtime_s (&parts, &seconds);
#else
  localtime_r (&seconds, &parts);
#endif
  char buffer[32];
  if (std::strftime (buffer, sizeof (buffer), "%Y-%m-%d %H:%M", &parts) == 0)
    {
      return "unknown-time";
    }
  return buffer;
}

void PrintRetrieved (const cortext::Cortext::Context &context,
                     std::size_t top)
{
  if (context.retrieved_memory.empty ())
    {
      std::cout << "  (no long-term memories surfaced)\n";
      return;
    }
  const std::size_t shown = std::min (top, context.retrieved_memory.size ());
  for (std::size_t i = 0; i < shown; ++i)
    {
      const auto &memory = context.retrieved_memory[i];
      std::cout << "  " << (i + 1) << ". [#" << memory.id << " · "
                << memory.source_id << " · "
                << FormatTimestamp (memory.timestamp) << " · rel "
                << std::fixed << std::setprecision (2) << memory.relevance
                << " · sal " << memory.salience << "] "
                << ClipForDisplay (MemoryText (memory), 160) << "\n";
    }
  if (context.retrieved_memory.size () > shown)
    {
      std::cout << "  ... " << (context.retrieved_memory.size () - shown)
                << " more retrieved\n";
    }
}

void PrintGauges (const cortext::Cortext::Context &context, bool stored)
{
  std::cout << "  ·";
  if (stored && context.output.stored_memory_id.has_value ())
    {
      std::cout << " stored #" << *context.output.stored_memory_id << " ·";
    }
  else if (stored)
    {
      std::cout << " not stored ·";
    }
  std::cout << " retrieved " << context.retrieved_memory.size () << " · wm "
            << context.working_memory.size () << " · focus " << std::fixed
            << std::setprecision (2) << context.output.effective_focus
            << " · coherence " << context.output.coherence << " · "
            << std::setprecision (0) << context.total_ms << " ms\n"
            << std::setprecision (2);
  if (context.should_interrupt)
    {
      std::cout << "  ! interrupt recommended: surfaced context is likely "
                   "worth acting on now\n";
    }
  if (context.consolidation_recommended || context.consolidation_required)
    {
      std::cout << "  ~ consolidation "
                << (context.consolidation_required ? "required" : "recommended")
                << " (run: consolidate)\n";
    }
}

int RunRemember (cortext::Cortext &engine, const CliOptions &options)
{
  if (options.command_args.size () == 1 && options.command_args[0] == "-")
    {
      std::size_t stored = 0;
      std::size_t lines = 0;
      std::string line;
      while (std::getline (std::cin, line))
        {
          if (line.empty ())
            {
              continue;
            }
          ++lines;
          auto context = engine.ProcessText (line, options.source_id,
                                             cortext::Retention::Durable);
          if (context.output.stored_memory_id.has_value ())
            {
              ++stored;
            }
        }
      engine.Flush ();
      std::cout << "ingested " << lines << " lines, " << stored
                << " stored as durable memories\n";
      return 0;
    }

  const std::string text = JoinArgs (options.command_args);
  if (text.empty ())
    {
      std::cerr << "remember: no text given\n";
      return 1;
    }
  auto context = engine.ProcessText (text, options.source_id,
                                     cortext::Retention::Durable);
  engine.Flush ();
  if (!context.retrieved_memory.empty ())
    {
      std::cout << "related context:\n";
      PrintRetrieved (context, options.top);
    }
  if (!options.quiet)
    {
      PrintGauges (context, /*stored=*/true);
    }
  return 0;
}

// Recall is ephemeral by default: the query triggers retrieval but is not
// stored, so queries never compete with real memories. --durable opts into
// treating the query itself as a stream event under a dedicated source.
cortext::Cortext::Context RecallOnce (cortext::Cortext &engine,
                                      const CliOptions &options,
                                      const std::string &query)
{
  if (!options.durable_recall)
    {
      return engine.ProcessText (query, options.source_id,
                                 cortext::Retention::Ephemeral);
    }
  const std::string source
      = options.source_id_overridden ? options.source_id : "cli/recall";
  return engine.ProcessText (query, source, cortext::Retention::Durable);
}

int RunRecall (cortext::Cortext &engine, const CliOptions &options)
{
  const std::string query = JoinArgs (options.command_args);
  if (query.empty ())
    {
      std::cerr << "recall: no query given\n";
      return 1;
    }
  auto context = RecallOnce (engine, options, query);
  engine.Flush ();
  PrintRetrieved (context, options.top);
  if (!options.quiet)
    {
      PrintGauges (context, /*stored=*/false);
    }
  return 0;
}

int RunConsolidate (cortext::Cortext &engine, const CliOptions &options)
{
  auto context = engine.Consolidate ();
  engine.Flush ();
  std::cout << "consolidation pass finished in " << std::fixed
            << std::setprecision (0) << context.total_ms << " ms\n"
            << std::setprecision (2);
  if (!options.quiet)
    {
      std::cout << "  · wm " << context.working_memory.size ()
                << " · further consolidation "
                << (context.consolidation_recommended ? "recommended" : "not needed")
                << "\n";
    }
  return 0;
}

int RunRepl (cortext::Cortext &engine, const CliOptions &options)
{
  std::cout << "cortext repl · db=" << options.db_path << " · source="
            << options.source_id << "\n"
            << "type to remember; /recall <query>, /consolidate, /stats, /quit\n";
  std::string line;
  cortext::Cortext::Context last_context;
  while (true)
    {
      std::cout << "> " << std::flush;
      if (!std::getline (std::cin, line))
        {
          std::cout << "\n";
          break;
        }
      if (line.empty ())
        {
          continue;
        }
      if (line == "/quit" || line == "/exit")
        {
          break;
        }
      if (line == "/consolidate")
        {
          last_context = engine.Consolidate ();
          std::cout << "  consolidated in " << std::fixed
                    << std::setprecision (0) << last_context.total_ms
                    << " ms\n"
                    << std::setprecision (2);
          continue;
        }
      if (line == "/stats")
        {
          PrintGauges (last_context, /*stored=*/false);
          continue;
        }
      if (line.rfind ("/recall ", 0) == 0)
        {
          const std::string query = line.substr (8);
          last_context = RecallOnce (engine, options, query);
          PrintRetrieved (last_context, options.top);
          if (!options.quiet)
            {
              PrintGauges (last_context, /*stored=*/false);
            }
          continue;
        }
      if (!line.empty () && line[0] == '/')
        {
          std::cout << "  unknown command: " << line << "\n";
          continue;
        }
      last_context = engine.ProcessText (line, options.source_id,
                                         cortext::Retention::Durable);
      if (!last_context.retrieved_memory.empty ())
        {
          PrintRetrieved (last_context, options.top);
        }
      if (!options.quiet)
        {
          PrintGauges (last_context, /*stored=*/true);
        }
    }
  engine.Flush ();
  return 0;
}

bool ParseDouble (const std::string &value, double &out)
{
  try
    {
      std::size_t consumed = 0;
      out = std::stod (value, &consumed);
      return consumed == value.size ();
    }
  catch (const std::exception &)
    {
      return false;
    }
}

int ParseOptions (int argc, char **argv, CliOptions &options)
{
  if (const char *env_db = std::getenv ("CORTEXT_CLI_DB"))
    {
      options.db_path = env_db;
    }
  if (const char *env_models = std::getenv ("CORTEXT_MODELS_DIR"))
    {
      options.models_dir = env_models;
    }

  std::vector<std::string> args (argv + 1, argv + argc);
  std::size_t i = 0;
  auto take_value = [&] (const std::string &flag) -> const std::string * {
    if (i + 1 >= args.size ())
      {
        std::cerr << flag << " requires a value\n";
        return nullptr;
      }
    return &args[++i];
  };

  for (; i < args.size (); ++i)
    {
      const std::string &arg = args[i];
      if (arg == "-h" || arg == "--help")
        {
          PrintUsage (std::cout);
          return -1;
        }
      if (arg == "--quiet")
        {
          options.quiet = true;
          continue;
        }
      if (arg == "--durable")
        {
          options.durable_recall = true;
          continue;
        }
      if (arg == "--db" || arg == "--models" || arg == "--source"
          || arg == "--focus" || arg == "--sensitivity" || arg == "--stability"
          || arg == "--top")
        {
          const std::string *value = take_value (arg);
          if (value == nullptr)
            {
              return 1;
            }
          if (arg == "--db")
            {
              options.db_path = *value;
            }
          else if (arg == "--models")
            {
              options.models_dir = *value;
            }
          else if (arg == "--source")
            {
              options.source_id = *value;
              options.source_id_overridden = true;
            }
          else if (arg == "--top")
            {
              options.top = static_cast<std::size_t> (
                  std::max (1L, std::atol (value->c_str ())));
            }
          else
            {
              double knob = 0.0;
              if (!ParseDouble (*value, knob) || knob < 0.0 || knob > 1.0)
                {
                  std::cerr << arg << " expects a number in [0,1]\n";
                  return 1;
                }
              if (arg == "--focus")
                {
                  options.focus = knob;
                }
              else if (arg == "--sensitivity")
                {
                  options.sensitivity = knob;
                }
              else
                {
                  options.stability = knob;
                }
            }
          continue;
        }
      if (!arg.empty () && arg[0] == '-' && arg != "-")
        {
          std::cerr << "unknown option: " << arg << "\n";
          PrintUsage (std::cerr);
          return 1;
        }
      if (options.command.empty ())
        {
          options.command = arg;
        }
      else
        {
          options.command_args.push_back (arg);
        }
    }

  if (options.command.empty ())
    {
      options.command = "repl";
    }
  return 0;
}

} // namespace

int main (int argc, char **argv)
{
  CliOptions options;
  const int parse_status = ParseOptions (argc, argv, options);
  if (parse_status != 0)
    {
      return parse_status < 0 ? 0 : parse_status;
    }

  if (options.command != "remember" && options.command != "recall"
      && options.command != "consolidate" && options.command != "repl")
    {
      std::cerr << "unknown command: " << options.command << "\n";
      PrintUsage (std::cerr);
      return 1;
    }

  cortext::Cortext::Config config;
  config.focus = options.focus;
  config.sensitivity = options.sensitivity;
  config.stability = options.stability;

  std::unique_ptr<cortext::Cortext> engine;
  try
    {
      engine = cortext::Cortext::Create (config, options.db_path,
                                         options.models_dir);
    }
  catch (const std::exception &error)
    {
      std::cerr << "failed to open cortext engine: " << error.what () << "\n"
                << "hint: pass --models <dir> containing AIST-87M-GGUF, or "
                   "set CORTEXT_AIST_MODEL_PATH\n";
      return 1;
    }

  try
    {
      if (options.command == "remember")
        {
          return RunRemember (*engine, options);
        }
      if (options.command == "recall")
        {
          return RunRecall (*engine, options);
        }
      if (options.command == "consolidate")
        {
          return RunConsolidate (*engine, options);
        }
      return RunRepl (*engine, options);
    }
  catch (const std::exception &error)
    {
      std::cerr << "cortext_cli error: " << error.what () << "\n";
      return 1;
    }
}
