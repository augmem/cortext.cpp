#include "../../src/deep_llm/deep_llm_factory.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

struct Options
{
  std::filesystem::path models_dir = "models";
  std::string backend = "all";
  int iterations = 5;
  int warmup = 1;
  int max_summary_words = 80;
};

struct SummaryFixture
{
  std::string name;
  std::vector<std::string> texts;
};

struct ExtractionFixture
{
  std::string name;
  std::string text;
};

struct BenchmarkStats
{
  double median_ms = 0.0;
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  size_t samples = 0;
};

struct BackendReport
{
  std::string name;
  std::filesystem::path summarizer_model_path;
  std::filesystem::path extractor_model_path;
  BenchmarkStats summarization;
  BenchmarkStats extraction;
  BenchmarkStats combined;
};

void
PrintUsage ()
{
  std::cout
      << "Usage: cortext_deep_llm_bench [options]\n"
      << "  --models-dir <path>         Model root (default: models)\n"
      << "  --backend all|gemma|lfm2|mixed    Deep backend selection (default: all)\n"
      << "  --iterations <n>            Timed iterations per fixture (default: 5)\n"
      << "  --warmup <n>                Warmup iterations per fixture (default: 1)\n"
      << "  --max-summary-words <n>     Word cap for summary runs (default: 80)\n"
      << "  --help                      Show this help\n";
}

Options
ParseArgs (int argc, char *argv[])
{
  Options opts;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if (arg == "--models-dir" && i + 1 < argc)
        {
          opts.models_dir = argv[++i];
        }
      else if (arg == "--backend" && i + 1 < argc)
        {
          opts.backend = argv[++i];
        }
      else if (arg == "--iterations" && i + 1 < argc)
        {
          opts.iterations = std::stoi (argv[++i]);
        }
      else if (arg == "--warmup" && i + 1 < argc)
        {
          opts.warmup = std::stoi (argv[++i]);
        }
      else if (arg == "--max-summary-words" && i + 1 < argc)
        {
          opts.max_summary_words = std::stoi (argv[++i]);
        }
      else if (arg == "--help" || arg == "-h")
        {
          PrintUsage ();
          std::exit (0);
        }
      else
        {
          throw std::runtime_error ("Unknown argument: " + arg);
        }
    }

  if (opts.iterations <= 0)
    {
      throw std::runtime_error ("--iterations must be > 0");
    }
  if (opts.warmup < 0)
    {
      throw std::runtime_error ("--warmup must be >= 0");
    }
  return opts;
}

nlohmann::json
BuildExtractionSchema ()
{
  return nlohmann::json::parse (R"({
    "type": "object",
    "properties": {
      "labels": {
        "type": "array",
        "minItems": 1,
        "items": {"type": "string"}
      },
      "relations": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "confidence": {"type": "number"}
          },
          "required": ["subject", "predicate", "object"]
        }
      }
    },
    "required": ["labels"]
  })");
}

std::vector<SummaryFixture>
BuildSummaryFixtures ()
{
  return {
    { "project_status",
      {
        "chat/user: We are migrating the analytics worker from cron to an "
        "event-driven queue because overnight spikes have started missing the "
        "6am reporting SLA.",
        "chat/assistant: We moved report generation behind a job dispatcher, "
        "but the backfill path still uses the legacy cron tables and may "
        "double-enqueue retries.",
        "chat/user: The current risk is duplicate invoices for enterprise "
        "accounts in Chicago and New York if replay jobs exceed the idempotency "
        "window.",
      } },
    { "meeting_notes",
      {
        "chat/user: Sarah confirmed that Acme will extend the pilot through "
        "June if we ship SSO, audit exports, and per-team retention settings.",
        "chat/assistant: Engineering estimated SSO at one sprint, audit "
        "exports at four days, and retention controls behind a feature flag.",
        "chat/user: Finance needs a revised forecast because the pilot is "
        "currently discounted and support costs are being tracked manually.",
      } },
    { "incident_review",
      {
        "chat/user: During Saturday's outage the API stayed healthy, but the "
        "search index lagged by forty minutes after the Kafka consumer fell "
        "behind.",
        "chat/assistant: Operators scaled the consumer group from three to "
        "eight workers and cleared the backlog without data loss.",
        "chat/user: We still need alerts for backlog age, disk pressure on the "
        "indexer nodes, and a runbook for replaying failed partitions.",
      } },
  };
}

std::vector<ExtractionFixture>
BuildExtractionFixtures ()
{
  return {
    { "customer_account",
      "Summary: Alice from Acme Corp is expanding the Seattle pilot into New "
      "York. She needs SSO, audit exports, and retention controls before the "
      "June renewal.\n"
      "Source text: Alice works at Acme Corp, manages the Seattle deployment, "
      "and is coordinating the New York launch with the security team." },
    { "incident_ops",
      "Summary: The search outage was caused by Kafka consumer lag. Operators "
      "scaled workers, restored indexing, and now need backlog-age alerts.\n"
      "Source text: The search team increased the consumer pool from three to "
      "eight workers and replayed failed partitions without losing events." },
    { "roadmap",
      "Summary: The analytics migration replaces cron with a queue-backed "
      "worker to protect the 6am reporting deadline and avoid duplicate "
      "invoices.\n"
      "Source text: Finance flagged duplicate invoice risk for enterprise "
      "accounts if replay jobs exceed the idempotency window during backfill." },
  };
}

std::string
BuildCombinedExtractionInput (const std::string &summary,
                              const std::vector<std::string> &source_texts)
{
  std::string combined = "Summary:\n" + summary + "\n\nSource text:\n";
  for (const auto &text : source_texts)
    {
      combined += "- ";
      combined += text;
      combined += "\n";
    }
  return combined;
}

double
ElapsedMillis (std::chrono::steady_clock::time_point start,
               std::chrono::steady_clock::time_point end)
{
  return std::chrono::duration<double, std::milli> (end - start).count ();
}

BenchmarkStats
ComputeStats (std::vector<double> samples)
{
  if (samples.empty ())
    {
      throw std::runtime_error ("No benchmark samples recorded");
    }

  std::sort (samples.begin (), samples.end ());
  BenchmarkStats stats;
  stats.samples = samples.size ();
  stats.min_ms = samples.front ();
  stats.max_ms = samples.back ();
  stats.mean_ms = std::accumulate (samples.begin (), samples.end (), 0.0)
                  / static_cast<double> (samples.size ());

  const size_t mid = samples.size () / 2;
  if ((samples.size () % 2) == 0)
    {
      stats.median_ms = 0.5 * (samples[mid - 1] + samples[mid]);
    }
  else
    {
      stats.median_ms = samples[mid];
    }
  return stats;
}

BenchmarkStats
BenchmarkSummarization (cortext::Summarizer &summarizer,
                        const std::vector<SummaryFixture> &fixtures,
                        int warmup, int iterations, int max_summary_words)
{
  for (int i = 0; i < warmup; ++i)
    {
      for (const auto &fixture : fixtures)
        {
          const auto summary = summarizer.SummarizeTextsLimited (
              fixture.texts, max_summary_words);
          if (summary.empty ())
            {
              throw std::runtime_error ("Warmup summarization returned empty text for "
                                        + fixture.name);
            }
        }
    }

  std::vector<double> samples;
  samples.reserve (static_cast<size_t> (iterations) * fixtures.size ());
  for (int i = 0; i < iterations; ++i)
    {
      for (const auto &fixture : fixtures)
        {
          const auto start = std::chrono::steady_clock::now ();
          const auto summary = summarizer.SummarizeTextsLimited (
              fixture.texts, max_summary_words);
          const auto end = std::chrono::steady_clock::now ();
          if (summary.empty ())
            {
              throw std::runtime_error ("Summarization returned empty text for "
                                        + fixture.name);
            }
          samples.push_back (ElapsedMillis (start, end));
        }
    }
  return ComputeStats (std::move (samples));
}

BenchmarkStats
BenchmarkExtraction (cortext::Extractor &extractor,
                     const std::vector<ExtractionFixture> &fixtures,
                     const nlohmann::json &schema, int warmup, int iterations)
{
  for (int i = 0; i < warmup; ++i)
    {
      for (const auto &fixture : fixtures)
        {
          const auto result = extractor.ExtractFromText (fixture.text, schema);
          if (result.labels.empty ())
            {
              throw std::runtime_error ("Warmup extraction returned no labels for "
                                        + fixture.name);
            }
        }
    }

  std::vector<double> samples;
  samples.reserve (static_cast<size_t> (iterations) * fixtures.size ());
  for (int i = 0; i < iterations; ++i)
    {
      for (const auto &fixture : fixtures)
        {
          const auto start = std::chrono::steady_clock::now ();
          const auto result = extractor.ExtractFromText (fixture.text, schema);
          const auto end = std::chrono::steady_clock::now ();
          if (result.labels.empty ())
            {
              throw std::runtime_error ("Extraction returned no labels for "
                                        + fixture.name);
            }
          samples.push_back (ElapsedMillis (start, end));
        }
    }
  return ComputeStats (std::move (samples));
}

BenchmarkStats
BenchmarkCombined (cortext::Summarizer &summarizer, cortext::Extractor &extractor,
                   const std::vector<SummaryFixture> &fixtures,
                   const nlohmann::json &schema, int warmup, int iterations,
                   int max_summary_words)
{
  for (int i = 0; i < warmup; ++i)
    {
      for (const auto &fixture : fixtures)
        {
          const auto summary = summarizer.SummarizeTextsLimited (
              fixture.texts, max_summary_words);
          if (summary.empty ())
            {
              throw std::runtime_error ("Warmup combined pass returned empty summary for "
                                        + fixture.name);
            }
          const auto result = extractor.ExtractFromText (
              BuildCombinedExtractionInput (summary, fixture.texts), schema);
          if (result.labels.empty ())
            {
              throw std::runtime_error ("Warmup combined pass returned no labels for "
                                        + fixture.name);
            }
        }
    }

  std::vector<double> samples;
  samples.reserve (static_cast<size_t> (iterations) * fixtures.size ());
  for (int i = 0; i < iterations; ++i)
    {
      for (const auto &fixture : fixtures)
        {
          const auto start = std::chrono::steady_clock::now ();
          const auto summary = summarizer.SummarizeTextsLimited (
              fixture.texts, max_summary_words);
          if (summary.empty ())
            {
              throw std::runtime_error ("Combined pass returned empty summary for "
                                        + fixture.name);
            }
          const auto result = extractor.ExtractFromText (
              BuildCombinedExtractionInput (summary, fixture.texts), schema);
          const auto end = std::chrono::steady_clock::now ();
          if (result.labels.empty ())
            {
              throw std::runtime_error ("Combined pass returned no labels for "
                                        + fixture.name);
            }
          samples.push_back (ElapsedMillis (start, end));
        }
    }
  return ComputeStats (std::move (samples));
}

std::vector<cortext::internal::DeepLlmBackend>
ResolveRequestedBackends (const std::string &backend)
{
  if (backend == "all")
    {
      return { cortext::internal::DeepLlmBackend::Gemma,
               cortext::internal::DeepLlmBackend::Lfm2,
               cortext::internal::DeepLlmBackend::Mixed };
    }
  if (backend == "gemma")
    {
      return { cortext::internal::DeepLlmBackend::Gemma };
    }
  if (backend == "lfm2")
    {
      return { cortext::internal::DeepLlmBackend::Lfm2 };
    }
  if (backend == "mixed")
    {
      return { cortext::internal::DeepLlmBackend::Mixed };
    }
  throw std::runtime_error ("Unsupported --backend value: " + backend);
}

void
PrintStats (const std::string &label, const BenchmarkStats &stats)
{
  std::cout << "  " << label << ": median=" << std::fixed
            << std::setprecision (2) << stats.median_ms << " ms"
            << ", mean=" << stats.mean_ms << " ms"
            << ", min=" << stats.min_ms << " ms"
            << ", max=" << stats.max_ms << " ms"
            << ", samples=" << stats.samples << "\n";
}

void
PrintReport (const BackendReport &report)
{
  std::cout << "\n[" << report.name << "]\n";
  std::cout << "  summarizer_model=" << report.summarizer_model_path << "\n";
  std::cout << "  extractor_model=" << report.extractor_model_path << "\n";
  PrintStats ("summarization", report.summarization);
  PrintStats ("extraction", report.extraction);
  PrintStats ("combined", report.combined);
}

} // namespace

int
main (int argc, char *argv[])
{
  try
    {
      const Options opts = ParseArgs (argc, argv);
      const auto summary_fixtures = BuildSummaryFixtures ();
      const auto extraction_fixtures = BuildExtractionFixtures ();
      const auto extraction_schema = BuildExtractionSchema ();

      std::cout << "=== Cortext Deep LLM Benchmark ===\n";
      std::cout << "models_dir=" << opts.models_dir << "\n";
      std::cout << "backend=" << opts.backend << "\n";
      std::cout << "iterations=" << opts.iterations << "\n";
      std::cout << "warmup=" << opts.warmup << "\n";
      std::cout << "max_summary_words=" << opts.max_summary_words << "\n";

      std::vector<BackendReport> reports;
      for (const auto backend : ResolveRequestedBackends (opts.backend))
        {
          std::string error;
          auto selection = cortext::internal::TryCreateDeepLlmSelection (
              opts.models_dir, backend, &error);
          if (!selection.has_value ())
            {
              std::cout << "\n["
                        << cortext::internal::DescribeDeepLlmBackend (backend)
                        << "] unavailable: " << error << "\n";
              continue;
            }

          auto resolved = std::move (*selection);
          BackendReport report;
          report.name = resolved.backend_name;
          report.summarizer_model_path = resolved.summarizer_model_path;
          report.extractor_model_path = resolved.extractor_model_path;
          report.summarization = BenchmarkSummarization (
              *resolved.summarizer, summary_fixtures, opts.warmup,
              opts.iterations, opts.max_summary_words);
          report.extraction = BenchmarkExtraction (
              *resolved.extractor, extraction_fixtures, extraction_schema,
              opts.warmup, opts.iterations);
          report.combined = BenchmarkCombined (
              *resolved.summarizer, *resolved.extractor, summary_fixtures,
              extraction_schema, opts.warmup, opts.iterations,
              opts.max_summary_words);
          reports.push_back (std::move (report));
        }

      if (reports.empty ())
        {
          throw std::runtime_error (
              "No requested deep LLM backend is available under "
              + opts.models_dir.string ());
        }

      for (const auto &report : reports)
        {
          PrintReport (report);
        }

      if (reports.size () > 1)
        {
          const auto fastest = std::min_element (
              reports.begin (), reports.end (),
              [] (const BackendReport &a, const BackendReport &b) {
                return a.combined.median_ms < b.combined.median_ms;
              });
          std::cout << "\nFastest combined median: " << fastest->name << " ("
                    << std::fixed << std::setprecision (2)
                    << fastest->combined.median_ms << " ms)\n";
        }
      return 0;
    }
  catch (const std::exception &e)
    {
      std::cerr << "cortext_deep_llm_bench failed: " << e.what () << "\n";
      return 1;
    }
}
