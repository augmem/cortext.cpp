#include "../../src/deep_llm/deep_llm_factory.hpp"
#include "../../src/deep_llm/lfm2_llama_backend.hpp"
#include "cortext/operations/label_utils.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <chrono>
#include <unordered_set>
#include <vector>

namespace
{

class ScopedEnvVar
{
public:
  ScopedEnvVar (const char *name, const std::string &value) : name_ (name)
  {
    const char *existing = std::getenv (name_);
    if (existing != nullptr)
      {
        had_value_ = true;
        old_value_ = existing;
      }
    setenv (name_, value.c_str (), 1);
  }

  ~ScopedEnvVar ()
  {
    if (had_value_)
      {
        setenv (name_, old_value_.c_str (), 1);
      }
    else
      {
        unsetenv (name_);
      }
  }

private:
  const char *name_;
  bool had_value_ = false;
  std::string old_value_;
};

struct Fixture
{
  std::string name;
  std::string text;
  std::vector<std::string> desired;
  std::vector<std::string> banned;
};

struct StyleStats
{
  std::string style;
  int desired_hits = 0;
  int banned_hits = 0;
  int labels_emitted = 0;
  int fixtures_with_labels = 0;
  int fixtures = 0;
  int overflow_penalty = 0;
  int score = 0;
  long long latency_ms = 0;
};

nlohmann::json
BuildSchema ()
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

std::string
Normalize (const std::string &value)
{
  return cortext::operations::NormalizeLabelKey (value);
}

bool
LooselyMatches (const std::string &lhs, const std::string &rhs)
{
  const std::string a = Normalize (lhs);
  const std::string b = Normalize (rhs);
  if (a.empty () || b.empty ())
    {
      return false;
    }
  return a == b || a.find (b) != std::string::npos
         || b.find (a) != std::string::npos;
}

std::string
JoinLabels (const cortext::operations::ExtractionResult &result)
{
  std::string out;
  for (size_t i = 0; i < result.labels.size (); ++i)
    {
      if (i > 0)
        {
          out += ", ";
        }
      out += result.labels[i].label;
    }
  return out;
}

} // namespace

int
main ()
{
  const auto model_path_opt = cortext::internal::ResolveLfm2ExtractorModelPath (
      std::filesystem::path ("models"));
  if (!model_path_opt.has_value ())
    {
      std::cerr << "No LFM2 extractor model found.\n";
      return 1;
    }

  cortext::Lfm2LlamaExtractor extractor (model_path_opt->string ());
  if (!extractor.IsAvailable ())
    {
      std::cerr << "LFM2 extractor failed to initialize.\n";
      return 1;
    }

  const auto schema = BuildSchema ();
  const std::vector<Fixture> fixtures = {
    { "interview_noise",
      "User: I need help getting ready for an interview tomorrow.\n"
      "Assistant: We can make a prep plan.\n"
      "User: I want more confidence, a clear explanation, and support for "
      "the prep steps.",
      { "interview", "confidence" },
      { "get", "getting", "ready", "explanation", "support" } },
    { "identity_work",
      "User: My name is Gabriel.\n"
      "Assistant: Noted.\n"
      "User: I live in Chicago and I am debugging a SQLite migration for "
      "Cortext.",
      { "Gabriel", "Chicago", "SQLite migration", "Cortext" },
      { "valid_start_ts", "valid_end_ts", "get", "ready" } },
    { "neighbor_housing",
      "User: My neighbor Sarah is helping me prepare documents.\n"
      "Assistant: For what?\n"
      "User: For a housing application in Logan Square.",
      { "Sarah", "housing application", "Logan Square" },
      { "prepare", "documents", "helping", "valid_start_ts",
        "valid_end_ts" } },
  };

  const std::vector<std::string> styles = { "baseline", "durable",
                                            "precision", "fewshot_durable",
                                            "fewshot_precision" };

  bool ok = true;
  for (const auto &style : styles)
    {
      ScopedEnvVar scoped_prompt ("CORTEXT_LFM2_EXTRACT_PROMPT_STYLE", style);
      StyleStats stats;
      stats.style = style;
      stats.fixtures = static_cast<int> (fixtures.size ());

      std::cout << "style=" << style << "\n";
      for (const auto &fixture : fixtures)
        {
          const auto started = std::chrono::steady_clock::now ();
          const auto result = extractor.ExtractFromText (fixture.text, schema);
          const auto finished = std::chrono::steady_clock::now ();
          stats.latency_ms += std::chrono::duration_cast<std::chrono::milliseconds> (
                                  finished - started)
                                  .count ();
          if (!result.labels.empty ())
            {
              stats.fixtures_with_labels += 1;
            }

          std::unordered_set<std::string> seen_desired;
          std::unordered_set<std::string> seen_banned;
          for (const auto &label : result.labels)
            {
              stats.labels_emitted += 1;
              for (const auto &desired : fixture.desired)
                {
                  if (LooselyMatches (label.label, desired))
                    {
                      seen_desired.insert (Normalize (desired));
                    }
                }
              for (const auto &banned : fixture.banned)
                {
                  if (LooselyMatches (label.label, banned))
                    {
                      seen_banned.insert (Normalize (banned));
                    }
                }
            }

          stats.desired_hits += static_cast<int> (seen_desired.size ());
          stats.banned_hits += static_cast<int> (seen_banned.size ());
          const int overflow = std::max (0,
                                         static_cast<int> (result.labels.size ())
                                             - 8);
          stats.overflow_penalty += overflow;

          std::cout << "fixture_" << fixture.name << "_labels="
                    << JoinLabels (result) << "\n";
          std::cout << "fixture_" << fixture.name << "_count="
                    << result.labels.size () << "\n";
          std::cout << "fixture_" << fixture.name << "_desired_hits="
                    << seen_desired.size () << "\n";
          std::cout << "fixture_" << fixture.name << "_banned_hits="
                    << seen_banned.size () << "\n";
        }

      stats.score = 3 * stats.desired_hits - 2 * stats.banned_hits
                    - stats.overflow_penalty;
      std::cout << "prompt_ablation_" << style << "_desired_hits="
                << stats.desired_hits << "\n";
      std::cout << "prompt_ablation_" << style << "_banned_hits="
                << stats.banned_hits << "\n";
      std::cout << "prompt_ablation_" << style << "_labels_emitted="
                << stats.labels_emitted << "\n";
      std::cout << "prompt_ablation_" << style << "_latency_ms="
                << stats.latency_ms << "\n";
      std::cout << "prompt_ablation_" << style << "_fixtures_with_labels="
                << stats.fixtures_with_labels << "\n";
      std::cout << "prompt_ablation_" << style << "_overflow_penalty="
                << stats.overflow_penalty << "\n";
      std::cout << "prompt_ablation_" << style << "_score=" << stats.score
                << "\n";

      ok &= (stats.fixtures_with_labels == stats.fixtures);
    }

  return ok ? 0 : 1;
}
