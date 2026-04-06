#include "../../src/deep_llm/deep_llm_factory.hpp"
#include "../../src/deep_llm/lfm2_llama_backend.hpp"
#include "cortext/operations/label_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
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
  std::vector<std::string> texts;
  std::vector<std::string> desired;
  std::vector<std::string> banned;
};

struct StyleStats
{
  std::string style;
  int desired_hits = 0;
  int banned_hits = 0;
  int summaries_emitted = 0;
  int fixtures = 0;
  int sentence_overflow_penalty = 0;
  int word_overflow_penalty = 0;
  int score = 0;
  long long latency_ms = 0;
};

std::string
Normalize (const std::string &value)
{
  return cortext::operations::NormalizeLabelKey (value);
}

bool
LooselyMatches (const std::string &haystack, const std::string &needle)
{
  const std::string a = Normalize (haystack);
  const std::string b = Normalize (needle);
  if (a.empty () || b.empty ())
    {
      return false;
    }
  return a == b || a.find (b) != std::string::npos
         || b.find (a) != std::string::npos;
}

int
CountSentences (const std::string &text)
{
  int count = 0;
  for (char ch : text)
    {
      if (ch == '.' || ch == '!' || ch == '?')
        {
          count += 1;
        }
    }
  return std::max (1, count);
}

int
CountWords (const std::string &text)
{
  std::istringstream input (text);
  int words = 0;
  std::string token;
  while (input >> token)
    {
      words += 1;
    }
  return words;
}

} // namespace

int
main ()
{
  const auto model_path_opt = cortext::internal::ResolveLfm2SummarizerModelPath (
      std::filesystem::path ("models"));
  if (!model_path_opt.has_value ())
    {
      std::cerr << "No LFM2 summarizer model found.\n";
      return 1;
    }

  cortext::Lfm2LlamaSummarizer summarizer (model_path_opt->string ());
  if (!summarizer.IsAvailable ())
    {
      std::cerr << "LFM2 summarizer failed to initialize.\n";
      return 1;
    }

  const std::vector<Fixture> fixtures = {
    { "identity_work",
      { "User: My name is Gabriel and I live in Chicago.",
        "Assistant: Noted.",
        "User: I am debugging a SQLite migration for Cortext before the "
        "demo." },
      { "Gabriel", "Chicago", "SQLite migration", "Cortext", "demo" },
      { "the user", "the assistant", "conversation", "excerpt", "discussed",
        "said", "asked", "user:", "assistant:", "final summary",
        "output summary", "prioritize durable facts" } },
    { "project_ops",
      { "User: Alice is leading the Acme pilot in Seattle.",
        "Assistant: The customer needs SSO and audit exports before June.",
        "User: Finance is revising the renewal forecast this week." },
      { "Alice", "Acme pilot", "Seattle", "SSO", "audit exports", "June",
        "renewal forecast" },
      { "the user", "the assistant", "conversation", "excerpt", "discussed",
        "said", "asked", "user:", "assistant:", "final summary",
        "output summary", "prioritize durable facts",
        "underlying facts and topics could include" } },
    { "neighbor_housing",
      { "User: My neighbor Sarah is helping with a housing application in "
        "Logan Square.",
        "Assistant: Missing pay stubs are delaying the submission.",
        "User: The deadline is next Tuesday." },
      { "Sarah", "housing application", "Logan Square", "pay stubs",
        "Tuesday" },
      { "the user", "the assistant", "conversation", "excerpt", "discussed",
        "said", "asked", "user:", "assistant:", "final summary",
        "output summary", "underlying facts and topics could include" } },
  };

  const std::vector<std::string> styles = { "baseline", "durable",
                                            "precision", "fewshot_durable",
                                            "fewshot_precision",
                                            "transcript_fewshot" };

  bool ok = true;
  for (const auto &style : styles)
    {
      ScopedEnvVar scoped_prompt ("CORTEXT_LFM2_SUMMARY_PROMPT_STYLE", style);
      StyleStats stats;
      stats.style = style;
      stats.fixtures = static_cast<int> (fixtures.size ());

      std::cout << "style=" << style << "\n";
      for (const auto &fixture : fixtures)
        {
          const auto started = std::chrono::steady_clock::now ();
          const std::string summary = summarizer.SummarizeTextsLimited (
              fixture.texts, 48);
          const auto finished = std::chrono::steady_clock::now ();
          stats.latency_ms += std::chrono::duration_cast<std::chrono::milliseconds> (
                                  finished - started)
                                  .count ();

          if (!summary.empty ())
            {
              stats.summaries_emitted += 1;
            }

          std::unordered_set<std::string> seen_desired;
          std::unordered_set<std::string> seen_banned;
          for (const auto &desired : fixture.desired)
            {
              if (LooselyMatches (summary, desired))
                {
                  seen_desired.insert (Normalize (desired));
                }
            }
          for (const auto &banned : fixture.banned)
            {
              if (LooselyMatches (summary, banned))
                {
                  seen_banned.insert (Normalize (banned));
                }
            }

          const int sentence_count = CountSentences (summary);
          const int word_count = CountWords (summary);
          stats.desired_hits += static_cast<int> (seen_desired.size ());
          stats.banned_hits += static_cast<int> (seen_banned.size ());
          stats.sentence_overflow_penalty += std::max (0, sentence_count - 3);
          stats.word_overflow_penalty += std::max (0, word_count - 48);

          std::cout << "fixture_" << fixture.name << "_summary=" << summary
                    << "\n";
          std::cout << "fixture_" << fixture.name << "_desired_hits="
                    << seen_desired.size () << "\n";
          std::cout << "fixture_" << fixture.name << "_banned_hits="
                    << seen_banned.size () << "\n";
          std::cout << "fixture_" << fixture.name << "_sentence_count="
                    << sentence_count << "\n";
          std::cout << "fixture_" << fixture.name << "_word_count="
                    << word_count << "\n";
        }

      stats.score = 2 * stats.desired_hits - 3 * stats.banned_hits
                    - stats.sentence_overflow_penalty
                    - stats.word_overflow_penalty;

      std::cout << "summary_prompt_ablation_" << style << "_desired_hits="
                << stats.desired_hits << "\n";
      std::cout << "summary_prompt_ablation_" << style << "_banned_hits="
                << stats.banned_hits << "\n";
      std::cout << "summary_prompt_ablation_" << style
                << "_summaries_emitted=" << stats.summaries_emitted << "\n";
      std::cout << "summary_prompt_ablation_" << style << "_latency_ms="
                << stats.latency_ms << "\n";
      std::cout << "summary_prompt_ablation_" << style
                << "_sentence_overflow_penalty="
                << stats.sentence_overflow_penalty << "\n";
      std::cout << "summary_prompt_ablation_" << style
                << "_word_overflow_penalty=" << stats.word_overflow_penalty
                << "\n";
      std::cout << "summary_prompt_ablation_" << style << "_score="
                << stats.score << "\n";

      ok &= (stats.summaries_emitted == stats.fixtures);
    }

  return ok ? 0 : 1;
}
