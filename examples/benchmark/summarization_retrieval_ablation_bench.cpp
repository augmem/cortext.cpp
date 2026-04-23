#include "cortext/consolidation_mode.hpp"
#include "cortext/cortext.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{

struct DetailScore
{
  int name = 0;
  int age = 0;
  int pitch = 0;
  int weekend = 0;
  int repetition = 0;

  int Total () const
  {
    return name + age + pitch + weekend + repetition;
  }
};

struct VariantResult
{
  std::string name;
  std::size_t retrieved = 0;
  std::size_t raw_hits = 0;
  std::size_t summary_hits = 0;
  DetailScore raw_details;
  DetailScore summary_details;
};

std::string
LowerAscii (std::string text)
{
  std::transform (text.begin (), text.end (), text.begin (), [] (char ch) {
    return static_cast<char> (
        std::tolower (static_cast<unsigned char> (ch)));
  });
  return text;
}

bool
ContainsAny (const std::string &text,
             const std::vector<std::string> &needles)
{
  for (const auto &needle : needles)
    {
      if (text.find (needle) != std::string::npos)
        {
          return true;
        }
    }
  return false;
}

DetailScore
ScoreDetails (const std::string &text)
{
  const std::string lower = LowerAscii (text);
  DetailScore score;
  score.name = ContainsAny (lower, { "maya" }) ? 1 : 0;
  score.age = ContainsAny (
                  lower,
                  { "11-year-old", "11 year old", "11 years old",
                    "eleven-year-old", "eleven year old", "eleven years old" })
                  ? 1
                  : 0;
  score.pitch = ContainsAny (lower, { "pitch" }) ? 1 : 0;
  score.weekend = ContainsAny (lower, { "weekend" }) ? 1 : 0;
  score.repetition = ContainsAny (
                         lower,
                         { "three times", "3 times", "mentioned the 11" })
                         ? 1
                         : 0;
  return score;
}

void
AccumulateDetails (DetailScore &dst, const DetailScore &src)
{
  dst.name = std::max (dst.name, src.name);
  dst.age = std::max (dst.age, src.age);
  dst.pitch = std::max (dst.pitch, src.pitch);
  dst.weekend = std::max (dst.weekend, src.weekend);
  dst.repetition = std::max (dst.repetition, src.repetition);
}

std::string
MemoryText (const cortext::Cortext::Context::Memory &memory)
{
  std::string text = memory.source_id;
  for (const auto &blob : memory.content)
    {
      if (!text.empty ())
        {
          text += "\n";
        }
      text.append (blob.begin (), blob.end ());
    }
  return text;
}

std::filesystem::path
MakeTempDbPath (const std::string &variant)
{
  const auto stamp = std::chrono::steady_clock::now ().time_since_epoch ().count ();
  auto path = std::filesystem::temp_directory_path ();
  path /= "cortext_summary_retrieval_" + variant + "_"
          + std::to_string (stamp) + ".db";
  return path;
}

VariantResult
RunVariant (const std::string &name, cortext::ConsolidationMode mode,
            bool consolidate)
{
  const auto db_path = MakeTempDbPath (name);
  std::filesystem::remove (db_path);
  std::filesystem::remove (db_path.string () + "-wal");
  std::filesystem::remove (db_path.string () + "-shm");

  cortext::Cortext::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto engine = cortext::Cortext::Create (cfg, db_path.string (), "models");
  std::uint64_t ts = 1700000000000ULL;

  const std::vector<std::string> event_turns = {
    "My niece Maya is 11 years old.",
    "This weekend Maya gave her first pitch at the youth robotics showcase.",
    "I mentioned the 11-year-old pitch three times because it mattered to me.",
    "Maya practiced the pitch three times before this weekend's showcase.",
  };

  for (const auto &turn : event_turns)
    {
      engine->ProcessTextAt (turn, "chat/user", ts);
      ts += 1000;
    }

  const std::vector<std::string> distractors = {
    "I also need to remember the grocery order and the library pickup.",
    "The weather changed quickly after lunch.",
    "We talked about database interfaces and object storage adapters.",
    "There was also a reminder to rebuild the chat target with j8.",
    "The soccer schedule moved to Tuesday evening.",
    "A neighbor asked about a housing application deadline.",
  };
  for (int round = 0; round < 3; ++round)
    {
      for (const auto &turn : distractors)
        {
          engine->ProcessTextAt (turn, "chat/user", ts);
          ts += 1000;
        }
    }

  if (consolidate)
    {
      engine->Consolidate (mode);
    }

  const auto ctx = engine->ProcessTextAt (
      "Who was the 11 year old who did a pitch this weekend, and how many "
      "times was it mentioned?",
      "chat/user", ts + 1000);

  VariantResult result;
  result.name = name;
  result.retrieved = ctx.retrieved_memory.size ();
  for (const auto &memory : ctx.retrieved_memory)
    {
      const auto details = ScoreDetails (MemoryText (memory));
      const bool is_summary = memory.source_id.rfind ("summary_", 0) == 0;
      if (is_summary)
        {
          result.summary_hits += (details.Total () > 0) ? 1U : 0U;
          AccumulateDetails (result.summary_details, details);
        }
      else
        {
          result.raw_hits += (details.Total () > 0) ? 1U : 0U;
          AccumulateDetails (result.raw_details, details);
        }
    }

  engine.reset ();
  std::filesystem::remove (db_path);
  std::filesystem::remove (db_path.string () + "-wal");
  std::filesystem::remove (db_path.string () + "-shm");
  return result;
}

void
PrintResult (const VariantResult &result)
{
  std::cout << "summarization_retrieval_" << result.name
            << "_retrieved=" << result.retrieved << "\n";
  std::cout << "summarization_retrieval_" << result.name
            << "_raw_hits=" << result.raw_hits << "\n";
  std::cout << "summarization_retrieval_" << result.name
            << "_summary_hits=" << result.summary_hits << "\n";
  std::cout << "summarization_retrieval_" << result.name
            << "_raw_detail_hits=" << result.raw_details.Total () << "\n";
  std::cout << "summarization_retrieval_" << result.name
            << "_summary_detail_hits=" << result.summary_details.Total ()
            << "\n";
  std::cout << "summarization_retrieval_" << result.name
            << "_raw_age=" << result.raw_details.age << "\n";
  std::cout << "summarization_retrieval_" << result.name
            << "_raw_pitch=" << result.raw_details.pitch << "\n";
  std::cout << "summarization_retrieval_" << result.name
            << "_raw_weekend=" << result.raw_details.weekend << "\n";
  std::cout << "summarization_retrieval_" << result.name
            << "_raw_repetition=" << result.raw_details.repetition << "\n";
  std::cout << "summarization_retrieval_" << result.name
            << "_summary_age=" << result.summary_details.age << "\n";
  std::cout << "summarization_retrieval_" << result.name
            << "_summary_pitch=" << result.summary_details.pitch << "\n";
  std::cout << "summarization_retrieval_" << result.name
            << "_summary_weekend=" << result.summary_details.weekend << "\n";
  std::cout << "summarization_retrieval_" << result.name
            << "_summary_repetition=" << result.summary_details.repetition
            << "\n";
}

} // namespace

int
main ()
{
  const std::vector<VariantResult> results = {
    RunVariant ("none", cortext::ConsolidationMode::Both, false),
    RunVariant ("shallow", cortext::ConsolidationMode::Shallow, true),
    RunVariant ("deep", cortext::ConsolidationMode::Deep, true),
  };

  for (const auto &result : results)
    {
      PrintResult (result);
    }

  return 0;
}
