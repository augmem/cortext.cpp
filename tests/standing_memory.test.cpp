#include "../src/operations/standing_memory_internal.hpp"

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <utility>
#include <vector>

namespace standing = cortext::operations::standing_memory_internal;

TEST_CASE ("Standing directive detector accepts clause-initial directives",
           "[standing]")
{
  REQUIRE (standing::IsStandingDirective (
      "Always include version numbers when I ask about libraries."));
  REQUIRE (standing::IsStandingDirective (
      "[2024/03/15 (Fri) 08:54] user: Always format code snippets with "
      "syntax highlighting."));
  REQUIRE (standing::IsStandingDirective (
      "Thanks for that. From now on, answer in French."));
  REQUIRE (standing::IsStandingDirective ("please never use emoji"));
  REQUIRE (standing::IsStandingDirective ("Please, always cite sources!"));
  REQUIRE (standing::IsStandingDirective (
      "Noted.\nI prefer short answers with bullet points."));
  REQUIRE (standing::IsStandingDirective (
      "Going forward: keep replies under 100 words."));
}

TEST_CASE ("Standing directive detector rejects mentions and long text",
           "[standing]")
{
  // The marker must open a clause and be a whole word.
  REQUIRE_FALSE (standing::IsStandingDirective (
      "I always forget my keys on Mondays."));
  REQUIRE_FALSE (standing::IsStandingDirective (
      "The alwaysOn flag is set in the config."));
  REQUIRE_FALSE (standing::IsStandingDirective (
      "Nevertheless the build passed."));
  REQUIRE_FALSE (standing::IsStandingDirective (""));
  // Long turns that contain a directive-looking clause are explanations.
  std::string long_text = "Always validate the input. ";
  while (long_text.size () <= standing::kMaxTextBytes)
    long_text += "The parser reads each record and checks its fields. ";
  REQUIRE_FALSE (standing::IsStandingDirective (long_text));
}

TEST_CASE ("Standing members rank by shared stems with set frequencies",
           "[standing]")
{
  const std::vector<std::pair<long long, std::string> > members = {
    { 1, standing::TermsOf ("Always include latency statistics when I ask "
                            "about query latency reduction.") },
    { 2, standing::TermsOf ("Always include version numbers when I ask "
                            "about libraries.") },
    { 3, standing::TermsOf ("Always include cost per hour when I ask about "
                            "pricing.") },
  };
  REQUIRE (members[1].second.find ("versio") != std::string::npos);
  const auto ranked = standing::RankByTerms (
      "Which library versions does the service depend on?", members);
  REQUIRE_FALSE (ranked.empty ());
  REQUIRE (ranked.front ().first == 2); // "librar", "versio"
  for (const auto &[id, score] : ranked)
    REQUIRE (id != 3);
  REQUIRE (standing::RankByTerms ("", members).empty ());
}
