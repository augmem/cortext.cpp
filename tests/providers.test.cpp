#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cortext/providers/openai_provider.hpp>
#include <cortext/providers/registry.hpp>

using namespace cortext;

TEST_CASE ("OpenAI-compatible provider URI resolves for generation roles",
           "[providers][openai]")
{
  std::string error;
  auto summarizer = providers::ResolveProvider (
      "openai://127.0.0.1:8000/v1/gemma4-e2b", providers::Role::Summarizer,
      &error);

  REQUIRE (summarizer != nullptr);
  REQUIRE (error.empty ());
  CHECK (summarizer->Identity ().scheme == "openai");
  CHECK (summarizer->Identity ().endpoint == "http://127.0.0.1:8000/v1");
  CHECK (summarizer->Identity ().model == "gemma4-e2b");
  CHECK (summarizer->GetCapabilities ().text);
  CHECK (summarizer->GetCapabilities ().constraints
         == providers::ConstraintSupport::ServerSchema);

  auto extractor = providers::ResolveProvider (
      "openai://localhost/v1/gemma4-e2b", providers::Role::Extractor,
      &error);
  REQUIRE (extractor != nullptr);
  CHECK (extractor->Identity ().endpoint == "http://localhost:8000/v1");
}

TEST_CASE ("OpenAI-compatible provider rejects unsupported roles",
           "[providers][openai]")
{
  std::string error;
  auto encoder = providers::ResolveProvider (
      "openai://127.0.0.1:8000/v1/gemma4-e2b", providers::Role::Encoder,
      &error);

  CHECK (encoder == nullptr);
  CHECK_THAT (error, Catch::Matchers::ContainsSubstring ("encoder role"));
}

TEST_CASE ("OpenAI-compatible provider is text-only",
           "[providers][openai]")
{
  providers::OpenAIProvider provider ("127.0.0.1:8000", "v1/gemma4-e2b");
  providers::GenerateRequest request;
  request.role = providers::Role::Summarizer;
  request.system_prompt = "summarize";
  providers::ContentPart part;
  part.kind = providers::ContentPart::Kind::AudioPcm16k;
  part.pcm = { 0.0f, 0.1f };
  request.parts.push_back (std::move (part));

  CHECK_THROWS_WITH (
      provider.Generate (request),
      Catch::Matchers::ContainsSubstring ("text parts only"));
}
