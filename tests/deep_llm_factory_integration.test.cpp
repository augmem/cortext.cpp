// Compiled only when models/gemma4-e2b-litert is present and LiteRT is
// enabled (see CMakeLists.txt). Once built, assertions are unconditional.

#include <catch2/catch_test_macros.hpp>

#include "../src/deep_llm/deep_llm_factory.hpp"

#include <filesystem>
#include <string>

namespace
{

std::string
FindModelPath (const std::string &relative_path)
{
  std::filesystem::path probe = std::filesystem::current_path ();
  for (int i = 0; i < 6; ++i)
    {
      const auto candidate = probe / relative_path;
      if (std::filesystem::exists (candidate))
        {
          return candidate.string ();
        }
      if (!probe.has_parent_path ())
        {
          break;
        }
      probe = probe.parent_path ();
    }
  return relative_path;
}

} // namespace

TEST_CASE ("Deep selection resolves the Gemma4 stack",
           "[deep_llm][integration]")
{
  auto selection = cortext::internal::TryCreateDeepLlmSelection (
      FindModelPath ("models"), nullptr);
  REQUIRE (selection.has_value ());
  CHECK (selection->backend_name == "Gemma/LiteRT-LM");
  CHECK (selection->summarizer_model_path.filename ()
         == std::filesystem::path ("gemma-4-E2B-it.litertlm"));
  CHECK (selection->extractor_model_path.filename ()
         == std::filesystem::path ("gemma-4-E2B-it.litertlm"));
}
