#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "../src/deep_llm/deep_llm_factory.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

class TempDir
{
public:
  TempDir ()
  {
    path_ = std::filesystem::temp_directory_path ()
            / std::filesystem::path ("cortext-deep-llm-test-XXXXXX");
    std::string templ = path_.string ();
    std::vector<char> buffer (templ.begin (), templ.end ());
    buffer.push_back ('\0');
    char *created = mkdtemp (buffer.data ());
    if (created == nullptr)
      {
        throw std::runtime_error ("mkdtemp failed");
      }
    path_ = created;
  }

  ~TempDir ()
  {
    std::error_code ec;
    std::filesystem::remove_all (path_, ec);
  }

  const std::filesystem::path &
  path () const
  {
    return path_;
  }

  std::filesystem::path
  Touch (const std::filesystem::path &relative)
  {
    const auto file_path = path_ / relative;
    std::filesystem::create_directories (file_path.parent_path ());
    std::ofstream out (file_path);
    out << "stub";
    out.close ();
    return file_path;
  }

private:
  std::filesystem::path path_;
};

} // namespace

TEST_CASE ("Gemma deep resolver requires Gemma4", "[deep_llm][resolution]")
{
  TempDir temp_dir;
  temp_dir.Touch ("gemma3n-e2b-litert/gemma-3n-E2B-it-int4.litertlm");

  CHECK_FALSE (
      cortext::internal::ResolveGemmaDeepLlmModelPath (temp_dir.path ())
          .has_value ());

  const auto gemma4
      = temp_dir.Touch ("gemma4-e2b-litert/gemma-4-E2B-it.litertlm");
  const auto resolved
      = cortext::internal::ResolveGemmaDeepLlmModelPath (temp_dir.path ());
  REQUIRE (resolved.has_value ());
  CHECK (std::filesystem::equivalent (*resolved, gemma4));
}

TEST_CASE ("Deep selection fails clearly without Gemma4",
           "[deep_llm][resolution]")
{
  TempDir temp_dir;
  temp_dir.Touch ("gemma3n-e2b-litert/gemma-3n-E2B-it-int4.litertlm");

  std::string error;
  auto selection = cortext::internal::TryCreateDeepLlmSelection (
      temp_dir.path (), &error);

  CHECK_FALSE (selection.has_value ());
#if !defined(CORTEXT_DISABLE_LITERT)
  CHECK_THAT (error, Catch::Matchers::ContainsSubstring ("gemma4-e2b-litert"));
#endif
}
