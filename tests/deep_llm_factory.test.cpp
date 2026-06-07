#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "../src/deep_llm/deep_llm_factory.hpp"
#include "../src/deep_llm/lfm2_llama_backend.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{

class ScopedEnvVar
{
public:
  explicit ScopedEnvVar (const char *name) : name_ (name)
  {
    const char *existing = std::getenv (name);
    if (existing != nullptr)
      {
        had_value_ = true;
        old_value_ = existing;
      }
    unsetenv (name_);
  }

  ScopedEnvVar (const char *name, const std::string &value) : ScopedEnvVar (name)
  {
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

nlohmann::json
BuildSupportedSchema ()
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

TEST_CASE ("Deep LLM backend override parsing", "[deep_llm][config]")
{
  ScopedEnvVar clear ("CORTEXT_DEEP_LLM_BACKEND");
  CHECK (cortext::internal::ResolveDeepLlmBackendOverride ()
         == cortext::internal::DeepLlmBackend::Auto);

  {
    ScopedEnvVar env ("CORTEXT_DEEP_LLM_BACKEND", "gemma");
    CHECK (cortext::internal::ResolveDeepLlmBackendOverride ()
           == cortext::internal::DeepLlmBackend::Gemma);
  }

  {
    ScopedEnvVar env ("CORTEXT_DEEP_LLM_BACKEND", "lfm2");
    CHECK (cortext::internal::ResolveDeepLlmBackendOverride ()
           == cortext::internal::DeepLlmBackend::Lfm2);
  }

  {
    ScopedEnvVar env ("CORTEXT_DEEP_LLM_BACKEND", "mixed");
    CHECK (cortext::internal::ResolveDeepLlmBackendOverride ()
           == cortext::internal::DeepLlmBackend::Mixed);
  }

  {
    ScopedEnvVar env ("CORTEXT_DEEP_LLM_BACKEND", "hybrid");
    CHECK (cortext::internal::ResolveDeepLlmBackendOverride ()
           == cortext::internal::DeepLlmBackend::Mixed);
  }

  {
    ScopedEnvVar env ("CORTEXT_DEEP_LLM_BACKEND", "invalid");
    CHECK_THROWS_AS (cortext::internal::ResolveDeepLlmBackendOverride (),
                     std::runtime_error);
  }
}

TEST_CASE ("LFM2 pinned Q4 model resolution", "[deep_llm][resolution]")
{
  ScopedEnvVar clear_summary ("CORTEXT_LFM2_SUMMARIZER_MODEL");
  ScopedEnvVar clear_extract ("CORTEXT_LFM2_EXTRACT_MODEL");
  TempDir temp_dir;

  auto transcript = temp_dir.Touch (
      "LFM2-2.6B-Transcript-GGUF/LFM2-2.6B-Transcript-Q4_K_M.gguf");
  auto extract = temp_dir.Touch (
      "LFM2-1.2B-Extract-GGUF/LFM2-1.2B-Extract-Q4_K_M.gguf");

  const auto resolved_transcript
      = cortext::internal::ResolveLfm2SummarizerModelPath (temp_dir.path ());
  const auto resolved_extract
      = cortext::internal::ResolveLfm2ExtractorModelPath (temp_dir.path ());

  REQUIRE (resolved_transcript.has_value ());
  REQUIRE (resolved_extract.has_value ());
  CHECK (std::filesystem::equivalent (*resolved_transcript, transcript));
  CHECK (std::filesystem::equivalent (*resolved_extract, extract));
}

TEST_CASE ("LFM2 env override takes precedence", "[deep_llm][resolution]")
{
  ScopedEnvVar clear_extract ("CORTEXT_LFM2_EXTRACT_MODEL");
  TempDir temp_dir;
  auto preferred = temp_dir.Touch ("override/LFM2-2.6B-Transcript-Q4_K_M.gguf");
  temp_dir.Touch ("LFM2-2.6B-Transcript-GGUF/LFM2-2.6B-Transcript-Q4_K_M.gguf");

  ScopedEnvVar env ("CORTEXT_LFM2_SUMMARIZER_MODEL", preferred.string ());
  const auto resolved
      = cortext::internal::ResolveLfm2SummarizerModelPath (temp_dir.path ());
  REQUIRE (resolved.has_value ());
  CHECK (std::filesystem::equivalent (*resolved, preferred));
}

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

TEST_CASE ("Auto deep backend does not fall back without Gemma4",
           "[deep_llm][resolution]")
{
  TempDir temp_dir;
  temp_dir.Touch ("gemma3n-e2b-litert/gemma-3n-E2B-it-int4.litertlm");
  temp_dir.Touch ("LFM2.5-350M-GGUF/LFM2.5-350M-Q4_K_M.gguf");
  temp_dir.Touch (
      "LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf");

  std::string error;
  auto selection = cortext::internal::TryCreateDeepLlmSelection (
      temp_dir.path (), cortext::internal::DeepLlmBackend::Auto, &error);

  CHECK_FALSE (selection.has_value ());
  CHECK_THAT (error, Catch::Matchers::ContainsSubstring ("gemma4-e2b-litert"));
  CHECK (error.find ("Liquid") == std::string::npos);
  CHECK (error.find ("Mixed") == std::string::npos);
}

TEST_CASE ("LFM2 extraction grammar supports current schema subset",
           "[deep_llm][grammar]")
{
  const std::string grammar
      = cortext::internal::BuildLfm2ExtractionGrammar (BuildSupportedSchema ());
  CHECK (grammar.find ("labels-kv") != std::string::npos);
  CHECK (grammar.find ("relations-kv") != std::string::npos);
  CHECK (grammar.find ("relations-item-confidence-kv") != std::string::npos);
  CHECK (grammar.find ("nonempty-string ::= ") != std::string::npos);
  CHECK (grammar.find ("labels ::= \"[\" space nonempty-string")
         != std::string::npos);
  CHECK (grammar.find ("root_body") == std::string::npos);
  CHECK (grammar.find ("string ::= \"\\\"\" char* \"\\\"\" space")
         != std::string::npos);
  CHECK (grammar.find ("chars ::= char chars |") == std::string::npos);
  CHECK (grammar.find ("space ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}")
         != std::string::npos);
}

TEST_CASE ("LFM2 extraction grammar permits empty labels when schema allows it",
           "[deep_llm][grammar]")
{
  auto schema = BuildSupportedSchema ();
  schema["properties"]["labels"]["minItems"] = 0;
  const std::string grammar
      = cortext::internal::BuildLfm2ExtractionGrammar (schema);
  CHECK (grammar.find (
             "(nonempty-string (\",\" space nonempty-string)*)?")
         != std::string::npos);
}

TEST_CASE ("LFM2 extraction grammar rejects unsupported schema",
           "[deep_llm][grammar]")
{
  auto schema = BuildSupportedSchema ();
  schema["required"] = nlohmann::json::array ({ "labels", "extra" });
  CHECK_THROWS_AS (cortext::internal::BuildLfm2ExtractionGrammar (schema),
                   std::runtime_error);
}

TEST_CASE ("LFM2 classes stay text-only", "[deep_llm][lfm2]")
{
  TempDir temp_dir;
  const auto dummy_model
      = temp_dir.Touch ("LFM2-2.6B-Transcript-Q4_K_M.gguf");
  const auto dummy_extract
      = temp_dir.Touch ("LFM2-1.2B-Extract-Q4_K_M.gguf");

  cortext::Lfm2LlamaSummarizer summarizer (dummy_model.string ());
  cortext::Lfm2LlamaExtractor extractor (dummy_extract.string ());

  float pcm[16] = { 0 };
  std::vector<cortext::AudioSegment> segments = { { pcm, 16 } };

  CHECK_FALSE (summarizer.IsAvailable ());
  CHECK_FALSE (extractor.IsAvailable ());
  CHECK_THROWS_WITH (summarizer.SummarizeAudio (pcm, 16),
                     Catch::Matchers::ContainsSubstring ("unsupported"));
  CHECK_THROWS_WITH (summarizer.SummarizeAudioSegments (segments),
                     Catch::Matchers::ContainsSubstring ("unsupported"));
  CHECK_THROWS_WITH (
      extractor.ExtractFromAudio (pcm, 16, BuildSupportedSchema ()),
      Catch::Matchers::ContainsSubstring ("unsupported"));
}

TEST_CASE ("LFM2 summarizer integration", "[deep_llm][lfm2][integration]")
{
  const std::string model_path = FindModelPath (
      "models/LFM2-2.6B-Transcript-GGUF/LFM2-2.6B-Transcript-Q4_K_M.gguf");
  if (!std::filesystem::exists (model_path))
    {
      SUCCEED ("Skipping - model not found at " << model_path);
      return;
    }

  cortext::Lfm2LlamaSummarizer summarizer (model_path);
  if (!summarizer.IsAvailable ())
    {
      SUCCEED ("Skipping - summarizer unavailable for " << model_path);
      return;
    }

  const std::string summary = summarizer.SummarizeTextsLimited (
      {
        "chat/user: Alice is leading the Acme pilot in Seattle.",
        "chat/assistant: The customer needs SSO and audit exports before June.",
        "chat/user: Finance is revising the renewal forecast this week.",
      },
      64);

  REQUIRE_FALSE (summary.empty ());
}

TEST_CASE ("LFM2 extractor integration", "[deep_llm][lfm2][integration]")
{
  const std::string model_path = FindModelPath (
      "models/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_K_M.gguf");
  if (!std::filesystem::exists (model_path))
    {
      SUCCEED ("Skipping - model not found at " << model_path);
      return;
    }

  cortext::Lfm2LlamaExtractor extractor (model_path);
  if (!extractor.IsAvailable ())
    {
      SUCCEED ("Skipping - extractor unavailable for " << model_path);
      return;
    }

  const auto result = extractor.ExtractFromText (
      "Alice works at Acme Corp in Seattle and is planning the New York "
      "rollout with the security team.",
      BuildSupportedSchema ());

  REQUIRE_FALSE (result.labels.empty ());
  for (const auto &label : result.labels)
    {
      CHECK_FALSE (label.label.empty ());
    }
  for (const auto &relation : result.relations)
    {
      CHECK_FALSE (relation.subject.empty ());
      CHECK_FALSE (relation.predicate.empty ());
      CHECK_FALSE (relation.object.empty ());
    }

  const std::vector<std::string> benchmark_like_inputs = {
    "Summary: Alice from Acme Corp is expanding the Seattle pilot into New "
    "York. She needs SSO, audit exports, and retention controls before the "
    "June renewal.\n"
    "Source text: Alice works at Acme Corp, manages the Seattle deployment, "
    "and is coordinating the New York launch with the security team.",
    "Summary: The search outage was caused by Kafka consumer lag. Operators "
    "scaled workers, restored indexing, and now need backlog-age alerts.\n"
    "Source text: The search team increased the consumer pool from three to "
    "eight workers and replayed failed partitions without losing events.",
    "Summary: The analytics migration replaces cron with a queue-backed "
    "worker to protect the 6am reporting deadline and avoid duplicate "
    "invoices.\n"
    "Source text: Finance flagged duplicate invoice risk for enterprise "
    "accounts if replay jobs exceed the idempotency window during backfill."
  };

  for (const auto &input : benchmark_like_inputs)
    {
      const auto benchmark_result
          = extractor.ExtractFromText (input, BuildSupportedSchema ());
      REQUIRE_FALSE (benchmark_result.labels.empty ());
      for (const auto &label : benchmark_result.labels)
        {
          CHECK_FALSE (label.label.empty ());
        }
    }
}

TEST_CASE ("Mixed backend selection integration", "[deep_llm][integration]")
{
  const std::string gemma_model = FindModelPath (
      "models/gemma4-e2b-litert/gemma-4-E2B-it.litertlm");
  const std::string extract_model = FindModelPath (
      "models/lfm2.5-350m-gguf/LFM2.5-350M-Q4_K_M.gguf");

  if (!std::filesystem::exists (gemma_model)
      || !std::filesystem::exists (extract_model))
    {
      SUCCEED ("Skipping - mixed backend assets not found");
      return;
    }

  auto selection
      = cortext::internal::TryCreateDeepLlmSelection (
          FindModelPath ("models"), cortext::internal::DeepLlmBackend::Mixed,
          nullptr);
  REQUIRE (selection.has_value ());
  CHECK (selection->backend_name == "Gemma+LFM2.5");
  CHECK (selection->summarizer_model_path.filename ()
         == std::filesystem::path ("gemma-4-E2B-it.litertlm"));
  CHECK (selection->extractor_model_path.filename ()
         == std::filesystem::path ("LFM2.5-350M-Q4_K_M.gguf"));
}

TEST_CASE ("Auto deep backend prefers Gemma4 selection", "[deep_llm][integration]")
{
  ScopedEnvVar clear_backend ("CORTEXT_DEEP_LLM_BACKEND");

  const std::string gemma_model = FindModelPath (
      "models/gemma4-e2b-litert/gemma-4-E2B-it.litertlm");

  if (!std::filesystem::exists (gemma_model))
    {
      SUCCEED ("Skipping - Gemma4 backend assets not found");
      return;
    }

  auto selection = cortext::internal::TryCreateDeepLlmSelection (
      FindModelPath ("models"), cortext::internal::DeepLlmBackend::Auto,
      nullptr);
  REQUIRE (selection.has_value ());
  CHECK (selection->backend_name == "Gemma/LiteRT-LM");
  CHECK (selection->summarizer_model_path.filename ()
         == std::filesystem::path ("gemma-4-E2B-it.litertlm"));
  CHECK (selection->extractor_model_path.filename ()
         == std::filesystem::path ("gemma-4-E2B-it.litertlm"));
}
