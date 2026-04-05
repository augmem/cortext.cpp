#include "cortext/extractor/gemma_extractor.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#if !defined(CORTEXT_DISABLE_LITERT)
#include "cortext/audio/gemma_audio.hpp"
#include "cortext/core/thread_config.hpp"
#include "cortext/extractor/json_schema_constraint.hpp"
#include "cortext/telemetry/telemetry.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wsign-compare"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "litert/c/litert_tensor_buffer.h"
#pragma clang diagnostic pop
#endif

namespace cortext
{

#if !defined(CORTEXT_DISABLE_LITERT)

namespace
{
std::optional<nlohmann::json>
TryParseJsonObject (const std::string &content)
{
  const auto start = content.find ('{');
  const auto end = content.rfind ('}');
  if (start == std::string::npos || end == std::string::npos || end <= start)
    {
      return std::nullopt;
    }
  try
    {
      return nlohmann::json::parse (content.substr (start, end - start + 1));
    }
  catch (const nlohmann::json::exception &)
    {
      return std::nullopt;
    }
}

std::string
BuildTextPrompt (const std::string &text)
{
  return std::string (
      "Extract labels, relations, and durable facts from the text below. "
      "Return only a single JSON object with keys \"labels\", \"relations\", and optional \"facts\". "
      "\"labels\" must be an array of non-empty strings copied from the text "
      "(no placeholders, no objects, no types/categories). "
      "Each relation must include non-empty \"subject\", \"predicate\", and "
      "\"object\" strings taken from the text. "
      "Each fact must include non-empty \"subject\", \"predicate\", and "
      "\"object\" strings taken from the text. "
      "If available, facts may also include integer millisecond "
      "\"valid_start_ts\" and \"valid_end_ts\" fields. "
      "Always return at least one label; if nothing is obvious, choose the "
      "single most salient term from the text.\n"
      "Text:\n")
      + text;
}

std::string
BuildAudioPrompt ()
{
  return std::string (
      "Extract labels, relations, and durable facts from the audio. "
      "Return only a single JSON object with keys \"labels\", \"relations\", and optional \"facts\". "
      "\"labels\" must be an array of non-empty strings drawn "
      "from the audio content (no placeholders, no objects, no types/categories). "
      "Each relation must include non-empty \"subject\", \"predicate\", and "
      "\"object\" strings drawn from the audio content. "
      "Each fact must include non-empty \"subject\", \"predicate\", and "
      "\"object\" strings drawn from the audio content. "
      "If available, facts may also include integer millisecond "
      "\"valid_start_ts\" and \"valid_end_ts\" fields. "
      "Always return at least one label; if nothing is obvious, choose the "
      "single most salient term from the audio.\n"
      "<start_of_audio>");
}

bool
TryCreateEngine (const std::string &model_path, litert::lm::Backend backend,
                 std::unique_ptr<litert::lm::Engine> &engine_out)
{
  auto model_assets = litert::lm::ModelAssets::Create (model_path);
  if (!model_assets.ok ())
    {
      return false;
    }

  auto settings = litert::lm::EngineSettings::CreateDefault (
      *std::move (model_assets), backend);
  if (!settings.ok ())
    {
      return false;
    }

  if (backend == litert::lm::Backend::CPU)
    {
      auto &executor_settings = settings->GetMutableMainExecutorSettings ();
      auto cpu_config_result
          = executor_settings.MutableBackendConfig<litert::lm::CpuConfig> ();
      if (cpu_config_result.ok ())
        {
          litert::lm::CpuConfig cpu_config = *cpu_config_result;
          cpu_config.number_of_threads = core::GetInferThreadCount ();
          executor_settings.SetBackendConfig (cpu_config);
        }
    }

  auto engine_result
      = litert::lm::Engine::CreateEngine (*std::move (settings));
  if (!engine_result.ok ())
    {
      return false;
    }

  engine_out = std::move (*engine_result);
  return true;
}

litert::TensorBuffer
BuildAudioTensor (const GemmaAudioFeatures &features)
{
  if (features.num_frames <= 0 || features.mel_bins <= 0)
    {
      throw std::runtime_error ("GemmaExtractor: Invalid audio feature shape");
    }

  const size_t element_count = features.mel_features.size ();
  const size_t byte_count = element_count * sizeof (float);
  void *host_buffer = nullptr;
  if (posix_memalign (&host_buffer, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                      byte_count)
          != 0
      || host_buffer == nullptr)
    {
      throw std::runtime_error ("GemmaExtractor: Audio buffer allocation failed");
    }
  std::memcpy (host_buffer, features.mel_features.data (), byte_count);

  LiteRtLayout layout{};
  layout.rank = 3;
  layout.has_strides = false;
  layout.dimensions[0] = 1;
  layout.dimensions[1] = static_cast<int32_t> (features.num_frames);
  layout.dimensions[2] = static_cast<int32_t> (features.mel_bins);

  LiteRtRankedTensorType tensor_type{};
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  tensor_type.layout = layout;

  LiteRtTensorBuffer buffer = nullptr;
  LiteRtStatus status = LiteRtCreateTensorBufferFromHostMemory (
      &tensor_type, host_buffer, byte_count, std::free, &buffer);
  if (status != kLiteRtStatusOk || buffer == nullptr)
    {
      std::free (host_buffer);
      throw std::runtime_error ("GemmaExtractor: Audio tensor creation failed");
    }

  return litert::TensorBuffer::WrapCObject (buffer, litert::OwnHandle::kYes);
}

operations::ExtractionResult
ParseExtractionResponse (const std::string &content)
{
  operations::ExtractionResult result;
  auto json_opt = TryParseJsonObject (content);
  if (!json_opt)
    {
      return result;
    }

  const auto &json_output = *json_opt;
  if (json_output.contains ("labels"))
    {
      for (const auto &label : json_output["labels"])
        {
          if (label.is_string ())
            {
              operations::ExtractedLabel e;
              e.label = label.get<std::string> ();
              e.salience = 0.5;
              result.labels.push_back (std::move (e));
            }
          else if (label.is_object ())
            {
              // Tolerate object-shaped labels if the model fails to follow the prompt.
              operations::ExtractedLabel e;
              e.label = label.value ("label", label.value ("name", ""));
              e.salience = 0.5;
              if (!e.label.empty ())
                {
                  result.labels.push_back (std::move (e));
                }
            }
        }
    }

  if (json_output.contains ("relations"))
    {
      for (const auto &relation : json_output["relations"])
        {
          operations::ExtractedRelation r;
          r.subject = relation.value ("subject", "");
          r.predicate = relation.value ("predicate", "");
          r.object = relation.value ("object", "");
          r.confidence = relation.value ("confidence", 0.5);
          result.relations.push_back (std::move (r));
        }
    }

  if (json_output.contains ("facts"))
    {
      for (const auto &fact : json_output["facts"])
        {
          operations::ExtractedFact f;
          f.subject = fact.value ("subject", "");
          f.predicate = fact.value ("predicate", "");
          f.object = fact.value ("object", "");
          f.confidence = fact.value ("confidence", 0.5);
          if (fact.contains ("valid_start_ts") && fact["valid_start_ts"].is_number ())
            {
              f.valid_start_ts = fact["valid_start_ts"].get<std::uint64_t> ();
            }
          if (fact.contains ("valid_end_ts") && fact["valid_end_ts"].is_number ())
            {
              f.valid_end_ts = fact["valid_end_ts"].get<std::uint64_t> ();
            }
          if (!f.subject.empty () && !f.predicate.empty () && !f.object.empty ())
            {
              result.facts.push_back (std::move (f));
            }
        }
    }

  return result;
}

bool
HasNonEmptyLabel (const operations::ExtractionResult &result)
{
  for (const auto &label : result.labels)
    {
      if (!label.label.empty ())
        {
          return true;
        }
    }
  return false;
}
} // namespace

struct GemmaExtractor::Impl
{
  std::unique_ptr<litert::lm::Engine> engine;
  bool available = false;
  std::string backend;

  explicit Impl (const std::string &model_path)
  {
    try
      {
        if (TryCreateEngine (model_path, litert::lm::Backend::CPU, engine))
          {
            backend = "cpu";
            available = true;
            return;
          }
        available = false;
      }
    catch (const std::exception &)
      {
        available = false;
      }
  }

  operations::ExtractionResult
  ExtractFromTextImpl (const std::string &text,
                       const nlohmann::json &schema)
  {
    if (!available || !engine)
      {
        throw std::runtime_error ("GemmaExtractor: Model not available");
      }

    constexpr int kMaxAttempts = 2;
    const std::string prompt = BuildTextPrompt (text);

    for (int attempt = 0; attempt < kMaxAttempts; ++attempt)
      {
        litert::lm::SessionConfig session_config
            = litert::lm::SessionConfig::CreateDefault ();
#if defined(__APPLE__)
        session_config.SetSamplerBackend (litert::lm::Backend::CPU);
#endif

        auto session_result = engine->CreateSession (session_config);
        if (!session_result.ok ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                throw std::runtime_error (
                    "GemmaExtractor: Failed to create session");
              }
            continue;
          }

        std::vector<litert::lm::InputData> inputs;
        inputs.emplace_back (litert::lm::InputText (prompt));

        auto prefill_status = (*session_result)->RunPrefill (inputs);
        if (!prefill_status.ok ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                const std::string err = prefill_status.ToString ();
                telemetry::LogWarn (
                    "cortext.extractor_prefill_failed",
                    { telemetry::Attribute::String ("backend", backend),
                      telemetry::Attribute::Int64 ("attempt", attempt + 1),
                      telemetry::Attribute::String ("error", err) });
                if (std::getenv ("CORTEXT_EXTRACTOR_DEBUG") != nullptr)
                  {
                    std::cerr << "GemmaExtractor prefill failed (backend="
                              << backend << "): " << err << std::endl;
                  }
                throw std::runtime_error ("GemmaExtractor: Prefill failed");
              }
            continue;
          }

        constexpr int kConstraintVocabSize = 1000000;
        auto &tokenizer = const_cast<litert::lm::Tokenizer &> (
            (*session_result)->GetTokenizer ());
        extractor::JsonSchemaConstraint constraint (schema, tokenizer,
                                                     kConstraintVocabSize);
        auto decode_config = litert::lm::DecodeConfig::CreateDefault ();
        decode_config.SetConstraint (&constraint);

        auto response = (*session_result)->RunDecode (decode_config);
        if (!response.ok ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                const std::string err = response.status ().ToString ();
                telemetry::LogWarn (
                    "cortext.extractor_decode_failed",
                    { telemetry::Attribute::String ("backend", backend),
                      telemetry::Attribute::Int64 ("attempt", attempt + 1),
                      telemetry::Attribute::String ("error", err) });
                if (std::getenv ("CORTEXT_EXTRACTOR_DEBUG") != nullptr)
                  {
                    std::cerr << "GemmaExtractor decode failed (backend="
                              << backend << "): " << err << std::endl;
                  }
                throw std::runtime_error ("GemmaExtractor: Decode failed");
              }
            continue;
          }

        const auto &texts = response->GetTexts ();
        if (texts.empty ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                telemetry::LogWarn (
                    "cortext.extractor_empty_output",
                    { telemetry::Attribute::String ("backend", backend),
                      telemetry::Attribute::Int64 ("attempt", attempt + 1) });
                if (std::getenv ("CORTEXT_EXTRACTOR_DEBUG") != nullptr)
                  {
                    std::cerr << "GemmaExtractor empty decode output (backend="
                              << backend << ")" << std::endl;
                  }
                throw std::runtime_error ("GemmaExtractor: Empty decode output");
              }
            continue;
          }

        if (std::getenv ("CORTEXT_EXTRACTOR_DEBUG") != nullptr)
          {
            std::cerr << "GemmaExtractor response (text): " << texts.front ()
                      << std::endl;
          }

        auto parsed = ParseExtractionResponse (texts.front ());
        if (HasNonEmptyLabel (parsed))
          {
            return parsed;
          }
      }

    telemetry::LogWarn (
        "cortext.extractor_invalid_output",
        { telemetry::Attribute::String ("backend", backend) });
    if (std::getenv ("CORTEXT_EXTRACTOR_DEBUG") != nullptr)
      {
        std::cerr << "GemmaExtractor invalid constrained labels (backend="
                  << backend << ")" << std::endl;
      }
    throw std::runtime_error (
        "GemmaExtractor: Failed to produce valid constrained labels");
  }

  operations::ExtractionResult
  ExtractFromAudioImpl (const float *pcm, size_t num_samples,
                        const nlohmann::json &schema)
  {
    if (!available || !engine)
      {
        throw std::runtime_error ("GemmaExtractor: Model not available");
      }

    // Preprocess audio to mel-spectrogram
    auto features = ExtractGemmaAudioFeatures (pcm, num_samples);

    constexpr int kMaxAttempts = 2;
    const std::string prompt = BuildAudioPrompt ();

    for (int attempt = 0; attempt < kMaxAttempts; ++attempt)
      {
        // Create session config with audio modality enabled
        litert::lm::SessionConfig session_config
            = litert::lm::SessionConfig::CreateDefault ();
        session_config.SetAudioModalityEnabled (true);
#if defined(__APPLE__)
        session_config.SetSamplerBackend (litert::lm::Backend::CPU);
#endif

        auto session_result = engine->CreateSession (session_config);
        if (!session_result.ok ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                throw std::runtime_error (
                    "GemmaExtractor: Failed to create session");
              }
            continue;
          }

        litert::TensorBuffer mel_tensor = BuildAudioTensor (features);

        std::vector<litert::lm::InputData> inputs;
        inputs.emplace_back (litert::lm::InputText (prompt));
        inputs.emplace_back (litert::lm::InputAudio (std::move (mel_tensor)));
        inputs.emplace_back (litert::lm::InputAudioEnd ());

        auto prefill_status = (*session_result)->RunPrefill (inputs);
        if (!prefill_status.ok ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                const std::string err = prefill_status.ToString ();
                telemetry::LogWarn (
                    "cortext.extractor_prefill_failed",
                    { telemetry::Attribute::String ("backend", backend),
                      telemetry::Attribute::Int64 ("attempt", attempt + 1),
                      telemetry::Attribute::String ("error", err) });
                if (std::getenv ("CORTEXT_EXTRACTOR_DEBUG") != nullptr)
                  {
                    std::cerr << "GemmaExtractor audio prefill failed (backend="
                              << backend << "): " << err << std::endl;
                  }
                throw std::runtime_error (
                    "GemmaExtractor: Audio prefill failed");
              }
            continue;
          }

        constexpr int kConstraintVocabSize = 1000000;
        auto &tokenizer = const_cast<litert::lm::Tokenizer &> (
            (*session_result)->GetTokenizer ());
        extractor::JsonSchemaConstraint constraint (schema, tokenizer,
                                                     kConstraintVocabSize);
        auto decode_config = litert::lm::DecodeConfig::CreateDefault ();
        decode_config.SetConstraint (&constraint);

        auto response = (*session_result)->RunDecode (decode_config);
        if (!response.ok ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                const std::string err = response.status ().ToString ();
                telemetry::LogWarn (
                    "cortext.extractor_decode_failed",
                    { telemetry::Attribute::String ("backend", backend),
                      telemetry::Attribute::Int64 ("attempt", attempt + 1),
                      telemetry::Attribute::String ("error", err) });
                if (std::getenv ("CORTEXT_EXTRACTOR_DEBUG") != nullptr)
                  {
                    std::cerr << "GemmaExtractor audio decode failed (backend="
                              << backend << "): " << err << std::endl;
                  }
                throw std::runtime_error (
                    "GemmaExtractor: Audio decode failed");
              }
            continue;
          }

        const auto &texts = response->GetTexts ();
        if (texts.empty ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                telemetry::LogWarn (
                    "cortext.extractor_empty_output",
                    { telemetry::Attribute::String ("backend", backend),
                      telemetry::Attribute::Int64 ("attempt", attempt + 1) });
                if (std::getenv ("CORTEXT_EXTRACTOR_DEBUG") != nullptr)
                  {
                    std::cerr << "GemmaExtractor empty audio decode output "
                                 "(backend="
                              << backend << ")" << std::endl;
                  }
                throw std::runtime_error (
                    "GemmaExtractor: Empty audio decode output");
              }
            continue;
          }

        if (std::getenv ("CORTEXT_EXTRACTOR_DEBUG") != nullptr)
          {
            std::cerr << "GemmaExtractor response (audio): " << texts.front ()
                      << std::endl;
          }

        auto parsed = ParseExtractionResponse (texts.front ());
        if (HasNonEmptyLabel (parsed))
          {
            return parsed;
          }
      }

    telemetry::LogWarn (
        "cortext.extractor_invalid_output",
        { telemetry::Attribute::String ("backend", backend) });
    if (std::getenv ("CORTEXT_EXTRACTOR_DEBUG") != nullptr)
      {
        std::cerr << "GemmaExtractor invalid constrained labels (backend="
                  << backend << ")" << std::endl;
      }
    throw std::runtime_error (
        "GemmaExtractor: Failed to produce valid constrained labels");
  }
};

GemmaExtractor::GemmaExtractor (const std::string &model_path)
    : impl_ (std::make_unique<Impl> (model_path))
{
}

GemmaExtractor::~GemmaExtractor () = default;

GemmaExtractor::GemmaExtractor (GemmaExtractor &&) noexcept = default;
GemmaExtractor &
GemmaExtractor::operator= (GemmaExtractor &&) noexcept = default;

operations::ExtractionResult
GemmaExtractor::ExtractFromText (const std::string &text,
                                 const nlohmann::json &schema)
{
  return impl_->ExtractFromTextImpl (text, schema);
}

operations::ExtractionResult
GemmaExtractor::ExtractFromAudio (const float *pcm, size_t num_samples,
                                  const nlohmann::json &schema)
{
  return impl_->ExtractFromAudioImpl (pcm, num_samples, schema);
}

bool
GemmaExtractor::IsAvailable () const
{
  return impl_ && impl_->available;
}

#else // CORTEXT_DISABLE_LITERT

// Stub implementation when LiteRT-LM is disabled
struct GemmaExtractor::Impl
{
};

GemmaExtractor::GemmaExtractor (const std::string & /*model_path*/)
    : impl_ (std::make_unique<Impl> ())
{
}

GemmaExtractor::~GemmaExtractor () = default;

GemmaExtractor::GemmaExtractor (GemmaExtractor &&) noexcept = default;
GemmaExtractor &
GemmaExtractor::operator= (GemmaExtractor &&) noexcept = default;

operations::ExtractionResult
GemmaExtractor::ExtractFromText (const std::string & /*text*/,
                                 const nlohmann::json & /*schema*/)
{
  throw std::runtime_error (
      "GemmaExtractor: LiteRT-LM disabled. Rebuild without CORTEXT_DISABLE_LITERT");
}

operations::ExtractionResult
GemmaExtractor::ExtractFromAudio (const float * /*pcm*/, size_t /*num_samples*/,
                                  const nlohmann::json & /*schema*/)
{
  throw std::runtime_error (
      "GemmaExtractor: LiteRT-LM disabled. Rebuild without CORTEXT_DISABLE_LITERT");
}

bool
GemmaExtractor::IsAvailable () const
{
  return false;
}

#endif // CORTEXT_DISABLE_LITERT

} // namespace cortext
