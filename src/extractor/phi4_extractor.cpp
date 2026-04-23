#include "cortext/extractor/phi4_extractor.hpp"

#include <stdexcept>
#include <string>

#if !defined(CORTEXT_DISABLE_OGA)
#include <ort_genai.h>
#endif

namespace cortext
{

#if !defined(CORTEXT_DISABLE_OGA)

namespace
{

bool
TryCreateModelWithProvider (
    const std::string &model_path, const char *provider,
    std::unique_ptr<OgaModel> &model, std::unique_ptr<OgaTokenizer> &tokenizer,
    std::unique_ptr<OgaMultiModalProcessor> &processor,
    std::string *error_out)
{
  try
    {
      auto config = OgaConfig::Create (model_path.c_str ());
      config->ClearProviders ();
      if (provider != nullptr && provider[0] != '\0')
        {
          config->AppendProvider (provider);
          if (std::string (provider) == "cuda")
            {
              config->SetProviderOption (provider, "enable_cuda_graph", "0");
            }
        }
      model = OgaModel::Create (*config);
      tokenizer = OgaTokenizer::Create (*model);
      processor = OgaMultiModalProcessor::Create (*model);
      return true;
    }
  catch (const std::exception &e)
    {
      if (error_out != nullptr)
        {
          *error_out = e.what ();
        }
      return false;
    }
}

} // namespace

struct Phi4Extractor::Impl
{
  std::unique_ptr<OgaModel> model;
  std::unique_ptr<OgaTokenizer> tokenizer;
  std::unique_ptr<OgaMultiModalProcessor> processor;
  bool available = false;

  explicit Impl (const std::string &model_path)
  {
    try
      {
        std::string error;
        const char *kGpuProviders[] = { "cuda", "dml", "rocm", "webgpu",
                                        "openvino", "qnn", "nvtensorrtrtx" };
        for (const char *provider : kGpuProviders)
          {
            if (TryCreateModelWithProvider (model_path, provider, model,
                                            tokenizer, processor, &error))
              {
                available = true;
                return;
              }
          }
        error.clear ();
        if (TryCreateModelWithProvider (model_path, nullptr, model, tokenizer,
                                        processor, &error))
          {
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
  ExtractWithSchema (const std::string &prompt, const nlohmann::json &schema)
  {
    if (!available)
      {
        throw std::runtime_error ("Phi4Extractor: Model not available");
      }

    auto params = OgaGeneratorParams::Create (*model);
    params->SetSearchOption ("max_length", 512);
    params->SetSearchOption ("temperature", 0.0);

    // Setup llguidance JSON schema constraint
    std::string schema_str = schema.dump ();
    // Prepend x-guidance config for strict whitespace handling
    if (schema_str.size () > 1 && schema_str[0] == '{')
      {
        schema_str = R"({"x-guidance":{"whitespace_flexible":false},)"
                     + schema_str.substr (1);
      }
    std::string guidance = "start: %json " + schema_str + "\n";
    params->SetGuidance ("lark_grammar", guidance.c_str (), false);

    // Tokenize prompt
    auto sequences = OgaSequences::Create ();
    tokenizer->Encode (prompt.c_str (), *sequences);

    // Generate
    auto generator = OgaGenerator::Create (*model, *params);
    generator->AppendTokenSequences (*sequences);
    while (!generator->IsDone ())
      {
        generator->GenerateNextToken ();
      }

    // Decode output
    const int32_t *output_data = generator->GetSequenceData (0);
    size_t output_count = generator->GetSequenceCount (0);
    auto decoded = tokenizer->Decode (output_data, output_count);
    std::string output_str (decoded);

    // Parse JSON output into ExtractionResult
    operations::ExtractionResult result;
    try
      {
        auto json_output = nlohmann::json::parse (output_str);

        // Extract labels
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

        // Extract relations
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
                if (fact.contains ("valid_start_ts")
                    && fact["valid_start_ts"].is_number ())
                  {
                    f.valid_start_ts
                        = fact["valid_start_ts"].get<std::uint64_t> ();
                  }
                if (fact.contains ("valid_end_ts")
                    && fact["valid_end_ts"].is_number ())
                  {
                    f.valid_end_ts
                        = fact["valid_end_ts"].get<std::uint64_t> ();
                  }
                if (!f.subject.empty () && !f.predicate.empty ()
                    && !f.object.empty ())
                  {
                    result.facts.push_back (std::move (f));
                  }
              }
          }
      }
    catch (const nlohmann::json::exception &)
      {
        // Failed to parse JSON, return empty result
      }

    return result;
  }

  operations::ExtractionResult
  ExtractFromAudioImpl (const float *pcm, size_t num_samples,
                        const nlohmann::json &schema)
  {
    if (!available)
      {
        throw std::runtime_error ("Phi4Extractor: Model not available");
      }

    auto params = OgaGeneratorParams::Create (*model);
    params->SetSearchOption ("max_length", 512);
    params->SetSearchOption ("temperature", 0.0);

    // Setup llguidance JSON schema constraint
    std::string schema_str = schema.dump ();
    if (schema_str.size () > 1 && schema_str[0] == '{')
      {
        schema_str = R"({"x-guidance":{"whitespace_flexible":false},)"
                     + schema_str.substr (1);
      }
    std::string guidance = "start: %json " + schema_str + "\n";
    params->SetGuidance ("lark_grammar", guidance.c_str (), false);

    // Create audio input using multimodal processor
    // Phi-4 expects 16kHz mono float32 audio
    const void *audio_data[] = { pcm };
    size_t audio_sizes[] = { num_samples * sizeof (float) };
    auto audios = OgaAudios::Load (audio_data, audio_sizes, 1);

    // Build prompt with audio
    std::string prompt = "<|audio|>Extract labels, relations, and durable "
                         "facts from this audio. Return JSON with \"labels\" "
                         "as an array of strings, \"relations\" as objects "
                         "with subject/predicate/object, and optional "
                         "\"facts\" as objects with subject/predicate/object "
                         "plus optional valid_start_ts/valid_end_ts integers. "
                         "Return an empty labels array when the audio has no "
                         "durable label.<|end|><|assistant|>";
    auto inputs = processor->ProcessAudios (prompt.c_str (), audios.get ());

    // Generate
    auto generator = OgaGenerator::Create (*model, *params);
    generator->SetInputs (*inputs);
    while (!generator->IsDone ())
      {
        generator->GenerateNextToken ();
      }

    // Decode output
    const int32_t *output_data = generator->GetSequenceData (0);
    size_t output_count = generator->GetSequenceCount (0);
    auto decoded = tokenizer->Decode (output_data, output_count);
    std::string output_str (decoded);

    // Parse JSON output into ExtractionResult
    operations::ExtractionResult result;
    try
      {
        auto json_output = nlohmann::json::parse (output_str);

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
                if (fact.contains ("valid_start_ts")
                    && fact["valid_start_ts"].is_number ())
                  {
                    f.valid_start_ts
                        = fact["valid_start_ts"].get<std::uint64_t> ();
                  }
                if (fact.contains ("valid_end_ts")
                    && fact["valid_end_ts"].is_number ())
                  {
                    f.valid_end_ts
                        = fact["valid_end_ts"].get<std::uint64_t> ();
                  }
                if (!f.subject.empty () && !f.predicate.empty ()
                    && !f.object.empty ())
                  {
                    result.facts.push_back (std::move (f));
                  }
              }
          }
      }
    catch (const nlohmann::json::exception &)
      {
        // Failed to parse JSON, return empty result
      }

    return result;
  }
};

Phi4Extractor::Phi4Extractor (const std::string &model_path)
    : impl_ (std::make_unique<Impl> (model_path))
{
}

Phi4Extractor::~Phi4Extractor () = default;

Phi4Extractor::Phi4Extractor (Phi4Extractor &&) noexcept = default;
Phi4Extractor &
Phi4Extractor::operator= (Phi4Extractor &&) noexcept = default;

operations::ExtractionResult
Phi4Extractor::ExtractFromText (const std::string &text,
                                const nlohmann::json &schema)
{
  std::string prompt
      = "<|user|>Extract labels, relations, and durable facts from: " + text
        + " Return JSON with \"labels\" as an array of strings, "
          "\"relations\" as objects with subject/predicate/object, and "
          "optional \"facts\" as objects with subject/predicate/object plus "
          "optional valid_start_ts/valid_end_ts integers. "
          "Return an empty labels array when the text has no durable label."
        + "<|end|><|assistant|>";
  return impl_->ExtractWithSchema (prompt, schema);
}

operations::ExtractionResult
Phi4Extractor::ExtractFromAudio (const float *pcm, size_t num_samples,
                                 const nlohmann::json &schema)
{
  return impl_->ExtractFromAudioImpl (pcm, num_samples, schema);
}

bool
Phi4Extractor::IsAvailable () const
{
  return impl_ && impl_->available;
}

#else // CORTEXT_DISABLE_OGA

// Stub implementation when OGA is disabled
struct Phi4Extractor::Impl
{
};

Phi4Extractor::Phi4Extractor (const std::string & /*model_path*/)
    : impl_ (std::make_unique<Impl> ())
{
}

Phi4Extractor::~Phi4Extractor () = default;

Phi4Extractor::Phi4Extractor (Phi4Extractor &&) noexcept = default;
Phi4Extractor &
Phi4Extractor::operator= (Phi4Extractor &&) noexcept = default;

operations::ExtractionResult
Phi4Extractor::ExtractFromText (const std::string & /*text*/,
                                const nlohmann::json & /*schema*/)
{
  throw std::runtime_error (
      "Phi4Extractor: OGA disabled. Rebuild without CORTEXT_DISABLE_OGA");
}

operations::ExtractionResult
Phi4Extractor::ExtractFromAudio (const float * /*pcm*/, size_t /*num_samples*/,
                                 const nlohmann::json & /*schema*/)
{
  throw std::runtime_error (
      "Phi4Extractor: OGA disabled. Rebuild without CORTEXT_DISABLE_OGA");
}

bool
Phi4Extractor::IsAvailable () const
{
  return false;
}

#endif // CORTEXT_DISABLE_OGA

} // namespace cortext
