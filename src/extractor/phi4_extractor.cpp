#include "cortext/extractor/phi4_extractor.hpp"

#include <stdexcept>

#if !defined(CORTEXT_DISABLE_OGA)
#include <ort_genai.h>
#endif

namespace cortext
{

#if !defined(CORTEXT_DISABLE_OGA)

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
        model = OgaModel::Create (model_path.c_str ());
        tokenizer = OgaTokenizer::Create (*model);
        processor = OgaMultiModalProcessor::Create (*model);
        available = true;
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

        // Extract entities
        if (json_output.contains ("entities"))
          {
            for (const auto &entity : json_output["entities"])
              {
                operations::ExtractedEntity e;
                e.name = entity.value ("name", "");
                e.type = entity.value ("type", "");
                e.salience = entity.value ("salience", 0.5);
                result.entities.push_back (std::move (e));
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
    std::string prompt = "<|audio|>Extract entities and relations from this "
                         "audio.<|end|><|assistant|>";
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

        if (json_output.contains ("entities"))
          {
            for (const auto &entity : json_output["entities"])
              {
                operations::ExtractedEntity e;
                e.name = entity.value ("name", "");
                e.type = entity.value ("type", "");
                e.salience = entity.value ("salience", 0.5);
                result.entities.push_back (std::move (e));
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
      = "<|user|>Extract entities and relations from: " + text
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
