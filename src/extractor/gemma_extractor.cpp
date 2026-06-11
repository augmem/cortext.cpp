#include "cortext/extractor/gemma_extractor.hpp"

#include "cortext/core/knobs.hpp"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

#if !defined(CORTEXT_DISABLE_LITERT)
#include "cortext/core/thread_config.hpp"
#include "cortext/extractor/json_schema_constraint.hpp"
#include "cortext/telemetry/telemetry.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wsign-compare"
#include "runtime/components/preprocessor/audio_preprocessor_miniaudio.h"
#include "runtime/components/preprocessor/stb_image_preprocessor.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
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
std::string
Trim (const std::string &text)
{
  size_t begin = 0;
  while (begin < text.size ()
         && std::isspace (static_cast<unsigned char> (text[begin])))
    {
      ++begin;
    }
  size_t end = text.size ();
  while (end > begin
         && std::isspace (static_cast<unsigned char> (text[end - 1])))
    {
      --end;
    }
  return text.substr (begin, end - begin);
}

bool
LooksLikeConcreteLooseLabel (const std::string &label)
{
  return label.find (' ') != std::string::npos
         || (!label.empty ()
             && std::isupper (static_cast<unsigned char> (label.front ())));
}

std::vector<std::string>
ParseLooseBraceLabels (const std::string &content)
{
  const size_t begin = content.find ('{');
  const size_t end = content.find ('}', begin == std::string::npos ? 0 : begin);
  if (begin == std::string::npos || end == std::string::npos || end <= begin)
    {
      return {};
    }

  std::vector<std::string> labels;
  std::stringstream stream (content.substr (begin + 1, end - begin - 1));
  std::string item;
  while (std::getline (stream, item, ','))
    {
      std::string label = Trim (item);
      const size_t colon = label.find (':');
      if (colon != std::string::npos)
        {
          label = Trim (label.substr (colon + 1));
        }
      if (!label.empty () && LooksLikeConcreteLooseLabel (label))
        {
          labels.push_back (label);
        }
    }
  return labels;
}

std::optional<nlohmann::json>
TryParseJsonObject (const std::string &content)
{
  for (std::size_t start = content.find ('{'); start != std::string::npos;
       start = content.find ('{', start + 1))
    {
      int depth = 0;
      bool in_string = false;
      bool escaped = false;
      for (std::size_t pos = start; pos < content.size (); ++pos)
        {
          const char c = content[pos];
          if (escaped)
            {
              escaped = false;
              continue;
            }
          if (c == '\\' && in_string)
            {
              escaped = true;
              continue;
            }
          if (c == '"')
            {
              in_string = !in_string;
              continue;
            }
          if (in_string)
            {
              continue;
            }
          if (c == '{')
            {
              ++depth;
            }
          else if (c == '}')
            {
              --depth;
              if (depth == 0)
                {
                  try
                    {
                      return nlohmann::json::parse (
                          content.substr (start, pos - start + 1));
                    }
                  catch (const nlohmann::json::exception &)
                    {
                      break;
                    }
                }
            }
        }
    }
  return std::nullopt;
}

std::string
BuildTextPrompt (const std::string &text)
{
  return std::string (
      "Extract labels, relations, and durable facts from the text below. "
      "Return only a single JSON object with keys \"labels\", \"relations\", and \"facts\". "
      "Always include all three keys; \"relations\" and \"facts\" must be empty arrays when the evidence states none. "
      "\"labels\" must be an array of non-empty strings copied from the text "
      "(no placeholders, no objects, no types/categories). "
      "Each relation must include non-empty \"subject\", \"predicate\", and "
      "\"object\" strings taken from the text. "
      "Each fact must include non-empty \"subject\", \"predicate\", and "
      "\"object\" strings taken from the text. "
      "If available, facts may also include integer millisecond "
      "\"valid_start_ts\" and \"valid_end_ts\" fields. "
      "Return an empty labels array when the text has no durable label.\n"
      "Text:\n")
      + text;
}

std::string
BuildAudioPrompt ()
{
  return std::string (
      "Extract labels, relations, and durable facts from the audio. "
      "Return only a single JSON object with keys \"labels\", \"relations\", and \"facts\". "
      "Always include all three keys; \"relations\" and \"facts\" must be empty arrays when the evidence states none. "
      "\"labels\" must be an array of non-empty strings drawn "
      "from the audio content (no placeholders, no objects, no types/categories). "
      "Each relation must include non-empty \"subject\", \"predicate\", and "
      "\"object\" strings drawn from the audio content. "
      "Each fact must include non-empty \"subject\", \"predicate\", and "
      "\"object\" strings drawn from the audio content. "
      "If available, facts may also include integer millisecond "
      "\"valid_start_ts\" and \"valid_end_ts\" fields. "
      "Return an empty labels array when the audio has no durable label.\n"
      "<start_of_audio>");
}

std::string
FormatCurrentLabels (const std::vector<std::string> &current_labels)
{
  if (current_labels.empty ())
    {
      return "none";
    }
  std::ostringstream out;
  for (size_t i = 0; i < current_labels.size (); ++i)
    {
      if (i > 0)
        {
          out << ", ";
        }
      out << current_labels[i];
    }
  return out.str ();
}

std::string
LabelPromptBoundsInstruction (const char *evidence_name)
{
  constexpr double kNeutralKnob = 0.5;
  const int min_labels = core::STMLTMDurableMinLabels (
      kNeutralKnob, kNeutralKnob, kNeutralKnob);
  const int max_labels = core::STMLTMDurableMaxLabels (
      kNeutralKnob, kNeutralKnob, kNeutralKnob);
  const int min_words = core::STMLTMLabelPromptMinWords (
      kNeutralKnob, kNeutralKnob, kNeutralKnob);
  const int max_words = core::STMLTMLabelPromptMaxWords (
      kNeutralKnob, kNeutralKnob, kNeutralKnob);

  std::ostringstream out;
  out << "When " << evidence_name << " contains enough anchors, return "
      << min_labels << "-" << max_labels << " labels. Prefer " << min_words
      << "-" << max_words
      << " word noun/event phrases unless the label is a proper name or a "
         "concrete object. ";
  return out.str ();
}

std::string
BuildLabelRefinementTextPrompt (
    const std::string &text,
    const std::vector<std::string> &current_labels)
{
  return std::string (
      "You are refining labels for one memory graph association. "
      "Use only the evidence to decide the final durable labels. "
	      "Current labels are untrusted candidates and may be wrong. "
	      "Keep a current label only if it appears in the evidence; remove every "
	      "unsupported or generic current label, and add missing concrete labels. "
		      "Return only a single JSON object with keys \"labels\", \"relations\", and \"facts\". "
      "Always include all three keys; \"relations\" and \"facts\" must be empty arrays when the evidence states none. "
		      "\"labels\" must be the final replacement set of non-empty source spans copied from the evidence. "
		      "Use durable memory anchors: named people, places, organizations, pets, specific objects, and short event phrases. "
		      + LabelPromptBoundsInstruction ("evidence")
		      +
		      "Every important word in a label must appear verbatim in the evidence; do not paraphrase, infer, summarize, or invent labels. "
	      "Do not return pronouns, chat roles, filler words, helper verbs, modal words, status phrases, or generic labels such as that, this, thing, get, go, make, might, idea, food, stuff, almost done, user, assistant. "
	      "\"facts\" should contain directly stated durable or episodic assertions that a person would later ask about: named people/pets, actions, preferences, locations, ownership, names, relationships, plans, and important events. "
	      "Each fact must use concise strings from the evidence, for example {\"subject\":\"Amelia\",\"predicate\":\"gave\",\"object\":\"money\"}. "
	      "Do not add facts for vague filler, uncertainty, reactions, generic categories, or anything not directly stated. "
	      "Every relation predicate must be one of: co_occurs, implies, contradicts, reinforces, causes, similar_to. "
	      "Every relation subject and object must exactly match one string in the final labels array; omit any relation whose endpoint is not a final label. "
	      "Use relations only for clear links stated by the evidence; otherwise return an empty relations array. "
	      "Use JSON array syntax exactly, for example {\"labels\":[\"Maria\",\"Bailey dog\",\"car crash\"],\"relations\":[],\"facts\":[{\"subject\":\"Maria\",\"predicate\":\"trained\",\"object\":\"Bailey dog\"}]}. "
	      "Do not return bare objects such as {dog: Bailey}. "
	      "Return an empty labels array when the evidence has no durable label.\n"
      "Current labels: ")
      + FormatCurrentLabels (current_labels) + "\nEvidence:\n" + text;
}

std::string
BuildLabelRefinementAudioPrompt (
    const std::vector<std::string> &current_labels)
{
  return std::string (
      "You are refining labels for one memory graph association from audio evidence. "
	      "Keep correct current labels, remove unsupported or generic labels, "
	      "and add missing concrete labels from the audio. "
		      "Return only a single JSON object with keys \"labels\", \"relations\", and \"facts\". "
      "Always include all three keys; \"relations\" and \"facts\" must be empty arrays when the evidence states none. "
		      "\"labels\" must be the final replacement set of non-empty source spans from the audio. "
		      "Use durable memory anchors: spoken names, places, organizations, pets, specific objects, and short event phrases. "
		      + LabelPromptBoundsInstruction ("audio")
		      +
		      "Every important word in a label must be spoken in the audio; do not paraphrase, infer, summarize, or invent labels. "
	      "Do not return pronouns, chat roles, filler words, helper verbs, modal words, status phrases, or generic labels such as that, this, thing, get, go, make, might, idea, food, stuff, almost done, user, assistant. "
	      "\"facts\" should contain directly stated durable or episodic assertions that a person would later ask about: named people/pets, actions, preferences, locations, ownership, names, relationships, plans, and important events. "
	      "Each fact must use concise strings from the audio, for example {\"subject\":\"Bailey\",\"predicate\":\"loves\",\"object\":\"tennis balls\"}. "
	      "Do not add facts for vague filler, uncertainty, reactions, generic categories, or anything not directly stated. "
	      "Every relation predicate must be one of: co_occurs, implies, contradicts, reinforces, causes, similar_to. "
	      "Every relation subject and object must exactly match one string in the final labels array; omit any relation whose endpoint is not a final label. "
	      "Use relations only for clear links stated by the audio; otherwise return an empty relations array. "
	      "Use JSON array syntax exactly, for example {\"labels\":[\"Bailey dog\",\"car crash\"],\"relations\":[],\"facts\":[{\"subject\":\"Bailey\",\"predicate\":\"loves\",\"object\":\"tennis balls\"}]}. "
	      "If this is speech, include spoken names, subjects, objects, places, and events. "
      "Return an empty labels array when the audio has no durable label.\n"
      "Current labels: ")
      + FormatCurrentLabels (current_labels) + "\n<start_of_audio>";
}

std::string
BuildLabelRefinementImagePrompt (
    const std::vector<std::string> &current_labels)
{
  return std::string (
      "You are refining labels for one memory graph association from image evidence. "
	      "Keep correct current labels, remove unsupported or generic labels, "
	      "and add missing concrete labels visible in the image. "
		      "Return only a single JSON object with keys \"labels\", \"relations\", and \"facts\". "
      "Always include all three keys; \"relations\" and \"facts\" must be empty arrays when the evidence states none. "
		      "\"labels\" must be the final replacement set of non-empty visible source spans. "
		      "Use durable memory anchors: visible people, places, organizations, pets, specific objects, and short event phrases. "
		      + LabelPromptBoundsInstruction ("image evidence")
		      +
		      "Every important word in a label must be visible or readable in the image; do not paraphrase, infer, summarize, or invent labels. "
	      "Do not return pronouns, chat roles, filler words, helper verbs, modal words, status phrases, or generic labels such as that, this, thing, get, go, make, might, idea, food, stuff, almost done, user, assistant. "
	      "\"facts\" should contain only directly visible/readable durable assertions such as names, locations, ownership, relationships, and important events. "
	      "Each fact must use concise visible/readable evidence strings. "
	      "Do not add facts for inferred intent, vague scenes, generic categories, or anything not directly visible/readable. "
	      "Every relation predicate must be one of: co_occurs, implies, contradicts, reinforces, causes, similar_to. "
	      "Every relation subject and object must exactly match one string in the final labels array; omit any relation whose endpoint is not a final label. "
	      "Use relations only for clear links visible in the image; otherwise return an empty relations array. "
	      "Use JSON array syntax exactly, for example {\"labels\":[\"golden retriever\",\"park sign\"],\"relations\":[],\"facts\":[]}. "
	      "Prefer visible people, objects, places, text, and events. "
      "Return an empty labels array when the image has no durable label.\n"
      "Current labels: ")
      + FormatCurrentLabels (current_labels) + "\n<start_of_image>";
}

litert::lm::AudioPreprocessorConfig
BuildGemma4AudioPreprocessorConfig ()
{
  return litert::lm::AudioPreprocessorConfig::CreateDefaultUsmConfig ();
}

litert::lm::ImagePreprocessParameter
BuildGemma4ImagePreprocessParameter ()
{
  litert::lm::ImagePreprocessParameter params;
  params.SetPatchifyConfig (
      litert::lm::ImagePreprocessParameter::PatchifyConfig {
          .patch_width = 16,
          .patch_height = 16,
          .max_num_patches = 2520,
          .pooling_kernel_size = 2 });
  return params;
}

litert::lm::InputAudio
PreprocessGemma4Audio (const float *pcm, size_t num_samples)
{
  auto preprocessor_result = litert::lm::AudioPreprocessorMiniAudio::Create (
      BuildGemma4AudioPreprocessorConfig ());
  if (!preprocessor_result.ok ())
    {
      throw std::runtime_error (
          "GemmaExtractor: Failed to create audio preprocessor");
    }

  std::vector<float> pcm_frames (pcm, pcm + num_samples);
  auto processed_audio = (*preprocessor_result)
                             ->Preprocess (litert::lm::InputAudio (
                                 std::move (pcm_frames)));
  (*preprocessor_result)->Reset ();
  if (!processed_audio.ok ())
    {
      throw std::runtime_error (
          "GemmaExtractor: Audio preprocessing failed");
    }
  return std::move (*processed_audio);
}

litert::lm::InputImage
PreprocessGemma4Image (const std::vector<unsigned char> &image_bytes)
{
  litert::lm::StbImagePreprocessor preprocessor;
  const std::string raw_image (
      reinterpret_cast<const char *> (image_bytes.data ()),
      image_bytes.size ());
  auto processed_image = preprocessor.Preprocess (
      litert::lm::InputImage (raw_image),
      BuildGemma4ImagePreprocessParameter ());
  if (!processed_image.ok ())
    {
      throw std::runtime_error (
          "GemmaExtractor: Image preprocessing failed");
    }
  return std::move (*processed_image);
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

  auto &executor_settings = settings->GetMutableMainExecutorSettings ();
  executor_settings.SetMaxNumImages (1);
  executor_settings.SetMaxNumTokens (2048);
  if (backend == litert::lm::Backend::CPU)
    {
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
      = litert::lm::EngineFactory::CreateDefault (*std::move (settings));
  if (!engine_result.ok ())
    {
      return false;
    }

  engine_out = std::move (*engine_result);
  return true;
}

bool
TryCreateMultimodalEngine (const std::string &model_path,
                           std::unique_ptr<litert::lm::Engine> &engine_out)
{
  auto model_assets = litert::lm::ModelAssets::Create (model_path);
  if (!model_assets.ok ())
    {
      return false;
    }

  auto settings = litert::lm::EngineSettings::CreateDefault (
      *std::move (model_assets), litert::lm::Backend::CPU,
      litert::lm::Backend::GPU, litert::lm::Backend::CPU);
  if (!settings.ok ())
    {
      return false;
    }

  auto &executor_settings = settings->GetMutableMainExecutorSettings ();
  executor_settings.SetMaxNumImages (1);
  executor_settings.SetMaxNumTokens (2048);
  auto cpu_config_result
      = executor_settings.MutableBackendConfig<litert::lm::CpuConfig> ();
  if (cpu_config_result.ok ())
    {
      litert::lm::CpuConfig cpu_config = *cpu_config_result;
      cpu_config.number_of_threads = core::GetInferThreadCount ();
      executor_settings.SetBackendConfig (cpu_config);
    }

  auto engine_result
      = litert::lm::EngineFactory::CreateDefault (*std::move (settings));
  if (!engine_result.ok ())
    {
      return false;
    }

  engine_out = std::move (*engine_result);
  return true;
}

} // namespace

// External linkage: shared with providers::ProviderExtractor (see
// cortext/extractor/extraction_parsing.hpp) so every transport parses model
// output identically.
operations::ExtractionResult
ParseExtractionResponse (const std::string &content)
{
  operations::ExtractionResult result;
  auto json_opt = TryParseJsonObject (content);
  if (!json_opt)
    {
      for (const auto &label : ParseLooseBraceLabels (content))
        {
          result.labels.push_back ({ label, 0.5 });
        }
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

  if (result.labels.empty () && json_output.contains ("subject")
      && json_output.contains ("object") && json_output["subject"].is_string ()
      && json_output["object"].is_string ())
    {
      const std::string subject = json_output.value ("subject", "");
      const std::string object = json_output.value ("object", "");
      if (!subject.empty ())
        {
          result.labels.push_back ({ subject, 0.5 });
        }
      if (!object.empty () && object != subject)
        {
          result.labels.push_back ({ object, 0.5 });
        }
    }

  if (json_output.contains ("relations"))
    {
      for (const auto &relation : json_output["relations"])
        {
          if (!relation.is_object ())
            {
              continue;
            }
          operations::ExtractedRelation r;
          r.subject = relation.value ("subject", "");
          r.predicate = relation.value ("predicate", "");
          r.object = relation.value ("object", "");
          r.confidence = relation.value ("confidence", 0.5);
          result.relations.push_back (std::move (r));
        }
    }
  else if (json_output.contains ("subject") && json_output.contains ("object")
           && json_output["subject"].is_string ()
           && json_output["object"].is_string ())
    {
      operations::ExtractedRelation r;
      r.subject = json_output.value ("subject", "");
      r.predicate = json_output.value ("predicate", "");
      r.object = json_output.value ("object", "");
      r.confidence = json_output.value ("confidence", 0.5);
      if (!r.subject.empty () && !r.predicate.empty () && !r.object.empty ())
        {
          result.relations.push_back (std::move (r));
        }
    }

  if (json_output.contains ("facts"))
    {
      for (const auto &fact : json_output["facts"])
        {
          if (!fact.is_object ())
            {
              continue;
            }
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

namespace
{

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

bool
SchemaAllowsEmptyLabels (const nlohmann::json &schema)
{
  if (!schema.contains ("properties") || !schema["properties"].is_object ())
    {
      return false;
    }
  const auto &properties = schema["properties"];
  if (!properties.contains ("labels") || !properties["labels"].is_object ())
    {
      return false;
    }
  return properties["labels"].value ("minItems", 0) == 0;
}
} // namespace

struct GemmaExtractor::Impl
{
  std::unique_ptr<litert::lm::Engine> engine;
  std::unique_ptr<litert::lm::Engine> multimodal_engine;
  bool available = false;
  std::string backend;
  std::string model_path;

  explicit Impl (const std::string &model_path)
      : model_path (model_path)
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
  RunTextPrompt (const std::string &prompt, const nlohmann::json &schema)
  {
    if (!available || !engine)
      {
        throw std::runtime_error ("GemmaExtractor: Model not available");
      }

    constexpr int kMaxAttempts = 2;
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
            engine->GetTokenizer ());
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
        if (HasNonEmptyLabel (parsed) || SchemaAllowsEmptyLabels (schema))
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
  ExtractFromTextImpl (const std::string &text,
                       const nlohmann::json &schema)
  {
    return RunTextPrompt (BuildTextPrompt (text), schema);
  }

  operations::ExtractionResult
  RefineLabelsFromTextImpl (const std::string &text,
                            const std::vector<std::string> &current_labels,
                            const nlohmann::json &schema)
  {
    return RunTextPrompt (BuildLabelRefinementTextPrompt (text, current_labels),
                          schema);
  }

  operations::ExtractionResult
  ExtractFromAudioImpl (const float *pcm, size_t num_samples,
                        const nlohmann::json &schema)
  {
    if (!multimodal_engine)
      {
        (void)TryCreateMultimodalEngine (model_path, multimodal_engine);
      }
    auto *active_engine = multimodal_engine ? multimodal_engine.get ()
                                            : engine.get ();
    if (!available || !active_engine)
      {
        throw std::runtime_error ("GemmaExtractor: Model not available");
      }

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

        auto session_result = active_engine->CreateSession (session_config);
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
        inputs.emplace_back (PreprocessGemma4Audio (pcm, num_samples));
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
            active_engine->GetTokenizer ());
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
        if (HasNonEmptyLabel (parsed) || SchemaAllowsEmptyLabels (schema))
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
  RefineLabelsFromAudioImpl (const float *pcm, size_t num_samples,
                             const std::vector<std::string> &current_labels,
                             const nlohmann::json &schema)
  {
    if (!multimodal_engine)
      {
        (void)TryCreateMultimodalEngine (model_path, multimodal_engine);
      }
    auto *active_engine = multimodal_engine ? multimodal_engine.get ()
                                            : engine.get ();
    if (!available || !active_engine)
      {
        throw std::runtime_error ("GemmaExtractor: Model not available");
      }

    constexpr int kMaxAttempts = 2;
    const std::string prompt = BuildLabelRefinementAudioPrompt (current_labels);

    for (int attempt = 0; attempt < kMaxAttempts; ++attempt)
      {
        litert::lm::SessionConfig session_config
            = litert::lm::SessionConfig::CreateDefault ();
        session_config.SetAudioModalityEnabled (true);
#if defined(__APPLE__)
        session_config.SetSamplerBackend (litert::lm::Backend::CPU);
#endif

        auto session_result = active_engine->CreateSession (session_config);
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
        inputs.emplace_back (PreprocessGemma4Audio (pcm, num_samples));
        inputs.emplace_back (litert::lm::InputAudioEnd ());

        auto prefill_status = (*session_result)->RunPrefill (inputs);
        if (!prefill_status.ok ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                throw std::runtime_error (
                    "GemmaExtractor: Audio label refinement prefill failed");
              }
            continue;
          }

        constexpr int kConstraintVocabSize = 1000000;
        auto &tokenizer = const_cast<litert::lm::Tokenizer &> (
            active_engine->GetTokenizer ());
        extractor::JsonSchemaConstraint constraint (schema, tokenizer,
                                                     kConstraintVocabSize);
        auto decode_config = litert::lm::DecodeConfig::CreateDefault ();
        decode_config.SetConstraint (&constraint);
        auto response = (*session_result)->RunDecode (decode_config);
        if (!response.ok ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                throw std::runtime_error (
                    "GemmaExtractor: Audio label refinement decode failed");
              }
            continue;
          }
        const auto &texts = response->GetTexts ();
        if (texts.empty ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                throw std::runtime_error (
                    "GemmaExtractor: Empty audio label refinement output");
              }
            continue;
          }
        auto parsed = ParseExtractionResponse (texts.front ());
        if (HasNonEmptyLabel (parsed) || SchemaAllowsEmptyLabels (schema))
          {
            return parsed;
          }
      }
    throw std::runtime_error (
        "GemmaExtractor: Failed to refine labels from audio");
  }

  operations::ExtractionResult
  RefineLabelsFromImageImpl (const std::vector<unsigned char> &image_bytes,
                             const std::vector<std::string> &current_labels,
                             const nlohmann::json &schema)
  {
    if (!multimodal_engine)
      {
        (void)TryCreateMultimodalEngine (model_path, multimodal_engine);
      }
    auto *active_engine = multimodal_engine ? multimodal_engine.get ()
                                            : engine.get ();
    if (!available || !active_engine)
      {
        throw std::runtime_error ("GemmaExtractor: Model not available");
      }
    if (image_bytes.empty ())
      {
        throw std::runtime_error ("GemmaExtractor: Empty image bytes");
      }

    constexpr int kMaxAttempts = 2;
    const std::string prompt = BuildLabelRefinementImagePrompt (current_labels);

    for (int attempt = 0; attempt < kMaxAttempts; ++attempt)
      {
        litert::lm::SessionConfig session_config
            = litert::lm::SessionConfig::CreateDefault ();
        session_config.SetVisionModalityEnabled (true);
#if defined(__APPLE__)
        session_config.SetSamplerBackend (litert::lm::Backend::CPU);
#endif

        auto session_result = active_engine->CreateSession (session_config);
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
        inputs.emplace_back (PreprocessGemma4Image (image_bytes));
        inputs.emplace_back (litert::lm::InputImageEnd ());

        auto prefill_status = (*session_result)->RunPrefill (inputs);
        if (!prefill_status.ok ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                throw std::runtime_error (
                    "GemmaExtractor: Image label refinement prefill failed");
              }
            continue;
          }

        constexpr int kConstraintVocabSize = 1000000;
        auto &tokenizer = const_cast<litert::lm::Tokenizer &> (
            active_engine->GetTokenizer ());
        extractor::JsonSchemaConstraint constraint (schema, tokenizer,
                                                     kConstraintVocabSize);
        auto decode_config = litert::lm::DecodeConfig::CreateDefault ();
        decode_config.SetConstraint (&constraint);
        auto response = (*session_result)->RunDecode (decode_config);
        if (!response.ok ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                throw std::runtime_error (
                    "GemmaExtractor: Image label refinement decode failed");
              }
            continue;
          }
        const auto &texts = response->GetTexts ();
        if (texts.empty ())
          {
            if (attempt + 1 >= kMaxAttempts)
              {
                throw std::runtime_error (
                    "GemmaExtractor: Empty image label refinement output");
              }
            continue;
          }
        auto parsed = ParseExtractionResponse (texts.front ());
        if (HasNonEmptyLabel (parsed) || SchemaAllowsEmptyLabels (schema))
          {
            return parsed;
          }
      }
    throw std::runtime_error (
        "GemmaExtractor: Failed to refine labels from image");
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

operations::ExtractionResult
GemmaExtractor::RefineLabelsFromText (
    const std::string &text, const std::vector<std::string> &current_labels,
    const nlohmann::json &schema)
{
  return impl_->RefineLabelsFromTextImpl (text, current_labels, schema);
}

operations::ExtractionResult
GemmaExtractor::RefineLabelsFromAudio (
    const float *pcm, size_t num_samples,
    const std::vector<std::string> &current_labels,
    const nlohmann::json &schema)
{
  return impl_->RefineLabelsFromAudioImpl (pcm, num_samples, current_labels,
                                          schema);
}

operations::ExtractionResult
GemmaExtractor::RefineLabelsFromImage (
    const std::vector<unsigned char> &image_bytes,
    const std::vector<std::string> &current_labels,
    const nlohmann::json &schema)
{
  return impl_->RefineLabelsFromImageImpl (image_bytes, current_labels, schema);
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

operations::ExtractionResult
GemmaExtractor::RefineLabelsFromText (
    const std::string & /*text*/,
    const std::vector<std::string> & /*current_labels*/,
    const nlohmann::json & /*schema*/)
{
  throw std::runtime_error (
      "GemmaExtractor: LiteRT-LM disabled. Rebuild without CORTEXT_DISABLE_LITERT");
}

operations::ExtractionResult
GemmaExtractor::RefineLabelsFromAudio (
    const float * /*pcm*/, size_t /*num_samples*/,
    const std::vector<std::string> & /*current_labels*/,
    const nlohmann::json & /*schema*/)
{
  throw std::runtime_error (
      "GemmaExtractor: LiteRT-LM disabled. Rebuild without CORTEXT_DISABLE_LITERT");
}

operations::ExtractionResult
GemmaExtractor::RefineLabelsFromImage (
    const std::vector<unsigned char> & /*image_bytes*/,
    const std::vector<std::string> & /*current_labels*/,
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
