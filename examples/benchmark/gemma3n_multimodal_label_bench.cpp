#include <cortext/extractor/json_schema_constraint.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "litert/c/litert_tensor_buffer.h"
#include "runtime/components/preprocessor/audio_preprocessor.h"
#include "runtime/components/preprocessor/audio_preprocessor_miniaudio.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor_settings.h"

namespace
{

using Clock = std::chrono::steady_clock;

struct Options
{
  std::filesystem::path model_path
      = "models/gemma3n-e2b-litert/gemma-3n-E2B-it-int4.litertlm";
  std::filesystem::path real_assets_dir = "build/real_multimodal_episode_assets";
  std::filesystem::path extended_assets_dir = "build/extended_label_graph_assets";
  std::filesystem::path output_dir = "build/gemma3n_multimodal_label_bench";
  int threads = 0;
  bool reload_per_case = false;
  bool source_media = false;
  std::string main_backend = "cpu";
  std::string vision_backend = "gpu";
  std::string audio_backend = "cpu";
  std::string sampler_backend = "cpu";
};

struct Case
{
  std::string case_id;
  std::string modality;
  std::string group;
  std::string text;
  std::filesystem::path path;
  std::set<std::string> expected;
};

struct CaseResult
{
  Case input;
  std::string raw_response;
  std::vector<std::string> labels;
  std::vector<std::string> matched_expected;
  bool parsed_json = false;
  bool label_hit = false;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double total_ms = 0.0;
  std::string error;
};

std::string
Lower (std::string value);

void
PrintUsage ()
{
  std::cout << "Usage: cortext_gemma3n_multimodal_label_bench"
            << " [--model <litertlm>] [--real-assets-dir <path>]"
            << " [--extended-assets-dir <path>] [--output-dir <path>]"
            << " [--threads <n>] [--reload-per-case]"
            << " [--source-media]"
            << " [--main-backend cpu|gpu] [--vision-backend cpu|gpu]"
            << " [--audio-backend cpu|gpu] [--sampler-backend cpu|gpu]\n";
}

Options
ParseArgs (int argc, char **argv)
{
  Options opts;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h")
        {
          PrintUsage ();
          std::exit (0);
        }
      if (arg == "--model" && i + 1 < argc)
        {
          opts.model_path = argv[++i];
        }
      else if (arg == "--real-assets-dir" && i + 1 < argc)
        {
          opts.real_assets_dir = argv[++i];
        }
      else if (arg == "--extended-assets-dir" && i + 1 < argc)
        {
          opts.extended_assets_dir = argv[++i];
        }
      else if (arg == "--output-dir" && i + 1 < argc)
        {
          opts.output_dir = argv[++i];
        }
      else if (arg == "--threads" && i + 1 < argc)
        {
          opts.threads = std::max (0, std::stoi (argv[++i]));
        }
      else if (arg == "--reload-per-case")
        {
          opts.reload_per_case = true;
        }
      else if (arg == "--source-media")
        {
          opts.source_media = true;
        }
      else if (arg == "--main-backend" && i + 1 < argc)
        {
          opts.main_backend = Lower (argv[++i]);
        }
      else if (arg == "--vision-backend" && i + 1 < argc)
        {
          opts.vision_backend = Lower (argv[++i]);
        }
      else if (arg == "--audio-backend" && i + 1 < argc)
        {
          opts.audio_backend = Lower (argv[++i]);
        }
      else if (arg == "--sampler-backend" && i + 1 < argc)
        {
          opts.sampler_backend = Lower (argv[++i]);
        }
      else
        {
          throw std::runtime_error ("Unknown or incomplete argument: " + arg);
        }
    }
  return opts;
}

litert::lm::Backend
ParseBackend (const std::string &value)
{
  if (value == "cpu")
    {
      return litert::lm::Backend::CPU;
    }
  if (value == "gpu")
    {
      return litert::lm::Backend::GPU;
    }
  throw std::runtime_error ("Unsupported LiteRT backend: " + value);
}

double
MsSince (Clock::time_point start)
{
  return std::chrono::duration<double, std::milli> (Clock::now () - start)
      .count ();
}

std::string
ReadString (const std::filesystem::path &path)
{
  std::ifstream in (path, std::ios::binary | std::ios::ate);
  if (!in)
    {
      throw std::runtime_error ("Failed to open " + path.string ());
    }
  const auto size = in.tellg ();
  in.seekg (0, std::ios::beg);
  std::string data (static_cast<std::size_t> (size), '\0');
  in.read (data.data (), static_cast<std::streamsize> (data.size ()));
  return data;
}

std::vector<float>
ReadFloat32 (const std::filesystem::path &path)
{
  const auto bytes = ReadString (path);
  if (bytes.size () % sizeof (float) != 0)
    {
      throw std::runtime_error ("Invalid float32 file: " + path.string ());
    }
  std::vector<float> out (bytes.size () / sizeof (float));
  std::memcpy (out.data (), bytes.data (), bytes.size ());
  return out;
}

std::string
Lower (std::string value)
{
  std::transform (value.begin (), value.end (), value.begin (),
                  [] (unsigned char c) {
                    return static_cast<char> (std::tolower (c));
                  });
  return value;
}

std::string
Normalize (const std::string &value)
{
  std::string out;
  out.reserve (value.size ());
  bool last_space = true;
  for (unsigned char c : value)
    {
      if (std::isalnum (c))
        {
          out.push_back (static_cast<char> (std::tolower (c)));
          last_space = false;
        }
      else if (!last_space)
        {
          out.push_back (' ');
          last_space = true;
        }
    }
  while (!out.empty () && out.back () == ' ')
    {
      out.pop_back ();
    }
  return out;
}

std::string
CsvEscape (const std::string &value)
{
  if (value.find_first_of (",\"\n\r") == std::string::npos)
    {
      return value;
    }
  std::string out = "\"";
  for (char c : value)
    {
      out += c;
      if (c == '"')
        {
          out += '"';
        }
    }
  out += '"';
  return out;
}

std::string
Join (const std::vector<std::string> &items)
{
  std::ostringstream out;
  for (std::size_t i = 0; i < items.size (); ++i)
    {
      if (i != 0)
        {
          out << "|";
        }
      out << items[i];
    }
  return out.str ();
}

std::string
JoinSet (const std::set<std::string> &items)
{
  std::ostringstream out;
  bool first = true;
  for (const auto &item : items)
    {
      if (!first)
        {
          out << "|";
        }
      first = false;
      out << item;
    }
  return out.str ();
}

nlohmann::json
BuildLabelSchema ()
{
  return {
    { "type", "object" },
    { "properties",
      { { "labels",
          { { "type", "array" },
            { "minItems", 0 },
            { "maxItems", 5 },
            { "items", { { "type", "string" } } } } } } },
    { "required", nlohmann::json::array ({ "labels" }) },
  };
}

std::optional<nlohmann::json>
TryParseObject (const std::string &content)
{
  const auto start = content.find ('{');
  const auto end = content.rfind ('}');
  if (start == std::string::npos || end == std::string::npos || end < start)
    {
      return std::nullopt;
    }
  const auto TryParse = [] (const std::string &candidate)
      -> std::optional<nlohmann::json> {
    try
      {
        return nlohmann::json::parse (candidate);
      }
    catch (const nlohmann::json::exception &)
      {
        return std::nullopt;
      }
  };

  const std::string candidate = content.substr (start, end - start + 1);
  if (auto parsed = TryParse (candidate))
    {
      return parsed;
    }

  if (candidate.size () >= 4 && candidate[0] == '{' && candidate[1] == '{'
      && candidate[candidate.size () - 1] == '}'
      && candidate[candidate.size () - 2] == '}')
    {
      if (auto parsed = TryParse (candidate.substr (1, candidate.size () - 2)))
        {
          return parsed;
        }
    }

  const auto labels_key = candidate.find ("\"labels\"");
  if (labels_key != std::string::npos)
    {
      const auto array_start = candidate.find ('[', labels_key);
      const auto array_end = candidate.find (']', array_start);
      if (array_start != std::string::npos && array_end != std::string::npos
          && array_end > array_start)
        {
          const std::string labels_object
              = "{\"labels\":"
                + candidate.substr (array_start, array_end - array_start + 1)
                + "}";
          if (auto parsed = TryParse (labels_object))
            {
              return parsed;
            }
        }
    }
  return std::nullopt;
}

std::vector<std::string>
ExtractLabels (const std::string &content, bool &parsed_json)
{
  parsed_json = false;
  std::vector<std::string> labels;
  auto object = TryParseObject (content);
  if (!object || !object->contains ("labels"))
    {
      return labels;
    }
  parsed_json = true;
  for (const auto &label : (*object)["labels"])
    {
      std::string text;
      if (label.is_string ())
        {
          text = label.get<std::string> ();
        }
      else if (label.is_object ())
        {
          text = label.value ("label", label.value ("name", ""));
        }
      text = Normalize (text);
      if (!text.empty ())
        {
          labels.push_back (text);
        }
    }
  std::sort (labels.begin (), labels.end ());
  labels.erase (std::unique (labels.begin (), labels.end ()), labels.end ());
  return labels;
}

std::vector<std::string>
MatchExpected (const std::vector<std::string> &labels,
               const std::set<std::string> &expected)
{
  std::vector<std::string> matches;
  for (const auto &want : expected)
    {
      const auto normalized_want = Normalize (want);
      for (const auto &label : labels)
        {
          if (label.find (normalized_want) != std::string::npos
              || normalized_want.find (label) != std::string::npos)
            {
              matches.push_back (want);
              break;
            }
        }
    }
  return matches;
}

litert::TensorBuffer
BuildImageTensor (const std::filesystem::path &path, int source_width,
                  int source_height)
{
  const auto bytes = ReadString (path);
  const std::size_t expected_size
      = static_cast<std::size_t> (source_width) * source_height * 3;
  if (bytes.size () != expected_size)
    {
      throw std::runtime_error ("Invalid RGB image byte length for "
                                + path.string ());
    }

  constexpr int kTargetWidth = 768;
  constexpr int kTargetHeight = 768;
  constexpr int kChannels = 3;
  const std::size_t element_count
      = static_cast<std::size_t> (kTargetWidth) * kTargetHeight * kChannels;
  const std::size_t byte_count = element_count * sizeof (float);

  void *host_buffer = nullptr;
  if (posix_memalign (&host_buffer, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                      byte_count)
          != 0
      || host_buffer == nullptr)
    {
      throw std::runtime_error ("Image tensor allocation failed");
    }

  auto *float_buffer = static_cast<float *> (host_buffer);
  for (int y = 0; y < kTargetHeight; ++y)
    {
      const int source_y = std::min (
          source_height - 1, (y * source_height) / kTargetHeight);
      for (int x = 0; x < kTargetWidth; ++x)
        {
          const int source_x = std::min (
              source_width - 1, (x * source_width) / kTargetWidth);
          const std::size_t source_offset
              = (static_cast<std::size_t> (source_y) * source_width
                 + static_cast<std::size_t> (source_x))
                * kChannels;
          const std::size_t target_offset
              = (static_cast<std::size_t> (y) * kTargetWidth
                 + static_cast<std::size_t> (x))
                * kChannels;
          for (int c = 0; c < kChannels; ++c)
            {
              float_buffer[target_offset + c]
                  = static_cast<unsigned char> (bytes[source_offset + c])
                    / 255.0f;
            }
        }
    }

  LiteRtLayout layout{};
  layout.rank = 4;
  layout.has_strides = false;
  layout.dimensions[0] = 1;
  layout.dimensions[1] = kTargetHeight;
  layout.dimensions[2] = kTargetWidth;
  layout.dimensions[3] = kChannels;

  LiteRtRankedTensorType tensor_type{};
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  tensor_type.layout = layout;

  LiteRtTensorBuffer buffer = nullptr;
  const LiteRtStatus status = LiteRtCreateTensorBufferFromHostMemory (
      &tensor_type, host_buffer, byte_count, std::free, &buffer);
  if (status != kLiteRtStatusOk || buffer == nullptr)
    {
      std::free (host_buffer);
      throw std::runtime_error ("Image tensor creation failed");
    }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  return litert::TensorBuffer::WrapCObject (buffer, litert::OwnHandle::kYes);
#pragma clang diagnostic pop
}

std::string
PromptFor (const Case &c)
{
  std::ostringstream prompt;
  prompt
      << "You are labeling one memory signal for a memory graph. "
      << "Return only JSON in this exact shape: {\"labels\":[\"...\"]}. "
      << "Use at most five short concrete labels. "
      << "Prefer visible or audible subjects, objects, places, names, and events. ";
  if (c.modality == "text")
    {
      prompt << "Label this text:\n" << c.text;
    }
  else if (c.modality == "image")
    {
      prompt << "Label this image.\n<start_of_image>";
    }
  else if (c.modality == "audio")
    {
      prompt << "Label this audio. If it is speech, include the spoken word or name.\n"
             << "<start_of_audio>";
    }
  return prompt.str ();
}

std::string
ConversationPromptFor (const Case &c)
{
  std::ostringstream prompt;
  prompt
      << "You are labeling one memory signal for a memory graph. "
      << "Return only JSON in this exact shape: {\"labels\":[\"...\"]}. "
      << "Use at most five short concrete labels. "
      << "Prefer visible or audible subjects, objects, places, names, and events. ";
  if (c.modality == "text")
    {
      prompt << "Label this text:\n" << c.text;
    }
  else if (c.modality == "image")
    {
      prompt << "Label this image.";
    }
  else if (c.modality == "audio")
    {
      prompt << "Label this audio. If it is speech, include the spoken word or name.";
    }
  return prompt.str ();
}

std::string
ExtractConversationText (const litert::lm::Message &message)
{
  if (!message.contains ("content"))
    {
      return {};
    }
  const auto &content = message["content"];
  if (content.is_string ())
    {
      return content.get<std::string> ();
    }
  if (content.is_array ())
    {
      std::ostringstream out;
      for (const auto &item : content)
        {
          if (item.is_string ())
            {
              out << item.get<std::string> ();
            }
          else if (item.is_object ())
            {
              if (item.contains ("text") && item["text"].is_string ())
                {
                  out << item["text"].get<std::string> ();
                }
              else if (item.contains ("content")
                       && item["content"].is_string ())
                {
                  out << item["content"].get<std::string> ();
                }
            }
        }
      return out.str ();
    }
  return {};
}

class Gemma3nLabeler
{
public:
  explicit Gemma3nLabeler (const Options &opts) : opts_ (opts)
  {
    auto model_assets = litert::lm::ModelAssets::Create (opts.model_path.string ());
    if (!model_assets.ok ())
      {
        throw std::runtime_error ("Failed to load model assets: "
                                  + model_assets.status ().ToString ());
      }

    auto settings = litert::lm::EngineSettings::CreateDefault (
        *std::move (model_assets), ParseBackend (opts.main_backend),
        ParseBackend (opts.vision_backend), ParseBackend (opts.audio_backend));
    if (!settings.ok ())
      {
        throw std::runtime_error ("Failed to create engine settings: "
                                  + settings.status ().ToString ());
      }

    auto &main_settings = settings->GetMutableMainExecutorSettings ();
    main_settings.SetMaxNumImages (1);
    main_settings.SetMaxNumTokens (2048);
    if (opts.threads > 0 && opts.main_backend == "cpu")
      {
        auto cpu_config_result
            = main_settings.MutableBackendConfig<litert::lm::CpuConfig> ();
        if (cpu_config_result.ok ())
          {
            litert::lm::CpuConfig cpu_config = *cpu_config_result;
            cpu_config.number_of_threads = opts.threads;
            main_settings.SetBackendConfig (cpu_config);
          }
      }

    auto engine_result
        = litert::lm::EngineFactory::CreateDefault (*std::move (settings));
    if (!engine_result.ok ())
      {
        throw std::runtime_error ("Failed to create engine: "
                                  + engine_result.status ().ToString ());
      }
    engine_ = std::move (*engine_result);
  }

  CaseResult Run (const Case &input)
  {
    CaseResult result;
    result.input = input;
    const auto total_start = Clock::now ();
    try
      {
        litert::lm::SessionConfig session_config
            = litert::lm::SessionConfig::CreateDefault ();
        if (input.modality == "image")
          {
            session_config.SetVisionModalityEnabled (true);
          }
        else if (input.modality == "audio")
          {
            session_config.SetAudioModalityEnabled (true);
          }

        session_config.SetSamplerBackend (ParseBackend (opts_.sampler_backend));

        auto session_result = engine_->CreateSession (session_config);
        if (!session_result.ok ())
          {
            throw std::runtime_error ("Failed to create session: "
                                      + session_result.status ().ToString ());
          }

        std::vector<litert::lm::InputData> inputs;
        inputs.emplace_back (litert::lm::InputText (PromptFor (input)));
        if (input.modality == "image")
          {
            inputs.emplace_back (
                litert::lm::InputImage (BuildImageTensor (input.path, 384, 384)));
            inputs.emplace_back (litert::lm::InputImageEnd ());
            inputs.emplace_back (litert::lm::InputText ("\n\n"));
          }
        else if (input.modality == "audio")
          {
            const auto pcm = ReadFloat32 (input.path);
            auto preprocessor_result
                = litert::lm::AudioPreprocessorMiniAudio::Create (
                    litert::lm::AudioPreprocessorConfig::CreateDefaultUsmConfig ());
            if (!preprocessor_result.ok ())
              {
                throw std::runtime_error ("Failed to create audio preprocessor: "
                                          + preprocessor_result.status ().ToString ());
              }
            auto processed_audio = (*preprocessor_result)
                                       ->Preprocess (litert::lm::InputAudio (pcm));
            if (!processed_audio.ok ())
              {
                throw std::runtime_error ("Audio preprocessing failed: "
                                          + processed_audio.status ().ToString ());
              }
            inputs.emplace_back (std::move (*processed_audio));
            inputs.emplace_back (litert::lm::InputAudioEnd ());
          }

        const auto prefill_start = Clock::now ();
        auto prefill_status = (*session_result)->RunPrefill (inputs);
        result.prefill_ms = MsSince (prefill_start);
        if (!prefill_status.ok ())
          {
            throw std::runtime_error ("Prefill failed: "
                                      + prefill_status.ToString ());
          }

        auto &tokenizer = const_cast<litert::lm::Tokenizer &> (
            engine_->GetTokenizer ());
        constexpr int kConstraintVocabSize = 1000000;
        const auto schema = BuildLabelSchema ();
        cortext::extractor::JsonSchemaConstraint constraint (
            schema, tokenizer, kConstraintVocabSize);
        auto decode_config = litert::lm::DecodeConfig::CreateDefault ();
        decode_config.SetConstraint (&constraint);

        const auto decode_start = Clock::now ();
        auto response = (*session_result)->RunDecode (decode_config);
        result.decode_ms = MsSince (decode_start);
        if (!response.ok ())
          {
            throw std::runtime_error ("Decode failed: "
                                      + response.status ().ToString ());
          }
        const auto &texts = response->GetTexts ();
        if (!texts.empty ())
          {
            result.raw_response = texts.front ();
          }
        result.labels = ExtractLabels (result.raw_response, result.parsed_json);
        result.matched_expected = MatchExpected (result.labels, input.expected);
        result.label_hit = !result.matched_expected.empty ();
      }
    catch (const std::exception &e)
      {
        result.error = e.what ();
      }
    result.total_ms = MsSince (total_start);
    return result;
  }

  CaseResult RunConversation (const Case &input)
  {
    CaseResult result;
    result.input = input;
    const auto total_start = Clock::now ();
    try
      {
        litert::lm::SessionConfig session_config
            = litert::lm::SessionConfig::CreateDefault ();
        if (input.modality == "image")
          {
            session_config.SetVisionModalityEnabled (true);
          }
        else if (input.modality == "audio")
          {
            session_config.SetAudioModalityEnabled (true);
          }
        session_config.SetSamplerBackend (ParseBackend (opts_.sampler_backend));

        auto config_result = litert::lm::ConversationConfig::Builder ()
                                 .SetSessionConfig (session_config)
                                 .Build (*engine_);
        if (!config_result.ok ())
          {
            throw std::runtime_error (
                "Failed to create conversation config: "
                + config_result.status ().ToString ());
          }

        auto conversation_result
            = litert::lm::Conversation::Create (*engine_, *config_result);
        if (!conversation_result.ok ())
          {
            throw std::runtime_error ("Failed to create conversation: "
                                      + conversation_result.status ().ToString ());
          }

        litert::lm::Message message = {
          { "role", "user" },
          { "content", ConversationPromptFor (input) }
        };
        if (input.modality == "image")
          {
            message["content"] = nlohmann::ordered_json::array ({
              { { "type", "text" }, { "text", ConversationPromptFor (input) } },
              { { "type", "image" }, { "path", input.path.string () } }
            });
          }
        else if (input.modality == "audio")
          {
            message["content"] = nlohmann::ordered_json::array ({
              { { "type", "text" }, { "text", ConversationPromptFor (input) } },
              { { "type", "audio" }, { "path", input.path.string () } }
            });
          }

        const auto infer_start = Clock::now ();
        auto response = (*conversation_result)->SendMessage (message);
        result.prefill_ms = MsSince (infer_start);
        if (!response.ok ())
          {
            throw std::runtime_error ("Conversation inference failed: "
                                      + response.status ().ToString ());
          }

        result.raw_response = ExtractConversationText (*response);
        result.labels = ExtractLabels (result.raw_response, result.parsed_json);
        result.matched_expected = MatchExpected (result.labels, input.expected);
        result.label_hit = !result.matched_expected.empty ();
      }
    catch (const std::exception &e)
      {
        result.error = e.what ();
      }
    result.total_ms = MsSince (total_start);
    return result;
  }

private:
  Options opts_;
  std::unique_ptr<litert::lm::Engine> engine_;
};

std::vector<Case>
BuildCases (const Options &opts)
{
  const auto real_raw = opts.real_assets_dir / "raw";
  const auto extended_raw = opts.extended_assets_dir / "raw";
  const auto real_source = opts.real_assets_dir / "source";
  const auto extended_source = opts.extended_assets_dir / "source";
  const auto real_image = opts.source_media ? real_source : real_raw;
  const auto extended_image = opts.source_media ? extended_source : extended_raw;
  // LiteRT-LM's current Gemma4 source media path handles JPEGs correctly, but
  // the local MiniAudio build rejects these OGG fixtures. Keep audio on the
  // existing real PCM fixture path while source-media mode exercises the
  // official image patchifier.
  const auto real_audio = real_raw;
  const auto extended_audio = extended_raw;
  const auto bailey_sentence_audio = real_audio / "bailey_sentence_16k_mono.f32";
  const auto bailey_audio = std::filesystem::exists (bailey_sentence_audio)
                                ? bailey_sentence_audio
                                : real_audio / "bailey_16k_mono.f32";
  return {
    { "dog_image", "image", "dog", "",
      real_image / (opts.source_media ? "dog.jpg" : "dog_384x384.rgb"),
      { "dog", "golden retriever" } },
    { "dog_text", "text", "dog", "That dog's name is Bailey.", {},
      { "dog", "bailey" } },
    { "dog_audio", "audio", "dog", "",
      bailey_audio,
      { "bailey" } },
    { "car_image", "image", "car_crash", "",
      real_image / (opts.source_media ? "car_crash.jpg"
                                      : "car_crash_384x384.rgb"),
      { "car", "crash" } },
    { "car_text", "text", "car_crash", "A car crashed into a tree.", {},
      { "car", "crash", "tree" } },
    { "crash_audio", "audio", "car_crash", "",
      real_audio / "crash_16k_mono.f32",
      { "crash" } },
    { "cat_image", "image", "cat", "",
      extended_image / (opts.source_media ? "cat.jpg" : "cat_384x384.rgb"),
      { "cat" } },
    { "cat_text", "text", "cat", "A cat is sitting and looking at the camera.",
      {}, { "cat" } },
    { "cat_audio", "audio", "cat", "",
      extended_audio / "cat_16k_mono.f32",
      { "cat" } },
    { "train_image", "image", "train", "",
      extended_image / (opts.source_media ? "train.jpg"
                                          : "train_384x384.rgb"),
      { "train" } },
    { "train_text", "text", "train", "A train is moving through the tunnel.",
      {}, { "train", "tunnel" } },
    { "train_audio", "audio", "train", "",
      extended_audio / "train_16k_mono.f32",
      { "train" } },
    { "bell_image", "image", "bell", "",
      extended_image / (opts.source_media ? "bell.jpg" : "bell_384x384.rgb"),
      { "bell" } },
    { "bell_text", "text", "bell", "A large bell is hanging in a display.",
      {}, { "bell" } },
    { "bell_audio", "audio", "bell", "",
      extended_audio / "bell_16k_mono.f32",
      { "bell" } },
  };
}

double
Mean (const std::vector<double> &values)
{
  if (values.empty ())
    {
      return 0.0;
    }
  return std::accumulate (values.begin (), values.end (), 0.0)
         / static_cast<double> (values.size ());
}

double
P95 (std::vector<double> values)
{
  if (values.empty ())
    {
      return 0.0;
    }
  std::sort (values.begin (), values.end ());
  const std::size_t index = static_cast<std::size_t> (
      std::ceil (0.95 * static_cast<double> (values.size ())) - 1.0);
  return values[std::min (index, values.size () - 1)];
}

nlohmann::json
Summarize (const Options &opts, const std::vector<CaseResult> &results)
{
  const std::string model_name = opts.model_path.filename ().string ();
  const std::string model_family
      = (model_name.find ("gemma-4") != std::string::npos
         || model_name.find ("Gemma-4") != std::string::npos
         || model_name.find ("Gemma4") != std::string::npos)
            ? "Gemma 4 E2B IT LiteRT"
            : "Gemma 3n E2B IT LiteRT";
  nlohmann::json by_modality = nlohmann::json::object ();
  std::map<std::string, std::vector<const CaseResult *>> grouped;
  for (const auto &result : results)
    {
      grouped[result.input.modality].push_back (&result);
    }
  for (const auto &[modality, rows] : grouped)
    {
      int parsed = 0;
      int hits = 0;
      int errors = 0;
      std::vector<double> latencies;
      for (const auto *row : rows)
        {
          parsed += row->parsed_json ? 1 : 0;
          hits += row->label_hit ? 1 : 0;
          errors += row->error.empty () ? 0 : 1;
          latencies.push_back (row->total_ms);
        }
      by_modality[modality] = {
        { "case_count", rows.size () },
        { "json_parse_rate",
          static_cast<double> (parsed) / static_cast<double> (rows.size ()) },
        { "label_hit_rate",
          static_cast<double> (hits) / static_cast<double> (rows.size ()) },
        { "error_count", errors },
        { "mean_ms", Mean (latencies) },
        { "p95_ms", P95 (latencies) },
      };
    }

  int parsed = 0;
  int hits = 0;
  int errors = 0;
  std::vector<double> total_latencies;
  for (const auto &result : results)
    {
      parsed += result.parsed_json ? 1 : 0;
      hits += result.label_hit ? 1 : 0;
      errors += result.error.empty () ? 0 : 1;
      total_latencies.push_back (result.total_ms);
    }

  return {
    { "benchmark", "gemma3n_multimodal_autoregressive_label_bench" },
    { "model_path", opts.model_path.string () },
    { "model_family", model_family },
    { "backend",
      "LiteRT-LM main=" + opts.main_backend + " vision="
          + opts.vision_backend + " audio=" + opts.audio_backend
          + " sampler=" + opts.sampler_backend },
    { "main_backend", opts.main_backend },
    { "vision_backend", opts.vision_backend },
    { "audio_backend", opts.audio_backend },
    { "sampler_backend", opts.sampler_backend },
    { "audio_preprocessor", "LiteRT-LM default USM MiniAudio" },
    { "threads", opts.threads },
    { "reload_per_case", opts.reload_per_case },
    { "source_media", opts.source_media },
    { "source", "benchmark-only" },
    { "real_media", true },
    { "case_count", results.size () },
    { "json_parse_rate",
      results.empty ()
          ? 0.0
          : static_cast<double> (parsed) / static_cast<double> (results.size ()) },
    { "label_hit_rate",
      results.empty ()
          ? 0.0
          : static_cast<double> (hits) / static_cast<double> (results.size ()) },
    { "error_count", errors },
    { "mean_ms", Mean (total_latencies) },
    { "p95_ms", P95 (total_latencies) },
    { "by_modality", by_modality },
  };
}

void
WriteCasesCsv (const std::filesystem::path &path,
               const std::vector<CaseResult> &results)
{
  std::ofstream out (path);
  out << "case_id,modality,group,expected,labels,matched_expected,label_hit,"
      << "parsed_json,prefill_ms,decode_ms,total_ms,error,raw_response\n";
  out << std::fixed << std::setprecision (3);
  for (const auto &result : results)
    {
      out << CsvEscape (result.input.case_id) << ','
          << CsvEscape (result.input.modality) << ','
          << CsvEscape (result.input.group) << ','
          << CsvEscape (JoinSet (result.input.expected)) << ','
          << CsvEscape (Join (result.labels)) << ','
          << CsvEscape (Join (result.matched_expected)) << ','
          << (result.label_hit ? 1 : 0) << ','
          << (result.parsed_json ? 1 : 0) << ','
          << result.prefill_ms << ',' << result.decode_ms << ','
          << result.total_ms << ',' << CsvEscape (result.error) << ','
          << CsvEscape (result.raw_response) << '\n';
    }
}

} // namespace

int
main (int argc, char **argv)
{
  try
    {
      const Options opts = ParseArgs (argc, argv);
      std::filesystem::create_directories (opts.output_dir);

      if (!std::filesystem::exists (opts.model_path))
        {
          throw std::runtime_error ("Model file not found: "
                                    + opts.model_path.string ());
        }

      std::vector<CaseResult> results;
      std::unique_ptr<Gemma3nLabeler> reusable_labeler;
      if (!opts.reload_per_case)
        {
          reusable_labeler = std::make_unique<Gemma3nLabeler> (opts);
        }
      for (const auto &input : BuildCases (opts))
        {
          const bool use_conversation
              = opts.source_media && input.modality == "image";
          if (opts.reload_per_case)
            {
              Gemma3nLabeler labeler (opts);
              results.push_back (use_conversation ? labeler.RunConversation (input)
                                                  : labeler.Run (input));
            }
          else
            {
              results.push_back (use_conversation
                                     ? reusable_labeler->RunConversation (input)
                                     : reusable_labeler->Run (input));
            }
          const auto &result = results.back ();
          std::cerr << result.input.case_id << " labels="
                    << Join (result.labels) << " hit="
                    << (result.label_hit ? "yes" : "no")
                    << " total_ms=" << result.total_ms;
          if (!result.error.empty ())
            {
              std::cerr << " error=" << result.error;
            }
          std::cerr << '\n';
        }

      nlohmann::json cases = nlohmann::json::array ();
      for (const auto &result : results)
        {
          cases.push_back ({
            { "case_id", result.input.case_id },
            { "modality", result.input.modality },
            { "group", result.input.group },
            { "expected", result.input.expected },
            { "labels", result.labels },
            { "matched_expected", result.matched_expected },
            { "label_hit", result.label_hit },
            { "parsed_json", result.parsed_json },
            { "prefill_ms", result.prefill_ms },
            { "decode_ms", result.decode_ms },
            { "total_ms", result.total_ms },
            { "error", result.error },
            { "raw_response", result.raw_response },
          });
        }

      const auto summary = Summarize (opts, results);
      {
        std::ofstream out (opts.output_dir
                           / "gemma3n_multimodal_label_results.json");
        out << summary.dump (2) << '\n';
      }
      {
        std::ofstream out (opts.output_dir
                           / "gemma3n_multimodal_label_cases.json");
        out << cases.dump (2) << '\n';
      }
      WriteCasesCsv (opts.output_dir / "gemma3n_multimodal_label_cases.csv",
                     results);
      {
        std::ofstream out (opts.output_dir
                           / "gemma3n_multimodal_label_failures.csv");
        out << "case_id,modality,group,expected,labels,error,raw_response\n";
        for (const auto &result : results)
          {
            if (result.label_hit && result.error.empty ())
              {
                continue;
              }
            out << CsvEscape (result.input.case_id) << ','
                << CsvEscape (result.input.modality) << ','
                << CsvEscape (result.input.group) << ','
                << CsvEscape (JoinSet (result.input.expected)) << ','
                << CsvEscape (Join (result.labels)) << ','
                << CsvEscape (result.error) << ','
                << CsvEscape (result.raw_response) << '\n';
          }
      }

      std::cout << summary.dump (2) << '\n';
      return summary.value ("error_count", 0) == 0 ? 0 : 2;
    }
  catch (const std::exception &e)
    {
      std::cerr << "error: " << e.what () << '\n';
      return 1;
    }
}
