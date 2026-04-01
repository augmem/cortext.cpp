#include "cortext/audio/sherpa_onnx.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#if defined(CORTEXT_ENABLE_SHERPA_ONNX)
#include "sherpa-onnx/c-api/cxx-api.h"
#endif

namespace cortext
{

#if defined(CORTEXT_ENABLE_SHERPA_ONNX)

namespace
{
void
EnsureNonEmpty (const std::string &value, const std::string &label)
{
  if (value.empty ())
    throw std::runtime_error ("SherpaOnnx: missing " + label);
}

sherpa_onnx::cxx::OfflineRecognizerConfig
BuildAsrConfig (const SherpaOnnxAsrConfig &config)
{
  sherpa_onnx::cxx::OfflineRecognizerConfig cfg;
  cfg.feat_config.sample_rate = config.sample_rate;
  cfg.model_config.tokens = config.tokens;
  cfg.model_config.bpe_vocab = config.bpe_vocab;
  cfg.model_config.num_threads = config.num_threads;
  cfg.model_config.provider = config.provider;
  cfg.model_config.model_type = config.model_type;

  if (config.model_type == "transducer")
    {
      EnsureNonEmpty (config.encoder, "transducer encoder");
      EnsureNonEmpty (config.decoder, "transducer decoder");
      EnsureNonEmpty (config.joiner, "transducer joiner");
      cfg.model_config.transducer.encoder = config.encoder;
      cfg.model_config.transducer.decoder = config.decoder;
      cfg.model_config.transducer.joiner = config.joiner;
    }
  else if (config.model_type == "paraformer")
    {
      EnsureNonEmpty (config.model, "paraformer model");
      cfg.model_config.paraformer.model = config.model;
    }
  else if (config.model_type == "nemo_ctc")
    {
      EnsureNonEmpty (config.model, "nemo_ctc model");
      cfg.model_config.nemo_ctc.model = config.model;
    }
  else if (config.model_type == "zipformer_ctc")
    {
      EnsureNonEmpty (config.model, "zipformer_ctc model");
      cfg.model_config.zipformer_ctc.model = config.model;
    }
  else if (config.model_type == "whisper")
    {
      EnsureNonEmpty (config.encoder, "whisper encoder");
      EnsureNonEmpty (config.decoder, "whisper decoder");
      cfg.model_config.whisper.encoder = config.encoder;
      cfg.model_config.whisper.decoder = config.decoder;
      cfg.model_config.whisper.language = config.language;
      cfg.model_config.whisper.task = "transcribe";
    }
  else
    {
      throw std::runtime_error ("SherpaOnnx: unsupported ASR model_type: "
                                + config.model_type);
    }

  return cfg;
}

sherpa_onnx::cxx::OfflineTtsConfig
BuildTtsConfig (const SherpaOnnxTtsConfig &config)
{
  sherpa_onnx::cxx::OfflineTtsConfig cfg;
  cfg.model.num_threads = config.num_threads;
  cfg.model.provider = config.provider;

  if (config.model_type == "vits")
    {
      EnsureNonEmpty (config.model, "vits model");
      EnsureNonEmpty (config.tokens, "vits tokens");
      cfg.model.vits.model = config.model;
      cfg.model.vits.tokens = config.tokens;
      cfg.model.vits.lexicon = config.lexicon;
      cfg.model.vits.data_dir = config.data_dir;
    }
  else if (config.model_type == "matcha")
    {
      EnsureNonEmpty (config.model, "matcha acoustic model");
      EnsureNonEmpty (config.vocoder, "matcha vocoder");
      EnsureNonEmpty (config.tokens, "matcha tokens");
      cfg.model.matcha.acoustic_model = config.model;
      cfg.model.matcha.vocoder = config.vocoder;
      cfg.model.matcha.tokens = config.tokens;
      cfg.model.matcha.lexicon = config.lexicon;
      cfg.model.matcha.data_dir = config.data_dir;
    }
  else if (config.model_type == "kokoro")
    {
      EnsureNonEmpty (config.model, "kokoro model");
      EnsureNonEmpty (config.tokens, "kokoro tokens");
      EnsureNonEmpty (config.voices, "kokoro voices");
      cfg.model.kokoro.model = config.model;
      cfg.model.kokoro.tokens = config.tokens;
      cfg.model.kokoro.voices = config.voices;
      cfg.model.kokoro.lexicon = config.lexicon;
      cfg.model.kokoro.data_dir = config.data_dir;
      cfg.model.kokoro.lang = config.language;
    }
  else if (config.model_type == "kitten")
    {
      EnsureNonEmpty (config.model, "kitten model");
      EnsureNonEmpty (config.tokens, "kitten tokens");
      EnsureNonEmpty (config.voices, "kitten voices");
      cfg.model.kitten.model = config.model;
      cfg.model.kitten.tokens = config.tokens;
      cfg.model.kitten.voices = config.voices;
      cfg.model.kitten.data_dir = config.data_dir;
    }
  else if (config.model_type == "zipvoice")
    {
      EnsureNonEmpty (config.encoder, "zipvoice encoder");
      EnsureNonEmpty (config.decoder, "zipvoice decoder");
      EnsureNonEmpty (config.vocoder, "zipvoice vocoder");
      EnsureNonEmpty (config.tokens, "zipvoice tokens");
      cfg.model.zipvoice.encoder = config.encoder;
      cfg.model.zipvoice.decoder = config.decoder;
      cfg.model.zipvoice.vocoder = config.vocoder;
      cfg.model.zipvoice.tokens = config.tokens;
      cfg.model.zipvoice.lexicon = config.lexicon;
      cfg.model.zipvoice.data_dir = config.data_dir;
    }
  else
    {
      throw std::runtime_error ("SherpaOnnx: unsupported TTS model_type: "
                                + config.model_type);
    }

  return cfg;
}

} // namespace

struct SherpaOnnxOfflineAsr::Impl
{
  explicit Impl (SherpaOnnxAsrConfig cfg)
      : config (std::move (cfg)), recognizer (sherpa_onnx::cxx::OfflineRecognizer::Create (
            BuildAsrConfig (config)))
  {
  }

  SherpaOnnxAsrConfig config;
  sherpa_onnx::cxx::OfflineRecognizer recognizer;
};

struct SherpaOnnxOfflineTts::Impl
{
  explicit Impl (SherpaOnnxTtsConfig cfg)
      : config (std::move (cfg)), tts (sherpa_onnx::cxx::OfflineTts::Create (
            BuildTtsConfig (config)))
  {
  }

  SherpaOnnxTtsConfig config;
  sherpa_onnx::cxx::OfflineTts tts;
};

#endif

SherpaOnnxOfflineAsr::SherpaOnnxOfflineAsr (SherpaOnnxAsrConfig config)
{
#if !defined(CORTEXT_ENABLE_SHERPA_ONNX)
  (void)config;
  throw std::runtime_error ("SherpaOnnxOfflineAsr requires CORTEXT_ENABLE_SHERPA_ONNX=ON");
#else
  impl_ = std::make_unique<Impl> (std::move (config));
#endif
}

SherpaOnnxOfflineAsr::~SherpaOnnxOfflineAsr () = default;

std::string
SherpaOnnxOfflineAsr::Transcribe (const float *pcm, std::size_t num_samples,
                                 int32_t sample_rate)
{
#if !defined(CORTEXT_ENABLE_SHERPA_ONNX)
  (void)pcm;
  (void)num_samples;
  (void)sample_rate;
  throw std::runtime_error ("SherpaOnnxOfflineAsr requires CORTEXT_ENABLE_SHERPA_ONNX=ON");
#else
  if (!impl_)
    throw std::runtime_error ("SherpaOnnxOfflineAsr not initialized");
  if (!pcm || num_samples == 0)
    throw std::runtime_error ("SherpaOnnxOfflineAsr: empty audio input");

  auto stream = impl_->recognizer.CreateStream ();
  stream.AcceptWaveform (sample_rate, pcm, static_cast<int32_t> (num_samples));
  impl_->recognizer.Decode (&stream);
  auto result = impl_->recognizer.GetResult (&stream);
  return result.text;
#endif
}

SherpaOnnxOfflineTts::SherpaOnnxOfflineTts (SherpaOnnxTtsConfig config)
{
#if !defined(CORTEXT_ENABLE_SHERPA_ONNX)
  (void)config;
  throw std::runtime_error ("SherpaOnnxOfflineTts requires CORTEXT_ENABLE_SHERPA_ONNX=ON");
#else
  impl_ = std::make_unique<Impl> (std::move (config));
#endif
}

SherpaOnnxOfflineTts::~SherpaOnnxOfflineTts () = default;

SherpaOnnxAudio
SherpaOnnxOfflineTts::Synthesize (const std::string &text, int32_t speaker_id)
{
#if !defined(CORTEXT_ENABLE_SHERPA_ONNX)
  (void)text;
  (void)speaker_id;
  throw std::runtime_error ("SherpaOnnxOfflineTts requires CORTEXT_ENABLE_SHERPA_ONNX=ON");
#else
  if (!impl_)
    throw std::runtime_error ("SherpaOnnxOfflineTts not initialized");
  if (text.empty ())
    throw std::runtime_error ("SherpaOnnxOfflineTts: text is empty");

  const float speed = impl_->config.speed;
  auto audio = impl_->tts.Generate (text, speaker_id, speed);
  SherpaOnnxAudio out;
  out.samples = std::move (audio.samples);
  out.sample_rate = audio.sample_rate;
  return out;
#endif
}

} // namespace cortext
