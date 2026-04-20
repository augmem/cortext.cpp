#include "sherpa-onnx/c-api/c-api.h"
#include "sherpa-onnx/c-api/cxx-api.h"
#include "whisper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct AudioBuffer {
  int32_t sample_rate = 0;
  std::vector<float> samples;
};

struct FmtChunk {
  uint16_t audio_format = 0;
  uint16_t channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
};

struct BackendStats {
  std::string name;
  double total_ms = 0.0;
  double stage1_ms = 0.0;
  double stage2_ms = 0.0;
  int segments = 0;
  std::size_t text_chars = 0;
};

uint16_t ReadLe16(const unsigned char* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t ReadLe32(const unsigned char* p) {
  return static_cast<uint32_t>(p[0] | (p[1] << 8) | (p[2] << 16)
                               | (p[3] << 24));
}

std::string Trim(std::string_view value) {
  std::size_t start = 0;
  while (start < value.size()
         && std::isspace(static_cast<unsigned char>(value[start]))) {
    ++start;
  }
  std::size_t end = value.size();
  while (end > start
         && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return std::string(value.substr(start, end - start));
}

AudioBuffer ReadWavFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open WAV file: " + path.string());
  }

  std::vector<unsigned char> data((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
  if (data.size() < 44) {
    throw std::runtime_error("WAV file too small: " + path.string());
  }
  if (std::memcmp(data.data(), "RIFF", 4) != 0
      || std::memcmp(data.data() + 8, "WAVE", 4) != 0) {
    throw std::runtime_error("Unsupported WAV header: " + path.string());
  }

  std::optional<FmtChunk> fmt;
  const unsigned char* pcm = nullptr;
  uint32_t pcm_size = 0;
  std::size_t offset = 12;
  while (offset + 8 <= data.size()) {
    const unsigned char* chunk = data.data() + offset;
    const uint32_t chunk_size = ReadLe32(chunk + 4);
    const std::size_t chunk_data = offset + 8;
    const std::size_t next = chunk_data + chunk_size + (chunk_size & 1u);
    if (next > data.size()) {
      break;
    }

    if (std::memcmp(chunk, "fmt ", 4) == 0 && chunk_size >= 16) {
      FmtChunk parsed;
      parsed.audio_format = ReadLe16(data.data() + chunk_data + 0);
      parsed.channels = ReadLe16(data.data() + chunk_data + 2);
      parsed.sample_rate = ReadLe32(data.data() + chunk_data + 4);
      parsed.bits_per_sample = ReadLe16(data.data() + chunk_data + 14);
      fmt = parsed;
    } else if (std::memcmp(chunk, "data", 4) == 0) {
      pcm = data.data() + chunk_data;
      pcm_size = chunk_size;
    }

    offset = next;
  }

  if (!fmt.has_value() || pcm == nullptr || pcm_size == 0) {
    throw std::runtime_error("Missing fmt/data chunk in WAV file: "
                             + path.string());
  }
  if (fmt->channels == 0) {
    throw std::runtime_error("Invalid channel count in WAV file: "
                             + path.string());
  }

  AudioBuffer out;
  out.sample_rate = static_cast<int32_t>(fmt->sample_rate);
  if (fmt->audio_format == 1 && fmt->bits_per_sample == 16) {
    const std::size_t total_samples = pcm_size / sizeof(int16_t);
    const std::size_t frames = total_samples / fmt->channels;
    out.samples.resize(frames);
    auto* src = reinterpret_cast<const int16_t*>(pcm);
    for (std::size_t i = 0; i < frames; ++i) {
      double mixed = 0.0;
      for (uint16_t ch = 0; ch < fmt->channels; ++ch) {
        mixed += static_cast<double>(src[i * fmt->channels + ch]) / 32768.0;
      }
      out.samples[i] = static_cast<float>(mixed / fmt->channels);
    }
    return out;
  }

  if (fmt->audio_format == 3 && fmt->bits_per_sample == 32) {
    const std::size_t total_samples = pcm_size / sizeof(float);
    const std::size_t frames = total_samples / fmt->channels;
    out.samples.resize(frames);
    auto* src = reinterpret_cast<const float*>(pcm);
    for (std::size_t i = 0; i < frames; ++i) {
      double mixed = 0.0;
      for (uint16_t ch = 0; ch < fmt->channels; ++ch) {
        mixed += static_cast<double>(src[i * fmt->channels + ch]);
      }
      out.samples[i] = static_cast<float>(mixed / fmt->channels);
    }
    return out;
  }

  throw std::runtime_error("Unsupported WAV format in " + path.string()
                           + " (format=" + std::to_string(fmt->audio_format)
                           + ", bits=" + std::to_string(fmt->bits_per_sample)
                           + ")");
}

template <typename Fn>
double MeasureMs(Fn&& fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

BackendStats RunSherpaBench(const std::filesystem::path& repo_root,
                            const AudioBuffer& audio,
                            int num_threads) {
  const auto sherpa_dir = repo_root / "models/sherpa-onnx";
  const auto asr_dir = sherpa_dir / "sherpa-onnx-whisper-tiny.en";
  const auto seg_model = sherpa_dir / "sherpa-onnx-pyannote-segmentation-3-0"
                         / "model.int8.onnx";
  const auto emb_model = sherpa_dir / "nemo_en_titanet_small.onnx";

  sherpa_onnx::cxx::OfflineRecognizerConfig recognizer_config;
  recognizer_config.feat_config.sample_rate = audio.sample_rate;
  recognizer_config.model_config.model_type = "whisper";
  recognizer_config.model_config.whisper.encoder
      = (asr_dir / "tiny.en-encoder.int8.onnx").string();
  recognizer_config.model_config.whisper.decoder
      = (asr_dir / "tiny.en-decoder.int8.onnx").string();
  recognizer_config.model_config.whisper.language = "en";
  recognizer_config.model_config.whisper.task = "transcribe";
  recognizer_config.model_config.tokens
      = (asr_dir / "tiny.en-tokens.txt").string();
  recognizer_config.model_config.num_threads = num_threads;
  recognizer_config.model_config.provider = "cpu";
  auto recognizer = sherpa_onnx::cxx::OfflineRecognizer::Create(recognizer_config);

  SherpaOnnxOfflineSpeakerDiarizationConfig diarization_config;
  std::memset(&diarization_config, 0, sizeof(diarization_config));
  const std::string seg_model_str = seg_model.string();
  const std::string emb_model_str = emb_model.string();
  diarization_config.segmentation.pyannote.model = seg_model_str.c_str();
  diarization_config.segmentation.num_threads = num_threads;
  diarization_config.segmentation.provider = "cpu";
  diarization_config.embedding.model = emb_model_str.c_str();
  diarization_config.embedding.num_threads = num_threads;
  diarization_config.embedding.provider = "cpu";
  diarization_config.clustering.num_clusters = 0;
  diarization_config.clustering.threshold = 0.72f;
  diarization_config.min_duration_on = 0.20f;
  diarization_config.min_duration_off = 0.15f;
  const SherpaOnnxOfflineSpeakerDiarization* diarizer
      = SherpaOnnxCreateOfflineSpeakerDiarization(&diarization_config);
  if (diarizer == nullptr) {
    throw std::runtime_error("Failed to create Sherpa diarizer");
  }

  BackendStats stats;
  stats.name = "Sherpa diarization";
  const auto* result = static_cast<const SherpaOnnxOfflineSpeakerDiarizationResult*>(nullptr);
  const auto* sorted = static_cast<const SherpaOnnxOfflineSpeakerDiarizationSegment*>(nullptr);
  stats.stage1_ms = MeasureMs([&] {
    result = SherpaOnnxOfflineSpeakerDiarizationProcess(
        diarizer, audio.samples.data(),
        static_cast<int32_t>(audio.samples.size()));
  });
  if (result == nullptr) {
    SherpaOnnxDestroyOfflineSpeakerDiarization(diarizer);
    throw std::runtime_error("Sherpa diarization returned null result");
  }

  stats.segments = SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result);
  sorted = SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result);
  stats.stage2_ms = MeasureMs([&] {
    for (int i = 0; i < stats.segments; ++i) {
      const auto& segment = sorted[i];
      const int32_t start_sample = std::max(
          0, static_cast<int32_t>(std::floor(segment.start * audio.sample_rate)));
      const int32_t end_sample = std::min(
          static_cast<int32_t>(audio.samples.size()),
          static_cast<int32_t>(std::ceil(segment.end * audio.sample_rate)));
      const int32_t segment_samples = end_sample - start_sample;
      if (segment_samples <= 0) {
        continue;
      }
      auto stream = recognizer.CreateStream();
      stream.AcceptWaveform(audio.sample_rate,
                            audio.samples.data() + start_sample,
                            segment_samples);
      recognizer.Decode(&stream);
      stats.text_chars += recognizer.GetResult(&stream).text.size();
    }
  });
  stats.total_ms = stats.stage1_ms + stats.stage2_ms;

  SherpaOnnxOfflineSpeakerDiarizationDestroySegment(sorted);
  SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);
  SherpaOnnxDestroyOfflineSpeakerDiarization(diarizer);
  return stats;
}

BackendStats RunWhisperBench(const std::filesystem::path& repo_root,
                             const AudioBuffer& audio,
                             int num_threads) {
  const auto model_path = repo_root / "models/whisper.cpp/ggml-small.en-tdrz.bin";
  whisper_context_params context_params = whisper_context_default_params();
  context_params.use_gpu = true;
  context_params.flash_attn = true;
  whisper_context* ctx
      = whisper_init_from_file_with_params(model_path.string().c_str(),
                                           context_params);
  if (ctx == nullptr) {
    throw std::runtime_error("Failed to initialize whisper.cpp model");
  }
  whisper_state* state = whisper_init_state(ctx);
  if (state == nullptr) {
    whisper_free(ctx);
    throw std::runtime_error("Failed to initialize whisper.cpp state");
  }

  whisper_full_params params
      = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
  params.n_threads = std::max(1, num_threads);
  params.translate = false;
  params.no_context = true;
  params.no_timestamps = false;
  params.single_segment = false;
  params.print_special = false;
  params.print_progress = false;
  params.print_realtime = false;
  params.print_timestamps = false;
  params.token_timestamps = false;
  params.max_tokens = 0;
  params.audio_ctx = 0;
  params.language = "en";
  params.tdrz_enable = true;

  BackendStats stats;
  stats.name = "Whisper.cpp tinydiarize";
  stats.total_ms = MeasureMs([&] {
    if (whisper_full_with_state(
            ctx, state, params, audio.samples.data(),
            static_cast<int>(audio.samples.size()))
        != 0) {
      throw std::runtime_error("whisper_full_with_state failed");
    }
  });
  stats.stage1_ms = stats.total_ms;
  stats.segments = whisper_full_n_segments_from_state(state);
  for (int i = 0; i < stats.segments; ++i) {
    const char* text = whisper_full_get_segment_text_from_state(state, i);
    stats.text_chars += Trim(text != nullptr ? text : "").size();
  }

  whisper_free_state(state);
  whisper_free(ctx);
  return stats;
}

void PrintStats(const BackendStats& stats, double audio_seconds) {
  const double rtf = audio_seconds > 0.0 ? stats.total_ms / (audio_seconds * 1000.0)
                                         : 0.0;
  std::cout << stats.name << '\n'
            << "  total_ms: " << std::fixed << std::setprecision(2)
            << stats.total_ms << '\n'
            << "  stage1_ms: " << stats.stage1_ms << '\n'
            << "  stage2_ms: " << stats.stage2_ms << '\n'
            << "  segments: " << stats.segments << '\n'
            << "  text_chars: " << stats.text_chars << '\n'
            << "  realtime_factor: " << rtf << "x\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3) {
      std::cerr << "usage: " << argv[0] << " <wav-path> [num_threads]\n";
      return 1;
    }

    const std::filesystem::path wav_path = argv[1];
    const int num_threads = argc >= 3 ? std::max(1, std::stoi(argv[2])) : 2;
    const std::filesystem::path repo_root = std::filesystem::current_path();

    const AudioBuffer audio = ReadWavFile(wav_path);
    if (audio.sample_rate != 16000) {
      throw std::runtime_error("Benchmark WAV must be 16 kHz mono PCM/float");
    }
    const double audio_seconds
        = static_cast<double>(audio.samples.size())
          / static_cast<double>(audio.sample_rate);

    std::cout << "audio: " << wav_path << '\n'
              << "duration_s: " << std::fixed << std::setprecision(3)
              << audio_seconds << '\n'
              << "threads: " << num_threads << "\n\n";

    const BackendStats sherpa = RunSherpaBench(repo_root, audio, num_threads);
    const BackendStats whisper = RunWhisperBench(repo_root, audio, num_threads);

    PrintStats(sherpa, audio_seconds);
    std::cout << '\n';
    PrintStats(whisper, audio_seconds);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "benchmark failed: " << ex.what() << '\n';
    return 1;
  }
}
