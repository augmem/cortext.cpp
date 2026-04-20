#include "voice_session.hpp"

#import <AVFoundation/AVFoundation.h>

#include "sherpa-onnx/c-api/c-api.h"
#include "sherpa-onnx/c-api/cxx-api.h"
#include "whisper.h"

#include "cortext/telemetry/telemetry.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace chat {

namespace {

bool FileExists(const std::filesystem::path& path) {
  std::error_code ec;
  return !path.empty() && std::filesystem::exists(path, ec);
}

bool UsingWhisperBackend(const VoiceSessionConfig& config) {
  return config.backend == "whisper";
}

std::string Trim(const std::string& value) {
  const auto start = value.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return {};
  }
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(start, end - start + 1);
}

float ComputeRms(const std::vector<float>& samples) {
  if (samples.empty()) {
    return 0.0f;
  }
  double sum = 0.0;
  for (float sample : samples) {
    sum += static_cast<double>(sample) * static_cast<double>(sample);
  }
  return static_cast<float>(std::sqrt(sum / static_cast<double>(samples.size())));
}

float CosineSimilarity(const std::vector<float>& lhs,
                       const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size() || lhs.empty()) {
    return 0.0f;
  }
  double dot = 0.0;
  double lhs_norm = 0.0;
  double rhs_norm = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    dot += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
    lhs_norm += static_cast<double>(lhs[i]) * static_cast<double>(lhs[i]);
    rhs_norm += static_cast<double>(rhs[i]) * static_cast<double>(rhs[i]);
  }
  if (lhs_norm <= 0.0 || rhs_norm <= 0.0) {
    return 0.0f;
  }
  return static_cast<float>(dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm)));
}

void NormalizeEmbedding(std::vector<float>& embedding) {
  double norm = 0.0;
  for (float value : embedding) {
    norm += static_cast<double>(value) * static_cast<double>(value);
  }
  if (norm <= 0.0) {
    return;
  }
  norm = std::sqrt(norm);
  for (float& value : embedding) {
    value = static_cast<float>(static_cast<double>(value) / norm);
  }
}

class MacAudioBridge {
public:
  MacAudioBridge(int sample_rate,
                 std::function<void(std::vector<float>&&)> on_samples,
                 std::function<void(const std::string&)> on_error)
      : sample_rate_(sample_rate),
        on_samples_(std::move(on_samples)),
        on_error_(std::move(on_error)) {
    capture_engine_ = [[AVAudioEngine alloc] init];
    playback_engine_ = [[AVAudioEngine alloc] init];
    player_node_ = [[AVAudioPlayerNode alloc] init];

    [playback_engine_ attachNode:player_node_];
    AVAudioMixerNode* mixer = [playback_engine_ mainMixerNode];
    AVAudioFormat* output_format = [mixer outputFormatForBus:0];
    playback_output_format_ = [output_format retain];
    [playback_engine_ connect:player_node_ to:mixer format:output_format];

    NSError* playback_error = nil;
    if (![playback_engine_ startAndReturnError:&playback_error]) {
      ReportError(playback_error, "Failed to start playback engine");
    }
  }

  ~MacAudioBridge() {
    StopCapture();
    StopPlayback();
    if (player_node_ != nil) {
      [player_node_ release];
      player_node_ = nil;
    }
    if (playback_output_format_ != nil) {
      [playback_output_format_ release];
      playback_output_format_ = nil;
    }
    if (playback_engine_ != nil) {
      [playback_engine_ stop];
      [playback_engine_ release];
      playback_engine_ = nil;
    }
    if (capture_converter_ != nil) {
      [capture_converter_ release];
      capture_converter_ = nil;
    }
    if (target_capture_format_ != nil) {
      [target_capture_format_ release];
      target_capture_format_ = nil;
    }
    if (capture_engine_ != nil) {
      [capture_engine_ release];
      capture_engine_ = nil;
    }
  }

  bool StartCapture() {
    @autoreleasepool {
      if (capturing_) {
        return true;
      }

      AVAudioInputNode* input = [capture_engine_ inputNode];
      if (input == nil) {
        ReportError(nil, "No audio input device is available");
        return false;
      }

      AVAudioFormat* input_format = [input inputFormatForBus:0];
      if (input_format == nil) {
        ReportError(nil, "Failed to query audio input format");
        return false;
      }

      if (target_capture_format_ != nil) {
        [target_capture_format_ release];
        target_capture_format_ = nil;
      }
      target_capture_format_ = [[AVAudioFormat alloc]
          initWithCommonFormat:AVAudioPCMFormatFloat32
                    sampleRate:sample_rate_
                      channels:1
                   interleaved:NO];
      if (capture_converter_ != nil) {
        [capture_converter_ release];
        capture_converter_ = nil;
      }
      capture_converter_ = [[AVAudioConverter alloc] initFromFormat:input_format
                                                           toFormat:target_capture_format_];
      if (capture_converter_ == nil) {
        ReportError(nil, "Failed to create microphone format converter");
        return false;
      }

      __block MacAudioBridge* weak_self = this;
      [input removeTapOnBus:0];
      [input installTapOnBus:0
                  bufferSize:1024
                      format:input_format
                       block:^(AVAudioPCMBuffer* buffer, AVAudioTime* when) {
        (void)when;
        if (weak_self == nullptr) {
          return;
        }
        weak_self->HandleCaptureBuffer(buffer, input_format);
      }];

      NSError* capture_error = nil;
      if (![capture_engine_ startAndReturnError:&capture_error]) {
        [input removeTapOnBus:0];
        ReportError(capture_error, "Failed to start microphone capture");
        return false;
      }

      capturing_ = true;
      return true;
    }
  }

  void StopCapture() {
    @autoreleasepool {
      if (!capturing_) {
        return;
      }
      AVAudioInputNode* input = [capture_engine_ inputNode];
      [input removeTapOnBus:0];
      [capture_engine_ stop];
      capturing_ = false;
    }
  }

  void PlayBlocking(const std::vector<float>& samples,
                    int32_t sample_rate,
                    std::atomic<bool>& cancelled) {
    if (samples.empty() || player_node_ == nil || playback_engine_ == nil
        || playback_output_format_ == nil) {
      return;
    }

    @autoreleasepool {
      std::unique_lock<std::mutex> lock(playback_mu_);
      playback_done_ = false;
      playback_cancelled_ = false;

      AVAudioFormat* source_format = [[[AVAudioFormat alloc]
          initWithCommonFormat:AVAudioPCMFormatFloat32
                    sampleRate:sample_rate
                      channels:1
                   interleaved:NO] autorelease];
      AVAudioPCMBuffer* source_buffer = [[[AVAudioPCMBuffer alloc]
          initWithPCMFormat:source_format
              frameCapacity:static_cast<AVAudioFrameCount>(samples.size())] autorelease];
      source_buffer.frameLength = static_cast<AVAudioFrameCount>(samples.size());
      std::memcpy(source_buffer.floatChannelData[0], samples.data(),
                  samples.size() * sizeof(float));

      AVAudioConverter* playback_converter = [[[AVAudioConverter alloc]
          initFromFormat:source_format
                 toFormat:playback_output_format_] autorelease];
      if (playback_converter == nil) {
        ReportError(nil, "Failed to create playback format converter");
        return;
      }

      const double ratio = playback_output_format_.sampleRate
                           / std::max(1.0, static_cast<double>(sample_rate));
      const AVAudioFrameCount target_frames
          = static_cast<AVAudioFrameCount>(std::ceil(samples.size() * ratio)) + 16;
      AVAudioPCMBuffer* buffer = [[[AVAudioPCMBuffer alloc]
          initWithPCMFormat:playback_output_format_
              frameCapacity:target_frames] autorelease];
      __block bool consumed = false;
      NSError* conversion_error = nil;
      AVAudioConverterInputBlock input_block
          = ^AVAudioBuffer* (AVAudioPacketCount inNumberOfPackets,
                             AVAudioConverterInputStatus* outStatus) {
        (void)inNumberOfPackets;
        if (consumed) {
          *outStatus = AVAudioConverterInputStatus_NoDataNow;
          return nil;
        }
        consumed = true;
        *outStatus = AVAudioConverterInputStatus_HaveData;
        return source_buffer;
      };

      const AVAudioConverterOutputStatus status
          = [playback_converter convertToBuffer:buffer
                                          error:&conversion_error
                             withInputFromBlock:input_block];
      if (status == AVAudioConverterOutputStatus_Error || conversion_error != nil
          || buffer.frameLength == 0) {
        ReportError(conversion_error, "Assistant playback conversion failed");
        return;
      }

      __block MacAudioBridge* weak_self = this;
      [player_node_ stop];
      [player_node_ scheduleBuffer:buffer completionHandler:^{
        if (weak_self == nullptr) {
          return;
        }
        std::lock_guard<std::mutex> guard(weak_self->playback_mu_);
        weak_self->playback_done_ = true;
        weak_self->playback_cv_.notify_all();
      }];
      [player_node_ play];

      playback_cv_.wait(lock, [&] {
        return playback_done_ || playback_cancelled_ || cancelled.load();
      });
      if (cancelled.load() || playback_cancelled_) {
        [player_node_ stop];
      }
    }
  }

  void StopPlayback() {
    std::lock_guard<std::mutex> lock(playback_mu_);
    playback_cancelled_ = true;
    playback_done_ = true;
    if (player_node_ != nil) {
      [player_node_ stop];
    }
    playback_cv_.notify_all();
  }

private:
  void HandleCaptureBuffer(AVAudioPCMBuffer* buffer, AVAudioFormat* input_format) {
    @autoreleasepool {
      if (buffer == nil || input_format == nil || capture_converter_ == nil
          || target_capture_format_ == nil) {
        return;
      }

      const double ratio = static_cast<double>(sample_rate_)
                           / std::max(1.0, input_format.sampleRate);
      const AVAudioFrameCount target_frames
          = static_cast<AVAudioFrameCount>(std::ceil(buffer.frameLength * ratio)) + 16;
      AVAudioPCMBuffer* converted = [[[AVAudioPCMBuffer alloc]
          initWithPCMFormat:target_capture_format_
              frameCapacity:target_frames] autorelease];
      if (converted == nil) {
        return;
      }

      __block bool consumed = false;
      NSError* conversion_error = nil;
      AVAudioConverterInputBlock input_block
          = ^AVAudioBuffer* (AVAudioPacketCount inNumberOfPackets,
                             AVAudioConverterInputStatus* outStatus) {
        (void)inNumberOfPackets;
        if (consumed) {
          *outStatus = AVAudioConverterInputStatus_NoDataNow;
          return nil;
        }
        consumed = true;
        *outStatus = AVAudioConverterInputStatus_HaveData;
        return buffer;
      };

      const AVAudioConverterOutputStatus status
          = [capture_converter_ convertToBuffer:converted
                                          error:&conversion_error
                             withInputFromBlock:input_block];
      if (status == AVAudioConverterOutputStatus_Error || conversion_error != nil) {
        ReportError(conversion_error, "Microphone conversion failed");
        return;
      }
      if (converted.frameLength == 0 || converted.floatChannelData == nil) {
        return;
      }

      std::vector<float> pcm(converted.floatChannelData[0],
                             converted.floatChannelData[0] + converted.frameLength);
      if (on_samples_) {
        on_samples_(std::move(pcm));
      }
    }
  }

  void ReportError(NSError* error, const char* prefix) {
    if (!on_error_) {
      return;
    }
    std::string message = prefix;
    if (error != nil && error.localizedDescription != nil) {
      message += ": ";
      message += [[error localizedDescription] UTF8String];
    }
    on_error_(message);
  }

  const int sample_rate_;
  std::function<void(std::vector<float>&&)> on_samples_;
  std::function<void(const std::string&)> on_error_;

  AVAudioEngine* capture_engine_ = nil;
  AVAudioConverter* capture_converter_ = nil;
  AVAudioFormat* target_capture_format_ = nil;
  AVAudioEngine* playback_engine_ = nil;
  AVAudioPlayerNode* player_node_ = nil;
  AVAudioFormat* playback_output_format_ = nil;

  std::mutex playback_mu_;
  std::condition_variable playback_cv_;
  bool playback_done_ = false;
  bool playback_cancelled_ = false;
  bool capturing_ = false;
};

}  // namespace

struct VoiceSession::Impl {
  explicit Impl(VoiceSessionConfig cfg)
      : config(std::move(cfg)),
        audio_bridge(config.sample_rate,
                     [this](std::vector<float>&& chunk) { EnqueueAudio(std::move(chunk)); },
                     [this](const std::string& error) { ReportError(error); }) {
    try {
      ValidateAssets();
      CreateAsr();
      CreateTts();
      CreateSpeakerAttribution();
      CreateSpeakerDiarization();
      available = true;
    } catch (const std::exception& ex) {
      ReportError(ex.what());
      available = false;
      return;
    }

    audio_thread = std::thread([this] { RunAudioLoop(); });
    tts_thread = std::thread([this] { RunTtsLoop(); });
  }

  ~Impl() {
    shutdown.store(true);
    StopListening();
    CancelAssistantReply();
    {
      std::lock_guard<std::mutex> lock(audio_mu);
      audio_cv.notify_all();
    }
    {
      std::lock_guard<std::mutex> lock(tts_mu);
      tts_cv.notify_all();
    }
    if (audio_thread.joinable()) {
      audio_thread.join();
    }
    if (tts_thread.joinable()) {
      tts_thread.join();
    }
    if (speaker_extractor != nullptr) {
      SherpaOnnxDestroySpeakerEmbeddingExtractor(speaker_extractor);
      speaker_extractor = nullptr;
    }
    if (whisper_state != nullptr) {
      whisper_free_state(whisper_state);
      whisper_state = nullptr;
    }
    if (whisper_ctx != nullptr) {
      whisper_free(whisper_ctx);
      whisper_ctx = nullptr;
    }
    if (speaker_diarizer != nullptr) {
      SherpaOnnxDestroyOfflineSpeakerDiarization(speaker_diarizer);
      speaker_diarizer = nullptr;
    }
  }

  void ValidateAssets() {
    std::vector<std::filesystem::path> required = {
        config.tts_model,
        config.tts_tokens,
        config.tts_voices,
        config.tts_data_dir,
    };
    if (UsingWhisperBackend(config)) {
      required.push_back(config.whisper_model);
    } else {
      required.push_back(config.asr_encoder);
      required.push_back(config.asr_decoder);
      required.push_back(config.asr_tokens);
    }
    for (const auto& path : required) {
      if (!FileExists(path)) {
        throw std::runtime_error("Missing voice asset: " + path.string());
      }
    }
  }

  void CreateAsr() {
    if (UsingWhisperBackend(config)) {
      whisper_context_params context_params = whisper_context_default_params();
      context_params.use_gpu = true;
      context_params.flash_attn = true;
      whisper_ctx = whisper_init_from_file_with_params(
          config.whisper_model.string().c_str(), context_params);
      if (whisper_ctx == nullptr) {
        throw std::runtime_error("Failed to initialize whisper.cpp model: "
                                 + config.whisper_model.string());
      }
      whisper_state = whisper_init_state(whisper_ctx);
      if (whisper_state == nullptr) {
        whisper_free(whisper_ctx);
        whisper_ctx = nullptr;
        throw std::runtime_error("Failed to initialize whisper.cpp state.");
      }
      return;
    }

    sherpa_onnx::cxx::OfflineRecognizerConfig recognizer_config;
    recognizer_config.feat_config.sample_rate = config.sample_rate;
    recognizer_config.model_config.model_type = "whisper";
    recognizer_config.model_config.whisper.encoder = config.asr_encoder.string();
    recognizer_config.model_config.whisper.decoder = config.asr_decoder.string();
    recognizer_config.model_config.whisper.language = "en";
    recognizer_config.model_config.whisper.task = "transcribe";
    recognizer_config.model_config.tokens = config.asr_tokens.string();
    recognizer_config.model_config.num_threads = config.num_threads;
    recognizer_config.model_config.provider = "cpu";

    recognizer = sherpa_onnx::cxx::OfflineRecognizer::Create(recognizer_config);
  }

  void CreateTts() {
    sherpa_onnx::cxx::OfflineTtsConfig tts_config;
    tts_config.model.num_threads = config.num_threads;
    tts_config.model.provider = "cpu";
    tts_config.model.kitten.model = config.tts_model.string();
    tts_config.model.kitten.tokens = config.tts_tokens.string();
    tts_config.model.kitten.voices = config.tts_voices.string();
    tts_config.model.kitten.data_dir = config.tts_data_dir.string();
    tts = sherpa_onnx::cxx::OfflineTts::Create(tts_config);
  }

  void CreateSpeakerAttribution() {
    if (UsingWhisperBackend(config)) {
      return;
    }
    if (config.speaker_embedding_model.empty()) {
      return;
    }
    if (!FileExists(config.speaker_embedding_model)) {
      ReportError("Speaker attribution model not found: "
                  + config.speaker_embedding_model.string());
      return;
    }
    const std::string model_path = config.speaker_embedding_model.string();
    SherpaOnnxSpeakerEmbeddingExtractorConfig speaker_config;
    std::memset(&speaker_config, 0, sizeof(speaker_config));
    speaker_config.model = model_path.c_str();
    speaker_config.num_threads = config.num_threads;
    speaker_config.provider = "cpu";

    speaker_extractor = SherpaOnnxCreateSpeakerEmbeddingExtractor(&speaker_config);
    if (speaker_extractor == nullptr) {
      ReportError("Speaker attribution is unavailable. Failed to create speaker embedding extractor.");
      return;
    }
    speaker_embedding_dim = SherpaOnnxSpeakerEmbeddingExtractorDim(speaker_extractor);
    if (speaker_embedding_dim <= 0) {
      SherpaOnnxDestroySpeakerEmbeddingExtractor(speaker_extractor);
      speaker_extractor = nullptr;
      ReportError("Speaker attribution is unavailable. Invalid embedding dimension.");
    }
  }

  void CreateSpeakerDiarization() {
    if (UsingWhisperBackend(config)) {
      return;
    }
    if (config.speaker_segmentation_model.empty()
        || config.speaker_embedding_model.empty()) {
      return;
    }
    if (!FileExists(config.speaker_segmentation_model)) {
      ReportError("Speaker segmentation model not found: "
                  + config.speaker_segmentation_model.string());
      return;
    }
    const std::string segmentation_model = config.speaker_segmentation_model.string();
    const std::string embedding_model = config.speaker_embedding_model.string();
    SherpaOnnxOfflineSpeakerDiarizationConfig diarization_config;
    std::memset(&diarization_config, 0, sizeof(diarization_config));
    diarization_config.segmentation.pyannote.model = segmentation_model.c_str();
    diarization_config.segmentation.num_threads = config.num_threads;
    diarization_config.segmentation.provider = "cpu";
    diarization_config.embedding.model = embedding_model.c_str();
    diarization_config.embedding.num_threads = config.num_threads;
    diarization_config.embedding.provider = "cpu";
    diarization_config.clustering.num_clusters = 0;
    diarization_config.clustering.threshold = 0.72f;
    diarization_config.min_duration_on = 0.20f;
    diarization_config.min_duration_off = 0.15f;

    speaker_diarizer
        = SherpaOnnxCreateOfflineSpeakerDiarization(&diarization_config);
    if (speaker_diarizer == nullptr) {
      ReportError("Speaker diarization is unavailable. Failed to create diarizer.");
      cortext::telemetry::LogError("chat.voice.diarization_init_failed");
    } else {
      cortext::telemetry::LogInfo("chat.voice.diarization_init", {
        cortext::telemetry::Attribute::String(
            "segmentation_model", config.speaker_segmentation_model.string()),
        cortext::telemetry::Attribute::String(
            "embedding_model", config.speaker_embedding_model.string()),
      });
    }
  }

  bool StartListening() {
    if (!available) {
      return false;
    }
    if (listening.exchange(true)) {
      return true;
    }
    const bool started = audio_bridge.StartCapture();
    if (!started) {
      listening.store(false);
      return false;
    }
    if (config.on_listening_changed) {
      config.on_listening_changed(true);
    }
    return true;
  }

  void StopListening() {
    if (!listening.exchange(false)) {
      return;
    }
    CancelAssistantReply();
    audio_bridge.StopCapture();
    ResetAsrState();
    if (config.on_partial_transcript) {
      config.on_partial_transcript("");
    }
    if (config.on_listening_changed) {
      config.on_listening_changed(false);
    }
  }

  void QueueAssistantText(const std::string& text) {
    const std::string trimmed = Trim(text);
    if (trimmed.empty() || !available) {
      return;
    }
    tts_cancelled.store(false);
    {
      std::lock_guard<std::mutex> lock(tts_mu);
      tts_queue.push_back(trimmed);
    }
    tts_cv.notify_one();
  }

  void CancelAssistantReply() {
    tts_cancelled.store(true);
    {
      std::lock_guard<std::mutex> lock(tts_mu);
      tts_queue.clear();
    }
    audio_bridge.StopPlayback();
    tts_cv.notify_all();
    if (assistant_playing.exchange(false) && config.on_playback_changed) {
      config.on_playback_changed(false);
    }
  }

  void EnqueueAudio(std::vector<float>&& chunk) {
    if (shutdown.load() || !listening.load() || chunk.empty()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(audio_mu);
      audio_queue.push_back(std::move(chunk));
    }
    audio_cv.notify_one();
  }

  void RunAudioLoop() {
    while (!shutdown.load()) {
      std::vector<float> chunk;
      {
        std::unique_lock<std::mutex> lock(audio_mu);
        audio_cv.wait(lock, [&] {
          return shutdown.load() || !audio_queue.empty();
        });
        if (shutdown.load()) {
          break;
        }
        chunk = std::move(audio_queue.front());
        audio_queue.pop_front();
      }
      ProcessAudioChunk(chunk);
    }
  }

  void ProcessAudioChunk(const std::vector<float>& chunk) {
    if (chunk.empty()) {
      return;
    }

    const float rms = ComputeRms(chunk);
    constexpr float kSpeechStartRms = 0.0125f;
    constexpr float kSpeechContinueRms = 0.0080f;
    const int silence_finalize_samples = static_cast<int>(config.sample_rate * 0.70f);
    const int min_utterance_samples = static_cast<int>(config.sample_rate * 0.25f);
    const int min_partial_samples = static_cast<int>(config.sample_rate * 0.35f);
    const int partial_step_samples = static_cast<int>(config.sample_rate * 0.45f);

    if (rms >= kSpeechStartRms && !speech_active.load()) {
      speech_active.store(true);
      trailing_silence_samples = 0;
      utterance_samples.clear();
      last_partial.clear();
      last_partial_sample_count = 0;
      CancelAssistantReply();
      if (config.on_user_speech_start) {
        config.on_user_speech_start();
      }
    }

    if (!speech_active.load()) {
      return;
    }

    utterance_samples.insert(utterance_samples.end(), chunk.begin(), chunk.end());
    if (rms >= kSpeechContinueRms) {
      trailing_silence_samples = 0;
    } else {
      trailing_silence_samples += static_cast<int>(chunk.size());
    }

    try {
      if (!recognizer.has_value()) {
        return;
      }

      const int voiced_samples = static_cast<int>(utterance_samples.size()) - trailing_silence_samples;
      if (voiced_samples >= min_partial_samples
          && voiced_samples - last_partial_sample_count >= partial_step_samples) {
        const std::string partial = TranscribeSamples(utterance_samples.data(),
                                                      static_cast<int32_t>(utterance_samples.size()));
        if (!partial.empty() && partial != last_partial) {
          last_partial = partial;
          if (config.on_partial_transcript) {
            config.on_partial_transcript(partial);
          }
        }
        last_partial_sample_count = voiced_samples;
      }

      if (trailing_silence_samples >= silence_finalize_samples
          && voiced_samples >= min_utterance_samples) {
        const int final_samples = std::max(0, voiced_samples);
        bool emitted = false;
        if (UsingWhisperBackend(config)) {
          emitted = WhisperTinydiarizeAndEmitSegments(utterance_samples.data(),
                                                      final_samples);
        } else if (speaker_diarizer != nullptr) {
          emitted = DiarizeAndEmitSegments(utterance_samples.data(), final_samples);
        }
        if (!emitted) {
          const std::string final_text
              = TranscribeSamples(utterance_samples.data(), final_samples);
          const std::string speaker_id
              = IdentifySpeaker(utterance_samples.data(), final_samples);
          if (!final_text.empty() && config.on_final_transcript) {
            config.on_final_transcript(VoiceFinalTranscript{final_text, speaker_id});
          }
        }
        ResetAsrState();
        if (config.on_partial_transcript) {
          config.on_partial_transcript("");
        }
      }
    } catch (const std::exception& ex) {
      ReportError(std::string("Voice ASR failed: ") + ex.what());
    }
  }

  void RunTtsLoop() {
    while (!shutdown.load()) {
      std::string text;
      {
        std::unique_lock<std::mutex> lock(tts_mu);
        tts_cv.wait(lock, [&] {
          return shutdown.load() || !tts_queue.empty();
        });
        if (shutdown.load()) {
          break;
        }
        text = std::move(tts_queue.front());
        tts_queue.pop_front();
      }

      if (tts_cancelled.exchange(false)) {
        continue;
      }

      try {
        auto audio = tts->Generate(text, 0, config.tts_speed);
        if (audio.samples.empty()) {
          continue;
        }
        assistant_playing.store(true);
        if (config.on_playback_changed) {
          config.on_playback_changed(true);
        }
        audio_bridge.PlayBlocking(audio.samples, audio.sample_rate, tts_cancelled);
      } catch (const std::exception& ex) {
        ReportError(std::string("Voice TTS failed: ") + ex.what());
      }

      if (assistant_playing.exchange(false) && config.on_playback_changed) {
        config.on_playback_changed(false);
      }
    }
  }

  std::string TranscribeSamples(const float* samples, int32_t sample_count) {
    if (UsingWhisperBackend(config)) {
      return TranscribeSamplesWithWhisper(samples, sample_count, false);
    }
    if (!recognizer.has_value() || samples == nullptr || sample_count <= 0) {
      return {};
    }
    auto offline_stream = recognizer->CreateStream();
    offline_stream.AcceptWaveform(config.sample_rate, samples, sample_count);
    recognizer->Decode(&offline_stream);
    return Trim(recognizer->GetResult(&offline_stream).text);
  }

  std::string TranscribeSamplesWithWhisper(const float* samples,
                                           int32_t sample_count,
                                           bool enable_tinydiarize) {
    if (whisper_ctx == nullptr || whisper_state == nullptr || samples == nullptr
        || sample_count <= 0) {
      return {};
    }

    whisper_full_params params
        = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = std::max(1, config.num_threads);
    params.translate = false;
    params.no_context = true;
    params.no_timestamps = !enable_tinydiarize;
    params.single_segment = false;
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.token_timestamps = false;
    params.max_tokens = 0;
    params.audio_ctx = 0;
    params.language = "en";
    params.tdrz_enable = enable_tinydiarize;

    if (whisper_full_with_state(
            whisper_ctx, whisper_state, params, samples, sample_count)
        != 0) {
      return {};
    }

    if (enable_tinydiarize) {
      return {};
    }

    const int segment_count = whisper_full_n_segments_from_state(whisper_state);
    std::string combined;
    for (int i = 0; i < segment_count; ++i) {
      const char* text
          = whisper_full_get_segment_text_from_state(whisper_state, i);
      const std::string trimmed = Trim(text != nullptr ? text : "");
      if (trimmed.empty()) {
        continue;
      }
      if (!combined.empty()) {
        combined.push_back(' ');
      }
      combined += trimmed;
    }
    return Trim(combined);
  }

  std::vector<float> ComputeSpeakerEmbedding(const float* samples,
                                             int32_t sample_count) {
    if (speaker_extractor == nullptr || samples == nullptr || sample_count <= 0) {
      return {};
    }
    const SherpaOnnxOnlineStream* stream
        = SherpaOnnxSpeakerEmbeddingExtractorCreateStream(speaker_extractor);
    if (stream == nullptr) {
      return {};
    }
    SherpaOnnxOnlineStreamAcceptWaveform(
        stream, config.sample_rate, samples, sample_count);
    SherpaOnnxOnlineStreamInputFinished(stream);
    if (!SherpaOnnxSpeakerEmbeddingExtractorIsReady(speaker_extractor, stream)) {
      SherpaOnnxDestroyOnlineStream(stream);
      return {};
    }
    const float* embedding_ptr
        = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(
            speaker_extractor, stream);
    if (embedding_ptr == nullptr) {
      SherpaOnnxDestroyOnlineStream(stream);
      return {};
    }
    std::vector<float> embedding(
        embedding_ptr, embedding_ptr + speaker_embedding_dim);
    SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(embedding_ptr);
    SherpaOnnxDestroyOnlineStream(stream);
    NormalizeEmbedding(embedding);
    return embedding;
  }

  std::string IdentifySpeaker(const float* samples, int32_t sample_count) {
    constexpr float kMinSpeakerSeconds = 0.45f;
    constexpr float kMatchThreshold = 0.72f;
    if (speaker_extractor == nullptr
        || sample_count
               < static_cast<int32_t>(config.sample_rate * kMinSpeakerSeconds)) {
      return {};
    }

    std::vector<float> embedding = ComputeSpeakerEmbedding(samples, sample_count);
    if (embedding.empty()) {
      return {};
    }

    float best_score = -1.0f;
    std::size_t best_index = 0;
    for (std::size_t i = 0; i < speaker_profiles.size(); ++i) {
      const float score = CosineSimilarity(embedding, speaker_profiles[i].embedding);
      if (score > best_score) {
        best_score = score;
        best_index = i;
      }
    }

    if (best_score >= kMatchThreshold && best_index < speaker_profiles.size()) {
      auto& profile = speaker_profiles[best_index];
      const float blend = 1.0f / static_cast<float>(profile.sample_count + 1);
      for (std::size_t i = 0; i < embedding.size(); ++i) {
        profile.embedding[i]
            = profile.embedding[i] * (1.0f - blend) + embedding[i] * blend;
      }
      NormalizeEmbedding(profile.embedding);
      profile.sample_count += 1;
      return profile.speaker_id;
    }

    SpeakerProfile profile;
    profile.speaker_id = "speaker:" + std::to_string(next_speaker_index++);
    profile.embedding = std::move(embedding);
    profile.sample_count = 1;
    speaker_profiles.push_back(std::move(profile));
    return speaker_profiles.back().speaker_id;
  }

  bool DiarizeAndEmitSegments(const float* samples, int32_t sample_count) {
    constexpr float kMinSegmentSeconds = 0.25f;
    if (speaker_diarizer == nullptr || samples == nullptr || sample_count <= 0
        || !config.on_final_transcript) {
      return false;
    }
    const auto* result = SherpaOnnxOfflineSpeakerDiarizationProcess(
        speaker_diarizer, samples, sample_count);
    if (result == nullptr) {
      cortext::telemetry::LogError("chat.voice.diarization_process_failed");
      return false;
    }

    const int32_t num_segments
        = SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result);
    const int32_t num_speakers
        = SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers(result);
    cortext::telemetry::LogInfo("chat.voice.diarization_result", {
      cortext::telemetry::Attribute::Int64("num_segments", num_segments),
      cortext::telemetry::Attribute::Int64("num_speakers", num_speakers),
      cortext::telemetry::Attribute::Double(
          "utterance_seconds",
          static_cast<double>(sample_count) / static_cast<double>(config.sample_rate)),
    });
    if (num_segments <= 0) {
      SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);
      return false;
    }

    const auto* segments
        = SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result);
    bool emitted = false;
    for (int32_t i = 0; i < num_segments; ++i) {
      const auto& segment = segments[i];
      const int32_t start_sample = std::max(
          0, static_cast<int32_t>(std::floor(segment.start * config.sample_rate)));
      const int32_t end_sample = std::min(
          sample_count,
          static_cast<int32_t>(std::ceil(segment.end * config.sample_rate)));
      const int32_t segment_samples = end_sample - start_sample;
      if (segment_samples
          < static_cast<int32_t>(config.sample_rate * kMinSegmentSeconds)) {
        continue;
      }
      const std::string text
          = TranscribeSamples(samples + start_sample, segment_samples);
      if (text.empty()) {
        continue;
      }
      std::string speaker_id
          = IdentifySpeaker(samples + start_sample, segment_samples);
      if (speaker_id.empty()) {
        speaker_id = "speaker:unknown";
      }
      if (config.on_segment_debug) {
        config.on_segment_debug(segment.start, segment.end, segment.speaker,
                                speaker_id, text);
      }
      cortext::telemetry::LogInfo("chat.voice.diarization_segment", {
        cortext::telemetry::Attribute::Double("start_s", segment.start),
        cortext::telemetry::Attribute::Double("end_s", segment.end),
        cortext::telemetry::Attribute::Int64("diarizer_speaker",
                                             static_cast<int64_t>(segment.speaker)),
        cortext::telemetry::Attribute::String("speaker_id", speaker_id),
        cortext::telemetry::Attribute::String(
            "text", text.size() <= 200 ? text : text.substr(0, 200) + "..."),
      });
      config.on_final_transcript(VoiceFinalTranscript{text, speaker_id});
      emitted = true;
    }

    SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segments);
    SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);
    return emitted;
  }

  bool WhisperTinydiarizeAndEmitSegments(const float* samples,
                                         int32_t sample_count) {
    if (whisper_ctx == nullptr || whisper_state == nullptr || samples == nullptr
        || sample_count <= 0 || !config.on_final_transcript) {
      return false;
    }

    (void)TranscribeSamplesWithWhisper(samples, sample_count, true);
    const int segment_count = whisper_full_n_segments_from_state(whisper_state);
    cortext::telemetry::LogInfo("chat.voice.whisper_tdrz_result", {
      cortext::telemetry::Attribute::Int64("num_segments", segment_count),
      cortext::telemetry::Attribute::Double(
          "utterance_seconds",
          static_cast<double>(sample_count)
              / static_cast<double>(config.sample_rate)),
    });
    if (segment_count <= 0) {
      return false;
    }

    bool emitted = false;
    int speaker_index = whisper_turn_speaker_index;
    for (int i = 0; i < segment_count; ++i) {
      const char* segment_text
          = whisper_full_get_segment_text_from_state(whisper_state, i);
      const std::string text = Trim(segment_text != nullptr ? segment_text : "");
      if (text.empty()) {
        continue;
      }

      const double start_s = 0.01 * static_cast<double>(
                                        whisper_full_get_segment_t0_from_state(
                                            whisper_state, i));
      const double end_s = 0.01 * static_cast<double>(
                                      whisper_full_get_segment_t1_from_state(
                                          whisper_state, i));
      const std::string speaker_id
          = "speaker:" + std::to_string(std::max(1, speaker_index));
      const bool speaker_turn_next
          = whisper_full_get_segment_speaker_turn_next_from_state(
              whisper_state, i);

      if (config.on_segment_debug) {
        config.on_segment_debug(static_cast<float>(start_s),
                                static_cast<float>(end_s),
                                speaker_index,
                                speaker_id,
                                text);
      }
      cortext::telemetry::LogInfo("chat.voice.whisper_tdrz_segment", {
        cortext::telemetry::Attribute::Double("start_s", start_s),
        cortext::telemetry::Attribute::Double("end_s", end_s),
        cortext::telemetry::Attribute::Int64("speaker_index", speaker_index),
        cortext::telemetry::Attribute::String("speaker_id", speaker_id),
        cortext::telemetry::Attribute::Bool("speaker_turn_next",
                                            speaker_turn_next),
      });
      config.on_final_transcript(VoiceFinalTranscript{text, speaker_id});
      emitted = true;

      if (speaker_turn_next) {
        speaker_index = speaker_index == 1 ? 2 : 1;
      }
    }

    whisper_turn_speaker_index = speaker_index;
    return emitted;
  }

  void ResetAsrState() {
    last_partial.clear();
    utterance_samples.clear();
    trailing_silence_samples = 0;
    last_partial_sample_count = 0;
    speech_active.store(false);
  }

  void ReportError(const std::string& error) {
    if (config.on_error) {
      config.on_error(error);
    }
  }

  VoiceSessionConfig config;
  bool available = false;

  std::atomic<bool> shutdown{false};
  std::atomic<bool> listening{false};
  std::atomic<bool> speech_active{false};
  std::atomic<bool> assistant_playing{false};
  std::atomic<bool> tts_cancelled{false};

  MacAudioBridge audio_bridge;
  std::optional<sherpa_onnx::cxx::OfflineRecognizer> recognizer;
  std::optional<sherpa_onnx::cxx::OfflineTts> tts;
  whisper_context* whisper_ctx = nullptr;
  whisper_state* whisper_state = nullptr;
  const SherpaOnnxSpeakerEmbeddingExtractor* speaker_extractor = nullptr;
  const SherpaOnnxOfflineSpeakerDiarization* speaker_diarizer = nullptr;
  int32_t speaker_embedding_dim = 0;
  int whisper_turn_speaker_index = 1;

  struct SpeakerProfile {
    std::string speaker_id;
    std::vector<float> embedding;
    int sample_count = 0;
  };
  std::vector<SpeakerProfile> speaker_profiles;
  int next_speaker_index = 1;

  std::mutex audio_mu;
  std::condition_variable audio_cv;
  std::deque<std::vector<float>> audio_queue;
  std::thread audio_thread;

  std::mutex tts_mu;
  std::condition_variable tts_cv;
  std::deque<std::string> tts_queue;
  std::thread tts_thread;

  std::string last_partial;
  std::vector<float> utterance_samples;
  int trailing_silence_samples = 0;
  int last_partial_sample_count = 0;
};

VoiceSession::VoiceSession(VoiceSessionConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

VoiceSession::~VoiceSession() = default;

bool VoiceSession::IsSupported() const {
  return true;
}

bool VoiceSession::IsAvailable() const {
  return impl_ && impl_->available;
}

bool VoiceSession::HasSpeakerAttribution() const {
  return impl_ && (impl_->speaker_diarizer != nullptr
                   || (UsingWhisperBackend(impl_->config)
                       && impl_->whisper_ctx != nullptr));
}

bool VoiceSession::Start() {
  return impl_ && impl_->StartListening();
}

void VoiceSession::Stop() {
  if (impl_) {
    impl_->StopListening();
  }
}

void VoiceSession::QueueAssistantText(const std::string& text) {
  if (impl_) {
    impl_->QueueAssistantText(text);
  }
}

void VoiceSession::CancelAssistantReply() {
  if (impl_) {
    impl_->CancelAssistantReply();
  }
}

}  // namespace chat
