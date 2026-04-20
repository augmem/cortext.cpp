#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace chat {

struct VoiceFinalTranscript {
  std::string text;
  std::string speaker_id;
};

struct VoiceSessionConfig {
  std::string backend = "sherpa";

  std::filesystem::path asr_encoder;
  std::filesystem::path asr_decoder;
  std::filesystem::path asr_joiner;
  std::filesystem::path asr_tokens;
  std::filesystem::path whisper_model;

  std::filesystem::path tts_model;
  std::filesystem::path tts_tokens;
  std::filesystem::path tts_voices;
  std::filesystem::path tts_data_dir;
  std::filesystem::path speaker_segmentation_model;
  std::filesystem::path speaker_embedding_model;

  int sample_rate = 16000;
  int num_threads = 2;
  float tts_speed = 1.0f;

  std::function<void(const std::string&)> on_partial_transcript;
  std::function<void(const VoiceFinalTranscript&)> on_final_transcript;
  std::function<void(float, float, int, const std::string&, const std::string&)> on_segment_debug;
  std::function<void()> on_user_speech_start;
  std::function<void(const std::string&)> on_error;
  std::function<void(bool)> on_listening_changed;
  std::function<void(bool)> on_playback_changed;
};

class VoiceSession {
public:
  explicit VoiceSession(VoiceSessionConfig config);
  ~VoiceSession();

  VoiceSession(const VoiceSession&) = delete;
  VoiceSession& operator=(const VoiceSession&) = delete;

  bool IsSupported() const;
  bool IsAvailable() const;
  bool HasSpeakerAttribution() const;
  bool Start();
  void Stop();

  void QueueAssistantText(const std::string& text);
  void CancelAssistantReply();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace chat
