#include "voice_session.hpp"

namespace chat {

struct VoiceSession::Impl {
  explicit Impl(VoiceSessionConfig cfg) : config(std::move(cfg)) {}

  VoiceSessionConfig config;
  bool available = false;
};

VoiceSession::VoiceSession(VoiceSessionConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

VoiceSession::~VoiceSession() = default;

bool VoiceSession::IsSupported() const {
  return false;
}

bool VoiceSession::IsAvailable() const {
  return impl_ && impl_->available;
}

bool VoiceSession::HasSpeakerAttribution() const {
  return false;
}

bool VoiceSession::Start() {
  if (impl_ && impl_->config.on_error) {
    impl_->config.on_error("Voice chat is only implemented for macOS builds.");
  }
  return false;
}

void VoiceSession::Stop() {}

void VoiceSession::QueueAssistantText(const std::string& text) {
  (void)text;
}

void VoiceSession::CancelAssistantReply() {}

}  // namespace chat
