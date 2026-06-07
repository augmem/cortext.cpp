#include "webcam_session.hpp"

#include <utility>

namespace chat {

struct WebcamSession::Impl {
  explicit Impl(WebcamSessionConfig cfg) : config(std::move(cfg)) {}

  WebcamSessionConfig config;
};

WebcamSession::WebcamSession(WebcamSessionConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

WebcamSession::~WebcamSession() = default;

bool WebcamSession::IsSupported() const {
  return false;
}

bool WebcamSession::IsAvailable() const {
  return false;
}

bool WebcamSession::Start() {
  if (impl_ && impl_->config.on_error) {
    impl_->config.on_error("Direct webcam capture is only implemented for macOS builds.");
  }
  return false;
}

void WebcamSession::Stop() {}

}  // namespace chat
