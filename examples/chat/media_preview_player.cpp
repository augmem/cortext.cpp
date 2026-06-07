#include "media_preview_player.hpp"

namespace chat {

struct MediaPreviewPlayer::Impl {
  std::atomic<bool> playing{false};
};

MediaPreviewPlayer::MediaPreviewPlayer() : impl_(std::make_unique<Impl>()) {}

MediaPreviewPlayer::~MediaPreviewPlayer() = default;

bool MediaPreviewPlayer::IsAvailable() const {
  return false;
}

void MediaPreviewPlayer::PlayPcmF32(std::vector<float> samples,
                                    std::int32_t sample_rate) {
  (void)samples;
  (void)sample_rate;
}

void MediaPreviewPlayer::Stop() {}

bool MediaPreviewPlayer::IsPlaying() const {
  return impl_ && impl_->playing.load();
}

}  // namespace chat
