#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace chat {

class MediaPreviewPlayer {
public:
  MediaPreviewPlayer();
  ~MediaPreviewPlayer();

  MediaPreviewPlayer(const MediaPreviewPlayer&) = delete;
  MediaPreviewPlayer& operator=(const MediaPreviewPlayer&) = delete;

  bool IsAvailable() const;
  void PlayPcmF32(std::vector<float> samples, std::int32_t sample_rate);
  void Stop();
  bool IsPlaying() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace chat
