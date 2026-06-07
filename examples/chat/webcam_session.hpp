#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace chat {

struct WebcamSessionVideoBridge;

struct WebcamFrame {
  std::vector<std::uint8_t> rgb;
  int width = 0;
  int height = 0;
  int channels = 3;
};

struct WebcamSessionConfig {
  int audio_sample_rate = 16000;
  int audio_chunk_ms = 2000;
  double video_fps = 1.0;

  std::function<void(WebcamFrame&&)> on_video_frame;
  std::function<void(std::vector<float>&&)> on_audio_chunk;
  std::function<void(const std::string&)> on_error;
  std::function<void(bool)> on_capture_changed;
};

class WebcamSession {
public:
  explicit WebcamSession(WebcamSessionConfig config);
  ~WebcamSession();

  WebcamSession(const WebcamSession&) = delete;
  WebcamSession& operator=(const WebcamSession&) = delete;

  bool IsSupported() const;
  bool IsAvailable() const;
  bool Start();
  void Stop();

private:
  struct Impl;
  friend struct WebcamSessionVideoBridge;

  std::unique_ptr<Impl> impl_;
};

}  // namespace chat
