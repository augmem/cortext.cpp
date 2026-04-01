#pragma once

#include <functional>
#include <memory>
#include <string>

namespace chat {

struct ImGuiAppConfig {
  std::string title = "Cortext Chat";
  int width = 1280;
  int height = 720;
  bool vsync = true;
};

class ImGuiApp {
public:
  explicit ImGuiApp(const ImGuiAppConfig& config = {});
  ~ImGuiApp();

  ImGuiApp(const ImGuiApp&) = delete;
  ImGuiApp& operator=(const ImGuiApp&) = delete;

  // Run the main loop, calling render_fn each frame.
  // Returns when the window is closed.
  void Run(std::function<void()> render_fn);

  // Request the application to exit (can be called from any thread).
  void RequestExit();

  // Check if the window should close.
  bool ShouldClose() const;

  // Get current window dimensions.
  int GetWidth() const;
  int GetHeight() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace chat
