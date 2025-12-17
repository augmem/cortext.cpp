#pragma once

#include <openai/openai.hpp>

#include <atomic>
#include <functional>
#include <optional>
#include <string>

namespace chat {

struct StreamingRequest {
  std::string model;
  openai::Json messages;
  double temperature = 0.7;
  std::atomic<bool>* cancel_flag = nullptr;
};

struct StreamingResult {
  std::string full_content;
  bool was_cancelled = false;
  std::optional<std::string> error;
};

// Callback invoked for each streamed token.
// - token: the new content fragment (may be empty for keep-alive)
// - done: true if this is the final callback (stream finished)
using TokenCallback = std::function<void(const std::string& token, bool done)>;

class StreamingChatClient {
public:
  StreamingChatClient(const std::string& api_key,
                      const std::string& base_url = "https://api.openai.com/v1/");

  // Streams a chat completion, calling token_callback for each received token.
  // Returns when the stream completes, is cancelled, or errors.
  StreamingResult Stream(const StreamingRequest& request,
                         const TokenCallback& token_callback);

private:
  struct CallbackContext {
    const TokenCallback* callback;
    std::atomic<bool>* cancel_flag;
    std::string buffer;  // Accumulates partial SSE data
    std::string full_content;
    bool done = false;
    std::optional<std::string> error;
  };

  static size_t WriteCallback(void* contents, size_t size, size_t nmemb,
                              void* userp);
  static void ProcessSSELine(const std::string& line, CallbackContext& ctx);

  std::string api_key_;
  std::string base_url_;
};

}  // namespace chat
