#include "streaming_client.hpp"

#include <curl/curl.h>

#include <cstring>
#include <sstream>

namespace chat {

namespace {

std::string TruncateForError(const std::string& value, size_t max_len) {
  if (value.size() <= max_len) {
    return value;
  }
  return value.substr(0, max_len) + "...";
}

}  // namespace

StreamingChatClient::StreamingChatClient(const std::string& api_key,
                                         const std::string& base_url)
    : api_key_(api_key), base_url_(base_url) {
  // Ensure base_url ends with /
  if (!base_url_.empty() && base_url_.back() != '/') {
    base_url_.push_back('/');
  }
}

size_t StreamingChatClient::WriteCallback(void* contents, size_t size,
                                          size_t nmemb, void* userp) {
  const size_t real_size = size * nmemb;
  auto* ctx = static_cast<CallbackContext*>(userp);

  // Check for cancellation
  if (ctx->cancel_flag && ctx->cancel_flag->load()) {
    return 0;  // Returning 0 aborts the transfer
  }

  // Append received data to buffer
  ctx->raw_body.append(static_cast<char*>(contents), real_size);
  ctx->buffer.append(static_cast<char*>(contents), real_size);

  // Process complete lines (SSE format: lines ending with \n)
  size_t pos;
  while ((pos = ctx->buffer.find('\n')) != std::string::npos) {
    std::string line = ctx->buffer.substr(0, pos);
    ctx->buffer.erase(0, pos + 1);

    // Remove trailing \r if present (CRLF line endings)
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    // Process the SSE line
    ProcessSSELine(line, *ctx);

    // Check for cancellation after processing
    if (ctx->cancel_flag && ctx->cancel_flag->load()) {
      return 0;
    }
  }

  return real_size;
}

void StreamingChatClient::ProcessSSELine(const std::string& line,
                                         CallbackContext& ctx) {
  // SSE format:
  // - Empty lines are event separators (ignore)
  // - Lines starting with "data: " contain JSON payload
  // - "data: [DONE]" signals end of stream

  if (line.empty()) {
    return;  // Event separator
  }

  // Only process data lines
  const std::string data_prefix = "data: ";
  if (line.compare(0, data_prefix.size(), data_prefix) != 0) {
    return;  // Not a data line (could be event:, id:, retry:, etc.)
  }

  std::string data = line.substr(data_prefix.size());

  // Check for end of stream
  if (data == "[DONE]") {
    if (!ctx.done && ctx.callback) {
      (*ctx.callback)("", true);
    }
    ctx.done = true;
    return;
  }

  // Parse JSON
  try {
    auto json = openai::Json::parse(data);
    if (auto usage = ParseStreamingUsageJson(json)) {
      ctx.usage = *usage;
    }

    const std::string content = ExtractStreamingDeltaText(json);
    if (!content.empty()) {
      ctx.full_content += content;
      if (ctx.callback) {
        (*ctx.callback)(content, false);
      }
    }

    if (HasStreamingFinishReason(json)) {
      if (!ctx.done && ctx.callback) {
        (*ctx.callback)("", true);
      }
      ctx.done = true;
    }

    // Check for errors in response
    if (json.contains("error")) {
      ctx.error = json["error"].dump();
    }
  } catch (const std::exception& e) {
    // JSON parse error - log but continue
    // Could be a partial message or non-JSON data
  }
}

StreamingResult StreamingChatClient::Stream(const StreamingRequest& request,
                                            const TokenCallback& token_callback) {
  StreamingResult result;
  CallbackContext ctx;
  ctx.callback = &token_callback;
  ctx.cancel_flag = request.cancel_flag;

  CURL* curl = curl_easy_init();
  if (!curl) {
    result.error = "Failed to initialize CURL";
    return result;
  }

  // Build request JSON
  openai::Json req_json;
  req_json["model"] = request.model;
  req_json["messages"] = request.messages;
  req_json["temperature"] = request.temperature;
  req_json["stream"] = true;
  req_json["stream_options"] = {{"include_usage", true}};

  std::string req_body = req_json.dump();
  std::string url = base_url_ + "chat/completions";

  // Set up headers
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");

  std::string auth_header = "Authorization: Bearer " + api_key_;
  headers = curl_slist_append(headers, auth_header.c_str());

  // Accept SSE
  headers = curl_slist_append(headers, "Accept: text/event-stream");

  // Configure CURL
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, req_body.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);

  // Disable buffering for real-time streaming
  curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 1024L);

  // Set reasonable timeouts
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
  // No operation timeout - streaming can take a while

  // Perform the request
  CURLcode res = curl_easy_perform(curl);

  // Check result
  if (res != CURLE_OK) {
    if (request.cancel_flag && request.cancel_flag->load()) {
      result.was_cancelled = true;
    } else if (res == CURLE_WRITE_ERROR && request.cancel_flag &&
               request.cancel_flag->load()) {
      // Write callback returned 0 to signal cancellation
      result.was_cancelled = true;
    } else {
      result.error = curl_easy_strerror(res);
    }
  }

  // Check HTTP status code
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  if (http_code >= 400 && !result.error.has_value()) {
    result.error = "HTTP error: " + std::to_string(http_code);
    if (ctx.error.has_value()) {
      result.error = *result.error + " - " + *ctx.error;
    } else if (!ctx.raw_body.empty()) {
      result.error = *result.error + " - " + TruncateForError(ctx.raw_body, 2000);
    }
  }

  // Transfer context error if present
  if (ctx.error.has_value() && !result.error.has_value()) {
    result.error = ctx.error;
  }

  // Set full content
  result.full_content = std::move(ctx.full_content);
  result.usage = ctx.usage;

  // Cleanup
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  return result;
}

}  // namespace chat
