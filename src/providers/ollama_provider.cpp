#include "cortext/providers/ollama_provider.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <netdb.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace cortext::providers
{

namespace
{

std::string
Base64Encode (const unsigned char *data, size_t size)
{
  static const char *kAlphabet
      = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve (((size + 2) / 3) * 4);
  for (size_t i = 0; i < size; i += 3)
    {
      unsigned int chunk = (unsigned int)data[i] << 16;
      if (i + 1 < size)
        {
          chunk |= (unsigned int)data[i + 1] << 8;
        }
      if (i + 2 < size)
        {
          chunk |= (unsigned int)data[i + 2];
        }
      out.push_back (kAlphabet[(chunk >> 18) & 0x3F]);
      out.push_back (kAlphabet[(chunk >> 12) & 0x3F]);
      out.push_back (i + 1 < size ? kAlphabet[(chunk >> 6) & 0x3F] : '=');
      out.push_back (i + 2 < size ? kAlphabet[chunk & 0x3F] : '=');
    }
  return out;
}

/// Encode float PCM (mono 16 kHz) into a 16-bit little-endian WAV container.
std::vector<unsigned char>
PcmToWav16k (const std::vector<float> &pcm)
{
  constexpr uint32_t kSampleRate = 16000;
  const uint32_t data_bytes = (uint32_t)pcm.size () * 2;
  std::vector<unsigned char> wav;
  wav.reserve (44 + data_bytes);

  auto push_u32 = [&wav] (uint32_t v) {
    wav.push_back ((unsigned char)(v & 0xFF));
    wav.push_back ((unsigned char)((v >> 8) & 0xFF));
    wav.push_back ((unsigned char)((v >> 16) & 0xFF));
    wav.push_back ((unsigned char)((v >> 24) & 0xFF));
  };
  auto push_u16 = [&wav] (uint16_t v) {
    wav.push_back ((unsigned char)(v & 0xFF));
    wav.push_back ((unsigned char)((v >> 8) & 0xFF));
  };
  auto push_tag = [&wav] (const char *tag) {
    wav.insert (wav.end (), tag, tag + 4);
  };

  push_tag ("RIFF");
  push_u32 (36 + data_bytes);
  push_tag ("WAVE");
  push_tag ("fmt ");
  push_u32 (16);
  push_u16 (1); // PCM
  push_u16 (1); // mono
  push_u32 (kSampleRate);
  push_u32 (kSampleRate * 2); // byte rate
  push_u16 (2);               // block align
  push_u16 (16);              // bits per sample
  push_tag ("data");
  push_u32 (data_bytes);
  for (float sample : pcm)
    {
      const float clamped = std::clamp (sample, -1.0F, 1.0F);
      const auto value = (int16_t)(clamped * 32767.0F);
      wav.push_back ((unsigned char)((uint16_t)value & 0xFF));
      wav.push_back ((unsigned char)(((uint16_t)value >> 8) & 0xFF));
    }
  return wav;
}

/// Minimal blocking HTTP/1.1 POST over a plain socket (Ollama is plain
/// HTTP on localhost / LAN). Handles Content-Length and chunked responses.
std::string
HttpPostJson (const std::string &host, const std::string &port,
              const std::string &path, const std::string &body,
              int timeout_ms)
{
  struct addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo *res = nullptr;
  if (getaddrinfo (host.c_str (), port.c_str (), &hints, &res) != 0)
    {
      throw std::runtime_error ("ollama: cannot resolve " + host);
    }

  int fd = -1;
  for (struct addrinfo *ai = res; ai != nullptr; ai = ai->ai_next)
    {
      fd = socket (ai->ai_family, ai->ai_socktype, ai->ai_protocol);
      if (fd < 0)
        {
          continue;
        }
      if (connect (fd, ai->ai_addr, ai->ai_addrlen) == 0)
        {
          break;
        }
      close (fd);
      fd = -1;
    }
  freeaddrinfo (res);
  if (fd < 0)
    {
      throw std::runtime_error ("ollama: cannot connect to " + host + ":"
                                + port);
    }

  struct timeval tv{};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  setsockopt (fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  setsockopt (fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);

  std::string request = "POST " + path + " HTTP/1.1\r\nHost: " + host
                        + "\r\nContent-Type: application/json\r\n"
                          "Connection: close\r\nContent-Length: "
                        + std::to_string (body.size ()) + "\r\n\r\n" + body;
  size_t sent = 0;
  while (sent < request.size ())
    {
      const ssize_t n
          = send (fd, request.data () + sent, request.size () - sent, 0);
      if (n <= 0)
        {
          close (fd);
          throw std::runtime_error ("ollama: send failed");
        }
      sent += (size_t)n;
    }

  std::string raw;
  char buffer[16384];
  for (;;)
    {
      const ssize_t n = recv (fd, buffer, sizeof buffer, 0);
      if (n < 0)
        {
          close (fd);
          throw std::runtime_error ("ollama: recv failed or timed out");
        }
      if (n == 0)
        {
          break;
        }
      raw.append (buffer, (size_t)n);
    }
  close (fd);

  const auto header_end = raw.find ("\r\n\r\n");
  if (header_end == std::string::npos)
    {
      throw std::runtime_error ("ollama: malformed http response");
    }
  const std::string headers = raw.substr (0, header_end);
  std::string payload = raw.substr (header_end + 4);

  const auto status_line_end = headers.find ("\r\n");
  const std::string status_line = headers.substr (0, status_line_end);
  if (status_line.find (" 200") == std::string::npos)
    {
      throw std::runtime_error ("ollama: http error: " + status_line + " "
                                + payload.substr (0, 256));
    }

  std::string lowered = headers;
  std::transform (lowered.begin (), lowered.end (), lowered.begin (),
                  [] (unsigned char c) { return (char)tolower (c); });
  if (lowered.find ("transfer-encoding: chunked") != std::string::npos)
    {
      std::string decoded;
      size_t pos = 0;
      while (pos < payload.size ())
        {
          const auto line_end = payload.find ("\r\n", pos);
          if (line_end == std::string::npos)
            {
              break;
            }
          const size_t chunk_size = std::stoul (
              payload.substr (pos, line_end - pos), nullptr, 16);
          if (chunk_size == 0)
            {
              break;
            }
          decoded.append (payload, line_end + 2, chunk_size);
          pos = line_end + 2 + chunk_size + 2;
        }
      payload = std::move (decoded);
    }
  return payload;
}

} // namespace

struct OllamaProvider::Impl
{
  std::string host;
  std::string port;
  std::string model;
  Capabilities capabilities;
  ProviderIdentity identity;
};

OllamaProvider::OllamaProvider (std::string authority, std::string model)
    : impl_ (std::make_unique<Impl> ())
{
  const auto colon = authority.rfind (':');
  if (colon == std::string::npos)
    {
      impl_->host = authority;
      impl_->port = "11434";
    }
  else
    {
      impl_->host = authority.substr (0, colon);
      impl_->port = authority.substr (colon + 1);
    }
  impl_->model = std::move (model);

  impl_->capabilities.text = true;
  // Multimodal acceptance depends on the served model; declared optimistic
  // here and verified by Health()/server errors. A later phase can consult
  // /api/show the way the judge harness does.
  impl_->capabilities.image = true;
  impl_->capabilities.audio = true;
  impl_->capabilities.constraints = ConstraintSupport::ServerSchema;
  // Socket-level timeouts surface as prompt failures (fail fast), and
  // seeded generation at temperature 0 reproduces on a fixed server build.
  impl_->capabilities.honors_deadline = true;
  impl_->capabilities.deterministic = true;

  impl_->identity.scheme = "ollama";
  impl_->identity.endpoint
      = "http://" + impl_->host + ":" + impl_->port;
  impl_->identity.model = impl_->model;
}

OllamaProvider::~OllamaProvider () = default;

bool
OllamaProvider::Health () const
{
  try
    {
      // /api/show doubles as a reachability + model-presence check.
      const std::string body
          = nlohmann::json{ { "model", impl_->model } }.dump ();
      (void)HttpPostJson (impl_->host, impl_->port, "/api/show", body, 5000);
      return true;
    }
  catch (const std::exception &)
    {
      return false;
    }
}

const Capabilities &
OllamaProvider::GetCapabilities () const
{
  return impl_->capabilities;
}

const ProviderIdentity &
OllamaProvider::Identity () const
{
  return impl_->identity;
}

GenerateResponse
OllamaProvider::Generate (const GenerateRequest &request)
{
  std::string text;
  std::vector<std::string> media;
  for (const auto &part : request.parts)
    {
      switch (part.kind)
        {
        case ContentPart::Kind::Text:
          if (!text.empty ())
            {
              text += "\n\n";
            }
          text += part.text;
          break;
        case ContentPart::Kind::ImageBytes:
          media.push_back (Base64Encode (part.bytes.data (),
                                         part.bytes.size ()));
          break;
        case ContentPart::Kind::AudioPcm16k:
          {
            const auto wav = PcmToWav16k (part.pcm);
            media.push_back (Base64Encode (wav.data (), wav.size ()));
          }
          break;
        }
    }

  nlohmann::json user_message{ { "role", "user" }, { "content", text } };
  if (!media.empty ())
    {
      // One binary multimodal field: Gemma 4 accepts image and WAV payloads
      // through `images` (same convention as the judge harness).
      user_message["images"] = media;
    }

  nlohmann::json body{
    { "model", impl_->model },
    { "messages",
      nlohmann::json::array (
          { nlohmann::json{ { "role", "system" },
                            { "content", request.system_prompt } },
            user_message }) },
    { "stream", false },
    { "think", false },
    { "options",
      nlohmann::json{ { "temperature", request.params.temperature } } },
  };
  if (request.params.max_tokens > 0)
    {
      body["options"]["num_predict"] = request.params.max_tokens;
    }
  if (request.params.seed >= 0)
    {
      body["options"]["seed"] = request.params.seed;
    }
  if (request.schema.has_value ())
    {
      body["format"] = *request.schema;
    }

  std::string last_error;
  const int attempts = std::max (1, request.params.max_retries);
  for (int attempt = 0; attempt < attempts; ++attempt)
    {
      try
        {
          // Corpus-derived text can carry invalid UTF-8 bytes; replace them
          // with U+FFFD instead of throwing mid-run (type_error.316).
          const std::string payload = HttpPostJson (
              impl_->host, impl_->port, "/api/chat",
              body.dump (-1, ' ', false,
                         nlohmann::json::error_handler_t::replace),
              request.params.timeout_ms);
          const auto parsed = nlohmann::json::parse (payload);
          GenerateResponse response;
          response.text
              = parsed.value ("message", nlohmann::json::object ())
                    .value ("content", "");
          response.truncated
              = parsed.value ("done_reason", "") == "length";
          response.metadata["model"] = parsed.value ("model", impl_->model);
          response.metadata["eval_count"]
              = std::to_string (parsed.value ("eval_count", 0));
          response.metadata["prompt_eval_count"]
              = std::to_string (parsed.value ("prompt_eval_count", 0));
          return response;
        }
      catch (const std::exception &e)
        {
          last_error = e.what ();
        }
    }
  throw std::runtime_error ("ollama generate failed after "
                            + std::to_string (attempts)
                            + " attempt(s): " + last_error);
}

std::unique_ptr<InferenceProvider>
OllamaProvider::Create (const ProviderUri &uri, Role role,
                        std::string *error_out)
{
  if (role == Role::Encoder)
    {
      if (error_out != nullptr)
        {
          *error_out = "ollama provider does not serve the encoder role yet";
        }
      return nullptr;
    }
  if (uri.authority.empty () || uri.path.empty ())
    {
      if (error_out != nullptr)
        {
          *error_out = "ollama uri must be ollama://host[:port]/model";
        }
      return nullptr;
    }
  return std::make_unique<OllamaProvider> (uri.authority, uri.path);
}

} // namespace cortext::providers
