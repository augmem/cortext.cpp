#include "cortext/providers/openai_provider.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <netdb.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

namespace cortext::providers
{

namespace
{

std::string
HttpJson (const std::string &method, const std::string &host,
          const std::string &port, const std::string &path,
          const std::string &body, int timeout_ms)
{
  struct addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo *res = nullptr;
  if (getaddrinfo (host.c_str (), port.c_str (), &hints, &res) != 0)
    {
      throw std::runtime_error ("openai-compatible: cannot resolve " + host);
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
      throw std::runtime_error ("openai-compatible: cannot connect to "
                                + host + ":" + port);
    }

  struct timeval tv{};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  setsockopt (fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  setsockopt (fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);

  std::string request = method + " " + path + " HTTP/1.1\r\nHost: " + host
                        + "\r\nConnection: close\r\n";
  if (!body.empty ())
    {
      request += "Content-Type: application/json\r\nContent-Length: "
                 + std::to_string (body.size ()) + "\r\n";
    }
  request += "\r\n";
  request += body;

  size_t sent = 0;
  while (sent < request.size ())
    {
      const ssize_t n
          = send (fd, request.data () + sent, request.size () - sent, 0);
      if (n <= 0)
        {
          close (fd);
          throw std::runtime_error ("openai-compatible: send failed");
        }
      sent += static_cast<size_t> (n);
    }

  std::string raw;
  char buffer[16384];
  for (;;)
    {
      const ssize_t n = recv (fd, buffer, sizeof buffer, 0);
      if (n < 0)
        {
          close (fd);
          throw std::runtime_error (
              "openai-compatible: recv failed or timed out");
        }
      if (n == 0)
        {
          break;
        }
      raw.append (buffer, static_cast<size_t> (n));
    }
  close (fd);

  const auto header_end = raw.find ("\r\n\r\n");
  if (header_end == std::string::npos)
    {
      throw std::runtime_error ("openai-compatible: malformed http response");
    }
  const std::string headers = raw.substr (0, header_end);
  std::string payload = raw.substr (header_end + 4);

  const auto status_line_end = headers.find ("\r\n");
  const std::string status_line = headers.substr (0, status_line_end);
  if (status_line.find (" 200") == std::string::npos)
    {
      throw std::runtime_error ("openai-compatible: http error: "
                                + status_line + " "
                                + payload.substr (0, 256));
    }

  std::string lowered = headers;
  std::transform (lowered.begin (), lowered.end (), lowered.begin (),
                  [] (unsigned char c) {
                    return static_cast<char> (std::tolower (c));
                  });
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

int
OpenAIBatchParallelism ()
{
  const char *raw = std::getenv ("CORTEXT_OPENAI_BATCH_PARALLELISM");
  if (raw == nullptr || *raw == '\0')
    {
      return 1;
    }
  try
    {
      return std::clamp (std::stoi (raw), 1, 64);
    }
  catch (const std::exception &)
    {
      return 1;
    }
}

} // namespace

struct OpenAIProvider::Impl
{
  std::string host;
  std::string port;
  std::string base_path = "/v1";
  std::string model;
  Capabilities capabilities;
  ProviderIdentity identity;
};

OpenAIProvider::OpenAIProvider (std::string authority, std::string path)
    : impl_ (std::make_unique<Impl> ())
{
  const auto colon = authority.rfind (':');
  if (colon == std::string::npos)
    {
      impl_->host = authority;
      impl_->port = "8000";
    }
  else
    {
      impl_->host = authority.substr (0, colon);
      impl_->port = authority.substr (colon + 1);
    }

  const auto slash = path.find ('/');
  if (slash == std::string::npos)
    {
      impl_->model = std::move (path);
    }
  else
    {
      impl_->base_path = "/" + path.substr (0, slash);
      impl_->model = path.substr (slash + 1);
    }
  if (impl_->model.empty ())
    {
      throw std::invalid_argument (
          "openai-compatible uri path must include a model name");
    }

  impl_->capabilities.text = true;
  impl_->capabilities.constraints = ConstraintSupport::ServerSchema;
  impl_->capabilities.honors_deadline = true;
  impl_->capabilities.deterministic = true;

  impl_->identity.scheme = "openai";
  impl_->identity.endpoint
      = "http://" + impl_->host + ":" + impl_->port + impl_->base_path;
  impl_->identity.model = impl_->model;
  impl_->identity.detail = "openai-compatible";
}

OpenAIProvider::~OpenAIProvider () = default;

bool
OpenAIProvider::Health () const
{
  try
    {
      (void)HttpJson ("GET", impl_->host, impl_->port,
                      impl_->base_path + "/models", "", 5000);
      return true;
    }
  catch (const std::exception &)
    {
      return false;
    }
}

const Capabilities &
OpenAIProvider::GetCapabilities () const
{
  return impl_->capabilities;
}

const ProviderIdentity &
OpenAIProvider::Identity () const
{
  return impl_->identity;
}

GenerateResponse
OpenAIProvider::Generate (const GenerateRequest &request)
{
  std::string text;
  for (const auto &part : request.parts)
    {
      if (part.kind != ContentPart::Kind::Text)
        {
          throw std::runtime_error (
              "openai-compatible provider currently supports text parts only");
        }
      if (!text.empty ())
        {
          text += "\n\n";
        }
      text += part.text;
    }

  nlohmann::json body{
    { "model", impl_->model },
    { "messages",
      nlohmann::json::array (
          { nlohmann::json{ { "role", "system" },
                            { "content", request.system_prompt } },
            nlohmann::json{ { "role", "user" }, { "content", text } } }) },
    { "temperature", request.params.temperature },
    { "stream", false },
  };
  if (request.params.max_tokens > 0)
    {
      body["max_tokens"] = request.params.max_tokens;
    }
  if (request.params.seed >= 0)
    {
      body["seed"] = request.params.seed;
    }
  if (request.schema.has_value ())
    {
      body["response_format"] = nlohmann::json{
        { "type", "json_schema" },
        { "json_schema",
          nlohmann::json{ { "name", "cortext_extraction" },
                          { "strict", true },
                          { "schema", *request.schema } } },
      };
    }

  std::string last_error;
  const int attempts = std::max (1, request.params.max_retries);
  for (int attempt = 0; attempt < attempts; ++attempt)
    {
      try
        {
          const auto payload = HttpJson (
              "POST", impl_->host, impl_->port,
              impl_->base_path + "/chat/completions",
              body.dump (-1, ' ', false,
                         nlohmann::json::error_handler_t::replace),
              request.params.timeout_ms);
          const auto parsed = nlohmann::json::parse (payload);
          const auto &choice = parsed.at ("choices").at (0);
          GenerateResponse response;
          response.text = choice.at ("message").value ("content", "");
          response.truncated
              = choice.value ("finish_reason", "") == "length";
          response.metadata["model"] = parsed.value ("model", impl_->model);
          if (parsed.contains ("usage"))
            {
              const auto &usage = parsed["usage"];
              response.metadata["prompt_tokens"] = std::to_string (
                  usage.value ("prompt_tokens", 0));
              response.metadata["completion_tokens"] = std::to_string (
                  usage.value ("completion_tokens", 0));
            }
          return response;
        }
      catch (const std::exception &e)
        {
          last_error = e.what ();
        }
    }
  throw std::runtime_error ("openai-compatible generate failed after "
                            + std::to_string (attempts)
                            + " attempt(s): " + last_error);
}

std::vector<GenerateResponse>
OpenAIProvider::GenerateBatch (const std::vector<GenerateRequest> &requests)
{
  const int parallelism = OpenAIBatchParallelism ();
  if (parallelism <= 1 || requests.size () <= 1)
    {
      return InferenceProvider::GenerateBatch (requests);
    }

  std::vector<GenerateResponse> responses (requests.size ());
  std::atomic<std::size_t> next_index { 0 };
  std::exception_ptr first_error;
  std::mutex error_mutex;

  const int worker_count
      = std::min<int> (parallelism, static_cast<int> (requests.size ()));
  std::vector<std::thread> workers;
  workers.reserve (static_cast<std::size_t> (worker_count));
  for (int i = 0; i < worker_count; ++i)
    {
      workers.emplace_back ([&] {
        for (;;)
          {
            const std::size_t index = next_index.fetch_add (1);
            if (index >= requests.size ())
              {
                break;
              }
            try
              {
                responses[index] = Generate (requests[index]);
              }
            catch (...)
              {
                std::lock_guard<std::mutex> lock (error_mutex);
                if (!first_error)
                  {
                    first_error = std::current_exception ();
                  }
              }
          }
      });
    }

  for (auto &worker : workers)
    {
      worker.join ();
    }
  if (first_error)
    {
      std::rethrow_exception (first_error);
    }
  return responses;
}

std::unique_ptr<InferenceProvider>
OpenAIProvider::Create (const ProviderUri &uri, Role role,
                        std::string *error_out)
{
  if (role == Role::Encoder)
    {
      if (error_out != nullptr)
        {
          *error_out = "openai-compatible provider does not serve the encoder "
                       "role yet";
        }
      return nullptr;
    }
  if (uri.authority.empty () || uri.path.empty ())
    {
      if (error_out != nullptr)
        {
          *error_out
              = "openai uri must be openai://host[:port]/v1/model";
        }
      return nullptr;
    }
  try
    {
      return std::make_unique<OpenAIProvider> (uri.authority, uri.path);
    }
  catch (const std::exception &e)
    {
      if (error_out != nullptr)
        {
          *error_out = e.what ();
        }
      return nullptr;
    }
}

} // namespace cortext::providers
