#pragma once

#include <cstddef>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace cortext::providers
{

/// @brief Role a provider fulfils inside the deep-LLM pipeline.
enum class Role
{
  Summarizer,
  Extractor,
  Encoder
};

/// @brief How a provider enforces structured (JSON-schema) output.
enum class ConstraintSupport
{
  None,           ///< Free-form generation only; caller must validate/retry.
  NativeGrammar,  ///< In-process grammar/constraint decoding (LiteRT, GBNF).
  ServerSchema    ///< Server-side schema enforcement (e.g. Ollama `format`).
};

/// @brief Static capabilities a provider declares at resolve time.
///
/// The factory verifies fitness here instead of discovering failures
/// mid-run (e.g. an encoder asked to caption an image, or an extractor
/// without any constrained-decoding path).
struct Capabilities
{
  bool text = false;
  bool image = false;
  bool audio = false;
  ConstraintSupport constraints = ConstraintSupport::None;
  /// Embedding output dimensions (empty for generation providers).
  std::vector<std::size_t> embedding_dims;
};

/// @brief Generation parameters shared across transports.
struct GenerateParams
{
  float temperature = 0.0F;
  int max_tokens = 0;       ///< <=0 means provider default.
  int seed = -1;            ///< <0 means provider default.
  int timeout_ms = 120000;
  int max_retries = 1;
};

/// @brief One input part of a request (text or raw media bytes).
struct ContentPart
{
  enum class Kind
  {
    Text,
    ImageBytes,
    AudioPcm16k ///< float32 mono 16 kHz samples in `pcm`.
  };

  Kind kind = Kind::Text;
  std::string text;
  std::vector<unsigned char> bytes; ///< Image payload when ImageBytes.
  std::vector<float> pcm;           ///< Audio samples when AudioPcm16k.
};

/// @brief A single generation request. The one primitive every transport
/// implements; Summarizer/Extractor adapters are built on top of it.
struct GenerateRequest
{
  Role role = Role::Summarizer;
  std::string system_prompt;
  std::vector<ContentPart> parts;
  /// When present the response MUST be JSON valid under this schema.
  std::optional<nlohmann::json> schema;
  GenerateParams params;
};

struct GenerateResponse
{
  std::string text;
  bool truncated = false;
  /// Transport-specific diagnostics (token counts, timings, model digest).
  std::map<std::string, std::string> metadata;
};

/// @brief Identity block recorded into eval artifacts (summary.json) so a
/// judge artifact always states which stack produced it.
struct ProviderIdentity
{
  std::string scheme;     ///< litert | gguf | ollama | openai | ...
  std::string endpoint;   ///< Model path or server URL.
  std::string model;      ///< Model name or file stem.
  std::string detail;     ///< Backend-specific extra (digest, quant, ...).
};

/// @brief Abstract inference transport (in-process engine or remote server).
class InferenceProvider
{
public:
  virtual ~InferenceProvider () = default;

  /// @brief Cheap readiness check (model loaded / server reachable).
  virtual bool Health () const = 0;

  virtual const Capabilities &GetCapabilities () const = 0;

  virtual const ProviderIdentity &Identity () const = 0;

  /// @brief Synchronous generation. Throws std::runtime_error on transport
  /// failure after exhausting GenerateParams::max_retries.
  virtual GenerateResponse Generate (const GenerateRequest &request) = 0;
};

} // namespace cortext::providers
