#pragma once

#include "cortext/providers/provider.hpp"
#include "cortext/providers/registry.hpp"

#include <memory>
#include <string>

namespace cortext::providers
{

/// @brief InferenceProvider over an Ollama server's native /api/chat.
///
/// Text parts join the message content; image bytes and audio (encoded to
/// 16 kHz mono WAV) ride the `images` field — the same convention the Julie
/// judge harness uses for Gemma 4 multimodal payloads. JSON-schema
/// constraints are enforced server-side via `format` (ServerSchema).
class OllamaProvider : public InferenceProvider
{
public:
  /// @param authority host[:port], e.g. "127.0.0.1:11434"
  /// @param model     model name, e.g. "gemma4:e4b"
  OllamaProvider (std::string authority, std::string model);
  ~OllamaProvider () override;

  bool Health () const override;
  const Capabilities &GetCapabilities () const override;
  const ProviderIdentity &Identity () const override;
  GenerateResponse Generate (const GenerateRequest &request) override;

  /// @brief Registry factory for the "ollama" scheme.
  static std::unique_ptr<InferenceProvider>
  Create (const ProviderUri &uri, Role role, std::string *error_out);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext::providers
