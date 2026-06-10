#pragma once

#include "cortext/providers/provider.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace cortext::providers
{

/// @brief Parsed provider URI.
///
/// Examples:
///   litert://models/gemma4-e2b-litert/gemma-4-E2B-it.litertlm
///   gguf://models/LFM2.5-1.2B-Instruct-GGUF
///   ollama://127.0.0.1:11435/gemma4:e4b
///   openai://api.example.com/v1/some-model
struct ProviderUri
{
  std::string scheme;
  std::string authority; ///< host[:port] for remote schemes, empty otherwise.
  std::string path;      ///< Model path (local) or model name (remote).
  std::string raw;
};

/// @brief Parse a provider URI. Returns std::nullopt on malformed input.
std::optional<ProviderUri> ParseProviderUri (const std::string &uri);

/// @brief Factory signature: build a provider for a parsed URI, or return
/// nullptr with `error_out` set.
using ProviderFactory = std::function<std::unique_ptr<InferenceProvider> (
    const ProviderUri &uri, Role role, std::string *error_out)>;

/// @brief Register a factory for a URI scheme ("ollama", "litert", ...).
/// Mirrors objstore_backend_resolve: registration is process-global and
/// idempotent; the last registration for a scheme wins.
void RegisterProviderFactory (const std::string &scheme,
                              ProviderFactory factory);

/// @brief Resolve a provider for `uri` and `role`.
///
/// Verifies the provider's declared Capabilities against the role
/// (extractors need a constrained-decoding path, encoders need embedding
/// dims) and returns nullptr with `error_out` set when unfit.
std::unique_ptr<InferenceProvider>
ResolveProvider (const std::string &uri, Role role, std::string *error_out);

/// @brief Environment override for a role, if set and non-empty.
///   CORTEXT_SUMMARIZER / CORTEXT_EXTRACTOR / CORTEXT_ENCODER
std::optional<std::string> RoleUriFromEnvironment (Role role);

} // namespace cortext::providers
