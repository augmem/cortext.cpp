#pragma once

#include "cortext/providers/provider.hpp"
#include "cortext/providers/registry.hpp"

#include <memory>
#include <string>

namespace cortext::providers
{

/// @brief Text-only OpenAI-compatible chat provider for vLLM/llama.cpp style
/// local servers.
///
/// URI shape: openai://host[:port]/v1/model-name. The first path component is
/// used as the API base path; the remaining path is the served model name.
class OpenAIProvider : public InferenceProvider
{
public:
  OpenAIProvider (std::string authority, std::string path);
  ~OpenAIProvider () override;

  bool Health () const override;
  const Capabilities &GetCapabilities () const override;
  const ProviderIdentity &Identity () const override;
  GenerateResponse Generate (const GenerateRequest &request) override;
  std::vector<GenerateResponse>
  GenerateBatch (const std::vector<GenerateRequest> &requests) override;

  static std::unique_ptr<InferenceProvider>
  Create (const ProviderUri &uri, Role role, std::string *error_out);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext::providers
