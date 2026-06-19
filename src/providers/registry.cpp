#include "cortext/providers/registry.hpp"

#include "cortext/providers/ollama_provider.hpp"
#include "cortext/providers/openai_provider.hpp"

#include <cstdlib>
#include <map>
#include <mutex>

namespace cortext::providers
{

namespace
{

std::mutex &
RegistryMutex ()
{
  static std::mutex mutex;
  return mutex;
}

std::map<std::string, ProviderFactory> &
Registry ()
{
  static std::map<std::string, ProviderFactory> registry;
  return registry;
}

} // namespace

std::optional<ProviderUri>
ParseProviderUri (const std::string &uri)
{
  const auto scheme_end = uri.find ("://");
  if (scheme_end == std::string::npos || scheme_end == 0)
    {
      return std::nullopt;
    }

  ProviderUri parsed;
  parsed.raw = uri;
  parsed.scheme = uri.substr (0, scheme_end);
  const std::string rest = uri.substr (scheme_end + 3);
  if (rest.empty ())
    {
      return std::nullopt;
    }

  // Remote schemes carry host[:port]/model; local schemes carry a path.
  const bool remote = parsed.scheme == "ollama" || parsed.scheme == "openai";
  if (remote)
    {
      const auto slash = rest.find ('/');
      if (slash == std::string::npos || slash + 1 >= rest.size ())
        {
          return std::nullopt;
        }
      parsed.authority = rest.substr (0, slash);
      parsed.path = rest.substr (slash + 1);
    }
  else
    {
      parsed.path = rest;
    }
  return parsed;
}

void
RegisterProviderFactory (const std::string &scheme, ProviderFactory factory)
{
  std::lock_guard<std::mutex> lock (RegistryMutex ());
  Registry ()[scheme] = std::move (factory);
}

namespace
{

// Built-in schemes are registered on first resolve. Explicit registration
// (rather than per-TU static initializers) survives static-library linking,
// where initializers in unreferenced translation units are dropped.
void
EnsureBuiltinProvidersRegistered ()
{
  static std::once_flag once;
  std::call_once (once, [] {
    RegisterProviderFactory ("ollama", &OllamaProvider::Create);
    RegisterProviderFactory ("openai", &OpenAIProvider::Create);
  });
}

} // namespace

std::unique_ptr<InferenceProvider>
ResolveProvider (const std::string &uri, Role role, std::string *error_out)
{
  EnsureBuiltinProvidersRegistered ();
  auto parsed = ParseProviderUri (uri);
  if (!parsed)
    {
      if (error_out != nullptr)
        {
          *error_out = "malformed provider uri: " + uri;
        }
      return nullptr;
    }

  ProviderFactory factory;
  {
    std::lock_guard<std::mutex> lock (RegistryMutex ());
    auto it = Registry ().find (parsed->scheme);
    if (it == Registry ().end ())
      {
        if (error_out != nullptr)
          {
            *error_out
                = "no provider registered for scheme: " + parsed->scheme;
          }
        return nullptr;
      }
    factory = it->second;
  }

  auto provider = factory (*parsed, role, error_out);
  if (provider == nullptr)
    {
      return nullptr;
    }
  // The same contract is enforced for every arrival path (here and for
  // directly injected implementations): the consumer states requirements,
  // the implementation declares capabilities, transport is irrelevant.
  if (!VerifyContract (provider->GetCapabilities (), ContractForRole (role),
                       error_out))
    {
      return nullptr;
    }
  return provider;
}

} // namespace cortext::providers
