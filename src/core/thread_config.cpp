#include "cortext/core/thread_config.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <thread>

namespace cortext::core
{

namespace
{

uint32_t
GetThreadCountFromEnv (const char *env_var)
{
  const char *val = std::getenv (env_var);
  if (val != nullptr)
    {
      try
        {
          int threads = std::stoi (val);
          if (threads > 0)
            {
              return static_cast<uint32_t> (threads);
            }
        }
      catch (...)
        {
          // Invalid value, fall through to default
        }
    }
  return DefaultThreadCount ();
}

} // namespace

uint32_t
DefaultThreadCount ()
{
  unsigned int cores = std::thread::hardware_concurrency ();
  if (cores == 0)
    {
      cores = 1; // Fallback if detection fails
    }
  return std::max (1u, cores / 3);
}

uint32_t
GetEmbedThreadCount ()
{
  return GetThreadCountFromEnv ("CORTEXT_EMBED_THREADS");
}

uint32_t
GetInferThreadCount ()
{
  return GetThreadCountFromEnv ("CORTEXT_INFER_THREADS");
}

} // namespace cortext::core
