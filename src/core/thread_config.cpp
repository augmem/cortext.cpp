#include "cortext/core/thread_config.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <thread>

namespace cortext::core
{

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
  const char *val = std::getenv ("CORTEXT_EMBED_THREADS");
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

uint32_t
GetInferThreadCount ()
{
  const char *val = std::getenv ("CORTEXT_INFER_THREADS");
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

} // namespace cortext::core
