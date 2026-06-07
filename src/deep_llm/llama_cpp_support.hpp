#pragma once

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#if defined(CORTEXT_ENABLE_LLAMA_CPP)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wc99-extensions"
#endif
#include CORTEXT_GGML_BACKEND_HEADER_PATH
#include CORTEXT_LLAMA_HEADER_PATH
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

namespace cortext::internal
{

#if !defined(CORTEXT_ENABLE_LLAMA_CPP)
using llama_sampler = void;
struct llama_vocab;
using llama_token = int32_t;
#endif

#if defined(CORTEXT_ENABLE_LLAMA_CPP)

inline void
LoadGgmlBackendsOnce ()
{
  static std::once_flag backend_once;
  std::call_once (backend_once, [] () { ggml_backend_load_all (); });
}

struct LlamaCppLogState
{
  ggml_log_level threshold = GGML_LOG_LEVEL_ERROR;
  std::atomic<bool> emit_continuation{ false };
};

inline ggml_log_level
ResolveLlamaCppLogLevel ()
{
  const char *value = std::getenv ("CORTEXT_LLAMA_CPP_LOG_LEVEL");
  if (value == nullptr || *value == '\0')
    {
      return GGML_LOG_LEVEL_ERROR;
    }

  std::string lower (value);
  for (char &ch : lower)
    {
      ch = static_cast<char> (std::tolower (static_cast<unsigned char> (ch)));
    }

  if (lower == "none" || lower == "silent" || lower == "quiet")
    {
      return GGML_LOG_LEVEL_NONE;
    }
  if (lower == "error")
    {
      return GGML_LOG_LEVEL_ERROR;
    }
  if (lower == "warn" || lower == "warning")
    {
      return GGML_LOG_LEVEL_WARN;
    }
  if (lower == "info")
    {
      return GGML_LOG_LEVEL_INFO;
    }
  if (lower == "debug" || lower == "trace")
    {
      return GGML_LOG_LEVEL_DEBUG;
    }

  return GGML_LOG_LEVEL_ERROR;
}

inline void
CortextLlamaCppLogCallback (ggml_log_level level, const char *text,
                            void *user_data)
{
  auto *state = static_cast<LlamaCppLogState *> (user_data);
  if (state == nullptr || text == nullptr)
    {
      return;
    }

  if (level == GGML_LOG_LEVEL_CONT)
    {
      if (state->emit_continuation.load ())
        {
          std::fputs (text, stderr);
          std::fflush (stderr);
        }
      return;
    }

  const bool emit = state->threshold != GGML_LOG_LEVEL_NONE
                    && level >= state->threshold;
  state->emit_continuation.store (emit);
  if (emit)
    {
      std::fputs (text, stderr);
      std::fflush (stderr);
    }
}

inline void
InstallLlamaCppLogFilter ()
{
  static std::once_flag once;
  static LlamaCppLogState state;
  std::call_once (once, [] () {
    state.threshold = ResolveLlamaCppLogLevel ();
    llama_log_set (CortextLlamaCppLogCallback, &state);
  });
}

#endif

inline constexpr bool kLlamaCppAvailable
#if defined(CORTEXT_ENABLE_LLAMA_CPP)
    = true;
#else
    = false;
#endif

;

} // namespace cortext::internal
