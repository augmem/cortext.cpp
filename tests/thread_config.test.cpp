#include <catch2/catch_test_macros.hpp>

#include "cortext/core/thread_config.hpp"

#include <cstdlib>
#include <thread>

TEST_CASE ("Thread configuration", "[core][thread_config]")
{
  SECTION ("DefaultThreadCount returns sensible value")
  {
    auto threads = cortext::core::DefaultThreadCount ();
    REQUIRE (threads >= 1);

    // Should be at most hardware_concurrency (if detected)
    unsigned int hw = std::thread::hardware_concurrency ();
    if (hw > 0)
      {
        CHECK (threads <= hw);
      }
  }

  SECTION ("GetEmbedThreadCount returns default when env not set")
  {
    // Unset the env var to ensure default behavior
    unsetenv ("CORTEXT_EMBED_THREADS");

    auto threads = cortext::core::GetEmbedThreadCount ();
    REQUIRE (threads >= 1);
    CHECK (threads == cortext::core::DefaultThreadCount ());
  }

  SECTION ("GetInferThreadCount returns default when env not set")
  {
    // Unset the env var to ensure default behavior
    unsetenv ("CORTEXT_INFER_THREADS");

    auto threads = cortext::core::GetInferThreadCount ();
    REQUIRE (threads >= 1);
    CHECK (threads == cortext::core::DefaultThreadCount ());
  }

  SECTION ("GetEmbedThreadCount respects environment variable")
  {
    setenv ("CORTEXT_EMBED_THREADS", "8", 1);

    auto threads = cortext::core::GetEmbedThreadCount ();
    CHECK (threads == 8);

    unsetenv ("CORTEXT_EMBED_THREADS");
  }

  SECTION ("GetInferThreadCount respects environment variable")
  {
    setenv ("CORTEXT_INFER_THREADS", "16", 1);

    auto threads = cortext::core::GetInferThreadCount ();
    CHECK (threads == 16);

    unsetenv ("CORTEXT_INFER_THREADS");
  }

  SECTION ("Invalid env var falls back to default")
  {
    setenv ("CORTEXT_EMBED_THREADS", "invalid", 1);

    auto threads = cortext::core::GetEmbedThreadCount ();
    CHECK (threads == cortext::core::DefaultThreadCount ());

    unsetenv ("CORTEXT_EMBED_THREADS");
  }

  SECTION ("Negative env var falls back to default")
  {
    setenv ("CORTEXT_INFER_THREADS", "-5", 1);

    auto threads = cortext::core::GetInferThreadCount ();
    CHECK (threads == cortext::core::DefaultThreadCount ());

    unsetenv ("CORTEXT_INFER_THREADS");
  }

  SECTION ("Zero env var falls back to default")
  {
    setenv ("CORTEXT_EMBED_THREADS", "0", 1);

    auto threads = cortext::core::GetEmbedThreadCount ();
    CHECK (threads == cortext::core::DefaultThreadCount ());

    unsetenv ("CORTEXT_EMBED_THREADS");
  }
}
