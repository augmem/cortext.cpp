#include <catch2/catch_test_macros.hpp>

#include "test_helpers.hpp"
#include "cortext/core/thread_config.hpp"

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
    cortext::testing::ScopedEnvVar embed_threads ("CORTEXT_EMBED_THREADS");

    auto threads = cortext::core::GetEmbedThreadCount ();
    REQUIRE (threads >= 1);
    CHECK (threads == cortext::core::DefaultThreadCount ());
  }

  SECTION ("GetInferThreadCount returns default when env not set")
  {
    cortext::testing::ScopedEnvVar infer_threads ("CORTEXT_INFER_THREADS");

    auto threads = cortext::core::GetInferThreadCount ();
    REQUIRE (threads >= 1);
    CHECK (threads == cortext::core::DefaultThreadCount ());
  }

  SECTION ("GetEmbedThreadCount respects environment variable")
  {
    cortext::testing::ScopedEnvVar embed_threads ("CORTEXT_EMBED_THREADS",
                                                  "8");

    auto threads = cortext::core::GetEmbedThreadCount ();
    CHECK (threads == 8);
  }

  SECTION ("GetInferThreadCount respects environment variable")
  {
    cortext::testing::ScopedEnvVar infer_threads ("CORTEXT_INFER_THREADS",
                                                  "16");

    auto threads = cortext::core::GetInferThreadCount ();
    CHECK (threads == 16);
  }

  SECTION ("Invalid env var falls back to default")
  {
    cortext::testing::ScopedEnvVar embed_threads ("CORTEXT_EMBED_THREADS",
                                                  "invalid");

    auto threads = cortext::core::GetEmbedThreadCount ();
    CHECK (threads == cortext::core::DefaultThreadCount ());
  }

  SECTION ("Negative env var falls back to default")
  {
    cortext::testing::ScopedEnvVar infer_threads ("CORTEXT_INFER_THREADS",
                                                  "-5");

    auto threads = cortext::core::GetInferThreadCount ();
    CHECK (threads == cortext::core::DefaultThreadCount ());
  }

  SECTION ("Zero env var falls back to default")
  {
    cortext::testing::ScopedEnvVar embed_threads ("CORTEXT_EMBED_THREADS",
                                                  "0");

    auto threads = cortext::core::GetEmbedThreadCount ();
    CHECK (threads == cortext::core::DefaultThreadCount ());
  }
}
