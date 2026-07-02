#include <catch2/catch_test_macros.hpp>

#include <cstdlib>

TEST_CASE ("test main scrubs external CORTEXT environment",
           "[env][hermetic]")
{
  REQUIRE (std::getenv ("CORTEXT_TEST_SENTINEL") == nullptr);
}
