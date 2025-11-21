#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>

using namespace cortext::core;

TEST_CASE ("NCtx and TauM ranges", "[core][knobs]")
{
  REQUIRE (NCtx (0.0) == Catch::Approx (32.0));
  REQUIRE (NCtx (1.0) == Catch::Approx (256.0));

  REQUIRE (TauM (0.0) == Catch::Approx (10.0));
  REQUIRE (TauM (1.0) == Catch::Approx (200.0));
}

TEST_CASE ("AlphaU decreases with T", "[core][knobs]")
{
  double a0 = AlphaU (0.0);
  double a1 = AlphaU (1.0);
  REQUIRE (a0 > a1);
  REQUIRE (a0 <= Catch::Approx (0.70));
  REQUIRE (a0 >= Catch::Approx (0.10));
}

TEST_CASE ("AlphaF monotonicity and bounds", "[core][knobs]")
{
  double low = AlphaF (0.0, 0.0);
  double highF = AlphaF (1.0, 0.0);
  double highU = AlphaF (0.5, 1.0);

  REQUIRE (highF >= low);
  REQUIRE (highU >= low);

  REQUIRE (low >= Catch::Approx (0.05));
  REQUIRE (highU <= Catch::Approx (0.50));
}
