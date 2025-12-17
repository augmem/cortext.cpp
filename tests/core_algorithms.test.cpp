#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/core/algorithms.hpp>

using namespace cortext::core;

TEST_CASE ("CosineSimilarity basic cases", "[core][algorithms]")
{
  Eigen::VectorXf a = Eigen::VectorXf::Ones (4);
  Eigen::VectorXf b = Eigen::VectorXf::Ones (4);
  REQUIRE (CosineSimilarity (a, b) == Catch::Approx (1.0));

  Eigen::VectorXf c (4);
  c << 1, 0, 0, 0;
  Eigen::VectorXf d (4);
  d << 0, 1, 0, 0;
  REQUIRE (CosineSimilarity (c, d) == Catch::Approx (0.0));

  Eigen::VectorXf e = 2.0f * a;
  REQUIRE (CosineSimilarity (a, e) == Catch::Approx (1.0));
}

TEST_CASE ("EWMA simple sequence", "[core][algorithms]")
{
  double prev = 0.0;
  double alpha = 0.5;
  double x1 = 1.0;
  double y1 = Ewma (prev, x1, alpha); // 0.5
  REQUIRE (y1 == Catch::Approx (0.5));

  double x2 = 0.0;
  double y2 = Ewma (y1, x2, alpha); // 0.25
  REQUIRE (y2 == Catch::Approx (0.25));
}

TEST_CASE ("Clamp and Lerp", "[core][algorithms]")
{
  REQUIRE (Clamp (5, 1, 10) == 5);
  REQUIRE (Clamp (0, 1, 10) == 1);
  REQUIRE (Clamp (20, 1, 10) == 10);

  REQUIRE (Lerp (0.0, 10.0, 0.0) == Catch::Approx (0.0));
  REQUIRE (Lerp (0.0, 10.0, 1.0) == Catch::Approx (10.0));
  REQUIRE (Lerp (0.0, 10.0, 0.3) == Catch::Approx (3.0));
}

TEST_CASE ("Map01 transforms cosine [-1,1] to [0,1]", "[core][algorithms]")
{
  // Reference: algorithms.md Section 2.1.2, lines 220-223
  // map01(z) = clamp((z + 1) / 2, 0, 1)

  // Boundary cases
  REQUIRE (Map01 (-1.0) == Catch::Approx (0.0));
  REQUIRE (Map01 (1.0) == Catch::Approx (1.0));
  REQUIRE (Map01 (0.0) == Catch::Approx (0.5));

  // Mid-points
  REQUIRE (Map01 (-0.5) == Catch::Approx (0.25));
  REQUIRE (Map01 (0.5) == Catch::Approx (0.75));

  // Values outside [-1,1] should be clamped
  REQUIRE (Map01 (-2.0) == Catch::Approx (0.0));
  REQUIRE (Map01 (2.0) == Catch::Approx (1.0));
}
