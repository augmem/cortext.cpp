#include <catch2/catch_test_macros.hpp>

#include <cortext/models/aist_gguf_encoder.hpp>

#include <cmath>
#include <vector>

TEST_CASE ("AIST Matryoshka truncation normalizes output",
           "[aist][gguf][unit]")
{
  std::vector<float> values{ 3.0F, 4.0F, 0.0F, 10.0F };
  auto truncated = cortext::TruncateAistMatryoshka (values, 2);
  REQUIRE (truncated.size () == 2);
  const double norm = std::sqrt (static_cast<double> (truncated[0])
                                     * truncated[0]
                                 + static_cast<double> (truncated[1])
                                       * truncated[1]);
  CHECK (std::abs (norm - 1.0) < 1.0e-5);
}
