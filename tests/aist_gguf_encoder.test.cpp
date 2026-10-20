#include <catch2/catch_test_macros.hpp>

#include <cortext/models/aist_gguf_encoder.hpp>

#include "../src/models/aist_semantic_cache_internal.hpp"

#include <cmath>
#include <string>
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

TEST_CASE ("AIST semantic cache evicts least recent entries by bytes",
           "[aist][gguf][unit]")
{
  using Cache = cortext::internal::AistSemanticCache;
  const std::vector<float> v (8, 1.0F);
  const std::size_t entry = Cache::EntryBytes ("a", v);
  Cache cache (3 * entry);
  cache.Put ("a", v);
  cache.Put ("b", v);
  cache.Put ("c", v);
  REQUIRE (cache.Size () == 3);
  REQUIRE (cache.Bytes () == 3 * entry);

  // Touching "a" makes "b" the least recent.
  REQUIRE (cache.Find ("a") != nullptr);
  cache.Put ("d", v);
  CHECK (cache.Size () == 3);
  CHECK (cache.Bytes () <= cache.CapacityBytes ());
  CHECK (cache.Find ("b") == nullptr);
  CHECK (cache.Find ("a") != nullptr);
  CHECK (cache.Find ("c") != nullptr);
  CHECK (cache.Find ("d") != nullptr);

  // Re-putting a key replaces it without double counting.
  cache.Put ("d", std::vector<float> (8, 2.0F));
  CHECK (cache.Size () == 3);
  CHECK (cache.Bytes () == 3 * entry);
  CHECK ((*cache.Find ("d"))[0] == 2.0F);

  // An entry larger than the budget is not kept and evicts nothing.
  cache.Put ("huge", std::vector<float> (4096, 0.0F));
  CHECK (cache.Find ("huge") == nullptr);
  CHECK (cache.Size () == 3);
}
