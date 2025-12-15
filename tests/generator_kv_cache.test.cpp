#include <catch2/catch_test_macros.hpp>

#include "cortext/generator/kv_cache.hpp"

#include <map>
#include <string>
#include <vector>

TEST_CASE ("KVCache construction", "[generator][kv_cache]")
{
  cortext::KVSpec spec{ 2, 2, 4, 1 }; // 2 layers, 2 heads, head_dim=4, batch=1

  SECTION ("Creates with specified capacity")
  {
    cortext::KVCache cache (spec, 8);
    CHECK (cache.Length () == 0);
  }

  SECTION ("Initial past inputs are empty")
  {
    cortext::KVCache cache (spec, 8);
    auto past = cache.PastInputs ();

    // Should have 2 layers * 2 (key + value) = 4 entries
    CHECK (past.size () == 4);

    // All should be empty (zero length in time dimension)
    for (const auto &[name, data] : past)
      {
        CHECK (data.empty ());
      }
  }
}

TEST_CASE ("KVCache init from present", "[generator][kv_cache]")
{
  cortext::KVSpec spec{ 2, 2, 4, 1 }; // 2 layers, 2 heads, head_dim=4, batch=1
  cortext::KVCache cache (spec, 16);

  // Create mock present KV with 3 tokens
  std::map<std::string, std::vector<float> > present;
  std::size_t tokens = 3;
  std::size_t kv_size = 1 * 2 * tokens * 4; // [B, H, T, D]

  for (int layer = 0; layer < 2; ++layer)
    {
      std::string key_name
          = "present_key_values." + std::to_string (layer) + ".key";
      std::string value_name
          = "present_key_values." + std::to_string (layer) + ".value";

      present[key_name] = std::vector<float> (kv_size, 1.0f * layer);
      present[value_name] = std::vector<float> (kv_size, 2.0f * layer);
    }

  std::vector<std::size_t> shape = { 1, 2, tokens, 4 };

  SECTION ("Initializes length correctly")
  {
    cache.InitFromPresent (present, shape);
    CHECK (cache.Length () == tokens);
  }

  SECTION ("Past inputs reflect initialized data")
  {
    cache.InitFromPresent (present, shape);
    auto past = cache.PastInputs ();

    CHECK (past.size () == 4);

    // Each should have data now
    for (const auto &[name, data] : past)
      {
        CHECK (data.size () == kv_size);
      }
  }
}

TEST_CASE ("KVCache append from present", "[generator][kv_cache]")
{
  cortext::KVSpec spec{ 2, 2, 4, 1 };
  cortext::KVCache cache (spec, 16);

  // Initialize with 2 tokens
  std::map<std::string, std::vector<float> > init_present;
  std::size_t init_tokens = 2;
  std::size_t init_kv_size = 1 * 2 * init_tokens * 4;

  for (int layer = 0; layer < 2; ++layer)
    {
      std::string key_name
          = "present_key_values." + std::to_string (layer) + ".key";
      std::string value_name
          = "present_key_values." + std::to_string (layer) + ".value";

      init_present[key_name] = std::vector<float> (init_kv_size, 1.0f);
      init_present[value_name] = std::vector<float> (init_kv_size, 2.0f);
    }

  std::vector<std::size_t> init_shape = { 1, 2, init_tokens, 4 };
  cache.InitFromPresent (init_present, init_shape);

  REQUIRE (cache.Length () == 2);

  SECTION ("Appending single token increases length")
  {
    // Create present with 1 new token (total 3, but we only append the last)
    std::map<std::string, std::vector<float> > append_present;
    std::size_t total_tokens = 3;
    std::size_t append_kv_size = 1 * 2 * total_tokens * 4;

    for (int layer = 0; layer < 2; ++layer)
      {
        std::string key_name
            = "present_key_values." + std::to_string (layer) + ".key";
        std::string value_name
            = "present_key_values." + std::to_string (layer) + ".value";

        append_present[key_name] = std::vector<float> (append_kv_size, 3.0f);
        append_present[value_name] = std::vector<float> (append_kv_size, 4.0f);
      }

    cache.AppendFromPresent (append_present);
    CHECK (cache.Length () == 3);
  }
}

TEST_CASE ("KVCache capacity growth", "[generator][kv_cache]")
{
  cortext::KVSpec spec{ 1, 1, 2, 1 }; // Minimal config
  cortext::KVCache cache (spec, 2);   // Start with small capacity

  SECTION ("EnsureCapacity grows when needed")
  {
    cache.EnsureCapacity (10);
    // Should not throw, capacity should be sufficient now
    CHECK (cache.Length () == 0);
  }

  SECTION ("EnsureCapacity with existing data preserves length")
  {
    // Initialize with 1 token
    std::map<std::string, std::vector<float> > present;
    present["present_key_values.0.key"] = { 1.0f, 2.0f };
    present["present_key_values.0.value"] = { 3.0f, 4.0f };

    cache.InitFromPresent (present, { 1, 1, 1, 2 });
    REQUIRE (cache.Length () == 1);

    cache.EnsureCapacity (20);
    CHECK (cache.Length () == 1);
  }
}

TEST_CASE ("KVCache clear", "[generator][kv_cache]")
{
  cortext::KVSpec spec{ 2, 2, 4, 1 };
  cortext::KVCache cache (spec, 8);

  // Initialize with some data
  std::map<std::string, std::vector<float> > present;
  std::size_t kv_size = 1 * 2 * 3 * 4;

  for (int layer = 0; layer < 2; ++layer)
    {
      std::string key_name
          = "present_key_values." + std::to_string (layer) + ".key";
      std::string value_name
          = "present_key_values." + std::to_string (layer) + ".value";

      present[key_name] = std::vector<float> (kv_size, 1.0f);
      present[value_name] = std::vector<float> (kv_size, 2.0f);
    }

  cache.InitFromPresent (present, { 1, 2, 3, 4 });
  REQUIRE (cache.Length () == 3);

  SECTION ("Clear clears length")
  {
    cache.Clear ();
    CHECK (cache.Length () == 0);
  }

  SECTION ("Clear clears past inputs")
  {
    cache.Clear ();
    auto past = cache.PastInputs ();

    for (const auto &[name, data] : past)
      {
        CHECK (data.empty ());
      }
  }
}
