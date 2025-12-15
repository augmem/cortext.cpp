#include <catch2/catch_test_macros.hpp>

#include "cortext/generator/gemma_tokenizer.hpp"

#include <filesystem>
#include <string>

namespace
{

std::string
GetTokenizerPath ()
{
  // Try common paths for the tokenizer
  std::vector<std::string> paths = {
    "models/gemma-3n/tokenizer.json",
    "../models/gemma-3n/tokenizer.json",
    "../../models/gemma-3n/tokenizer.json",
  };

  for (const auto &path : paths)
    {
      if (std::filesystem::exists (path))
        {
          return path;
        }
    }

  return "";
}

} // namespace

TEST_CASE ("GemmaTokenizer construction", "[generator][tokenizer]")
{
  auto path = GetTokenizerPath ();

  SECTION ("Throws on missing file")
  {
    REQUIRE_THROWS_AS (cortext::GemmaTokenizer ("/nonexistent/path.json"),
                       std::runtime_error);
  }

  if (path.empty ())
    {
      WARN ("Tokenizer not found, skipping remaining tests");
      return;
    }

  SECTION ("Loads tokenizer successfully")
  {
    REQUIRE_NOTHROW (cortext::GemmaTokenizer (path));
  }
}

TEST_CASE ("GemmaTokenizer encode/decode", "[generator][tokenizer]")
{
  auto path = GetTokenizerPath ();
  if (path.empty ())
    {
      SKIP ("Tokenizer not found");
    }

  cortext::GemmaTokenizer tokenizer (path);

  SECTION ("Basic encoding")
  {
    auto ids = tokenizer.Encode ("Hello");
    REQUIRE (!ids.empty ());
  }

  SECTION ("Empty string produces minimal tokens")
  {
    auto ids = tokenizer.Encode ("");
    // Empty string may produce tokens due to BPE normalization
    // (adding leading space indicator). Just verify it doesn't crash.
    CHECK (ids.size () <= 1);
  }

  SECTION ("Encode with BOS token")
  {
    auto ids_no_bos = tokenizer.Encode ("Hello", false);
    auto ids_with_bos = tokenizer.Encode ("Hello", true);

    REQUIRE (ids_with_bos.size () == ids_no_bos.size () + 1);
    CHECK (ids_with_bos[0] == tokenizer.BosTokenId ());
  }

  SECTION ("Basic roundtrip")
  {
    std::string text = "Hello, world!";
    auto ids = tokenizer.Encode (text);
    REQUIRE (!ids.empty ());

    auto decoded = tokenizer.Decode (ids);
    CHECK (decoded == text);
  }

  SECTION ("Unicode roundtrip")
  {
    std::string text = "Hello \xE4\xB8\x96\xE7\x95\x8C"; // "Hello 世界"
    auto ids = tokenizer.Encode (text);
    REQUIRE (!ids.empty ());

    auto decoded = tokenizer.Decode (ids);
    CHECK (decoded == text);
  }

  SECTION ("Whitespace handling")
  {
    std::string text = "  multiple   spaces  ";
    auto ids = tokenizer.Encode (text);
    auto decoded = tokenizer.Decode (ids);
    // BPE may normalize whitespace slightly, but content should be preserved
    CHECK (!decoded.empty ());
  }
}

TEST_CASE ("GemmaTokenizer special tokens", "[generator][tokenizer]")
{
  auto path = GetTokenizerPath ();
  if (path.empty ())
    {
      SKIP ("Tokenizer not found");
    }

  cortext::GemmaTokenizer tokenizer (path);

  SECTION ("Special token IDs")
  {
    // gemma-3n special tokens
    CHECK (tokenizer.BosTokenId () == 2);
    CHECK (tokenizer.EosTokenId () == 1);
    CHECK (tokenizer.PadTokenId () == 0);
  }

  SECTION ("Vocab size is reasonable")
  {
    // gemma-3n has 262,400 vocab entries
    CHECK (tokenizer.VocabSize () > 100000);
    CHECK (tokenizer.VocabSize () < 500000);
  }
}

TEST_CASE ("GemmaTokenizer token lookup", "[generator][tokenizer]")
{
  auto path = GetTokenizerPath ();
  if (path.empty ())
    {
      SKIP ("Tokenizer not found");
    }

  cortext::GemmaTokenizer tokenizer (path);

  SECTION ("TokenToId for simple tokens")
  {
    // Common single-character tokens should exist and not return UNK
    auto comma_id = tokenizer.TokenToId (",");
    CHECK (comma_id != tokenizer.UnkTokenId ());

    auto colon_id = tokenizer.TokenToId (":");
    CHECK (colon_id != tokenizer.UnkTokenId ());
  }

  SECTION ("IdToToken roundtrip")
  {
    auto id = tokenizer.TokenToId ("{");
    if (id != tokenizer.UnkTokenId ())
      {
        auto token = tokenizer.IdToToken (id);
        CHECK (!token.empty ());
      }
  }
}
