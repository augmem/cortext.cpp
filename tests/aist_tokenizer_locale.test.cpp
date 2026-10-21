// The AIST text tokenizer must classify bytes the same way whatever the
// process locale is: bindings run under different locales (Python sets the
// user's UTF-8 locale, the C++ runner, Go and Dart stay in "C"), and a store
// written by one binding is queried by another.

#include <catch2/catch_test_macros.hpp>

#include <cortext/models/aist_gguf_encoder.hpp>

#include <clocale>
#include <filesystem>
#include <string>
#include <vector>

namespace
{

// The model is resolved from the source tree, whatever directory the tests
// run from (ctest runs them from the build directory).
const std::filesystem::path kSource = CORTEXT_TEST_SOURCE_DIR;

/// @brief Sets LC_ALL for the scope and restores the previous locale.
class ScopedLocale
{
public:
  explicit ScopedLocale (const char *name)
  {
    const char *previous = std::setlocale (LC_ALL, nullptr);
    previous_ = previous != nullptr ? previous : "C";
    active_ = std::setlocale (LC_ALL, name) != nullptr;
  }
  ~ScopedLocale () { std::setlocale (LC_ALL, previous_.c_str ()); }
  ScopedLocale (const ScopedLocale &) = delete;
  ScopedLocale &operator= (const ScopedLocale &) = delete;

  bool
  Active () const
  {
    return active_;
  }

private:
  std::string previous_;
  bool active_ = false;
};

struct Encoded
{
  std::vector<std::vector<int>> ids;
  std::vector<std::vector<float>> embeddings;
};

// A fresh encoder per locale: the semantic cache is per instance, so each
// embedding below is computed under the locale being tested.
Encoded
EncodeAll (const std::vector<std::string> &texts)
{
  const auto model = cortext::ResolveAistGgufModelPath (kSource / "models");
  REQUIRE (model.has_value ());
  cortext::AistGgufConfig config;
  config.model_path = model->string ();
  cortext::AistGgufEncoder encoder (config);
  REQUIRE (encoder.IsRuntimeAvailable ());
  Encoded out;
  for (const std::string &text : texts)
    {
      out.ids.push_back (encoder.DebugTokenizeText (text));
      std::vector<float> embedding;
      encoder.EncodeText (text, embedding);
      out.embeddings.push_back (std::move (embedding));
    }
  return out;
}

} // namespace

TEST_CASE ("AIST tokenizer ids and embeddings do not depend on the locale",
           "[aist][gguf][integration][tokenizer]")
{
  const std::vector<std::string> texts{
    "B\xc3\xba" "scame alojamiento en Sevilla",
    "Cr\xc3\xa8me br\xc3\xbbl\xc3\xa9" "e, gr\xc3\xbc\xc3\x9f" "e aus "
    "M\xc3\xbcnchen.",
    "Ol\xc3\xa1, n\xc3\xa3o \xc3\xa9 f\xc3\xa1" "cil: \xc3\x87" "A "
    "\xc3\x86\xc3\x98\xc3\x85 \xc3\xb1\xc3\x91 \xc3\xbf",
    "Plain ASCII stays the same: tabs\tand\vfeeds\fend.",
  };

  Encoded c_locale;
  {
    ScopedLocale locale ("C");
    REQUIRE (locale.Active ());
    c_locale = EncodeAll (texts);
  }

  // The canonical ids are the "C" locale ones: every byte >= 0x80 is its own
  // [UNK] (100) and splits the word around it.
  CHECK (c_locale.ids[0]
         == std::vector<int>{ 101, 1038, 100, 100, 8040, 14074, 2632, 22918,
                              28883, 4372, 29363, 102 });

  int utf8_locales = 0;
  for (const char *name : { "en_US.UTF-8", "C.UTF-8", "en_US.utf8" })
    {
      ScopedLocale locale (name);
      if (!locale.Active ())
        {
          continue;
        }
      ++utf8_locales;
      INFO ("locale " << name);
      const Encoded utf8 = EncodeAll (texts);
      for (std::size_t i = 0; i < texts.size (); ++i)
        {
          INFO ("text " << i);
          CHECK (utf8.ids[i] == c_locale.ids[i]);
          // bit-identical, not near: the same ids run the same graph
          const bool same_embedding
              = utf8.embeddings[i] == c_locale.embeddings[i];
          CHECK (same_embedding);
        }
    }
  if (utf8_locales == 0)
    {
      SKIP ("no UTF-8 locale is installed");
    }
}
