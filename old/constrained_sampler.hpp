#pragma once

#include "cortext/generator/gemma_tokenizer.hpp"
#include "cortext/generator/json_decoder.hpp"

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace cortext
{

/// @brief Constrains token sampling based on JSON Schema and parser state.
///
/// Complete port of json_decoder.py SchemaConstrainedSampler including:
/// - Token vocabulary building for JSON structural elements
/// - Logit masking based on parser state
/// - Enum/field name prefix matching
/// - Boost for terminator tokens
/// - oneOf discriminator handling
///
/// Works with any schema by consulting the StreamingJSONParser for
/// allowed continuations and mapping them to token IDs.
class SchemaConstrainedSampler
{
public:
  /// @brief Initialize sampler with tokenizer and parser.
  /// @param tokenizer Tokenizer for token ID mapping.
  /// @param parser Parser tracking JSON state.
  SchemaConstrainedSampler (const GemmaTokenizer &tokenizer,
                            StreamingJSONParser &parser);

  ~SchemaConstrainedSampler ();

  SchemaConstrainedSampler (const SchemaConstrainedSampler &) = delete;
  SchemaConstrainedSampler &operator= (const SchemaConstrainedSampler &)
      = delete;
  SchemaConstrainedSampler (SchemaConstrainedSampler &&) noexcept;
  SchemaConstrainedSampler &operator= (SchemaConstrainedSampler &&) noexcept;

  /// @brief Mask logits to only schema-valid tokens.
  /// @param logits Raw logits from model [vocab_size].
  /// @return Constrained logits with invalid tokens set to -inf.
  ///
  /// Also applies boosts to terminator tokens and preferred fields.
  std::vector<float> ConstrainLogits (const std::vector<float> &logits) const;

  /// @brief Get token IDs for a text string.
  /// @param text Text to tokenize.
  /// @return Vector of token IDs (excluding BOS).
  std::vector<int64_t> GetTokenIds (const std::string &text) const;

  /// @brief Get vocabulary size.
  std::size_t VocabSize () const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
