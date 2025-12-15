#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cortext
{

/// @brief BPE tokenizer for gemma-3n models.
///
/// Parses HuggingFace tokenizer.json format and implements byte-pair encoding
/// for text tokenization. Handles the SentencePiece-style normalization where
/// spaces are replaced with ▁ (U+2581).
///
/// Special tokens:
/// - PAD = 0, EOS = 1, BOS = 2, UNK = 3
///
/// Usage:
/// @code
/// GemmaTokenizer tok("models/gemma-3n/tokenizer.json");
/// auto ids = tok.Encode("Hello, world!");
/// auto text = tok.Decode(ids);
/// @endcode
class GemmaTokenizer
{
public:
  /// @brief Construct tokenizer from HuggingFace tokenizer.json file.
  /// @param tokenizer_json_path Path to tokenizer.json file.
  /// @throws std::runtime_error if file cannot be read or parsed.
  explicit GemmaTokenizer (const std::string &tokenizer_json_path);

  ~GemmaTokenizer ();

  GemmaTokenizer (const GemmaTokenizer &) = delete;
  GemmaTokenizer &operator= (const GemmaTokenizer &) = delete;
  GemmaTokenizer (GemmaTokenizer &&) noexcept;
  GemmaTokenizer &operator= (GemmaTokenizer &&) noexcept;

  /// @brief Encode text to token IDs using BPE.
  /// @param text Input text to tokenize.
  /// @param add_bos If true, prepend BOS token. Default false.
  /// @return Vector of token IDs.
  ///
  /// Text is normalized (spaces → ▁) before BPE encoding.
  /// Does NOT add BOS/EOS tokens by default - caller should add if needed.
  std::vector<int64_t> Encode (const std::string &text,
                               bool add_bos = false) const;

  /// @brief Decode token IDs back to text.
  /// @param ids Token IDs to decode.
  /// @return Decoded text with ▁ replaced by spaces.
  ///
  /// Special tokens (BOS, EOS, PAD) are omitted from output.
  std::string Decode (const std::vector<int64_t> &ids) const;

  /// @brief Get vocabulary size.
  std::size_t VocabSize () const;

  /// @brief Get BOS (beginning of sequence) token ID.
  int64_t BosTokenId () const;

  /// @brief Get EOS (end of sequence) token ID.
  int64_t EosTokenId () const;

  /// @brief Get PAD (padding) token ID.
  int64_t PadTokenId () const;

  /// @brief Get UNK (unknown) token ID.
  int64_t UnkTokenId () const;

  /// @brief Get single token ID for a string, or UNK if not found.
  int64_t TokenToId (const std::string &token) const;

  /// @brief Get string for a token ID, or empty string if invalid.
  std::string IdToToken (int64_t id) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
