#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include "absl/status/statusor.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/tokenizer.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "cortext/generator/json_decoder.hpp"

namespace cortext::extractor
{

/// @brief JSON-schema constraint for LiteRT-LM decoding.
class JsonSchemaConstraint : public litert::lm::Constraint
{
public:
  class JsonState : public litert::lm::Constraint::State
  {
  public:
    std::vector<int> token_ids;
    std::string decoded_text;
  };

  JsonSchemaConstraint (nlohmann::json schema,
                        litert::lm::Tokenizer &tokenizer,
                        int vocabulary_size);

  std::unique_ptr<State> Start () const override;
  bool IsEnded (const State &state) const override;
  int GetVocabularySize () const override;

  absl::StatusOr<std::unique_ptr<State>>
  ComputeNext (const State &state, int token) const override;

  absl::StatusOr<std::unique_ptr<litert::lm::Bitmap>>
  ComputeBitmap (const State &state) const override;

private:
  struct TokenVocabulary
  {
    explicit TokenVocabulary (litert::lm::Tokenizer &tokenizer);

    std::vector<int> TokenIdsForText (const std::string &text) const;
    std::vector<int> UnionTokenIds (
        const std::vector<std::string> &texts) const;

    litert::lm::Tokenizer &tokenizer;
    std::unordered_map<std::string, std::vector<int>> json_tokens;
  };

  struct AllowedTokens
  {
    bool allow_all = false;
    std::unordered_set<int> ids;
  };

  AllowedTokens BuildAllowedTokens (const StreamingJSONParser &parser) const;
  std::optional<StreamingJSONParser>
  BuildParser (const std::string &buffer, ParserStatus &status) const;

  nlohmann::json schema_;
  litert::lm::Tokenizer &tokenizer_;
  int vocabulary_size_;
  TokenVocabulary vocab_;
};

} // namespace cortext::extractor
