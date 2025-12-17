#pragma once

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cortext
{

/// @brief Status of JSON parsing.
enum class ParserStatus
{
  kValidPartial, ///< Valid so far, not complete
  kComplete,     ///< Valid and schema satisfied
  kInvalid       ///< Invalid JSON structure
};

/// @brief JSON parser state machine states.
enum class JSONState
{
  kExpectingObjectStart,  ///< Expecting '{'
  kExpectingKeyOrClose,   ///< Expecting '"key"' or '}'
  kExpectingKeyString,    ///< Inside key string
  kExpectingColon,        ///< Expecting ':'
  kExpectingValue,        ///< Expecting value start
  kInStringValue,         ///< Inside string value
  kInNumberValue,         ///< Inside number value
  kInBooleanValue,        ///< Inside true/false
  kInNullValue,           ///< Inside null
  kInArray,               ///< Inside array
  kInNestedObject,        ///< Inside nested object
  kExpectingCommaOrClose, ///< Expecting ',' or '}'
  kInvalid                ///< Invalid state
};

/// @brief oneOf discriminator information.
struct OneOfDiscriminator
{
  std::string prop;
  std::vector<nlohmann::json> values_by_branch;
  std::set<nlohmann::json> value_set;
};

/// @brief Streaming JSON parser with full schema validation.
///
/// Complete port of json_decoder.py StreamingJSONParser including:
/// - State machine for incremental JSON parsing
/// - Schema validation (types, required fields, enums)
/// - oneOf branch handling with discriminator detection
/// - Heuristic branch selection (fewest required, then fewest properties)
/// - Enum/const constraint validation
///
/// Validates JSON token-by-token against a JSON Schema and detects
/// when schema requirements are satisfied to stop generation early.
class StreamingJSONParser
{
public:
  /// @brief Initialize parser with JSON Schema.
  /// @param schema JSON Schema defining expected structure.
  explicit StreamingJSONParser (const nlohmann::json &schema);

  ~StreamingJSONParser ();

  StreamingJSONParser (const StreamingJSONParser &) = delete;
  StreamingJSONParser &operator= (const StreamingJSONParser &) = delete;
  StreamingJSONParser (StreamingJSONParser &&) noexcept;
  StreamingJSONParser &operator= (StreamingJSONParser &&) noexcept;

  /// @brief Add token and update state.
  /// @param token_text Token text to add.
  /// @return ParserStatus indicating current state.
  ParserStatus AddToken (const std::string &token_text);

  /// @brief Get tokens allowed at current position based on schema.
  /// @return Vector of allowed token strings/characters.
  ///
  /// Returns "ANY" as special marker when string content is unconstrained.
  std::vector<std::string> GetAllowedTokens () const;

  /// @brief Get parsed JSON object.
  /// @return Parsed object (may be incomplete if not satisfied).
  const nlohmann::json &GetParsedObject () const;

  /// @brief Get current parser state.
  JSONState State () const;

  /// @brief Get accumulated buffer.
  const std::string &Buffer () const;

  /// @brief Reset parser to initial state.
  void Reset ();

  // --- Accessors for sampler integration ---

  /// @brief Get current field being parsed (if any).
  const std::string &CurrentKey () const;

  /// @brief Get partial key buffer during key string parsing.
  const std::string &PartialKey () const;

  /// @brief Get partial value buffer during value parsing.
  const std::string &PartialValue () const;

  /// @brief Get set of fields that have been seen.
  const std::set<std::string> &SeenFields () const;

  /// @brief Get set of required fields from schema.
  const std::set<std::string> &RequiredFields () const;

  /// @brief Get expected type for current value (from schema).
  const std::optional<std::string> &CurrentValueType () const;

  /// @brief Get enum values for current value (if any).
  const std::optional<std::vector<nlohmann::json>> &CurrentEnumValues () const;

  /// @brief Get enum type for current value.
  const std::optional<std::string> &CurrentEnumType () const;

  // --- oneOf state accessors ---

  /// @brief Get oneOf branches (empty if no oneOf in schema).
  const std::vector<nlohmann::json> &OneOfBranches () const;

  /// @brief Get active branch index (nullopt if not locked).
  std::optional<int> ActiveBranchIndex () const;

  /// @brief Get oneOf discriminator (nullopt if none detected).
  const std::optional<OneOfDiscriminator> &OneOfDiscriminatorInfo () const;

  /// @brief Get union of properties across all oneOf branches.
  const std::set<std::string> &OneOfUnionProperties () const;

  /// @brief Check if in oneOf union mode (tie case).
  bool IsOneOfUnionMode () const;

  /// @brief Get preferred fields before branch lock.
  const std::vector<std::string> &PreferredFieldsPrelock () const;

  /// @brief Get the current schema (branch schema if locked, else top-level).
  const nlohmann::json &CurrentSchema () const;

  /// @brief Get effective required fields at current point.
  std::set<std::string> EffectiveRequiredFields () const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cortext
