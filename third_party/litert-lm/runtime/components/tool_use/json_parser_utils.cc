// Copyright 2025 The Google AI Edge Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/components/tool_use/json_parser_utils.h"

#include <string>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ANTLRInputStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTreeWalker.h"
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "AntlrJsonLexer.h"
#include "AntlrJsonParser.h"
#include "AntlrJsonParserBaseListener.h"
#include "runtime/components/tool_use/parser_common.h"

namespace litert::lm {

namespace {

absl::StatusOr<nlohmann::ordered_json> ParseJsonArray(
    antlr_json_tool_call_parser::AntlrJsonParser::ArrayContext* array_ctx);

absl::StatusOr<nlohmann::ordered_json> ParseJsonObject(
    antlr_json_tool_call_parser::AntlrJsonParser::ObjectContext* object_ctx);

// Parses a JSON value context into a nlohmann::ordered_json.
absl::StatusOr<nlohmann::ordered_json> ParseJsonValue(
    antlr_json_tool_call_parser::AntlrJsonParser::ValueContext* value_ctx) {
  if (value_ctx == nullptr) {
    return nlohmann::ordered_json();
  }

  if (value_ctx->STRING()) {
    return nlohmann::ordered_json(
        std::string(StripQuotes(value_ctx->getText())));
  } else if (value_ctx->NUMBER()) {
    double double_value;
    // JSON numbers can be ints or floats, SimpleAtod handles both.
    if (!absl::SimpleAtod(value_ctx->getText(), &double_value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse number: ", value_ctx->getText()));
    }
    return nlohmann::ordered_json(double_value);
  } else if (value_ctx->object()) {
    return ParseJsonObject(value_ctx->object());
  } else if (value_ctx->array()) {
    return ParseJsonArray(value_ctx->array());
  } else if (value_ctx->BOOLEAN()) {
    // JSON booleans are lowercase 'true' or 'false'.
    return nlohmann::ordered_json(value_ctx->getText() == "true");
  } else if (value_ctx->NONE()) {
    return nlohmann::ordered_json(nullptr);
  } else {
    // Should not happen if the grammar is correct and covers all value types.
    return absl::InternalError(
        absl::StrCat("Unhandled JSON value type: ", value_ctx->getText()));
  }
}

// Parses a JSON array context into a nlohmann::ordered_json array.
absl::StatusOr<nlohmann::ordered_json> ParseJsonArray(
    antlr_json_tool_call_parser::AntlrJsonParser::ArrayContext* array_ctx) {
  nlohmann::ordered_json list_value = nlohmann::ordered_json::array();
  if (array_ctx == nullptr) {
    return list_value;  // Return empty list for null context
  }

  for (antlr_json_tool_call_parser::AntlrJsonParser::ValueContext* value :
       array_ctx->value()) {
    absl::StatusOr<nlohmann::ordered_json> parsed_value = ParseJsonValue(value);
    if (!parsed_value.ok()) {
      return parsed_value.status();
    }
    list_value.push_back(std::move(parsed_value).value());
  }
  return list_value;
}

// Parses a JSON object context into a nlohmann::ordered_json object.
absl::StatusOr<nlohmann::ordered_json> ParseJsonObject(
    antlr_json_tool_call_parser::AntlrJsonParser::ObjectContext* object_ctx) {
  nlohmann::ordered_json struct_value = nlohmann::ordered_json::object();
  if (object_ctx == nullptr) {
    return struct_value;  // Return empty struct for null context
  }

  for (antlr_json_tool_call_parser::AntlrJsonParser::PairContext* pair_ctx :
       object_ctx->pair()) {
    if (pair_ctx == nullptr || pair_ctx->STRING() == nullptr ||
        pair_ctx->value() == nullptr) {
      // Skip invalid pairs, though this might indicate a parsing issue.
      ABSL_LOG(WARNING) << "Skipping invalid pair in JSON object.";
      continue;
    }

    std::string key_text =
        std::string(StripQuotes(pair_ctx->STRING()->getText()));
    if (key_text.empty()) {
      return absl::InvalidArgumentError("JSON object key cannot be empty.");
    }

    if (struct_value.contains(key_text)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate key in JSON object: ", key_text));
    }

    absl::StatusOr<nlohmann::ordered_json> parsed_value =
        ParseJsonValue(pair_ctx->value());
    if (!parsed_value.ok()) {
      return absl::Status(
          parsed_value.status().code(),
          absl::StrCat("Error parsing value for key '", key_text,
                       "': ", parsed_value.status().message()));
    }
    struct_value[key_text] = std::move(parsed_value).value();
  }
  return struct_value;
}

class JsonListener
    : public antlr_json_tool_call_parser::AntlrJsonParserBaseListener {
 public:
  JsonListener() : status_(false) {};
  void enterFunctionCall(
      antlr_json_tool_call_parser::AntlrJsonParser::FunctionCallContext* ctx)
      override;
  void enterFunctionCallList(
      antlr_json_tool_call_parser::AntlrJsonParser::FunctionCallListContext*
          ctx) override {
    if (ctx == nullptr) {
      return;
    }
    if (ctx->OPEN_BRACKET() != nullptr && ctx->CLOSE_BRACKET() != nullptr &&
        ctx->functionCall().empty()) {
      status_ = true;
    }
  }

  const nlohmann::ordered_json& tool_calls() const { return tool_calls_; }
  bool status() const { return status_; }

 private:
  nlohmann::ordered_json tool_calls_ = nlohmann::ordered_json::array();
  bool status_;
};

void JsonListener::enterFunctionCall(
    antlr_json_tool_call_parser::AntlrJsonParser::FunctionCallContext* ctx) {
  if (ctx == nullptr) {
    return;
  }
  nlohmann::ordered_json tool_call;
  if (ctx->fullFunctionCall()) {
    antlr_json_tool_call_parser::AntlrJsonParser::FullFunctionCallContext*
        fcContext = ctx->fullFunctionCall();
    if (fcContext == nullptr || fcContext->functionNamePair() == nullptr ||
        fcContext->functionNamePair()->getText().empty()) {
      return;
    }
    tool_call["name"] =
        (StripQuotes(fcContext->functionNamePair()->STRING()->getText()));
    antlr_json_tool_call_parser::AntlrJsonParser::FunctionArgsPairContext*
        argsPair = fcContext->functionArgsPair();
    if (argsPair == nullptr) {
      return;
    }
    absl::StatusOr<nlohmann::ordered_json> parsed_args =
        ParseJsonObject(argsPair->object());
    if (!parsed_args.ok()) {
      status_ = false;
      return;
    }
    tool_call["arguments"] = std::move(parsed_args).value();
    tool_calls_.push_back(tool_call);
    status_ = true;
  } else if (ctx->emptyFunctionCall()) {
    status_ = true;
  } else {
    return;
  }
}

}  // namespace

absl::StatusOr<nlohmann::ordered_json> ParseJsonExpression(
    absl::string_view text) {
  antlr4::ANTLRInputStream input(std::string(text.begin(), text.end()));
  antlr_json_tool_call_parser::AntlrJsonLexer lexer(&input);
  lexer.removeErrorListeners();
  DefaultErrorListener lexer_error_listener;
  lexer.addErrorListener(&lexer_error_listener);

  antlr4::CommonTokenStream tokens(&lexer);
  tokens.fill();  // Consume all tokens from the lexer.

  if (!lexer_error_listener.status()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to lexer JSON input.", text));
  }
  antlr_json_tool_call_parser::AntlrJsonParser parser(&tokens);
  parser.removeErrorListeners();
  DefaultErrorListener parser_error_listener;
  parser.addErrorListener(&parser_error_listener);

  // Start parsing from the 'json' rule.
  antlr_json_tool_call_parser::AntlrJsonParser::JsonContext* json_ctx =
      parser.json();

  if (!parser_error_listener.status() || parser.getNumberOfSyntaxErrors() > 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse JSON input.", text));
  }

  if (json_ctx == nullptr) {
    return absl::InvalidArgumentError("Parsing resulted in a null context.");
  }

  JsonListener listener;
  antlr4::tree::ParseTreeWalker::DEFAULT.walk(&listener, json_ctx);

  if (!listener.status()) {
    // Listener reported one or more errors.
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse tool call", text));
  }

  return listener.tool_calls();
}

}  // namespace litert::lm
