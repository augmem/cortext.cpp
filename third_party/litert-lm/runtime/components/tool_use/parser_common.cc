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

#include "runtime/components/tool_use/parser_common.h"

#include <cstddef>
#include <exception>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ANTLRErrorListener.h"
#include "Parser.h"
#include "Recognizer.h"
#include "Token.h"
#include "atn/ATNConfigSet.h"
#include "dfa/DFA.h"
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/proto/tool_call.pb.h"

namespace litert::lm {

namespace {

nlohmann::ordered_json StructToJson(const proto::Struct& struct_value);
nlohmann::ordered_json ListToJson(const proto::ListValue& list_value);

nlohmann::ordered_json ValueToJson(const proto::Value& value) {
  switch (value.kind_case()) {
    case proto::Value::kNullValue:
      return nlohmann::ordered_json();
    case proto::Value::kNumberValue:
      return nlohmann::ordered_json(value.number_value());
    case proto::Value::kStringValue:
      return nlohmann::ordered_json(value.string_value());
    case proto::Value::kBoolValue:
      return nlohmann::ordered_json(value.bool_value());
    case proto::Value::kStructValue:
      return StructToJson(value.struct_value());
    case proto::Value::kListValue:
      return ListToJson(value.list_value());
    default:
      return nlohmann::ordered_json();
  }
}

nlohmann::ordered_json StructToJson(const proto::Struct& struct_value) {
  nlohmann::ordered_json struct_json = nlohmann::ordered_json::object();
  for (const proto::Field& field : struct_value.fields()) {
    struct_json[field.name()] = ValueToJson(field.value());
  }
  return struct_json;
}

nlohmann::ordered_json ListToJson(const proto::ListValue& list_value) {
  nlohmann::ordered_json list_json = nlohmann::ordered_json::array();
  for (const proto::Value& value : list_value.values()) {
    list_json.push_back(ValueToJson(value));
  }
  return list_json;
}

}  // namespace

void DefaultErrorListener::syntaxError(antlr4::Recognizer* recognizer,
                                       antlr4::Token* offendingSymbol,
                                       size_t line, size_t charPositionInLine,
                                       const std::string& msg,
                                       std::exception_ptr e) {
  status_ = false;
}

void DefaultErrorListener::reportAmbiguity(antlr4::Parser* recognizer,
                                           const antlr4::dfa::DFA& dfa,
                                           size_t startIndex, size_t stopIndex,
                                           bool exact,
                                           const antlrcpp::BitSet& ambigAlts,
                                           antlr4::atn::ATNConfigSet* configs) {
  status_ = false;
}

void DefaultErrorListener::reportAttemptingFullContext(
    antlr4::Parser* recognizer, const antlr4::dfa::DFA& dfa, size_t startIndex,
    size_t stopIndex, const antlrcpp::BitSet& conflictingAlts,
    antlr4::atn::ATNConfigSet* configs) {
  status_ = false;
}

void DefaultErrorListener::reportContextSensitivity(
    antlr4::Parser* recognizer, const antlr4::dfa::DFA& dfa, size_t startIndex,
    size_t stopIndex, size_t prediction, antlr4::atn::ATNConfigSet* configs) {
  status_ = false;
}

absl::string_view StripQuotes(absl::string_view text) {
  if (text.size() < 2 || (text.front() != '"' && text.front() != '\'') ||
      text.back() != text.front()) {
    return text;
  }
  return text.substr(1, text.size() - 2);
}

nlohmann::ordered_json ToolCallsToJson(const proto::ToolCalls& tool_calls) {
  nlohmann::ordered_json tool_calls_json = nlohmann::ordered_json::array();
  for (const proto::ToolCall& tool_call : tool_calls.tool_calls()) {
    nlohmann::ordered_json tool_call_json = nlohmann::ordered_json::object();
    tool_call_json["name"] = tool_call.name();
    tool_call_json["arguments"] = nlohmann::ordered_json::object();
    for (const auto& field : tool_call.arguments().fields()) {
      tool_call_json["arguments"][field.name()] = ValueToJson(field.value());
    }
    tool_calls_json.push_back(std::move(tool_call_json));
  }
  return tool_calls_json;
}

}  // namespace litert::lm
