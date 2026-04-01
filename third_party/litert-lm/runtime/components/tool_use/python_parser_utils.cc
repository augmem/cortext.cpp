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

#include "runtime/components/tool_use/python_parser_utils.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ANTLRInputStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTree.h"
#include "tree/ParseTreeWalker.h"
#include "tree/TerminalNode.h"
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "AntlrPythonLexer.h"
#include "AntlrPythonParser.h"
#include "AntlrPythonParserBaseListener.h"
#include "runtime/components/tool_use/parser_common.h"

namespace litert::lm {

namespace {

absl::StatusOr<nlohmann::ordered_json> ParsePythonList(
    antlr_python_tool_call_parser::AntlrPythonParser::ListContext* list);

absl::StatusOr<nlohmann::ordered_json> ParsePythonDict(
    antlr_python_tool_call_parser::AntlrPythonParser::DictContext* dict);

absl::StatusOr<nlohmann::ordered_json> ParsePythonObject(
    antlr_python_tool_call_parser::AntlrPythonParser::ObjectContext* object);

absl::StatusOr<nlohmann::ordered_json> ParsePythonValue(
    antlr_python_tool_call_parser::AntlrPythonParser::ValueContext* value) {
  if (value == nullptr) {
    return nlohmann::ordered_json();
  }
  std::string text = value->getText();
  if (value->INT()) {
    int int_value;
    if (!absl::SimpleAtoi(text, &int_value)) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid int: ", text));
    }
    return nlohmann::ordered_json(int_value);
  } else if (value->FLOAT()) {
    double double_value;
    if (!absl::SimpleAtod(text, &double_value)) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid float: ", text));
    }
    return nlohmann::ordered_json(double_value);
  } else if (value->STRING()) {
    return nlohmann::ordered_json(std::string(StripQuotes(text)));
  } else if (value->BOOL()) {
    return nlohmann::ordered_json(text == "True");
  } else if (value->NONE()) {
    return nlohmann::ordered_json(nullptr);
  } else if (value->list()) {
    return ParsePythonList(value->list());
  } else if (value->dict()) {
    return ParsePythonDict(value->dict());
  } else if (value->object()) {
    return ParsePythonObject(value->object());
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown value type: ", text));
}

absl::StatusOr<nlohmann::ordered_json> ParsePythonList(
    antlr_python_tool_call_parser::AntlrPythonParser::ListContext* list) {
  nlohmann::ordered_json list_json = nlohmann::ordered_json::array();
  if (list == nullptr) {
    return list_json;
  }
  for (antlr_python_tool_call_parser::AntlrPythonParser::ValueContext* value :
       list->value()) {
    auto parsed_value = ParsePythonValue(value);
    if (!parsed_value.ok()) {
      return parsed_value.status();
    }
    list_json.push_back(std::move(parsed_value).value());
  }
  return list_json;
}

absl::StatusOr<nlohmann::ordered_json> ParsePythonDict(
    antlr_python_tool_call_parser::AntlrPythonParser::DictContext* dict) {
  nlohmann::ordered_json dict_json = nlohmann::ordered_json::object();
  if (dict == nullptr) {
    return dict_json;
  }
  const auto& keys = dict->key();
  const auto& values = dict->value();
  const size_t num_pairs = std::min(keys.size(), values.size());

  for (size_t i = 0; i < num_pairs; ++i) {
    antlr_python_tool_call_parser::AntlrPythonParser::KeyContext* key_node =
        keys[i];
    antlr_python_tool_call_parser::AntlrPythonParser::ValueContext* value_ctx =
        values[i];

    if (key_node == nullptr || value_ctx == nullptr) {
      continue;
    }
    std::string key_text = std::string(StripQuotes(key_node->getText()));
    if (dict_json.contains(key_text)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate key: ", key_text));
    }
    auto parsed_value = ParsePythonValue(value_ctx);
    if (!parsed_value.ok()) {
      return parsed_value.status();
    }
    dict_json[key_text] = std::move(parsed_value).value();
  }
  return dict_json;
}

absl::StatusOr<nlohmann::ordered_json> ParsePythonObject(
    antlr_python_tool_call_parser::AntlrPythonParser::ObjectContext* object) {
  nlohmann::ordered_json object_json = nlohmann::ordered_json::object();
  if (object == nullptr || object->NAME() == nullptr) {
    return object_json;
  }
  std::string object_name = object->NAME()->getText();
  object_json["__type__"] = object_name;
  if (object->argValExpr()) {
    for (const auto& arg_val : object->argValExpr()->argVal()) {
      if (arg_val == nullptr || arg_val->NAME() == nullptr ||
          arg_val->NAME()->getText().empty()) {
        continue;
      }
      std::string name = arg_val->NAME()->getText();
      if (arg_val->value() == nullptr || arg_val->value()->getText().empty()) {
        continue;
      }
      if (object_json.contains(name)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Duplicate key: ", name));
      }
      auto parsed_value = ParsePythonValue(arg_val->value());
      if (!parsed_value.ok()) {
        return parsed_value.status();
      }
      object_json[name] = std::move(parsed_value).value();
    }
  }
  return object_json;
}

class PythonListener
    : public antlr_python_tool_call_parser::AntlrPythonParserBaseListener {
 public:
  void enterFunctionCall(
      antlr_python_tool_call_parser::AntlrPythonParser::FunctionCallContext*
          ctx) override;
  const nlohmann::ordered_json& tool_calls() const { return tool_calls_; }
  bool status() const { return status_; }

 private:
  nlohmann::ordered_json tool_calls_ = nlohmann::ordered_json::array();
  bool status_ = true;
};

void PythonListener::enterFunctionCall(
    antlr_python_tool_call_parser::AntlrPythonParser::FunctionCallContext*
        ctx) {
  if (ctx == nullptr) {
    status_ = false;
    return;
  }
  nlohmann::ordered_json tool_call;
  if (ctx->fullFunctionCall()) {
    antlr_python_tool_call_parser::AntlrPythonParser::FullFunctionCallContext*
        fcContext = ctx->fullFunctionCall();
    if (fcContext == nullptr || fcContext->NAME() == nullptr ||
        fcContext->NAME()->getText().empty()) {
      status_ = false;
      return;
    }
    tool_call["name"] = (fcContext->NAME()->getText());
    antlr_python_tool_call_parser::AntlrPythonParser::ArgValExprContext*
        argVals = fcContext->argValExpr();
    if (argVals == nullptr) {
      status_ = false;
      return;
    }
    for (antlr_python_tool_call_parser::AntlrPythonParser::ArgValContext*
             argValue : argVals->argVal()) {
      if (argValue == nullptr || argValue->NAME() == nullptr ||
          argValue->NAME()->getText().empty()) {
        status_ = false;
        return;
      }
      std::string name = argValue->NAME()->getText();
      antlr_python_tool_call_parser::AntlrPythonParser::ValueContext* value =
          argValue->value();
      if (value == nullptr || value->getText().empty()) {
        status_ = false;
        return;
      }
      auto parsed_value = ParsePythonValue(value);
      if (!parsed_value.ok()) {
        status_ = false;
        return;
      }
      if (tool_call.contains("arguments") &&
          tool_call["arguments"].contains(name)) {
        // Duplicate arg name.
        status_ = false;
        return;
      }
      // The parsed_value is already nlohmann::ordered_json.
      tool_call["arguments"][name] = parsed_value.value();
    }
  } else if (ctx->emptyFunctionCall()) {
    if (ctx->emptyFunctionCall()->NAME() == nullptr) {
      status_ = false;
      return;
    }
    tool_call["name"] = (ctx->emptyFunctionCall()->NAME()->getText());
  } else {
    status_ = false;
    return;
  }

  tool_calls_.push_back(tool_call);
  status_ = true;
}

}  // namespace

absl::StatusOr<nlohmann::ordered_json> ParsePythonExpression(
    absl::string_view text) {
  if (text.empty()) {
    return nlohmann::ordered_json::array();
  }
  antlr4::ANTLRInputStream input(std::string(text.begin(), text.end()));
  antlr_python_tool_call_parser::AntlrPythonLexer lexer(&input);
  lexer.removeErrorListeners();
  DefaultErrorListener error_listener;
  lexer.addErrorListener(&error_listener);
  antlr4::CommonTokenStream tokens(&lexer);
  tokens.fill();
  if (!error_listener.status()) {
    // Lexer reported one or more errors.
    return absl::InvalidArgumentError(
        "Failed to parse tool call: Lexer reported errors.");
  }
  antlr_python_tool_call_parser::AntlrPythonParser parser(&tokens);
  parser.removeErrorListeners();
  DefaultErrorListener parser_error_listener;
  parser.addErrorListener(&parser_error_listener);
  antlr4::tree::ParseTree* tree = parser.main();
  if (!parser_error_listener.status()) {
    // Parser reported one or more errors.
    return absl::InvalidArgumentError(
        "Failed to parse tool call: Parser reported errors.");
  }
  PythonListener listener;
  antlr4::tree::ParseTreeWalker::DEFAULT.walk(&listener, tree);

  if (!listener.status()) {
    // Listener reported one or more errors.
    return absl::InvalidArgumentError(
        "Failed to parse tool call: Listener reported errors.");
  }

  return listener.tool_calls();
}

}  // namespace litert::lm
