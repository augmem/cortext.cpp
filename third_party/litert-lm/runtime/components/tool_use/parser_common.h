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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PARSER_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PARSER_COMMON_H_

#include <cstddef>
#include <exception>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ANTLRErrorListener.h"
#include "Parser.h"
#include "Recognizer.h"
#include "Token.h"
#include "atn/ATNConfigSet.h"
#include "dfa/DFA.h"
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/proto/tool_call.pb.h"

namespace litert::lm {

class DefaultErrorListener final : public antlr4::ANTLRErrorListener {
 public:
  DefaultErrorListener() : status_(true) {}

  void syntaxError(antlr4::Recognizer* recognizer,
                   antlr4::Token* offendingSymbol, size_t line,
                   size_t charPositionInLine, const std::string& msg,
                   std::exception_ptr e) override;

  void reportAmbiguity(antlr4::Parser* recognizer, const antlr4::dfa::DFA& dfa,
                       size_t startIndex, size_t stopIndex, bool exact,
                       const antlrcpp::BitSet& ambigAlts,
                       antlr4::atn::ATNConfigSet* configs) override;

  void reportAttemptingFullContext(antlr4::Parser* recognizer,
                                   const antlr4::dfa::DFA& dfa,
                                   size_t startIndex, size_t stopIndex,
                                   const antlrcpp::BitSet& conflictingAlts,
                                   antlr4::atn::ATNConfigSet* configs) override;

  void reportContextSensitivity(antlr4::Parser* recognizer,
                                const antlr4::dfa::DFA& dfa, size_t startIndex,
                                size_t stopIndex, size_t prediction,
                                antlr4::atn::ATNConfigSet* configs) override;
  bool status() const { return status_; }

 private:
  bool status_;
};

absl::string_view StripQuotes(absl::string_view text);

nlohmann::ordered_json ToolCallsToJson(const proto::ToolCalls& tool_calls);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PARSER_COMMON_H_
