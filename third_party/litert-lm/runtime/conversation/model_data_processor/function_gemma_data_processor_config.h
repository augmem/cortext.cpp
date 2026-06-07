// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_FUNCTION_GEMMA_DATA_PROCESSOR_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_FUNCTION_GEMMA_DATA_PROCESSOR_CONFIG_H_

#include <string>

namespace litert::lm {

// Config for FunctionGemmaDataProcessor.
struct FunctionGemmaDataProcessorConfig {
  // The constraint mode when constrained decoding is enabled.
  enum class ConstraintMode {
    kUnspecified,
    // Both function call and text output are allowed.
    kTextAndOr,
    // Only function call is allowed.
    kFunctionCallOnly,
  };
  // Signifies the beginning of a tool call.
  std::string code_fence_start = "<start_function_call>";
  // Signifies the end of tool call.
  std::string code_fence_end = "<end_function_call>";
  std::string open_quote = "<escape>";
  std::string close_quote = "<escape>";
  std::string function_response_start = "<start_function_response>";
  // The syntax type of the tool call.
  std::string syntax_type = "fc";
  // Whether to escape the fence strings for regex matching.
  bool escape_fence_strings = true;
  // An optional regex to match each line of the tool code block.
  std::string tool_code_regex = "";
  // Whether to use the chat template for applying FC format.
  bool use_template_for_fc_format = false;
  // The constraint mode when constrained decoding is enabled. Default is
  // kTextAndOr.
  ConstraintMode constraint_mode = ConstraintMode::kTextAndOr;
};

// Arguments for FunctionGemmaDataProcessor.
struct FunctionGemmaDataProcessorArguments {};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_FUNCTION_GEMMA_DATA_PROCESSOR_CONFIG_H_
