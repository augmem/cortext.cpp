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

#include <string>

#include <gtest/gtest.h>
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/rust/parsers.rs.h"

namespace litert::lm {
namespace {

TEST(ConvertJsonValueTest, HandlesComplexStructure) {
  // We use parse_json_expression to create a JsonValue because JsonValue
  // is an opaque CXX type and cannot be constructed from C++.
  // parse_json_expression expects a "Tool Call" structure (name + arguments).
  const std::string kJsonInput = R"json({
    "name": "test_tool",
    "arguments": {
      "null_field": null,
      "bool_true": true,
      "bool_false": false,
      "number_int": 42,
      "number_float": 3.14,
      "string_val": "hello",
      "array_mixed": [1, "two", false],
      "object_nested": {
        "inner_key": "inner_val"
      }
    }
  })json";

  const auto result = parse_json_expression(kJsonInput);
  ASSERT_TRUE(result.is_ok)
      << "Failed to parse setup JSON: " << std::string(result.error);
  ASSERT_EQ(result.tool_calls.size(), 1);

  // Extract the "arguments" field which contains our test data.
  const auto& tool_call = result.tool_calls[0];
  const auto arguments = tool_call.object_get("arguments");

  // Convert from JsonValue to nlohmann::ordered_json.
  nlohmann::ordered_json converted = ConvertJsonValue(*arguments);

  // Verify primitive types.
  EXPECT_TRUE(converted["null_field"].is_null());
  EXPECT_EQ(converted["bool_true"], true);
  EXPECT_EQ(converted["bool_false"], false);
  EXPECT_EQ(converted["number_int"], 42);
  EXPECT_NEAR(converted["number_float"].get<double>(), 3.14, 0.0001);
  EXPECT_EQ(converted["string_val"], "hello");

  // Verify array.
  ASSERT_TRUE(converted["array_mixed"].is_array());
  ASSERT_EQ(converted["array_mixed"].size(), 3);
  EXPECT_EQ(converted["array_mixed"][0], 1);
  EXPECT_EQ(converted["array_mixed"][1], "two");
  EXPECT_EQ(converted["array_mixed"][2], false);

  // Verify nested object.
  ASSERT_TRUE(converted["object_nested"].is_object());
  EXPECT_EQ(converted["object_nested"]["inner_key"], "inner_val");
}

}  // namespace
}  // namespace litert::lm
