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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/test_utils.h"  // NOLINT

namespace {

using ::litert::lm::ParseJsonExpression;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(JsonParserUtilsTest, ParseSingleToolCall) {
  EXPECT_THAT(ParseJsonExpression(R"json({
                "name": "print",
                "arguments": {
                  "x": 1
                }
              })json"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "print",
                "arguments": {
                  "x": 1
                }
              }])json")));
}

TEST(JsonParserUtilsTest, ParseMultipleToolCalls) {
  EXPECT_THAT(ParseJsonExpression(R"json([
                {
                  "name": "func1",
                  "arguments": {
                    "a": "hello"
                  }
                },
                {
                  "name": "func2",
                  "arguments": {
                    "b": true,
                    "c": null
                  }
                }
              ])json"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([
                {
                  "name": "func1",
                  "arguments": {
                    "a": "hello"
                  }
                },
                {
                  "name": "func2",
                  "arguments": {
                    "b": true,
                    "c": null
                  }
                }
              ])json")));
}

TEST(JsonParserUtilsTest, ParseEmptyArguments) {
  EXPECT_THAT(ParseJsonExpression(R"json({
                "name": "empty_args_func",
                "arguments": {}
              })json"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "empty_args_func",
                "arguments": {}
              }])json")));
}

TEST(JsonParserUtilsTest, ParseComplexArguments) {
  EXPECT_THAT(ParseJsonExpression(R"json({
                "name": "complex_func",
                "arguments": {
                  "str_arg": "value",
                  "int_arg": 123,
                  "float_arg": -4.5,
                  "bool_arg": false,
                  "null_arg": null,
                  "list_arg": [1, "two", true, null, {"nested_key": "nested_val"}],
                  "obj_arg": {
                    "key1": "val1",
                    "key2": [10, 20]
                  }
                }
              })json"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "complex_func",
                "arguments": {
                  "str_arg": "value",
                  "int_arg": 123,
                  "float_arg": -4.5,
                  "bool_arg": false,
                  "null_arg": null,
                  "list_arg": [
                    1,
                    "two",
                    true,
                    null,
                    {
                      "nested_key": "nested_val"
                    }
                  ],
                  "obj_arg": {
                    "key1": "val1",
                    "key2": [
                      10,
                      20
                    ]
                  }
                }
              }])json")));
}

TEST(JsonParserUtilsTest, InvalidJson) {
  EXPECT_THAT(ParseJsonExpression(R"json({
                "name": "bad_tool_call",
                "arguments": {
                  "x": 1, // Missing closing brace
              })json"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonParserUtilsTest, MissingName) {
  EXPECT_THAT(ParseJsonExpression(R"json({
                "arguments": { "x": 1 }
              })json"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonParserUtilsTest, ArgumentsNotObject) {
  EXPECT_THAT(ParseJsonExpression(R"json({
                "name": "wrong_args",
                "arguments": [1, 2, 3]
              })json"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonParserUtilsTest, EmptyInputString) {
  EXPECT_THAT(ParseJsonExpression(""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonParserUtilsTest, EmptyJsonArray) {
  EXPECT_THAT(ParseJsonExpression("[]"),
              IsOkAndHolds(nlohmann::ordered_json::parse("[]")));
}

}  // namespace
