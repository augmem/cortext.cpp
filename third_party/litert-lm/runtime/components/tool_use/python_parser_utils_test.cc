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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/test_utils.h"  // NOLINT

namespace {

using ::litert::lm::ParsePythonExpression;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(PythonParserUtilsTest, ParseSingleToolCall) {
  EXPECT_THAT(ParsePythonExpression("function_name(x='hello')"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": "hello"
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseMultipleToolCalls) {
  EXPECT_THAT(ParsePythonExpression("[func_1(x='hello'), func_2(y=2)]"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "func_1",
                "arguments": {
                  "x": "hello"
                }
              },
              {
                "name": "func_2",
                "arguments": {
                  "y": 2
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseMultipleToolCallsOnSeparateLines) {
  EXPECT_THAT(ParsePythonExpression(R"(func_1(x='hello')
func_2(y=2))"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "func_1",
                "arguments": {
                  "x": "hello"
                }
              },
              {
                "name": "func_2",
                "arguments": {
                  "y": 2
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseEmptyList) {
  EXPECT_THAT(ParsePythonExpression("[]"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([])json")));
}

TEST(PythonParserUtilsTest, ParseEmptyString) {
  EXPECT_THAT(ParsePythonExpression(""),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([])json")));
}

TEST(PythonParserUtilsTest, ParseInvalidToolCall) {
  EXPECT_THAT(ParsePythonExpression("invalid_tool_call"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PythonParserUtilsTest, ParseNoArguments) {
  EXPECT_THAT(ParsePythonExpression("function_name()"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {}
              }])json")));
}

TEST(PythonParserUtilsTest, ParseIntegerArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=1)"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": 1
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseNegativeIntegerArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=-1)"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": -1
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseFloatArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=1.1)"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": 1.1
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseNegativeFloatArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=-1.1)"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": -1.1
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseNegativeFloatNoLeadingDigit) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=-.1)"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": -0.1
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseBooleanArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=True)"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": true
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseStringArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x='hello')"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": "hello"
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseListArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=[1, 2, 3])"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": [1, 2, 3]
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseMixedTypesListArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=[1, 2, 'hello'])"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": [1, 2, "hello"]
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseNestedListArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=[[1, 2], [3, 4]])"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": [[1, 2], [3, 4]]
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseEmptyListArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=[])"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": []
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseDictArgument) {
  EXPECT_THAT(ParsePythonExpression(
                  "function_name(x={'hello': 'world', 'foo': 'bar'})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": {"hello": "world", "foo": "bar"}
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseNestedDictArgument) {
  EXPECT_THAT(
      ParsePythonExpression(
          "function_name(x={'outer': {'hello': 'world', 'foo': 'bar'}})"),
      IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": {"outer": {"hello": "world", "foo": "bar"}}
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseEmptyDictArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x={})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": {}
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseNoneArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=None)"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": null
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseLiteralStringNoneArgument) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=\"None\")"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": "None"
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseObjectArgument) {
  EXPECT_THAT(
      ParsePythonExpression("function_name(x=Obj(hello='world', foo=1))"),
      IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": {
                    "__type__": "Obj",
                    "hello": "world",
                    "foo": 1
                  }
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseListOfDictsArgument) {
  EXPECT_THAT(ParsePythonExpression(
                  "function_name(x=[{'hello': 'world', 'foo': 'bar'}, "
                  "{'foo': 1, 'bar': 2}])"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": [
                    {
                      "hello": "world",
                      "foo": "bar"
                    },
                    {
                      "foo": 1,
                      "bar": 2
                    }
                  ]
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseArgumentsWithTrailingComma) {
  EXPECT_THAT(ParsePythonExpression("function_name(x='hello',)"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": "hello"
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseMultipleToolCallsWithTrailingComma) {
  EXPECT_THAT(ParsePythonExpression("[func_1(x='hello'), func_2(y=2),]"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "func_1",
                "arguments": {
                  "x": "hello"
                }
              },
              {
                "name": "func_2",
                "arguments": {
                  "y": 2
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseListArgumentWithTrailingComma) {
  EXPECT_THAT(ParsePythonExpression("function_name(x=[1, 2, 3,])"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": [1, 2, 3]
                }
              }])json")));
}

TEST(PythonParserUtilsTest, ParseDictArgumentWithTrailingComma) {
  EXPECT_THAT(ParsePythonExpression(
                  "function_name(x={'hello': 'world', 'foo': 'bar',})"),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json([{
                "name": "function_name",
                "arguments": {
                  "x": {"hello": "world", "foo": "bar"}
                }
              }])json")));
}

TEST(PythonParserUtilsTest, PositionalArgumentsAreInvalid) {
  EXPECT_THAT(ParsePythonExpression("function_name(1, 2, 3)"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PythonParserUtilsTest, InvalidArgType) {
  EXPECT_THAT(ParsePythonExpression("function_name(a=dadssadasd)"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PythonParserUtilsTest, InvalidArgExpression) {
  EXPECT_THAT(ParsePythonExpression("function_name(a=2*2)"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
