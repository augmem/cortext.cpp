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

#include "runtime/components/tool_use/fc_tool_format_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(FcToolFormatUtilsTest, FormatValueAsFcString) {
  EXPECT_THAT(FormatValueAsFc(nlohmann::ordered_json("string value")),
              IsOkAndHolds("<escape>string value<escape>"));
}

TEST(FcToolFormatUtilsTest, FormatValueAsFcInteger) {
  EXPECT_THAT(FormatValueAsFc(nlohmann::ordered_json(123)),
              IsOkAndHolds("123"));
}

TEST(FcToolFormatUtilsTest, FormatValueAsFcFloat) {
  EXPECT_THAT(FormatValueAsFc(nlohmann::ordered_json(1.23)),
              IsOkAndHolds("1.23"));
}

TEST(FcToolFormatUtilsTest, FormatValueAsFcTrue) {
  EXPECT_THAT(FormatValueAsFc(nlohmann::ordered_json(true)),
              IsOkAndHolds("true"));
}

TEST(FcToolFormatUtilsTest, FormatValueAsFcFalse) {
  EXPECT_THAT(FormatValueAsFc(nlohmann::ordered_json(false)),
              IsOkAndHolds("false"));
}

TEST(FcToolFormatUtilsTest, FormatValueAsFcNull) {
  EXPECT_THAT(FormatValueAsFc(nlohmann::ordered_json(nullptr)),
              IsOkAndHolds("null"));
}

TEST(FcToolFormatUtilsTest, FormatValueAsFcObject) {
  EXPECT_THAT(FormatValueAsFc(
                  nlohmann::ordered_json::parse(R"json({"key": "value"})json")),
              IsOkAndHolds("{key:<escape>value<escape>}"));
}

TEST(FcToolFormatUtilsTest, FormatValueAsFcArray) {
  EXPECT_THAT(
      FormatValueAsFc(nlohmann::ordered_json::parse(R"json([1, "two"])json")),
      IsOkAndHolds("[1,<escape>two<escape>]"));
}

TEST(FcToolFormatUtilsTest, FormatValueAsFcObjectComplex) {
  EXPECT_THAT(FormatValueAsFc(nlohmann::ordered_json::parse(
                  R"json({
                    "string_value": "foo",
                    "number_value": 123,
                    "boolean_value": true,
                    "null_value": null,
                    "nested_object": {"key": "value"},
                    "nested_array": [4, 5, 6],
                    "nested_array_of_objects": [
                      {"key1": 7},
                      {"key2": 8}
                    ]
                  })json")),
              IsOkAndHolds("{"
                           "string_value:<escape>foo<escape>,"
                           "number_value:123,"
                           "boolean_value:true,"
                           "null_value:null,"
                           "nested_object:{key:<escape>value<escape>},"
                           "nested_array:[4,5,6],"
                           "nested_array_of_objects:[{key1:7},{key2:8}]"
                           "}"));
}

TEST(FcToolFormatUtilsTest, FormatTool) {
  EXPECT_THAT(
      FormatToolAsFc(nlohmann::ordered_json::parse(R"json({
        "name": "search",
        "description": "Returns a list of web pages.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The search query."
            }
          }
        }
      })json")),
      IsOkAndHolds("declaration:search{"
                   "description:<escape>Returns a list of web pages.<escape>,"
                   "parameters:{"
                   "type:<escape>OBJECT<escape>,"
                   "properties:{"
                   "query:{"
                   "type:<escape>STRING<escape>,"
                   "description:<escape>The search query.<escape>"
                   "}"  // query
                   "}"  // properties
                   "}"  // parameters
                   "}"  // declaration
                   ));
}

TEST(FcToolFormatUtilsTest, FormatToolWithEmptyParameters) {
  EXPECT_THAT(
      FormatToolAsFc(nlohmann::ordered_json::parse(R"json({
        "name": "search",
        "description": "Returns a list of web pages.",
        "parameters": {}
      })json")),
      IsOkAndHolds("declaration:search{"
                   "description:<escape>Returns a list of web pages.<escape>,"
                   "parameters:{}"
                   "}"  // declaration
                   ));
}

TEST(FcToolFormatUtilsTest, FormatToolWithMultipleParameters) {
  EXPECT_THAT(
      FormatToolAsFc(nlohmann::ordered_json::parse(R"json({
        "name": "search",
        "description": "Returns a list of web pages.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The search query."
            },
            "max_results": {
              "type": "integer",
              "description": "The maximum number of results."
            }
          }
        }
      })json")),
      IsOkAndHolds("declaration:search{"
                   "description:<escape>Returns a list of web pages.<escape>,"
                   "parameters:{"
                   "type:<escape>OBJECT<escape>,"
                   "properties:{"
                   "query:{"
                   "type:<escape>STRING<escape>,"
                   "description:<escape>The search query.<escape>"
                   "},"  // query
                   "max_results:{"
                   "type:<escape>INTEGER<escape>,"
                   "description:<escape>The maximum number of results.<escape>"
                   "}"  // max_results
                   "}"  // properties
                   "}"  // parameters
                   "}"  // declaration
                   ));
}

TEST(FcToolFormatUtilsTest, FormatToolWithRequiredParameters) {
  EXPECT_THAT(
      FormatToolAsFc(nlohmann::ordered_json::parse(R"json({
        "name": "search",
        "description": "Returns a list of web pages.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The search query."
            },
            "max_results": {
              "type": "integer",
              "description": "The maximum number of results."
            }
          },
          "required": ["query"]
        }
      })json")),
      IsOkAndHolds("declaration:search{"
                   "description:<escape>Returns a list of web pages.<escape>,"
                   "parameters:{"
                   "type:<escape>OBJECT<escape>,"
                   "properties:{"
                   "query:{"
                   "type:<escape>STRING<escape>,"
                   "description:<escape>The search query.<escape>"
                   "},"  // query
                   "max_results:{"
                   "type:<escape>INTEGER<escape>,"
                   "description:<escape>The maximum number of results.<escape>}"
                   "},"  // max_results
                   "required:[<escape>query<escape>]"
                   "}"  // parameters
                   "}"  // declaration
                   ));
}

TEST(FcToolFormatUtilsTest, FormatToolInvalidInputType) {
  // Input is not an object.
  EXPECT_THAT(FormatToolAsFc(nlohmann::ordered_json(123)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Tool must be a JSON object. Got: number"));
}

TEST(FcToolFormatUtilsTest, FormatToolMissingName) {
  // Missing "name" field.
  EXPECT_THAT(FormatToolAsFc(nlohmann::ordered_json::parse(R"json({
                "description": "A tool without a name."
              })json")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Tool name is required and must be a string."));
}

TEST(FcToolFormatUtilsTest, FormatToolInvalidNameType) {
  // "name" field is not a string.
  EXPECT_THAT(FormatToolAsFc(nlohmann::ordered_json::parse(R"json({
                "name": 123,
                "description": "A tool with an invalid name type."
              })json")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Tool name is required and must be a string."));
}

TEST(FcToolFormatUtilsTest, FormatToolAlternativeFormat) {
  EXPECT_THAT(
      FormatToolAsFc(nlohmann::ordered_json::parse(R"json({
        "type": "function",
        "function": {
          "name": "search",
          "description": "Returns a list of web pages.",
          "parameters": {
            "type": "object",
            "properties": {
              "query": {
                "type": "string",
                "description": "The search query."
              }
            }
          }
        }
      })json")),
      IsOkAndHolds("declaration:search{"
                   "description:<escape>Returns a list of web pages.<escape>,"
                   "parameters:{"
                   "type:<escape>OBJECT<escape>,"
                   "properties:{"
                   "query:{"
                   "type:<escape>STRING<escape>,"
                   "description:<escape>The search query.<escape>"
                   "}"  // query
                   "}"  // properties
                   "}"  // parameters
                   "}"  // declaration
                   ));
}

}  // namespace
}  // namespace litert::lm
