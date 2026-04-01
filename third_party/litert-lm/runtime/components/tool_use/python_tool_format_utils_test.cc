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

#include "runtime/components/tool_use/python_tool_format_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/test_utils.h"  // NOLINT

namespace {

using ::litert::lm::FormatToolAsPython;
using ::litert::lm::FormatValueAsPython;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonString) {
  nlohmann::ordered_json value = "string value";
  EXPECT_THAT(FormatValueAsPython(value), IsOkAndHolds(R"("string value")"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonInteger) {
  nlohmann::ordered_json value = 123;
  EXPECT_THAT(FormatValueAsPython(value), IsOkAndHolds("123"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonFloat) {
  nlohmann::ordered_json value = 1.23;
  EXPECT_THAT(FormatValueAsPython(value), IsOkAndHolds("1.23"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonTrue) {
  nlohmann::ordered_json value = true;
  EXPECT_THAT(FormatValueAsPython(value), IsOkAndHolds("True"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonFalse) {
  nlohmann::ordered_json value = false;
  EXPECT_THAT(FormatValueAsPython(value), IsOkAndHolds("False"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonNull) {
  nlohmann::ordered_json value = nullptr;
  EXPECT_THAT(FormatValueAsPython(value), IsOkAndHolds("None"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonObject) {
  nlohmann::ordered_json value =
      nlohmann::ordered_json::parse(R"({"key": "value"})");
  EXPECT_THAT(FormatValueAsPython(value), IsOkAndHolds(R"({"key": "value"})"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonArray) {
  nlohmann::ordered_json value = nlohmann::ordered_json::parse(R"([1, "two"])");
  EXPECT_THAT(FormatValueAsPython(value), IsOkAndHolds(R"([1, "two"])"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonDict) {
  nlohmann::ordered_json object =
      nlohmann::ordered_json::parse(R"({"key1": "value1", "key2": 2})");
  EXPECT_THAT(FormatValueAsPython(object),
              IsOkAndHolds(R"({"key1": "value1", "key2": 2})"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonInstance) {
  nlohmann::ordered_json value = nlohmann::ordered_json::parse(
      R"({"type": "Object", "key1": "value1", "key2": "value2"})");
  EXPECT_THAT(FormatValueAsPython(value),
              IsOkAndHolds(R"(Object(key1="value1", key2="value2"))"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonDictNested) {
  nlohmann::ordered_json object = nlohmann::ordered_json::parse(
      R"({"key1": "value1", "key2": {"nested_key": "nested_value"}})");
  EXPECT_THAT(
      FormatValueAsPython(object),
      IsOkAndHolds(
          R"({"key1": "value1", "key2": {"nested_key": "nested_value"}})"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonDictWithArray) {
  nlohmann::ordered_json object = nlohmann::ordered_json::parse(
      R"({"key1": "value1", "key2": [1, "two"]})");
  EXPECT_THAT(FormatValueAsPython(object),
              IsOkAndHolds(R"({"key1": "value1", "key2": [1, "two"]})"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonInstanceNested) {
  nlohmann::ordered_json object = nlohmann::ordered_json::parse(
      R"json({
        "type": "Object",
        "arg1": "value1",
        "arg2": {
          "nested_key": "nested_value"
        }
      })json");
  EXPECT_THAT(
      FormatValueAsPython(object),
      IsOkAndHolds(
          R"(Object(arg1="value1", arg2={"nested_key": "nested_value"}))"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonInstanceWithArray) {
  nlohmann::ordered_json object = nlohmann::ordered_json::parse(
      R"({"type": "Object", "arg1": "value1", "arg2": [1, "two"]})");
  EXPECT_THAT(FormatValueAsPython(object),
              IsOkAndHolds(R"(Object(arg1="value1", arg2=[1, "two"]))"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonNestedArray) {
  nlohmann::ordered_json array =
      nlohmann::ordered_json::parse(R"([1, [2, 3], [4, [5, 6]]])");
  EXPECT_THAT(FormatValueAsPython(array),
              IsOkAndHolds(R"([1, [2, 3], [4, [5, 6]]])"));
}

TEST(PythonToolFormatUtilsTest, FormatValueAsPythonArrayWithObjects) {
  nlohmann::ordered_json array = nlohmann::ordered_json::parse(
      R"([{"key1": "value1"}, {"key2": "value2"}])");
  EXPECT_THAT(FormatValueAsPython(array),
              IsOkAndHolds(R"([{"key1": "value1"}, {"key2": "value2"}])"));
}

TEST(PythonToolFormatUtilsTest, FormatToolWithStringParameter) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "name": "test_tool",
      "description": "This is a test tool.",
      "parameters": {
        "properties": {
          "test_param_1": {
            "type": "string",
            "description": "First parameter."
          }
        }
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: str | None = None,
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
  """
)"));
}

TEST(PythonToolFormatUtilsTest, FormatToolWithMultipleParameters) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "name": "test_tool",
      "description": "This is a test tool.",
      "parameters": {
        "properties": {
          "test_param_1": {
            "type": "string",
            "description": "First parameter."
          },
          "test_param_2": {
            "type": "string",
            "description": "Second parameter."
          }
        }
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: str | None = None,
    test_param_2: str | None = None,
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
    test_param_2: Second parameter.
  """
)"));
}

TEST(PythonToolFormatUtilsTest, FormatToolWithRequiredParameters) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "name": "test_tool",
      "description": "This is a test tool.",
      "parameters": {
        "properties": {
          "test_param_1": {
            "type": "string",
            "description": "First parameter."
          },
          "test_param_2": {
            "type": "string",
            "description": "Second parameter."
          }
        },
        "required": ["test_param_1"]
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: str,
    test_param_2: str | None = None,
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
    test_param_2: Second parameter.
  """
)"));
}

TEST(PythonToolFormatUtilsTest, FormatToolWithArrayParameter) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "name": "test_tool",
      "description": "This is a test tool.",
      "parameters": {
        "properties": {
          "test_param_1": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "First parameter."
          }
        }
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: list[str] | None = None,
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
  """
)"));
}

TEST(PythonToolFormatUtilsTest, FormatToolWithObjectParameter) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "name": "test_tool",
      "description": "This is a test tool.",
      "parameters": {
        "properties": {
          "test_param_1": {
            "type": "object",
            "properties": {
              "field_1": {
                "type": "string"
              }
            },
            "description": "First parameter."
          }
        }
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: dict | None = None,
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
  """
)"));
}

TEST(PythonToolFormatUtilsTest, FormatToolWithMixedParameters) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "name": "test_tool",
      "description": "This is a test tool.",
      "parameters": {
        "properties": {
          "test_param_1": {
            "type": "string",
            "description": "First parameter."
          },
          "test_param_2": {
            "type": "object",
            "properties": {
              "field_1": {
                "type": "string"
              }
            },
            "description": "Second parameter."
          },
          "test_param_3": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "Third parameter."
          }
        },
        "required": ["test_param_1", "test_param_2", "test_param_3"]
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: str,
    test_param_2: dict,
    test_param_3: list[str],
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
    test_param_2: Second parameter.
    test_param_3: Third parameter.
  """
)"));
}

TEST(PythonToolFormatUtilsTest, FormatToolAlternativeFormat) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json(
    {
      "type": "function",
      "function": {
        "name": "test_tool",
        "description": "This is a test tool.",
        "parameters": {
          "properties": {
            "test_param_1": {
              "type": "string",
              "description": "First parameter."
            }
          }
        }
      }
    }
  )json");
  EXPECT_THAT(FormatToolAsPython(tool), IsOkAndHolds(R"(def test_tool(
    test_param_1: str | None = None,
) -> dict:
  """This is a test tool.

  Args:
    test_param_1: First parameter.
  """
)"));
}

}  // namespace
