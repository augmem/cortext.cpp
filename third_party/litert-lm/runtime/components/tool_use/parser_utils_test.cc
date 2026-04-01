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

#include "runtime/components/tool_use/parser_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::IsOkAndHolds;

TEST(ParserUtilsTest, GetSyntaxType) {
  EXPECT_EQ(GetSyntaxType("python"), SyntaxType::kPython);
  EXPECT_EQ(GetSyntaxType("json"), SyntaxType::kJson);
  EXPECT_EQ(GetSyntaxType("unknown"), SyntaxType::kUnknown);
}

TEST(ParserUtilsTest, EmptyResponse) {
  EXPECT_THAT(ParseTextAndToolCalls("",
                                    /*code_fence_start=*/"```tool_code\n",
                                    /*code_fence_end=*/"\n```",
                                    /*syntax_type=*/SyntaxType::kPython),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "content":[
                  {
                    "type": "text",
                    "text": ""
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, TextOnly) {
  EXPECT_THAT(ParseTextAndToolCalls("This is some text.",
                                    /*code_fence_start=*/"```tool_code\n",
                                    /*code_fence_end=*/"\n```",
                                    /*syntax_type=*/SyntaxType::kPython),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "content":[
                  {
                    "type": "text",
                    "text": "This is some text."
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, ParsePythonToolCall) {
  EXPECT_THAT(ParseTextAndToolCalls(R"(```tool_code
tool_name(x=1)
```)",
                                    /*code_fence_start=*/"```tool_code\n",
                                    /*code_fence_end=*/"\n```",
                                    /*syntax_type=*/SyntaxType::kPython),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, ParsePythonParallelToolCalls) {
  EXPECT_THAT(ParseTextAndToolCalls(R"(```tool_code
tool_1(x=1)
tool_2(y=2)
```)",
                                    /*code_fence_start=*/"```tool_code\n",
                                    /*code_fence_end=*/"\n```",
                                    /*syntax_type=*/SyntaxType::kPython),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_1",
                      "arguments": {
                        "x": 1
                      }
                    }
                  },
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_2",
                      "arguments": {
                        "y": 2
                      }
                    }
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, ParseTextAndPythonToolCalls) {
  EXPECT_THAT(ParseTextAndToolCalls(
                  R"(This is some text.
```tool_code
tool_name(x=1)
```)",
                  /*code_fence_start=*/"```tool_code\n",
                  /*code_fence_end=*/"\n```",
                  /*syntax_type=*/SyntaxType::kPython),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "content": [
                  {
                    "type": "text",
                    "text": "This is some text.\n"
                  }
                ],
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, ParsePythonCallWithRegex) {
  EXPECT_THAT(
      ParseTextAndToolCalls(R"(```tool_code
print(tool_name(x=1))
```)",
                            /*code_fence_start=*/"```tool_code\n",
                            /*code_fence_end=*/"\n```",
                            /*syntax_type=*/SyntaxType::kPython,
                            /*escape_fence_strings=*/true,
                            /*tool_code_regex=*/R"(print\((.+\(.*\))\))"),
      IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, ParseJsonToolCall) {
  EXPECT_THAT(ParseTextAndToolCalls(R"(```tool_code
[{"name": "tool_name", "arguments": {"x": 1}}]
```)",
                                    /*code_fence_start=*/"```tool_code\n",
                                    /*code_fence_end=*/"\n```",
                                    /*syntax_type=*/SyntaxType::kJson),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, ParseJsonParallelToolCalls) {
  EXPECT_THAT(ParseTextAndToolCalls(
                  R"(```tool_code
[
  {"name": "tool_1", "arguments": {"x": 1}},
  {"name": "tool_2", "arguments": {"y": 2}}
]
```)",
                  /*code_fence_start=*/"```tool_code\n",
                  /*code_fence_end=*/"\n```",
                  /*syntax_type=*/SyntaxType::kJson),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_1",
                      "arguments": {
                        "x": 1
                      }
                    }
                  },
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_2",
                      "arguments": {
                        "y": 2
                      }
                    }
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, ParseTextAndJsonToolCalls) {
  EXPECT_THAT(ParseTextAndToolCalls(
                  R"(This is some text.
```tool_code
[{"name": "tool_name", "arguments": {"x": 1}}]
```)",
                  /*code_fence_start=*/"```tool_code\n",
                  /*code_fence_end=*/"\n```",
                  /*syntax_type=*/SyntaxType::kJson),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "content": [
                  {
                    "type": "text",
                    "text": "This is some text.\n"
                  }
                ],
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, ParseTextThenCodeThenText) {
  EXPECT_THAT(ParseTextAndToolCalls(
                  R"(This is some text.
```tool_code
tool_1(x=1)
```
This is some more text.
)",
                  /*code_fence_start=*/"```tool_code\n",
                  /*code_fence_end=*/"\n```",
                  /*syntax_type=*/SyntaxType::kPython),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "content": [
                  {
                    "type": "text",
                    "text": "This is some text.\n"
                  },
                  {
                    "type": "text",
                    "text": "\nThis is some more text.\n"
                  }
                ],
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_1",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, ParseTextThenCodeThenTextThenCode) {
  EXPECT_THAT(ParseTextAndToolCalls(
                  R"(This is some text.
```tool_code
tool_1(x=1)
```
This is some more text.
```tool_code
tool_2(y=2)
```)",
                  /*code_fence_start=*/"```tool_code\n",
                  /*code_fence_end=*/"\n```",
                  /*syntax_type=*/SyntaxType::kPython),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                  "content": [
                    {
                      "type": "text",
                      "text": "This is some text.\n"
                    },
                    {
                      "type": "text",
                      "text": "\nThis is some more text.\n"
                    }
                  ],
                  "tool_calls": [
                    {
                      "type": "function",
                      "function": {
                        "name": "tool_1",
                        "arguments": {
                          "x": 1
                        }
                      }
                    },
                    {
                      "type": "function",
                      "function": {
                        "name": "tool_2",
                        "arguments": {
                          "y": 2
                        }
                      }
                    }
                  ]
                })json")));
}

TEST(ParserUtilsTest, ParseTextThenParallelToolCalls) {
  EXPECT_THAT(ParseTextAndToolCalls(
                  R"(This is some text.
```tool_code
tool_1(x=1)
tool_2(y=2)
```)",
                  /*code_fence_start=*/"```tool_code\n",
                  /*code_fence_end=*/"\n```",
                  /*syntax_type=*/SyntaxType::kPython),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "content": [
                  {
                    "type": "text",
                    "text": "This is some text.\n"
                  }
                ],
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_1",
                      "arguments": {
                        "x": 1
                      }
                    }
                  },
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_2",
                      "arguments": {
                        "y": 2
                      }
                    }
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, ParseTextThenParallelToolCallsThenText) {
  EXPECT_THAT(ParseTextAndToolCalls(
                  R"(This is some text.
```tool_code
tool_1(x=1)
tool_2(y=2)
```
This is some more text.
)",
                  /*code_fence_start=*/"```tool_code\n",
                  /*code_fence_end=*/"\n```",
                  /*syntax_type=*/SyntaxType::kPython),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "content": [
                  {
                    "type": "text",
                    "text": "This is some text.\n"
                  },
                  {
                    "type": "text",
                    "text": "\nThis is some more text.\n"
                  }
                ],
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_1",
                      "arguments": {
                        "x": 1
                      }
                    }
                  },
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_2",
                      "arguments": {
                        "y": 2
                      }
                    }
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, ParseTextThenCodeThenCode) {
  EXPECT_THAT(ParseTextAndToolCalls(
                  R"(This is some text.
```tool_code
tool_1(x=1)
```
```tool_code
tool_2(y=2)
```
This is some more text.
)",
                  /*code_fence_start=*/"```tool_code\n",
                  /*code_fence_end=*/"\n```",
                  /*syntax_type=*/SyntaxType::kPython),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "content": [
                  {
                    "type": "text",
                    "text": "This is some text.\n"
                  },
                  {
                    "type": "text",
                    "text": "\n"
                  },
                  {
                    "type": "text",
                    "text": "\nThis is some more text.\n"
                  }
                ],
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_1",
                      "arguments": {
                        "x": 1
                      }
                    }
                  },
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_2",
                      "arguments": {
                        "y": 2
                      }
                    }
                  }
                ]
              })json")));
}

TEST(ParserUtilsTest, IncompleteToolCodeIsText) {
  // Missing the closing ```.
  EXPECT_THAT(ParseTextAndToolCalls(
                  R"(```tool_code
tool_name(x=1)
)",
                  /*code_fence_start=*/"```tool_code\n",
                  /*code_fence_end=*/"\n```",
                  /*syntax_type=*/SyntaxType::kPython),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "content":[
                  {
                    "type": "text",
                    "text": "```tool_code\ntool_name(x=1)\n"
                  }
                ]
              })json")));
}

}  // namespace
}  // namespace litert::lm
