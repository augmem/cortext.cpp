# Copyright 2025 The Google AI Edge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_antlr//antlr:antlr4.bzl", "antlr")

package(
    default_visibility = [
        ":default_visibility",
    ],
)

package_group(
    name = "default_visibility",
    packages = [
        "//runtime/components/tool_use/...",
    ],
)

antlr(
    name = "antlr_python_tool_call_parser_source",
    srcs = [
        "AntlrPythonLexer.g4",
        "AntlrPythonParser.g4",
    ],
    language = "Cpp",
    listener = True,
    package = "antlr_python_tool_call_parser",
)
antlr(
    name = "antlr_json_tool_call_parser_source",
    srcs = [
        "AntlrJsonLexer.g4",
        "AntlrJsonParser.g4",
    ],
    language = "Cpp",
    listener = True,
    package = "antlr_json_tool_call_parser",
)
cc_library(
    name = "antlr_python_tool_call_parser",
    srcs = [
        ":antlr_python_tool_call_parser_source",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":antlr_python_tool_call_parser_source",
        "@antlr4//:cpp",
    ],
)
cc_library(
    name = "antlr_json_tool_call_parser",
    srcs = [
        ":antlr_json_tool_call_parser_source",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":antlr_json_tool_call_parser_source",
        "@antlr4//:cpp",
    ],
)
