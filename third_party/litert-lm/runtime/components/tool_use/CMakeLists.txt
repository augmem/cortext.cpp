# Copyright 2026 Google LLC.
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


set(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(antlr)

add_litertlm_library(runtime_components_tool_use_json_parser_utils STATIC
  json_parser_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::JsonUtils
  ALIAS runtime_components_tool_use_json_parser_utils)

target_include_directories(runtime_components_tool_use_json_parser_utils
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${GENERATED_SRC_DIR}
)

target_link_libraries(runtime_components_tool_use_json_parser_utils
  PUBLIC
    LiteRTLM::Runtime::Components::ToolUse::ParserCommon
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
    LITERTLM_DEPS
)

add_litertlm_library(runtime_components_tool_use_parser_common STATIC
  parser_common.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::ParserCommon
  ALIAS runtime_components_tool_use_parser_common)

target_include_directories(runtime_components_tool_use_parser_common
  PRIVATE
    ${LITERTLM_INCLUDE_PATHS}
    ${GENERATED_SRC_DIR}
)

target_link_libraries(runtime_components_tool_use_parser_common
  PUBLIC
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
    LITERTLM_DEPS
)

add_litertlm_library(runtime_components_tool_use_parser_utils STATIC
  parser_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::ParserUtils
  ALIAS runtime_components_tool_use_parser_utils)

target_include_directories(runtime_components_tool_use_parser_utils
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_tool_use_parser_utils
  PUBLIC
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Python
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Fc

    LiteRTLM::Runtime::Components::ToolUse::JsonUtils
    LiteRTLM::Runtime::Components::ToolUse::PythonUtils
    LiteRTLM::Runtime::Components::ToolUse::FcUtils

    runtime_util_litert_status_util
    LITERTLM_DEPS
)

add_litertlm_library(runtime_components_tool_use_python_parser_utils STATIC
  python_parser_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::PythonUtils
  ALIAS runtime_components_tool_use_python_parser_utils)

target_include_directories(runtime_components_tool_use_python_parser_utils
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_tool_use_python_parser_utils
  PUBLIC
    LiteRTLM::Runtime::Components::ToolUse::ParserCommon
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Python
    LITERTLM_DEPS
)

add_litertlm_library(runtime_components_tool_use_python_tool_format_utils STATIC
  python_tool_format_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::PythonFormatUtils
  ALIAS runtime_components_tool_use_python_tool_format_utils)

target_include_directories(runtime_components_tool_use_python_tool_format_utils
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_tool_use_python_tool_format_utils
  PUBLIC
    runtime_util_litert_status_util
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
    LITERTLM_DEPS
)


add_litertlm_library(runtime_components_tool_use_fc_parser_utils STATIC
  fc_parser_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::FcUtils
  ALIAS runtime_components_tool_use_fc_parser_utils)

target_include_directories(runtime_components_tool_use_fc_parser_utils
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_tool_use_fc_parser_utils
  PUBLIC
    LiteRTLM::Runtime::Components::ToolUse::ParserCommon
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Fc
    LITERTLM_DEPS
)

add_litertlm_library(runtime_components_tool_use_fc_tool_format_utils STATIC
  fc_tool_format_utils.cc
)
add_library(LiteRTLM::Runtime::Components::ToolUse::FcFormatUtils
  ALIAS runtime_components_tool_use_fc_tool_format_utils)

target_include_directories(runtime_components_tool_use_fc_tool_format_utils
  PRIVATE
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_components_tool_use_fc_tool_format_utils
  PUBLIC
    runtime_util_litert_status_util
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
    LITERTLM_DEPS
)


add_library(runtime_components_tool_use_libs INTERFACE)
add_library(LiteRTLM::Runtime::Components::ToolUse
  ALIAS runtime_components_tool_use_libs)

target_link_libraries(runtime_components_tool_use_libs INTERFACE
  LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
  LiteRTLM::Runtime::Components::ToolUse::Antlr::Python
  LiteRTLM::Runtime::Components::ToolUse::JsonUtils
  LiteRTLM::Runtime::Components::ToolUse::ParserCommon
  LiteRTLM::Runtime::Components::ToolUse::ParserUtils
  LiteRTLM::Runtime::Components::ToolUse::PythonUtils
  LiteRTLM::Runtime::Components::ToolUse::PythonFormatUtils
  LiteRTLM::Runtime::Components::ToolUse::FcFormatUtils
  LiteRTLM::Runtime::Components::ToolUse::FcFormatUtils

)