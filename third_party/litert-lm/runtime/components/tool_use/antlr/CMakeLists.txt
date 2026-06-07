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

# TODO(totero): Refactor anltr generation step into generators.cmake file.
# --- ANTLR GENERATION ---
set(ANTLR_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/antlr_generated)
file(MAKE_DIRECTORY ${ANTLR_GENERATED_DIR})

add_custom_command(
    OUTPUT
        ${ANTLR_GENERATED_DIR}/AntlrJsonLexer.h
        ${ANTLR_GENERATED_DIR}/AntlrJsonParser.h
        ${ANTLR_GENERATED_DIR}/AntlrJsonLexer.cpp
        ${ANTLR_GENERATED_DIR}/AntlrJsonParser.cpp
        ${ANTLR_GENERATED_DIR}/AntlrJsonParserBaseListener.h
        ${ANTLR_GENERATED_DIR}/AntlrJsonParserListener.h
    COMMAND java -jar ${ANTLR_JAR_PATH}
            -Dlanguage=Cpp
            -listener -visitor
            -o ${ANTLR_GENERATED_DIR}
            -package antlr_json_tool_call_parser
            ${PKG_ROOT}/AntlrJsonLexer.g4
            ${PKG_ROOT}/AntlrJsonParser.g4
    DEPENDS ${PKG_ROOT}/AntlrJsonLexer.g4 ${PKG_ROOT}/AntlrJsonParser.g4
    COMMENT "Generating JSON Parser C++ code..."
)

add_litertlm_library(antlr_json_tool_call_parser STATIC
    ${ANTLR_GENERATED_DIR}/AntlrJsonLexer.cpp
    ${ANTLR_GENERATED_DIR}/AntlrJsonParser.cpp
)
add_library(LiteRTLM::Runtime::Components::ToolUse::Antlr::Json ALIAS antlr_json_tool_call_parser)

target_include_directories(antlr_json_tool_call_parser
    PUBLIC
        ${ANTLR_GENERATED_DIR}
        ${PKG_ROOT}
)

target_link_libraries(antlr_json_tool_call_parser
    PUBLIC
        antlr4_static
)

add_custom_command(
    OUTPUT
        ${ANTLR_GENERATED_DIR}/AntlrPythonLexer.h
        ${ANTLR_GENERATED_DIR}/AntlrPythonParser.h
        ${ANTLR_GENERATED_DIR}/AntlrPythonLexer.cpp
        ${ANTLR_GENERATED_DIR}/AntlrPythonParser.cpp
        ${ANTLR_GENERATED_DIR}/AntlrPythonParserBaseListener.h
        ${ANTLR_GENERATED_DIR}/AntlrPythonParserListener.h
    COMMAND java -jar ${ANTLR_JAR_PATH}
            -Dlanguage=Cpp
            -listener -visitor
            -o ${ANTLR_GENERATED_DIR}
            -package antlr_python_tool_call_parser
            ${PKG_ROOT}/AntlrPythonLexer.g4
            ${PKG_ROOT}/AntlrPythonParser.g4
    DEPENDS ${PKG_ROOT}/AntlrPythonLexer.g4 ${PKG_ROOT}/AntlrPythonParser.g4
    COMMENT "Generating Python Parser C++ code..."
)

add_litertlm_library(antlr_python_tool_call_parser STATIC
    ${ANTLR_GENERATED_DIR}/AntlrPythonLexer.cpp
    ${ANTLR_GENERATED_DIR}/AntlrPythonParser.cpp
)
add_library(LiteRTLM::Runtime::Components::ToolUse::Antlr::Python ALIAS antlr_python_tool_call_parser)

target_include_directories(antlr_python_tool_call_parser
    PUBLIC
        ${ANTLR_GENERATED_DIR}
        ${PKG_ROOT}
)

target_link_libraries(antlr_python_tool_call_parser
    PUBLIC
        antlr4_static
)

add_custom_command(
    OUTPUT
        ${ANTLR_GENERATED_DIR}/AntlrFcLexer.h
        ${ANTLR_GENERATED_DIR}/AntlrFcParser.h
        ${ANTLR_GENERATED_DIR}/AntlrFcLexer.cpp
        ${ANTLR_GENERATED_DIR}/AntlrFcParser.cpp
        ${ANTLR_GENERATED_DIR}/AntlrFcParserBaseListener.h
        ${ANTLR_GENERATED_DIR}/AntlrFcParserListener.h
    COMMAND java -jar ${ANTLR_JAR_PATH}
            -Dlanguage=Cpp
            -listener -visitor
            -o ${ANTLR_GENERATED_DIR}
            -package antlr_python_tool_call_parser
            ${PKG_ROOT}/AntlrFcLexer.g4
            ${PKG_ROOT}/AntlrFcParser.g4
    DEPENDS ${PKG_ROOT}/AntlrFcLexer.g4 ${PKG_ROOT}/AntlrFcParser.g4
    COMMENT "Generating Fc Parser C++ code..."
)

add_litertlm_library(antlr_fc_tool_call_parser STATIC
    ${ANTLR_GENERATED_DIR}/AntlrFcLexer.cpp
    ${ANTLR_GENERATED_DIR}/AntlrFcParser.cpp
)
add_library(LiteRTLM::Runtime::Components::ToolUse::Antlr::Fc ALIAS antlr_fc_tool_call_parser)

target_include_directories(antlr_fc_tool_call_parser
    PUBLIC
        ${ANTLR_GENERATED_DIR}
        ${PKG_ROOT}
)

target_link_libraries(antlr_fc_tool_call_parser
    PUBLIC
        antlr4_static
)

add_library(runtime_components_tool_use_antlr_libs INTERFACE)
add_library(LiteRTLM::Runtime::Components::ToolUse::Antlr ALIAS runtime_components_tool_use_antlr_libs)

target_link_libraries(runtime_components_tool_use_antlr_libs INTERFACE
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Json
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Python
    LiteRTLM::Runtime::Components::ToolUse::Antlr::Fc
)