# Copyright 2024 sqlite-objstore
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

macro(objstore_configure_static_analysis)
    if(OBJSTORE_ENABLE_CLANG_TIDY)
        find_program(OBJSTORE_CLANG_TIDY_EXE NAMES clang-tidy)
        if(NOT OBJSTORE_CLANG_TIDY_EXE)
            message(FATAL_ERROR "clang-tidy requested but not found on PATH")
        endif()
        set(CMAKE_C_CLANG_TIDY
            "${OBJSTORE_CLANG_TIDY_EXE};-warnings-as-errors=*;-extra-arg=-std=c17"
            CACHE STRING "clang-tidy invocation for C sources" FORCE)
        set(CMAKE_CXX_CLANG_TIDY
            "${OBJSTORE_CLANG_TIDY_EXE};-warnings-as-errors=*;-extra-arg=-std=c++20"
            CACHE STRING "clang-tidy invocation for CXX sources" FORCE)
        set(CMAKE_EXPORT_COMPILE_COMMANDS ON
            CACHE BOOL "Export compile_commands.json for tooling" FORCE)
    endif()

    if(OBJSTORE_ENABLE_CPPCHECK)
        find_program(OBJSTORE_CPPCHECK_EXE NAMES cppcheck)
        if(NOT OBJSTORE_CPPCHECK_EXE)
            message(FATAL_ERROR "cppcheck requested but not found on PATH")
        endif()
        set(CMAKE_C_CPPCHECK
            "${OBJSTORE_CPPCHECK_EXE};--std=c17;--enable=warning,performance;--error-exitcode=2;--inline-suppr"
            CACHE STRING "cppcheck invocation for C sources" FORCE)
        set(CMAKE_CXX_CPPCHECK
            "${OBJSTORE_CPPCHECK_EXE};--std=c++20;--enable=warning,performance;--error-exitcode=2;--inline-suppr"
            CACHE STRING "cppcheck invocation for CXX sources" FORCE)
        set(CMAKE_EXPORT_COMPILE_COMMANDS ON
            CACHE BOOL "Export compile_commands.json for tooling" FORCE)
    endif()
endmacro()

