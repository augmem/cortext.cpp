# Copyright 2025 The ODML Authors.
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
# Custom Bazel target for Cortext - exports only litert::lm API symbols
# This file is appended to the LiteRT-LM BUILD file by cmake/litert_append_build.cmake
#
# Uses -exported_symbols_list to hide internal dependencies (like protobuf)
# that would conflict with system libraries (e.g., OpenTelemetry OTLP).

load("@rules_cc//cc:defs.bzl", "cc_binary")

# Exported symbols list file (copied by cmake/litert_append_build.cmake)
exports_files(["litert-exported-symbols.txt"])

# C++ shared library with controlled symbol exports
# Uses cc_binary with linkshared=True to create a .so/.dylib
#
# On macOS: Uses -exported_symbols_list to hide protobuf symbols
# On Linux: Uses version script to hide protobuf symbols
cc_binary(
    name = "liblitert_lm_cpp.so",
    deps = [
        "//runtime/engine:litert_lm_lib",
    ],
    linkshared = True,
    linkopts = select({
        "@platforms//os:macos": [
            "-Wl,-exported_symbols_list,$(location :litert-exported-symbols.txt)",
        ],
        "@platforms//os:linux": [
            # Linux uses version script (TODO: create litert-version-script.map)
            # "-Wl,--version-script=$(location :litert-version-script.map)",
        ],
        "//conditions:default": [],
    }),
    additional_linker_inputs = [":litert-exported-symbols.txt"],
    visibility = ["//visibility:public"],
)
