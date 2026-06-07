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

# Custom Bazel target for Cortext - builds LiteRT-LM as a shared library
# This file is appended to the LiteRT-LM BUILD file by cmake/litert_append_build.cmake
#
# Symbol visibility is controlled by cortext's CMake build (hidden by default),
# so we no longer need to filter exports here.

load("@rules_cc//cc:defs.bzl", "cc_binary")

# C++ shared library
cc_binary(
    name = "liblitert_lm_cpp.so",
    deps = [
        "//runtime/engine:litert_lm_lib",
    ],
    linkshared = True,
    visibility = ["//visibility:public"],
)
