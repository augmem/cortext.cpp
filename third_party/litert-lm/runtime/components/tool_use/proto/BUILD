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

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//rust:defs.bzl", "rust_proto_library")

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

proto_library(
    name = "tool_call_proto",
    srcs = ["tool_call.proto"],
)

cc_proto_library(
    name = "tool_call_cc_proto",
    deps = [":tool_call_proto"],
)

rust_proto_library(
    name = "tool_call_rust_proto",
    deps = [":tool_call_proto"],
)
