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

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

licenses(["notice"])

cc_library(
    name = "bitmap",
    hdrs = ["bitmap.h"],
)

cc_library(
    name = "constraint",
    hdrs = ["constraint.h"],
    deps = [
        ":bitmap",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "constraint_provider",
    hdrs = ["constraint_provider.h"],
    deps = [
        ":constraint",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "constrained_decoder",
    srcs = ["constrained_decoder.cc"],
    hdrs = ["constrained_decoder.h"],
    deps = [
        ":constraint",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "//runtime/util:convert_tensor_buffer",
        "//runtime/util:litert_status_util",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_layout",
            "@litert//litert/cc:litert_macros",
            "@litert//litert/cc:litert_model",
            "@litert//litert/cc:litert_tensor_buffer",
        ],
    }),
)

cc_library(
    name = "fake_constraint",
    srcs = ["fake_constraint.cc"],
    hdrs = ["fake_constraint.h"],
    deps = [
        ":bitmap",
        ":constraint",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
