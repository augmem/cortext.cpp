# Copyright 2025 Google LLC
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

load("@rules_kotlin//kotlin:jvm.bzl", "kt_jvm_library")
load("@rules_java//java:java_binary.bzl", "java_binary")

kt_jvm_library(
    name = "main_lib",
    srcs = ["Main.kt"],
    deps = ["//kotlin/java/com/google/ai/edge/litertlm:litertlm-jvm"],
)

# Interactive chat with a model using the LiteRT-LM Kotlin API on JVM (on Linux, MacOS, etc).
#
# User can try the Kotlin API (function calling, multi-modality) more seamlessly without an Android
# device.
#
# To run it with bazel:
# bazel run -c opt //kotlin/java/com/google/ai/edge/litertlm/example:main -- <abs_model_path>
#
# To build a standalone binary:
# bazel build -c opt //kotlin/java/com/google/ai/edge/litertlm/example:main_deploy.jar
# ./bazel-bin/kotlin/java/com/google/ai/edge/litertlm/example/main_deploy.jar -- <abs_model_apth>
java_binary(
    name = "main",
    jvm_flags = ["--enable-native-access=ALL-UNNAMED"],  # it is expect to access native code.
    main_class = "com.google.ai.edge.litertlm.example.MainKt",
    runtime_deps = [
        ":main_lib",
    ],
)
