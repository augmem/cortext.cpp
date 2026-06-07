// Copyright 2026 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_IDS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_IDS_H_

namespace litert::lm {

// Common resource IDs are used to identify resources in the ResourceRegistry.
// Resource IDs are int values that are used to identify resources in the
// ResourceRegistry.
// The assigned values are intentionally spaced out to ensure we can add support
// for potentially multiple instances of the same resource type in the future.
enum ResourceId {
  LLM_EXECUTOR_0_RID = 0,
  VISION_EXECUTOR_0_RID = 20,
  AUDIO_EXECUTOR_0_RID = 40,
  TOKENIZER_0_RID = 60,
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_IDS_H_
