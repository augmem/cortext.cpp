// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_FACTORY_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// Create an instance of LlmExecutor for LiteRT compiled models. Supports both
// statically and dynamically shaped models.
// Args:
//   executor_settings: Settings for the executor.
//   lrt_env: The LiteRT environment.
//   resources: The model resources.
absl::StatusOr<std::unique_ptr<LlmExecutor>>
CreateLlmLiteRtCompiledModelExecutor(LlmExecutorSettings executor_settings,
                                     Environment& lrt_env,
                                     ModelResources& resources);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_COMPILED_MODEL_EXECUTOR_FACTORY_H_
