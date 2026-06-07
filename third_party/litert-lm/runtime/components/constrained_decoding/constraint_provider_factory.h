#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_CONSTRAINT_PROVIDER_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_CONSTRAINT_PROVIDER_FACTORY_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/tokenizer.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<ConstraintProvider>> CreateConstraintProvider(
    const ConstraintProviderConfig& constraint_provider_config,
    const Tokenizer& tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_CONSTRAINT_PROVIDER_FACTORY_H_
