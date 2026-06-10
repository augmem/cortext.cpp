#include "cortext/providers/provider.hpp"

namespace cortext::providers
{

InferenceContract
ContractForRole (Role role)
{
  InferenceContract contract;
  switch (role)
    {
    case Role::Summarizer:
      contract.needs_text = true;
      contract.needs_deadline = true;
      break;
    case Role::Extractor:
      contract.needs_text = true;
      contract.needs_deadline = true;
      // Extraction output feeds schema-validated parsing; free-form-only
      // implementations cannot honor it.
      contract.min_constraints = ConstraintSupport::NativeGrammar;
      break;
    case Role::Encoder:
      contract.needs_text = true;
      // Runtime label/similarity space (see kEmbeddingDim).
      contract.embedding_dim = 256;
      break;
    }
  return contract;
}

bool
VerifyContract (const Capabilities &capabilities,
                const InferenceContract &contract, std::string *error_out)
{
  auto fail = [error_out] (const char *message) {
    if (error_out != nullptr)
      {
        *error_out = message;
      }
    return false;
  };

  if (contract.needs_text && !capabilities.text)
    {
      return fail ("contract requires text support");
    }
  if (contract.needs_image && !capabilities.image)
    {
      return fail ("contract requires image support");
    }
  if (contract.needs_audio && !capabilities.audio)
    {
      return fail ("contract requires audio support");
    }
  if (contract.min_constraints != ConstraintSupport::None
      && capabilities.constraints == ConstraintSupport::None)
    {
      return fail ("contract requires a constrained-decoding mechanism "
                   "(native grammar or server-side schema)");
    }
  if (contract.embedding_dim != 0)
    {
      bool found = false;
      for (std::size_t dim : capabilities.embedding_dims)
        {
          if (dim == contract.embedding_dim)
            {
              found = true;
              break;
            }
        }
      if (!found)
        {
          return fail ("contract requires an unsupported embedding "
                       "dimension");
        }
    }
  if (contract.needs_deadline && !capabilities.honors_deadline)
    {
      return fail ("contract requires fail-fast deadline behavior");
    }
  if (contract.needs_determinism && !capabilities.deterministic)
    {
      return fail ("contract requires deterministic generation");
    }
  return true;
}

} // namespace cortext::providers
