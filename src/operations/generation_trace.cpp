#include "cortext/operations/generation_trace.hpp"

#include "cortext/buffered_write_instruction.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/processor/operation_context.hpp"
#include <vector>

namespace cortext::operations
{

namespace
{

/// @brief Compute semantic drift between current and previous embedding.
inline double
ComputeSemanticDrift (const std::deque<Eigen::VectorXf> &embeddings)
{
  if (embeddings.size () < 2)
    return 0.0;

  const Eigen::VectorXf &current = embeddings.back ();
  const Eigen::VectorXf &prev = embeddings[embeddings.size () - 2];

  if (current.size () != prev.size () || current.size () == 0)
    return 0.0;

  // Cosine distance: 1 - cosine_similarity
  double similarity = core::CosineSimilarity (current, prev);
  return 1.0 - similarity;
}

} // namespace

void
RecordGenerationTrace::Execute (OperationContext &context) const
{
  auto &p_ctx = context.GetProcessorContext ();

  if (p_ctx.recent_context_embeddings.empty ())
    return;

  const Eigen::VectorXf &emb = p_ctx.recent_context_embeddings.back ();
  double drift = ComputeSemanticDrift (p_ctx.recent_context_embeddings);

  // Convert Eigen vector to std::vector<float> for BLOB storage
  std::vector<float> emb_blob (emb.data (), emb.data () + emb.size ());

  BufferedWriteInstruction op;
  op.query = "INSERT INTO generation_trace (embedding, timestamp, "
             "semantic_drift) VALUES (?, ?, ?)";
  op.params = { emb_blob,
                static_cast<long long> (context.GetSignal ().timestamp),
                drift };

  context.AddWriteInstruction (std::move (op));
}

} // namespace cortext::operations
