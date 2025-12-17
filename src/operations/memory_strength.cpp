#include "cortext/operations/memory_strength.hpp"

#include "cortext/buffered_write_instruction.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/schema.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace cortext::operations
{

void
UpdateMemoryStrength::Execute (OperationContext &context) const
{
  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();

  const double F = cfg.focus;
  const double S = cfg.sensitivity;
  const double T = cfg.stability;
  const double alpha = core::AlphaS (S, p_ctx.u_t);
  const double half_life = core::BaseHalfLifePrior (T);
  const double lambda
      = std::log (constants::kTwo) / std::max (half_life, constants::kNormEpsilon);
  const double cutoff = core::PeripheryCutoff (T);

  // Evict weak memories below periphery cutoff first, so the last buffered
  // instruction after this operation reflects the latest reinforcement UPDATE.
  {
    BufferedWriteInstruction op;
    op.query = "DELETE FROM embeddings WHERE strength < ?";
    op.params = { cutoff };
    context.AddWriteInstruction (std::move (op));
  }

  const auto &events = context.GetMemoryUsageEvents ();
  for (const auto &e : events)
    {
      const double used_flag = e.used ? 1.0 : 0.0;
      const double cg_event
          = e.contextual_gain.value_or (constants::kNormalizedMin);
      const long long id = static_cast<long long> (e.embedding_id);

      // Ensure row exists for this embedding_id when feedback is present.
      // This avoids recreating rows for purely non-used events with no gain.
      if (e.used || e.contextual_gain.has_value ())
        {
          const long long ts
              = static_cast<long long> (context.GetSignal ().timestamp);

          // Ensure embeddings row has default values for strength/frequency columns
          // (embeddings table already exists from storage, update if needed)
          BufferedWriteInstruction op;
          op.query = "UPDATE embeddings "
                     "SET strength = COALESCE(strength, 1.0), "
                     "    contextual_gain = COALESCE(contextual_gain, 0.0), "
                     "    use_frequency = COALESCE(use_frequency, 0.0), "
                     "    lability_state = COALESCE(lability_state, 0.0), "
                     "    last_access = COALESCE(last_access, ?) "
                     "WHERE embedding_id = ?";
          op.params = { ts, id };
          context.AddWriteInstruction (std::move (op));

          // Insert into memory_feedback for count columns
          BufferedWriteInstruction op2;
          op2.query = "INSERT INTO memory_feedback "
                      "(embedding_id, retrieved_count, used_count, last_used) "
                      "SELECT ?, 0, 0, 0 "
                      "WHERE NOT EXISTS (SELECT 1 FROM "
                      "memory_feedback WHERE "
                      "embedding_id = ?)";
          op2.params = { id, id };
          context.AddWriteInstruction (std::move (op2));
        }

      // Algorithm 14 + 18:
      // - Increment retrieved/used counts and accumulate contextual_gain
      // - Update use_frequency via EWMA with α_S schedule and used_flag
      // - Apply reinforcement (Alg 14) and influence-weighted term (Alg 18)
      //   influence_factor = ( (used_count + used_flag) / max(1,
      //   retrieved_count + 1) )
      //                      * clamp(cg_event, -1, +1)
      //   influence_term = F * map01(influence_factor) = F *
      //   ((influence_factor + 1)/2)
      {
        // Apply influence term only when feedback is available (used or cg
        // provided)
        const double gate_influence
            = (e.used || e.contextual_gain.has_value ()) ? 1.0 : 0.0;
        const long long ts
            = static_cast<long long> (context.GetSignal ().timestamp);

        // First, update memory_feedback for count columns
        BufferedWriteInstruction op_mf;
        op_mf.query
            = "UPDATE memory_feedback "
              "SET "
              "  retrieved_count = retrieved_count + 1, "
              "  used_count = used_count + ?, "
              "  last_used = CASE WHEN ? > 0 THEN ? ELSE last_used END "
              "WHERE embedding_id = ?";
        op_mf.params = { used_flag, used_flag, ts, id };
        context.AddWriteInstruction (std::move (op_mf));

        // Then, update embeddings for strength/frequency columns
        // Use subquery to get retrieved_count and used_count from memory_feedback
        // Algorithm 14: Apply exponential decay with half-life semantics
        // strength_t = strength_{t-1} × exp(-λ × Δt) + S × use_frequency + F × influence
        BufferedWriteInstruction op_em;
        op_em.query
            = "UPDATE embeddings "
              "SET "
              "  contextual_gain = contextual_gain + ?, "
              "  use_frequency = (1.0 - ?) * use_frequency + ? * ?, "
              "  strength = MAX(0.0, "
              // Exponential decay: strength × exp(-λ × Δt) where Δt is seconds
              "    strength * exp(-? * MAX(0.0, ? - COALESCE(last_access, ?)) / 1000.0) "
              "    + (? * ((1.0 - ?) * use_frequency + ? * ?)) " // reinforcement (S × EWMA)
              "    + (? * (? * ((" // influence gate * F * ((
              "           (CASE WHEN ? < -1.0 THEN -1.0 "
              "                 WHEN ? >  1.0 THEN  1.0 "
              "                 ELSE ? END) "
              "           * (COALESCE((SELECT used_count FROM memory_feedback WHERE embedding_id = ?), 0) * 1.0 / "
              "              CASE WHEN COALESCE((SELECT retrieved_count FROM memory_feedback WHERE embedding_id = ?), 1) < 1 "
              "                   THEN 1 ELSE COALESCE((SELECT retrieved_count FROM memory_feedback WHERE embedding_id = ?), 1) END)"
              "           + 1.0) / 2.0))) " // map01: (influence_factor + 1) / 2
              "  ), "
              "  last_access = ? "
              "WHERE embedding_id = ?";
        // Placeholders order:
        //  1: cg_event  (contextual_gain += ?)
        //  2: alpha
        //  3: alpha
        //  4: used_flag
        //  5: lambda (for exp decay)
        //  6: ts (current timestamp)
        //  7: ts (fallback if last_access is NULL)
        //  8: S
        //  9: alpha
        //  10: alpha
        //  11: used_flag
        //  12: gate_influence
        //  13: F
        //  14: cg_event (clamp lower)
        //  15: cg_event (clamp upper)
        //  16: cg_event (original)
        //  17: id (for used_count subquery)
        //  18: id (for retrieved_count subquery check)
        //  19: id (for retrieved_count subquery value)
        //  20: ts (for updating last_access)
        //  21: id (WHERE)
        op_em.params = { cg_event, alpha,          alpha,    used_flag,
                         lambda,   ts,             ts,       S,
                         alpha,    alpha,          used_flag, gate_influence,
                         F,        cg_event,       cg_event, cg_event,
                         id,       id,             id,       ts,
                         id };
        context.AddWriteInstruction (std::move (op_em));
      }
    }
}

void
UpdateMemoryStrength::CollectSchema (cortext::store::SchemaRegistry &registry) const
{
  (void)registry; // Relies on core memory_feedback
}

} // namespace cortext::operations
