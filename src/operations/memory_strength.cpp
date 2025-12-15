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
    op.query = "DELETE FROM embeddings_meta WHERE strength < ?";
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
          // Insert into embeddings_meta for strength/frequency columns
          BufferedWriteInstruction op;
          op.query = "INSERT INTO embeddings_meta "
                     "(embedding_id, strength, contextual_gain, use_frequency, "
                     "lability_state) "
                     "SELECT ?, 1.0, 0.0, 0.0, 0.0 "
                     "WHERE NOT EXISTS (SELECT 1 FROM "
                     "embeddings_meta WHERE "
                     "embedding_id = ?)";
          op.params = { id, id };
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

        // Then, update embeddings_meta for strength/frequency columns
        // Use subquery to get retrieved_count and used_count from memory_feedback
        BufferedWriteInstruction op_em;
        op_em.query
            = "UPDATE embeddings_meta "
              "SET "
              "  contextual_gain = contextual_gain + ?, "
              "  use_frequency = (1.0 - ?) * use_frequency + ? * ?, "
              "  strength = MAX(0.0, "
              "    strength "
              "    + (? * ((1.0 - ?) * use_frequency + ? * ?)) " // reinforcement (S × EWMA)
              "    + (? * (? * ((" // influence gate * F * ((
              "           (CASE WHEN ? < -1.0 THEN -1.0 "
              "                 WHEN ? >  1.0 THEN  1.0 "
              "                 ELSE ? END) "
              "           * (COALESCE((SELECT used_count FROM memory_feedback WHERE embedding_id = ?), 0) * 1.0 / "
              "              CASE WHEN COALESCE((SELECT retrieved_count FROM memory_feedback WHERE embedding_id = ?), 1) < 1 "
              "                   THEN 1 ELSE COALESCE((SELECT retrieved_count FROM memory_feedback WHERE embedding_id = ?), 1) END)"
              "           + 1.0) / 2.0))) " // map01: (influence_factor + 1) / 2
              "    - ?"
              "  ) "
              "WHERE embedding_id = ?";
        // Placeholders order:
        //  1: cg_event  (contextual_gain += ?)
        //  2: alpha
        //  3: alpha
        //  4: used_flag
        //  5: S
        //  6: alpha
        //  7: alpha
        //  8: used_flag
        //  9: gate_influence
        //  10: F
        //  11: cg_event (clamp lower)
        //  12: cg_event (clamp upper)
        //  13: cg_event (original)
        //  14: id (for used_count subquery)
        //  15: id (for retrieved_count subquery check)
        //  16: id (for retrieved_count subquery value)
        //  17: lambda
        //  18: id (WHERE)
        op_em.params = { cg_event,       alpha,    alpha,    used_flag,
                         S,              alpha,    alpha,    used_flag,
                         gate_influence, F,        cg_event, cg_event,
                         cg_event,       id,       id,       id,
                         lambda,         id };
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
