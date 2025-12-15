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
    op.query = "DELETE FROM memory_feedback WHERE strength < ?";
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
          BufferedWriteInstruction op;
          op.query = "INSERT INTO memory_feedback "
                     "(embedding_id, strength, retrieved_count, used_count, "
                     "contextual_gain, use_frequency, last_used, lability_state) "
                     "SELECT ?, 1.0, 0, 0, 0.0, 0.0, 0, 0.0 "
                     "WHERE NOT EXISTS (SELECT 1 FROM "
                     "memory_feedback WHERE "
                     "embedding_id = ?)";
          op.params = { id, id };
          context.AddWriteInstruction (std::move (op));
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
        BufferedWriteInstruction op;
        op.query
            = "UPDATE memory_feedback "
              "SET "
              "  retrieved_count = retrieved_count + 1, "
              "  used_count = used_count + ?, "
              "  contextual_gain = contextual_gain + ?, "
              "  use_frequency = (1.0 - ?) * use_frequency + ? * ?, "
              "  last_used = CASE WHEN ? > 0 THEN ? ELSE last_used END, "
              "  strength = MAX(0.0, "
              "    strength "
              "    + (? * ((1.0 - ?) * use_frequency + ? * ?)) " // reinforcement (S × EWMA)
              "    + (? * (? * (" // influence gate * F *
              "           (CASE WHEN ? < -1.0 THEN -1.0 "
              "                 WHEN ? >  1.0 THEN  1.0 "
              "                 ELSE ? END) "
              "           * (used_count * 1.0 / "
              "              CASE WHEN retrieved_count < 1 "
              "                   THEN 1 ELSE retrieved_count END)"
              "       ))) "
              "    - ?"
              "  ) "
              "WHERE embedding_id = ?";
        // Placeholders order:
        //  1: used_flag (used_count += ?)
        //  2: cg_event  (contextual_gain += ?)
        //  3: alpha
        //  4: alpha
        //  5: used_flag
        //  6: used_flag (for last_used gate)
        //  7: ts (last_used)
        //  8: S
        //  9: alpha
        //  10: alpha
        //  11: used_flag
        //  12: gate_influence
        //  13: F
        //  14: cg_event (clamp lower)
        //  15: cg_event (clamp upper)
        //  16: cg_event (original)
        //  17: lambda
        //  18: id
        op.params = { used_flag,      cg_event,  alpha,    alpha,
                      used_flag,      used_flag, ts,       S,
                      alpha,          alpha,     used_flag,
                      gate_influence, F,         cg_event, cg_event,
                      cg_event,       lambda,    id };
        context.AddWriteInstruction (std::move (op));
      }
    }
}

void
UpdateMemoryStrength::CollectSchema (cortext::store::SchemaRegistry &registry) const
{
  (void)registry; // Relies on core memory_feedback
}

} // namespace cortext::operations
