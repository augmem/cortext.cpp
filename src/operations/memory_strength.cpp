#include "cortext/operations/memory_strength.hpp"

#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/schema.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace cortext::operations
{

void
UpdateMemoryStrength::Execute (OperationContext &context, Transaction &tx) const
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

  // Evict weak memories below periphery cutoff first
  auto eviction_result = tx.Execute ("DELETE FROM embeddings WHERE strength < ?", { cutoff });
  const int64_t eviction_count = eviction_result.size ();

  int64_t update_count = 0;

  const auto &events = context.GetMemoryUsageEvents ();
  for (const auto &e : events)
    {
      const double used_flag = e.used ? 1.0 : 0.0;
      const double cg_event
          = e.contextual_gain.value_or (constants::kNormalizedMin);
      const long long id = static_cast<long long> (e.embedding_id);

      // Ensure row exists for this embedding_id when feedback is present.
      if (e.used || e.contextual_gain.has_value ())
        {
          const long long ts
              = static_cast<long long> (context.GetSignal ().timestamp);

          // Ensure embeddings row has default values for strength/frequency columns
          tx.Execute ("UPDATE embeddings "
                      "SET strength = COALESCE(strength, 1.0), "
                      "    contextual_gain = COALESCE(contextual_gain, 0.0), "
                      "    use_frequency = COALESCE(use_frequency, 0.0), "
                      "    lability_state = COALESCE(lability_state, 0.0), "
                      "    last_access = COALESCE(last_access, ?) "
                      "WHERE embedding_id = ?",
                      { ts, id });

          // Insert into memory_feedback for count columns
          tx.Execute ("INSERT INTO memory_feedback "
                      "(embedding_id, retrieved_count, used_count, last_used) "
                      "SELECT ?, 0, 0, 0 "
                      "WHERE NOT EXISTS (SELECT 1 FROM "
                      "memory_feedback WHERE "
                      "embedding_id = ?)",
                      { id, id });
        }

      // Algorithm 14 + 18:
      const double gate_influence
          = (e.used || e.contextual_gain.has_value ()) ? 1.0 : 0.0;
      const long long ts
          = static_cast<long long> (context.GetSignal ().timestamp);

      // First, update memory_feedback for count columns
      tx.Execute ("UPDATE memory_feedback "
                  "SET "
                  "  retrieved_count = retrieved_count + 1, "
                  "  used_count = used_count + ?, "
                  "  last_used = CASE WHEN ? > 0 THEN ? ELSE last_used END "
                  "WHERE embedding_id = ?",
                  { used_flag, used_flag, ts, id });

      // Then, update embeddings for strength/frequency columns
      tx.Execute (
          "UPDATE embeddings "
          "SET "
          "  contextual_gain = contextual_gain + ?, "
          "  use_frequency = (1.0 - ?) * use_frequency + ? * ?, "
          "  strength = MAX(0.0, "
          "    strength * exp(-? * MAX(0.0, ? - COALESCE(last_access, ?)) / 1000.0) "
          "    + (? * ((1.0 - ?) * use_frequency + ? * ?)) "
          "    + (? * (? * (("
          "           (CASE WHEN ? < -1.0 THEN -1.0 "
          "                 WHEN ? >  1.0 THEN  1.0 "
          "                 ELSE ? END) "
          "           * (COALESCE((SELECT used_count FROM memory_feedback WHERE embedding_id = ?), 0) * 1.0 / "
          "              CASE WHEN COALESCE((SELECT retrieved_count FROM memory_feedback WHERE embedding_id = ?), 1) < 1 "
          "                   THEN 1 ELSE COALESCE((SELECT retrieved_count FROM memory_feedback WHERE embedding_id = ?), 1) END)"
          "           + 1.0) / 2.0))) "
          "  ), "
          "  last_access = ? "
          "WHERE embedding_id = ?",
          { cg_event,       alpha,    alpha,    used_flag, lambda,
            ts,             ts,       S,        alpha,     alpha,
            used_flag,      gate_influence, F,        cg_event, cg_event,
            cg_event,       id,       id,       id,        ts,
            id });

      ++update_count;
    }

  telemetry::LogDebug ("cortext.memory_strength",
                       { telemetry::Attribute::Int64 ("update_count",
                                                      update_count),
                         telemetry::Attribute::Int64 ("eviction_count",
                                                      eviction_count) });
}

void
UpdateMemoryStrength::CollectSchema (cortext::store::SchemaRegistry &registry) const
{
  (void)registry; // Relies on core memory_feedback
}

} // namespace cortext::operations
