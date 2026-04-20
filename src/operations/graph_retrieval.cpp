#include "cortext/operations/graph_retrieval.hpp"

#include "constructive_recall_internal.hpp"
#include "../store/facts.hpp"
#include "retrieval_debug_state.hpp"
#include "temporal_retrieval.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/core/sparse.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cctype>
#include <cstdlib>
#include <sstream>

namespace cortext::operations
{

namespace
{
bool
EnvFlag (const char *name)
{
  const char *value = std::getenv (name);
  if (!value)
    {
      return false;
    }
  std::string s (value);
  std::transform (s.begin (), s.end (), s.begin (),
                  [] (unsigned char c) { return static_cast<char> (std::tolower (c)); });
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

long long
AnyToInt64 (const std::any &value)
{
  if (!value.has_value ())
    {
      return 0;
    }
  if (value.type () == typeid (long long))
    {
      return std::any_cast<long long> (value);
    }
  if (value.type () == typeid (int))
    {
      return static_cast<long long> (std::any_cast<int> (value));
    }
  return 0;
}

double
AnyToDouble (const std::any &value, double fallback = 0.0)
{
  if (!value.has_value ())
    {
      return fallback;
    }
  if (value.type () == typeid (double))
    {
      return std::any_cast<double> (value);
    }
  if (value.type () == typeid (float))
    {
      return static_cast<double> (std::any_cast<float> (value));
    }
  if (value.type () == typeid (long long))
    {
      return static_cast<double> (std::any_cast<long long> (value));
    }
  if (value.type () == typeid (int))
    {
      return static_cast<double> (std::any_cast<int> (value));
    }
  return fallback;
}

std::string
AnyToString (const std::any &value)
{
  if (!value.has_value () || value.type () != typeid (std::string))
    {
      return {};
    }
  return std::any_cast<std::string> (value);
}

std::optional<long long>
AnyToOptionalInt64 (const std::any &value)
{
  if (!value.has_value ())
    {
      return std::nullopt;
    }
  if (value.type () == typeid (long long))
    {
      return std::any_cast<long long> (value);
    }
  if (value.type () == typeid (int))
    {
      return static_cast<long long> (std::any_cast<int> (value));
  }
  return std::nullopt;
}

const char *
MetacognitiveModeLabel (ProcessorContext::MetacognitiveMode mode)
{
  switch (mode)
    {
    case ProcessorContext::MetacognitiveMode::Normal:
      return "normal";
    case ProcessorContext::MetacognitiveMode::TotRecovery:
      return "tot_recovery";
    case ProcessorContext::MetacognitiveMode::UnknownCaution:
      return "unknown_caution";
    }
  return "normal";
}

store::FactRecord
BuildFactRecord (const std::map<std::string, std::any> &row)
{
  store::FactRecord record;
  if (auto it = row.find ("fact_id"); it != row.end ())
    record.fact_id = AnyToInt64 (it->second);
  if (auto it = row.find ("subject"); it != row.end ())
    record.subject = AnyToString (it->second);
  if (auto it = row.find ("predicate"); it != row.end ())
    record.predicate = AnyToString (it->second);
  if (auto it = row.find ("object"); it != row.end ())
    record.object = AnyToString (it->second);
  if (auto it = row.find ("canonical_subject"); it != row.end ())
    record.canonical_subject = AnyToString (it->second);
  if (auto it = row.find ("canonical_predicate"); it != row.end ())
    record.canonical_predicate = AnyToString (it->second);
  if (auto it = row.find ("canonical_object"); it != row.end ())
    record.canonical_object = AnyToString (it->second);
  if (auto it = row.find ("valid_start_ts"); it != row.end ())
    record.valid_start_ts = AnyToOptionalInt64 (it->second);
  if (auto it = row.find ("valid_end_ts"); it != row.end ())
    record.valid_end_ts = AnyToOptionalInt64 (it->second);
  if (auto it = row.find ("recorded_at_ts"); it != row.end ())
    record.recorded_at_ts = AnyToInt64 (it->second);
  if (auto it = row.find ("superseded_at_ts"); it != row.end ())
    record.superseded_at_ts = AnyToOptionalInt64 (it->second);
  if (auto it = row.find ("confidence"); it != row.end ())
    record.confidence = AnyToDouble (it->second, 0.5);
  if (auto it = row.find ("summary_memory_id"); it != row.end ())
    record.summary_memory_id = AnyToInt64 (it->second);
  if (auto it = row.find ("embedding_id"); it != row.end ())
    record.embedding_id = AnyToInt64 (it->second);
  if (auto it = row.find ("fact_text"); it != row.end ())
    record.fact_text = AnyToString (it->second);
  if (auto it = row.find ("is_current"); it != row.end ())
    record.is_current = AnyToInt64 (it->second) != 0;
  if (auto it = row.find ("evidence_count"); it != row.end ())
    record.evidence_count = static_cast<int> (AnyToInt64 (it->second));
  if (auto it = row.find ("support_mass"); it != row.end ())
    record.support_mass = AnyToDouble (it->second, 0.0);
  if (auto it = row.find ("source_diversity"); it != row.end ())
    record.source_diversity = static_cast<int> (AnyToInt64 (it->second));
  if (auto it = row.find ("contradiction_mass"); it != row.end ())
    record.contradiction_mass = AnyToDouble (it->second, 0.0);
  if (auto it = row.find ("confirmation_count"); it != row.end ())
    record.confirmation_count = static_cast<int> (AnyToInt64 (it->second));
  if (auto it = row.find ("challenge_count"); it != row.end ())
    record.challenge_count = static_cast<int> (AnyToInt64 (it->second));
  if (auto it = row.find ("compressed_support_count"); it != row.end ())
    record.compressed_support_count = static_cast<int> (AnyToInt64 (it->second));
  if (auto it = row.find ("last_confirmation_ts"); it != row.end ())
    record.last_confirmation_ts = AnyToInt64 (it->second);
  if (auto it = row.find ("last_challenge_ts"); it != row.end ())
    record.last_challenge_ts = AnyToOptionalInt64 (it->second);
  if (auto it = row.find ("severity_class"); it != row.end ())
    record.severity_class = AnyToString (it->second);
  if (auto it = row.find ("lifecycle_state"); it != row.end ())
    record.lifecycle_state
        = store::ParseFactLifecycleState (AnyToString (it->second));
  if (auto it = row.find ("archived_at"); it != row.end ())
    record.archived_at = AnyToOptionalInt64 (it->second);
  if (auto it = row.find ("last_maintenance_ts"); it != row.end ())
    record.last_maintenance_ts = AnyToInt64 (it->second);
  return record;
}

std::string
FactPairKey (const std::string &canonical_subject,
             const std::string &canonical_predicate)
{
  return canonical_subject + "|" + canonical_predicate;
}

double
PredicateCriticality (const std::string &canonical_predicate)
{
  const auto contains = [&canonical_predicate] (const char *needle) {
    return canonical_predicate.find (needle) != std::string::npos;
  };
  if (contains ("medication") || contains ("caregiver") || contains ("location")
      || contains ("schedule") || contains ("appointment")
      || contains ("safety") || contains ("destination")
      || contains ("pickup") || contains ("lives_in"))
    {
      return 1.0;
    }
  if (contains ("routine") || contains ("household")
      || contains ("home") || contains ("works_at"))
    {
      return 0.75;
    }
  if (contains ("prefer") || contains ("likes") || contains ("favorite"))
    {
      return 0.45;
    }
  return 0.60;
}

double
PredicateRoutineAffinity (const std::string &canonical_predicate)
{
  const auto contains = [&canonical_predicate] (const char *needle) {
    return canonical_predicate.find (needle) != std::string::npos;
  };
  if (contains ("schedule") || contains ("routine") || contains ("home")
      || contains ("household") || contains ("works_at"))
    {
      return 1.0;
    }
  if (contains ("prefer") || contains ("likes") || contains ("favorite"))
    {
      return 0.8;
    }
  if (contains ("medication") || contains ("caregiver")
      || contains ("lives_in") || contains ("location"))
    {
      return 0.35;
    }
  return 0.15;
}

double
EvidenceWeight (const std::string &evidence_type, double support_weight)
{
  double type_weight = 0.75;
  if (evidence_type == "episodic")
    {
      type_weight = 1.0;
    }
  else if (evidence_type == "summary")
    {
      type_weight = 0.88;
    }
  return core::Clamp (type_weight * core::Clamp (support_weight, 0.25, 1.0),
                      0.0, 1.0);
}

struct LinkedFactEvidence
{
  store::FactRecord record;
  store::FactScore fact_score;
  double query_similarity = 0.0;
  std::string evidence_type;
  double support_weight = 1.0;
  bool stale = false;
};

struct CandidateFactScore
{
  double boost = 0.0;
  double stale_penalty = 0.0;
  int linked_fact_count = 0;
};

double
FactBoostMultiplier (const temporal::RetrievalAblationOverride &override)
{
  if (!override.fact_boost_strength.has_value ())
    {
      return 1.0;
    }
  switch (*override.fact_boost_strength)
    {
    case temporal::FactBoostStrength::Off:
      return 0.0;
    case temporal::FactBoostStrength::Weak:
      return 0.55;
    case temporal::FactBoostStrength::Strong:
      return 1.45;
    }
  return 1.0;
}

double
StalePenaltyMultiplier (const temporal::RetrievalAblationOverride &override)
{
  if (!override.stale_penalty_strength.has_value ())
    {
      return 1.0;
    }
  switch (*override.stale_penalty_strength)
    {
    case temporal::StalePenaltyStrength::Off:
      return 0.0;
    case temporal::StalePenaltyStrength::Moderate:
      return 1.0;
    case temporal::StalePenaltyStrength::Strong:
      return 1.75;
    }
  return 1.0;
}

temporal::ProvenanceMode
ResolveProvenanceMode (const temporal::RetrievalAblationOverride &override)
{
  return override.provenance_mode.value_or (
      temporal::ProvenanceMode::DirectLinkOnly);
}

const char *
FactBoostModeLabel (const temporal::RetrievalAblationOverride &override)
{
  return override.fact_boost_strength.has_value ()
             ? temporal::ToString (*override.fact_boost_strength)
             : "default";
}

const char *
StalePenaltyModeLabel (const temporal::RetrievalAblationOverride &override)
{
  return override.stale_penalty_strength.has_value ()
             ? temporal::ToString (*override.stale_penalty_strength)
             : "default";
}

temporal::RoutineRecencyMode
ResolveRoutineRecencyMode (const temporal::RetrievalAblationOverride &override)
{
  return override.routine_recency_mode.value_or (
      temporal::RoutineRecencyMode::Balanced);
}

const char *
RoutineRecencyModeLabel (const temporal::RetrievalAblationOverride &override)
{
  return override.routine_recency_mode.has_value ()
             ? temporal::ToString (*override.routine_recency_mode)
             : "default";
}

double
RoutineRecencyBoostAdjustment (const store::FactRecord &record,
                               const store::FactScore &score,
                               temporal::RetrievalMode mode,
                               const temporal::RetrievalAblationOverride &override)
{
  if (mode != temporal::RetrievalMode::Current)
    {
      return 1.0;
    }

  const double routine_affinity
      = PredicateRoutineAffinity (record.canonical_predicate);
  switch (ResolveRoutineRecencyMode (override))
    {
    case temporal::RoutineRecencyMode::Balanced:
      return 1.0;
    case temporal::RoutineRecencyMode::RoutineBiased:
      return core::Clamp (
          1.0 + 0.40 * routine_affinity * score.routine_support
              - 0.16 * score.recency_support,
          0.70, 1.45);
    case temporal::RoutineRecencyMode::RecencyBiased:
      return core::Clamp (
          1.0 + 0.46 * score.recency_support
              - 0.18 * routine_affinity * score.routine_support,
          0.70, 1.55);
    }
  return 1.0;
}

double
RoutineRecencyStaleAdjustment (
    const store::FactRecord &record, const store::FactScore &score,
    temporal::RetrievalMode mode,
    const temporal::RetrievalAblationOverride &override)
{
  if (mode != temporal::RetrievalMode::Current)
    {
      return 1.0;
    }

  const double routine_affinity
      = PredicateRoutineAffinity (record.canonical_predicate);
  switch (ResolveRoutineRecencyMode (override))
    {
    case temporal::RoutineRecencyMode::Balanced:
      return 1.0;
    case temporal::RoutineRecencyMode::RoutineBiased:
      return core::Clamp (
          1.0 - 0.65 * routine_affinity * score.routine_support, 0.35, 1.05);
    case temporal::RoutineRecencyMode::RecencyBiased:
      return core::Clamp (
          1.0 + 0.45 * std::max (score.recency_support, 0.35)
              + 0.24 * routine_affinity,
          0.95, 1.75);
    }
  return 1.0;
}

CandidateFactScore
ScoreCandidateFacts (const std::vector<LinkedFactEvidence> &links,
                     double focus, double sensitivity, double stability,
                     temporal::RetrievalMode mode,
                     const temporal::RetrievalAblationOverride &override)
{
  CandidateFactScore out;
  if (links.empty ())
    {
      return out;
    }

  const double f_eff = core::FocusBias (focus);
  const double s_eff = core::SensitivityBias (sensitivity);
  const double t_eff = core::Clamp (stability, 0.0, 1.0);
  const double fact_weight = core::Lerp (0.08, 0.28, f_eff)
                             * core::Lerp (0.90, 1.10, s_eff)
                             * core::Lerp (1.05, 0.95, t_eff)
                             * FactBoostMultiplier (override);
  const double stale_weight = core::Lerp (0.05, 0.20, f_eff)
                              * core::Lerp (0.90, 1.05, s_eff)
                              * StalePenaltyMultiplier (override);

  for (const auto &link : links)
    {
      const double similarity = core::Clamp (link.query_similarity, 0.0, 1.0);
      const double confidence
          = core::Clamp (link.record.confidence, 0.0, 1.0);
      const double provenance
          = EvidenceWeight (link.evidence_type, link.support_weight);
      const double criticality
          = PredicateCriticality (link.record.canonical_predicate);
      const double lifecycle
          = core::Clamp (link.fact_score.lifecycle_support, 0.0, 1.0);

      if (link.stale)
        {
          if (mode == temporal::RetrievalMode::Current)
            {
              const double stale_adjust = RoutineRecencyStaleAdjustment (
                  link.record, link.fact_score, mode, override);
              const double penalty
                  = stale_weight
                    * (0.60 * similarity + 0.40 * confidence)
                    * criticality * provenance
                    * std::max (0.55, lifecycle) * stale_adjust;
              out.stale_penalty = std::max (out.stale_penalty, penalty);
            }
          continue;
        }

      out.linked_fact_count++;
      const double support = core::Clamp (link.fact_score.evidence_support, 0.0,
                                          1.0);
      const double temporal_match
          = core::Clamp (link.fact_score.temporal_match, 0.0, 1.0);
      const double routine_recency_adjust = RoutineRecencyBoostAdjustment (
          link.record, link.fact_score, mode, override);
      const double signal = (0.45 * similarity + 0.35 * confidence
                             + 0.20 * support)
                            * (0.55 + 0.45 * temporal_match)
                            * criticality * provenance * lifecycle
                            * routine_recency_adjust;
      out.boost = std::max (out.boost, fact_weight * signal);
    }

  return out;
}
} // namespace

void
GraphAugmentedRetrieveCandidates::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  retrieval_debug::ClearLastSelectedEmbeddingOrder ();
  retrieval_debug::ClearLastRankedCandidates ();
  auto elapsed_ms = [] (const std::chrono::steady_clock::time_point &start,
                        const std::chrono::steady_clock::time_point &end) {
    return std::chrono::duration_cast<
        std::chrono::duration<double, std::milli> > (end - start)
        .count ();
  };

  // Check streaming pacing gate - skip retrieval if not triggered
  if (!context.GetShouldCheckRetrieval ())
    {
      return;
    }

  Store *store = context.GetStore ();
  if (!store)
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &signal = context.GetSignal ();

  if (p_ctx.memory_stream.empty ())
    {
      return;
    }

  // Section 7.6: Use the current accumulator centroid (μ_acc) as query.
  // Cache prior to any accumulator reset to keep retrieval grounded in the
  // current unit's evolving context.
  auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
  if (acc_it == p_ctx.accumulator_states.end ()
      || acc_it->second.mu_acc.size () == 0)
    {
      return;
    }
  Eigen::VectorXf q = acc_it->second.mu_acc;
  Eigen::VectorXf q_ctx
      = (acc_it->second.c_t.size () > 0) ? acc_it->second.c_t : q;

  const double ctx_mix = core::RetrievalContextMix (cfg.focus);
  if (ctx_mix > 0.0 && !p_ctx.recent_context_embeddings.empty ()
      && q.size () > 0)
    {
      Eigen::VectorXf mean_ctx = Eigen::VectorXf::Zero (q.size ());
      int count = 0;
      for (const auto &v : p_ctx.recent_context_embeddings)
        {
          if (v.size () == q.size ())
            {
              mean_ctx += v;
              ++count;
            }
        }
      if (count > 0)
        {
          mean_ctx /= static_cast<float> (count);
          q = q * static_cast<float> (1.0 - ctx_mix)
              + mean_ctx * static_cast<float> (ctx_mix);
        }
    }

  if (q.size () == 0)
    {
      return;
    }
  const float q_norm = q.norm ();
  if (q_norm > 0.0f)
    {
      q /= q_norm;
    }
  if (q_ctx.size () > 0)
    {
      const float ctx_norm = q_ctx.norm ();
      if (ctx_norm > 1e-9f)
        {
          q_ctx /= ctx_norm;
        }
    }

  const double f_eff = core::FocusBias (cfg.focus);
  const double s_eff = core::SensitivityBias (cfg.sensitivity);
  const int base_k = std::max (1, core::MaxResults (cfg.focus));
  const int base_depth = std::max (1, core::GraphDepth (cfg.stability));
  const double min_edge_weight = core::MinEdgeWeight (cfg.focus);
  const int k_key = core::SparseKeySize (cfg.focus);
  const std::string sparse_key = core::SparseKey (q, k_key);
  const double s_eff_seed = s_eff;
  const auto retrieval_override = temporal::GetRetrievalOverride ();
  const auto requested_retrieval_mode
      = retrieval_override.mode.value_or (temporal::RetrievalMode::Current);
  const auto ablation_override = temporal::GetRetrievalAblationOverride ();
  const auto retrieval_mode
      = temporal::ResolveRetrievalMode (requested_retrieval_mode);
  const std::uint64_t retrieval_ts
      = temporal::ResolveRetrievalTimestamp (requested_retrieval_mode,
                                            retrieval_override.timestamp,
                                            signal.timestamp);
  const auto fact_query_mode = temporal::ResolveFactQueryMode (
      temporal::ToFactQueryMode (retrieval_mode));
  const bool fact_layer_enabled = temporal::IsFactLayerEnabled ();
  const bool disable_source_conf = EnvFlag ("CORTEXT_DISABLE_SOURCE_CONF");
  const bool disable_predictive_bonus
      = EnvFlag ("CORTEXT_DISABLE_PREDICTIVE_RETRIEVAL_BONUS");
  const bool disable_procedural_proactive
      = EnvFlag ("CORTEXT_DISABLE_PROCEDURAL_PROACTIVE_RETRIEVAL");
  const bool disable_tot_recovery
      = EnvFlag ("CORTEXT_DISABLE_METACOG_TOT_RECOVERY");
  const bool disable_unknown_caution
      = EnvFlag ("CORTEXT_DISABLE_METACOG_UNKNOWN_CAUTION");
  const bool disable_constructive_recall
      = constructive_recall::Disabled ();

  if (p_ctx.metacognitive_mode_expires_at > 0
      && signal.timestamp >= p_ctx.metacognitive_mode_expires_at)
    {
      p_ctx.metacognitive_mode = ProcessorContext::MetacognitiveMode::Normal;
      p_ctx.metacognitive_mode_expires_at = 0;
      p_ctx.metacognitive_certainty_satisfied = false;
    }

  auto metacognitive_mode = p_ctx.metacognitive_mode;
  const double metacognitive_confidence
      = core::Clamp01 (p_ctx.metacognitive_confidence);
  const double metacognitive_fok_threshold
      = core::FOKThreshold (cfg.focus);
  const double tot_confidence_scale = core::Clamp (
      (metacognitive_confidence - metacognitive_fok_threshold)
          / std::max (constants::kNormEpsilon,
                      1.0 - metacognitive_fok_threshold),
      0.0, 1.0);
  const double unknown_caution_scale = 1.0 - metacognitive_confidence;
  if (metacognitive_mode == ProcessorContext::MetacognitiveMode::TotRecovery
      && disable_tot_recovery)
    {
      metacognitive_mode = ProcessorContext::MetacognitiveMode::Normal;
    }
  if (metacognitive_mode
          == ProcessorContext::MetacognitiveMode::UnknownCaution
      && disable_unknown_caution)
    {
      metacognitive_mode = ProcessorContext::MetacognitiveMode::Normal;
    }

  int k = base_k;
  int depth = base_depth;
  if (metacognitive_mode == ProcessorContext::MetacognitiveMode::TotRecovery)
    {
      k = std::max (
          base_k,
          static_cast<int> (
              std::ceil ((1.15 + 0.35 * tot_confidence_scale) * base_k)));
      depth = std::min (3, base_depth + (tot_confidence_scale >= 0.2 ? 1 : 0));
    }
  const double predictive_weight
      = disable_predictive_bonus
            ? 0.0
            : core::Lerp (0.05, 0.20, f_eff)
                  * core::Lerp (1.0, 0.85, cfg.stability);

  // Seed vector retrieval via sqlite-vec KNN query.
  struct Scored
  {
    long long embedding_id;
    long long memory_id;
    long long created_at;
    double score;
    double ctx_score;
    double source_confidence;
    double proc_score;
    double pre_activation;
    double predictive_bonus;
    int source_contradictions;
    double emotion_intensity;
    double arousal_avg;
    double fact_boost;
    double fact_stale_penalty;
    int linked_fact_count;
    bool is_association;
    bool is_label;
    Eigen::VectorXf vec;
    Eigen::VectorXf ctx;
  };
  std::vector<Scored> seeds;
  seeds.reserve (static_cast<size_t> (k));
  int64_t procedural_seed_count = 0;

  const std::string latest_reconstruction_join
      = disable_constructive_recall
            ? ""
            : "LEFT JOIN ("
              "  SELECT mr.memory_id, mr.embedding_id, mr.created_at, mr.uncertainty "
              "  FROM memory_reconstructions mr "
              "  JOIN ("
              "    SELECT memory_id, MAX(reconstruction_id) AS reconstruction_id "
              "    FROM memory_reconstructions "
              "    GROUP BY memory_id"
              "  ) latest ON latest.reconstruction_id = mr.reconstruction_id"
              ") rc ON rc.memory_id = m.memory_id ";
  const char *current_embedding_expr
      = disable_constructive_recall ? "m.embedding_id"
                                    : "COALESCE(rc.embedding_id, m.embedding_id)";

  // Convert query vector to std::vector<float> for parameter binding.
  std::vector<float> q_vec (q.data (), q.data () + q.size ());

  std::unordered_set<long long> seen_seed_embeddings;

  auto append_seeds = [&] (const std::string &sql,
                           const std::vector<std::any> &params,
                           const char *timing_key) {
    const auto t_start = std::chrono::steady_clock::now ();
    auto rows = store->Execute (sql, params);
    const auto t_end = std::chrono::steady_clock::now ();
    if (timing_key)
      {
        context.AddOperationTiming (timing_key, elapsed_ms (t_start, t_end));
      }
    for (const auto &row : rows)
      {
        auto it_emb_id = row.find ("base_embedding_id");
        if (it_emb_id == row.end ())
          {
            it_emb_id = row.find ("embedding_id");
          }
        auto it_dist = row.find ("distance");
        auto it_created = row.find ("created_at");
        auto it_mem_id = row.find ("memory_id");
        auto it_kind = row.find ("kind");
        auto it_preact = row.find ("pre_activation");
        if (it_emb_id == row.end ())
          continue;
        if (it_emb_id->second.type () != typeid (long long))
          continue;

        const long long emb_id = std::any_cast<long long> (it_emb_id->second);
        if (!seen_seed_embeddings.insert (emb_id).second)
          {
            continue;
          }

        // Prefer cosine similarity for scoring to align with manuscript logic.
        double sim = 0.0;
        if (it_dist != row.end ()
                 && it_dist->second.type () == typeid (double))
          {
            const double dist = std::any_cast<double> (it_dist->second);
            sim = 1.0 / (1.0 + dist);
          }

        long long mem_id = 0;
        if (it_mem_id != row.end ())
          {
            mem_id = store::AnyToLongLong (it_mem_id->second).value_or (0);
          }
        if (mem_id == 0)
          {
            auto sig_rows = store->Execute (
                "SELECT memory_id FROM signals WHERE embedding_id = ? LIMIT 1",
                { emb_id });
            if (!sig_rows.empty ())
              {
                auto it_sig = sig_rows[0].find ("memory_id");
                if (it_sig != sig_rows[0].end ())
                  {
                    mem_id = store::AnyToLongLong (it_sig->second).value_or (0);
                  }
              }
          }

        long long created_at = 0;
        if (it_created != row.end ()
            && it_created->second.type () == typeid (long long))
          {
            created_at = std::any_cast<long long> (it_created->second);
          }

        bool is_association = false;
        bool is_label = false;
        if (it_kind != row.end () && it_kind->second.type () == typeid (std::string))
          {
            const std::string kind
                = std::any_cast<std::string> (it_kind->second);
            is_association = (kind == "ASSOCIATION");
            is_label = (kind == "LABEL");
          }

        const double pre_activation
            = (it_preact != row.end ()) ? AnyToDouble (it_preact->second, 0.0) : 0.0;

        seeds.push_back (Scored{ emb_id,
                                 mem_id,
                                 created_at,
                                 sim,
                                 0.0,
                                 1.0,
                                 0.0,
                                 pre_activation,
                                 predictive_weight * pre_activation,
                                 0,
                                 0.0,
                                 0.0,
                                 0.0,
                                 0.0,
                                 0,
                                 is_association,
                                 is_label,
                                 Eigen::VectorXf (),
                                 Eigen::VectorXf () });
      }
  };

  const std::string seed_sql
      = std::string (
            "SELECT m.embedding_id AS base_embedding_id, e.distance, "
            "COALESCE(m.created_at, e.created_at, 0) AS created_at, "
            "m.memory_id, m.kind, COALESCE(m.pre_activation, 0.0) AS pre_activation "
            "FROM memories m ")
        + latest_reconstruction_join
        + "JOIN embeddings e ON e.embedding_id = "
        + current_embedding_expr
        + " "
          "WHERE e.embedding MATCH ? AND k = ? "
          "AND m.kind != 'WORKING' "
          "AND (m.kind != 'ASSOCIATION' OR EXISTS ("
          "  SELECT 1 FROM associations a "
          "  WHERE a.source_memory_id = m.memory_id "
          "    AND a.edge_type = 'derived_from'))";
  append_seeds (seed_sql, { q_vec, static_cast<long long> (k) },
                "GraphRetrieve.seed_sql");

  const double k_summary_raw
      = core::Lerp (2.0, 6.0, s_eff_seed) * core::Lerp (1.0, 0.75, f_eff)
        * core::Lerp (1.0, 0.85, cfg.stability);
  const int k_summary
      = std::max (1, static_cast<int> (std::round (k_summary_raw)));
  const auto &summary_cache = p_ctx.summary_cache;
  if (k_summary > 0 && !summary_cache.empty ())
    {
      const auto t_start = std::chrono::steady_clock::now ();
      std::vector<std::pair<double, size_t>> ranked;
      ranked.reserve (summary_cache.size ());
      for (size_t i = 0; i < summary_cache.size (); ++i)
        {
          const auto &entry = summary_cache[i];
          if (entry.embedding.size () != q.size ())
            {
              continue;
            }
          if (entry.embedding_norm <= 1e-9f)
            {
              continue;
            }
          const double sim
              = static_cast<double> (entry.embedding.dot (q))
                / static_cast<double> (entry.embedding_norm);
          ranked.emplace_back (sim, i);
        }
      const int take
          = std::min (k_summary, static_cast<int> (ranked.size ()));
      if (take > 0)
        {
          auto mid = ranked.begin () + take;
          std::nth_element (
              ranked.begin (), mid, ranked.end (),
              [] (const auto &a, const auto &b) { return a.first > b.first; });
          std::sort (ranked.begin (), mid,
                     [] (const auto &a, const auto &b) { return a.first > b.first; });
          for (int i = 0; i < take; ++i)
            {
              const auto &entry = summary_cache[ranked[i].second];
              if (!seen_seed_embeddings.insert (entry.embedding_id).second)
                {
                  continue;
                }
              seeds.push_back (Scored{
                  entry.embedding_id,
                  entry.memory_id,
                  0,
                  ranked[i].first,
                  0.0,
                  1.0,
                  0.0,
                  0.0,
                  0.0,
                  0,
                  0.0,
                  0.0,
                  0.0,
                  0.0,
                  0,
                  entry.is_association,
                  entry.is_label,
                  entry.embedding,
                  Eigen::VectorXf () });
            }
        }
      const auto t_end = std::chrono::steady_clock::now ();
      context.AddOperationTiming ("GraphRetrieve.summary_cache",
                                  elapsed_ms (t_start, t_end));
    }

  if (cfg.procedural_enabled && !disable_procedural_proactive
      && !sparse_key.empty ())
    {
      auto pit = p_ctx.procedural_store.find (sparse_key);
      if (pit != p_ctx.procedural_store.end () && !pit->second.empty ())
        {
          std::vector<std::pair<double, long long>> ranked_proc;
          ranked_proc.reserve (pit->second.size ());
          constexpr double kProceduralSeedMin = 0.55;
          for (const auto &[memory_id, score] : pit->second)
            {
              if (memory_id <= 0)
                {
                  continue;
                }
              const double clamped = core::Clamp (score, 0.0, 1.0);
              if (clamped < kProceduralSeedMin)
                {
                  continue;
                }
              ranked_proc.emplace_back (clamped, memory_id);
            }
          if (!ranked_proc.empty ())
            {
              std::sort (ranked_proc.begin (), ranked_proc.end (),
                         [] (const auto &a, const auto &b) {
                           return a.first > b.first;
                         });
              const int k_proc
                  = std::max (
                      1, static_cast<int> (std::round (
                             core::Lerp (1.0, 4.0, s_eff)
                             * core::Lerp (1.0, 0.75, cfg.stability))));
              if (static_cast<int> (ranked_proc.size ()) > k_proc)
                {
                  ranked_proc.resize (static_cast<size_t> (k_proc));
                }

              std::ostringstream sql;
              sql << "SELECT m.memory_id, "
                     << current_embedding_expr
                     << " AS embedding_id, "
                        "COALESCE(m.created_at, 0) AS created_at, "
                        "m.kind, COALESCE(m.pre_activation, 0.0) AS pre_activation "
                        "FROM memories m "
                  << latest_reconstruction_join
                  << "WHERE m.memory_id IN (";
              std::vector<std::any> params;
              params.reserve (ranked_proc.size ());
              bool first = true;
              for (const auto &[score, memory_id] : ranked_proc)
                {
                  (void)score;
                  if (!first)
                    {
                      sql << ",";
                    }
                  first = false;
                  sql << "?";
                  params.push_back (memory_id);
                }
              sql << ")";

              const auto t_start = std::chrono::steady_clock::now ();
              auto rows = store->Execute (sql.str (), params);
              const auto t_end = std::chrono::steady_clock::now ();
              context.AddOperationTiming ("GraphRetrieve.procedural_seed_sql",
                                          elapsed_ms (t_start, t_end));

              std::unordered_map<long long, double> proc_scores;
              proc_scores.reserve (ranked_proc.size ());
              for (const auto &[score, memory_id] : ranked_proc)
                {
                  proc_scores.emplace (memory_id, score);
                }

              for (const auto &row : rows)
                {
                  const long long mem_id = AnyToInt64 (row.at ("memory_id"));
                  const long long emb_id = AnyToInt64 (row.at ("embedding_id"));
                  if (mem_id <= 0 || emb_id <= 0
                      || !seen_seed_embeddings.insert (emb_id).second)
                    {
                      continue;
                    }
                  const long long created_at = AnyToInt64 (row.at ("created_at"));
                  const std::string kind = AnyToString (row.at ("kind"));
                  const bool is_association = (kind == "ASSOCIATION");
                  const bool is_label = (kind == "LABEL");
                  const double pre_activation = core::Clamp (
                      AnyToDouble (row.at ("pre_activation"), 0.0), 0.0, 1.0);
                  const double proc_score
                      = proc_scores.count (mem_id) == 1 ? proc_scores[mem_id] : 0.0;

                  seeds.push_back (Scored{ emb_id,
                                           mem_id,
                                           created_at,
                                           0.0,
                                           0.0,
                                           1.0,
                                           proc_score,
                                           pre_activation,
                                           predictive_weight * pre_activation,
                                           0,
                                           0.0,
                                           0.0,
                                           0.0,
                                           0.0,
                                           0,
                                           is_association,
                                           is_label,
                                           Eigen::VectorXf (),
                                           Eigen::VectorXf () });
                  ++procedural_seed_count;
                }
            }
        }
    }

  struct MatchedFact
  {
    store::FactRecord record;
    store::FactScore score;
    double query_similarity = 0.0;
  };
  std::vector<MatchedFact> matched_facts;
  std::unordered_map<long long, std::vector<LinkedFactEvidence>>
      candidate_fact_links;
  std::unordered_map<std::string, double> current_pair_similarity;
  std::unordered_map<std::string, std::string> current_pair_object;

  const int k_fact = std::max (
      2, static_cast<int> (std::round (core::Lerp (12.0, 3.0, f_eff)
                                       * core::Lerp (1.0, 0.85,
                                                     cfg.stability))));
  const int k_fact_search
      = std::max (32, std::min (256, k_fact * 12));
  {
    const auto t_start = std::chrono::steady_clock::now ();
    std::ostringstream sql;
    sql << "SELECT fc.fact_id, fc.embedding_id, e.distance, "
           "fa.subject, fa.predicate, fa.object, "
           "fa.canonical_subject, fa.canonical_predicate, fa.canonical_object, "
           "fa.valid_start_ts, fa.valid_end_ts, fa.recorded_at_ts, "
           "fa.superseded_at_ts, fa.confidence, fa.summary_memory_id, "
           "COALESCE(fc.fact_text, '') AS fact_text, "
           "COALESCE(fc.is_current, 0) AS is_current, "
           "COALESCE(fa.support_mass, 0.0) AS support_mass, "
           "COALESCE(fa.source_diversity, 0) AS source_diversity, "
           "COALESCE(fa.contradiction_mass, 0.0) AS contradiction_mass, "
           "COALESCE(fa.confirmation_count, 0) AS confirmation_count, "
           "COALESCE(fa.challenge_count, 0) AS challenge_count, "
           "COALESCE(fa.compressed_support_count, 0) AS compressed_support_count, "
           "COALESCE(fa.last_confirmation_ts, 0) AS last_confirmation_ts, "
           "fa.last_challenge_ts, "
           "COALESCE(fa.severity_class, 'medium') AS severity_class, "
           "COALESCE(fa.lifecycle_state, 'active') AS lifecycle_state, "
           "fa.archived_at, "
           "COALESCE(fa.last_maintenance_ts, 0) AS last_maintenance_ts, "
           "(SELECT COUNT(*) FROM fact_evidence fe "
           " LEFT JOIN memories m ON m.memory_id = fe.source_memory_id "
           " WHERE fe.fact_id = fa.fact_id AND m.memory_id IS NOT NULL) "
           " AS evidence_count "
           "FROM embeddings e "
           "JOIN fact_cache fc ON fc.embedding_id = e.embedding_id "
           "JOIN fact_assertions fa ON fa.fact_id = fc.fact_id "
           "WHERE e.embedding MATCH ? AND k = ? ";
    switch (retrieval_mode)
      {
      case temporal::RetrievalMode::Current:
        sql << "AND COALESCE(fa.lifecycle_state, 'active') != 'archived' "
               "AND (fa.valid_start_ts IS NULL OR fa.valid_start_ts <= ?) "
               "AND (fa.valid_end_ts IS NULL OR fa.valid_end_ts > ?) "
               "AND fa.recorded_at_ts <= ? "
               "AND (fa.superseded_at_ts IS NULL OR fa.superseded_at_ts > ?) ";
        break;
      case temporal::RetrievalMode::ValidAt:
        sql << "AND (fa.valid_start_ts IS NULL OR fa.valid_start_ts <= ?) "
               "AND (fa.valid_end_ts IS NULL OR fa.valid_end_ts > ?) ";
        break;
      case temporal::RetrievalMode::KnownAt:
        sql << "AND fa.recorded_at_ts <= ? "
               "AND (fa.superseded_at_ts IS NULL OR fa.superseded_at_ts > ?) ";
        break;
      }
    sql << "ORDER BY e.distance ASC";

    std::vector<std::any> params
        = { q_vec, static_cast<long long> (k_fact_search) };
    switch (retrieval_mode)
      {
      case temporal::RetrievalMode::Current:
        params.push_back (static_cast<long long> (retrieval_ts));
        params.push_back (static_cast<long long> (retrieval_ts));
        params.push_back (static_cast<long long> (retrieval_ts));
        params.push_back (static_cast<long long> (retrieval_ts));
        break;
      case temporal::RetrievalMode::ValidAt:
        params.push_back (static_cast<long long> (retrieval_ts));
        params.push_back (static_cast<long long> (retrieval_ts));
        break;
      case temporal::RetrievalMode::KnownAt:
        params.push_back (static_cast<long long> (retrieval_ts));
        params.push_back (static_cast<long long> (retrieval_ts));
        break;
      }

    auto rows = store->Execute (sql.str (), params);
    const auto t_end = std::chrono::steady_clock::now ();
    context.AddOperationTiming ("GraphRetrieve.fact_seed_sql",
                                elapsed_ms (t_start, t_end));

    std::unordered_set<long long> seen_fact_ids;
    for (const auto &row : rows)
      {
        const store::FactRecord record = BuildFactRecord (row);
        if (record.fact_id <= 0 || !seen_fact_ids.insert (record.fact_id).second)
          {
            continue;
          }
        const auto it_dist = row.find ("distance");
        const double query_similarity
            = 1.0
              / (1.0
                 + (it_dist != row.end () ? AnyToDouble (it_dist->second, 0.0)
                                          : 0.0));
        matched_facts.push_back (
            { record, store::ScoreFactRecord (record, fact_query_mode,
                                              retrieval_ts),
              query_similarity });
        if (retrieval_mode == temporal::RetrievalMode::Current)
          {
            const std::string key
                = FactPairKey (record.canonical_subject,
                               record.canonical_predicate);
            auto it = current_pair_similarity.find (key);
            if (it == current_pair_similarity.end ())
              {
                current_pair_similarity.emplace (key, query_similarity);
              }
            else
              {
                it->second = std::max (it->second, query_similarity);
              }
            current_pair_object[key] = record.canonical_object;
          }
      }
  }

  if (fact_layer_enabled && !matched_facts.empty ())
    {
      std::ostringstream sql;
      sql << "SELECT fe.fact_id, fe.source_memory_id, fe.evidence_type, "
             "fe.support_weight FROM fact_evidence fe WHERE fe.fact_id IN (";
      std::vector<std::any> params;
      bool first = true;
      for (const auto &fact : matched_facts)
        {
          if (!first)
            {
              sql << ",";
            }
          first = false;
          sql << "?";
          params.push_back (fact.record.fact_id);
        }
      sql << ")";

      const auto t_start = std::chrono::steady_clock::now ();
      auto rows = store->Execute (sql.str (), params);
      const auto t_end = std::chrono::steady_clock::now ();
      context.AddOperationTiming ("GraphRetrieve.fact_evidence_sql",
                                  elapsed_ms (t_start, t_end));

      std::unordered_map<long long, const MatchedFact *> fact_lookup;
      for (const auto &fact : matched_facts)
        {
          fact_lookup.emplace (fact.record.fact_id, &fact);
        }
      for (const auto &row : rows)
        {
          const long long fact_id = AnyToInt64 (row.at ("fact_id"));
          const long long source_memory_id
              = AnyToInt64 (row.at ("source_memory_id"));
          auto it = fact_lookup.find (fact_id);
          if (it == fact_lookup.end () || source_memory_id <= 0)
            {
              continue;
            }
          candidate_fact_links[source_memory_id].push_back (
              { it->second->record,
                it->second->score,
                it->second->query_similarity,
                AnyToString (row.at ("evidence_type")),
                AnyToDouble (row.at ("support_weight"), 1.0),
                false });
        }
    }

  if (fact_layer_enabled
      && retrieval_mode == temporal::RetrievalMode::Current
      && !current_pair_similarity.empty ())
    {
      std::ostringstream sql;
      sql << "SELECT fa.fact_id, fa.subject, fa.predicate, fa.object, "
             "fa.canonical_subject, fa.canonical_predicate, fa.canonical_object, "
             "fa.valid_start_ts, fa.valid_end_ts, fa.recorded_at_ts, "
             "fa.superseded_at_ts, fa.confidence, fa.summary_memory_id, "
             "COALESCE(fc.embedding_id, 0) AS embedding_id, "
             "COALESCE(fc.fact_text, '') AS fact_text, "
             "COALESCE(fc.is_current, 0) AS is_current, "
             "COALESCE(fa.support_mass, 0.0) AS support_mass, "
             "COALESCE(fa.source_diversity, 0) AS source_diversity, "
             "COALESCE(fa.contradiction_mass, 0.0) AS contradiction_mass, "
             "COALESCE(fa.confirmation_count, 0) AS confirmation_count, "
             "COALESCE(fa.challenge_count, 0) AS challenge_count, "
             "COALESCE(fa.compressed_support_count, 0) AS compressed_support_count, "
             "COALESCE(fa.last_confirmation_ts, 0) AS last_confirmation_ts, "
             "fa.last_challenge_ts, "
             "COALESCE(fa.severity_class, 'medium') AS severity_class, "
             "COALESCE(fa.lifecycle_state, 'active') AS lifecycle_state, "
             "fa.archived_at, "
             "COALESCE(fa.last_maintenance_ts, 0) AS last_maintenance_ts, "
             "(SELECT COUNT(*) FROM fact_evidence fe2 "
             " LEFT JOIN memories m2 ON m2.memory_id = fe2.source_memory_id "
             " WHERE fe2.fact_id = fa.fact_id AND m2.memory_id IS NOT NULL) "
             " AS evidence_count, "
             "fe.source_memory_id, fe.evidence_type, fe.support_weight "
             "FROM fact_assertions fa "
             "JOIN fact_evidence fe ON fe.fact_id = fa.fact_id "
             "LEFT JOIN fact_cache fc ON fc.fact_id = fa.fact_id "
             "WHERE (";
      std::vector<std::any> params;
      bool first = true;
      for (const auto &fact : matched_facts)
        {
          if (!first)
            {
              sql << " OR ";
            }
          first = false;
          sql << "(fa.canonical_subject = ? AND fa.canonical_predicate = ?)";
          params.push_back (fact.record.canonical_subject);
          params.push_back (fact.record.canonical_predicate);
        }
      sql << ") AND NOT ("
             "(fa.valid_start_ts IS NULL OR fa.valid_start_ts <= ?) "
             "AND (fa.valid_end_ts IS NULL OR fa.valid_end_ts > ?) "
             "AND fa.recorded_at_ts <= ? "
             "AND (fa.superseded_at_ts IS NULL OR fa.superseded_at_ts > ?))";
      params.push_back (static_cast<long long> (retrieval_ts));
      params.push_back (static_cast<long long> (retrieval_ts));
      params.push_back (static_cast<long long> (retrieval_ts));
      params.push_back (static_cast<long long> (retrieval_ts));

      const auto t_start = std::chrono::steady_clock::now ();
      auto rows = store->Execute (sql.str (), params);
      const auto t_end = std::chrono::steady_clock::now ();
      context.AddOperationTiming ("GraphRetrieve.fact_stale_sql",
                                  elapsed_ms (t_start, t_end));

      for (const auto &row : rows)
        {
          const store::FactRecord record = BuildFactRecord (row);
          const long long source_memory_id
              = AnyToInt64 (row.at ("source_memory_id"));
          if (source_memory_id <= 0)
            {
              continue;
            }
          const std::string key
              = FactPairKey (record.canonical_subject, record.canonical_predicate);
          auto it = current_pair_similarity.find (key);
          if (it == current_pair_similarity.end ())
            {
              continue;
            }
          candidate_fact_links[source_memory_id].push_back (
              { record,
                store::ScoreFactRecord (record, store::FactQueryMode::Current,
                                        retrieval_ts),
                it->second,
                AnyToString (row.at ("evidence_type")),
                AnyToDouble (row.at ("support_weight"), 1.0),
                true });

          auto object_it = current_pair_object.find (key);
          const bool is_conflicting_object
              = object_it != current_pair_object.end ()
                && record.canonical_object != object_it->second;
          if (!is_conflicting_object)
            {
              continue;
            }

        }
    }

  if (seeds.empty () && matched_facts.empty ())
    {
      return;
    }

  // Update last retrieval timestamp for consolidation idle-gating.
  p_ctx.last_retrieval_ts = signal.timestamp;
  p_ctx.metacognitive_certainty_satisfied = false;

  // V2: Graph expansion from seed memory IDs using ASSOCIATIONS table.
  std::unordered_set<long long> expanded_memory_ids;
  for (const auto &s : seeds)
    {
      if (s.memory_id > 0)
        {
          expanded_memory_ids.insert (s.memory_id);
        }
    }
  for (const auto &[memory_id, links] : candidate_fact_links)
    {
      if (!links.empty () && memory_id > 0)
        {
          expanded_memory_ids.insert (memory_id);
        }
    }

  // Add index-store seeds for pattern completion.
  if (!sparse_key.empty ())
    {
      auto it = p_ctx.index_store.find (sparse_key);
      if (it != p_ctx.index_store.end ())
        {
          for (const auto mem_id : it->second)
            {
              if (mem_id > 0)
                expanded_memory_ids.insert (mem_id);
            }
        }
    }

  // Expand via ASSOCIATIONS graph traversal
  auto get_double = [] (const std::any &v, double def) -> double {
    if (v.type () == typeid (double))
      return std::any_cast<double> (v);
    if (v.type () == typeid (float))
      return static_cast<double> (std::any_cast<float> (v));
    if (v.type () == typeid (int))
      return static_cast<double> (std::any_cast<int> (v));
    if (v.type () == typeid (long long))
      return static_cast<double> (std::any_cast<long long> (v));
    return def;
  };

  if (!expanded_memory_ids.empty ())
    {
      try
        {
          const auto t_start = std::chrono::steady_clock::now ();
          // Build SQL with a VALUES list for seeds.
          const char *edge_filter
              = cfg.reinforcement_enabled ? "" : " AND a.edge_type != 'reinforces' ";
          std::string sql = "WITH RECURSIVE seed(id) AS (VALUES ";
          std::vector<std::any> params;
          params.reserve (expanded_memory_ids.size () + 2);
          bool first = true;
          for (long long mem_id : expanded_memory_ids)
            {
              if (!first)
                sql += ", ";
              first = false;
              sql += "(?)";
              params.push_back (mem_id);
            }
          sql += "), expand(id, depth) AS ("
                 "  SELECT id, 0 FROM seed "
                 "  UNION "
                 "  SELECT a.target_memory_id, expand.depth+1 "
                 "  FROM associations a JOIN expand ON a.source_memory_id = expand.id "
                 "  WHERE expand.depth < ? AND a.weight >= ? ";
          sql += edge_filter;
          sql += "  UNION "
                 "  SELECT a.source_memory_id, expand.depth+1 "
                 "  FROM associations a JOIN expand ON a.target_memory_id = expand.id "
                 "  WHERE expand.depth < ? AND a.weight >= ? ";
          sql += edge_filter;
          sql += ") "
                 "SELECT DISTINCT id FROM expand;";
          params.push_back (static_cast<long long> (depth));
          params.push_back (min_edge_weight);
          params.push_back (static_cast<long long> (depth));
          params.push_back (min_edge_weight);

          auto exp_rows = store->Execute (sql, params);
          const auto t_end = std::chrono::steady_clock::now ();
          context.AddOperationTiming ("GraphRetrieve.expand_sql",
                                      elapsed_ms (t_start, t_end));
          for (const auto &r : exp_rows)
            {
              auto it = r.find ("id");
              if (it != r.end () && it->second.type () == typeid (long long))
                {
                  expanded_memory_ids.insert (std::any_cast<long long> (it->second));
                }
            }
        }
      catch (...)
        {
          // No associations or query failed; use seed-only behavior.
        }
    }

  const double min_assoc_raw
      = core::Lerp (3.0, 1.0, f_eff) * core::Lerp (1.0, 0.85, cfg.stability);
  const double min_label_raw
      = core::Lerp (0.0, 3.0, s_eff) * core::Lerp (1.0, 0.85, f_eff);
  int min_assoc = static_cast<int> (std::round (min_assoc_raw));
  int min_label = static_cast<int> (std::round (min_label_raw));
  if (min_assoc < 0)
    min_assoc = 0;
  if (min_label < 0)
    min_label = 0;

  if (!expanded_memory_ids.empty () && (min_assoc > 0 || min_label > 0))
    {
      std::unordered_set<long long> summary_ids;
      if (min_assoc > 0)
        {
          try
            {
              const auto t_start = std::chrono::steady_clock::now ();
              std::string sql =
                  "SELECT DISTINCT a.source_memory_id AS id "
                  "FROM associations a "
                  "JOIN memories m ON m.memory_id = a.source_memory_id "
                  "WHERE a.edge_type = 'derived_from' "
                  "AND a.target_memory_id IN (";
              std::vector<std::any> params;
              params.reserve (expanded_memory_ids.size ());
              bool first = true;
              for (const long long id : expanded_memory_ids)
                {
                  if (!first)
                    sql += ",";
                  first = false;
                  sql += "?";
                  params.push_back (id);
                }
              sql += ") AND m.kind = 'ASSOCIATION'";
              auto rows = store->Execute (sql, params);
              const auto t_end = std::chrono::steady_clock::now ();
              context.AddOperationTiming ("GraphRetrieve.summary_assoc_sql",
                                          elapsed_ms (t_start, t_end));
              for (const auto &r : rows)
                {
                  auto it = r.find ("id");
                  if (it != r.end ()
                      && it->second.type () == typeid (long long))
                    {
                      const long long mem_id
                          = std::any_cast<long long> (it->second);
                      if (mem_id > 0)
                        {
                          summary_ids.insert (mem_id);
                          expanded_memory_ids.insert (mem_id);
                        }
                    }
                }
            }
          catch (...)
            {
            }
        }

      if (min_label > 0 && !summary_ids.empty ())
        {
          try
            {
              const auto t_start = std::chrono::steady_clock::now ();
              std::string sql =
                  "SELECT DISTINCT a.target_memory_id AS id "
                  "FROM associations a "
                  "JOIN memories m ON m.memory_id = a.target_memory_id "
                  "WHERE a.edge_type = 'has_label' "
                  "AND a.source_memory_id IN (";
              std::vector<std::any> params;
              params.reserve (summary_ids.size ());
              bool first = true;
              for (const long long id : summary_ids)
                {
                  if (!first)
                    sql += ",";
                  first = false;
                  sql += "?";
                  params.push_back (id);
                }
              sql += ") AND m.kind = 'LABEL'";
              auto rows = store->Execute (sql, params);
              const auto t_end = std::chrono::steady_clock::now ();
              context.AddOperationTiming ("GraphRetrieve.summary_label_sql",
                                          elapsed_ms (t_start, t_end));
              for (const auto &r : rows)
                {
                  auto it = r.find ("id");
                  if (it != r.end ()
                      && it->second.type () == typeid (long long))
                    {
                      const long long mem_id
                          = std::any_cast<long long> (it->second);
                      if (mem_id > 0)
                        {
                          expanded_memory_ids.insert (mem_id);
                        }
                    }
                }
            }
          catch (...)
            {
            }
        }
    }

  // Fetch embeddings for expanded memory_ids via MEMORIES table
  std::unordered_map<long long, Eigen::VectorXf> out;
  std::vector<Scored> scored;
  scored.reserve (expanded_memory_ids.size ());

  bool fetched_any = false;
  if (!expanded_memory_ids.empty ())
    {
      try
        {
          const auto t_start = std::chrono::steady_clock::now ();
          std::string sql
              = std::string (
                    "SELECT m.memory_id, "
                    "m.embedding_id AS base_embedding_id, "
                    "e.embedding, "
                    "COALESCE(m.created_at, e.created_at, 0) AS created_at, "
                    "m.context, m.source_origin, m.source_reliability, "
                    "m.source_contradiction_count, m.emotional_intensity, "
                    "m.s_arousal_avg, m.kind, "
                    "COALESCE(m.pre_activation, 0.0) AS pre_activation "
                    "FROM memories m ")
                + latest_reconstruction_join
                + "JOIN embeddings e "
                  "ON e.embedding_id = "
                + current_embedding_expr
                + " "
                  "WHERE m.memory_id IN (";
          std::vector<std::any> params;
          params.reserve (expanded_memory_ids.size ());
          bool first = true;
          for (const long long id : expanded_memory_ids)
            {
              if (!first)
                sql += ",";
              first = false;
              sql += "?";
              params.push_back (id);
            }
          sql += ");";
          auto fetch_rows = store->Execute (sql, params);
          const auto t_end = std::chrono::steady_clock::now ();
          context.AddOperationTiming ("GraphRetrieve.fetch_embeddings_sql",
                                      elapsed_ms (t_start, t_end));
          for (const auto &row : fetch_rows)
            {
              auto it_mem_id = row.find ("memory_id");
              auto it_emb_id = row.find ("base_embedding_id");
              if (it_emb_id == row.end ())
                {
                  it_emb_id = row.find ("embedding_id");
                }
              auto it_emb = row.find ("embedding");
              auto it_created = row.find ("created_at");
              auto it_ctx = row.find ("context");
              auto it_origin = row.find ("source_origin");
              auto it_rel = row.find ("source_reliability");
              auto it_contra = row.find ("source_contradiction_count");
              auto it_emotion = row.find ("emotional_intensity");
              auto it_arousal = row.find ("s_arousal_avg");
              auto it_kind = row.find ("kind");
              auto it_preact = row.find ("pre_activation");
              if (it_mem_id == row.end () || it_emb_id == row.end () || it_emb == row.end ())
                continue;
              if (it_mem_id->second.type () != typeid (long long))
                continue;
              if (it_emb_id->second.type () != typeid (long long))
                continue;
              const long long mem_id = std::any_cast<long long> (it_mem_id->second);
              const long long emb_id = std::any_cast<long long> (it_emb_id->second);
              Eigen::VectorXf v;
              if (!core::DecodeFloatBlob (
                      it_emb->second, static_cast<int> (q.size ()), v))
                continue;
              const double sim = core::CosineSimilarity (q, v);
              long long created_at = 0;
              if (it_created != row.end () && it_created->second.type () == typeid (long long))
                {
                  created_at = std::any_cast<long long> (it_created->second);
                }
              Eigen::VectorXf ctx_vec;
              if (it_ctx != row.end () && it_ctx->second.has_value ())
                {
                  core::DecodeFloatBlob (
                      it_ctx->second, static_cast<int> (q_ctx.size ()), ctx_vec);
                }
              const double ctx_sim
                  = (ctx_vec.size () == q_ctx.size () && q_ctx.size () > 0)
                        ? core::CosineSimilarity (q_ctx, ctx_vec)
                        : 0.0;

              std::string origin;
              if (it_origin != row.end () && it_origin->second.type () == typeid (std::string))
                {
                  origin = std::any_cast<std::string> (it_origin->second);
                }

              double base_rel = 0.7;
              if (it_rel != row.end () && it_rel->second.type () == typeid (double))
                {
                  base_rel = std::any_cast<double> (it_rel->second);
                }
              if (origin == "user")
                base_rel = std::max (base_rel, 0.8);
              else if (origin == "assistant")
                base_rel = std::max (base_rel, 0.6);
              else if (origin == "system")
                base_rel = std::max (base_rel, 0.9);

              int contradiction_count = 0;
              if (it_contra != row.end () && it_contra->second.type () == typeid (long long))
                {
                  contradiction_count
                      = static_cast<int> (std::any_cast<long long> (it_contra->second));
                }

              double memory_emotion = 0.0;
              if (it_emotion != row.end () && it_emotion->second.has_value ())
                {
                  memory_emotion
                      = core::Clamp (get_double (it_emotion->second, 0.0), 0.0, 1.0);
                }
              double memory_arousal = 0.0;
              if (it_arousal != row.end () && it_arousal->second.has_value ())
                {
                  memory_arousal
                      = core::Clamp (get_double (it_arousal->second, 0.0), 0.0, 1.0);
                }
              bool is_association = false;
              bool is_label = false;
              if (it_kind != row.end () && it_kind->second.type () == typeid (std::string))
                {
                  const std::string kind
                      = std::any_cast<std::string> (it_kind->second);
                  is_association = (kind == "ASSOCIATION");
                  is_label = (kind == "LABEL");
                }

              double age_s = 0.0;
              if (created_at > 0
                  && signal.timestamp > static_cast<uint64_t> (created_at))
                {
                  age_s = static_cast<double> (signal.timestamp - created_at) / 1000.0;
                }
              const double freshness = std::exp (-age_s / 3600.0);
              const double source_conf
                  = core::Clamp (base_rel * (1.0 - 0.15 * contradiction_count)
                                     * (0.7 + 0.3 * freshness),
                                 0.0, 1.0);

              double proc_score = 0.0;
              if (cfg.procedural_enabled && !sparse_key.empty ())
                {
                  auto pit = p_ctx.procedural_store.find (sparse_key);
                  if (pit != p_ctx.procedural_store.end ())
                    {
                      auto mit = pit->second.find (mem_id);
                      if (mit != pit->second.end ())
                        {
                          proc_score = mit->second;
                        }
                    }
                }

              const double pre_activation
                  = (it_preact != row.end ())
                        ? core::Clamp (get_double (it_preact->second, 0.0), 0.0, 1.0)
                        : 0.0;

              scored.push_back (Scored{ emb_id,
                                        mem_id,
                                        created_at,
                                        sim,
                                        ctx_sim,
                                        source_conf,
                                        proc_score,
                                        pre_activation,
                                        predictive_weight * pre_activation,
                                        contradiction_count,
                                        memory_emotion,
                                        memory_arousal,
                                        0.0,
                                        0.0,
                                        0,
                                        is_association,
                                        is_label,
                                        v,
                                        ctx_vec });
              fetched_any = true;
            }
        }
      catch (...)
        {
        }
    }

  auto is_overlap_wm = [&p_ctx] (const Eigen::VectorXf &v,
                                 double dup_thresh) -> bool {
    if (v.size () == 0)
      {
        return false;
      }
    for (const auto &slot : p_ctx.wm_slots)
      {
        if (slot.embedding.size () == v.size () && slot.embedding.size () > 0)
          {
            const double sim = core::CosineSimilarity (slot.embedding, v);
            if (sim >= dup_thresh)
              {
                return true;
              }
          }
      }
    return false;
  };

  const double dup_thresh = core::DupThresh (cfg.focus, cfg.stability);
  const double dup_thresh_summary = core::Clamp (
      dup_thresh * core::Lerp (1.35, 1.10, f_eff), 0.0, 0.99);
  const bool bypass_summary_overlap = (f_eff < 0.75);
  const std::uint64_t write_exclusion_ts
      = context.GetWriteExclusionTs ().value_or (signal.timestamp);
  const auto stored_id = context.GetStoredEmbeddingId ();
  const auto weights
      = core::RetrievalDiversificationWeights (cfg.focus, cfg.sensitivity,
                                               cfg.stability);
  const double w_rel = weights.first;
  double w_div = weights.second;
  if (metacognitive_mode == ProcessorContext::MetacognitiveMode::TotRecovery)
    {
      w_div *= core::Lerp (0.95, 0.75, tot_confidence_scale);
    }
  const double w_ctx = core::Lerp (0.15, 0.35, f_eff)
                       * core::Lerp (1.0, 0.85, s_eff);
  const double w_proc = core::Lerp (0.10, 0.25, s_eff);
  const double w_emotion
      = cfg.affect_retrieval
            ? core::RetrievalEmotionWeight (cfg.sensitivity)
            : 0.0;
  const double assoc_boost
      = core::AssociationBoost (cfg.focus, cfg.sensitivity, cfg.stability);
  const double label_boost = core::Lerp (0.05, 0.18, s_eff);

  const double salience = core::Clamp (
      context.GetMetric (operations::Metric::salience).value_or (0.0), 0.0, 1.0);
  const double emotion_intensity
      = core::Clamp (context.GetEmotionIntensity (), 0.0, 1.0);
  const double arousal = core::Clamp (context.GetArousal (), 0.0, 1.0);
  const double s_affect = core::AffectSensitivityBias (cfg.sensitivity);
  const double w_arousal_raw = core::Lerp (0.30, 0.55, s_affect);
  const double w_emotion_raw = core::Lerp (0.30, 0.55, s_affect);
  const double w_salience_raw = core::Lerp (0.10, 0.20, s_affect);
  const double w_aff_sum = std::max (constants::kNormEpsilon,
                                     w_arousal_raw + w_emotion_raw
                                       + w_salience_raw);
  const double w_aff_arousal = w_arousal_raw / w_aff_sum;
  const double w_aff_emotion = w_emotion_raw / w_aff_sum;
  const double w_aff_salience = w_salience_raw / w_aff_sum;
  const double affect_gain = core::AffectGain (cfg.sensitivity);
  const double affect_drive
      = core::Clamp (affect_gain
                         * (w_aff_arousal * arousal
                            + w_aff_emotion * emotion_intensity
                            + w_aff_salience * salience),
                     0.0, 1.0);
  const double w_mem_emotion_raw = core::Lerp (0.60, 0.80, s_eff);
  const double w_mem_arousal_raw = core::Lerp (0.20, 0.40, s_eff);
  const double w_mem_sum = std::max (constants::kNormEpsilon,
                                     w_mem_emotion_raw + w_mem_arousal_raw);
  const double w_mem_emotion = w_mem_emotion_raw / w_mem_sum;
  const double w_mem_arousal = w_mem_arousal_raw / w_mem_sum;

  const double source_thresh = core::Lerp (0.15, 0.45, cfg.stability);
  struct FilterStats
  {
    int64_t total = 0;
    int64_t skipped_stored = 0;
    int64_t skipped_write_exclusion = 0;
    int64_t skipped_overlap = 0;
    int64_t skipped_source_conf = 0;
    int64_t kept = 0;
    int64_t kept_assoc = 0;
    int64_t kept_label = 0;
    int64_t kept_other = 0;
  };

  auto filter_candidates = [&] (const std::vector<Scored> &candidates,
                                 bool enforce_write_exclusion,
                                 bool enforce_source_conf,
                                 FilterStats *stats) {
    std::vector<Scored> eligible;
    eligible.reserve (candidates.size ());
    for (const auto &s : candidates)
      {
        if (stats)
          {
            stats->total++;
          }
        if (stored_id.has_value () && s.embedding_id == *stored_id)
          {
            if (stats)
              {
                stats->skipped_stored++;
              }
            continue;
          }
        const bool summary_like = (s.is_association || s.is_label);
        if (enforce_write_exclusion && !summary_like && s.created_at > 0
            && static_cast<std::uint64_t> (s.created_at) >= write_exclusion_ts)
          {
            if (stats)
              {
                stats->skipped_write_exclusion++;
              }
            continue;
          }
        const double cand_dup_thresh
            = summary_like ? dup_thresh_summary : dup_thresh;
        if (!(summary_like && bypass_summary_overlap)
            && is_overlap_wm (s.vec, cand_dup_thresh))
          {
            if (stats)
              {
                stats->skipped_overlap++;
              }
            continue;
          }
        if (!disable_source_conf && enforce_source_conf
            && s.source_confidence < source_thresh)
          {
            if (stats)
              {
                stats->skipped_source_conf++;
              }
            continue;
          }
        eligible.push_back (s);
        if (stats)
          {
            stats->kept++;
            if (s.is_association)
              {
                stats->kept_assoc++;
              }
            else if (s.is_label)
              {
                stats->kept_label++;
              }
            else
              {
                stats->kept_other++;
              }
          }
      }
    return eligible;
  };

  auto apply_fact_scores = [&] (std::vector<Scored> &candidates) {
    for (auto &candidate : candidates)
      {
        if (!fact_layer_enabled || candidate.memory_id <= 0)
          {
            continue;
          }
        std::vector<LinkedFactEvidence> links;
        auto it = candidate_fact_links.find (candidate.memory_id);
        if (it != candidate_fact_links.end ())
          {
            links = it->second;
          }
        if (links.empty ()
            && ResolveProvenanceMode (ablation_override)
                   == temporal::ProvenanceMode::AnyFactMatch)
          {
            const double candidate_support = core::Clamp (
                std::max ({ candidate.score, candidate.ctx_score,
                            candidate.proc_score }),
                0.25, 1.0);
            const std::size_t cap = std::min<std::size_t> (matched_facts.size (),
                                                           2);
            for (std::size_t i = 0; i < cap; ++i)
              {
                links.push_back ({ matched_facts[i].record,
                                   matched_facts[i].score,
                                   matched_facts[i].query_similarity,
                                   "semantic",
                                   candidate_support,
                                   false });
              }
          }
        if (links.empty ())
          {
            continue;
          }
        const auto score = ScoreCandidateFacts (
            links, cfg.focus, cfg.sensitivity, cfg.stability, retrieval_mode,
            ablation_override);
        candidate.fact_boost = score.boost;
        candidate.fact_stale_penalty = score.stale_penalty;
        candidate.linked_fact_count = score.linked_fact_count;
      }
  };

  auto base_score = [&] (const Scored &s) {
    const double relevance = std::max (0.0, s.score);
    const double ctx_sim = std::max (0.0, s.ctx_score);
    const double proc_sim = std::max (0.0, s.proc_score);
    const double memory_affect
        = w_mem_emotion * s.emotion_intensity + w_mem_arousal * s.arousal_avg;
    const double emotion_bonus = affect_drive * memory_affect;
    const double association_bonus = s.is_association ? assoc_boost : 0.0;
    const double label_bonus = s.is_label ? label_boost : 0.0;
    return w_rel * relevance + w_ctx * ctx_sim + w_proc * proc_sim
           + w_emotion * emotion_bonus + s.predictive_bonus + association_bonus
           + label_bonus
           + s.fact_boost - s.fact_stale_penalty;
  };

  auto apply_unknown_caution = [&] (std::vector<Scored> &selected) {
    p_ctx.metacognitive_certainty_satisfied = false;
    if (metacognitive_mode
            != ProcessorContext::MetacognitiveMode::UnknownCaution)
      {
        return;
      }
    if (selected.empty ())
      {
        return;
      }
    double best_score = -1e9;
    for (const auto &candidate : selected)
      {
        best_score = std::max (best_score, base_score (candidate));
      }
    if (best_score <= 0.0)
      {
        selected.clear ();
        return;
      }
    const double cutoff
        = core::CertaintyRequirement (cfg.stability) * best_score
          * core::Lerp (1.0, 1.15, unknown_caution_scale);
    selected.erase (
        std::remove_if (selected.begin (), selected.end (),
                        [&] (const Scored &candidate) {
                          return base_score (candidate) < cutoff;
                        }),
        selected.end ());
    p_ctx.metacognitive_certainty_satisfied = !selected.empty ();
  };

  auto select_diversified = [&] (const std::vector<Scored> &candidates, int k,
                                 const std::vector<Scored> &initial) {
    std::vector<Scored> selected;
    if (candidates.empty () || k <= 0)
      {
        return selected;
      }
    const size_t n = candidates.size ();
    std::vector<bool> used (n, false);
    selected.reserve (std::min (n, static_cast<size_t> (k)));

    std::unordered_map<long long, size_t> candidate_index;
    candidate_index.reserve (n);
    for (size_t i = 0; i < n; ++i)
      {
        candidate_index.emplace (candidates[i].embedding_id, i);
      }

    for (const auto &seed : initial)
      {
        auto it = candidate_index.find (seed.embedding_id);
        if (it == candidate_index.end ())
          {
            continue;
          }
        const size_t idx = it->second;
        if (used[idx])
          {
            continue;
          }
        used[idx] = true;
        selected.push_back (candidates[idx]);
      }

    for (int iter = static_cast<int> (selected.size ()); iter < k; ++iter)
      {
        double best_score = -1e9;
        int best_idx = -1;
        for (size_t i = 0; i < n; ++i)
          {
            if (used[i])
              {
                continue;
              }
            const double core_score = base_score (candidates[i]);
            double max_redundancy = 0.0;
            for (const auto &sel : selected)
              {
                const double sim
                    = core::CosineSimilarity (candidates[i].vec, sel.vec);
                max_redundancy = std::max (max_redundancy, std::max (0.0, sim));
              }
            const double mmr = core_score - w_div * max_redundancy;
            if (mmr > best_score)
              {
                best_score = mmr;
                best_idx = static_cast<int> (i);
              }
          }
        if (best_idx < 0)
          {
            break;
          }
        used[best_idx] = true;
        selected.push_back (candidates[best_idx]);
      }
    return selected;
  };

  auto reinstate_context = [&] (const std::vector<Scored> &selected) {
    if (selected.empty ())
      {
        return;
      }
    Eigen::VectorXf mean_ctx;
    int ctx_count = 0;
    for (const auto &s : selected)
      {
        if (s.ctx.size () == q_ctx.size () && q_ctx.size () > 0)
          {
            if (mean_ctx.size () == 0)
              {
                mean_ctx = s.ctx;
              }
            else
              {
                mean_ctx += s.ctx;
              }
            ++ctx_count;
          }
      }
    if (ctx_count > 0)
      {
        mean_ctx /= static_cast<float> (ctx_count);
        const float norm = mean_ctx.norm ();
        if (norm > 1e-9f)
          {
            mean_ctx /= norm;
          }
        if (acc_it != p_ctx.accumulator_states.end ())
          {
            auto &acc = acc_it->second;
            const double alpha = core::Lerp (0.20, 0.05, cfg.stability);
            if (acc.c_t.size () != mean_ctx.size ())
              {
                acc.c_t = mean_ctx;
              }
            else
              {
                acc.c_t = acc.c_t * static_cast<float> (1.0 - alpha)
                          + mean_ctx * static_cast<float> (alpha);
              }
            const float acc_norm = acc.c_t.norm ();
            if (acc_norm > 1e-9f)
              {
                acc.c_t /= acc_norm;
              }
          }
      }
  };

  auto append_reconstruction_versions = [&] (
                                            const std::vector<Scored> &selected) {
    if (disable_constructive_recall || !store || selected.empty ())
      {
        return;
      }
    for (const auto &candidate : selected)
      {
        if (candidate.memory_id <= 0 || candidate.vec.size () != q.size ()
            || candidate.vec.size () == 0)
          {
            continue;
          }

        const double source_conf
            = core::Clamp (candidate.source_confidence, 0.0, 1.0);
        const double uncertainty = core::Clamp (
            0.45 * (1.0 - source_conf)
                + 0.35 * (1.0 - std::max (0.0, candidate.score))
                + 0.20 * (1.0 - std::max (0.0, candidate.ctx_score)),
            0.0, 1.0);
        const double blend = core::Clamp (0.05 + 0.20 * uncertainty, 0.05,
                                          0.25);

        Eigen::VectorXf reconstructed
            = static_cast<float> (1.0 - blend) * candidate.vec
              + static_cast<float> (blend * 0.75) * q;
        if (q_ctx.size () == q.size () && q_ctx.size () > 0)
          {
            reconstructed += static_cast<float> (blend * 0.25) * q_ctx;
          }
        const float norm = reconstructed.norm ();
        if (norm <= 1e-9f)
          {
            continue;
          }
        reconstructed /= norm;

        const auto blob_id
            = constructive_recall::LoadCurrentBlobId (tx, candidate.memory_id);
        constructive_recall::AppendReconstructionWithEmbedding (
            tx, candidate.memory_id, reconstructed, blob_id,
            static_cast<long long> (signal.timestamp), uncertainty, "retrieval",
            source_conf, std::max (0.0, candidate.ctx_score));
      }
  };

  int64_t reinforcement_candidate_count = 0;
  if (!fetched_any)
    {
      if (!seeds.empty ())
        {
          const auto t_start = std::chrono::steady_clock::now ();
          for (auto &s : seeds)
            {
              const auto current = constructive_recall::LoadCurrentEmbedding (
                  tx, s.memory_id, s.embedding_id,
                  static_cast<int> (q.size ()));
              if (!current.has_value ())
                {
                  continue;
                }
              s.vec = *current;
              if (s.vec.size () == q.size () && s.vec.size () > 0)
                {
                  s.score = core::CosineSimilarity (q, s.vec);
                }
            }
          const auto t_end = std::chrono::steady_clock::now ();
          context.AddOperationTiming ("GraphRetrieve.seed_fetch_sql",
                                      elapsed_ms (t_start, t_end));
        }
      for (auto &s : seeds)
        {
          if (s.vec.size () == q.size () && s.vec.size () > 0)
            {
              s.score = core::CosineSimilarity (q, s.vec);
            }
        }
      apply_fact_scores (seeds);
      const auto t_score_start = std::chrono::steady_clock::now ();
      FilterStats strict_stats;
      auto eligible = filter_candidates (seeds, true, true, &strict_stats);
      FilterStats relax_stats;
      if (eligible.empty ()
          && metacognitive_mode
                 != ProcessorContext::MetacognitiveMode::UnknownCaution)
        {
          eligible = filter_candidates (seeds, false, false, &relax_stats);
        }
      auto selected = select_diversified (eligible, k, {});
      std::stable_sort (selected.begin (), selected.end (),
                        [&] (const Scored &a, const Scored &b) {
                          return base_score (a) > base_score (b);
                        });
      apply_unknown_caution (selected);
      const auto t_score_end = std::chrono::steady_clock::now ();
      context.AddOperationTiming ("GraphRetrieve.score", elapsed_ms (t_score_start, t_score_end));
      for (const auto &s : selected)
        {
          out.emplace (s.embedding_id, s.vec);
        }
      {
        std::vector<long long> selected_order;
        selected_order.reserve (selected.size ());
        std::vector<retrieval_debug::RankedCandidate> ranked_candidates;
        ranked_candidates.reserve (selected.size ());
        for (const auto &s : selected)
          {
            selected_order.push_back (s.embedding_id);
            ranked_candidates.push_back (
                { s.embedding_id,
                  s.memory_id,
                  base_score (s),
                  std::max (0.0, s.score),
                  std::max (0.0, s.proc_score),
                  s.predictive_bonus,
                  s.pre_activation,
                  s.fact_boost,
                  s.fact_stale_penalty,
                  s.linked_fact_count });
          }
        retrieval_debug::SetLastSelectedEmbeddingOrder (selected_order);
        retrieval_debug::SetLastRankedCandidates (ranked_candidates);
      }
      append_reconstruction_versions (selected);
      context.SetRetrievedMemoryEmbeddings (std::move (out));
      reinstate_context (selected);

      {
        std::vector<long long> seed_mem_ids;
        seed_mem_ids.reserve (selected.size ());
        for (const auto &s : selected)
          {
            if (s.memory_id > 0)
              {
                seed_mem_ids.push_back (s.memory_id);
              }
          }
        reinforcement_candidate_count
            = static_cast<int64_t> (seed_mem_ids.size ());
      }

      return;
    }

  apply_fact_scores (scored);
  FilterStats strict_stats;
  const auto t_score_start = std::chrono::steady_clock::now ();
  auto eligible = filter_candidates (scored, true, true, &strict_stats);
  FilterStats relax_stats;
  if (eligible.empty ()
      && metacognitive_mode
             != ProcessorContext::MetacognitiveMode::UnknownCaution)
    {
      eligible = filter_candidates (scored, false, false, &relax_stats);
    }
  if (min_assoc > k)
    min_assoc = k;
  if (min_label > k - min_assoc)
    min_label = k - min_assoc;
  int min_proc = 0;
  if (cfg.procedural_enabled && !disable_procedural_proactive)
    {
      min_proc = std::min (1, k - min_assoc - min_label);
    }

  auto pick_top = [&] (const std::vector<Scored> &candidates,
                       int count,
                       const std::function<bool(const Scored &)> &pred) {
    std::vector<std::pair<double, Scored>> ranked;
    ranked.reserve (candidates.size ());
    for (const auto &cand : candidates)
      {
        if (!pred (cand))
          {
            continue;
          }
        ranked.emplace_back (base_score (cand), cand);
      }
    std::sort (ranked.begin (), ranked.end (),
               [] (const auto &a, const auto &b) { return a.first > b.first; });
    std::vector<Scored> out;
    for (int i = 0; i < count && i < static_cast<int> (ranked.size ()); ++i)
      {
        out.push_back (ranked[i].second);
      }
    return out;
  };

  std::vector<Scored> initial;
  initial.reserve (static_cast<size_t> (min_assoc + min_label));
  std::unordered_set<long long> initial_ids;
  if (min_assoc > 0)
    {
      auto assoc_seeds = pick_top (
          eligible, min_assoc,
          [] (const Scored &s) { return s.is_association; });
      for (const auto &seed : assoc_seeds)
        {
          if (initial_ids.insert (seed.embedding_id).second)
            {
              initial.push_back (seed);
            }
        }
    }
  if (min_label > 0)
    {
      auto label_seeds = pick_top (
          eligible, min_label,
          [] (const Scored &s) { return s.is_label; });
      for (const auto &seed : label_seeds)
        {
          if (initial_ids.insert (seed.embedding_id).second)
            {
              initial.push_back (seed);
            }
        }
    }
  if (min_proc > 0)
    {
      auto proc_seeds = pick_top (
          eligible, min_proc,
          [] (const Scored &s) { return s.proc_score >= 0.55; });
      for (const auto &seed : proc_seeds)
        {
          if (initial_ids.insert (seed.embedding_id).second)
            {
              initial.push_back (seed);
            }
        }
    }

  auto selected = select_diversified (eligible, k, initial);
  std::stable_sort (selected.begin (), selected.end (),
                    [&] (const Scored &a, const Scored &b) {
                      return base_score (a) > base_score (b);
                    });
  apply_unknown_caution (selected);
  const auto t_score_end = std::chrono::steady_clock::now ();
  context.AddOperationTiming ("GraphRetrieve.score", elapsed_ms (t_score_start, t_score_end));
  for (const auto &s : selected)
    {
      out.emplace (s.embedding_id, s.vec);
    }
  {
    std::vector<long long> selected_order;
    selected_order.reserve (selected.size ());
    std::vector<retrieval_debug::RankedCandidate> ranked_candidates;
    ranked_candidates.reserve (selected.size ());
    for (const auto &s : selected)
      {
        selected_order.push_back (s.embedding_id);
        ranked_candidates.push_back (
            { s.embedding_id,
              s.memory_id,
              base_score (s),
              std::max (0.0, s.score),
              std::max (0.0, s.proc_score),
              s.predictive_bonus,
              s.pre_activation,
              s.fact_boost,
              s.fact_stale_penalty,
              s.linked_fact_count });
      }
    retrieval_debug::SetLastSelectedEmbeddingOrder (selected_order);
    retrieval_debug::SetLastRankedCandidates (ranked_candidates);
  }

  append_reconstruction_versions (selected);
  context.SetRetrievedMemoryEmbeddings (std::move (out));
  reinstate_context (selected);

  {
    std::vector<long long> retrieved_mem_ids;
    retrieved_mem_ids.reserve (selected.size ());
    for (const auto &s : selected)
      {
        if (s.memory_id > 0)
          {
            retrieved_mem_ids.push_back (s.memory_id);
          }
      }
    reinforcement_candidate_count
        = static_cast<int64_t> (retrieved_mem_ids.size ());
  }

  // Debug logging
  auto count_kinds = [] (const std::vector<Scored> &items,
                         int64_t &assoc, int64_t &label, int64_t &other) {
    assoc = 0;
    label = 0;
    other = 0;
    for (const auto &s : items)
      {
        if (s.is_association)
          {
            assoc++;
          }
        else if (s.is_label)
          {
            label++;
          }
        else
          {
            other++;
          }
      }
  };

  int64_t scored_assoc = 0, scored_label = 0, scored_other = 0;
  int64_t eligible_assoc = 0, eligible_label = 0, eligible_other = 0;
  int64_t selected_assoc = 0, selected_label = 0, selected_other = 0;
  int64_t initial_assoc = 0, initial_label = 0, initial_other = 0;
  int64_t fact_linked_selected = 0;
  double selected_predictive_bonus_sum = 0.0;
  double selected_preactivation_sum = 0.0;
  double selected_fact_boost_sum = 0.0;
  double selected_stale_penalty_sum = 0.0;
  count_kinds (scored, scored_assoc, scored_label, scored_other);
  count_kinds (eligible, eligible_assoc, eligible_label, eligible_other);
  count_kinds (selected, selected_assoc, selected_label, selected_other);
  count_kinds (initial, initial_assoc, initial_label, initial_other);
  for (const auto &s : selected)
    {
      if (s.fact_boost > 0.0 || s.fact_stale_penalty > 0.0)
        {
          fact_linked_selected++;
        }
      selected_predictive_bonus_sum += s.predictive_bonus;
      selected_preactivation_sum += s.pre_activation;
      selected_fact_boost_sum += s.fact_boost;
      selected_stale_penalty_sum += s.fact_stale_penalty;
    }

  telemetry::LogDebug ("cortext.graph_retrieval", {
    telemetry::Attribute::Double ("dup_thresh", dup_thresh),
    telemetry::Attribute::Int64 ("write_exclusion_ts", static_cast<int64_t> (write_exclusion_ts)),
    telemetry::Attribute::Bool ("stored_id_present", stored_id.has_value ()),
    telemetry::Attribute::Int64 ("selected_count", static_cast<int64_t> (selected.size ())),
    telemetry::Attribute::Int64 ("seed_count", static_cast<int64_t> (seeds.size ())),
    telemetry::Attribute::Int64 ("procedural_seed_count", procedural_seed_count),
    telemetry::Attribute::Int64 ("expansion_depth", static_cast<int64_t> (depth)),
    telemetry::Attribute::Int64 ("final_candidate_count", static_cast<int64_t> (scored.size ())),
    telemetry::Attribute::Int64 ("scored_assoc_count", scored_assoc),
    telemetry::Attribute::Int64 ("scored_label_count", scored_label),
    telemetry::Attribute::Int64 ("eligible_assoc_count", eligible_assoc),
    telemetry::Attribute::Int64 ("eligible_label_count", eligible_label),
    telemetry::Attribute::Int64 ("selected_assoc_count", selected_assoc),
    telemetry::Attribute::Int64 ("selected_label_count", selected_label),
    telemetry::Attribute::Int64 ("initial_assoc_count", initial_assoc),
    telemetry::Attribute::Int64 ("initial_label_count", initial_label),
    telemetry::Attribute::Int64 ("filtered_total", strict_stats.total),
    telemetry::Attribute::Int64 ("filtered_stored", strict_stats.skipped_stored),
    telemetry::Attribute::Int64 ("filtered_write_exclusion", strict_stats.skipped_write_exclusion),
    telemetry::Attribute::Int64 ("filtered_overlap", strict_stats.skipped_overlap),
    telemetry::Attribute::Int64 ("filtered_source_conf", strict_stats.skipped_source_conf),
    telemetry::Attribute::Int64 ("filtered_kept", strict_stats.kept),
    telemetry::Attribute::Double ("retrieval_w_rel", w_rel),
    telemetry::Attribute::Double ("retrieval_w_div", w_div),
    telemetry::Attribute::Double ("retrieval_w_ctx", w_ctx),
    telemetry::Attribute::Double ("retrieval_w_proc", w_proc),
    telemetry::Attribute::Double ("retrieval_predictive_weight",
                                  predictive_weight),
    telemetry::Attribute::Int64 ("reinforcement_candidate_count",
                                 reinforcement_candidate_count),
    telemetry::Attribute::Bool ("reinforcement_enabled",
                                cfg.reinforcement_enabled),
    telemetry::Attribute::String ("metacognitive_mode",
                                  MetacognitiveModeLabel (
                                      metacognitive_mode)),
    telemetry::Attribute::Double ("metacognitive_confidence",
                                  metacognitive_confidence),
    telemetry::Attribute::Double ("metacognitive_tot_confidence_scale",
                                  tot_confidence_scale),
    telemetry::Attribute::Double ("metacognitive_unknown_caution_scale",
                                  unknown_caution_scale),
    telemetry::Attribute::Bool ("metacognitive_certainty_satisfied",
                                p_ctx.metacognitive_certainty_satisfied),
    telemetry::Attribute::String ("temporal_mode",
                                  temporal::ToString (retrieval_mode)),
    telemetry::Attribute::String ("requested_temporal_mode",
                                  temporal::ToString (
                                      requested_retrieval_mode)),
    telemetry::Attribute::Int64 ("temporal_query_ts",
                                 static_cast<int64_t> (retrieval_ts)),
    telemetry::Attribute::Bool ("fact_layer_enabled", fact_layer_enabled),
    telemetry::Attribute::Bool (
        "history_enabled",
        ablation_override.history_enabled.value_or (true)),
    telemetry::Attribute::String (
        "fact_boost_mode",
        FactBoostModeLabel (ablation_override)),
    telemetry::Attribute::String (
        "stale_penalty_mode",
        StalePenaltyModeLabel (ablation_override)),
    telemetry::Attribute::String (
        "provenance_mode",
        temporal::ToString (ResolveProvenanceMode (ablation_override))),
    telemetry::Attribute::String (
        "routine_recency_mode",
        RoutineRecencyModeLabel (ablation_override)),
    telemetry::Attribute::Int64 ("fact_seed_count",
                                 static_cast<int64_t> (matched_facts.size ())),
    telemetry::Attribute::Int64 ("fact_linked_memory_count",
                                 static_cast<int64_t> (
                                     candidate_fact_links.size ())),
    telemetry::Attribute::Int64 ("fact_linked_selected_count",
                                 fact_linked_selected),
    telemetry::Attribute::Double (
        "predictive_bonus_mean",
        selected.empty () ? 0.0
                          : selected_predictive_bonus_sum
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "selected_preactivation_mean",
        selected.empty () ? 0.0
                          : selected_preactivation_sum
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "fact_mean_boost",
        selected.empty () ? 0.0
                          : selected_fact_boost_sum
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "fact_mean_stale_penalty",
        selected.empty () ? 0.0
                          : selected_stale_penalty_sum
                                / static_cast<double> (selected.size ()))
  });
}

} // namespace cortext::operations
