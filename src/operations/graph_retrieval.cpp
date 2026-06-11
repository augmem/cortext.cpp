#include "cortext/operations/graph_retrieval.hpp"

#include "constructive_recall_internal.hpp"
#include "../store/facts.hpp"
#include "eviction_ablation.hpp"
#include "retrieval_debug_state.hpp"
#include "storage_pressure.hpp"
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
#include <limits>
#include <sstream>
#include <map>
#include <set>

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

std::string
CanonicalQueryText (const std::string &text)
{
  std::string out;
  out.reserve (text.size ());
  bool previous_space = false;
  for (unsigned char c : text)
    {
      if (std::isalnum (c) != 0)
        {
          out.push_back (static_cast<char> (std::tolower (c)));
          previous_space = false;
        }
      else if (!out.empty () && !previous_space)
        {
          out.push_back (' ');
          previous_space = true;
        }
    }
  if (!out.empty () && out.back () == ' ')
    {
      out.pop_back ();
    }
  return out;
}

bool
CanonicalPhraseAppears (const std::string &canonical_query,
                        const std::string &canonical_label)
{
  if (canonical_query.empty () || canonical_label.empty ())
    {
      return false;
    }
  return (" " + canonical_query + " ")
             .find (" " + canonical_label + " ")
         != std::string::npos;
}

bool
IsRouteStopword (const std::string &token)
{
  static const std::set<std::string> stopwords = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "how", "i", "if", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "she", "so", "that",
    "the", "their", "them", "then", "there", "they", "this", "to",
    "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "with", "you", "your"
  };
  return stopwords.count (token) > 0;
}

std::vector<std::string>
RouteTokens (const std::string &text, int min_chars)
{
  std::vector<std::string> tokens;
  std::istringstream in (CanonicalQueryText (text));
  std::string token;
  const size_t min_token_chars = static_cast<size_t> (std::max (1, min_chars));
  while (in >> token)
    {
      if (token.size () < min_token_chars || IsRouteStopword (token))
        {
          continue;
        }
      tokens.push_back (token);
    }
  std::sort (tokens.begin (), tokens.end ());
  tokens.erase (std::unique (tokens.begin (), tokens.end ()), tokens.end ());
  return tokens;
}

double
RouteTokenOverlapScore (const std::vector<std::string> &query_tokens,
                        const std::string &text, double focus,
                        double sensitivity, double stability, int min_chars)
{
  if (query_tokens.empty () || text.empty ())
    {
      return 0.0;
    }
  const auto text_tokens = RouteTokens (text, min_chars);
  if (text_tokens.empty ())
    {
      return 0.0;
    }
  int overlap = 0;
  for (const auto &token : query_tokens)
    {
      if (std::binary_search (text_tokens.begin (), text_tokens.end (),
                              token))
        {
          ++overlap;
        }
    }
  if (overlap == 0)
    {
      return 0.0;
    }
  const double query_coverage
      = static_cast<double> (overlap)
        / static_cast<double> (query_tokens.size ());
  const double source_coverage
      = static_cast<double> (overlap)
        / static_cast<double> (std::min (query_tokens.size (),
                                         text_tokens.size ()));
  const double query_weight = core::RetrievalTokenOverlapQueryWeight (
      focus, sensitivity, stability);
  return core::Clamp (query_weight * query_coverage
                          + (1.0 - query_weight) * source_coverage,
                      0.0, 1.0);
}

std::string
SignalTextPayload (const Signal &signal)
{
  if (!signal.payload.has_value () || signal.payload->empty ())
    {
      return {};
    }
  if (signal.modality != "text" && signal.mimetype != "text/plain")
    {
      return {};
    }
  return std::string (signal.payload->begin (), signal.payload->end ());
}

std::string
BuildWorkingMemoryTextPayload (Store &store, const ProcessorContext &p_ctx,
                               int max_slots, int max_chars)
{
  if (max_slots <= 0 || max_chars <= 0
      || (p_ctx.wm_slots.empty () && p_ctx.wm_recent_slots.empty ()))
    {
      return {};
    }

  std::string out;
  int slots_seen = 0;
  bool budget_exhausted = false;
  // Associative slots usually duplicate recent-ring content; the same blob
  // is appended once.
  std::set<std::vector<unsigned char>> seen_blobs;
  auto append_slot_text = [&] (const ProcessorContext::WMSlot &slot) {
    bool added_slot_text = false;
    for (auto rec_it = slot.signal_records.rbegin ();
         rec_it != slot.signal_records.rend (); ++rec_it)
      {
        if (rec_it->blob_id.empty ())
          {
            continue;
          }
        if (!seen_blobs.insert (rec_it->blob_id).second)
          {
            continue;
          }
        if (!rec_it->modality.empty () && rec_it->modality != "text"
            && rec_it->mime != "text/plain")
          {
            continue;
          }
        try
          {
            auto rows = store.Execute ("SELECT objstore_get(?1) AS payload",
                                       { rec_it->blob_id });
            if (rows.empty () || rows[0].count ("payload") == 0)
              {
                continue;
              }
            const auto payload = store::BlobFromAny (rows[0].at ("payload"));
            if (payload.empty ())
              {
                continue;
              }
            if (!out.empty ())
              {
                out.push_back ('\n');
              }
            const int remaining = max_chars - static_cast<int> (out.size ());
            if (remaining <= 0)
              {
                budget_exhausted = true;
                return;
              }
            const int take = std::min<int> (
                remaining, static_cast<int> (payload.size ()));
            out.append (reinterpret_cast<const char *> (payload.data ()),
                        static_cast<size_t> (take));
            added_slot_text = true;
            if (static_cast<int> (out.size ()) >= max_chars)
              {
                budget_exhausted = true;
                return;
              }
          }
        catch (...)
          {
          }
      }
    if (added_slot_text)
      {
        ++slots_seen;
      }
  };

  // The recent FIFO slice carries the newest conversational turns; walk it
  // newest-first before the associative slots.
  for (auto slot_it = p_ctx.wm_recent_slots.rbegin ();
       slot_it != p_ctx.wm_recent_slots.rend () && slots_seen < max_slots
       && !budget_exhausted;
       ++slot_it)
    {
      append_slot_text (*slot_it);
    }
  for (auto slot_it = p_ctx.wm_slots.rbegin ();
       slot_it != p_ctx.wm_slots.rend () && slots_seen < max_slots
       && !budget_exhausted;
       ++slot_it)
    {
      append_slot_text (*slot_it);
    }
  return out;
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

double
ResolveResurfacingDecayScale (
    Transaction &tx,
    const temporal::RetrievalAblationOverride &retrieval_override,
    double stability)
{
  const auto mode = temporal::ResolveResurfacingDecayMode (retrieval_override);
  if (mode == temporal::ResurfacingDecayMode::TimeOnly)
    {
      return 1.0;
    }

  const auto eviction_override = eviction::GetEvictionAblationOverride ();
  const auto pressure_state
      = pressure::ComputeStoragePressureState (tx, eviction_override);
  switch (mode)
    {
    case temporal::ResurfacingDecayMode::PressureGate:
      return pressure::GateScale (
          pressure_state, core::RetrievalPressureGateLowScale (stability));
    case temporal::ResurfacingDecayMode::PressureRamp:
      return pressure::RampScale (
          pressure_state, core::RetrievalPressureRampLowScale (stability));
    case temporal::ResurfacingDecayMode::TimeOnly:
      break;
    }
  return 1.0;
}

double
ApplyFreshnessPenaltyScale (double freshness, double penalty_scale)
{
  return core::Clamp (1.0 - penalty_scale * (1.0 - freshness), 0.0, 1.0);
}

store::FactRecord
BuildFactRecord (const std::map<std::string, std::any> &row, double focus,
                 double sensitivity, double stability)
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
    record.confidence = AnyToDouble (
        it->second,
        core::RetrievalFactMissingConfidence (focus, sensitivity, stability));
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
EvidenceWeight (const std::string &evidence_type, double support_weight,
                double focus, double sensitivity, double stability)
{
  const double type_weight = core::FactRetrievalEvidenceTypeWeight (
      focus, sensitivity, stability, evidence_type.c_str ());
  return core::Clamp (
      type_weight
          * core::Clamp (
              support_weight,
              core::FactRetrievalEvidenceSupportFloor (focus, sensitivity,
                                                       stability),
              1.0),
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
FactBoostMultiplier (const temporal::RetrievalAblationOverride &override,
                     double focus, double sensitivity, double stability)
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
      return core::RetrievalFactBoostWeakMultiplier (focus, sensitivity,
                                                     stability);
    case temporal::FactBoostStrength::Strong:
      return core::RetrievalFactBoostStrongMultiplier (focus, sensitivity,
                                                       stability);
    }
  return 1.0;
}

double
StalePenaltyMultiplier (const temporal::RetrievalAblationOverride &override,
                        double focus, double sensitivity, double stability)
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
      return core::RetrievalStalePenaltyStrongMultiplier (focus, sensitivity,
                                                          stability);
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
                               const temporal::RetrievalAblationOverride &override,
                               double focus, double sensitivity,
                               double stability)
{
  if (mode != temporal::RetrievalMode::Current)
    {
      return 1.0;
    }

  const double routine_affinity
      = store::PredicateRoutineAffinity (record.canonical_predicate, focus,
                                         sensitivity, stability);
  const auto policy = core::RetrievalRoutineRecencyAdjustmentPolicy (
      focus, sensitivity, stability);
  switch (ResolveRoutineRecencyMode (override))
    {
    case temporal::RoutineRecencyMode::Balanced:
      return 1.0;
    case temporal::RoutineRecencyMode::RoutineBiased:
      return core::Clamp (
          1.0 + policy.routine_boost_weight * routine_affinity
                    * score.routine_support
              - policy.routine_recency_penalty_weight * score.recency_support,
          policy.routine_min_multiplier, policy.routine_max_multiplier);
    case temporal::RoutineRecencyMode::RecencyBiased:
      return core::Clamp (
          1.0 + policy.recency_boost_weight * score.recency_support
              - policy.recency_routine_penalty_weight * routine_affinity
                    * score.routine_support,
          policy.recency_min_multiplier, policy.recency_max_multiplier);
    }
  return 1.0;
}

double
RoutineRecencyStaleAdjustment (
    const store::FactRecord &record, const store::FactScore &score,
    temporal::RetrievalMode mode,
    const temporal::RetrievalAblationOverride &override, double focus,
    double sensitivity, double stability)
{
  if (mode != temporal::RetrievalMode::Current)
    {
      return 1.0;
    }

  const double routine_affinity
      = store::PredicateRoutineAffinity (record.canonical_predicate, focus,
                                         sensitivity, stability);
  const auto policy = core::RetrievalRoutineRecencyAdjustmentPolicy (
      focus, sensitivity, stability);
  switch (ResolveRoutineRecencyMode (override))
    {
    case temporal::RoutineRecencyMode::Balanced:
      return 1.0;
    case temporal::RoutineRecencyMode::RoutineBiased:
      return core::Clamp (
          1.0 - policy.stale_routine_penalty_weight * routine_affinity
                    * score.routine_support,
          policy.stale_routine_min_multiplier,
          policy.stale_routine_max_multiplier);
    case temporal::RoutineRecencyMode::RecencyBiased:
      return core::Clamp (
          1.0
              + policy.stale_recency_boost_weight
                    * std::max (score.recency_support,
                                policy.stale_recency_floor)
              + policy.stale_recency_routine_weight * routine_affinity,
          policy.stale_recency_min_multiplier,
          policy.stale_recency_max_multiplier);
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

  const auto fact_candidate_policy
      = core::RetrievalFactCandidateScoringWeights (focus, sensitivity,
                                                    stability);
  const double fact_weight = fact_candidate_policy.boost_weight
                             * FactBoostMultiplier (override, focus,
                                                    sensitivity, stability);
  const double stale_weight = fact_candidate_policy.stale_penalty_weight
                              * StalePenaltyMultiplier (override, focus,
                                                        sensitivity,
                                                        stability);
  const auto stale_policy
      = core::RetrievalFactStaleScoringPolicy (focus, sensitivity, stability);
  const auto boost_policy
      = core::RetrievalFactBoostScoringPolicy (focus, sensitivity, stability);

  for (const auto &link : links)
    {
      const double similarity = core::Clamp (link.query_similarity, 0.0, 1.0);
      const double confidence
          = core::Clamp (link.record.confidence, 0.0, 1.0);
      const double provenance
          = EvidenceWeight (link.evidence_type, link.support_weight, focus,
                            sensitivity, stability);
      const double criticality = store::PredicateCriticality (
          link.record.canonical_predicate, focus, sensitivity, stability);
      const double lifecycle
          = core::Clamp (link.fact_score.lifecycle_support, 0.0, 1.0);

      if (link.stale)
        {
          if (mode == temporal::RetrievalMode::Current)
            {
              const double stale_adjust = RoutineRecencyStaleAdjustment (
                  link.record, link.fact_score, mode, override, focus,
                  sensitivity, stability);
              const double penalty
                  = stale_weight
                    * (stale_policy.similarity_weight * similarity
                       + stale_policy.confidence_weight * confidence)
                    * criticality * provenance
                    * std::max (stale_policy.lifecycle_floor, lifecycle)
                    * stale_adjust;
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
          link.record, link.fact_score, mode, override, focus, sensitivity,
          stability);
      const double signal = (boost_policy.similarity_weight * similarity
                             + boost_policy.confidence_weight * confidence
                             + boost_policy.support_weight * support)
                            * ((1.0 - boost_policy.temporal_weight)
                               + boost_policy.temporal_weight * temporal_match)
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
  retrieval_debug::ClearLastRejectedCandidates ();
  retrieval_debug::ClearLastEvidencePackets ();
  retrieval_debug::ClearLastRetrievalSummary ();
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

  if (EnvFlag ("CORTEXT_ABLATION_DISABLE_LTM_RETRIEVAL"))
    {
      context.AddOperationTiming (
          "GraphRetrieve.ablation_disable_ltm_retrieval", 0.0);
      return;
    }

  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &signal = context.GetSignal ();

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

  const double ctx_mix = core::RetrievalContextMix (
      cfg.focus, cfg.sensitivity, cfg.stability);
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

  const int base_seed_k = std::max (1, core::RetrievalMaxResults (cfg.focus));
  const int base_k = std::max (
      1, core::RetrievalGraphExpandedRagMaxItems (cfg.focus, cfg.stability));
  const int base_depth = std::max (1, core::RetrievalGraphDepth (cfg.stability));
  const double min_edge_weight = core::MinEdgeWeight (cfg.focus);
  const int k_key
      = core::SparseKeySize (cfg.focus, cfg.sensitivity, cfg.stability);
  const std::string sparse_key = core::SparseKey (q, k_key);
  const auto retrieval_override = temporal::GetRetrievalOverride ();
  const auto requested_retrieval_mode
      = retrieval_override.mode.value_or (temporal::RetrievalMode::Current);
  const auto ablation_override = temporal::GetRetrievalAblationOverride ();
  const auto resurfacing_decay_mode
      = temporal::ResolveResurfacingDecayMode (ablation_override);
  const auto retrieval_mode
      = temporal::ResolveRetrievalMode (requested_retrieval_mode);
  const std::uint64_t retrieval_ts
      = temporal::ResolveRetrievalTimestamp (requested_retrieval_mode,
                                            retrieval_override.timestamp,
                                            signal.timestamp);
  const double resurfacing_decay_scale
      = ResolveResurfacingDecayScale (tx, ablation_override, cfg.stability);
  const auto fact_query_mode = temporal::ResolveFactQueryMode (
      temporal::ToFactQueryMode (retrieval_mode));
  const bool fact_layer_enabled
      = temporal::IsFactLayerEnabled () && !EnvFlag ("CORTEXT_DISABLE_FACTS");
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

  int seed_k = base_seed_k;
  int k = base_k;
  int depth = base_depth;
  if (metacognitive_mode == ProcessorContext::MetacognitiveMode::TotRecovery)
    {
      const double tot_expansion_factor = core::RetrievalTotExpansionFactor (
          cfg.focus, cfg.sensitivity, cfg.stability, tot_confidence_scale);
      seed_k = std::max (
          base_seed_k,
          static_cast<int> (std::ceil (tot_expansion_factor * base_seed_k)));
      k = std::max (
          base_k,
          static_cast<int> (std::ceil (tot_expansion_factor * base_k)));
      depth = std::min (
          core::RetrievalTotMaxDepth (cfg.focus, cfg.stability),
          base_depth
              + (tot_confidence_scale
                         >= core::RetrievalTotDepthConfidenceThreshold (
                             cfg.focus, cfg.sensitivity, cfg.stability)
                     ? 1
                     : 0));
    }
  const double predictive_weight
      = disable_predictive_bonus
            ? 0.0
            : core::RetrievalPredictiveBonusWeight (
                  cfg.focus, cfg.sensitivity, cfg.stability);
  const bool preconsolidated_label_graph_enabled
      = !EnvFlag ("CORTEXT_DISABLE_PRECONSOLIDATED_LABEL_GRAPH");
  const double label_graph_weight
      = preconsolidated_label_graph_enabled
            ? core::RetrievalPreconsolidatedLabelGraphWeight (
                cfg.focus, cfg.sensitivity, cfg.stability)
            : 0.0;
  const int label_graph_top_labels = std::max (
      1, core::RetrievalPreconsolidatedLabelGraphTopLabels (cfg.focus,
                                                            cfg.stability));
  const double label_graph_min_query_score
      = core::RetrievalPreconsolidatedLabelGraphMinQueryScore (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const double label_graph_relation_weight
      = core::RetrievalPreconsolidatedLabelRelationWeight (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const double label_graph_degree_damping
      = core::RetrievalPreconsolidatedLabelGraphDegreeDamping (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const int label_graph_seed_sources = std::max (
      0, core::RetrievalPreconsolidatedLabelGraphSeedSources (cfg.focus,
                                                             cfg.stability));
  const int durable_source_text_seed_sources = std::max (
      0, core::RetrievalDurableSourceTextSeedSources (cfg.focus,
                                                     cfg.stability));
  const int durable_source_text_search_limit = std::max (
      durable_source_text_seed_sources,
      core::RetrievalDurableSourceTextSearchLimit (
          cfg.focus, cfg.sensitivity, cfg.stability));
  const double durable_source_text_min_score
      = core::RetrievalDurableSourceTextMinScore (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const int durable_source_text_max_bytes = std::max (
      1, core::RetrievalDurableSourceTextMaxBytes (
             cfg.focus, cfg.sensitivity, cfg.stability));
  const int fact_text_seed_count = std::max (
      0, core::RetrievalFactTextSeedCount (
             cfg.focus, cfg.sensitivity, cfg.stability));
  const int fact_text_search_limit = std::max (
      fact_text_seed_count,
      core::RetrievalFactTextSearchLimit (
          cfg.focus, cfg.sensitivity, cfg.stability,
          std::max (fact_text_seed_count, 1)));
  const double fact_text_seed_min_score
      = core::RetrievalFactTextSeedMinScore (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const bool source_seed_graph_expansion_enabled
      = !EnvFlag ("CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION");
  const bool temporal_retrieval_enabled
      = !EnvFlag ("CORTEXT_DISABLE_TEMPORAL_RETRIEVAL");
  const int graph_expanded_temporal_window
      = temporal_retrieval_enabled
            ? std::max (0, core::RetrievalGraphExpandedRagTemporalWindow (
                               cfg.focus, cfg.stability))
            : 0;
  const double graph_expanded_graph_weight
      = core::RetrievalGraphExpandedRagGraphWeight (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const double graph_expanded_relation_weight
      = core::RetrievalGraphExpandedRagRelationWeight (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const double graph_expanded_fact_weight
      = core::RetrievalGraphExpandedRagFactWeight (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const double temporal_rank_tau_seconds
      = core::RetrievalTemporalRankTauSeconds (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const double temporal_rank_weight
      = temporal_retrieval_enabled
            ? core::RetrievalTemporalRankWeight (
                cfg.focus, cfg.sensitivity, cfg.stability)
            : 0.0;
  const double procedural_seed_min_score
      = core::RetrievalProceduralSeedMinScore (cfg.focus, cfg.sensitivity,
                                               cfg.stability);
  const int text_query_wm_slots
      = std::max (0, core::RetrievalTextQueryWMSlots (cfg.focus,
                                                      cfg.stability));
  const int text_query_wm_chars
      = std::max (0, core::RetrievalTextQueryWMChars (cfg.focus,
                                                     cfg.stability));
  const int label_text_route_limit = core::RetrievalLabelTextRouteLimit (
      cfg.focus, cfg.sensitivity, cfg.stability, label_graph_top_labels);
  const int graph_expansion_row_limit
      = core::RetrievalGraphExpansionRowLimit (
          cfg.focus, cfg.sensitivity, cfg.stability, base_k);
  const int graph_expansion_fanout = core::RetrievalGraphExpansionFanout (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const int label_graph_fanout = core::RetrievalLabelGraphFanout (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const int fact_evidence_fanout = core::RetrievalFactEvidenceFanout (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const int route_token_min_chars = core::RetrievalRouteTokenMinChars (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const bool label_token_text_route_enabled
      = core::RetrievalLabelTokenTextRouteEnabled (
          cfg.focus, cfg.sensitivity, cfg.stability);
  std::string query_text_payload = SignalTextPayload (signal);
  int appended_wm_chars = 0;
  int query_text_token_count = 0;
  if (text_query_wm_slots > 0 && text_query_wm_chars > 0)
    {
      const std::string wm_text = BuildWorkingMemoryTextPayload (
          *store, p_ctx, text_query_wm_slots, text_query_wm_chars);
      if (!wm_text.empty ())
        {
          if (!query_text_payload.empty ())
            {
              query_text_payload.push_back ('\n');
            }
          query_text_payload += wm_text;
          appended_wm_chars = static_cast<int> (wm_text.size ());
        }
    }
	  query_text_token_count
	      = static_cast<int> (
          RouteTokens (query_text_payload, route_token_min_chars).size ());

  auto temporal_rank_score = [&] (long long event_ts) {
    if (temporal_rank_weight <= 0.0 || temporal_rank_tau_seconds <= 0.0
        || signal.timestamp == 0 || event_ts <= 0)
      {
        return 0.0;
      }
    if (static_cast<std::uint64_t> (event_ts) >= signal.timestamp)
      {
        return 1.0;
      }
    const double age_s
        = static_cast<double> (signal.timestamp
                               - static_cast<std::uint64_t> (event_ts))
          / 1000.0;
    return core::Clamp (std::exp (-age_s / temporal_rank_tau_seconds), 0.0,
                        1.0);
  };

  // Seed vector retrieval via sqlite-vec KNN query.
  struct Scored
  {
    long long embedding_id;
    long long memory_id;
    long long created_at;
    long long last_access;
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
    double label_graph_boost;
    int label_match_count;
    double durable_source_boost;
    int durable_source_count;
    bool is_association;
    bool is_label;
    Eigen::VectorXf vec;
    Eigen::VectorXf ctx;
    double temporal_score = 0.0;
    long long retrieved_count = 0;
    long long used_count = 0;
    std::string source_id;
    std::string modality;
  };
  std::vector<Scored> seeds;
  seeds.reserve (static_cast<size_t> (seed_k));
  std::unordered_map<long long, double> durable_source_text_seed_scores;
  std::unordered_map<long long, double> procedural_seed_scores;
  int64_t procedural_seed_count = 0;
  int64_t hierarchical_label_seed_count = 0;

  const std::string latest_reconstruction_join
      = disable_constructive_recall
            ? ""
            : "LEFT JOIN memory_reconstructions rc "
              "  ON rc.memory_id = m.memory_id "
              " AND rc.reconstruction_id = ("
              "   SELECT mr2.reconstruction_id "
              "   FROM memory_reconstructions mr2 "
              "   WHERE mr2.memory_id = m.memory_id "
              "   ORDER BY mr2.reconstruction_id DESC "
              "   LIMIT 1"
              " ) ";
  const char *current_embedding_expr
      = disable_constructive_recall ? "m.embedding_id"
                                    : "COALESCE(rc.embedding_id, m.embedding_id)";

  // Convert query vector to std::vector<float> for parameter binding.
  std::vector<float> q_vec (q.data (), q.data () + q.size ());
  const int seed_search_k = core::RetrievalSeedSearchK (
      cfg.focus, cfg.sensitivity, cfg.stability, seed_k);

  std::unordered_set<long long> seen_seed_embeddings;
  const auto &summary_cache = p_ctx.summary_cache;
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
        auto it_last_access = row.find ("last_access");
        auto it_retrieved_count = row.find ("retrieved_count");
        auto it_used_count = row.find ("used_count");
        auto it_event_ts = row.find ("event_ts");
        auto it_mem_id = row.find ("memory_id");
        auto it_kind = row.find ("kind");
        auto it_source_id = row.find ("source_id");
        auto it_modality = row.find ("modality");
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
            sim = core::RetrievalVectorDistanceScore (
                dist, cfg.focus, cfg.sensitivity, cfg.stability);
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
        long long last_access = 0;
        if (it_last_access != row.end ()
            && it_last_access->second.type () == typeid (long long))
          {
            last_access = std::any_cast<long long> (it_last_access->second);
          }
        const long long retrieved_count
            = std::max (0LL,
                        (it_retrieved_count != row.end ())
                            ? store::AnyToLongLong (it_retrieved_count->second)
                                  .value_or (0)
                            : 0LL);
        const long long used_count
            = std::max (0LL,
                        (it_used_count != row.end ())
                            ? store::AnyToLongLong (it_used_count->second)
                                  .value_or (0)
                            : 0LL);
        long long event_ts = created_at;
        if (it_event_ts != row.end ()
            && it_event_ts->second.type () == typeid (long long))
          {
            event_ts = std::any_cast<long long> (it_event_ts->second);
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
        const std::string memory_source_id
            = it_source_id != row.end () ? AnyToString (it_source_id->second)
                                         : std::string ();
        const std::string modality
            = it_modality != row.end () ? AnyToString (it_modality->second)
                                        : std::string ();

        seeds.push_back (Scored{ emb_id,
                                 mem_id,
                                 created_at,
                                 last_access,
                                 sim,
                                 0.0,
                                 core::RetrievalSeedFallbackSourceConfidence (
                                     cfg.focus, cfg.sensitivity,
                                     cfg.stability),
                                 0.0,
                                 pre_activation,
                                 predictive_weight * pre_activation,
                                 0,
                                 0.0,
                                 0.0,
                                 0.0,
                                 0.0,
                                 0,
                                 0.0,
                                 0,
                                 0.0,
                                 0,
                                 is_association,
                                 is_label,
                                 Eigen::VectorXf (),
                                 Eigen::VectorXf (),
                                 0.0,
                                 0LL,
                                 0LL,
                                 std::string (),
                                 std::string () });
        seeds.back ().temporal_score = temporal_rank_score (event_ts);
        seeds.back ().retrieved_count = retrieved_count;
        seeds.back ().used_count = used_count;
        seeds.back ().source_id = memory_source_id;
        seeds.back ().modality = modality;
      }
  };

  std::string seed_sql;
  if (disable_constructive_recall)
    {
      seed_sql
          = "WITH seed AS ("
            "  SELECT embedding_id, distance, created_at "
            "  FROM embeddings "
            "  WHERE embedding MATCH ? AND k = ? "
            "  ORDER BY distance"
            ") "
            "SELECT m.embedding_id AS base_embedding_id, seed.distance, "
              "COALESCE(m.created_at, seed.created_at, 0) AS created_at, "
              "COALESCE(m.last_access, 0) AS last_access, "
              "COALESCE(m.retrieved_count, 0) AS retrieved_count, "
              "COALESCE(m.used_count, 0) AS used_count, "
              "COALESCE(NULLIF(m.start_ts, 0), m.created_at, "
            "         seed.created_at, 0) AS event_ts, "
            "m.memory_id, m.kind, m.source_id, m.modality, "
            "COALESCE(m.pre_activation, 0.0) AS pre_activation "
            "FROM seed "
            "JOIN memories m ON m.embedding_id = seed.embedding_id "
            "WHERE m.kind != 'WORKING' "
            "AND m.kind != 'LABEL' "
            "AND (m.kind != 'ASSOCIATION' OR EXISTS ("
            "  SELECT 1 FROM associations a "
            "  WHERE a.source_memory_id = m.memory_id "
            "    AND a.edge_type = 'derived_from')) "
            "ORDER BY seed.distance";
    }
  else
    {
      seed_sql
          = "WITH seed AS ("
            "  SELECT embedding_id, distance, created_at "
            "  FROM embeddings "
            "  WHERE embedding MATCH ? AND k = ? "
            "  ORDER BY distance"
            "), seed_memory AS ("
            "  SELECT seed.embedding_id AS seed_embedding_id, seed.distance, "
            "         seed.created_at AS seed_created_at, "
            "         m.memory_id, m.embedding_id AS base_embedding_id "
            "  FROM seed "
            "  JOIN memories m ON m.embedding_id = seed.embedding_id "
            "  UNION ALL "
            "  SELECT seed.embedding_id AS seed_embedding_id, seed.distance, "
            "         seed.created_at AS seed_created_at, "
            "         mr.memory_id, m.embedding_id AS base_embedding_id "
            "  FROM seed "
            "  JOIN memory_reconstructions mr "
            "       ON mr.embedding_id = seed.embedding_id "
            "  JOIN memories m ON m.memory_id = mr.memory_id "
            "  WHERE mr.reconstruction_id = ("
            "    SELECT MAX(mr2.reconstruction_id) "
            "    FROM memory_reconstructions mr2 "
            "    WHERE mr2.memory_id = mr.memory_id"
            "  )"
            ") "
            "SELECT sm.base_embedding_id, sm.distance, "
              "COALESCE(m.created_at, sm.seed_created_at, 0) AS created_at, "
              "COALESCE(m.last_access, 0) AS last_access, "
              "COALESCE(m.retrieved_count, 0) AS retrieved_count, "
              "COALESCE(m.used_count, 0) AS used_count, "
              "COALESCE(NULLIF(m.start_ts, 0), m.created_at, "
            "         sm.seed_created_at, 0) AS event_ts, "
            "m.memory_id, m.kind, m.source_id, m.modality, "
            "COALESCE(m.pre_activation, 0.0) AS pre_activation "
            "FROM seed_memory sm "
            "JOIN memories m ON m.memory_id = sm.memory_id "
            "WHERE m.kind != 'WORKING' "
            "AND m.kind != 'LABEL' "
            "AND (m.kind != 'ASSOCIATION' OR EXISTS ("
            "  SELECT 1 FROM associations a "
            "  WHERE a.source_memory_id = m.memory_id "
            "    AND a.edge_type = 'derived_from')) "
            "ORDER BY sm.distance";
    }
  append_seeds (seed_sql, { q_vec, static_cast<long long> (seed_search_k) },
                "GraphRetrieve.seed_sql");

  const int k_summary = core::RetrievalSummaryLabelSeedCount (
      cfg.focus, cfg.sensitivity, cfg.stability);
  auto build_ranked_query_label_ids = [&] {
    std::map<long long, double> query_label_scores;
    if (!preconsolidated_label_graph_enabled || label_graph_weight <= 0.0)
      {
        return std::vector<std::pair<double, long long>> {};
      }
    auto add_dynamic_label_by_key = [&] (const std::string &label_key,
                                         double query_score) {
      if (label_key.empty () || query_score <= 0.0)
        {
          return;
        }
      try
        {
          auto rows = store->Execute (
              "SELECT memory_id FROM memories "
              "WHERE kind = 'LABEL' AND source_id = ? LIMIT 1",
              { label_key });
          if (rows.empty ())
            {
              return;
            }
          const long long label_id = AnyToInt64 (rows[0].at ("memory_id"));
          if (label_id <= 0)
            {
              return;
            }
          auto &score = query_label_scores[label_id];
          score = std::max (score, query_score);
        }
      catch (...)
        {
        }
    };
    try
      {
        bool label_bank_attached = false;
        auto dbs = store->Execute ("PRAGMA database_list", {});
        for (const auto &row : dbs)
          {
            auto it = row.find ("name");
            if (it != row.end () && it->second.type () == typeid (std::string)
                && std::any_cast<std::string> (it->second)
                       == "cortext_label_bank")
              {
                label_bank_attached = true;
                break;
              }
          }
	        if (label_bank_attached)
	          {
	            const int static_label_k
	                = core::RetrievalLabelBankStaticSearchK (
	                    cfg.focus, cfg.sensitivity, cfg.stability,
	                    label_graph_top_labels);
            auto rows = store->Execute (
                "SELECT key, distance "
                "FROM cortext_label_bank.label_bank_vec "
                "WHERE embedding MATCH ? AND k = ? "
                "ORDER BY distance",
                { q_vec, static_cast<long long> (static_label_k) });
            for (const auto &row : rows)
              {
                auto key_it = row.find ("key");
                if (key_it == row.end ()
                    || key_it->second.type () != typeid (std::string))
                  {
                    continue;
                  }
                const double distance
                    = row.count ("distance")
                              && row.at ("distance").type () == typeid (double)
                          ? std::any_cast<double> (row.at ("distance"))
                          : 0.0;
                const double query_score = core::Clamp (
                    core::RetrievalVectorDistanceScore (
                        distance, cfg.focus, cfg.sensitivity, cfg.stability),
                    0.0, 1.0);
                if (query_score >= label_graph_min_query_score)
                  {
                    add_dynamic_label_by_key (
                        std::any_cast<std::string> (key_it->second),
                        query_score);
                  }
              }
          }
      }
    catch (...)
      {
      }
    for (const auto &entry : summary_cache)
      {
        if (!entry.is_label || entry.embedding.size () != q.size ()
            || entry.embedding_norm <= 1e-9f)
          {
            continue;
          }
        const double sim
            = static_cast<double> (entry.embedding.dot (q))
              / static_cast<double> (entry.embedding_norm);
        const double query_score = core::Clamp (std::max (0.0, sim), 0.0, 1.0);
        if (query_score < label_graph_min_query_score)
          {
            continue;
          }
        auto &score = query_label_scores[entry.memory_id];
        score = std::max (score, query_score);
      }

    const std::string query_text = CanonicalQueryText (query_text_payload);
      if (!query_text.empty ())
        {
          try
            {
	              auto rows = store->Execute (
	                "SELECT memory_id, COALESCE(label, source_id, '') AS label_text "
	                "FROM memories WHERE kind = 'LABEL' "
	                "ORDER BY created_at DESC, memory_id DESC LIMIT ?",
	                { static_cast<long long> (label_text_route_limit) });
            for (const auto &row : rows)
              {
                const long long label_id = AnyToInt64 (row.at ("memory_id"));
                const std::string label_text = AnyToString (
                    row.at ("label_text"));
                if (label_id <= 0 || label_text.empty ())
                  {
                    continue;
                  }
                const std::string label_query_text = CanonicalQueryText (
                    label_text);
                double text_score = 0.0;
                if (CanonicalPhraseAppears (query_text, label_query_text))
                  {
                    text_score = 1.0;
                  }
	                else if (label_token_text_route_enabled)
	                  {
	                    const auto label_query_tokens = RouteTokens (
	                        query_text, route_token_min_chars);
	                    text_score = RouteTokenOverlapScore (
	                        label_query_tokens, label_text, cfg.focus,
	                        cfg.sensitivity, cfg.stability,
	                        route_token_min_chars);
	                  }
                if (text_score <= 0.0)
                  {
                    continue;
                  }
                auto &score = query_label_scores[label_id];
                score = std::max (score, text_score);
              }
          }
        catch (...)
          {
          }
      }

    std::vector<std::pair<double, long long>> ranked_query_labels;
    ranked_query_labels.reserve (query_label_scores.size ());
    for (const auto &[label_id, score] : query_label_scores)
      {
        ranked_query_labels.emplace_back (score, label_id);
      }
    std::sort (ranked_query_labels.begin (), ranked_query_labels.end (),
               [] (const auto &a, const auto &b) { return a.first > b.first; });
    if (static_cast<int> (ranked_query_labels.size ()) > label_graph_top_labels)
      {
        ranked_query_labels.resize (
            static_cast<size_t> (label_graph_top_labels));
      }
    return ranked_query_labels;
  };
  auto ranked_query_label_ids = build_ranked_query_label_ids;
  if (k_summary > 0 && !summary_cache.empty ())
    {
      const auto t_start = std::chrono::steady_clock::now ();
      std::vector<std::pair<double, size_t>> ranked;
      ranked.reserve (summary_cache.size ());
      for (size_t i = 0; i < summary_cache.size (); ++i)
        {
	          const auto &entry = summary_cache[i];
	          if (entry.is_label)
	            {
	              continue;
	            }
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
                  0,
                  ranked[i].first,
                  0.0,
                  core::RetrievalSeedFallbackSourceConfidence (
                      cfg.focus, cfg.sensitivity, cfg.stability),
                  0.0,
                  0.0,
                  0.0,
                  0,
                  0.0,
                  0.0,
	                  0.0,
	                  0.0,
	                  0,
	                  0.0,
	                  0,
	                  0.0,
	                  0,
	                  entry.is_association,
	                  entry.is_label,
	                  entry.embedding,
                  Eigen::VectorXf (),
                  0.0,
                  0LL,
                  0LL,
                  std::string (),
                  std::string () });
            }
        }
      const auto t_end = std::chrono::steady_clock::now ();
      context.AddOperationTiming ("GraphRetrieve.summary_cache",
                                  elapsed_ms (t_start, t_end));
    }

  if (core::RetrievalClusterLabelEnabled (cfg.focus, cfg.sensitivity,
                                          cfg.stability)
      && !summary_cache.empty ())
    {
      const auto t_start = std::chrono::steady_clock::now ();
      const int requested_clusters = std::max (
          1, core::RetrievalClusterLabelK (cfg.focus, cfg.stability));
      const int cluster_take = std::max (
          1, core::RetrievalClusterLabelM (cfg.focus, cfg.stability));
      const int labels_per_cluster = std::max (
          1, core::RetrievalClusterLabelN (cfg.focus, cfg.sensitivity,
                                           cfg.stability));

      std::vector<size_t> label_indices;
      label_indices.reserve (summary_cache.size ());
      for (size_t i = 0; i < summary_cache.size (); ++i)
        {
          const auto &entry = summary_cache[i];
          if (!entry.is_label || entry.embedding.size () != q.size ()
              || entry.embedding_norm <= 1e-9f)
            {
              continue;
            }
          label_indices.push_back (i);
        }

      const int label_count = static_cast<int> (label_indices.size ());
      if (label_count > 0)
        {
          const int cluster_count = std::min (requested_clusters, label_count);
          std::vector<Eigen::VectorXf> centroids;
          centroids.reserve (static_cast<size_t> (cluster_count));
          for (int c = 0; c < cluster_count; ++c)
            {
              const int idx = (c * label_count) / cluster_count;
              centroids.push_back (
                  summary_cache[label_indices[static_cast<size_t> (idx)]].embedding);
            }

          std::vector<int> assignments (static_cast<size_t> (label_count), 0);
          const int cluster_iterations = std::max (
              1, core::STMLabelClusterIterations (cfg.focus, cfg.sensitivity,
                                                  cfg.stability));
          for (int iter = 0; iter < cluster_iterations; ++iter)
            {
              bool changed = false;
              for (int i = 0; i < label_count; ++i)
                {
                  const auto &v
                      = summary_cache[label_indices[static_cast<size_t> (i)]].embedding;
                  int best_cluster = 0;
                  double best_dist = std::numeric_limits<double>::max ();
                  for (int c = 0; c < cluster_count; ++c)
                    {
                      const double dist = static_cast<double> (
                          (v - centroids[static_cast<size_t> (c)]).squaredNorm ());
                      if (dist < best_dist)
                        {
                          best_dist = dist;
                          best_cluster = c;
                        }
                    }
                  if (assignments[static_cast<size_t> (i)] != best_cluster)
                    {
                      assignments[static_cast<size_t> (i)] = best_cluster;
                      changed = true;
                    }
                }

              std::vector<Eigen::VectorXf> next;
              std::vector<int> counts (static_cast<size_t> (cluster_count), 0);
              next.reserve (static_cast<size_t> (cluster_count));
              for (int c = 0; c < cluster_count; ++c)
                {
                  next.push_back (Eigen::VectorXf::Zero (q.size ()));
                }
              for (int i = 0; i < label_count; ++i)
                {
                  const int c = assignments[static_cast<size_t> (i)];
                  next[static_cast<size_t> (c)]
                      += summary_cache[label_indices[static_cast<size_t> (i)]].embedding;
                  counts[static_cast<size_t> (c)]++;
                }
              for (int c = 0; c < cluster_count; ++c)
                {
                  if (counts[static_cast<size_t> (c)] > 0)
                    {
                      next[static_cast<size_t> (c)]
                          /= static_cast<float> (counts[static_cast<size_t> (c)]);
                    }
                  else
                    {
                      next[static_cast<size_t> (c)] = centroids[static_cast<size_t> (c)];
                    }
                }
              centroids = std::move (next);
              if (!changed)
                {
                  break;
                }
            }

          std::vector<std::pair<double, int>> ranked_clusters;
          ranked_clusters.reserve (static_cast<size_t> (cluster_count));
          for (int c = 0; c < cluster_count; ++c)
            {
              ranked_clusters.emplace_back (
                  static_cast<double> ((centroids[static_cast<size_t> (c)] - q)
                                           .squaredNorm ()),
                  c);
            }
          std::sort (ranked_clusters.begin (), ranked_clusters.end (),
                     [] (const auto &a, const auto &b) {
                       return a.first < b.first;
                     });

          const int take_clusters
              = std::min (cluster_take, static_cast<int> (ranked_clusters.size ()));
          for (int rank = 0; rank < take_clusters; ++rank)
            {
              const int cluster_id = ranked_clusters[static_cast<size_t> (rank)].second;
              std::vector<std::pair<double, size_t>> ranked_labels;
              for (int i = 0; i < label_count; ++i)
                {
                  if (assignments[static_cast<size_t> (i)] != cluster_id)
                    {
                      continue;
                    }
                  const size_t cache_idx = label_indices[static_cast<size_t> (i)];
                  const auto &entry = summary_cache[cache_idx];
                  const double sim
                      = static_cast<double> (entry.embedding.dot (q))
                        / static_cast<double> (entry.embedding_norm);
                  ranked_labels.emplace_back (sim, cache_idx);
                }
              std::sort (ranked_labels.begin (), ranked_labels.end (),
                         [] (const auto &a, const auto &b) {
                           return a.first > b.first;
                         });
              const int take_labels
                  = std::min (labels_per_cluster,
                              static_cast<int> (ranked_labels.size ()));
              for (int i = 0; i < take_labels; ++i)
                {
                  const auto &[sim, cache_idx] = ranked_labels[static_cast<size_t> (i)];
                  const auto &entry = summary_cache[cache_idx];
                  if (!seen_seed_embeddings.insert (entry.embedding_id).second)
                    {
                      continue;
                    }
                  seeds.push_back (Scored{
                      entry.embedding_id,
                      entry.memory_id,
                      0,
                      0,
                      sim,
                      0.0,
                      core::RetrievalSeedFallbackSourceConfidence (
                          cfg.focus, cfg.sensitivity, cfg.stability),
                      0.0,
                      0.0,
                      0.0,
                      0,
                      0.0,
                      0.0,
	                      0.0,
	                      0.0,
	                      0,
	                      0.0,
	                      0,
	                      0.0,
	                      0,
	                      false,
	                      true,
	                      entry.embedding,
                      Eigen::VectorXf (),
                      0.0,
                      0LL,
                      0LL,
                      std::string (),
                      std::string () });
                  hierarchical_label_seed_count++;
                }
            }
        }

      const auto t_end = std::chrono::steady_clock::now ();
      context.AddOperationTiming ("GraphRetrieve.cluster_label_cache",
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
          for (const auto &[memory_id, score] : pit->second)
            {
              if (memory_id <= 0)
                {
                  continue;
                }
              const double clamped = core::Clamp (score, 0.0, 1.0);
              if (clamped < procedural_seed_min_score)
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
              const int k_proc = core::RetrievalProceduralSeedFanout (
                  cfg.focus, cfg.sensitivity, cfg.stability);
              if (static_cast<int> (ranked_proc.size ()) > k_proc)
                {
                  ranked_proc.resize (static_cast<size_t> (k_proc));
                }

              std::ostringstream sql;
              sql << "SELECT m.memory_id, "
                     << current_embedding_expr
                     << " AS embedding_id, "
                        "COALESCE(m.created_at, 0) AS created_at, "
                        "COALESCE(m.last_access, 0) AS last_access, "
                        "COALESCE(m.retrieved_count, 0) AS retrieved_count, "
                        "COALESCE(m.used_count, 0) AS used_count, "
                        "COALESCE(NULLIF(m.start_ts, 0), m.created_at, 0) "
                        "AS event_ts, "
                        "m.kind, m.source_id, m.modality, "
                        "COALESCE(m.pre_activation, 0.0) AS pre_activation "
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
                  if (mem_id <= 0 || emb_id <= 0)
                    {
                      continue;
                    }
                  const long long created_at = AnyToInt64 (row.at ("created_at"));
                  const long long last_access = AnyToInt64 (row.at ("last_access"));
                  const long long retrieved_count
                      = std::max (0LL, AnyToInt64 (row.at ("retrieved_count")));
                  const long long used_count
                      = std::max (0LL, AnyToInt64 (row.at ("used_count")));
                  const long long event_ts = AnyToInt64 (row.at ("event_ts"));
                  const std::string kind = AnyToString (row.at ("kind"));
                  const bool is_association = (kind == "ASSOCIATION");
                  const bool is_label = (kind == "LABEL");
                  const double pre_activation = core::Clamp (
                      AnyToDouble (row.at ("pre_activation"), 0.0), 0.0, 1.0);
                  const std::string memory_source_id
                      = AnyToString (row.at ("source_id"));
                  const std::string modality = AnyToString (row.at ("modality"));
                  const double proc_score
                      = proc_scores.count (mem_id) == 1 ? proc_scores[mem_id] : 0.0;
                  if (proc_score > 0.0)
                    {
                      auto &slot = procedural_seed_scores[mem_id];
                      slot = std::max (slot, proc_score);
                    }
                  if (!seen_seed_embeddings.insert (emb_id).second)
                    {
                      for (auto &seed : seeds)
                        {
                          if (seed.embedding_id != emb_id)
                            {
                              continue;
                            }
                          seed.proc_score
                              = std::max (seed.proc_score, proc_score);
                          seed.pre_activation
                              = std::max (seed.pre_activation, pre_activation);
                          seed.predictive_bonus
                              = std::max (seed.predictive_bonus,
                                          predictive_weight * pre_activation);
                          seed.last_access
                              = std::max (seed.last_access, last_access);
                          seed.retrieved_count = std::max (
                              seed.retrieved_count, retrieved_count);
                          seed.used_count
                              = std::max (seed.used_count, used_count);
                          if (seed.source_id.empty ())
                            {
                              seed.source_id = memory_source_id;
                            }
                          if (seed.modality.empty ())
                            {
                              seed.modality = modality;
                            }
                          break;
                        }
                      if (proc_score > 0.0)
                        {
                          ++procedural_seed_count;
                        }
                      continue;
                    }

                  seeds.push_back (Scored{ emb_id,
                                           mem_id,
                                           created_at,
                                           last_access,
                                           0.0,
                                           0.0,
                                           core::RetrievalSeedFallbackSourceConfidence (
                                               cfg.focus, cfg.sensitivity,
                                               cfg.stability),
                                           proc_score,
                                           pre_activation,
                                           predictive_weight * pre_activation,
                                           0,
                                           0.0,
                                           0.0,
                                           0.0,
                                           0.0,
                                           0,
                                           0.0,
                                           0,
                                           0.0,
                                           0,
                                           is_association,
                                           is_label,
                                           Eigen::VectorXf (),
                                           Eigen::VectorXf (),
                                           0.0,
                                           0LL,
                                           0LL,
                                           memory_source_id,
                                           modality });
                  seeds.back ().temporal_score = temporal_rank_score (event_ts);
                  seeds.back ().retrieved_count = retrieved_count;
                  seeds.back ().used_count = used_count;
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
  std::unordered_set<long long> seen_fact_ids;
  int fact_text_candidate_count = 0;
  int fact_text_rejected_low_score_count = 0;
  int fact_text_match_count = 0;
  double fact_text_best_score = 0.0;

  auto add_matched_fact = [&] (const store::FactRecord &record,
                               double query_similarity) {
    if (record.fact_id <= 0 || !seen_fact_ids.insert (record.fact_id).second)
      {
        return;
      }
	    matched_facts.push_back (
	        { record, store::ScoreFactRecord (record, fact_query_mode,
	                                          retrieval_ts, cfg.focus,
	                                          cfg.sensitivity, cfg.stability),
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
  };

  const int k_fact = core::RetrievalFactVectorSeedCount (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const int k_fact_search = core::RetrievalFactVectorSearchK (
      cfg.focus, cfg.sensitivity, cfg.stability, k_fact);
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

    for (const auto &row : rows)
      {
        const store::FactRecord record = BuildFactRecord (
            row, cfg.focus, cfg.sensitivity, cfg.stability);
        const auto it_dist = row.find ("distance");
        const double query_similarity
            = core::RetrievalVectorDistanceScore (
                it_dist != row.end () ? AnyToDouble (it_dist->second, 0.0)
                                      : 0.0,
                cfg.focus, cfg.sensitivity, cfg.stability);
        add_matched_fact (record, query_similarity);
      }
  }

  if (fact_layer_enabled && fact_text_seed_count > 0
      && fact_text_seed_min_score > 0.0 && !query_text_payload.empty ())
    {
      const auto query_tokens = RouteTokens (
          query_text_payload, route_token_min_chars);
      if (!query_tokens.empty ())
        {
          std::ostringstream sql;
          sql << "SELECT fa.fact_id, fa.subject, fa.predicate, fa.object, "
                 "fa.canonical_subject, fa.canonical_predicate, "
                 "fa.canonical_object, fa.valid_start_ts, fa.valid_end_ts, "
                 "fa.recorded_at_ts, fa.superseded_at_ts, fa.confidence, "
                 "fa.summary_memory_id, COALESCE(fc.embedding_id, 0) AS embedding_id, "
                 "COALESCE(fc.fact_text, fa.subject || ' ' || fa.predicate || ' ' || fa.object) "
                 "AS fact_text, COALESCE(fc.is_current, 0) AS is_current, "
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
                 "AS evidence_count "
                 "FROM fact_assertions fa "
                 "LEFT JOIN fact_cache fc ON fc.fact_id = fa.fact_id "
                 "WHERE EXISTS (SELECT 1 FROM fact_evidence fe "
                 "              JOIN memories m ON m.memory_id = fe.source_memory_id "
                 "              WHERE fe.fact_id = fa.fact_id) ";
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
	          sql << "ORDER BY fa.recorded_at_ts DESC, fa.fact_id DESC LIMIT ? ";

	          std::vector<std::any> params;
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
	          params.push_back (static_cast<long long> (fact_text_search_limit));

	          const auto t_start = std::chrono::steady_clock::now ();
          auto rows = store->Execute (sql.str (), params);
          const auto t_end = std::chrono::steady_clock::now ();
          context.AddOperationTiming ("GraphRetrieve.fact_text_seed_sql",
                                      elapsed_ms (t_start, t_end));
          fact_text_candidate_count = static_cast<int> (rows.size ());

          struct TextMatchedFact
          {
            store::FactRecord record;
            double score = 0.0;
          };
          std::vector<TextMatchedFact> text_matches;
          text_matches.reserve (rows.size ());
          for (const auto &row : rows)
            {
              const auto record = BuildFactRecord (
                  row, cfg.focus, cfg.sensitivity, cfg.stability);
	              const double score = RouteTokenOverlapScore (
	                  query_tokens, record.fact_text, cfg.focus, cfg.sensitivity,
	                  cfg.stability, route_token_min_chars);
              fact_text_best_score = std::max (fact_text_best_score, score);
              if (score < fact_text_seed_min_score)
                {
                  ++fact_text_rejected_low_score_count;
                  continue;
                }
              text_matches.push_back ({ record, score });
            }
          std::sort (text_matches.begin (), text_matches.end (),
                     [] (const TextMatchedFact &a,
                         const TextMatchedFact &b) {
                       if (a.score != b.score)
                         {
                           return a.score > b.score;
                         }
                       return a.record.fact_id > b.record.fact_id;
                     });
          if (text_matches.size ()
              > static_cast<size_t> (fact_text_seed_count))
            {
              text_matches.resize (
                  static_cast<size_t> (fact_text_seed_count));
            }
          fact_text_match_count = static_cast<int> (text_matches.size ());
          for (const auto &match : text_matches)
            {
              add_matched_fact (match.record, match.score);
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
      sql << ") ORDER BY fe.fact_id ASC, fe.support_weight DESC, "
             "fe.source_memory_id DESC LIMIT ?";
      params.push_back (static_cast<long long> (
          std::max (1, static_cast<int> (matched_facts.size ()))
          * std::max (1, fact_evidence_fanout)));

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
                AnyToDouble (
                    row.at ("support_weight"),
                    core::RetrievalFactMissingEvidenceSupportWeight (
                        cfg.focus, cfg.sensitivity, cfg.stability)),
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
      sql << " ORDER BY fa.confidence DESC, fe.support_weight DESC, "
             "COALESCE(fa.recorded_at_ts, 0) DESC, fe.source_memory_id DESC "
             "LIMIT ?";
      params.push_back (static_cast<long long> (
          core::RetrievalFactStaleExpansionLimit (
              cfg.focus, cfg.sensitivity, cfg.stability,
              static_cast<int> (matched_facts.size ()))));

      const auto t_start = std::chrono::steady_clock::now ();
      auto rows = store->Execute (sql.str (), params);
      const auto t_end = std::chrono::steady_clock::now ();
      context.AddOperationTiming ("GraphRetrieve.fact_stale_sql",
                                  elapsed_ms (t_start, t_end));

      for (const auto &row : rows)
        {
          const store::FactRecord record = BuildFactRecord (
              row, cfg.focus, cfg.sensitivity, cfg.stability);
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
	                                        retrieval_ts, cfg.focus,
	                                        cfg.sensitivity, cfg.stability),
	                it->second,
                AnyToString (row.at ("evidence_type")),
                AnyToDouble (
                    row.at ("support_weight"),
                    core::RetrievalFactMissingEvidenceSupportWeight (
                        cfg.focus, cfg.sensitivity, cfg.stability)),
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
  std::vector<long long> source_seed_memory_ids;
  std::unordered_set<long long> seen_source_seed_memory_ids;
  std::unordered_map<long long, double> source_seed_expansion_scores;
  struct SourceTemporalAnchor
  {
    long long memory_id = 0;
    std::string source_id;
    long long start_ts = 0;
  };
  std::vector<SourceTemporalAnchor> source_temporal_anchors;
  std::unordered_set<std::string> seen_source_temporal_anchors;
  int64_t wm_temporal_anchor_count = 0;
  auto add_source_temporal_anchor = [&] (long long memory_id,
                                         const std::string &source_id,
                                         long long start_ts,
                                         bool from_working_memory) {
    if (source_id.empty () || start_ts <= 0)
      {
        return;
      }
    if (from_working_memory && signal.timestamp > 0
        && static_cast<std::uint64_t> (start_ts) >= signal.timestamp)
      {
        return;
      }
    const std::string key = source_id + "\n" + std::to_string (start_ts);
    if (!seen_source_temporal_anchors.insert (key).second)
      {
        return;
      }
    source_temporal_anchors.push_back ({ memory_id, source_id, start_ts });
    if (from_working_memory)
      {
        ++wm_temporal_anchor_count;
      }
  };
  for (const auto &s : seeds)
    {
      if (s.memory_id > 0)
        {
          expanded_memory_ids.insert (s.memory_id);
          if (!s.is_association && !s.is_label
              && seen_source_seed_memory_ids.insert (s.memory_id).second)
            {
              source_seed_memory_ids.push_back (s.memory_id);
            }
        }
    }
  for (const auto &[memory_id, links] : candidate_fact_links)
    {
      if (!links.empty () && memory_id > 0)
        {
          expanded_memory_ids.insert (memory_id);
        }
    }

  if (source_seed_graph_expansion_enabled
      && (!source_seed_memory_ids.empty () || !p_ctx.wm_slots.empty ()))
    {
      try
        {
          const auto t_start = std::chrono::steady_clock::now ();
          if (!source_seed_memory_ids.empty ())
            {
              std::string seed_sql
                  = "SELECT memory_id, source_id, start_ts FROM memories "
                    "WHERE memory_id IN (";
              std::vector<std::any> seed_params;
              seed_params.reserve (source_seed_memory_ids.size ());
              for (size_t i = 0; i < source_seed_memory_ids.size (); ++i)
                {
                  if (i > 0)
                    seed_sql += ",";
                  seed_sql += "?";
                  seed_params.push_back (source_seed_memory_ids[i]);
                }
              seed_sql += ") AND kind NOT IN ('WORKING', 'LABEL', 'ASSOCIATION')";
              const auto seed_rows = store->Execute (seed_sql, seed_params);
              for (const auto &seed_row : seed_rows)
                {
                  add_source_temporal_anchor (
                      AnyToInt64 (seed_row.at ("memory_id")),
                      AnyToString (seed_row.at ("source_id")),
                      AnyToInt64 (seed_row.at ("start_ts")), false);
                }
            }
          for (const auto &slot : p_ctx.wm_slots)
            {
              add_source_temporal_anchor (
                  slot.memory_id, slot.source_id, slot.start_ts, true);
              for (const auto &record : slot.signal_records)
                {
                  add_source_temporal_anchor (
                      slot.memory_id, slot.source_id,
                      static_cast<long long> (record.timestamp), true);
                }
            }
          if (graph_expanded_temporal_window > 0)
            {
              for (const auto &anchor : source_temporal_anchors)
                {
                  if (anchor.source_id.empty () || anchor.start_ts <= 0)
                    {
                      continue;
                    }
                  // Top-N by ABS(start_ts - anchor) is the merge of the N
                  // nearest rows at-or-below the anchor and the N nearest
                  // above it; both halves are index range scans on
                  // idx_memories_source_start (no full per-source sort).
                  // The outer ORDER BY uses the original keys, so result
                  // set and order are unchanged.
                  const long long temporal_limit = static_cast<long long> (
                      graph_expanded_temporal_window);
                  const bool has_ts_cap = signal.timestamp > 0;
                  const long long ts_cap
                      = has_ts_cap
                            ? static_cast<long long> (
                                  std::min<std::uint64_t> (
                                      signal.timestamp,
                                      static_cast<std::uint64_t> (
                                          std::numeric_limits<
                                              long long>::max ())))
                            : 0;
                  const char *ts_cap_clause
                      = has_ts_cap ? "  AND start_ts < ? " : "";
                  std::string temporal_sql
                      = std::string (
                            "SELECT memory_id, "
                            "       ABS(start_ts - ?) AS distance, start_ts "
                            "FROM ("
                            "  SELECT memory_id, start_ts FROM ("
                            "    SELECT memory_id, start_ts FROM memories "
                            "    WHERE source_id = ? AND memory_id != ? "
                            "      AND kind NOT IN ('WORKING', 'LABEL', "
                            "'ASSOCIATION') "
                            "      AND start_ts <= ? ")
                        + ts_cap_clause
                        + "    ORDER BY start_ts DESC LIMIT ?) "
                          "  UNION ALL "
                          "  SELECT memory_id, start_ts FROM ("
                          "    SELECT memory_id, start_ts FROM memories "
                          "    WHERE source_id = ? AND memory_id != ? "
                          "      AND kind NOT IN ('WORKING', 'LABEL', "
                          "'ASSOCIATION') "
                          "      AND start_ts > ? "
                        + ts_cap_clause
                        + "    ORDER BY start_ts ASC LIMIT ?) "
                          ") "
                          "ORDER BY distance ASC, start_ts DESC "
                          "LIMIT ?";
                  std::vector<std::any> temporal_params;
                  temporal_params.reserve (12);
                  temporal_params.push_back (anchor.start_ts);
                  temporal_params.push_back (anchor.source_id);
                  temporal_params.push_back (anchor.memory_id);
                  temporal_params.push_back (anchor.start_ts);
                  if (has_ts_cap)
                    {
                      temporal_params.push_back (ts_cap);
                    }
                  temporal_params.push_back (temporal_limit);
                  temporal_params.push_back (anchor.source_id);
                  temporal_params.push_back (anchor.memory_id);
                  temporal_params.push_back (anchor.start_ts);
                  if (has_ts_cap)
                    {
                      temporal_params.push_back (ts_cap);
                    }
                  temporal_params.push_back (temporal_limit);
                  temporal_params.push_back (temporal_limit);
                  auto rows = store->Execute (temporal_sql, temporal_params);
                  int rank = 0;
                  for (const auto &row : rows)
                    {
                      const long long memory_id
                          = AnyToInt64 (row.at ("memory_id"));
                      if (memory_id <= 0)
                        continue;
                      expanded_memory_ids.insert (memory_id);
                      auto &score = source_seed_expansion_scores[memory_id];
                      score = std::max (
                          score,
                          core::RetrievalGraphExpandedRagTemporalRankScore (
                              cfg.focus, cfg.sensitivity, cfg.stability, rank));
                      ++rank;
                    }
                }
            }
          const auto t_end = std::chrono::steady_clock::now ();
          context.AddOperationTiming (
              "GraphRetrieve.source_seed_temporal_expansion",
              elapsed_ms (t_start, t_end));
        }
      catch (...)
        {
        }

      if (graph_expanded_graph_weight > 0.0
          && !source_seed_memory_ids.empty ())
        {
          try
            {
              const auto t_start = std::chrono::steady_clock::now ();
              std::string sql = "WITH seed(id) AS (VALUES ";
              std::vector<std::any> params;
	              params.reserve (source_seed_memory_ids.size () + 5);
              for (size_t i = 0; i < source_seed_memory_ids.size (); ++i)
                {
                  if (i > 0)
                    sql += ",";
                  sql += "(?)";
                  params.push_back (source_seed_memory_ids[i]);
                }
              params.push_back (graph_expanded_relation_weight);
              params.push_back (graph_expanded_relation_weight);
              sql += "), seed_cue AS ("
                     "  SELECT df.source_memory_id AS cue_id, seed.id AS seed_id, "
                     "         MAX(df.weight) AS score "
                     "  FROM seed "
                     "  JOIN associations df ON df.target_memory_id = seed.id "
                     "    AND df.edge_type = 'derived_from' "
                     "  JOIN memories cm ON cm.memory_id = df.source_memory_id "
                     "    AND cm.kind = 'ASSOCIATION' "
                     "  GROUP BY df.source_memory_id, seed.id "
                     "), seed_label AS ("
                     "  SELECT hl.target_memory_id AS label_id, "
                     "         MAX(sc.score * hl.weight) AS score "
                     "  FROM seed_cue sc "
                     "  JOIN associations hl ON hl.source_memory_id = sc.cue_id "
                     "    AND hl.edge_type = 'has_label' "
                     "  JOIN memories lm ON lm.memory_id = hl.target_memory_id "
                     "    AND lm.kind = 'LABEL' "
                     "  GROUP BY hl.target_memory_id "
                     "), related_label AS ("
                     "  SELECT r.target_memory_id AS label_id, "
                     "         MAX(sl.score * ? * r.weight) AS score "
                     "  FROM seed_label sl "
                     "  JOIN associations r ON r.source_memory_id = sl.label_id "
                     "    AND r.edge_type IN ('co_occurs', 'implies', "
                     "                        'contradicts', 'reinforces', "
                     "                        'causes', 'similar_to') "
                     "  JOIN memories lm ON lm.memory_id = r.target_memory_id "
                     "    AND lm.kind = 'LABEL' "
                     "  GROUP BY r.target_memory_id "
                     "  UNION ALL "
                     "  SELECT r.source_memory_id AS label_id, "
                     "         MAX(sl.score * ? * r.weight) AS score "
                     "  FROM seed_label sl "
                     "  JOIN associations r ON r.target_memory_id = sl.label_id "
                     "    AND r.edge_type IN ('co_occurs', 'implies', "
                     "                        'contradicts', 'reinforces', "
                     "                        'causes', 'similar_to') "
                     "  JOIN memories lm ON lm.memory_id = r.source_memory_id "
                     "    AND lm.kind = 'LABEL' "
                     "  GROUP BY r.source_memory_id "
                     "), routed_label AS ("
                     "  SELECT label_id, score FROM seed_label "
                     "  UNION ALL "
                     "  SELECT label_id, score FROM related_label "
                     "), routed_cue AS ("
                     "  SELECT hl.source_memory_id AS cue_id, "
                     "         MAX(rl.score * hl.weight) AS score "
                     "  FROM routed_label rl "
                     "  JOIN associations hl ON hl.target_memory_id = rl.label_id "
                     "    AND hl.edge_type = 'has_label' "
                     "  JOIN memories cm ON cm.memory_id = hl.source_memory_id "
                     "    AND cm.kind = 'ASSOCIATION' "
                     "  GROUP BY hl.source_memory_id "
                     ") "
                     "SELECT df.target_memory_id AS source_memory_id, "
                     "       rc.cue_id AS cue_id, "
                     "       MAX(rc.score * df.weight) AS score "
                     "FROM routed_cue rc "
                     "JOIN associations df ON df.source_memory_id = rc.cue_id "
                     "  AND df.edge_type = 'derived_from' "
                     "JOIN memories sm ON sm.memory_id = df.target_memory_id "
	                     "WHERE sm.kind NOT IN ('WORKING', 'LABEL', 'ASSOCIATION') "
	                     "GROUP BY df.target_memory_id, rc.cue_id "
	                     "ORDER BY score DESC, df.target_memory_id DESC "
	                     "LIMIT ?";
		              params.push_back (static_cast<long long> (std::min (
		                  graph_expansion_row_limit,
		                  label_graph_fanout
		                      * std::max (
		                          1, static_cast<int> (
		                                 source_seed_memory_ids.size ())))));
              auto rows = store->Execute (sql, params);
              for (const auto &row : rows)
                {
                  const long long memory_id
                      = AnyToInt64 (row.at ("source_memory_id"));
                  const long long cue_id = AnyToInt64 (row.at ("cue_id"));
                  const double score = AnyToDouble (row.at ("score"), 0.0);
                  if (memory_id > 0 && score > 0.0)
                    {
                      expanded_memory_ids.insert (memory_id);
                      auto &slot = source_seed_expansion_scores[memory_id];
                      slot = std::max (
                          slot, graph_expanded_graph_weight
                                    * core::Clamp (score, 0.0, 1.0));
                    }
                  if (cue_id > 0)
                    {
                      expanded_memory_ids.insert (cue_id);
                    }
                }
              const auto t_end = std::chrono::steady_clock::now ();
              context.AddOperationTiming (
                  "GraphRetrieve.source_seed_label_relation_expansion",
                  elapsed_ms (t_start, t_end));
            }
          catch (...)
            {
            }
        }

      if (fact_layer_enabled && graph_expanded_fact_weight > 0.0
          && !source_seed_memory_ids.empty ())
        {
          try
            {
              const auto t_start = std::chrono::steady_clock::now ();
              std::string sql = "WITH seed(id) AS (VALUES ";
              std::vector<std::any> params;
	              params.reserve (source_seed_memory_ids.size () + 1);
              for (size_t i = 0; i < source_seed_memory_ids.size (); ++i)
                {
                  if (i > 0)
                    sql += ",";
                  sql += "(?)";
                  params.push_back (source_seed_memory_ids[i]);
                }
              sql += ") "
                     "SELECT fe2.source_memory_id AS source_memory_id, "
                     "       MAX(fe.support_weight * fe2.support_weight) AS score "
                     "FROM seed "
                     "JOIN fact_evidence fe ON fe.source_memory_id = seed.id "
                     "JOIN fact_assertions fa ON fa.fact_id = fe.fact_id "
                     "JOIN fact_evidence fe2 ON fe2.fact_id = fe.fact_id "
                     "JOIN memories m ON m.memory_id = fe2.source_memory_id "
	                     "WHERE COALESCE(fa.lifecycle_state, 'active') = 'active' "
	                     "  AND m.kind NOT IN ('WORKING', 'LABEL', 'ASSOCIATION') "
	                     "GROUP BY fe2.source_memory_id "
	                     "ORDER BY score DESC, fe2.source_memory_id DESC "
	                     "LIMIT ?";
	              params.push_back (
	                  static_cast<long long> (graph_expansion_row_limit));
              auto rows = store->Execute (sql, params);
              for (const auto &row : rows)
                {
                  const long long memory_id
                      = AnyToInt64 (row.at ("source_memory_id"));
                  const double score = AnyToDouble (row.at ("score"), 0.0);
                  if (memory_id <= 0 || score <= 0.0)
                    continue;
                  expanded_memory_ids.insert (memory_id);
                  auto &slot = source_seed_expansion_scores[memory_id];
                  slot = std::max (
                      slot, graph_expanded_fact_weight
                                * core::Clamp (score, 0.0, 1.0));
                }
              const auto t_end = std::chrono::steady_clock::now ();
              context.AddOperationTiming (
                  "GraphRetrieve.source_seed_fact_expansion",
                  elapsed_ms (t_start, t_end));
            }
          catch (...)
            {
            }
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

  if (preconsolidated_label_graph_enabled && label_graph_weight > 0.0
      && label_graph_seed_sources > 0)
    {
      const auto query_labels = ranked_query_label_ids ();
      if (!query_labels.empty ())
        {
          try
            {
              const auto t_start = std::chrono::steady_clock::now ();
              std::string sql
                  = "WITH query_labels(id, qscore) AS (VALUES ";
              std::vector<std::any> params;
              params.reserve (query_labels.size () * 2 + 3);
              for (size_t i = 0; i < query_labels.size (); ++i)
                {
                  if (i > 0)
                    {
                      sql += ",";
                    }
                  sql += "(?, ?)";
                  params.push_back (query_labels[i].second);
                  params.push_back (query_labels[i].first);
                }
              params.push_back (label_graph_relation_weight);
              params.push_back (label_graph_relation_weight);
	              params.push_back (static_cast<long long> (
	                  std::min (label_graph_seed_sources, label_graph_fanout)));
              sql += "), direct_cue AS ("
                     "  SELECT hl.source_memory_id AS cue_id, "
                     "         q.qscore AS score "
                     "  FROM query_labels q "
                     "  JOIN associations hl ON hl.target_memory_id = q.id "
                     "    AND hl.edge_type = 'has_label' "
                     "), related_label AS ("
                     "  SELECT r.target_memory_id AS label_id, "
                     "         q.qscore * ? * r.weight AS score "
                     "  FROM query_labels q "
                     "  JOIN associations r ON r.source_memory_id = q.id "
                     "    AND r.edge_type IN ('co_occurs', 'implies', "
                     "                        'contradicts', 'reinforces', "
                     "                        'causes', 'similar_to') "
                     "  UNION ALL "
                     "  SELECT r.source_memory_id AS label_id, "
                     "         q.qscore * ? * r.weight AS score "
                     "  FROM query_labels q "
                     "  JOIN associations r ON r.target_memory_id = q.id "
                     "    AND r.edge_type IN ('co_occurs', 'implies', "
                     "                        'contradicts', 'reinforces', "
                     "                        'causes', 'similar_to') "
                     "), related_cue AS ("
                     "  SELECT hl.source_memory_id AS cue_id, "
                     "         rl.score AS score "
                     "  FROM related_label rl "
                     "  JOIN memories lm ON lm.memory_id = rl.label_id "
                     "    AND lm.kind = 'LABEL' "
                     "  JOIN associations hl ON hl.target_memory_id = rl.label_id "
                     "    AND hl.edge_type = 'has_label' "
                     "), cue AS ("
                     "  SELECT cue_id, MAX(score) AS score "
                     "  FROM ("
                     "    SELECT cue_id, score FROM direct_cue "
                     "    UNION ALL "
                     "    SELECT cue_id, score FROM related_cue "
                     "  ) "
                     "  GROUP BY cue_id "
                     ") "
                     "SELECT cue.cue_id AS cue_id, "
                     "       df.target_memory_id AS source_id "
                     "FROM cue "
                     "JOIN memories cm ON cm.memory_id = cue.cue_id "
                     "  AND cm.kind = 'ASSOCIATION' "
                     "JOIN associations df ON df.source_memory_id = cue.cue_id "
                     "  AND df.edge_type = 'derived_from' "
                     "ORDER BY cue.score DESC, cm.created_at DESC "
                     "LIMIT ?";

              auto rows = store->Execute (sql, params);
              const auto t_end = std::chrono::steady_clock::now ();
              context.AddOperationTiming (
                  "GraphRetrieve.label_graph_seed_sources",
                  elapsed_ms (t_start, t_end));
              for (const auto &row : rows)
                {
                  const long long cue_id = store::AnyToLongLong (
                      row.at ("cue_id")).value_or (0);
                  const long long source_id = store::AnyToLongLong (
                      row.at ("source_id")).value_or (0);
                  if (cue_id > 0)
                    {
                      expanded_memory_ids.insert (cue_id);
                    }
                  if (source_id > 0)
                    {
                      expanded_memory_ids.insert (source_id);
                    }
                }
            }
          catch (...)
            {
            }
        }
    }

  if (preconsolidated_label_graph_enabled && label_graph_weight > 0.0
      && durable_source_text_seed_sources > 0 && !query_text_payload.empty ())
    {
      const auto query_tokens = RouteTokens (
          query_text_payload, route_token_min_chars);
      if (!query_tokens.empty ())
        {
          try
            {
              const auto t_start = std::chrono::steady_clock::now ();
              auto rows = store->Execute (
                  "SELECT DISTINCT df.target_memory_id AS source_id, "
                  "       cue.memory_id AS cue_id, "
                  "       lm.memory_id AS label_id, "
                  "       sm.blob_id AS blob_id, "
                  "       COUNT(DISTINCT hl.target_memory_id) AS label_count "
                  "FROM associations df "
                  "JOIN memories cue ON cue.memory_id = df.source_memory_id "
                  "  AND cue.kind = 'ASSOCIATION' "
                  "JOIN associations hl ON hl.source_memory_id = cue.memory_id "
                  "  AND hl.edge_type = 'has_label' "
                  "JOIN memories lm ON lm.memory_id = hl.target_memory_id "
                  "  AND lm.kind = 'LABEL' "
                  "JOIN memories sm ON sm.memory_id = df.target_memory_id "
	                  "WHERE df.edge_type = 'derived_from' "
	                  "  AND sm.kind NOT IN ('WORKING', 'LABEL', 'ASSOCIATION') "
	                  "  AND sm.blob_id IS NOT NULL "
	                  "  AND sm.modality = 'text' "
	                  "GROUP BY df.target_memory_id, cue.memory_id, "
	                  "         lm.memory_id, sm.blob_id "
	                  "ORDER BY sm.start_ts DESC, df.target_memory_id DESC "
	                  "LIMIT ?",
	                  { static_cast<long long> (
	                      durable_source_text_search_limit) });
              struct TextSeed
              {
                double score = 0.0;
                long long source_id = 0;
                long long cue_id = 0;
                long long label_id = 0;
              };
              std::vector<TextSeed> ranked_sources;
              ranked_sources.reserve (rows.size ());
              for (const auto &row : rows)
                {
                  const long long source_id
                      = store::AnyToLongLong (row.at ("source_id")).value_or (0);
                  if (source_id <= 0)
                    {
                      continue;
                    }
                  const auto blob_id = store::BlobFromAny (row.at ("blob_id"));
                  if (blob_id.empty ())
                    {
                      continue;
                    }
	                  auto payload_rows = store->Execute (
	                      "SELECT substr(objstore_get(?1), 1, ?2) AS payload",
	                      { blob_id,
	                        static_cast<long long> (
	                            durable_source_text_max_bytes) });
                  if (payload_rows.empty ()
                      || payload_rows[0].count ("payload") == 0)
                    {
                      continue;
                    }
                  const auto payload
                      = store::BlobFromAny (payload_rows[0].at ("payload"));
                  if (payload.empty ())
                    {
                      continue;
                    }
                  const std::string source_text (
                      reinterpret_cast<const char *> (payload.data ()),
                      payload.size ());
	                  double score = RouteTokenOverlapScore (
	                      query_tokens, source_text, cfg.focus, cfg.sensitivity,
	                      cfg.stability, route_token_min_chars);
                  const int label_count = static_cast<int> (
                      store::AnyToLongLong (row.at ("label_count"))
                          .value_or (1));
                  const double label_support = core::Clamp (
                      std::log1p (static_cast<double> (label_count))
                          / std::log (core::RetrievalDurableSourceSupportSaturationCount (
                              cfg.focus, cfg.stability)),
                      0.0, 1.0);
                  const double text_base_weight
                      = core::RetrievalDurableSourceTextBaseWeight (
                          cfg.focus, cfg.sensitivity, cfg.stability);
                  score = core::Clamp (
                      score
                          * (text_base_weight
                             + (1.0 - text_base_weight) * label_support),
                      0.0, 1.0);
                  if (score < durable_source_text_min_score)
                    {
                      continue;
                    }
                  const long long cue_id
                      = store::AnyToLongLong (row.at ("cue_id")).value_or (0);
                  const long long label_id
                      = store::AnyToLongLong (row.at ("label_id")).value_or (0);
                  ranked_sources.push_back ({ score, source_id, cue_id,
                                               label_id });
                }
              std::sort (
                  ranked_sources.begin (), ranked_sources.end (),
                  [] (const auto &a, const auto &b) {
                    if (a.score == b.score)
                      {
                        return a.source_id < b.source_id;
                      }
                    return a.score > b.score;
                  });
              if (static_cast<int> (ranked_sources.size ())
                  > durable_source_text_seed_sources)
                {
                  ranked_sources.resize (
                      static_cast<size_t> (durable_source_text_seed_sources));
                }
              for (const auto &seed : ranked_sources)
                {
                  if (seed.source_id > 0)
                    {
                      expanded_memory_ids.insert (seed.source_id);
                      auto &slot
                          = durable_source_text_seed_scores[seed.source_id];
                      slot = std::max (slot, seed.score);
                    }
                  if (seed.cue_id > 0)
                    {
                      expanded_memory_ids.insert (seed.cue_id);
                      auto &slot = durable_source_text_seed_scores[seed.cue_id];
                      slot = std::max (slot, seed.score);
                    }
                  if (seed.label_id > 0)
                    {
                      expanded_memory_ids.insert (seed.label_id);
                      auto &slot
                          = durable_source_text_seed_scores[seed.label_id];
                      slot = std::max (slot, seed.score);
                    }
                }
              const auto t_end = std::chrono::steady_clock::now ();
              context.AddOperationTiming (
                  "GraphRetrieve.durable_source_text_seed",
                  elapsed_ms (t_start, t_end));
            }
          catch (...)
            {
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
          const char *edge_filter_outer
              = cfg.reinforcement_enabled ? "" : " AND a.edge_type != 'reinforces' ";
          const char *edge_filter_inner
              = cfg.reinforcement_enabled ? ""
                                           : " AND a2.edge_type != 'reinforces' ";
          std::string sql = "WITH RECURSIVE seed(id) AS (VALUES ";
          std::vector<std::any> params;
          params.reserve (expanded_memory_ids.size () + 9);
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
                 "  WHERE expand.depth < ? AND a.weight >= ? "
                 "  AND a.rowid IN ("
                 "    SELECT a2.rowid FROM associations a2 "
                 "    WHERE a2.source_memory_id = expand.id "
                 "      AND a2.weight >= ? ";
          sql += edge_filter_inner;
          sql += "    ORDER BY a2.weight DESC, "
                 "             COALESCE(a2.last_reinforced, 0) DESC, "
                 "             a2.target_memory_id ASC "
                 "    LIMIT ?"
                 "  ) ";
          sql += edge_filter_outer;
          sql += "  UNION "
                 "  SELECT a.source_memory_id, expand.depth+1 "
                 "  FROM associations a JOIN expand ON a.target_memory_id = expand.id "
                 "  WHERE expand.depth < ? AND a.weight >= ? "
                 "  AND a.rowid IN ("
                 "    SELECT a2.rowid FROM associations a2 "
                 "    WHERE a2.target_memory_id = expand.id "
                 "      AND a2.weight >= ? ";
          sql += edge_filter_inner;
          sql += "    ORDER BY a2.weight DESC, "
                 "             COALESCE(a2.last_reinforced, 0) DESC, "
                 "             a2.source_memory_id ASC "
                 "    LIMIT ?"
                 "  ) ";
          sql += edge_filter_outer;
          sql += ") "
                 "SELECT DISTINCT id FROM expand LIMIT ?;";
          params.push_back (static_cast<long long> (depth));
          params.push_back (min_edge_weight);
          params.push_back (min_edge_weight);
          params.push_back (static_cast<long long> (graph_expansion_fanout));
          params.push_back (static_cast<long long> (depth));
          params.push_back (min_edge_weight);
          params.push_back (min_edge_weight);
          params.push_back (static_cast<long long> (graph_expansion_fanout));
          params.push_back (
              static_cast<long long> (graph_expansion_row_limit));

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

  const auto graph_expansion_evidence
      = core::RetrievalGraphExpansionEvidenceCounts (
          cfg.focus, cfg.sensitivity, cfg.stability);
  int min_assoc = graph_expansion_evidence.min_association_count;
  int min_label = graph_expansion_evidence.min_label_count;

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
                    "COALESCE(m.last_access, 0) AS last_access, "
                    "COALESCE(m.retrieved_count, 0) AS retrieved_count, "
                    "COALESCE(m.used_count, 0) AS used_count, "
                    "COALESCE(NULLIF(m.start_ts, 0), m.created_at, "
                    "         e.created_at, 0) AS event_ts, "
                    "m.context, m.source_reliability, "
                    "m.source_contradiction_count, m.emotional_intensity, "
                    "m.s_arousal_avg, m.kind, "
                    "m.source_id, m.modality, "
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
                auto it_last_access = row.find ("last_access");
                auto it_retrieved_count = row.find ("retrieved_count");
                auto it_used_count = row.find ("used_count");
                auto it_event_ts = row.find ("event_ts");
              auto it_ctx = row.find ("context");
              auto it_rel = row.find ("source_reliability");
              auto it_contra = row.find ("source_contradiction_count");
              auto it_emotion = row.find ("emotional_intensity");
              auto it_arousal = row.find ("s_arousal_avg");
              auto it_kind = row.find ("kind");
              auto it_source_id = row.find ("source_id");
              auto it_modality = row.find ("modality");
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
                long long last_access = 0;
                if (it_last_access != row.end ()
                    && it_last_access->second.type () == typeid (long long))
                  {
                    last_access = std::any_cast<long long> (it_last_access->second);
                  }
                const long long retrieved_count
                    = std::max (0LL,
                                (it_retrieved_count != row.end ())
                                    ? store::AnyToLongLong (
                                          it_retrieved_count->second)
                                          .value_or (0)
                                    : 0LL);
                const long long used_count
                    = std::max (0LL,
                                (it_used_count != row.end ())
                                    ? store::AnyToLongLong (it_used_count->second)
                                          .value_or (0)
                                    : 0LL);
                long long event_ts = created_at;
              if (it_event_ts != row.end ()
                  && it_event_ts->second.type () == typeid (long long))
                {
                  event_ts = std::any_cast<long long> (it_event_ts->second);
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

              double base_rel = core::SourceReliabilityPrior (
                  cfg.focus, cfg.sensitivity, cfg.stability);
              if (it_rel != row.end () && it_rel->second.type () == typeid (double))
                {
                  base_rel = std::any_cast<double> (it_rel->second);
                }

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
              const double freshness = std::exp (
                  -age_s
                  / core::RetrievalSourceFreshnessTauSeconds (
                      cfg.focus, cfg.sensitivity, cfg.stability));
              const double effective_freshness
                  = ApplyFreshnessPenaltyScale (freshness,
                                               resurfacing_decay_scale);
              const double freshness_weight
                  = core::RetrievalSourceFreshnessWeight (
                      cfg.focus, cfg.sensitivity, cfg.stability);
              const double source_conf
                  = core::Clamp (
                      base_rel
                          * (1.0
                             - core::RetrievalSourceContradictionPenalty (
                                 cfg.focus, cfg.sensitivity, cfg.stability)
                                   * contradiction_count)
                          * ((1.0 - freshness_weight)
                             + freshness_weight * effective_freshness),
                      0.0, 1.0);

                double proc_score = 0.0;
                if (cfg.procedural_enabled && !disable_procedural_proactive
                    && !sparse_key.empty ())
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
                auto proc_seed_it = procedural_seed_scores.find (mem_id);
                if (proc_seed_it != procedural_seed_scores.end ())
                  {
                    proc_score = std::max (proc_score, proc_seed_it->second);
                  }

              const double pre_activation
                  = (it_preact != row.end ())
                        ? core::Clamp (get_double (it_preact->second, 0.0), 0.0, 1.0)
                        : 0.0;
              const std::string memory_source_id
                  = it_source_id != row.end ()
                        ? AnyToString (it_source_id->second)
                        : std::string ();
              const std::string modality
                  = it_modality != row.end ()
                        ? AnyToString (it_modality->second)
                        : std::string ();

              scored.push_back (Scored{ emb_id,
                                        mem_id,
                                        created_at,
                                        last_access,
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
                                          0.0,
                                          0,
                                          0.0,
                                          0,
                                          is_association,
                                          is_label,
                                          v,
                                        ctx_vec,
                                        0.0,
                                        0LL,
                                        0LL,
                                        std::string (),
                                        std::string () });
                scored.back ().temporal_score = temporal_rank_score (event_ts);
                scored.back ().retrieved_count = retrieved_count;
                scored.back ().used_count = used_count;
                scored.back ().source_id = memory_source_id;
                scored.back ().modality = modality;
                fetched_any = true;
            }
        }
      catch (...)
        {
        }
    }

  if (!durable_source_text_seed_scores.empty ())
    {
      for (auto &candidate : scored)
        {
          auto it = durable_source_text_seed_scores.find (candidate.memory_id);
          if (it == durable_source_text_seed_scores.end ())
            {
              continue;
            }
          candidate.label_graph_boost
              = std::max (candidate.label_graph_boost, it->second);
          candidate.label_match_count
              = std::max (candidate.label_match_count, 1);
          candidate.durable_source_count
              = std::max (candidate.durable_source_count, 1);
        }
    }

  if (!source_seed_expansion_scores.empty ())
    {
      for (auto &candidate : scored)
        {
          auto it = source_seed_expansion_scores.find (candidate.memory_id);
          if (it == source_seed_expansion_scores.end ())
            {
              continue;
            }
          candidate.durable_source_boost
              = std::max (candidate.durable_source_boost,
                          core::Clamp (it->second, 0.0, 1.0));
          candidate.durable_source_count
              = std::max (candidate.durable_source_count, 1);
        }
    }

  auto max_wm_overlap = [&p_ctx] (const Eigen::VectorXf &v) -> double {
    if (v.size () == 0)
      {
        return 0.0;
      }
    double max_similarity = 0.0;
    for (const auto &slot : p_ctx.wm_slots)
      {
        if (slot.embedding.size () == v.size () && slot.embedding.size () > 0)
          {
            const double sim = core::CosineSimilarity (slot.embedding, v);
            max_similarity = std::max (max_similarity, sim);
          }
      }
    return max_similarity;
  };

  auto is_overlap_wm = [&max_wm_overlap] (const Eigen::VectorXf &v,
                                          double dup_thresh) -> bool {
    if (v.size () == 0)
      {
        return false;
      }
    return max_wm_overlap (v) >= dup_thresh;
  };

  const double dup_thresh = core::DupThresh (cfg.focus, cfg.stability);
  const double dup_thresh_summary = core::Clamp (
      dup_thresh
          * core::RetrievalSummaryDuplicateThresholdScale (
              cfg.focus, cfg.sensitivity, cfg.stability),
      0.0,
      core::RetrievalSummaryDuplicateThresholdCap (
          cfg.focus, cfg.sensitivity, cfg.stability));
  const bool bypass_summary_overlap = core::RetrievalBypassSummaryOverlap (
      cfg.focus, cfg.sensitivity, cfg.stability);
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
      w_div *= core::RetrievalTotDiversificationScale (
          cfg.focus, cfg.sensitivity, cfg.stability, tot_confidence_scale);
    }
  const auto candidate_blend_weights
      = core::RetrievalCandidateBlendScoringWeights (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const double w_ctx = candidate_blend_weights.context;
  const double w_proc = candidate_blend_weights.procedural;
  const double w_emotion
      = cfg.affect_retrieval
            ? core::RetrievalEmotionWeight (cfg.sensitivity)
            : 0.0;
  const double assoc_boost
      = core::AssociationBoost (cfg.focus, cfg.sensitivity, cfg.stability);
  const double label_boost = candidate_blend_weights.label_boost;
  const bool durable_source_scoring_enabled
      = !EnvFlag ("CORTEXT_DISABLE_DURABLE_SOURCE_SET_RETRIEVAL");
  const double durable_source_weight
      = durable_source_scoring_enabled
            ? core::RetrievalDurableSourceSetWeight (
                cfg.focus, cfg.sensitivity, cfg.stability)
            : 0.0;
  const double durable_source_min_score
      = core::RetrievalDurableSourceSetMinScore (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const double source_backed_boost_floor = std::max (
      durable_source_min_score,
      core::RetrievalSourceBackedBoostFloor (
          cfg.focus, cfg.sensitivity, cfg.stability));
  const int durable_source_min_topk = std::max (
      0, core::RetrievalDurableSourceMinTopK (cfg.focus, cfg.stability));
  const bool recent_retrieval_inhibition_enabled
      = EnvFlag ("CORTEXT_ENABLE_RECENT_RETRIEVAL_INHIBITION");
  const double recent_retrieval_inhibition_weight
      = recent_retrieval_inhibition_enabled
            ? core::RetrievalRecentInhibitionWeight (
                cfg.focus, cfg.sensitivity, cfg.stability)
            : 0.0;
  const double recent_retrieval_inhibition_tau_seconds
      = core::RetrievalRecentInhibitionTauSeconds (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const bool base_level_availability_enabled
      = EnvFlag ("CORTEXT_ENABLE_BASE_LEVEL_AVAILABILITY");
  const double base_level_availability_weight
      = base_level_availability_enabled
            ? core::RetrievalBaseLevelAvailabilityWeight (
                cfg.focus, cfg.sensitivity, cfg.stability)
            : 0.0;
  const double base_level_availability_tau_seconds
      = core::RetrievalBaseLevelAvailabilityTauSeconds (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const double base_level_availability_count_saturation
      = core::RetrievalBaseLevelAvailabilityCountSaturation (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const bool partial_matching_enabled
      = EnvFlag ("CORTEXT_ENABLE_PARTIAL_MATCHING_PENALTY");
  const double partial_match_penalty_weight
      = partial_matching_enabled
            ? core::RetrievalPartialMatchPenaltyWeight (
                cfg.focus, cfg.sensitivity, cfg.stability)
            : 0.0;
  const double partial_match_contradiction_saturation
      = core::RetrievalPartialMatchContradictionSaturation (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const double partial_match_source_mismatch_weight
      = core::RetrievalPartialMatchSourceMismatchWeight (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const double partial_match_modality_mismatch_weight
      = core::RetrievalPartialMatchModalityMismatchWeight (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const bool evidence_blending_enabled
      = EnvFlag ("CORTEXT_ENABLE_EVIDENCE_BLENDING");
  const double evidence_blend_tie_margin
      = core::RetrievalEvidenceBlendTieMargin (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const double evidence_blend_temperature
      = core::RetrievalEvidenceBlendTemperature (
          cfg.focus, cfg.sensitivity, cfg.stability);
  const int evidence_blend_max_members
      = evidence_blending_enabled
            ? core::RetrievalEvidenceBlendMaxMembers (
                cfg.focus, cfg.sensitivity, cfg.stability)
            : 0;
  std::string signal_modality = signal.modality;
  if (signal_modality.empty () && signal.mimetype == "text/plain")
    {
      signal_modality = "text";
    }

  const double salience = core::Clamp (
      context.GetMetric (operations::Metric::salience).value_or (0.0), 0.0, 1.0);
  const double emotion_intensity
      = core::Clamp (context.GetEmotionIntensity (), 0.0, 1.0);
  const double arousal = core::Clamp (context.GetArousal (), 0.0, 1.0);
  const auto affect_weights = core::AffectDriveWeights (cfg.sensitivity);
  const double affect_gain = core::AffectGain (cfg.sensitivity);
  const double affect_drive
      = core::Clamp (affect_gain
                         * (affect_weights.arousal_weight * arousal
                            + affect_weights.emotion_weight * emotion_intensity
                            + affect_weights.salience_weight * salience),
                     0.0, 1.0);
  const auto memory_affect_weights
      = core::RetrievalMemoryAffectScoringWeights (cfg.sensitivity);
  const double w_mem_emotion = memory_affect_weights.emotion;
  const double w_mem_arousal = memory_affect_weights.arousal;

  auto base_level_availability = [&] (const Scored &s) {
    if (!base_level_availability_enabled
        || base_level_availability_weight <= 0.0)
      {
        return 0.0;
      }
    const long long retrieved_count = std::max (0LL, s.retrieved_count);
    const long long used_count = std::max (0LL, s.used_count);
    const double exposure_count
        = static_cast<double> (retrieved_count)
          + 2.0 * static_cast<double> (used_count);
    if (exposure_count <= 0.0)
      {
        return 0.0;
      }

    const double saturation
        = std::max (base_level_availability_count_saturation, 1.0);
    const double count_support = core::Clamp (
        std::log1p (exposure_count) / std::log1p (saturation), 0.0, 1.0);
    const long long access_ts = s.last_access > 0 ? s.last_access : s.created_at;
    double recency = 0.5;
    if (access_ts > 0)
      {
        if (signal.timestamp > static_cast<std::uint64_t> (access_ts))
          {
            const double age_s
                = static_cast<double> (
                      signal.timestamp - static_cast<std::uint64_t> (access_ts))
                  / 1000.0;
            recency = std::exp (
                -age_s
                / std::max (base_level_availability_tau_seconds,
                            constants::kNormEpsilon));
          }
        else
          {
            recency = 1.0;
          }
      }

    const double use_ratio
        = retrieved_count > 0
              ? static_cast<double> (used_count)
                    / static_cast<double> (retrieved_count)
              : (used_count > 0 ? 1.0 : 0.0);
    const double use_quality = core::Clamp (0.75 + 0.25 * use_ratio, 0.75,
                                            1.0);
    return base_level_availability_weight * count_support
           * core::Clamp (recency, 0.0, 1.0) * use_quality;
  };

  auto recent_retrieval_inhibition = [&] (const Scored &s) {
    if (!recent_retrieval_inhibition_enabled
        || recent_retrieval_inhibition_weight <= 0.0 || s.last_access <= 0
        || signal.timestamp <= static_cast<std::uint64_t> (s.last_access))
      {
        return 0.0;
      }
    const double age_s
        = static_cast<double> (signal.timestamp
                               - static_cast<std::uint64_t> (s.last_access))
          / 1000.0;
    const double decay = std::exp (
        -age_s
        / std::max (recent_retrieval_inhibition_tau_seconds,
                    constants::kNormEpsilon));
    return -recent_retrieval_inhibition_weight
           * core::Clamp (decay, 0.0, 1.0);
  };

  auto partial_match_penalty = [&] (const Scored &s) {
    double penalty = std::max (0.0, s.fact_stale_penalty);
    if (!partial_matching_enabled || partial_match_penalty_weight <= 0.0)
      {
        return -penalty;
      }

    const double contradiction_support = core::Clamp (
        static_cast<double> (std::max (0, s.source_contradictions))
            / std::max (partial_match_contradiction_saturation, 1.0),
        0.0, 1.0);
    penalty += partial_match_penalty_weight * contradiction_support;

    const bool summary_like = s.is_association || s.is_label;
    if (!summary_like && !signal.source_id.empty () && !s.source_id.empty ()
        && s.source_id != signal.source_id)
      {
        penalty += partial_match_penalty_weight
                   * partial_match_source_mismatch_weight;
      }
    if (!summary_like && !signal_modality.empty () && !s.modality.empty ()
        && s.modality != signal_modality)
      {
        penalty += partial_match_penalty_weight
                   * partial_match_modality_mismatch_weight;
      }

    return -core::Clamp (penalty, 0.0, 1.0);
  };

  const double source_thresh = core::RetrievalSourceConfidenceThreshold (
      cfg.focus, cfg.sensitivity, cfg.stability, resurfacing_decay_scale);
  const double certainty_requirement
      = core::RetrievalUnknownCautionCertaintyRequirement (
          cfg.focus, cfg.sensitivity, cfg.stability, resurfacing_decay_scale);
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
        const bool source_backed_by_graph
            = s.durable_source_boost >= source_backed_boost_floor
              || (s.durable_source_count > 0
                  && s.label_graph_boost >= source_backed_boost_floor);
        if (!disable_source_conf && enforce_source_conf
            && !source_backed_by_graph && s.source_confidence < source_thresh)
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
                core::RetrievalAnyFactMatchSupportFloor (
                    cfg.focus, cfg.sensitivity, cfg.stability),
                1.0);
            const std::size_t cap = std::min<std::size_t> (
                matched_facts.size (),
                static_cast<std::size_t> (core::RetrievalAnyFactMatchLinkCap (
                    cfg.focus, cfg.sensitivity, cfg.stability)));
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

  auto apply_label_graph_scores = [&] (std::vector<Scored> &candidates) {
    if (!preconsolidated_label_graph_enabled || label_graph_weight <= 0.0
        || candidates.empty ())
      {
        return;
      }

    const auto t_start = std::chrono::steady_clock::now ();
    std::vector<std::pair<double, long long>> ranked_query_labels
        = build_ranked_query_label_ids ();
    if (ranked_query_labels.empty ())
      {
        return;
      }

    double query_score_sum = 0.0;
    std::vector<long long> candidate_memory_ids;
    candidate_memory_ids.reserve (candidates.size ());
    std::unordered_set<long long> seen_candidate_memory_ids;
    for (const auto &[score, label_memory_id] : ranked_query_labels)
      {
        (void)label_memory_id;
        query_score_sum += score;
      }
    for (const auto &candidate : candidates)
      {
        if (candidate.memory_id <= 0 || candidate.is_label)
          {
            continue;
          }
        if (seen_candidate_memory_ids.insert (candidate.memory_id).second)
          {
            candidate_memory_ids.push_back (candidate.memory_id);
          }
      }
    if (candidate_memory_ids.empty () || query_score_sum <= 1e-9)
      {
        return;
      }

    try
      {
        std::string sql
            = "WITH query_labels(id, qscore) AS (VALUES ";
        std::vector<std::any> params;
        params.reserve (ranked_query_labels.size () * 2
                        + candidate_memory_ids.size () + 3);
        for (size_t i = 0; i < ranked_query_labels.size (); ++i)
          {
            if (i > 0)
              {
                sql += ",";
              }
            sql += "(?, ?)";
            params.push_back (ranked_query_labels[i].second);
            params.push_back (ranked_query_labels[i].first);
          }
        sql += "), candidate(id) AS (VALUES ";
        for (size_t i = 0; i < candidate_memory_ids.size (); ++i)
          {
            if (i > 0)
              {
                sql += ",";
              }
            sql += "(?)";
            params.push_back (candidate_memory_ids[i]);
          }
	        params.push_back (label_graph_relation_weight);
	        params.push_back (label_graph_relation_weight);
	        params.push_back (label_graph_degree_damping);
	        sql += "), candidate_labels AS ("
               "  SELECT c.id AS candidate_memory_id, "
               "         a.target_memory_id AS label_memory_id, "
               "         MAX(a.weight) AS label_weight "
               "  FROM candidate c "
               "  JOIN associations a ON a.source_memory_id = c.id "
               "    AND a.edge_type = 'has_label' "
               "  GROUP BY c.id, a.target_memory_id "
               "  UNION ALL "
               "  SELECT c.id AS candidate_memory_id, "
               "         hl.target_memory_id AS label_memory_id, "
               "         MAX(dl.weight * hl.weight) AS label_weight "
               "  FROM candidate c "
               "  JOIN associations dl ON dl.target_memory_id = c.id "
               "    AND dl.edge_type = 'derived_from' "
               "  JOIN associations hl ON hl.source_memory_id = dl.source_memory_id "
               "    AND hl.edge_type = 'has_label' "
	               "  GROUP BY c.id, hl.target_memory_id "
	               "), relation_labels AS ("
               "  SELECT cl.candidate_memory_id, "
               "         r.target_memory_id AS label_memory_id, "
               "         MAX(? * cl.label_weight * r.weight) AS label_weight "
               "  FROM candidate_labels cl "
               "  JOIN memories src_label ON src_label.memory_id = cl.label_memory_id "
               "    AND src_label.kind = 'LABEL' "
               "  JOIN associations r ON r.source_memory_id = cl.label_memory_id "
               "    AND r.edge_type IN ('co_occurs', 'implies', 'contradicts', "
               "                        'reinforces', 'causes', 'similar_to') "
               "  JOIN memories target_label ON target_label.memory_id = r.target_memory_id "
               "    AND target_label.kind = 'LABEL' "
               "  GROUP BY cl.candidate_memory_id, r.target_memory_id "
               "  UNION ALL "
               "  SELECT cl.candidate_memory_id, "
               "         r.source_memory_id AS label_memory_id, "
               "         MAX(? * cl.label_weight * r.weight) AS label_weight "
               "  FROM candidate_labels cl "
               "  JOIN memories target_label ON target_label.memory_id = cl.label_memory_id "
               "    AND target_label.kind = 'LABEL' "
               "  JOIN associations r ON r.target_memory_id = cl.label_memory_id "
               "    AND r.edge_type IN ('co_occurs', 'implies', 'contradicts', "
               "                        'reinforces', 'causes', 'similar_to') "
               "  JOIN memories src_label ON src_label.memory_id = r.source_memory_id "
               "    AND src_label.kind = 'LABEL' "
               "  GROUP BY cl.candidate_memory_id, r.source_memory_id "
               "), matched AS ("
               "  SELECT candidate_memory_id, label_memory_id, "
               "         MAX(label_weight) AS label_weight "
               "  FROM ("
               "    SELECT candidate_memory_id, label_memory_id, label_weight "
               "    FROM candidate_labels "
               "    UNION ALL "
               "    SELECT candidate_memory_id, label_memory_id, label_weight "
               "    FROM relation_labels "
	               "  ) "
	               "  GROUP BY candidate_memory_id, label_memory_id "
	               "), matched_labels AS ("
	               "  SELECT DISTINCT label_memory_id AS label_id FROM matched "
	               "), label_degree AS ("
	               "  SELECT label_id, SUM(degree) AS degree "
	               "  FROM ("
	               "    SELECT a.target_memory_id AS label_id, COUNT(*) AS degree "
	               "    FROM associations a "
	               "    JOIN matched_labels ml ON ml.label_id = a.target_memory_id "
	               "    WHERE a.edge_type = 'has_label' "
	               "    GROUP BY a.target_memory_id "
	               "    UNION ALL "
	               "    SELECT a.source_memory_id AS label_id, COUNT(*) AS degree "
	               "    FROM associations a "
	               "    JOIN matched_labels ml ON ml.label_id = a.source_memory_id "
	               "    WHERE a.edge_type IN ('co_occurs', 'implies', 'contradicts', "
	               "                          'reinforces', 'causes', 'similar_to') "
	               "    GROUP BY a.source_memory_id "
	               "    UNION ALL "
	               "    SELECT a.target_memory_id AS label_id, COUNT(*) AS degree "
	               "    FROM associations a "
	               "    JOIN matched_labels ml ON ml.label_id = a.target_memory_id "
	               "    WHERE a.edge_type IN ('co_occurs', 'implies', 'contradicts', "
	               "                          'reinforces', 'causes', 'similar_to') "
	               "    GROUP BY a.target_memory_id "
	               "  ) "
	               "  GROUP BY label_id "
	               "), label_specificity AS ("
	               "  SELECT ml.label_id, "
	               "         1.0 / (1.0 + ? * COALESCE(ld.degree, 0)) AS specificity "
	               "  FROM matched_labels ml "
	               "  LEFT JOIN label_degree ld ON ld.label_id = ml.label_id "
	               ") "
               "SELECT m.candidate_memory_id AS memory_id, "
               "       SUM(q.qscore * m.label_weight "
               "           * COALESCE(ls.specificity, 1.0)) AS raw_boost, "
               "       COUNT(*) AS label_match_count "
               "FROM matched m "
               "JOIN query_labels q ON q.id = m.label_memory_id "
               "LEFT JOIN label_specificity ls ON ls.label_id = m.label_memory_id "
               "GROUP BY m.candidate_memory_id";

        auto rows = store->Execute (sql, params);
        std::unordered_map<long long, double> boosts;
        std::unordered_map<long long, int> label_matches;
        boosts.reserve (rows.size ());
        label_matches.reserve (rows.size ());
        for (const auto &row : rows)
          {
            const auto it_memory = row.find ("memory_id");
            const auto it_raw = row.find ("raw_boost");
            if (it_memory == row.end () || it_raw == row.end ())
              {
                continue;
              }
            const long long memory_id = AnyToInt64 (it_memory->second);
            if (memory_id <= 0)
              {
                continue;
              }
            const double raw_boost = AnyToDouble (it_raw->second, 0.0);
            boosts[memory_id]
                = core::Clamp (raw_boost / query_score_sum, 0.0, 1.0);
            const auto it_match_count = row.find ("label_match_count");
            if (it_match_count != row.end ())
              {
                label_matches[memory_id]
                    = static_cast<int> (AnyToInt64 (it_match_count->second));
              }
          }
        for (auto &candidate : candidates)
          {
            auto it = boosts.find (candidate.memory_id);
            if (it != boosts.end ())
              {
                candidate.label_graph_boost
                    = std::max (candidate.label_graph_boost, it->second);
                auto match_it = label_matches.find (candidate.memory_id);
                if (match_it != label_matches.end ())
                  {
                    candidate.label_match_count
                        = std::max (candidate.label_match_count,
                                    match_it->second);
                  }
              }
          }
      }
    catch (...)
      {
      }
    const auto t_end = std::chrono::steady_clock::now ();
    context.AddOperationTiming ("GraphRetrieve.preconsolidated_label_graph",
                                elapsed_ms (t_start, t_end));
  };

  auto apply_durable_source_scores = [&] (std::vector<Scored> &candidates) {
    if (!durable_source_scoring_enabled || durable_source_weight <= 0.0
        || candidates.empty ())
      {
        return;
      }

    std::vector<long long> candidate_memory_ids;
    candidate_memory_ids.reserve (candidates.size ());
    std::unordered_set<long long> seen_candidate_memory_ids;
    for (const auto &candidate : candidates)
      {
        if (candidate.memory_id <= 0
            || !(candidate.is_association || candidate.is_label))
          {
            continue;
          }
        if (seen_candidate_memory_ids.insert (candidate.memory_id).second)
          {
            candidate_memory_ids.push_back (candidate.memory_id);
          }
      }
    if (candidate_memory_ids.empty ())
      {
        return;
      }

    const auto t_start = std::chrono::steady_clock::now ();
    try
      {
        const int durable_source_link_limit
            = core::RetrievalDurableSourceLinkFanout (
                cfg.focus, cfg.sensitivity, cfg.stability,
                static_cast<int> (candidate_memory_ids.size ()));
        std::string sql = "WITH candidate(id) AS (VALUES ";
        std::vector<std::any> params;
        params.reserve (candidate_memory_ids.size () + 1);
        for (size_t i = 0; i < candidate_memory_ids.size (); ++i)
          {
            if (i > 0)
              {
                sql += ",";
              }
            sql += "(?)";
            params.push_back (candidate_memory_ids[i]);
          }
        sql += "), source_links AS ("
               "  SELECT c.id AS candidate_memory_id, "
               "         df.target_memory_id AS source_memory_id, "
               "         MAX(df.weight) AS link_weight "
               "  FROM candidate c "
               "  JOIN memories cm ON cm.memory_id = c.id "
               "  JOIN associations df ON df.source_memory_id = c.id "
               "    AND df.edge_type = 'derived_from' "
               "  WHERE cm.kind = 'ASSOCIATION' "
               "  GROUP BY c.id, df.target_memory_id "
               "  UNION ALL "
               "  SELECT c.id AS candidate_memory_id, "
               "         df.target_memory_id AS source_memory_id, "
               "         MAX(hl.weight * df.weight) AS link_weight "
               "  FROM candidate c "
               "  JOIN memories cm ON cm.memory_id = c.id "
               "  JOIN associations hl ON hl.target_memory_id = c.id "
               "    AND hl.edge_type = 'has_label' "
               "  JOIN associations df ON df.source_memory_id = hl.source_memory_id "
               "    AND df.edge_type = 'derived_from' "
               "  WHERE cm.kind = 'LABEL' "
               "  GROUP BY c.id, df.target_memory_id "
               "), source_links_ranked AS ("
               "  SELECT candidate_memory_id, source_memory_id, "
               "         MAX(link_weight) AS link_weight "
               "  FROM source_links "
               "  GROUP BY candidate_memory_id, source_memory_id "
               "  ORDER BY link_weight DESC, candidate_memory_id DESC, "
               "           source_memory_id DESC "
               "  LIMIT ? "
               ") "
               "SELECT sl.candidate_memory_id, sl.source_memory_id, "
               "       MAX(sl.link_weight) AS link_weight, e.embedding "
               "FROM source_links_ranked sl "
               "JOIN memories sm ON sm.memory_id = sl.source_memory_id "
               "JOIN embeddings e ON e.embedding_id = sm.embedding_id "
               "WHERE sm.kind NOT IN ('WORKING', 'LABEL', 'ASSOCIATION') "
               "GROUP BY sl.candidate_memory_id, sl.source_memory_id";
        params.push_back (static_cast<long long> (durable_source_link_limit));

        auto rows = store->Execute (sql, params);
        std::unordered_map<long long, double> max_source_score;
        std::unordered_map<long long, int> source_counts;
        max_source_score.reserve (candidate_memory_ids.size ());
        source_counts.reserve (candidate_memory_ids.size ());
        for (const auto &row : rows)
          {
            const auto it_candidate = row.find ("candidate_memory_id");
            const auto it_embedding = row.find ("embedding");
            if (it_candidate == row.end () || it_embedding == row.end ())
              {
                continue;
              }
            const long long candidate_memory_id
                = AnyToInt64 (it_candidate->second);
            if (candidate_memory_id <= 0)
              {
                continue;
              }
            Eigen::VectorXf source_vec;
            if (!core::DecodeFloatBlob (
                    it_embedding->second, static_cast<int> (q.size ()),
                    source_vec))
              {
                continue;
              }
            const double link_weight = core::Clamp (
                AnyToDouble (row.at ("link_weight"),
                             core::DerivedSourceFallbackEdgeWeight (
                                 cfg.focus, cfg.sensitivity, cfg.stability)),
                0.0, 1.0);
            const double sim
                = core::Clamp (std::max (0.0, core::CosineSimilarity (q, source_vec))
                                   * link_weight,
                               0.0, 1.0);
            auto score_it = max_source_score.find (candidate_memory_id);
            if (score_it == max_source_score.end ())
              {
                max_source_score.emplace (candidate_memory_id, sim);
              }
            else
              {
                score_it->second = std::max (score_it->second, sim);
              }
            source_counts[candidate_memory_id]++;
          }

        for (auto &candidate : candidates)
          {
            auto score_it = max_source_score.find (candidate.memory_id);
            if (score_it == max_source_score.end ())
              {
                continue;
              }
            const int source_count
                = std::max (0, source_counts[candidate.memory_id]);
            const double count_support = core::Clamp (
                std::log1p (static_cast<double> (source_count))
                    / std::log (core::RetrievalDurableSourceSupportSaturationCount (
                        cfg.focus, cfg.stability)),
                0.0, 1.0);
            const double source_base_weight
                = core::RetrievalDurableSourceSetBaseWeight (
                    cfg.focus, cfg.sensitivity, cfg.stability);
            const double source_score
                = core::Clamp (score_it->second
                                   * (source_base_weight
                                      + (1.0 - source_base_weight)
                                            * count_support),
                               0.0, 1.0);
            if (source_score < durable_source_min_score)
              {
                continue;
              }
            candidate.durable_source_boost
                = std::max (candidate.durable_source_boost, source_score);
            candidate.durable_source_count
                = std::max (candidate.durable_source_count, source_count);
          }
      }
    catch (...)
      {
      }
    const auto t_end = std::chrono::steady_clock::now ();
    context.AddOperationTiming ("GraphRetrieve.durable_source_set",
                                elapsed_ms (t_start, t_end));
  };

  auto activation_ledger = [&] (const Scored &s) {
    const double relevance = std::max (0.0, s.score);
    const double ctx_sim = std::max (0.0, s.ctx_score);
    const double proc_sim = std::max (0.0, s.proc_score);
    const double memory_affect
        = w_mem_emotion * s.emotion_intensity + w_mem_arousal * s.arousal_avg;
    const double emotion_bonus = affect_drive * memory_affect;
    const double association_bonus = s.is_association ? assoc_boost : 0.0;
    const double label_bonus = s.is_label ? label_boost : 0.0;
    const double temporal_bonus
        = s.is_label ? 0.0 : temporal_rank_weight * s.temporal_score;
    const double availability = base_level_availability (s);
    retrieval_debug::ActivationLedger ledger;
    ledger.base_level = w_rel * relevance + temporal_bonus + availability;
    ledger.spreading_activation
        = w_ctx * ctx_sim + w_emotion * emotion_bonus + association_bonus
          + label_bonus + label_graph_weight * s.label_graph_boost
          + durable_source_weight * s.durable_source_boost + s.fact_boost;
    ledger.partial_match_penalty = partial_match_penalty (s);
    ledger.recent_inhibition = recent_retrieval_inhibition (s);
    ledger.utility = w_proc * proc_sim + s.predictive_bonus;
    ledger.exploration_noise = 0.0;
    ledger.activation_total
        = ledger.base_level + ledger.spreading_activation
          + ledger.partial_match_penalty + ledger.recent_inhibition
          + ledger.utility + ledger.exploration_noise;
    return ledger;
  };

  auto base_score = [&] (const Scored &s) {
    return activation_ledger (s).activation_total;
  };

  auto ranked_candidate = [&] (const Scored &s) {
    const auto activation = activation_ledger (s);
    retrieval_debug::RankedCandidate candidate;
    candidate.embedding_id = s.embedding_id;
    candidate.memory_id = s.memory_id;
    candidate.score = activation.activation_total;
    candidate.relevance = std::max (0.0, s.score);
    candidate.proc_score = std::max (0.0, s.proc_score);
    candidate.predictive_bonus = s.predictive_bonus;
    candidate.pre_activation = s.pre_activation;
    candidate.fact_boost = s.fact_boost;
    candidate.fact_stale_penalty = s.fact_stale_penalty;
    candidate.linked_fact_count = s.linked_fact_count;
    candidate.label_graph_boost = s.label_graph_boost;
    candidate.label_match_count = s.label_match_count;
    candidate.durable_source_boost = s.durable_source_boost;
    candidate.durable_source_count = s.durable_source_count;
    candidate.temporal_score = s.temporal_score;
    candidate.activation = activation;
    return candidate;
  };

  auto build_evidence_packets = [&] (const std::vector<Scored> &selected) {
    std::vector<retrieval_debug::EvidencePacket> packets;
    if (!evidence_blending_enabled || selected.size () < 2
        || evidence_blend_max_members < 2)
      {
        return packets;
      }

    const double best_score = base_score (selected.front ());
    if (!std::isfinite (best_score))
      {
        return packets;
      }

    std::vector<std::pair<int, const Scored *>> members;
    members.reserve (
        static_cast<size_t> (std::max (2, evidence_blend_max_members)));
    for (int rank = 0; rank < static_cast<int> (selected.size ()); ++rank)
      {
        const auto &candidate = selected[static_cast<size_t> (rank)];
        const double score = base_score (candidate);
        if (!std::isfinite (score))
          {
            continue;
          }
        const double gap = best_score - score;
        if (gap > evidence_blend_tie_margin)
          {
            break;
          }
        members.emplace_back (rank, &candidate);
        if (static_cast<int> (members.size ()) >= evidence_blend_max_members)
          {
            break;
          }
      }
    if (members.size () < 2)
      {
        return packets;
      }

    const double temperature = std::max (evidence_blend_temperature,
                                         constants::kNormEpsilon);
    double unnormalized_sum = 0.0;
    std::vector<double> unnormalized;
    unnormalized.reserve (members.size ());
    for (const auto &[rank, candidate] : members)
      {
        (void)rank;
        const double weight = std::exp (
            (base_score (*candidate) - best_score) / temperature);
        unnormalized.push_back (weight);
        unnormalized_sum += weight;
      }
    if (unnormalized_sum <= constants::kNormEpsilon)
      {
        return packets;
      }

    retrieval_debug::EvidencePacket packet;
    packet.packet_id = 1;
    packet.consumer = "constructive_recall_summary";
    packet.reason = "near_tied_top_candidates";
    packet.tie_margin = evidence_blend_tie_margin;
    packet.temperature = temperature;
    packet.score_span = best_score - base_score (*members.back ().second);
    packet.members.reserve (members.size ());
    for (size_t i = 0; i < members.size (); ++i)
      {
        const auto &[rank, candidate] = members[i];
        const double weight = unnormalized[i] / unnormalized_sum;
        packet.activation_total += weight * base_score (*candidate);
        packet.members.push_back (
            { rank,
              candidate->embedding_id,
              candidate->memory_id,
              weight,
              base_score (*candidate),
              activation_ledger (*candidate) });
      }
    packets.push_back (std::move (packet));
    return packets;
  };

  auto append_evidence_blend_reconstructions =
      [&] (const std::vector<Scored> &selected,
           const std::vector<retrieval_debug::EvidencePacket> &packets) {
    if (!evidence_blending_enabled || disable_constructive_recall
        || packets.empty ())
      {
        return;
      }

    std::unordered_map<long long, const Scored *> selected_by_memory_id;
    selected_by_memory_id.reserve (selected.size ());
    for (const auto &candidate : selected)
      {
        if (candidate.memory_id > 0)
          {
            selected_by_memory_id.emplace (candidate.memory_id, &candidate);
          }
      }

    for (const auto &packet : packets)
      {
        if (packet.members.size () < 2)
          {
            continue;
          }
        const long long anchor_memory_id = packet.members.front ().memory_id;
        if (anchor_memory_id <= 0)
          {
            continue;
          }

        Eigen::VectorXf blended = Eigen::VectorXf::Zero (q.size ());
        double total_weight = 0.0;
        double source_confidence = 0.0;
        double context_similarity = 0.0;
        for (const auto &member : packet.members)
          {
            auto it = selected_by_memory_id.find (member.memory_id);
            if (it == selected_by_memory_id.end ())
              {
                continue;
              }
            const Scored &candidate = *it->second;
            if (candidate.vec.size () != q.size () || candidate.vec.size () == 0
                || member.weight <= 0.0)
              {
                continue;
              }
            blended += static_cast<float> (member.weight) * candidate.vec;
            total_weight += member.weight;
            source_confidence += member.weight
                                 * core::Clamp (
                                     candidate.source_confidence, 0.0, 1.0);
            context_similarity += member.weight
                                  * std::max (0.0, candidate.ctx_score);
          }
        if (total_weight <= constants::kNormEpsilon)
          {
            continue;
          }
        blended /= static_cast<float> (total_weight);
        const float norm = blended.norm ();
        if (norm <= 1e-9f)
          {
            continue;
          }
        blended /= norm;

        const double anchor_weight
            = core::Clamp (packet.members.front ().weight, 0.0, 1.0);
        const double uncertainty = core::Clamp (1.0 - anchor_weight, 0.0, 1.0);
        const auto blob_id
            = constructive_recall::LoadCurrentBlobId (tx, anchor_memory_id);
        constructive_recall::AppendReconstructionWithEmbedding (
            tx, anchor_memory_id, blended, blob_id,
            static_cast<long long> (signal.timestamp), uncertainty,
            "evidence_blend",
            core::Clamp (source_confidence / total_weight, 0.0, 1.0),
            core::Clamp (context_similarity / total_weight, 0.0, 1.0));
      }
  };

  auto collect_filter_rejections =
      [&] (const std::vector<Scored> &candidates,
           bool enforce_write_exclusion,
           bool enforce_source_conf,
           const std::string &stage) {
        std::vector<retrieval_debug::RejectedCandidate> rejected;
        rejected.reserve (candidates.size ());
        for (const auto &s : candidates)
          {
            std::string reason;
            double observed = 0.0;
            double threshold = 0.0;
            if (stored_id.has_value () && s.embedding_id == *stored_id)
              {
                reason = "stored_embedding";
                observed = static_cast<double> (s.embedding_id);
                threshold = static_cast<double> (*stored_id);
              }
            else
              {
                const bool summary_like = (s.is_association || s.is_label);
                if (enforce_write_exclusion && !summary_like
                    && s.created_at > 0
                    && static_cast<std::uint64_t> (s.created_at)
                           >= write_exclusion_ts)
                  {
                    reason = "write_exclusion";
                    observed = static_cast<double> (s.created_at);
                    threshold = static_cast<double> (write_exclusion_ts);
                  }
                else
                  {
                    const double cand_dup_thresh
                        = summary_like ? dup_thresh_summary : dup_thresh;
                    const double wm_overlap = max_wm_overlap (s.vec);
                    if (!(summary_like && bypass_summary_overlap)
                        && wm_overlap >= cand_dup_thresh)
                      {
                        reason = "working_memory_overlap";
                        observed = wm_overlap;
                        threshold = cand_dup_thresh;
                      }
                    else
                      {
                        const bool source_backed_by_graph
                            = s.durable_source_boost
                                  >= source_backed_boost_floor
                              || (s.durable_source_count > 0
                                  && s.label_graph_boost
                                         >= source_backed_boost_floor);
                        if (!disable_source_conf && enforce_source_conf
                            && !source_backed_by_graph
                            && s.source_confidence < source_thresh)
                          {
                            reason = "source_confidence";
                            observed = s.source_confidence;
                            threshold = source_thresh;
                          }
                      }
                  }
              }

            if (!reason.empty ())
              {
                rejected.push_back (
                    { ranked_candidate (s), reason, stage, observed,
                      threshold });
              }
          }
        return rejected;
      };

  auto append_unselected_rejections =
      [&] (std::vector<retrieval_debug::RejectedCandidate> &rejected,
           const std::vector<Scored> &eligible,
           const std::vector<Scored> &selected) {
        std::unordered_set<long long> selected_embeddings;
        selected_embeddings.reserve (selected.size ());
        double selected_min_score = 0.0;
        bool have_selected = false;
        for (const auto &s : selected)
          {
            selected_embeddings.insert (s.embedding_id);
            const double score = base_score (s);
            selected_min_score
                = have_selected ? std::min (selected_min_score, score) : score;
            have_selected = true;
          }
        for (const auto &s : eligible)
          {
            if (selected_embeddings.count (s.embedding_id) > 0)
              {
                continue;
              }
            rejected.push_back (
                { ranked_candidate (s),
                  "not_selected",
                  "selection",
                  base_score (s),
                  have_selected ? selected_min_score : 0.0 });
          }
      };

  auto is_source_backed = [&] (const Scored &candidate) {
    return candidate.durable_source_boost >= source_backed_boost_floor
           || (candidate.durable_source_count > 0
               && candidate.label_graph_boost >= source_backed_boost_floor);
  };

  auto is_source_backed_memory = [&] (const Scored &candidate) {
    return !candidate.is_association && !candidate.is_label
           && is_source_backed (candidate);
  };


  auto enforce_durable_source_floor =
      [&] (std::vector<Scored> &selected,
           const std::vector<Scored> &eligible) {
    if (durable_source_min_topk <= 0 || selected.empty () || eligible.empty ())
      {
        return;
      }
    const int target_count = std::min (
        durable_source_min_topk, static_cast<int> (selected.size ()));
    int source_backed_count = 0;
    std::unordered_set<long long> selected_embeddings;
    selected_embeddings.reserve (selected.size ());
    for (const auto &candidate : selected)
      {
        selected_embeddings.insert (candidate.embedding_id);
        if (is_source_backed_memory (candidate))
          {
            ++source_backed_count;
          }
      }
    while (source_backed_count < target_count)
      {
        int best_eligible_idx = -1;
        double best_eligible_score = -1e9;
        for (int i = 0; i < static_cast<int> (eligible.size ()); ++i)
          {
            const auto &candidate = eligible[static_cast<size_t> (i)];
            if (!is_source_backed_memory (candidate)
                || selected_embeddings.count (candidate.embedding_id) > 0)
              {
                continue;
              }
            const double score = base_score (candidate);
            if (score > best_eligible_score)
              {
                best_eligible_score = score;
                best_eligible_idx = i;
              }
          }
        if (best_eligible_idx < 0)
          {
            break;
          }

        int replace_idx = -1;
        double replace_score = 1e9;
        for (int i = 0; i < static_cast<int> (selected.size ()); ++i)
          {
            const auto &candidate = selected[static_cast<size_t> (i)];
            if (is_source_backed_memory (candidate))
              {
                continue;
              }
            // Proactive procedural selections hold reserved slots: they are
            // semantically dissimilar by design (routine recall, not
            // similarity recall) and would otherwise always be the cheapest
            // replacement victim.
            if (candidate.proc_score >= procedural_seed_min_score)
              {
                continue;
              }
            const double score = base_score (candidate);
            if (score < replace_score)
              {
                replace_score = score;
                replace_idx = i;
              }
          }
        if (replace_idx < 0)
          {
            break;
          }

        selected_embeddings.erase (
            selected[static_cast<size_t> (replace_idx)].embedding_id);
        selected[static_cast<size_t> (replace_idx)]
            = eligible[static_cast<size_t> (best_eligible_idx)];
        selected_embeddings.insert (
            selected[static_cast<size_t> (replace_idx)].embedding_id);
        ++source_backed_count;
      }
    std::stable_sort (selected.begin (), selected.end (),
                      [&] (const Scored &a, const Scored &b) {
                        return base_score (a) > base_score (b);
                      });
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
    const double cutoff = certainty_requirement * best_score
                          * core::RetrievalUnknownCautionCutoffMultiplier (
                              cfg.focus, cfg.sensitivity, cfg.stability,
                              unknown_caution_scale);
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
            const double alpha = core::RetrievalContextReinstatementAlpha (
                cfg.focus, cfg.sensitivity, cfg.stability);
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
        const auto reconstruction_policy
            = core::RetrievalConstructiveReconstructionPolicy (
                cfg.focus, cfg.sensitivity, cfg.stability);
        const double uncertainty = core::Clamp (
            reconstruction_policy.source_confidence_weight * (1.0 - source_conf)
                + reconstruction_policy.candidate_score_weight
                      * (1.0 - std::max (0.0, candidate.score))
                + reconstruction_policy.context_score_weight
                      * (1.0 - std::max (0.0, candidate.ctx_score)),
            0.0, 1.0);
        const double blend = core::Clamp (
            reconstruction_policy.min_blend
                + (reconstruction_policy.max_blend
                   - reconstruction_policy.min_blend)
                      * uncertainty,
            reconstruction_policy.min_blend, reconstruction_policy.max_blend);

        Eigen::VectorXf reconstructed
            = static_cast<float> (1.0 - blend) * candidate.vec
              + static_cast<float> (
                    blend * reconstruction_policy.query_weight)
                    * q;
        if (q_ctx.size () == q.size () && q_ctx.size () > 0)
          {
            reconstructed
                += static_cast<float> (
                       blend * (1.0 - reconstruction_policy.query_weight))
                   * q_ctx;
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

  // Reserved-slot seeding for diversified selection: high-evidence
  // association/label candidates and qualifying procedural candidates are
  // admitted ahead of MMR competition. Procedural seeds are proactive by
  // design - they surface routine memories regardless of semantic
  // similarity, so they must not have to win a similarity-ranked slot.
  // Both selection paths (vector-backed and the no-fetch fallback) use this.
  auto build_reserved_initial = [&] (const std::vector<Scored> &eligible) {
    const int reserve_assoc = std::min (min_assoc, k);
    const int reserve_label = std::min (min_label, k - reserve_assoc);
    int reserve_proc = 0;
    if (cfg.procedural_enabled && !disable_procedural_proactive)
      {
        reserve_proc = core::RetrievalProceduralSeedReserveCount (
            cfg.focus, cfg.sensitivity, cfg.stability,
            k - reserve_assoc - reserve_label);
      }

    auto pick_top = [&] (const std::vector<Scored> &candidates, int count,
                         const std::function<bool (const Scored &)> &pred) {
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
                 [] (const auto &a, const auto &b)
                 { return a.first > b.first; });
      std::vector<Scored> out;
      for (int i = 0; i < count && i < static_cast<int> (ranked.size ());
           ++i)
        {
          out.push_back (ranked[i].second);
        }
      return out;
    };

    std::vector<Scored> initial;
    initial.reserve (
        static_cast<size_t> (reserve_assoc + reserve_label + reserve_proc));
    std::unordered_set<long long> initial_ids;
    if (reserve_assoc > 0)
      {
        auto assoc_seeds = pick_top (
            eligible, reserve_assoc,
            [] (const Scored &s) { return s.is_association; });
        for (const auto &seed : assoc_seeds)
          {
            if (initial_ids.insert (seed.embedding_id).second)
              {
                initial.push_back (seed);
              }
          }
      }
    if (reserve_label > 0)
      {
        auto label_seeds = pick_top (
            eligible, reserve_label,
            [] (const Scored &s) { return s.is_label; });
        for (const auto &seed : label_seeds)
          {
            if (initial_ids.insert (seed.embedding_id).second)
              {
                initial.push_back (seed);
              }
          }
      }
    if (reserve_proc > 0)
      {
        auto proc_seeds = pick_top (
            eligible, reserve_proc,
            [procedural_seed_min_score] (const Scored &s)
            { return s.proc_score >= procedural_seed_min_score; });
        for (const auto &seed : proc_seeds)
          {
            if (initial_ids.insert (seed.embedding_id).second)
              {
                initial.push_back (seed);
              }
          }
      }
    return initial;
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
      apply_label_graph_scores (seeds);
      apply_durable_source_scores (seeds);
      const auto t_score_start = std::chrono::steady_clock::now ();
      FilterStats strict_stats;
      auto eligible = filter_candidates (seeds, true, true, &strict_stats);
      FilterStats relax_stats;
      bool used_relaxed_filter = false;
      if (eligible.empty ()
          && metacognitive_mode
                 != ProcessorContext::MetacognitiveMode::UnknownCaution)
        {
          eligible = filter_candidates (seeds, false, false, &relax_stats);
          used_relaxed_filter = true;
        }
      auto selected
          = select_diversified (eligible, k, build_reserved_initial (eligible));
      std::stable_sort (selected.begin (), selected.end (),
                        [&] (const Scored &a, const Scored &b) {
                          return base_score (a) > base_score (b);
                        });
      enforce_durable_source_floor (selected, eligible);
      apply_unknown_caution (selected);
      const auto evidence_packets = build_evidence_packets (selected);
      const auto t_score_end = std::chrono::steady_clock::now ();
      context.AddOperationTiming ("GraphRetrieve.score",
                                  elapsed_ms (t_score_start, t_score_end));
      for (const auto &s : selected)
        {
          out.emplace (s.embedding_id, s.vec);
        }
      {
        std::vector<long long> selected_order;
        selected_order.reserve (selected.size ());
        std::vector<retrieval_debug::RankedCandidate> ranked_candidates;
        ranked_candidates.reserve (selected.size ());
        auto rejected_candidates = collect_filter_rejections (
            seeds, !used_relaxed_filter, !used_relaxed_filter,
            used_relaxed_filter ? "relaxed_filter" : "strict_filter");
        append_unselected_rejections (rejected_candidates, eligible,
                                      selected);
        int rejected_filter_count = 0;
        int rejected_selection_count = 0;
        for (const auto &entry : rejected_candidates)
          {
            if (entry.stage == "selection")
              {
                ++rejected_selection_count;
              }
            else
              {
                ++rejected_filter_count;
              }
          }
        for (const auto &s : selected)
          {
            selected_order.push_back (s.embedding_id);
            ranked_candidates.push_back (ranked_candidate (s));
          }
        int candidate_fact_link_row_count = 0;
        for (const auto &[memory_id, links] : candidate_fact_links)
          {
            (void)memory_id;
            candidate_fact_link_row_count
                += static_cast<int> (links.size ());
          }
        int selected_fact_linked_count = 0;
        for (const auto &s : selected)
          {
            if (s.fact_boost > 0.0 || s.fact_stale_penalty > 0.0
                || s.linked_fact_count > 0)
              {
                ++selected_fact_linked_count;
              }
          }
        int evidence_packet_member_count = 0;
        for (const auto &packet : evidence_packets)
          {
            evidence_packet_member_count
                += static_cast<int> (packet.members.size ());
          }
        retrieval_debug::SetLastRetrievalSummary (
            { fact_layer_enabled,
              static_cast<int> (matched_facts.size ()),
              static_cast<int> (candidate_fact_links.size ()),
              candidate_fact_link_row_count,
              selected_fact_linked_count,
              query_text_token_count,
              text_query_wm_slots,
              appended_wm_chars,
              fact_text_candidate_count,
              fact_text_rejected_low_score_count,
              fact_text_match_count,
              fact_text_best_score,
              static_cast<int> (rejected_candidates.size ()),
              rejected_filter_count,
              rejected_selection_count,
              static_cast<int> (evidence_packets.size ()),
              evidence_packet_member_count });
        retrieval_debug::SetLastSelectedEmbeddingOrder (selected_order);
        retrieval_debug::SetLastRankedCandidates (ranked_candidates);
        retrieval_debug::SetLastRejectedCandidates (rejected_candidates);
        retrieval_debug::SetLastEvidencePackets (evidence_packets);
      }
      append_reconstruction_versions (selected);
      append_evidence_blend_reconstructions (selected, evidence_packets);
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
  apply_label_graph_scores (scored);
  apply_durable_source_scores (scored);
  FilterStats strict_stats;
  const auto t_score_start = std::chrono::steady_clock::now ();
  auto eligible = filter_candidates (scored, true, true, &strict_stats);
  FilterStats relax_stats;
  bool used_relaxed_filter = false;
  if (eligible.empty ()
      && metacognitive_mode
             != ProcessorContext::MetacognitiveMode::UnknownCaution)
    {
      eligible = filter_candidates (scored, false, false, &relax_stats);
      used_relaxed_filter = true;
    }
  const auto initial = build_reserved_initial (eligible);
  auto selected = select_diversified (eligible, k, initial);
  std::stable_sort (selected.begin (), selected.end (),
                    [&] (const Scored &a, const Scored &b) {
                      return base_score (a) > base_score (b);
                    });
  enforce_durable_source_floor (selected, eligible);
  apply_unknown_caution (selected);
  const auto evidence_packets = build_evidence_packets (selected);
  const auto t_score_end = std::chrono::steady_clock::now ();
  context.AddOperationTiming ("GraphRetrieve.score",
                              elapsed_ms (t_score_start, t_score_end));
  for (const auto &s : selected)
    {
      out.emplace (s.embedding_id, s.vec);
    }
  {
    std::vector<long long> selected_order;
    selected_order.reserve (selected.size ());
    std::vector<retrieval_debug::RankedCandidate> ranked_candidates;
    ranked_candidates.reserve (selected.size ());
    auto rejected_candidates = collect_filter_rejections (
        scored, !used_relaxed_filter, !used_relaxed_filter,
        used_relaxed_filter ? "relaxed_filter" : "strict_filter");
    append_unselected_rejections (rejected_candidates, eligible, selected);
    int rejected_filter_count = 0;
    int rejected_selection_count = 0;
    for (const auto &entry : rejected_candidates)
      {
        if (entry.stage == "selection")
          {
            ++rejected_selection_count;
          }
        else
          {
            ++rejected_filter_count;
          }
      }
    for (const auto &s : selected)
      {
        selected_order.push_back (s.embedding_id);
        ranked_candidates.push_back (ranked_candidate (s));
      }
    int candidate_fact_link_row_count = 0;
    for (const auto &[memory_id, links] : candidate_fact_links)
      {
        (void)memory_id;
        candidate_fact_link_row_count += static_cast<int> (links.size ());
      }
    int selected_fact_linked_count = 0;
    for (const auto &s : selected)
      {
        if (s.fact_boost > 0.0 || s.fact_stale_penalty > 0.0
            || s.linked_fact_count > 0)
          {
            ++selected_fact_linked_count;
          }
      }
    int evidence_packet_member_count = 0;
    for (const auto &packet : evidence_packets)
      {
        evidence_packet_member_count
            += static_cast<int> (packet.members.size ());
      }
    retrieval_debug::SetLastRetrievalSummary (
        { fact_layer_enabled,
          static_cast<int> (matched_facts.size ()),
          static_cast<int> (candidate_fact_links.size ()),
          candidate_fact_link_row_count,
          selected_fact_linked_count,
          query_text_token_count,
          text_query_wm_slots,
          appended_wm_chars,
          fact_text_candidate_count,
          fact_text_rejected_low_score_count,
          fact_text_match_count,
          fact_text_best_score,
          static_cast<int> (rejected_candidates.size ()),
          rejected_filter_count,
          rejected_selection_count,
          static_cast<int> (evidence_packets.size ()),
          evidence_packet_member_count });
    retrieval_debug::SetLastSelectedEmbeddingOrder (selected_order);
    retrieval_debug::SetLastRankedCandidates (ranked_candidates);
    retrieval_debug::SetLastRejectedCandidates (rejected_candidates);
    retrieval_debug::SetLastEvidencePackets (evidence_packets);
  }

  append_reconstruction_versions (selected);
  append_evidence_blend_reconstructions (selected, evidence_packets);
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
  double selected_label_graph_boost_sum = 0.0;
  double selected_durable_source_boost_sum = 0.0;
  double selected_temporal_score_sum = 0.0;
  double selected_activation_base_level_sum = 0.0;
  double selected_activation_spreading_sum = 0.0;
  double selected_activation_partial_penalty_sum = 0.0;
  double selected_activation_recent_inhibition_sum = 0.0;
  double selected_activation_utility_sum = 0.0;
  double selected_activation_exploration_noise_sum = 0.0;
  double selected_activation_total_sum = 0.0;
  int64_t evidence_packet_member_count = 0;
  for (const auto &packet : evidence_packets)
    {
      evidence_packet_member_count
          += static_cast<int64_t> (packet.members.size ());
    }
  count_kinds (scored, scored_assoc, scored_label, scored_other);
  count_kinds (eligible, eligible_assoc, eligible_label, eligible_other);
  count_kinds (selected, selected_assoc, selected_label, selected_other);
  count_kinds (initial, initial_assoc, initial_label, initial_other);
  for (const auto &s : selected)
    {
      const auto activation = activation_ledger (s);
      if (s.fact_boost > 0.0 || s.fact_stale_penalty > 0.0)
        {
          fact_linked_selected++;
        }
      selected_predictive_bonus_sum += s.predictive_bonus;
      selected_preactivation_sum += s.pre_activation;
      selected_fact_boost_sum += s.fact_boost;
      selected_stale_penalty_sum += s.fact_stale_penalty;
      selected_label_graph_boost_sum += s.label_graph_boost;
      selected_durable_source_boost_sum += s.durable_source_boost;
      selected_temporal_score_sum += s.temporal_score;
      selected_activation_base_level_sum += activation.base_level;
      selected_activation_spreading_sum += activation.spreading_activation;
      selected_activation_partial_penalty_sum
          += activation.partial_match_penalty;
      selected_activation_recent_inhibition_sum
          += activation.recent_inhibition;
      selected_activation_utility_sum += activation.utility;
      selected_activation_exploration_noise_sum
          += activation.exploration_noise;
      selected_activation_total_sum += activation.activation_total;
    }

  telemetry::LogDebug ("cortext.graph_retrieval", {
    telemetry::Attribute::Double ("dup_thresh", dup_thresh),
    telemetry::Attribute::Int64 ("write_exclusion_ts", static_cast<int64_t> (write_exclusion_ts)),
    telemetry::Attribute::Bool ("stored_id_present", stored_id.has_value ()),
      telemetry::Attribute::Int64 ("selected_count", static_cast<int64_t> (selected.size ())),
      telemetry::Attribute::Int64 ("seed_count", static_cast<int64_t> (seeds.size ())),
      telemetry::Attribute::Int64 ("seed_k", static_cast<int64_t> (seed_k)),
      telemetry::Attribute::Int64 ("selected_k", static_cast<int64_t> (k)),
      telemetry::Attribute::Int64 ("procedural_seed_count", procedural_seed_count),
    telemetry::Attribute::Int64 ("hierarchical_label_seed_count",
                                 hierarchical_label_seed_count),
    telemetry::Attribute::Int64 ("wm_temporal_anchor_count",
                                 wm_temporal_anchor_count),
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
    telemetry::Attribute::Double ("retrieval_temporal_rank_weight",
                                  temporal_rank_weight),
    telemetry::Attribute::Double ("retrieval_temporal_rank_tau_seconds",
                                  temporal_rank_tau_seconds),
    telemetry::Attribute::Bool ("recent_retrieval_inhibition_enabled",
                                recent_retrieval_inhibition_enabled),
    telemetry::Attribute::Double ("recent_retrieval_inhibition_weight",
                                  recent_retrieval_inhibition_weight),
      telemetry::Attribute::Double (
          "recent_retrieval_inhibition_tau_seconds",
          recent_retrieval_inhibition_tau_seconds),
      telemetry::Attribute::Bool ("base_level_availability_enabled",
                                  base_level_availability_enabled),
      telemetry::Attribute::Double ("base_level_availability_weight",
                                    base_level_availability_weight),
      telemetry::Attribute::Double (
          "base_level_availability_tau_seconds",
          base_level_availability_tau_seconds),
      telemetry::Attribute::Double (
          "base_level_availability_count_saturation",
          base_level_availability_count_saturation),
    telemetry::Attribute::Bool ("partial_matching_enabled",
                                partial_matching_enabled),
    telemetry::Attribute::Double ("partial_match_penalty_weight",
                                  partial_match_penalty_weight),
    telemetry::Attribute::Double (
        "partial_match_contradiction_saturation",
        partial_match_contradiction_saturation),
    telemetry::Attribute::Double (
        "partial_match_source_mismatch_weight",
        partial_match_source_mismatch_weight),
    telemetry::Attribute::Double (
        "partial_match_modality_mismatch_weight",
        partial_match_modality_mismatch_weight),
    telemetry::Attribute::Bool ("evidence_blending_enabled",
                                evidence_blending_enabled),
    telemetry::Attribute::Double ("evidence_blend_tie_margin",
                                  evidence_blend_tie_margin),
    telemetry::Attribute::Double ("evidence_blend_temperature",
                                  evidence_blend_temperature),
    telemetry::Attribute::Int64 (
        "evidence_blend_max_members",
        static_cast<int64_t> (evidence_blend_max_members)),
    telemetry::Attribute::Int64 (
        "evidence_packet_count",
        static_cast<int64_t> (evidence_packets.size ())),
    telemetry::Attribute::Int64 ("evidence_packet_member_count",
                                 evidence_packet_member_count),
    telemetry::Attribute::Bool ("preconsolidated_label_graph_enabled",
                                preconsolidated_label_graph_enabled),
    telemetry::Attribute::Double ("preconsolidated_label_graph_weight",
                                  label_graph_weight),
    telemetry::Attribute::Bool ("durable_source_set_enabled",
                                durable_source_scoring_enabled),
    telemetry::Attribute::Double ("durable_source_set_weight",
                                  durable_source_weight),
    telemetry::Attribute::Int64 (
        "preconsolidated_label_graph_top_labels",
        static_cast<int64_t> (label_graph_top_labels)),
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
        "resurfacing_decay_mode",
        temporal::ToString (resurfacing_decay_mode)),
    telemetry::Attribute::Double ("resurfacing_decay_scale",
                                  resurfacing_decay_scale),
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
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "label_graph_mean_boost",
        selected.empty () ? 0.0
                          : selected_label_graph_boost_sum
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "durable_source_set_mean_boost",
        selected.empty () ? 0.0
                          : selected_durable_source_boost_sum
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "temporal_rank_mean_score",
        selected.empty () ? 0.0
                          : selected_temporal_score_sum
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "activation_base_level_mean",
        selected.empty () ? 0.0
                          : selected_activation_base_level_sum
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "activation_spreading_mean",
        selected.empty () ? 0.0
                          : selected_activation_spreading_sum
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "activation_partial_match_penalty_mean",
        selected.empty () ? 0.0
                          : selected_activation_partial_penalty_sum
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "activation_recent_inhibition_mean",
        selected.empty () ? 0.0
                          : selected_activation_recent_inhibition_sum
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "activation_utility_mean",
        selected.empty () ? 0.0
                          : selected_activation_utility_sum
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "activation_exploration_noise_mean",
        selected.empty () ? 0.0
                          : selected_activation_exploration_noise_sum
                                / static_cast<double> (selected.size ())),
    telemetry::Attribute::Double (
        "activation_total_mean",
        selected.empty () ? 0.0
                          : selected_activation_total_sum
                                / static_cast<double> (selected.size ()))
  });
}

} // namespace cortext::operations
