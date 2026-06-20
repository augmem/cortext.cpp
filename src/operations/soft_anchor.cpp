#include "cortext/operations/soft_anchor.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/processor/processor_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace cortext::operations
{
namespace
{

struct SignalViews
{
  Eigen::VectorXf semantic;
  Eigen::VectorXf entity;
  Eigen::VectorXf full;
  double q_semantic = 1.0;
  double q_entity = 0.5;
  double q_full = 1.0;
};

struct CandidateScore
{
  size_t index = 0;
  double raw = 0.0;
  double posterior = 0.0;
  double s_semantic = 0.0;
  double s_entity = 0.0;
  double s_full = 0.0;
  double margin = 0.0;
};

double
Clamp01 (double value)
{
  if (!std::isfinite (value))
    {
      return 0.0;
    }
  return std::max (0.0, std::min (1.0, value));
}

Eigen::VectorXf
Normalize (const Eigen::VectorXf &v)
{
  if (v.size () == 0)
    {
      return Eigen::VectorXf ();
    }
  const float norm = v.norm ();
  if (norm <= 1e-9f || !std::isfinite (norm))
    {
      return Eigen::VectorXf::Zero (v.size ());
    }
  return v / norm;
}

Eigen::VectorXf
SliceNormalize (const Eigen::VectorXf &v, Eigen::Index offset,
                Eigen::Index count)
{
  if (offset < 0 || count <= 0 || offset + count > v.size ())
    {
      return Eigen::VectorXf ();
    }
  return Normalize (v.segment (offset, count));
}

SignalViews
ExtractViews (const Eigen::VectorXf &embedding, double focus,
              double sensitivity, double stability)
{
  SignalViews views;
  views.full = Normalize (embedding);
  if (embedding.size () >= 1536)
    {
      views.semantic = SliceNormalize (embedding, 0, 768);
      views.entity = SliceNormalize (embedding, 768, 768);
      views.full = SliceNormalize (embedding, 0, 1536);
      views.q_entity = 1.0;
    }
  else
    {
      // Existing production embeddings are single-view. Keep Soft Anchor usable
      // without changing retrieval embeddings; entity quality remains weaker.
      views.semantic = views.full;
      views.entity = views.full;
      views.q_entity = core::SoftAnchorSingleViewEntityQuality (
          focus, sensitivity, stability);
    }
  return views;
}

double
Map01Cosine (const Eigen::VectorXf &a, const Eigen::VectorXf &b)
{
  if (a.size () == 0 || b.size () == 0 || a.size () != b.size ())
    {
      return 0.0;
    }
  const float an = a.norm ();
  const float bn = b.norm ();
  if (an <= 1e-9f || bn <= 1e-9f)
    {
      return 0.0;
    }
  const double cosine = static_cast<double> (a.dot (b) / (an * bn));
  return Clamp01 ((cosine + 1.0) * 0.5);
}

double
Entropy01 (const std::vector<double> &probabilities)
{
  if (probabilities.size () <= 1)
    {
      return 0.0;
    }
  double entropy = 0.0;
  for (const double p : probabilities)
    {
      if (p > 1e-12)
        {
          entropy -= p * std::log (p);
        }
    }
  return Clamp01 (entropy / std::log (static_cast<double> (probabilities.size ())));
}

std::vector<double>
Softmax (const std::vector<double> &values, double beta)
{
  if (values.empty ())
    {
      return {};
    }
  const double max_value = *std::max_element (values.begin (), values.end ());
  std::vector<double> exps;
  exps.reserve (values.size ());
  double sum = 0.0;
  for (const double value : values)
    {
      const double e = std::exp (beta * (value - max_value));
      exps.push_back (e);
      sum += e;
    }
  if (sum <= 1e-12)
    {
      const double uniform = 1.0 / static_cast<double> (values.size ());
      return std::vector<double> (values.size (), uniform);
    }
  for (double &e : exps)
    {
      e /= sum;
    }
  return exps;
}

std::string
JoinMemoryIds (const std::deque<long long> &ids)
{
  std::ostringstream out;
  bool first = true;
  for (const auto id : ids)
    {
      if (!first)
        {
          out << ",";
        }
      first = false;
      out << id;
    }
  return out.str ();
}

std::string
MemoryTier (int age, int working_ttl, int active_ttl)
{
  if (age <= working_ttl)
    {
      return "WM";
    }
  if (age <= active_ttl)
    {
      return "STM";
    }
  return "LTM";
}

double
Median (std::vector<double> values)
{
  if (values.empty ())
    {
      return 0.0;
    }
  std::sort (values.begin (), values.end ());
  return values[values.size () / 2];
}

void
BlendCentroid (Eigen::VectorXf &centroid, const Eigen::VectorXf &value,
               double alpha)
{
  if (value.size () == 0)
    {
      return;
    }
  if (centroid.size () != value.size ())
    {
      centroid = value;
      return;
    }
  centroid = Normalize ((1.0 - alpha) * centroid + alpha * value);
}

void
PersistAnchor (Transaction &tx,
               const ProcessorContext::SoftAnchorState &anchor,
               uint64_t updated_at)
{
  tx.Execute (
      "INSERT OR REPLACE INTO soft_anchors "
      "(anchor_id, status, semantic_centroid, entity_centroid, full_centroid, "
      " semantic_radius, entity_radius, full_radius, source_id, first_step, "
      " last_step, first_ts, last_ts, last_boundary_id, anchor_strength, "
      " support_count, contradiction_count, recent_memory_ids, updated_at) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { anchor.anchor_id,
        anchor.status,
        store::EigenToFloatVec (anchor.semantic_centroid),
        store::EigenToFloatVec (anchor.entity_centroid),
        store::EigenToFloatVec (anchor.full_centroid),
        anchor.semantic_radius,
        anchor.entity_radius,
        anchor.full_radius,
        anchor.last_source_id,
        static_cast<long long> (anchor.first_step),
        static_cast<long long> (anchor.last_step),
        static_cast<long long> (anchor.first_ts),
        static_cast<long long> (anchor.last_ts),
        static_cast<long long> (anchor.last_boundary_id),
        anchor.anchor_strength,
        static_cast<long long> (anchor.support_count),
        static_cast<long long> (anchor.contradiction_count),
        JoinMemoryIds (anchor.recent_memory_ids),
        static_cast<long long> (updated_at) });
}

void
PersistLink (Transaction &tx, const ProcessorContext::SoftAnchorLink &link,
             uint64_t created_at)
{
  tx.Execute (
      "INSERT OR REPLACE INTO soft_anchor_links "
      "(memory_id, anchor_id, anchor_strength, anchor_label, evidence_kind, "
      " memory_tier, score, margin, entropy, support_count, "
      " contradiction_count, created_step, updated_step, created_at) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { link.memory_id,
        link.anchor_id,
        link.anchor_strength,
        link.anchor_label,
        link.evidence_kind,
        link.memory_tier,
        link.score,
        link.margin,
        link.entropy,
        static_cast<long long> (link.support_count),
        static_cast<long long> (link.contradiction_count),
        static_cast<long long> (link.created_step),
        static_cast<long long> (link.updated_step),
        static_cast<long long> (created_at) });
}

} // namespace

void
UpdateSoftAnchor::Execute (OperationContext &context, Transaction &tx) const
{
  auto &processor = context.GetProcessorContext ();
  processor.soft_anchor_enabled = true;
  processor.soft_anchor_last_links.clear ();
  processor.soft_anchor_last_link_count = 0;
  processor.soft_anchor_last_create_count = 0;
  processor.soft_anchor_last_update_count = 0;
  processor.soft_anchor_last_none_count = 0;
  const auto started = std::chrono::steady_clock::now ();

  const auto memory_id = context.GetStoredMemoryId ();
  if (!memory_id.has_value () || *memory_id <= 0)
    {
      processor.soft_anchor_last_none_count = 1;
      return;
    }

  const auto &signal = context.GetSignal ();
  const Eigen::VectorXf *embedding = &signal.embedding;
  if (signal.soft_anchor_embedding.has_value ()
      && signal.soft_anchor_embedding->size () > 0)
    {
      embedding = &(*signal.soft_anchor_embedding);
    }
  else if (const auto &representative = context.GetRepresentativeEmbedding ();
           representative.has_value () && representative->size () > 0)
    {
      embedding = &(*representative);
    }
  if (embedding->size () == 0)
    {
      processor.soft_anchor_last_none_count = 1;
      return;
    }

  const double F = Clamp01 (context.GetConfig ().focus);
  const double S = Clamp01 (context.GetConfig ().sensitivity);
  const double T = Clamp01 (context.GetConfig ().stability);
  const SignalViews views = ExtractViews (*embedding, F, S, T);
  const int step = processor.signals_processed;
  const int working_ttl = std::max (
      1, core::SoftAnchorHardBoundaryRetainSteps (F, S, T));
  const int active_ttl = std::max (1, core::SoftAnchorActiveTTLSteps (T));
  const std::size_t recent_memory_limit = static_cast<std::size_t> (
      std::max (1, core::SoftAnchorRecentMemoryLimit (F, S, T)));
  const double support_raw_threshold
      = core::SoftAnchorSupportRawThreshold (F, S, T);
  const double support_margin_threshold
      = core::SoftAnchorSupportMarginThreshold (F, S, T);
  const double support_entropy_max = core::SoftAnchorSupportEntropyMax (F, S, T);
  const double contradiction_semantic_threshold
      = core::SoftAnchorContradictionSemanticThreshold (F, S, T);
  const double contradiction_entity_max
      = core::SoftAnchorContradictionEntityMax (F, S, T);
  const double durable_margin_threshold
      = core::SoftAnchorDurableMarginThreshold (F, S, T);
  const double durable_entropy_max
      = core::SoftAnchorDurableEntropyMax (F, S, T);
  const double boundary_score = context.GetBoundaryScore ().value_or (0.0);
  const int64_t boundary_id = processor.episode_start_ts > 0
                                  ? static_cast<int64_t> (processor.episode_start_ts)
                                  : 0;

  const auto view_weights = core::SoftAnchorViewScoringWeights (
      F, S, T, views.q_semantic, views.q_entity, views.q_full);
  const auto candidate_weights
      = core::SoftAnchorCandidateScoringWeights (F, S, T);
  const auto decision_policy = core::SoftAnchorDecisionPolicyForKnobs (F, S, T);

  std::vector<CandidateScore> candidates;
  candidates.reserve (processor.soft_anchor_states.size ());
  std::vector<double> raw_scores;
  for (size_t i = 0; i < processor.soft_anchor_states.size (); ++i)
    {
      auto &anchor = processor.soft_anchor_states[i];
      if (anchor.status == "rejected")
        {
          continue;
        }
      const int age = std::max (0, step - anchor.last_step);
      if (anchor.status != "durable"
          && static_cast<double> (age)
                 > static_cast<double> (active_ttl)
                       * core::SoftAnchorActiveDecayMultiplier (F, S, T))
        {
          anchor.status = "decayed";
        }

      CandidateScore score;
      score.index = i;
      score.s_semantic = Map01Cosine (views.semantic, anchor.semantic_centroid);
      score.s_entity = Map01Cosine (views.entity, anchor.entity_centroid);
      score.s_full = Map01Cosine (views.full, anchor.full_centroid);
      const double view = view_weights.semantic * score.s_semantic
                          + view_weights.entity * score.s_entity
                          + view_weights.full * score.s_full;
      const double source = core::SoftAnchorSourceAffinity (
          F, S, T, anchor.last_source_id == signal.source_id);
      const double tau = core::SoftAnchorRecencyTauSteps (T);
      const double recency = std::exp (-std::log (2.0) * age / tau);
      const double k_support = core::SoftAnchorSupportSaturation (T);
      const double support = 1.0 - std::exp (
                                -static_cast<double> (anchor.support_count)
                                    / k_support);
      const double k_contra = core::SoftAnchorContradictionSaturation (S);
      const double contra = 1.0 - std::exp (
                               -static_cast<double> (anchor.contradiction_count)
                                   / k_contra);
      const double boundary_shift
          = anchor.last_boundary_id == boundary_id
                ? 0.0
                : Clamp01 (boundary_score
                           * core::SoftAnchorBoundaryShiftScale (F, T));
      score.raw = Clamp01 (candidate_weights.view * view
                           + candidate_weights.source * source
                           + candidate_weights.recency * recency
                           + candidate_weights.support * support
                           - candidate_weights.boundary * boundary_shift
                           - candidate_weights.contradiction * contra);
      candidates.push_back (score);
      raw_scores.push_back (score.raw);
    }

  const bool has_real_candidates = !raw_scores.empty ();
  const double best_real = raw_scores.empty ()
                               ? 0.0
                               : *std::max_element (raw_scores.begin (),
                                                   raw_scores.end ());
  const double specificity = Clamp01 (best_real - Median (raw_scores));
  const double real_entropy = raw_scores.empty () ? 0.0 : Entropy01 (
      Softmax (raw_scores, core::SoftAnchorSoftmaxTemperature (F)));
  const double generic = core::SoftAnchorGenericScore (
      F, S, T, views.q_entity, real_entropy, specificity);
  const double info = Clamp01 (
                          view_weights.semantic * views.q_semantic
                          + view_weights.entity * views.q_entity
                          + view_weights.full * views.q_full)
                      * (1.0 - generic);
  const double top_contra = 0.0;
  const double no_real_penalty = has_real_candidates ? (1.0 - best_real)
                                                     : (1.0 - info);
  const double score_none = core::SoftAnchorNoneScore (
      F, S, T, generic, real_entropy, no_real_penalty, top_contra);
  const double score_new = core::SoftAnchorNewScore (
      F, S, T, info, boundary_score, best_real, generic);

  std::vector<double> hypothesis_scores = raw_scores;
  const size_t none_index = hypothesis_scores.size ();
  hypothesis_scores.push_back (score_none);
  const size_t new_index = hypothesis_scores.size ();
  hypothesis_scores.push_back (score_new);
  const auto probabilities = Softmax (
      hypothesis_scores, core::SoftAnchorSoftmaxTemperature (F));
  const double entropy = Entropy01 (probabilities);
  const auto winner = static_cast<size_t> (std::distance (
      probabilities.begin (),
      std::max_element (probabilities.begin (), probabilities.end ())));

  if ((winner == none_index)
      || generic >= decision_policy.generic_hard_threshold)
    {
      processor.soft_anchor_last_none_count = 1;
    }
  else if (winner == new_index
           && probabilities[new_index] >= decision_policy.new_threshold)
    {
      ProcessorContext::SoftAnchorState anchor;
      anchor.anchor_id = "soft_anchor_"
                         + std::to_string (processor.soft_anchor_next_id++);
      anchor.status = "provisional";
      anchor.last_source_id = signal.source_id;
      anchor.semantic_centroid = views.semantic;
      anchor.entity_centroid = views.entity;
      anchor.full_centroid = views.full;
      anchor.anchor_strength = probabilities[new_index];
      anchor.support_count = 1;
      anchor.first_step = step;
      anchor.last_step = step;
      anchor.first_ts = signal.timestamp;
      anchor.last_ts = signal.timestamp;
      anchor.last_boundary_id = boundary_id;
      anchor.recent_memory_ids.push_back (*memory_id);
      processor.soft_anchor_states.push_back (std::move (anchor));
      processor.soft_anchor_last_create_count = 1;

      ProcessorContext::SoftAnchorLink link;
      link.memory_id = *memory_id;
      link.anchor_id = processor.soft_anchor_states.back ().anchor_id;
      link.anchor_strength = probabilities[new_index];
      link.anchor_label = "tentative";
      link.evidence_kind = "new";
      link.memory_tier = "WM";
      link.score = score_new;
      link.margin = probabilities[new_index]
                    - *std::max_element (probabilities.begin (),
                                         probabilities.end () - 1);
      link.entropy = entropy;
      link.support_count = 1;
      link.created_step = step;
      link.updated_step = step;
      processor.soft_anchor_last_links.push_back (std::move (link));
    }
  else if (!candidates.empty ())
    {
      for (size_t i = 0; i < candidates.size (); ++i)
        {
          candidates[i].posterior = probabilities[i];
        }
      std::sort (candidates.begin (), candidates.end (),
                 [] (const CandidateScore &a, const CandidateScore &b) {
                   return a.posterior > b.posterior;
                 });

      const double second = candidates.size () > 1 ? candidates[1].posterior
                                                   : std::max (
                                                         probabilities[none_index],
                                                         probabilities[new_index]);
      const double top_margin = candidates[0].posterior - second;
      const bool ambiguous = candidates.size () > 1
                             && (top_margin < decision_policy.margin_min
                                 || entropy
                                        >= decision_policy.ambiguous_entropy);
      const int keep = decision_policy.keep_count;
      for (int kept = 0;
           kept < keep && kept < static_cast<int> (candidates.size ());
           ++kept)
        {
          const auto &candidate = candidates[static_cast<size_t> (kept)];
          if (candidate.posterior < decision_policy.keep_threshold
              || candidate.raw < decision_policy.tentative_threshold)
            {
              continue;
            }
          auto &anchor = processor.soft_anchor_states[candidate.index];
          const int age = std::max (0, step - anchor.last_step);
          ProcessorContext::SoftAnchorLink link;
          link.memory_id = *memory_id;
          link.anchor_id = anchor.anchor_id;
          link.anchor_strength = candidate.posterior;
          link.anchor_label = ambiguous ? "ambiguous" : "tentative";
          link.evidence_kind = "continued";
          link.memory_tier = MemoryTier (age, working_ttl, active_ttl);
          link.score = candidate.raw;
          link.margin = kept == 0 ? top_margin : candidate.posterior - second;
          link.entropy = entropy;
          link.support_count = anchor.support_count;
          link.contradiction_count = anchor.contradiction_count;
          link.created_step = anchor.first_step;
          link.updated_step = step;
          processor.soft_anchor_last_links.push_back (std::move (link));
        }

      if (!processor.soft_anchor_last_links.empty ())
        {
          auto &anchor = processor.soft_anchor_states[candidates[0].index];
          const double alpha = decision_policy.centroid_alpha;
          BlendCentroid (anchor.semantic_centroid, views.semantic, alpha);
          BlendCentroid (anchor.entity_centroid, views.entity, alpha);
          BlendCentroid (anchor.full_centroid, views.full, alpha);
          anchor.semantic_radius = std::max (
              anchor.semantic_radius, 1.0 - candidates[0].s_semantic);
          anchor.entity_radius = std::max (
              anchor.entity_radius, 1.0 - candidates[0].s_entity);
          anchor.full_radius = std::max (
              anchor.full_radius, 1.0 - candidates[0].s_full);
          anchor.anchor_strength = std::max (anchor.anchor_strength,
                                             candidates[0].posterior);
          anchor.last_source_id = signal.source_id;
          anchor.last_step = step;
          anchor.last_ts = signal.timestamp;
          anchor.last_boundary_id = boundary_id;
          anchor.recent_memory_ids.push_back (*memory_id);
          while (anchor.recent_memory_ids.size () > recent_memory_limit)
            {
              anchor.recent_memory_ids.pop_front ();
            }
          if (candidates[0].raw >= support_raw_threshold
              && top_margin >= support_margin_threshold
              && entropy <= support_entropy_max)
            {
              anchor.support_count += 1;
            }
          if (candidates[0].s_semantic >= contradiction_semantic_threshold
              && candidates[0].s_entity <= contradiction_entity_max)
            {
              anchor.contradiction_count += 1;
            }
          const int support_required = decision_policy.durable_support_required;
          if (anchor.support_count >= support_required
              && anchor.contradiction_count == 0
              && top_margin >= durable_margin_threshold
              && entropy <= durable_entropy_max)
            {
              anchor.status = "durable";
              processor.soft_anchor_last_links.front ().anchor_label = "durable";
            }
          else if (anchor.status == "provisional")
            {
              anchor.status = "active";
            }
          processor.soft_anchor_last_update_count = 1;
        }
    }

  for (const auto &link : processor.soft_anchor_last_links)
    {
      PersistLink (tx, link, signal.timestamp);
    }
  for (const auto &anchor : processor.soft_anchor_states)
    {
      PersistAnchor (tx, anchor, signal.timestamp);
    }

  processor.soft_anchor_last_link_count
      = static_cast<int> (processor.soft_anchor_last_links.size ());
  processor.soft_anchor_last_state_count
      = static_cast<int> (processor.soft_anchor_states.size ());
  processor.soft_anchor_update_count += 1;
  const auto elapsed = std::chrono::duration<double, std::micro> (
      std::chrono::steady_clock::now () - started);
  processor.soft_anchor_last_update_us = elapsed.count ();
  processor.soft_anchor_total_update_us += elapsed.count ();
  processor.soft_anchor_recent_update_us.push_back (elapsed.count ());
  while (processor.soft_anchor_recent_update_us.size () > 512)
    {
      processor.soft_anchor_recent_update_us.pop_front ();
    }
}

} // namespace cortext::operations
