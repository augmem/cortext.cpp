#include "cortext/operations/emotion_cascade.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

namespace cortext::operations
{

EmotionCascadeParams
EmotionCascadeParams::FromKnobs (double /*F*/, double S, double /*T*/)
{
  EmotionCascadeParams p;
  p.cascade_radius = core::CascadeRadius (S);
  p.cascade_decay = core::CascadeDecay (S);
  return p;
}

namespace
{

/// @brief Adds a write instruction to the transaction.
void
Add (Transaction &tx, const std::string &q,
     const std::vector<std::any> &p = {})
{
  tx.Execute (q, p);
}

/// @brief Structure for emotional source memories.
struct EmotionalSource
{
  long long embedding_id;
  double intensity;
  double arousal;
  double valence;
  double half_life_bonus;
  int cascade_radius;
  double cascade_decay;
};

/// @brief Load recently tagged high-intensity emotional memories.
/// v2: Uses memories table (emotional fields merged from emotional_tags)
std::vector<EmotionalSource>
LoadEmotionalSources (Store *store, long long recent_window_ts,
                      double theta_intensity, double theta_arousal,
                      const EmotionCascadeParams &fallback)
{
  std::vector<EmotionalSource> sources;

  // v2: Query memories for high-intensity flashbulb memories
  // Only process recently tagged ones (within consolidation window)
  auto rows = store->Execute (
      "SELECT embedding_id, emotional_intensity, half_life_bonus, "
      "       cascade_radius, cascade_decay "
      "FROM memories "
      "WHERE flashbulb = 1 AND emotional_intensity >= ?1 "
      "AND s_arousal_avg >= ?2 "
      "AND created_at >= ?3 "
      "ORDER BY emotional_intensity DESC",
      { theta_intensity, theta_arousal, recent_window_ts });

  for (const auto &row : rows)
    {
      auto it_id = row.find ("embedding_id");
      auto it_intensity = row.find ("emotional_intensity");

      if (it_id == row.end () || it_intensity == row.end ())
        {
          continue;
        }

      if (it_id->second.type () != typeid (long long))
        {
          continue;
        }

      EmotionalSource s;
      s.embedding_id = std::any_cast<long long> (it_id->second);

      if (it_intensity->second.type () == typeid (double))
        {
          s.intensity = std::any_cast<double> (it_intensity->second);
        }
      else
        {
          continue;
        }

      // v2: arousal/valence not stored per-memory in v2, use defaults
      s.arousal = 0.5;
      s.valence = 0.5;

      auto it_bonus = row.find ("half_life_bonus");
      if (it_bonus != row.end () && it_bonus->second.type () == typeid (double))
        {
          s.half_life_bonus = std::any_cast<double> (it_bonus->second);
        }
      else
        {
          s.half_life_bonus = 1.0;
        }

      auto it_radius = row.find ("cascade_radius");
      if (it_radius != row.end () && it_radius->second.type () == typeid (long long))
        {
          s.cascade_radius
              = static_cast<int> (std::any_cast<long long> (it_radius->second));
        }
      else
        {
          s.cascade_radius = fallback.cascade_radius;
        }

      auto it_decay = row.find ("cascade_decay");
      if (it_decay != row.end () && it_decay->second.type () == typeid (double))
        {
          s.cascade_decay = std::any_cast<double> (it_decay->second);
        }
      else
        {
          s.cascade_decay = fallback.cascade_decay;
        }

      sources.push_back (s);
    }

  return sources;
}

/// @brief Structure for cascade neighbor.
struct CascadeNeighbor
{
  long long embedding_id;
  int depth;
};

/// @brief Find neighbor embedding IDs via graph traversal.
/// V2: Traverses ASSOCIATIONS table using memory_ids, returns embedding_ids.
std::vector<CascadeNeighbor>
FindCascadeNeighbors (Store *store, long long source_embedding_id, int max_depth)
{
  std::vector<CascadeNeighbor> neighbors;

  // V2: First find the memory_id for this embedding_id
  auto mem_rows = store->Execute (
      "SELECT memory_id FROM memories WHERE embedding_id = ?",
      { source_embedding_id });

  if (mem_rows.empty ())
    {
      return neighbors;
    }

  long long source_memory_id = 0;
  auto it_mem = mem_rows[0].find ("memory_id");
  if (it_mem != mem_rows[0].end () && it_mem->second.type () == typeid (long long))
    {
      source_memory_id = std::any_cast<long long> (it_mem->second);
    }
  else
    {
      return neighbors;
    }

  // V2: Traverse ASSOCIATIONS via recursive CTE using memory_ids
  auto rows = store->Execute (
      "WITH RECURSIVE cascade(memory_id, depth) AS ( "
      "  SELECT ?1, 0 "
      "  UNION "
      "  SELECT a.target_memory_id, cascade.depth + 1 "
      "  FROM associations a "
      "  JOIN cascade ON a.source_memory_id = cascade.memory_id "
      "  WHERE cascade.depth < ?2 "
      "  UNION "
      "  SELECT a.source_memory_id, cascade.depth + 1 "
      "  FROM associations a "
      "  JOIN cascade ON a.target_memory_id = cascade.memory_id "
      "  WHERE cascade.depth < ?2 "
      ") "
      "SELECT DISTINCT m.embedding_id, MIN(c.depth) AS min_depth "
      "FROM cascade c "
      "JOIN memories m ON c.memory_id = m.memory_id "
      "WHERE c.depth > 0 AND m.embedding_id IS NOT NULL "
      "GROUP BY m.embedding_id",
      { source_memory_id, static_cast<long long> (max_depth) });

  for (const auto &row : rows)
    {
      auto it_id = row.find ("embedding_id");
      auto it_depth = row.find ("min_depth");

      if (it_id == row.end () || it_depth == row.end ())
        {
          continue;
        }

      if (it_id->second.type () != typeid (long long))
        {
          continue;
        }

      CascadeNeighbor n;
      n.embedding_id = std::any_cast<long long> (it_id->second);

      if (it_depth->second.type () == typeid (long long))
        {
          n.depth = static_cast<int> (std::any_cast<long long> (it_depth->second));
        }
      else
        {
          n.depth = 1;
        }

      neighbors.push_back (n);
    }

  return neighbors;
}

} // namespace

void
PropagateEmotionalCascade::Execute (OperationContext &context, Transaction &tx) const
{
  Store *store = context.GetStore ();
  if (!store)
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  auto params
      = EmotionCascadeParams::FromKnobs (cfg.focus, cfg.sensitivity, cfg.stability);

  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);

  // Define the stability-derived lookback window for emotional sources.
  const int consolidation_interval
      = core::EmotionCascadeWindowSeconds (cfg.stability);
  const long long recent_window_ms
      = static_cast<long long> (consolidation_interval) * 1000LL;
  const long long recent_window_ts = std::max (0LL, now_ts - recent_window_ms);
  const double theta_intensity = core::ThetaIntensity (cfg.sensitivity);
  const double theta_arousal = core::ThetaArousal (cfg.sensitivity);

  // Load high-intensity emotional sources
  auto sources
      = LoadEmotionalSources (store, recent_window_ts, theta_intensity,
                              theta_arousal, params);
  const double intensity_floor = core::CascadeIntensityFloor (
      cfg.focus, cfg.sensitivity, cfg.stability);

  if (sources.empty ())
    {
      telemetry::LogDebug("cortext.emotion_cascade", {
        telemetry::Attribute::Int64("cascade_sources", 0),
        telemetry::Attribute::Int64("max_hops", 0)
      });
      return;
    }

  // Track which embeddings we've already processed to avoid duplicates
  std::unordered_set<long long> processed;
  int max_hops = 0;

  for (const auto &src : sources)
    {
      // Use the source's own cascade parameters if available,
      // otherwise fall back to knob-derived
      const int radius = (src.cascade_radius > 0) ? src.cascade_radius
                                                  : params.cascade_radius;
      const double decay = (src.cascade_decay > 0.0) ? src.cascade_decay
                                                     : params.cascade_decay;

      if (radius > max_hops)
        max_hops = radius;

      // Find neighbors via graph traversal
      auto neighbors = FindCascadeNeighbors (store, src.embedding_id, radius);

      for (const auto &neighbor : neighbors)
        {
          // Skip if already processed from another source
          if (processed.count (neighbor.embedding_id) > 0)
            {
              continue;
            }
          processed.insert (neighbor.embedding_id);

          // Compute decayed intensity based on hop distance
          const double decayed_intensity
              = src.intensity * std::pow (decay, neighbor.depth);

          // Compute decayed half_life_bonus
          const double decayed_bonus
              = src.half_life_bonus * std::pow (decay, neighbor.depth);

          // Skip if intensity has decayed too much
          if (decayed_intensity < intensity_floor)
            {
              continue;
            }

          // v2: Update emotional cascade fields in memories table
          // Preserve max intensity and half_life_bonus if already higher
          Add (tx,
               "UPDATE memories "
               "SET emotional_intensity = MAX(emotional_intensity, ?1), "
               "    half_life_bonus = MAX(half_life_bonus, ?2) "
               "WHERE embedding_id = ?3",
               { decayed_intensity, decayed_bonus, neighbor.embedding_id });
        }
    }

  telemetry::LogDebug("cortext.emotion_cascade", {
    telemetry::Attribute::Int64("cascade_sources", static_cast<long long>(sources.size())),
    telemetry::Attribute::Int64("max_hops", max_hops)
  });
}

} // namespace cortext::operations
