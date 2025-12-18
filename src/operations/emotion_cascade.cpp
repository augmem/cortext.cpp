#include "cortext/operations/emotion_cascade.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
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
std::vector<EmotionalSource>
LoadEmotionalSources (Store *store, long long recent_window_ts)
{
  std::vector<EmotionalSource> sources;

  // Query emotional tags for high-intensity memories
  // Only process recently tagged ones (within consolidation window)
  auto rows = store->Execute (
      "SELECT embedding_id, intensity, arousal, valence, half_life_bonus, "
      "       cascade_radius, cascade_decay "
      "FROM emotional_tags "
      "WHERE intensity >= 0.5 AND ts >= ?1 "
      "ORDER BY intensity DESC",
      { recent_window_ts });

  for (const auto &row : rows)
    {
      auto it_id = row.find ("embedding_id");
      auto it_intensity = row.find ("intensity");

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

      // Parse optional fields with defaults
      auto it_arousal = row.find ("arousal");
      if (it_arousal != row.end () && it_arousal->second.type () == typeid (double))
        {
          s.arousal = std::any_cast<double> (it_arousal->second);
        }
      else
        {
          s.arousal = 0.5;
        }

      auto it_valence = row.find ("valence");
      if (it_valence != row.end () && it_valence->second.type () == typeid (double))
        {
          s.valence = std::any_cast<double> (it_valence->second);
        }
      else
        {
          s.valence = 0.5;
        }

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
          s.cascade_radius = 1;
        }

      auto it_decay = row.find ("cascade_decay");
      if (it_decay != row.end () && it_decay->second.type () == typeid (double))
        {
          s.cascade_decay = std::any_cast<double> (it_decay->second);
        }
      else
        {
          s.cascade_decay = 0.5;
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
std::vector<CascadeNeighbor>
FindCascadeNeighbors (Store *store, long long source_embedding_id, int max_depth)
{
  std::vector<CascadeNeighbor> neighbors;

  // Use recursive CTE to traverse graph from source node
  std::string source_node = "emb:" + std::to_string (source_embedding_id);

  auto rows = store->Execute (
      "WITH RECURSIVE cascade(id, depth) AS ( "
      "  SELECT ?1, 0 "
      "  UNION "
      "  SELECT ge.target_id, cascade.depth + 1 "
      "  FROM graph_edges ge "
      "  JOIN cascade ON ge.source_id = cascade.id "
      "  WHERE cascade.depth < ?2 "
      "  UNION "
      "  SELECT ge.source_id, cascade.depth + 1 "
      "  FROM graph_edges ge "
      "  JOIN cascade ON ge.target_id = cascade.id "
      "  WHERE cascade.depth < ?2 "
      ") "
      "SELECT DISTINCT id, MIN(depth) AS min_depth "
      "FROM cascade "
      "WHERE depth > 0 AND id LIKE 'emb:%' "
      "GROUP BY id",
      { source_node, static_cast<long long> (max_depth) });

  for (const auto &row : rows)
    {
      auto it_id = row.find ("id");
      auto it_depth = row.find ("min_depth");

      if (it_id == row.end () || it_depth == row.end ())
        {
          continue;
        }

      if (it_id->second.type () != typeid (std::string))
        {
          continue;
        }

      std::string node_id = std::any_cast<std::string> (it_id->second);

      // Extract embedding_id from "emb:123" format
      if (node_id.size () <= 4 || node_id.substr (0, 4) != "emb:")
        {
          continue;
        }

      CascadeNeighbor n;
      try
        {
          n.embedding_id = std::stoll (node_id.substr (4));
        }
      catch (...)
        {
          continue;
        }

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

  // Define recent window for emotional sources (last consolidation interval)
  const int consolidation_interval
      = core::ConsolidationIntervalSeconds (cfg.stability);
  const long long recent_window_ts = now_ts - consolidation_interval;

  // Load high-intensity emotional sources
  auto sources = LoadEmotionalSources (store, recent_window_ts);

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
          if (decayed_intensity < 0.1)
            {
              continue;
            }

          // Update or insert emotional tag for neighbor
          // Use INSERT OR REPLACE to update if exists, preserving max intensity
          Add (tx,
               "INSERT INTO emotional_tags "
               "(embedding_id, flashbulb, intensity, arousal, valence, "
               " half_life_bonus, detail_suppression, gist_components, "
               " cascade_radius, cascade_decay, flashbulb_threshold_eff, ts) "
               "VALUES (?1, 0, ?2, ?3, ?4, ?5, 0.0, 0, 0, 0.0, 0.0, ?6) "
               "ON CONFLICT (embedding_id) DO UPDATE SET "
               "intensity = MAX(emotional_tags.intensity, excluded.intensity), "
               "half_life_bonus = MAX(emotional_tags.half_life_bonus, excluded.half_life_bonus), "
               "ts = excluded.ts",
               { neighbor.embedding_id, decayed_intensity, src.arousal, src.valence,
                 decayed_bonus, now_ts });
        }
    }

  telemetry::LogDebug("cortext.emotion_cascade", {
    telemetry::Attribute::Int64("cascade_sources", static_cast<long long>(sources.size())),
    telemetry::Attribute::Int64("max_hops", max_hops)
  });
}

} // namespace cortext::operations
