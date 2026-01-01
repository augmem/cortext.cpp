#include "cortext/operations/reconsolidation.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace cortext::operations
{

namespace
{
constexpr double kDriftClampMax = 0.3;
constexpr double kDriftSkipEpsilon = 0.001;
constexpr double kRippleStrengthMin = 0.01;     // Min lability to propagate ripple
constexpr double kRippleDriftCapFactor = 0.5;  // Ripple drift capped at 50% of primary

/// @brief Normalizes a vector to unit length.
inline Eigen::VectorXf
Unit (const Eigen::VectorXf &v)
{
  const double n = v.norm ();
  if (n <= constants::kNormEpsilon)
    {
      return v;
    }
  return v / static_cast<float> (n);
}

inline std::vector<float>
ToFloatVector (const Eigen::VectorXf &v)
{
  std::vector<float> out;
  out.resize (static_cast<size_t> (v.size ()));
  for (int i = 0; i < v.size (); ++i)
    {
      out[static_cast<size_t> (i)] = v[i];
    }
  return out;
}

/// @brief Neighbor info returned from graph traversal.
struct NeighborInfo
{
  long long embedding_id;
  int depth;
};

/// @brief Queries graph neighbors of a given embedding via iterative BFS.
/// V2: Uses ASSOCIATIONS table with memory_ids instead of graph_edges.
/// Returns neighbors connected via 'co_occurs' or 'reinforces' edges
/// up to max_depth hops away.
inline std::vector<NeighborInfo>
QueryGraphNeighbors (Store *store, long long embedding_id, int max_depth)
{
  std::vector<NeighborInfo> neighbors;
  if (!store || max_depth < 1)
    {
      return neighbors;
    }

  // V2: First find the memory_id for this embedding_id
  auto mem_rows = store->Execute (
      "SELECT memory_id FROM memories WHERE embedding_id = ?",
      { embedding_id });

  if (mem_rows.empty ())
    {
      return neighbors;
    }

  long long seed_memory_id = 0;
  auto it_mem = mem_rows[0].find ("memory_id");
  if (it_mem != mem_rows[0].end () && it_mem->second.type () == typeid (long long))
    {
      seed_memory_id = std::any_cast<long long> (it_mem->second);
    }
  else
    {
      return neighbors;
    }

  // V2: Use BFS to traverse ASSOCIATIONS
  std::unordered_set<long long> visited;
  std::vector<std::pair<long long, int>> frontier; // (memory_id, depth)

  visited.insert (seed_memory_id);
  frontier.push_back ({ seed_memory_id, 0 });

  size_t frontier_idx = 0;
  while (frontier_idx < frontier.size ())
    {
      const auto &[current_mem_id, current_depth] = frontier[frontier_idx];
      ++frontier_idx;

      if (current_depth >= max_depth)
        {
          continue;
        }

      // V2: Query edges from/to this memory via ASSOCIATIONS
      auto rows = store->Execute (
          "SELECT source_memory_id, target_memory_id FROM associations "
          "WHERE (source_memory_id = ? OR target_memory_id = ?) "
          "AND edge_type IN ('co_occurs', 'reinforces')",
          { current_mem_id, current_mem_id });

      for (const auto &row : rows)
        {
          auto src_it = row.find ("source_memory_id");
          auto tgt_it = row.find ("target_memory_id");
          if (src_it == row.end () || tgt_it == row.end ())
            {
              continue;
            }

          long long src_id = 0, tgt_id = 0;
          if (src_it->second.type () == typeid (long long))
            {
              src_id = std::any_cast<long long> (src_it->second);
            }
          if (tgt_it->second.type () == typeid (long long))
            {
              tgt_id = std::any_cast<long long> (tgt_it->second);
            }

          // The neighbor is the other end of the edge
          long long neighbor_mem_id = (src_id == current_mem_id) ? tgt_id : src_id;

          if (visited.count (neighbor_mem_id))
            {
              continue;
            }
          visited.insert (neighbor_mem_id);

          int neighbor_depth = current_depth + 1;
          frontier.push_back ({ neighbor_mem_id, neighbor_depth });

          // Look up embedding_id for this neighbor memory
          auto emb_rows = store->Execute (
              "SELECT embedding_id FROM memories WHERE memory_id = ?",
              { neighbor_mem_id });

          if (!emb_rows.empty ())
            {
              auto emb_it = emb_rows[0].find ("embedding_id");
              if (emb_it != emb_rows[0].end ()
                  && emb_it->second.type () == typeid (long long))
                {
                  neighbors.push_back (
                      { std::any_cast<long long> (emb_it->second), neighbor_depth });
                }
            }
        }
    }

  return neighbors;
}

/// @brief Loads an embedding from embeddings by ID.
inline std::optional<Eigen::VectorXf>
LoadEmbedding (Store *store, long long embedding_id, int expected_dim = 256)
{
  if (!store)
    {
      return std::nullopt;
    }

  auto rows = store->Execute (
      "SELECT embedding FROM embeddings WHERE embedding_id = ?",
      { embedding_id });

  if (rows.empty ())
    {
      return std::nullopt;
    }

  auto it = rows[0].find ("embedding");
  if (it == rows[0].end ())
    {
      return std::nullopt;
    }

  // Use DecodeFloatBlob to handle blob data from sqlite-vec
  Eigen::VectorXf result;
  if (core::DecodeFloatBlob (it->second, expected_dim, result))
    {
      return result;
    }

  return std::nullopt;
}

/// @brief Writes ripple updates for a neighbor embedding (v2 schema).
inline void
WriteNeighborUpdates (Transaction &tx, long long embedding_id,
                      double lability, long long timestamp,
                      const Eigen::VectorXf &blended)
{
  // Update lability_state and lability_ts in memories table (v2: merged from
  // memory_feedback)
  tx.Execute ("UPDATE memories "
              "SET lability_state = COALESCE(?, 0.0), lability_ts = ? "
              "WHERE embedding_id = ?",
              { lability, timestamp, embedding_id });

  // Update embeddings with blended embedding
  // Note: sqlite-vec virtual tables don't support UPDATE on vector columns,
  // so we use DELETE + INSERT instead
  tx.Execute ("DELETE FROM embeddings WHERE embedding_id = ?",
              { embedding_id });

  // v2: Minimal embeddings table (embedding_id, embedding, created_at)
  tx.Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
              "VALUES (?, ?, ?)",
              { embedding_id, ToFloatVector (blended), timestamp });
}

} // namespace

void
ApplyReconsolidation::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  Store *store = context.GetStore ();

  // Require a current context embedding to drift toward.
  if (p_ctx.recent_context_embeddings.empty ())
    {
      return;
    }
  const Eigen::VectorXf &x_cur = p_ctx.recent_context_embeddings.back ();
  const Eigen::VectorXf u_cur = Unit (x_cur);

  const auto &retrieved = context.GetRetrievedMemoryEmbeddings ();
  if (retrieved.empty ())
    {
      return;
    }

  const double S = cfg.sensitivity;
  const double T = cfg.stability;
  const double recon_gain = core::ReconsolidationGain (T);
  const double ripple_decay = core::RippleDecay (T);
  const int ripple_depth = core::RippleDepth (T);
  const double tau_labile = core::TauLabile (T);
  const double lability_susc = core::LabilitySusceptibility (S, T);

  double max_drift = 0.0;
  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);

  // Track which embeddings we've already processed (avoid duplicate ripple)
  std::unordered_set<long long> processed_ids;

  for (const auto &kv : retrieved)
    {
      const long long embedding_id = kv.first;
      const Eigen::VectorXf &m = kv.second;
      if (m.size () == 0 || m.size () != u_cur.size ())
        {
          continue;
        }

      // Mark as processed to avoid duplicate ripple
      processed_ids.insert (embedding_id);

      const Eigen::VectorXf u_m = Unit (m);
      // contextual_relevance ∈ [0,1]
      double contextual_relevance = core::CosineSimilarity (u_m, u_cur);
      if (contextual_relevance < constants::kNormalizedMin)
        contextual_relevance = constants::kNormalizedMin;
      if (contextual_relevance > constants::kNormalizedMax)
        contextual_relevance = constants::kNormalizedMax;

      double current_lability = lability_susc;
      double stored_lability = 0.0;
      long long lability_ts = 0;
      auto lability_rows = tx.Execute (
          "SELECT lability_state, lability_ts FROM memories WHERE embedding_id = ?",
          { embedding_id });
      if (!lability_rows.empty ())
        {
          const auto &lab_row = lability_rows[0];
          auto state_it = lab_row.find ("lability_state");
          if (state_it != lab_row.end ())
            {
              if (state_it->second.type () == typeid (double))
                {
                  stored_lability = std::any_cast<double> (state_it->second);
                }
              else if (state_it->second.type () == typeid (float))
                {
                  stored_lability
                      = static_cast<double> (std::any_cast<float> (state_it->second));
                }
            }
          auto ts_it = lab_row.find ("lability_ts");
          if (ts_it != lab_row.end ())
            {
              if (ts_it->second.type () == typeid (long long))
                {
                  lability_ts = std::any_cast<long long> (ts_it->second);
                }
              else if (ts_it->second.type () == typeid (int))
                {
                  lability_ts = std::any_cast<int> (ts_it->second);
                }
            }
        }

      if (lability_ts > 0 && now_ts > lability_ts)
        {
          const double dt_s
              = static_cast<double> (now_ts - lability_ts) / 1000.0;
          const double base
              = (stored_lability > 0.0) ? stored_lability : lability_susc;
          current_lability
              = base
                * std::exp (-dt_s
                            / std::max (tau_labile,
                                        constants::kNormEpsilon));
        }
      else if (stored_lability > 0.0)
        {
          current_lability = stored_lability;
        }

      double drift_mag
          = (1.0 - T) * S * current_lability * contextual_relevance;
      drift_mag *= recon_gain;
      // Safety clamp
      drift_mag = std::min (kDriftClampMax,
                            std::max (constants::kNormalizedMin, drift_mag));
      max_drift = std::max (max_drift, drift_mag);

      // Note: embeddings row already exists - no need to ensure it exists

      // v2: Update lability_state, original_centroid, and lability_ts in
      // memories table (merged from memory_feedback)
      tx.Execute ("UPDATE memories "
                  "SET lability_state = COALESCE(?, 0.0), "
                  "    original_centroid = COALESCE(original_centroid, ?), "
                  "    lability_ts = ? "
                  "WHERE embedding_id = ?",
                  { current_lability, ToFloatVector (u_m), now_ts,
                    embedding_id });

      if (drift_mag < kDriftSkipEpsilon)
        {
          continue; // No embedding update when drift too small
        }

      // Blend toward current context and normalize.
      Eigen::VectorXf blended = static_cast<float> (1.0 - drift_mag) * u_m
                                + static_cast<float> (drift_mag) * u_cur;
      blended = Unit (blended);

      // Update embeddings with blended embedding
      // Note: sqlite-vec virtual tables don't support UPDATE on vector columns,
      // so we use DELETE + INSERT instead
      tx.Execute ("DELETE FROM embeddings WHERE embedding_id = ?",
                  { embedding_id });

      // v2: Minimal embeddings table (embedding_id, embedding, created_at)
      tx.Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { embedding_id, ToFloatVector (blended), now_ts });

      // --- BEGIN RIPPLE PROPAGATION ---
      // Only propagate ripple if store is available and primary drift occurred
      if (!store)
        {
          continue;
        }

      // Query graph neighbors of this reconsolidated memory
      auto neighbors = QueryGraphNeighbors (store, embedding_id, ripple_depth);

      for (const auto &neighbor : neighbors)
        {
          // Skip if already processed (either as primary or via earlier ripple)
          if (processed_ids.count (neighbor.embedding_id))
            {
              continue;
            }
          processed_ids.insert (neighbor.embedding_id);

          // Compute decayed lability for this neighbor
          // neighbor_lability = source_lability * pow(ripple_decay, depth)
          double neighbor_lability
              = current_lability
                * std::pow (ripple_decay, static_cast<double> (neighbor.depth));

          // Skip if lability too low
          if (neighbor_lability < kRippleStrengthMin)
            {
              continue;
            }

          // Load neighbor embedding
          auto neighbor_emb = LoadEmbedding (store, neighbor.embedding_id);
          if (!neighbor_emb)
            {
              continue;
            }

          // Compute neighbor drift (proportional to decayed lability)
          double neighbor_drift = neighbor_lability * recon_gain;
          // Cap ripple drift at reduced factor compared to primary
          neighbor_drift = std::min (kDriftClampMax * kRippleDriftCapFactor,
                                     std::max (kDriftSkipEpsilon, neighbor_drift));

          if (neighbor_drift < kDriftSkipEpsilon)
            {
              continue;
            }

          // Blend neighbor toward current context
          Eigen::VectorXf u_neighbor = Unit (*neighbor_emb);
          Eigen::VectorXf neighbor_blended
              = static_cast<float> (1.0 - neighbor_drift) * u_neighbor
                + static_cast<float> (neighbor_drift) * u_cur;
          neighbor_blended = Unit (neighbor_blended);

          // Write updates for this neighbor
          WriteNeighborUpdates (tx, neighbor.embedding_id, neighbor_lability,
                                now_ts, neighbor_blended);
        }
      // --- END RIPPLE PROPAGATION ---
    }

  (void)max_drift;
}

} // namespace cortext::operations
