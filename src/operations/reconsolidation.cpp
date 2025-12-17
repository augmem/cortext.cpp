#include "cortext/operations/reconsolidation.hpp"

#include "cortext/buffered_write_instruction.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/schema.hpp"
#include "cortext/store/schema_helpers.hpp"
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
constexpr double kUncertaintyBumpCap = 0.2;
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

/// @brief Extracts embedding ID from a graph node ID string (e.g., "emb:123" →
/// 123).
inline std::optional<long long>
ParseEmbeddingId (const std::string &node_id)
{
  if (node_id.rfind ("emb:", 0) == 0)
    {
      try
        {
          return std::stoll (node_id.substr (4));
        }
      catch (...)
        {
          return std::nullopt;
        }
    }
  return std::nullopt;
}

/// @brief Neighbor info returned from graph traversal.
struct NeighborInfo
{
  long long embedding_id;
  int depth;
};

/// @brief Queries graph neighbors of a given embedding via iterative BFS.
/// Returns neighbors connected via 'co_occurs_with' or 'reinforces' edges
/// up to max_depth hops away.
inline std::vector<NeighborInfo>
QueryGraphNeighbors (Store *store, long long embedding_id, int max_depth)
{
  std::vector<NeighborInfo> neighbors;
  if (!store || max_depth < 1)
    {
      return neighbors;
    }

  // Use BFS to find neighbors up to max_depth hops
  std::unordered_set<std::string> visited;
  std::vector<std::pair<std::string, int>> frontier; // (node_id, depth)

  std::string seed_node = "emb:" + std::to_string (embedding_id);
  visited.insert (seed_node);
  frontier.push_back ({ seed_node, 0 });

  size_t frontier_idx = 0;
  while (frontier_idx < frontier.size ())
    {
      const auto &[current_node, current_depth] = frontier[frontier_idx];
      ++frontier_idx;

      if (current_depth >= max_depth)
        {
          continue;
        }

      // Query edges from/to this node
      std::string query
          = "SELECT source_id, target_id FROM graph_edges "
            "WHERE (source_id = ? OR target_id = ?) "
            "AND edge_type IN ('co_occurs_with', 'reinforces')";

      auto rows = store->Execute (query, { current_node, current_node });
      for (const auto &row : rows)
        {
          auto src_it = row.find ("source_id");
          auto tgt_it = row.find ("target_id");
          if (src_it == row.end () || tgt_it == row.end ())
            {
              continue;
            }

          std::string src, tgt;
          if (src_it->second.type () == typeid (std::string))
            {
              src = std::any_cast<std::string> (src_it->second);
            }
          if (tgt_it->second.type () == typeid (std::string))
            {
              tgt = std::any_cast<std::string> (tgt_it->second);
            }

          // The neighbor is the other end of the edge
          std::string neighbor_node = (src == current_node) ? tgt : src;

          if (visited.count (neighbor_node))
            {
              continue;
            }
          visited.insert (neighbor_node);

          int neighbor_depth = current_depth + 1;
          frontier.push_back ({ neighbor_node, neighbor_depth });

          // Parse embedding ID and add to results
          auto parsed_id = ParseEmbeddingId (neighbor_node);
          if (parsed_id)
            {
              neighbors.push_back ({ *parsed_id, neighbor_depth });
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

/// @brief Writes ripple updates for a neighbor embedding.
inline void
WriteNeighborUpdates (OperationContext &context, long long embedding_id,
                      double lability, long long timestamp,
                      const Eigen::VectorXf &blended)
{
  // Ensure memory_feedback row exists
  {
    BufferedWriteInstruction op;
    op.query = "INSERT INTO memory_feedback "
               "(embedding_id, retrieved_count, used_count, last_used) "
               "SELECT ?, 0, 0, 0 "
               "WHERE NOT EXISTS (SELECT 1 FROM "
               "memory_feedback WHERE embedding_id = ?)";
    op.params = { embedding_id, embedding_id };
    context.AddWriteInstruction (std::move (op));
  }

  // Update lability_state in embeddings (unified table)
  {
    BufferedWriteInstruction op;
    op.query = "UPDATE embeddings "
               "SET lability_state = ? "
               "WHERE embedding_id = ?";
    op.params = { lability, embedding_id };
    context.AddWriteInstruction (std::move (op));
  }

  // Update lability_ts in memory_feedback
  {
    BufferedWriteInstruction op;
    op.query = "UPDATE memory_feedback "
               "SET lability_ts = ? "
               "WHERE embedding_id = ?";
    op.params = { timestamp, embedding_id };
    context.AddWriteInstruction (std::move (op));
  }

  // Update embeddings with blended embedding
  // Note: sqlite-vec virtual tables don't support UPDATE on vector columns,
  // so we use DELETE + INSERT instead, preserving all metadata fields
  {
    BufferedWriteInstruction del_op;
    del_op.query = "DELETE FROM embeddings WHERE embedding_id = ?";
    del_op.params = { embedding_id };
    context.AddWriteInstruction (std::move (del_op));

    BufferedWriteInstruction ins_op;
    ins_op.query = std::string ("INSERT INTO embeddings (")
                   + store::kEmbeddingsReconsolidateColumns + ") VALUES ("
                   + store::kEmbeddingsReconsolidateDefaults + ")";
    ins_op.params = { embedding_id, ToFloatVector (blended), lability };
    context.AddWriteInstruction (std::move (ins_op));
  }
}

} // namespace

void
ApplyReconsolidation::Execute (OperationContext &context) const
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

      // Without DB reads available here, assume current lability based on
      // susceptibility (elapsed≈0 at retrieval time).
      const double current_lability = lability_susc;

      double drift_mag
          = (1.0 - T) * S * current_lability * contextual_relevance;
      drift_mag *= recon_gain;
      // Safety clamp
      drift_mag = std::min (kDriftClampMax,
                            std::max (constants::kNormalizedMin, drift_mag));
      max_drift = std::max (max_drift, drift_mag);

      // Note: embeddings row already exists - no need to ensure it exists

      // Ensure memory_feedback row exists for retrieval tracking.
      {
        BufferedWriteInstruction op;
        op.query = "INSERT INTO memory_feedback "
                   "(embedding_id, retrieved_count, used_count, last_used) "
                   "SELECT ?, 0, 0, 0 "
                   "WHERE NOT EXISTS (SELECT 1 FROM "
                   "memory_feedback WHERE embedding_id = ?)";
        op.params = { embedding_id, embedding_id };
        context.AddWriteInstruction (std::move (op));
      }

      // Update lability_state in embeddings (unified table).
      {
        BufferedWriteInstruction op;
        op.query = "UPDATE embeddings "
                   "SET lability_state = ? "
                   "WHERE embedding_id = ?";
        op.params = { current_lability, embedding_id };
        context.AddWriteInstruction (std::move (op));
      }

      // Update original_embedding and lability_ts in memory_feedback.
      {
        BufferedWriteInstruction op;
        op.query = "UPDATE memory_feedback "
                   "SET original_embedding = COALESCE(original_embedding, ?), "
                   "    lability_ts = ? "
                   "WHERE embedding_id = ?";
        op.params = { ToFloatVector (u_m), now_ts, embedding_id };
        context.AddWriteInstruction (std::move (op));
      }

      if (drift_mag < kDriftSkipEpsilon)
        {
          continue; // No embedding update when drift too small
        }

      // Blend toward current context and normalize.
      Eigen::VectorXf blended = static_cast<float> (1.0 - drift_mag) * u_m
                                + static_cast<float> (drift_mag) * u_cur;
      blended = Unit (blended);

      // Update embeddings (unified table for vector storage and metadata).
      // Note: sqlite-vec virtual tables don't support UPDATE on vector columns,
      // so we use DELETE + INSERT instead, preserving all metadata fields
      {
        BufferedWriteInstruction del_op;
        del_op.query = "DELETE FROM embeddings WHERE embedding_id = ?";
        del_op.params = { embedding_id };
        context.AddWriteInstruction (std::move (del_op));

        BufferedWriteInstruction ins_op;
        ins_op.query = std::string ("INSERT INTO embeddings (")
                       + store::kEmbeddingsReconsolidateColumns + ") VALUES ("
                       + store::kEmbeddingsReconsolidateDefaults + ")";
        ins_op.params = { embedding_id, ToFloatVector (blended), current_lability };
        context.AddWriteInstruction (std::move (ins_op));
      }

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
          WriteNeighborUpdates (context, neighbor.embedding_id, neighbor_lability,
                                now_ts, neighbor_blended);
        }
      // --- END RIPPLE PROPAGATION ---
    }

  // Increase uncertainty proportional to max drift (cap 0.2).
  const double bump = std::min (kUncertaintyBumpCap, max_drift);
  p_ctx.u_t = core::Clamp (p_ctx.u_t + bump, constants::kNormalizedMin,
                           constants::kNormalizedMax);
}

void
ApplyReconsolidation::CollectSchema (cortext::store::SchemaRegistry &registry) const
{
  // Reconsolidation relies on memory_feedback and embeddings, which are core tables.
  // However, if we want to be explicit about table requirements or indexes for lability,
  // we could add them here. For now, core tables are handled by RegisterCoreSchema.
  (void)registry;
}

} // namespace cortext::operations
