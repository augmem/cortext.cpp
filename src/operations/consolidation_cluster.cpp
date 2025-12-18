#include "cortext/operations/consolidation_cluster.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <vector>

namespace cortext::operations
{

ConsolidationClusterParams
ConsolidationClusterParams::FromKnobs (double F, double /*S*/, double /*T*/)
{
  ConsolidationClusterParams p;
  p.merge_threshold = core::MergeThreshold (F);
  p.min_cluster_size = core::MinClusterSize (F);
  p.max_clusters = 100; // Reasonable cap
  return p;
}

void
ConsolidationCluster::Execute (OperationContext &context, Transaction &tx) const
{
  if (!context.GetConsolidationShouldStart ())
    {
      return;
    }

  Store *store = context.GetStore ();
  if (!store)
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  auto params = ConsolidationClusterParams::FromKnobs (cfg.focus, cfg.sensitivity,
                                                       cfg.stability);

  // v2: Load candidates with embeddings from consolidation_candidates joined
  // with embeddings (via memories for consistency).
  auto rows = store->Execute (
      "SELECT cc.embedding_id, cc.score, e.embedding "
      "FROM consolidation_candidates cc "
      "JOIN memories m ON cc.embedding_id = m.embedding_id "
      "JOIN embeddings e ON m.embedding_id = e.embedding_id "
      "ORDER BY cc.score ASC",
      {});

  if (rows.empty ())
    {
      return;
    }

  // 2. Parse embeddings into memory.
  struct Candidate
  {
    long long embedding_id;
    double score;
    Eigen::VectorXf embedding;
    bool assigned = false;
  };
  std::vector<Candidate> items;
  items.reserve (rows.size ());

  // Determine embedding dimension from first valid embedding.
  int emb_dim = 256; // Default assumption

  for (const auto &row : rows)
    {
      auto it_id = row.find ("embedding_id");
      auto it_score = row.find ("score");
      auto it_emb = row.find ("embedding");

      if (it_id == row.end () || it_score == row.end () || it_emb == row.end ())
        {
          continue;
        }

      if (it_id->second.type () != typeid (long long))
        {
          continue;
        }

      Candidate c;
      c.embedding_id = std::any_cast<long long> (it_id->second);

      // Score can be double or int depending on DB.
      if (it_score->second.type () == typeid (double))
        {
          c.score = std::any_cast<double> (it_score->second);
        }
      else if (it_score->second.type () == typeid (long long))
        {
          c.score = static_cast<double> (
              std::any_cast<long long> (it_score->second));
        }
      else
        {
          continue;
        }

      // Decode embedding blob.
      if (!core::DecodeFloatBlob (it_emb->second, emb_dim, c.embedding))
        {
          continue;
        }
      emb_dim = static_cast<int> (c.embedding.size ());

      items.push_back (std::move (c));
    }

  if (items.empty ())
    {
      return;
    }

  // 3. Greedy clustering (single-linkage style).
  int next_cluster_id = 0;
  std::vector<ClusterInfo> clusters;

  for (size_t i = 0; i < items.size (); ++i)
    {
      if (items[i].assigned)
        {
          continue;
        }

      // Start new cluster with this item.
      ClusterInfo cluster;
      cluster.cluster_id = next_cluster_id++;
      cluster.embedding_ids.push_back (items[i].embedding_id);
      cluster.avg_score = items[i].score;

      // Initialize centroid.
      Eigen::VectorXf centroid = items[i].embedding;
      items[i].assigned = true;
      int cluster_count = 1;

      // Find all candidates similar to centroid.
      for (size_t j = i + 1; j < items.size (); ++j)
        {
          if (items[j].assigned)
            {
              continue;
            }

          double sim = core::CosineSimilarity (centroid, items[j].embedding);
          if (sim > params.merge_threshold)
            {
              // Add to cluster.
              cluster.embedding_ids.push_back (items[j].embedding_id);

              // Update running average score.
              double old_avg = cluster.avg_score;
              cluster.avg_score
                  = (old_avg * cluster_count + items[j].score)
                    / (cluster_count + 1);

              // Update centroid as running mean.
              centroid = (centroid * static_cast<float> (cluster_count)
                          + items[j].embedding)
                         / static_cast<float> (cluster_count + 1);

              items[j].assigned = true;
              ++cluster_count;
            }
        }

      // Store centroid as std::vector<float>.
      cluster.centroid.resize (static_cast<size_t> (centroid.size ()));
      for (int k = 0; k < centroid.size (); ++k)
        {
          cluster.centroid[static_cast<size_t> (k)] = centroid (k);
        }

      // Only keep clusters meeting minimum size.
      if (static_cast<int> (cluster.embedding_ids.size ())
          >= params.min_cluster_size)
        {
          clusters.push_back (std::move (cluster));
        }

      if (static_cast<int> (clusters.size ()) >= params.max_clusters)
        {
          break;
        }
    }

  // 4. Pass clusters to next operation via context.
  const int cluster_count = static_cast<int>(clusters.size());
  context.SetConsolidationClusters (std::move (clusters));

  telemetry::LogDebug("cortext.consolidation_cluster", {
    telemetry::Attribute::Int64("cluster_count", cluster_count)
  });
}

} // namespace cortext::operations
