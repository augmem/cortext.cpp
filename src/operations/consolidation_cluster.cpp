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
  (void)tx;
  (void)tx;
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

  // v2: Load candidates from context (in-memory passing)
  const auto &input_candidates = context.GetConsolidationCandidates ();
  if (input_candidates.empty ())
    {
      return;
    }

  // 2. Prepare candidates for clustering.
  struct Candidate
  {
    long long embedding_id;
    double score;
    Eigen::VectorXf embedding;
    bool assigned = false;
  };
  std::vector<Candidate> items;
  items.reserve (input_candidates.size ());

  for (const auto &ic : input_candidates)
    {
      Candidate c;
      c.embedding_id = ic.embedding_id;
      c.score = ic.score;
      c.embedding = ic.embedding;
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
