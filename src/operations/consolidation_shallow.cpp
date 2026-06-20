#include "cortext/internal/cancellation.hpp"
#include "cortext/operations/consolidation_shallow.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace cortext::operations
{

namespace
{

struct LabelNode
{
  long long memory_id = 0;
  Eigen::VectorXf embedding;
};

std::string
GenerateShallowId (uint64_t ts, int counter)
{
  std::ostringstream ss;
  ss << "association_" << ts << "_" << counter;
  return ss.str ();
}

void
AddWrite (Transaction &tx, const std::string &q,
          const std::vector<std::any> &p = {})
{
  tx.Execute (q, p);
}

std::vector<LabelNode>
LoadLabelNodes (Transaction &tx, int expected_dim)
{
  std::vector<LabelNode> labels;
  auto rows = tx.Execute (
      "SELECT m.memory_id, e.embedding "
      "FROM memories m "
      "JOIN embeddings e ON m.embedding_id = e.embedding_id "
      "WHERE m.kind = 'LABEL'",
      {});

  for (const auto &row : rows)
    {
      auto it_id = row.find ("memory_id");
      auto it_emb = row.find ("embedding");
      if (it_id == row.end () || it_emb == row.end ())
        {
          continue;
        }
      if (it_id->second.type () != typeid (long long))
        {
          continue;
        }

      LabelNode node;
      node.memory_id = std::any_cast<long long> (it_id->second);
      if (!core::DecodeFloatBlob (it_emb->second, expected_dim, node.embedding))
        {
          continue;
        }
      labels.push_back (std::move (node));
    }
  return labels;
}

} // namespace

void
ConsolidationShallow::Execute (OperationContext &context, Transaction &tx) const
{
  if (!context.GetConsolidationShouldStart ())
    {
      return;
    }

  const auto &clusters = context.GetConsolidationClusters ();
  if (clusters.empty ())
    {
      return;
    }

  if (!context.GetStore ())
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  const int max_labels = core::ShallowConsolidationMaxLabels (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const double label_threshold = core::ShallowConsolidationLabelMinSimilarity (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const double derived_source_edge_weight = core::DerivedSourceEdgeWeight (
      cfg.focus, cfg.sensitivity, cfg.stability);

  const int expected_dim = std::max (
      1, static_cast<int> (clusters[0].centroid.size ()));
  auto label_nodes = LoadLabelNodes (tx, expected_dim);

  const uint64_t now_ts = context.GetSignal ().timestamp;
  int association_counter = 0;
  long long labels_attached = 0;

  for (const auto &cluster : clusters)
    {
      internal::ThrowIfStopRequested ();
      if (cluster.centroid.empty ())
        {
          continue;
        }
      std::string association_id = GenerateShallowId (now_ts,
                                                      association_counter++);

      // Insert centroid embedding.
      std::vector<float> centroid_blob = cluster.centroid;
      AddWrite (tx,
                "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
                { centroid_blob, static_cast<long long> (now_ts) });
      auto id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
      long long centroid_embedding_id = 0;
      if (!id_rows.empty () && id_rows[0].count ("id"))
        {
          auto val = id_rows[0].at ("id");
          if (val.type () == typeid (long long))
            {
              centroid_embedding_id = std::any_cast<long long> (val);
            }
          else if (val.type () == typeid (int))
            {
              centroid_embedding_id = std::any_cast<int> (val);
            }
        }

      // Create ASSOCIATION memory without a blob (shallow centroid).
      AddWrite (tx,
                "INSERT INTO memories "
                "(embedding_id, source_id, kind, label, start_ts, n_signals, created_at) "
                "VALUES (?, ?, 'ASSOCIATION', ?, ?, ?, ?)",
                { centroid_embedding_id, association_id, std::string (),
                  static_cast<long long> (now_ts),
                  static_cast<long long> (cluster.embedding_ids.size ()),
                  static_cast<long long> (now_ts) });

      auto mem_id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
      long long centroid_memory_id = 0;
      if (!mem_id_rows.empty () && mem_id_rows[0].count ("id"))
        {
          auto val = mem_id_rows[0].at ("id");
          if (val.type () == typeid (long long))
            {
              centroid_memory_id = std::any_cast<long long> (val);
            }
          else if (val.type () == typeid (int))
            {
              centroid_memory_id = std::any_cast<int> (val);
            }
        }

      Eigen::VectorXf centroid_vec;
      bool has_centroid_vec = false;
      if (!cluster.centroid.empty ())
        {
          centroid_vec = Eigen::VectorXf (
              static_cast<Eigen::Index> (cluster.centroid.size ()));
          for (size_t i = 0; i < cluster.centroid.size (); ++i)
            {
              centroid_vec (static_cast<Eigen::Index> (i))
                  = cluster.centroid[i];
            }
          has_centroid_vec = true;
        }
      if (has_centroid_vec && centroid_memory_id > 0
          && centroid_embedding_id > 0)
        {
          context.GetProcessorContext ().UpsertAssociationCache (
              centroid_memory_id, centroid_embedding_id, centroid_vec, true);
        }

      // Update cluster_id and derived_from edges.
      for (long long emb_id : cluster.embedding_ids)
        {
          internal::ThrowIfStopRequested ();
          AddWrite (tx,
                    "UPDATE memories SET cluster_id = ? "
                    "WHERE embedding_id = ?",
                    { cluster.cluster_id, emb_id });

          auto src_rows = tx.Execute (
              "SELECT memory_id FROM memories WHERE embedding_id = ?",
              { emb_id });
          if (!src_rows.empty () && src_rows[0].count ("memory_id"))
            {
              auto val = src_rows[0].at ("memory_id");
              long long src_memory_id = 0;
              if (val.type () == typeid (long long))
                {
                  src_memory_id = std::any_cast<long long> (val);
                }
              else if (val.type () == typeid (int))
                {
                  src_memory_id = std::any_cast<int> (val);
                }
              if (src_memory_id > 0 && centroid_memory_id > 0)
                {
                  AddWrite (tx,
                            "INSERT OR IGNORE INTO associations "
                            "(source_memory_id, target_memory_id, edge_type, weight) "
                            "VALUES (?, ?, 'derived_from', ?)",
                            { centroid_memory_id, src_memory_id,
                              derived_source_edge_weight });
                }
            }
        }

      // Attach labels by embedding similarity (if label nodes exist).
      if (!label_nodes.empty () && centroid_memory_id > 0 && has_centroid_vec)
        {
          std::vector<std::pair<double, long long>> candidates;
          candidates.reserve (label_nodes.size ());
          for (const auto &label : label_nodes)
            {
              if (label.embedding.size () != centroid_vec.size ())
                {
                  continue;
                }
              const double cos = core::CosineSimilarity (centroid_vec,
                                                         label.embedding);
              if (cos >= label_threshold)
                {
                  candidates.emplace_back (cos, label.memory_id);
                }
            }

          if (!candidates.empty ())
            {
              const int k = std::min (max_labels,
                                      static_cast<int> (candidates.size ()));
              std::partial_sort (candidates.begin (),
                                 candidates.begin () + k,
                                 candidates.end (),
                                 [] (const auto &a, const auto &b) {
                                   return a.first > b.first;
                                 });
              for (int i = 0; i < k; ++i)
                {
                  const double weight01 = core::Map01 (candidates[i].first);
                  AddWrite (tx,
                            "INSERT OR REPLACE INTO associations "
                            "(source_memory_id, target_memory_id, edge_type, weight) "
                            "VALUES (?, ?, 'has_label', ?)",
                            { centroid_memory_id, candidates[i].second,
                              weight01 });
                  labels_attached += 1;
                }
            }
        }
    }

  telemetry::LogInfo ("cortext.consolidation_shallow", {
    telemetry::Attribute::Int64 ("cluster_count",
                                 static_cast<long long> (clusters.size ())),
    telemetry::Attribute::Int64 ("labels_attached", labels_attached),
    telemetry::Attribute::Int64 ("max_labels", max_labels)
  });
}

} // namespace cortext::operations
