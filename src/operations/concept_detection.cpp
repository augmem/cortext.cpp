#include "cortext/operations/concept_detection.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <Eigen/Dense>
#include <any>
#include <cmath>
#include <string>
#include <vector>

namespace cortext::operations
{

ConceptDetectionParams
ConceptDetectionParams::FromKnobs (double /*F*/, double /*S*/, double T)
{
  ConceptDetectionParams p;
  p.min_episodes = core::MinEpisodesForConcept (T);
  p.entity_frequency_threshold = core::EntityFrequencyThreshold (T);
  return p;
}

namespace
{

/// @brief Executes a write query on the transaction.
void
Add (Transaction &tx, const std::string &q,
     const std::vector<std::any> &p = {})
{
  tx.Execute (q, p);
}

/// @brief Structure for candidate concepts.
struct ConceptCandidate
{
  std::string entity_name;
  int episode_count;
  int total_mentions;
  std::vector<std::string> summary_ids;
};

/// @brief Load candidate concepts that meet episode and frequency thresholds.
std::vector<ConceptCandidate>
LoadConceptCandidates (Store *store, int min_episodes, int frequency_threshold)
{
  std::vector<ConceptCandidate> candidates;

  // Query entities that appear across multiple episodes with sufficient frequency
  // Join through consolidation_sources and memories to get episode_id
  auto rows = store->Execute (
      "SELECT e.name, "
      "       COUNT(DISTINCT COALESCE(m.episode_id, 0)) AS episode_count, "
      "       COUNT(*) AS total_mentions "
      "FROM extraction_entities e "
      "JOIN consolidation_sources cs ON e.summary_id = cs.summary_id "
      "LEFT JOIN memories m ON cs.source_embedding_id = m.embedding_id "
      "GROUP BY e.name "
      "HAVING episode_count >= ?1 AND total_mentions >= ?2",
      { static_cast<long long> (min_episodes),
        static_cast<long long> (frequency_threshold) });

  for (const auto &row : rows)
    {
      auto it_name = row.find ("name");
      auto it_episodes = row.find ("episode_count");
      auto it_mentions = row.find ("total_mentions");

      if (it_name == row.end () || it_episodes == row.end ()
          || it_mentions == row.end ())
        {
          continue;
        }

      if (it_name->second.type () != typeid (std::string))
        {
          continue;
        }

      ConceptCandidate c;
      c.entity_name = std::any_cast<std::string> (it_name->second);

      if (it_episodes->second.type () == typeid (long long))
        {
          c.episode_count
              = static_cast<int> (std::any_cast<long long> (it_episodes->second));
        }
      else
        {
          continue;
        }

      if (it_mentions->second.type () == typeid (long long))
        {
          c.total_mentions
              = static_cast<int> (std::any_cast<long long> (it_mentions->second));
        }
      else
        {
          continue;
        }

      candidates.push_back (std::move (c));
    }

  // For each candidate, get the summary IDs they appear in
  for (auto &c : candidates)
    {
      auto summary_rows = store->Execute (
          "SELECT DISTINCT summary_id FROM extraction_entities WHERE name = ?",
          { c.entity_name });

      for (const auto &sr : summary_rows)
        {
          auto it = sr.find ("summary_id");
          if (it != sr.end () && it->second.type () == typeid (std::string))
            {
              c.summary_ids.push_back (std::any_cast<std::string> (it->second));
            }
        }
    }

  return candidates;
}

/// @brief Compute concept centroid from related summary centroids.
/// Returns true if centroid was computed successfully.
bool
ComputeConceptCentroid (Store *store,
                        const std::vector<std::string> &summary_ids,
                        Eigen::VectorXf &centroid_out)
{
  if (summary_ids.empty ())
    {
      return false;
    }

  constexpr int kEmbeddingDim = 256;
  Eigen::VectorXf sum = Eigen::VectorXf::Zero (kEmbeddingDim);
  int count = 0;

  for (const auto &sid : summary_ids)
    {
      auto rows = store->Execute (
          "SELECT centroid FROM consolidation_summaries WHERE summary_id = ?",
          { sid });

      for (const auto &row : rows)
        {
          auto it = row.find ("centroid");
          if (it == row.end ())
            {
              continue;
            }

          Eigen::VectorXf centroid;
          if (core::DecodeFloatBlob (it->second, kEmbeddingDim, centroid))
            {
              sum += centroid;
              ++count;
            }
        }
    }

  if (count == 0)
    {
      return false;
    }

  centroid_out = sum / static_cast<float> (count);
  return true;
}

} // namespace

void
DetectConceptNodes::Execute (OperationContext &context, Transaction &tx) const
{
  Store *store = context.GetStore ();
  if (!store)
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  auto params
      = ConceptDetectionParams::FromKnobs (cfg.focus, cfg.sensitivity, cfg.stability);

  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);

  // Load candidate concepts meeting thresholds
  auto candidates
      = LoadConceptCandidates (store, params.min_episodes,
                               params.entity_frequency_threshold);

  if (candidates.empty ())
    {
      telemetry::LogDebug("cortext.concept_detection", {
        telemetry::Attribute::Int64("new_concepts_count", 0)
      });
      return;
    }

  int new_concepts_count = 0;

  for (const auto &c : candidates)
    {
      const std::string concept_node_id = "concept:" + c.entity_name;

      // Check if concept node already exists
      auto existing = store->Execute (
          "SELECT node_id FROM graph_nodes WHERE node_id = ?",
          { concept_node_id });

      if (!existing.empty ())
        {
          // Concept already exists, skip
          continue;
        }

      new_concepts_count++;

      // Compute concept centroid from related summary centroids
      Eigen::VectorXf centroid;
      if (!ComputeConceptCentroid (store, c.summary_ids, centroid))
        {
          continue;
        }

      // Convert centroid to vector<float> for storage
      std::vector<float> centroid_vec (centroid.data (),
                                       centroid.data () + centroid.size ());

      // Insert concept embedding (v2: minimal table)
      Add (tx,
           "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
           { centroid_vec, now_ts });

      // Get the auto-assigned embedding_id
      auto id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
      long long embedding_id = 0;
      if (!id_rows.empty () && id_rows[0].count ("id"))
        {
          auto val = id_rows[0].at ("id");
          if (val.type () == typeid (long long))
            {
              embedding_id = std::any_cast<long long> (val);
            }
          else if (val.type () == typeid (int))
            {
              embedding_id = std::any_cast<int> (val);
            }
        }

      // Create MEMORIES row with kind='LABEL' for concept (v2 schema)
      Add (tx,
           "INSERT INTO memories "
           "(embedding_id, source_id, kind, label, start_ts, created_at) "
           "VALUES (?, ?, 'LABEL', ?, ?, ?)",
           { embedding_id, concept_node_id, c.entity_name, now_ts, now_ts });

      // Create concept node in graph_nodes (legacy table for backward
      // compatibility)
      Add (tx,
           "INSERT OR IGNORE INTO graph_nodes "
           "(node_id, type, label, embedding_id, created_at) "
           "VALUES (?1, 'concept', ?2, ?3, ?4)",
           { concept_node_id, c.entity_name, embedding_id, now_ts });

      // Create 'generalizes' edges from concept to related entities
      const std::string entity_node_id = "entity:" + c.entity_name;
      Add (tx,
           "INSERT OR REPLACE INTO graph_edges "
           "(source_id, target_id, edge_type, weight, last_reinforced) "
           "VALUES (?1, ?2, 'generalizes', ?3, ?4)",
           { concept_node_id, entity_node_id,
             static_cast<double> (c.total_mentions), now_ts });

      // Link concept to all summaries it appears in
      for (const auto &sid : c.summary_ids)
        {
          Add (tx,
               "INSERT OR REPLACE INTO graph_edges "
               "(source_id, target_id, edge_type, weight, last_reinforced) "
               "VALUES (?1, ?2, 'abstracted_from', 1.0, ?3)",
               { concept_node_id, "summary:" + sid, now_ts });
        }
    }

  telemetry::LogDebug("cortext.concept_detection", {
    telemetry::Attribute::Int64("new_concepts_count", new_concepts_count)
  });
}

} // namespace cortext::operations
