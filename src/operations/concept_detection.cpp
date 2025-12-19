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
  std::vector<long long> association_memory_ids; // V2: memory_ids of ASSOCIATION memories
};

/// @brief Load candidate concepts that meet episode and frequency thresholds.
/// V2 schema: LABEL memories are entities, ASSOCIATION memories are summaries,
/// ASSOCIATIONS with edge_type='derived_from' link them.
std::vector<ConceptCandidate>
LoadConceptCandidates (Store *store, int min_episodes, int frequency_threshold)
{
  std::vector<ConceptCandidate> candidates;

  // V2: Query LABEL memories (entities) that appear across multiple episodes
  // with sufficient frequency. Join through ASSOCIATIONS (derived_from) to get
  // the ASSOCIATION memories (summaries) they were derived from, then to source
  // memories for episode_id.
  auto rows = store->Execute (
      "SELECT label_mem.label AS name, "
      "       COUNT(DISTINCT COALESCE(src_mem.episode_id, 0)) AS episode_count, "
      "       COUNT(*) AS total_mentions "
      "FROM memories label_mem "
      "JOIN associations a1 ON a1.target_memory_id = label_mem.memory_id "
      "  AND a1.edge_type = 'derived_from' "
      "JOIN memories assoc_mem ON a1.source_memory_id = assoc_mem.memory_id "
      "  AND assoc_mem.kind = 'ASSOCIATION' "
      "JOIN associations a2 ON a2.source_memory_id = assoc_mem.memory_id "
      "  AND a2.edge_type = 'derived_from' "
      "JOIN memories src_mem ON a2.target_memory_id = src_mem.memory_id "
      "WHERE label_mem.kind = 'LABEL' "
      "GROUP BY label_mem.label "
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

  // For each candidate, get the ASSOCIATION memory_ids they appear in
  for (auto &c : candidates)
    {
      // V2: Find ASSOCIATION memories linked to this LABEL via derived_from
      auto assoc_rows = store->Execute (
          "SELECT DISTINCT assoc_mem.memory_id "
          "FROM memories label_mem "
          "JOIN associations a ON a.target_memory_id = label_mem.memory_id "
          "  AND a.edge_type = 'derived_from' "
          "JOIN memories assoc_mem ON a.source_memory_id = assoc_mem.memory_id "
          "  AND assoc_mem.kind = 'ASSOCIATION' "
          "WHERE label_mem.kind = 'LABEL' AND label_mem.label = ?",
          { c.entity_name });

      for (const auto &ar : assoc_rows)
        {
          auto it = ar.find ("memory_id");
          if (it != ar.end () && it->second.type () == typeid (long long))
            {
              c.association_memory_ids.push_back (
                  std::any_cast<long long> (it->second));
            }
        }
    }

  return candidates;
}

/// @brief Compute concept centroid from related ASSOCIATION memory embeddings.
/// V2: ASSOCIATION memories store embeddings directly via embedding_id.
/// Returns true if centroid was computed successfully.
bool
ComputeConceptCentroid (Store *store,
                        const std::vector<long long> &association_memory_ids,
                        Eigen::VectorXf &centroid_out)
{
  if (association_memory_ids.empty ())
    {
      return false;
    }

  constexpr int kEmbeddingDim = 256;
  Eigen::VectorXf sum = Eigen::VectorXf::Zero (kEmbeddingDim);
  int count = 0;

  for (const long long mem_id : association_memory_ids)
    {
      // V2: Get embedding from MEMORIES -> embeddings join
      auto rows = store->Execute (
          "SELECT e.embedding FROM memories m "
          "JOIN embeddings e ON m.embedding_id = e.embedding_id "
          "WHERE m.memory_id = ?",
          { mem_id });

      for (const auto &row : rows)
        {
          auto it = row.find ("embedding");
          if (it == row.end ())
            {
              continue;
            }

          Eigen::VectorXf emb;
          if (core::DecodeFloatBlob (it->second, kEmbeddingDim, emb))
            {
              sum += emb;
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
      telemetry::LogDebug ("cortext.concept_detection",
                           { telemetry::Attribute::Int64 ("new_concepts_count", 0) });
      return;
    }

  int new_concepts_count = 0;

  for (const auto &c : candidates)
    {
      const std::string concept_source_id = "concept:" + c.entity_name;

      // V2: Check if concept LABEL memory already exists
      auto existing = store->Execute (
          "SELECT memory_id FROM memories "
          "WHERE kind = 'LABEL' AND source_id = ?",
          { concept_source_id });

      if (!existing.empty ())
        {
          // Concept already exists, skip
          continue;
        }

      new_concepts_count++;

      // Compute concept centroid from related ASSOCIATION memory embeddings
      Eigen::VectorXf centroid;
      if (!ComputeConceptCentroid (store, c.association_memory_ids, centroid))
        {
          continue;
        }

      // Convert centroid to vector<float> for storage
      std::vector<float> centroid_vec (centroid.data (),
                                       centroid.data () + centroid.size ());

      // Insert concept embedding
      Add (tx, "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
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

      // V2: Create MEMORIES row with kind='LABEL' for concept
      Add (tx,
           "INSERT INTO memories "
           "(embedding_id, source_id, kind, label, start_ts, created_at) "
           "VALUES (?, ?, 'LABEL', ?, ?, ?)",
           { embedding_id, concept_source_id, c.entity_name, now_ts, now_ts });

      // Get the concept memory_id
      auto mem_id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
      long long concept_memory_id = 0;
      if (!mem_id_rows.empty () && mem_id_rows[0].count ("id"))
        {
          auto val = mem_id_rows[0].at ("id");
          if (val.type () == typeid (long long))
            {
              concept_memory_id = std::any_cast<long long> (val);
            }
          else if (val.type () == typeid (int))
            {
              concept_memory_id = std::any_cast<int> (val);
            }
        }

      // V2: Find the entity LABEL memory for 'generalizes' edge
      auto entity_rows = store->Execute (
          "SELECT memory_id FROM memories "
          "WHERE kind = 'LABEL' AND label = ? AND source_id != ?",
          { c.entity_name, concept_source_id });

      for (const auto &er : entity_rows)
        {
          auto it = er.find ("memory_id");
          if (it != er.end () && it->second.type () == typeid (long long))
            {
              long long entity_memory_id = std::any_cast<long long> (it->second);

              // V2: Create 'generalizes' edge in ASSOCIATIONS
              Add (tx,
                   "INSERT OR REPLACE INTO associations "
                   "(source_memory_id, target_memory_id, edge_type, weight, "
                   "last_reinforced) "
                   "VALUES (?, ?, 'generalizes', ?, ?)",
                   { concept_memory_id, entity_memory_id,
                     static_cast<double> (c.total_mentions), now_ts });
            }
        }

      // V2: Link concept to all ASSOCIATION memories it appears in
      for (const long long assoc_mem_id : c.association_memory_ids)
        {
          Add (tx,
               "INSERT OR REPLACE INTO associations "
               "(source_memory_id, target_memory_id, edge_type, weight, "
               "last_reinforced) "
               "VALUES (?, ?, 'abstracted_from', 1.0, ?)",
               { concept_memory_id, assoc_mem_id, now_ts });
        }
    }

  telemetry::LogDebug (
      "cortext.concept_detection",
      { telemetry::Attribute::Int64 ("new_concepts_count", new_concepts_count) });
}

} // namespace cortext::operations
