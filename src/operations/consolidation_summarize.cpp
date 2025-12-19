#include "cortext/operations/consolidation_summarize.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/summarizer/summarizer.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <any>
#include <ctime>
#include <sstream>
#include <string>
#include <vector>

namespace cortext::operations
{

namespace
{

/// @brief Generate a unique summary ID.
std::string
GenerateSummaryId (uint64_t ts, int counter)
{
  std::ostringstream ss;
  ss << "summary_" << ts << "_" << counter;
  return ss.str ();
}

/// @brief Execute a write query on the transaction.
void
AddWrite (Transaction &tx, const std::string &q,
          const std::vector<std::any> &p = {})
{
  tx.Execute (q, p);
}

/// @brief Decode blob to string if possible.
std::string
BlobToString (const std::any &val)
{
  if (val.type () == typeid (std::vector<unsigned char>))
    {
      const auto &blob = std::any_cast<std::vector<unsigned char>> (val);
      return std::string (blob.begin (), blob.end ());
    }
  if (val.type () == typeid (std::string))
    {
      return std::any_cast<std::string> (val);
    }
  return {};
}

/// @brief Extract blob bytes from any.
std::vector<unsigned char>
GetBlobBytes (const std::any &val)
{
  if (val.type () == typeid (std::vector<unsigned char>))
    {
      return std::any_cast<std::vector<unsigned char>> (val);
    }
  return {};
}

} // namespace

ConsolidationSummarizeParams
ConsolidationSummarizeParams::FromKnobs (double F, double /*S*/, double /*T*/)
{
  ConsolidationSummarizeParams p;
  p.min_cluster_size_for_extraction = core::MinClusterSizeForExtraction (F);
  return p;
}

void
ConsolidationSummarize::Execute (OperationContext &context, Transaction &tx) const
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

  Store *store = context.GetStore ();
  if (!store)
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  auto params
      = ConsolidationSummarizeParams::FromKnobs (cfg.focus, cfg.sensitivity,
                                                  cfg.stability);
  const uint64_t now_ts = context.GetSignal ().timestamp;

  std::vector<ExtractionRequest> extraction_requests;
  int summary_counter = 0;

  // Get summarizer (may be null if OGA disabled)
  Summarizer *summarizer = context.GetSummarizer ();

  for (const auto &cluster : clusters)
    {
      std::string summary_id = GenerateSummaryId (now_ts, summary_counter++);

      // Collect all source texts and find most representative memory
      std::vector<std::string> source_texts;
      std::string best_text;
      double best_sim = -1.0;

      // Convert cluster centroid to Eigen for comparison.
      Eigen::VectorXf centroid_eigen (
          static_cast<Eigen::Index> (cluster.centroid.size ()));
      for (size_t i = 0; i < cluster.centroid.size (); ++i)
        {
          centroid_eigen (static_cast<Eigen::Index> (i)) = cluster.centroid[i];
        }

      for (long long emb_id : cluster.embedding_ids)
        {
          // Query embedding and blob_id for this memory.
          auto rows = store->Execute (
              "SELECT e.embedding, m.blob_id "
              "FROM embeddings e "
              "LEFT JOIN memories m ON e.embedding_id = m.embedding_id "
              "WHERE e.embedding_id = ?",
              { emb_id });

          if (rows.empty ())
            {
              continue;
            }

          const auto &row = rows[0];
          auto it_emb = row.find ("embedding");
          if (it_emb == row.end ())
            {
              continue;
            }

          // Decode embedding.
          Eigen::VectorXf emb;
          if (!core::DecodeFloatBlob (it_emb->second,
                                       static_cast<int> (cluster.centroid.size ()),
                                       emb))
            {
              continue;
            }

          // Try to fetch text from objstore.
          std::string text;
          auto it_blob = row.find ("blob_id");
          if (it_blob != row.end ())
            {
              auto blob_id = GetBlobBytes (it_blob->second);
              if (!blob_id.empty ())
                {
                  auto data_rows = store->Execute (
                      "SELECT objstore_get(?1) AS data", { blob_id });
                  if (!data_rows.empty ())
                    {
                      auto it_data = data_rows[0].find ("data");
                      if (it_data != data_rows[0].end ())
                        {
                          text = BlobToString (it_data->second);
                        }
                    }
                }
            }

          // Collect text for summarizer
          if (!text.empty ())
            {
              source_texts.push_back (text);
            }

          // Track best (most representative) for extractive fallback
          double sim = core::CosineSimilarity (centroid_eigen, emb);
          if (sim > best_sim)
            {
              best_sim = sim;
              best_text = text;
            }
        }

      // Use abstractive summarizer if available, else extractive fallback
      std::string summary_text;
      if (summarizer && summarizer->IsAvailable () && !source_texts.empty ())
        {
          try
            {
              summary_text = summarizer->SummarizeTexts (source_texts);
            }
          catch (const std::exception &)
            {
              // Fallback to extractive on error
              summary_text = best_text;
            }
        }
      else
        {
          summary_text = best_text;
        }

      // 2. Convert centroid to blob for storage.
      std::vector<float> centroid_blob = cluster.centroid;

      // 3. Create embeddings entry for centroid (v2: minimal table).
      AddWrite (tx,
                "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
                { centroid_blob, static_cast<long long> (now_ts) });

      // Get the auto-assigned embedding_id
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

      // 4. Create MEMORIES row for centroid with kind='ASSOCIATION' (v2 schema).
      // Store summary_text as label, n_signals as cluster size.
      AddWrite (tx,
                "INSERT INTO memories "
                "(embedding_id, source_id, kind, label, start_ts, n_signals, created_at) "
                "VALUES (?, ?, 'ASSOCIATION', ?, ?, ?, ?)",
                { centroid_embedding_id, summary_id, summary_text,
                  static_cast<long long> (now_ts),
                  static_cast<long long> (cluster.embedding_ids.size ()),
                  static_cast<long long> (now_ts) });

      // Get the memory_id for the centroid memory
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

      // 5. Update cluster_id in memories for source embeddings and create
      // ASSOCIATIONS (derived_from edges).
      for (long long emb_id : cluster.embedding_ids)
        {
          // Update cluster_id
          AddWrite (tx,
                    "UPDATE memories SET cluster_id = ? "
                    "WHERE embedding_id = ?",
                    { cluster.cluster_id, emb_id });

          // Get source memory_id for this embedding
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

              // Create derived_from edge: centroid <- source
              if (src_memory_id > 0 && centroid_memory_id > 0)
                {
                  AddWrite (tx,
                            "INSERT OR IGNORE INTO associations "
                            "(source_memory_id, target_memory_id, edge_type, weight) "
                            "VALUES (?, ?, 'derived_from', 1.0)",
                            { centroid_memory_id, src_memory_id });
                }
            }
        }

      // 7. Queue extraction if cluster is large enough.
      if (static_cast<int> (cluster.embedding_ids.size ())
          >= params.min_cluster_size_for_extraction)
        {
          ExtractionRequest req;
          req.summary_id = summary_id;
          req.summary_text = summary_text;
          req.cluster_size = static_cast<int> (cluster.embedding_ids.size ());
          req.created_at = now_ts;
          req.source_texts = source_texts;  // Reuse already collected texts
          extraction_requests.push_back (std::move (req));
        }
    }

  // 8. Pass extraction requests to next operation.
  const int jobs_queued = static_cast<int>(extraction_requests.size());
  context.SetExtractionRequests (std::move (extraction_requests));

  telemetry::LogDebug("cortext.consolidation_summarize", {
    telemetry::Attribute::Int64("summary_count", summary_counter),
    telemetry::Attribute::Int64("extraction_jobs_queued", jobs_queued)
  });
}

} // namespace cortext::operations
