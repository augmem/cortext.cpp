#include "cortext/operations/consolidation_summarize.hpp"

#include "cortext/buffered_write_instruction.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/summarizer/summarizer.hpp"
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

/// @brief Add a buffered write instruction.
void
AddWrite (OperationContext &ctx, const std::string &q,
          const std::vector<std::any> &p = {})
{
  BufferedWriteInstruction op;
  op.query = q;
  op.params = p;
  ctx.AddWriteInstruction (std::move (op));
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
ConsolidationSummarize::Execute (OperationContext &context) const
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
              "SELECT ve.embedding, m.blob_id "
              "FROM vec_embeddings ve "
              "LEFT JOIN memories m ON ve.embedding_id = m.embedding_id "
              "WHERE ve.embedding_id = ?",
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

      // 3. Insert into consolidation_summaries.
      AddWrite (context,
                "INSERT INTO consolidation_summaries"
                "(summary_id, summary_text, centroid, cluster_size) "
                "VALUES(?, ?, ?, ?)",
                { summary_id, summary_text, centroid_blob,
                  static_cast<long long> (cluster.embedding_ids.size ()) });

      // 4. Insert source mappings into consolidation_sources.
      for (long long emb_id : cluster.embedding_ids)
        {
          AddWrite (context,
                    "INSERT INTO consolidation_sources"
                    "(summary_id, source_embedding_id) "
                    "VALUES(?, ?)",
                    { summary_id, emb_id });
        }

      // 5. Create vec_embeddings entry for centroid.
      // Get next embedding_id via MAX + 1.
      AddWrite (context,
                "INSERT INTO vec_embeddings(embedding_id, embedding) "
                "SELECT COALESCE(MAX(embedding_id), 0) + 1, ? "
                "FROM vec_embeddings",
                { centroid_blob });

      // 6. Update cluster_id in embeddings_meta for source embeddings.
      for (long long emb_id : cluster.embedding_ids)
        {
          AddWrite (context,
                    "UPDATE embeddings_meta SET cluster_id = ? "
                    "WHERE embedding_id = ?",
                    { cluster.cluster_id, emb_id });
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

  // 8. Clear processed candidates.
  AddWrite (context,
            "DELETE FROM consolidation_candidates WHERE embedding_id IN "
            "(SELECT source_embedding_id FROM consolidation_sources)");

  // 9. Pass extraction requests to next operation.
  context.SetExtractionRequests (std::move (extraction_requests));
}

} // namespace cortext::operations
