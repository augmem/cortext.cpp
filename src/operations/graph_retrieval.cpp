#include "cortext/operations/graph_retrieval.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cortext::operations
{

namespace
{
inline bool
StartsWith (const std::string &s, const char *prefix)
{
  const std::string p (prefix);
  return s.size () >= p.size () && s.compare (0, p.size (), p) == 0;
}

bool
AnyToBytes (const std::any &v, const unsigned char *&data, size_t &len)
{
  if (v.type () == typeid (std::vector<char>))
    {
      const auto &blob = std::any_cast<const std::vector<char> &> (v);
      data = reinterpret_cast<const unsigned char *> (blob.data ());
      len = blob.size ();
      return true;
    }
  if (v.type () == typeid (std::vector<unsigned char>))
    {
      const auto &blob = std::any_cast<const std::vector<unsigned char> &> (v);
      data = blob.data ();
      len = blob.size ();
      return true;
    }
  return false;
}

bool
DecodeFloatBlob (const std::any &v, int dim, Eigen::VectorXf &out)
{
  const unsigned char *data = nullptr;
  size_t len = 0;
  if (!AnyToBytes (v, data, len))
    {
      return false;
    }
  const size_t want = static_cast<size_t> (dim) * sizeof (float);
  if (dim <= 0 || len != want)
    {
      return false;
    }
  out.resize (dim);
  for (int i = 0; i < dim; ++i)
    {
      float f = 0.0f;
      std::memcpy (&f, data + static_cast<size_t> (i) * sizeof (float),
                   sizeof (float));
      out[i] = f;
    }
  return true;
}

std::string
EmbNodeId (long long embedding_id)
{
  return std::string ("emb:") + std::to_string (embedding_id);
}

} // namespace

void
GraphAugmentedRetrieveCandidates::Execute (OperationContext &context) const
{
  Store *store = context.GetStore ();
  if (!store)
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();
  const Eigen::VectorXf &q = context.GetSignal ().embedding;
  if (q.size () == 0)
    {
      return;
    }

  const int k = std::max (1, core::MaxResults (cfg.focus));
  const int depth = std::max (1, core::GraphDepth (cfg.focus));

  // Seed vector retrieval (linear scan over embeddings table).
  struct Scored
  {
    long long id;
    double score;
    Eigen::VectorXf vec;
  };
  std::vector<Scored> seeds;
  seeds.reserve (static_cast<size_t> (k));

  try
    {
      auto rows = store->Execute ("SELECT embedding_id, embedding FROM embeddings",
                                  {});
      for (const auto &row : rows)
        {
          auto it_id = row.find ("embedding_id");
          auto it_emb = row.find ("embedding");
          if (it_id == row.end () || it_emb == row.end ())
            continue;
          if (it_id->second.type () != typeid (long long))
            continue;

          const long long id = std::any_cast<long long> (it_id->second);
          Eigen::VectorXf v;
          if (!DecodeFloatBlob (it_emb->second, q.size (), v))
            continue;
          const double sim = core::CosineSimilarity (q, v);
          seeds.push_back (Scored{ id, sim, v });
        }
    }
  catch (...)
    {
      return;
    }

  if (seeds.empty ())
    {
      return;
    }

  std::sort (seeds.begin (), seeds.end (),
             [] (const Scored &a, const Scored &b) {
               return a.score > b.score;
             });
  if (static_cast<int> (seeds.size ()) > k)
    {
      seeds.resize (static_cast<size_t> (k));
    }

  // Update last retrieval timestamp for consolidation idle-gating.
  p_ctx.last_retrieval_ts = context.GetSignal ().timestamp;

  // Graph expansion from seed embedding node IDs.
  std::vector<std::string> seed_nodes;
  seed_nodes.reserve (seeds.size ());
  for (const auto &s : seeds)
    {
      seed_nodes.push_back (EmbNodeId (s.id));
    }

  std::unordered_set<std::string> expanded_nodes;
  for (const auto &s : seed_nodes)
    {
      expanded_nodes.insert (s);
    }

  try
    {
      // Build SQL with a VALUES list for seeds.
      std::string sql = "WITH RECURSIVE seed(id) AS (VALUES ";
      std::vector<std::any> params;
      params.reserve (seed_nodes.size () + 2);
      for (size_t i = 0; i < seed_nodes.size (); ++i)
        {
          if (i > 0)
            sql += ", ";
          sql += "(?)";
          params.push_back (seed_nodes[i]);
        }
      sql += "), expand(id, depth) AS ("
             "  SELECT id, 0 FROM seed "
             "  UNION "
             "  SELECT ge.target_id, expand.depth+1 "
             "  FROM graph_edges ge JOIN expand ON ge.source_id = expand.id "
             "  WHERE expand.depth < ? "
             "  UNION "
             "  SELECT ge.source_id, expand.depth+1 "
             "  FROM graph_edges ge JOIN expand ON ge.target_id = expand.id "
             "  WHERE expand.depth < ? "
             ") "
             "SELECT DISTINCT id FROM expand;";
      params.push_back (static_cast<long long> (depth));
      params.push_back (static_cast<long long> (depth));

      auto rows = store->Execute (sql, params);
      for (const auto &r : rows)
        {
          auto it = r.find ("id");
          if (it != r.end () && it->second.type () == typeid (std::string))
            {
              expanded_nodes.insert (std::any_cast<std::string> (it->second));
            }
        }
    }
  catch (...)
    {
      // No graph tables or query failed; use seed-only behavior.
    }

  // Map expanded graph node ids to embedding_ids via graph_nodes (and by emb: prefix).
  std::unordered_set<long long> expanded_embedding_ids;
  expanded_embedding_ids.reserve (expanded_nodes.size ());
  for (const auto &n : expanded_nodes)
    {
      if (StartsWith (n, "emb:"))
        {
          const char *p = n.c_str () + 4;
          char *end = nullptr;
          const long long id = std::strtoll (p, &end, 10);
          if (end != p)
            expanded_embedding_ids.insert (id);
        }
    }

  // Also include any nodes that have embedding_id set.
  if (!expanded_nodes.empty ())
    {
      try
        {
          std::string sql = "SELECT embedding_id FROM graph_nodes WHERE node_id IN (";
          std::vector<std::any> params;
          params.reserve (expanded_nodes.size ());
          bool first = true;
          for (const auto &nid : expanded_nodes)
            {
              if (!first)
                sql += ",";
              first = false;
              sql += "?";
              params.push_back (nid);
            }
          sql += ") AND embedding_id IS NOT NULL;";

          auto rows = store->Execute (sql, params);
          for (const auto &r : rows)
            {
              auto it = r.find ("embedding_id");
              if (it != r.end () && it->second.type () == typeid (long long))
                {
                  expanded_embedding_ids.insert (
                      std::any_cast<long long> (it->second));
                }
            }
        }
      catch (...)
        {
        }
    }

  // Fetch embeddings for expanded ids and re-rank.
  std::unordered_map<long long, Eigen::VectorXf> out;
  std::vector<Scored> scored;
  scored.reserve (expanded_embedding_ids.size ());

  // If we cannot fetch, at least return seeds.
  bool fetched_any = false;
  if (!expanded_embedding_ids.empty ())
    {
      try
        {
          std::string sql = "SELECT embedding_id, embedding FROM embeddings WHERE embedding_id IN (";
          std::vector<std::any> params;
          params.reserve (expanded_embedding_ids.size ());
          bool first = true;
          for (const long long id : expanded_embedding_ids)
            {
              if (!first)
                sql += ",";
              first = false;
              sql += "?";
              params.push_back (id);
            }
          sql += ");";
          auto rows = store->Execute (sql, params);
          for (const auto &row : rows)
            {
              auto it_id = row.find ("embedding_id");
              auto it_emb = row.find ("embedding");
              if (it_id == row.end () || it_emb == row.end ())
                continue;
              if (it_id->second.type () != typeid (long long))
                continue;
              const long long id = std::any_cast<long long> (it_id->second);
              Eigen::VectorXf v;
              if (!DecodeFloatBlob (it_emb->second, q.size (), v))
                continue;
              const double sim = core::CosineSimilarity (q, v);
              scored.push_back (Scored{ id, sim, v });
              fetched_any = true;
            }
        }
      catch (...)
        {
        }
    }

  if (!fetched_any)
    {
      for (const auto &s : seeds)
        {
          out.emplace (s.id, s.vec);
        }
      context.SetRetrievedMemoryEmbeddings (std::move (out));
      return;
    }

  std::sort (scored.begin (), scored.end (),
             [] (const Scored &a, const Scored &b) {
               return a.score > b.score;
             });
  if (static_cast<int> (scored.size ()) > k)
    {
      scored.resize (static_cast<size_t> (k));
    }
  for (const auto &s : scored)
    {
      out.emplace (s.id, s.vec);
    }

  context.SetRetrievedMemoryEmbeddings (std::move (out));
}

} // namespace cortext::operations

