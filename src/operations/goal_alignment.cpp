#include "cortext/operations/goal_alignment.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <string>
#include <unordered_set>
#include <vector>

namespace cortext::operations
{

void
ComputeGoalAlignment::Execute (OperationContext &context, Transaction &tx) const
{
  Store *store = context.GetStore ();
  if (!store)
    {
      return;
    }

  const Eigen::VectorXf &q = context.GetSignal ().embedding;
  if (q.size () == 0)
    {
      return;
    }

  // 1) Read goal node ids.
  std::vector<std::string> goals;
  try
    {
      auto rows = store->Execute ("SELECT node_id FROM goal_nodes", {});
      for (const auto &r : rows)
        {
          auto it = r.find ("node_id");
          if (it != r.end () && it->second.type () == typeid (std::string))
            {
              goals.push_back (std::any_cast<std::string> (it->second));
            }
        }
    }
  catch (...)
    {
      return;
    }

  if (goals.empty ())
    {
      return;
    }

  // 2) Expand by 1 hop over graph_edges (undirected).
  std::unordered_set<std::string> nodes;
  for (const auto &g : goals)
    nodes.insert (g);

  try
    {
      std::string sql = "SELECT source_id, target_id FROM graph_edges WHERE source_id IN (";
      std::vector<std::any> params;
      params.reserve (goals.size () * 2);
      bool first = true;
      for (const auto &g : goals)
        {
          if (!first)
            sql += ",";
          first = false;
          sql += "?";
          params.push_back (g);
        }
      sql += ") OR target_id IN (";
      first = true;
      for (const auto &g : goals)
        {
          if (!first)
            sql += ",";
          first = false;
          sql += "?";
          params.push_back (g);
        }
      sql += ");";

      auto rows = store->Execute (sql, params);
      for (const auto &r : rows)
        {
          auto s_it = r.find ("source_id");
          auto t_it = r.find ("target_id");
          if (s_it != r.end () && s_it->second.type () == typeid (std::string))
            nodes.insert (std::any_cast<std::string> (s_it->second));
          if (t_it != r.end () && t_it->second.type () == typeid (std::string))
            nodes.insert (std::any_cast<std::string> (t_it->second));
        }
    }
  catch (...)
    {
      // If graph tables are missing, use goal-only context.
    }

  if (nodes.empty ())
    {
      return;
    }

  // 3) Map nodes -> embedding_id
  std::vector<long long> embedding_ids;
  try
    {
      std::string sql = "SELECT embedding_id FROM graph_nodes WHERE node_id IN (";
      std::vector<std::any> params;
      params.reserve (nodes.size ());
      bool first = true;
      for (const auto &nid : nodes)
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
              embedding_ids.push_back (std::any_cast<long long> (it->second));
            }
        }
    }
  catch (...)
    {
      return;
    }

  if (embedding_ids.empty ())
    {
      return;
    }

  // 4) Fetch embeddings and compute mean vector.
  Eigen::VectorXf sum = Eigen::VectorXf::Zero (q.size ());
  int count = 0;

  try
    {
      std::string sql
          = "SELECT embedding_id, embedding FROM embeddings WHERE embedding_id IN (";
      std::vector<std::any> params;
      params.reserve (embedding_ids.size ());
      bool first = true;
      for (const long long id : embedding_ids)
        {
          if (!first)
            sql += ",";
          first = false;
          sql += "?";
          params.push_back (id);
        }
      sql += ");";

      auto rows = store->Execute (sql, params);
      for (const auto &r : rows)
        {
          auto it = r.find ("embedding");
          if (it == r.end ())
            continue;
          Eigen::VectorXf v;
          if (!core::DecodeFloatBlob (it->second, q.size (), v))
            continue;
          sum += v;
          count += 1;
        }
    }
  catch (...)
    {
      return;
    }

  if (count <= 0)
    {
      return;
    }

  const Eigen::VectorXf mean = sum / static_cast<float> (count);
  const double cos = core::CosineSimilarity (q, mean);
  const double ga = core::Clamp (
      constants::kCosineToAlignmentScale
          * (cos + constants::kCosineToAlignmentOffset),
      constants::kNormalizedMin, constants::kNormalizedMax);
  context.SetMetric (operations::Metric::goal_alignment, ga);

  telemetry::LogDebug("cortext.goal_alignment", {
    telemetry::Attribute::Int64("goal_node_count", static_cast<long long>(goals.size())),
    telemetry::Attribute::Double("alignment_score", ga)
  });
}

} // namespace cortext::operations

