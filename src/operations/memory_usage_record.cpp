#include "cortext/operations/memory_usage_record.hpp"

#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"

#include <any>
#include <chrono>
#include <map>
#include <string>

namespace cortext::operations
{

namespace
{

long long
GetInt64 (const std::map<std::string, std::any> &row, const char *key,
          long long fallback)
{
  const auto it = row.find (key);
  if (it == row.end ())
    return fallback;
  return store::AnyToLongLong (it->second).value_or (fallback);
}

} // namespace

void
RecordMemoryUsage::Execute (OperationContext &context, Transaction &tx) const
{
  if (context.GetSignal ().retention == Retention::Ephemeral)
    {
      return;
    }
  auto &p_ctx = context.GetProcessorContext ();
  const long long ts = static_cast<long long> (context.GetSignal ().timestamp);
  for (const auto &e : context.GetMemoryUsageEvents ())
    {
      const long long used_flag = e.used ? 1 : 0;
      const long long embedding_id = static_cast<long long> (e.embedding_id);
      const long long memory_id = static_cast<long long> (e.memory_id);
      const bool has_memory_id = memory_id > 0;
      const std::string lookup_column
          = has_memory_id ? "memory_id" : "embedding_id";
      const long long lookup_id = has_memory_id ? memory_id : embedding_id;

      const auto start = std::chrono::steady_clock::now ();
      auto rows = tx.Execute (
          "SELECT memory_id, retrieved_count, used_count FROM memories "
          "WHERE " + lookup_column + " = ?",
          { lookup_id });
      if (rows.empty ())
        {
          continue;
        }
      const long long row_memory_id
          = GetInt64 (rows[0], "memory_id", memory_id);
      const long long retrieved_new
          = GetInt64 (rows[0], "retrieved_count", 0) + 1;
      const long long used_new
          = GetInt64 (rows[0], "used_count", 0) + used_flag;
      tx.Execute ("UPDATE memories "
                  "SET retrieved_count = ?, "
                  "    used_count = ?, "
                  "    last_used = CASE WHEN ? > 0 THEN ? ELSE last_used END, "
                  "    last_access = ? "
                  "WHERE " + lookup_column + " = ?",
                  { retrieved_new, used_new, used_flag, ts, ts, lookup_id });
      if (row_memory_id > 0)
        {
          p_ctx.UpdateRetrievalSurfaceUsageByMemory (row_memory_id,
                                                     retrieved_new, used_new,
                                                     ts);
        }
      else
        {
          p_ctx.UpdateRetrievalSurfaceUsageByEmbedding (
              embedding_id, retrieved_new, used_new, ts);
        }
      context.AddOperationTiming (
          "MemoryUsage.record_sql",
          std::chrono::duration<double, std::milli> (
              std::chrono::steady_clock::now () - start)
              .count ());
    }
}

} // namespace cortext::operations
