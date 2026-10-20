#include "member_vector_index_internal.hpp"

#include "cortext/core/utils.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <queue>

namespace cortext::operations::member_vector_index_internal
{
namespace
{

// Member rows of grouped LONG_TERM memories past a signal id. CROSS JOIN
// keeps signals first: a range on its primary key, so a refresh reads only
// rows stored since the last one.
constexpr const char *kMemberRowsSql
    = "SELECT s.signal_id AS signal_id, v.embedding AS embedding "
      "FROM signals s "
      "CROSS JOIN memories m ON m.memory_id = s.memory_id "
      "CROSS JOIN memory_signal_vectors v ON v.blob_id = s.blob_id "
      "WHERE s.signal_id > ? AND m.kind = 'LONG_TERM' AND m.n_signals > 1 "
      "ORDER BY s.signal_id";

// The highest LONG_TERM signal row (any memory size) above a signal id:
// walks down from the top of the signals key, past at most the working-memory
// tail. Refreshes read up to it and resume above it, so the long-term rows
// that are not members (single-signal memories) are read once, not on every
// query. Rows above it can only be working rows, which never become members
// (a closed unit is written as new LONG_TERM rows), so nothing is skipped.
constexpr const char *kTopLongTermRowSql
    = "SELECT s.signal_id AS signal_id "
      "FROM signals s CROSS JOIN memories m ON m.memory_id = s.memory_id "
      "WHERE s.signal_id > ? AND m.kind = 'LONG_TERM' "
      "ORDER BY s.signal_id DESC LIMIT 1";

// Appends one member as int8 codes (normalised, one scale). `vec` is
// normalised in place.
void
Append (internal::MemberVectors &index, long long signal_id,
        Eigen::VectorXf &vec)
{
  index.read_through = std::max (index.read_through, signal_id);
  const float norm = vec.norm ();
  if (norm <= 0.0f)
    return;
  vec /= norm;
  const float max_abs = vec.cwiseAbs ().maxCoeff ();
  const float scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
  if (scale <= 0.0f)
    return;
  index.signal_ids.push_back (signal_id);
  index.scales.push_back (scale);
  for (int i = 0; i < index.dim; ++i)
    index.codes.push_back (static_cast<std::int8_t> (
        std::lround (std::clamp (vec[i] / scale, -127.0f, 127.0f))));
}

// The warm read: every member row of the database at `path`, on its own
// read-only connection. Dimension from the first vector's size. Empty
// (dim 0) on any failure.
internal::MemberVectors
ReadAll (const std::string &path)
{
  const auto start = std::chrono::steady_clock::now ();
  internal::MemberVectors out;
  sqlite3 *db = nullptr;
  bool ok = sqlite3_open_v2 (path.c_str (), &db,
                             SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX,
                             nullptr)
            == SQLITE_OK;
  // One read transaction, so the mark and the rows come from one snapshot.
  ok = ok && sqlite3_exec (db, "BEGIN", nullptr, nullptr, nullptr) == SQLITE_OK;
  sqlite3_stmt *stmt = nullptr;
  long long mark = 0;
  ok = ok
       && sqlite3_prepare_v2 (db, kTopLongTermRowSql, -1, &stmt, nullptr)
              == SQLITE_OK
       && sqlite3_bind_int64 (stmt, 1, 0) == SQLITE_OK;
  int rc = ok ? sqlite3_step (stmt) : SQLITE_ERROR;
  if (rc == SQLITE_ROW)
    mark = sqlite3_column_int64 (stmt, 0);
  ok = ok && (rc == SQLITE_ROW || rc == SQLITE_DONE);
  sqlite3_finalize (stmt);
  stmt = nullptr;
  ok = ok
       && sqlite3_prepare_v2 (db, kMemberRowsSql, -1, &stmt, nullptr)
              == SQLITE_OK
       && sqlite3_bind_int64 (stmt, 1, 0) == SQLITE_OK;
  Eigen::VectorXf vec;
  rc = SQLITE_DONE;
  while (ok && (rc = sqlite3_step (stmt)) == SQLITE_ROW)
    {
      const long long signal_id = sqlite3_column_int64 (stmt, 0);
      if (signal_id > mark)
        break;
      const void *blob = sqlite3_column_blob (stmt, 1);
      const int bytes = sqlite3_column_bytes (stmt, 1);
      if (signal_id <= 0 || !blob || bytes <= 0
          || bytes % static_cast<int> (sizeof (float)) != 0)
        continue;
      const int dim = bytes / static_cast<int> (sizeof (float));
      if (out.dim == 0)
        out.dim = dim;
      if (dim != out.dim)
        {
          out.read_through = std::max (out.read_through, signal_id);
          continue;
        }
      vec.resize (dim);
      std::memcpy (vec.data (), blob, static_cast<std::size_t> (bytes));
      Append (out, signal_id, vec);
    }
  ok = ok && (rc == SQLITE_DONE || rc == SQLITE_ROW);
  sqlite3_finalize (stmt);
  if (db)
    sqlite3_exec (db, "COMMIT", nullptr, nullptr, nullptr);
  sqlite3_close (db);
  if (ok)
    out.read_through = std::max (out.read_through, mark);
  if (!ok)
    {
      telemetry::AddCounter ("cortext.member_index.warm_failed_total", 1);
      return {};
    }
  telemetry::RecordHistogram (
      "cortext.member_index.warm_duration_ms",
      std::chrono::duration<double, std::milli> (
          std::chrono::steady_clock::now () - start)
          .count ());
  return out;
}

} // namespace

void
StartWarm (internal::MemberVectorIndex &index, const std::string &path)
{
  if (path.empty () || index.warm.valid () || index.read_through > 0)
    return;
  index.warm = std::async (std::launch::async, ReadAll, path);
}

std::size_t
Refresh (internal::MemberVectorIndex &index, Transaction &tx, int dim,
         double *warm_wait_ms)
{
  if (dim <= 0)
    return 0;
  if (index.warm.valid ())
    {
      const auto start = std::chrono::steady_clock::now ();
      auto warmed = index.warm.get ();
      if (warm_wait_ms)
        *warm_wait_ms = std::chrono::duration<double, std::milli> (
                            std::chrono::steady_clock::now () - start)
                            .count ();
      // Rows read since open are all past the warm read's snapshot, so the
      // refresh below continues where it ended.
      // A warm read with no members has no dimension of its own; its mark
      // still holds.
      if (warmed.signal_ids.empty () && warmed.read_through > 0)
        warmed.dim = dim;
      if (warmed.dim == dim && index.read_through == 0)
        static_cast<internal::MemberVectors &> (index) = std::move (warmed);
    }
  if (index.dim != dim)
    {
      index = internal::MemberVectorIndex ();
      index.dim = dim;
    }
  // Read the new mark first: rows above it can only be working rows.
  const auto top = tx.Execute (kTopLongTermRowSql, { index.read_through });
  if (top.empty ())
    return 0;
  const long long mark
      = store::AnyToLongLong (top.front ().at ("signal_id")).value_or (0);
  if (mark <= index.read_through)
    return 0;
  const auto rows = tx.Execute (kMemberRowsSql, { index.read_through });
  index.read_through = mark;
  Eigen::VectorXf vec;
  std::size_t added = 0;
  for (const auto &row : rows)
    {
      const auto id_it = row.find ("signal_id");
      const auto emb_it = row.find ("embedding");
      if (id_it == row.end () || emb_it == row.end ())
        continue;
      const long long signal_id
          = store::AnyToLongLong (id_it->second).value_or (0);
      if (signal_id <= 0 || signal_id > mark)
        continue;
      index.read_through = std::max (index.read_through, signal_id);
      if (!core::DecodeFloatBlob (emb_it->second, dim, vec))
        continue;
      const std::size_t before = index.signal_ids.size ();
      Append (index, signal_id, vec);
      added += index.signal_ids.size () - before;
    }
  return added;
}

std::vector<std::pair<long long, float> >
Search (const internal::MemberVectorIndex &index, const Eigen::VectorXf &query,
        std::size_t limit)
{
  std::vector<std::pair<long long, float> > out;
  const int dim = index.dim;
  if (limit == 0 || dim <= 0 || query.size () != dim
      || index.signal_ids.empty ())
    return out;
  const float norm = query.norm ();
  if (norm <= 0.0f)
    return out;
  // The query is quantized like the members, so each dot product is an
  // int32 sum the compiler vectorizes (a float sum may not be reordered).
  const Eigen::VectorXf q = query / norm;
  const float q_max = q.cwiseAbs ().maxCoeff ();
  if (q_max <= 0.0f)
    return out;
  const float q_scale = q_max / 127.0f;
  std::vector<std::int16_t> q_codes (static_cast<std::size_t> (dim));
  for (int i = 0; i < dim; ++i)
    q_codes[static_cast<std::size_t> (i)] = static_cast<std::int16_t> (
        std::lround (std::clamp (q[i] / q_scale, -127.0f, 127.0f)));
  const std::int16_t *qc = q_codes.data ();
  // Min-heap of the best `limit` (score, row).
  using Item = std::pair<float, std::size_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item> > best;
  const std::int8_t *codes = index.codes.data ();
  for (std::size_t row = 0; row < index.signal_ids.size (); ++row)
    {
      const float scale = index.scales[row];
      if (scale <= 0.0f)
        continue;
      const std::int8_t *c = codes + row * static_cast<std::size_t> (dim);
      std::int32_t dot = 0;
      for (int i = 0; i < dim; ++i)
        dot += static_cast<std::int32_t> (qc[i]) * c[i];
      const float score = static_cast<float> (dot) * q_scale * scale;
      if (best.size () < limit)
        best.emplace (score, row);
      else if (score > best.top ().first)
        {
          best.pop ();
          best.emplace (score, row);
        }
    }
  out.reserve (best.size ());
  while (!best.empty ())
    {
      out.emplace_back (index.signal_ids[best.top ().second],
                        best.top ().first);
      best.pop ();
    }
  std::reverse (out.begin (), out.end ());
  return out;
}

void
Drop (internal::MemberVectorIndex &index, long long signal_id)
{
  const auto it = std::lower_bound (index.signal_ids.begin (),
                                    index.signal_ids.end (), signal_id);
  if (it != index.signal_ids.end () && *it == signal_id)
    index.scales[static_cast<std::size_t> (it - index.signal_ids.begin ())]
        = 0.0f;
}

} // namespace cortext::operations::member_vector_index_internal
