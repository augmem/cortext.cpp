#include "member_vector_index_internal.hpp"

#include "cortext/core/utils.hpp"
#include "cortext/store/utils.hpp"

#include <algorithm>
#include <cmath>
#include <queue>

namespace cortext::operations::member_vector_index_internal
{

std::size_t
Refresh (internal::MemberVectorIndex &index, Transaction &tx, int dim)
{
  if (dim <= 0)
    return 0;
  if (index.dim != dim)
    {
      index = internal::MemberVectorIndex ();
      index.dim = dim;
    }
  const auto rows = tx.Execute (
      "SELECT s.signal_id AS signal_id, v.embedding AS embedding "
      // CROSS JOIN keeps signals first: a range on its primary key, so a
      // refresh reads only rows stored since the last one.
      "FROM signals s "
      "CROSS JOIN memories m ON m.memory_id = s.memory_id "
      "CROSS JOIN memory_signal_vectors v ON v.blob_id = s.blob_id "
      "WHERE s.signal_id > ? AND m.kind = 'LONG_TERM' AND m.n_signals > 1 "
      "ORDER BY s.signal_id",
      { index.read_through });
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
      if (signal_id <= 0)
        continue;
      index.read_through = std::max (index.read_through, signal_id);
      if (!core::DecodeFloatBlob (emb_it->second, dim, vec))
        continue;
      const float norm = vec.norm ();
      if (norm <= 0.0f)
        continue;
      vec /= norm;
      const float max_abs = vec.cwiseAbs ().maxCoeff ();
      const float scale = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
      if (scale <= 0.0f)
        continue;
      index.signal_ids.push_back (signal_id);
      index.scales.push_back (scale);
      for (int i = 0; i < dim; ++i)
        index.codes.push_back (static_cast<std::int8_t> (
            std::lround (std::clamp (vec[i] / scale, -127.0f, 127.0f))));
      ++added;
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
