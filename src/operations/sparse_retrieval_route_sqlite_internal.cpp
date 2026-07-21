#include "sparse_retrieval_route_sqlite_internal.hpp"

#include "sparse_retrieval_route_internal.hpp"

#include <cortext/store/store.hpp>
#include <cortext/store/utils.hpp>

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cortext::operations::sparse_retrieval_route_sqlite_internal
{

namespace
{

constexpr int kSchemaVersion = 1;

struct NodeRow
{
  long long memory_id = 0;
  std::vector<float> embedding;
  int level = 0;
  std::vector<std::vector<long long>> links;
  bool active = false;
};

std::optional<std::vector<float>>
Normalize (const Eigen::VectorXf &embedding, int embedding_dim)
{
  if (embedding_dim <= 0 || embedding.size () != embedding_dim)
    return std::nullopt;
  double norm_squared = 0.0;
  std::vector<float> normalized (static_cast<std::size_t> (embedding_dim));
  for (Eigen::Index index = 0; index < embedding.size (); ++index)
    {
      const float value = embedding[index];
      if (!std::isfinite (value))
        return std::nullopt;
      normalized[static_cast<std::size_t> (index)] = value;
      norm_squared += static_cast<double> (value) * value;
    }
  if (!(norm_squared > 0.0) || !std::isfinite (norm_squared))
    return std::nullopt;
  const float inverse_norm
      = static_cast<float> (1.0 / std::sqrt (norm_squared));
  for (float &value : normalized)
    value *= inverse_norm;
  return normalized;
}

float
Distance (const std::vector<float> &left, const std::vector<float> &right)
{
  double dot = 0.0;
  for (std::size_t index = 0; index < left.size (); ++index)
    dot += static_cast<double> (left[index]) * right[index];
  return static_cast<float> (1.0 - dot);
}

int
DeterministicLevel (long long memory_id, std::size_t maximum_level)
{
  std::uint64_t value = static_cast<std::uint64_t> (memory_id)
                        + 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  value ^= value >> 31U;
  int level = 0;
  while (level < static_cast<int> (maximum_level)
         && (value & 0x3ULL) == 0)
    {
      ++level;
      value >>= 2U;
    }
  return level;
}

std::vector<unsigned char>
EncodeEmbedding (const Eigen::VectorXf &embedding)
{
  std::vector<unsigned char> bytes (
      static_cast<std::size_t> (embedding.size ()) * sizeof (float));
  if (!bytes.empty ())
    std::memcpy (bytes.data (), embedding.data (), bytes.size ());
  return bytes;
}

std::vector<unsigned char>
EncodeEmbedding (const std::vector<float> &embedding)
{
  std::vector<unsigned char> bytes (embedding.size () * sizeof (float));
  if (!bytes.empty ())
    std::memcpy (bytes.data (), embedding.data (), bytes.size ());
  return bytes;
}

std::optional<std::vector<float>>
DecodeEmbedding (const std::any &value, int embedding_dim)
{
  const auto bytes = store::BlobFromAny (value);
  if (embedding_dim <= 0
      || bytes.size ()
             != static_cast<std::size_t> (embedding_dim) * sizeof (float))
    return std::nullopt;
  std::vector<float> embedding (static_cast<std::size_t> (embedding_dim));
  std::memcpy (embedding.data (), bytes.data (), bytes.size ());
  for (const float component : embedding)
    if (!std::isfinite (component))
      return std::nullopt;
  return embedding;
}

void
AppendUint32 (std::vector<unsigned char> &bytes, std::uint32_t value)
{
  for (std::size_t byte = 0; byte < sizeof (value); ++byte)
    bytes.push_back (
        static_cast<unsigned char> ((value >> (byte * 8)) & 0xffU));
}

void
AppendInt64 (std::vector<unsigned char> &bytes, long long value)
{
  const auto encoded = static_cast<std::uint64_t> (value);
  for (std::size_t byte = 0; byte < sizeof (encoded); ++byte)
    bytes.push_back (static_cast<unsigned char> (
        (encoded >> (byte * 8)) & 0xffU));
}

std::optional<std::uint32_t>
ReadUint32 (const std::vector<unsigned char> &bytes, std::size_t &offset)
{
  if (bytes.size () - std::min (bytes.size (), offset)
      < sizeof (std::uint32_t))
    return std::nullopt;
  std::uint32_t value = 0;
  for (std::size_t byte = 0; byte < sizeof (value); ++byte)
    value |= static_cast<std::uint32_t> (bytes[offset++]) << (byte * 8);
  return value;
}

std::optional<long long>
ReadInt64 (const std::vector<unsigned char> &bytes, std::size_t &offset)
{
  if (bytes.size () - std::min (bytes.size (), offset)
      < sizeof (std::uint64_t))
    return std::nullopt;
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < sizeof (value); ++byte)
    value |= static_cast<std::uint64_t> (bytes[offset++]) << (byte * 8);
  return static_cast<long long> (value);
}

std::vector<unsigned char>
EncodeLinks (const std::vector<std::vector<long long>> &links)
{
  std::vector<unsigned char> bytes;
  AppendUint32 (bytes, static_cast<std::uint32_t> (links.size ()));
  for (const auto &level : links)
    {
      AppendUint32 (bytes, static_cast<std::uint32_t> (level.size ()));
      for (const long long memory_id : level)
        AppendInt64 (bytes, memory_id);
    }
  return bytes;
}

std::vector<unsigned char>
EncodeIdentityIds (const std::vector<long long> &memory_ids)
{
  std::vector<unsigned char> bytes;
  bytes.reserve (sizeof (std::uint32_t)
                 + memory_ids.size () * sizeof (std::uint64_t));
  AppendUint32 (bytes, static_cast<std::uint32_t> (memory_ids.size ()));
  for (const long long memory_id : memory_ids)
    AppendInt64 (bytes, memory_id);
  return bytes;
}

std::optional<std::vector<long long>>
DecodeIdentityIds (const std::any &value, std::size_t capacity)
{
  const auto bytes = store::BlobFromAny (value);
  if (bytes.empty ())
    return std::vector<long long>{};
  std::size_t offset = 0;
  const auto count = ReadUint32 (bytes, offset);
  if (!count || *count == 0 || *count > capacity)
    return std::nullopt;
  std::vector<long long> memory_ids;
  memory_ids.reserve (*count);
  std::unordered_set<long long> unique;
  unique.reserve (*count);
  for (std::uint32_t index = 0; index < *count; ++index)
    {
      const auto memory_id = ReadInt64 (bytes, offset);
      if (!memory_id || *memory_id <= 0 || !unique.insert (*memory_id).second)
        return std::nullopt;
      memory_ids.push_back (*memory_id);
    }
  if (offset != bytes.size ())
    return std::nullopt;
  return memory_ids;
}

std::optional<std::vector<std::vector<long long>>>
DecodeLinks (const std::any &value, int expected_level,
             const Parameters &parameters)
{
  if (expected_level < 0
      || static_cast<std::size_t> (expected_level) > parameters.maximum_level)
    return std::nullopt;
  const auto bytes = store::BlobFromAny (value);
  std::size_t offset = 0;
  const auto level_count = ReadUint32 (bytes, offset);
  if (!level_count
      || *level_count != static_cast<std::uint32_t> (expected_level + 1))
    return std::nullopt;
  std::vector<std::vector<long long>> links (*level_count);
  for (std::size_t level_index = 0; level_index < links.size ();
       ++level_index)
    {
      auto &level = links[level_index];
      const auto count = ReadUint32 (bytes, offset);
      const std::size_t link_limit
          = level_index == 0 ? parameters.row_addressed_level_zero_links
                             : parameters.row_addressed_neighbor_count;
      if (!count || *count > link_limit)
        return std::nullopt;
      level.reserve (*count);
      for (std::uint32_t index = 0; index < *count; ++index)
        {
          const auto memory_id = ReadInt64 (bytes, offset);
          if (!memory_id || *memory_id <= 0)
            return std::nullopt;
          level.push_back (*memory_id);
        }
    }
  if (offset != bytes.size ())
    return std::nullopt;
  return links;
}

std::string
Placeholders (std::size_t count)
{
  std::string result;
  for (std::size_t index = 0; index < count; ++index)
    {
      if (index != 0)
        result += ',';
      result += '?';
    }
  return result;
}

long long
Int64 (const std::map<std::string, std::any> &row, const std::string &key,
       long long fallback = 0)
{
  const auto value = row.find (key);
  return value == row.end ()
             ? fallback
             : store::AnyToLongLong (value->second).value_or (fallback);
}

std::optional<NodeRow>
DecodeNode (const std::map<std::string, std::any> &row, int embedding_dim,
            const Parameters &parameters)
{
  NodeRow node;
  node.memory_id = Int64 (row, "memory_id");
  node.level = static_cast<int> (Int64 (row, "level", -1));
  node.active = Int64 (row, "active", -1) == 1;
  const auto embedding = row.find ("embedding");
  const auto links = row.find ("links");
  if (node.memory_id <= 0 || embedding == row.end () || links == row.end ())
    return std::nullopt;
  auto decoded_embedding
      = DecodeEmbedding (embedding->second, embedding_dim);
  auto decoded_links = DecodeLinks (links->second, node.level, parameters);
  if (!decoded_embedding || !decoded_links)
    return std::nullopt;
  node.embedding = std::move (*decoded_embedding);
  node.links = std::move (*decoded_links);
  return node;
}

std::optional<std::vector<NodeRow>>
FetchNodes (Store &store, int embedding_dim,
            const Parameters &parameters,
            long long generation,
            const std::vector<long long> &memory_ids)
{
  if (memory_ids.empty ())
    return std::vector<NodeRow>{};
  std::vector<std::any> params;
  params.reserve (memory_ids.size () + 1);
  params.emplace_back (generation);
  for (const long long memory_id : memory_ids)
    params.emplace_back (memory_id);
  const auto rows = store.Execute (
      "SELECT memory_id, embedding, level, links, active "
      "FROM cortext_sparse_route_nodes WHERE generation = ? AND memory_id IN ("
          + Placeholders (memory_ids.size ()) + ")",
      params);
  std::vector<NodeRow> decoded;
  decoded.reserve (rows.size ());
  for (const auto &row : rows)
    {
      auto node = DecodeNode (row, embedding_dim, parameters);
      if (!node)
        return std::nullopt;
      decoded.push_back (std::move (*node));
    }
  return decoded;
}

} // namespace

struct Route::Impl
{
  Impl (Store &route_store, int dimension, long long route_entry_memory_id,
        int route_max_level, long long route_active_count,
        long long route_generation, Parameters route_parameters,
        bool route_building = false,
        long long route_build_cursor = 0)
      : store (&route_store), embedding_dim (dimension),
        entry_memory_id (route_entry_memory_id), max_level (route_max_level),
        active_count (route_active_count), generation (route_generation),
        building (route_building), build_cursor (route_build_cursor),
        parameters (std::move (route_parameters))
  {
    activation_search_ef = parameters.search_ef;
    activation_search_node_budget = parameters.search_node_budget;
  }

  Store *store = nullptr;
  int embedding_dim = 0;
  long long entry_memory_id = 0;
  int max_level = 0;
  long long activation_entry_memory_id = 0;
  int activation_entry_level = 0;
  long long active_count = 0;
  long long generation = 0;
  bool building = false;
  long long build_cursor = 0;
  Parameters parameters;
  mutable std::size_t activation_search_ef = 0;
  mutable std::size_t activation_search_node_budget = 0;
  std::unordered_map<long long, std::vector<float>> delta_embeddings;
  std::unordered_set<long long> removed;
  std::unordered_map<long long, std::shared_ptr<const NodeRow>> node_cache;
  std::deque<long long> node_cache_order;
  std::vector<float> activation_centroid;
  std::vector<long long> activation_identity_ids;
  mutable std::size_t last_search_node_rows = 0;
  mutable std::size_t last_activation_snapshot_rows = 0;
  mutable std::size_t last_activation_snapshot_cache_miss_rows = 0;
  mutable std::size_t last_activation_frontier_seed_count = 0;
  mutable std::size_t last_search_distance_evaluations = 0;
  mutable int last_search_failure_code = 0;
  mutable int last_seal_failure_code = 0;
  mutable std::mutex mutex;
};

Route::Route (std::unique_ptr<Impl> impl) : impl_ (std::move (impl)) {}

Route::~Route () = default;

std::shared_ptr<Route>
Route::Create (
    Store &store, int embedding_dim,
    const std::vector<std::pair<long long, Eigen::VectorXf>> &entries,
    const sparse_retrieval_route_internal::Route &hnsw_route,
    Parameters parameters)
{
  if (embedding_dim <= 0)
    return nullptr;
  try
    {
      const auto snapshot = hnsw_route.Snapshot ();
      if (!snapshot || snapshot->nodes.size () != entries.size ())
        return nullptr;
      std::unordered_set<long long> expected;
      expected.reserve (entries.size ());
      for (const auto &[memory_id, embedding] : entries)
        if (memory_id <= 0 || !Normalize (embedding, embedding_dim)
            || !expected.insert (memory_id).second)
          return nullptr;

      auto tx = store.Begin ();
      tx->Execute ("DELETE FROM cortext_sparse_route_nodes");
      tx->Execute ("DELETE FROM cortext_sparse_route_meta");
      tx->Execute ("DELETE FROM cortext_sparse_route_build");
      tx->Execute ("DELETE FROM cortext_sparse_route_dirty");
      for (const auto &node : snapshot->nodes)
        {
          if (!node.active || expected.erase (node.memory_id) != 1
              || node.embedding.size () != embedding_dim
              || node.level < 0
              || node.links.size ()
                     != static_cast<std::size_t> (node.level + 1))
            return nullptr;
          tx->Execute (
              "INSERT INTO cortext_sparse_route_nodes("
              "memory_id, embedding, level, links, active, generation) "
              "VALUES(?, ?, ?, ?, 1, 1)",
              { node.memory_id, EncodeEmbedding (node.embedding),
                static_cast<long long> (node.level), EncodeLinks (node.links) });
        }
      if (!expected.empty ())
        return nullptr;
      tx->Execute (
          "INSERT INTO cortext_sparse_route_meta("
          "singleton, schema_version, embedding_dim, entry_memory_id, "
          "max_level, active_count, generation) VALUES(1, ?, ?, ?, ?, ?, 1)",
          { static_cast<long long> (kSchemaVersion),
            static_cast<long long> (embedding_dim),
            snapshot->entry_memory_id,
            static_cast<long long> (snapshot->max_level),
            static_cast<long long> (snapshot->nodes.size ()) });
      tx->Commit ();
      return std::shared_ptr<Route> (new Route (std::make_unique<Impl> (
          store, embedding_dim, snapshot->entry_memory_id,
          snapshot->max_level,
          static_cast<long long> (snapshot->nodes.size ()), 1,
          std::move (parameters))));
    }
  catch (...)
    {
      return nullptr;
    }
}

std::shared_ptr<Route>
Route::Open (Store &store, int embedding_dim, Parameters parameters)
{
  if (embedding_dim <= 0)
    return nullptr;
  try
    {
      const auto meta = store.Execute (
          "SELECT meta.schema_version, meta.embedding_dim, "
          "meta.entry_memory_id, meta.max_level, meta.active_count, "
          "meta.generation, meta.activation_entry_memory_id, "
          "activation_generation, COALESCE(activation_centroid, X'') AS "
          "activation_centroid, COALESCE(activation_identity_ids, X'') AS "
          "activation_identity_ids, "
          "(SELECT active FROM cortext_sparse_route_nodes entry "
          " WHERE entry.memory_id = meta.entry_memory_id "
          " AND entry.generation = meta.generation) AS entry_active, "
          "(SELECT level FROM cortext_sparse_route_nodes entry "
          " WHERE entry.memory_id = meta.entry_memory_id "
          " AND entry.generation = meta.generation) AS entry_level "
          "FROM cortext_sparse_route_meta meta WHERE singleton = 1");
      if (meta.size () != 1
          || Int64 (meta.front (), "schema_version") != kSchemaVersion
          || Int64 (meta.front (), "embedding_dim") != embedding_dim)
        return nullptr;
      const long long entry_memory_id
          = Int64 (meta.front (), "entry_memory_id", -1);
      const long long max_level = Int64 (meta.front (), "max_level", -1);
      const long long active_count
          = Int64 (meta.front (), "active_count", -1);
      const long long generation
          = Int64 (meta.front (), "generation", -1);
      const long long activation_entry
          = Int64 (meta.front (), "activation_entry_memory_id", -1);
      const long long activation_generation
          = Int64 (meta.front (), "activation_generation", -1);
      const long long entry_active
          = Int64 (meta.front (), "entry_active", -1);
      const long long entry_level
          = Int64 (meta.front (), "entry_level", -1);
      const auto activation_centroid_bytes
          = store::BlobFromAny (meta.front ().at ("activation_centroid"));
      const auto activation_centroid
          = activation_centroid_bytes.empty ()
                ? std::optional<std::vector<float>>{}
                : DecodeEmbedding (meta.front ().at ("activation_centroid"),
                                   embedding_dim);
      const auto activation_identity_bytes
          = store::BlobFromAny (meta.front ().at ("activation_identity_ids"));
      const auto activation_identity_ids = DecodeIdentityIds (
          meta.front ().at ("activation_identity_ids"),
          parameters.activation_snapshot_capacity);
      if (entry_memory_id < 0 || max_level < 0
          || max_level > static_cast<long long> (parameters.maximum_level)
          || active_count < 0 || generation < 0 || activation_entry < 0
          || activation_generation < 0
          || (active_count == 0
              && (entry_memory_id != 0 || max_level != 0))
          || (active_count > 0
              && (entry_memory_id == 0 || entry_active != 1
                  || entry_level != max_level)))
        return nullptr;
      const bool has_activation = activation_entry > 0;
      if (has_activation
          && (activation_generation != generation || !activation_centroid
              || !activation_identity_ids || activation_identity_ids->empty ()
              || activation_identity_ids->front () != activation_entry))
        return nullptr;
      if (!has_activation
          && (activation_generation != 0
              || !activation_centroid_bytes.empty ()
              || !activation_identity_bytes.empty ()))
        return nullptr;
      auto impl = std::make_unique<Impl> (
          store, embedding_dim, entry_memory_id, static_cast<int> (max_level),
          active_count, generation, std::move (parameters));
      if (has_activation)
        {
          const auto activation_rows = FetchNodes (
              store, embedding_dim, impl->parameters, generation,
              *activation_identity_ids);
          if (!activation_rows
              || activation_rows->size () != activation_identity_ids->size ()
              || std::any_of (
                  activation_rows->begin (), activation_rows->end (),
                  [] (const NodeRow &row) { return !row.active; }))
            return nullptr;
          const auto entry = std::find_if (
              activation_rows->begin (), activation_rows->end (),
              [activation_entry] (const NodeRow &row) {
                return row.memory_id == activation_entry;
              });
          if (entry == activation_rows->end ())
            return nullptr;
          impl->activation_entry_memory_id = activation_entry;
          impl->activation_entry_level = entry->level;
          impl->activation_centroid = *activation_centroid;
          impl->activation_identity_ids = *activation_identity_ids;
        }
      return std::shared_ptr<Route> (new Route (std::move (impl)));
    }
  catch (...)
    {
      return nullptr;
    }
}

std::shared_ptr<Route>
Route::OpenOrBeginBuild (Store &store, int embedding_dim,
                         Parameters parameters)
{
  if (embedding_dim <= 0)
    return nullptr;
  try
    {
      const auto build = store.Execute (
          "SELECT schema_version, embedding_dim, cursor_memory_id, "
          "entry_memory_id, max_level, active_count, generation "
          "FROM cortext_sparse_route_build WHERE singleton = 1");
      if (build.size () == 1
          && Int64 (build.front (), "schema_version") == kSchemaVersion
          && Int64 (build.front (), "embedding_dim") == embedding_dim)
        {
          const long long cursor
              = Int64 (build.front (), "cursor_memory_id", -1);
          const long long entry
              = Int64 (build.front (), "entry_memory_id", -1);
          const long long max_level
              = Int64 (build.front (), "max_level", -1);
          const long long active_count
              = Int64 (build.front (), "active_count", -1);
          const long long generation
              = Int64 (build.front (), "generation", -1);
          if (cursor >= 0 && entry >= 0 && max_level >= 0
              && max_level
                     <= static_cast<long long> (parameters.maximum_level)
              && active_count >= 0 && generation > 0)
            return std::shared_ptr<Route> (new Route (
                std::make_unique<Impl> (
                    store, embedding_dim, entry,
                    static_cast<int> (max_level), active_count, generation,
                    parameters, true, cursor)));
        }

      // A missing build record means the prior partial generation was not
      // sealed before a crash. Start a new isolated generation; stale rows
      // remain unreachable because every fetch is generation-qualified.
      const auto generation_rows = store.Execute (
          "SELECT COALESCE(MAX(generation), 0) + 1 AS generation "
          "FROM cortext_sparse_route_nodes");
      const long long generation
          = generation_rows.empty ()
                ? 1
                : std::max (1LL, Int64 (generation_rows.front (),
                                         "generation", 1));
      auto tx = store.Begin ();
      tx->Execute ("DELETE FROM cortext_sparse_route_meta");
      tx->Execute ("DELETE FROM cortext_sparse_route_build");
      tx->Execute ("DELETE FROM cortext_sparse_route_dirty");
      tx->Execute (
          "INSERT INTO cortext_sparse_route_build("
          "singleton, schema_version, embedding_dim, cursor_memory_id, "
          "entry_memory_id, max_level, active_count, generation) "
          "VALUES(1, ?, ?, 0, 0, 0, 0, ?)",
          { static_cast<long long> (kSchemaVersion),
            static_cast<long long> (embedding_dim), generation });
      tx->Commit ();
      return std::shared_ptr<Route> (new Route (std::make_unique<Impl> (
          store, embedding_dim, 0, 0, 0, generation,
          std::move (parameters), true, 0)));
    }
  catch (...)
    {
      return nullptr;
    }
}

bool
Route::Upsert (long long memory_id, const Eigen::VectorXf &embedding)
{
  if (!impl_ || !impl_->store || memory_id <= 0)
    return false;
  auto normalized = Normalize (embedding, impl_->embedding_dim);
  if (!normalized)
    return false;
  // This statement joins the caller's authoritative signal transaction when
  // one is active. The durable identity journal lets a restart restage the
  // latest value from the authoritative current surface without rebuilding
  // the unchanged graph or reopening stale rows as current.
  try
    {
      impl_->store->Execute (
          "INSERT OR IGNORE INTO cortext_sparse_route_dirty(memory_id) "
          "VALUES(?)",
          { memory_id });
      // Live changes stay only in SQLite until a consolidation edge stages a
      // knob-bounded slice. Retrieval falls back to the exact SQL surface
      // while any journal row remains, so correctness does not require an
      // unbounded in-memory delta on either active or unpublished routes.
      return true;
    }
  catch (...)
    {
      return false;
    }
}

bool
Route::Remove (long long memory_id)
{
  if (!impl_ || !impl_->store || memory_id <= 0)
    return false;
  try
    {
      impl_->store->Execute (
          "INSERT OR IGNORE INTO cortext_sparse_route_dirty(memory_id) "
          "VALUES(?)",
          { memory_id });
      return true;
    }
  catch (...)
    {
      return false;
    }
}

bool
Route::StagePendingUpsert (long long memory_id,
                           const Eigen::VectorXf &embedding)
{
  if (!impl_ || memory_id <= 0)
    return false;
  auto normalized = Normalize (embedding, impl_->embedding_dim);
  if (!normalized)
    return false;
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      impl_->removed.erase (memory_id);
      impl_->delta_embeddings[memory_id] = std::move (*normalized);
      return true;
    }
  catch (...)
    {
      return false;
    }
}

bool
Route::StagePendingRemove (long long memory_id)
{
  if (!impl_ || memory_id <= 0)
    return false;
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      impl_->delta_embeddings.erase (memory_id);
      impl_->removed.insert (memory_id);
      return true;
    }
  catch (...)
    {
      return false;
    }
}

std::optional<std::vector<long long>>
Route::Search (const Eigen::VectorXf &query, std::size_t route_capacity) const
{
  return SearchWithEnvelope (query, route_capacity, false, false);
}

std::optional<std::vector<long long>>
Route::SearchActivated (const Eigen::VectorXf &query) const
{
  return SearchWithEnvelope (query, impl_ ? impl_->parameters.route_capacity
                                         : 0,
                             false, true);
}

std::optional<std::vector<long long>>
Route::SearchWithEnvelope (const Eigen::VectorXf &query,
                           std::size_t route_capacity,
                           bool construction_search,
                           bool return_all_activated,
                           bool rebuild_activation) const
{
  if (!impl_ || !impl_->store || route_capacity == 0)
    return std::nullopt;
  route_capacity = std::min (route_capacity,
                             impl_->parameters.route_capacity);
  auto normalized = Normalize (query, impl_->embedding_dim);
  if (!normalized)
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      impl_->last_search_failure_code = 1;
      return std::nullopt;
    }
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      impl_->last_search_failure_code = 0;
      impl_->last_search_node_rows = 0;
      impl_->last_activation_snapshot_rows = 0;
      impl_->last_activation_snapshot_cache_miss_rows = 0;
      impl_->last_activation_frontier_seed_count = 0;
      impl_->last_search_distance_evaluations = 0;
      const bool use_construction_envelope
          = impl_->building || construction_search;
      const std::size_t search_node_budget
          = use_construction_envelope
                ? impl_->parameters.backfill_search_node_budget
                : impl_->activation_search_node_budget;
      const std::size_t search_effort
          = use_construction_envelope
                ? impl_->parameters.backfill_search_ef
                : impl_->activation_search_ef;
      if (impl_->delta_embeddings.size () > search_node_budget)
        {
          impl_->last_search_failure_code = 2;
          return std::nullopt;
        }
      const std::size_t persisted_node_budget
          = use_construction_envelope
                ? search_node_budget
                : search_node_budget - impl_->delta_embeddings.size ();
      const std::size_t activation_identity_target
          = use_construction_envelope
                ? search_node_budget
                : impl_->parameters.activation_identity_target;
      const std::size_t node_fetch_budget = persisted_node_budget;
      struct QueryNode
      {
        std::shared_ptr<const NodeRow> row;
        std::optional<float> distance;
      };
      std::unordered_map<long long, QueryNode> cache;
      cache.reserve (node_fetch_budget + activation_identity_target);
      std::vector<std::shared_ptr<const NodeRow>> activation_nodes;
      if (return_all_activated && !use_construction_envelope
          && !rebuild_activation && !impl_->activation_identity_ids.empty ())
        {
          std::vector<long long> missing;
          missing.reserve (impl_->activation_identity_ids.size ());
          activation_nodes.reserve (impl_->activation_identity_ids.size ());
          for (const long long memory_id : impl_->activation_identity_ids)
            {
              const auto cached = impl_->node_cache.find (memory_id);
              if (cached == impl_->node_cache.end ())
                missing.push_back (memory_id);
              else
                {
                  activation_nodes.push_back (cached->second);
                  cache.emplace (memory_id,
                                 QueryNode{ cached->second, {} });
                }
            }
          const auto fetched = FetchNodes (
              *impl_->store, impl_->embedding_dim, impl_->parameters,
              impl_->generation, missing);
          if (!fetched)
            {
              impl_->last_search_failure_code = 10;
              return std::nullopt;
            }
          impl_->last_activation_snapshot_cache_miss_rows
              = fetched->size ();
          for (auto node : *fetched)
            {
              const long long memory_id = node.memory_id;
              auto shared
                  = std::make_shared<const NodeRow> (std::move (node));
              impl_->node_cache[memory_id] = shared;
              impl_->node_cache_order.push_back (memory_id);
              activation_nodes.push_back (shared);
              cache.emplace (memory_id, QueryNode{ std::move (shared), {} });
            }
          impl_->last_activation_snapshot_rows = activation_nodes.size ();
          if (impl_->last_activation_snapshot_rows
                  > impl_->parameters.activation_snapshot_capacity
              || impl_->last_activation_snapshot_rows
                     != impl_->activation_identity_ids.size ())
            {
              impl_->last_search_failure_code = 11;
              return std::nullopt;
            }
          while (impl_->node_cache.size ()
                     > impl_->parameters.shadow_node_capacity
                 && !impl_->node_cache_order.empty ())
            {
              const long long oldest = impl_->node_cache_order.front ();
              impl_->node_cache_order.pop_front ();
              impl_->node_cache.erase (oldest);
            }
        }
      std::unordered_set<long long> canonical_ids;
      canonical_ids.reserve (node_fetch_budget);
      std::size_t canonical_node_rows = 0;
      auto persisted_query_distance = [&] (const long long memory_id) {
        auto &node = cache.at (memory_id);
        if (node.distance)
          return *node.distance;
        node.distance = Distance (*normalized, node.row->embedding);
        ++impl_->last_search_distance_evaluations;
        return *node.distance;
      };
      auto delta_query_distance = [&] (const std::vector<float> &embedding) {
        const float distance = Distance (*normalized, embedding);
        ++impl_->last_search_distance_evaluations;
        return distance;
      };
      auto fetch = [&] (std::vector<long long> memory_ids) -> bool {
        std::sort (memory_ids.begin (), memory_ids.end ());
        memory_ids.erase (std::unique (memory_ids.begin (), memory_ids.end ()),
                          memory_ids.end ());
        memory_ids.erase (
            std::remove_if (memory_ids.begin (), memory_ids.end (),
                            [&] (const long long memory_id) {
                              return cache.count (memory_id) != 0;
                            }),
            memory_ids.end ());
        std::vector<long long> missing;
        missing.reserve (memory_ids.size ());
        for (const long long memory_id : memory_ids)
          {
            if (canonical_node_rows + missing.size () >= node_fetch_budget)
              break;
            const auto cached = impl_->node_cache.find (memory_id);
            if (cached == impl_->node_cache.end ())
              missing.push_back (memory_id);
            else
              {
                cache.emplace (memory_id,
                               QueryNode{ cached->second, {} });
                canonical_ids.insert (memory_id);
                ++canonical_node_rows;
              }
          }
        const std::size_t remaining
            = node_fetch_budget - canonical_node_rows;
        if (missing.size () > remaining)
          missing.resize (remaining);
        const auto nodes
            = FetchNodes (*impl_->store, impl_->embedding_dim,
                          impl_->parameters, impl_->generation, missing);
        // Links can point into an older unpublished generation while a
        // bounded rebuild is in progress. Those nodes are simply not part of
        // this generation's traversal surface.
        if (!nodes)
          {
            impl_->last_search_failure_code = 3;
            return false;
          }
        for (auto node : *nodes)
          {
            const long long memory_id = node.memory_id;
            auto shared
                = std::make_shared<const NodeRow> (std::move (node));
            impl_->node_cache[memory_id] = shared;
            impl_->node_cache_order.push_back (memory_id);
            cache.emplace (memory_id, QueryNode{ std::move (shared), {} });
            canonical_ids.insert (memory_id);
            ++canonical_node_rows;
          }
        while (impl_->node_cache.size ()
                   > impl_->parameters.shadow_node_capacity
               && !impl_->node_cache_order.empty ())
          {
            const long long oldest = impl_->node_cache_order.front ();
            impl_->node_cache_order.pop_front ();
            impl_->node_cache.erase (oldest);
          }
        impl_->last_search_node_rows = canonical_node_rows;
        return true;
      };

      // Every query retains the canonical maximum-level entry and uses the
      // current F/S/T-derived HNSW envelope: 8C after consolidation, ramping
      // by R to 9C. The independently A-bounded consolidation snapshot owns
      // level-zero fill locality without replacing canonical hierarchy descent.
      long long current_id = impl_->entry_memory_id;
      const long long activation_entry_id
          = !use_construction_envelope && !rebuild_activation
                    && impl_->activation_entry_memory_id > 0
                ? impl_->activation_entry_memory_id
                : 0;
      if (current_id > 0
          && !fetch ({ current_id, activation_entry_id }))
        {
          impl_->last_search_failure_code = 4;
          return std::nullopt;
        }
      if (current_id > 0 && cache.count (current_id) == 0)
        {
          impl_->last_search_failure_code = 5;
          return std::nullopt;
        }
      float current_distance
          = current_id > 0
                ? persisted_query_distance (current_id)
                : std::numeric_limits<float>::infinity ();
      for (int level = impl_->max_level; level > 0; --level)
        {
          bool improved = true;
          std::size_t hops = 0;
          while (improved
                 && hops++ < impl_->parameters.hnsw.query_effort)
            {
              improved = false;
              const auto current = cache.find (current_id);
              if (current == cache.end ()
                  || current->second.row->level < level)
                break;
              const auto &neighbors
                  = current->second.row
                        ->links[static_cast<std::size_t> (level)];
              if (!fetch (neighbors))
                {
                  impl_->last_search_failure_code = 6;
                  return std::nullopt;
                }
              for (const long long neighbor_id : neighbors)
                {
                  const auto neighbor = cache.find (neighbor_id);
                  if (neighbor == cache.end ())
                    continue;
                  const float distance
                      = persisted_query_distance (neighbor_id);
                  if (distance < current_distance
                      || (distance == current_distance
                          && neighbor_id < current_id))
                    {
                      current_distance = distance;
                      current_id = neighbor_id;
                      improved = true;
                    }
                }
            }
        }

      using Ranked = std::pair<float, long long>;
      const std::size_t best_capacity
          = return_all_activated ? node_fetch_budget : route_capacity;
      std::priority_queue<Ranked, std::vector<Ranked>, std::greater<Ranked>>
          frontier;
      std::priority_queue<Ranked> best;
      std::unordered_set<long long> visited;
      visited.reserve (search_node_budget + activation_nodes.size ());
      auto seed_frontier = [&] (const long long memory_id) {
        if (memory_id <= 0 || !visited.insert (memory_id).second)
          return false;
        const auto node = cache.find (memory_id);
        if (node == cache.end ())
          return false;
        const float distance = persisted_query_distance (memory_id);
        frontier.emplace (distance, memory_id);
        if (node->second.row->active
            && impl_->removed.count (memory_id) == 0
            && impl_->delta_embeddings.count (memory_id) == 0)
          best.emplace (distance, memory_id);
        return true;
      };
      seed_frontier (current_id);
      // Consolidation persists an A-bounded activation snapshot in SQLite.
      // Treat its identities as deterministic level-zero entry points so the
      // changed centroid actually steers the next sparse walk; their rows are
      // charged to the separate A envelope and never consume the current
      // 8C-to-9C traversal budget.
      if (!rebuild_activation)
        for (const long long memory_id : impl_->activation_identity_ids)
          if (seed_frontier (memory_id))
            ++impl_->last_activation_frontier_seed_count;

      std::size_t expanded_count = 0;
      while (!frontier.empty () && canonical_node_rows < node_fetch_budget
             && expanded_count < search_effort)
        {
          if (best.size () >= best_capacity
              && frontier.top ().first > best.top ().first)
            break;
          std::vector<Ranked> expansion;
          expansion.reserve (impl_->parameters.search_expansion_batch);
          while (!frontier.empty ()
                 && expansion.size ()
                        < impl_->parameters.search_expansion_batch
                 && expanded_count + expansion.size () < search_effort)
            {
              if (best.size () >= best_capacity
                  && frontier.top ().first > best.top ().first)
                break;
              expansion.push_back (frontier.top ());
              frontier.pop ();
            }
          expanded_count += expansion.size ();
          std::vector<long long> neighbor_ids;
          for (const auto &[distance, memory_id] : expansion)
            {
              (void)distance;
              const auto node = cache.find (memory_id);
              if (node == cache.end ()
                  || node->second.row->links.empty ())
                continue;
              for (const long long neighbor_id
                   : node->second.row->links.front ())
                if (visited.insert (neighbor_id).second)
                  neighbor_ids.push_back (neighbor_id);
            }
          if (!fetch (neighbor_ids))
            {
              impl_->last_search_failure_code = 7;
              return std::nullopt;
            }
          for (const long long memory_id : neighbor_ids)
            {
              const auto node = cache.find (memory_id);
              if (node == cache.end ())
                continue;
              const float distance = persisted_query_distance (memory_id);
              if (best.size () < best_capacity
                  || distance <= best.top ().first)
                frontier.emplace (distance, memory_id);
              if (node->second.row->active
                  && impl_->removed.count (memory_id) == 0
                  && impl_->delta_embeddings.count (memory_id) == 0
                  && (best.size () < best_capacity
                      || distance <= best.top ().first))
                {
                  best.emplace (distance, memory_id);
                  if (best.size () > best_capacity)
                    best.pop ();
                }
            }
        }

      // HNSW's distance stop can exhaust a locally reachable frontier before
      // the knob-derived node envelope, including when an otherwise-current
      // node has lost every reciprocal incoming edge. Complete the routing
      // envelope with a deterministic, indexed slice of the same authoritative
      // SQLite graph. This is not a second retrieval path: the rows join the
      // same exact rerank below, fetch() still enforces the per-query ceiling,
      // and only activation_identity_target identities leave this route.
      const std::size_t minimum_persisted_rows
          = return_all_activated
                ? node_fetch_budget
                : std::min (
                      node_fetch_budget,
                      route_capacity + impl_->removed.size ()
                                  > impl_->delta_embeddings.size ()
                              ? route_capacity + impl_->removed.size ()
                                    - impl_->delta_embeddings.size ()
                              : 0);
      auto fill_ordered_generation_slice
          = [&] (const char *comparison, long long pivot) -> bool {
        if (canonical_node_rows >= minimum_persisted_rows)
          return true;
        const auto rows = impl_->store->Execute (
            std::string (
                "SELECT memory_id FROM cortext_sparse_route_nodes "
                "WHERE generation = ? AND active = 1 AND memory_id ")
                + comparison
                + " ? ORDER BY memory_id LIMIT ?",
            { impl_->generation, pivot,
              static_cast<long long> (minimum_persisted_rows) });
        std::vector<long long> memory_ids;
        memory_ids.reserve (rows.size ());
        for (const auto &row : rows)
          {
            const long long memory_id = Int64 (row, "memory_id", 0);
            if (memory_id <= 0)
              return false;
            memory_ids.push_back (memory_id);
          }
        return fetch (std::move (memory_ids));
      };
      const long long fill_pivot
          = activation_entry_id > 0 ? activation_entry_id : current_id;
      if (fill_pivot > 0
          && (!fill_ordered_generation_slice (">=", fill_pivot)
              || !fill_ordered_generation_slice ("<", fill_pivot)))
        {
          impl_->last_search_failure_code = 9;
          return std::nullopt;
        }

      std::vector<Ranked> ranked;
      ranked.reserve (cache.size () + impl_->delta_embeddings.size ());
      for (const auto &[memory_id, node] : cache)
        if (canonical_ids.count (memory_id) != 0 && node.row->active
            && impl_->removed.count (memory_id) == 0
            && impl_->delta_embeddings.count (memory_id) == 0)
          ranked.emplace_back (persisted_query_distance (memory_id),
                               memory_id);
      for (const auto &[memory_id, embedding] : impl_->delta_embeddings)
        ranked.emplace_back (delta_query_distance (embedding), memory_id);
      std::sort (ranked.begin (), ranked.end ());
      std::vector<Ranked> protected_canonical;
      if (return_all_activated && !use_construction_envelope
          && !rebuild_activation)
        {
          const std::size_t protected_count
              = std::min (ranked.size (),
                          impl_->parameters.family_exact_comparison_limit);
          protected_canonical.assign (
              ranked.begin (), ranked.begin () + protected_count);
        }
      // A successful consolidation snapshots at most A identities selected
      // under the current 8C-to-9C route. Querying that fixed snapshot is an
      // independent A-bounded exact layer. The 2C canonical semantic core is
      // protected, while an R=max(2,B/16) portion of A may move with the
      // consolidation snapshot before the final exact rerank.
      if (return_all_activated && !use_construction_envelope
          && !rebuild_activation && !impl_->activation_identity_ids.empty ())
        {
          std::unordered_set<long long> already_ranked;
          already_ranked.reserve (ranked.size ());
          for (const auto &[distance, memory_id] : ranked)
            {
              (void)distance;
              already_ranked.insert (memory_id);
            }
          for (const long long memory_id : impl_->activation_identity_ids)
            {
              const auto node = cache.find (memory_id);
              if (node != cache.end () && node->second.row->active
                  && impl_->removed.count (memory_id) == 0
                  && impl_->delta_embeddings.count (memory_id) == 0
                  && already_ranked.insert (memory_id).second)
              {
                ranked.emplace_back (persisted_query_distance (memory_id),
                                     memory_id);
              }
            }
        }
      std::sort (ranked.begin (), ranked.end ());
      const std::size_t result_limit
          = return_all_activated ? activation_identity_target
                                 : route_capacity;
      if (!protected_canonical.empty () && ranked.size () > result_limit)
        {
          std::unordered_set<long long> selected_ids;
          selected_ids.reserve (result_limit);
          std::vector<Ranked> selected;
          selected.reserve (result_limit);
          for (const auto &item : protected_canonical)
            if (selected_ids.insert (item.second).second)
              selected.push_back (item);
          // A is 2C + 2B. Preserve the exact 2C query core, then reserve only
          // the independently knob-derived B/16 reciprocal-update budget from
          // the consolidation-centroid snapshot before filling the remainder
          // by exact query rank. This lets consolidation move the activated
          // neighborhood without allowing the centroid to displace a whole
          // backfill batch of query-relevant identities.
          const std::size_t centroid_quota = std::min (
              impl_->parameters.reciprocal_update_count,
              result_limit - selected.size ());
          std::size_t centroid_selected = 0;
          for (const long long memory_id : impl_->activation_identity_ids)
            {
              if (centroid_selected >= centroid_quota)
                break;
              const auto node = cache.find (memory_id);
              if (node == cache.end () || !node->second.row->active
                  || impl_->removed.count (memory_id) != 0
                  || impl_->delta_embeddings.count (memory_id) != 0
                  || !selected_ids.insert (memory_id).second)
                continue;
              selected.emplace_back (persisted_query_distance (memory_id),
                                     memory_id);
              ++centroid_selected;
            }
          for (const auto &item : ranked)
            if (selected.size () < result_limit
                && selected_ids.insert (item.second).second)
              selected.push_back (item);
          std::sort (selected.begin (), selected.end ());
          ranked = std::move (selected);
        }
      else if (ranked.size () > result_limit)
        ranked.resize (result_limit);
      std::vector<long long> result;
      result.reserve (ranked.size ());
      for (const auto &[distance, memory_id] : ranked)
        {
          (void)distance;
          result.push_back (memory_id);
        }
      if (!impl_->building && !construction_search && !rebuild_activation)
        {
          impl_->activation_search_ef = std::min (
              impl_->parameters.search_ef,
              impl_->activation_search_ef
                  + impl_->parameters.activation_search_ef_step);
          impl_->activation_search_node_budget = std::min (
              impl_->parameters.search_node_budget,
              impl_->activation_search_node_budget
                  + impl_->parameters.activation_search_node_budget_step);
        }
      return result;
    }
  catch (...)
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      impl_->last_search_failure_code = 8;
      return std::nullopt;
    }
}

bool
Route::Recenter (const Eigen::VectorXf &consolidation_embedding)
{
  if (!impl_ || !impl_->store)
    return false;
  const auto activated = SearchWithEnvelope (
      consolidation_embedding, impl_->parameters.route_capacity, false, true,
      true);
  if (!activated || activated->empty ())
    return ActiveCount () == 0;
  const auto normalized
      = Normalize (consolidation_embedding, impl_->embedding_dim);
  if (!normalized)
    return false;
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      const auto rows = FetchNodes (
          *impl_->store, impl_->embedding_dim, impl_->parameters,
          impl_->generation, { activated->front () });
      if (!rows || rows->size () != 1 || !rows->front ().active)
        return false;
      auto tx = impl_->store->Begin ();
      tx->Execute (
          "UPDATE cortext_sparse_route_meta SET "
          "activation_entry_memory_id = ?, activation_generation = ?, "
          "activation_centroid = ?, activation_identity_ids = ? "
          "WHERE singleton = 1 AND generation = ?",
          { rows->front ().memory_id, impl_->generation,
            EncodeEmbedding (*normalized), EncodeIdentityIds (*activated),
            impl_->generation });
      tx->Commit ();
      impl_->activation_entry_memory_id = rows->front ().memory_id;
      impl_->activation_entry_level = rows->front ().level;
      impl_->activation_centroid = *normalized;
      impl_->activation_identity_ids = *activated;
      impl_->activation_search_ef
          = impl_->parameters.activation_search_ef_min;
      impl_->activation_search_node_budget
          = impl_->parameters.activation_search_node_budget_min;
      return true;
    }
  catch (...)
    {
      return false;
    }
}

bool
Route::Seal (const sparse_retrieval_route_internal::Route *hnsw_route)
{
  if (!impl_ || !impl_->store)
    return false;
  std::unordered_map<long long, std::vector<long long>> neighbor_proposals;
  if (!hnsw_route)
    {
      impl_->last_seal_failure_code = 10;
      std::vector<std::pair<long long, Eigen::VectorXf>> queries;
      {
        std::lock_guard<std::mutex> lock (impl_->mutex);
        if (impl_->delta_embeddings.empty () && impl_->removed.empty ())
          return true;
        queries.reserve (impl_->delta_embeddings.size ());
        for (const auto &[memory_id, embedding] : impl_->delta_embeddings)
          {
            Eigen::VectorXf query (
                static_cast<Eigen::Index> (embedding.size ()));
            std::copy (embedding.begin (), embedding.end (), query.data ());
            queries.emplace_back (memory_id, std::move (query));
          }
      }
      for (const auto &[memory_id, query] : queries)
        {
          auto candidates = SearchWithEnvelope (
              query, impl_->parameters.row_addressed_neighbor_count
                         + queries.size (),
              true, false);
          if (!candidates)
            {
              std::lock_guard<std::mutex> lock (impl_->mutex);
              impl_->last_seal_failure_code
                  = 100 + impl_->last_search_failure_code;
              return false;
            }
          candidates->erase (
              std::remove (candidates->begin (), candidates->end (),
                           memory_id),
              candidates->end ());
          neighbor_proposals.emplace (memory_id, std::move (*candidates));
        }
      impl_->last_seal_failure_code = 0;
    }
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      impl_->last_seal_failure_code = 0;
      if (impl_->delta_embeddings.empty () && impl_->removed.empty ())
        return true;
      // Metadata and the durable dirty identities remain visible until the
      // row updates, refreshed metadata, and journal clear commit atomically.
      // An interrupted seal can therefore restage only the committed delta.

      std::vector<long long> roots;
      roots.reserve (impl_->delta_embeddings.size () + impl_->removed.size ());
      for (const auto &[memory_id, embedding] : impl_->delta_embeddings)
        {
          (void)embedding;
          roots.push_back (memory_id);
        }
      roots.insert (roots.end (), impl_->removed.begin (), impl_->removed.end ());
      std::sort (roots.begin (), roots.end ());
      roots.erase (std::unique (roots.begin (), roots.end ()), roots.end ());
      if (!hnsw_route)
        {
          impl_->last_seal_failure_code = 60;
          const auto prior_rows
              = FetchNodes (*impl_->store, impl_->embedding_dim,
                            impl_->parameters, impl_->generation, roots);
          if (!prior_rows)
            return false;
          std::unordered_map<long long, NodeRow> prior_by_id;
          prior_by_id.reserve (prior_rows->size ());
          for (auto node : *prior_rows)
            prior_by_id.emplace (node.memory_id, std::move (node));

          // Existing adjacency is derived from the stored embedding. Keeping
          // those links after an embedding moves can make the row-addressed
          // route silently miss its exact target. Invalidate the generation
          // and let the bounded backfill rebuild it instead of publishing a
          // mixed old-topology/new-vector snapshot.
          bool topology_requires_rebuild = false;
          for (const auto &[memory_id, embedding] : impl_->delta_embeddings)
            {
              const auto prior = prior_by_id.find (memory_id);
              if (prior == prior_by_id.end () || !prior->second.active)
                continue;
              constexpr float kEmbeddingMovementEpsilon = 1e-6f;
              topology_requires_rebuild
                  = prior->second.embedding.size () != embedding.size ()
                    || Distance (embedding, prior->second.embedding)
                           > kEmbeddingMovementEpsilon;
              if (topology_requires_rebuild)
                break;
            }
          if (topology_requires_rebuild)
            {
              impl_->last_seal_failure_code = 62;
              auto tx = impl_->store->Begin ();
              tx->Execute (
                  impl_->building
                      ? "DELETE FROM cortext_sparse_route_build "
                        "WHERE singleton = 1"
                      : "DELETE FROM cortext_sparse_route_meta "
                        "WHERE singleton = 1");
              tx->Execute ("DELETE FROM cortext_sparse_route_dirty");
              tx->Commit ();
              impl_->node_cache.clear ();
              impl_->node_cache_order.clear ();
              impl_->delta_embeddings.clear ();
              impl_->removed.clear ();
              impl_->entry_memory_id = 0;
              impl_->max_level = 0;
              impl_->active_count = 0;
              return false;
            }

          std::unordered_map<long long, NodeRow> updates;
          updates.reserve (roots.size ()
                           * (impl_->parameters.reciprocal_update_count + 1));
          const bool clear_activation_snapshot
              = !impl_->building
                && (impl_->removed.count (
                        impl_->activation_entry_memory_id)
                        != 0
                    || std::any_of (
                        impl_->activation_identity_ids.begin (),
                        impl_->activation_identity_ids.end (),
                        [&] (long long memory_id) {
                          return impl_->removed.count (memory_id) != 0;
                        }));
          long long next_active_count = impl_->active_count;
          long long next_entry_memory_id = impl_->entry_memory_id;
          int next_max_level = impl_->max_level;
          const long long next_generation = impl_->generation;

          for (const long long memory_id : roots)
            {
              const auto removed = impl_->removed.count (memory_id) != 0;
              const auto delta = impl_->delta_embeddings.find (memory_id);
              const auto prior = prior_by_id.find (memory_id);
              if (removed)
                {
                  if (prior == prior_by_id.end ())
                    continue;
                  NodeRow node = prior->second;
                  if (node.active)
                    --next_active_count;
                  node.active = false;
                  updates[memory_id] = std::move (node);
                  continue;
                }
              if (delta == impl_->delta_embeddings.end ())
                continue;

              if (prior != prior_by_id.end ())
                {
                  NodeRow node = prior->second;
                  if (!node.active)
                    ++next_active_count;
                  node.embedding = delta->second;
                  node.active = true;
                  updates[memory_id] = std::move (node);
                  if (next_entry_memory_id == 0
                      || prior->second.level > next_max_level
                      || (prior->second.level == next_max_level
                          && memory_id < next_entry_memory_id))
                    {
                      next_entry_memory_id = memory_id;
                      next_max_level = prior->second.level;
                    }
                  continue;
                }

              std::vector<long long> proposed
                  = neighbor_proposals[memory_id];
              const auto proposed_rows = FetchNodes (
                  *impl_->store, impl_->embedding_dim, impl_->parameters,
                  impl_->generation, proposed);
              if (!proposed_rows)
                return false;
              std::unordered_map<long long, NodeRow> proposed_by_id;
              proposed_by_id.reserve (proposed_rows->size ());
              for (auto node : *proposed_rows)
                proposed_by_id.emplace (node.memory_id, std::move (node));

              struct Neighbor
              {
                float distance = 0.0f;
                long long memory_id = 0;
                NodeRow node;
              };
              std::vector<Neighbor> neighbors;
              neighbors.reserve (proposed.size ());
              for (const long long candidate_id : proposed)
                {
                  if (candidate_id == memory_id
                      || impl_->removed.count (candidate_id) != 0)
                    continue;
                  NodeRow candidate;
                  const auto updated = updates.find (candidate_id);
                  if (updated != updates.end ())
                    candidate = updated->second;
                  else
                    {
                      const auto persisted
                          = proposed_by_id.find (candidate_id);
                      if (persisted == proposed_by_id.end ())
                        continue;
                      candidate = persisted->second;
                    }
                  if (!candidate.active || candidate.links.empty ())
                    continue;
                  neighbors.push_back (
                      { Distance (delta->second, candidate.embedding),
                        candidate_id, std::move (candidate) });
                }
              std::sort (neighbors.begin (), neighbors.end (),
                         [] (const Neighbor &left, const Neighbor &right) {
                           return left.distance < right.distance
                                  || (left.distance == right.distance
                                      && left.memory_id < right.memory_id);
                         });
              NodeRow node;
              node.memory_id = memory_id;
              node.embedding = delta->second;
              node.level = DeterministicLevel (
                  memory_id, impl_->parameters.maximum_level);
              node.links.resize (static_cast<std::size_t> (node.level + 1));
              node.active = true;
              for (int level = 0; level <= node.level; ++level)
                {
                  auto &links
                      = node.links[static_cast<std::size_t> (level)];
                  const std::size_t link_limit
                      = level == 0
                            ? impl_->parameters
                                  .row_addressed_level_zero_links
                            : impl_->parameters
                                  .row_addressed_neighbor_count;
                  links.reserve (std::min (link_limit, neighbors.size ()));
                  for (const auto &neighbor : neighbors)
                    {
                      if (neighbor.node.level < level)
                        continue;
                      links.push_back (neighbor.memory_id);
                      if (links.size () >= link_limit)
                        break;
                    }
                }
              updates[memory_id] = node;
              ++next_active_count;
              if (next_entry_memory_id == 0 || node.level > next_max_level)
                {
                  next_entry_memory_id = memory_id;
                  next_max_level = node.level;
                }

              for (int level = 0; level <= node.level; ++level)
                {
                  std::size_t reciprocal_count = 0;
                  for (const auto &candidate : neighbors)
                    {
                      if (candidate.node.level < level)
                        continue;
                      NodeRow neighbor = candidate.node;
                      const auto updated = updates.find (
                          candidate.memory_id);
                      if (updated != updates.end ())
                        neighbor = updated->second;
                      auto &level_links
                          = neighbor.links[static_cast<std::size_t> (level)];
                      if (std::find (level_links.begin (), level_links.end (),
                                     memory_id)
                          == level_links.end ())
                        {
                          const std::size_t link_limit
                              = level == 0
                                    ? impl_->parameters
                                          .row_addressed_level_zero_links
                                    : impl_->parameters
                                          .row_addressed_neighbor_count;
                          if (level_links.size () < link_limit)
                            level_links.push_back (memory_id);
                          else if (!level_links.empty ())
                            level_links.back () = memory_id;
                        }
                      updates[neighbor.memory_id] = std::move (neighbor);
                      if (++reciprocal_count
                          >= impl_->parameters.reciprocal_update_count)
                        break;
                    }
                }
            }

          if (next_active_count < 0)
            return false;

          if (impl_->removed.count (impl_->entry_memory_id) != 0)
            {
              long long replacement_memory_id = 0;
              int replacement_level = -1;
              const auto consider_replacement
                  = [&] (long long memory_id, int level) {
                      if (memory_id <= 0
                          || impl_->removed.count (memory_id) != 0)
                        return;
                      const auto updated = updates.find (memory_id);
                      if (updated != updates.end ()
                          && !updated->second.active)
                        return;
                      if (level > replacement_level
                          || (level == replacement_level
                              && (replacement_memory_id == 0
                                  || memory_id < replacement_memory_id)))
                        {
                          replacement_memory_id = memory_id;
                          replacement_level = level;
                        }
                    };
              for (const auto &[memory_id, node] : updates)
                if (node.active)
                  consider_replacement (memory_id, node.level);

              // At most |removed| leading rows can be excluded from the
              // generation-qualified level index, so |removed|+1 finds the
              // best surviving persisted entry without a store-sized scan.
              const auto candidates = impl_->store->Execute (
                  "SELECT memory_id, level "
                  "FROM cortext_sparse_route_nodes "
                  "INDEXED BY idx_sparse_route_nodes_entry "
                  "WHERE generation = ? AND active = 1 "
                  "ORDER BY level DESC, memory_id ASC LIMIT ?",
                  { impl_->generation,
                    static_cast<long long> (impl_->removed.size () + 1) });
              for (const auto &candidate : candidates)
                consider_replacement (
                    Int64 (candidate, "memory_id"),
                    static_cast<int> (Int64 (candidate, "level")));

              if (next_active_count == 0)
                {
                  next_entry_memory_id = 0;
                  next_max_level = 0;
                }
              else if (replacement_memory_id > 0)
                {
                  next_entry_memory_id = replacement_memory_id;
                  next_max_level = replacement_level;
                }
              else
                return false;
            }

          std::vector<long long> update_ids;
          update_ids.reserve (updates.size ());
          for (const auto &[memory_id, node] : updates)
            {
              (void)node;
              update_ids.push_back (memory_id);
            }
          std::sort (update_ids.begin (), update_ids.end ());
          impl_->last_seal_failure_code = 61;
          auto tx = impl_->store->Begin ();
          for (const long long memory_id : update_ids)
            {
              const auto &node = updates.at (memory_id);
              if (node.memory_id <= 0
                  || node.embedding.size ()
                         != static_cast<std::size_t> (impl_->embedding_dim)
                  || node.level < 0
                  || node.links.size ()
                         != static_cast<std::size_t> (node.level + 1))
                return false;
              tx->Execute (
                  "INSERT INTO cortext_sparse_route_nodes("
                  "memory_id, embedding, level, links, active, generation) "
                  "VALUES(?, ?, ?, ?, ?, ?) "
                  "ON CONFLICT(memory_id) DO UPDATE SET "
                  "embedding = excluded.embedding, level = excluded.level, "
                  "links = excluded.links, active = excluded.active, "
                  "generation = excluded.generation",
                  { node.memory_id, EncodeEmbedding (node.embedding),
                    static_cast<long long> (node.level),
                    EncodeLinks (node.links),
                    static_cast<long long> (node.active ? 1 : 0),
                    next_generation });
            }
          if (impl_->building)
            tx->Execute (
                "INSERT INTO cortext_sparse_route_build("
                "singleton, schema_version, embedding_dim, "
                "cursor_memory_id, entry_memory_id, max_level, "
                "active_count, generation) VALUES(1, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(singleton) DO UPDATE SET "
                "schema_version = excluded.schema_version, "
                "embedding_dim = excluded.embedding_dim, "
                "cursor_memory_id = excluded.cursor_memory_id, "
                "entry_memory_id = excluded.entry_memory_id, "
                "max_level = excluded.max_level, "
                "active_count = excluded.active_count, "
                "generation = excluded.generation",
                { static_cast<long long> (kSchemaVersion),
                  static_cast<long long> (impl_->embedding_dim),
                  impl_->build_cursor, next_entry_memory_id,
                  static_cast<long long> (next_max_level),
                  next_active_count, next_generation });
          else
            {
              tx->Execute (
                  "INSERT INTO cortext_sparse_route_meta("
                  "singleton, schema_version, embedding_dim, "
                  "entry_memory_id, max_level, active_count, generation) "
                  "VALUES(1, ?, ?, ?, ?, ?, ?) "
                  "ON CONFLICT(singleton) DO UPDATE SET "
                  "schema_version = excluded.schema_version, "
                  "embedding_dim = excluded.embedding_dim, "
                  "entry_memory_id = excluded.entry_memory_id, "
                  "max_level = excluded.max_level, "
                  "active_count = excluded.active_count, "
                  "generation = excluded.generation",
                  { static_cast<long long> (kSchemaVersion),
                    static_cast<long long> (impl_->embedding_dim),
                    next_entry_memory_id,
                    static_cast<long long> (next_max_level),
                    next_active_count, next_generation });
              if (clear_activation_snapshot)
                tx->Execute (
                    "UPDATE cortext_sparse_route_meta SET "
                    "activation_entry_memory_id = 0, "
                    "activation_generation = 0, activation_centroid = NULL, "
                    "activation_identity_ids = NULL WHERE singleton = 1");
            }
          for (const long long memory_id : roots)
            tx->Execute (
                "DELETE FROM cortext_sparse_route_dirty WHERE memory_id = ?",
                { memory_id });
          tx->Commit ();
          impl_->last_seal_failure_code = 0;
          impl_->entry_memory_id = next_entry_memory_id;
          impl_->max_level = next_max_level;
          impl_->active_count = next_active_count;
          impl_->generation = next_generation;
          if (clear_activation_snapshot)
            {
              impl_->activation_entry_memory_id = 0;
              impl_->activation_entry_level = 0;
              impl_->activation_centroid.clear ();
              impl_->activation_identity_ids.clear ();
            }
          // A seal can invalidate cached rows out of FIFO order. Clear the
          // bookkeeping with the cache so stale order entries cannot grow
          // across repeated update/search cycles.
          impl_->node_cache.clear ();
          impl_->node_cache_order.clear ();
          impl_->delta_embeddings.clear ();
          impl_->removed.clear ();
          return true;
        }
      impl_->last_seal_failure_code = 20;
      const auto snapshot = hnsw_route->Snapshot (roots);
      if (!snapshot)
        {
          impl_->last_seal_failure_code = 2;
          return false;
        }
      impl_->last_seal_failure_code = 0;

      impl_->last_seal_failure_code = 30;
      const auto prior_rows
          = FetchNodes (*impl_->store, impl_->embedding_dim,
                        impl_->parameters, impl_->generation, roots);
      if (!prior_rows)
        {
          impl_->last_seal_failure_code = 3;
          return false;
        }
      impl_->last_seal_failure_code = 0;
      std::unordered_map<long long, bool> prior_active;
      for (const auto &node : *prior_rows)
        prior_active.emplace (node.memory_id, node.active);
      long long next_active_count = impl_->active_count;
      for (const long long memory_id : roots)
        {
          const bool was_active = prior_active[memory_id];
          const bool is_active
              = impl_->delta_embeddings.count (memory_id) != 0
                && impl_->removed.count (memory_id) == 0;
          next_active_count += static_cast<long long> (is_active)
                               - static_cast<long long> (was_active);
        }
      if (next_active_count < 0)
        {
          impl_->last_seal_failure_code = 4;
          return false;
        }

      const bool clear_activation_snapshot
          = !impl_->building
            && (impl_->removed.count (impl_->activation_entry_memory_id) != 0
                || std::any_of (
                    impl_->activation_identity_ids.begin (),
                    impl_->activation_identity_ids.end (),
                    [&] (long long memory_id) {
                      return impl_->removed.count (memory_id) != 0;
                    }));

      const long long next_generation = impl_->generation;
      impl_->last_seal_failure_code = 40;
      auto tx = impl_->store->Begin ();
      for (const auto &node : snapshot->nodes)
        {
          if (node.memory_id <= 0
              || node.embedding.size () != impl_->embedding_dim
              || node.level < 0
              || static_cast<std::size_t> (node.level)
                     > impl_->parameters.maximum_level
              || node.links.size ()
                     != static_cast<std::size_t> (node.level + 1))
            throw std::runtime_error ("invalid HNSW sparse route snapshot");
          tx->Execute (
              "INSERT INTO cortext_sparse_route_nodes("
              "memory_id, embedding, level, links, active, generation) "
              "VALUES(?, ?, ?, ?, ?, ?) ON CONFLICT(memory_id) DO UPDATE SET "
              "embedding = excluded.embedding, level = excluded.level, "
              "links = excluded.links, active = excluded.active, "
              "generation = excluded.generation",
              { node.memory_id, EncodeEmbedding (node.embedding),
                static_cast<long long> (node.level), EncodeLinks (node.links),
                static_cast<long long> (node.active ? 1 : 0), next_generation });
        }
      const std::string metadata_table
          = impl_->building ? "cortext_sparse_route_build"
                            : "cortext_sparse_route_meta";
      const std::string cursor_column
          = impl_->building ? "cursor_memory_id, " : "";
      const std::string cursor_placeholder
          = impl_->building ? "?, " : "";
      std::vector<std::any> metadata_params{
        static_cast<long long> (kSchemaVersion),
        static_cast<long long> (impl_->embedding_dim)
      };
      if (impl_->building)
        metadata_params.emplace_back (impl_->build_cursor);
      metadata_params.insert (
          metadata_params.end (),
          { snapshot->entry_memory_id,
            static_cast<long long> (snapshot->max_level), next_active_count,
            next_generation });
      tx->Execute (
          "INSERT INTO " + metadata_table + "("
          "singleton, schema_version, embedding_dim, " + cursor_column
          + "entry_memory_id, max_level, active_count, generation) "
            "VALUES(1, ?, ?, " + cursor_placeholder + "?, ?, ?, ?) "
            "ON CONFLICT(singleton) DO UPDATE SET "
            "schema_version = excluded.schema_version, "
            "embedding_dim = excluded.embedding_dim, "
            + (impl_->building
                   ? "cursor_memory_id = excluded.cursor_memory_id, "
                   : "")
            + "entry_memory_id = excluded.entry_memory_id, "
              "max_level = excluded.max_level, "
              "active_count = excluded.active_count, "
              "generation = excluded.generation",
          metadata_params);
      if (clear_activation_snapshot)
        tx->Execute (
            "UPDATE cortext_sparse_route_meta SET "
            "activation_entry_memory_id = 0, activation_generation = 0, "
            "activation_centroid = NULL, activation_identity_ids = NULL "
            "WHERE singleton = 1");
      for (const long long memory_id : roots)
        tx->Execute (
            "DELETE FROM cortext_sparse_route_dirty WHERE memory_id = ?",
            { memory_id });
      tx->Commit ();
      impl_->last_seal_failure_code = 0;
      impl_->entry_memory_id = snapshot->entry_memory_id;
      impl_->max_level = snapshot->max_level;
      impl_->active_count = next_active_count;
      impl_->generation = next_generation;
      if (clear_activation_snapshot)
        {
          impl_->activation_entry_memory_id = 0;
          impl_->activation_entry_level = 0;
          impl_->activation_centroid.clear ();
          impl_->activation_identity_ids.clear ();
        }
      // A seal can invalidate cached rows out of FIFO order. Clear the
      // bookkeeping with the cache so stale order entries cannot grow
      // across repeated update/search cycles.
      impl_->node_cache.clear ();
      impl_->node_cache_order.clear ();
      impl_->delta_embeddings.clear ();
      impl_->removed.clear ();
      return true;
    }
  catch (...)
    {
      if (impl_ && impl_->last_seal_failure_code == 0)
        impl_->last_seal_failure_code = 50;
      return false;
    }
}

bool
Route::Invalidate ()
{
  if (!impl_ || !impl_->store)
    return false;
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      auto tx = impl_->store->Begin ();
      tx->Execute (
          impl_->building
              ? "DELETE FROM cortext_sparse_route_build WHERE singleton = 1"
              : "DELETE FROM cortext_sparse_route_meta WHERE singleton = 1");
      tx->Execute ("DELETE FROM cortext_sparse_route_dirty");
      tx->Commit ();
      return true;
    }
  catch (...)
    {
      return false;
    }
}

bool
Route::SetBuildCursor (long long memory_id)
{
  if (!impl_ || !impl_->building || memory_id < 0)
    return false;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  impl_->build_cursor = memory_id;
  return true;
}

bool
Route::PublishBuild (std::size_t expected_active_count)
{
  if (!impl_ || !impl_->store || !impl_->building)
    return false;
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      if (!impl_->delta_embeddings.empty () || !impl_->removed.empty ()
          || impl_->entry_memory_id <= 0
          || impl_->active_count
                 != static_cast<long long> (expected_active_count))
        return false;
      auto tx = impl_->store->Begin ();
      tx->Execute ("DELETE FROM cortext_sparse_route_meta");
      tx->Execute (
          "INSERT INTO cortext_sparse_route_meta("
          "singleton, schema_version, embedding_dim, entry_memory_id, "
          "max_level, active_count, generation) VALUES(1, ?, ?, ?, ?, ?, ?)",
          { static_cast<long long> (kSchemaVersion),
            static_cast<long long> (impl_->embedding_dim),
            impl_->entry_memory_id, static_cast<long long> (impl_->max_level),
            impl_->active_count, impl_->generation });
      tx->Execute ("DELETE FROM cortext_sparse_route_build");
      tx->Execute ("DELETE FROM cortext_sparse_route_dirty");
      tx->Commit ();
      impl_->building = false;
      return true;
    }
  catch (...)
    {
      return false;
    }
}

std::size_t
Route::DeltaSize () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->delta_embeddings.size () + impl_->removed.size ();
}

std::size_t
Route::ActiveCount () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return static_cast<std::size_t> (std::max (0LL, impl_->active_count));
}

long long
Route::BuildCursor () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->building ? impl_->build_cursor : 0;
}

bool
Route::IsBuilding () const
{
  if (!impl_)
    return false;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->building;
}

std::optional<std::vector<long long>>
Route::DirtyMemoryIds (std::size_t limit) const
{
  if (!impl_ || !impl_->store)
    return std::vector<long long>{};
  if (limit == 0
      || limit > static_cast<std::size_t> (
                     std::numeric_limits<long long>::max ()))
    return std::nullopt;
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      const auto rows = impl_->store->Execute (
          "SELECT memory_id FROM cortext_sparse_route_dirty "
          "ORDER BY memory_id LIMIT ?",
          { static_cast<long long> (limit) });
      std::vector<long long> memory_ids;
      memory_ids.reserve (rows.size ());
      for (const auto &row : rows)
        {
          const long long memory_id = Int64 (row, "memory_id", 0);
          if (memory_id <= 0)
            return std::nullopt;
          memory_ids.push_back (memory_id);
        }
      return memory_ids;
    }
  catch (...)
    {
      return std::nullopt;
    }
}

bool
Route::HasDirtyMemoryIds () const
{
  if (!impl_ || !impl_->store)
    return false;
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      return !impl_->store
                  ->Execute (
                      "SELECT 1 AS present FROM cortext_sparse_route_dirty "
                      "LIMIT 1")
                  .empty ();
    }
  catch (...)
    {
      // Fail closed to exact retrieval when journal state cannot be read.
      return true;
    }
}

std::size_t
Route::RestartRowsLoaded () const
{
  return impl_ ? 1 : 0;
}

std::size_t
Route::LastSearchNodeRows () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->last_search_node_rows;
}

std::size_t
Route::LastActivationSnapshotRows () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->last_activation_snapshot_rows;
}

std::size_t
Route::LastActivationSnapshotCacheMissRows () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->last_activation_snapshot_cache_miss_rows;
}

std::size_t
Route::LastActivationFrontierSeedCount () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->last_activation_frontier_seed_count;
}

std::size_t
Route::LastSearchDistanceEvaluations () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->last_search_distance_evaluations;
}

std::size_t
Route::CachedNodeRows () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->node_cache.size ();
}

std::size_t
Route::CacheOrderRows () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->node_cache_order.size ();
}

int
Route::LastSearchFailureCode () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->last_search_failure_code;
}

int
Route::LastSealFailureCode () const
{
  if (!impl_)
    return -1;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->last_seal_failure_code;
}

long long
Route::ActivationEntryMemoryId () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->activation_entry_memory_id;
}

std::size_t
Route::ActivationIdentityCount () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->activation_identity_ids.size ();
}

std::vector<long long>
Route::ActivationIdentityIds () const
{
  if (!impl_)
    return {};
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->activation_identity_ids;
}

std::size_t
Route::ActivationSearchEffort () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->activation_search_ef;
}

std::size_t
Route::ActivationSearchNodeBudget () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->activation_search_node_budget;
}

} // namespace cortext::operations::sparse_retrieval_route_sqlite_internal
