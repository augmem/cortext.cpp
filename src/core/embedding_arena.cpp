#include "cortext/internal/embedding_arena.hpp"

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace cortext::internal
{
namespace
{

constexpr std::uint32_t kChunkShift = 10;
constexpr std::uint32_t kChunkRows = 1u << kChunkShift;
constexpr std::uint32_t kChunkMask = kChunkRows - 1;
constexpr std::size_t kMaxChunks = std::size_t{ 1 } << 16;
constexpr std::size_t kMaxPools = 64;
constexpr std::size_t kRowAlignmentFloats = 16; // 64-byte rows
constexpr std::align_val_t kChunkAlignment{ 64 };

struct ChunkMeta
{
  std::array<std::atomic<std::uint32_t>, kChunkRows> refs{};
  std::array<float, kChunkRows> norms{};
  std::array<std::uint64_t, kChunkRows> hashes{};
  std::array<bool, kChunkRows> live{};
};

struct ChunkDeleter
{
  void
  operator() (float *chunk) const noexcept
  {
    ::operator delete (chunk, kChunkAlignment);
  }
};

/// Rows of one dimension. Chunks are allocated on demand and never move or
/// shrink, so readers index them without the mutex while they hold a handle.
struct Pool
{
  explicit Pool (Eigen::Index dimension)
      : dim (dimension),
        stride ((static_cast<std::size_t> (dimension) + kRowAlignmentFloats - 1)
                / kRowAlignmentFloats * kRowAlignmentFloats),
        chunks (new std::atomic<float *>[kMaxChunks]),
        meta (new std::atomic<ChunkMeta *>[kMaxChunks])
  {
    for (std::size_t index = 0; index < kMaxChunks; ++index)
      {
        chunks[index].store (nullptr, std::memory_order_relaxed);
        meta[index].store (nullptr, std::memory_order_relaxed);
      }
  }

  ~Pool ()
  {
    for (std::size_t index = 0; index < chunk_count; ++index)
      {
        ChunkDeleter () (chunks[index].load (std::memory_order_relaxed));
        delete meta[index].load (std::memory_order_relaxed);
      }
  }

  float *
  Row (std::uint32_t slot) const noexcept
  {
    return chunks[slot >> kChunkShift].load (std::memory_order_acquire)
           + static_cast<std::size_t> (slot & kChunkMask) * stride;
  }

  ChunkMeta &
  Meta (std::uint32_t slot) const noexcept
  {
    return *meta[slot >> kChunkShift].load (std::memory_order_acquire);
  }

  const Eigen::Index dim;
  const std::size_t stride;
  std::unique_ptr<std::atomic<float *>[]> chunks;
  std::unique_ptr<std::atomic<ChunkMeta *>[]> meta;

  std::mutex mutex;
  std::unordered_multimap<std::uint64_t, std::uint32_t> by_hash;
  std::vector<std::uint32_t> free_slots;
  std::uint32_t next_slot = 0;
  std::size_t chunk_count = 0;
  std::size_t live_rows = 0;
};

std::array<std::atomic<Pool *>, kMaxPools> g_pools{};
std::mutex g_pools_mutex;
std::atomic<std::uint64_t> g_rows_written{ 0 };
std::atomic<std::uint64_t> g_rows_shared{ 0 };

std::uint64_t
HashRow (const float *values, Eigen::Index size) noexcept
{
  std::uint64_t hash = 0xcbf29ce484222325ull ^ static_cast<std::uint64_t> (size);
  for (Eigen::Index index = 0; index < size; ++index)
    {
      std::uint32_t word = 0;
      std::memcpy (&word, values + index, sizeof (word));
      hash = (hash ^ word) * 0x100000001b3ull;
      hash ^= hash >> 29;
    }
  return hash;
}

/// Returns the 1-based pool index for @p dimension, creating the pool once.
/// Pools are process-lifetime: they are intentionally never destroyed so a
/// handle released during static destruction still finds its row.
std::uint32_t
PoolIndexFor (Eigen::Index dimension)
{
  for (std::size_t index = 0; index < kMaxPools; ++index)
    {
      Pool *pool = g_pools[index].load (std::memory_order_acquire);
      if (pool == nullptr)
        break;
      if (pool->dim == dimension)
        return static_cast<std::uint32_t> (index + 1);
    }
  std::lock_guard<std::mutex> lock (g_pools_mutex);
  for (std::size_t index = 0; index < kMaxPools; ++index)
    {
      Pool *pool = g_pools[index].load (std::memory_order_acquire);
      if (pool == nullptr)
        {
          g_pools[index].store (new Pool (dimension),
                                std::memory_order_release);
          return static_cast<std::uint32_t> (index + 1);
        }
      if (pool->dim == dimension)
        return static_cast<std::uint32_t> (index + 1);
    }
  throw std::length_error ("embedding arena: too many distinct dimensions");
}

Pool &
PoolAt (std::uint32_t pool_index) noexcept
{
  return *g_pools[pool_index - 1].load (std::memory_order_acquire);
}

std::uint32_t
AllocateSlotLocked (Pool &pool)
{
  if (!pool.free_slots.empty ())
    {
      const std::uint32_t slot = pool.free_slots.back ();
      pool.free_slots.pop_back ();
      return slot;
    }
  const std::uint32_t slot = pool.next_slot;
  const std::size_t chunk = slot >> kChunkShift;
  if (chunk >= kMaxChunks)
    throw std::length_error ("embedding arena: row capacity exhausted");
  if (chunk == pool.chunk_count)
    {
      auto *rows = static_cast<float *> (::operator new (
          static_cast<std::size_t> (kChunkRows) * pool.stride * sizeof (float),
          kChunkAlignment));
      std::memset (rows, 0,
                   static_cast<std::size_t> (kChunkRows) * pool.stride
                       * sizeof (float));
      auto *meta = new ChunkMeta ();
      pool.chunks[chunk].store (rows, std::memory_order_release);
      pool.meta[chunk].store (meta, std::memory_order_release);
      ++pool.chunk_count;
    }
  ++pool.next_slot;
  return slot;
}

} // namespace

SharedEmbedding::SharedEmbedding (const Eigen::VectorXf &values)
    : SharedEmbedding (values.data (), values.size ())
{
}

SharedEmbedding::SharedEmbedding (const float *values, Eigen::Index size)
{
  if (size <= 0 || values == nullptr)
    return;
  const std::uint32_t pool_index = PoolIndexFor (size);
  Pool &pool = PoolAt (pool_index);
  const std::uint64_t hash = HashRow (values, size);
  const std::size_t bytes = static_cast<std::size_t> (size) * sizeof (float);
  std::lock_guard<std::mutex> lock (pool.mutex);
  const auto [begin, end] = pool.by_hash.equal_range (hash);
  for (auto it = begin; it != end; ++it)
    {
      if (std::memcmp (pool.Row (it->second), values, bytes) == 0)
        {
          // A row whose count already reached zero is resurrected here; its
          // pending release re-checks the count under this mutex and keeps it.
          pool.Meta (it->second).refs[it->second & kChunkMask].fetch_add (
              1, std::memory_order_relaxed);
          g_rows_shared.fetch_add (1, std::memory_order_relaxed);
          pool_ = pool_index;
          slot_ = it->second;
          return;
        }
    }
  const std::uint32_t slot = AllocateSlotLocked (pool);
  float *row = pool.Row (slot);
  std::memcpy (row, values, bytes);
  ChunkMeta &meta = pool.Meta (slot);
  const std::uint32_t offset = slot & kChunkMask;
  meta.norms[offset] = ConstMap (row, size).norm ();
  meta.hashes[offset] = hash;
  meta.live[offset] = true;
  meta.refs[offset].store (1, std::memory_order_relaxed);
  pool.by_hash.emplace (hash, slot);
  ++pool.live_rows;
  g_rows_written.fetch_add (1, std::memory_order_relaxed);
  pool_ = pool_index;
  slot_ = slot;
}

SharedEmbedding::SharedEmbedding (const SharedEmbedding &other) noexcept
    : pool_ (other.pool_), slot_ (other.slot_)
{
  if (pool_ != 0)
    PoolAt (pool_).Meta (slot_).refs[slot_ & kChunkMask].fetch_add (
        1, std::memory_order_relaxed);
}

SharedEmbedding::SharedEmbedding (SharedEmbedding &&other) noexcept
    : pool_ (other.pool_), slot_ (other.slot_)
{
  other.pool_ = 0;
  other.slot_ = 0;
}

SharedEmbedding &
SharedEmbedding::operator= (const SharedEmbedding &other) noexcept
{
  if (this != &other)
    {
      SharedEmbedding copy (other);
      *this = std::move (copy);
    }
  return *this;
}

SharedEmbedding &
SharedEmbedding::operator= (SharedEmbedding &&other) noexcept
{
  if (this != &other)
    {
      Release ();
      pool_ = other.pool_;
      slot_ = other.slot_;
      other.pool_ = 0;
      other.slot_ = 0;
    }
  return *this;
}

SharedEmbedding::~SharedEmbedding () { Release (); }

void
SharedEmbedding::Release () noexcept
{
  if (pool_ == 0)
    return;
  Pool &pool = PoolAt (pool_);
  const std::uint32_t slot = slot_;
  pool_ = 0;
  slot_ = 0;
  ChunkMeta &meta = pool.Meta (slot);
  const std::uint32_t offset = slot & kChunkMask;
  if (meta.refs[offset].fetch_sub (1, std::memory_order_acq_rel) != 1)
    return;
  std::lock_guard<std::mutex> lock (pool.mutex);
  if (!meta.live[offset]
      || meta.refs[offset].load (std::memory_order_acquire) != 0)
    return;
  const auto [begin, end] = pool.by_hash.equal_range (meta.hashes[offset]);
  for (auto it = begin; it != end; ++it)
    if (it->second == slot)
      {
        pool.by_hash.erase (it);
        break;
      }
  meta.live[offset] = false;
  --pool.live_rows;
  try
    {
      pool.free_slots.push_back (slot);
    }
  catch (...)
    {
      // Without free-list room the slot is leaked, never reused while live.
    }
}

Eigen::Index
SharedEmbedding::size () const noexcept
{
  return pool_ == 0 ? 0 : PoolAt (pool_).dim;
}

const float *
SharedEmbedding::data () const noexcept
{
  return pool_ == 0 ? nullptr : PoolAt (pool_).Row (slot_);
}

SharedEmbedding::ConstMap
SharedEmbedding::View () const noexcept
{
  return ConstMap (data (), size ());
}

Eigen::VectorXf
SharedEmbedding::ToVector () const
{
  return Eigen::VectorXf (View ());
}

float
SharedEmbedding::Norm () const noexcept
{
  return pool_ == 0 ? 0.0f
                    : PoolAt (pool_).Meta (slot_).norms[slot_ & kChunkMask];
}

EmbeddingArenaStats
GetEmbeddingArenaStats ()
{
  EmbeddingArenaStats stats;
  for (std::size_t index = 0; index < kMaxPools; ++index)
    {
      Pool *pool = g_pools[index].load (std::memory_order_acquire);
      if (pool == nullptr)
        break;
      std::lock_guard<std::mutex> lock (pool->mutex);
      stats.live_rows += pool->live_rows;
      stats.reserved_rows += pool->chunk_count * kChunkRows;
      stats.resident_bytes
          += pool->chunk_count
                 * (static_cast<std::size_t> (kChunkRows) * pool->stride
                        * sizeof (float)
                    + sizeof (ChunkMeta))
             + pool->by_hash.size ()
                   * (sizeof (std::uint64_t) + sizeof (std::uint32_t)
                      + 2 * sizeof (void *))
             + pool->free_slots.capacity () * sizeof (std::uint32_t);
    }
  stats.rows_written = g_rows_written.load (std::memory_order_relaxed);
  stats.rows_shared = g_rows_shared.load (std::memory_order_relaxed);
  return stats;
}

} // namespace cortext::internal
