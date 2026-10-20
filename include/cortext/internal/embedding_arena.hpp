#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>

namespace cortext::internal
{

/// @brief Handle to one immutable float32 row in the process-wide embedding
/// arena.
///
/// The arena keeps exactly one resident copy of every distinct vector held by
/// a handle: rows are content-addressed (bitwise), reference-counted, and
/// never mutated after they are written, so a copied handle is a
/// copy-on-write snapshot that costs one reference increment. Rows of one
/// dimension live in 64-byte-aligned chunks that never move, so a row
/// pointer stays valid for as long as any handle to it is alive.
///
/// Every resident embedding consumer (retrieval surface, historical and
/// current search surfaces, rollback snapshots) holds handles instead of
/// owning vectors, and scans read rows from the arena directly.
///
/// Thread safety: handles may be created, copied, and destroyed from any
/// thread; row contents are immutable.
class SharedEmbedding
{
public:
  using ConstMap = Eigen::Map<const Eigen::VectorXf, Eigen::Aligned16>;

  SharedEmbedding () noexcept = default;

  /// @brief Acquires the row bitwise-equal to @p values (writing it when no
  /// such row is resident). An empty vector yields an empty handle.
  /// Implicit: a handle is a value-semantic, immutable copy of the vector.
  SharedEmbedding (const Eigen::VectorXf &values); // NOLINT(google-explicit-constructor)

  /// @brief Same as the vector overload for @p size contiguous floats.
  SharedEmbedding (const float *values, Eigen::Index size);

  SharedEmbedding (const SharedEmbedding &other) noexcept;
  SharedEmbedding (SharedEmbedding &&other) noexcept;
  SharedEmbedding &operator= (const SharedEmbedding &other) noexcept;
  SharedEmbedding &operator= (SharedEmbedding &&other) noexcept;
  ~SharedEmbedding ();

  /// @brief Row dimension, 0 for an empty handle.
  Eigen::Index size () const noexcept;

  bool
  empty () const noexcept
  {
    return pool_ == 0;
  }

  /// @brief First float of the row (16-byte aligned), nullptr when empty.
  const float *data () const noexcept;

  /// @brief Read-only Eigen view of the row; arithmetic over it matches the
  /// same arithmetic over an owning aligned Eigen::VectorXf bit for bit.
  ConstMap View () const noexcept;

  /// @brief Owning copy of the row.
  Eigen::VectorXf ToVector () const;

  /// @brief Euclidean norm of the row as Eigen computes VectorXf::norm ().
  float Norm () const noexcept;

  /// @brief True when both handles name the same resident row.
  bool
  SameRow (const SharedEmbedding &other) const noexcept
  {
    return pool_ == other.pool_ && slot_ == other.slot_;
  }

private:
  void Release () noexcept;

  std::uint32_t pool_ = 0; ///< 1-based pool index, 0 = empty
  std::uint32_t slot_ = 0;
};

/// @brief Process-wide arena occupancy (all dimensions).
struct EmbeddingArenaStats
{
  std::size_t live_rows = 0;     ///< Distinct rows referenced by handles
  std::size_t reserved_rows = 0; ///< Rows backed by allocated chunks
  std::size_t resident_bytes = 0; ///< Bytes of allocated row chunks + side tables
  std::uint64_t rows_written = 0; ///< Cumulative rows written
  std::uint64_t rows_shared = 0;  ///< Cumulative acquisitions served by an existing row
};

EmbeddingArenaStats GetEmbeddingArenaStats ();

} // namespace cortext::internal
