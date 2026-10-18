// tests/embedding_arena.test.cpp
#include "../src/operations/family_embedding_features_internal.hpp"
#include "../src/operations/historical_surface_search_cache_internal.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cortext/internal/embedding_arena.hpp>

#include <Eigen/Dense>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using cortext::internal::GetEmbeddingArenaStats;
using cortext::internal::SharedEmbedding;

namespace
{

Eigen::VectorXf
RandomVector (std::mt19937 &rng, Eigen::Index size)
{
  std::normal_distribution<float> normal (0.0f, 1.0f);
  Eigen::VectorXf values (size);
  for (Eigen::Index index = 0; index < size; ++index)
    values[index] = normal (rng);
  return values;
}

bool
SameBits (float a, float b)
{
  return std::memcmp (&a, &b, sizeof (float)) == 0;
}

bool
SameBits (const Eigen::VectorXf &a, const Eigen::VectorXf &b)
{
  return a.size () == b.size ()
         && std::memcmp (a.data (), b.data (),
                         static_cast<std::size_t> (a.size ()) * sizeof (float))
                == 0;
}

} // namespace

TEST_CASE ("Embedding arena keeps one resident row per distinct vector",
           "[embedding_arena]")
{
  std::mt19937 rng (7);
  // A dimension no other test uses keeps the occupancy deltas exact.
  constexpr Eigen::Index kDim = 61;
  const Eigen::VectorXf a = RandomVector (rng, kDim);
  const Eigen::VectorXf b = RandomVector (rng, kDim);
  const auto before = GetEmbeddingArenaStats ();
  {
    SharedEmbedding first (a);
    SharedEmbedding same (a);
    SharedEmbedding other (b);
    REQUIRE (first.SameRow (same));
    REQUIRE_FALSE (first.SameRow (other));
    REQUIRE (first.size () == kDim);
    REQUIRE (SameBits (first.ToVector (), a));
    REQUIRE (reinterpret_cast<std::uintptr_t> (first.data ()) % 16 == 0);
    REQUIRE (SameBits (first.Norm (), a.norm ()));
    REQUIRE (GetEmbeddingArenaStats ().live_rows == before.live_rows + 2);

    // Copies are snapshots of the same immutable row.
    SharedEmbedding copy = first;
    REQUIRE (copy.SameRow (first));
    first = SharedEmbedding ();
    REQUIRE (first.empty ());
    REQUIRE (SameBits (copy.ToVector (), a));
    REQUIRE (GetEmbeddingArenaStats ().live_rows == before.live_rows + 2);

    // Bitwise addressing: -0.0 and +0.0 are different rows.
    Eigen::VectorXf positive_zero = Eigen::VectorXf::Zero (kDim);
    Eigen::VectorXf negative_zero = positive_zero;
    negative_zero[3] = -0.0f;
    SharedEmbedding pz (positive_zero);
    SharedEmbedding nz (negative_zero);
    REQUIRE_FALSE (pz.SameRow (nz));
    REQUIRE (pz.Norm () == 0.0f);
  }
  const auto after = GetEmbeddingArenaStats ();
  REQUIRE (after.live_rows == before.live_rows);
  REQUIRE (after.rows_written >= before.rows_written + 4);

  // A released row's slot is reused instead of growing the arena.
  {
    SharedEmbedding reused (RandomVector (rng, kDim));
    REQUIRE (GetEmbeddingArenaStats ().reserved_rows == after.reserved_rows);
  }
  REQUIRE (SharedEmbedding (Eigen::VectorXf ()).empty ());
  REQUIRE (SharedEmbedding ().size () == 0);
  REQUIRE (SharedEmbedding ().data () == nullptr);
}

TEST_CASE ("Arena row distances match the former flat-buffer scans bit for bit",
           "[embedding_arena]")
{
  using RowMajorMatrix
      = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  namespace cache = cortext::operations::historical_surface_search_cache_internal;
  std::mt19937 rng (11);
  for (const Eigen::Index dim : { Eigen::Index{ 256 }, Eigen::Index{ 7 },
                                  Eigen::Index{ 130 } })
    {
      constexpr Eigen::Index kRows = 300;
      std::vector<float> flat;
      std::vector<SharedEmbedding> rows;
      std::vector<Eigen::VectorXf> owned;
      for (Eigen::Index row = 0; row < kRows; ++row)
        {
          owned.push_back (RandomVector (rng, dim));
          flat.insert (flat.end (), owned.back ().data (),
                       owned.back ().data () + dim);
          rows.emplace_back (owned.back ());
        }
      const Eigen::VectorXf query = RandomVector (rng, dim);
      const Eigen::Map<const RowMajorMatrix> matrix (flat.data (), kRows, dim);
      Eigen::VectorXf scan (kRows);
      scan = (matrix.rowwise () - query.transpose ()).rowwise ().squaredNorm ();
      for (Eigen::Index row = 0; row < kRows; ++row)
        {
          const auto &shared = rows[static_cast<std::size_t> (row)];
          const auto &vector = owned[static_cast<std::size_t> (row)];
          REQUIRE (SameBits (cache::RowSquaredDistance (shared, query),
                             scan[row]));
          REQUIRE (SameBits (
              cache::RowBlockSquaredDistance (shared, query),
              (matrix.row (row) - query.transpose ()).squaredNorm ()));
          REQUIRE (SameBits ((shared.View () - query).squaredNorm (),
                             (vector - query).squaredNorm ()));
          REQUIRE (SameBits (shared.View ().dot (query), vector.dot (query)));
          REQUIRE (SameBits (shared.View ().norm (), vector.norm ()));
          REQUIRE (SameBits (shared.Norm (), vector.norm ()));
        }
    }
}

TEST_CASE ("Family norm side table expands to the full features bit for bit",
           "[embedding_arena][family]")
{
  namespace family = cortext::operations::family_embedding_features_internal;
  std::mt19937 rng (13);
  for (const Eigen::Index dim : { Eigen::Index{ 256 }, Eigen::Index{ 6 } })
    for (int sample = 0; sample < 50; ++sample)
      {
        const SharedEmbedding row (RandomVector (rng, dim));
        const auto full = family::Build (row.ToVector ());
        const auto expanded = family::Expand (
            row.data (), row.size (), family::BuildNorms (row.data (),
                                                          row.size ()));
        REQUIRE (SameBits (full.normalized, expanded.normalized));
        REQUIRE (full.normalized_squared_norm
                 == expanded.normalized_squared_norm);
        REQUIRE (full.block_norms == expanded.block_norms);
        REQUIRE (full.hadamard_block_norms == expanded.hadamard_block_norms);
        REQUIRE (full.has_hadamard_features == expanded.has_hadamard_features);
      }
  const SharedEmbedding zero (Eigen::VectorXf::Zero (8).eval ());
  REQUIRE (family::Expand (zero.data (), zero.size (),
                           family::BuildNorms (zero.data (), zero.size ()))
               .normalized.size ()
           == 0);
}
