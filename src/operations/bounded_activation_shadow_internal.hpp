#pragma once

#include "cortext/signal.hpp"
#include "cortext/processor/processor_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "../experimental_env.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cortext::operations::bounded_activation_shadow_internal
{

constexpr std::size_t kMaximumEmbeddingDimension = 256;
constexpr std::size_t kMaximumRoots = 32;
constexpr std::size_t kMaximumLeafBeam = 8;
constexpr std::size_t kMaximumCandidateIdentities = 8 * (1 + 12) * 40;

enum class Failure
{
  None,
  EmbeddingDimension,
  NonfiniteEmbedding,
  PrepareInjected,
  PublishInjected,
  DuplicateGeneration,
  GenerationGap,
  GenerationRegression,
  DigestMismatch,
  CapacityInvariant,
  WorkBound,
  RebuildFailure
};

#if defined(CORTEXT_TESTING)
inline std::atomic<int> g_failure_stage { 0 };

inline bool
ConsumeFailureStage (int stage)
{
  int expected = stage;
  return g_failure_stage.compare_exchange_strong (
      expected, 0, std::memory_order_relaxed);
}
#endif

inline const char *
FailureName (Failure failure)
{
  switch (failure)
    {
    case Failure::None:
      return "none";
    case Failure::EmbeddingDimension:
      return "embedding-dimension";
    case Failure::NonfiniteEmbedding:
      return "nonfinite-embedding";
    case Failure::PrepareInjected:
      return "prepare-injected";
    case Failure::PublishInjected:
      return "publish-injected";
    case Failure::DuplicateGeneration:
      return "duplicate-generation";
    case Failure::GenerationGap:
      return "generation-gap";
    case Failure::GenerationRegression:
      return "generation-regression";
    case Failure::DigestMismatch:
      return "digest-mismatch";
    case Failure::CapacityInvariant:
      return "capacity-invariant";
    case Failure::WorkBound:
      return "work-bound";
    case Failure::RebuildFailure:
      return "rebuild-failure";
    }
  return "unknown";
}

struct Parameters
{
  std::size_t roots = 0;
  std::size_t leaf_capacity = 0;
  std::size_t children_per_root = 0;
  std::size_t root_beam = 0;
  std::size_t leaf_beam = 0;
  std::size_t representatives_per_leaf = 0;
  std::size_t neighbor_degree = 0;
  std::size_t activated_leaves_per_consolidation = 0;
  std::size_t consolidation_interval = 0;
  float centroid_alpha = 0.0F;
  float root_creation_squared_distance = 0.0F;
  float leaf_creation_squared_distance = 0.0F;
  std::size_t fixed_embedding_slots = 0;
  std::size_t fixed_neighbor_slots = 0;
  std::size_t normal_comparison_bound = 0;
  std::size_t recall_comparison_bound = 0;
  std::size_t consolidation_comparison_bound = 0;
};

inline double
ClampKnob (double value)
{
  return std::max (0.0, std::min (1.0, value));
}

inline std::size_t
Rounded (double value)
{
  return static_cast<std::size_t> (std::lround (value));
}

inline Parameters
DeriveParameters (double focus, double sensitivity, double stability)
{
  const double F = ClampKnob (focus);
  const double S = ClampKnob (sensitivity);
  const double T = ClampKnob (stability);
  Parameters result;
  result.roots = 8 + Rounded (24 * S);
  result.leaf_capacity = 128 + Rounded (512 * F + 512 * T);
  result.children_per_root
      = (result.leaf_capacity + result.roots - 1) / result.roots;
  result.root_beam = 1 + Rounded (3 * S);
  result.leaf_beam = 2 + Rounded (6 * F);
  result.representatives_per_leaf = 8 + Rounded (32 * F);
  result.neighbor_degree = 4 + Rounded (8 * S);
  result.activated_leaves_per_consolidation = 8 + Rounded (24 * F);
  result.consolidation_interval = 256 + Rounded (512 * T);
  result.centroid_alpha = static_cast<float> (0.01 + 0.09 * (1.0 - T));
  result.root_creation_squared_distance
      = static_cast<float> (0.01 + 0.02 * (1.0 - S));
  result.leaf_creation_squared_distance
      = static_cast<float> (0.001 + 0.008 * (1.0 - S));
  result.fixed_embedding_slots
      = result.roots + result.leaf_capacity
        + result.leaf_capacity * result.representatives_per_leaf;
  result.fixed_neighbor_slots
      = result.leaf_capacity * result.neighbor_degree;
  result.normal_comparison_bound
      = result.roots + result.root_beam * result.children_per_root
        + 2 * result.representatives_per_leaf
        + result.children_per_root;
  result.recall_comparison_bound
      = result.roots + result.root_beam * result.children_per_root
        + result.leaf_beam * (1 + result.neighbor_degree)
              * result.representatives_per_leaf;
  result.consolidation_comparison_bound
      = 2 * result.leaf_capacity * result.roots
        + result.activated_leaves_per_consolidation
              * (result.roots
                 + result.root_beam * result.children_per_root
                 + result.neighbor_degree
                 + result.neighbor_degree * result.neighbor_degree);
  return result;
}

struct FixedEmbedding
{
  std::array<float, kMaximumEmbeddingDimension> values {};
  std::uint16_t dimension = 0;
};

inline bool
Normalize (const Eigen::VectorXf &source, FixedEmbedding &target)
{
  if (source.size () <= 0
      || static_cast<std::size_t> (source.size ())
             > kMaximumEmbeddingDimension)
    return false;
  double norm_squared = 0.0;
  for (Eigen::Index index = 0; index < source.size (); ++index)
    {
      const float value = source[index];
      if (!std::isfinite (value))
        return false;
      norm_squared += static_cast<double> (value) * value;
    }
  const double inverse_norm
      = 1.0 / std::max (std::sqrt (norm_squared), 1.0e-12);
  target.dimension = static_cast<std::uint16_t> (source.size ());
  std::fill (target.values.begin (), target.values.end (), 0.0F);
  for (Eigen::Index index = 0; index < source.size (); ++index)
    target.values[static_cast<std::size_t> (index)]
        = static_cast<float> (source[index] * inverse_norm);
  return true;
}

inline bool
Normalize (FixedEmbedding &embedding)
{
  double norm_squared = 0.0;
  for (std::size_t index = 0; index < embedding.dimension; ++index)
    norm_squared += static_cast<double> (embedding.values[index])
                    * embedding.values[index];
  if (!std::isfinite (norm_squared))
    return false;
  const double inverse_norm
      = 1.0 / std::max (std::sqrt (norm_squared), 1.0e-12);
  for (std::size_t index = 0; index < embedding.dimension; ++index)
    embedding.values[index]
        = static_cast<float> (embedding.values[index] * inverse_norm);
  return true;
}

inline float
SquaredDistance (const FixedEmbedding &left, const FixedEmbedding &right)
{
  if (left.dimension == 0 || left.dimension != right.dimension)
    return 2.0F;
  double dot = 0.0;
  for (std::size_t index = 0; index < left.dimension; ++index)
    dot += static_cast<double> (left.values[index]) * right.values[index];
  return static_cast<float> (std::max (0.0, 2.0 - 2.0 * dot));
}

inline std::uint64_t
FnvAppend (std::uint64_t hash, const void *bytes, std::size_t size)
{
  const auto *data = static_cast<const unsigned char *> (bytes);
  for (std::size_t index = 0; index < size; ++index)
    {
      hash ^= static_cast<std::uint64_t> (data[index]);
      hash *= 1099511628211ULL;
    }
  return hash;
}

template <typename Value>
inline std::uint64_t
FnvValue (std::uint64_t hash, const Value &value)
{
  return FnvAppend (hash, &value, sizeof (value));
}

inline std::uint64_t
EmbeddingIdentity (const FixedEmbedding &embedding)
{
  std::uint64_t hash = 1469598103934665603ULL;
  hash = FnvValue (hash, embedding.dimension);
  return FnvAppend (hash, embedding.values.data (),
                    embedding.dimension * sizeof (float));
}

struct Root
{
  FixedEmbedding centroid;
  std::uint64_t count = 0;
};

struct Leaf
{
  FixedEmbedding centroid;
  std::uint64_t count = 0;
  std::uint64_t last_active = 0;
  std::uint64_t epoch_activity = 0;
  std::uint16_t root = 0;
  std::uint16_t representative_count = 0;
  std::uint16_t neighbor_count = 0;
};

struct Representative
{
  FixedEmbedding embedding;
  std::uint64_t identity = 0;
  std::uint64_t event_index = 0;
  std::uint64_t last_active = 0;
  bool occupied = false;
};

struct JournalEvent
{
  FixedEmbedding embedding;
  std::uint64_t generation = 0;
  std::uint64_t event_index = 0;
  std::uint64_t embedding_identity = 0;
  std::uint64_t timestamp = 0;
  std::uint64_t digest = 0;
  std::uint64_t recall_comparisons = 0;
};

struct Metrics
{
  std::uint64_t normal_comparisons = 0;
  std::uint64_t normal_comparisons_max = 0;
  std::uint64_t recall_comparisons = 0;
  std::uint64_t recall_comparisons_max = 0;
  std::uint64_t consolidation_comparisons = 0;
  std::uint64_t consolidation_comparisons_max = 0;
  std::uint64_t consolidation_count = 0;
  std::uint64_t representative_replacements = 0;
  std::uint64_t allocation_after_initialization_count = 0;
  std::uint64_t restart_rows_visited = 0;
  double restart_rebuild_ms = 0.0;
  bool restart_rebuild_linear_history = false;
  bool restart_production_gate = false;
};

struct State
{
  explicit State (Parameters derived) : parameters (derived)
  {
    roots.resize (parameters.roots);
    leaves.resize (parameters.leaf_capacity);
    representatives.resize (parameters.leaf_capacity
                            * parameters.representatives_per_leaf);
    representative_pair_distances.resize (
        parameters.leaf_capacity * parameters.representatives_per_leaf
        * parameters.representatives_per_leaf,
        std::numeric_limits<float>::infinity ());
    neighbors.resize (parameters.leaf_capacity * parameters.neighbor_degree,
                      0);
    root_distances.resize (parameters.roots, 0.0F);
    root_accumulator.resize (
        parameters.roots * kMaximumEmbeddingDimension, 0.0);
    root_accumulator_counts.resize (parameters.roots, 0);
    root_child_counts.resize (parameters.roots, 0);
    assignment_distances.resize (parameters.leaf_capacity * parameters.roots,
                                 0.0F);
    assignment_margins.resize (parameters.leaf_capacity, 0.0F);
    assignment_order.resize (parameters.leaf_capacity, 0);
    candidate_seen.resize (parameters.leaf_capacity, 0);
    candidate_indices.resize (parameters.leaf_capacity, 0);
    transaction_journal.resize (parameters.consolidation_interval);
    initialized = true;
    const auto vector_bytes
        = roots.capacity () * sizeof (Root)
          + leaves.capacity () * sizeof (Leaf)
          + representatives.capacity () * sizeof (Representative)
          + representative_pair_distances.capacity () * sizeof (float)
          + neighbors.capacity () * sizeof (std::uint16_t)
          + root_distances.capacity () * sizeof (float)
          + root_accumulator.capacity () * sizeof (double)
          + root_accumulator_counts.capacity () * sizeof (std::uint64_t)
          + root_child_counts.capacity () * sizeof (std::size_t)
          + assignment_distances.capacity () * sizeof (float)
          + assignment_margins.capacity () * sizeof (float)
          + assignment_order.capacity () * sizeof (std::uint16_t)
          + candidate_seen.capacity () * sizeof (unsigned char)
          + candidate_indices.capacity () * sizeof (std::uint16_t)
          + transaction_journal.capacity () * sizeof (JournalEvent);
    allocated_bytes = vector_bytes;
  }

  Parameters parameters;
  bool initialized = false;
  bool available = false;
  bool disabled = false;
  bool restart_rebuild_required = false;
  Failure failure = Failure::None;
  std::uint64_t last_published_generation = 0;
  std::uint64_t last_event_index = 0;
  std::uint64_t last_event_digest = 0;
  std::size_t root_count = 0;
  std::size_t leaf_count = 0;
  std::size_t allocated_bytes = 0;
  Metrics metrics;
  std::vector<Root> roots;
  std::vector<Leaf> leaves;
  std::vector<Representative> representatives;
  std::vector<float> representative_pair_distances;
  std::vector<std::uint16_t> neighbors;
  std::vector<float> root_distances;
  std::vector<double> root_accumulator;
  std::vector<std::uint64_t> root_accumulator_counts;
  std::vector<std::size_t> root_child_counts;
  std::vector<float> assignment_distances;
  std::vector<float> assignment_margins;
  std::vector<std::uint16_t> assignment_order;
  std::vector<unsigned char> candidate_seen;
  std::vector<std::uint16_t> candidate_indices;
  std::vector<JournalEvent> transaction_journal;
  std::size_t transaction_journal_count = 0;
  std::size_t transaction_depth = 0;
};

struct PreparedObservation
{
  std::shared_ptr<State> state;
  FixedEmbedding embedding;
  std::uint64_t generation = 0;
  std::uint64_t event_index = 0;
  std::uint64_t embedding_identity = 0;
  std::uint64_t timestamp = 0;
  std::uint64_t digest = 0;
  std::size_t candidate_count = 0;
  std::uint64_t recall_comparisons = 0;
  std::array<std::uint64_t, kMaximumCandidateIdentities>
      candidate_identities {};
  std::size_t transaction_journal_start = 0;
  bool transaction_root = false;
  bool transaction_scope_active = false;
  bool valid = false;
};

struct PublishResult
{
  bool published = false;
  bool disabled = false;
  Failure failure = Failure::None;
  std::uint64_t generation = 0;
  std::uint64_t normal_comparisons = 0;
  std::uint64_t consolidation_comparisons = 0;
};

struct Snapshot
{
  Parameters parameters;
  bool available = false;
  bool disabled = false;
  bool restart_rebuild_required = false;
  Failure failure = Failure::None;
  std::uint64_t generation = 0;
  std::uint64_t event_index = 0;
  std::size_t root_count = 0;
  std::size_t leaf_count = 0;
  std::size_t allocated_bytes = 0;
  Metrics metrics;
  std::uint64_t state_digest = 0;
};

struct WorkSnapshot
{
  bool available = false;
  bool disabled = false;
  Failure failure = Failure::None;
  std::uint64_t generation = 0;
  std::uint64_t event_index = 0;
  std::size_t root_count = 0;
  std::size_t leaf_count = 0;
  Metrics metrics;
};

struct RankedIndex
{
  float distance = std::numeric_limits<float>::infinity ();
  std::uint16_t index = 0;
};

template <std::size_t Capacity>
inline void
InsertRanked (std::array<RankedIndex, Capacity> &ranked,
              std::size_t &ranked_count, std::size_t limit,
              RankedIndex candidate)
{
  limit = std::min (limit, Capacity);
  std::size_t insertion = 0;
  while (insertion < ranked_count
         && (ranked[insertion].distance < candidate.distance
             || (ranked[insertion].distance == candidate.distance
                 && ranked[insertion].index < candidate.index)))
    ++insertion;
  if (insertion >= limit)
    return;
  const std::size_t new_count = std::min (limit, ranked_count + 1);
  for (std::size_t index = new_count; index > insertion + 1; --index)
    ranked[index - 1] = ranked[index - 2];
  ranked[insertion] = candidate;
  ranked_count = new_count;
}

inline Representative &
RepresentativeAt (State &state, std::size_t leaf, std::size_t slot)
{
  return state.representatives[
      leaf * state.parameters.representatives_per_leaf + slot];
}

inline const Representative &
RepresentativeAt (const State &state, std::size_t leaf, std::size_t slot)
{
  return state.representatives[
      leaf * state.parameters.representatives_per_leaf + slot];
}

inline float &
PairDistanceAt (State &state, std::size_t leaf, std::size_t left,
                std::size_t right)
{
  const auto E = state.parameters.representatives_per_leaf;
  return state.representative_pair_distances[(leaf * E + left) * E + right];
}

inline std::uint16_t &
NeighborAt (State &state, std::size_t leaf, std::size_t slot)
{
  return state.neighbors[leaf * state.parameters.neighbor_degree + slot];
}

inline std::uint16_t
NeighborAt (const State &state, std::size_t leaf, std::size_t slot)
{
  return state.neighbors[leaf * state.parameters.neighbor_degree + slot];
}

inline void
Disable (State &state, Failure failure)
{
  state.available = false;
  state.disabled = true;
  state.failure = failure;
  state.root_count = 0;
  state.leaf_count = 0;
}

inline std::uint64_t
EventDigest (const State &state, std::uint64_t generation,
             std::uint64_t event_index, std::uint64_t embedding_identity,
             std::uint64_t timestamp)
{
  std::uint64_t hash = 1469598103934665603ULL;
  hash = FnvValue (hash, generation);
  hash = FnvValue (hash, event_index);
  hash = FnvValue (hash, embedding_identity);
  hash = FnvValue (hash, timestamp);
  hash = FnvValue (hash, state.parameters.roots);
  hash = FnvValue (hash, state.parameters.leaf_capacity);
  hash = FnvValue (hash, state.parameters.root_beam);
  hash = FnvValue (hash, state.parameters.leaf_beam);
  hash = FnvValue (hash, state.parameters.representatives_per_leaf);
  hash = FnvValue (hash, state.parameters.neighbor_degree);
  return hash;
}

inline bool
RootSelected (std::uint16_t root,
              const std::array<RankedIndex, kMaximumRoots> &ranked,
              std::size_t count)
{
  for (std::size_t index = 0; index < count; ++index)
    if (ranked[index].index == root)
      return true;
  return false;
}

inline std::uint64_t
Route (State &state, const FixedEmbedding &query,
       PreparedObservation &prepared)
{
  if (state.root_count == 0 || state.leaf_count == 0)
    return 0;
  std::array<RankedIndex, kMaximumRoots> selected_roots {};
  std::size_t selected_root_count = 0;
  for (std::size_t root = 0; root < state.root_count; ++root)
    InsertRanked (selected_roots, selected_root_count,
                  state.parameters.root_beam,
                  { SquaredDistance (query, state.roots[root].centroid),
                    static_cast<std::uint16_t> (root) });

  std::array<RankedIndex, kMaximumLeafBeam> selected_leaves {};
  std::size_t selected_leaf_count = 0;
  std::size_t compared_leaves = 0;
  for (std::size_t leaf = 0; leaf < state.leaf_count; ++leaf)
    {
      if (!RootSelected (state.leaves[leaf].root, selected_roots,
                         selected_root_count))
        continue;
      ++compared_leaves;
      InsertRanked (selected_leaves, selected_leaf_count,
                    state.parameters.leaf_beam,
                    { SquaredDistance (query, state.leaves[leaf].centroid),
                      static_cast<std::uint16_t> (leaf) });
    }

  std::fill (state.candidate_seen.begin (),
             state.candidate_seen.begin ()
                 + static_cast<std::ptrdiff_t> (state.leaf_count),
             0);
  std::size_t expanded_count = 0;
  for (std::size_t index = 0; index < selected_leaf_count; ++index)
    {
      const auto leaf = selected_leaves[index].index;
      if (!state.candidate_seen[leaf])
        {
          state.candidate_seen[leaf] = 1;
          state.candidate_indices[expanded_count++] = leaf;
        }
      for (std::size_t slot = 0;
           slot < state.leaves[leaf].neighbor_count; ++slot)
        {
          const auto neighbor = NeighborAt (state, leaf, slot);
          if (neighbor < state.leaf_count && !state.candidate_seen[neighbor])
            {
              state.candidate_seen[neighbor] = 1;
              state.candidate_indices[expanded_count++] = neighbor;
            }
        }
    }

  std::size_t representative_comparisons = 0;
  for (std::size_t expanded = 0; expanded < expanded_count; ++expanded)
    {
      const auto leaf = state.candidate_indices[expanded];
      for (std::size_t slot = 0;
           slot < state.leaves[leaf].representative_count; ++slot)
        {
          const auto &representative = RepresentativeAt (state, leaf, slot);
          ++representative_comparisons;
          bool duplicate = false;
          for (std::size_t prior = 0; prior < prepared.candidate_count; ++prior)
            if (prepared.candidate_identities[prior]
                == representative.identity)
              {
                duplicate = true;
                break;
              }
          if (!duplicate
              && prepared.candidate_count
                     < prepared.candidate_identities.size ())
            prepared.candidate_identities[prepared.candidate_count++]
                = representative.identity;
        }
    }
  return state.root_count + compared_leaves + representative_comparisons;
}

inline std::uint64_t
AddLeaf (State &state, const FixedEmbedding &embedding,
         std::uint64_t identity, std::uint64_t event_index,
         std::size_t root)
{
  if (state.leaf_count >= state.parameters.leaf_capacity)
    return 0;
  const std::size_t leaf_index = state.leaf_count++;
  auto &leaf = state.leaves[leaf_index];
  leaf = {};
  leaf.centroid = embedding;
  leaf.count = 1;
  leaf.last_active = event_index;
  leaf.epoch_activity = 1;
  leaf.root = static_cast<std::uint16_t> (root);
  leaf.representative_count = 1;
  auto &representative = RepresentativeAt (state, leaf_index, 0);
  representative = { embedding, identity, event_index, event_index, true };
  std::array<RankedIndex, 12> selected {};
  std::size_t selected_count = 0;
  std::uint64_t comparisons = 0;
  for (std::size_t other = 0; other + 1 < state.leaf_count; ++other)
    {
      if (state.leaves[other].root != root)
        continue;
      ++comparisons;
      InsertRanked (selected, selected_count, state.parameters.neighbor_degree,
                    { SquaredDistance (embedding,
                                       state.leaves[other].centroid),
                      static_cast<std::uint16_t> (other) });
    }
  leaf.neighbor_count = static_cast<std::uint16_t> (selected_count);
  for (std::size_t slot = 0; slot < selected_count; ++slot)
    NeighborAt (state, leaf_index, slot) = selected[slot].index;
  return comparisons;
}

inline std::uint64_t
UpdateRepresentatives (State &state, std::size_t leaf_index,
                       const FixedEmbedding &embedding,
                       std::uint64_t identity, std::uint64_t event_index)
{
  auto &leaf = state.leaves[leaf_index];
  const auto E = state.parameters.representatives_per_leaf;
  std::array<float, 40> distances {};
  std::uint64_t comparisons = leaf.representative_count;
  std::size_t nearest = 0;
  float nearest_distance = std::numeric_limits<float>::infinity ();
  for (std::size_t slot = 0; slot < leaf.representative_count; ++slot)
    {
      distances[slot] = SquaredDistance (
          embedding, RepresentativeAt (state, leaf_index, slot).embedding);
      if (distances[slot] < nearest_distance)
        {
          nearest_distance = distances[slot];
          nearest = slot;
        }
    }
  if (leaf.representative_count > 0)
    RepresentativeAt (state, leaf_index, nearest).last_active = event_index;
  if (leaf.representative_count < E)
    {
      const std::size_t slot = leaf.representative_count++;
      RepresentativeAt (state, leaf_index, slot)
          = { embedding, identity, event_index, event_index, true };
      for (std::size_t other = 0; other < slot; ++other)
        {
          PairDistanceAt (state, leaf_index, slot, other) = distances[other];
          PairDistanceAt (state, leaf_index, other, slot) = distances[other];
        }
      return comparisons;
    }

  float redundant_distance = std::numeric_limits<float>::infinity ();
  std::size_t redundant_left = 0;
  std::size_t redundant_right = 0;
  for (std::size_t left = 0; left < E; ++left)
    for (std::size_t right = left + 1; right < E; ++right)
      if (PairDistanceAt (state, leaf_index, left, right)
          < redundant_distance)
        {
          redundant_distance
              = PairDistanceAt (state, leaf_index, left, right);
          redundant_left = left;
          redundant_right = right;
        }
  if (nearest_distance <= redundant_distance)
    return comparisons;
  const std::size_t replacement
      = RepresentativeAt (state, leaf_index, redundant_left).last_active
                <= RepresentativeAt (state, leaf_index, redundant_right)
                       .last_active
            ? redundant_left
            : redundant_right;
  RepresentativeAt (state, leaf_index, replacement)
      = { embedding, identity, event_index, event_index, true };
  for (std::size_t other = 0; other < E; ++other)
    {
      const float distance
          = other == replacement
                ? std::numeric_limits<float>::infinity ()
                : SquaredDistance (
                      embedding,
                      RepresentativeAt (state, leaf_index, other).embedding);
      if (other != replacement)
        ++comparisons;
      PairDistanceAt (state, leaf_index, replacement, other) = distance;
      PairDistanceAt (state, leaf_index, other, replacement) = distance;
    }
  ++state.metrics.representative_replacements;
  return comparisons;
}

inline void
UpdateCentroid (FixedEmbedding &centroid, const FixedEmbedding &embedding,
                float alpha)
{
  for (std::size_t index = 0; index < centroid.dimension; ++index)
    centroid.values[index]
        = (1.0F - alpha) * centroid.values[index]
          + alpha * embedding.values[index];
  Normalize (centroid);
}

inline std::uint64_t
Ingest (State &state, const FixedEmbedding &embedding,
        std::uint64_t identity, std::uint64_t event_index)
{
  std::uint64_t comparisons = 0;
  if (state.root_count == 0)
    {
      state.roots[0].centroid = embedding;
      state.roots[0].count = 1;
      state.root_count = 1;
      comparisons += AddLeaf (state, embedding, identity, event_index, 0);
      return comparisons;
    }

  std::array<RankedIndex, kMaximumRoots> selected_roots {};
  std::size_t selected_root_count = 0;
  float nearest_root_distance = std::numeric_limits<float>::infinity ();
  std::size_t nearest_root = 0;
  for (std::size_t root = 0; root < state.root_count; ++root)
    {
      const float distance
          = SquaredDistance (embedding, state.roots[root].centroid);
      state.root_distances[root] = distance;
      ++comparisons;
      if (distance < nearest_root_distance)
        {
          nearest_root_distance = distance;
          nearest_root = root;
        }
      InsertRanked (selected_roots, selected_root_count,
                    state.parameters.root_beam,
                    { distance, static_cast<std::uint16_t> (root) });
    }
  if (state.root_count < state.parameters.roots
      && nearest_root_distance
             > state.parameters.root_creation_squared_distance)
    {
      const std::size_t root = state.root_count++;
      state.roots[root].centroid = embedding;
      state.roots[root].count = 1;
      comparisons += AddLeaf (state, embedding, identity, event_index, root);
      return comparisons;
    }

  std::size_t nearest_leaf = state.leaf_count;
  float nearest_leaf_distance = std::numeric_limits<float>::infinity ();
  for (std::size_t leaf = 0; leaf < state.leaf_count; ++leaf)
    {
      if (!RootSelected (state.leaves[leaf].root, selected_roots,
                         selected_root_count))
        continue;
      const float distance
          = SquaredDistance (embedding, state.leaves[leaf].centroid);
      ++comparisons;
      if (distance < nearest_leaf_distance)
        {
          nearest_leaf_distance = distance;
          nearest_leaf = leaf;
        }
    }
  if (nearest_leaf == state.leaf_count)
    {
      comparisons += AddLeaf (state, embedding, identity, event_index,
                              nearest_root);
      return comparisons;
    }

  if (state.leaf_count < state.parameters.leaf_capacity
      && nearest_leaf_distance
             > state.parameters.leaf_creation_squared_distance)
    {
      std::fill (state.root_child_counts.begin (),
                 state.root_child_counts.end (), 0);
      for (std::size_t leaf = 0; leaf < state.leaf_count; ++leaf)
        ++state.root_child_counts[state.leaves[leaf].root];
      std::array<RankedIndex, kMaximumRoots> ordered_roots {};
      std::size_t ordered_count = 0;
      for (std::size_t root = 0; root < state.root_count; ++root)
        InsertRanked (ordered_roots, ordered_count, state.root_count,
                      { state.root_distances[root],
                        static_cast<std::uint16_t> (root) });
      for (std::size_t index = 0; index < ordered_count; ++index)
        {
          const auto root = ordered_roots[index].index;
          if (state.root_child_counts[root]
              < state.parameters.children_per_root)
            {
              comparisons += AddLeaf (state, embedding, identity, event_index,
                                      root);
              return comparisons;
            }
        }
    }

  comparisons += UpdateRepresentatives (state, nearest_leaf, embedding,
                                        identity, event_index);
  auto &leaf = state.leaves[nearest_leaf];
  UpdateCentroid (leaf.centroid, embedding, state.parameters.centroid_alpha);
  ++leaf.count;
  leaf.last_active = event_index;
  ++leaf.epoch_activity;
  return comparisons;
}

inline std::uint64_t
BalancedAssignmentPass (State &state)
{
  std::fill (state.root_child_counts.begin (),
             state.root_child_counts.end (), 0);
  std::fill (state.root_accumulator.begin (),
             state.root_accumulator.end (), 0.0);
  std::fill (state.root_accumulator_counts.begin (),
             state.root_accumulator_counts.end (), 0);
  std::uint64_t comparisons = 0;
  for (std::size_t leaf = 0; leaf < state.leaf_count; ++leaf)
    {
      float nearest = std::numeric_limits<float>::infinity ();
      float second_nearest = std::numeric_limits<float>::infinity ();
      for (std::size_t root = 0; root < state.root_count; ++root)
        {
          const float distance
              = SquaredDistance (state.leaves[leaf].centroid,
                                 state.roots[root].centroid);
          state.assignment_distances[leaf * state.parameters.roots + root]
              = distance;
          if (distance < nearest)
            {
              second_nearest = nearest;
              nearest = distance;
            }
          else if (distance < second_nearest)
            second_nearest = distance;
          ++comparisons;
        }
      state.assignment_margins[leaf]
          = state.root_count > 1 ? second_nearest - nearest : 1.0F;
      state.assignment_order[leaf] = static_cast<std::uint16_t> (leaf);
    }
  std::sort (
      state.assignment_order.begin (),
      state.assignment_order.begin ()
          + static_cast<std::ptrdiff_t> (state.leaf_count),
      [&] (std::uint16_t left, std::uint16_t right) {
        const float left_margin = state.assignment_margins[left];
        const float right_margin = state.assignment_margins[right];
        return left_margin > right_margin
               || (left_margin == right_margin && left < right);
      });

  for (std::size_t order = 0; order < state.leaf_count; ++order)
    {
      const auto leaf = state.assignment_order[order];
      std::array<RankedIndex, kMaximumRoots> ranked {};
      std::size_t ranked_count = 0;
      for (std::size_t root = 0; root < state.root_count; ++root)
        InsertRanked (
            ranked, ranked_count, state.root_count,
            { state.assignment_distances[
                  static_cast<std::size_t> (leaf) * state.parameters.roots
                  + root],
              static_cast<std::uint16_t> (root) });
      std::size_t destination = state.root_count;
      for (std::size_t index = 0; index < ranked_count; ++index)
        if (state.root_child_counts[ranked[index].index]
            < state.parameters.children_per_root)
          {
            destination = ranked[index].index;
            break;
          }
      if (destination == state.root_count)
        {
          Disable (state, Failure::CapacityInvariant);
          return comparisons;
        }
      state.leaves[leaf].root = static_cast<std::uint16_t> (destination);
      ++state.root_child_counts[destination];
      const auto weight = state.leaves[leaf].count;
      state.root_accumulator_counts[destination] += weight;
      for (std::size_t dimension = 0;
           dimension < state.leaves[leaf].centroid.dimension; ++dimension)
        state.root_accumulator[
            destination * kMaximumEmbeddingDimension + dimension]
            += static_cast<double> (state.leaves[leaf].centroid.values[dimension])
               * weight;
    }
  for (std::size_t root = 0; root < state.root_count; ++root)
    {
      if (state.root_accumulator_counts[root] == 0)
        continue;
      auto &centroid = state.roots[root].centroid;
      for (std::size_t dimension = 0; dimension < centroid.dimension;
           ++dimension)
        centroid.values[dimension] = static_cast<float> (
            state.root_accumulator[root * kMaximumEmbeddingDimension
                                   + dimension]
            / state.root_accumulator_counts[root]);
      Normalize (centroid);
      state.roots[root].count = state.root_accumulator_counts[root];
    }
  return comparisons;
}

inline std::uint64_t
RefreshNeighbors (State &state, std::size_t leaf_index)
{
  std::array<RankedIndex, kMaximumRoots> selected_roots {};
  std::size_t selected_root_count = 0;
  for (std::size_t root = 0; root < state.root_count; ++root)
    InsertRanked (selected_roots, selected_root_count,
                  state.parameters.root_beam,
                  { SquaredDistance (state.leaves[leaf_index].centroid,
                                     state.roots[root].centroid),
                    static_cast<std::uint16_t> (root) });
  std::fill (state.candidate_seen.begin (),
             state.candidate_seen.begin ()
                 + static_cast<std::ptrdiff_t> (state.leaf_count),
             0);
  state.candidate_seen[leaf_index] = 1;
  std::size_t candidate_count = 0;
  for (std::size_t leaf = 0; leaf < state.leaf_count; ++leaf)
    if (!state.candidate_seen[leaf]
        && RootSelected (state.leaves[leaf].root, selected_roots,
                         selected_root_count))
      {
        state.candidate_seen[leaf] = 1;
        state.candidate_indices[candidate_count++]
            = static_cast<std::uint16_t> (leaf);
      }
  const auto add_candidate = [&] (std::size_t candidate) {
    if (candidate < state.leaf_count && !state.candidate_seen[candidate])
      {
        state.candidate_seen[candidate] = 1;
        state.candidate_indices[candidate_count++]
            = static_cast<std::uint16_t> (candidate);
      }
  };
  const auto old_neighbor_count = state.leaves[leaf_index].neighbor_count;
  for (std::size_t slot = 0; slot < old_neighbor_count; ++slot)
    {
      const auto neighbor = NeighborAt (state, leaf_index, slot);
      add_candidate (neighbor);
      if (neighbor >= state.leaf_count)
        continue;
      for (std::size_t nested = 0;
           nested < state.leaves[neighbor].neighbor_count; ++nested)
        add_candidate (NeighborAt (state, neighbor, nested));
    }
  std::array<RankedIndex, 12> selected {};
  std::size_t selected_count = 0;
  for (std::size_t index = 0; index < candidate_count; ++index)
    {
      const auto candidate = state.candidate_indices[index];
      InsertRanked (selected, selected_count, state.parameters.neighbor_degree,
                    { SquaredDistance (state.leaves[leaf_index].centroid,
                                       state.leaves[candidate].centroid),
                      candidate });
    }
  state.leaves[leaf_index].neighbor_count
      = static_cast<std::uint16_t> (selected_count);
  for (std::size_t slot = 0; slot < selected_count; ++slot)
    NeighborAt (state, leaf_index, slot) = selected[slot].index;
  return state.root_count + candidate_count;
}

inline std::uint64_t
Consolidate (State &state)
{
  std::uint64_t comparisons = BalancedAssignmentPass (state);
  if (state.disabled)
    return comparisons;
  comparisons += BalancedAssignmentPass (state);
  if (state.disabled)
    return comparisons;
  std::array<RankedIndex, 32> active {};
  std::size_t active_count = 0;
  for (std::size_t leaf = 0; leaf < state.leaf_count; ++leaf)
    {
      if (state.leaves[leaf].epoch_activity == 0)
        continue;
      const float rank
          = -static_cast<float> (state.leaves[leaf].epoch_activity);
      InsertRanked (active, active_count,
                    state.parameters.activated_leaves_per_consolidation,
                    { rank, static_cast<std::uint16_t> (leaf) });
    }
  for (std::size_t index = 0; index < active_count; ++index)
    comparisons += RefreshNeighbors (state, active[index].index);
  for (std::size_t leaf = 0; leaf < state.leaf_count; ++leaf)
    state.leaves[leaf].epoch_activity = 0;
  ++state.metrics.consolidation_count;
  state.metrics.consolidation_comparisons = comparisons;
  state.metrics.consolidation_comparisons_max
      = std::max (state.metrics.consolidation_comparisons_max, comparisons);
  if (comparisons > state.parameters.consolidation_comparison_bound)
    Disable (state, Failure::WorkBound);
  return comparisons;
}

inline std::uint64_t
StateDigest (const State &state)
{
  std::uint64_t hash = 1469598103934665603ULL;
  hash = FnvValue (hash, state.last_published_generation);
  hash = FnvValue (hash, state.last_event_index);
  hash = FnvValue (hash, state.root_count);
  hash = FnvValue (hash, state.leaf_count);
  for (std::size_t root = 0; root < state.root_count; ++root)
    {
      hash = FnvValue (hash, state.roots[root].centroid.dimension);
      hash = FnvAppend (
          hash, state.roots[root].centroid.values.data (),
          state.roots[root].centroid.dimension * sizeof (float));
      hash = FnvValue (hash, state.roots[root].count);
    }
  for (std::size_t leaf = 0; leaf < state.leaf_count; ++leaf)
    {
      const auto &current = state.leaves[leaf];
      hash = FnvValue (hash, current.centroid.dimension);
      hash = FnvAppend (hash, current.centroid.values.data (),
                        current.centroid.dimension * sizeof (float));
      hash = FnvValue (hash, current.count);
      hash = FnvValue (hash, current.last_active);
      hash = FnvValue (hash, current.epoch_activity);
      hash = FnvValue (hash, current.root);
      hash = FnvValue (hash, current.representative_count);
      hash = FnvValue (hash, current.neighbor_count);
      for (std::size_t slot = 0;
           slot < state.leaves[leaf].representative_count; ++slot)
        {
          const auto &representative
              = RepresentativeAt (state, leaf, slot);
          hash = FnvValue (hash, representative.embedding.dimension);
          hash = FnvAppend (
              hash, representative.embedding.values.data (),
              representative.embedding.dimension * sizeof (float));
          hash = FnvValue (hash, representative.identity);
          hash = FnvValue (hash, representative.event_index);
          hash = FnvValue (hash, representative.last_active);
          hash = FnvValue (hash, representative.occupied);
        }
      for (std::size_t slot = 0;
           slot < state.leaves[leaf].neighbor_count; ++slot)
        {
          const auto neighbor = NeighborAt (state, leaf, slot);
          hash = FnvValue (hash, neighbor);
        }
    }
  return hash;
}

inline PreparedObservation
PrepareWithGenerationOffset (const std::shared_ptr<State> &state,
                             const Signal &signal,
                             std::size_t generation_offset)
{
  PreparedObservation prepared;
  prepared.state = state;
  if (!state || !state->available || state->disabled
      || signal.retention == Retention::Ephemeral
      || signal.force_consolidation)
    return prepared;
#if defined(CORTEXT_TESTING)
  if (ConsumeFailureStage (1))
    {
      Disable (*state, Failure::PrepareInjected);
      return prepared;
    }
#endif
  if (!Normalize (signal.embedding, prepared.embedding))
    {
      Disable (*state,
               signal.embedding.size () <= 0
                       || static_cast<std::size_t> (signal.embedding.size ())
                              > kMaximumEmbeddingDimension
                   ? Failure::EmbeddingDimension
                   : Failure::NonfiniteEmbedding);
      return prepared;
    }
  prepared.generation
      = state->last_published_generation + generation_offset + 1;
  prepared.event_index = state->last_event_index + generation_offset + 1;
  prepared.embedding_identity = EmbeddingIdentity (prepared.embedding);
  prepared.timestamp = signal.timestamp;
  prepared.digest = EventDigest (*state, prepared.generation,
                                 prepared.event_index,
                                 prepared.embedding_identity,
                                 prepared.timestamp);
  prepared.recall_comparisons = Route (*state, prepared.embedding, prepared);
  if (prepared.recall_comparisons > state->parameters.recall_comparison_bound)
    {
      Disable (*state, Failure::WorkBound);
      return prepared;
    }
  prepared.valid = true;
  return prepared;
}

inline PreparedObservation
Prepare (const std::shared_ptr<State> &state, const Signal &signal)
{
  return PrepareWithGenerationOffset (state, signal, 0);
}

inline PreparedObservation
PrepareForCanonicalTransaction (const std::shared_ptr<State> &state,
                                const Signal &signal,
                                bool root_transaction)
{
  PreparedObservation prepared;
  prepared.state = state;
  if (!state || !state->available || state->disabled
      || signal.retention == Retention::Ephemeral
      || signal.force_consolidation)
    return prepared;
  if ((root_transaction && state->transaction_depth != 0)
      || (!root_transaction && state->transaction_depth == 0)
      || state->transaction_journal_count
             >= state->transaction_journal.size ())
    {
      Disable (*state, Failure::CapacityInvariant);
      return prepared;
    }
  if (root_transaction)
    state->transaction_journal_count = 0;
  const auto journal_start = state->transaction_journal_count;
  prepared = PrepareWithGenerationOffset (state, signal, journal_start);
  if (!prepared.valid)
    return prepared;
  auto &journal = state->transaction_journal[state->transaction_journal_count];
  journal = { prepared.embedding,
              prepared.generation,
              prepared.event_index,
              prepared.embedding_identity,
              prepared.timestamp,
              prepared.digest,
              prepared.recall_comparisons };
  ++state->transaction_journal_count;
  ++state->transaction_depth;
  prepared.transaction_journal_start = journal_start;
  prepared.transaction_root = root_transaction;
  prepared.transaction_scope_active = true;
  return prepared;
}

inline PublishResult
PublishAfterPersistentCommit (PreparedObservation &prepared)
{
  PublishResult result;
  if (!prepared.valid || !prepared.state)
    return result;
  auto &state = *prepared.state;
  result.generation = prepared.generation;
  if (!state.available || state.disabled)
    {
      result.disabled = state.disabled;
      result.failure = state.failure;
      return result;
    }
  const auto expected = state.last_published_generation + 1;
  if (prepared.generation < expected)
    {
      Disable (state, prepared.generation == state.last_published_generation
                          ? Failure::DuplicateGeneration
                          : Failure::GenerationRegression);
    }
  else if (prepared.generation > expected)
    Disable (state, Failure::GenerationGap);
  else if (prepared.digest
           != EventDigest (state, prepared.generation, prepared.event_index,
                           prepared.embedding_identity,
                           prepared.timestamp))
    Disable (state, Failure::DigestMismatch);
#if defined(CORTEXT_TESTING)
  if (!state.disabled && ConsumeFailureStage (2))
    Disable (state, Failure::PublishInjected);
#endif
  if (state.disabled)
    {
      result.disabled = true;
      result.failure = state.failure;
      prepared.valid = false;
      return result;
    }
  try
    {
      result.normal_comparisons
          = Ingest (state, prepared.embedding, prepared.embedding_identity,
                    prepared.event_index);
      state.metrics.normal_comparisons = result.normal_comparisons;
      state.metrics.normal_comparisons_max
          = std::max (state.metrics.normal_comparisons_max,
                      result.normal_comparisons);
      state.metrics.recall_comparisons = prepared.recall_comparisons;
      state.metrics.recall_comparisons_max
          = std::max (state.metrics.recall_comparisons_max,
                      prepared.recall_comparisons);
      if (result.normal_comparisons
          > state.parameters.normal_comparison_bound)
        Disable (state, Failure::WorkBound);
      if (!state.disabled)
        {
          state.last_published_generation = prepared.generation;
          state.last_event_index = prepared.event_index;
          state.last_event_digest = prepared.digest;
          if (state.last_event_index % state.parameters.consolidation_interval
              == 0)
            result.consolidation_comparisons = Consolidate (state);
        }
    }
  catch (...)
    {
      Disable (state, Failure::PublishInjected);
    }
  result.published = !state.disabled;
  result.disabled = state.disabled;
  result.failure = state.failure;
  prepared.valid = false;
  return result;
}

inline void
DiscardCanonicalTransaction (PreparedObservation &prepared)
{
  if (!prepared.transaction_scope_active || !prepared.state)
    return;
  auto &state = *prepared.state;
  state.transaction_journal_count
      = std::min (state.transaction_journal_count,
                  prepared.transaction_journal_start);
  if (state.transaction_depth > 0)
    --state.transaction_depth;
  if (prepared.transaction_root)
    {
      state.transaction_depth = 0;
      state.transaction_journal_count = 0;
    }
  prepared.transaction_scope_active = false;
  prepared.valid = false;
}

/// Clears a prepared journal entry when control leaves Process before the
/// canonical transaction reaches its explicit commit or rollback path.
class CanonicalTransactionJournalScope final
{
public:
  explicit CanonicalTransactionJournalScope (PreparedObservation &prepared)
      : prepared_ (prepared)
  {
  }

  ~CanonicalTransactionJournalScope () noexcept
  {
    DiscardCanonicalTransaction (prepared_);
  }

  CanonicalTransactionJournalScope (const CanonicalTransactionJournalScope &)
      = delete;
  CanonicalTransactionJournalScope &
  operator= (const CanonicalTransactionJournalScope &) = delete;

private:
  PreparedObservation &prepared_;
};

inline PublishResult
PublishAfterCanonicalTransactionCommit (PreparedObservation &prepared)
{
  PublishResult result;
  if (!prepared.transaction_scope_active || !prepared.state)
    return result;
  auto state = prepared.state;
  if (!prepared.transaction_root)
    {
      if (state->transaction_depth == 0)
        Disable (*state, Failure::CapacityInvariant);
      else
        --state->transaction_depth;
      prepared.transaction_scope_active = false;
      prepared.valid = false;
      result.disabled = state->disabled;
      result.failure = state->failure;
      return result;
    }
  if (state->transaction_depth != 1)
    {
      Disable (*state, Failure::CapacityInvariant);
      state->transaction_depth = 0;
      state->transaction_journal_count = 0;
      prepared.transaction_scope_active = false;
      prepared.valid = false;
      result.disabled = true;
      result.failure = state->failure;
      return result;
    }
  const auto event_count = state->transaction_journal_count;
  for (std::size_t index = 0; index < event_count && !state->disabled;
       ++index)
    {
      const auto &journal = state->transaction_journal[index];
      PreparedObservation event;
      event.state = state;
      event.embedding = journal.embedding;
      event.generation = journal.generation;
      event.event_index = journal.event_index;
      event.embedding_identity = journal.embedding_identity;
      event.timestamp = journal.timestamp;
      event.digest = journal.digest;
      event.recall_comparisons = journal.recall_comparisons;
      event.valid = true;
      result = PublishAfterPersistentCommit (event);
    }
  state->transaction_depth = 0;
  state->transaction_journal_count = 0;
  prepared.transaction_scope_active = false;
  prepared.valid = false;
  return result;
}

struct RegistryState
{
  std::atomic<std::size_t> enabled_count { 0 };
  std::mutex mutex;
  std::unordered_map<const ProcessorContext *, std::shared_ptr<State>> states;
};

inline RegistryState &
Registry ()
{
  static auto *registry = new RegistryState ();
  return *registry;
}

inline std::shared_ptr<State>
Find (const ProcessorContext &context)
{
  auto &registry = Registry ();
  if (registry.enabled_count.load (std::memory_order_relaxed) == 0)
    return nullptr;
  std::lock_guard<std::mutex> lock (registry.mutex);
  const auto found = registry.states.find (&context);
  return found == registry.states.end () ? nullptr : found->second;
}

inline long long
ExtractCount (const std::vector<std::map<std::string, std::any>> &rows)
{
  if (rows.empty ())
    return 0;
  const auto found = rows.front ().find ("n");
  if (found == rows.front ().end ())
    return 0;
  if (found->second.type () == typeid (long long))
    return std::any_cast<long long> (found->second);
  if (found->second.type () == typeid (int64_t))
    return std::any_cast<int64_t> (found->second);
  if (found->second.type () == typeid (int))
    return std::any_cast<int> (found->second);
  return 0;
}

inline void
RebuildFromPersistentAuthority (State &state, Store &store)
{
  const auto started = std::chrono::steady_clock::now ();
  try
    {
      const auto rows = store.Execute (
          "SELECT s.signal_id, s.timestamp, "
          "COALESCE(a.embedding, e.embedding) AS embedding "
          "FROM signals s "
          "LEFT JOIN cortext_active_signal_embeddings a "
          "ON a.signal_id = s.signal_id "
          "LEFT JOIN embeddings e "
          "ON e.embedding_id = s.embedding_id "
          "ORDER BY s.signal_id");
      const auto authoritative_signal_count
          = ExtractCount (
              store.Execute ("SELECT COUNT(*) AS n FROM signals"));
      if (authoritative_signal_count < 0
          || static_cast<std::size_t> (authoritative_signal_count)
                 != rows.size ())
        {
          Disable (state, Failure::RebuildFailure);
          return;
        }
      state.available = true;
      state.restart_rebuild_required = false;
      std::uint64_t event_index = 0;
      for (const auto &row : rows)
        {
          const auto embedding_it = row.find ("embedding");
          if (embedding_it == row.end () || !embedding_it->second.has_value ())
            {
              Disable (state, Failure::RebuildFailure);
              return;
            }
          const auto bytes = store::BlobFromAny (embedding_it->second);
          if (bytes.empty () || bytes.size () % sizeof (float) != 0)
            {
              Disable (state, Failure::RebuildFailure);
              return;
            }
          const auto dimension = bytes.size () / sizeof (float);
          if (dimension == 0 || dimension > kMaximumEmbeddingDimension)
            {
              Disable (state, Failure::EmbeddingDimension);
              return;
            }
          Eigen::VectorXf embedding (static_cast<Eigen::Index> (dimension));
          std::memcpy (embedding.data (), bytes.data (), bytes.size ());
          FixedEmbedding normalized;
          if (!Normalize (embedding, normalized))
            {
              Disable (state, Failure::NonfiniteEmbedding);
              return;
            }
          ++event_index;
          const auto identity = EmbeddingIdentity (normalized);
          const auto comparisons
              = Ingest (state, normalized, identity, event_index);
          state.metrics.normal_comparisons = comparisons;
          state.metrics.normal_comparisons_max
              = std::max (state.metrics.normal_comparisons_max, comparisons);
          if (comparisons > state.parameters.normal_comparison_bound)
            {
              Disable (state, Failure::WorkBound);
              return;
            }
          if (event_index % state.parameters.consolidation_interval == 0)
            Consolidate (state);
          if (state.disabled)
            return;
        }
      state.last_published_generation = event_index;
      state.last_event_index = event_index;
      state.metrics.restart_rows_visited = rows.size ();
      state.metrics.restart_rebuild_linear_history = !rows.empty ();
      state.metrics.restart_production_gate = rows.empty ();
      state.metrics.restart_rebuild_ms
          = std::chrono::duration_cast<
                std::chrono::duration<double, std::milli>> (
                std::chrono::steady_clock::now () - started)
                .count ();
    }
  catch (...)
    {
      Disable (state, Failure::RebuildFailure);
    }
}

inline std::shared_ptr<State>
Initialize (const ProcessorContext &context, Store *store, double focus,
            double sensitivity, double stability)
{
  if (!internal::experimental_env::Flag (
          "CORTEXT_BOUNDED_ACTIVATION_SHADOW"))
    return nullptr;
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  auto &state = registry.states[&context];
  if (state)
    return state;
  state = std::make_shared<State> (
      DeriveParameters (focus, sensitivity, stability));
  registry.enabled_count.fetch_add (1, std::memory_order_relaxed);
  if (!store)
    {
      state->available = true;
      return state;
    }
  try
    {
      const long long row_count
          = ExtractCount (store->Execute ("SELECT COUNT(*) AS n FROM signals"));
      if (row_count == 0)
        state->available = true;
      else
        {
          state->restart_rebuild_required = true;
          if (internal::experimental_env::Flag (
                  "CORTEXT_BOUNDED_ACTIVATION_SHADOW_REBUILD"))
            RebuildFromPersistentAuthority (*state, *store);
        }
    }
  catch (...)
    {
      Disable (*state, Failure::RebuildFailure);
    }
  return state;
}

inline void
Erase (const ProcessorContext &context)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  if (registry.states.erase (&context) > 0)
    registry.enabled_count.fetch_sub (1, std::memory_order_relaxed);
}

inline Snapshot
ReadSnapshot (const State &state)
{
  return { state.parameters,
           state.available,
           state.disabled,
           state.restart_rebuild_required,
           state.failure,
           state.last_published_generation,
           state.last_event_index,
           state.root_count,
           state.leaf_count,
           state.allocated_bytes,
           state.metrics,
           StateDigest (state) };
}

inline std::vector<Snapshot>
Snapshots ()
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  std::vector<Snapshot> snapshots;
  snapshots.reserve (registry.states.size ());
  for (const auto &[_, state] : registry.states)
    snapshots.push_back (ReadSnapshot (*state));
  return snapshots;
}

inline std::optional<WorkSnapshot>
SingleWorkSnapshot ()
{
  auto &registry = Registry ();
  if (registry.enabled_count.load (std::memory_order_relaxed) == 0)
    return std::nullopt;
  std::lock_guard<std::mutex> lock (registry.mutex);
  if (registry.states.size () != 1 || !registry.states.begin ()->second)
    return std::nullopt;
  const auto &state = *registry.states.begin ()->second;
  return WorkSnapshot { state.available,
                        state.disabled,
                        state.failure,
                        state.last_published_generation,
                        state.last_event_index,
                        state.root_count,
                        state.leaf_count,
                        state.metrics };
}

#if defined(CORTEXT_TESTING)
inline void
SetFailureStageForTest (int stage)
{
  g_failure_stage.store (stage, std::memory_order_relaxed);
}

inline std::vector<Snapshot>
SnapshotsForTest ()
{
  return Snapshots ();
}

inline std::size_t
RegistrySizeForTest ()
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  return registry.states.size ();
}
#endif

} // namespace cortext::operations::bounded_activation_shadow_internal
