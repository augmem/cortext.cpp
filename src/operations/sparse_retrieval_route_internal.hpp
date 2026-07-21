#pragma once

#include "sparse_retrieval_knobs_internal.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace cortext::operations::sparse_retrieval_route_internal
{

struct Parameters
{
  std::size_t minimum_capacity = 0;
  std::size_t graph_neighbor_count = 0;
  std::size_t construction_effort = 0;
  std::size_t query_effort = 0;
};

inline Parameters
DeriveParameters (double focus, double sensitivity, double stability)
{
  return {
    static_cast<std::size_t> (
        sparse_retrieval_knobs_internal::RouteCapacity (
            focus, sensitivity, stability))
        * 2,
    static_cast<std::size_t> (sparse_retrieval_knobs_internal::
                                  GraphNeighborCount (
        focus, sensitivity, stability)),
    static_cast<std::size_t> (sparse_retrieval_knobs_internal::
                                  ConstructionEffort (
        focus, sensitivity, stability)),
    static_cast<std::size_t> (sparse_retrieval_knobs_internal::QueryEffort (
        focus, sensitivity, stability)),
  };
}

inline Parameters
DefaultParameters ()
{
  return DeriveParameters (0.5, 0.5, 0.5);
}

struct NodeSnapshot
{
  long long memory_id = 0;
  Eigen::VectorXf embedding;
  int level = 0;
  bool active = false;
  std::vector<std::vector<long long>> links;
};

struct GraphSnapshot
{
  long long entry_memory_id = 0;
  int max_level = 0;
  std::vector<NodeSnapshot> nodes;
};

/// Private, modality/source-agnostic HNSW route over the current memory
/// surface. The route only proposes memory IDs; callers retain authoritative
/// timestamp, supersession, kind, family, and final-rank checks.
class Route
{
public:
  static std::shared_ptr<Route> Create (
      int embedding_dim,
      const std::vector<std::pair<long long, Eigen::VectorXf>> &entries,
      Parameters parameters);

  ~Route ();

  Route (const Route &) = delete;
  Route &operator= (const Route &) = delete;

  bool Upsert (long long memory_id, const Eigen::VectorXf &embedding);
  bool Remove (long long memory_id);
  bool Sync (
      const std::vector<std::pair<long long, Eigen::VectorXf>> &entries);
  bool SealDelta ();

  std::optional<std::vector<long long>> Search (
      const Eigen::VectorXf &query, std::size_t route_capacity) const;

  /// Capture exact adjacency from the primary HNSW graph. When roots are
  /// supplied, the snapshot contains the roots and their direct graph
  /// neighbors so an incremental persistent mirror can update every node
  /// whose reciprocal links may have changed.
  std::optional<GraphSnapshot> Snapshot (
      const std::vector<long long> &roots = {}) const;

  std::size_t Size () const;
  std::size_t DeltaSize () const;

private:
  struct Impl;

  explicit Route (std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

} // namespace cortext::operations::sparse_retrieval_route_internal
