#include "sparse_retrieval_route_internal.hpp"

#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cortext::operations::sparse_retrieval_route_internal
{

namespace
{

constexpr std::size_t kFirstSeed = 100;
constexpr std::size_t kSecondSeed = 777;

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
InnerProductDistance (const std::vector<float> &left,
                      const std::vector<float> &right)
{
  double dot = 0.0;
  for (std::size_t index = 0; index < left.size (); ++index)
    dot += static_cast<double> (left[index]) * right[index];
  return static_cast<float> (1.0 - dot);
}

} // namespace

struct Route::Impl
{
  explicit Impl (int dimension, std::size_t initial_capacity,
                 Parameters parameters_value)
      : embedding_dim (dimension),
        capacity (std::max (parameters_value.minimum_capacity,
                            initial_capacity)),
        parameters (std::move (parameters_value)),
        space (static_cast<std::size_t> (dimension)),
        first (&space, capacity, parameters.graph_neighbor_count,
               parameters.construction_effort, kFirstSeed),
        second (&space, capacity, parameters.graph_neighbor_count,
                parameters.construction_effort, kSecondSeed)
  {
    first.setEf (parameters.query_effort);
    second.setEf (parameters.query_effort);
  }

  bool AddSealed (long long memory_id, const Eigen::VectorXf &embedding)
  {
    auto normalized = Normalize (embedding, embedding_dim);
    if (!normalized || memory_id <= 0
        || sealed_embeddings.count (memory_id) != 0)
      return false;
    const auto label = static_cast<hnswlib::labeltype> (memory_id);
    first.addPoint (normalized->data (), label);
    second.addPoint (normalized->data (), label);
    sealed_embeddings.emplace (memory_id, std::move (*normalized));
    return true;
  }

  struct SealedFilter : hnswlib::BaseFilterFunctor
  {
    explicit SealedFilter (const Impl &owner) : owner_ (owner) {}

    bool operator() (hnswlib::labeltype label) override
    {
      const long long memory_id = static_cast<long long> (label);
      return owner_.delta_embeddings.count (memory_id) == 0
             && owner_.removed.count (memory_id) == 0
             && owner_.sealed_removed.count (memory_id) == 0;
    }

  private:
    const Impl &owner_;
  };

  int embedding_dim = 0;
  std::size_t capacity = 0;
  Parameters parameters;
  hnswlib::InnerProductSpace space;
  hnswlib::HierarchicalNSW<float> first;
  hnswlib::HierarchicalNSW<float> second;
  std::unordered_map<long long, std::vector<float>> sealed_embeddings;
  std::unordered_map<long long, std::vector<float>> delta_embeddings;
  std::unordered_set<long long> removed;
  std::unordered_set<long long> sealed_removed;
  mutable std::mutex mutex;
};

Route::Route (std::unique_ptr<Impl> impl) : impl_ (std::move (impl)) {}

Route::~Route () = default;

std::shared_ptr<Route>
Route::Create (
    int embedding_dim,
    const std::vector<std::pair<long long, Eigen::VectorXf>> &entries,
    Parameters parameters)
{
  if (embedding_dim <= 0 || parameters.minimum_capacity == 0
      || parameters.graph_neighbor_count == 0
      || parameters.construction_effort == 0
      || parameters.query_effort == 0)
    return nullptr;
  try
    {
      auto route = std::shared_ptr<Route> (new Route (std::make_unique<Impl> (
          embedding_dim, entries.size () * 2, std::move (parameters))));
      for (const auto &[memory_id, embedding] : entries)
        if (!route->impl_->AddSealed (memory_id, embedding))
          return nullptr;
      return route;
    }
  catch (...)
    {
      return nullptr;
    }
}

bool
Route::Upsert (long long memory_id, const Eigen::VectorXf &embedding)
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
Route::Remove (long long memory_id)
{
  if (!impl_ || memory_id <= 0)
    return false;
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      impl_->delta_embeddings.erase (memory_id);
      if (impl_->sealed_embeddings.count (memory_id) != 0
          && impl_->sealed_removed.count (memory_id) == 0)
        impl_->removed.insert (memory_id);
      return true;
    }
  catch (...)
    {
      return false;
    }
}

bool
Route::Sync (
    const std::vector<std::pair<long long, Eigen::VectorXf>> &entries)
{
  if (!impl_)
    return false;
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      std::unordered_set<long long> incoming;
      incoming.reserve (entries.size ());
      for (const auto &[memory_id, embedding] : entries)
        {
          auto normalized = Normalize (embedding, impl_->embedding_dim);
          if (memory_id <= 0 || !normalized
              || !incoming.insert (memory_id).second)
            return false;
          const auto sealed = impl_->sealed_embeddings.find (memory_id);
          if (sealed != impl_->sealed_embeddings.end ()
              && impl_->sealed_removed.count (memory_id) == 0
              && sealed->second == *normalized)
            {
              impl_->delta_embeddings.erase (memory_id);
              impl_->removed.erase (memory_id);
              continue;
            }
          impl_->removed.erase (memory_id);
          impl_->delta_embeddings[memory_id] = std::move (*normalized);
        }
      for (const auto &[memory_id, embedding] : impl_->sealed_embeddings)
        {
          (void)embedding;
          if (incoming.count (memory_id) == 0
              && impl_->sealed_removed.count (memory_id) == 0)
            impl_->removed.insert (memory_id);
        }
      for (auto current = impl_->delta_embeddings.begin ();
           current != impl_->delta_embeddings.end ();)
        {
          if (incoming.count (current->first) == 0)
            current = impl_->delta_embeddings.erase (current);
          else
            ++current;
        }
      return true;
    }
  catch (...)
    {
      return false;
    }
}

bool
Route::SealDelta ()
{
  if (!impl_)
    return false;
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      std::vector<long long> removed (impl_->removed.begin (),
                                      impl_->removed.end ());
      std::sort (removed.begin (), removed.end ());
      std::vector<std::pair<long long, const std::vector<float> *>> delta;
      delta.reserve (impl_->delta_embeddings.size ());
      std::size_t new_label_count = 0;
      for (const auto &[memory_id, embedding] : impl_->delta_embeddings)
        {
          delta.emplace_back (memory_id, &embedding);
          if (impl_->sealed_embeddings.count (memory_id) == 0)
            ++new_label_count;
        }
      std::sort (delta.begin (), delta.end (), [] (const auto &left,
                                                   const auto &right) {
        return left.first < right.first;
      });

      const std::size_t required
          = impl_->first.cur_element_count + new_label_count;
      if (required > impl_->capacity)
        {
          const std::size_t new_capacity
              = std::max (required, impl_->capacity * 2);
          impl_->first.resizeIndex (new_capacity);
          impl_->second.resizeIndex (new_capacity);
          impl_->capacity = new_capacity;
        }

      for (const long long memory_id : removed)
        {
          const auto label = static_cast<hnswlib::labeltype> (memory_id);
          impl_->first.markDelete (label);
          impl_->second.markDelete (label);
          impl_->sealed_removed.insert (memory_id);
        }
      for (const auto &[memory_id, embedding] : delta)
        {
          const auto label = static_cast<hnswlib::labeltype> (memory_id);
          impl_->first.addPoint (embedding->data (), label);
          impl_->second.addPoint (embedding->data (), label);
          impl_->sealed_embeddings[memory_id] = *embedding;
          impl_->sealed_removed.erase (memory_id);
        }
      impl_->removed.clear ();
      impl_->delta_embeddings.clear ();
      return true;
    }
  catch (...)
    {
      return false;
    }
}

std::optional<std::vector<long long>>
Route::Search (const Eigen::VectorXf &query,
               std::size_t route_capacity) const
{
  if (!impl_ || route_capacity == 0)
    return std::nullopt;
  auto normalized = Normalize (query, impl_->embedding_dim);
  if (!normalized)
    return std::nullopt;
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      if (impl_->sealed_embeddings.empty ()
          && impl_->delta_embeddings.empty ())
        return std::vector<long long>{};
      const std::size_t per_graph
          = std::min (route_capacity, impl_->sealed_embeddings.size ());
      std::unordered_set<long long> labels;
      labels.reserve (per_graph * 2 + impl_->delta_embeddings.size ());
      Impl::SealedFilter filter (*impl_);
      if (per_graph > 0)
        {
          for (const auto *index : { &impl_->first, &impl_->second })
            {
              auto found = index->searchKnn (normalized->data (), per_graph,
                                             &filter);
              while (!found.empty ())
                {
                  labels.insert (
                      static_cast<long long> (found.top ().second));
                  found.pop ();
                }
            }
        }
      for (const auto &[memory_id, embedding] : impl_->delta_embeddings)
        {
          (void)embedding;
          labels.insert (memory_id);
        }
      struct Candidate
      {
        float distance = 0.0f;
        long long memory_id = 0;
      };
      std::vector<Candidate> candidates;
      candidates.reserve (labels.size ());
      for (const long long memory_id : labels)
        {
          const auto delta = impl_->delta_embeddings.find (memory_id);
          const auto sealed = impl_->sealed_embeddings.find (memory_id);
          const std::vector<float> *embedding
              = delta != impl_->delta_embeddings.end ()
                    ? &delta->second
                    : sealed != impl_->sealed_embeddings.end ()
                          && impl_->removed.count (memory_id) == 0
                          && impl_->sealed_removed.count (memory_id) == 0
                    ? &sealed->second
                    : nullptr;
          if (!embedding)
            continue;
          candidates.push_back (
              { InnerProductDistance (*normalized, *embedding),
                memory_id });
        }
      std::sort (candidates.begin (), candidates.end (),
                 [] (const Candidate &left, const Candidate &right) {
                   return left.distance < right.distance
                          || (left.distance == right.distance
                              && left.memory_id < right.memory_id);
                 });
      if (candidates.size () > route_capacity)
        candidates.resize (route_capacity);
      std::vector<long long> memory_ids;
      memory_ids.reserve (candidates.size ());
      for (const auto &candidate : candidates)
        memory_ids.push_back (candidate.memory_id);
      return memory_ids;
    }
  catch (...)
    {
      return std::nullopt;
    }
}

std::optional<GraphSnapshot>
Route::Snapshot (const std::vector<long long> &roots) const
{
  if (!impl_)
    return std::nullopt;
  try
    {
      std::lock_guard<std::mutex> lock (impl_->mutex);
      GraphSnapshot snapshot;
      if (impl_->first.cur_element_count == 0)
        return snapshot;

      const auto entry_internal = impl_->first.enterpoint_node_;
      snapshot.entry_memory_id = static_cast<long long> (
          impl_->first.getExternalLabel (entry_internal));
      snapshot.max_level = impl_->first.maxlevel_;

      std::unordered_set<hnswlib::tableint> selected;
      if (roots.empty ())
        {
          selected.reserve (impl_->first.cur_element_count);
          for (hnswlib::tableint internal_id = 0;
               internal_id < impl_->first.cur_element_count; ++internal_id)
            selected.insert (internal_id);
        }
      else
        {
          selected.reserve (
              roots.size () * (impl_->parameters.graph_neighbor_count + 1));
          for (const long long memory_id : roots)
            {
              const auto found = impl_->first.label_lookup_.find (
                  static_cast<hnswlib::labeltype> (memory_id));
              if (found == impl_->first.label_lookup_.end ())
                continue;
              selected.insert (found->second);
              const int level = impl_->first.element_levels_[found->second];
              for (int graph_level = 0; graph_level <= level; ++graph_level)
                {
                  const auto neighbors = impl_->first.getConnectionsWithLock (
                      found->second, graph_level);
                  selected.insert (neighbors.begin (), neighbors.end ());
                }
            }
        }

      std::vector<hnswlib::tableint> ordered (selected.begin (),
                                               selected.end ());
      std::sort (ordered.begin (), ordered.end (), [&] (const auto left,
                                                         const auto right) {
        return impl_->first.getExternalLabel (left)
               < impl_->first.getExternalLabel (right);
      });
      snapshot.nodes.reserve (ordered.size ());
      for (const hnswlib::tableint internal_id : ordered)
        {
          const long long memory_id = static_cast<long long> (
              impl_->first.getExternalLabel (internal_id));
          const auto embedding = impl_->sealed_embeddings.find (memory_id);
          if (memory_id <= 0 || embedding == impl_->sealed_embeddings.end ())
            return std::nullopt;
          NodeSnapshot node;
          node.memory_id = memory_id;
          node.embedding = Eigen::Map<const Eigen::VectorXf> (
              embedding->second.data (), impl_->embedding_dim);
          node.level = impl_->first.element_levels_[internal_id];
          node.active = impl_->sealed_removed.count (memory_id) == 0
                        && impl_->removed.count (memory_id) == 0;
          node.links.resize (static_cast<std::size_t> (node.level + 1));
          for (int graph_level = 0; graph_level <= node.level; ++graph_level)
            {
              const auto neighbors = impl_->first.getConnectionsWithLock (
                  internal_id, graph_level);
              auto &links = node.links[static_cast<std::size_t> (graph_level)];
              links.reserve (neighbors.size ());
              for (const hnswlib::tableint neighbor : neighbors)
                links.push_back (static_cast<long long> (
                    impl_->first.getExternalLabel (neighbor)));
            }
          snapshot.nodes.push_back (std::move (node));
        }
      return snapshot;
    }
  catch (...)
    {
      return std::nullopt;
    }
}

std::size_t
Route::Size () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  std::size_t size = impl_->delta_embeddings.size ();
  for (const auto &[memory_id, embedding] : impl_->sealed_embeddings)
    {
      (void)embedding;
      if (impl_->delta_embeddings.count (memory_id) == 0
          && impl_->removed.count (memory_id) == 0
          && impl_->sealed_removed.count (memory_id) == 0)
        ++size;
    }
  return size;
}

std::size_t
Route::DeltaSize () const
{
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock (impl_->mutex);
  return impl_->delta_embeddings.size () + impl_->removed.size ();
}

} // namespace cortext::operations::sparse_retrieval_route_internal
