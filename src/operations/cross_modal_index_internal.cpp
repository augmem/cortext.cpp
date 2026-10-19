#include "cross_modal_index_internal.hpp"

#include "cortext/core/utils.hpp"

#include <any>
#include <cstring>
#include <string>
#include <typeinfo>

namespace cortext::operations::cross_modal_index_internal
{
namespace
{

struct State
{
  std::map<std::string, Population> by_modality;
};

struct RegistryState
{
  std::mutex mutex;
  std::unordered_map<const ProcessorContext *, std::shared_ptr<State>> states;
};

RegistryState &
Registry ()
{
  // Process-lifetime owner; SignalProcessor teardown erases its entry.
  static RegistryState *registry = new RegistryState ();
  return *registry;
}

long long
Int64 (const std::map<std::string, std::any> &row, const char *key)
{
  const auto it = row.find (key);
  if (it == row.end () || it->second.type () != typeid (long long))
    return 0;
  return std::any_cast<long long> (it->second);
}

// LONG_TERM signal rows of one modality above a signal id, in id order.
// CROSS JOIN pins signals as the outer loop (a primary-key range); left to
// the planner, the modality index drives and every sync visits the whole
// modality.
constexpr const char *kRowsAbove
    = "SELECT s.signal_id AS signal_id, s.memory_id AS memory_id, "
      "v.embedding AS embedding "
      "FROM signals s "
      "CROSS JOIN memory_signal_vectors v ON v.blob_id = s.blob_id "
      "CROSS JOIN memories m ON m.memory_id = s.memory_id "
      "WHERE s.signal_id > ? AND v.modality = ? AND m.kind = 'LONG_TERM' "
      "ORDER BY s.signal_id";

// The scan mark row, if it is still the same LONG_TERM signal row (any
// modality).
constexpr const char *kMarkRow
    = "SELECT s.memory_id AS memory_id "
      "FROM signals s CROSS JOIN memories m ON m.memory_id = s.memory_id "
      "WHERE s.signal_id = ? AND m.kind = 'LONG_TERM'";

// The highest LONG_TERM signal row (any modality) above the mark: walks down
// from the top of the signals key, past at most the working-memory tail.
constexpr const char *kTopRowAbove
    = "SELECT s.signal_id AS signal_id, s.memory_id AS memory_id "
      "FROM signals s CROSS JOIN memories m ON m.memory_id = s.memory_id "
      "WHERE s.signal_id > ? AND m.kind = 'LONG_TERM' "
      "ORDER BY s.signal_id DESC LIMIT 1";

void
Append (Population &population, Transaction &tx, const std::string &modality)
{
  // Read the new mark first: rows above it can only be working rows, which
  // never join the population, so nothing is missed between the reads.
  const auto top = tx.Execute (kTopRowAbove, { population.mark_signal_id });
  if (top.empty ())
    return;
  const long long next_mark = Int64 (top.front (), "signal_id");
  const long long next_mark_memory = Int64 (top.front (), "memory_id");
  Eigen::VectorXf vec;
  tx.ExecuteEach (
      kRowsAbove, { population.mark_signal_id, modality },
      [&] (const std::map<std::string, std::any> &row) {
        const long long signal_id = Int64 (row, "signal_id");
        const auto emb = row.find ("embedding");
        if (signal_id > next_mark || emb == row.end ()
            || !core::DecodeFloatBlob (emb->second, population.dim, vec))
          return;
        const std::size_t offset = population.rows.size ();
        population.rows.resize (offset
                                + static_cast<std::size_t> (population.dim));
        std::memcpy (population.rows.data () + offset, vec.data (),
                     sizeof (float)
                         * static_cast<std::size_t> (population.dim));
        population.norms.push_back (vec.norm ());
        population.memory_ids.push_back (Int64 (row, "memory_id"));
      });
  population.mark_signal_id = next_mark;
  population.mark_memory_id = next_mark_memory;
}

bool
MarkIntact (const Population &population, Transaction &tx)
{
  if (population.mark_signal_id == 0)
    return true;
  const auto rows = tx.Execute (kMarkRow, { population.mark_signal_id });
  return rows.size () == 1
         && Int64 (rows.front (), "memory_id") == population.mark_memory_id;
}

} // namespace

void
ScoreModality (const ProcessorContext &ctx, Transaction &tx,
               const std::string &modality, const Eigen::VectorXf &query,
               std::vector<std::pair<long long, double>> &out)
{
  const int dim = static_cast<int> (query.size ());
  if (dim <= 0)
    return;
  std::shared_ptr<State> state;
  {
    auto &registry = Registry ();
    std::lock_guard<std::mutex> lock (registry.mutex);
    auto &slot = registry.states[&ctx];
    if (!slot)
      slot = std::make_shared<State> ();
    state = slot;
  }
  auto &population = state->by_modality[modality];
  if (population.dim != dim || !MarkIntact (population, tx))
    {
      population = Population{};
      population.dim = dim;
    }
  try
    {
      Append (population, tx, modality);
    }
  catch (...)
    {
      // A partial append is a valid prefix only if the store finished the
      // read; start over next time instead of guessing.
      population = Population{};
      throw;
    }

  // Former path: Cosine (query, vec) = double (q.dot (v)) /
  // double (q.norm () * v.norm ()), zero when either side is empty or the
  // denominator <= 1e-12.
  const float query_norm = query.norm ();
  using AlignedRow = Eigen::Map<const Eigen::VectorXf, Eigen::Aligned16>;
  out.reserve (out.size () + population.size ());
  for (std::size_t i = 0; i < population.size (); ++i)
    {
      const AlignedRow row (population.rows.data ()
                                + i * static_cast<std::size_t> (dim),
                            dim);
      const double denom
          = static_cast<double> (query_norm * population.norms[i]);
      const double cosine
          = denom <= 1e-12 ? 0.0
                           : static_cast<double> (query.dot (row)) / denom;
      out.emplace_back (population.memory_ids[i], cosine);
    }
}

std::size_t
ResidentRows (const ProcessorContext &ctx, const std::string &modality)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  const auto it = registry.states.find (&ctx);
  if (it == registry.states.end ())
    return 0;
  const auto population = it->second->by_modality.find (modality);
  return population == it->second->by_modality.end ()
             ? 0
             : population->second.size ();
}

void
Erase (const ProcessorContext &ctx)
{
  auto &registry = Registry ();
  std::lock_guard<std::mutex> lock (registry.mutex);
  registry.states.erase (&ctx);
}

} // namespace cortext::operations::cross_modal_index_internal
