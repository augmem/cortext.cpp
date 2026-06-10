// Steady-state live-loop benchmark for profiling the core operation
// pipeline. Drives the exact production DynamicOperationSet (BuildPipelineRoot)
// through one long-lived store with deterministic synthetic embeddings, so
// CPU profiles concentrate on the memory algorithms rather than per-scenario
// setup, encoder inference, or LLM consolidation backends.
//
// Output is fully deterministic: aggregate counters and checksums are
// printed so before/after comparisons can assert bit-identical semantics
// across implementation-level optimizations.

#include <cortext/clock.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace cortext
{
// Production pipeline factory defined in src/cortext.cpp.
std::unique_ptr<IOperation> BuildPipelineRoot (bool probe_mode);
} // namespace cortext

namespace
{

constexpr int kEmbeddingDim = 256;
constexpr int kTopicCount = 16;
constexpr int kTopicRunLength = 25;
constexpr int kDefaultSignalCount = 4000;
constexpr int kConsolidationInterval = 500;
constexpr std::uint64_t kStartTimestampMs = 1000000;
constexpr std::uint64_t kStepMs = 1000;

// Deterministic 64-bit LCG (MMIX constants).
class Lcg
{
public:
  explicit Lcg (std::uint64_t seed) : state_ (seed) {}

  std::uint64_t
  Next ()
  {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return state_;
  }

  // Uniform in [-1, 1).
  float
  NextSymmetric ()
  {
    return static_cast<float> (
               static_cast<double> (Next () >> 11)
               / static_cast<double> (1ULL << 53))
               * 2.0f
           - 1.0f;
  }

private:
  std::uint64_t state_;
};

class BenchEncoder : public cortext::Encoder
{
public:
  void
  EncodeText (const std::string &, std::vector<float> &out) override
  {
    out.assign (kEmbeddingDim, 0.0f);
    out[0] = 1.0f;
  }

  void
  EncodeAudio (const float *, std::size_t, std::vector<float> &out) override
  {
    out.assign (kEmbeddingDim, 0.0f);
    out[0] = 1.0f;
  }

  void
  EncodeImage (const std::uint8_t *, int, int, int,
               std::vector<float> &out) override
  {
    out.assign (kEmbeddingDim, 0.0f);
    out[0] = 1.0f;
  }
};

// Fixed clock so any wall-clock reads inside the pipeline are deterministic.
class FixedClock : public cortext::Clock
{
public:
  std::uint64_t
  NowMillis () const override
  {
    return now_ms_;
  }

  void
  Set (std::uint64_t ms)
  {
    now_ms_ = ms;
  }

private:
  std::uint64_t now_ms_ = kStartTimestampMs;
};

Eigen::VectorXf
MakeTopicCentroid (int topic, Lcg &rng)
{
  (void)topic;
  Eigen::VectorXf v (kEmbeddingDim);
  for (int i = 0; i < kEmbeddingDim; ++i)
    {
      v[i] = rng.NextSymmetric ();
    }
  v.normalize ();
  return v;
}

long long
CountRows (cortext::Store &store, const char *table)
{
  auto rows = store.Execute (std::string ("SELECT COUNT(*) AS cnt FROM ")
                                 + table,
                             {});
  return std::any_cast<long long> (rows[0].at ("cnt"));
}

} // namespace

int
main (int argc, char **argv)
{
  int signal_count = kDefaultSignalCount;
  if (argc > 1)
    {
      signal_count = std::max (1, std::atoi (argv[1]));
    }

  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (":memory:"));
  cortext::store::ApplyMigrations (*store);

  BenchEncoder encoder;
  auto clock = std::make_shared<FixedClock> ();

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &encoder;
  cfg.clock = clock;

  cortext::SignalProcessor processor (cfg, store,
                                      cortext::BuildPipelineRoot (false),
                                      nullptr);

  Lcg rng (0x5DEECE66DULL);
  std::vector<Eigen::VectorXf> topics;
  topics.reserve (kTopicCount);
  for (int t = 0; t < kTopicCount; ++t)
    {
      topics.push_back (MakeTopicCentroid (t, rng));
    }

  long long writes = 0;
  long long boundaries = 0;
  long long interrupts = 0;
  long long candidates_total = 0;
  double threshold_sum = 0.0;
  double focus_sum = 0.0;

  const auto started = std::chrono::steady_clock::now ();

  std::uint64_t ts = kStartTimestampMs;
  for (int i = 0; i < signal_count; ++i)
    {
      ts += kStepMs;
      clock->Set (ts);

      // Topic schedule: blocks of kTopicRunLength signals; every fourth
      // block revisits an earlier topic to exercise retrieval and
      // reinforcement against established memories.
      const int block = i / kTopicRunLength;
      int topic = block % kTopicCount;
      if ((block % 4) == 3)
        {
          topic = (block / 4) % kTopicCount;
        }

      Eigen::VectorXf e = topics[static_cast<std::size_t> (topic)];
      for (int d = 0; d < kEmbeddingDim; ++d)
        {
          e[d] += 0.25f * rng.NextSymmetric ();
        }
      e.normalize ();

      cortext::Signal signal;
      signal.embedding = e;
      signal.timestamp = ts;
      signal.source_id = "bench/steady";
      signal.modality = "text";

      if ((i + 1) % kConsolidationInterval == 0)
        {
          signal.consolidation_mode = cortext::ConsolidationMode::Shallow;
        }

      const auto out = processor.Process (signal);
      writes += out.write_decision ? 1 : 0;
      boundaries += out.at_boundary ? 1 : 0;
      interrupts += out.interrupt_allowed ? 1 : 0;
      candidates_total
          += static_cast<long long> (out.candidate_memory_ids.size ());
      threshold_sum += out.threshold_T_dynamic;
      focus_sum += out.effective_focus;
    }
  processor.Flush ();

  const auto elapsed
      = std::chrono::duration_cast<std::chrono::duration<double> > (
            std::chrono::steady_clock::now () - started)
            .count ();

  std::cout << std::fixed << std::setprecision (6);
  std::cout << "steady_signals=" << signal_count << "\n";
  std::cout << "steady_writes=" << writes << "\n";
  std::cout << "steady_boundaries=" << boundaries << "\n";
  std::cout << "steady_interrupts=" << interrupts << "\n";
  std::cout << "steady_candidates_total=" << candidates_total << "\n";
  std::cout << "steady_threshold_sum=" << threshold_sum << "\n";
  std::cout << "steady_effective_focus_sum=" << focus_sum << "\n";
  std::cout << "steady_final_memories=" << CountRows (*store, "memories")
            << "\n";
  std::cout << "steady_final_associations="
            << CountRows (*store, "associations") << "\n";
  std::cout << "steady_final_embeddings=" << CountRows (*store, "embeddings")
            << "\n";

  // Timing lines are intentionally written to stderr so stdout stays
  // byte-stable for before/after semantic comparisons.
  std::cerr << std::fixed << std::setprecision (3);
  std::cerr << "steady_elapsed_s=" << elapsed << "\n";
  std::cerr << "steady_signals_per_s="
            << (elapsed > 0.0 ? signal_count / elapsed : 0.0) << "\n";

  return 0;
}
