// tests/implicit_feedback.test.cpp
#include <Eigen/Dense>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/detect_memory_usage.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;
using cortext::operations::DetectMemoryUsage;

namespace
{

constexpr int kEmbeddingDim = 256;

/// @brief Creates a 256-dim unit vector with value at specified index.
static Eigen::VectorXf
MakeUnitVec256 (int idx)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[idx] = 1.0f;
  return v;
}

static Signal
MakeSignal (const Eigen::VectorXf &emb, uint64_t ts = 1)
{
  Signal s;
  s.embedding = emb;
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

// Helper op to populate the retrieval cache with embeddings.
class SetupRetrievalCacheOp : public IOperation
{
public:
  SetupRetrievalCacheOp (
      std::vector<ProcessorContext::CachedRetrieval> cached_retrievals)
      : cached_retrievals_ (std::move (cached_retrievals))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    pctx.recent_retrievals_cache.clear ();
    for (const auto &cr : cached_retrievals_)
      {
        pctx.recent_retrievals_cache.push_back (cr);
      }
  }

private:
  std::vector<ProcessorContext::CachedRetrieval> cached_retrievals_;
};

// Helper op to capture and verify the generated MemoryUsageEvents.
class CaptureUsageEventsOp : public IOperation
{
public:
  CaptureUsageEventsOp (
      std::vector<OperationContext::MemoryUsageEvent> &out_events)
      : out_events_ (out_events)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    out_events_ = ctx.GetMemoryUsageEvents ();
  }

private:
  std::vector<OperationContext::MemoryUsageEvent> &out_events_;
};

} // namespace

TEST_CASE ("DetectMemoryUsage detects high similarity as used",
           "[operations][feedback]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;       // Low focus → low threshold (0.5)
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Create a cached retrieval with the same unit vector as signal
  const Eigen::VectorXf vec = MakeUnitVec256 (0);
  std::vector<ProcessorContext::CachedRetrieval> cached = {
    { 42LL, vec, 100 } // embedding_id=42, retrieved at ts=100
  };

  std::vector<OperationContext::MemoryUsageEvent> events;

  auto setup = std::make_unique<SetupRetrievalCacheOp> (cached);
  auto detect = std::make_unique<DetectMemoryUsage> ();
  auto capture = std::make_unique<CaptureUsageEventsOp> (events);
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (setup), std::move (detect), std::move (capture));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  // Process signal with identical embedding (similarity = 1.0)
  auto s = MakeSignal (vec, /*ts=*/150);
  processor.Process (s);

  REQUIRE (events.size () == 1);
  CHECK (events[0].embedding_id == 42LL);
  CHECK (events[0].used == true);
  REQUIRE (events[0].contextual_gain.has_value ());
  CHECK (events[0].contextual_gain.value () > 0.99); // similarity ~1.0
}

TEST_CASE ("DetectMemoryUsage marks low similarity as not used",
           "[operations][feedback]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;       // High focus → high threshold (0.8)
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Create cached retrieval with orthogonal vector
  const Eigen::VectorXf cached_vec = MakeUnitVec256 (0);
  const Eigen::VectorXf signal_vec = MakeUnitVec256 (1); // orthogonal
  std::vector<ProcessorContext::CachedRetrieval> cached = {
    { 99LL, cached_vec, 100 }
  };

  std::vector<OperationContext::MemoryUsageEvent> events;

  auto setup = std::make_unique<SetupRetrievalCacheOp> (cached);
  auto detect = std::make_unique<DetectMemoryUsage> ();
  auto capture = std::make_unique<CaptureUsageEventsOp> (events);
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (setup), std::move (detect), std::move (capture));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  // Process signal with orthogonal embedding (similarity = 0.0)
  auto s = MakeSignal (signal_vec, /*ts=*/150);
  processor.Process (s);

  REQUIRE (events.size () == 1);
  CHECK (events[0].embedding_id == 99LL);
  CHECK (events[0].used == false);
  CHECK_FALSE (events[0].contextual_gain.has_value ()); // no gain for unused
}

TEST_CASE ("DetectMemoryUsage filters stale cache entries by stability",
           "[operations][feedback]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.0; // Low stability → short cache duration (30s)

  const Eigen::VectorXf vec = MakeUnitVec256 (0);

  // Create cached retrieval that is too old (100 seconds ago, beyond 30s limit)
  std::vector<ProcessorContext::CachedRetrieval> cached = {
    { 7LL, vec, 50000 } // retrieved at ts=50s (ms timestamps)
  };

  std::vector<OperationContext::MemoryUsageEvent> events;

  auto setup = std::make_unique<SetupRetrievalCacheOp> (cached);
  auto detect = std::make_unique<DetectMemoryUsage> ();
  auto capture = std::make_unique<CaptureUsageEventsOp> (events);
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (setup), std::move (detect), std::move (capture));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  // Process signal at ts=150s (100 seconds after retrieval, > 30s cache
  // duration)
  auto s = MakeSignal (vec, /*ts=*/150000);
  processor.Process (s);

  // Entry should be filtered out due to age
  CHECK (events.empty ());
}

TEST_CASE ("DetectMemoryUsage handles multiple cached retrievals",
           "[operations][feedback]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;       // Medium threshold (~0.65)
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0; // High stability → long cache duration (300s)

  const Eigen::VectorXf signal_vec = MakeUnitVec256 (0);
  const Eigen::VectorXf similar_vec = MakeUnitVec256 (0); // identical
  const Eigen::VectorXf dissimilar_vec = MakeUnitVec256 (100); // orthogonal

  std::vector<ProcessorContext::CachedRetrieval> cached = {
    { 1LL, similar_vec, 100 },    // should be marked used
    { 2LL, dissimilar_vec, 100 }, // should NOT be marked used
    { 3LL, similar_vec, 100 }     // should be marked used
  };

  std::vector<OperationContext::MemoryUsageEvent> events;

  auto setup = std::make_unique<SetupRetrievalCacheOp> (cached);
  auto detect = std::make_unique<DetectMemoryUsage> ();
  auto capture = std::make_unique<CaptureUsageEventsOp> (events);
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (setup), std::move (detect), std::move (capture));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  auto s = MakeSignal (signal_vec, /*ts=*/150);
  processor.Process (s);

  REQUIRE (events.size () == 3);

  // Find events by embedding_id
  auto find_event = [&events] (long long id) {
    for (const auto &e : events)
      {
        if (e.embedding_id == id)
          return e;
      }
    return OperationContext::MemoryUsageEvent{ -1, false, std::nullopt };
  };

  auto e1 = find_event (1LL);
  auto e2 = find_event (2LL);
  auto e3 = find_event (3LL);

  CHECK (e1.used == true);
  CHECK (e2.used == false);
  CHECK (e3.used == true);
}

TEST_CASE ("DetectMemoryUsage no-ops with empty cache",
           "[operations][feedback]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Empty cache
  std::vector<ProcessorContext::CachedRetrieval> cached = {};

  std::vector<OperationContext::MemoryUsageEvent> events;

  auto setup = std::make_unique<SetupRetrievalCacheOp> (cached);
  auto detect = std::make_unique<DetectMemoryUsage> ();
  auto capture = std::make_unique<CaptureUsageEventsOp> (events);
  auto ops = std::make_unique<cortext::OperationSet> (
      std::move (setup), std::move (detect), std::move (capture));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  const Eigen::VectorXf vec = MakeUnitVec256 (0);
  auto s = MakeSignal (vec, /*ts=*/150);
  processor.Process (s);

  CHECK (events.empty ());
}
