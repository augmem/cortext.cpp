// tests/implicit_feedback.test.cpp
#include <Eigen/Dense>
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/detect_memory_usage.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <unordered_map>

using namespace cortext;
using cortext::operations::DetectMemoryUsage;

namespace
{

constexpr int kEmbeddingDim = 8;

static Eigen::VectorXf
MakeUnitVec (int idx)
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

// Helper op to seed retrieval candidates and interrupt decision.
class SetupRetrievalContextOp : public IOperation
{
public:
  SetupRetrievalContextOp (
      std::unordered_map<long long, Eigen::VectorXf> candidates,
      bool interrupt_allowed,
      std::optional<long long> selected_id)
      : candidates_ (std::move (candidates)),
        interrupt_allowed_ (interrupt_allowed),
        selected_id_ (selected_id)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetRetrievedMemoryEmbeddings (candidates_);
    ctx.SetInterruptAllowed (interrupt_allowed_);
    ctx.SetSelectedCandidateId (selected_id_);
  }

private:
  std::unordered_map<long long, Eigen::VectorXf> candidates_;
  bool interrupt_allowed_;
  std::optional<long long> selected_id_;
};

// Helper op to capture MemoryUsageEvents.
class CaptureUsageEventsOp : public IOperation
{
public:
  explicit CaptureUsageEventsOp (
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

TEST_CASE ("DetectMemoryUsage marks selected candidate as used",
           "[operations][feedback]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  const Eigen::VectorXf signal_vec = MakeUnitVec (0);
  const Eigen::VectorXf cand_a = MakeUnitVec (0);
  const Eigen::VectorXf cand_b = MakeUnitVec (1);

  std::unordered_map<long long, Eigen::VectorXf> candidates = {
    { 42LL, cand_a },
    { 99LL, cand_b }
  };

  std::vector<OperationContext::MemoryUsageEvent> events;

  auto setup = std::make_unique<SetupRetrievalContextOp> (
      candidates, /*interrupt_allowed=*/true, /*selected_id=*/42LL);
  auto detect = std::make_unique<DetectMemoryUsage> ();
  auto capture = std::make_unique<CaptureUsageEventsOp> (events);
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (setup), std::move (detect), std::move (capture));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (signal_vec, /*ts=*/150));

  REQUIRE (events.size () == 2);
  bool saw_used = false;
  for (const auto &ev : events)
    {
      if (ev.embedding_id == 42LL)
        {
          REQUIRE (ev.used == true);
          REQUIRE (ev.contextual_gain.has_value ());
          CHECK (ev.contextual_gain.value () > 0.99);
          saw_used = true;
        }
      else if (ev.embedding_id == 99LL)
        {
          REQUIRE (ev.used == false);
        }
    }
  REQUIRE (saw_used);
}

TEST_CASE ("DetectMemoryUsage marks none used when interrupt denied",
           "[operations][feedback]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  const Eigen::VectorXf signal_vec = MakeUnitVec (0);
  std::unordered_map<long long, Eigen::VectorXf> candidates = {
    { 7LL, MakeUnitVec (0) }
  };

  std::vector<OperationContext::MemoryUsageEvent> events;

  auto setup = std::make_unique<SetupRetrievalContextOp> (
      candidates, /*interrupt_allowed=*/false, /*selected_id=*/7LL);
  auto detect = std::make_unique<DetectMemoryUsage> ();
  auto capture = std::make_unique<CaptureUsageEventsOp> (events);
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (setup), std::move (detect), std::move (capture));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (signal_vec, /*ts=*/200));

  REQUIRE (events.size () == 1);
  REQUIRE (events[0].used == false);
}

TEST_CASE ("DetectMemoryUsage no-ops with empty candidates",
           "[operations][feedback]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  std::vector<OperationContext::MemoryUsageEvent> events;

  auto setup = std::make_unique<SetupRetrievalContextOp> (
      std::unordered_map<long long, Eigen::VectorXf>{},
      /*interrupt_allowed=*/false,
      std::nullopt);
  auto detect = std::make_unique<DetectMemoryUsage> ();
  auto capture = std::make_unique<CaptureUsageEventsOp> (events);
  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::move (setup), std::move (detect), std::move (capture));

  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (MakeUnitVec (0), /*ts=*/300));

  REQUIRE (events.empty ());
}
