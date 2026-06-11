#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/operations/boundary.hpp>
#include <cortext/operations/coherence.hpp>
#include <cortext/operations/accumulator.hpp>
#include <cortext/operations/streaming_pacing.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace cortext;

namespace
{

// Tags for contract-aggregation tests.
struct TagA
{
};
struct TagB
{
};
struct TagC
{
};

struct RecordOp : IOperation
{
  RecordOp () = default;
  RecordOp (std::vector<int> *order, int id) : order (order), id (id) {}

  void
  Execute (OperationContext & /*ctx*/, Transaction & /*tx*/) const override
  {
    if (order)
      {
        order->push_back (id);
      }
  }

  std::vector<int> *order = nullptr;
  int id = 0;
};

struct RecordOp2 : RecordOp
{
  using RecordOp::RecordOp;
};

struct RecordOp3 : RecordOp
{
  using RecordOp::RecordOp;
};

struct ProducesA : Operation<Requires<>, Satisfies<TagA> >
{
  void
  Execute (OperationContext &, Transaction &) const override
  {
  }
};

struct ConsumesAProducesB : Operation<Requires<TagA>, Satisfies<TagB> >
{
  void
  Execute (OperationContext &, Transaction &) const override
  {
  }
};

struct ConsumesC : Operation<Requires<TagC>, Satisfies<> >
{
  void
  Execute (OperationContext &, Transaction &) const override
  {
  }
};

} // namespace

TEST_CASE ("OperationSet executes members in order", "[operation_set]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  OperationContext ctx (s, pctx, cfg);

  std::vector<int> order;
  OperationSet set (RecordOp (&order, 1), RecordOp2 (&order, 2),
                     RecordOp3 (&order, 3));
  set.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (order == std::vector<int>{ 1, 2, 3 });
}

TEST_CASE ("OperationSet records per-operation timings under distinct names",
           "[operation_set]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  OperationContext ctx (s, pctx, cfg);

  OperationSet<RecordOp, RecordOp2> set;
  set.Execute (ctx, cortext::testing::GetNullTransaction ());

  const auto &timings = ctx.GetOperationTimings ();
  REQUIRE (timings.size () == 2);
}

TEST_CASE ("Nested operation sets execute inline without a stage timing entry",
           "[operation_set]")
{
  Signal s;
  s.embedding = Eigen::VectorXf::Zero (2);
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  OperationContext ctx (s, pctx, cfg);

  std::vector<int> order;
  using Stage = OperationSet<RecordOp, RecordOp2>;
  OperationSet<Stage, RecordOp3> root (
      Stage (RecordOp (&order, 1), RecordOp2 (&order, 2)),
      RecordOp3 (&order, 3));
  root.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (order == std::vector<int>{ 1, 2, 3 });
  // Member ops are timed; the nested Stage wrapper itself is not.
  REQUIRE (ctx.GetOperationTimings ().size () == 3);
}

TEST_CASE ("OperationSet aggregates contracts at compile time", "[operation_set]")
{
  // Internally satisfied chain: A produced, then consumed.
  using Chained = OperationSet<ProducesA, ConsumesAProducesB>;
  STATIC_REQUIRE (Chained::kContractsComplete);
  STATIC_REQUIRE (IsSelfContained<Chained>);
  STATIC_REQUIRE (std::is_same_v<Chained::Input, Requires<> >);

  // Out of order: the consumer precedes the producer, so TagA is external.
  using Misordered = OperationSet<ConsumesAProducesB, ProducesA>;
  STATIC_REQUIRE (std::is_same_v<Misordered::Input, Requires<TagA> >);
  STATIC_REQUIRE (!IsSelfContained<Misordered>);

  // Unmet requirement surfaces as external input.
  using NeedsC = OperationSet<ProducesA, ConsumesC>;
  STATIC_REQUIRE (std::is_same_v<NeedsC::Input, Requires<TagC> >);

  // Nested operation sets propagate contracts upward.
  using Nested = OperationSet<OperationSet<ProducesA>, ConsumesAProducesB>;
  STATIC_REQUIRE (IsSelfContained<Nested>);

  // Unported members (no contract) mark aggregation incomplete.
  using Mixed = OperationSet<ProducesA, RecordOp>;
  STATIC_REQUIRE (!Mixed::kContractsComplete);
  STATIC_REQUIRE (!IsSelfContained<Mixed>);
}

TEST_CASE ("real operations declare validated contracts",
           "[operation_set][contracts]")
{
  using cortext::operations::CheckStreamingPacing;
  using cortext::operations::ComputeCoherence;
  using cortext::operations::DetectBoundary;
  using cortext::operations::UpdateAccumulator;

  // Producer before consumer is self-contained.
  using Ordered = OperationSet<ComputeCoherence, UpdateAccumulator>;
  STATIC_REQUIRE (Ordered::kContractsComplete);
  STATIC_REQUIRE (IsSelfContained<Ordered>);

  // A real consumer without its producer surfaces the missing tag: the
  // streaming-pacing gate needs the boundary flush decision.
  using MissingProducer = OperationSet<CheckStreamingPacing>;
  STATIC_REQUIRE (MissingProducer::kContractsComplete);
  STATIC_REQUIRE (!IsSelfContained<MissingProducer>);
  STATIC_REQUIRE (cortext::operation_set_detail::Contains<
                  cortext::tags::FlushRequired,
                  MissingProducer::Input>::value);

  // Supplying the producer (boundary detection needs its own inputs, so
  // the set is still not self-contained, but the flush requirement is now
  // met internally).
  using WithProducer = OperationSet<DetectBoundary, CheckStreamingPacing>;
  STATIC_REQUIRE (!cortext::operation_set_detail::Contains<
                  cortext::tags::FlushRequired,
                  WithProducer::Input>::value);
}

namespace
{

struct ForkTagA;
struct ForkTagB;
struct ForkTagC;

class ForkBranchA final
    : public Operation<Requires<>, Satisfies<ForkTagA> >
{
public:
  ForkBranchA () = default;
  ForkBranchA (std::atomic<bool> *mine, std::atomic<bool> *other,
               std::atomic<bool> *both_ran)
      : mine_ (mine), other_ (other), both_ran_ (both_ran)
  {
  }
  void
  Execute (OperationContext &, Transaction &) const override
  {
    if (!mine_)
      {
        return;
      }
    mine_->store (true);
    const auto deadline
        = std::chrono::steady_clock::now () + std::chrono::seconds (5);
    while (!other_->load ())
      {
        if (std::chrono::steady_clock::now () > deadline)
          {
            return; // both_ran_ stays false -> test fails visibly
          }
        std::this_thread::yield ();
      }
    both_ran_->store (true);
  }

private:
  std::atomic<bool> *mine_ = nullptr;
  std::atomic<bool> *other_ = nullptr;
  std::atomic<bool> *both_ran_ = nullptr;
};

class ForkBranchB final
    : public Operation<Requires<>, Satisfies<ForkTagB> >
{
public:
  ForkBranchB () = default;
  ForkBranchB (std::atomic<bool> *mine, std::atomic<bool> *other)
      : mine_ (mine), other_ (other)
  {
  }
  void
  Execute (OperationContext &, Transaction &) const override
  {
    if (!mine_)
      {
        return;
      }
    mine_->store (true);
    const auto deadline
        = std::chrono::steady_clock::now () + std::chrono::seconds (5);
    while (!other_->load ())
      {
        if (std::chrono::steady_clock::now () > deadline)
          {
            return;
          }
        std::this_thread::yield ();
      }
  }

private:
  std::atomic<bool> *mine_ = nullptr;
  std::atomic<bool> *other_ = nullptr;
};

class ForkThrows final : public Operation<Requires<>, Satisfies<ForkTagC> >
{
public:
  void
  Execute (OperationContext &, Transaction &) const override
  {
    throw std::runtime_error ("branch failure");
  }
};

class ForkConsumesA final
    : public Operation<Requires<ForkTagA>, Satisfies<> >
{
public:
  void
  Execute (OperationContext &, Transaction &) const override
  {
  }
};

class ForkTouchesTx final : public Operation<Requires<>, Satisfies<ForkTagA> >
{
public:
  ForkTouchesTx () = default;
  explicit ForkTouchesTx (int calls) : calls_ (calls) {}
  void
  Execute (OperationContext &, Transaction &tx) const override
  {
    for (int i = 0; i < calls_; ++i)
      {
        tx.Execute ("probe", {});
      }
  }

private:
  int calls_ = 0;
};

class ForkTouchesTxB final
    : public Operation<Requires<>, Satisfies<ForkTagB> >
{
public:
  ForkTouchesTxB () = default;
  explicit ForkTouchesTxB (int calls) : calls_ (calls) {}
  void
  Execute (OperationContext &, Transaction &tx) const override
  {
    for (int i = 0; i < calls_; ++i)
      {
        tx.Execute ("probe", {});
      }
  }

private:
  int calls_ = 0;
};

/// Counts overlapping Execute calls; overlap proves a serialization hole.
class OverlapProbeTransaction final : public Transaction
{
public:
  std::unique_ptr<Transaction>
  Begin () override
  {
    return nullptr;
  }
  std::vector<std::map<std::string, std::any> >
  Execute (const std::string &, const std::vector<std::any> &) override
  {
    const int now = ++in_flight_;
    max_in_flight_.store (std::max (max_in_flight_.load (), now));
    std::this_thread::sleep_for (std::chrono::microseconds (200));
    --in_flight_;
    ++total_calls_;
    return {};
  }
  void
  Commit () override
  {
  }
  void
  Rollback () override
  {
  }

  std::atomic<int> in_flight_{ 0 };
  std::atomic<int> max_in_flight_{ 0 };
  std::atomic<int> total_calls_{ 0 };
};

} // namespace

TEST_CASE ("Fork and Join contracts encode fork-join correctness",
           "[operation_set][fork]")
{
  using TheFork = Fork<ForkBranchA, ForkBranchB>;
  using TheJoin = Join<ForkBranchA, ForkBranchB>;

  // Fork satisfies nothing; outputs appear at the Join.
  STATIC_REQUIRE (std::is_same_v<TheFork::Output, Satisfies<> >);
  STATIC_REQUIRE (
      cortext::operation_set_detail::Contains<ForkTagA,
                                              TheJoin::Output>::value);
  STATIC_REQUIRE (
      cortext::operation_set_detail::Contains<ForkTagB,
                                              TheJoin::Output>::value);

  // A consumer between Fork and Join leaves the requirement external; a
  // consumer after Join is satisfied.
  using ConsumerBeforeJoin
      = OperationSet<TheFork, ForkConsumesA, TheJoin>;
  STATIC_REQUIRE (!IsSelfContained<ConsumerBeforeJoin>);
  using ConsumerAfterJoin = OperationSet<TheFork, TheJoin, ForkConsumesA>;
  STATIC_REQUIRE (IsSelfContained<ConsumerAfterJoin>);

  // Conflict proofs: disjoint branches fork; overlapping outputs or a
  // branch consuming a sibling's output do not.
  STATIC_REQUIRE (ForkBranchesConflictFree<ForkBranchA, ForkBranchB>);
  STATIC_REQUIRE (!ForkBranchesConflictFree<ForkBranchA, ForkBranchA>);
  STATIC_REQUIRE (!ForkBranchesConflictFree<ForkBranchA, ForkConsumesA>);
}

TEST_CASE ("Fork runs branches concurrently and Join is the barrier",
           "[operation_set][fork]")
{
  Signal s;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  OperationContext ctx (s, pctx, cfg);

  std::atomic<bool> a_started{ false };
  std::atomic<bool> b_started{ false };
  std::atomic<bool> both_ran{ false };

  // Each branch waits for the other: completion proves true concurrency.
  OperationSet<Fork<ForkBranchA, ForkBranchB>, Join<ForkBranchA, ForkBranchB> >
      set (Fork<ForkBranchA, ForkBranchB> (
               ForkBranchA (&a_started, &b_started, &both_ran),
               ForkBranchB (&b_started, &a_started)),
           Join<ForkBranchA, ForkBranchB> ());
  set.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (a_started.load ());
  REQUIRE (b_started.load ());
  REQUIRE (both_ran.load ());
  REQUIRE_FALSE (ctx.HasPendingForks ());
}

TEST_CASE ("Join rethrows the first branch failure", "[operation_set][fork]")
{
  Signal s;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  OperationContext ctx (s, pctx, cfg);

  OperationSet<Fork<ForkThrows, ForkBranchB>, Join<ForkThrows, ForkBranchB> >
      set;
  REQUIRE_THROWS_AS (
      set.Execute (ctx, cortext::testing::GetNullTransaction ()),
      std::runtime_error);
  REQUIRE_FALSE (ctx.HasPendingForks ());
}

TEST_CASE ("Fork serializes transaction access per method call",
           "[operation_set][fork]")
{
  Signal s;
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  OperationContext ctx (s, pctx, cfg);

  OverlapProbeTransaction probe_tx;
  constexpr int kCalls = 50;
  OperationSet<Fork<ForkTouchesTx, ForkTouchesTxB>,
               Join<ForkTouchesTx, ForkTouchesTxB> >
      set{ Fork<ForkTouchesTx, ForkTouchesTxB>{ ForkTouchesTx (kCalls),
                                                 ForkTouchesTxB (kCalls) },
           Join<ForkTouchesTx, ForkTouchesTxB>{} };
  set.Execute (ctx, probe_tx);

  REQUIRE (probe_tx.total_calls_.load () == 2 * kCalls);
  REQUIRE (probe_tx.max_in_flight_.load () == 1);
}
