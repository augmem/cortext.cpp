// tests/operations_competition.test.cpp
#include <Eigen/Dense>
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/competition.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <unordered_map>
#include <vector>

using namespace cortext;
using cortext::operations::ApplyRetrievalCompetition;

namespace
{

// Helper op to preload current context and retrieved embeddings into context.
class SetupCompetitionInputsOp : public IOperation
{
public:
  SetupCompetitionInputsOp (Eigen::VectorXf cur,
                            std::unordered_map<long long, Eigen::VectorXf> r)
      : cur_ (std::move (cur)), retrieved_ (std::move (r))
  {
  }
  void
  Execute (OperationContext &ctx) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    pctx.recent_context_embeddings.clear ();
    pctx.recent_context_embeddings.push_back (cur_);
    ctx.SetRetrievedMemoryEmbeddings (retrieved_);
  }

private:
  Eigen::VectorXf cur_;
  std::unordered_map<long long, Eigen::VectorXf> retrieved_;
};

static Signal
MakeSignal (const Eigen::VectorXf &emb, uint64_t ts)
{
  Signal s;
  s.embedding = emb;
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

inline Eigen::VectorXf
Norm (Eigen::VectorXf v)
{
  const float n = v.norm ();
  if (n <= 1e-9f)
    return v;
  return v / n;
}

} // namespace

TEST_CASE ("Alg21 inhibits near losers but not distant ones",
           "[operations][competition]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // Choose knobs to get k≈3 winners and meaningful inhibition radius.
  SignalProcessor::Config cfg;
  cfg.focus = 1.0;       // winners_k ~ 3, high inhibition radius 0.85
  cfg.sensitivity = 1.0; // stronger lateral strength
  cfg.stability = 0.0;   // larger base suppression per retrieval

  // Base context (unit x-axis).
  const Eigen::VectorXf ctx
      = Norm ((Eigen::VectorXf (3) << 1.0f, 0.0f, 0.0f).finished ());

  // Winners: very close to context
  const Eigen::VectorXf w1
      = Norm ((Eigen::VectorXf (3) << 0.99f, 0.05f, 0.0f).finished ());
  const Eigen::VectorXf w2
      = Norm ((Eigen::VectorXf (3) << 0.98f, 0.06f, 0.0f).finished ());
  const Eigen::VectorXf w3
      = Norm ((Eigen::VectorXf (3) << 0.95f, 0.10f, 0.0f).finished ());

  // Near-loser: close to winners (cos ~0.9+ with them) but slightly lower
  const Eigen::VectorXf near_l
      = Norm ((Eigen::VectorXf (3) << 0.88f, 0.47f, 0.0f).finished ());
  // Distant: orthogonal-ish
  const Eigen::VectorXf far_l
      = Norm ((Eigen::VectorXf (3) << 0.0f, 1.0f, 0.0f).finished ());

  std::unordered_map<long long, Eigen::VectorXf> retrieved{
    { 10LL, w1 }, { 11LL, w2 }, { 12LL, w3 }, { 13LL, near_l }, { 20LL, far_l }
  };

  auto setup = std::make_unique<SetupCompetitionInputsOp> (ctx, retrieved);
  auto apply = std::make_unique<ApplyRetrievalCompetition> ();
  auto pipeline
      = std::make_unique<OperationSet> (std::move (setup), std::move (apply));

  SignalProcessor processor (cfg, store, std::move (pipeline));
  processor.Process (MakeSignal (ctx, /*ts=*/100));
  processor.Flush ();

  // Near loser suppressed
  {
    auto rows = store->Execute (
        "SELECT strength FROM memory_feedback WHERE embedding_id = ?",
        { 13LL });
    REQUIRE (rows.size () == 1);
    const double strength = std::any_cast<double> (rows[0].at ("strength"));
    REQUIRE (strength < 1.0);
  }
  // Distant unchanged (row may not exist if never touched; tolerate both
  // cases)
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS c FROM memory_feedback WHERE embedding_id = ?",
        { 20LL });
    REQUIRE (rows.size () == 1);
    const auto cnt = std::any_cast<long long> (rows[0].at ("c"));
    if (cnt == 1)
      {
        auto r2 = store->Execute (
            "SELECT strength FROM memory_feedback WHERE embedding_id = ?",
            { 20LL });
        REQUIRE (r2.size () == 1);
        const double s2 = std::any_cast<double> (r2[0].at ("strength"));
        REQUIRE (s2 == Catch::Approx (1.0));
      }
  }
}

TEST_CASE ("Alg21 recovery restores strength over time",
           "[operations][competition][recovery]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // Higher stability → longer recovery_time (600s); we'll advance ts by >=600.
  SignalProcessor::Config cfg;
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 1.0; // recovery_time = 600s

  const Eigen::VectorXf ctx
      = Norm ((Eigen::VectorXf (3) << 1.0f, 0.0f, 0.0f).finished ());
  const Eigen::VectorXf w1
      = Norm ((Eigen::VectorXf (3) << 0.99f, 0.05f, 0.0f).finished ());
  const Eigen::VectorXf w2
      = Norm ((Eigen::VectorXf (3) << 0.98f, 0.06f, 0.0f).finished ());
  const Eigen::VectorXf w3
      = Norm ((Eigen::VectorXf (3) << 0.95f, 0.12f, 0.0f).finished ());
  const Eigen::VectorXf near_l
      = Norm ((Eigen::VectorXf (3) << 0.90f, 0.44f, 0.0f).finished ());

  std::unordered_map<long long, Eigen::VectorXf> retrieved{
    { 1LL, w1 }, { 3LL, w2 }, { 4LL, w3 }, { 2LL, near_l }
  };

  // First pass to apply suppression.
  {
    auto setup = std::make_unique<SetupCompetitionInputsOp> (ctx, retrieved);
    auto apply = std::make_unique<ApplyRetrievalCompetition> ();
    auto pipeline = std::make_unique<OperationSet> (std::move (setup),
                                                    std::move (apply));
    SignalProcessor processor (cfg, store, std::move (pipeline));
    processor.Process (MakeSignal (ctx, /*ts=*/1000));
    processor.Flush ();
  }

  double strength_after_supp = 0.0;
  {
    auto rows = store->Execute (
        "SELECT strength FROM memory_feedback WHERE embedding_id = ?",
        { 2LL });
    REQUIRE (rows.size () == 1);
    strength_after_supp = std::any_cast<double> (rows[0].at ("strength"));
    REQUIRE (strength_after_supp < 1.0);
  }

  // Second pass well after recovery_time (>=600s) should restore some
  // strength.
  {
    cfg.focus = 0.0; // make all candidates winners (k=7) → no new suppression
    auto setup = std::make_unique<SetupCompetitionInputsOp> (ctx, retrieved);
    auto apply = std::make_unique<ApplyRetrievalCompetition> ();
    auto pipeline = std::make_unique<OperationSet> (std::move (setup),
                                                    std::move (apply));
    SignalProcessor processor (cfg, store, std::move (pipeline));
    processor.Process (MakeSignal (ctx, /*ts=*/2000)); // +1000s
    processor.Flush ();
  }

  {
    auto rows = store->Execute (
        "SELECT strength FROM memory_feedback WHERE embedding_id = ?",
        { 2LL });
    REQUIRE (rows.size () == 1);
    const double strength_after_recovery
        = std::any_cast<double> (rows[0].at ("strength"));
    REQUIRE (strength_after_recovery > strength_after_supp);
    REQUIRE (strength_after_recovery <= 1.0);
  }
}
