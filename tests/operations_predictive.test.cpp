// tests/operations_predictive.test.cpp
#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/predictive.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <unordered_map>
#include <vector>

using namespace cortext;
using cortext::operations::ApplyPredictivePreActivation;

namespace
{

// Helper op to preload recent context trajectory and retrieved embeddings.
class SetupPredictiveInputsOp : public IOperation
{
public:
  SetupPredictiveInputsOp (std::vector<Eigen::VectorXf> recent,
                           std::unordered_map<long long, Eigen::VectorXf> r)
      : recent_ (std::move (recent)), retrieved_ (std::move (r))
  {
  }

  void
  Execute (OperationContext &ctx) const override
  {
    auto &pctx = ctx.GetProcessorContext ();
    pctx.recent_context_embeddings.clear ();
    for (const auto &e : recent_)
      {
        pctx.recent_context_embeddings.push_back (e);
      }
    ctx.SetRetrievedMemoryEmbeddings (retrieved_);
  }

private:
  std::vector<Eigen::VectorXf> recent_;
  std::unordered_map<long long, Eigen::VectorXf> retrieved_;
};

inline Eigen::VectorXf
Norm (Eigen::VectorXf v)
{
  const float n = v.norm ();
  if (n <= 1e-9f)
    return v;
  return v / n;
}

static Signal
MakeSignal (const Eigen::VectorXf &emb, uint64_t ts)
{
  Signal s;
  s.embedding = emb;
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

} // namespace

TEST_CASE ("Alg22 boosts predicted-aligned candidates",
           "[operations][predictive]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;       // trajectory_samples ~ 3, conf ~ 0.5
  cfg.sensitivity = 0.5; // moderate surprise sens
  cfg.stability = 0.5;   // moderate decay → moderate delta

  // Recent context shows a gentle trend toward +y.
  const Eigen::VectorXf e0
      = Norm ((Eigen::VectorXf (3) << 1.0f, 0.0f, 0.0f).finished ());
  const Eigen::VectorXf e1
      = Norm ((Eigen::VectorXf (3) << 1.0f, 0.10f, 0.0f).finished ());
  const Eigen::VectorXf e2
      = Norm ((Eigen::VectorXf (3) << 1.0f, 0.20f, 0.0f).finished ());
  std::vector<Eigen::VectorXf> recent{ e0, e1, e2 };

  // Candidate aligned with predicted (near average of recent).
  const Eigen::VectorXf aligned
      = Norm ((Eigen::VectorXf (3) << 1.0f, 0.22f, 0.0f).finished ());
  // Orthogonal-ish candidate.
  const Eigen::VectorXf orth
      = Norm ((Eigen::VectorXf (3) << 0.0f, 1.0f, 0.0f).finished ());

  std::unordered_map<long long, Eigen::VectorXf> retrieved{ { 101LL, aligned },
                                                            { 202LL, orth } };

  auto setup = std::make_unique<SetupPredictiveInputsOp> (recent, retrieved);
  auto apply = std::make_unique<ApplyPredictivePreActivation> ();
  auto pipeline
      = std::make_unique<OperationSet> (std::move (setup), std::move (apply));

  SignalProcessor processor (cfg, store, std::move (pipeline));
  processor.Process (MakeSignal (e2, /*ts=*/123));
  processor.Flush ();

  // Aligned candidate should have strength > 1.0
  {
    auto rows = store->Execute (
        "SELECT strength FROM memory_feedback WHERE embedding_id = ?",
        { 101LL });
    REQUIRE (rows.size () == 1);
    const double strength = std::any_cast<double> (rows[0].at ("strength"));
    REQUIRE (strength > 1.0);
  }
  // Orthogonal may not be touched; accept either no row or unchanged.
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS c FROM memory_feedback WHERE embedding_id = ?",
        { 202LL });
    REQUIRE (rows.size () == 1);
    const auto cnt = std::any_cast<long long> (rows[0].at ("c"));
    if (cnt == 1)
      {
        auto r2 = store->Execute (
            "SELECT strength FROM memory_feedback WHERE embedding_id = ?",
            { 202LL });
        REQUIRE (r2.size () == 1);
        const double s2 = std::any_cast<double> (r2[0].at ("strength"));
        REQUIRE (s2 == Catch::Approx (1.0));
      }
  }
}

TEST_CASE ("Alg22 respects prediction confidence threshold",
           "[operations][predictive][threshold]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cfg.focus = 1.0;       // conf_thresh ~ 0.7
  cfg.sensitivity = 0.0; // no surprise modulation
  cfg.stability = 0.5;

  // Recent contexts along x-axis.
  const Eigen::VectorXf e0
      = Norm ((Eigen::VectorXf (3) << 1.0f, 0.0f, 0.0f).finished ());
  const Eigen::VectorXf e1
      = Norm ((Eigen::VectorXf (3) << 1.0f, 0.0f, 0.0f).finished ());
  const Eigen::VectorXf e2
      = Norm ((Eigen::VectorXf (3) << 1.0f, 0.0f, 0.0f).finished ());
  std::vector<Eigen::VectorXf> recent{ e0, e1, e2 };

  // Candidate with cosine 0.6 vs x-axis (below 0.7 threshold).
  const Eigen::VectorXf below
      = Norm ((Eigen::VectorXf (3) << 0.6f, 0.8f, 0.0f).finished ());

  std::unordered_map<long long, Eigen::VectorXf> retrieved{ { 303LL, below } };

  auto setup = std::make_unique<SetupPredictiveInputsOp> (recent, retrieved);
  auto apply = std::make_unique<ApplyPredictivePreActivation> ();
  auto pipeline
      = std::make_unique<OperationSet> (std::move (setup), std::move (apply));

  SignalProcessor processor (cfg, store, std::move (pipeline));
  processor.Process (MakeSignal (e2, /*ts=*/456));
  processor.Flush ();

  // Expect no row (or unchanged strength if row exists).
  auto rows = store->Execute (
      "SELECT COUNT(*) AS c FROM memory_feedback WHERE embedding_id = ?",
      { 303LL });
  REQUIRE (rows.size () == 1);
  const auto cnt = std::any_cast<long long> (rows[0].at ("c"));
  if (cnt == 1)
    {
      auto r2 = store->Execute (
          "SELECT strength FROM memory_feedback WHERE embedding_id = ?",
          { 303LL });
      REQUIRE (r2.size () == 1);
      const double s = std::any_cast<double> (r2[0].at ("strength"));
      REQUIRE (s == Catch::Approx (1.0));
    }
}
