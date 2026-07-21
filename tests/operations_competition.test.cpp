// tests/operations_competition.test.cpp
#include "test_helpers.hpp"
#include "../src/operations/rif_active_epoch_cache_internal.hpp"
#include "../src/operations/rif_state_internal.hpp"
#include "../src/operations/signal_record_rollback_internal.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/competition.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace cortext::store
{
void DebugApplyCoreMigrationsThroughForTest (Store &store,
                                             std::int64_t maximum_id);
}

using namespace cortext;
using cortext::operations::ApplyRetrievalCompetition;

namespace
{

constexpr int kEmbeddingDim = 256;

std::size_t
MidpointRifRowBatch ()
{
  return cortext::operations::rif_active_epoch_cache_internal::DeriveLimits (
             0.5, 0.5, 0.5)
      .row_batch_size;
}

inline Eigen::VectorXf
Norm (Eigen::VectorXf v)
{
  const float n = v.norm ();
  if (n <= 1e-9f)
    return v;
  return v / n;
}

// Create a 256-dim embedding with values at specified indices
inline Eigen::VectorXf
Make256DEmb (std::initializer_list<std::pair<int, float>> values)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  for (const auto &[idx, val] : values)
    {
      if (idx >= 0 && idx < kEmbeddingDim)
        v[idx] = val;
    }
  return Norm (v);
}

// Helper op to seed embeddings and memories into the v2 database.
class SeedEmbeddingsOp : public IOperation
{
public:
  explicit SeedEmbeddingsOp (std::unordered_map<long long, Eigen::VectorXf> embs)
      : embeddings_ (std::move (embs))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto *store = ctx.GetStore ();
    auto now_ts = cortext::testing::NowMs ();
    for (const auto &[id, emb] : embeddings_)
      {
        std::vector<float> vec (emb.data (), emb.data () + emb.size ());
        // v2: Insert into embeddings (minimal vec0 table)
        store->Execute (
            "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
            "VALUES(?, ?, ?)",
            { id, vec, now_ts });
        // v2: Insert into memories (comprehensive metadata)
        store->Execute (
            "INSERT OR REPLACE INTO memories("
            "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
            "s_max, s_avg, strength, use_frequency, stability, connectivity, drift_mag, "
            "influence, sustained_influence, contextual_gain, redundancy, "
            "pre_activation, lability_state, suppression_count, created_at) "
            "VALUES(?, ?, 'test', 'LONG_TERM', ?, 1, 'text', 0.5, 0.5, "
            "1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, ?)",
            { id, id, now_ts, now_ts });
      }
  }

private:
  std::unordered_map<long long, Eigen::VectorXf> embeddings_;
};

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
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
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

class SetNeuromodOp : public IOperation
{
public:
  explicit SetNeuromodOp (double ne) : ne_ (ne) {}

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.GetProcessorContext ().neuromod_ne = ne_;
  }

private:
  double ne_ = 0.0;
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

} // namespace

TEST_CASE ("Alg21 inhibits near losers but not distant ones",
           "[operations][competition]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // Choose knobs to get k≈3 winners and meaningful inhibition radius.
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;       // winners_k ~ 3, high inhibition radius 0.85
  cfg.sensitivity = 1.0; // stronger lateral strength
  cfg.stability = 0.0;   // larger base suppression per retrieval

  // Base context (unit x-axis) - 256D vectors
  const Eigen::VectorXf ctx = Make256DEmb ({ { 0, 1.0f } });

  // Winners: very close to context
  const Eigen::VectorXf w1 = Make256DEmb ({ { 0, 0.99f }, { 1, 0.05f } });
  const Eigen::VectorXf w2 = Make256DEmb ({ { 0, 0.98f }, { 1, 0.06f } });
  const Eigen::VectorXf w3 = Make256DEmb ({ { 0, 0.95f }, { 1, 0.10f } });

  // Near-loser: close to winners (cos ~0.9+ with them) but slightly lower
  const Eigen::VectorXf near_l = Make256DEmb ({ { 0, 0.88f }, { 1, 0.47f } });
  // Distant: orthogonal-ish
  const Eigen::VectorXf far_l = Make256DEmb ({ { 1, 1.0f } });

  std::unordered_map<long long, Eigen::VectorXf> retrieved{
    { 10LL, w1 }, { 11LL, w2 }, { 12LL, w3 }, { 13LL, near_l }, { 20LL, far_l }
  };

  auto seed = std::make_unique<SeedEmbeddingsOp> (retrieved);
  auto setup = std::make_unique<SetupCompetitionInputsOp> (ctx, retrieved);
  auto apply = std::make_unique<ApplyRetrievalCompetition> ();
  auto pipeline = std::make_unique<DynamicOperationSet> (
      std::move (seed), std::move (setup), std::move (apply));

  SignalProcessor processor (cfg, store, std::move (pipeline));
  processor.Process (MakeSignal (ctx, /*ts=*/100));
  processor.Flush ();

  // v2: Near loser suppressed - query memories table
  {
    auto rows = store->Execute (
        "SELECT strength FROM memories WHERE memory_id = ?",
        { 13LL });
    REQUIRE (rows.size () == 1);
    const double strength = std::any_cast<double> (rows[0].at ("strength"));
    REQUIRE (strength < 1.0);
  }
  // v2: Distant unchanged - query memories table
  {
    auto rows = store->Execute (
        "SELECT COUNT(*) AS c FROM memories WHERE memory_id = ?",
        { 20LL });
    REQUIRE (rows.size () == 1);
    const auto cnt = std::any_cast<long long> (rows[0].at ("c"));
    if (cnt == 1)
      {
        auto r2 = store->Execute (
            "SELECT strength FROM memories WHERE memory_id = ?",
            { 20LL });
        REQUIRE (r2.size () == 1);
        const double s2 = std::any_cast<double> (r2[0].at ("strength"));
        REQUIRE (s2 == Catch::Approx (1.0));
      }
  }
}

TEST_CASE ("Alg21 structured retrieval suppresses shared embedding loser by memory id",
           "[operations][competition]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  const Eigen::VectorXf ctx_vec = Make256DEmb ({ { 0, 1.0f } });
  const Eigen::VectorXf w1 = Make256DEmb ({ { 0, 0.99f }, { 1, 0.05f } });
  const Eigen::VectorXf w2 = Make256DEmb ({ { 0, 0.98f }, { 1, 0.06f } });
  const Eigen::VectorXf w3 = Make256DEmb ({ { 0, 0.95f }, { 1, 0.10f } });
  const Eigen::VectorXf loser = Make256DEmb ({ { 0, 0.88f }, { 1, 0.47f } });

  std::unordered_map<long long, Eigen::VectorXf> base {
    { 1LL, w1 }, { 2LL, w2 }, { 3LL, w3 }, { 420LL, loser }
  };
  SeedEmbeddingsOp seed (base);
  Signal signal = MakeSignal (ctx_vec, 100);
  ProcessorContext pctx;
  OperationContext ctx (signal, pctx, cfg, store.get ());
  seed.Execute (ctx, cortext::testing::GetNullTransaction ());
  cortext::testing::SeedMemoryV2 (*store, 100LL, 420LL, "loser",
                                  "LONG_TERM", 1.0, 1);
  cortext::testing::SeedMemoryV2 (*store, 101LL, 420LL, "sibling",
                                  "LONG_TERM", 1.0, 1);

  pctx.recent_context_embeddings.push_back (ctx_vec);
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 1LL, w1 },
                                                      { 2LL, w2 },
                                                      { 3LL, w3 },
                                                      { 420LL, loser } });
  ctx.SetRetrievedMemoryCandidates (
      std::vector<OperationContext::RetrievedMemoryCandidate>{
        { 1LL, 1LL, w1, 1.0 },
        { 2LL, 2LL, w2, 1.0 },
        { 3LL, 3LL, w3, 1.0 },
        { 100LL, 420LL, loser, 0.5 } });

  ApplyRetrievalCompetition op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT memory_id, suppression FROM memories "
      "WHERE memory_id IN (100, 101) ORDER BY memory_id",
      {});
  REQUIRE (rows.size () == 2);
  REQUIRE (std::any_cast<double> (rows[0].at ("suppression")) > 0.0);
  REQUIRE (std::any_cast<double> (rows[1].at ("suppression")) == 0.0);
}

TEST_CASE ("Alg21 memory-scoped RIF recovery survives embedding fork",
           "[operations][competition][recovery]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  const Eigen::VectorXf ctx_vec = Make256DEmb ({ { 0, 1.0f } });
  cortext::testing::SeedEmbeddingV2 (*store, 420LL, ctx_vec, 1);
  cortext::testing::SeedEmbeddingV2 (*store, 421LL, ctx_vec, 2);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 421LL, "forked",
                                  "LONG_TERM", 1.0, 1);
  store->Execute (
      "UPDATE memories SET suppression = ?, suppression_ts = ?, "
      "strength = ? WHERE memory_id = ?",
      { 0.5, 1LL, 0.5, 100LL });
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);

  ProcessorContext pctx;
  pctx.recent_context_embeddings.push_back (ctx_vec);
  pctx.retrieval_suppression_memory_ids.insert (100LL);
  auto signal = MakeSignal (ctx_vec, 1'000'000);
  OperationContext ctx (signal, pctx, cfg, store.get ());
  ctx.SetRetrievedMemoryEmbeddings (
      std::unordered_map<long long, Eigen::VectorXf>{ { 421LL, ctx_vec } });
  ctx.SetRetrievedMemoryCandidates (
      std::vector<OperationContext::RetrievedMemoryCandidate>{
        { 100LL, 421LL, ctx_vec, 1.0 } });

  auto tx = store->Begin ();
  ApplyRetrievalCompetition op;
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT suppression, strength FROM rif_effective_memories "
      "WHERE memory_id = ?",
      { 100LL });
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<double> (rows[0].at ("suppression"))
           == Catch::Approx (0.0));
  REQUIRE (std::any_cast<double> (rows[0].at ("strength")) > 0.5);
  REQUIRE (pctx.retrieval_suppression_memory_ids.empty ());
}

TEST_CASE ("Alg21 RIF recovery UPDATE does not violate NOT NULL constraint",
           "[operations][competition][rif_state]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;

  const Eigen::VectorXf ctx = Make256DEmb ({ { 0, 1.0f } });
  const Eigen::VectorXf w1 = Make256DEmb ({ { 0, 0.99f }, { 1, 0.05f } });
  const Eigen::VectorXf w2 = Make256DEmb ({ { 0, 0.98f }, { 1, 0.06f } });
  const Eigen::VectorXf w3 = Make256DEmb ({ { 0, 0.95f }, { 1, 0.10f } });
  const Eigen::VectorXf loser = Make256DEmb ({ { 0, 0.88f }, { 1, 0.47f } });

  std::unordered_map<long long, Eigen::VectorXf> retrieved{
    { 1LL, w1 }, { 2LL, w2 }, { 3LL, w3 }, { 4LL, loser }
  };

  {
    auto seed = std::make_unique<SeedEmbeddingsOp> (retrieved);
    auto setup = std::make_unique<SetupCompetitionInputsOp> (ctx, retrieved);
    auto apply = std::make_unique<ApplyRetrievalCompetition> ();
    auto pipeline = std::make_unique<DynamicOperationSet> (
        std::move (seed), std::move (setup), std::move (apply));
    SignalProcessor processor (cfg, store, std::move (pipeline));
    processor.Process (MakeSignal (ctx, /*ts=*/1000));
    processor.Flush ();
  }

  // v2: RIF state is inline on memories table
  auto rif_rows
      = store->Execute ("SELECT memory_id, suppression FROM memories WHERE suppression > 0");
  REQUIRE (rif_rows.size () >= 1);
  for (const auto &row : rif_rows)
    {
      REQUIRE (row.count ("suppression") == 1);
      const double supp = std::any_cast<double> (row.at ("suppression"));
      REQUIRE (supp >= 0.0);
    }

  {
    auto setup = std::make_unique<SetupCompetitionInputsOp> (ctx, retrieved);
    auto apply = std::make_unique<ApplyRetrievalCompetition> ();
    auto pipeline
        = std::make_unique<DynamicOperationSet> (std::move (setup), std::move (apply));
    SignalProcessor processor (cfg, store, std::move (pipeline));
    REQUIRE_NOTHROW (processor.Process (MakeSignal (ctx, /*ts=*/2000)));
    processor.Flush ();
  }

  // v2: RIF state is inline on memories table
  rif_rows
      = store->Execute ("SELECT memory_id, suppression FROM memories WHERE suppression > 0");
  for (const auto &row : rif_rows)
    {
      REQUIRE (row.count ("suppression") == 1);
      const double supp = std::any_cast<double> (row.at ("suppression"));
      REQUIRE (supp >= 0.0);
    }
}

TEST_CASE ("Alg21 recovery restores strength over time",
           "[operations][competition][recovery]")
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  // Higher stability -> the longest knob-derived recovery time (1800s).
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 1.0; // recovery_time = 600s

  // 256D vectors
  const Eigen::VectorXf ctx = Make256DEmb ({ { 0, 1.0f } });
  const Eigen::VectorXf w1 = Make256DEmb ({ { 0, 0.99f }, { 1, 0.05f } });
  const Eigen::VectorXf w2 = Make256DEmb ({ { 0, 0.98f }, { 1, 0.06f } });
  const Eigen::VectorXf w3 = Make256DEmb ({ { 0, 0.95f }, { 1, 0.12f } });
  const Eigen::VectorXf near_l = Make256DEmb ({ { 0, 0.90f }, { 1, 0.44f } });

  std::unordered_map<long long, Eigen::VectorXf> retrieved{
    { 1LL, w1 }, { 3LL, w2 }, { 4LL, w3 }, { 2LL, near_l }
  };

  // First pass to apply suppression (seed embeddings first).
  {
    auto seed = std::make_unique<SeedEmbeddingsOp> (retrieved);
    auto setup = std::make_unique<SetupCompetitionInputsOp> (ctx, retrieved);
    auto apply = std::make_unique<ApplyRetrievalCompetition> ();
    auto pipeline = std::make_unique<DynamicOperationSet> (
        std::move (seed), std::move (setup), std::move (apply));
    SignalProcessor processor (
        cfg, store,
        operations::signal_record_rollback_internal::
            MarkEngineOwnedJournalAware (std::move (pipeline)));
    processor.Process (MakeSignal (ctx, /*ts=*/1000));
    processor.Flush ();
  }

  // v2: Query memories table for strength
  double strength_after_supp = 0.0;
  {
    auto rows = store->Execute (
        "SELECT strength FROM memories WHERE memory_id = ?",
        { 2LL });
    REQUIRE (rows.size () == 1);
    strength_after_supp = std::any_cast<double> (rows[0].at ("strength"));
    REQUIRE (strength_after_supp < 1.0);
  }

  // One second later should recover only a small fraction.
  double strength_after_partial_recovery = 0.0;
  {
    cfg.focus = 0.0; // make all candidates winners (k=7) → no new suppression
    auto setup = std::make_unique<SetupCompetitionInputsOp> (ctx, retrieved);
    auto apply = std::make_unique<ApplyRetrievalCompetition> ();
    auto pipeline = std::make_unique<DynamicOperationSet> (std::move (setup),
                                                    std::move (apply));
    SignalProcessor processor (
        cfg, store,
        operations::signal_record_rollback_internal::
            MarkEngineOwnedJournalAware (std::move (pipeline)));
    processor.Process (MakeSignal (ctx, /*ts=*/2000));
    processor.Flush ();
  }

  {
    auto rows = store->Execute (
        "SELECT strength, suppression FROM rif_effective_memories "
        "WHERE memory_id = ?",
        { 2LL });
    REQUIRE (rows.size () == 1);
    strength_after_partial_recovery
        = std::any_cast<double> (rows[0].at ("strength"));
    const double suppression_after_partial
        = std::any_cast<double> (rows[0].at ("suppression"));
    REQUIRE (strength_after_partial_recovery > strength_after_supp);
    REQUIRE (suppression_after_partial > 0.0);
  }

  // Advance beyond 1800 seconds to complete recovery.
  {
    auto setup = std::make_unique<SetupCompetitionInputsOp> (ctx, retrieved);
    auto apply = std::make_unique<ApplyRetrievalCompetition> ();
    auto pipeline = std::make_unique<DynamicOperationSet> (std::move (setup),
                                                    std::move (apply));
    SignalProcessor processor (
        cfg, store,
        operations::signal_record_rollback_internal::
            MarkEngineOwnedJournalAware (std::move (pipeline)));
    processor.Process (MakeSignal (ctx, /*ts=*/1802000));
    processor.Flush ();
  }

  {
    auto rows = store->Execute (
        "SELECT strength, suppression FROM rif_effective_memories "
        "WHERE memory_id = ?",
        { 2LL });
    REQUIRE (rows.size () == 1);
    const double strength_after_full_recovery
        = std::any_cast<double> (rows[0].at ("strength"));
    const double suppression_after_full
        = std::any_cast<double> (rows[0].at ("suppression"));
    REQUIRE (strength_after_full_recovery
             > strength_after_partial_recovery);
    REQUIRE (strength_after_full_recovery <= 1.0);
    REQUIRE (suppression_after_full == Catch::Approx (0.0).margin (1e-4));
  }
}

TEST_CASE ("Lazy RIF clock preserves eager recurrence without active scans",
           "[operations][competition][rif_state][lazy]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  cortext::testing::InitializeCoreSchema (*store);
  store->Execute (
      "INSERT INTO memories(memory_id, source_id, kind, start_ts, n_signals, "
      "modality, strength, suppression, suppression_ts, created_at) "
      "VALUES(1, 'source-a', 'LONG_TERM', 1000, 1, 'audio', "
      "0.5, 0.5, 1000, 1000)");
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);

  {
    auto tx = store->Begin ();
    const auto result
        = cortext::operations::rif_state_internal::AdvanceRecovery (
            *tx, 2000, 10000.0, MidpointRifRowBatch ());
    REQUIRE_FALSE (result.generation_reset);
    REQUIRE (result.expired_rows == 0);
    tx->Commit ();
  }
  auto effective = store->Execute (
      "SELECT strength, suppression, suppression_ts "
      "FROM rif_effective_memories WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (effective[0].at ("strength"))
           == Catch::Approx (0.55));
  REQUIRE (std::any_cast<double> (effective[0].at ("suppression"))
           == Catch::Approx (0.45));
  REQUIRE (std::any_cast<long long> (effective[0].at ("suppression_ts"))
           == 2000);
  auto physical = store->Execute (
      "SELECT strength, suppression, suppression_ts FROM memories "
      "WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (physical[0].at ("strength")) == 0.5);
  REQUIRE (std::any_cast<double> (physical[0].at ("suppression")) == 0.5);
  REQUIRE (std::any_cast<long long> (physical[0].at ("suppression_ts"))
           == 1000);

  {
    auto tx = store->Begin ();
    cortext::operations::rif_state_internal::AdvanceRecovery (
        *tx, 3000, 10000.0, MidpointRifRowBatch ());
    cortext::operations::rif_state_internal::SuppressMemory (
        *tx, 1, 0.1, 3000);
    tx->Commit ();
  }
  effective = store->Execute (
      "SELECT strength, suppression, suppression_ts "
      "FROM rif_effective_memories WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (effective[0].at ("strength"))
           == Catch::Approx (0.495));
  REQUIRE (std::any_cast<double> (effective[0].at ("suppression"))
           == Catch::Approx (0.505));
  REQUIRE (std::any_cast<long long> (effective[0].at ("suppression_ts"))
           == 3000);

  {
    auto tx = store->Begin ();
    const auto result
        = cortext::operations::rif_state_internal::AdvanceRecovery (
            *tx, 13000, 10000.0, MidpointRifRowBatch ());
    REQUIRE (result.generation_reset);
    tx->Commit ();
  }
  effective = store->Execute (
      "SELECT strength, suppression, suppression_ts "
      "FROM rif_effective_memories WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (effective[0].at ("strength"))
           == Catch::Approx (1.0));
  REQUIRE (std::any_cast<double> (effective[0].at ("suppression")) == 0.0);
  REQUIRE (std::any_cast<long long> (effective[0].at ("suppression_ts"))
           == 13000);
}

TEST_CASE ("Lazy RIF recovery work is independent of the active row count",
           "[operations][competition][rif_state][lazy][bounded]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  cortext::testing::InitializeCoreSchema (*store);
  auto tx = store->Begin ();
  for (long long memory_id = 1; memory_id <= 2088; ++memory_id)
    {
      tx->Execute (
          "INSERT INTO memories(memory_id, source_id, kind, start_ts, "
          "n_signals, modality, strength, suppression, suppression_ts, "
          "created_at) VALUES(?, ?, 'LONG_TERM', 1000, 1, ?, "
          "0.5, 0.5, 1000, 1000)",
          { memory_id, std::string ("source-") + std::to_string (memory_id % 4),
            memory_id % 3 == 0 ? std::string ("image")
            : memory_id % 3 == 1 ? std::string ("text")
                                 : std::string ("audio") });
    }
  tx->Commit ();
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);

  tx = store->Begin ();
  const auto result
      = cortext::operations::rif_state_internal::AdvanceRecovery (
          *tx, 1001, 600000.0, MidpointRifRowBatch ());
  tx->Commit ();
  REQUIRE_FALSE (result.generation_reset);
  REQUIRE (result.expired_rows == 0);
  REQUIRE (result.retired_rows == 0);
  auto active = store->Execute (
      "SELECT COUNT(*) AS count FROM rif_active_state");
  REQUIRE (std::any_cast<long long> (active[0].at ("count")) == 2088);
  auto physical = store->Execute (
      "SELECT COUNT(*) AS count FROM memories "
      "WHERE strength = 0.5 AND suppression = 0.5 "
      "AND suppression_ts = 1000");
  REQUIRE (std::any_cast<long long> (physical[0].at ("count")) == 2088);
}

TEST_CASE ("Lazy RIF freezes the exact eager value at the active threshold",
           "[operations][competition][rif_state][lazy][threshold]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  cortext::testing::InitializeCoreSchema (*store);
  store->Execute (
      "INSERT INTO memories(memory_id, source_id, kind, start_ts, n_signals, "
      "modality, strength, suppression, suppression_ts, created_at) "
      "VALUES(1, 'source-a', 'LONG_TERM', 1000, 1, 'image', "
      "0.8, 1.1e-9, 1000, 1000)");
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);

  auto tx = store->Begin ();
  auto result = cortext::operations::rif_state_internal::AdvanceRecovery (
      *tx, 2000, 10000.0, MidpointRifRowBatch ());
  tx->Commit ();
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);
  REQUIRE (result.expired_rows == 1);

  auto row = store->Execute (
      "SELECT strength, suppression, suppression_ts FROM memories "
      "WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (row[0].at ("suppression"))
           == Catch::Approx (0.99e-9));
  REQUIRE (std::any_cast<double> (row[0].at ("strength"))
           == Catch::Approx (0.80000000011));
  REQUIRE (std::any_cast<long long> (row[0].at ("suppression_ts")) == 2000);
  REQUIRE (store->Execute (
               "SELECT memory_id FROM rif_active_state WHERE memory_id = 1")
               .empty ());

  tx = store->Begin ();
  result = cortext::operations::rif_state_internal::AdvanceRecovery (
      *tx, 3000, 10000.0, MidpointRifRowBatch ());
  tx->Commit ();
  REQUIRE (result.expired_rows == 0);
  const auto frozen = store->Execute (
      "SELECT strength, suppression, suppression_ts FROM memories "
      "WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (frozen[0].at ("suppression"))
           == std::any_cast<double> (row[0].at ("suppression")));
  REQUIRE (std::any_cast<double> (frozen[0].at ("strength"))
           == std::any_cast<double> (row[0].at ("strength")));
  REQUIRE (std::any_cast<long long> (frozen[0].at ("suppression_ts"))
           == 2000);
}

TEST_CASE ("Lazy RIF retires full-recovery generations in bounded batches",
           "[operations][competition][rif_state][lazy][generation]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  cortext::testing::InitializeCoreSchema (*store);
  const auto row_batch_size = MidpointRifRowBatch ();
  auto tx = store->Begin ();
  for (long long memory_id = 1;
       memory_id <= static_cast<long long> (2 * row_batch_size + 2);
       ++memory_id)
    tx->Execute (
        "INSERT INTO memories(memory_id, source_id, kind, start_ts, "
        "n_signals, modality, strength, suppression, suppression_ts, "
        "created_at) VALUES(?, 'source', 'LONG_TERM', 1000, 1, 'text', "
        "0.5, 0.5, 1000, 1000)",
        { memory_id });
  tx->Commit ();

  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);
  tx = store->Begin ();
  const auto first
      = cortext::operations::rif_state_internal::AdvanceRecovery (
          *tx, 12000, 10000.0, MidpointRifRowBatch ());
  tx->Commit ();
  REQUIRE (first.generation_reset);
  REQUIRE (first.retired_rows == row_batch_size);

  tx = store->Begin ();
  const auto second
      = cortext::operations::rif_state_internal::AdvanceRecovery (
          *tx, 13000, 10000.0, MidpointRifRowBatch ());
  tx->Commit ();
  REQUIRE_FALSE (second.generation_reset);
  REQUIRE (second.retired_rows == row_batch_size);

  tx = store->Begin ();
  const auto third
      = cortext::operations::rif_state_internal::AdvanceRecovery (
          *tx, 14000, 10000.0, MidpointRifRowBatch ());
  tx->Commit ();
  REQUIRE (third.retired_rows == 2);
  REQUIRE (store->Execute ("SELECT 1 FROM rif_active_state LIMIT 1").empty ());
  REQUIRE (store->Execute (
               "SELECT 1 FROM rif_generation_resets LIMIT 1").empty ());
  const auto values = store->Execute (
      "SELECT COUNT(*) AS n FROM memories WHERE strength = 1.0 "
      "AND suppression = 0.0 AND suppression_ts = 12000");
  REQUIRE (std::any_cast<long long> (values[0].at ("n"))
           == static_cast<long long> (2 * row_batch_size + 2));
}

TEST_CASE ("Lazy RIF reanchors absolute strength writes",
           "[operations][competition][rif_state][lazy][write]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  cortext::testing::InitializeCoreSchema (*store);
  store->Execute (
      "INSERT INTO memories(memory_id, source_id, kind, start_ts, n_signals, "
      "modality, strength, suppression, suppression_ts, created_at) "
      "VALUES(1, 'source-a', 'LONG_TERM', 1000, 1, 'text', "
      "0.5, 0.5, 1000, 1000)");
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);
  auto tx = store->Begin ();
  cortext::operations::rif_state_internal::AdvanceRecovery (
      *tx, 2000, 10000.0, MidpointRifRowBatch ());
  tx->Execute ("UPDATE memories SET strength = 0.2 WHERE memory_id = 1");
  cortext::operations::rif_state_internal::RefreshActiveStrengthWhere (
      *tx, "memory_id = ?", { 1LL });
  tx->Commit ();

  auto effective = store->Execute (
      "SELECT strength, suppression FROM rif_effective_memories "
      "WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (effective[0].at ("strength"))
           == Catch::Approx (0.2));
  REQUIRE (std::any_cast<double> (effective[0].at ("suppression"))
           == Catch::Approx (0.45));

  tx = store->Begin ();
  cortext::operations::rif_state_internal::AdvanceRecovery (
      *tx, 3000, 10000.0, MidpointRifRowBatch ());
  tx->Commit ();
  effective = store->Execute (
      "SELECT strength, suppression FROM rif_effective_memories "
      "WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (effective[0].at ("suppression"))
           == Catch::Approx (0.405));
  REQUIRE (std::any_cast<double> (effective[0].at ("strength"))
           == Catch::Approx (0.245));
}

TEST_CASE ("RIF active epoch publishes only after persistent commit and "
           "recovers from publication failure",
           "[operations][competition][rif_state][active_epoch]")
{
  namespace cache
      = cortext::operations::rif_active_epoch_cache_internal;
  namespace rif = cortext::operations::rif_state_internal;
  auto store = cortext::SQLiteStore::Create (":memory:");
  cortext::testing::InitializeCoreSchema (*store);
  store->Execute (
      "INSERT INTO memories(memory_id, source_id, kind, start_ts, n_signals, "
      "modality, strength, suppression, suppression_ts, created_at) "
      "VALUES(1, 'source-a', 'LONG_TERM', 1000, 1, 'text', "
      "0.5, 0.5, 1000, 1000)");
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);

  cache::State epoch;
  cache::Ensure (epoch, *store, cache::DeriveLimits (0.5, 0.5, 0.5));
  REQUIRE (epoch.valid);
  REQUIRE (epoch.active_rows == 1);
  REQUIRE (epoch.database);
  auto cached = epoch.database->Execute (
      "SELECT memory_id FROM active_state WHERE memory_id = 1");
  REQUIRE (cached.size () == 1);
  REQUIRE (std::any_cast<long long> (cached[0].at ("memory_id")) == 1);

  auto tx = store->Begin ();
  const auto recovered = rif::AdvanceRecovery (
      *tx, 2000, 10000.0, MidpointRifRowBatch ());
  cache::StageClock (epoch, recovered.clock.generation,
                     recovered.clock.log_factor, recovered.clock.last_ts);
  cache::StageMemories (epoch, recovered.changed_memory_ids);
  // The shadow identity ledger remains at its previously committed state
  // while the durable transaction is open.
  cached = epoch.database->Execute (
      "SELECT memory_id FROM active_state WHERE memory_id = 1");
  REQUIRE (cached.size () == 1);
  tx->Commit ();

  cache::SetPublishFailureStageForTest (2);
  const auto publication
      = cache::PublishAfterPersistentCommit (epoch, *store);
  REQUIRE (publication.published);
  REQUIRE (publication.rebuilt);
  REQUIRE (publication.recovered_from_failure);
  REQUIRE (epoch.valid);
  const auto effective = store->Execute (
      "SELECT strength, suppression FROM rif_effective_memories "
      "WHERE memory_id = 1");
  cached = epoch.database->Execute (
      "SELECT memory_id FROM active_state WHERE memory_id = 1");
  REQUIRE (std::any_cast<double> (effective[0].at ("suppression"))
           == Catch::Approx (0.45));
  REQUIRE (cached.size () == 1);

  cache::State restarted;
  cache::Ensure (restarted, *store, cache::DeriveLimits (0.5, 0.5, 0.5));
  REQUIRE (restarted.active_rows == epoch.active_rows);
  REQUIRE (restarted.generation == epoch.generation);
  REQUIRE (restarted.allocated_bytes <= restarted.limits.allocated_bytes);
}

TEST_CASE ("RIF active epoch row batches follow nine F S T points and "
           "preserve a live B plus one mutation",
           "[operations][competition][rif_state][active_epoch][knobs]")
{
  namespace cache
      = cortext::operations::rif_active_epoch_cache_internal;
  struct Point
  {
    double focus;
    double sensitivity;
    double stability;
  };
  const std::array<Point, 9> points { {
    { 0.5, 0.5, 0.5 },
    { 0.0, 0.0, 0.0 },
    { 1.0, 1.0, 1.0 },
    { 0.0, 0.5, 0.5 },
    { 1.0, 0.5, 0.5 },
    { 0.5, 0.0, 0.5 },
    { 0.5, 1.0, 0.5 },
    { 0.5, 0.5, 0.0 },
    { 0.5, 0.5, 1.0 },
  } };

  for (const auto &point : points)
    {
      auto store = cortext::SQLiteStore::Create (":memory:");
      cortext::testing::InitializeCoreSchema (*store);
      const auto limits = cache::DeriveLimits (
          point.focus, point.sensitivity, point.stability);
      const auto live_mutation_count = limits.row_batch_size + 1;
      auto tx = store->Begin ();
      for (std::size_t index = 0; index < live_mutation_count; ++index)
        {
          const long long memory_id = static_cast<long long> (index + 1);
          const std::string source_id
              = index % 2 == 0 ? "opaque/source-a" : "opaque/source-b";
          const std::string modality
              = index % 3 == 0 ? "text"
                : index % 3 == 1 ? "audio"
                                 : "image";
          tx->Execute (
              "INSERT INTO memories(memory_id, source_id, kind, start_ts, "
              "n_signals, modality, strength, suppression, suppression_ts, "
              "created_at) VALUES(?, ?, 'LONG_TERM', 1000, 1, ?, 0.5, "
              "0.5, 1000, 1000)",
              { memory_id, source_id, modality });
        }
      tx->Commit ();
      cortext::operations::rif_state_internal::RebuildFromMaterialized (
          *store);

      cache::State epoch;
      cache::Ensure (epoch, *store, limits);
      epoch.pending_memory_ids.reserve (live_mutation_count);
      for (std::size_t index = 0; index < live_mutation_count; ++index)
        cache::StageMemory (epoch, static_cast<long long> (index + 1));
      const auto publication = cache::PublishAfterPersistentCommit (
          epoch, *store);
      REQUIRE (publication.published);
      REQUIRE_FALSE (publication.rebuilt);
      REQUIRE (publication.changed_rows == live_mutation_count);
      REQUIRE (publication.maximum_statement_rows
               == limits.row_batch_size);
      REQUIRE (epoch.row_batch_high_water == limits.row_batch_size);
      REQUIRE (epoch.database->Execute (
                   "SELECT memory_id FROM active_state ORDER BY memory_id")
                   .size ()
               == live_mutation_count);
    }
}

TEST_CASE ("RIF consolidation starts an empty knob-derived epoch without "
           "copying durable active history",
           "[operations][competition][rif_state][active_epoch][consolidation]"
           "[knobs][regression]")
{
  namespace cache
      = cortext::operations::rif_active_epoch_cache_internal;
  struct Point
  {
    double focus;
    double sensitivity;
    double stability;
  };
  const std::array<Point, 9> points { {
    { 0.5, 0.5, 0.5 },
    { 0.0, 0.0, 0.0 },
    { 1.0, 1.0, 1.0 },
    { 0.0, 0.5, 0.5 },
    { 1.0, 0.5, 0.5 },
    { 0.5, 0.0, 0.5 },
    { 0.5, 1.0, 0.5 },
    { 0.5, 0.5, 0.0 },
    { 0.5, 0.5, 1.0 },
  } };

  for (const auto &point : points)
    {
      CAPTURE (point.focus, point.sensitivity, point.stability);
      auto store = cortext::SQLiteStore::Create (":memory:");
      cortext::testing::InitializeCoreSchema (*store);
      const auto limits = cache::DeriveLimits (
          point.focus, point.sensitivity, point.stability);
      const std::size_t durable_history_rows = limits.row_batch_size + 1;
      auto transaction = store->Begin ();
      for (std::size_t index = 0; index < durable_history_rows; ++index)
        {
          const long long memory_id = static_cast<long long> (index + 1);
          const std::string source_id
              = index % 2 == 0 ? "opaque/source-a" : "opaque/source-b";
          const std::string modality
              = index % 3 == 0 ? "text"
                : index % 3 == 1 ? "audio"
                                 : "image";
          transaction->Execute (
              "INSERT INTO memories(memory_id, source_id, kind, start_ts, "
              "n_signals, modality, strength, suppression, suppression_ts, "
              "created_at) VALUES(?, ?, 'LONG_TERM', 1000, 1, ?, 0.5, "
              "0.5, 1000, 1000)",
              { memory_id, source_id, modality });
        }
      transaction->Commit ();
      cortext::operations::rif_state_internal::RebuildFromMaterialized (
          *store);

      cache::State epoch;
      cache::Ensure (epoch, *store, limits);
      REQUIRE (epoch.active_rows == durable_history_rows);
      REQUIRE (epoch.database->Execute (
                   "SELECT memory_id FROM active_state ORDER BY memory_id")
                   .size ()
               == durable_history_rows);

      cache::StageRebuild (epoch, true);
      const auto reset
          = cache::PublishAfterPersistentCommit (epoch, *store);
      REQUIRE (reset.published);
      REQUIRE (reset.rebuilt);
      REQUIRE_FALSE (reset.recovered_from_failure);
      REQUIRE (reset.changed_rows == 0);
      REQUIRE (reset.maximum_statement_rows == 0);
      REQUIRE (epoch.active_rows == 0);
      REQUIRE (epoch.event_count == 0);
      REQUIRE (epoch.mutation_count == 0);
      REQUIRE (epoch.row_batch_high_water == 0);
      REQUIRE (epoch.database->Execute (
                   "SELECT memory_id FROM active_state ORDER BY memory_id")
                   .empty ());
      REQUIRE (cortext::operations::rif_state_internal::CountRows (
                   *store,
                   "SELECT COUNT(*) AS row_count FROM rif_active_state")
               == static_cast<long long> (durable_history_rows));

      cache::StageMemory (epoch, 1);
      const auto next_event
          = cache::PublishAfterPersistentCommit (epoch, *store);
      REQUIRE (next_event.published);
      REQUIRE_FALSE (next_event.rebuilt);
      REQUIRE (next_event.changed_rows == 1);
      REQUIRE (next_event.maximum_statement_rows == 1);
      REQUIRE (epoch.active_rows == 1);
      REQUIRE (epoch.database->Execute (
                   "SELECT memory_id FROM active_state WHERE memory_id = 1")
                   .size ()
               == 1);
    }
}

TEST_CASE ("Lazy RIF migration preserves divergent historical timestamps",
           "[operations][competition][rif_state][migration_fallback]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  cortext::store::DebugApplyCoreMigrationsThroughForTest (*store, 27);
  store->Execute (
      "INSERT INTO memories(memory_id, source_id, kind, start_ts, n_signals, "
      "modality, strength, suppression, suppression_ts, created_at) VALUES"
      "(1, 'source-a', 'LONG_TERM', 1000, 1, 'text', 0.5, 0.5, 1000, 1000),"
      "(2, 'source-b', 'LONG_TERM', 1500, 1, 'audio', 0.5, 0.5, 1500, 1500)");
  cortext::store::ApplyMigrations (*store);
  REQUIRE (cortext::operations::rif_state_internal::CountRows (
               *store,
               "SELECT COUNT(*) AS row_count FROM rif_active_state") == 2);
  cortext::operations::rif_active_epoch_cache_internal::State epoch;
  cortext::operations::rif_active_epoch_cache_internal::Ensure (
      epoch, *store,
      cortext::operations::rif_active_epoch_cache_internal::DeriveLimits (
          0.5, 0.5, 0.5));
  REQUIRE (epoch.calibration_memory_ids == std::vector<long long> { 1 });

  auto tx = store->Begin ();
  const auto result
      = cortext::operations::rif_state_internal::AdvanceRecovery (
          *tx, 2000, 10000.0, epoch.calibration_memory_ids,
          epoch.limits.row_batch_size);
  tx->Commit ();
  REQUIRE (result.changed_memory_ids.size () == 1);
  const auto rows = store->Execute (
      "SELECT memory_id, strength, suppression FROM rif_effective_memories "
      "ORDER BY memory_id");
  REQUIRE (rows.size () == 2);
  REQUIRE (std::any_cast<double> (rows[0].at ("strength"))
           == Catch::Approx (0.55));
  REQUIRE (std::any_cast<double> (rows[0].at ("suppression"))
           == Catch::Approx (0.45));
  REQUIRE (std::any_cast<double> (rows[1].at ("strength"))
           == Catch::Approx (0.525));
  REQUIRE (std::any_cast<double> (rows[1].at ("suppression"))
           == Catch::Approx (0.475));

  tx = store->Begin ();
  const auto continued
      = cortext::operations::rif_state_internal::AdvanceRecovery (
          *tx, 3000, 10000.0, MidpointRifRowBatch ());
  tx->Commit ();
  REQUIRE (continued.changed_memory_ids.empty ());
  const auto continued_rows = store->Execute (
      "SELECT memory_id, strength, suppression FROM rif_effective_memories "
      "ORDER BY memory_id");
  REQUIRE (std::any_cast<double> (continued_rows[0].at ("strength"))
           == Catch::Approx (0.595));
  REQUIRE (std::any_cast<double> (continued_rows[0].at ("suppression"))
           == Catch::Approx (0.405));
  REQUIRE (std::any_cast<double> (continued_rows[1].at ("strength"))
           == Catch::Approx (0.5725));
  REQUIRE (std::any_cast<double> (continued_rows[1].at ("suppression"))
           == Catch::Approx (0.4275));
}

TEST_CASE ("Lazy RIF compatibility rebuilds prune retired reset markers",
           "[operations][competition][rif_state][compatibility][pruning]")
{
  auto store = cortext::SQLiteStore::Create (":memory:");
  cortext::testing::InitializeCoreSchema (*store);
  store->Execute (
      "INSERT INTO memories(memory_id, source_id, kind, start_ts, n_signals, "
      "modality, strength, suppression, suppression_ts, created_at) "
      "VALUES(1, 'source-a', 'LONG_TERM', 1000, 1, 'text', "
      "0.5, 0.5, 1000, 1000)");
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);

  auto tx = store->Begin ();
  tx->Execute (
      "INSERT INTO rif_generation_resets(generation, reset_ts) "
      "VALUES(1, 2000)");
  tx->Execute (
      "UPDATE rif_recovery_clock SET generation = 2, last_ts = 2000 "
      "WHERE singleton = 1");
  cortext::operations::rif_state_internal::MaterializeAllAndClear (*tx);
  tx->Commit ();
  REQUIRE (store->Execute (
               "SELECT 1 FROM rif_generation_resets LIMIT 1").empty ());

  store->Execute (
      "INSERT INTO rif_generation_resets(generation, reset_ts) "
      "VALUES(2, 3000)");
  store->Execute (
      "UPDATE rif_recovery_clock SET generation = 3, last_ts = 3000 "
      "WHERE singleton = 1");
  cortext::operations::rif_state_internal::RebuildFromMaterialized (*store);
  REQUIRE (store->Execute (
               "SELECT 1 FROM rif_generation_resets LIMIT 1").empty ());
}

TEST_CASE ("High NE increases retrieval competition suppression",
           "[operations][competition][neuromod]")
{
  auto run_case = [] (bool disable_scale) {
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 1.0;
    cfg.sensitivity = 1.0;
    cfg.stability = 0.0;

    const Eigen::VectorXf ctx = Make256DEmb ({ { 0, 1.0f } });
    const Eigen::VectorXf w1 = Make256DEmb ({ { 0, 0.99f }, { 1, 0.05f } });
    const Eigen::VectorXf w2 = Make256DEmb ({ { 0, 0.98f }, { 1, 0.06f } });
    const Eigen::VectorXf w3 = Make256DEmb ({ { 0, 0.95f }, { 1, 0.10f } });
    const Eigen::VectorXf loser = Make256DEmb ({ { 0, 0.88f }, { 1, 0.47f } });

    std::unordered_map<long long, Eigen::VectorXf> retrieved{
      { 10LL, w1 }, { 11LL, w2 }, { 12LL, w3 }, { 13LL, loser }
    };

    auto seed = std::make_unique<SeedEmbeddingsOp> (retrieved);
    auto setup = std::make_unique<SetupCompetitionInputsOp> (ctx, retrieved);
    auto set_ne = std::make_unique<SetNeuromodOp> (1.0);
    auto apply = std::make_unique<ApplyRetrievalCompetition> ();
    auto pipeline = std::make_unique<DynamicOperationSet> (
        std::move (seed), std::move (setup), std::move (set_ne),
        std::move (apply));

    cortext::SignalProcessor processor (cfg, store, std::move (pipeline));
    if (disable_scale)
      {
        cortext::testing::ScopedEnvVar disable (
            "CORTEXT_DISABLE_NEUROMOD_COMPETITION_SCALE", "1");
        processor.Process (MakeSignal (ctx, 100));
      }
    else
      {
        processor.Process (MakeSignal (ctx, 100));
      }
    processor.Flush ();

    auto rows = store->Execute (
        "SELECT strength FROM memories WHERE memory_id = ?",
        { 13LL });
    REQUIRE (rows.size () == 1);
    return std::any_cast<double> (rows[0].at ("strength"));
  };

  const double strength_scaled = run_case (false);
  const double strength_unscaled = run_case (true);
#if defined(CORTEXT_EXPERIMENT_HOOKS)
  REQUIRE (strength_scaled < strength_unscaled);
#else
  REQUIRE (strength_scaled == Catch::Approx (strength_unscaled));
#endif
}
