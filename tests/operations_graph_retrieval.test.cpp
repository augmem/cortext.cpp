// tests/operations_graph_retrieval.test.cpp
#include "test_helpers.hpp"
#include "../src/operations/eviction_ablation.hpp"
#include "../src/operations/retrieval_debug_state.hpp"
#include "../src/operations/temporal_retrieval.hpp"
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cortext/core/sparse.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <cmath>
#include <optional>

using namespace cortext;
using cortext::operations::GraphAugmentedRetrieveCandidates;
namespace temporal = cortext::operations::temporal;
namespace eviction = cortext::operations::eviction;

namespace
{
constexpr int kEmbeddingDim = 256;

/// @brief Creates a 256-dim unit vector with value at index 0.
static Eigen::VectorXf
UnitVec256 (float first_val)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = first_val;
  const float n = v.norm ();
  if (n > 1e-9f)
    v /= n;
  return v;
}

/// @brief Creates a 256-dim unit vector with value at index 1.
static Eigen::VectorXf
UnitVec256Second (float second_val)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[1] = second_val;
  const float n = v.norm ();
  if (n > 1e-9f)
    v /= n;
  return v;
}

static Eigen::VectorXf
BlendVec256 (float first_val, float second_val)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = first_val;
  v[1] = second_val;
  const float n = v.norm ();
  if (n > 1e-9f)
    v /= n;
  return v;
}

/// @brief Converts Eigen vector to std::vector<float> for DB storage.
static std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
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

class ForceRetrievalGateOp : public IOperation
{
public:
  explicit ForceRetrievalGateOp (
      std::optional<ProcessorContext::MetacognitiveMode> metacognitive_mode
          = std::nullopt,
      std::optional<double> metacognitive_confidence = std::nullopt)
      : metacognitive_mode_ (metacognitive_mode),
        metacognitive_confidence_ (metacognitive_confidence)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetShouldCheckRetrieval (true);
    auto &p_ctx = ctx.GetProcessorContext ();
    if (p_ctx.memory_stream.empty ())
      {
        p_ctx.memory_stream.push_back (ctx.GetSignal ().embedding);
      }
    auto &acc = p_ctx.accumulator_states[ctx.GetSignal ().source_id];
    acc.mu_acc = ctx.GetSignal ().embedding;
    acc.c_t = ctx.GetSignal ().embedding;
    if (metacognitive_mode_.has_value ())
      {
        p_ctx.metacognitive_mode = *metacognitive_mode_;
        p_ctx.metacognitive_mode_expires_at = ctx.GetSignal ().timestamp + 1000;
        p_ctx.metacognitive_certainty_satisfied = false;
      }
    if (metacognitive_confidence_.has_value ())
      {
        p_ctx.metacognitive_confidence = *metacognitive_confidence_;
      }
  }

private:
  std::optional<ProcessorContext::MetacognitiveMode> metacognitive_mode_;
  std::optional<double> metacognitive_confidence_;
};

class SeedProceduralStoreOp : public IOperation
{
public:
  SeedProceduralStoreOp (long long memory_id, double score)
      : memory_id_ (memory_id), score_ (score)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    const int key_size = core::SparseKeySize (ctx.GetConfig ().focus);
    const auto &signal = ctx.GetSignal ();
    const std::string key = core::SparseKey (signal.embedding, key_size);
    ctx.GetProcessorContext ().procedural_store[key][memory_id_] = score_;
  }

private:
  long long memory_id_ = 0;
  double score_ = 0.0;
};

void
SetMemorySourceMetadata (Store &store, long long memory_id,
                         const std::string &origin, double reliability,
                         int contradictions)
{
  store.Execute ("UPDATE memories "
                 "SET source_origin = ?, source_reliability = ?, "
                 "source_contradiction_count = ? "
                 "WHERE memory_id = ?",
                 { origin, reliability, static_cast<long long> (contradictions),
                   memory_id });
}
} // namespace

TEST_CASE ("V2: Alg31 expands vector seeds via ASSOCIATIONS and returns expanded ids",
           "[operations][graph][alg31][v2]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  // Initialize core schema
  cortext::testing::InitializeCoreSchema (*store);

  // Create 256-dim embeddings for vec0 compatibility.
  Eigen::VectorXf emb1 = UnitVec256 (1.0f);       // First dimension = 1
  Eigen::VectorXf emb2 = UnitVec256Second (1.0f); // Second dimension = 1

  // id=1 aligns with query, id=2 does not.
  // V2: Insert into embeddings (minimal vec0 table)
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 1LL, ToFloatVec (emb1), 0LL });
  store->Execute ("INSERT INTO embeddings(embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { 2LL, ToFloatVec (emb2), 0LL });

  // V2: Insert into memories (comprehensive metadata)
  store->Execute ("INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
                  "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
                  "VALUES (?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, 1.0, 0)",
                  { 1LL, 1LL });
  store->Execute ("INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
                  "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
                  "VALUES (?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, 1.0, 0)",
                  { 2LL, 2LL });

  // V2: Create ASSOCIATIONS edge directly between memories (no intermediate label)
  // memory_id 1 -> memory_id 2 via 'co_occurs' edge
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'co_occurs', 1.0)",
      { 1LL, 2LL });

  // Low focus => depth=2 (GraphDepth(F)), ensuring expansion reaches memory 2.
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto ops = std::make_unique<OperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  auto out = processor.Process (MakeSignal (UnitVec256 (1.0f), 10));

  // Should include both seed (1) and expanded (2).
  bool has1 = false, has2 = false;
  for (const auto id : out.candidate_memory_ids)
    {
      if (id == 1LL)
        has1 = true;
      if (id == 2LL)
        has2 = true;
    }
  REQUIRE (has1);
  REQUIRE (has2);
}

TEST_CASE ("Predictive pre-activation changes retrieval ranking",
           "[operations][graph][predictive]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = UnitVec256 (1.0f);
  Eigen::VectorXf predictive_target = Eigen::VectorXf::Zero (kEmbeddingDim);
  predictive_target[0] = 0.8f;
  predictive_target[1] = 0.6f;
  predictive_target.normalize ();
  Eigen::VectorXf raw_best = Eigen::VectorXf::Zero (kEmbeddingDim);
  raw_best[0] = 0.95f;
  raw_best[1] = 0.3122499f;
  raw_best.normalize ();

  cortext::testing::SeedEmbeddingV2 (*store, 11LL, predictive_target, 1);
  cortext::testing::SeedMemoryV2 (*store, 11LL, 11LL, "test", "LONG_TERM",
                                  1.0, 1);
  cortext::testing::SeedEmbeddingV2 (*store, 22LL, raw_best, 1);
  cortext::testing::SeedMemoryV2 (*store, 22LL, 22LL, "test", "LONG_TERM", 1.0,
                                  1);
  store->Execute ("UPDATE memories SET pre_activation = 1.0 WHERE memory_id = ?",
                  { 11LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto run = [&] {
    cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
    auto ops = std::make_unique<OperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (query, 10));
    processor.Flush ();
    return cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  };

  {
    cortext::testing::ScopedEnvVar disable (
        "CORTEXT_DISABLE_PREDICTIVE_RETRIEVAL_BONUS", "1");
    const auto ranked = run ();
    REQUIRE (ranked.size () >= 2);
    REQUIRE (ranked.front ().memory_id == 22LL);
    REQUIRE (ranked.front ().predictive_bonus == Catch::Approx (0.0));
  }

  {
    cortext::testing::ScopedEnvVar enable (
        "CORTEXT_DISABLE_PREDICTIVE_RETRIEVAL_BONUS");
    const auto ranked = run ();
    REQUIRE (ranked.size () >= 2);
    REQUIRE (ranked.front ().memory_id == 11LL);
    REQUIRE (ranked.front ().predictive_bonus > 0.0);
    REQUIRE (ranked.front ().pre_activation == Catch::Approx (1.0));
  }
}

TEST_CASE ("TOT recovery expands graph traversal depth",
           "[operations][graph][metacognitive][tot]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = UnitVec256 (1.0f);
  cortext::testing::SeedEmbeddingV2 (*store, 1LL, query, 1);
  cortext::testing::SeedMemoryV2 (*store, 1LL, 1LL, "test", "LONG_TERM", 1.0,
                                  1);
  for (long long id = 2; id <= 4; ++id)
    {
      cortext::testing::SeedEmbeddingV2 (*store, id, UnitVec256Second (1.0f), 1);
      cortext::testing::SeedMemoryV2 (*store, id, id, "test", "ASSOCIATION",
                                      1.0, 1);
    }
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'co_occurs', 1.0)",
      { 1LL, 2LL });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'co_occurs', 1.0)",
      { 2LL, 3LL });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'co_occurs', 1.0)",
      { 3LL, 4LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0;

  auto run = [&] {
    auto ops = std::make_unique<OperationSet> (
        std::make_unique<ForceRetrievalGateOp> (
            ProcessorContext::MetacognitiveMode::TotRecovery, 1.0),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    auto out = processor.Process (MakeSignal (query, 10));
    processor.Flush ();
    return out.candidate_memory_ids;
  };

  {
    cortext::testing::ScopedEnvVar disable (
        "CORTEXT_DISABLE_METACOG_TOT_RECOVERY", "1");
    const auto ids = run ();
    REQUIRE (std::find (ids.begin (), ids.end (), 4LL) == ids.end ());
  }

  {
    cortext::testing::ScopedEnvVar enable ("CORTEXT_DISABLE_METACOG_TOT_RECOVERY");
    const auto ids = run ();
    REQUIRE (std::find (ids.begin (), ids.end (), 4LL) != ids.end ());
  }
}

TEST_CASE ("TOT recovery strength scales with metacognitive confidence",
           "[operations][graph][metacognitive][tot]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = UnitVec256 (1.0f);
  cortext::testing::SeedEmbeddingV2 (*store, 1LL, query, 1);
  cortext::testing::SeedMemoryV2 (*store, 1LL, 1LL, "test", "LONG_TERM", 1.0,
                                  1);
  for (long long id = 2; id <= 4; ++id)
    {
      cortext::testing::SeedEmbeddingV2 (*store, id, UnitVec256Second (1.0f), 1);
      cortext::testing::SeedMemoryV2 (*store, id, id, "test", "ASSOCIATION",
                                      1.0, 1);
    }
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'co_occurs', 1.0)",
      { 1LL, 2LL });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'co_occurs', 1.0)",
      { 2LL, 3LL });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'co_occurs', 1.0)",
      { 3LL, 4LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0;

  auto run = [&] (double confidence) {
    auto ops = std::make_unique<OperationSet> (
        std::make_unique<ForceRetrievalGateOp> (
            ProcessorContext::MetacognitiveMode::TotRecovery, confidence),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    auto out = processor.Process (MakeSignal (query, 10));
    processor.Flush ();
    return out.candidate_memory_ids;
  };

  const auto low_ids = run (0.05);
  const auto high_ids = run (1.0);
  REQUIRE (std::find (low_ids.begin (), low_ids.end (), 4LL) == low_ids.end ());
  REQUIRE (std::find (high_ids.begin (), high_ids.end (), 4LL) != high_ids.end ());
}

TEST_CASE ("Procedural proactive retrieval surfaces learned routine memory",
           "[operations][graph][procedural]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = BlendVec256 (1.0f, 0.0f);
  const Eigen::VectorXf routine_target = BlendVec256 (0.30f, 0.9539392f);
  cortext::testing::SeedEmbeddingV2 (*store, 500LL, routine_target, 1);
  cortext::testing::SeedMemoryV2 (*store, 500LL, 500LL, "test", "LONG_TERM",
                                  1.0, 1);

  for (int i = 0; i < 9; ++i)
    {
      const float x = 0.44f - 0.01f * static_cast<float> (i);
      const float y = std::sqrt (std::max (0.0f, 1.0f - x * x));
      const long long id = 600LL + i;
      cortext::testing::SeedEmbeddingV2 (*store, id, BlendVec256 (x, y), 1);
      cortext::testing::SeedMemoryV2 (*store, id, id, "test", "LONG_TERM", 1.0,
                                      1);
    }

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  cfg.procedural_enabled = true;

  auto run = [&] (bool disable_proactive) {
    cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
    auto ops = std::make_unique<OperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<SeedProceduralStoreOp> (500LL, 1.0),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    std::optional<cortext::testing::ScopedEnvVar> disable_guard;
    if (disable_proactive)
      {
        disable_guard.emplace (
            "CORTEXT_DISABLE_PROCEDURAL_PROACTIVE_RETRIEVAL", "1");
      }
    processor.Process (MakeSignal (query, 10));
    processor.Flush ();
    return cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  };

  const auto ranked_off = run (true);
  const auto ranked_on = run (false);

  bool off_has_target = false;
  bool on_has_target = false;
  for (const auto &candidate : ranked_off)
    {
      if (candidate.memory_id == 500LL)
        {
          off_has_target = true;
        }
    }
  for (const auto &candidate : ranked_on)
    {
      if (candidate.memory_id == 500LL)
        {
          on_has_target = true;
        }
    }

  REQUIRE (off_has_target == false);
  REQUIRE (on_has_target == true);
  REQUIRE_FALSE (ranked_on.empty ());
  auto it_target = std::find_if (
      ranked_on.begin (), ranked_on.end (),
      [] (const auto &candidate) { return candidate.memory_id == 500LL; });
  REQUIRE (it_target != ranked_on.end ());
  REQUIRE (it_target->proc_score == Catch::Approx (1.0).margin (1e-6));
}

TEST_CASE ("Unknown caution suppresses relaxed fallback retrieval",
           "[operations][graph][metacognitive][unknown]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = UnitVec256 (1.0f);
  cortext::testing::SeedEmbeddingV2 (*store, 77LL, query, 1);
  cortext::testing::SeedMemoryV2 (*store, 77LL, 77LL, "test", "LONG_TERM", 1.0,
                                  1);
  SetMemorySourceMetadata (*store, 77LL, "external", 0.10, 2);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto run = [&] {
    auto ops = std::make_unique<OperationSet> (
        std::make_unique<ForceRetrievalGateOp> (
            ProcessorContext::MetacognitiveMode::UnknownCaution, 0.0),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    auto out = processor.Process (MakeSignal (query, 10));
    processor.Flush ();
    return out.candidate_memory_ids;
  };

  {
    cortext::testing::ScopedEnvVar disable (
        "CORTEXT_DISABLE_METACOG_UNKNOWN_CAUTION", "1");
    const auto ids = run ();
    REQUIRE (std::find (ids.begin (), ids.end (), 77LL) != ids.end ());
  }

  {
    cortext::testing::ScopedEnvVar enable (
        "CORTEXT_DISABLE_METACOG_UNKNOWN_CAUTION");
    const auto ids = run ();
    REQUIRE (ids.empty ());
  }
}

TEST_CASE ("Pressure-weighted resurfacing preserves old but relevant memories at low pressure",
           "[operations][graph][ablation][resurfacing]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = UnitVec256 (1.0f);
  cortext::testing::SeedEmbeddingV2 (*store, 77LL, query, 1);
  cortext::testing::SeedMemoryV2 (*store, 77LL, 77LL, "test", "LONG_TERM", 1.0,
                                  1);
  SetMemorySourceMetadata (*store, 77LL, "external", 0.44, 2);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto run = [&] (std::optional<temporal::ResurfacingDecayMode> decay_mode,
                  long long used_storage_bytes) {
    temporal::RetrievalAblationOverride retrieval_override;
    retrieval_override.resurfacing_decay_mode = decay_mode;
    temporal::ScopedRetrievalAblationOverride retrieval_guard (
        retrieval_override);

    eviction::EvictionAblationOverride eviction_override;
    eviction_override.storage_gate_enabled = true;
    eviction_override.min_storage_bytes = 1000;
    eviction_override.used_storage_bytes = used_storage_bytes;
    eviction::ScopedEvictionAblationOverride eviction_guard (
        eviction_override);

    auto ops = std::make_unique<OperationSet> (
        std::make_unique<ForceRetrievalGateOp> (
            ProcessorContext::MetacognitiveMode::UnknownCaution, 0.0),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    auto out = processor.Process (
        MakeSignal (query, 168ULL * 60ULL * 60ULL * 1000ULL + 1000ULL));
    processor.Flush ();
    return out.candidate_memory_ids;
  };

  const auto default_ids = run (std::nullopt, 100);
  const auto time_only_ids
      = run (temporal::ResurfacingDecayMode::TimeOnly, 100);
  const auto pressure_weighted_ids
      = run (temporal::ResurfacingDecayMode::PressureRamp, 100);

  REQUIRE (default_ids == pressure_weighted_ids);
  REQUIRE (time_only_ids.empty ());
  REQUIRE (std::find (pressure_weighted_ids.begin (),
                      pressure_weighted_ids.end (),
                      77LL)
           != pressure_weighted_ids.end ());
}

TEST_CASE ("Pressure-weighted resurfacing converges to time-only under high pressure",
           "[operations][graph][ablation][resurfacing]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = UnitVec256 (1.0f);
  cortext::testing::SeedEmbeddingV2 (*store, 88LL, query, 1);
  cortext::testing::SeedMemoryV2 (*store, 88LL, 88LL, "test", "LONG_TERM", 1.0,
                                  1);
  SetMemorySourceMetadata (*store, 88LL, "external", 0.44, 2);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto run = [&] (std::optional<temporal::ResurfacingDecayMode> decay_mode) {
    temporal::RetrievalAblationOverride retrieval_override;
    retrieval_override.resurfacing_decay_mode = decay_mode;
    temporal::ScopedRetrievalAblationOverride retrieval_guard (
        retrieval_override);

    eviction::EvictionAblationOverride eviction_override;
    eviction_override.storage_gate_enabled = true;
    eviction_override.min_storage_bytes = 1000;
    eviction_override.used_storage_bytes = 1000;
    eviction::ScopedEvictionAblationOverride eviction_guard (
        eviction_override);

    auto ops = std::make_unique<OperationSet> (
        std::make_unique<ForceRetrievalGateOp> (
            ProcessorContext::MetacognitiveMode::UnknownCaution, 0.0),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    auto out = processor.Process (
        MakeSignal (query, 168ULL * 60ULL * 60ULL * 1000ULL + 1000ULL));
    processor.Flush ();
    return out.candidate_memory_ids;
  };

  const auto default_ids = run (std::nullopt);
  const auto time_only_ids
      = run (temporal::ResurfacingDecayMode::TimeOnly);
  const auto pressure_weighted_ids
      = run (temporal::ResurfacingDecayMode::PressureRamp);

  REQUIRE (default_ids == pressure_weighted_ids);
  REQUIRE (time_only_ids.empty ());
  REQUIRE (pressure_weighted_ids.empty ());
}

TEST_CASE ("Pressure-weighted resurfacing under low pressure is independent of stability",
           "[operations][graph][ablation][resurfacing]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = UnitVec256 (1.0f);
  cortext::testing::SeedEmbeddingV2 (*store, 99LL, query, 1);
  cortext::testing::SeedMemoryV2 (*store, 99LL, 99LL, "test", "LONG_TERM", 1.0,
                                  1);
  SetMemorySourceMetadata (*store, 99LL, "external", 0.46, 2);

  auto run = [&] (double stability) {
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 0.5;
    cfg.sensitivity = 0.5;
    cfg.stability = stability;

    temporal::RetrievalAblationOverride retrieval_override;
    retrieval_override.resurfacing_decay_mode
        = temporal::ResurfacingDecayMode::PressureRamp;
    temporal::ScopedRetrievalAblationOverride retrieval_guard (
        retrieval_override);

    eviction::EvictionAblationOverride eviction_override;
    eviction_override.storage_gate_enabled = true;
    eviction_override.min_storage_bytes = 1000;
    eviction_override.used_storage_bytes = 100;
    eviction::ScopedEvictionAblationOverride eviction_guard (
        eviction_override);

    auto ops = std::make_unique<OperationSet> (
        std::make_unique<ForceRetrievalGateOp> (
            ProcessorContext::MetacognitiveMode::UnknownCaution, 0.0),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    auto out = processor.Process (
        MakeSignal (query, 168ULL * 60ULL * 60ULL * 1000ULL + 1000ULL));
    processor.Flush ();
    return out.candidate_memory_ids;
  };

  const auto low_t_ids = run (0.2);
  const auto high_t_ids = run (0.9);

  REQUIRE (low_t_ids == high_t_ids);
  REQUIRE (std::find (low_t_ids.begin (), low_t_ids.end (), 99LL)
           != low_t_ids.end ());
}
