// tests/operations_graph_retrieval.test.cpp
#include "test_helpers.hpp"
#include "../src/operations/eviction_ablation.hpp"
#include "../src/operations/retrieval_debug_state.hpp"
#include "../src/operations/temporal_retrieval.hpp"
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cortext/core/sparse.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

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

static Signal
MakeTextSignal (const Eigen::VectorXf &emb, const std::string &text,
                uint64_t ts)
{
  Signal s = MakeSignal (emb, ts);
  s.modality = "text";
  s.mimetype = "text/plain";
  s.payload = std::vector<unsigned char> (text.begin (), text.end ());
  return s;
}

class ForceRetrievalGateOp : public IOperation
{
public:
  explicit ForceRetrievalGateOp (
      std::optional<ProcessorContext::MetacognitiveMode> metacognitive_mode
          = std::nullopt,
      std::optional<double> metacognitive_confidence = std::nullopt,
      bool seed_memory_stream = true)
      : metacognitive_mode_ (metacognitive_mode),
        metacognitive_confidence_ (metacognitive_confidence),
        seed_memory_stream_ (seed_memory_stream)
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ctx.SetShouldCheckRetrieval (true);
    auto &p_ctx = ctx.GetProcessorContext ();
    if (seed_memory_stream_ && p_ctx.memory_stream.empty ())
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
  bool seed_memory_stream_ = true;
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
    const int key_size = core::SparseKeySize (
        ctx.GetConfig ().focus, ctx.GetConfig ().sensitivity,
        ctx.GetConfig ().stability);
    const auto &signal = ctx.GetSignal ();
    const std::string key = core::SparseKey (signal.embedding, key_size);
    ctx.GetProcessorContext ().procedural_store[key][memory_id_] = score_;
  }

private:
  long long memory_id_ = 0;
  double score_ = 0.0;
};

class SeedSummaryCacheOp : public IOperation
{
public:
  struct Entry
  {
    long long memory_id = 0;
    long long embedding_id = 0;
    Eigen::VectorXf embedding;
    bool is_association = false;
    bool is_label = false;
  };

  explicit SeedSummaryCacheOp (std::vector<Entry> entries)
      : entries_ (std::move (entries))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    auto &p_ctx = ctx.GetProcessorContext ();
    for (const auto &entry : entries_)
      {
        p_ctx.UpsertSummaryCache (entry.memory_id, entry.embedding_id,
                                  entry.embedding, entry.is_association,
                                  entry.is_label);
      }
  }

private:
  std::vector<Entry> entries_;
};

class SeedWorkingMemoryAnchorOp : public IOperation
{
public:
  SeedWorkingMemoryAnchorOp (long long memory_id,
                             const std::string &source_id,
                             long long start_ts,
                             Eigen::VectorXf embedding)
      : memory_id_ (memory_id), source_id_ (source_id), start_ts_ (start_ts),
        embedding_ (std::move (embedding))
  {
  }

  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    ProcessorContext::WMSlot slot;
    slot.memory_id = memory_id_;
    slot.source_id = source_id_;
    slot.modality = "text";
    slot.start_ts = start_ts_;
    slot.embedding = embedding_;
    slot.strength = 1.0;
    SignalRecord record;
    record.timestamp = static_cast<uint64_t> (start_ts_);
    record.modality = "text";
    record.mime = "text/plain";
    record.embedding = embedding_;
    slot.signal_records.push_back (std::move (record));
    ctx.GetProcessorContext ().wm_slots.push_back (std::move (slot));
  }

private:
  long long memory_id_ = 0;
  std::string source_id_;
  long long start_ts_ = 0;
  Eigen::VectorXf embedding_;
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

  auto ops = std::make_unique<DynamicOperationSet> (
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

TEST_CASE ("Graph retrieval uses durable DB seeds after process restart",
           "[operations][graph][restart]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = UnitVec256 (1.0f);
  const Eigen::VectorXf other = UnitVec256Second (1.0f);
  store->Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?)",
      { 1LL, ToFloatVec (query), 0LL });
  store->Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?)",
      { 2LL, ToFloatVec (other), 0LL });
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
      "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, 1.0, 0)",
      { 1LL, 1LL });
  store->Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
      "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, 'test', 'LONG_TERM', 0, 1, 'text', 0.5, 0.5, 1.0, 0)",
      { 2LL, 2LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (std::nullopt, std::nullopt,
                                              false),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  auto out = processor.Process (MakeSignal (query, 10));

  REQUIRE (std::find (out.candidate_memory_ids.begin (),
                      out.candidate_memory_ids.end (),
                      1LL) != out.candidate_memory_ids.end ());
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
  cortext::testing::ScopedEnvVar disable_source_seed_expansion (
      "CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION", "1");

  auto run = [&] {
    cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
    auto ops = std::make_unique<DynamicOperationSet> (
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

TEST_CASE ("Graph retrieval scores durable labels by derived source set",
           "[operations][graph][durable_source]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = UnitVec256 (1.0f);
  const Eigen::VectorXf label_vec = BlendVec256 (0.55f, 0.45f);
  const Eigen::VectorXf unrelated = UnitVec256Second (1.0f);

  cortext::testing::SeedEmbeddingV2 (*store, 100LL, query, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 200LL, unrelated, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 300LL, label_vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 100LL, "stream/main",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 200LL,
                                  "associative_cue_test", "ASSOCIATION",
                                  1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 300LL, 300LL, "label/gallery",
                                  "LABEL", 1.0, 1000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'derived_from', 1.0), (?, ?, 'has_label', 1.0)",
      { 200LL, 100LL, 200LL, 300LL });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;

  auto run = [&] {
    cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (query, 2000));
    processor.Flush ();
    return cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  };

  std::vector<cortext::operations::retrieval_debug::RankedCandidate> ranked;
  ranked = run ();
  auto label_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 300LL;
      });
  REQUIRE (label_it != ranked.end ());
  REQUIRE (label_it->durable_source_count == 1);
  REQUIRE (label_it->durable_source_boost > 0.80);

  {
    cortext::testing::ScopedEnvVar disabled (
        "CORTEXT_DISABLE_DURABLE_SOURCE_SET_RETRIEVAL", "1");
    ranked = run ();
  }
  label_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 300LL;
      });
  REQUIRE (label_it != ranked.end ());
  REQUIRE (label_it->durable_source_count == 0);
  REQUIRE (label_it->durable_source_boost == 0.0);
}

TEST_CASE ("Graph retrieval propagates label graph boost across durable label relations",
           "[operations][graph][label_relation]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf source_vec = UnitVec256 (1.0f);
  const Eigen::VectorXf mcdonalds_vec = UnitVec256 (1.0f);
  const Eigen::VectorXf chicago_vec = UnitVec256Second (1.0f);

  cortext::testing::SeedEmbeddingV2 (*store, 100LL, source_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 200LL, source_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 300LL, mcdonalds_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 400LL, chicago_vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 100LL, "stream/main",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 200LL,
                                  "associative_cue_test", "ASSOCIATION",
                                  1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 300LL, 300LL, "mcdonalds",
                                  "LABEL", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 400LL, 400LL, "chicago",
                                  "LABEL", 1.0, 1000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'derived_from', 1.0), "
      "       (?, ?, 'has_label', 1.0), "
      "       (?, ?, 'co_occurs', 1.0)",
      { 200LL, 100LL, 200LL, 300LL, 300LL, 400LL });

  auto run = [&] {
    cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
    std::vector<SeedSummaryCacheOp::Entry> labels;
    labels.push_back ({ 300LL, 300LL, mcdonalds_vec, false, true });
    labels.push_back ({ 400LL, 400LL, chicago_vec, false, true });
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<SeedSummaryCacheOp> (std::move (labels)),
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 0.5;
    cfg.sensitivity = 1.0;
    cfg.stability = 0.5;
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (chicago_vec, 2000));
    processor.Flush ();
    return cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  };

  const auto ranked = run ();
  auto source_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 100LL;
      });
  REQUIRE (source_it != ranked.end ());
  REQUIRE (source_it->label_match_count == 1);
  REQUIRE (source_it->label_graph_boost > 0.30);
}

TEST_CASE ("Graph retrieval seeds source memories from durable label graph",
           "[operations][graph][label_relation][seed]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query_vec = UnitVec256 (1.0f);
  const Eigen::VectorXf source_vec = UnitVec256Second (1.0f);

  cortext::testing::SeedEmbeddingV2 (*store, 100LL, source_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 200LL, source_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 300LL, query_vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 100LL, "stream/source",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 200LL,
                                  "associative_cue_test", "ASSOCIATION",
                                  1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 300LL, 300LL, "river park",
                                  "LABEL", 1.0, 1000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'derived_from', 1.0), (?, ?, 'has_label', 1.0)",
      { 200LL, 100LL, 200LL, 300LL });

  for (long long i = 0; i < 20; ++i)
    {
      const long long id = 1000LL + i;
      cortext::testing::SeedEmbeddingV2 (*store, id, query_vec, 1000);
      cortext::testing::SeedMemoryV2 (*store, id, id,
                                      "stream/distractor" + std::to_string (i),
                                      "LONG_TERM", 1.0, 1000);
    }

  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  std::vector<SeedSummaryCacheOp::Entry> labels;
  labels.push_back ({ 300LL, 300LL, query_vec, false, true });
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<SeedSummaryCacheOp> (std::move (labels)),
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query_vec, 2000));
  processor.Flush ();

  const auto ranked
      = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  auto source_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 100LL;
      });
  REQUIRE (source_it != ranked.end ());
  REQUIRE (source_it->label_match_count == 1);
  REQUIRE (source_it->label_graph_boost > 0.0);
}

TEST_CASE ("Graph retrieval maps text query labels back to durable sources",
           "[operations][graph][label_relation][text]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query_vec = UnitVec256 (1.0f);
  const Eigen::VectorXf source_vec = UnitVec256Second (1.0f);
  const Eigen::VectorXf label_vec = UnitVec256Second (1.0f);

  cortext::testing::SeedEmbeddingV2 (*store, 100LL, source_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 200LL, source_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 300LL, label_vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 100LL, "stream/source",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 200LL,
                                  "associative_cue_test", "ASSOCIATION",
                                  1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 300LL, 300LL, "river park",
                                  "LABEL", 1.0, 1000);
  store->Execute ("UPDATE memories SET label = 'River Park' WHERE memory_id = 300",
                  {});
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'derived_from', 1.0), (?, ?, 'has_label', 1.0)",
      { 200LL, 100LL, 200LL, 300LL });

  for (long long i = 0; i < 20; ++i)
    {
      const long long id = 1000LL + i;
      cortext::testing::SeedEmbeddingV2 (*store, id, query_vec, 1000);
      cortext::testing::SeedMemoryV2 (*store, id, id,
                                      "stream/distractor" + std::to_string (i),
                                      "LONG_TERM", 1.0, 1000);
    }

	  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
	  auto ops = std::make_unique<DynamicOperationSet> (
	      std::make_unique<ForceRetrievalGateOp> (),
	      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeTextSignal (query_vec, "what happened at River Park?",
                                     2000));
  processor.Flush ();

  const auto ranked
      = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  auto source_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 100LL;
      });
  REQUIRE (source_it != ranked.end ());
  REQUIRE (source_it->label_match_count == 1);
  REQUIRE (source_it->label_graph_boost > 0.0);
}

TEST_CASE ("Graph retrieval maps token-overlap text queries to durable labels",
           "[operations][graph][label_relation][text][token]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query_vec = UnitVec256 (1.0f);
  const Eigen::VectorXf source_vec = UnitVec256Second (1.0f);
  const Eigen::VectorXf label_vec = UnitVec256Second (1.0f);

  cortext::testing::SeedEmbeddingV2 (*store, 100LL, source_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 200LL, source_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 300LL, label_vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 100LL, "stream/source",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 200LL,
                                  "associative_cue_test", "ASSOCIATION",
                                  1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 300LL, 300LL,
                                  "amelia just gave me money", "LABEL",
                                  1.0, 1000);
  store->Execute (
      "UPDATE memories SET label = 'Amelia just gave me money' "
      "WHERE memory_id = 300",
      {});
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'derived_from', 1.0), (?, ?, 'has_label', 1.0)",
      { 200LL, 100LL, 200LL, 300LL });

  for (long long i = 0; i < 20; ++i)
    {
      const long long id = 1000LL + i;
      cortext::testing::SeedEmbeddingV2 (*store, id, query_vec, 1000);
      cortext::testing::SeedMemoryV2 (*store, id, id,
                                      "stream/distractor" + std::to_string (i),
                                      "LONG_TERM", 1.0, 1000);
    }

	  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
	  auto ops = std::make_unique<DynamicOperationSet> (
	      std::make_unique<ForceRetrievalGateOp> (),
	      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeTextSignal (query_vec, "who gave money?", 2000));
  processor.Flush ();

  const auto ranked
      = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  auto source_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 100LL;
      });
  REQUIRE (source_it != ranked.end ());
  REQUIRE (source_it->label_match_count == 1);
  REQUIRE (source_it->label_graph_boost > 0.50);
}

TEST_CASE ("Graph retrieval expands source seeds through durable label relations",
           "[operations][graph][label_relation][source_seed]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query_vec = UnitVec256 (1.0f);
  const Eigen::VectorXf related_vec = UnitVec256Second (1.0f);
  const Eigen::VectorXf label_vec = BlendVec256 (0.3f, 0.7f);

  cortext::testing::SeedEmbeddingV2 (*store, 100LL, query_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 101LL, related_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 200LL, query_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 201LL, related_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 300LL, label_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 301LL, label_vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 100LL, "stream/source",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 101LL, 101LL, "stream/related",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 200LL,
                                  "associative_cue_source", "ASSOCIATION",
                                  1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 201LL, 201LL,
                                  "associative_cue_related", "ASSOCIATION",
                                  1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 300LL, 300LL, "river park",
                                  "LABEL", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 301LL, 301LL, "playground",
                                  "LABEL", 1.0, 1000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'derived_from', 1.0), "
      "       (?, ?, 'has_label', 1.0), "
      "       (?, ?, 'co_occurs', 1.0), "
      "       (?, ?, 'derived_from', 1.0), "
      "       (?, ?, 'has_label', 1.0)",
      { 200LL, 100LL, 200LL, 300LL, 300LL, 301LL, 201LL, 101LL,
        201LL, 301LL });

  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query_vec, 5000));
  processor.Flush ();

  const auto ranked
      = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  auto related_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 101LL;
      });
  REQUIRE (related_it != ranked.end ());
  REQUIRE (related_it->durable_source_count > 0);
  REQUIRE (related_it->durable_source_boost > 0.0);
}

TEST_CASE (
    "Graph retrieval preserves source seeds when consolidated summaries exist",
    "[operations][graph][source_seed][consolidated]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query_vec = UnitVec256 (1.0f);
  const Eigen::VectorXf related_vec = UnitVec256Second (1.0f);
  const Eigen::VectorXf label_vec = BlendVec256 (0.3f, 0.7f);

  cortext::testing::SeedEmbeddingV2 (*store, 100LL, query_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 101LL, related_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 200LL, query_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 201LL, related_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 300LL, label_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 301LL, label_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 900LL, related_vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 100LL, "stream/source",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 101LL, 101LL, "stream/related",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 200LL,
                                  "associative_cue_source", "ASSOCIATION",
                                  1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 201LL, 201LL,
                                  "associative_cue_related", "ASSOCIATION",
                                  1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 300LL, 300LL, "river park",
                                  "LABEL", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 301LL, 301LL, "playground",
                                  "LABEL", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 900LL, 900LL,
                                  "daily/consolidated-summary", "LONG_TERM",
                                  1.0, 1000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'derived_from', 1.0), "
      "       (?, ?, 'has_label', 1.0), "
      "       (?, ?, 'co_occurs', 1.0), "
      "       (?, ?, 'derived_from', 1.0), "
      "       (?, ?, 'has_label', 1.0)",
      { 200LL, 100LL, 200LL, 300LL, 300LL, 301LL, 201LL, 101LL,
        201LL, 301LL });

  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<SeedSummaryCacheOp> (
          std::vector<SeedSummaryCacheOp::Entry> {
              { 900LL, 900LL, related_vec, false, false } }),
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query_vec, 5000));
  processor.Flush ();

  const auto ranked
      = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  auto source_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 100LL;
      });
  auto related_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 101LL;
      });
  REQUIRE (source_it != ranked.end ());
  REQUIRE (related_it != ranked.end ());
  REQUIRE (related_it->durable_source_count > 0);
  REQUIRE (related_it->durable_source_boost > 0.0);
}

TEST_CASE ("Graph retrieval expands temporal neighbors from active working memory",
           "[operations][graph][source_seed][wm_temporal]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query_vec = UnitVec256 (1.0f);
  const Eigen::VectorXf target_vec = UnitVec256Second (1.0f);
  const Eigen::VectorXf wm_vec = UnitVec256 (-1.0f);

  for (long long i = 0; i < 540; ++i)
    {
      const long long id = 1000LL + i;
      cortext::testing::SeedEmbeddingV2 (*store, id, query_vec, 1000 + i);
      cortext::testing::SeedMemoryV2 (*store, id, id,
                                      "stream/distractor" + std::to_string (i),
                                      "LONG_TERM", 1.0, 1000 + i);
    }

  cortext::testing::SeedEmbeddingV2 (*store, 200LL, target_vec, 1990);
  cortext::testing::SeedEmbeddingV2 (*store, 201LL, wm_vec, 2000);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 200LL, "stream/main",
                                  "LONG_TERM", 1.0, 1990);
  cortext::testing::SeedMemoryV2 (*store, 201LL, 201LL, "stream/main",
                                  "WORKING", 1.0, 2000);

  auto run = [&] {
    cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<SeedWorkingMemoryAnchorOp> (201LL, "stream/main",
                                                     2000LL, wm_vec),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 1.0;
    cfg.sensitivity = 0.5;
    cfg.stability = 0.5;
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (query_vec, 3000));
    processor.Flush ();
    return cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  };

  auto ranked = run ();
  auto target_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 200LL;
      });
  auto working_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 201LL;
      });
  REQUIRE (target_it != ranked.end ());
  REQUIRE (target_it->durable_source_count > 0);
  REQUIRE (target_it->durable_source_boost > 0.0);
  REQUIRE (working_it == ranked.end ());

  {
    cortext::testing::ScopedEnvVar disabled (
        "CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION", "1");
    ranked = run ();
  }
  target_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 200LL;
      });
  REQUIRE (target_it == ranked.end ());
}

TEST_CASE ("Graph retrieval keeps active working-memory temporal expansion source-local",
           "[operations][graph][source_seed][wm_temporal]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query_vec = UnitVec256 (1.0f);
  const Eigen::VectorXf target_vec = UnitVec256Second (1.0f);
  const Eigen::VectorXf wm_vec = UnitVec256 (-1.0f);

  for (long long i = 0; i < 24; ++i)
    {
      const long long id = 1000LL + i;
      cortext::testing::SeedEmbeddingV2 (*store, id, query_vec, 1000 + i);
      cortext::testing::SeedMemoryV2 (*store, id, id,
                                      "stream/distractor" + std::to_string (i),
                                      "LONG_TERM", 1.0, 1000 + i);
    }

  cortext::testing::SeedEmbeddingV2 (*store, 200LL, target_vec, 1998);
  cortext::testing::SeedEmbeddingV2 (*store, 201LL, wm_vec, 2000);
  cortext::testing::SeedEmbeddingV2 (*store, 203LL, target_vec, 1998);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 200LL, "stream/reply",
                                  "LONG_TERM", 1.0, 1998);
  cortext::testing::SeedMemoryV2 (*store, 201LL, 201LL, "stream/main",
                                  "WORKING", 1.0, 2000);
  cortext::testing::SeedMemoryV2 (*store, 203LL, 203LL, "stream/main",
                                  "LONG_TERM", 1.0, 1998);

  auto run = [&] {
    cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<SeedWorkingMemoryAnchorOp> (201LL, "stream/main",
                                                     2000LL, wm_vec),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 1.0;
    cfg.sensitivity = 0.5;
    cfg.stability = 0.5;
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (query_vec, 3000));
    processor.Flush ();
    return cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  };

  auto ranked = run ();
  auto different_source_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 200LL;
      });
  auto working_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 201LL;
      });
  auto same_source_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 203LL;
      });
  if (different_source_it != ranked.end ())
    {
      REQUIRE (different_source_it->durable_source_boost == 0.0);
    }
  REQUIRE (same_source_it != ranked.end ());
  REQUIRE (same_source_it->durable_source_count > 0);
  REQUIRE (same_source_it->durable_source_boost > 0.0);
  REQUIRE (working_it == ranked.end ());

  {
    cortext::testing::ScopedEnvVar disabled (
        "CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION", "1");
    ranked = run ();
  }
  same_source_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 203LL;
      });
  REQUIRE (same_source_it == ranked.end ());
}

TEST_CASE ("Graph retrieval uses broad source seeds but compact knob-derived output",
           "[operations][graph][source_seed][compact]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query_vec = UnitVec256 (1.0f);
  for (long long i = 0; i < 40; ++i)
    {
      const long long id = 1000LL + i;
      cortext::testing::SeedEmbeddingV2 (*store, id, query_vec, 1000 + i);
      cortext::testing::SeedMemoryV2 (*store, id, id,
                                      "stream/source" + std::to_string (i),
                                      "LONG_TERM", 1.0, 1000 + i);
    }

  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.35;
  cfg.sensitivity = 0.65;
  cfg.stability = 0.60;
  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query_vec, 5000));
  processor.Flush ();

  const auto ranked
      = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  const int compact_k = std::max (
      1, cortext::core::RetrievalGraphExpandedRagMaxItems (cfg.focus,
                                                           cfg.stability));
  const int seed_k = std::max (
      1, cortext::core::RetrievalMaxResults (cfg.focus));
  REQUIRE (seed_k > compact_k);
  REQUIRE (!ranked.empty ());
  REQUIRE (static_cast<int> (ranked.size ()) <= compact_k);
}

TEST_CASE ("Graph retrieval routes text queries through source-backed durable labels",
           "[operations][graph][label_relation][text][source]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query_vec = UnitVec256 (1.0f);
  const Eigen::VectorXf source_vec = UnitVec256Second (1.0f);
  const Eigen::VectorXf label_vec = UnitVec256Second (1.0f);

  cortext::testing::SeedEmbeddingV2 (*store, 100LL, source_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 200LL, source_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 300LL, label_vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 100LL, "stream/source",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 200LL,
                                  "associative_cue_test", "ASSOCIATION",
                                  1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 300LL, 300LL, "dinner errand",
                                  "LABEL", 1.0, 1000);
  store->Execute ("UPDATE memories SET label = 'dinner errand' WHERE memory_id = 300",
                  {});
  const std::string source_text
      = "We ate noodles beside Moon Plaza after class.";
  const std::vector<unsigned char> payload (source_text.begin (),
                                            source_text.end ());
  auto blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                   { payload });
  REQUIRE (blob_rows.size () == 1);
  const auto blob_id = cortext::store::BlobFromAny (blob_rows[0].at ("id"));
  REQUIRE (!blob_id.empty ());
  store->Execute ("UPDATE memories SET blob_id = ? WHERE memory_id = 100",
                  { blob_id });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'derived_from', 1.0), (?, ?, 'has_label', 1.0)",
      { 200LL, 100LL, 200LL, 300LL });

  for (long long i = 0; i < 20; ++i)
    {
      const long long id = 1000LL + i;
      cortext::testing::SeedEmbeddingV2 (*store, id, query_vec, 1000);
      cortext::testing::SeedMemoryV2 (*store, id, id,
                                      "stream/distractor" + std::to_string (i),
                                      "LONG_TERM", 1.0, 1000);
    }

  auto run = [&] {
    cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 0.5;
    cfg.sensitivity = 1.0;
    cfg.stability = 0.5;
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (
        MakeTextSignal (query_vec, "what happened with noodles at Moon Plaza?",
                        2000));
    processor.Flush ();
    return cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  };

  std::vector<cortext::operations::retrieval_debug::RankedCandidate> ranked;
  ranked = run ();
  auto source_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 100LL;
      });
  auto label_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 300LL;
      });
  REQUIRE (!ranked.empty ());
  REQUIRE ((ranked.front ().memory_id == 100LL
            || ranked.front ().memory_id == 300LL));
  REQUIRE (label_it != ranked.end ());
  REQUIRE (label_it->label_match_count == 1);
  REQUIRE (label_it->label_graph_boost > 0.60);
  REQUIRE (source_it != ranked.end ());
  REQUIRE (source_it->label_match_count == 1);
  REQUIRE (source_it->label_graph_boost > 0.60);

  {
    cortext::testing::ScopedEnvVar disabled (
        "CORTEXT_DISABLE_PRECONSOLIDATED_LABEL_GRAPH", "1");
    ranked = run ();
  }
  source_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 100LL;
      });
  REQUIRE (source_it != ranked.end ());
  REQUIRE (source_it->label_graph_boost == 0.0);
}

TEST_CASE ("Graph retrieval damps high-degree label graph boosts",
           "[operations][graph][label_relation][generic]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query_vec = UnitVec256 (1.0f);
  const Eigen::VectorXf filler_vec = UnitVec256Second (1.0f);

  cortext::testing::SeedEmbeddingV2 (*store, 100LL, query_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 101LL, query_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 200LL, query_vec, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 201LL, query_vec, 1000);
  cortext::testing::SeedMemoryV2 (*store, 100LL, 100LL, "stream/generic",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 101LL, 101LL, "stream/specific",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 200LL, 200LL, "generic",
                                  "LABEL", 1.0, 1000);
  cortext::testing::SeedMemoryV2 (*store, 201LL, 201LL, "specific",
                                  "LABEL", 1.0, 1000);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (?, ?, 'has_label', 1.0), (?, ?, 'has_label', 1.0)",
      { 100LL, 200LL, 101LL, 201LL });

  for (long long i = 0; i < 40; ++i)
    {
      const long long label_id = 300LL + i;
      cortext::testing::SeedEmbeddingV2 (*store, label_id, filler_vec, 1000);
      cortext::testing::SeedMemoryV2 (*store, label_id, label_id, "filler",
                                      "LABEL", 1.0, 1000);
      store->Execute (
          "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
          "VALUES (?, ?, 'co_occurs', 1.0)",
          { 200LL, label_id });
    }

  cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
  std::vector<SeedSummaryCacheOp::Entry> labels;
  labels.push_back ({ 200LL, 200LL, query_vec, false, true });
  labels.push_back ({ 201LL, 201LL, query_vec, false, true });
  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<SeedSummaryCacheOp> (std::move (labels)),
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<GraphAugmentedRetrieveCandidates> ());
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.5;
  SignalProcessor processor (cfg, store, std::move (ops));
  processor.Process (MakeSignal (query_vec, 2000));
  processor.Flush ();

  const auto ranked
      = cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  auto generic_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 100LL;
      });
  auto specific_it = std::find_if (
      ranked.begin (), ranked.end (), [] (const auto &candidate) {
        return candidate.memory_id == 101LL;
      });
  REQUIRE (generic_it != ranked.end ());
  REQUIRE (specific_it != ranked.end ());
  REQUIRE (generic_it->label_match_count == 1);
  REQUIRE (specific_it->label_match_count == 1);
  REQUIRE (generic_it->label_graph_boost < specific_it->label_graph_boost);
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
    auto ops = std::make_unique<DynamicOperationSet> (
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
    auto ops = std::make_unique<DynamicOperationSet> (
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
  const Eigen::VectorXf routine_target = BlendVec256 (-1.0f, 0.0f);
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
    std::optional<cortext::testing::ScopedEnvVar> disable_guard;
    if (disable_proactive)
      {
        disable_guard.emplace (
            "CORTEXT_DISABLE_PROCEDURAL_PROACTIVE_RETRIEVAL", "1");
      }
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<SeedProceduralStoreOp> (500LL, 1.0),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (query, 10));
    processor.Flush ();
    return cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  };

  const auto ranked_off = run (true);
  const auto ranked_on = run (false);
  const double procedural_seed_min_score
      = core::RetrievalProceduralSeedMinScore (cfg.focus, cfg.sensitivity,
                                               cfg.stability);

  bool off_has_procedural_target = false;
  bool on_has_procedural_target = false;
  for (const auto &candidate : ranked_off)
    {
      if (candidate.memory_id == 500LL
          && candidate.proc_score >= procedural_seed_min_score)
        {
          off_has_procedural_target = true;
        }
    }
  for (const auto &candidate : ranked_on)
    {
      if (candidate.memory_id == 500LL
          && candidate.proc_score >= procedural_seed_min_score)
        {
          on_has_procedural_target = true;
        }
    }

  REQUIRE (off_has_procedural_target == false);
  REQUIRE (on_has_procedural_target == true);
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
    auto ops = std::make_unique<DynamicOperationSet> (
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

TEST_CASE ("Temporal retrieval rank bias favors recent relevant memories",
           "[operations][graph][temporal][recency]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  constexpr long long kNow = 4'000'000LL;
  constexpr long long kOldTs = kNow - 60LL * 60LL * 1000LL;
  constexpr long long kRecentTs = kNow - 1000LL;

  const Eigen::VectorXf query = UnitVec256 (1.0f);
  const Eigen::VectorXf old_vec = query;
  const Eigen::VectorXf recent_vec = BlendVec256 (
      0.95f, std::sqrt (std::max (0.0f, 1.0f - 0.95f * 0.95f)));

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, old_vec, kOldTs);
  cortext::testing::SeedMemoryV2 (*store, 10LL, 10LL, "test", "LONG_TERM",
                                  1.0, kOldTs);
  cortext::testing::SeedEmbeddingV2 (*store, 20LL, recent_vec, kRecentTs);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 20LL, "test", "LONG_TERM",
                                  1.0, kRecentTs);
  SetMemorySourceMetadata (*store, 10LL, "external", 1.0, 0);
  SetMemorySourceMetadata (*store, 20LL, "external", 1.0, 0);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  auto run = [&] (bool disable_temporal) {
    cortext::operations::retrieval_debug::ClearLastRankedCandidates ();
    std::optional<cortext::testing::ScopedEnvVar> disable_guard;
    if (disable_temporal)
      {
        disable_guard.emplace ("CORTEXT_DISABLE_TEMPORAL_RETRIEVAL", "1");
      }
    auto ops = std::make_unique<DynamicOperationSet> (
        std::make_unique<ForceRetrievalGateOp> (),
        std::make_unique<GraphAugmentedRetrieveCandidates> ());
    SignalProcessor processor (cfg, store, std::move (ops));
    processor.Process (MakeSignal (query, static_cast<uint64_t> (kNow)));
    processor.Flush ();
    return cortext::operations::retrieval_debug::GetLastRankedCandidates ();
  };

  const auto temporal_ranked = run (false);
  REQUIRE_FALSE (temporal_ranked.empty ());
  REQUIRE (temporal_ranked.front ().memory_id == 20LL);
  REQUIRE (temporal_ranked.front ().temporal_score > 0.95);

  const auto no_temporal_ranked = run (true);
  REQUIRE_FALSE (no_temporal_ranked.empty ());
  REQUIRE (no_temporal_ranked.front ().memory_id == 10LL);
  REQUIRE (no_temporal_ranked.front ().temporal_score == Catch::Approx (0.0));
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

    auto ops = std::make_unique<DynamicOperationSet> (
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

    auto ops = std::make_unique<DynamicOperationSet> (
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

TEST_CASE ("Pressure-weighted resurfacing under low pressure follows stability",
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

    auto ops = std::make_unique<DynamicOperationSet> (
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

  REQUIRE (std::find (high_t_ids.begin (), high_t_ids.end (), 99LL)
           != high_t_ids.end ());
  REQUIRE (high_t_ids.size () >= low_t_ids.size ());
}
