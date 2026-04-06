#include "test_helpers.hpp"

#include "../src/operations/retrieval_debug_state.hpp"
#include "../src/operations/temporal_retrieval.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <sstream>

using namespace cortext;

namespace
{

constexpr int kEmbeddingDim = 256;

Eigen::VectorXf
MakeVec (float first, float second = 0.0f)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = first;
  v[1] = second;
  const float norm = v.norm ();
  if (norm > 1e-9f)
    {
      v /= norm;
    }
  return v;
}

std::vector<float>
ToFloatVec (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

Signal
MakeSignal (const Eigen::VectorXf &embedding, std::uint64_t ts)
{
  Signal s;
  s.embedding = embedding;
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

class ForceRetrievalGateOp : public IOperation
{
public:
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
  }
};

void
SeedMemoryWithEmbedding (Store &store, long long memory_id, long long embedding_id,
                         const Eigen::VectorXf &embedding,
                         const std::string &source_id, long long start_ts)
{
  cortext::testing::SeedEmbeddingV2 (store, embedding_id, ToFloatVec (embedding),
                                     start_ts);
  cortext::testing::SeedMemoryV2 (store, memory_id, embedding_id, source_id,
                                  "LONG_TERM", 1.0, start_ts);
}

void
InsertFactAssertion (Store &store, long long fact_id, const std::string &subject,
                     const std::string &predicate, const std::string &object,
                     const std::string &canonical_subject,
                     const std::string &canonical_predicate,
                     const std::string &canonical_object,
                     std::optional<long long> valid_start_ts,
                     std::optional<long long> valid_end_ts,
                     long long recorded_at_ts,
                     std::optional<long long> superseded_at_ts,
                     double confidence, long long summary_memory_id)
{
  store.Execute (
      "INSERT INTO fact_assertions "
      "(fact_id, subject, predicate, object, canonical_subject, "
      " canonical_predicate, canonical_object, valid_start_ts, valid_end_ts, "
      " recorded_at_ts, superseded_at_ts, confidence, summary_memory_id, "
      " created_at) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { fact_id,
        subject,
        predicate,
        object,
        canonical_subject,
        canonical_predicate,
        canonical_object,
        valid_start_ts.has_value () ? std::any (*valid_start_ts) : std::any (),
        valid_end_ts.has_value () ? std::any (*valid_end_ts) : std::any (),
        recorded_at_ts,
        superseded_at_ts.has_value () ? std::any (*superseded_at_ts)
                                      : std::any (),
        confidence,
        summary_memory_id,
        recorded_at_ts });
}

void
InsertFactCache (Store &store, long long fact_id, long long embedding_id,
                 const std::string &fact_text, bool is_current,
                 std::optional<long long> valid_start_ts,
                 std::optional<long long> valid_end_ts,
                 long long recorded_at_ts,
                 std::optional<long long> superseded_at_ts,
                 long long updated_at)
{
  store.Execute (
      "INSERT INTO fact_cache "
      "(fact_id, embedding_id, fact_text, is_current, valid_start_ts, "
      " valid_end_ts, recorded_at_ts, superseded_at_ts, updated_at) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { fact_id,
        embedding_id,
        fact_text,
        is_current ? 1LL : 0LL,
        valid_start_ts.has_value () ? std::any (*valid_start_ts) : std::any (),
        valid_end_ts.has_value () ? std::any (*valid_end_ts) : std::any (),
        recorded_at_ts,
        superseded_at_ts.has_value () ? std::any (*superseded_at_ts)
                                      : std::any (),
        updated_at });
}

void
LinkFactEvidence (Store &store, long long fact_id, long long source_memory_id,
                  const std::string &evidence_type, double support_weight)
{
  store.Execute (
      "INSERT INTO fact_evidence "
      "(fact_id, source_memory_id, evidence_type, support_weight) "
      "VALUES (?, ?, ?, ?)",
      { fact_id, source_memory_id, evidence_type, support_weight });
}

std::vector<long long>
RunRetrieval (std::shared_ptr<Store> store, const Eigen::VectorXf &query,
              std::uint64_t signal_ts,
              operations::temporal::RetrievalMode mode,
              std::optional<std::uint64_t> retrieval_ts,
              std::optional<operations::temporal::RetrievalAblationOverride>
                  ablation_override = std::nullopt)
{
  SignalProcessor::Config cfg;
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cortext::testing::RequireEncoder (cfg);

  auto ops = std::make_unique<OperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<operations::GraphAugmentedRetrieveCandidates> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  operations::temporal::ScopedRetrievalOverride guard (mode, retrieval_ts);
  std::optional<operations::temporal::ScopedRetrievalAblationOverride>
      ablation_guard;
  if (ablation_override.has_value ())
    {
      ablation_guard.emplace (*ablation_override);
    }
  operations::retrieval_debug::ClearLastSelectedEmbeddingOrder ();
  (void)processor.Process (MakeSignal (query, signal_ts));
  return operations::retrieval_debug::GetLastSelectedEmbeddingOrder ();
}

} // namespace

TEST_CASE ("Graph retrieval current mode prefers active fact evidence",
           "[operations][graph][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.98f, 0.20f),
                           "summary-sarah", 1000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.80f, 0.60f),
                           "summary-emily", 5000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 101LL, ToFloatVec (query), 7000LL);

  InsertFactAssertion (*store, 100LL, "Alice", "caregiver", "Sarah", "alice",
                       "caregiver", "sarah", 1000LL, 5000LL, 3000LL, 7000LL,
                       0.8, 1LL);
  InsertFactAssertion (*store, 101LL, "Alice", "caregiver", "Emily", "alice",
                       "caregiver", "emily", 5000LL, std::nullopt, 7000LL,
                       std::nullopt, 0.9, 2LL);
  InsertFactCache (*store, 100LL, 1LL, "Alice caregiver Sarah", false, 1000LL,
                   5000LL, 3000LL, 7000LL, 7000LL);
  InsertFactCache (*store, 101LL, 101LL, "Alice caregiver Emily", true, 5000LL,
                   std::nullopt, 7000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 100LL, 1LL, "summary", 1.0);
  LinkFactEvidence (*store, 101LL, 2LL, "summary", 1.0);

  const auto order = RunRetrieval (store, query, 8000ULL,
                                   operations::temporal::RetrievalMode::Current,
                                   8000ULL);
  REQUIRE_FALSE (order.empty ());
  REQUIRE (order.front () == 2LL);
}

TEST_CASE ("Graph retrieval valid_at mode surfaces historical fact evidence",
           "[operations][graph][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.82f, 0.57f),
                           "summary-austin", 1000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.97f, 0.24f),
                           "summary-denver", 5000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 102LL, ToFloatVec (query), 7000LL);

  InsertFactAssertion (*store, 200LL, "Alice", "lives_in", "Austin", "alice",
                       "lives_in", "austin", 1000LL, 5000LL, 2000LL, 7000LL,
                       0.8, 1LL);
  InsertFactAssertion (*store, 201LL, "Alice", "lives_in", "Denver", "alice",
                       "lives_in", "denver", 5000LL, std::nullopt, 7000LL,
                       std::nullopt, 0.9, 2LL);
  InsertFactCache (*store, 200LL, 102LL, "Alice lives_in Austin", false, 1000LL,
                   5000LL, 2000LL, 7000LL, 7000LL);
  InsertFactCache (*store, 201LL, 2LL, "Alice lives_in Denver", true, 5000LL,
                   std::nullopt, 7000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 200LL, 1LL, "summary", 1.0);
  LinkFactEvidence (*store, 201LL, 2LL, "summary", 1.0);

  const auto order = RunRetrieval (store, query, 8000ULL,
                                   operations::temporal::RetrievalMode::ValidAt,
                                   4000ULL);
  REQUIRE_FALSE (order.empty ());
  REQUIRE (order.front () == 1LL);
}

TEST_CASE ("Graph retrieval known_at mode reconstructs delayed knowledge",
           "[operations][graph][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.84f, 0.54f),
                           "summary-austin", 1000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.96f, 0.28f),
                           "summary-denver", 5000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 103LL, ToFloatVec (query), 7000LL);

  InsertFactAssertion (*store, 300LL, "Alice", "lives_in", "Austin", "alice",
                       "lives_in", "austin", 1000LL, 5000LL, 2000LL, 7000LL,
                       0.8, 1LL);
  InsertFactAssertion (*store, 301LL, "Alice", "lives_in", "Denver", "alice",
                       "lives_in", "denver", 5000LL, std::nullopt, 7000LL,
                       std::nullopt, 0.9, 2LL);
  InsertFactCache (*store, 300LL, 103LL, "Alice lives_in Austin", false, 1000LL,
                   5000LL, 2000LL, 7000LL, 7000LL);
  InsertFactCache (*store, 301LL, 2LL, "Alice lives_in Denver", true, 5000LL,
                   std::nullopt, 7000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 300LL, 1LL, "summary", 1.0);
  LinkFactEvidence (*store, 301LL, 2LL, "summary", 1.0);

  const auto order = RunRetrieval (store, query, 8000ULL,
                                   operations::temporal::RetrievalMode::KnownAt,
                                   6000ULL);
  REQUIRE_FALSE (order.empty ());
  REQUIRE (order.front () == 1LL);
}

TEST_CASE ("Graph retrieval boosts provenance-linked fact evidence over distractor",
           "[operations][graph][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.80f, 0.60f),
                           "linked-memory", 5000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.99f, 0.12f),
                           "semantic-distractor", 5000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 104LL, ToFloatVec (query), 7000LL);

  InsertFactAssertion (*store, 400LL, "Alice", "caregiver", "Emily", "alice",
                       "caregiver", "emily", 5000LL, std::nullopt, 7000LL,
                       std::nullopt, 0.95, 1LL);
  InsertFactCache (*store, 400LL, 104LL, "Alice caregiver Emily", true, 5000LL,
                   std::nullopt, 7000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 400LL, 1LL, "summary", 1.0);

  const auto order = RunRetrieval (store, query, 8000ULL,
                                   operations::temporal::RetrievalMode::Current,
                                   8000ULL);
  REQUIRE_FALSE (order.empty ());
  REQUIRE (order.front () == 1LL);
}

TEST_CASE ("Graph retrieval fact-off ablation removes fact-linked rescue",
           "[operations][graph][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.80f, 0.60f),
                           "linked-memory", 5000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.99f, 0.12f),
                           "semantic-distractor", 5000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 107LL, ToFloatVec (query), 7000LL);

  InsertFactAssertion (*store, 401LL, "Alice", "caregiver", "Emily", "alice",
                       "caregiver", "emily", 5000LL, std::nullopt, 7000LL,
                       std::nullopt, 0.95, 1LL);
  InsertFactCache (*store, 401LL, 107LL, "Alice caregiver Emily", true, 5000LL,
                   std::nullopt, 7000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 401LL, 1LL, "summary", 1.0);

  const auto baseline = RunRetrieval (
      store, query, 8000ULL, operations::temporal::RetrievalMode::Current,
      8000ULL);
  REQUIRE_FALSE (baseline.empty ());
  REQUIRE (baseline.front () == 1LL);

  operations::temporal::RetrievalAblationOverride override;
  override.fact_layer_enabled = false;
  const auto ablated = RunRetrieval (
      store, query, 8000ULL, operations::temporal::RetrievalMode::Current,
      8000ULL, override);
  REQUIRE_FALSE (ablated.empty ());
  REQUIRE (ablated.front () == 2LL);

  const auto &ranked = operations::retrieval_debug::GetLastRankedCandidates ();
  REQUIRE_FALSE (ranked.empty ());
  REQUIRE (ranked.front ().fact_boost == 0.0);
  REQUIRE (ranked.front ().fact_stale_penalty == 0.0);
}

TEST_CASE ("Graph retrieval current-only ablation collapses historical mode",
           "[operations][graph][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.82f, 0.57f),
                           "summary-austin", 1000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.97f, 0.24f),
                           "summary-denver", 5000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 108LL, ToFloatVec (query), 7000LL);

  InsertFactAssertion (*store, 402LL, "Alice", "lives_in", "Austin", "alice",
                       "lives_in", "austin", 1000LL, 5000LL, 2000LL, 7000LL,
                       0.8, 1LL);
  InsertFactAssertion (*store, 403LL, "Alice", "lives_in", "Denver", "alice",
                       "lives_in", "denver", 5000LL, std::nullopt, 7000LL,
                       std::nullopt, 0.9, 2LL);
  InsertFactCache (*store, 402LL, 108LL, "Alice lives_in Austin", false, 1000LL,
                   5000LL, 2000LL, 7000LL, 7000LL);
  InsertFactCache (*store, 403LL, 2LL, "Alice lives_in Denver", true, 5000LL,
                   std::nullopt, 7000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 402LL, 1LL, "summary", 1.0);
  LinkFactEvidence (*store, 403LL, 2LL, "summary", 1.0);

  const auto historical = RunRetrieval (
      store, query, 8000ULL, operations::temporal::RetrievalMode::ValidAt,
      4000ULL);
  REQUIRE_FALSE (historical.empty ());
  REQUIRE (historical.front () == 1LL);

  operations::temporal::RetrievalAblationOverride override;
  override.history_enabled = false;
  const auto current_only = RunRetrieval (
      store, query, 8000ULL, operations::temporal::RetrievalMode::ValidAt,
      4000ULL, override);
  REQUIRE_FALSE (current_only.empty ());
  REQUIRE (current_only.front () == 2LL);
}

TEST_CASE ("Graph retrieval any-fact-match provenance can lift unlinked distractor",
           "[operations][graph][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.80f, 0.60f),
                           "linked-memory", 5000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.99f, 0.12f),
                           "semantic-distractor", 5000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 109LL, ToFloatVec (query), 7000LL);

  InsertFactAssertion (*store, 404LL, "Alice", "caregiver", "Emily", "alice",
                       "caregiver", "emily", 5000LL, std::nullopt, 7000LL,
                       std::nullopt, 0.95, 1LL);
  InsertFactCache (*store, 404LL, 109LL, "Alice caregiver Emily", true, 5000LL,
                   std::nullopt, 7000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 404LL, 1LL, "summary", 1.0);

  const auto linked_only = RunRetrieval (
      store, query, 8000ULL, operations::temporal::RetrievalMode::Current,
      8000ULL);
  REQUIRE_FALSE (linked_only.empty ());
  REQUIRE (linked_only.front () == 1LL);

  operations::temporal::RetrievalAblationOverride override;
  override.provenance_mode
      = operations::temporal::ProvenanceMode::AnyFactMatch;
  const auto unrestricted = RunRetrieval (
      store, query, 8000ULL, operations::temporal::RetrievalMode::Current,
      8000ULL, override);
  REQUIRE_FALSE (unrestricted.empty ());
  REQUIRE (unrestricted.front () == 2LL);
}

TEST_CASE ("Graph retrieval ignores low-confidence conflicting noise",
           "[operations][graph][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.82f, 0.58f),
                           "caregiver-sarah", 1000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.98f, 0.19f),
                           "caregiver-emily-noise", 5000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 105LL, ToFloatVec (query), 7000LL);

  InsertFactAssertion (*store, 500LL, "Alice", "caregiver", "Sarah", "alice",
                       "caregiver", "sarah", 1000LL, std::nullopt, 3000LL,
                       std::nullopt, 0.95, 1LL);
  InsertFactAssertion (*store, 501LL, "Alice", "caregiver", "Emily", "alice",
                       "caregiver", "emily", 5000LL, std::nullopt, 7000LL,
                       7000LL, 0.35, 2LL);
  InsertFactCache (*store, 500LL, 105LL, "Alice caregiver Sarah", true, 1000LL,
                   std::nullopt, 3000LL, std::nullopt, 8000LL);
  InsertFactCache (*store, 501LL, 2LL, "Alice caregiver Emily", false, 5000LL,
                   std::nullopt, 7000LL, 7000LL, 8000LL);
  LinkFactEvidence (*store, 500LL, 1LL, "summary", 1.0);
  LinkFactEvidence (*store, 501LL, 2LL, "summary", 1.0);

  const auto order = RunRetrieval (store, query, 8000ULL,
                                   operations::temporal::RetrievalMode::Current,
                                   8000ULL);
  REQUIRE_FALSE (order.empty ());
  REQUIRE (order.front () == 1LL);
}

TEST_CASE ("Graph retrieval repeated changes do not regress to stale evidence",
           "[operations][graph][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.88f, 0.47f),
                           "austin-old", 1000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.84f, 0.54f),
                           "denver-mid", 5000LL);
  SeedMemoryWithEmbedding (*store, 3LL, 3LL, MakeVec (0.79f, 0.61f),
                           "austin-current", 9000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 106LL, ToFloatVec (query), 11000LL);

  InsertFactAssertion (*store, 600LL, "Alice", "lives_in", "Austin", "alice",
                       "lives_in", "austin", 1000LL, 5000LL, 3000LL, 7000LL,
                       0.90, 1LL);
  InsertFactAssertion (*store, 601LL, "Alice", "lives_in", "Denver", "alice",
                       "lives_in", "denver", 5000LL, 9000LL, 7000LL, 11000LL,
                       0.94, 2LL);
  InsertFactAssertion (*store, 602LL, "Alice", "lives_in", "Austin", "alice",
                       "lives_in", "austin", 9000LL, std::nullopt, 11000LL,
                       std::nullopt, 0.97, 3LL);
  InsertFactCache (*store, 600LL, 1LL, "Alice lives_in Austin", false, 1000LL,
                   5000LL, 3000LL, 7000LL, 12000LL);
  InsertFactCache (*store, 601LL, 106LL, "Alice lives_in Denver", false, 5000LL,
                   9000LL, 7000LL, 11000LL, 12000LL);
  InsertFactCache (*store, 602LL, 3LL, "Alice lives_in Austin", true, 9000LL,
                   std::nullopt, 11000LL, std::nullopt, 12000LL);
  LinkFactEvidence (*store, 600LL, 1LL, "summary", 1.0);
  LinkFactEvidence (*store, 601LL, 2LL, "summary", 1.0);
  LinkFactEvidence (*store, 602LL, 3LL, "summary", 1.0);

  const auto current_order = RunRetrieval (
      store, query, 12000ULL, operations::temporal::RetrievalMode::Current,
      12000ULL);
  {
    std::ostringstream trace;
    const auto &ranked
        = operations::retrieval_debug::GetLastRankedCandidates ();
    for (const auto &candidate : ranked)
      {
        trace << "[mem=" << candidate.memory_id << ", emb="
              << candidate.embedding_id << ", score=" << candidate.score
              << ", rel=" << candidate.relevance
              << ", boost=" << candidate.fact_boost
              << ", stale=" << candidate.fact_stale_penalty
              << ", links=" << candidate.linked_fact_count << "]";
      }
    UNSCOPED_INFO ("current ranked candidates " << trace.str ());
  }
  REQUIRE_FALSE (current_order.empty ());
  REQUIRE (current_order.front () == 3LL);

  const auto valid_mid_order = RunRetrieval (
      store, query, 12000ULL, operations::temporal::RetrievalMode::ValidAt,
      8000ULL);
  REQUIRE_FALSE (valid_mid_order.empty ());
  REQUIRE (valid_mid_order.front () == 2LL);
}

TEST_CASE ("Graph retrieval stale-penalty ablation exposes stale resurfacing",
           "[operations][graph][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.99f, 0.10f),
                           "caregiver-sarah", 1000LL);
  SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.72f, 0.69f),
                           "caregiver-emily", 5000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 110LL, ToFloatVec (query), 7000LL);

  InsertFactAssertion (*store, 405LL, "Alice", "caregiver", "Sarah", "alice",
                       "caregiver", "sarah", 1000LL, 5000LL, 3000LL, 7000LL,
                       0.8, 1LL);
  InsertFactAssertion (*store, 406LL, "Alice", "caregiver", "Emily", "alice",
                       "caregiver", "emily", 5000LL, std::nullopt, 7000LL,
                       std::nullopt, 0.9, 2LL);
  InsertFactCache (*store, 405LL, 1LL, "Alice caregiver Sarah", false, 1000LL,
                   5000LL, 3000LL, 7000LL, 7000LL);
  InsertFactCache (*store, 406LL, 110LL, "Alice caregiver Emily", true, 5000LL,
                   std::nullopt, 7000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 405LL, 1LL, "summary", 1.0);
  LinkFactEvidence (*store, 406LL, 2LL, "summary", 1.0);

  operations::temporal::RetrievalAblationOverride off;
  off.stale_penalty_strength
      = operations::temporal::StalePenaltyStrength::Off;
  const auto stale_off = RunRetrieval (
      store, query, 8000ULL, operations::temporal::RetrievalMode::Current,
      8000ULL, off);
  REQUIRE_FALSE (stale_off.empty ());
  REQUIRE (stale_off.front () == 1LL);

  operations::temporal::RetrievalAblationOverride strong;
  strong.stale_penalty_strength
      = operations::temporal::StalePenaltyStrength::Strong;
  const auto stale_strong = RunRetrieval (
      store, query, 8000ULL, operations::temporal::RetrievalMode::Current,
      8000ULL, strong);
  REQUIRE_FALSE (stale_strong.empty ());
  REQUIRE (stale_strong.front () == 2LL);
}

TEST_CASE ("Graph retrieval routine-recency presets separate routine from current exception",
           "[operations][graph][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 100LL, 100LL, MakeVec (0.90f, 0.44f),
                           "routine-lunch", 1000LL);
  SeedMemoryWithEmbedding (*store, 101LL, 101LL, MakeVec (0.83f, 0.55f),
                           "doctor-exception", 5000LL);
  SeedMemoryWithEmbedding (*store, 102LL, 102LL, MakeVec (0.83f, 0.56f),
                           "filler-1", 4500LL);
  SeedMemoryWithEmbedding (*store, 103LL, 103LL, MakeVec (0.81f, 0.59f),
                           "filler-2", 4600LL);
  cortext::testing::SeedEmbeddingV2 (*store, 108LL, ToFloatVec (query), 7000LL);

  InsertFactAssertion (*store, 800LL, "Alice", "schedule", "Lunch at noon",
                       "alice", "schedule", "lunch_at_noon", 1000LL,
                       std::nullopt, 2000LL, std::nullopt, 0.88, 100LL);
  InsertFactAssertion (*store, 801LL, "Alice", "schedule", "Doctor at noon",
                       "alice", "schedule", "doctor_at_noon", 5000LL, 6000LL,
                       5000LL, std::nullopt, 0.84, 101LL);
  InsertFactCache (*store, 800LL, 100LL, "Alice schedule Lunch at noon", true,
                   1000LL, std::nullopt, 2000LL, std::nullopt, 7000LL);
  InsertFactCache (*store, 801LL, 108LL, "Alice schedule Doctor at noon", true,
                   5000LL, 6000LL, 5000LL, std::nullopt, 7000LL);
  LinkFactEvidence (*store, 800LL, 100LL, "summary", 1.0);
  LinkFactEvidence (*store, 801LL, 101LL, "summary", 1.0);

  operations::temporal::RetrievalAblationOverride routine_override;
  routine_override.routine_recency_mode
      = operations::temporal::RoutineRecencyMode::RoutineBiased;
  const auto routine_order = RunRetrieval (
      store, query, 5500ULL, operations::temporal::RetrievalMode::Current,
      5500ULL, routine_override);

  operations::temporal::RetrievalAblationOverride recency_override;
  recency_override.routine_recency_mode
      = operations::temporal::RoutineRecencyMode::RecencyBiased;
  const auto recency_order = RunRetrieval (
      store, query, 5500ULL, operations::temporal::RetrievalMode::Current,
      5500ULL, recency_override);

  REQUIRE_FALSE (routine_order.empty ());
  REQUIRE_FALSE (recency_order.empty ());
  REQUIRE (routine_order.front () == 100LL);
  REQUIRE (recency_order.front () == 101LL);
}

TEST_CASE ("Graph retrieval ignores archived facts in current mode but preserves them historically",
           "[operations][graph][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
  SeedMemoryWithEmbedding (*store, 200LL, 200LL, MakeVec (0.90f, 0.44f),
                           "archived-home", 1000LL);
  SeedMemoryWithEmbedding (*store, 201LL, 201LL, MakeVec (0.92f, 0.39f),
                           "current-distractor", 6000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 202LL, ToFloatVec (query), 7000LL);

  InsertFactAssertion (*store, 900LL, "Alice", "home", "Maple House", "alice",
                       "home", "maple_house", 1000LL, 5000LL, 2000LL,
                       std::nullopt, 0.96, 200LL);
  InsertFactCache (*store, 900LL, 202LL, "Alice home Maple House", false,
                   1000LL, 5000LL, 2000LL, std::nullopt, 8000LL);
  LinkFactEvidence (*store, 900LL, 200LL, "summary", 1.0);
  store->Execute (
      "UPDATE fact_assertions SET lifecycle_state = 'archived', archived_at = ?, "
      "support_mass = 0.85, severity_class = 'low' WHERE fact_id = ?",
      { 8000LL, 900LL });

  const auto current_order = RunRetrieval (
      store, query, 9000ULL, operations::temporal::RetrievalMode::Current,
      9000ULL);
  REQUIRE_FALSE (current_order.empty ());
  REQUIRE (current_order.front () == 201LL);

  const auto valid_order = RunRetrieval (
      store, query, 9000ULL, operations::temporal::RetrievalMode::ValidAt,
      4000ULL);
  REQUIRE_FALSE (valid_order.empty ());
  REQUIRE (valid_order.front () == 200LL);
}
