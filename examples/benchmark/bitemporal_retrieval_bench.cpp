#include "../../src/operations/retrieval_debug_state.hpp"
#include "../../src/operations/temporal_retrieval.hpp"

#include <cortext/operations/graph_retrieval.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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

cortext::Signal
MakeSignal (const Eigen::VectorXf &embedding, std::uint64_t ts)
{
  cortext::Signal s;
  s.embedding = embedding;
  s.timestamp = ts;
  s.source_id = "bench";
  return s;
}

class ForceRetrievalGateOp : public cortext::IOperation
{
public:
  void
  Execute (cortext::OperationContext &ctx,
           cortext::Transaction & /*tx*/) const override
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

class BenchEncoder final : public cortext::Encoder
{
public:
  void
  EncodeText (const std::string & /*text*/,
              std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeAudio (const float * /*pcm*/, std::size_t /*num_samples*/,
               std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeImage (const std::uint8_t * /*data*/, int /*width*/, int /*height*/,
               int /*channels*/, std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }
};

void
SeedMemoryWithEmbedding (cortext::Store &store, long long memory_id,
                         long long embedding_id, const Eigen::VectorXf &embedding,
                         const std::string &source_id, long long start_ts)
{
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?)",
      { embedding_id, ToFloatVec (embedding), start_ts });
  store.Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, "
      "start_ts, n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, ?, 'ASSOCIATION', ?, 1, 'text', 0.5, 0.5, 1.0, ?)",
      { memory_id, embedding_id, source_id, start_ts, start_ts });
}

void
InsertFactAssertion (cortext::Store &store, long long fact_id,
                     const std::string &subject, const std::string &predicate,
                     const std::string &object,
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
InsertFactCache (cortext::Store &store, long long fact_id, long long embedding_id,
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
LinkFactEvidence (cortext::Store &store, long long fact_id, long long memory_id,
                  const std::string &evidence_type, double support_weight)
{
  store.Execute (
      "INSERT INTO fact_evidence "
      "(fact_id, source_memory_id, evidence_type, support_weight) "
      "VALUES (?, ?, ?, ?)",
      { fact_id, memory_id, evidence_type, support_weight });
}

std::vector<long long>
RunRetrieval (std::shared_ptr<cortext::Store> store, const Eigen::VectorXf &query,
              std::uint64_t signal_ts,
              cortext::operations::temporal::RetrievalMode mode,
              std::optional<std::uint64_t> retrieval_ts)
{
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 1.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  static BenchEncoder encoder;
  cfg.encoder = &encoder;

  auto ops = std::make_unique<cortext::DynamicOperationSet> (
      std::make_unique<ForceRetrievalGateOp> (),
      std::make_unique<cortext::operations::GraphAugmentedRetrieveCandidates> ());
  cortext::SignalProcessor processor (cfg, store, std::move (ops));

  cortext::operations::temporal::ScopedRetrievalOverride guard (mode,
                                                                retrieval_ts);
  cortext::operations::retrieval_debug::ClearLastSelectedEmbeddingOrder ();
  (void)processor.Process (MakeSignal (query, signal_ts));
  return cortext::operations::retrieval_debug::GetLastSelectedEmbeddingOrder ();
}

struct ScenarioResult
{
  std::string name;
  bool passed = false;
  long long expected_top = 0;
  long long actual_top = 0;
};

} // namespace

int
main ()
{
  std::vector<ScenarioResult> results;

  {
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
    SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.98f, 0.20f),
                             "caregiver-sarah", 1000LL);
    SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.80f, 0.60f),
                             "caregiver-emily", 5000LL);
    SeedMemoryWithEmbedding (*store, 3LL, 3LL, MakeVec (0.99f, 0.12f),
                             "caregiver-distractor", 5000LL);
    store->Execute (
        "INSERT INTO embeddings(embedding_id, embedding, created_at) "
        "VALUES (?, ?, ?)",
        { 101LL, ToFloatVec (query), 7000LL });
    InsertFactAssertion (*store, 100LL, "Alice", "caregiver", "Sarah", "alice",
                         "caregiver", "sarah", 1000LL, 5000LL, 3000LL, 7000LL,
                         0.8, 1LL);
    InsertFactAssertion (*store, 101LL, "Alice", "caregiver", "Emily", "alice",
                         "caregiver", "emily", 5000LL, std::nullopt, 7000LL,
                         std::nullopt, 0.95, 2LL);
    InsertFactCache (*store, 100LL, 1LL, "Alice caregiver Sarah", false, 1000LL,
                     5000LL, 3000LL, 7000LL, 7000LL);
    InsertFactCache (*store, 101LL, 101LL, "Alice caregiver Emily", true,
                     5000LL, std::nullopt, 7000LL, std::nullopt, 8000LL);
    LinkFactEvidence (*store, 100LL, 1LL, "summary", 1.0);
    LinkFactEvidence (*store, 101LL, 2LL, "summary", 1.0);

    const auto order = RunRetrieval (
        store, query, 8000ULL,
        cortext::operations::temporal::RetrievalMode::Current, 8000ULL);
    results.push_back ({ "current_caregiver", !order.empty () && order.front () == 2LL,
                         2LL, order.empty () ? 0LL : order.front () });
  }

  {
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
    SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.82f, 0.57f),
                             "location-austin", 1000LL);
    SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.97f, 0.24f),
                             "location-denver", 5000LL);
    store->Execute (
        "INSERT INTO embeddings(embedding_id, embedding, created_at) "
        "VALUES (?, ?, ?)",
        { 102LL, ToFloatVec (query), 7000LL });
    InsertFactAssertion (*store, 200LL, "Alice", "lives_in", "Austin", "alice",
                         "lives_in", "austin", 1000LL, 5000LL, 2000LL, 7000LL,
                         0.8, 1LL);
    InsertFactAssertion (*store, 201LL, "Alice", "lives_in", "Denver", "alice",
                         "lives_in", "denver", 5000LL, std::nullopt, 7000LL,
                         std::nullopt, 0.9, 2LL);
    InsertFactCache (*store, 200LL, 102LL, "Alice lives_in Austin", false,
                     1000LL, 5000LL, 2000LL, 7000LL, 7000LL);
    InsertFactCache (*store, 201LL, 2LL, "Alice lives_in Denver", true, 5000LL,
                     std::nullopt, 7000LL, std::nullopt, 8000LL);
    LinkFactEvidence (*store, 200LL, 1LL, "summary", 1.0);
    LinkFactEvidence (*store, 201LL, 2LL, "summary", 1.0);

    const auto order = RunRetrieval (
        store, query, 8000ULL,
        cortext::operations::temporal::RetrievalMode::ValidAt, 4000ULL);
    results.push_back ({ "valid_at_location", !order.empty () && order.front () == 1LL,
                         1LL, order.empty () ? 0LL : order.front () });
  }

  {
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
    SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.84f, 0.54f),
                             "known-austin", 1000LL);
    SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.96f, 0.28f),
                             "known-denver", 5000LL);
    store->Execute (
        "INSERT INTO embeddings(embedding_id, embedding, created_at) "
        "VALUES (?, ?, ?)",
        { 103LL, ToFloatVec (query), 7000LL });
    InsertFactAssertion (*store, 300LL, "Alice", "lives_in", "Austin", "alice",
                         "lives_in", "austin", 1000LL, 5000LL, 2000LL, 7000LL,
                         0.8, 1LL);
    InsertFactAssertion (*store, 301LL, "Alice", "lives_in", "Denver", "alice",
                         "lives_in", "denver", 5000LL, std::nullopt, 7000LL,
                         std::nullopt, 0.9, 2LL);
    InsertFactCache (*store, 300LL, 103LL, "Alice lives_in Austin", false,
                     1000LL, 5000LL, 2000LL, 7000LL, 7000LL);
    InsertFactCache (*store, 301LL, 2LL, "Alice lives_in Denver", true, 5000LL,
                     std::nullopt, 7000LL, std::nullopt, 8000LL);
    LinkFactEvidence (*store, 300LL, 1LL, "summary", 1.0);
    LinkFactEvidence (*store, 301LL, 2LL, "summary", 1.0);

    const auto order = RunRetrieval (
        store, query, 8000ULL,
        cortext::operations::temporal::RetrievalMode::KnownAt, 6000ULL);
    results.push_back ({ "known_at_delayed_move",
                         !order.empty () && order.front () == 1LL, 1LL,
                         order.empty () ? 0LL : order.front () });
  }

  {
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*store);
    const Eigen::VectorXf query = MakeVec (1.0f, 0.0f);
    SeedMemoryWithEmbedding (*store, 1LL, 1LL, MakeVec (0.80f, 0.60f),
                             "linked-schedule", 5000LL);
    SeedMemoryWithEmbedding (*store, 2LL, 2LL, MakeVec (0.99f, 0.12f),
                             "schedule-distractor", 5000LL);
    store->Execute (
        "INSERT INTO embeddings(embedding_id, embedding, created_at) "
        "VALUES (?, ?, ?)",
        { 104LL, ToFloatVec (query), 7000LL });
    InsertFactAssertion (*store, 400LL, "Alice", "appointment_location",
                         "Clinic", "alice", "appointment_location", "clinic",
                         5000LL, std::nullopt, 7000LL, std::nullopt, 0.95,
                         1LL);
    InsertFactCache (*store, 400LL, 104LL, "Alice appointment_location Clinic",
                     true, 5000LL, std::nullopt, 7000LL, std::nullopt, 8000LL);
    LinkFactEvidence (*store, 400LL, 1LL, "summary", 1.0);

    const auto order = RunRetrieval (
        store, query, 8000ULL,
        cortext::operations::temporal::RetrievalMode::Current, 8000ULL);
    results.push_back ({ "provenance_linked_schedule",
                         !order.empty () && order.front () == 1LL, 1LL,
                         order.empty () ? 0LL : order.front () });
  }

  int passed = 0;
  for (const auto &result : results)
    {
      passed += result.passed ? 1 : 0;
      std::cout << result.name << ": expected_top=" << result.expected_top
                << " actual_top=" << result.actual_top
                << " pass=" << (result.passed ? "1" : "0") << "\n";
    }
  std::cout << "scenario_count=" << results.size () << "\n";
  std::cout << "pass_count=" << passed << "\n";
  std::cout << "accuracy="
            << (results.empty ()
                    ? 0.0
                    : static_cast<double> (passed)
                          / static_cast<double> (results.size ()))
            << "\n";
  return passed == static_cast<int> (results.size ()) ? 0 : 1;
}
