#include "test_helpers.hpp"

#include "../src/operations/temporal_retrieval.hpp"
#include "../src/store/facts.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/process_extraction_results.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/sqlite_store.hpp>

using namespace cortext;

namespace
{

constexpr int kEmbeddingDim = 256;

Signal
MakeSignal (uint64_t ts)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

class FixedEncoder : public Encoder
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
    EncodeText ("", out_embedding);
  }

  void
  EncodeImage (const std::uint8_t * /*data*/, int /*width*/, int /*height*/,
               int /*channels*/, std::vector<float> &out_embedding) override
  {
    EncodeText ("", out_embedding);
  }
};

void
SeedSummary (Store &store, long long embedding_id, long long memory_id,
             const std::string &summary_id, long long start_ts)
{
  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (store, embedding_id, unit_embedding,
                                     start_ts);
  cortext::testing::SeedMemoryV2 (store, memory_id, embedding_id, summary_id,
                                  "ASSOCIATION", 1.0, start_ts);
}

void
RunExtraction (Store &store, ProcessorContext &pctx, FixedEncoder &encoder,
               uint64_t signal_ts, double sensitivity = 0.5,
               double stability = 0.5)
{
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = sensitivity;
  cfg.stability = stability;
  cfg.encoder = &encoder;

  Signal s = MakeSignal (signal_ts);
  OperationContext ctx (s, pctx, cfg, &store);

  operations::ProcessExtractionResults op;
  auto tx = store.Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();
}

} // namespace

TEST_CASE ("ProcessExtractionResults persists bitemporal facts and fact cache",
           "[operations][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);

  ProcessorContext pctx;
  FixedEncoder encoder;

  operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.labels.push_back ({ "Alice", 0.0 });
  extraction.facts.push_back (
      { "Alice", "lives_in", "Austin", 0.9, std::nullopt, std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (extraction));

  RunExtraction (*store, pctx, encoder, 3000ULL);

  auto fact_rows = store->Execute (
      "SELECT canonical_subject, canonical_predicate, canonical_object, "
      "valid_start_ts, recorded_at_ts, confidence "
      "FROM fact_assertions",
      {});
  REQUIRE (fact_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (
               fact_rows[0].at ("canonical_subject"))
           == "alice");
  REQUIRE (std::any_cast<std::string> (
               fact_rows[0].at ("canonical_predicate"))
           == "lives_in");
  REQUIRE (std::any_cast<std::string> (
               fact_rows[0].at ("canonical_object"))
           == "austin");
  REQUIRE (cortext::testing::GetInt64 (fact_rows[0], "valid_start_ts")
           == 1000LL);
  REQUIRE (cortext::testing::GetInt64 (fact_rows[0], "recorded_at_ts")
           == 3000LL);
  REQUIRE (cortext::testing::GetDouble (fact_rows[0], "confidence")
           == Catch::Approx (0.9));

  auto cache_rows = store->Execute (
      "SELECT fact_text, is_current, embedding_id FROM fact_cache", {});
  REQUIRE (cache_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (cache_rows[0].at ("fact_text"))
           == "Alice lives_in Austin");
  REQUIRE (cortext::testing::GetInt64 (cache_rows[0], "is_current") == 1LL);
  REQUIRE (cortext::testing::GetInt64 (cache_rows[0], "embedding_id") > 0LL);
}

TEST_CASE ("Duplicate fact assertions merge and accumulate evidence",
           "[operations][facts]")
{
  cortext::testing::ScopedEnvVar clear_evidence_confidence (
      "CORTEXT_ENABLE_EVIDENCE_CONFIDENCE");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);
  SeedSummary (*store, 11LL, 21LL, "summary-2", 2000LL);

  ProcessorContext pctx;
  FixedEncoder encoder;

  operations::ExtractionResult first;
  first.summary_id = "summary-1";
  first.labels.push_back ({ "Alice", 0.0 });
  first.facts.push_back (
      { "Alice", "works_at", "Acme", 0.6, std::nullopt, std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (first));
  RunExtraction (*store, pctx, encoder, 3000ULL);

  operations::ExtractionResult second;
  second.summary_id = "summary-2";
  second.labels.push_back ({ "Acme", 0.0 });
  second.facts.push_back (
      { "Alice", "works_at", "Acme", 0.8, std::nullopt, std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (second));
  RunExtraction (*store, pctx, encoder, 4000ULL);

  auto rows = store->Execute (
      "SELECT COUNT(*) AS c, MAX(confidence) AS confidence "
      "FROM fact_assertions",
      {});
  REQUIRE (rows.size () == 1);
  REQUIRE (cortext::testing::GetInt64 (rows[0], "c") == 1LL);
  REQUIRE (cortext::testing::GetDouble (rows[0], "confidence")
           == Catch::Approx (0.8));

  auto tx = store->Begin ();
  cortext::store::FactQuery query;
  query.mode = cortext::store::FactQueryMode::Current;
  query.canonical_subject = "alice";
  query.canonical_predicate = "works_at";
  query.timestamp = 5000ULL;
  const auto facts = cortext::store::QueryFacts (*tx, query);
  REQUIRE (facts.size () == 1);
	  REQUIRE (facts[0].evidence_count == 2);
}

TEST_CASE ("Evidence confidence revises fact confidence for independent evidence",
           "[operations][facts][evidence-confidence]")
{
  cortext::testing::ScopedEnvVar enable_evidence_confidence (
      "CORTEXT_ENABLE_EVIDENCE_CONFIDENCE", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);
  SeedSummary (*store, 11LL, 21LL, "summary-2", 2000LL);

  ProcessorContext pctx;
  FixedEncoder encoder;

  operations::ExtractionResult first;
  first.summary_id = "summary-1";
  first.facts.push_back (
      { "Alice", "works_at", "Acme", 0.6, std::nullopt, std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (first));
  RunExtraction (*store, pctx, encoder, 3000ULL);

  operations::ExtractionResult second;
  second.summary_id = "summary-2";
  second.facts.push_back (
      { "Alice", "works_at", "Acme", 0.6, std::nullopt, std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (second));
  RunExtraction (*store, pctx, encoder, 4000ULL);

  auto rows = store->Execute (
      "SELECT confidence FROM fact_assertions "
      "WHERE canonical_subject = 'alice' AND canonical_predicate = 'works_at' "
      "  AND canonical_object = 'acme'",
      {});
  REQUIRE (rows.size () == 1);
  REQUIRE (cortext::testing::GetDouble (rows[0], "confidence") > 0.6);

  auto tx = store->Begin ();
  cortext::store::FactQuery query;
  query.mode = cortext::store::FactQueryMode::Current;
  query.canonical_subject = "alice";
  query.canonical_predicate = "works_at";
  query.timestamp = 5000ULL;
  const auto facts = cortext::store::QueryFacts (*tx, query);
  REQUIRE (facts.size () == 1);
  REQUIRE (facts[0].evidence_count == 2);
}

TEST_CASE ("Evidence confidence ignores duplicate fact evidence stamps",
           "[operations][facts][evidence-confidence]")
{
  cortext::testing::ScopedEnvVar enable_evidence_confidence (
      "CORTEXT_ENABLE_EVIDENCE_CONFIDENCE", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);

  ProcessorContext pctx;
  FixedEncoder encoder;

  operations::ExtractionResult first;
  first.summary_id = "summary-1";
  first.facts.push_back (
      { "Alice", "works_at", "Acme", 0.6, std::nullopt, std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (first));
  RunExtraction (*store, pctx, encoder, 3000ULL);

  operations::ExtractionResult duplicate;
  duplicate.summary_id = "summary-1";
  duplicate.facts.push_back (
      { "Alice", "works_at", "Acme", 0.6, std::nullopt, std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (duplicate));
  RunExtraction (*store, pctx, encoder, 4000ULL);

  auto rows = store->Execute (
      "SELECT confidence FROM fact_assertions "
      "WHERE canonical_subject = 'alice' AND canonical_predicate = 'works_at' "
      "  AND canonical_object = 'acme'",
      {});
  REQUIRE (rows.size () == 1);
  REQUIRE (cortext::testing::GetDouble (rows[0], "confidence")
           == Catch::Approx (0.6));

  auto evidence_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM fact_evidence",
      {});
  REQUIRE (cortext::testing::GetInt64 (evidence_rows[0], "c") == 1LL);
}

TEST_CASE ("Bitemporal queries distinguish valid time from known time",
           "[operations][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);
  SeedSummary (*store, 11LL, 21LL, "summary-2", 5000LL);

  ProcessorContext pctx;
  FixedEncoder encoder;

  operations::ExtractionResult first;
  first.summary_id = "summary-1";
  first.labels.push_back ({ "Alice", 0.0 });
  first.facts.push_back (
      { "Alice", "lives_in", "Austin", 0.7, std::uint64_t (1000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (first));
  RunExtraction (*store, pctx, encoder, 3000ULL);

  operations::ExtractionResult second;
  second.summary_id = "summary-2";
  second.labels.push_back ({ "Alice", 0.0 });
  second.facts.push_back (
      { "Alice", "lives_in", "Denver", 0.9, std::uint64_t (5000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (second));
  RunExtraction (*store, pctx, encoder, 7000ULL);

  auto tx = store->Begin ();

  cortext::store::FactQuery current_query;
  current_query.mode = cortext::store::FactQueryMode::Current;
  current_query.canonical_subject = "alice";
  current_query.canonical_predicate = "lives_in";
  current_query.timestamp = 8000ULL;
  auto current = cortext::store::QueryFacts (*tx, current_query);
  REQUIRE (current.size () == 1);
  REQUIRE (current[0].object == "Denver");

  cortext::store::FactQuery valid_query;
  valid_query.mode = cortext::store::FactQueryMode::ValidAt;
  valid_query.canonical_subject = "alice";
  valid_query.canonical_predicate = "lives_in";
  valid_query.timestamp = 6000ULL;
  auto valid_at = cortext::store::QueryFacts (*tx, valid_query);
  REQUIRE (valid_at.size () == 1);
  REQUIRE (valid_at[0].object == "Denver");

  cortext::store::FactQuery known_query;
  known_query.mode = cortext::store::FactQueryMode::KnownAt;
  known_query.canonical_subject = "alice";
  known_query.canonical_predicate = "lives_in";
  known_query.timestamp = 6000ULL;
  auto known_at = cortext::store::QueryFacts (*tx, known_query);
  REQUIRE (known_at.size () == 1);
  REQUIRE (known_at[0].object == "Austin");

  auto past_current = cortext::store::QueryFacts (
      *tx, { cortext::store::FactQueryMode::ValidAt, std::string ("alice"),
             std::string ("lives_in"), 4000ULL });
  REQUIRE (past_current.size () == 1);
  REQUIRE (past_current[0].object == "Austin");
}

TEST_CASE ("Conflicting facts supersede earlier open assertions",
           "[operations][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);
  SeedSummary (*store, 11LL, 21LL, "summary-2", 5000LL);

  ProcessorContext pctx;
  FixedEncoder encoder;

  operations::ExtractionResult first;
  first.summary_id = "summary-1";
  first.facts.push_back (
      { "Alice", "caregiver", "Sarah", 0.7, std::nullopt, std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (first));
  RunExtraction (*store, pctx, encoder, 3000ULL);

  operations::ExtractionResult second;
  second.summary_id = "summary-2";
  second.facts.push_back (
      { "Alice", "caregiver", "Emily", 0.9, std::uint64_t (5000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (second));
  RunExtraction (*store, pctx, encoder, 7000ULL);

  auto fact_rows = store->Execute (
      "SELECT canonical_object, valid_start_ts, valid_end_ts, recorded_at_ts, "
      "superseded_at_ts "
      "FROM fact_assertions "
      "WHERE canonical_subject = 'alice' AND canonical_predicate = 'caregiver' "
      "ORDER BY recorded_at_ts ASC",
      {});
  REQUIRE (fact_rows.size () == 2);

  REQUIRE (std::any_cast<std::string> (fact_rows[0].at ("canonical_object"))
           == "sarah");
  REQUIRE (cortext::testing::GetInt64 (fact_rows[0], "valid_start_ts")
           == 1000LL);
  REQUIRE (cortext::testing::GetInt64 (fact_rows[0], "valid_end_ts")
           == 5000LL);
  REQUIRE (cortext::testing::GetInt64 (fact_rows[0], "superseded_at_ts")
           == 7000LL);

  REQUIRE (std::any_cast<std::string> (fact_rows[1].at ("canonical_object"))
           == "emily");
  REQUIRE (cortext::testing::GetInt64 (fact_rows[1], "valid_start_ts")
           == 5000LL);

  auto tx = store->Begin ();
  auto current = cortext::store::QueryFacts (
      *tx, { cortext::store::FactQueryMode::Current, std::string ("alice"),
             std::string ("caregiver"), 8000ULL });
  REQUIRE (current.size () == 1);
  REQUIRE (current[0].object == "Emily");
}

TEST_CASE ("Facts without explicit timestamps fall back to summary start time",
           "[operations][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedSummary (*store, 10LL, 20LL, "summary-1", 1234LL);

  ProcessorContext pctx;
  FixedEncoder encoder;

  operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.facts.push_back (
      { "Alice", "takes_medication", "Donepezil", 0.85, std::nullopt,
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (extraction));
  RunExtraction (*store, pctx, encoder, 3000ULL);

  auto fact_rows = store->Execute (
      "SELECT valid_start_ts, valid_end_ts FROM fact_assertions "
      "WHERE canonical_subject = 'alice' "
      "  AND canonical_predicate = 'takes_medication' "
      "  AND canonical_object = 'donepezil'",
      {});
  REQUIRE (fact_rows.size () == 1);
  REQUIRE (cortext::testing::GetInt64 (fact_rows[0], "valid_start_ts")
           == 1234LL);
  REQUIRE (fact_rows[0].at ("valid_end_ts").type () == typeid (std::nullptr_t));
}

TEST_CASE ("Fact scoring favors current active assertions over superseded ones",
           "[operations][facts]")
{
  cortext::store::FactRecord current;
  current.confidence = 0.9;
  current.evidence_count = 3;
  current.recorded_at_ts = 3000;
  current.valid_start_ts = 1000;
  current.is_current = true;

  cortext::store::FactRecord superseded = current;
  superseded.is_current = false;
  superseded.superseded_at_ts = 7000;

  const auto current_score = cortext::store::ScoreFactRecord (
      current, cortext::store::FactQueryMode::Current, 8000ULL);
  const auto superseded_score = cortext::store::ScoreFactRecord (
      superseded, cortext::store::FactQueryMode::Current, 8000ULL);

  REQUIRE (current_score.temporal_match == Catch::Approx (1.0));
  REQUIRE (superseded_score.temporal_match == Catch::Approx (0.0));
  REQUIRE (current_score.combined > superseded_score.combined);
}

TEST_CASE ("Fact scoring routine-recency presets separate routine and current exception facts",
           "[operations][facts]")
{
  cortext::store::FactRecord routine;
  routine.canonical_predicate = "schedule";
  routine.canonical_object = "lunch_at_noon";
  routine.confidence = 0.88;
  routine.evidence_count = 2;
  routine.recorded_at_ts = 2000;
  routine.valid_start_ts = 1000;
  routine.is_current = true;

  cortext::store::FactRecord current_exception = routine;
  current_exception.canonical_object = "doctor_at_noon";
  current_exception.confidence = 0.84;
  current_exception.evidence_count = 1;
  current_exception.recorded_at_ts = 5000;
  current_exception.valid_start_ts = 5000;
  current_exception.valid_end_ts = 6000;

  const auto balanced_routine = cortext::store::ScoreFactRecord (
      routine, cortext::store::FactQueryMode::Current, 5500ULL);
  const auto balanced_exception = cortext::store::ScoreFactRecord (
      current_exception, cortext::store::FactQueryMode::Current, 5500ULL);

  cortext::operations::temporal::RetrievalAblationOverride routine_override;
  routine_override.routine_recency_mode
      = cortext::operations::temporal::RoutineRecencyMode::RoutineBiased;
  cortext::operations::temporal::ScopedRetrievalAblationOverride routine_guard (
      routine_override);
  const auto routine_biased_routine = cortext::store::ScoreFactRecord (
      routine, cortext::store::FactQueryMode::Current, 5500ULL);
  const auto routine_biased_exception = cortext::store::ScoreFactRecord (
      current_exception, cortext::store::FactQueryMode::Current, 5500ULL);

  cortext::operations::temporal::RetrievalAblationOverride recency_override;
  recency_override.routine_recency_mode
      = cortext::operations::temporal::RoutineRecencyMode::RecencyBiased;
  cortext::operations::temporal::ScopedRetrievalAblationOverride recency_guard (
      recency_override);
  const auto recency_biased_routine = cortext::store::ScoreFactRecord (
      routine, cortext::store::FactQueryMode::Current, 5500ULL);
  const auto recency_biased_exception = cortext::store::ScoreFactRecord (
      current_exception, cortext::store::FactQueryMode::Current, 5500ULL);
  const double balanced_gap
      = balanced_routine.combined - balanced_exception.combined;
  const double routine_gap
      = routine_biased_routine.combined - routine_biased_exception.combined;
  const double recency_gap
      = recency_biased_routine.combined - recency_biased_exception.combined;

  REQUIRE (routine_biased_routine.combined > balanced_routine.combined);
  REQUIRE (routine_biased_exception.combined < balanced_exception.combined);
  REQUIRE (recency_biased_exception.combined > balanced_exception.combined);
  REQUIRE (routine_gap > balanced_gap);
  REQUIRE (recency_gap < balanced_gap);
  REQUIRE (routine_biased_routine.combined > routine_biased_exception.combined);
  REQUIRE (recency_biased_exception.combined > recency_biased_routine.combined);
}

TEST_CASE ("Low-confidence conflicting facts do not supersede stronger current assertions",
           "[operations][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);
  SeedSummary (*store, 11LL, 21LL, "summary-2", 5000LL);

  ProcessorContext pctx;
  FixedEncoder encoder;

  operations::ExtractionResult first;
  first.summary_id = "summary-1";
  first.facts.push_back (
      { "Alice", "caregiver", "Sarah", 0.95, std::uint64_t (1000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (first));
  RunExtraction (*store, pctx, encoder, 3000ULL, 0.10, 0.80);

  operations::ExtractionResult noisy;
  noisy.summary_id = "summary-2";
  noisy.facts.push_back (
      { "Alice", "caregiver", "Emily", 0.35, std::uint64_t (5000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (noisy));
  RunExtraction (*store, pctx, encoder, 7000ULL, 0.10, 0.80);

  auto rows = store->Execute (
      "SELECT canonical_object, superseded_at_ts "
      "FROM fact_assertions "
      "WHERE canonical_subject = 'alice' AND canonical_predicate = 'caregiver' "
      "ORDER BY recorded_at_ts ASC",
      {});
  REQUIRE (rows.size () == 2);
  REQUIRE (std::any_cast<std::string> (rows[0].at ("canonical_object"))
           == "sarah");
  REQUIRE (rows[0].at ("superseded_at_ts").type () == typeid (std::nullptr_t));
  REQUIRE (std::any_cast<std::string> (rows[1].at ("canonical_object"))
           == "emily");
  REQUIRE (cortext::testing::GetInt64 (rows[1], "superseded_at_ts")
           == 7000LL);

  auto tx = store->Begin ();
  auto current = cortext::store::QueryFacts (
      *tx, { cortext::store::FactQueryMode::Current, std::string ("alice"),
             std::string ("caregiver"), 8000ULL });
  REQUIRE (current.size () == 1);
  REQUIRE (current[0].object == "Sarah");
}

TEST_CASE ("Stronger later conflict supersedes after weaker noise is ignored",
           "[operations][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);
  SeedSummary (*store, 11LL, 21LL, "summary-2", 5000LL);
  SeedSummary (*store, 12LL, 22LL, "summary-3", 9000LL);

  ProcessorContext pctx;
  FixedEncoder encoder;

  operations::ExtractionResult first;
  first.summary_id = "summary-1";
  first.facts.push_back (
      { "Alice", "caregiver", "Sarah", 0.95, std::uint64_t (1000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (first));
  RunExtraction (*store, pctx, encoder, 3000ULL, 0.10, 0.80);

  operations::ExtractionResult noisy;
  noisy.summary_id = "summary-2";
  noisy.facts.push_back (
      { "Alice", "caregiver", "Emily", 0.35, std::uint64_t (5000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (noisy));
  RunExtraction (*store, pctx, encoder, 7000ULL, 0.10, 0.80);

  operations::ExtractionResult correction;
  correction.summary_id = "summary-3";
  correction.facts.push_back (
      { "Alice", "caregiver", "Emily", 0.96, std::uint64_t (9000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (correction));
  RunExtraction (*store, pctx, encoder, 9000ULL, 0.10, 0.80);

  auto rows = store->Execute (
      "SELECT canonical_object, valid_end_ts, superseded_at_ts "
      "FROM fact_assertions "
      "WHERE canonical_subject = 'alice' AND canonical_predicate = 'caregiver' "
      "ORDER BY recorded_at_ts ASC",
      {});
  REQUIRE (rows.size () == 3);
  REQUIRE (std::any_cast<std::string> (rows[0].at ("canonical_object"))
           == "sarah");
  REQUIRE (cortext::testing::GetInt64 (rows[0], "valid_end_ts") == 9000LL);
  REQUIRE (cortext::testing::GetInt64 (rows[0], "superseded_at_ts")
           == 9000LL);
  REQUIRE (cortext::testing::GetInt64 (rows[1], "superseded_at_ts")
           == 7000LL);
  REQUIRE (std::any_cast<std::string> (rows[2].at ("canonical_object"))
           == "emily");
  REQUIRE (rows[2].at ("superseded_at_ts").type () == typeid (std::nullptr_t));

  auto tx = store->Begin ();
  auto current = cortext::store::QueryFacts (
      *tx, { cortext::store::FactQueryMode::Current, std::string ("alice"),
             std::string ("caregiver"), 10000ULL });
  REQUIRE (current.size () == 1);
  REQUIRE (current[0].object == "Emily");

  auto known_at = cortext::store::QueryFacts (
      *tx, { cortext::store::FactQueryMode::KnownAt, std::string ("alice"),
             std::string ("caregiver"), 8000ULL });
  REQUIRE (known_at.size () == 1);
  REQUIRE (known_at[0].object == "Sarah");
}

TEST_CASE ("Repeated legitimate changes preserve ordered fact history",
           "[operations][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);
  SeedSummary (*store, 11LL, 21LL, "summary-2", 5000LL);
  SeedSummary (*store, 12LL, 22LL, "summary-3", 9000LL);

  ProcessorContext pctx;
  FixedEncoder encoder;

  operations::ExtractionResult first;
  first.summary_id = "summary-1";
  first.facts.push_back (
      { "Alice", "lives_in", "Austin", 0.90, std::uint64_t (1000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (first));
  RunExtraction (*store, pctx, encoder, 3000ULL);

  operations::ExtractionResult second;
  second.summary_id = "summary-2";
  second.facts.push_back (
      { "Alice", "lives_in", "Denver", 0.94, std::uint64_t (5000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (second));
  RunExtraction (*store, pctx, encoder, 7000ULL);

  operations::ExtractionResult third;
  third.summary_id = "summary-3";
  third.facts.push_back (
      { "Alice", "lives_in", "Austin", 0.97, std::uint64_t (9000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (third));
  RunExtraction (*store, pctx, encoder, 11000ULL);

  auto tx = store->Begin ();
  auto current = cortext::store::QueryFacts (
      *tx, { cortext::store::FactQueryMode::Current, std::string ("alice"),
             std::string ("lives_in"), 12000ULL });
  REQUIRE (current.size () == 1);
  REQUIRE (current[0].object == "Austin");

  auto valid_mid = cortext::store::QueryFacts (
      *tx, { cortext::store::FactQueryMode::ValidAt, std::string ("alice"),
             std::string ("lives_in"), 8000ULL });
  REQUIRE (valid_mid.size () == 1);
  REQUIRE (valid_mid[0].object == "Denver");

  auto valid_late = cortext::store::QueryFacts (
      *tx, { cortext::store::FactQueryMode::ValidAt, std::string ("alice"),
             std::string ("lives_in"), 10000ULL });
  REQUIRE (valid_late.size () == 1);
  REQUIRE (valid_late[0].object == "Austin");

  auto rows = store->Execute (
      "SELECT canonical_object, valid_start_ts, valid_end_ts "
      "FROM fact_assertions "
      "WHERE canonical_subject = 'alice' AND canonical_predicate = 'lives_in' "
      "ORDER BY recorded_at_ts ASC",
      {});
  REQUIRE (rows.size () == 3);
  REQUIRE (std::any_cast<std::string> (rows[0].at ("canonical_object"))
           == "austin");
  REQUIRE (cortext::testing::GetInt64 (rows[0], "valid_end_ts") == 5000LL);
  REQUIRE (std::any_cast<std::string> (rows[1].at ("canonical_object"))
           == "denver");
  REQUIRE (cortext::testing::GetInt64 (rows[1], "valid_end_ts") == 9000LL);
  REQUIRE (std::any_cast<std::string> (rows[2].at ("canonical_object"))
           == "austin");
  REQUIRE (rows[2].at ("valid_end_ts").type () == typeid (std::nullptr_t));
}

TEST_CASE ("Fact lifecycle archives then deletes unsupported low-severity facts",
           "[operations][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);

  ProcessorContext pctx;
  FixedEncoder encoder;

  operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.facts.push_back (
      { "Alice", "likes", "Green tea", 0.75, std::nullopt, std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (extraction));
  RunExtraction (*store, pctx, encoder, 3000ULL);

  auto fact_rows = store->Execute (
      "SELECT fact_id FROM fact_assertions WHERE canonical_subject = 'alice' "
      "AND canonical_predicate = 'likes' AND canonical_object = 'green_tea'",
      {});
  REQUIRE (fact_rows.size () == 1);
  const long long fact_id = cortext::testing::GetInt64 (fact_rows[0], "fact_id");

  store->Execute ("DELETE FROM memories WHERE memory_id = ?", { 20LL });

  {
    auto tx = store->Begin ();
    cortext::store::MaintainFactLifecycle (*tx, &encoder, 9000ULL, { fact_id });
    tx->Commit ();
  }

  auto archived_rows = store->Execute (
      "SELECT lifecycle_state, archived_at FROM fact_assertions WHERE fact_id = ?",
      { fact_id });
  REQUIRE (archived_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (archived_rows[0].at ("lifecycle_state"))
           == "archived");
  REQUIRE (cortext::testing::GetInt64 (archived_rows[0], "archived_at")
           == 9000LL);

  {
    auto tx = store->Begin ();
    cortext::store::MaintainFactLifecycle (*tx, &encoder, 25000ULL,
                                           { fact_id });
    tx->Commit ();
  }

  auto deleted_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM fact_assertions WHERE fact_id = ?",
      { fact_id });
  REQUIRE (cortext::testing::GetInt64 (deleted_rows[0], "c") == 0LL);
}

TEST_CASE ("High-severity superseded facts survive evidence loss as archived history",
           "[operations][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);
  SeedSummary (*store, 11LL, 21LL, "summary-2", 5000LL);

  ProcessorContext pctx;
  FixedEncoder encoder;

  operations::ExtractionResult first;
  first.summary_id = "summary-1";
  first.facts.push_back (
      { "Alice", "caregiver", "Sarah", 0.9, std::uint64_t (1000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (first));
  RunExtraction (*store, pctx, encoder, 3000ULL);

  operations::ExtractionResult second;
  second.summary_id = "summary-2";
  second.facts.push_back (
      { "Alice", "caregiver", "Emily", 0.95, std::uint64_t (5000),
        std::nullopt });
  pctx.pending_extraction_results.push_back (std::move (second));
  RunExtraction (*store, pctx, encoder, 7000ULL);

  auto sarah_rows = store->Execute (
      "SELECT fact_id FROM fact_assertions WHERE canonical_subject = 'alice' "
      "AND canonical_predicate = 'caregiver' AND canonical_object = 'sarah'",
      {});
  REQUIRE (sarah_rows.size () == 1);
  const long long sarah_fact_id
      = cortext::testing::GetInt64 (sarah_rows[0], "fact_id");

  store->Execute ("DELETE FROM memories WHERE memory_id = ?", { 20LL });

  {
    auto tx = store->Begin ();
    cortext::store::MaintainFactLifecycle (*tx, &encoder, 15000ULL,
                                           { sarah_fact_id });
    tx->Commit ();
  }

  auto lifecycle_rows = store->Execute (
      "SELECT lifecycle_state FROM fact_assertions WHERE fact_id = ?",
      { sarah_fact_id });
  REQUIRE (lifecycle_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (lifecycle_rows[0].at ("lifecycle_state"))
           == "archived");

  auto tx = store->Begin ();
  const auto valid = cortext::store::QueryFacts (
      *tx, { cortext::store::FactQueryMode::ValidAt, std::string ("alice"),
             std::string ("caregiver"), 4000ULL });
  REQUIRE (valid.size () == 1);
  REQUIRE (valid[0].object == "Sarah");
}

TEST_CASE ("Repeated confirmations are compressed by default lifecycle scoring",
           "[operations][facts]")
{
  auto build_store = [] () {
    auto unique_store = SQLiteStore::Create (":memory:");
    auto shared = std::shared_ptr<Store> (std::move (unique_store));
    cortext::testing::InitializeCoreSchema (*shared);
    SeedSummary (*shared, 10LL, 20LL, "summary-1", 1000LL);
    SeedSummary (*shared, 11LL, 21LL, "summary-2", 2000LL);
    SeedSummary (*shared, 12LL, 22LL, "summary-3", 3000LL);
    SeedSummary (*shared, 13LL, 23LL, "summary-4", 4000LL);
    return shared;
  };

  auto run_sequence = [&] (std::shared_ptr<Store> store) {
    ProcessorContext pctx;
    FixedEncoder encoder;
    const std::vector<std::string> summary_ids = {
      "summary-1", "summary-2", "summary-3", "summary-4"
    };
    uint64_t ts = 5000ULL;
    for (const auto &summary_id : summary_ids)
      {
        operations::ExtractionResult extraction;
        extraction.summary_id = summary_id;
        extraction.facts.push_back (
            { "Alice", "home", "Maple House", 0.82, std::nullopt,
              std::nullopt });
        pctx.pending_extraction_results.push_back (std::move (extraction));
        RunExtraction (*store, pctx, encoder, ts);
        ts += 1000ULL;
      }
  };

  auto default_store = build_store ();
  run_sequence (default_store);

  cortext::store::FactLifecycleOptions uncompressed;
  uncompressed.compress_repeated_confirmations = false;
  cortext::store::ScopedFactLifecycleOptions uncompressed_guard (uncompressed);
  auto uncompressed_store = build_store ();
  run_sequence (uncompressed_store);

  auto default_rows = default_store->Execute (
      "SELECT support_mass, confirmation_count FROM fact_assertions "
      "WHERE canonical_subject = 'alice' AND canonical_predicate = 'home'",
      {});
  auto uncompressed_rows = uncompressed_store->Execute (
      "SELECT support_mass, confirmation_count FROM fact_assertions "
      "WHERE canonical_subject = 'alice' AND canonical_predicate = 'home'",
      {});
  REQUIRE (default_rows.size () == 1);
  REQUIRE (uncompressed_rows.size () == 1);
  REQUIRE (cortext::testing::GetInt64 (default_rows[0], "confirmation_count")
           == 4LL);
  REQUIRE (cortext::testing::GetInt64 (uncompressed_rows[0],
                                       "confirmation_count")
           == 4LL);
  REQUIRE (cortext::testing::GetDouble (default_rows[0], "support_mass")
           < cortext::testing::GetDouble (uncompressed_rows[0],
                                          "support_mass"));
}
