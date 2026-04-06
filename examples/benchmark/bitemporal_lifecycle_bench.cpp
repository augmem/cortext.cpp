#include "../../src/store/facts.hpp"

#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/process_extraction_results.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{

constexpr int kEmbeddingDim = 256;

cortext::Signal
MakeSignal (std::uint64_t ts)
{
  cortext::Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.timestamp = ts;
  s.source_id = "bench";
  return s;
}

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
SeedSummary (cortext::Store &store, long long embedding_id, long long memory_id,
             const std::string &summary_id, long long start_ts)
{
  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 1.0f;
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES (?, ?, ?)",
      { embedding_id, embedding, start_ts });
  store.Execute (
      "INSERT INTO memories(memory_id, embedding_id, source_id, kind, start_ts, "
      "n_signals, modality, s_max, s_avg, strength, created_at) "
      "VALUES (?, ?, ?, 'ASSOCIATION', ?, 1, 'text', 0.5, 0.5, 1.0, ?)",
      { memory_id, embedding_id, summary_id, start_ts, start_ts });
}

void
RunExtraction (cortext::Store &store, cortext::ProcessorContext &pctx,
               BenchEncoder &encoder,
               cortext::operations::ExtractionResult result,
               std::uint64_t signal_ts)
{
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &encoder;

  pctx.pending_extraction_results.push_back (std::move (result));
  cortext::OperationContext ctx (MakeSignal (signal_ts), pctx, cfg, &store);
  cortext::operations::ProcessExtractionResults op;
  auto tx = store.Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();
}

void
MaintainFact (cortext::Store &store, BenchEncoder &encoder, std::uint64_t ts,
              long long fact_id)
{
  auto tx = store.Begin ();
  cortext::store::MaintainFactLifecycle (*tx, &encoder, ts, { fact_id });
  tx->Commit ();
}

long long
LookupFactId (cortext::Store &store, const std::string &subject,
              const std::string &predicate, const std::string &object)
{
  auto rows = store.Execute (
      "SELECT fact_id FROM fact_assertions "
      "WHERE canonical_subject = ? AND canonical_predicate = ? "
      "AND canonical_object = ?",
      { subject, predicate, object });
  if (rows.empty ())
    {
      return 0;
    }
  if (rows[0].at ("fact_id").type () == typeid (long long))
    {
      return std::any_cast<long long> (rows[0].at ("fact_id"));
    }
  return std::any_cast<int> (rows[0].at ("fact_id"));
}

std::string
LookupLifecycleState (cortext::Store &store, long long fact_id)
{
  auto rows = store.Execute (
      "SELECT lifecycle_state FROM fact_assertions WHERE fact_id = ?",
      { fact_id });
  if (rows.empty ())
    {
      return {};
    }
  return std::any_cast<std::string> (rows[0].at ("lifecycle_state"));
}

double
LookupSupportMass (cortext::Store &store, long long fact_id)
{
  auto rows = store.Execute (
      "SELECT support_mass FROM fact_assertions WHERE fact_id = ?",
      { fact_id });
  if (rows.empty ())
    {
      return 0.0;
    }
  if (rows[0].at ("support_mass").type () == typeid (double))
    {
      return std::any_cast<double> (rows[0].at ("support_mass"));
    }
  return static_cast<double> (std::any_cast<float> (rows[0].at ("support_mass")));
}

bool
FactExists (cortext::Store &store, long long fact_id)
{
  auto rows = store.Execute (
      "SELECT COUNT(*) AS c FROM fact_assertions WHERE fact_id = ?",
      { fact_id });
  if (rows.empty ())
    {
      return false;
    }
  const auto &v = rows[0].at ("c");
  if (v.type () == typeid (long long))
    {
      return std::any_cast<long long> (v) > 0;
    }
  return std::any_cast<int> (v) > 0;
}

std::string
ValidObjectAt (cortext::Store &store, const std::string &subject,
               const std::string &predicate, std::uint64_t ts)
{
  auto tx = store.Begin ();
  const auto facts = cortext::store::QueryFacts (
      *tx, { cortext::store::FactQueryMode::ValidAt, subject, predicate, ts });
  if (facts.empty ())
    {
      return {};
    }
  return facts.front ().object;
}

struct ScenarioResult
{
  std::string name;
  bool passed = false;
};

ScenarioResult
SupportDecayScenario ()
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  BenchEncoder encoder;
  cortext::ProcessorContext pctx;
  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);

  cortext::operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.facts.push_back (
      { "Alice", "likes", "Green tea", 0.75, std::nullopt, std::nullopt });
  RunExtraction (*store, pctx, encoder, std::move (extraction), 3000ULL);
  const long long fact_id = LookupFactId (*store, "alice", "likes", "green_tea");
  store->Execute ("DELETE FROM memories WHERE memory_id = ?", { 20LL });

  MaintainFact (*store, encoder, 9000ULL, fact_id);
  const std::string default_state = LookupLifecycleState (*store, fact_id);

  auto unique_store_no_decay = cortext::SQLiteStore::Create (":memory:");
  auto no_decay_store
      = std::shared_ptr<cortext::Store> (std::move (unique_store_no_decay));
  cortext::store::ApplyMigrations (*no_decay_store);
  cortext::ProcessorContext no_decay_pctx;
  SeedSummary (*no_decay_store, 10LL, 20LL, "summary-1", 1000LL);

  cortext::store::FactLifecycleOptions no_decay;
  no_decay.support_decay_enabled = false;
  cortext::store::ScopedFactLifecycleOptions no_decay_guard (no_decay);
  cortext::operations::ExtractionResult no_decay_extraction;
  no_decay_extraction.summary_id = "summary-1";
  no_decay_extraction.facts.push_back (
      { "Alice", "likes", "Green tea", 0.75, std::nullopt, std::nullopt });
  RunExtraction (*no_decay_store, no_decay_pctx, encoder,
                 std::move (no_decay_extraction), 3000ULL);
  const long long no_decay_fact_id
      = LookupFactId (*no_decay_store, "alice", "likes", "green_tea");
  no_decay_store->Execute ("DELETE FROM memories WHERE memory_id = ?",
                           { 20LL });
  MaintainFact (*no_decay_store, encoder, 9000ULL, no_decay_fact_id);
  const std::string no_decay_state
      = LookupLifecycleState (*no_decay_store, no_decay_fact_id);

  return { "support_decay", default_state == "archived"
                                && no_decay_state != "archived" };
}

ScenarioResult
ArchiveVsDeleteScenario ()
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  BenchEncoder encoder;
  cortext::ProcessorContext pctx;
  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);

  cortext::operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.facts.push_back (
      { "Alice", "likes", "Green tea", 0.75, std::nullopt, std::nullopt });
  RunExtraction (*store, pctx, encoder, std::move (extraction), 3000ULL);
  const long long fact_id = LookupFactId (*store, "alice", "likes", "green_tea");
  store->Execute ("DELETE FROM memories WHERE memory_id = ?", { 20LL });
  MaintainFact (*store, encoder, 9000ULL, fact_id);
  MaintainFact (*store, encoder, 25000ULL, fact_id);
  const bool default_deleted = !FactExists (*store, fact_id);

  auto unique_store_keep = cortext::SQLiteStore::Create (":memory:");
  auto keep_store
      = std::shared_ptr<cortext::Store> (std::move (unique_store_keep));
  cortext::store::ApplyMigrations (*keep_store);
  cortext::ProcessorContext keep_pctx;
  SeedSummary (*keep_store, 10LL, 20LL, "summary-1", 1000LL);

  cortext::store::FactLifecycleOptions keep_archived;
  keep_archived.allow_delete = false;
  cortext::store::ScopedFactLifecycleOptions keep_guard (keep_archived);
  cortext::operations::ExtractionResult keep_extraction;
  keep_extraction.summary_id = "summary-1";
  keep_extraction.facts.push_back (
      { "Alice", "likes", "Green tea", 0.75, std::nullopt, std::nullopt });
  RunExtraction (*keep_store, keep_pctx, encoder, std::move (keep_extraction),
                 3000ULL);
  const long long keep_fact_id
      = LookupFactId (*keep_store, "alice", "likes", "green_tea");
  keep_store->Execute ("DELETE FROM memories WHERE memory_id = ?", { 20LL });
  MaintainFact (*keep_store, encoder, 9000ULL, keep_fact_id);
  MaintainFact (*keep_store, encoder, 25000ULL, keep_fact_id);

  return { "archive_vs_delete",
           default_deleted && FactExists (*keep_store, keep_fact_id)
               && LookupLifecycleState (*keep_store, keep_fact_id) == "archived" };
}

ScenarioResult
CompressionScenario ()
{
  auto build_store = [] () {
    auto unique_store = cortext::SQLiteStore::Create (":memory:");
    auto shared = std::shared_ptr<cortext::Store> (std::move (unique_store));
    cortext::store::ApplyMigrations (*shared);
    SeedSummary (*shared, 10LL, 20LL, "summary-1", 1000LL);
    SeedSummary (*shared, 11LL, 21LL, "summary-2", 2000LL);
    SeedSummary (*shared, 12LL, 22LL, "summary-3", 3000LL);
    SeedSummary (*shared, 13LL, 23LL, "summary-4", 4000LL);
    return shared;
  };

  auto run_sequence = [&] (const std::shared_ptr<cortext::Store> &store,
                           BenchEncoder &encoder) {
    cortext::ProcessorContext pctx;
    const std::vector<std::string> summary_ids = {
      "summary-1", "summary-2", "summary-3", "summary-4"
    };
    std::uint64_t ts = 5000ULL;
    for (const auto &summary_id : summary_ids)
      {
        cortext::operations::ExtractionResult extraction;
        extraction.summary_id = summary_id;
        extraction.facts.push_back (
            { "Alice", "home", "Maple House", 0.82, std::nullopt,
              std::nullopt });
        RunExtraction (*store, pctx, encoder, std::move (extraction), ts);
        ts += 1000ULL;
      }
  };

  BenchEncoder encoder;
  auto compressed_store = build_store ();
  run_sequence (compressed_store, encoder);
  const long long compressed_id
      = LookupFactId (*compressed_store, "alice", "home", "maple_house");

  cortext::store::FactLifecycleOptions uncompressed;
  uncompressed.compress_repeated_confirmations = false;
  cortext::store::ScopedFactLifecycleOptions uncompressed_guard (uncompressed);
  auto uncompressed_store = build_store ();
  run_sequence (uncompressed_store, encoder);
  const long long uncompressed_id
      = LookupFactId (*uncompressed_store, "alice", "home", "maple_house");

  return { "compression",
           LookupSupportMass (*compressed_store, compressed_id)
               < LookupSupportMass (*uncompressed_store, uncompressed_id) };
}

ScenarioResult
EvidenceLossResilienceScenario ()
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  BenchEncoder encoder;
  cortext::ProcessorContext pctx;
  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);
  SeedSummary (*store, 11LL, 21LL, "summary-2", 5000LL);

  cortext::operations::ExtractionResult first;
  first.summary_id = "summary-1";
  first.facts.push_back (
      { "Alice", "caregiver", "Sarah", 0.90, std::uint64_t (1000),
        std::nullopt });
  RunExtraction (*store, pctx, encoder, std::move (first), 3000ULL);

  cortext::operations::ExtractionResult second;
  second.summary_id = "summary-2";
  second.facts.push_back (
      { "Alice", "caregiver", "Emily", 0.95, std::uint64_t (5000),
        std::nullopt });
  RunExtraction (*store, pctx, encoder, std::move (second), 7000ULL);

  const long long sarah_id = LookupFactId (*store, "alice", "caregiver", "sarah");
  store->Execute ("DELETE FROM memories WHERE memory_id = ?", { 20LL });
  MaintainFact (*store, encoder, 15000ULL, sarah_id);
  const std::string default_valid = ValidObjectAt (*store, "alice", "caregiver",
                                                   4000ULL);

  auto unique_store_drop = cortext::SQLiteStore::Create (":memory:");
  auto drop_store
      = std::shared_ptr<cortext::Store> (std::move (unique_store_drop));
  cortext::store::ApplyMigrations (*drop_store);
  cortext::ProcessorContext drop_pctx;
  SeedSummary (*drop_store, 10LL, 20LL, "summary-1", 1000LL);
  SeedSummary (*drop_store, 11LL, 21LL, "summary-2", 5000LL);

  cortext::store::FactLifecycleOptions drop_history;
  drop_history.preserve_high_severity_history = false;
  cortext::store::ScopedFactLifecycleOptions drop_guard (drop_history);
  cortext::operations::ExtractionResult drop_first;
  drop_first.summary_id = "summary-1";
  drop_first.facts.push_back (
      { "Alice", "caregiver", "Sarah", 0.90, std::uint64_t (1000),
        std::nullopt });
  RunExtraction (*drop_store, drop_pctx, encoder, std::move (drop_first),
                 3000ULL);

  cortext::operations::ExtractionResult drop_second;
  drop_second.summary_id = "summary-2";
  drop_second.facts.push_back (
      { "Alice", "caregiver", "Emily", 0.95, std::uint64_t (5000),
        std::nullopt });
  RunExtraction (*drop_store, drop_pctx, encoder, std::move (drop_second),
                 7000ULL);

  const long long drop_sarah_id
      = LookupFactId (*drop_store, "alice", "caregiver", "sarah");
  drop_store->Execute ("DELETE FROM memories WHERE memory_id = ?", { 20LL });
  MaintainFact (*drop_store, encoder, 16000ULL, drop_sarah_id);
  MaintainFact (*drop_store, encoder, 40000ULL, drop_sarah_id);
  const std::string drop_valid
      = ValidObjectAt (*drop_store, "alice", "caregiver", 4000ULL);

  return { "evidence_loss_resilience",
           default_valid == "Sarah" && drop_valid != "Sarah" };
}

ScenarioResult
SeverityRetentionScenario ()
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);

  BenchEncoder encoder;
  cortext::ProcessorContext pctx;
  SeedSummary (*store, 10LL, 20LL, "summary-1", 1000LL);
  SeedSummary (*store, 11LL, 21LL, "summary-2", 2000LL);

  cortext::operations::ExtractionResult preference;
  preference.summary_id = "summary-1";
  preference.facts.push_back (
      { "Alice", "likes", "Green tea", 0.75, std::nullopt, std::nullopt });
  RunExtraction (*store, pctx, encoder, std::move (preference), 3000ULL);

  cortext::operations::ExtractionResult caregiver;
  caregiver.summary_id = "summary-2";
  caregiver.facts.push_back (
      { "Alice", "caregiver", "Emily", 0.90, std::uint64_t (2000),
        std::nullopt });
  RunExtraction (*store, pctx, encoder, std::move (caregiver), 4000ULL);

  const long long preference_id
      = LookupFactId (*store, "alice", "likes", "green_tea");
  const long long caregiver_id
      = LookupFactId (*store, "alice", "caregiver", "emily");
  store->Execute ("DELETE FROM memories WHERE memory_id IN (?, ?)",
                  { 20LL, 21LL });
  MaintainFact (*store, encoder, 10000ULL, preference_id);
  MaintainFact (*store, encoder, 10000ULL, caregiver_id);
  MaintainFact (*store, encoder, 40000ULL, preference_id);
  MaintainFact (*store, encoder, 40000ULL, caregiver_id);

  return { "severity_retention",
           !FactExists (*store, preference_id) && FactExists (*store, caregiver_id) };
}

} // namespace

int
main ()
{
  const std::vector<ScenarioResult> results = {
    SupportDecayScenario (),
    ArchiveVsDeleteScenario (),
    CompressionScenario (),
    EvidenceLossResilienceScenario (),
    SeverityRetentionScenario (),
  };

  int pass_count = 0;
  for (const auto &result : results)
    {
      std::cout << result.name << ": " << (result.passed ? "pass" : "fail")
                << "\n";
      if (result.passed)
        {
          ++pass_count;
        }
    }

  std::cout << "scenario_count=" << results.size () << "\n";
  std::cout << "pass_count=" << pass_count << "\n";
  std::cout << "accuracy="
            << (results.empty ()
                    ? 0.0
                    : static_cast<double> (pass_count)
                          / static_cast<double> (results.size ()))
            << "\n";

  return pass_count == static_cast<int> (results.size ()) ? 0 : 1;
}
