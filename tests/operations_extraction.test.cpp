// tests/operations_extraction.test.cpp
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/consolidation.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <string>

using namespace cortext;
using cortext::operations::EnqueueExtractionJobs;

namespace
{

static Signal
MakeSignal (uint64_t ts)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

static void
CreateInputTables (const std::shared_ptr<Store> &store)
{
  store->Execute ("CREATE TABLE IF NOT EXISTS consolidation_summaries ("
                  "summary_id TEXT PRIMARY KEY,"
                  "summary_text TEXT,"
                  "cluster_size INTEGER)");
  store->Execute ("CREATE TABLE IF NOT EXISTS consolidation_sources ("
                  "summary_id TEXT,"
                  "source_text TEXT)");
}

} // namespace

TEST_CASE ("Alg29c gating by stability disables enqueue when T <= 0.2",
           "[operations][extraction][alg29c]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  CreateInputTables (store);
  // Seed one eligible-looking summary (but T will gate it off).
  store->Execute (
      "INSERT INTO consolidation_summaries(summary_id, summary_text, "
      "cluster_size) VALUES (?,?,?)",
      { std::string ("s1"), std::string ("summary 1"), 100LL });
  store->Execute ("INSERT INTO consolidation_sources(summary_id, source_text) "
                  "VALUES (?,?)",
                  { std::string ("s1"), std::string ("src a") });

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.2; // gate off

  auto op = std::make_unique<EnqueueExtractionJobs> ();
  auto ops = std::make_unique<OperationSet> (std::move (op));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (10'000ULL));
  processor.Flush ();

  // Jobs table should exist but have zero rows.
  store->Execute ("CREATE TABLE IF NOT EXISTS extraction_jobs ("
                  "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                  "summary_id TEXT UNIQUE, prompt TEXT, status TEXT, "
                  "created_at INTEGER)");
  auto rows = store->Execute ("SELECT COUNT(*) AS c FROM extraction_jobs");
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 0LL);
}

TEST_CASE ("Alg29c enqueues job when enabled and cluster_size >= min",
           "[operations][extraction][alg29c]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  CreateInputTables (store);

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.3; // enabled

  // With F=0.5: min_cluster ≈ round(lerp(3,10,0.5)) = 7
  store->Execute (
      "INSERT INTO consolidation_summaries(summary_id, summary_text, "
      "cluster_size) VALUES (?,?,?)",
      { std::string ("s2"), std::string ("summary 2"), 7LL });
  store->Execute ("INSERT INTO consolidation_sources(summary_id, source_text) "
                  "VALUES (?,?), (?,?)",
                  { std::string ("s2"), std::string ("src b1"),
                    std::string ("s2"), std::string ("src b2") });

  auto op = std::make_unique<EnqueueExtractionJobs> ();
  auto ops = std::make_unique<OperationSet> (std::move (op));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (20'000ULL));
  processor.Flush ();

  auto rows = store->Execute ("SELECT summary_id, status FROM extraction_jobs");
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (rows[0].at ("summary_id")) == "s2");
  REQUIRE (std::any_cast<std::string> (rows[0].at ("status")) == "queued");
}

TEST_CASE ("Alg29c batches up to ExtractionBatchSize(T) per run",
           "[operations][extraction][alg29c]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  CreateInputTables (store);

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.0; // batch size = 8

  // Create 10 eligible summaries to exceed batch size.
  for (int i = 0; i < 10; ++i)
    {
      const std::string id = "s" + std::to_string (100 + i);
      store->Execute ("INSERT INTO consolidation_summaries(summary_id, "
                      "summary_text, cluster_size) VALUES (?,?,?)",
                      { id, std::string ("summary "), 10LL });
      store->Execute (
          "INSERT INTO consolidation_sources(summary_id, source_text) "
          "VALUES (?,?)",
          { id, std::string ("ctx") });
    }

  auto op = std::make_unique<EnqueueExtractionJobs> ();
  auto ops = std::make_unique<OperationSet> (std::move (op));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (30'000ULL));
  processor.Flush ();

  auto rows = store->Execute ("SELECT COUNT(*) AS c FROM extraction_jobs");
  REQUIRE (rows.size () == 1);
  // Expect only 8 rows due to batching at T=0.0.
  REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 8LL);
}

TEST_CASE ("Alg29c prompt contains summary and concatenated context",
           "[operations][extraction][alg29c]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  CreateInputTables (store);

  SignalProcessor::Config cfg;
  cfg.focus = 0.4;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.6; // enabled

  store->Execute (
      "INSERT INTO consolidation_summaries(summary_id, summary_text, "
      "cluster_size) VALUES (?,?,?)",
      { std::string ("s9"), std::string ("hello world"), 10LL });
  store->Execute ("INSERT INTO consolidation_sources(summary_id, source_text) "
                  "VALUES (?,?), (?,?)",
                  { std::string ("s9"), std::string ("line A"),
                    std::string ("s9"), std::string ("line B") });

  auto op = std::make_unique<EnqueueExtractionJobs> ();
  auto ops = std::make_unique<OperationSet> (std::move (op));
  SignalProcessor processor (cfg, store, std::move (ops));

  processor.Process (MakeSignal (40'000ULL));
  processor.Flush ();

  auto rows = store->Execute (
      "SELECT prompt FROM extraction_jobs WHERE summary_id=?", { std::string ("s9") });
  REQUIRE (rows.size () == 1);
  const auto prompt = std::any_cast<std::string> (rows[0].at ("prompt"));
  REQUIRE (prompt.find ("Summary: hello world") != std::string::npos);
  REQUIRE (prompt.find ("Context: ") != std::string::npos);
  REQUIRE (prompt.find ("line A") != std::string::npos);
  REQUIRE (prompt.find ("line B") != std::string::npos);
}

TEST_CASE ("Alg29c is idempotent on repeated runs",
           "[operations][extraction][alg29c]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  CreateInputTables (store);

  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.7; // enabled

  store->Execute (
      "INSERT INTO consolidation_summaries(summary_id, summary_text, "
      "cluster_size) VALUES (?,?,?)",
      { std::string ("sx"), std::string ("summary X"), 10LL });
  store->Execute ("INSERT INTO consolidation_sources(summary_id, source_text) "
                  "VALUES (?,?)",
                  { std::string ("sx"), std::string ("ctx") });

  auto op1 = std::make_unique<EnqueueExtractionJobs> ();
  auto ops1 = std::make_unique<OperationSet> (std::move (op1));
  SignalProcessor processor1 (cfg, store, std::move (ops1));
  processor1.Process (MakeSignal (50'000ULL));
  processor1.Flush ();

  auto op2 = std::make_unique<EnqueueExtractionJobs> ();
  auto ops2 = std::make_unique<OperationSet> (std::move (op2));
  SignalProcessor processor2 (cfg, store, std::move (ops2));
  processor2.Process (MakeSignal (51'000ULL));
  processor2.Flush ();

  auto rows = store->Execute ("SELECT COUNT(*) AS c FROM extraction_jobs WHERE summary_id=?",
                              { std::string ("sx") });
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<long long> (rows[0].at ("c")) == 1LL);
}
