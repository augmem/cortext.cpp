// tests/operations_memory_usage_record.test.cpp
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/memory_usage_record.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/sqlite_store.hpp>

using cortext::OperationContext;
using cortext::Signal;
using cortext::SignalProcessor;

namespace
{

constexpr int kEmbeddingDim = 256;

Signal
MakeSignal (uint64_t ts)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (4);
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

struct Fixture
{
  std::shared_ptr<cortext::Store> store;
  SignalProcessor::Config cfg;
  cortext::ProcessorContext pctx;

  Fixture ()
  {
    store = std::shared_ptr<cortext::Store> (
        cortext::SQLiteStore::Create (":memory:"));
    cortext::testing::InitializeCoreSchema (*store);
    cortext::testing::RequireEncoder (cfg);
    std::vector<float> vec (kEmbeddingDim, 0.0f);
    vec[0] = 1.0f;
    cortext::testing::SeedEmbeddingV2 (*store, 700LL, vec, 1);
    cortext::testing::SeedMemoryV2 (*store, 70LL, 700LL, "test",
                                    "LONG_TERM", 1.0, 1);
    cortext::ProcessorContext::RetrievalSurfaceEntry entry;
    entry.memory_id = 70LL;
    entry.embedding_id = 700LL;
    entry.created_at = 1LL;
    entry.start_ts = 1LL;
    entry.kind = "LONG_TERM";
    entry.source_id = "test";
    entry.modality = "text";
    entry.embedding = Eigen::Map<Eigen::VectorXf> (vec.data (), vec.size ());
    pctx.UpsertRetrievalSurface (std::move (entry));
  }

  void
  Record (OperationContext::MemoryUsageEvent ev, uint64_t ts)
  {
    auto signal = MakeSignal (ts);
    OperationContext ctx (signal, pctx, cfg, store.get ());
    ctx.SetMemoryUsageEvents ({ ev });
    cortext::operations::RecordMemoryUsage op;
    auto tx = store->Begin ();
    op.Execute (ctx, *tx);
    tx->Commit ();
  }

  std::map<std::string, std::any>
  Row ()
  {
    auto rows = store->Execute (
        "SELECT retrieved_count, used_count, last_used, last_access "
        "FROM memories WHERE memory_id = 70",
        {});
    REQUIRE (rows.size () == 1);
    return rows[0];
  }
};

} // namespace

TEST_CASE ("RecordMemoryUsage counts used retrievals by memory id and "
           "mirrors the retrieval surface",
           "[operations][memory_usage]")
{
  Fixture f;
  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 701LL; // memory id wins over a stale embedding id
  ev.memory_id = 70LL;
  ev.used = true;
  ev.contextual_gain = 1.0;
  f.Record (ev, 100);

  const auto row = f.Row ();
  REQUIRE (std::any_cast<long long> (row.at ("retrieved_count")) == 1LL);
  REQUIRE (std::any_cast<long long> (row.at ("used_count")) == 1LL);
  REQUIRE (std::any_cast<long long> (row.at ("last_used")) == 100LL);
  REQUIRE (std::any_cast<long long> (row.at ("last_access")) == 100LL);

  const auto cache_it = f.pctx.retrieval_surface_index.find (70LL);
  REQUIRE (cache_it != f.pctx.retrieval_surface_index.end ());
  const auto &cached = f.pctx.retrieval_surface_cache[cache_it->second];
  REQUIRE (cached.retrieved_count == 1LL);
  REQUIRE (cached.used_count == 1LL);
  REQUIRE (cached.last_access == 100LL);
}

TEST_CASE ("RecordMemoryUsage counts an unused retrieval by embedding id",
           "[operations][memory_usage]")
{
  Fixture f;
  OperationContext::MemoryUsageEvent ev{};
  ev.embedding_id = 700LL;
  ev.used = false;
  f.Record (ev, 200);
  f.Record (ev, 300);

  const auto row = f.Row ();
  REQUIRE (std::any_cast<long long> (row.at ("retrieved_count")) == 2LL);
  REQUIRE (std::any_cast<long long> (row.at ("used_count")) == 0LL);
  const auto &last_used = row.at ("last_used");
  const bool last_used_set
      = last_used.has_value () && last_used.type () == typeid (long long);
  REQUIRE_FALSE (last_used_set);
  REQUIRE (std::any_cast<long long> (row.at ("last_access")) == 300LL);
}
