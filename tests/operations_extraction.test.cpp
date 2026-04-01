// tests/operations_extraction.test.cpp
#include "test_helpers.hpp"
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/consolidation.hpp>
#include <cortext/operations/process_extraction_results.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <string>

using namespace cortext;
using cortext::operations::EnqueueExtractionJobs;

namespace
{

constexpr int kEmbeddingDim = 256;

static Signal
MakeSignal (uint64_t ts)
{
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.timestamp = ts;
  s.source_id = "test";
  return s;
}

static std::vector<operations::ExtractionRequest>
MakeRequests (int count)
{
  std::vector<operations::ExtractionRequest> requests;
  requests.reserve (static_cast<size_t> (count));
  for (int i = 0; i < count; ++i)
    {
      operations::ExtractionRequest req;
      req.summary_id = "s" + std::to_string (i);
      req.summary_text = "summary " + std::to_string (i);
      req.source_texts = { "ctx a", "ctx b" };
      req.cluster_size = 10;
      req.created_at = 0;
      requests.push_back (std::move (req));
    }
  return requests;
}

class FixedEncoder : public Encoder
{
public:
  explicit FixedEncoder (std::vector<float> embedding)
      : embedding_ (std::move (embedding))
  {
  }

  void
  EncodeText (const std::string & /*text*/,
              std::vector<float> &out_embedding) override
  {
    out_embedding = embedding_;
  }

  void
  EncodeAudio (const float * /*pcm*/, std::size_t /*num_samples*/,
               std::vector<float> &out_embedding) override
  {
    out_embedding = embedding_;
  }

  void
  EncodeImage (const std::uint8_t * /*data*/, int /*width*/, int /*height*/,
               int /*channels*/, std::vector<float> &out_embedding) override
  {
    out_embedding = embedding_;
  }

private:
  std::vector<float> embedding_;
};

} // namespace

TEST_CASE ("EnqueueExtractionJobs respects consolidation gate",
           "[operations][extraction][alg29c]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.2;

  Signal s = MakeSignal (10'000ULL);
  OperationContext ctx (s, pctx, cfg);
  ctx.SetConsolidationShouldStart (false);
  ctx.SetExtractionRequests (MakeRequests (3));

  int call_count = 0;
  operations::ExtractionCallback cb
      = [&call_count] (const std::vector<operations::ExtractionRequest> &)
  {
    ++call_count;
  };
  ctx.SetExtractionCallback (&cb);

  EnqueueExtractionJobs op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (call_count == 0);
}

TEST_CASE ("EnqueueExtractionJobs forwards requests when gate opens",
           "[operations][extraction][alg29c]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.3;

  Signal s = MakeSignal (20'000ULL);
  OperationContext ctx (s, pctx, cfg);
  ctx.SetConsolidationShouldStart (true);
  ctx.SetExtractionRequests (MakeRequests (1));

  std::vector<operations::ExtractionRequest> captured;
  operations::ExtractionCallback cb
      = [&captured] (const std::vector<operations::ExtractionRequest> &batch)
  {
    captured = batch;
  };
  ctx.SetExtractionCallback (&cb);

  EnqueueExtractionJobs op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (captured.size () == 1);
  REQUIRE (captured[0].summary_id == "s0");
}

TEST_CASE ("Alg29c batches up to ExtractionBatchSize(T) per run",
           "[operations][extraction][alg29c]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.0; // batch size = 8

  Signal s = MakeSignal (30'000ULL);
  OperationContext ctx (s, pctx, cfg);
  ctx.SetConsolidationShouldStart (true);
  ctx.SetExtractionRequests (MakeRequests (10));

  std::vector<operations::ExtractionRequest> captured;
  operations::ExtractionCallback cb
      = [&captured] (const std::vector<operations::ExtractionRequest> &batch)
  {
    captured = batch;
  };
  ctx.SetExtractionCallback (&cb);

  EnqueueExtractionJobs op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Expect only 8 rows due to batching at T=0.0.
  REQUIRE (captured.size () == 8);
}

TEST_CASE ("Alg32 caps jobs to MaxExtractionsPerCycle(T)",
           "[operations][extraction][alg32]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0; // enabled, batch size=32, max_extractions_per_cycle=5

  Signal s = MakeSignal (60'000ULL);
  OperationContext ctx (s, pctx, cfg);
  ctx.SetConsolidationShouldStart (true);
  ctx.SetExtractionRequests (MakeRequests (10));

  std::vector<operations::ExtractionRequest> captured;
  operations::ExtractionCallback cb
      = [&captured] (const std::vector<operations::ExtractionRequest> &batch)
  {
    captured = batch;
  };
  ctx.SetExtractionCallback (&cb);

  EnqueueExtractionJobs op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (captured.size () == 5);
}

TEST_CASE ("EnqueueExtractionJobs forwards summary and sources",
           "[operations][extraction][alg29c]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.4;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.6;

  Signal s = MakeSignal (40'000ULL);
  OperationContext ctx (s, pctx, cfg);
  ctx.SetConsolidationShouldStart (true);

  operations::ExtractionRequest req;
  req.summary_id = "s9";
  req.summary_text = "hello world";
  req.source_texts = { "line A", "line B" };
  req.cluster_size = 10;
  req.created_at = 0;
  ctx.SetExtractionRequests ({ req });

  std::vector<operations::ExtractionRequest> captured;
  operations::ExtractionCallback cb
      = [&captured] (const std::vector<operations::ExtractionRequest> &batch)
  {
    captured = batch;
  };
  ctx.SetExtractionCallback (&cb);

  EnqueueExtractionJobs op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  REQUIRE (captured.size () == 1);
  REQUIRE (captured[0].summary_text == "hello world");
  REQUIRE (captured[0].source_texts.size () == 2);
  REQUIRE (captured[0].source_texts[0] == "line A");
  REQUIRE (captured[0].source_texts[1] == "line B");
}

TEST_CASE ("Alg29c is idempotent on repeated runs",
           "[operations][extraction][alg29c]")
{
  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.7; // enabled

  int call_count = 0;
  operations::ExtractionCallback cb
      = [&call_count] (const std::vector<operations::ExtractionRequest> &)
  {
    ++call_count;
  };

  Signal s1 = MakeSignal (50'000ULL);
  OperationContext ctx1 (s1, pctx, cfg);
  ctx1.SetConsolidationShouldStart (true);
  ctx1.SetExtractionRequests (MakeRequests (1));
  ctx1.SetExtractionCallback (&cb);

  EnqueueExtractionJobs op;
  op.Execute (ctx1, cortext::testing::GetNullTransaction ());

  Signal s2 = MakeSignal (50'000ULL);
  OperationContext ctx2 (s2, pctx, cfg);
  ctx2.SetConsolidationShouldStart (true);
  ctx2.SetExtractionRequests (MakeRequests (1));
  ctx2.SetExtractionCallback (&cb);

  // Second call within the extraction interval should be suppressed.
  op.Execute (ctx2, cortext::testing::GetNullTransaction ());

  REQUIRE (call_count == 1);
}

TEST_CASE ("ProcessExtractionResults computes label salience from embeddings",
           "[operations][extraction][labels]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "summary-1",
                                 "ASSOCIATION", 1.0, 1000LL);

  FixedEncoder encoder (unit_embedding);

  SignalProcessor::Config cfg;

  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.0;
  cfg.encoder = &encoder;

  ProcessorContext pctx;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  for (int i = 0; i < 5; ++i)
    {
      extraction.labels.push_back ({ "Acme", 0.0 });
    }
  pctx.pending_extraction_results.push_back (std::move (extraction));

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT s_max, label FROM memories WHERE kind = 'LABEL' AND source_id = ?",
      { std::string ("acme") });
  REQUIRE (rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (rows[0].at ("label")) == "Acme");
  const double s_max = cortext::testing::GetDouble (rows[0], "s_max");
  REQUIRE (s_max == Catch::Approx (1.0));
}
