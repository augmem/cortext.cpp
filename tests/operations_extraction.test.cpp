// tests/operations_extraction.test.cpp
#include "test_helpers.hpp"
#include <any>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cortext/encoder/encoder.hpp>
#include <cortext/extractor/extractor.hpp>
#include <cortext/extractor/gemma_extractor.hpp>
#include <cortext/operations/consolidation.hpp>
#include <cortext/operations/label_utils.hpp>
#include <cortext/operations/process_extraction_results.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

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

class FixedExtractor : public Extractor
{
public:
  explicit FixedExtractor (std::vector<std::string> labels)
      : labels_ (std::move (labels))
  {
  }

  FixedExtractor (std::vector<std::string> labels,
                  std::vector<operations::ExtractedRelation> relations)
      : labels_ (std::move (labels)), relations_ (std::move (relations))
  {
  }

  operations::ExtractionResult
  ExtractFromText (const std::string & /*text*/,
                   const nlohmann::json & /*schema*/) override
  {
    operations::ExtractionResult result;
    for (const auto &label : labels_)
      {
        result.labels.push_back ({ label, 1.0 });
      }
    result.relations = relations_;
    return result;
  }

  operations::ExtractionResult
  ExtractFromAudio (const float * /*pcm*/, size_t /*num_samples*/,
                    const nlohmann::json & /*schema*/) override
  {
    return ExtractFromText ("", nlohmann::json::object ());
  }

  bool
  IsAvailable () const override
  {
    return true;
  }

private:
  std::vector<std::string> labels_;
  std::vector<operations::ExtractedRelation> relations_;
};

struct LabelEdgeMetrics
{
  int true_positive = 0;
  int false_positive = 0;
  int false_negative = 0;
  int edge_count = 0;

  double
  Precision () const
  {
    const int denom = true_positive + false_positive;
    return denom > 0 ? static_cast<double> (true_positive) / denom : 0.0;
  }

  double
  Recall () const
  {
    const int denom = true_positive + false_negative;
    return denom > 0 ? static_cast<double> (true_positive) / denom : 0.0;
  }
};

static void
SeedAssociationWithLabels (Store &store,
                           const std::vector<std::string> &labels)
{
  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;
  cortext::testing::SeedEmbeddingV2 (store, 10LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (store, 20LL, 10LL, "associative_cue_1",
                                  "ASSOCIATION", 1.0, 1000LL);

  long long next_embedding_id = 100LL;
  long long next_memory_id = 200LL;
  for (const auto &label : labels)
    {
      const std::string key = operations::NormalizeLabelKey (label);
      std::vector<float> label_embedding (kEmbeddingDim, 0.0f);
      label_embedding[0] = 1.0f;
      cortext::testing::SeedEmbeddingV2 (store, next_embedding_id,
                                         label_embedding, 1000LL);
      store.Execute (
          "INSERT OR REPLACE INTO memories "
          "(memory_id, embedding_id, source_id, kind, label, start_ts, "
          "n_signals, modality, s_max, s_avg, strength, created_at) "
          "VALUES (?, ?, ?, 'LABEL', ?, 1000, 1, 'text', 1.0, 1.0, 1.0, 1000)",
          { next_memory_id, next_embedding_id, key, label });
      store.Execute (
          "INSERT OR REPLACE INTO associations "
          "(source_memory_id, target_memory_id, edge_type, weight) "
          "VALUES (20, ?, 'has_label', 1.0)",
          { next_memory_id });
      ++next_embedding_id;
      ++next_memory_id;
    }
}

static LabelEdgeMetrics
MeasureLabelEdges (Store &store, const std::set<std::string> &truth)
{
  auto rows = store.Execute (
      "SELECT l.source_id FROM associations a "
      "JOIN memories l ON l.memory_id = a.target_memory_id "
      "WHERE a.source_memory_id = 20 AND a.edge_type = 'has_label' "
      "  AND l.kind = 'LABEL'",
      {});
  LabelEdgeMetrics metrics;
  metrics.edge_count = static_cast<int> (rows.size ());
  std::set<std::string> observed;
  for (const auto &row : rows)
    {
      auto it = row.find ("source_id");
      if (it == row.end () || it->second.type () != typeid (std::string))
        {
          continue;
        }
      const std::string key
          = operations::NormalizeLabelKey (std::any_cast<std::string> (it->second));
      observed.insert (key);
      if (truth.find (key) != truth.end ())
        {
          ++metrics.true_positive;
        }
      else
        {
          ++metrics.false_positive;
        }
    }
  for (const auto &label : truth)
    {
      if (observed.find (label) == observed.end ())
        {
          ++metrics.false_negative;
        }
    }
  return metrics;
}

static std::string
FindModelPath (const std::string &relative_path)
{
  std::filesystem::path probe = std::filesystem::current_path ();
  for (int i = 0; i < 6; ++i)
    {
      const auto candidate = probe / relative_path;
      if (std::filesystem::exists (candidate))
        {
          return candidate.string ();
        }
      if (!probe.has_parent_path ())
        {
          break;
        }
      probe = probe.parent_path ();
    }
  return relative_path;
}

static LabelEdgeMetrics
RunBlobConsolidationLabelCase (
    Extractor *extractor, const std::vector<std::string> &provisional_labels,
    const std::vector<std::string> &refined_labels,
    const std::set<std::string> &truth, const std::string &evidence_text,
    bool replacement_path)
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);
  SeedAssociationWithLabels (*store, provisional_labels);

  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;
  FixedEncoder encoder (unit_embedding);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.48;
  cfg.sensitivity = 0.94;
  cfg.stability = 0.88;
  cfg.encoder = &encoder;
  cfg.extractor = replacement_path ? extractor : nullptr;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = replacement_path ? extractor : nullptr;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  if (replacement_path)
    {
      operations::ExtractionRequest req;
      req.summary_id = "associative_cue_1";
      req.summary_text = evidence_text;
      req.source_texts = { evidence_text };
      req.current_labels = provisional_labels;
      req.cluster_size = 5;
      req.created_at = 2000ULL;
      ctx.SetExtractionRequests ({ req });
    }
  else
    {
      operations::ExtractionResult extraction;
      extraction.summary_id = "associative_cue_1";
      for (const auto &label : refined_labels)
        {
          extraction.labels.push_back ({ label, 1.0 });
        }
      pctx.pending_extraction_results.push_back (std::move (extraction));
    }

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  return MeasureLabelEdges (*store, truth);
}

static nlohmann::json
BuildExtractionSchema ()
{
  return nlohmann::json::parse (R"({
    "type": "object",
    "properties": {
      "labels": {
        "type": "array",
        "minItems": 0,
        "items": {"type": "string"}
      },
      "relations": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "confidence": {"type": "number"}
          },
          "required": ["subject", "predicate", "object"]
        }
      },
      "facts": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "confidence": {"type": "number"},
            "valid_start_ts": {"type": "integer"},
            "valid_end_ts": {"type": "integer"}
          },
          "required": ["subject", "predicate", "object"]
        }
      }
    },
    "required": ["labels", "relations"]
  })");
}

static std::vector<float>
ReadFloat32File (const std::string &path)
{
  std::ifstream in (path, std::ios::binary);
  REQUIRE (in.good ());
  in.seekg (0, std::ios::end);
  const auto bytes = in.tellg ();
  REQUIRE (bytes >= 0);
  REQUIRE (static_cast<std::streamoff> (bytes) % sizeof (float) == 0);
  in.seekg (0, std::ios::beg);
  std::vector<float> values (
      static_cast<size_t> (bytes) / sizeof (float));
  in.read (reinterpret_cast<char *> (values.data ()), bytes);
  REQUIRE ((in.good () || in.eof ()));
  return values;
}

static std::vector<unsigned char>
ReadBytesFile (const std::string &path)
{
  std::ifstream in (path, std::ios::binary);
  REQUIRE (in.good ());
  in.seekg (0, std::ios::end);
  const auto bytes = in.tellg ();
  REQUIRE (bytes >= 0);
  in.seekg (0, std::ios::beg);
  std::vector<unsigned char> values (static_cast<size_t> (bytes));
  in.read (reinterpret_cast<char *> (values.data ()), bytes);
  REQUIRE ((in.good () || in.eof ()));
  return values;
}

static std::vector<std::string>
ExtractLabelStrings (const operations::ExtractionResult &result)
{
  std::vector<std::string> labels;
  labels.reserve (result.labels.size ());
  for (const auto &label : result.labels)
    {
      labels.push_back (label.label);
    }
  return labels;
}

static bool
HasLabelContaining (const std::vector<std::string> &labels,
                    const std::string &needle)
{
  const std::string expected = operations::NormalizeLabelKey (needle);
  for (const auto &label : labels)
    {
      if (operations::NormalizeLabelKey (label).find (expected)
          != std::string::npos)
        {
          return true;
        }
    }
  return false;
}

static void
PrintLabels (const std::string &modality,
             const std::vector<std::string> &labels)
{
  std::cout << modality << ",";
  for (size_t i = 0; i < labels.size (); ++i)
    {
      if (i > 0)
        {
          std::cout << "|";
        }
      std::cout << labels[i];
    }
  std::cout << std::endl;
}

} // namespace

TEST_CASE ("EnqueueExtractionJobs respects consolidation gate",
           "[operations][extraction][alg29c]")
{
  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
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
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
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
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
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
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
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
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
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
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
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
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
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

TEST_CASE ("ProcessExtractionResults stores 1536-dim label encoder output as retrieval-sized embeddings",
           "[operations][extraction][labels]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> summary_embedding (kEmbeddingDim, 0.0f);
  summary_embedding[0] = 1.0f;

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, summary_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "summary-1",
                                  "ASSOCIATION", 1.0, 1000LL);

  std::vector<float> encoder_embedding (1536, 0.0f);
  encoder_embedding[0] = 1.0f;
  encoder_embedding[300] = 1.0f;
  FixedEncoder encoder (encoder_embedding);

    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder (cfg);
    cfg.focus = 0.75;
    cfg.sensitivity = 0.95;
    cfg.stability = 0.85;
    cfg.encoder = &encoder;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.labels.push_back ({ "Retrieval Sized Label", 0.0 });
  pctx.pending_extraction_results.push_back (std::move (extraction));

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT length(e.embedding) AS bytes "
      "FROM memories m JOIN embeddings e ON e.embedding_id = m.embedding_id "
      "WHERE m.kind = 'LABEL' AND m.source_id = ?",
      { std::string ("retrieval sized label") });
  REQUIRE (rows.size () == 1);
  REQUIRE (cortext::testing::GetInt64 (rows[0], "bytes")
           == static_cast<long long> (kEmbeddingDim * sizeof (float)));
}

TEST_CASE ("ProcessExtractionResults preserves multiple distinct labels from one extraction result",
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
  cfg.stability = 1.0;
  cfg.encoder = &encoder;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.labels.push_back ({ "Gabriel", 0.0 });
  extraction.labels.push_back ({ "Chicago", 0.0 });
  extraction.labels.push_back ({ "Cortext", 0.0 });
  pctx.pending_extraction_results.push_back (std::move (extraction));

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto label_rows = store->Execute (
      "SELECT source_id, label FROM memories WHERE kind = 'LABEL' ORDER BY source_id",
      {});
  REQUIRE (label_rows.size () == 3);
  REQUIRE (std::any_cast<std::string> (label_rows[0].at ("source_id")) == "chicago");
  REQUIRE (std::any_cast<std::string> (label_rows[1].at ("source_id")) == "cortext");
  REQUIRE (std::any_cast<std::string> (label_rows[2].at ("source_id")) == "gabriel");

  auto edge_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM associations WHERE edge_type = 'has_label'",
      {});
  REQUIRE (cortext::testing::GetInt64 (edge_rows[0], "c") == 3);
}

TEST_CASE ("ProcessExtractionResults materializes relation endpoints as durable labels",
           "[operations][extraction][labels][relations]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "summary-1",
                                  "ASSOCIATION", 1.0, 1000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 30LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 30LL, 30LL,
                                  "Maria met Bailey at River Park.",
                                  "LONG_TERM", 1.0, 1000LL);
  std::vector<unsigned char> source_blob = { 'M', 'a', 'r', 'i', 'a' };
  auto blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                   { source_blob });
  REQUIRE (blob_rows.size () == 1);
  store->Execute ("UPDATE memories SET blob_id = ? WHERE memory_id = 30",
                  { blob_rows[0].at ("id") });
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (20, 30, 'derived_from', 1.0)",
      {});

  FixedEncoder encoder (unit_embedding);

	    SignalProcessor::Config cfg;
	    cortext::testing::RequireEncoder (cfg);
	    cfg.focus = 0.75;
	    cfg.sensitivity = 0.95;
	    cfg.stability = 0.85;
	    cfg.encoder = &encoder;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.labels.push_back ({ "Maria", 0.0 });
  extraction.relations.push_back (
      { "Maria", "reinforces", "Dog", 0.75 });
  pctx.pending_extraction_results.push_back (std::move (extraction));

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto label_rows = store->Execute (
      "SELECT source_id FROM memories WHERE kind = 'LABEL' ORDER BY source_id",
      {});
  REQUIRE (label_rows.size () == 2);
  REQUIRE (std::any_cast<std::string> (label_rows[0].at ("source_id")) == "dog");
  REQUIRE (std::any_cast<std::string> (label_rows[1].at ("source_id")) == "maria");

  auto relation_rows = store->Execute (
      "SELECT a.edge_type, a.weight "
      "FROM associations a "
      "JOIN memories s ON s.memory_id = a.source_memory_id "
      "JOIN memories o ON o.memory_id = a.target_memory_id "
      "WHERE s.source_id = 'maria' AND o.source_id = 'dog'",
      {});
  REQUIRE (relation_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (relation_rows[0].at ("edge_type"))
           == "reinforces");
  REQUIRE (cortext::testing::GetDouble (relation_rows[0], "weight")
           == Catch::Approx (0.75));
}

TEST_CASE ("ProcessExtractionResults audits relation skip reasons",
           "[operations][extraction][labels][relations]")
{
  cortext::testing::ScopedEnvVar enable_audit ("CORTEXT_STM_LTM_AUDIT", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "summary-1",
                                  "ASSOCIATION", 1.0, 1000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 30LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 30LL, 30LL,
                                  "source-memory-1", "LONG_TERM", 1.0,
                                  1000LL);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (20, 30, 'derived_from', 1.0)",
      {});
  store->Execute (
      "CREATE TABLE stm_ltm_relabel_audit ("
      "summary_id TEXT PRIMARY KEY, created_at INTEGER, cluster_size INTEGER, "
      "source_memory_count INTEGER, source_text_count INTEGER, "
      "source_blob_count INTEGER, source_memory_ids TEXT, "
      "stm_graph_count INTEGER, stm_item_count INTEGER, "
      "stm_label_edge_count INTEGER, current_label_count INTEGER, "
      "current_labels TEXT, refined_label_count INTEGER DEFAULT 0, "
      "refined_labels TEXT DEFAULT '', kept_label_count INTEGER DEFAULT 0, "
      "added_label_count INTEGER DEFAULT 0, removed_label_count INTEGER DEFAULT 0, "
      "removed_labels TEXT DEFAULT '', has_label_edges_after INTEGER DEFAULT 0, "
      "derived_from_edges INTEGER DEFAULT 0, relation_count INTEGER DEFAULT 0, "
      "relation_edges_created INTEGER DEFAULT 0, "
      "relation_edges_skipped_missing_endpoint INTEGER DEFAULT 0, "
      "relation_edges_skipped_unsupported_predicate INTEGER DEFAULT 0, "
      "fact_assertions_touched INTEGER DEFAULT 0, "
      "source_memories_with_content INTEGER DEFAULT 0"
      ")",
      {});
  store->Execute (
      "INSERT INTO stm_ltm_relabel_audit "
      "(summary_id, created_at, cluster_size, source_memory_count, "
      "source_text_count, source_blob_count, source_memory_ids, "
      "stm_graph_count, stm_item_count, stm_label_edge_count, "
      "current_label_count, current_labels) "
      "VALUES ('summary-1', 1000, 1, 1, 1, 0, '20', 1, 1, 1, 0, '')",
      {});

  FixedEncoder encoder (unit_embedding);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.encoder = &encoder;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.labels.push_back ({ "Maria", 0.0 });
  extraction.relations.push_back (
      { "Maria", "reinforces", "Dog", 0.75 });
  extraction.relations.push_back (
      { "Maria", "unsupported", "Dog", 0.75 });
  extraction.relations.push_back (
      { "Maria", "", "Dog", 0.75 });
  extraction.relations.push_back (
      { "Maria", "!!!", "Dog", 0.75 });
  pctx.pending_extraction_results.push_back (std::move (extraction));

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto audit_rows = store->Execute (
      "SELECT relation_count, relation_edges_created, "
      "relation_edges_skipped_non_durable_endpoint, "
      "relation_edges_skipped_missing_endpoint, "
      "relation_edges_skipped_unsupported_predicate "
      "FROM stm_ltm_relabel_audit WHERE summary_id = 'summary-1'",
      {});
  REQUIRE (audit_rows.size () == 1);
  REQUIRE (cortext::testing::GetInt64 (audit_rows[0], "relation_count") == 4);
  REQUIRE (cortext::testing::GetInt64 (audit_rows[0], "relation_edges_created")
           == 4);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "relation_edges_skipped_non_durable_endpoint")
           == 0);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "relation_edges_skipped_missing_endpoint")
           == 0);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "relation_edges_skipped_unsupported_predicate")
           == 0);
}

TEST_CASE ("ProcessExtractionResults drops relations with non durable endpoints",
           "[operations][extraction][labels][relations]")
{
  cortext::testing::ScopedEnvVar enable_audit ("CORTEXT_STM_LTM_AUDIT", "1");
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "summary-1",
                                  "ASSOCIATION", 1.0, 1000LL);
  store->Execute (
      "CREATE TABLE stm_ltm_relabel_audit ("
      "summary_id TEXT PRIMARY KEY, created_at INTEGER, cluster_size INTEGER, "
      "source_memory_count INTEGER, source_text_count INTEGER, "
      "source_blob_count INTEGER, source_memory_ids TEXT, "
      "stm_graph_count INTEGER, stm_item_count INTEGER, "
      "stm_label_edge_count INTEGER, current_label_count INTEGER, "
      "current_labels TEXT, refined_label_count INTEGER DEFAULT 0, "
      "refined_labels TEXT DEFAULT '', kept_label_count INTEGER DEFAULT 0, "
      "added_label_count INTEGER DEFAULT 0, removed_label_count INTEGER DEFAULT 0, "
      "removed_labels TEXT DEFAULT '', has_label_edges_after INTEGER DEFAULT 0, "
      "derived_from_edges INTEGER DEFAULT 0, relation_count INTEGER DEFAULT 0, "
      "relation_edges_created INTEGER DEFAULT 0, "
      "relation_edges_skipped_non_durable_endpoint INTEGER DEFAULT 0, "
      "relation_edges_skipped_missing_endpoint INTEGER DEFAULT 0, "
      "relation_edges_skipped_unsupported_predicate INTEGER DEFAULT 0, "
      "relation_endpoint_direct_hits INTEGER DEFAULT 0, "
      "relation_endpoint_repair_hits INTEGER DEFAULT 0, "
      "relation_endpoint_created_labels INTEGER DEFAULT 0, "
      "relation_endpoint_relation_backed_labels INTEGER DEFAULT 0, "
      "relation_endpoint_rejected_count INTEGER DEFAULT 0, "
      "relation_endpoint_rejected_non_durable INTEGER DEFAULT 0, "
      "relation_endpoint_rejected_ungrounded INTEGER DEFAULT 0, "
      "fact_assertions_touched INTEGER DEFAULT 0, "
      "source_memories_with_content INTEGER DEFAULT 0"
      ")",
      {});
  store->Execute (
      "INSERT INTO stm_ltm_relabel_audit "
      "(summary_id, created_at, cluster_size, source_memory_count, "
      "source_text_count, source_blob_count, source_memory_ids, "
      "stm_graph_count, stm_item_count, stm_label_edge_count, "
      "current_label_count, current_labels) "
      "VALUES ('summary-1', 1000, 1, 1, 1, 0, '20', 1, 1, 1, 0, '')",
      {});

  FixedEncoder encoder (unit_embedding);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.encoder = &encoder;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.labels.push_back ({ "Maria", 0.0 });
  extraction.relations.push_back (
      { "Maria", "co_occurs", "thing", 0.8 });
  pctx.pending_extraction_results.push_back (std::move (extraction));

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto relation_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM associations WHERE edge_type = 'co_occurs'",
      {});
  REQUIRE (cortext::testing::GetInt64 (relation_rows[0], "c") == 0);

  auto filler_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE kind = 'LABEL' AND source_id = 'thing'",
      {});
  REQUIRE (cortext::testing::GetInt64 (filler_rows[0], "c") == 0);

  auto audit_rows = store->Execute (
      "SELECT relation_count, relation_edges_created, "
      "relation_edges_skipped_non_durable_endpoint, "
      "relation_edges_skipped_missing_endpoint, "
      "relation_endpoint_rejected_non_durable "
      "FROM stm_ltm_relabel_audit WHERE summary_id = 'summary-1'",
      {});
  REQUIRE (audit_rows.size () == 1);
  REQUIRE (cortext::testing::GetInt64 (audit_rows[0], "relation_count") == 1);
  REQUIRE (cortext::testing::GetInt64 (audit_rows[0], "relation_edges_created")
           == 0);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "relation_edges_skipped_non_durable_endpoint")
           == 1);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "relation_edges_skipped_missing_endpoint")
           == 0);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "relation_endpoint_rejected_non_durable")
           == 1);
}

TEST_CASE ("ProcessExtractionResults repairs relation endpoints to admitted labels",
           "[operations][extraction][labels][relations]")
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
  cfg.encoder = &encoder;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.labels.push_back ({ "Maria", 0.0 });
  extraction.labels.push_back ({ "Bailey dog", 0.0 });
  extraction.relations.push_back (
      { "Maria", "co_occurs", "Dog", 0.8 });
  pctx.pending_extraction_results.push_back (std::move (extraction));

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto dog_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE kind = 'LABEL' AND source_id = 'dog'",
      {});
  REQUIRE (cortext::testing::GetInt64 (dog_rows[0], "c") == 0);

  auto relation_rows = store->Execute (
      "SELECT a.edge_type, a.weight "
      "FROM associations a "
      "JOIN memories s ON s.memory_id = a.source_memory_id "
      "JOIN memories o ON o.memory_id = a.target_memory_id "
      "WHERE s.source_id = 'maria' AND o.source_id = 'bailey dog'",
      {});
  REQUIRE (relation_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (relation_rows[0].at ("edge_type"))
           == "co_occurs");
  REQUIRE (cortext::testing::GetDouble (relation_rows[0], "weight")
           == Catch::Approx (0.8));
}

TEST_CASE ("ProcessExtractionResults repairs non-durable relation endpoints before rejection",
           "[operations][extraction][labels][relations]")
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
  cfg.encoder = &encoder;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.labels.push_back ({ "Maria", 0.0 });
  extraction.labels.push_back ({ "Bailey dog", 0.0 });
  extraction.relations.push_back (
      { "Maria", "co_occurs", "dog", 0.8 });
  pctx.pending_extraction_results.push_back (std::move (extraction));

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto dog_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE kind = 'LABEL' AND source_id = 'dog'",
      {});
  REQUIRE (cortext::testing::GetInt64 (dog_rows[0], "c") == 0);

  auto relation_rows = store->Execute (
      "SELECT a.edge_type, a.weight "
      "FROM associations a "
      "JOIN memories s ON s.memory_id = a.source_memory_id "
      "JOIN memories o ON o.memory_id = a.target_memory_id "
      "WHERE s.source_id = 'maria' AND o.source_id = 'bailey dog'",
      {});
  REQUIRE (relation_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (relation_rows[0].at ("edge_type"))
           == "co_occurs");
  REQUIRE (cortext::testing::GetDouble (relation_rows[0], "weight")
           == Catch::Approx (0.8));
}

TEST_CASE ("ProcessExtractionResults repairs relation endpoints to current STM labels",
           "[operations][extraction][labels][relations]")
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
  FixedExtractor extractor (
      { "Maria" },
      { operations::ExtractedRelation { "Maria", "co_occurs", "Dog", 0.8 } });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.12;
  cfg.sensitivity = 0.83;
  cfg.stability = 0.98;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text = "Maria was there.";
  req.source_texts = { "Maria was there." };
  req.current_labels = { "Bailey dog" };
  req.cluster_size = 5;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto dog_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE kind = 'LABEL' AND source_id = 'dog'",
      {});
  REQUIRE (cortext::testing::GetInt64 (dog_rows[0], "c") == 0);

  auto relation_rows = store->Execute (
      "SELECT a.edge_type, a.weight "
      "FROM associations a "
      "JOIN memories s ON s.memory_id = a.source_memory_id "
      "JOIN memories o ON o.memory_id = a.target_memory_id "
      "WHERE s.source_id = 'maria' AND o.source_id = 'bailey dog'",
      {});
  REQUIRE (relation_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (relation_rows[0].at ("edge_type"))
           == "co_occurs");
  REQUIRE (cortext::testing::GetDouble (relation_rows[0], "weight")
           == Catch::Approx (0.8));
}

TEST_CASE ("ProcessExtractionResults admits high confidence relation-backed endpoints",
           "[operations][extraction][labels][relations]")
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
  FixedExtractor extractor (
      { "Maria" },
      { operations::ExtractedRelation { "Maria", "co_occurs", "Dog", 0.8 } });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.48;
  cfg.sensitivity = 0.76;
  cfg.stability = 0.88;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text = "Maria was there.";
  req.source_texts = { "Maria was there." };
  req.cluster_size = 5;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto relation_rows = store->Execute (
      "SELECT a.edge_type, a.weight "
      "FROM associations a "
      "JOIN memories s ON s.memory_id = a.source_memory_id "
      "JOIN memories o ON o.memory_id = a.target_memory_id "
      "WHERE s.source_id = 'maria' AND o.source_id = 'dog'",
      {});
  REQUIRE (relation_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (relation_rows[0].at ("edge_type"))
           == "co_occurs");
  REQUIRE (cortext::testing::GetDouble (relation_rows[0], "weight")
           == Catch::Approx (0.8));
}

TEST_CASE ("ProcessExtractionResults admits relation endpoints from blob-backed extraction",
           "[operations][extraction][labels][relations]")
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
  FixedExtractor extractor (
      { "Maria" },
      { operations::ExtractedRelation { "Maria", "near", "Dog", 0.8 } });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.81;
  cfg.sensitivity = 0.87;
  cfg.stability = 0.46;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text = "Maria was there.";
  req.source_texts = { "Maria was there." };
  req.source_blobs.push_back (
      operations::ExtractionSourceBlob { { 1, 2, 3 }, "image", "image/jpeg" });
  req.cluster_size = 5;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto relation_rows = store->Execute (
      "SELECT a.edge_type "
      "FROM associations a "
      "JOIN memories s ON s.memory_id = a.source_memory_id "
      "JOIN memories o ON o.memory_id = a.target_memory_id "
      "WHERE s.source_id = 'maria' AND o.source_id = 'dog'",
      {});
  REQUIRE (relation_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (relation_rows[0].at ("edge_type"))
           == "co_occurs");
}

TEST_CASE ("ProcessExtractionResults rejects transcript filler labels",
           "[operations][extraction][labels]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "associative_cue_1",
                                 "ASSOCIATION", 1.0, 1000LL);

  FixedEncoder encoder (unit_embedding);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.encoder = &encoder;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionResult extraction;
  extraction.summary_id = "associative_cue_1";
  extraction.labels.push_back ({ "ya", 0.0 });
  extraction.labels.push_back ({ "know", 0.0 });
  extraction.labels.push_back ({ "User", 0.0 });
  extraction.labels.push_back ({ "Assistant", 0.0 });
  extraction.labels.push_back ({ "thanks", 0.0 });
  extraction.labels.push_back ({ "that", 0.0 });
  extraction.labels.push_back ({ "this", 0.0 });
  extraction.labels.push_back ({ "thing", 0.0 });
  extraction.labels.push_back ({ "get", 0.0 });
  extraction.labels.push_back ({ "go", 0.0 });
  extraction.labels.push_back ({ "make", 0.0 });
  extraction.labels.push_back ({ "might", 0.0 });
  extraction.labels.push_back ({ "idea", 0.0 });
  extraction.labels.push_back ({ "food", 0.0 });
  extraction.labels.push_back ({ "stuff", 0.0 });
  extraction.labels.push_back ({ "almost done", 0.0 });
  extraction.labels.push_back ({ "Gabriel", 0.0 });
  pctx.pending_extraction_results.push_back (std::move (extraction));

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto label_rows = store->Execute (
      "SELECT source_id, label FROM memories WHERE kind = 'LABEL'",
      {});
  REQUIRE (label_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (label_rows[0].at ("source_id"))
           == "gabriel");

  auto edge_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM associations WHERE edge_type = 'has_label'",
      {});
  REQUIRE (cortext::testing::GetInt64 (edge_rows[0], "c") == 1);
}

TEST_CASE ("Blob-aware consolidation replaces noisy preconsolidated label edges",
           "[operations][extraction][labels][ablation][blob-consolidation]")
{
  const std::set<std::string> truth = { "maria", "dog", "car" };
  const std::vector<std::string> provisional_labels
      = { "Maria", "Dog", "Steve", "Office", "Crash" };
  const std::vector<std::string> refined_labels
      = { "Maria", "Dog", "Car" };
  const std::string evidence_text
      = "Maria trained the dog after seeing a car crash.";

  auto run_case = [&] (bool replacement_path) {
    auto unique_store = SQLiteStore::Create (":memory:");
    auto store = std::shared_ptr<Store> (std::move (unique_store));
    cortext::testing::InitializeCoreSchema (*store);
    SeedAssociationWithLabels (*store, provisional_labels);

    std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
    unit_embedding[0] = 1.0f;
    FixedEncoder encoder (unit_embedding);
	    FixedExtractor extractor (refined_labels);

	    SignalProcessor::Config cfg;
	    cortext::testing::RequireEncoder (cfg);
	    cfg.focus = 0.75;
	    cfg.sensitivity = 0.95;
	    cfg.stability = 0.85;
	    cfg.encoder = &encoder;
    cfg.extractor = replacement_path ? &extractor : nullptr;

    ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
    pctx.extractor = replacement_path ? &extractor : nullptr;
    Signal s = MakeSignal (2000ULL);
    OperationContext ctx (s, pctx, cfg, store.get ());

    if (replacement_path)
      {
        operations::ExtractionRequest req;
        req.summary_id = "associative_cue_1";
        req.summary_text = evidence_text;
        req.source_texts = { evidence_text };
        req.current_labels = provisional_labels;
        req.cluster_size = 5;
        req.created_at = 2000ULL;
        ctx.SetExtractionRequests ({ req });
      }
    else
      {
        operations::ExtractionResult extraction;
        extraction.summary_id = "associative_cue_1";
        for (const auto &label : refined_labels)
          {
            extraction.labels.push_back ({ label, 1.0 });
          }
        pctx.pending_extraction_results.push_back (std::move (extraction));
      }

    operations::ProcessExtractionResults op;
    auto tx = store->Begin ();
    op.Execute (ctx, *tx);
    tx->Commit ();

    return MeasureLabelEdges (*store, truth);
  };

  const auto append_only = run_case (false);
  const auto replacement = run_case (true);

  std::cout << "\n[blob consolidation label pruning ablation]\n"
            << "path,edges,tp,fp,fn,precision,recall\n"
            << "append_only_before_blob_refinement,"
            << append_only.edge_count << "," << append_only.true_positive
            << "," << append_only.false_positive << ","
            << append_only.false_negative << "," << append_only.Precision ()
            << "," << append_only.Recall () << "\n"
            << "replacement_after_blob_refinement,"
            << replacement.edge_count << "," << replacement.true_positive
            << "," << replacement.false_positive << ","
            << replacement.false_negative << "," << replacement.Precision ()
            << "," << replacement.Recall () << "\n";

  REQUIRE (append_only.edge_count == 6);
  REQUIRE (append_only.true_positive == 3);
  REQUIRE (append_only.false_positive == 3);
  REQUIRE (append_only.false_negative == 0);
  REQUIRE (append_only.Precision () == Catch::Approx (0.5));
  REQUIRE (append_only.Recall () == Catch::Approx (1.0));

  REQUIRE (replacement.edge_count == 3);
  REQUIRE (replacement.true_positive == 3);
  REQUIRE (replacement.false_positive == 0);
  REQUIRE (replacement.false_negative == 0);
  REQUIRE (replacement.Precision () == Catch::Approx (1.0));
  REQUIRE (replacement.Recall () == Catch::Approx (1.0));
}

TEST_CASE ("STM relabel replacement keeps a small durable current-label floor",
           "[operations][extraction][labels]")
{
  const std::set<std::string> truth = { "maria", "dog", "car" };
  const std::vector<std::string> provisional_labels
      = { "Maria", "Dog", "Car", "User", "idea" };
  const std::vector<std::string> under_admitted_refined_labels = { "Maria" };
  const std::string evidence_text = "Maria put the dog in the car.";
  FixedExtractor extractor (under_admitted_refined_labels);

  const auto replacement = RunBlobConsolidationLabelCase (
      &extractor, provisional_labels, under_admitted_refined_labels, truth,
      evidence_text, true);

  REQUIRE (replacement.edge_count == 3);
  REQUIRE (replacement.true_positive == 3);
  REQUIRE (replacement.false_positive == 0);
  REQUIRE (replacement.false_negative == 0);
  REQUIRE (replacement.Precision () == Catch::Approx (1.0));
  REQUIRE (replacement.Recall () == Catch::Approx (1.0));
}

TEST_CASE ("STM relabel current-label floor does not treat text blobs as multimodal grounding",
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
  FixedExtractor extractor ({ "Maria" });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.53;
  cfg.sensitivity = 0.79;
  cfg.stability = 0.99;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text = "Maria was there.";
  req.source_texts = { "Maria was there." };
  req.source_blobs.push_back (operations::ExtractionSourceBlob {
      { 'M', 'a', 'r', 'i', 'a' }, "text", "text/plain" });
  req.current_labels = { "Maria", "Alfred" };
  req.cluster_size = 1;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto label_rows = store->Execute (
      "SELECT source_id FROM memories WHERE kind = 'LABEL' "
      "ORDER BY source_id",
      {});
  REQUIRE (label_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (label_rows[0].at ("source_id"))
           == "maria");
}

TEST_CASE ("STM relabel source-span floor admits grounded labels when extractor under-admits",
           "[operations][extraction][labels]")
{
  cortext::testing::ScopedEnvVar enable_audit ("CORTEXT_STM_LTM_AUDIT", "1");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "summary-1",
                                  "ASSOCIATION", 1.0, 1000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 30LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 30LL, 30LL,
                                  "source-memory-1", "LONG_TERM", 1.0,
                                  1000LL);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (20, 30, 'derived_from', 1.0)",
      {});
  store->Execute (
      "CREATE TABLE stm_ltm_relabel_audit ("
      "summary_id TEXT PRIMARY KEY, created_at INTEGER, cluster_size INTEGER, "
      "source_memory_count INTEGER, source_text_count INTEGER, "
      "source_blob_count INTEGER, source_memory_ids TEXT, "
      "stm_graph_count INTEGER, stm_item_count INTEGER, "
      "stm_label_edge_count INTEGER, current_label_count INTEGER, "
      "current_labels TEXT, refined_label_count INTEGER DEFAULT 0, "
      "refined_labels_in_selected_evidence INTEGER DEFAULT 0, "
      "extraction_label_candidate_count INTEGER DEFAULT 0, "
      "source_span_candidate_count INTEGER DEFAULT 0, "
      "labels_inserted_from_current_floor INTEGER DEFAULT 0, "
      "labels_inserted_from_source_span_floor INTEGER DEFAULT 0, "
      "relation_edges_created INTEGER DEFAULT 0)",
      {});
  store->Execute (
      "INSERT INTO stm_ltm_relabel_audit "
      "(summary_id, created_at, cluster_size, source_memory_count, "
      "source_text_count, source_blob_count, source_memory_ids, "
      "stm_graph_count, stm_item_count, stm_label_edge_count, "
      "current_label_count, current_labels) "
      "VALUES ('summary-1', 1000, 1, 1, 1, 0, '20', 1, 1, 0, 0, '')",
      {});

  FixedEncoder encoder (unit_embedding);
  FixedExtractor extractor ({});

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.12;
  cfg.sensitivity = 0.83;
  cfg.stability = 0.98;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text = "Maria met Bailey at River Park. delivery was 7 days.";
  req.source_texts = { "Maria met Bailey at River Park. delivery was 7 days." };
  req.cluster_size = 1;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto label_rows = store->Execute (
      "SELECT source_id FROM memories WHERE kind = 'LABEL' "
      "ORDER BY source_id",
      {});
  std::set<std::string> labels;
  for (const auto &row : label_rows)
    {
      labels.insert (std::any_cast<std::string> (row.at ("source_id")));
  }
  CAPTURE (labels);
  REQUIRE (label_rows.size () == 4);
  REQUIRE (labels.count ("maria") == 1);
  REQUIRE (labels.count ("bailey") == 1);
  REQUIRE (labels.count ("river park") == 1);
  REQUIRE (labels.count ("maria met bailey at river park") == 1);
  REQUIRE (labels.count ("7 days") == 0);
  REQUIRE (labels.count ("delivery") == 0);

  auto audit_rows = store->Execute (
      "SELECT extraction_label_candidate_count, source_span_candidate_count, "
      "labels_inserted_from_current_floor, "
      "labels_inserted_from_source_span_floor, "
      "label_cooccurrence_edges_created, relation_edges_created, "
      "refined_label_count, "
      "refined_labels_in_selected_evidence, "
      "durable_ltm_nodes_with_source, durable_ltm_nodes_missing_source, "
      "durable_ltm_source_link_pairs "
      "FROM stm_ltm_relabel_audit WHERE summary_id = 'summary-1'",
      {});
  REQUIRE (audit_rows.size () == 1);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "extraction_label_candidate_count")
           == 0);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "source_span_candidate_count")
           == 4);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "labels_inserted_from_current_floor")
           == 0);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "labels_inserted_from_source_span_floor")
           == 4);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "label_cooccurrence_edges_created")
           == 6);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "relation_edges_created")
           == 6);
  REQUIRE (cortext::testing::GetInt64 (audit_rows[0], "refined_label_count")
           == 4);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "refined_labels_in_selected_evidence")
           == 4);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "durable_ltm_nodes_with_source")
           == 4);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "durable_ltm_nodes_missing_source")
           == 0);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "durable_ltm_source_link_pairs")
           == 4);

  auto cooccurrence_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM associations WHERE edge_type = 'co_occurs'",
      {});
  REQUIRE (cortext::testing::GetInt64 (cooccurrence_rows[0], "c") == 6);
}

TEST_CASE ("STM relabel source-span complement enriches labels after extractor meets floor",
           "[operations][extraction][labels]")
{
  cortext::testing::ScopedEnvVar enable_audit ("CORTEXT_STM_LTM_AUDIT", "1");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "summary-1",
                                  "ASSOCIATION", 1.0, 1000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 30LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 30LL, 30LL,
                                  "source-memory-1", "LONG_TERM", 1.0,
                                  1000LL);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (20, 30, 'derived_from', 1.0)",
      {});
  store->Execute (
      "CREATE TABLE stm_ltm_relabel_audit ("
      "summary_id TEXT PRIMARY KEY, created_at INTEGER, cluster_size INTEGER, "
      "source_memory_count INTEGER, source_text_count INTEGER, "
      "source_blob_count INTEGER, source_memory_ids TEXT, "
      "stm_graph_count INTEGER, stm_item_count INTEGER, "
      "stm_label_edge_count INTEGER, current_label_count INTEGER, "
      "current_labels TEXT, refined_label_count INTEGER DEFAULT 0, "
      "source_span_candidate_count INTEGER DEFAULT 0, "
      "labels_inserted_from_source_span_floor INTEGER DEFAULT 0)",
      {});
  store->Execute (
      "INSERT INTO stm_ltm_relabel_audit "
      "(summary_id, created_at, cluster_size, source_memory_count, "
      "source_text_count, source_blob_count, source_memory_ids, "
      "stm_graph_count, stm_item_count, stm_label_edge_count, "
      "current_label_count, current_labels) "
      "VALUES ('summary-1', 1000, 1, 1, 1, 0, '20', 1, 1, 0, 0, '')",
      {});

  FixedEncoder encoder (unit_embedding);
  FixedExtractor extractor ({ "Maria", "Bailey", "River Park" });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.48;
  cfg.sensitivity = 0.76;
  cfg.stability = 0.88;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text
      = "Maria met Bailey at River Park. Blue Cafe was near Red Bridge.";
  req.source_texts = { req.summary_text };
  req.cluster_size = 1;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto label_rows = store->Execute (
      "SELECT source_id FROM memories WHERE kind = 'LABEL' "
      "ORDER BY source_id",
      {});
  std::set<std::string> labels;
  std::ostringstream labels_debug;
  for (const auto &row : label_rows)
    {
      const auto value = std::any_cast<std::string> (row.at ("source_id"));
      labels_debug << "label='" << value << "'\n";
      labels.insert (value);
    }
  INFO (labels_debug.str ());
  REQUIRE (labels.size () == 5);
  REQUIRE (labels.count ("maria") == 1);
  REQUIRE (labels.count ("bailey") == 1);
  REQUIRE (labels.count ("river park") == 1);
  REQUIRE (labels.count ("blue cafe") == 1);
  REQUIRE (labels.count ("red bridge") == 1);

  auto audit_rows = store->Execute (
      "SELECT source_span_candidate_count, "
      "labels_inserted_from_source_span_floor, refined_label_count "
      "FROM stm_ltm_relabel_audit WHERE summary_id = 'summary-1'",
      {});
  REQUIRE (audit_rows.size () == 1);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "source_span_candidate_count")
           == 5);
  REQUIRE (cortext::testing::GetInt64 (
               audit_rows[0], "labels_inserted_from_source_span_floor")
           == 2);
  REQUIRE (cortext::testing::GetInt64 (audit_rows[0], "refined_label_count")
           == 5);
}

TEST_CASE ("STM relabel floor prioritizes source-span anchors over current labels",
           "[operations][extraction][labels]")
{
  cortext::testing::ScopedEnvVar enable_audit ("CORTEXT_STM_LTM_AUDIT", "1");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "summary-1",
                                  "ASSOCIATION", 1.0, 1000LL);

  FixedEncoder encoder (unit_embedding);
  FixedExtractor extractor ({});

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.81;
  cfg.sensitivity = 0.87;
  cfg.stability = 0.46;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text
      = "The cart situation near Taco Bell happened after school project.";
  req.source_texts = { req.summary_text };
  req.current_labels = { "cart situation", "school project" };
  req.cluster_size = 1;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto label_rows = store->Execute (
      "SELECT source_id FROM memories WHERE kind = 'LABEL' "
      "ORDER BY source_id",
      {});
  std::set<std::string> labels;
  for (const auto &row : label_rows)
    {
      labels.insert (std::any_cast<std::string> (row.at ("source_id")));
    }
  REQUIRE (labels.size () == 2);
  REQUIRE (labels.count ("taco bell") == 1);
  REQUIRE (labels.count ("cart situation") == 0);
  REQUIRE (labels.count ("school project") == 0);
}

TEST_CASE ("STM relabel promotes accepted event labels into source-backed facts",
           "[operations][extraction][labels][facts]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "summary-1",
                                  "ASSOCIATION", 1.0, 1000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 30LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 30LL, 30LL, "source-memory-1",
                                  "LONG_TERM", 1.0, 1000LL);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (20, 30, 'derived_from', 1.0)",
      {});

  FixedEncoder encoder (unit_embedding);
  FixedExtractor extractor ({});

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.53;
  cfg.sensitivity = 0.79;
  cfg.stability = 0.99;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text = "Amelia gave money at Taco Bell.";
  req.source_texts = { req.summary_text };
  req.cluster_size = 1;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto fact_rows = store->Execute (
      "SELECT canonical_subject, canonical_predicate, canonical_object "
      "FROM fact_assertions",
      {});
  REQUIRE (fact_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (
               fact_rows[0].at ("canonical_subject"))
           == "amelia");
  REQUIRE (std::any_cast<std::string> (
               fact_rows[0].at ("canonical_predicate"))
           == "gave");
  REQUIRE (std::any_cast<std::string> (
               fact_rows[0].at ("canonical_object"))
           == "money");

  auto evidence_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM fact_evidence",
      {});
  REQUIRE (cortext::testing::GetInt64 (evidence_rows[0], "c") == 2);
}

TEST_CASE ("Fact writes can be disabled natively during relabel processing",
           "[operations][extraction][labels][facts]")
{
  cortext::testing::ScopedEnvVar disable_facts ("CORTEXT_DISABLE_FACTS", "1");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "summary-1",
                                  "ASSOCIATION", 1.0, 1000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 30LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 30LL, 30LL, "source-memory-1",
                                  "LONG_TERM", 1.0, 1000LL);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (20, 30, 'derived_from', 1.0)",
      {});

  FixedEncoder encoder (unit_embedding);
  FixedExtractor extractor ({});

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.53;
  cfg.sensitivity = 0.79;
  cfg.stability = 0.99;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text = "Amelia gave money at Taco Bell.";
  req.source_texts = { req.summary_text };
  req.cluster_size = 1;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto fact_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM fact_assertions",
      {});
  REQUIRE (cortext::testing::GetInt64 (fact_rows[0], "c") == 0);

  auto evidence_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM fact_evidence",
      {});
  REQUIRE (cortext::testing::GetInt64 (evidence_rows[0], "c") == 0);
}

TEST_CASE ("STM relabel source-span candidates include contextual event anchors",
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
  FixedExtractor extractor ({});

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.52;
  cfg.sensitivity = 0.72;
  cfg.stability = 1.0;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text
      = "The cart situation near Taco Bell got weird after school.";
  req.source_texts = { req.summary_text };
  req.cluster_size = 1;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto label_rows = store->Execute (
      "SELECT source_id FROM memories WHERE kind = 'LABEL' "
      "ORDER BY source_id",
      {});
  std::set<std::string> labels;
  for (const auto &row : label_rows)
    {
      labels.insert (std::any_cast<std::string> (row.at ("source_id")));
    }
  REQUIRE (labels.count ("cart situation near taco bell") == 1);
  REQUIRE (labels.count ("taco bell") == 1);
  REQUIRE (labels.count ("cart") == 0);
}

TEST_CASE ("STM relabel source-span candidates prioritize event action anchors",
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
  FixedExtractor extractor ({});

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.81;
  cfg.sensitivity = 0.87;
  cfg.stability = 0.46;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text
      = "Amelia just gave me money at Taco Bell after school.";
  req.source_texts = { req.summary_text };
  req.cluster_size = 1;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto label_rows = store->Execute (
      "SELECT source_id FROM memories WHERE kind = 'LABEL' "
      "ORDER BY source_id",
      {});
  std::set<std::string> labels;
  for (const auto &row : label_rows)
    {
      labels.insert (std::any_cast<std::string> (row.at ("source_id")));
    }
  REQUIRE (labels.count ("amelia gave money") == 1);
  REQUIRE (labels.count ("gave money") == 1);

  auto fact_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM fact_assertions "
      "WHERE canonical_subject = 'amelia' "
      "  AND canonical_predicate = 'gave' "
      "  AND canonical_object = 'money'",
      {});
  REQUIRE_FALSE (fact_rows.empty ());
  REQUIRE (cortext::testing::GetInt64 (fact_rows[0], "c") == 1);
}

TEST_CASE ("STM relabel source-span candidates skip generic commercial fragments",
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
  FixedExtractor extractor ({ "Bathroom Vanity Light" });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.48;
  cfg.sensitivity = 0.63;
  cfg.stability = 0.98;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text
      = "Big cart had off brand bathroom vanity light near Red Bridge.";
  req.source_texts = { req.summary_text };
  req.cluster_size = 1;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto label_rows = store->Execute (
      "SELECT source_id FROM memories WHERE kind = 'LABEL' "
      "ORDER BY source_id",
      {});
  std::set<std::string> labels;
  for (const auto &row : label_rows)
    {
      labels.insert (std::any_cast<std::string> (row.at ("source_id")));
    }
  REQUIRE (labels.count ("bathroom vanity light") == 1);
  REQUIRE (labels.count ("red bridge") == 1);
  REQUIRE (labels.count ("big cart") == 0);
  REQUIRE (labels.count ("off brand") == 0);
}

TEST_CASE ("STM relabel skips sentence-start discourse labels",
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
  FixedExtractor extractor (
      { "Good", "Sorry", "All", "Zz", "Cart", "Light", "Might have",
        "Might have to come back", "turning around", "Taco Bell",
        "Black Vanity Light", "cart situation" });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.27;
  cfg.sensitivity = 0.55;
  cfg.stability = 0.95;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text
      = "Good. Sorry about all that. Might have to come back after turning "
        "around. The cart situation was near Taco Bell. The black vanity "
        "light stayed in the room.";
  req.source_texts = { req.summary_text };
  req.cluster_size = 1;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto label_rows = store->Execute (
      "SELECT source_id FROM memories WHERE kind = 'LABEL' "
      "ORDER BY source_id",
      {});
  std::set<std::string> labels;
  for (const auto &row : label_rows)
    {
      labels.insert (std::any_cast<std::string> (row.at ("source_id")));
    }
  REQUIRE (labels.count ("taco bell") == 1);
  REQUIRE (labels.count ("black vanity light") == 1);
  REQUIRE (labels.count ("cart situation") == 1);
  REQUIRE (labels.count ("good") == 0);
  REQUIRE (labels.count ("sorry") == 0);
  REQUIRE (labels.count ("all") == 0);
  REQUIRE (labels.count ("zz") == 0);
  REQUIRE (labels.count ("cart") == 0);
  REQUIRE (labels.count ("light") == 0);
  REQUIRE (labels.count ("might have") == 0);
  REQUIRE (labels.count ("might have to come back") == 0);
  REQUIRE (labels.count ("turning around") == 0);
}

TEST_CASE ("STM relabel admits multi-token labels when evidence contains all tokens",
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
  FixedExtractor extractor ({ "car crash" });

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 0.94;
  cfg.stability = 0.0;
  cfg.encoder = &encoder;
  cfg.extractor = &extractor;

  ProcessorContext pctx;
  cortext::testing::SeedMatureLabelContrastBank (pctx, kEmbeddingDim);
  pctx.extractor = &extractor;
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, pctx, cfg, store.get ());

  operations::ExtractionRequest req;
  req.summary_id = "summary-1";
  req.summary_text = "Maria saw the damaged car after the crash.";
  req.source_texts = { "Maria saw the damaged car after the crash." };
  req.cluster_size = 1;
  req.created_at = 2000ULL;
  ctx.SetExtractionRequests ({ req });

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto label_rows = store->Execute (
      "SELECT source_id FROM memories WHERE kind = 'LABEL' "
      "ORDER BY source_id",
      {});
  REQUIRE (label_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (label_rows[0].at ("source_id"))
           == "car crash");
}

TEST_CASE ("Gemma4 live blob-aware consolidation label ablation",
           "[.][gemma4-live][blob-consolidation]")
{
  const std::string model_path = FindModelPath (
      "models/gemma4-e2b-litert/gemma-4-E2B-it.litertlm");
  if (!std::filesystem::exists (model_path))
    {
      SKIP ("Gemma 4 model not found at " << model_path);
    }

  cortext::GemmaExtractor extractor (model_path);
  if (!extractor.IsAvailable ())
    {
      SKIP ("Gemma 4 extractor unavailable for " << model_path);
    }

  const std::set<std::string> truth = { "maria", "dog", "car crash" };
  const std::vector<std::string> provisional_labels
      = { "Maria", "Dog", "Steve", "Office", "Crash" };
  const std::string evidence_text
      = "Maria trained the dog after seeing a car crash.";

  const auto live_result = extractor.RefineLabelsFromText (
      evidence_text, provisional_labels, BuildExtractionSchema ());
  std::vector<std::string> live_labels;
  live_labels.reserve (live_result.labels.size ());
  for (const auto &label : live_result.labels)
    {
      live_labels.push_back (label.label);
    }

  const auto append_only = RunBlobConsolidationLabelCase (
      nullptr, provisional_labels, live_labels, truth, evidence_text, false);
  FixedExtractor live_result_extractor (live_labels);
  const auto gemma4_replacement = RunBlobConsolidationLabelCase (
      &live_result_extractor, provisional_labels, live_labels, truth,
      evidence_text, true);

  std::cout << "\n[gemma4 live blob consolidation label pruning ablation]\n"
            << "model," << model_path << "\n";
  std::cout << "gemma4_labels,";
  for (size_t i = 0; i < live_labels.size (); ++i)
    {
      if (i > 0)
        {
          std::cout << "|";
        }
      std::cout << live_labels[i];
    }
  std::cout << "\n"
            << "path,edges,tp,fp,fn,precision,recall\n"
            << "append_only_before_blob_refinement,"
            << append_only.edge_count << "," << append_only.true_positive
            << "," << append_only.false_positive << ","
            << append_only.false_negative << "," << append_only.Precision ()
            << "," << append_only.Recall () << "\n"
            << "gemma4_replacement_after_blob_refinement,"
            << gemma4_replacement.edge_count << ","
            << gemma4_replacement.true_positive << ","
            << gemma4_replacement.false_positive << ","
            << gemma4_replacement.false_negative << ","
            << gemma4_replacement.Precision () << ","
            << gemma4_replacement.Recall () << "\n";

  REQUIRE (append_only.false_positive > 0);
  CHECK (gemma4_replacement.false_positive < append_only.false_positive);
  CHECK (gemma4_replacement.edge_count <= append_only.edge_count);
}

TEST_CASE ("Gemma4 live label refinement covers text audio and image",
           "[.][gemma4-live][multimodal][blob-consolidation]")
{
  const std::string model_path = FindModelPath (
      "models/gemma4-e2b-litert/gemma-4-E2B-it.litertlm");
  if (!std::filesystem::exists (model_path))
    {
      SKIP ("Gemma 4 model not found at " << model_path);
    }

  cortext::GemmaExtractor text_extractor (model_path);
  cortext::GemmaExtractor image_extractor (model_path);
  cortext::GemmaExtractor audio_extractor (model_path);
  if (!text_extractor.IsAvailable () || !image_extractor.IsAvailable ()
      || !audio_extractor.IsAvailable ())
    {
      SKIP ("Gemma 4 extractor unavailable for " << model_path);
    }

  const auto schema = BuildExtractionSchema ();
  std::cout << "\n[gemma4 live multimodal label refinement]\n"
            << "model," << model_path << "\n"
            << "modality,labels\n";

  const auto text_current_labels
      = std::vector<std::string> { "Maria", "Dog", "car crash", "Office" };
  const auto text_result = text_extractor.RefineLabelsFromText (
      "Maria saw the dog after the car crash.", text_current_labels, schema);
  const auto text_labels = ExtractLabelStrings (text_result);
  PrintLabels ("text", text_labels);

  const auto image_current_labels
      = std::vector<std::string> { "Dog", "Office", "generic" };
  const std::string image_path = FindModelPath (
      "build/real_multimodal_episode_assets/source/dog.jpg");
  if (!std::filesystem::exists (image_path))
    {
      SKIP ("Image fixture not found at " << image_path);
    }
  const auto image_bytes = ReadBytesFile (image_path);
  const auto image_result = image_extractor.RefineLabelsFromImage (
      image_bytes, image_current_labels, schema);
  const auto image_labels = ExtractLabelStrings (image_result);
  PrintLabels ("image", image_labels);

  const auto audio_current_labels
      = std::vector<std::string> { "crash", "Office", "generic" };
  const std::string audio_path = FindModelPath (
      "build/real_multimodal_episode_assets/raw/crash_16k_mono.f32");
  if (!std::filesystem::exists (audio_path))
    {
      SKIP ("Audio fixture not found at " << audio_path);
    }
  const auto pcm = ReadFloat32File (audio_path);
  const auto audio_result = audio_extractor.RefineLabelsFromAudio (
      pcm.data (), pcm.size (), audio_current_labels, schema);
  const auto audio_labels = ExtractLabelStrings (audio_result);
  PrintLabels ("audio", audio_labels);

  CHECK (HasLabelContaining (text_labels, "Maria"));
  CHECK (HasLabelContaining (audio_labels, "crash"));
  CHECK (HasLabelContaining (image_labels, "dog"));
}

TEST_CASE ("Cold start declines optional label admission paths",
           "[operations][extraction][labels][coldstart]")
{
  // With an immature contrast bank (fewer than kContrastMinBankSize
  // labels) the optional admission paths decline: relation-endpoint
  // creation and floor fill wait until candidates can be vetted, while
  // the primary extractor path still admits so the bank can seed.
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  std::vector<float> unit_embedding (kEmbeddingDim, 0.0f);
  unit_embedding[0] = 1.0f;

  cortext::testing::SeedEmbeddingV2 (*store, 10LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 20LL, 10LL, "summary-1",
                                  "ASSOCIATION", 1.0, 1000LL);
  cortext::testing::SeedEmbeddingV2 (*store, 30LL, unit_embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (*store, 30LL, 30LL,
                                  "Maria met the dog at River Park.",
                                  "LONG_TERM", 1.0, 1000LL);
  store->Execute (
      "INSERT INTO associations(source_memory_id, target_memory_id, edge_type, weight) "
      "VALUES (20, 30, 'derived_from', 1.0)",
      {});

  FixedEncoder encoder (unit_embedding);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.75;
  cfg.sensitivity = 0.95;
  cfg.stability = 0.85;
  cfg.encoder = &encoder;

  ProcessorContext cold_pctx; // intentionally empty contrast bank
  Signal s = MakeSignal (2000ULL);
  OperationContext ctx (s, cold_pctx, cfg, store.get ());

  operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  extraction.labels.push_back ({ "Maria", 0.0 });
  extraction.relations.push_back ({ "Maria", "reinforces", "Dog", 0.75 });
  cold_pctx.pending_extraction_results.push_back (std::move (extraction));

  operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  // The extractor label seeds the bank.
  auto label_rows = store->Execute (
      "SELECT source_id FROM memories WHERE kind = 'LABEL' ORDER BY source_id",
      {});
  REQUIRE (label_rows.size () == 1);
  REQUIRE (std::any_cast<std::string> (label_rows[0].at ("source_id"))
           == "maria");

  // The relation endpoint is NOT manufactured into a label at cold start,
  // so the relation edge has no target and is skipped.
  auto relation_rows = store->Execute (
      "SELECT a.edge_type FROM associations a "
      "JOIN memories s ON s.memory_id = a.source_memory_id "
      "JOIN memories o ON o.memory_id = a.target_memory_id "
      "WHERE s.source_id = 'maria' AND o.source_id = 'dog'",
      {});
  REQUIRE (relation_rows.empty ());
}
