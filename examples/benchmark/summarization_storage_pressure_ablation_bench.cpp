#include "../../src/operations/eviction_ablation.hpp"
#include "../../src/deep_llm/deep_llm_factory.hpp"
#include "include/benchmark_text_encoder.hpp"

#include <Eigen/Dense>
#include <cortext/consolidation_mode.hpp>
#include <cortext/operations/consolidation_summarize.hpp>
#include <cortext/operations/process_extraction_results.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <any>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{

struct LiveModels
{
  explicit LiveModels (const std::string &models_dir)
      : encoder (models_dir),
        deep_llm (cortext::internal::CreateDeepLlmSelection (
            std::filesystem::path (models_dir)))
  {
  }

  cortext::benchmark::EmbeddingGemmaBenchEncoder encoder;
  cortext::internal::DeepLlmSelection deep_llm;
};

cortext::Signal
MakeSignal (uint64_t ts)
{
  cortext::Signal s;
  s.embedding = Eigen::VectorXf::Ones (256);
  s.timestamp = ts;
  s.source_id = "bench/consolidation";
  s.consolidation_mode = cortext::ConsolidationMode::Both;
  return s;
}

void
SeedEmbedding (cortext::Store *store, long long id, const Eigen::VectorXf &emb)
{
  std::vector<float> vec (emb.data (), emb.data () + emb.size ());
  store->Execute (
      "INSERT OR REPLACE INTO embeddings(embedding_id, embedding, created_at) "
      "VALUES(?, ?, ?)",
      { id, vec, 1000LL });
  store->Execute (
      "INSERT OR REPLACE INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, end_ts, n_signals, "
      "modality, s_max, s_avg, strength, redundancy, created_at) "
      "VALUES(?, ?, 'bench', 'LONG_TERM', ?, ?, 1, 'text', 0.5, 0.5, "
      "1.0, 0.0, ?)",
      { id, id, 1000LL + id, 1000LL + id, 1000LL + id });
}

void
SeedMemoryText (cortext::Store *store, long long memory_id,
                const std::string &text)
{
  std::vector<unsigned char> data_blob (text.begin (), text.end ());
  auto blob_rows
      = store->Execute ("SELECT objstore_put(?1) AS id", { data_blob });
  if (blob_rows.empty () || blob_rows[0].count ("id") == 0)
    {
      throw std::runtime_error ("objstore_put failed");
    }
  std::vector<unsigned char> blob_id;
  const auto &raw_blob = blob_rows[0].at ("id");
  if (raw_blob.type () == typeid (std::vector<unsigned char>))
    {
      blob_id = std::any_cast<std::vector<unsigned char>> (raw_blob);
    }
  else if (raw_blob.type () == typeid (std::vector<char>))
    {
      const auto chars = std::any_cast<std::vector<char>> (raw_blob);
      blob_id.assign (chars.begin (), chars.end ());
    }
  if (blob_id.empty ())
    {
      throw std::runtime_error ("objstore_put returned empty id");
    }
  store->Execute ("UPDATE memories SET blob_id = ? WHERE memory_id = ?",
                  { blob_id, memory_id });
}

long long
CountSummaries (cortext::Store *store)
{
  auto rows = store->Execute (
      // Deep summaries are LONG_TERM memories (the ASSOCIATION form was
      // migrated away; see store/schema.cpp).
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE kind = 'LONG_TERM' AND source_id LIKE 'summary_%'",
      {});
  if (rows.empty ())
    {
      return 0;
    }
  return std::any_cast<long long> (rows[0].at ("c"));
}

long long
CountCues (cortext::Store *store)
{
  auto rows = store->Execute (
      "SELECT COUNT(*) AS c FROM memories "
      "WHERE kind = 'ASSOCIATION' AND source_id LIKE 'associative_cue_%'",
      {});
  if (rows.empty ())
    {
      return 0;
    }
  return std::any_cast<long long> (rows[0].at ("c"));
}

long long
CountLabels (cortext::Store *store)
{
  auto rows = store->Execute (
      "SELECT COUNT(*) AS c FROM memories WHERE kind = 'LABEL'", {});
  if (rows.empty ())
    {
      return 0;
    }
  return std::any_cast<long long> (rows[0].at ("c"));
}

long long
CountLabelEdges (cortext::Store *store)
{
  auto rows = store->Execute (
      "SELECT COUNT(*) AS c FROM associations WHERE edge_type = 'has_label'",
      {});
  if (rows.empty ())
    {
      return 0;
    }
  return std::any_cast<long long> (rows[0].at ("c"));
}

std::string
ListLabels (cortext::Store *store)
{
  auto rows = store->Execute (
      "SELECT label FROM memories WHERE kind = 'LABEL' ORDER BY label", {});
  std::ostringstream out;
  for (std::size_t i = 0; i < rows.size (); ++i)
    {
      if (i > 0)
        {
          out << ",";
        }
      out << std::any_cast<std::string> (rows[i].at ("label"));
    }
  return out.str ();
}

struct Scenario
{
  std::string name;
  long long used_bytes = 0;
  bool protected_evidence = false;
  long long expected_summaries = 0;
  long long expected_cues = 0;
  long long min_expected_labels = 1;
  long long min_expected_label_edges = 1;
  long long max_expected_labels = -1;
  long long max_expected_label_edges = -1;
  std::vector<std::string> source_texts;
};

struct Result
{
  std::string name;
  long long summaries = 0;
  long long cues = 0;
  long long labels = 0;
  long long label_edges = 0;
  std::size_t extraction_requests = 0;
  std::string label_values;
  bool ok = false;
};

Result
RunScenario (const Scenario &scenario, LiveModels &models)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));

  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.0;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  cfg.encoder = &models.encoder;

  auto init_ops = std::make_unique<cortext::DynamicOperationSet> ();
  cortext::SignalProcessor init_processor (cfg, store, std::move (init_ops));
  init_processor.Process (MakeSignal (1));
  init_processor.Flush ();

  const std::vector<std::string> source_texts
      = scenario.source_texts.empty ()
            ? std::vector<std::string> {
                "User: Gabriel lives in Chicago and is testing Cortext consolidation labels.",
                "Assistant: The Cortext memory graph should extract labels from raw memories.",
                "User: Storage pressure should not decide whether labels are extracted.",
              }
            : scenario.source_texts;
  for (std::size_t i = 0; i < source_texts.size (); ++i)
    {
      const long long id = static_cast<long long> (i + 1);
      SeedEmbedding (store.get (), id,
                     models.encoder.EncodeTextEigen (source_texts[i]));
      SeedMemoryText (store.get (), id, source_texts[i]);
    }
  store->Execute (
      "UPDATE memories SET redundancy = 1.0, strength = 0.0, "
      "emotional_intensity = 0.0, s_emotion_max = 0.0, s_arousal_avg = 0.0",
      {});

  if (scenario.protected_evidence)
    {
      store->Execute (
          "UPDATE memories SET strength = 0.08, emotional_intensity = 1.0",
          {});
      store->Execute (
          "INSERT INTO fact_assertions "
          "(fact_id, subject, predicate, object, canonical_subject, "
          "canonical_predicate, canonical_object, recorded_at_ts, confidence, "
          "summary_memory_id, created_at) "
          "VALUES (1, 'gabriel', 'uses', 'cortext', 'gabriel', 'uses', "
          "'cortext', 1000, 0.9, 0, 1000)",
          {});
      store->Execute (
          "INSERT INTO fact_evidence "
          "(fact_id, source_memory_id, evidence_type, support_weight) "
          "VALUES (1, 1, 'source', 1.0)",
          {});
      store->Execute (
          "INSERT INTO fact_evidence "
          "(fact_id, source_memory_id, evidence_type, support_weight) "
          "VALUES (1, 2, 'source', 1.0)",
          {});
      store->Execute (
          "INSERT INTO fact_evidence "
          "(fact_id, source_memory_id, evidence_type, support_weight) "
          "VALUES (1, 3, 'source', 1.0)",
          {});
    }

  cortext::ProcessorContext pctx;
  pctx.summarizer = models.deep_llm.summarizer.get ();
  pctx.extractor = models.deep_llm.extractor.get ();
  pctx.last_consolidation_ts = 100000;

  cortext::Signal s = MakeSignal (3000);
  cortext::OperationContext ctx (s, pctx, cfg, store.get ());
  ctx.SetConsolidationShouldStart (true);

  cortext::ClusterInfo cluster;
  cluster.cluster_id = 1;
  cluster.embedding_ids = { 1, 2, 3 };
  cluster.centroid = std::vector<float> (256, 0.5f);
  cluster.avg_score = 0.3;
  ctx.SetConsolidationClusters ({ cluster });

  cortext::operations::eviction::EvictionAblationOverride override;
  override.storage_gate_enabled = true;
  override.min_storage_bytes = 1000;
  override.used_storage_bytes = scenario.used_bytes;
  cortext::operations::eviction::ScopedEvictionAblationOverride pressure (
      override);

  cortext::operations::ConsolidationSummarize summarize;
  cortext::operations::ProcessExtractionResults process_extraction;
  auto tx = store->Begin ();
  summarize.Execute (ctx, *tx);
  const std::size_t extraction_requests = ctx.GetExtractionRequests ().size ();
  process_extraction.Execute (ctx, *tx);
  tx->Commit ();

  Result result;
  result.name = scenario.name;
  result.summaries = CountSummaries (store.get ());
  result.cues = CountCues (store.get ());
  result.labels = CountLabels (store.get ());
  result.label_edges = CountLabelEdges (store.get ());
  result.label_values = ListLabels (store.get ());
  result.extraction_requests = extraction_requests;
  result.ok = result.summaries == scenario.expected_summaries
              && result.cues == scenario.expected_cues
              && result.extraction_requests == 1
              && result.labels >= scenario.min_expected_labels
              && result.label_edges >= scenario.min_expected_label_edges
              && (scenario.max_expected_labels < 0
                  || result.labels <= scenario.max_expected_labels)
              && (scenario.max_expected_label_edges < 0
                  || result.label_edges <= scenario.max_expected_label_edges);
  return result;
}

} // namespace

int
main ()
{
  LiveModels models ("models");
  std::cout << "encoder_backend=" << models.encoder.backend_name () << "\n";
  std::cout << "deep_llm_backend=" << models.deep_llm.backend_name << "\n";
  std::cout << "extractor_model="
            << models.deep_llm.extractor_model_path.string () << "\n";
  std::cout << "summarizer_model="
            << models.deep_llm.summarizer_model_path.string () << "\n";

  const std::vector<Scenario> scenarios = {
    { "low_pressure", 1, false, 1, 1 },
    { "high_pressure_compressible", 10000, false, 1, 1 },
    { "high_pressure_protected_evidence", 10000, true, 1, 1 },
    { "filler_only", 1, false, 1, 1, 0, 0, 0, 0,
      {
        "User: ya know.",
        "Assistant: okay.",
        "User: ya, know.",
      } },
    { "banter_only", 1, false, 1, 1, 0, 0, 0, 0,
      {
        "User: thanks.",
        "Assistant: sure.",
        "User: okay thanks.",
      } },
    { "mixed_durable_filler", 1, false, 1, 1, 1, 1,
      -1, -1,
      {
        "User: ya know, Maya is 11 years old.",
        "Assistant: okay.",
        "User: Maya gave a pitch this weekend.",
      } },
  };

  bool ok = true;
  std::cout << "scenario extraction_requests summaries cues labels "
               "has_label_edges label_values ok\n";
  for (const auto &scenario : scenarios)
    {
      const Result result = RunScenario (scenario, models);
      ok = ok && result.ok;
      std::cout << result.name << " " << result.extraction_requests << " "
                << result.summaries << " " << result.cues << " "
                << result.labels << " " << result.label_edges << " "
                << (result.label_values.empty () ? "-" : result.label_values)
                << " "
                << (result.ok ? 1 : 0) << "\n";
    }

  std::cout << "consolidation_label_pressure_ablation_ok=" << (ok ? 1 : 0)
            << "\n";
  return ok ? 0 : 1;
}
