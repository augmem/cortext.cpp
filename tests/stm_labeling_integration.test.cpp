// Compiled only when models/AIST-87M-GGUF is present (see CMakeLists.txt).
// Verifies short-term-graph labeling on the real encoder in the runtime
// similarity space (256-d Matryoshka view). LTM labeling consumes these
// edges (AugmentLabelsFromShortTermGraph), so this is the upstream half of
// the label-quality pipeline.

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/models/aist_gguf_encoder.hpp>
#include <cortext/operations/short_term_memory_shadow.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

using namespace cortext;

namespace
{

std::filesystem::path
RepoRoot ()
{
  std::filesystem::path p = std::filesystem::current_path ();
  for (int i = 0; i < 6; ++i)
    {
      if (std::filesystem::exists (p / "CMakeLists.txt")
          && std::filesystem::exists (p / "models"))
        {
          return p;
        }
      if (!p.has_parent_path ())
        {
          break;
        }
      p = p.parent_path ();
    }
  return std::filesystem::current_path ().parent_path ();
}

AistGgufEncoder &
RealEncoder ()
{
  static AistGgufEncoder encoder = [] {
    auto resolved = ResolveAistGgufModelPath (RepoRoot () / "models");
    REQUIRE (resolved.has_value ());
    AistGgufConfig config;
    config.model_path = resolved->string ();
    config.context_length = 64;
    return AistGgufEncoder (config);
  }();
  REQUIRE (encoder.IsRuntimeAvailable ());
  return encoder;
}

/// Runtime similarity space: 256-d Matryoshka view, renormalized (mirrors
/// RetrievalEmbeddingView in src/cortext.cpp).
Eigen::VectorXf
RuntimeView (const std::string &text)
{
  std::vector<float> full;
  RealEncoder ().EncodeText (text, full);
  REQUIRE (full.size () >= 256);
  full.resize (256);
  Eigen::VectorXf v (256);
  for (int i = 0; i < 256; ++i)
    {
      v (i) = full[static_cast<size_t> (i)];
    }
  const float norm = v.norm ();
  REQUIRE (norm > 1e-9f);
  return v / norm;
}

void
SeedLabel (Store &store, ProcessorContext &p_ctx, long long memory_id,
           long long embedding_id, const std::string &label)
{
  const Eigen::VectorXf embedding = RuntimeView (label);
  cortext::testing::SeedEmbeddingV2 (store, embedding_id, embedding, 1000LL);
  cortext::testing::SeedMemoryV2 (store, memory_id, embedding_id, label,
                                  "LABEL", 1.0, 1000LL);
  store.Execute ("UPDATE memories SET label = ? WHERE memory_id = ?",
                 { label, memory_id });
  p_ctx.UpsertSummaryCache (memory_id, embedding_id, embedding, false, true);
}

Signal
MakeTextSignal (const std::string &text, const std::string &source_id,
                uint64_t ts)
{
  Signal s;
  s.source_id = source_id;
  s.timestamp = ts;
  s.embedding = RuntimeView (text);
  return s;
}

double
MaxEdgeWeight (const std::deque<ProcessorContext::ShadowLabelEdge> &edges)
{
  double max_weight = 0.0;
  for (const auto &edge : edges)
    {
      max_weight = std::max (max_weight, edge.weight);
    }
  return max_weight;
}

} // namespace

TEST_CASE ("STM labeling routes real text to the right labels and keeps "
           "banter below the consolidation gate",
           "[stm][labels][aist][integration]")
{
  cortext::testing::ScopedEnvVar enable ("CORTEXT_STM_SHADOW_ENABLE", "1");
  cortext::testing::ScopedEnvVar clear_disable ("CORTEXT_STM_SHADOW_DISABLE");

  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));
  cortext::testing::InitializeCoreSchema (*store);

  ProcessorContext p_ctx;
  SeedLabel (*store, p_ctx, 101LL, 201LL, "kitchen renovation");
  SeedLabel (*store, p_ctx, 102LL, 202LL, "kitchen appliances");
  SeedLabel (*store, p_ctx, 103LL, 203LL, "soccer practice");
  SeedLabel (*store, p_ctx, 104LL, 204LL, "soccer tournament");
  SeedLabel (*store, p_ctx, 105LL, 205LL, "medication schedule");
  SeedLabel (*store, p_ctx, 106LL, 206LL, "tax documents");

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  operations::UpdateShortTermMemoryShadow op;
  const double consolidation_gate
      = core::STMLabelConsolidationMinSimilarity (cfg.focus, cfg.sensitivity,
                                                  cfg.stability);

  // Related signal: should route to the kitchen labels, not sports/admin.
  {
    auto tx = store->Begin ();
    const Signal signal = MakeTextSignal (
        "We picked out new cabinets and countertops for the kitchen today",
        "stm-real/kitchen", 2000ULL);
    OperationContext ctx (signal, p_ctx, cfg, store.get ());
    op.Execute (ctx, *tx);
    tx->Commit ();
  }
  const auto &kitchen_edges
      = p_ctx.short_term_graphs["stm-real/kitchen"].label_edges;
  REQUIRE_FALSE (kitchen_edges.empty ());

  std::ostringstream edges_debug;
  edges_debug << "consolidation_gate=" << consolidation_gate << "\n";
  const ProcessorContext::ShadowLabelEdge *top_edge = nullptr;
  for (const auto &edge : kitchen_edges)
    {
      edges_debug << "kitchen edge label='" << edge.label
                  << "' weight=" << edge.weight << "\n";
      if (top_edge == nullptr || edge.weight > top_edge->weight)
        {
          top_edge = &edge;
        }
    }
  INFO (edges_debug.str ());

  REQUIRE (top_edge != nullptr);
  CHECK (top_edge->label.rfind ("kitchen", 0) == 0);

  // No unrelated label may outrank the on-topic ones.
  for (const auto &edge : kitchen_edges)
    {
      if (edge.label.rfind ("kitchen", 0) != 0)
        {
          CHECK (edge.weight < top_edge->weight);
        }
    }

  // Banter: edges may exist (admission floor is permissive by design), but
  // every edge must fall below the consolidation gate so none of it can
  // reach LTM labels through AugmentLabelsFromShortTermGraph.
  {
    auto tx = store->Begin ();
    const Signal signal = MakeTextSignal ("okay thanks sounds good",
                                          "stm-real/banter", 3000ULL);
    OperationContext ctx (signal, p_ctx, cfg, store.get ());
    op.Execute (ctx, *tx);
    tx->Commit ();
  }
  const auto &banter_edges
      = p_ctx.short_term_graphs["stm-real/banter"].label_edges;
  std::ostringstream banter_debug;
  for (const auto &edge : banter_edges)
    {
      banter_debug << "banter edge label='" << edge.label
                   << "' weight=" << edge.weight << "\n";
    }
  INFO (banter_debug.str ());
  CHECK (MaxEdgeWeight (banter_edges) < consolidation_gate);

  // The related signal's best label must clear the consolidation gate, or
  // STM labels can never reach LTM at all.
  CHECK (top_edge->weight >= consolidation_gate);
}
