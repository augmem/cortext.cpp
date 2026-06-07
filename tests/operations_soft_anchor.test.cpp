#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/operations/soft_anchor.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>

using namespace cortext;
using cortext::operations::UpdateSoftAnchor;
using cortext::store::AnyToLongLong;

namespace
{

Eigen::VectorXf
MakeEsVector (float semantic_value, float entity_value, float full_tail_value)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (1536);
  v[0] = semantic_value;
  v[768] = entity_value;
  v[1200] = full_tail_value;
  return v;
}

long long
CountRows (Store &store, const std::string &table)
{
  auto rows = store.Execute ("SELECT COUNT(*) AS cnt FROM " + table, {});
  REQUIRE (rows.size () == 1);
  auto count = AnyToLongLong (rows[0].at ("cnt"));
  REQUIRE (count.has_value ());
  return *count;
}

OperationContext
MakeSoftAnchorContext (Signal &signal, ProcessorContext &processor,
                       const SignalProcessor::Config &config, Store &store,
                       long long memory_id)
{
  OperationContext context (signal, processor, config, &store);
  context.SetStoredMemoryId (memory_id);
  context.SetRepresentativeEmbedding (signal.embedding);
  context.SetBoundaryScore (0.0);
  return context;
}

} // namespace

TEST_CASE ("Soft Anchor forms by default",
           "[operations][soft_anchor]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  REQUIRE (unique_store != nullptr);
  testing::InitializeCoreSchema (*unique_store);

  Signal signal;
  signal.embedding = MakeEsVector (1.0f, 1.0f, 0.2f);
  signal.timestamp = 1000;
  signal.source_id = "chat-a";
  signal.modality = "text";

  ProcessorContext processor;
  SignalProcessor::Config config;
  testing::RequireEncoder (config);
  config.focus = 0.5;
  config.sensitivity = 0.5;
  config.stability = 0.5;

  auto context = MakeSoftAnchorContext (signal, processor, config,
                                        *unique_store, 101);
  UpdateSoftAnchor op;
  auto tx = unique_store->Begin ();
  op.Execute (context, *tx);
  tx->Commit ();

  REQUIRE (processor.soft_anchor_enabled);
  REQUIRE (processor.soft_anchor_states.size () == 1);
  REQUIRE (processor.soft_anchor_last_create_count == 1);
  REQUIRE (processor.soft_anchor_last_links.size () == 1);
  REQUIRE (CountRows (*unique_store, "soft_anchors") == 1);
  REQUIRE (CountRows (*unique_store, "soft_anchor_links") == 1);
}

TEST_CASE ("Soft Anchor creates a provisional anchor from first ES signal",
           "[operations][soft_anchor]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  REQUIRE (unique_store != nullptr);
  testing::InitializeCoreSchema (*unique_store);

  Signal signal;
  signal.embedding = MakeEsVector (1.0f, 1.0f, 0.2f);
  signal.timestamp = 1000;
  signal.source_id = "chat-a";
  signal.modality = "text";

  ProcessorContext processor;
  SignalProcessor::Config config;
  testing::RequireEncoder (config);
  config.focus = 0.5;
  config.sensitivity = 0.5;
  config.stability = 0.5;

  auto context = MakeSoftAnchorContext (signal, processor, config,
                                        *unique_store, 101);
  UpdateSoftAnchor op;
  auto tx = unique_store->Begin ();
  op.Execute (context, *tx);
  tx->Commit ();

  REQUIRE (processor.soft_anchor_enabled);
  REQUIRE (processor.soft_anchor_states.size () == 1);
  REQUIRE (processor.soft_anchor_last_create_count == 1);
  REQUIRE (processor.soft_anchor_last_update_count == 0);
  REQUIRE (processor.soft_anchor_last_link_count == 1);
  REQUIRE (processor.soft_anchor_last_links.size () == 1);
  REQUIRE (processor.soft_anchor_last_links.front ().memory_id == 101);
  REQUIRE (processor.soft_anchor_last_links.front ().anchor_label
           == "tentative");
  REQUIRE (processor.soft_anchor_last_links.front ().evidence_kind == "new");
  REQUIRE (CountRows (*unique_store, "soft_anchors") == 1);
  REQUIRE (CountRows (*unique_store, "soft_anchor_links") == 1);
}

TEST_CASE ("Soft Anchor updates an existing anchor from a similar ES signal",
           "[operations][soft_anchor]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  REQUIRE (unique_store != nullptr);
  testing::InitializeCoreSchema (*unique_store);

  ProcessorContext processor;
  SignalProcessor::Config config;
  testing::RequireEncoder (config);
  config.focus = 0.5;
  config.sensitivity = 0.5;
  config.stability = 0.5;

  UpdateSoftAnchor op;
  Signal first;
  first.embedding = MakeEsVector (1.0f, 1.0f, 0.2f);
  first.timestamp = 1000;
  first.source_id = "chat-a";
  first.modality = "text";
  {
    auto context = MakeSoftAnchorContext (first, processor, config,
                                          *unique_store, 101);
    auto tx = unique_store->Begin ();
    op.Execute (context, *tx);
    tx->Commit ();
  }
  REQUIRE (processor.soft_anchor_states.size () == 1);
  const std::string anchor_id = processor.soft_anchor_states.front ().anchor_id;

  processor.signals_processed = 1;
  Signal second;
  second.embedding = MakeEsVector (0.98f, 0.99f, 0.2f);
  second.timestamp = 2000;
  second.source_id = "chat-a";
  second.modality = "text";
  {
    auto context = MakeSoftAnchorContext (second, processor, config,
                                          *unique_store, 102);
    auto tx = unique_store->Begin ();
    op.Execute (context, *tx);
    tx->Commit ();
  }

  REQUIRE (processor.soft_anchor_states.size () == 1);
  REQUIRE (processor.soft_anchor_states.front ().anchor_id == anchor_id);
  REQUIRE (processor.soft_anchor_last_create_count == 0);
  REQUIRE (processor.soft_anchor_last_update_count == 1);
  REQUIRE (processor.soft_anchor_last_link_count >= 1);
  REQUIRE (processor.soft_anchor_last_links.front ().memory_id == 102);
  REQUIRE (processor.soft_anchor_last_links.front ().anchor_id == anchor_id);
  REQUIRE (processor.soft_anchor_last_links.front ().evidence_kind
           == "continued");
  REQUIRE (CountRows (*unique_store, "soft_anchors") == 1);
  REQUIRE (CountRows (*unique_store, "soft_anchor_links") == 2);
}
