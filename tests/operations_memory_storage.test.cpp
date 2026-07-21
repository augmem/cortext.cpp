#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/core/algorithms.hpp>
#include <cortext/core/knobs.hpp>
#include <cortext/operations/memory_storage.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/accumulator_state.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/processor/operation_set.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>
#include "../src/operations/constructive_recall_internal.hpp"
#include "../src/operations/active_signal_embedding_ring_internal.hpp"
#include "../src/operations/execution_cache_sidecar_internal.hpp"
#include "../src/operations/historical_surface_search_cache_internal.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <tuple>

using namespace cortext;
using cortext::operations::MemoryStorage;
using cortext::store::AnyToLongLong;
using cortext::store::BlobFromAny;

namespace
{

constexpr int kEmbeddingDim = 256;

Eigen::VectorXf
UnitVec (int dim)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[dim] = 1.0f;
  return v;
}

Eigen::VectorXf
VectorWithCosineToDim0 (float cosine)
{
  Eigen::VectorXf v = Eigen::VectorXf::Zero (kEmbeddingDim);
  v[0] = cosine;
  v[1] = std::sqrt (std::max (0.0f, 1.0f - cosine * cosine));
  return v;
}

class SeedStorageInputsOp : public IOperation
{
public:
  void
  Execute (OperationContext &ctx, Transaction & /*tx*/) const override
  {
    const auto &signal = ctx.GetSignal ();
    AccumulatorState acc;
    acc.mu_acc = signal.embedding;
    acc.c_t = signal.embedding;
    acc.n_signals = 1;
    acc.s_sum = 0.6;
    acc.s_max = 0.6;
    acc.t_start = signal.timestamp;

    SignalRecord rec;
    rec.embedding = signal.embedding;
    rec.timestamp = signal.timestamp;
    rec.modality = signal.modality;
    rec.mime = signal.mimetype;
    rec.score = 0.6;
    rec.serial_position = 0;
    acc.signals.push_back (std::move (rec));

    ctx.GetProcessorContext ().accumulator_states[signal.source_id]
        = std::move (acc);
    ctx.SetAccumulatorWriteDecision (true);
    ctx.SetRepresentativeEmbedding (signal.embedding);
  }
};

/// @brief RAII wrapper for a temporary database file.
class ScopedTempDb
{
public:
  ScopedTempDb ()
  {
    path_ = cortext::testing::UniqueTempPath ("memory_storage_test_",
                                              ".db").string ();
    std::filesystem::remove (path_);
    store_ = SQLiteStore::Create (path_.c_str ());

    // Initialize core schema
    cortext::testing::InitializeCoreSchema (*store_);
  }

  ~ScopedTempDb ()
  {
    store_.reset ();
    std::filesystem::remove (path_);
  }

  Store *
  get () const
  {
    return store_.get ();
  }

private:
  std::string path_;
  std::unique_ptr<Store> store_;
};

void
RequireDisabledCurrentSurfaceCacheParity (const char *disabled_flag)
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  cortext::testing::ScopedEnvVar disabled (disabled_flag, "1");
  const bool hook_active
      = std::strcmp (disabled_flag,
                     "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL") == 0
            ? operations::constructive_recall::Disabled ()
            : operations::constructive_recall::
                  CurrentSurfaceWritesDisabled ();
  if (!hook_active)
    {
      SKIP ("current-surface write hook is disabled in this build");
    }
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  Signal signal;
  signal.embedding = UnitVec (0);
  signal.timestamp = 2000;
  signal.source_id = "disabled/current/source";
  signal.modality = "text";
  signal.mimetype = "text/plain";

  ProcessorContext pctx;
  REQUIRE (cache::Reset (pctx, {}));
  AccumulatorState acc;
  acc.mu_acc = signal.embedding;
  acc.c_t = signal.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.8;
  acc.s_max = 0.8;
  acc.t_start = signal.timestamp;
  SignalRecord rec;
  rec.embedding = signal.embedding;
  rec.timestamp = signal.timestamp;
  rec.modality = signal.modality;
  rec.mime = signal.mimetype;
  rec.score = 0.8;
  rec.serial_position = 0;
  acc.signals.push_back (std::move (rec));
  pctx.accumulator_states[signal.source_id] = std::move (acc);

  OperationContext ctx (signal, pctx, cfg, store);
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (signal.embedding);
  MemoryStorage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  REQUIRE (ctx.GetStoredMemoryId ().has_value ());
  const auto current_rows = store->Execute (
      "SELECT memory_id FROM current_memory_embeddings WHERE memory_id = ?",
      { *ctx.GetStoredMemoryId () });
  REQUIRE (current_rows.empty ());
  const auto owner = cache::Find (pctx);
  REQUIRE (owner != nullptr);
  REQUIRE (owner->current_memory_index.count (*ctx.GetStoredMemoryId ()) == 0);
  REQUIRE (pctx.retrieval_surface_index.count (*ctx.GetStoredMemoryId ()) == 1);
  cache::Erase (pctx);
}

} // namespace

TEST_CASE ("MemoryStorage uses default SqlObjectStore inside processor "
           "savepoints",
           "[operations][memory_storage][object_store]")
{
  auto unique_store = SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<Store> (std::move (unique_store));

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  auto ops = std::make_unique<DynamicOperationSet> (
      std::make_unique<SeedStorageInputsOp> (),
      std::make_unique<MemoryStorage> ());
  SignalProcessor processor (cfg, store, std::move (ops));

  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.timestamp = 12345;
  s.source_id = "processor-object-source";
  const std::string payload = "stored through default object store";
  s.payload = std::vector<unsigned char> (payload.begin (), payload.end ());
  s.modality = "text";
  s.mimetype = "text/plain";

  const auto out = processor.Process (s);
  REQUIRE (out.stored_memory_id.has_value ());
  REQUIRE (out.stored_signal_id.has_value ());
  REQUIRE (out.operation_ms.at (
               "MemoryStorage.supersession_current_rows_visited")
           == 0.0);
  REQUIRE (out.operation_ms.at (
               "MemoryStorage.supersession_historical_rows_visited")
           == 0.0);
  REQUIRE (out.operation_ms.at (
               "MemoryStorage.supersession_rows_visited_activity")
           == 0.0);
  REQUIRE (out.operation_ms.at (
               "MemoryStorage.supersession_sql_fallback_count")
           == 3.0);

  auto memory_rows = store->Execute (
      "SELECT blob_id FROM memories WHERE memory_id = ?",
      { *out.stored_memory_id });
  REQUIRE (memory_rows.size () == 1);
  const auto blob_id = BlobFromAny (memory_rows[0].at ("blob_id"));
  REQUIRE_FALSE (blob_id.empty ());

  auto object_rows = store->Execute (
      "SELECT data FROM objstore_data WHERE id = ?",
      { blob_id });
  REQUIRE (object_rows.size () == 1);
  REQUIRE (BlobFromAny (object_rows[0].at ("data")) == *s.payload);
}

TEST_CASE ("MemoryStorage stores payload when write_decision is true",
           "[operations][memory_storage]")
{
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.timestamp = 12345;
  s.source_id = "test-source";
  s.payload = std::vector<unsigned char>{ 'h', 'e', 'l', 'l', 'o' };
  s.modality = "text";
  s.mimetype = "text/plain";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.9;
  cfg.sensitivity = 0.1;
  cfg.stability = 0.8;

  // Set up accumulator state required by memory storage
  AccumulatorState acc;
  acc.mu_acc = s.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.5;
  acc.s_max = 0.5;
  acc.t_start = s.timestamp - 1000;
  {
    SignalRecord rec;
    rec.embedding = s.embedding;
    rec.timestamp = s.timestamp;
    rec.modality = s.modality;
    rec.mime = s.mimetype;
    rec.score = 0.5;
    rec.serial_position = 0;
    auto blob_rows = store->Execute ("SELECT objstore_put(?1) AS id",
                                     { *s.payload });
    REQUIRE (blob_rows.size () == 1);
    rec.blob_id = BlobFromAny (blob_rows[0].at ("id"));
    acc.signals.push_back (std::move (rec));
  }
  pctx.accumulator_states[s.source_id] = std::move (acc);

  OperationContext ctx (s, pctx, cfg, store);
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (s.embedding);

  MemoryStorage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  // Verify stored_embedding_id is set
  auto stored_id = ctx.GetStoredEmbeddingId ();
  REQUIRE (stored_id.has_value ());
  REQUIRE (*stored_id > 0);
  auto stored_signal_id = ctx.GetStoredSignalId ();
  REQUIRE (stored_signal_id.has_value ());
  REQUIRE (*stored_signal_id > 0);

  // Verify embedding was inserted
  auto emb_rows
      = store->Execute ("SELECT * FROM embeddings WHERE embedding_id = ?",
                        { *stored_id });
  REQUIRE (emb_rows.size () == 1);

  // Verify memories was inserted (savepoint commits directly)
  auto idx_rows = store->Execute (
      "SELECT * FROM memories WHERE embedding_id = ?", { *stored_id });
  REQUIRE (idx_rows.size () == 1);
  const auto memory_id
      = AnyToLongLong (idx_rows[0].at ("memory_id")).value_or (0);
  REQUIRE (memory_id > 0);
  REQUIRE (std::any_cast<double> (idx_rows[0].at ("source_reliability"))
           == Catch::Approx (cortext::core::SourceReliabilityPrior (
               cfg.focus, cfg.sensitivity, cfg.stability)));
  REQUIRE (std::any_cast<double> (idx_rows[0].at ("source_reliability"))
           != Catch::Approx (0.7));

  // Verify signal row inserted with its own embedding + blob
  auto sig_rows = store->Execute (
      "SELECT signal_id, embedding_id, blob_id FROM signals WHERE memory_id = ?",
      { memory_id });
  REQUIRE (sig_rows.size () == 1);
  REQUIRE (AnyToLongLong (sig_rows[0].at ("signal_id")) == stored_signal_id);
  const auto sig_emb_id = AnyToLongLong (sig_rows[0].at ("embedding_id"));
  REQUIRE (sig_emb_id.has_value ());
  auto sig_blob_id = BlobFromAny (sig_rows[0].at ("blob_id"));
  REQUIRE (!sig_blob_id.empty ());

  // v2: Verify memories has all expected columns (strength, use_frequency now on memories)
  auto fb_rows = store->Execute (
      "SELECT strength, stability, use_frequency FROM memories "
      "WHERE embedding_id = ?", { *stored_id });
  REQUIRE (fb_rows.size () == 1);
  REQUIRE (std::any_cast<double> (fb_rows[0].at ("strength"))
           == Catch::Approx (core::MemoryInitialStrengthPolicy (
               cfg.focus, cfg.sensitivity, cfg.stability)));
  REQUIRE (std::any_cast<double> (fb_rows[0].at ("stability"))
           == Catch::Approx (core::MemoryInitialStabilityPolicy (
               cfg.focus, cfg.sensitivity, cfg.stability)));

  // No buffered instructions since we use savepoints
}

TEST_CASE ("MemoryStorage initializes scalar memory fields from knobs",
           "[operations][memory_storage][knobs]")
{
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.timestamp = 56789;
  s.source_id = "knob-source";
  s.modality = "text";
  s.mimetype = "text/plain";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.0;
  cfg.sensitivity = 1.0;
  cfg.stability = 0.0;

  AccumulatorState acc;
  acc.mu_acc = s.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.4;
  acc.s_max = 0.4;
  acc.t_start = s.timestamp - 500;
  {
    SignalRecord rec;
    rec.embedding = s.embedding;
    rec.timestamp = s.timestamp;
    rec.modality = s.modality;
    rec.mime = s.mimetype;
    rec.score = 0.4;
    rec.serial_position = 0;
    acc.signals.push_back (std::move (rec));
  }
  pctx.accumulator_states[s.source_id] = std::move (acc);

  OperationContext ctx (s, pctx, cfg, store);
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (s.embedding);

  MemoryStorage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  auto stored_id = ctx.GetStoredEmbeddingId ();
  REQUIRE (stored_id.has_value ());

  auto rows = store->Execute (
      "SELECT strength, stability FROM memories WHERE embedding_id = ?",
      { *stored_id });
  REQUIRE (rows.size () == 1);

  const double expected_strength = core::MemoryInitialStrengthPolicy (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const double expected_stability = core::MemoryInitialStabilityPolicy (
      cfg.focus, cfg.sensitivity, cfg.stability);
  REQUIRE (expected_strength < 1.0);
  REQUIRE (expected_stability < 1.0);
  REQUIRE (std::any_cast<double> (rows[0].at ("strength"))
           == Catch::Approx (expected_strength));
  REQUIRE (std::any_cast<double> (rows[0].at ("stability"))
           == Catch::Approx (expected_stability));
}

TEST_CASE ("MemoryStorage discards when write_decision is false",
           "[operations][memory_storage]")
{
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.timestamp = 12345;
  s.source_id = "test-source";
  s.payload = std::vector<unsigned char>{ 'h', 'e', 'l', 'l', 'o' };
  s.modality = "text";
  s.mimetype = "text/plain";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;


  OperationContext ctx (s, pctx, cfg, store);
  ctx.SetAccumulatorWriteDecision (false); // Rejected

  MemoryStorage op;
  op.Execute (ctx, cortext::testing::GetNullTransaction ());

  // Verify stored_embedding_id is NOT set
  auto stored_id = ctx.GetStoredEmbeddingId ();
  REQUIRE_FALSE (stored_id.has_value ());

  // Verify no buffered instructions

  // Verify no embeddings were inserted
  auto emb_rows = store->Execute ("SELECT COUNT(*) AS cnt FROM embeddings", {});
  REQUIRE (emb_rows.size () == 1);
  auto cnt = AnyToLongLong (emb_rows[0].at ("cnt"));
  REQUIRE (cnt.has_value ());
  REQUIRE (*cnt == 0);
}

TEST_CASE ("MemoryStorage stores memory even when no payload",
           "[operations][memory_storage]")
{
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.timestamp = 12345;
  s.source_id = "test-source";
  // No payload set
  s.modality = "text";
  s.mimetype = "text/plain";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Set up accumulator state required by memory storage
  AccumulatorState acc;
  acc.mu_acc = s.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.5;
  acc.s_max = 0.5;
  acc.t_start = s.timestamp - 1000;
  {
    SignalRecord rec;
    rec.embedding = s.embedding;
    rec.timestamp = s.timestamp;
    rec.modality = s.modality;
    rec.mime = s.mimetype;
    rec.score = 0.5;
    rec.serial_position = 0;
    acc.signals.push_back (std::move (rec));
  }
  pctx.accumulator_states[s.source_id] = std::move (acc);

  OperationContext ctx (s, pctx, cfg, store);
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (s.embedding);

  MemoryStorage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  // Verify stored_embedding_id IS set (memory is stored, just no payload blob)
  auto stored_id = ctx.GetStoredEmbeddingId ();
  REQUIRE (stored_id.has_value ());
  REQUIRE (*stored_id > 0);

  // v2: Verify memories row was inserted with null blob_id
  auto mem_rows = store->Execute (
      "SELECT blob_id FROM memories WHERE embedding_id = ?",
      { *stored_id });
  REQUIRE (mem_rows.size () == 1);
  // blob_id should be null for no payload

  // Signals should exist but have no blob_id
  auto sig_rows = store->Execute (
      "SELECT blob_id FROM signals WHERE memory_id = (SELECT memory_id FROM memories WHERE embedding_id = ?)",
      { *stored_id });
  REQUIRE (sig_rows.size () == 1);
  auto sig_blob_id = BlobFromAny (sig_rows[0].at ("blob_id"));
  REQUIRE (sig_blob_id.empty ());
}

TEST_CASE ("MemoryStorage keeps exact signal vectors out of the global route",
           "[operations][memory_storage][embedding_population][modality]")
{
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  ProcessorContext pctx;
  MemoryStorage op;

  const std::vector<std::string> sources = {
    "opaque/source/alpha", "opaque/source/beta"
  };
  const std::vector<std::string> modalities = { "text", "audio", "image" };
  std::vector<Eigen::VectorXf> expected_signal_embeddings;
  std::vector<long long> expected_memory_embedding_ids;

  for (std::size_t source_index = 0; source_index < sources.size ();
       ++source_index)
    {
      Signal signal;
      signal.embedding = UnitVec (static_cast<int> (source_index));
      signal.timestamp = 1000 + source_index * 100;
      signal.source_id = sources[source_index];
      signal.modality = modalities.front ();
      signal.mimetype = "text/plain";

      AccumulatorState acc;
      acc.mu_acc = signal.embedding;
      acc.c_t = signal.embedding;
      acc.n_signals = static_cast<int> (modalities.size ());
      acc.s_sum = 1.8;
      acc.s_max = 0.8;
      acc.t_start = signal.timestamp;
      for (std::size_t modality_index = 0;
           modality_index < modalities.size (); ++modality_index)
        {
          SignalRecord rec;
          rec.embedding = UnitVec (
              static_cast<int> (source_index * modalities.size ()
                                + modality_index));
          rec.timestamp = signal.timestamp + modality_index;
          rec.modality = modalities[modality_index];
          rec.mime = modalities[modality_index] == "text"
                         ? "text/plain"
                         : "application/octet-stream";
          rec.score = 0.6;
          rec.serial_position = static_cast<int> (modality_index);
          expected_signal_embeddings.push_back (rec.embedding);
          acc.signals.push_back (std::move (rec));
        }
      pctx.accumulator_states[signal.source_id] = std::move (acc);

      OperationContext ctx (signal, pctx, cfg, store);
      ctx.SetAccumulatorWriteDecision (true);
      ctx.SetRepresentativeEmbedding (signal.embedding);
      auto tx = store->Begin ();
      op.Execute (ctx, *tx);
      tx->Commit ();

      REQUIRE (ctx.GetStoredEmbeddingId ().has_value ());
      expected_memory_embedding_ids.push_back (*ctx.GetStoredEmbeddingId ());
    }

  const auto embedding_count_rows
      = store->Execute ("SELECT COUNT(*) AS n FROM embeddings");
  REQUIRE (embedding_count_rows.size () == 1);
  REQUIRE (AnyToLongLong (embedding_count_rows[0].at ("n"))
           == static_cast<long long> (sources.size ()));

  const auto signal_rows = store->Execute (
      "SELECT s.source_id, s.embedding_id, a.embedding, a.capacity "
      "FROM signals s "
      "LEFT JOIN cortext_active_signal_embeddings a "
      "ON a.signal_id = s.signal_id "
      "ORDER BY s.signal_id");
  REQUIRE (signal_rows.size () == expected_signal_embeddings.size ());
  for (std::size_t index = 0; index < signal_rows.size (); ++index)
    {
      const std::size_t source_index = index / modalities.size ();
      REQUIRE (std::any_cast<std::string> (
                   signal_rows[index].at ("source_id"))
               == sources[source_index]);
      REQUIRE (AnyToLongLong (signal_rows[index].at ("embedding_id"))
               == expected_memory_embedding_ids[source_index]);
      REQUIRE (AnyToLongLong (signal_rows[index].at ("capacity"))
               == operations::active_signal_embedding_ring_internal::Capacity (
                      cfg.focus, cfg.sensitivity, cfg.stability));
      const auto bytes = BlobFromAny (signal_rows[index].at ("embedding"));
      REQUIRE (bytes.size ()
               == static_cast<std::size_t> (kEmbeddingDim)
                      * sizeof (float));
      Eigen::VectorXf restored (kEmbeddingDim);
      std::memcpy (restored.data (), bytes.data (), bytes.size ());
      REQUIRE (restored.isApprox (expected_signal_embeddings[index], 0.0f));
    }
}

TEST_CASE ("Active signal embedding ring obeys knob-derived capacities",
           "[operations][memory_storage][embedding_population][knobs]")
{
  namespace ring = operations::active_signal_embedding_ring_internal;
  std::vector<std::tuple<double, double, double>> points;
  for (const double focus : { 0.0, 0.5, 1.0 })
    for (const double sensitivity : { 0.0, 0.5, 1.0 })
      for (const double stability : { 0.0, 0.5, 1.0 })
        points.emplace_back (focus, sensitivity, stability);

  REQUIRE (ring::Capacity (0.0, 0.0, 0.0) == 64);
  REQUIRE (ring::Capacity (0.5, 0.5, 0.5) == 151);
  REQUIRE (ring::Capacity (1.0, 1.0, 1.0) == 266);

  auto store = SQLiteStore::Create (":memory:");
  cortext::testing::InitializeCoreSchema (*store);
  for (const auto &[focus, sensitivity, stability] : points)
    {
      store->Execute ("DELETE FROM cortext_active_signal_embeddings");
      store->Execute ("DELETE FROM signals");
      const int capacity = ring::Capacity (focus, sensitivity, stability);
      REQUIRE (capacity
               == operations::sparse_retrieval_knobs_internal::
                      ActiveSignalEmbeddingCapacity (
                          focus, sensitivity, stability));
      REQUIRE (capacity
               >= static_cast<int> (core::NCtx (stability)
                                    + core::KCtx (stability)));

      auto tx = store->Begin ();
      for (long long signal_id = 1; signal_id <= capacity + 3; ++signal_id)
        {
          tx->Execute (
              "INSERT INTO signals(signal_id, source_id, embedding_id, "
              "timestamp, modality, created_at) VALUES (?, ?, 1, ?, ?, ?)",
              { signal_id,
                signal_id % 2 == 0 ? std::string ("opaque/even")
                                    : std::string ("opaque/odd"),
                signal_id,
                signal_id % 3 == 0 ? std::string ("image")
                                    : (signal_id % 3 == 1
                                           ? std::string ("text")
                                           : std::string ("audio")),
                signal_id });
          std::vector<float> embedding (kEmbeddingDim, 0.0f);
          embedding[static_cast<std::size_t> (signal_id) % kEmbeddingDim]
              = 1.0f;
          ring::Upsert (*tx, signal_id, embedding, signal_id, focus,
                        sensitivity, stability);
        }
      tx->Commit ();

      const auto rows = store->Execute (
          "SELECT COUNT(*) AS n, MIN(signal_id) AS first_id, "
          "MAX(signal_id) AS last_id, MIN(capacity) AS min_capacity, "
          "MAX(capacity) AS max_capacity "
          "FROM cortext_active_signal_embeddings");
      REQUIRE (rows.size () == 1);
      REQUIRE (AnyToLongLong (rows[0].at ("n")) == capacity);
      REQUIRE (AnyToLongLong (rows[0].at ("first_id")) == 4);
      REQUIRE (AnyToLongLong (rows[0].at ("last_id")) == capacity + 3);
      REQUIRE (AnyToLongLong (rows[0].at ("min_capacity")) == capacity);
      REQUIRE (AnyToLongLong (rows[0].at ("max_capacity")) == capacity);
    }

  store->Execute ("DELETE FROM cortext_active_signal_embeddings");
  store->Execute ("DELETE FROM signals");
  auto resized_tx = store->Begin ();
  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 1.0f;
  for (long long signal_id = 1; signal_id <= 131; ++signal_id)
    {
      resized_tx->Execute (
          "INSERT INTO signals(signal_id, source_id, embedding_id, timestamp, "
          "modality, created_at) VALUES (?, 'opaque/resize', 1, ?, 'text', ?)",
          { signal_id, signal_id, signal_id });
      if (signal_id <= 130)
        ring::Upsert (*resized_tx, signal_id, embedding, signal_id, 0.5, 0.5,
                      0.5);
      else
        ring::Upsert (*resized_tx, signal_id, embedding, signal_id, 0.0, 0.0,
                      0.0);
    }
  resized_tx->Commit ();
  const auto resized_rows = store->Execute (
      "SELECT COUNT(*) AS n, MIN(signal_id) AS first_id, "
      "MAX(signal_id) AS last_id, MIN(capacity) AS min_capacity, "
      "MAX(capacity) AS max_capacity "
      "FROM cortext_active_signal_embeddings");
  REQUIRE (resized_rows.size () == 1);
  REQUIRE (AnyToLongLong (resized_rows[0].at ("n")) == 64);
  REQUIRE (AnyToLongLong (resized_rows[0].at ("first_id")) == 68);
  REQUIRE (AnyToLongLong (resized_rows[0].at ("last_id")) == 131);
  REQUIRE (AnyToLongLong (resized_rows[0].at ("min_capacity")) == 64);
  REQUIRE (AnyToLongLong (resized_rows[0].at ("max_capacity")) == 64);
}

TEST_CASE ("Active signal embedding ring follows timestamp recency across "
           "out-of-order writes and deletion gaps",
           "[operations][memory_storage][embedding_population][recency]"
           "[regression]")
{
  namespace ring = operations::active_signal_embedding_ring_internal;
  auto store = SQLiteStore::Create (":memory:");
  cortext::testing::InitializeCoreSchema (*store);
  const int capacity = ring::Capacity (0.0, 0.0, 0.0);
  REQUIRE (capacity == 64);

  std::vector<float> embedding (kEmbeddingDim, 0.0f);
  embedding[0] = 1.0f;
  auto tx = store->Begin ();
  for (long long signal_id = 1; signal_id <= capacity; ++signal_id)
    {
      const long long timestamp = 1000 + signal_id;
      tx->Execute (
          "INSERT INTO signals(signal_id, source_id, embedding_id, "
          "timestamp, modality, created_at) VALUES (?, 'opaque/ring', 1, "
          "?, 'text', ?)",
          { signal_id, timestamp, timestamp });
      ring::Upsert (*tx, signal_id, embedding, timestamp, 0.0, 0.0, 0.0);
    }
  tx->Execute (
      "INSERT INTO signals(signal_id, source_id, embedding_id, timestamp, "
      "modality, created_at) VALUES(65, 'opaque/ring', 1, 500, 'audio', "
      "500)");
  ring::Upsert (*tx, 65, embedding, 500, 0.0, 0.0, 0.0);
  tx->Commit ();

  auto rows = store->Execute (
      "SELECT COUNT(*) AS n, MIN(signal_id) AS first_id, "
      "MAX(signal_id) AS last_id FROM cortext_active_signal_embeddings");
  REQUIRE (AnyToLongLong (rows[0].at ("n")) == capacity);
  REQUIRE (AnyToLongLong (rows[0].at ("first_id")) == 1);
  REQUIRE (AnyToLongLong (rows[0].at ("last_id")) == capacity);

  // A deletion gap accepts the next exact vector. Once full again, timestamp
  // recency—not signal identity—owns the replacement frontier.
  store->Execute (
      "DELETE FROM cortext_active_signal_embeddings WHERE signal_id = ?",
      { static_cast<long long> (capacity) });
  tx = store->Begin ();
  tx->Execute (
      "INSERT INTO signals(signal_id, source_id, embedding_id, timestamp, "
      "modality, created_at) VALUES(66, 'opaque/ring', 1, 600, 'image', "
      "600)");
  ring::Upsert (*tx, 66, embedding, 600, 0.0, 0.0, 0.0);
  tx->Commit ();
  rows = store->Execute (
      "SELECT COUNT(*) AS n, SUM(signal_id = 66) AS gap_fill "
      "FROM cortext_active_signal_embeddings");
  REQUIRE (AnyToLongLong (rows[0].at ("n")) == capacity);
  REQUIRE (AnyToLongLong (rows[0].at ("gap_fill")) == 1);

  tx = store->Begin ();
  tx->Execute (
      "INSERT INTO signals(signal_id, source_id, embedding_id, timestamp, "
      "modality, created_at) VALUES(67, 'opaque/ring', 1, 2000, 'text', "
      "2000)");
  ring::Upsert (*tx, 67, embedding, 2000, 0.0, 0.0, 0.0);
  tx->Commit ();
  rows = store->Execute (
      "SELECT COUNT(*) AS n, SUM(signal_id = 66) AS old_present, "
      "SUM(signal_id = 67) AS new_present "
      "FROM cortext_active_signal_embeddings");
  REQUIRE (AnyToLongLong (rows[0].at ("n")) == capacity);
  REQUIRE (AnyToLongLong (rows[0].at ("old_present")) == 0);
  REQUIRE (AnyToLongLong (rows[0].at ("new_present")) == 1);
}

TEST_CASE ("An emptied exact signal ring does not rebuild from flattened "
           "memory centroids",
           "[operations][memory_storage][embedding_population][regression]")
{
  namespace ring = operations::active_signal_embedding_ring_internal;
  auto store = SQLiteStore::Create (":memory:");
  cortext::testing::InitializeCoreSchema (*store);
  const int capacity = ring::Capacity (0.0, 0.0, 0.0);

  const Eigen::VectorXf aggregate = UnitVec (0);
  cortext::testing::SeedEmbeddingV2 (*store, 100, aggregate, 1000);
  cortext::testing::SeedMemoryV2 (
      *store, 10, 100, "opaque/flattened", "LONG_TERM", 1.0, 1000);
  store->Execute (
      "INSERT INTO signals(signal_id, memory_id, source_id, embedding_id, "
      "timestamp, modality, created_at) "
      "VALUES(1, 10, 'opaque/flattened', 100, 1000, 'audio', 1000)");
  std::vector<float> prior_exact (kEmbeddingDim, 0.0f);
  prior_exact[1] = 1.0f;
  auto tx = store->Begin ();
  ring::InsertAtCapacity (*tx, 1, prior_exact, 1000, capacity);
  tx->Commit ();

  // Simulate a later cascade that removes every currently retained exact
  // vector while an older flattened signal row remains authoritative.
  store->Execute ("DELETE FROM cortext_active_signal_embeddings");
  store->Execute (
      "INSERT INTO signals(signal_id, memory_id, source_id, embedding_id, "
      "timestamp, modality, created_at) "
      "VALUES(2, 10, 'opaque/new', 100, 2000, 'image', 2000)");
  std::vector<float> newest_exact (kEmbeddingDim, 0.0f);
  newest_exact[2] = 1.0f;
  tx = store->Begin ();
  ring::Upsert (*tx, 2, newest_exact, 2000, 0.0, 0.0, 0.0);
  tx->Commit ();

  const auto rows = store->Execute (
      "SELECT signal_id, embedding FROM cortext_active_signal_embeddings");
  REQUIRE (rows.size () == 1);
  REQUIRE (AnyToLongLong (rows.front ().at ("signal_id")) == 2);
  const auto bytes = BlobFromAny (rows.front ().at ("embedding"));
  REQUIRE (bytes.size () == static_cast<std::size_t> (kEmbeddingDim)
                                * sizeof (float));
  Eigen::VectorXf restored (kEmbeddingDim);
  std::memcpy (restored.data (), bytes.data (), bytes.size ());
  REQUIRE (restored.isApprox (UnitVec (2), 0.0f));
}

TEST_CASE ("MemoryStorage writes modality-agnostic supersedes edges",
           "[operations][memory_storage][supersession][fanout_cache]")
{
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  const Eigen::VectorXf stale_embedding = UnitVec (0);
  const Eigen::VectorXf correction_embedding = VectorWithCosineToDim0 (0.94f);
  cortext::testing::SeedEmbeddingV2 (*store, 100, stale_embedding, 1000);
  cortext::testing::SeedMemoryV2 (*store, 10, 100, "belief/source",
                                  "LONG_TERM", 1.0, 1000);

  Signal s;
  s.embedding = correction_embedding;
  s.timestamp = 2000;
  s.source_id = "belief/source";
  s.modality = "image";
  s.mimetype = "image/custom-test";

  ProcessorContext pctx;
  pctx.UpsertRetrievalSurface (
      { 10, 100, 1000, 1000, 0, 0, 0, 0, "LONG_TERM", "belief/source",
        "text", -1.0, 0, 0.0, 0.0, 0.0, false, true,
        stale_embedding });
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);

  AccumulatorState acc;
  acc.mu_acc = s.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.8;
  acc.s_max = 0.8;
  acc.t_start = s.timestamp;
  {
    SignalRecord rec;
    rec.embedding = s.embedding;
    rec.timestamp = s.timestamp;
    rec.modality = s.modality;
    rec.mime = s.mimetype;
    rec.score = 0.8;
    rec.serial_position = 0;
    acc.signals.push_back (std::move (rec));
  }
  pctx.accumulator_states[s.source_id] = std::move (acc);
  pctx.association_fanout_cache.valid = true;

  OperationContext ctx (s, pctx, cfg, store);
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (s.embedding);

  MemoryStorage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  REQUIRE (ctx.GetStoredMemoryId ().has_value ());
  const long long correction_memory_id = *ctx.GetStoredMemoryId ();
  auto edge_rows = store->Execute (
      "SELECT weight FROM associations "
      "WHERE source_memory_id = ? "
      "  AND target_memory_id = ? "
      "  AND edge_type = 'supersedes'",
      { correction_memory_id, 10LL });
  REQUIRE (edge_rows.size () == 1);
  REQUIRE (std::any_cast<double> (edge_rows[0].at ("weight")) > 0.0);
  REQUIRE (pctx.association_fanout_cache.valid);
  REQUIRE (pctx.association_fanout_cache.out_by_source
               .at (correction_memory_id)
               .front ().memory_id
           == 10LL);
  const auto sidecar
      = operations::execution_cache_sidecar_internal::Find (pctx);
  REQUIRE (sidecar);
  REQUIRE (sidecar->supersession_eligibility.valid);
  REQUIRE (sidecar->supersession_eligibility.activation_ts_by_target.at (10LL)
           == 2000LL);

  auto stale_rows = store->Execute (
      "SELECT source_contradiction_count FROM memories WHERE memory_id = ?",
      { 10LL });
  REQUIRE (stale_rows.size () == 1);
  REQUIRE (AnyToLongLong (stale_rows[0].at ("source_contradiction_count"))
           == 1LL);

}

TEST_CASE ("Typed supersession candidates retain borrowed cache vectors",
           "[operations][memory_storage][supersession][cache]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  ProcessorContext pctx;
  const Eigen::VectorXf zero = Eigen::VectorXf::Zero (kEmbeddingDim);
  REQUIRE (cache::Reset (
      pctx, { { 100, 10, 1000, "LONG_TERM", "source/10", zero } }));

  auto owner = cache::Find (pctx);
  REQUIRE (owner != nullptr);
  const Eigen::VectorXf *borrowed = &owner->entries[0].embedding;
  cache::SupersessionCandidateRows result;
  result.cache_owner = owner;
  result.rows.push_back ({ 10, borrowed, std::nullopt });

  cache::Erase (pctx);
  owner.reset ();
  REQUIRE (cache::Find (pctx) == nullptr);
  REQUIRE (result.rows[0].Embedding () == borrowed);
  REQUIRE (result.rows[0].Embedding ()->size () == kEmbeddingDim);
  REQUIRE (result.rows[0].Embedding ()->norm () == 0.0f);

  cache::SupersessionCandidate owned;
  owned.memory_id = 11;
  owned.owned_embedding = Eigen::VectorXf::Ones (4);
  REQUIRE (owned.Embedding () == &*owned.owned_embedding);
  REQUIRE (owned.Embedding ()->size () == 4);

  cache::SupersessionCandidate invalid;
  invalid.memory_id = 12;
  REQUIRE (invalid.Embedding () == nullptr);
}

TEST_CASE ("Historical cache tracks only supersession-eligible embeddings",
           "[operations][memory_storage][supersession][coverage]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  ProcessorContext pctx;
  REQUIRE (cache::Reset (
      pctx,
      { { 100, 10, 1000, "LONG_TERM", "source/a", UnitVec (0) },
        { 101, 11, 1001, "ASSOCIATION", "source/b", UnitVec (1) },
        { 102, 0, 0, std::string (), std::string (), UnitVec (2) } }));
  auto state = cache::Find (pctx);
  REQUIRE (state != nullptr);
  REQUIRE (state->supersession_entry_indices.size () == 2);

  cache::Append (
      pctx, { 103, 12, 1002, "ASSOCIATION", "source/c", UnitVec (3) });
  cache::Append (
      pctx, { 104, 0, 0, std::string (), std::string (), UnitVec (4) });
  cache::RemoveEmbedding (pctx, 100);
  state = cache::Find (pctx);
  REQUIRE (state != nullptr);
  REQUIRE (state->supersession_entry_indices.size () == 2);
  for (const std::size_t index : state->supersession_entry_indices)
    {
      REQUIRE (index < state->entries.size ());
      REQUIRE (cache::IsSupersessionCandidateEntry (state->entries[index]));
      REQUIRE (state->supersession_index_positions.at (index)
               < state->supersession_entry_indices.size ());
    }
  cache::Erase (pctx);
}

TEST_CASE ("Shared embeddings retain every supersession memory sibling",
           "[operations][memory_storage][supersession][coverage]"
           "[shared-embedding]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  const Eigen::VectorXf stale_embedding = UnitVec (0);
  const Eigen::VectorXf correction_embedding = VectorWithCosineToDim0 (0.94f);
  const int candidate_limit = std::max (
      1, core::SupersessionCandidateLimit (cfg.focus, cfg.sensitivity,
                                           cfg.stability));
  const int max_edges = core::SupersessionMaxEdges (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const int sibling_count = candidate_limit + 1;
  cortext::testing::SeedEmbeddingV2 (*store, 100, stale_embedding, 1000);

  ProcessorContext pctx;
  std::vector<cache::Entry> historical_entries;
  std::vector<cache::Entry> current_entries;
  historical_entries.reserve (static_cast<std::size_t> (sibling_count));
  current_entries.reserve (static_cast<std::size_t> (sibling_count));
  for (int index = 0; index < sibling_count; ++index)
    {
      const long long sibling_id = 10LL + index;
      const long long start_ts = 1000LL + index;
      const std::string source_id
          = "belief/sibling/" + std::to_string (index);
      cortext::testing::SeedMemoryV2 (*store, sibling_id, 100, source_id,
                                      "LONG_TERM", 1.0, start_ts);
      ProcessorContext::RetrievalSurfaceEntry current;
      current.memory_id = sibling_id;
      current.embedding_id = 100;
      current.start_ts = start_ts;
      current.kind = "LONG_TERM";
      current.embedding = stale_embedding;
      pctx.UpsertRetrievalSurface (std::move (current));
      historical_entries.push_back (
          { 100, sibling_id, start_ts, "LONG_TERM", source_id,
            stale_embedding });
      current_entries.push_back (
          { 100, sibling_id, start_ts, "LONG_TERM", source_id,
            stale_embedding, 100 });
    }
  REQUIRE (cache::Reset (pctx, std::move (historical_entries),
                         std::move (current_entries)));
  const auto cache_state = cache::Find (pctx);
  REQUIRE (cache_state);
  REQUIRE (cache_state->supersession_entry_by_memory.contains (10));
  REQUIRE (cache_state->supersession_entry_by_memory.contains (
      9LL + sibling_count));
  REQUIRE (cache_state->supersession_embedding_fanout);
  REQUIRE_FALSE (cache::CurrentPopulationCoversHistorical (
      *cache_state, -1));

  Signal signal;
  signal.embedding = correction_embedding;
  signal.timestamp = 2000;
  signal.source_id = "belief/correction";
  signal.modality = "text";
  signal.mimetype = "text/plain";
  AccumulatorState acc;
  acc.mu_acc = signal.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.8;
  acc.s_max = 0.8;
  acc.t_start = signal.timestamp;
  SignalRecord record;
  record.embedding = signal.embedding;
  record.timestamp = signal.timestamp;
  record.modality = signal.modality;
  record.mime = signal.mimetype;
  record.score = 0.8;
  acc.signals.push_back (std::move (record));
  pctx.accumulator_states[signal.source_id] = std::move (acc);

  OperationContext ctx (signal, pctx, cfg, store);
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (signal.embedding);
  MemoryStorage operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Commit ();
  REQUIRE (ctx.GetStoredMemoryId ());
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_current_rows_visited")
           == static_cast<double> (sibling_count));
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_rows_visited_activity")
           == 1.0);
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_historical_coverage_proven")
           == 0.0);
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_sql_fallback_count")
           == 0.0);
  const auto edges = store->Execute (
      "SELECT target_memory_id FROM associations "
      "WHERE source_memory_id = ? AND edge_type = 'supersedes' "
      "ORDER BY target_memory_id",
      { *ctx.GetStoredMemoryId () });
  REQUIRE (edges.size () == static_cast<std::size_t> (max_edges));
  for (int index = 0; index < max_edges; ++index)
    REQUIRE (AnyToLongLong (edges[static_cast<std::size_t> (index)].at (
                                "target_memory_id"))
             == 10LL + sibling_count - max_edges + index);
  cache::Erase (pctx);
}

TEST_CASE ("Exact empty supersession coverage skips the recent SQL fallback",
           "[operations][memory_storage][supersession][coverage]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  Signal signal;
  signal.embedding = UnitVec (0);
  signal.timestamp = 2000;
  signal.source_id = "fresh/source";
  signal.modality = "text";
  signal.mimetype = "text/plain";

  ProcessorContext pctx;
  REQUIRE (cache::Reset (pctx, {}));
  AccumulatorState acc;
  acc.mu_acc = signal.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.8;
  acc.s_max = 0.8;
  acc.t_start = signal.timestamp;
  SignalRecord record;
  record.embedding = signal.embedding;
  record.timestamp = signal.timestamp;
  record.modality = signal.modality;
  record.mime = signal.mimetype;
  record.score = 0.8;
  acc.signals.push_back (std::move (record));
  pctx.accumulator_states[signal.source_id] = std::move (acc);

  OperationContext ctx (signal, pctx, cfg, store);
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (signal.embedding);
  MemoryStorage operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Commit ();

  REQUIRE (ctx.GetStoredMemoryId ());
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_current_rows_visited")
           == 0.0);
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_historical_rows_visited")
           == 0.0);
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_rows_visited_activity")
           == 0.0);
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_historical_coverage_proven")
           == 1.0);
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_sql_fallback_count")
           == 0.0);
  cache::Erase (pctx);
}

TEST_CASE ("Large supersession population uses the knob-bounded SQLite route",
           "[operations][memory_storage][supersession][hnsw][sqlite][knobs]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  namespace sqlite_route
      = operations::sparse_retrieval_route_sqlite_internal;
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 1.0;
  cfg.sensitivity = 0.0;
  cfg.stability = 0.0;
  const auto parameters = sqlite_route::DeriveParameters (
      cfg.focus, cfg.sensitivity, cfg.stability);
  REQUIRE (parameters.route_capacity == 512);
  REQUIRE (parameters.backfill_batch_size == 128);
  const std::size_t population = parameters.route_capacity + 1;
  const int candidate_limit = core::SupersessionCandidateLimit (
      cfg.focus, cfg.sensitivity, cfg.stability);
  REQUIRE (candidate_limit == 67);
  REQUIRE (static_cast<std::size_t> (candidate_limit) < population);
  const long long expected_target_id = 10 + candidate_limit;

  ProcessorContext pctx;
  std::vector<cache::Entry> historical_entries;
  std::vector<cache::Entry> current_entries;
  std::vector<std::pair<long long, Eigen::VectorXf>> route_entries;
  historical_entries.reserve (population);
  current_entries.reserve (population);
  route_entries.reserve (population);
  const Eigen::VectorXf target = VectorWithCosineToDim0 (0.94f);
  for (std::size_t index = 0; index < population; ++index)
    {
      const long long memory_id = 10 + static_cast<long long> (index);
      const long long embedding_id = 1000 + static_cast<long long> (index);
      const bool blocks_unfiltered_top_k
          = index < static_cast<std::size_t> (candidate_limit);
      const long long start_ts
          = blocks_unfiltered_top_k
                ? 20000 + static_cast<long long> (index)
                : 1000 + static_cast<long long> (index);
      Eigen::VectorXf embedding = Eigen::VectorXf::Zero (kEmbeddingDim);
      if (index < static_cast<std::size_t> (candidate_limit))
        embedding = UnitVec (0);
      else if (index == static_cast<std::size_t> (candidate_limit))
        embedding = target;
      else
        {
          embedding[1 + static_cast<Eigen::Index> (
                            index % (kEmbeddingDim - 1))]
              = 1.0f;
        }
      const std::string source_id
          = "opaque/" + std::to_string (index % 4);
      cortext::testing::SeedEmbeddingV2 (
          *store, embedding_id, embedding, start_ts);
      cortext::testing::SeedMemoryV2 (
          *store, memory_id, embedding_id, source_id, "LONG_TERM", 1.0,
          start_ts);
      ProcessorContext::RetrievalSurfaceEntry surface;
      surface.memory_id = memory_id;
      surface.embedding_id = embedding_id;
      surface.start_ts = start_ts;
      surface.kind = "LONG_TERM";
      surface.embedding = embedding;
      pctx.UpsertRetrievalSurface (std::move (surface));
      historical_entries.push_back (
          { embedding_id, memory_id, start_ts, "LONG_TERM", source_id,
            embedding });
      current_entries.push_back (
          { embedding_id, memory_id, start_ts, "LONG_TERM", source_id,
            embedding, embedding_id });
      route_entries.emplace_back (memory_id, std::move (embedding));
    }
  REQUIRE (cache::Reset (pctx, std::move (historical_entries),
                         std::move (current_entries)));
  cache::SetProcessorSurfaceComplete (pctx, true);
  cache::SetSparseRouteParameters (pctx, parameters);
  auto state = cache::FindMutable (pctx);
  REQUIRE (state);
  auto hnsw = cache::EnsureSparseRoute (*state);
  REQUIRE (hnsw);
  state->sqlite_sparse_route = sqlite_route::Route::Create (
      *store, kEmbeddingDim, route_entries, *hnsw, parameters);
  REQUIRE (state->sqlite_sparse_route);

  Signal signal;
  signal.embedding = UnitVec (0);
  signal.timestamp = 10000;
  signal.source_id = "opaque/correction";
  signal.modality = "image";
  signal.mimetype = "image/test";
  AccumulatorState acc;
  acc.mu_acc = signal.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.8;
  acc.s_max = 0.8;
  acc.t_start = signal.timestamp;
  SignalRecord record;
  record.embedding = signal.embedding;
  record.timestamp = signal.timestamp;
  record.modality = signal.modality;
  record.mime = signal.mimetype;
  record.score = 0.8;
  acc.signals.push_back (std::move (record));
  pctx.accumulator_states[signal.source_id] = std::move (acc);

  OperationContext ctx (signal, pctx, cfg, store);
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (signal.embedding);
  MemoryStorage operation;
  auto tx = store->Begin ();
  operation.Execute (ctx, *tx);
  tx->Commit ();

  REQUIRE (ctx.GetStoredMemoryId ());
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_current_rows_visited")
           <= static_cast<double> (parameters.activation_identity_target));
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_sparse_route_node_rows")
           <= static_cast<double> (parameters.search_node_budget));
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_sparse_route_distance_evaluations")
           <= ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_sparse_route_node_rows"));
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_historical_rows_visited")
           == static_cast<double> (population + 1));
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_historical_coverage_proven")
           == 1.0);
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_sql_fallback_count")
           == 0.0);
  const auto edge_rows = store->Execute (
      "SELECT 1 FROM associations WHERE source_memory_id = ? "
      "AND target_memory_id = ? AND edge_type = 'supersedes'",
      { *ctx.GetStoredMemoryId (), expected_target_id });
  REQUIRE (edge_rows.size () == 1);
  cache::Erase (pctx);
}

TEST_CASE ("Equivalent current population proves historical coverage in O1",
           "[operations][memory_storage][supersession][coverage]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  ProcessorContext pctx;
  REQUIRE (cache::Reset (
      pctx,
      { { 100, 10, 1000, "LONG_TERM", "source/a", UnitVec (0) },
        { 101, 11, 1001, "LONG_TERM", "source/b", UnitVec (1) } },
      { { 100, 10, 0, std::string (), std::string (), UnitVec (0), 100 },
        { 101, 11, 0, std::string (), std::string (), UnitVec (1), 101 } }));
  auto state = cache::Find (pctx);
  REQUIRE (state != nullptr);
  REQUIRE (cache::CurrentPopulationCoversHistorical (*state, -1));

  cache::Append (
      pctx, { 102, 12, 1002, "LONG_TERM", "source/c", UnitVec (2) });
  state = cache::Find (pctx);
  REQUIRE (state != nullptr);
  REQUIRE (cache::CurrentPopulationCoversHistorical (*state, 12));
  REQUIRE_FALSE (cache::CurrentPopulationCoversHistorical (*state, 10));

  cache::UpsertCurrent (
      pctx,
      { 102, 12, 0, std::string (), std::string (), UnitVec (2), 102 });
  state = cache::Find (pctx);
  REQUIRE (cache::CurrentPopulationCoversHistorical (*state, -1));

  cache::UpsertCurrent (
      pctx,
      { 200, 10, 0, std::string (), std::string (), UnitVec (3), 100 });
  state = cache::Find (pctx);
  REQUIRE_FALSE (cache::CurrentPopulationCoversHistorical (*state, -1));
  cache::UpsertCurrent (
      pctx,
      { 100, 10, 0, std::string (), std::string (), UnitVec (0), 100 });
  state = cache::Find (pctx);
  REQUIRE (cache::CurrentPopulationCoversHistorical (*state, -1));
  cache::Erase (pctx);

  ProcessorContext reversed;
  REQUIRE (cache::Reset (
      reversed,
      { { 100, 11, 1000, "LONG_TERM", "source/a", UnitVec (0) },
        { 101, 10, 1001, "LONG_TERM", "source/b", UnitVec (1) } },
      { { 100, 11, 0, std::string (), std::string (), UnitVec (0), 100 },
        { 101, 10, 0, std::string (), std::string (), UnitVec (1), 101 } }));
  state = cache::Find (reversed);
  REQUIRE (state != nullptr);
  REQUIRE_FALSE (cache::CurrentPopulationCoversHistorical (*state, -1));
  cache::Erase (reversed);
}

TEST_CASE ("Historical surface cache survives sequential thread migration and "
           "erases across threads",
           "[operations][memory_storage][cache][lifecycle]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  const std::size_t baseline_size = cache::RegistrySizeForTest ();
  alignas (ProcessorContext) std::byte storage[sizeof (ProcessorContext)];
  auto *pctx = std::construct_at (
      reinterpret_cast<ProcessorContext *> (storage));
  const Eigen::VectorXf first = UnitVec (0);
  const Eigen::VectorXf second = UnitVec (1);
  REQUIRE (cache::Reset (
      *pctx, { { 100, 10, 1000, "LONG_TERM", "source/10", first } }));
  REQUIRE (cache::RegistrySizeForTest () == baseline_size + 1);

  std::atomic<bool> worker_found{ false };
  std::thread migrate ([&] {
    const auto owner = cache::Find (*pctx);
    worker_found.store (owner != nullptr && owner->entries.size () == 1,
                        std::memory_order_relaxed);
    cache::Append (
        *pctx, { 101, 11, 2000, "LONG_TERM", "source/11", second });
  });
  migrate.join ();

  REQUIRE (worker_found.load (std::memory_order_relaxed));
  auto owner = cache::Find (*pctx);
  REQUIRE (owner != nullptr);
  REQUIRE (owner->entries.size () == 2);
  owner.reset ();

  std::thread teardown ([&] { cache::Erase (*pctx); });
  teardown.join ();
  REQUIRE (cache::Find (*pctx) == nullptr);
  REQUIRE (cache::RegistrySizeForTest () == baseline_size);
  std::destroy_at (pctx);

  pctx = std::construct_at (reinterpret_cast<ProcessorContext *> (storage));
  REQUIRE (cache::Find (*pctx) == nullptr);
  REQUIRE (cache::Reset (
      *pctx, { { 102, 12, 3000, "LONG_TERM", "source/12", first } }));
  cache::Erase (*pctx);
  std::destroy_at (pctx);
  REQUIRE (cache::RegistrySizeForTest () == baseline_size);
}

TEST_CASE ("Precomputed supersession query norm preserves cosine results",
           "[operations][memory_storage][supersession][cosine]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  auto require_equivalent = [] (const Eigen::VectorXf &query,
                                const Eigen::VectorXf &target) {
    const double expected = core::CosineSimilarity (query, target);
    const double actual = cache::SupersessionCosineSimilarity (
        query, query.norm (), target);
    REQUIRE (actual == expected);
  };

  const Eigen::VectorXf empty;
  require_equivalent (empty, empty);
  require_equivalent (Eigen::VectorXf::Ones (4),
                      Eigen::VectorXf::Ones (3));
  require_equivalent (Eigen::VectorXf::Zero (4),
                      Eigen::VectorXf::Ones (4));
  require_equivalent (Eigen::VectorXf::Ones (4),
                      Eigen::VectorXf::Zero (4));

  const Eigen::VectorXf query = UnitVec (0);
  require_equivalent (query, UnitVec (0));
  require_equivalent (query, -UnitVec (0));
  require_equivalent (query, VectorWithCosineToDim0 (0.37f));

  SignalProcessor::Config cfg;
  const float threshold = static_cast<float> (
      core::SupersessionSimilarityThreshold (
          cfg.focus, cfg.sensitivity, cfg.stability));
  require_equivalent (
      query, VectorWithCosineToDim0 (
                 std::nextafter (
                     threshold, -std::numeric_limits<float>::infinity ())));
  require_equivalent (
      query, VectorWithCosineToDim0 (
                 std::nextafter (
                     threshold, std::numeric_limits<float>::infinity ())));

  const Eigen::VectorXf tied = VectorWithCosineToDim0 (0.52f);
  const double query_norm = query.norm ();
  const double first
      = cache::SupersessionCosineSimilarity (query, query_norm, tied);
  const double second
      = cache::SupersessionCosineSimilarity (query, query_norm, tied);
  REQUIRE (first == second);
}

TEST_CASE ("MemoryStorage preserves SQL KNN behavior with signal-only decoys",
           "[operations][memory_storage][supersession][knn_population]")
{
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  const Eigen::VectorXf stale_embedding = UnitVec (0);
  const Eigen::VectorXf correction_embedding = VectorWithCosineToDim0 (0.94f);
  cortext::testing::SeedEmbeddingV2 (*store, 100, stale_embedding, 1000);
  cortext::testing::SeedMemoryV2 (*store, 10, 100, "belief/stale",
                                  "LONG_TERM", 1.0, 1000);
  cortext::testing::SeedEmbeddingV2 (*store, 101, correction_embedding, 1000);
  cortext::testing::SeedMemoryV2 (*store, 11, 101, "belief/duplicate",
                                  "LONG_TERM", 1.0, 1000);

  const int candidate_limit = std::max (
      1, core::SupersessionCandidateLimit (
             cfg.focus, cfg.sensitivity, cfg.stability));
  Eigen::VectorXf decoy_embedding = correction_embedding;
  decoy_embedding[2] = 0.01f;
  decoy_embedding.normalize ();
  for (int i = 0; i < candidate_limit; ++i)
    {
      cortext::testing::SeedEmbeddingV2 (
          *store, 1000LL + i, decoy_embedding, 1000);
    }

  const auto sql_candidates = store->Execute (
      "SELECT m.memory_id FROM embeddings e "
      "JOIN memories m ON m.embedding_id = e.embedding_id "
      "WHERE e.embedding MATCH ? AND k = ? "
      "  AND m.memory_id != ? "
      "  AND m.embedding_id IS NOT NULL "
      "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
      "  AND COALESCE(m.start_ts, 0) < ?",
      { std::vector<float> (correction_embedding.data (),
                            correction_embedding.data ()
                                + correction_embedding.size ()),
        static_cast<long long> (candidate_limit), -1LL, 2000LL });
  REQUIRE (std::none_of (
      sql_candidates.begin (), sql_candidates.end (), [] (const auto &row) {
        return AnyToLongLong (row.at ("memory_id")) == 10LL;
      }));

  Signal s;
  s.embedding = correction_embedding;
  s.timestamp = 2000;
  s.source_id = "belief/source";
  s.modality = "text";
  s.mimetype = "text/plain";

  ProcessorContext pctx;
  ProcessorContext::RetrievalSurfaceEntry stale_surface;
  stale_surface.memory_id = 10;
  stale_surface.embedding_id = 100;
  stale_surface.start_ts = 1000;
  stale_surface.kind = "LONG_TERM";
  stale_surface.embedding = stale_embedding;
  pctx.UpsertRetrievalSurface (std::move (stale_surface));
  ProcessorContext::RetrievalSurfaceEntry duplicate_surface;
  duplicate_surface.memory_id = 11;
  duplicate_surface.embedding_id = 101;
  duplicate_surface.start_ts = 1000;
  duplicate_surface.kind = "LONG_TERM";
  duplicate_surface.embedding = correction_embedding;
  pctx.UpsertRetrievalSurface (std::move (duplicate_surface));
  std::vector<operations::historical_surface_search_cache_internal::Entry>
      historical_entries = {
        { 100, 10, 1000, "LONG_TERM", "belief/stale", stale_embedding },
        { 101, 11, 1000, "LONG_TERM", "belief/duplicate",
          correction_embedding }
      };
  for (int i = 0; i < candidate_limit; ++i)
    {
      historical_entries.push_back (
          { 1000LL + i, 0, 0, std::string (), std::string (),
            decoy_embedding });
    }
  REQUIRE (operations::historical_surface_search_cache_internal::Reset (
      pctx, std::move (historical_entries),
      { { 100, 10, 0, std::string (), std::string (), stale_embedding },
        { 101, 11, 0, std::string (), std::string (),
          correction_embedding } }));
  AccumulatorState acc;
  acc.mu_acc = s.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.8;
  acc.s_max = 0.8;
  acc.t_start = s.timestamp;
  SignalRecord rec;
  rec.embedding = s.embedding;
  rec.timestamp = s.timestamp;
  rec.modality = s.modality;
  rec.mime = s.mimetype;
  rec.score = 0.8;
  rec.serial_position = 0;
  acc.signals.push_back (std::move (rec));
  pctx.accumulator_states[s.source_id] = std::move (acc);

  OperationContext ctx (s, pctx, cfg, store);
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (s.embedding);
  MemoryStorage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  REQUIRE (ctx.GetStoredMemoryId ().has_value ());
  REQUIRE (ctx.GetOperationTimings ().at (
               "MemoryStorage.supersession_historical_coverage_proven")
           == 1.0);
  const auto edge_rows = store->Execute (
      "SELECT 1 FROM associations "
      "WHERE source_memory_id = ? AND target_memory_id = ? "
      "  AND edge_type = 'supersedes'",
      { *ctx.GetStoredMemoryId (), 10LL });
  // The private cache mirrors the eligible memory population returned by the
  // sqlite-vec query. Orphan/signal-only embeddings do not hide this candidate
  // in either path.
  REQUIRE (edge_rows.size () == 1);
  operations::historical_surface_search_cache_internal::Erase (pctx);
}

TEST_CASE ("Runtime reconstruction does not restore signal-only cutoff pollution",
           "[operations][memory_storage][supersession][reconstruction][knn_population]")
{
  namespace cache
      = operations::historical_surface_search_cache_internal;
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  const int candidate_limit = std::max (
      1, core::SupersessionCandidateLimit (
             cfg.focus, cfg.sensitivity, cfg.stability));
  REQUIRE (candidate_limit >= 4);
  const Eigen::VectorXf target_embedding = VectorWithCosineToDim0 (0.94f);
  const Eigen::VectorXf closer_embedding = VectorWithCosineToDim0 (0.99f);
  const Eigen::VectorXf reconstruction_embedding
      = VectorWithCosineToDim0 (0.999f);
  cortext::testing::SeedEmbeddingV2 (*store, 100, target_embedding, 1000);
  cortext::testing::SeedMemoryV2 (*store, 10, 100, "belief/target",
                                  "LONG_TERM", 1.0, 1000);

  std::vector<cache::Entry> initial_entries = {
    { 100, 10, 1000, "LONG_TERM", "belief/target", target_embedding }
  };
  for (int i = 0; i < candidate_limit - 3; ++i)
    {
      const long long embedding_id = 200 + i;
      const long long memory_id = 20 + i;
      cortext::testing::SeedEmbeddingV2 (
          *store, embedding_id, closer_embedding, 1000);
      cortext::testing::SeedMemoryV2 (
          *store, memory_id, embedding_id,
          "belief/closer/" + std::to_string (i), "LONG_TERM", 1.0, 1000);
      initial_entries.push_back (
          { embedding_id, memory_id, 1000, "LONG_TERM",
            "belief/closer/" + std::to_string (i), closer_embedding });
    }

  ProcessorContext pctx;
  REQUIRE (cache::Reset (pctx, std::move (initial_entries)));
  operations::constructive_recall::ReconstructionUpdatePolicy policy;
  policy.update_current_surface = false;
  auto reconstruction_tx = store->Begin ();
  const long long reconstruction_id
      = operations::constructive_recall::AppendReconstructionWithEmbedding (
          *reconstruction_tx, 10, reconstruction_embedding, {}, 1500, 0.1,
          "test", 1.0, 1.0, policy, &pctx);
  REQUIRE (reconstruction_id > 0);
  reconstruction_tx->Commit ();
  const auto reconstruction_rows = store->Execute (
      "SELECT embedding_id FROM memory_reconstructions "
      "WHERE reconstruction_id = ?",
      { reconstruction_id });
  REQUIRE (reconstruction_rows.size () == 1);
  const long long reconstruction_embedding_id = AnyToLongLong (
      reconstruction_rows[0].at ("embedding_id")).value_or (0);
  REQUIRE (reconstruction_embedding_id > 0);
  const auto cache_owner = cache::Find (pctx);
  REQUIRE (cache_owner != nullptr);
  REQUIRE (cache_owner->embedding_index.count (reconstruction_embedding_id)
           == 1);
  REQUIRE (cache_owner->entries[
               cache_owner->embedding_index.at (reconstruction_embedding_id)]
               .memory_id
           == 0);

  Signal signal;
  signal.embedding = UnitVec (0);
  signal.timestamp = 2000;
  signal.source_id = "belief/correction";
  signal.modality = "text";
  signal.mimetype = "text/plain";
  AccumulatorState acc;
  acc.mu_acc = signal.embedding;
  acc.c_t = signal.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.8;
  acc.s_max = 0.8;
  acc.t_start = signal.timestamp;
  SignalRecord rec;
  rec.embedding = signal.embedding;
  rec.timestamp = signal.timestamp;
  rec.modality = signal.modality;
  rec.mime = signal.mimetype;
  rec.score = 0.8;
  rec.serial_position = 0;
  acc.signals.push_back (std::move (rec));
  pctx.accumulator_states[signal.source_id] = std::move (acc);

  OperationContext ctx (signal, pctx, cfg, store);
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (signal.embedding);
  MemoryStorage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();
  REQUIRE (ctx.GetStoredMemoryId ().has_value ());

  const auto sql_candidates = store->Execute (
      "SELECT m.memory_id FROM embeddings e "
      "JOIN memories m ON m.embedding_id = e.embedding_id "
      "WHERE e.embedding MATCH ? AND k = ? "
      "  AND m.memory_id != ? "
      "  AND m.embedding_id IS NOT NULL "
      "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
      "  AND COALESCE(m.start_ts, 0) < ?",
      { std::vector<float> (signal.embedding.data (),
                            signal.embedding.data ()
                                + signal.embedding.size ()),
        static_cast<long long> (candidate_limit), *ctx.GetStoredMemoryId (),
        2000LL });
  REQUIRE (std::any_of (
      sql_candidates.begin (), sql_candidates.end (), [] (const auto &row) {
        return AnyToLongLong (row.at ("memory_id")) == 10LL;
      }));
  const auto target_edges = store->Execute (
      "SELECT 1 FROM associations "
      "WHERE source_memory_id = ? AND target_memory_id = ? "
      "  AND edge_type = 'supersedes'",
      { *ctx.GetStoredMemoryId (), 10LL });
  REQUIRE (target_edges.size () == 1);
  cache::Erase (pctx);
}

TEST_CASE ("MemoryStorage current cache mirrors disabled SQL current paths",
           "[operations][memory_storage][cache][current_surface]")
{
  SECTION ("current-surface writes disabled")
    {
      RequireDisabledCurrentSurfaceCacheParity (
          "CORTEXT_DISABLE_CURRENT_MEMORY_SURFACE_WRITES");
    }
  SECTION ("constructive recall disabled")
    {
      RequireDisabledCurrentSurfaceCacheParity (
          "CORTEXT_DISABLE_CONSTRUCTIVE_RECALL");
    }
}

TEST_CASE ("MemoryStorage stores payload in objstore and retrieves it",
           "[operations][memory_storage]")
{
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  const std::string test_text = "Hello, world!";
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  s.timestamp = 99999;
  s.source_id = "objstore-test";
  s.payload = std::vector<unsigned char> (test_text.begin (), test_text.end ());
  s.modality = "text";
  s.mimetype = "text/plain";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cortext::testing::RequireEncoder (cfg);
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;

  // Set up accumulator state required by memory storage
  AccumulatorState acc;
  acc.mu_acc = s.embedding;
  acc.n_signals = 1;
  acc.s_sum = 0.5;
  acc.s_max = 0.5;
  acc.t_start = s.timestamp - 1000;
  {
    SignalRecord rec;
    rec.embedding = s.embedding;
    rec.timestamp = s.timestamp;
    rec.modality = s.modality;
    rec.mime = s.mimetype;
    rec.score = 0.5;
    rec.serial_position = 0;
    rec.payload = *s.payload;
    acc.signals.push_back (std::move (rec));
  }
  pctx.accumulator_states[s.source_id] = std::move (acc);

  OperationContext ctx (s, pctx, cfg, store);
  ctx.SetAccumulatorWriteDecision (true);
  ctx.SetRepresentativeEmbedding (s.embedding);

  MemoryStorage op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  REQUIRE (pctx.accumulator_states.at (s.source_id)
               .signals.at (0)
               .payload.empty ());

  auto stored_id = ctx.GetStoredEmbeddingId ();
  REQUIRE (stored_id.has_value ());

  // Savepoint commits directly, no need to execute buffered writes

  // v2: Retrieve blob_id from memories
  auto idx_rows = store->Execute (
      "SELECT blob_id FROM memories WHERE embedding_id = ?",
      { *stored_id });
  REQUIRE (idx_rows.size () == 1);
  auto blob_id = BlobFromAny (idx_rows[0].at ("blob_id"));
  REQUIRE (!blob_id.empty ());

  // Retrieve payload from objstore
  auto payload_rows
      = store->Execute ("SELECT objstore_get(?1) AS data", { blob_id });
  REQUIRE (payload_rows.size () == 1);
  auto retrieved_blob = BlobFromAny (payload_rows[0].at ("data"));
  std::string retrieved_text (retrieved_blob.begin (), retrieved_blob.end ());
  REQUIRE (retrieved_text == test_text);

  // Verify signals row references a blob and matches payload
  auto sig_rows = store->Execute (
      "SELECT blob_id FROM signals WHERE memory_id = (SELECT memory_id FROM memories WHERE embedding_id = ?)",
      { *stored_id });
  REQUIRE (sig_rows.size () == 1);
  auto sig_blob_id = BlobFromAny (sig_rows[0].at ("blob_id"));
  REQUIRE (!sig_blob_id.empty ());
  auto sig_payload_rows
      = store->Execute ("SELECT objstore_get(?1) AS data", { sig_blob_id });
  REQUIRE (sig_payload_rows.size () == 1);
  auto sig_retrieved_blob = BlobFromAny (sig_payload_rows[0].at ("data"));
  std::string sig_retrieved_text (sig_retrieved_blob.begin (),
                                  sig_retrieved_blob.end ());
  REQUIRE (sig_retrieved_text == test_text);
}
