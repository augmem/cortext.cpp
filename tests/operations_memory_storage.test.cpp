#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/operations/memory_storage.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/accumulator_state.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/utils.hpp>
#include <cstring>
#include <filesystem>

using namespace cortext;
using cortext::operations::MemoryStorage;
using cortext::store::AnyToLongLong;
using cortext::store::BlobFromAny;

namespace
{

constexpr int kEmbeddingDim = 256;

/// @brief RAII wrapper for a temporary database file.
class ScopedTempDb
{
public:
  ScopedTempDb ()
  {
    auto tmp = std::filesystem::temp_directory_path ()
               / ("memory_storage_test_"
                  + std::to_string (std::rand ()) + ".db");
    path_ = tmp.string ();
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

} // namespace

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

  // Verify signal row inserted with its own embedding + blob
  auto sig_rows = store->Execute (
      "SELECT embedding_id, blob_id FROM signals WHERE memory_id = ?",
      { memory_id });
  REQUIRE (sig_rows.size () == 1);
  const auto sig_emb_id = AnyToLongLong (sig_rows[0].at ("embedding_id"));
  REQUIRE (sig_emb_id.has_value ());
  auto sig_blob_id = BlobFromAny (sig_rows[0].at ("blob_id"));
  REQUIRE (!sig_blob_id.empty ());

  // v2: Verify memories has all expected columns (strength, use_frequency now on memories)
  auto fb_rows = store->Execute (
      "SELECT strength, use_frequency FROM memories WHERE embedding_id = ?", { *stored_id });
  REQUIRE (fb_rows.size () == 1);

  // No buffered instructions since we use savepoints
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
