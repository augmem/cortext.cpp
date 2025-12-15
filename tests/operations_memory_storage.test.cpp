#include <catch2/catch_test_macros.hpp>
#include <cortext/operations/memory_storage.hpp>
#include <cortext/processor.hpp>
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

    // Create required schema tables
    store_->Execute ("CREATE VIRTUAL TABLE IF NOT EXISTS objstore USING "
                     "objstore()",
                     {});
    store_->Execute ("CREATE TABLE IF NOT EXISTS embeddings ("
                     "  embedding_id INTEGER PRIMARY KEY,"
                     "  embedding BLOB"
                     ")",
                     {});
    store_->Execute ("CREATE TABLE IF NOT EXISTS memories ("
                     "  embedding_id INTEGER PRIMARY KEY,"
                     "  modality TEXT,"
                     "  mime TEXT,"
                     "  source_id TEXT,"
                     "  timestamp INTEGER,"
                     "  width INTEGER DEFAULT 0,"
                     "  height INTEGER DEFAULT 0,"
                     "  channels INTEGER DEFAULT 0,"
                     "  sample_rate INTEGER DEFAULT 0,"
                     "  num_samples INTEGER DEFAULT 0,"
                     "  blob_id BLOB"
                     ")",
                     {});
    store_->Execute ("CREATE TABLE IF NOT EXISTS embeddings_meta ("
                     "  embedding_id INTEGER PRIMARY KEY,"
                     "  strength REAL DEFAULT 1.0,"
                     "  contextual_gain REAL DEFAULT 0.0,"
                     "  use_frequency REAL DEFAULT 0.0,"
                     "  lability_state REAL DEFAULT 0.0,"
                     "  created_at INTEGER"
                     ")",
                     {});
    // vec0 virtual table for KNN search (using float[3] for test embeddings)
    store_->Execute ("CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings "
                     "USING vec0(embedding_id INTEGER PRIMARY KEY, "
                     "embedding float[3])",
                     {});
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
  s.embedding = Eigen::VectorXf::Ones (3);
  s.timestamp = 12345;
  s.source_id = "test-source";
  s.payload = std::vector<unsigned char>{ 'h', 'e', 'l', 'l', 'o' };
  s.modality = "text";
  s.mimetype = "text/plain";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  OperationContext ctx (s, pctx, cfg, buf, store);
  ctx.SetWriteDecision (true);

  MemoryStorage op;
  op.Execute (ctx);

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

  // Verify embeddings_meta was inserted
  auto fb_rows = store->Execute (
      "SELECT * FROM embeddings_meta WHERE embedding_id = ?", { *stored_id });
  REQUIRE (fb_rows.size () == 1);

  // No buffered instructions since we use savepoints
  REQUIRE (buf.empty ());
}

TEST_CASE ("MemoryStorage discards when write_decision is false",
           "[operations][memory_storage]")
{
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  Signal s;
  s.embedding = Eigen::VectorXf::Ones (3);
  s.timestamp = 12345;
  s.source_id = "test-source";
  s.payload = std::vector<unsigned char>{ 'h', 'e', 'l', 'l', 'o' };
  s.modality = "text";
  s.mimetype = "text/plain";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  OperationContext ctx (s, pctx, cfg, buf, store);
  ctx.SetWriteDecision (false); // Rejected

  MemoryStorage op;
  op.Execute (ctx);

  // Verify stored_embedding_id is NOT set
  auto stored_id = ctx.GetStoredEmbeddingId ();
  REQUIRE_FALSE (stored_id.has_value ());

  // Verify no buffered instructions
  REQUIRE (buf.empty ());

  // Verify no embeddings were inserted
  auto emb_rows = store->Execute ("SELECT COUNT(*) AS cnt FROM embeddings", {});
  REQUIRE (emb_rows.size () == 1);
  auto cnt = AnyToLongLong (emb_rows[0].at ("cnt"));
  REQUIRE (cnt.has_value ());
  REQUIRE (*cnt == 0);
}

TEST_CASE ("MemoryStorage does nothing when no payload",
           "[operations][memory_storage]")
{
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  Signal s;
  s.embedding = Eigen::VectorXf::Ones (3);
  s.timestamp = 12345;
  s.source_id = "test-source";
  // No payload set
  s.modality = "text";
  s.mimetype = "text/plain";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  OperationContext ctx (s, pctx, cfg, buf, store);
  ctx.SetWriteDecision (true);

  MemoryStorage op;
  op.Execute (ctx);

  // Verify stored_embedding_id is NOT set
  auto stored_id = ctx.GetStoredEmbeddingId ();
  REQUIRE_FALSE (stored_id.has_value ());

  // Verify no buffered instructions
  REQUIRE (buf.empty ());
}

TEST_CASE ("MemoryStorage stores payload in objstore and retrieves it",
           "[operations][memory_storage]")
{
  ScopedTempDb db;
  Store *store = db.get ();
  REQUIRE (store != nullptr);

  const std::string test_text = "Hello, world!";
  Signal s;
  s.embedding = Eigen::VectorXf::Ones (3);
  s.timestamp = 99999;
  s.source_id = "objstore-test";
  s.payload = std::vector<unsigned char> (test_text.begin (), test_text.end ());
  s.modality = "text";
  s.mimetype = "text/plain";

  ProcessorContext pctx;
  SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 0.5;
  std::vector<BufferedWriteInstruction> buf;

  OperationContext ctx (s, pctx, cfg, buf, store);
  ctx.SetWriteDecision (true);

  MemoryStorage op;
  op.Execute (ctx);

  auto stored_id = ctx.GetStoredEmbeddingId ();
  REQUIRE (stored_id.has_value ());

  // Savepoint commits directly, no need to execute buffered writes
  REQUIRE (buf.empty ());

  // Retrieve blob_id from memories
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
}
