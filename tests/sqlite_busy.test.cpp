#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <sqlite3.h>
#include <string>
#include <thread>

#include <cortext/cortext.hpp>
#include <cortext/signal.hpp>
#include <cortext/store/sqlite_store.hpp>

// A store's busy_timeout must cover every lock its transactions take. A
// deferred BEGIN reads first and takes the WAL write lock on its first write;
// SQLite does not run the busy handler for that read-to-write upgrade, so a
// write lock held for a moment by any other connection (another process, a
// backup or replication tool, the background checkpointer's wal-index
// recovery) failed the signal at once with "database is locked". Root
// transactions now BEGIN IMMEDIATE, where the busy handler applies.

namespace
{

struct BusyTempDb
{
  BusyTempDb ()
  {
    const auto stamp = std::chrono::high_resolution_clock::now ()
                           .time_since_epoch ()
                           .count ();
    path = (std::filesystem::temp_directory_path ()
            / ("cortext_busy_" + std::to_string (stamp) + ".db"))
               .string ();
  }
  ~BusyTempDb ()
  {
    std::error_code ec;
    for (const char *suffix : { "", "-wal", "-shm" })
      std::filesystem::remove (path + suffix, ec);
  }
  std::string path;
};

// Another connection that holds the database's write lock (BEGIN IMMEDIATE,
// no changes) for `hold` and then releases it.
class WriteLockHolder
{
public:
  explicit WriteLockHolder (const std::string &path)
  {
    REQUIRE (sqlite3_open (path.c_str (), &db_) == SQLITE_OK);
    sqlite3_busy_timeout (db_, 5000);
  }
  ~WriteLockHolder ()
  {
    if (thread_.joinable ())
      thread_.join ();
    sqlite3_close (db_);
  }

  // Returns once the lock is held; the lock is released `hold` later.
  void
  HoldFor (std::chrono::milliseconds hold)
  {
    if (thread_.joinable ())
      thread_.join ();
    held_ = false;
    thread_ = std::thread ([this, hold] {
      const int rc = sqlite3_exec (db_, "BEGIN IMMEDIATE", nullptr, nullptr,
                                   nullptr);
      {
        std::lock_guard<std::mutex> lock (mutex_);
        held_ = true;
        begin_rc_ = rc;
      }
      cv_.notify_all ();
      std::this_thread::sleep_for (hold);
      sqlite3_exec (db_, "COMMIT", nullptr, nullptr, nullptr);
    });
    std::unique_lock<std::mutex> lock (mutex_);
    cv_.wait (lock, [this] { return held_; });
    REQUIRE (begin_rc_ == SQLITE_OK);
  }

  void
  Join ()
  {
    if (thread_.joinable ())
      thread_.join ();
  }

private:
  sqlite3 *db_ = nullptr;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool held_ = false;
  int begin_rc_ = SQLITE_OK;
};

double
MillisSince (std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double, std::milli> (
             std::chrono::steady_clock::now () - start)
      .count ();
}

} // namespace

TEST_CASE ("A transaction that reads before it writes waits out another "
           "connection's write lock",
           "[store][wal][sqlite_busy]")
{
  BusyTempDb db;
  auto store = cortext::SQLiteStore::Create (db.path);
  store->Execute ("CREATE TABLE rows (id INTEGER PRIMARY KEY, value TEXT)");
  store->Execute ("INSERT INTO rows (value) VALUES ('seed')");

  WriteLockHolder other (db.path);
  other.HoldFor (std::chrono::milliseconds (250));

  const auto start = std::chrono::steady_clock::now ();
  auto tx = store->Begin ();
  const auto rows = tx->Execute ("SELECT COUNT(*) AS n FROM rows");
  REQUIRE (rows.size () == 1);
  REQUIRE_NOTHROW (
      tx->Execute ("INSERT INTO rows (value) VALUES ('after read')"));
  REQUIRE_NOTHROW (tx->Commit ());
  other.Join ();
  // It waited for the lock instead of failing (busy_timeout is 5 s).
  REQUIRE (MillisSince (start) >= 150.0);

  const auto count = store->Execute ("SELECT COUNT(*) AS n FROM rows");
  REQUIRE (std::any_cast<long long> (count[0].at ("n")) == 2);
}

TEST_CASE ("A nested savepoint that writes after reading waits out another "
           "connection's write lock",
           "[store][wal][sqlite_busy]")
{
  BusyTempDb db;
  auto store = cortext::SQLiteStore::Create (db.path);
  store->Execute ("CREATE TABLE rows (id INTEGER PRIMARY KEY, value TEXT)");

  WriteLockHolder other (db.path);
  other.HoldFor (std::chrono::milliseconds (250));

  auto root = store->Begin ();
  (void)root->Execute ("SELECT COUNT(*) AS n FROM rows");
  auto nested = root->Begin ();
  REQUIRE_NOTHROW (nested->Execute ("INSERT INTO rows (value) VALUES ('n')"));
  REQUIRE_NOTHROW (nested->Commit ());
  REQUIRE_NOTHROW (root->Commit ());
  other.Join ();
  const auto count = store->Execute ("SELECT COUNT(*) AS n FROM rows");
  REQUIRE (std::any_cast<long long> (count[0].at ("n")) == 1);
}

TEST_CASE ("Engine signals wait out another connection's write lock",
           "[sqlite_busy][engine]")
{
  BusyTempDb db;
  cortext::Cortext::Config cfg;
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (db.path.c_str ()));
  auto ctx = cortext::Cortext::Create (cfg, store);
  std::uint64_t t = 1000;
  // Warm the engine so the next signals read before they write.
  for (int i = 0; i < 6; ++i)
    (void)ctx->ProcessTextAt ("The blue box holds item " + std::to_string (i)
                                  + ".",
                              "chat/main", t += 1000,
                              cortext::Retention::Natural);
  ctx->Flush ();

  WriteLockHolder other (db.path);
  for (const auto retention :
       { cortext::Retention::Natural, cortext::Retention::Durable,
         cortext::Retention::Ephemeral })
    {
      INFO ("retention " << static_cast<int> (retention));
      other.HoldFor (std::chrono::milliseconds (200));
      REQUIRE_NOTHROW (ctx->ProcessTextAt ("Where is item 3 kept?",
                                           "chat/main", t += 1000,
                                           retention));
      other.Join ();
    }
  other.HoldFor (std::chrono::milliseconds (200));
  REQUIRE_NOTHROW (ctx->Flush ());
  other.Join ();
}

// Soak: ingest with interleaved Ephemeral queries, Flush and frequent
// background checkpoints while another connection takes the write lock for a
// moment every few signals. Hidden: ~1 min.
TEST_CASE ("Engine ingest with interleaved Ephemeral queries never reports "
           "a locked database",
           "[.][sqlite_busy][engine_stress]")
{
  cortext::testing::ScopedEnvVar wal_autocheckpoint (
      "CORTEXT_SQLITE_WAL_AUTOCHECKPOINT", "8");
  BusyTempDb db;
  cortext::Cortext::Config cfg;
  auto store = std::shared_ptr<cortext::Store> (
      cortext::SQLiteStore::Create (db.path.c_str ()));
  auto ctx = cortext::Cortext::Create (cfg, store);
  WriteLockHolder other (db.path);
  std::uint64_t t = 1000;
  const int n = 600;
  for (int i = 0; i < n; ++i)
    {
      INFO ("signal " << i);
      if (i % 7 == 3)
        other.HoldFor (std::chrono::milliseconds (5));
      const auto retention = (i % 3 == 0) ? cortext::Retention::Durable
                                          : cortext::Retention::Natural;
      REQUIRE_NOTHROW (ctx->ProcessTextAt (
          "Fact " + std::to_string (i) + ": item "
              + std::to_string (i * 7 % 97) + " sits in drawer "
              + std::to_string (i % 13) + ".",
          "chat/main", t += 1000, retention));
      if (i % 4 == 0)
        for (int q = 0; q < 3; ++q)
          {
            if (q == 1)
              other.HoldFor (std::chrono::milliseconds (5));
            REQUIRE_NOTHROW (ctx->ProcessTextAt (
                "where is item " + std::to_string ((i + q) % 97),
                "eval/query", t, cortext::Retention::Ephemeral));
          }
      if (i % 25 == 0)
        REQUIRE_NOTHROW (ctx->Flush ());
    }
  other.Join ();
}
