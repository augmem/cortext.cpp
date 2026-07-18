// tests/migration_core.test.cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../src/operations/consolidation_throughput_state_internal.hpp"
#include "test_helpers.hpp"
#include <cortext/processor.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/processor/operation_set.hpp>
#include <any>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace cortext;

namespace
{

class RecordingMigrationStore : public Store
{
public:
    std::vector<std::string> queries;
    int begin_count = 0;

    std::vector<std::map<std::string, std::any> >
    Execute(const std::string& query,
            const std::vector<std::any>& params = {}) override
    {
        (void)params;
        queries.push_back(query);
        return {};
    }

    std::unique_ptr<Transaction> Begin() override
    {
        ++begin_count;
        throw StoreError("ApplyMigrations should not call Store::Begin");
    }

    void Commit() override
    {
        queries.push_back("Store::Commit");
    }

    void Rollback() override
    {
        queries.push_back("Store::Rollback");
    }

    void Close() override
    {
    }
};

} // namespace

namespace cortext::store
{
std::set<int64_t> DebugGetAppliedMigrationIdsForTest (Store &store);
void DebugApplyCoreMigrationsThroughForTest (Store &store, int64_t maximum_id);
}

TEST_CASE("Migrations lock the full pass before reading applied ids",
          "[schema][migration]") {
    RecordingMigrationStore store;

    cortext::store::ApplyMigrations(store);

    REQUIRE(store.begin_count == 0);
    REQUIRE(store.queries.size() > 3);
    REQUIRE(store.queries[0] == "BEGIN IMMEDIATE");
    REQUIRE(store.queries[1].find(
                "CREATE TABLE IF NOT EXISTS cortext_schema_migrations") == 0);
    REQUIRE(store.queries[2] == "SELECT id FROM cortext_schema_migrations");
    REQUIRE(store.queries.back() == "COMMIT");
}

TEST_CASE("Migrations apply core tables automatically", "[schema][migration]") {
    auto unique_store = SQLiteStore::Create(":memory:");
    auto store = std::shared_ptr<Store>(std::move(unique_store));

    // No operations, just core processor
    auto ops = std::make_unique<DynamicOperationSet>();
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder(cfg);
    SignalProcessor processor(cfg, store, std::move(ops));

    auto rows = store->Execute(
        "SELECT name FROM sqlite_master WHERE type='table'", {});
    
    auto has = [&](const std::string& name) {
        for(const auto& r : rows) {
            auto it = r.find("name");
            if(it != r.end() && std::any_cast<std::string>(it->second) == name) return true;
        }
        return false;
    };

    // Core tables from migration 0 (v2 schema)
    REQUIRE(has("memories"));
    REQUIRE(has("state"));        // v2 unified state
    REQUIRE(has("accumulators")); // v2 accumulator state
    REQUIRE(has("signals"));      // v2 signals table
    REQUIRE(has("episodes"));     // v2 episodes table
    REQUIRE(has("memory_reconstructions"));
    REQUIRE(has("meta_learning_coeffs"));
    REQUIRE(has("cortext_schema_migrations"));
    // embeddings is a virtual table (vec0), check it separately
    auto emb_rows = store->Execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'", {});
    REQUIRE(emb_rows.size() == 1);
}

TEST_CASE("Migrations track version history", "[schema][migration]") {
    auto unique_store = SQLiteStore::Create(":memory:");
    auto store = std::shared_ptr<Store>(std::move(unique_store));

    auto ops = std::make_unique<DynamicOperationSet>();
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder(cfg);
    SignalProcessor processor(cfg, store, std::move(ops));

    auto rows = store->Execute("SELECT id, description FROM cortext_schema_migrations ORDER BY id");
    REQUIRE(rows.size() >= 1);

    // Core migration is ID 0 (clean slate schema). Note: SQLiteStore returns integers as long long.
    auto id_val = rows[0].at("id");
    long long id = -1;
    if(id_val.type() == typeid(long long)) {
        id = std::any_cast<long long>(id_val);
    } else if(id_val.type() == typeid(int)) {
        id = std::any_cast<int>(id_val);
    }

    REQUIRE(id == 0);
}

TEST_CASE("Migration 25 adds persisted consolidation throughput state",
          "[schema][migration][consolidation]") {
    auto store = SQLiteStore::Create(":memory:");
    cortext::store::ApplyMigrations(*store);

    const auto ids = cortext::store::DebugGetAppliedMigrationIdsForTest(*store);
    REQUIRE(ids.count(25) == 1);

    auto columns = store->Execute("PRAGMA table_info(state)");
    std::set<std::string> names;
    for (const auto& row : columns) {
        names.insert(std::any_cast<std::string>(row.at("name")));
    }
    REQUIRE(names.count("consolidation_rate_floor") == 1);
    REQUIRE(names.count("consolidation_rate_peak") == 1);

    store->Execute(
        "INSERT INTO state(id, consolidation_rate_floor, "
        "consolidation_rate_peak) VALUES(1, 2.5, 9.5)");
    const auto rows = store->Execute(
        "SELECT consolidation_rate_floor, consolidation_rate_peak "
        "FROM state WHERE id = 1");
    REQUIRE(std::any_cast<double>(rows[0].at("consolidation_rate_floor"))
            == 2.5);
    REQUIRE(std::any_cast<double>(rows[0].at("consolidation_rate_peak"))
            == 9.5);
}

TEST_CASE("Migration 26 adds consolidation throughput initialization",
          "[schema][migration][consolidation]") {
    auto store = SQLiteStore::Create(":memory:");
    cortext::store::ApplyMigrations(*store);

    const auto ids = cortext::store::DebugGetAppliedMigrationIdsForTest(*store);
    REQUIRE(ids.count(26) == 1);

    auto columns = store->Execute("PRAGMA table_info(state)");
    std::set<std::string> names;
    for (const auto& row : columns) {
        names.insert(std::any_cast<std::string>(row.at("name")));
    }
    REQUIRE(names.count("consolidation_rate_initialized") == 1);

    store->Execute(
        "INSERT INTO state(id, rate_ticks, consolidation_rate_floor, "
        "consolidation_rate_peak, consolidation_rate_initialized) "
        "VALUES(1, 1, 0.0, 0.0, 1)");
    const auto rows = store->Execute(
        "SELECT consolidation_rate_initialized FROM state WHERE id = 1");
    REQUIRE(std::any_cast<long long>(
                rows[0].at("consolidation_rate_initialized")) == 1);
}

TEST_CASE("Migration 26 leaves pre-25 throughput range uninitialized",
          "[schema][migration][consolidation]") {
    auto store = SQLiteStore::Create(":memory:");
    cortext::store::DebugApplyCoreMigrationsThroughForTest(*store, 24);
    store->Execute(
        "INSERT INTO state(id, signals_processed, m_rate, rate_ticks) "
        "VALUES(1, 1, 70.5, 1)");

    cortext::store::ApplyMigrations(*store);

    const auto rows = store->Execute(
        "SELECT m_rate, consolidation_rate_floor, "
        "consolidation_rate_peak, consolidation_rate_initialized "
        "FROM state WHERE id = 1");
    REQUIRE(rows.size() == 1);
    REQUIRE(std::any_cast<double>(rows[0].at("consolidation_rate_floor"))
            == 0.0);
    REQUIRE(std::any_cast<double>(rows[0].at("consolidation_rate_peak"))
            == 0.0);
    REQUIRE(std::any_cast<long long>(
                rows[0].at("consolidation_rate_initialized")) == 0);

    ProcessorContext context;
    namespace throughput
        = operations::consolidation_throughput_state_internal;
    throughput::Reset(context, {
        std::any_cast<double>(rows[0].at("consolidation_rate_floor")),
        std::any_cast<double>(rows[0].at("consolidation_rate_peak")),
        std::any_cast<long long>(
            rows[0].at("consolidation_rate_initialized")) != 0,
    });
    const double first_new_rate =
        std::any_cast<double>(rows[0].at("m_rate"));
    throughput::Observe(context, first_new_rate, 0.5, 0.5, 0.5);
    const auto observed = throughput::Find(context);
    REQUIRE(observed.initialized);
    REQUIRE(observed.floor == first_new_rate);
    REQUIRE(observed.peak == first_new_rate);
    throughput::Erase(context);
}

TEST_CASE("Migration 27 adds an armed consolidation throughput latch",
          "[schema][migration][consolidation]") {
    auto store = SQLiteStore::Create(":memory:");
    cortext::store::ApplyMigrations(*store);

    const auto ids = cortext::store::DebugGetAppliedMigrationIdsForTest(*store);
    REQUIRE(ids.count(27) == 1);
    const auto columns = store->Execute("PRAGMA table_info(state)");
    std::set<std::string> names;
    for (const auto& row : columns) {
        names.insert(std::any_cast<std::string>(row.at("name")));
    }
    REQUIRE(names.count("consolidation_rate_armed") == 1);

    store->Execute("INSERT INTO state(id) VALUES(1)");
    const auto rows = store->Execute(
        "SELECT consolidation_rate_armed FROM state WHERE id = 1");
    REQUIRE(std::any_cast<long long>(rows[0].at("consolidation_rate_armed"))
            == 1);
}

TEST_CASE("Migration 28 seeds exact lazy RIF recovery state",
          "[schema][migration][rif]") {
    auto store = SQLiteStore::Create(":memory:");
    cortext::store::DebugApplyCoreMigrationsThroughForTest(*store, 27);
    store->Execute(
        "INSERT INTO memories(memory_id, source_id, kind, start_ts, "
        "n_signals, modality, strength, suppression, suppression_ts, "
        "created_at) VALUES(1, 'source-a', 'LONG_TERM', 100, 1, "
        "'text', 0.6, 0.4, 100, 100)");

    cortext::store::ApplyMigrations(*store);

    const auto ids = cortext::store::DebugGetAppliedMigrationIdsForTest(*store);
    REQUIRE(ids.count(28) == 1);
    auto objects = store->Execute(
        "SELECT type, name FROM sqlite_master WHERE name IN ("
        "'rif_recovery_clock', 'rif_generation_resets', "
        "'rif_active_state', 'rif_effective_memories', "
        "'idx_rif_active_due', 'rif_memories_insert_active', "
        "'rif_memories_update_active', 'rif_memories_update_inactive', "
        "'rif_memories_update_strength')");
    REQUIRE(objects.size() == 4);
    REQUIRE(store->Execute(
        "SELECT name FROM sqlite_temp_master "
        "WHERE type = 'view' AND name = 'rif_effective_memories'").size()
        == 1);

    auto clock = store->Execute(
        "SELECT generation, log_factor, last_ts FROM rif_recovery_clock");
    REQUIRE(clock.size() == 1);
    REQUIRE(std::any_cast<long long>(clock[0].at("generation")) == 1);
    REQUIRE(std::any_cast<double>(clock[0].at("log_factor")) == 0.0);
    REQUIRE(std::any_cast<long long>(clock[0].at("last_ts")) == 100);

    auto active = store->Execute(
        "SELECT generation, anchor_suppression, recovery_total, "
        "anchor_log_factor, expires_log_factor "
        "FROM rif_active_state WHERE memory_id = 1");
    REQUIRE(active.size() == 1);
    REQUIRE(std::any_cast<long long>(active[0].at("generation")) == 1);
    REQUIRE(std::any_cast<double>(active[0].at("anchor_suppression"))
            == Catch::Approx(0.4));
    REQUIRE(std::any_cast<double>(active[0].at("recovery_total"))
            == Catch::Approx(1.0));
    REQUIRE(std::any_cast<double>(active[0].at("anchor_log_factor"))
            == 0.0);
    REQUIRE(std::any_cast<double>(active[0].at("expires_log_factor")) < 0.0);

    auto effective = store->Execute(
        "SELECT strength, suppression, suppression_ts "
        "FROM rif_effective_memories WHERE memory_id = 1");
    REQUIRE(effective.size() == 1);
    REQUIRE(std::any_cast<double>(effective[0].at("strength"))
            == Catch::Approx(0.6));
    REQUIRE(std::any_cast<double>(effective[0].at("suppression"))
            == Catch::Approx(0.4));
    REQUIRE(std::any_cast<long long>(effective[0].at("suppression_ts"))
            == 100);

    const auto persistent_extras = store->Execute(
        "SELECT name FROM sqlite_master "
        "WHERE name LIKE 'rif_%' "
        "  AND name NOT IN ('rif_recovery_clock', 'rif_generation_resets', "
        "                   'rif_active_state', 'idx_rif_active_due')");
    REQUIRE(persistent_extras.empty());
}

TEST_CASE("Migration 28 resumes an exact partial schema and is idempotent",
          "[schema][migration][rif][partial]") {
    auto store = SQLiteStore::Create(":memory:");
    cortext::store::DebugApplyCoreMigrationsThroughForTest(*store, 27);
    store->Execute(
        "CREATE TABLE rif_recovery_clock("
        "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
        "generation INTEGER NOT NULL CHECK(generation > 0), "
        "log_factor REAL NOT NULL, last_ts INTEGER NOT NULL)");
    store->Execute(
        "INSERT INTO memories(memory_id, source_id, kind, start_ts, "
        "n_signals, modality, strength, suppression, suppression_ts, "
        "created_at) VALUES(1, 'partial', 'LONG_TERM', 100, 1, "
        "'image', 0.75, 0.25, 100, 100)");

    cortext::store::ApplyMigrations(*store);
    const auto first_ids
        = cortext::store::DebugGetAppliedMigrationIdsForTest(*store);
    REQUIRE(first_ids.count(28) == 1);
    const auto first = store->Execute(
        "SELECT generation, log_factor, last_ts FROM rif_recovery_clock");
    REQUIRE(first.size() == 1);
    const auto first_active = store->Execute(
        "SELECT memory_id FROM rif_active_state ORDER BY memory_id");
    REQUIRE(first_active.size() == 1);

    cortext::store::ApplyMigrations(*store);
    const auto second = store->Execute(
        "SELECT generation, log_factor, last_ts FROM rif_recovery_clock");
    const auto second_active = store->Execute(
        "SELECT memory_id FROM rif_active_state ORDER BY memory_id");
    REQUIRE(second.size() == 1);
    REQUIRE(second_active.size() == 1);
    REQUIRE(std::any_cast<long long>(second[0].at("generation"))
            == std::any_cast<long long>(first[0].at("generation")));
    REQUIRE(std::any_cast<double>(second[0].at("log_factor"))
            == std::any_cast<double>(first[0].at("log_factor")));
    REQUIRE(std::any_cast<long long>(second[0].at("last_ts"))
            == std::any_cast<long long>(first[0].at("last_ts")));
    REQUIRE(std::any_cast<long long>(second_active[0].at("memory_id"))
            == std::any_cast<long long>(first_active[0].at("memory_id")));
}

TEST_CASE("Migration 28 fails atomically on an incompatible partial schema",
          "[schema][migration][rif][partial][rollback]") {
    auto store = SQLiteStore::Create(":memory:");
    cortext::store::DebugApplyCoreMigrationsThroughForTest(*store, 27);
    store->Execute("CREATE TABLE rif_recovery_clock(singleton INTEGER)");

    REQUIRE_THROWS(cortext::store::ApplyMigrations(*store));
    const auto ids
        = cortext::store::DebugGetAppliedMigrationIdsForTest(*store);
    REQUIRE(ids.count(28) == 0);
    REQUIRE(store->Execute(
        "SELECT name FROM sqlite_master WHERE name = 'rif_active_state'")
        .empty());
    REQUIRE(store->Execute(
        "SELECT name FROM sqlite_master WHERE name = 'rif_effective_memories'")
        .empty());
}

TEST_CASE("Migrations preserve 64-bit applied migration ids",
          "[schema][migration]") {
    auto store = SQLiteStore::Create(":memory:");
    cortext::store::ApplyMigrations(*store);

    const int64_t large_id = 3000000000LL;
    store->Execute(
        "INSERT INTO cortext_schema_migrations (id, description, applied_at) "
        "VALUES (?, ?, 0)",
        {large_id, std::string("synthetic large id")});

    const auto ids = cortext::store::DebugGetAppliedMigrationIdsForTest(*store);
    REQUIRE(ids.count(large_id) == 1);
}

TEST_CASE("Migrations are idempotent", "[schema][migration]") {
    auto unique_store = SQLiteStore::Create(":memory:");
    auto store = std::shared_ptr<Store>(std::move(unique_store));

    // First run
    {
        auto ops = std::make_unique<DynamicOperationSet>();
        SignalProcessor::Config cfg;
        cortext::testing::RequireEncoder(cfg);
        SignalProcessor p1(cfg, store, std::move(ops));
    }

    // Second run with same store
    {
        auto ops = std::make_unique<DynamicOperationSet>();
        SignalProcessor::Config cfg;
        cortext::testing::RequireEncoder(cfg);
        SignalProcessor p2(cfg, store, std::move(ops));
    }

    // Should not throw and tables should still exist
    auto count = store->Execute("SELECT COUNT(*) as c FROM cortext_schema_migrations");
    long long c = 0;
    auto c_val = count[0].at("c");
    if(c_val.type() == typeid(long long)) c = std::any_cast<long long>(c_val);
    else if(c_val.type() == typeid(int)) c = std::any_cast<int>(c_val);
    
    REQUIRE(c >= 1);
}

TEST_CASE("Migrations create graph retrieval lookup indexes", "[schema][migration]") {
    auto unique_store = SQLiteStore::Create(":memory:");
    auto store = std::shared_ptr<Store>(std::move(unique_store));

    auto ops = std::make_unique<DynamicOperationSet>();
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder(cfg);
    SignalProcessor processor(cfg, store, std::move(ops));

    auto rows = store->Execute(
        "SELECT name FROM sqlite_master WHERE type='index'",
        {});
    auto has_index = [&] (const std::string& name) {
        for(const auto& r : rows) {
            auto it = r.find("name");
            if(it != r.end() && std::any_cast<std::string>(it->second) == name) return true;
        }
        return false;
    };

    REQUIRE(has_index("idx_memories_embedding"));
    REQUIRE(has_index("idx_signals_embedding"));
    REQUIRE(has_index("idx_signals_source_ts_serial"));
    REQUIRE(has_index("idx_memories_kind_source"));
    REQUIRE(has_index("idx_memories_source_start"));
    REQUIRE(has_index("idx_memories_last_access"));
    REQUIRE(has_index("idx_memories_label_created"));
    REQUIRE(has_index("idx_memories_created_desc"));
    REQUIRE(has_index("idx_memories_ltm_strength_created"));
    REQUIRE(has_index("idx_memories_pre_activation_embedding_active"));
    REQUIRE_FALSE(has_index("idx_memories_strength"));
    REQUIRE_FALSE(has_index("idx_memories_working"));
    REQUIRE(has_index("idx_memories_working_active"));
    REQUIRE(has_index("idx_memories_working_closed_end"));
    REQUIRE(has_index("idx_associations_edge_source_target"));
    REQUIRE(has_index("idx_associations_edge_target_source"));
    REQUIRE(has_index("idx_associations_source_weight"));
    REQUIRE(has_index("idx_associations_target_weight"));

    auto last_access_sql_rows = store->Execute(
        "SELECT sql FROM sqlite_master WHERE type='index' "
        "AND name='idx_memories_last_access'",
        {});
    REQUIRE(last_access_sql_rows.size() == 1);
    auto last_access_sql = std::any_cast<std::string>(
        last_access_sql_rows[0].at("sql"));
    REQUIRE(last_access_sql.find("kind != 'WORKING'") != std::string::npos);

    auto recent_view_rows = store->Execute(
        "SELECT sql FROM sqlite_master WHERE type='view' "
        "AND name='recent_retrievals'",
        {});
    REQUIRE(recent_view_rows.size() == 1);
    auto recent_view_sql = std::any_cast<std::string>(
        recent_view_rows[0].at("sql"));
    REQUIRE(recent_view_sql.find("kind != 'WORKING'") != std::string::npos);

    auto preactivation_sql_rows = store->Execute(
        "SELECT sql FROM sqlite_master WHERE type='index' "
        "AND name='idx_memories_pre_activation_embedding_active'",
        {});
    REQUIRE(preactivation_sql_rows.size() == 1);
    auto preactivation_sql = std::any_cast<std::string>(
        preactivation_sql_rows[0].at("sql"));
    REQUIRE(preactivation_sql.find("embedding_id") != std::string::npos);
    REQUIRE(preactivation_sql.find("pre_activation > 0.0") != std::string::npos);
}

TEST_CASE("Migrations normalize role-derived source metadata",
          "[schema][migration]") {
    auto unique_store = SQLiteStore::Create(":memory:");
    auto store = std::shared_ptr<Store>(std::move(unique_store));

    auto ops = std::make_unique<DynamicOperationSet>();
    SignalProcessor::Config cfg;
    cortext::testing::RequireEncoder(cfg);
    SignalProcessor processor(cfg, store, std::move(ops));

    std::vector<float> embedding(256, 0.0f);
    embedding[0] = 1.0f;
    store->Execute(
        "INSERT INTO embeddings (embedding_id, embedding, created_at) "
        "VALUES (?, ?, ?)",
        {100LL, embedding, 1000LL});
    store->Execute(
        "INSERT INTO memories (memory_id, embedding_id, source_id, kind, "
        "label, start_ts, n_signals, modality, source_origin, "
        "source_reliability, created_at) "
        "VALUES (?, ?, ?, 'ASSOCIATION', ?, 1000, 1, 'text', ?, ?, 1000)",
        {100LL, 100LL, std::string("association_1000_0"),
         std::string("association text"), std::string("assistant"), 0.6});

    store->Execute("DELETE FROM cortext_schema_migrations WHERE id = 8");
    cortext::store::ApplyMigrations(*store);

    auto rows = store->Execute(
        "SELECT kind, source_origin, source_reliability "
        "FROM memories WHERE memory_id = ?",
        {100LL});
    REQUIRE(rows.size() == 1);
    REQUIRE(std::any_cast<std::string>(rows[0].at("kind")) == "ASSOCIATION");
    REQUIRE(std::any_cast<std::string>(rows[0].at("source_origin"))
            == "source");
    REQUIRE(std::any_cast<double>(rows[0].at("source_reliability"))
            == 0.7);
}
