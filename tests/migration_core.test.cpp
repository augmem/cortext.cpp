// tests/migration_core.test.cpp
#include <catch2/catch_test_macros.hpp>
#include "test_helpers.hpp"
#include <cortext/processor.hpp>
#include <cortext/store/sqlite_store.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/processor/operation_set.hpp>
#include <memory>

using namespace cortext;

TEST_CASE("Migrations apply core tables automatically", "[schema][migration]") {
    auto unique_store = SQLiteStore::Create(":memory:");
    auto store = std::shared_ptr<Store>(std::move(unique_store));

    // No operations, just core processor
    auto ops = std::make_unique<OperationSet>();
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
    REQUIRE(has("cortext_schema_migrations"));
    // embeddings is a virtual table (vec0), check it separately
    auto emb_rows = store->Execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'", {});
    REQUIRE(emb_rows.size() == 1);
}

TEST_CASE("Migrations track version history", "[schema][migration]") {
    auto unique_store = SQLiteStore::Create(":memory:");
    auto store = std::shared_ptr<Store>(std::move(unique_store));

    auto ops = std::make_unique<OperationSet>();
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

TEST_CASE("Migrations are idempotent", "[schema][migration]") {
    auto unique_store = SQLiteStore::Create(":memory:");
    auto store = std::shared_ptr<Store>(std::move(unique_store));

    // First run
    {
        auto ops = std::make_unique<OperationSet>();
        SignalProcessor::Config cfg;
        cortext::testing::RequireEncoder(cfg);
        SignalProcessor p1(cfg, store, std::move(ops));
    }

    // Second run with same store
    {
        auto ops = std::make_unique<OperationSet>();
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
