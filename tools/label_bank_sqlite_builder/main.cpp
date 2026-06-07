#include "cortext/operations/label_utils.hpp"
#include "cortext/store/extension_loader.hpp"

#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

constexpr int kEmbeddingDim = 256;

void
CheckSqlite (int rc, sqlite3 *db, const std::string &context)
{
  if (rc != SQLITE_OK && rc != SQLITE_DONE && rc != SQLITE_ROW)
    {
      throw std::runtime_error (
          context + ": " + (db ? sqlite3_errmsg (db) : "unknown sqlite error"));
    }
}

uint64_t
Fnv1aUpdate (uint64_t hash, const char *data, std::streamsize size)
{
  constexpr uint64_t kPrime = 1099511628211ULL;
  for (std::streamsize i = 0; i < size; ++i)
    {
      hash ^= static_cast<unsigned char> (data[i]);
      hash *= kPrime;
    }
  return hash;
}

std::string
FileHash (const fs::path &path)
{
  std::ifstream in (path, std::ios::binary);
  if (!in.is_open ())
    {
      throw std::runtime_error ("failed to open " + path.string ());
    }
  uint64_t hash = 1469598103934665603ULL;
  char buf[64 * 1024];
  while (in.good ())
    {
      in.read (buf, sizeof (buf));
      const auto n = in.gcount ();
      if (n > 0)
        {
          hash = Fnv1aUpdate (hash, buf, n);
        }
    }
  return "fnv1a64:" + std::to_string (hash);
}

void
Exec (sqlite3 *db, const std::string &sql)
{
  char *err = nullptr;
  const int rc = sqlite3_exec (db, sql.c_str (), nullptr, nullptr, &err);
  if (rc != SQLITE_OK)
    {
      std::string message = err ? err : "unknown sqlite error";
      sqlite3_free (err);
      throw std::runtime_error (message + " while executing: " + sql);
    }
}

void
BindText (sqlite3_stmt *stmt, int index, const std::string &value)
{
  sqlite3_bind_text (stmt, index, value.c_str (),
                     static_cast<int> (value.size ()), SQLITE_TRANSIENT);
}

void
InsertMeta (sqlite3 *db, const std::string &key, const std::string &value)
{
  sqlite3_stmt *stmt = nullptr;
  CheckSqlite (
      sqlite3_prepare_v2 (
          db,
          "INSERT INTO label_bank_meta(key, value) VALUES (?, ?)",
          -1, &stmt, nullptr),
      db, "prepare meta insert");
  BindText (stmt, 1, key);
  BindText (stmt, 2, value);
  CheckSqlite (sqlite3_step (stmt), db, "insert meta");
  sqlite3_finalize (stmt);
}

} // namespace

int
main (int argc, char **argv)
{
  fs::path metadata_path = "data/label_bank/metadata.json";
  fs::path output_path = "data/label_bank/label_bank.sqlite";
  for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
      auto value = [&] {
        if (i + 1 >= argc)
          {
            throw std::runtime_error ("missing value for " + arg);
          }
        return std::string (argv[++i]);
      };
      if (arg == "--metadata")
        {
          metadata_path = value ();
        }
      else if (arg == "--output")
        {
          output_path = value ();
        }
      else
        {
          throw std::runtime_error ("unknown argument: " + arg);
        }
    }

  nlohmann::json metadata = nlohmann::json::parse (
      std::ifstream (metadata_path));
  const int embedding_dim = metadata.value ("embedding_dim", kEmbeddingDim);
  if (embedding_dim != kEmbeddingDim)
    {
      throw std::runtime_error ("expected 256-dimensional label embeddings");
    }
  const fs::path labels_path
      = metadata_path.parent_path () / metadata.value ("labels_file", "");
  const std::string content_hash = FileHash (labels_path);

  fs::create_directories (output_path.parent_path ());
  const fs::path tmp_path = output_path.string () + ".tmp";
  fs::remove (tmp_path);

  sqlite3 *db = nullptr;
  CheckSqlite (sqlite3_open (tmp_path.string ().c_str (), &db), db,
               "open output");
  cortext::RegisterBuiltInExtensions ();
  cortext::RegisterBuiltInExtensionsOnDb (db);

  try
    {
      Exec (db, "PRAGMA journal_mode=OFF");
      Exec (db, "PRAGMA synchronous=OFF");
      Exec (db,
            "CREATE TABLE label_bank_meta("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)");
      Exec (db,
            "CREATE VIRTUAL TABLE label_bank_vec USING vec0("
            "label_id INTEGER PRIMARY KEY,"
            "embedding float[256],"
            "+key TEXT,"
            "+label TEXT)");

      sqlite3_stmt *insert = nullptr;
      CheckSqlite (
          sqlite3_prepare_v2 (
              db,
              "INSERT INTO label_bank_vec(label_id, embedding, key, label) "
              "VALUES (?, ?, ?, ?)",
              -1, &insert, nullptr),
          db, "prepare label insert");

      Exec (db, "BEGIN");
      std::ifstream labels_in (labels_path);
      if (!labels_in.is_open ())
        {
          throw std::runtime_error ("failed to open " + labels_path.string ());
        }
      std::string line;
      int label_id = 0;
      int skipped = 0;
      while (std::getline (labels_in, line))
        {
          if (line.empty ())
            {
              continue;
            }
          nlohmann::json row = nlohmann::json::parse (line, nullptr, false);
          if (row.is_discarded ())
            {
              ++skipped;
              continue;
            }
          std::string label = row.value ("label", "");
          std::string key = row.value (
              "key", cortext::operations::NormalizeLabelKey (label));
          auto embedding_json = row.find ("embedding");
          if (label.empty () || key.empty () || embedding_json == row.end ()
              || !embedding_json->is_array ()
              || static_cast<int> (embedding_json->size ()) != kEmbeddingDim)
            {
              ++skipped;
              continue;
            }
          std::vector<float> embedding;
          embedding.reserve (kEmbeddingDim);
          for (const auto &v : *embedding_json)
            {
              embedding.push_back (v.get<float> ());
            }
          ++label_id;
          sqlite3_bind_int64 (insert, 1, label_id);
          sqlite3_bind_blob (insert, 2, embedding.data (),
                             static_cast<int> (embedding.size ()
                                               * sizeof (float)),
                             SQLITE_TRANSIENT);
          BindText (insert, 3, key);
          BindText (insert, 4, label);
          CheckSqlite (sqlite3_step (insert), db, "insert label");
          sqlite3_reset (insert);
          sqlite3_clear_bindings (insert);
        }
      sqlite3_finalize (insert);
      Exec (db, "COMMIT");

      InsertMeta (db, "count", std::to_string (label_id));
      InsertMeta (db, "embedding_dim", std::to_string (kEmbeddingDim));
      InsertMeta (db, "content_hash", content_hash);
      InsertMeta (db, "generated_by",
                  "tools/label_bank_sqlite_builder");
      InsertMeta (db, "source_labels_file", labels_path.string ());
      InsertMeta (db, "jsonl_generated_by",
                  metadata.value ("generated_by", ""));
      InsertMeta (db, "jsonl_skipped_count", std::to_string (skipped));
      sqlite3_close (db);
      fs::rename (tmp_path, output_path);

      nlohmann::json out;
      out["output"] = output_path.string ();
      out["labels"] = label_id;
      out["skipped"] = skipped;
      out["embedding_dim"] = kEmbeddingDim;
      out["content_hash"] = content_hash;
      std::cout << out.dump (2) << "\n";
      return 0;
    }
  catch (...)
    {
      sqlite3_exec (db, "ROLLBACK", nullptr, nullptr, nullptr);
      sqlite3_close (db);
      fs::remove (tmp_path);
      throw;
    }
}
