#include "cortext/operations/label_bank.hpp"

#include "cortext/core/knobs.hpp"
#include "cortext/operations/label_utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <any>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <Eigen/Dense>

#include <nlohmann/json.hpp>

namespace cortext::operations
{

namespace
{

constexpr int kEmbeddingDim = 256;

long long
LastInsertRowId (Transaction &tx)
{
  auto rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
  if (rows.empty () || !rows[0].count ("id"))
    {
      return 0;
    }
  auto val = rows[0].at ("id");
  if (val.type () == typeid (long long))
    {
      return std::any_cast<long long> (val);
    }
  if (val.type () == typeid (int))
    {
      return static_cast<long long> (std::any_cast<int> (val));
    }
  return 0;
}

struct LabelEntry
{
  std::string label;
  std::string key;
  std::vector<float> embedding;
};

Eigen::VectorXf
ToEigenVector (const std::vector<float> &v)
{
  if (v.empty ())
    {
      return Eigen::VectorXf ();
    }
  Eigen::VectorXf out (
      static_cast<Eigen::Index> (v.size ()));
  for (size_t i = 0; i < v.size (); ++i)
    {
      out (static_cast<Eigen::Index> (i)) = v[i];
    }
  return out;
}

std::optional<std::pair<std::vector<LabelEntry>, int>>
LoadLabelBank (const std::string &path, int *skipped)
{
  if (skipped)
    {
      *skipped = 0;
    }
  std::ifstream meta_in (path);
  if (!meta_in.is_open ())
    {
      return std::nullopt;
    }

  std::string labels_file = path;
  int embedding_dim = kEmbeddingDim;

  if (path.size () >= 5
      && path.substr (path.size () - 5) == ".json")
    {
      nlohmann::json meta;
      meta_in >> meta;
      labels_file = meta.value ("labels_file", "");
      embedding_dim = meta.value ("embedding_dim", kEmbeddingDim);
      if (labels_file.empty ())
        {
          return std::nullopt;
        }
      const auto base = std::filesystem::path (path).parent_path ();
      labels_file = (base / labels_file).string ();
    }

  std::ifstream labels_in (labels_file);
  if (!labels_in.is_open ())
    {
      return std::nullopt;
    }

  std::vector<LabelEntry> entries;
  std::string line;
  while (std::getline (labels_in, line))
    {
      if (line.empty ())
        {
          continue;
        }
      nlohmann::json row = nlohmann::json::parse (line, nullptr, false);
      if (row.is_discarded ())
        {
          if (skipped)
            {
              (*skipped)++;
            }
          continue;
        }
      LabelEntry entry;
      entry.label = row.value ("label", "");
      entry.key = row.value ("key", NormalizeLabelKey (entry.label));
      if (entry.label.empty () || entry.key.empty ())
        {
          if (skipped)
            {
              (*skipped)++;
            }
          continue;
        }
      auto emb = row.find ("embedding");
      if (emb == row.end () || !emb->is_array ())
        {
          if (skipped)
            {
              (*skipped)++;
            }
          continue;
        }
      entry.embedding.reserve (emb->size ());
      for (const auto &v : *emb)
        {
          entry.embedding.push_back (v.get<float> ());
        }
      if (static_cast<int> (entry.embedding.size ()) != embedding_dim)
        {
          if (skipped)
            {
              (*skipped)++;
            }
          continue;
        }
      entries.push_back (std::move (entry));
    }

  return std::make_optional (
      std::make_pair (std::move (entries), embedding_dim));
}

} // namespace

void
SeedLabelBank::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  if (p_ctx.label_bank_seeded)
    {
      return;
    }
  p_ctx.label_bank_seeded = true;

  const auto &cfg = context.GetConfig ();
  const std::string &label_bank_path = cfg.label_bank_path;
  if (label_bank_path.empty ())
    {
      return;
    }
  int skipped = 0;
  auto loaded = LoadLabelBank (label_bank_path, &skipped);
  if (!loaded.has_value ())
    {
      telemetry::LogWarn (
          "Label bank skipped: missing or invalid metadata",
          { telemetry::Attribute::String ("component", "label_bank"),
            telemetry::Attribute::String ("path", label_bank_path) });
      return;
    }

  auto &entries = loaded->first;
  if (entries.empty ())
    {
      telemetry::LogWarn (
          "Label bank skipped: no labels loaded",
          { telemetry::Attribute::String ("component", "label_bank"),
            telemetry::Attribute::String ("path", label_bank_path) });
      return;
    }

  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);
  const double salience = core::LabelBankSalience (
      cfg.focus, cfg.sensitivity, cfg.stability);
  int inserted = 0;
  int updated = 0;
  int embeddings_added = 0;

  auto &label_cache = p_ctx.label_embedding_cache;
  label_cache.reserve (label_cache.size () + entries.size ());

  struct ExistingLabel
  {
    long long memory_id = 0;
    long long embedding_id = 0;
  };
  std::unordered_map<std::string, ExistingLabel> existing_labels;
  try
    {
      auto existing_rows = tx.Execute (
          "SELECT memory_id, embedding_id, source_id "
          "FROM memories WHERE kind = 'LABEL'",
          {});
      existing_labels.reserve (existing_rows.size ());
      for (const auto &row : existing_rows)
        {
          auto it_key = row.find ("source_id");
          if (it_key == row.end ()
              || it_key->second.type () != typeid (std::string))
            {
              continue;
            }
          ExistingLabel ex;
          auto it_mem = row.find ("memory_id");
          auto it_emb = row.find ("embedding_id");
          if (it_mem != row.end ())
            {
              if (it_mem->second.type () == typeid (long long))
                {
                  ex.memory_id = std::any_cast<long long> (it_mem->second);
                }
              else if (it_mem->second.type () == typeid (int))
                {
                  ex.memory_id = std::any_cast<int> (it_mem->second);
                }
            }
          if (it_emb != row.end () && it_emb->second.type () != typeid (std::nullptr_t))
            {
              if (it_emb->second.type () == typeid (long long))
                {
                  ex.embedding_id = std::any_cast<long long> (it_emb->second);
                }
              else if (it_emb->second.type () == typeid (int))
                {
                  ex.embedding_id = std::any_cast<int> (it_emb->second);
                }
            }
          existing_labels.emplace (std::any_cast<std::string> (it_key->second),
                                   ex);
        }
    }
  catch (...)
    {
    }

  for (const auto &entry : entries)
    {
      const std::string &label_key = entry.key;
      const std::string &label_text = entry.label;

      long long memory_id = 0;
      long long embedding_id = 0;
      auto it_existing = existing_labels.find (label_key);
      if (it_existing != existing_labels.end ())
        {
          memory_id = it_existing->second.memory_id;
          embedding_id = it_existing->second.embedding_id;
        }

      if (embedding_id == 0)
        {
          tx.Execute ("INSERT INTO embeddings (embedding, created_at) "
                      "VALUES (?, ?)",
                      { entry.embedding, now_ts });
          embedding_id = LastInsertRowId (tx);
          if (embedding_id == 0)
            {
              skipped++;
              continue;
            }
          embeddings_added++;
        }

      if (memory_id == 0)
        {
          tx.Execute (
              "INSERT INTO memories "
              "(embedding_id, source_id, kind, label, start_ts, s_max, created_at) "
              "VALUES (?, ?, 'LABEL', ?, ?, ?, ?)",
              { embedding_id > 0 ? std::any (embedding_id) : std::any (),
                label_key, label_text, now_ts, salience, now_ts });
          memory_id = LastInsertRowId (tx);
          if (memory_id == 0)
            {
              skipped++;
              continue;
            }
          inserted++;
          existing_labels[label_key] = { memory_id, embedding_id };
        }
      else if (embedding_id > 0)
        {
          tx.Execute ("UPDATE memories SET embedding_id = ? WHERE memory_id = ?",
                      { embedding_id, memory_id });
          updated++;
          existing_labels[label_key] = { memory_id, embedding_id };
        }

      label_cache[label_key] = entry.embedding;
      if (memory_id > 0 && embedding_id > 0)
        {
          Eigen::VectorXf emb = ToEigenVector (entry.embedding);
          p_ctx.UpsertSummaryCache (memory_id, embedding_id, emb, false, true);
        }
    }

  telemetry::LogInfo (
      "cortext.label_bank_seed",
      { telemetry::Attribute::String ("path", label_bank_path),
        telemetry::Attribute::Int64 ("labels_loaded",
                                     static_cast<int64_t> (entries.size ())),
        telemetry::Attribute::Int64 ("labels_inserted", inserted),
        telemetry::Attribute::Int64 ("labels_updated", updated),
        telemetry::Attribute::Int64 ("embeddings_added", embeddings_added),
        telemetry::Attribute::Int64 ("labels_skipped", skipped) });
}

} // namespace cortext::operations
