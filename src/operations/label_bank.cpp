#include "cortext/operations/label_bank.hpp"

#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <algorithm>
#include <any>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace cortext::operations
{

namespace
{

constexpr const char *kLabelBankSchema = "cortext_label_bank";

/// Distinguishes a wrong-embedding-space bank (fatal) from ordinary attach
/// failures (skippable) in LoadLabelBank::Execute's error handling.
struct LabelBankPinMismatch : std::runtime_error
{
  using std::runtime_error::runtime_error;
};

std::filesystem::path
ResolveLabelBankSqlite (const std::string &path)
{
  std::filesystem::path requested (path);
  if (requested.extension () == ".sqlite" || requested.extension () == ".db")
    {
      return requested;
    }
  if (requested.filename () == "metadata.json" || requested.extension () == ".json")
    {
      return requested.parent_path () / "label_bank.sqlite";
    }
  return requested;
}

std::string
ToAttachUri (const std::filesystem::path &path)
{
  std::filesystem::path absolute = std::filesystem::absolute (path);
  std::string uri = "file:" + absolute.string ();
  std::replace (uri.begin (), uri.end (), ' ', '+');
  uri += "?mode=ro&immutable=1";
  return uri;
}

bool
IsAttached (Transaction &tx)
{
  auto rows = tx.Execute ("PRAGMA database_list", {});
  for (const auto &row : rows)
    {
      auto it = row.find ("name");
      if (it != row.end () && it->second.type () == typeid (std::string)
          && std::any_cast<std::string> (it->second) == kLabelBankSchema)
        {
          return true;
        }
    }
  return false;
}

long long
ReadLabelCount (Transaction &tx)
{
  try
    {
      auto rows = tx.Execute (
          "SELECT COUNT(*) AS n FROM cortext_label_bank.label_bank_vec", {});
      if (rows.empty ())
        {
          return 0;
        }
      auto it = rows[0].find ("n");
      if (it == rows[0].end ())
        {
          return 0;
        }
      if (it->second.type () == typeid (long long))
        {
          return std::any_cast<long long> (it->second);
        }
      if (it->second.type () == typeid (int))
        {
          return std::any_cast<int> (it->second);
        }
    }
  catch (...)
    {
    }
  return 0;
}

std::string
ReadMeta (Transaction &tx, const std::string &key)
{
  try
    {
      auto rows = tx.Execute (
          "SELECT value FROM cortext_label_bank.label_bank_meta "
          "WHERE key = ? LIMIT 1",
          { key });
      if (rows.empty ())
        {
          return {};
        }
      auto it = rows[0].find ("value");
      if (it != rows[0].end () && it->second.type () == typeid (std::string))
        {
          return std::any_cast<std::string> (it->second);
        }
    }
  catch (...)
    {
    }
  return {};
}

} // namespace

void
LoadLabelBank::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  if (p_ctx.label_bank_loaded)
    {
      return;
    }
  p_ctx.label_bank_loaded = true;

  const auto &cfg = context.GetConfig ();
  if (cfg.label_bank_path.empty ())
    {
      return;
    }

  const auto sqlite_path = ResolveLabelBankSqlite (cfg.label_bank_path);
  std::error_code ec;
  if (!std::filesystem::exists (sqlite_path, ec))
    {
      telemetry::LogWarn (
          "Label bank skipped: SQLite asset not found",
          { telemetry::Attribute::String ("component", "label_bank"),
            telemetry::Attribute::String ("path", sqlite_path.string ()) });
      return;
    }

  try
    {
      if (!IsAttached (tx))
        {
          tx.Execute (
              "ATTACH DATABASE ? AS cortext_label_bank",
              { ToAttachUri (sqlite_path) });
        }

      const long long label_count = ReadLabelCount (tx);
      const std::string content_hash = ReadMeta (tx, "content_hash");

      // Embedding-model pin: the bank's vectors must come from the same
      // encoder as the live database's. Both pins are reachable in this
      // transaction (the bank is attached to the main connection). A
      // mismatch is fatal — it cannot be skipped like a missing bank,
      // because a wrong-space bank silently corrupts label salience.
      const std::string bank_pin = ReadMeta (tx, "embedding_model_pin");
      std::string active_pin;
      try
        {
          auto pin_rows = tx.Execute (
              "SELECT fingerprint FROM main.cortext_embedding_model_pin "
              "WHERE id = 1",
              {});
          if (!pin_rows.empty ())
            {
              auto it = pin_rows[0].find ("fingerprint");
              if (it != pin_rows[0].end ()
                  && it->second.type () == typeid (std::string))
                {
                  active_pin = std::any_cast<std::string> (it->second);
                }
            }
        }
      catch (const std::exception &)
        {
          // Live database predates pinning; nothing to compare against.
        }
      if (!active_pin.empty () && bank_pin != active_pin)
        {
          throw LabelBankPinMismatch (
              "label bank embedding-model pin mismatch: bank='"
              + (bank_pin.empty () ? std::string ("<unpinned>") : bank_pin)
              + "' active='" + active_pin
              + "'. Regenerate the label bank with the active encoder "
                "(tools/label_bank_generator + label_bank_sqlite_builder).");
        }
      telemetry::LogInfo (
          "cortext.label_bank_load",
          { telemetry::Attribute::String ("path", sqlite_path.string ()),
            telemetry::Attribute::String ("mode", "attached_vector_search"),
            telemetry::Attribute::String ("content_hash", content_hash),
            telemetry::Attribute::Int64 ("labels_available", label_count) });
    }
  catch (const LabelBankPinMismatch &)
    {
      // Wrong-space banks must not degrade to "skipped": fail the process.
      throw;
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Label bank skipped: attach failed",
          { telemetry::Attribute::String ("component", "label_bank"),
            telemetry::Attribute::String ("path", sqlite_path.string ()),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

} // namespace cortext::operations
