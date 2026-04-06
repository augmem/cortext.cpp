#include <cortext/encoder/encoder.hpp>
#include <cortext/operations/extraction.hpp>
#include <cortext/operations/process_extraction_results.hpp>
#include <cortext/processor.hpp>
#include <cortext/processor/operation_context.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{
constexpr int kEmbeddingDim = 256;

class BenchEncoder : public cortext::Encoder
{
public:
  void
  EncodeText (const std::string &, std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeAudio (const float *, std::size_t,
               std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeImage (const std::uint8_t *, int, int, int,
               std::vector<float> &out_embedding) override
  {
    out_embedding.assign (kEmbeddingDim, 0.0f);
    out_embedding[0] = 1.0f;
  }
};

class ScopedEnvVar
{
public:
  explicit ScopedEnvVar (const char *name) : name_ (name)
  {
    const char *existing = std::getenv (name);
    if (existing != nullptr)
      {
        had_value_ = true;
        old_value_ = existing;
      }
    unsetenv (name_);
  }

  ScopedEnvVar (const char *name, const std::string &value) : ScopedEnvVar (name)
  {
    setenv (name_, value.c_str (), 1);
  }

  ~ScopedEnvVar ()
  {
    if (had_value_)
      {
        setenv (name_, old_value_.c_str (), 1);
      }
    else
      {
        unsetenv (name_);
      }
  }

private:
  const char *name_;
  bool had_value_ = false;
  std::string old_value_;
};

void
SeedSummary (cortext::Store &store)
{
  const std::vector<float> embedding (kEmbeddingDim, 0.0f);
  store.Execute (
      "INSERT INTO embeddings(embedding_id, embedding, created_at) VALUES(?, ?, ?)",
      { 10LL, embedding, 1000LL });
  store.Execute (
      "INSERT INTO memories("
      "memory_id, embedding_id, source_id, kind, start_ts, n_signals, modality, "
      "s_max, s_avg, strength, created_at) "
      "VALUES(?, ?, ?, 'ASSOCIATION', ?, 1, 'text', 1.0, 1.0, 1.0, ?)",
      { 20LL, 10LL, std::string ("summary-1"), 1000LL, 1000LL });
}

long long
GetInt64 (const std::map<std::string, std::any> &row, const std::string &key)
{
  auto it = row.find (key);
  if (it == row.end ())
    {
      return 0;
    }
  if (it->second.type () == typeid (long long))
    {
      return std::any_cast<long long> (it->second);
    }
  if (it->second.type () == typeid (int))
    {
      return static_cast<long long> (std::any_cast<int> (it->second));
    }
  return 0;
}

struct StudyResult
{
  long long label_count = 0;
  long long edge_count = 0;
};

StudyResult
RunStudy (bool legacy_gate, const std::vector<std::string> &labels)
{
  auto unique_store = cortext::SQLiteStore::Create (":memory:");
  auto store = std::shared_ptr<cortext::Store> (std::move (unique_store));
  cortext::store::ApplyMigrations (*store);
  SeedSummary (*store);

  BenchEncoder encoder;
  cortext::SignalProcessor::Config cfg;
  cfg.focus = 0.5;
  cfg.sensitivity = 0.5;
  cfg.stability = 1.0;
  cfg.encoder = &encoder;

  cortext::ProcessorContext pctx;
  cortext::Signal signal;
  signal.timestamp = 2000ULL;
  signal.source_id = "bench";
  signal.embedding = Eigen::VectorXf::Ones (kEmbeddingDim);
  cortext::OperationContext ctx (signal, pctx, cfg, store.get ());

  cortext::operations::ExtractionResult extraction;
  extraction.summary_id = "summary-1";
  for (const auto &label : labels)
    {
      extraction.labels.push_back ({ label, 0.0 });
    }
  pctx.pending_extraction_results.push_back (std::move (extraction));

  ScopedEnvVar gate_env ("CORTEXT_ABLATE_LEGACY_LABEL_GATE",
                         legacy_gate ? "1" : "0");
  cortext::operations::ProcessExtractionResults op;
  auto tx = store->Begin ();
  op.Execute (ctx, *tx);
  tx->Commit ();

  StudyResult result;
  auto label_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM memories WHERE kind = 'LABEL'", {});
  auto edge_rows = store->Execute (
      "SELECT COUNT(*) AS c FROM associations WHERE edge_type = 'has_label'", {});
  result.label_count = GetInt64 (label_rows[0], "c");
  result.edge_count = GetInt64 (edge_rows[0], "c");
  return result;
}

bool
Check (const char *name, bool condition)
{
  std::cout << name << "=" << (condition ? 1 : 0) << "\n";
  return condition;
}

} // namespace

int
main ()
{
  bool ok = true;

  const auto sparse_legacy
      = RunStudy (true, { "Gabriel", "Chicago", "Cortext" });
  const auto sparse_current
      = RunStudy (false, { "Gabriel", "Chicago", "Cortext" });
  std::cout << "label_sparse_legacy_count=" << sparse_legacy.label_count << "\n";
  std::cout << "label_sparse_current_count=" << sparse_current.label_count << "\n";
  std::cout << "label_sparse_legacy_edges=" << sparse_legacy.edge_count << "\n";
  std::cout << "label_sparse_current_edges=" << sparse_current.edge_count << "\n";
  ok &= Check ("label_sparse_preserved_more",
               sparse_current.label_count > sparse_legacy.label_count);
  ok &= Check ("label_sparse_edges_preserved_more",
               sparse_current.edge_count > sparse_legacy.edge_count);

  const auto repeated_legacy
      = RunStudy (true, { "Gabriel", "Gabriel", "Gabriel", "Gabriel", "Gabriel",
                          "Gabriel", "Gabriel", "Gabriel", "Gabriel", "Gabriel",
                          "Gabriel", "Gabriel", "Gabriel", "Gabriel", "Gabriel" });
  const auto repeated_current
      = RunStudy (false, { "Gabriel", "Gabriel", "Gabriel", "Gabriel", "Gabriel",
                           "Gabriel", "Gabriel", "Gabriel", "Gabriel", "Gabriel",
                           "Gabriel", "Gabriel", "Gabriel", "Gabriel", "Gabriel" });
  std::cout << "label_repeated_legacy_count=" << repeated_legacy.label_count << "\n";
  std::cout << "label_repeated_current_count=" << repeated_current.label_count << "\n";
  ok &= Check ("label_repeated_dedup_stable",
               repeated_current.label_count == 1 && repeated_legacy.label_count == 1);

  return ok ? 0 : 1;
}
