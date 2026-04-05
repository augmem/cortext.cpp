#include "../../src/store/facts.hpp"

#include <cortext/encoder/encoder.hpp>
#include <cortext/store/schema.hpp>
#include <cortext/store/sqlite_store.hpp>

#include <Eigen/Dense>

#include <any>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{

class BenchEncoder : public cortext::Encoder
{
public:
  void
  EncodeText (const std::string & /*text*/,
              std::vector<float> &out_embedding) override
  {
    out_embedding.assign (256, 0.0f);
    out_embedding[0] = 1.0f;
  }

  void
  EncodeAudio (const float * /*pcm*/, std::size_t /*num_samples*/,
               std::vector<float> &out_embedding) override
  {
    EncodeText ("", out_embedding);
  }

  void
  EncodeImage (const std::uint8_t * /*data*/, int /*width*/, int /*height*/,
               int /*channels*/, std::vector<float> &out_embedding) override
  {
    EncodeText ("", out_embedding);
  }
};

double
ElapsedMs (std::chrono::steady_clock::time_point start,
           std::chrono::steady_clock::time_point end)
{
  return std::chrono::duration<double, std::milli> (end - start).count ();
}

long long
LastInsertId (cortext::Transaction &tx)
{
  auto rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
  if (rows.empty ())
    {
      return 0;
    }
  auto it = rows[0].find ("id");
  if (it == rows[0].end () || !it->second.has_value ())
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
  return 0;
}

} // namespace

int
main (int argc, char *argv[])
{
  int fact_count = 1000;
  int iterations = 5;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if (arg == "--facts" && i + 1 < argc)
        {
          fact_count = std::atoi (argv[++i]);
        }
      else if (arg == "--iterations" && i + 1 < argc)
        {
          iterations = std::atoi (argv[++i]);
        }
      else if (arg == "--help" || arg == "-h")
        {
          std::cout << "Usage: cortext_fact_store_bench [--facts N] "
                       "[--iterations N]\n";
          return 0;
        }
    }

  BenchEncoder encoder;
  double insert_total_ms = 0.0;
  double current_query_total_ms = 0.0;
  double valid_query_total_ms = 0.0;
  double known_query_total_ms = 0.0;

  for (int iter = 0; iter < iterations; ++iter)
    {
      auto store = cortext::SQLiteStore::Create (":memory:");
      cortext::store::ApplyMigrations (*store);

      auto tx = store->Begin ();
      tx->Execute (
          "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
          { std::vector<float> (256, 0.0f), 1000LL });
      const long long summary_embedding_id = LastInsertId (*tx);

      auto t_insert_start = std::chrono::steady_clock::now ();
      for (int i = 0; i < fact_count; ++i)
        {
          const long long summary_memory_id = 1000LL + i;
          const long long summary_start_ts = 1000LL + static_cast<long long> (i);
          tx->Execute (
              "INSERT INTO memories "
              "(memory_id, embedding_id, source_id, kind, start_ts, n_signals, "
              " modality, s_max, s_avg, created_at) "
              "VALUES (?, ?, ?, 'ASSOCIATION', ?, 1, 'text', 0.5, 0.5, ?)",
              { summary_memory_id, summary_embedding_id,
                std::string ("summary-") + std::to_string (i), summary_start_ts,
                summary_start_ts });

          tx->Execute (
              "INSERT INTO fact_assertions "
              "(subject, predicate, object, canonical_subject, "
              " canonical_predicate, canonical_object, valid_start_ts, "
              " recorded_at_ts, confidence, summary_memory_id, created_at) "
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              { std::string ("Alice"),
                std::string ("works_at"),
                std::string ("Company-") + std::to_string (i),
                std::string ("alice"),
                std::string ("works_at"),
                std::string ("company_") + std::to_string (i),
                summary_start_ts,
                5000LL + i,
                0.8,
                summary_memory_id,
                5000LL + i });
          const long long fact_id = LastInsertId (*tx);
          tx->Execute (
              "INSERT INTO fact_evidence "
              "(fact_id, source_memory_id, evidence_type, support_weight) "
              "VALUES (?, ?, 'summary', 1.0)",
              { fact_id, summary_memory_id });
          cortext::store::RefreshFactCache (*tx, &encoder, fact_id, 8000ULL);
        }
      auto t_insert_end = std::chrono::steady_clock::now ();
      insert_total_ms += ElapsedMs (t_insert_start, t_insert_end);
      tx->Commit ();

      auto read_tx = store->Begin ();

      auto t_current_start = std::chrono::steady_clock::now ();
      auto current = cortext::store::QueryFacts (
          *read_tx, { cortext::store::FactQueryMode::Current,
                      std::string ("alice"), std::string ("works_at"),
                      8000ULL });
      auto t_current_end = std::chrono::steady_clock::now ();
      current_query_total_ms += ElapsedMs (t_current_start, t_current_end);

      auto t_valid_start = std::chrono::steady_clock::now ();
      auto valid = cortext::store::QueryFacts (
          *read_tx, { cortext::store::FactQueryMode::ValidAt,
                      std::string ("alice"), std::string ("works_at"),
                      1500ULL });
      auto t_valid_end = std::chrono::steady_clock::now ();
      valid_query_total_ms += ElapsedMs (t_valid_start, t_valid_end);

      auto t_known_start = std::chrono::steady_clock::now ();
      auto known = cortext::store::QueryFacts (
          *read_tx, { cortext::store::FactQueryMode::KnownAt,
                      std::string ("alice"), std::string ("works_at"),
                      6000ULL });
      auto t_known_end = std::chrono::steady_clock::now ();
      known_query_total_ms += ElapsedMs (t_known_start, t_known_end);

      std::cout << "iteration=" << (iter + 1)
                << " inserted=" << fact_count
                << " current=" << current.size ()
                << " valid=" << valid.size ()
                << " known=" << known.size () << "\n";
    }

  std::cout << std::fixed << std::setprecision (2);
  std::cout << "fact_count=" << fact_count
            << " iterations=" << iterations << "\n";
  std::cout << "insert_mean_ms=" << (insert_total_ms / iterations) << "\n";
  std::cout << "current_query_mean_ms="
            << (current_query_total_ms / iterations) << "\n";
  std::cout << "valid_query_mean_ms="
            << (valid_query_total_ms / iterations) << "\n";
  std::cout << "known_query_mean_ms="
            << (known_query_total_ms / iterations) << "\n";
  return 0;
}
