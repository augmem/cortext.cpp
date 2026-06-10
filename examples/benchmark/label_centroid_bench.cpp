#include <cortext/models/aist_gguf_encoder.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using Json = nlohmann::json;

struct Options
{
  std::filesystem::path source_records
      = "build/label_centroid_sources/label_centroid_source_records.jsonl";
  std::filesystem::path output_dir = "build/label_centroid_bench";
  std::filesystem::path models_dir = "models";
  std::filesystem::path model_path;
  std::string source_filter = "salt.csv";
  int max_labels = 0;
  int train_prompts_per_label = 44;
  int max_eval_prompts_per_label = 0;
};

struct LabelRecord
{
  std::string label_id;
  std::string source;
  std::string kind;
  std::string label;
  int source_example_count = 0;
  std::vector<std::string> prompts;
};

struct EncodedPrompt
{
  std::string label_id;
  std::string prompt;
  std::vector<float> embedding;
};

struct LabelCentroid
{
  LabelRecord record;
  std::vector<float> full;
};

struct EvalRow
{
  std::string view;
  std::string label_id;
  std::string prompt;
  int rank = 0;
  double target_score = 0.0;
  double top_score = 0.0;
  std::string top_label_id;
  std::string top_label;
  bool top1 = false;
  bool top5 = false;
  bool top10 = false;
};

void
Usage ()
{
  std::cout
      << "Usage: cortext_label_centroid_bench [options]\n"
      << "  --source-records PATH       label_centroid_source_records.jsonl\n"
      << "  --output-dir PATH           output directory\n"
      << "  --models PATH               models directory\n"
      << "  --model PATH                explicit ES-AIST GGUF path\n"
      << "  --source-filter SOURCE      default: salt.csv\n"
      << "  --max-labels N              deterministic cap, 0=all\n"
      << "  --train-prompts-per-label N default: 44\n"
      << "  --max-eval-prompts-per-label N default: 0=all remaining\n";
}

Options
ParseArgs (int argc, char **argv)
{
  Options opts;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      auto require_value = [&] () -> std::string {
        if (i + 1 >= argc)
          {
            throw std::runtime_error ("Missing value for " + arg);
          }
        return argv[++i];
      };
      if (arg == "--source-records")
        {
          opts.source_records = require_value ();
        }
      else if (arg == "--output-dir")
        {
          opts.output_dir = require_value ();
        }
      else if (arg == "--models")
        {
          opts.models_dir = require_value ();
        }
      else if (arg == "--model")
        {
          opts.model_path = require_value ();
        }
      else if (arg == "--source-filter")
        {
          opts.source_filter = require_value ();
        }
      else if (arg == "--max-labels")
        {
          opts.max_labels = std::stoi (require_value ());
        }
      else if (arg == "--train-prompts-per-label")
        {
          opts.train_prompts_per_label = std::stoi (require_value ());
        }
      else if (arg == "--max-eval-prompts-per-label")
        {
          opts.max_eval_prompts_per_label = std::stoi (require_value ());
        }
      else if (arg == "--help" || arg == "-h")
        {
          Usage ();
          std::exit (0);
        }
      else
        {
          throw std::runtime_error ("Unknown argument: " + arg);
        }
    }
  return opts;
}

std::string
CsvEscape (const std::string &s)
{
  if (s.find_first_of (",\"\n\r") == std::string::npos)
    {
      return s;
    }
  std::string out = "\"";
  for (char c : s)
    {
      if (c == '"')
        {
          out += "\"\"";
        }
      else
        {
          out += c;
        }
    }
  out += '"';
  return out;
}

double
Millis (std::chrono::steady_clock::duration duration)
{
  return std::chrono::duration<double, std::milli> (duration).count ();
}

void
Normalize (std::vector<float> &values)
{
  double norm_sq = 0.0;
  for (float value : values)
    {
      norm_sq += static_cast<double> (value) * static_cast<double> (value);
    }
  if (norm_sq <= 0.0)
    {
      return;
    }
  const auto inv = static_cast<float> (1.0 / std::sqrt (norm_sq));
  for (float &value : values)
    {
      value *= inv;
    }
}

std::vector<float>
MeanNormalized (const std::vector<std::vector<float>> &vectors)
{
  if (vectors.empty ())
    {
      return {};
    }
  std::vector<float> out (vectors.front ().size (), 0.0F);
  for (const auto &vector : vectors)
    {
      if (vector.size () != out.size ())
        {
          throw std::runtime_error ("Embedding dimension mismatch");
        }
      for (std::size_t i = 0; i < vector.size (); ++i)
        {
          out[i] += vector[i];
        }
    }
  const auto inv = 1.0F / static_cast<float> (vectors.size ());
  for (float &value : out)
    {
      value *= inv;
    }
  Normalize (out);
  return out;
}

std::vector<float>
SliceView (const std::vector<float> &embedding, const std::string &view)
{
  if (view == "full_key")
    {
      std::vector<float> out = embedding;
      Normalize (out);
      return out;
    }
  std::size_t begin = 0;
  std::size_t end = embedding.size ();
  if (view == "semantic_key")
    {
      begin = 0;
      end = std::min<std::size_t> (768, embedding.size ());
    }
  else if (view == "entity_key")
    {
      begin = std::min<std::size_t> (768, embedding.size ());
      end = std::min<std::size_t> (1536, embedding.size ());
    }
  else if (view == "prefix_768")
    {
      begin = 0;
      end = std::min<std::size_t> (768, embedding.size ());
    }
  else if (view == "prefix_1536")
    {
      begin = 0;
      end = std::min<std::size_t> (1536, embedding.size ());
    }
  else
    {
      throw std::runtime_error ("Unknown view: " + view);
    }
  std::vector<float> out;
  if (begin < end)
    {
      out.assign (embedding.begin () + static_cast<std::ptrdiff_t> (begin),
                  embedding.begin () + static_cast<std::ptrdiff_t> (end));
    }
  Normalize (out);
  return out;
}

double
Dot (const std::vector<float> &a, const std::vector<float> &b)
{
  const std::size_t n = std::min (a.size (), b.size ());
  double out = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      out += static_cast<double> (a[i]) * static_cast<double> (b[i]);
    }
  return out;
}

std::filesystem::path
ResolveModel (const Options &opts)
{
  if (!opts.model_path.empty ())
    {
      if (!std::filesystem::exists (opts.model_path))
        {
          throw std::runtime_error ("Model path does not exist: "
                                    + opts.model_path.string ());
        }
      return opts.model_path;
    }
  if (const char *env = std::getenv ("CORTEXT_ES_AIST_MODEL_PATH"))
    {
      std::filesystem::path path (env);
      if (!std::filesystem::exists (path))
        {
          throw std::runtime_error ("CORTEXT_ES_AIST_MODEL_PATH does not exist: "
                                    + path.string ());
        }
      return path;
    }
  auto resolved = cortext::ResolveEssAistGgufModelPath (opts.models_dir);
  if (!resolved)
    {
      throw std::runtime_error (
          "ES-AIST/ESS-AIST GGUF not found. Use --model or --models.");
    }
  return *resolved;
}

std::vector<LabelRecord>
LoadRecords (const Options &opts)
{
  std::ifstream in (opts.source_records);
  if (!in)
    {
      throw std::runtime_error ("Failed to open " + opts.source_records.string ());
    }
  std::vector<LabelRecord> records;
  std::string line;
  while (std::getline (in, line))
    {
      if (line.empty ())
        {
          continue;
        }
      Json json = Json::parse (line, nullptr, false);
      if (json.is_discarded ())
        {
          continue;
        }
      const std::string source = json.value ("source", "");
      if (!opts.source_filter.empty () && source != opts.source_filter)
        {
          continue;
        }
      LabelRecord record;
      record.label_id = json.value ("label_id", "");
      record.source = source;
      record.kind = json.value ("kind", "");
      record.label = json.value ("label", "");
      record.source_example_count = json.value ("source_example_count", 0);
      if (json.contains ("prompts") && json["prompts"].is_array ())
        {
          for (const auto &prompt : json["prompts"])
            {
              if (prompt.is_string ())
                {
                  record.prompts.push_back (prompt.get<std::string> ());
                }
            }
        }
      if (!record.label_id.empty ()
          && static_cast<int> (record.prompts.size ())
                 > opts.train_prompts_per_label)
        {
          records.push_back (std::move (record));
        }
      if (opts.max_labels > 0
          && static_cast<int> (records.size ()) >= opts.max_labels)
        {
          break;
        }
    }
  return records;
}

std::vector<float>
EncodeText (cortext::AaitGgufEncoder &encoder, const std::string &text)
{
  std::vector<float> embedding;
  encoder.EncodeText (text, embedding);
  Normalize (embedding);
  return embedding;
}

std::vector<std::string>
Views ()
{
  return { "semantic_key", "entity_key", "full_key" };
}

std::pair<int, std::vector<std::pair<double, int>>>
RankLabels (const std::vector<float> &query,
            const std::vector<LabelCentroid> &centroids,
            const std::vector<std::vector<float>> &centroid_views,
            int target_index)
{
  std::vector<std::pair<double, int>> scores;
  scores.reserve (centroids.size ());
  for (std::size_t i = 0; i < centroids.size (); ++i)
    {
      scores.push_back ({ Dot (query, centroid_views[i]),
                          static_cast<int> (i) });
    }
  std::sort (scores.begin (), scores.end (),
             [] (const auto &a, const auto &b) {
               if (a.first != b.first)
                 {
                   return a.first > b.first;
                 }
               return a.second < b.second;
             });
  int rank = static_cast<int> (scores.size ());
  for (std::size_t i = 0; i < scores.size (); ++i)
    {
      if (scores[i].second == target_index)
        {
          rank = static_cast<int> (i) + 1;
          break;
        }
    }
  return { rank, scores };
}

Json
VectorPreview (const std::vector<float> &values)
{
  Json out = Json::array ();
  const auto n = std::min<std::size_t> (values.size (), 16);
  for (std::size_t i = 0; i < n; ++i)
    {
      out.push_back (values[i]);
    }
  return out;
}

void
WriteCentroids (const std::filesystem::path &path,
                const std::vector<LabelCentroid> &centroids)
{
  std::ofstream out (path);
  for (const auto &centroid : centroids)
    {
      Json json{
        { "label_id", centroid.record.label_id },
        { "source", centroid.record.source },
        { "kind", centroid.record.kind },
        { "label", centroid.record.label },
        { "dim", centroid.full.size () },
        { "semantic_range", Json::array ({ 0, 768 }) },
        { "entity_range", Json::array ({ 768, 1536 }) },
        { "full_range", Json::array ({ 0, 1536 }) },
        { "full_centroid", centroid.full },
      };
      out << json.dump () << '\n';
    }
}

void
WriteCases (const std::filesystem::path &path,
            const std::vector<EvalRow> &rows)
{
  std::ofstream out (path);
  out << "view,label_id,prompt,rank,target_score,top_score,top_label_id,"
         "top_label,top1,top5,top10\n";
  for (const auto &row : rows)
    {
      out << row.view << ',' << CsvEscape (row.label_id) << ','
          << CsvEscape (row.prompt) << ',' << row.rank << ','
          << row.target_score << ',' << row.top_score << ','
          << CsvEscape (row.top_label_id) << ',' << CsvEscape (row.top_label)
          << ',' << (row.top1 ? 1 : 0) << ',' << (row.top5 ? 1 : 0)
          << ',' << (row.top10 ? 1 : 0) << '\n';
    }
}

Json
SummarizeRows (const std::vector<EvalRow> &rows)
{
  std::map<std::string, std::vector<const EvalRow *>> by_view;
  for (const auto &row : rows)
    {
      by_view[row.view].push_back (&row);
    }

  Json out = Json::object ();
  for (const auto &[view, view_rows] : by_view)
    {
      double top1 = 0.0;
      double top5 = 0.0;
      double top10 = 0.0;
      double mrr = 0.0;
      double mean_rank = 0.0;
      for (const auto *row : view_rows)
        {
          top1 += row->top1 ? 1.0 : 0.0;
          top5 += row->top5 ? 1.0 : 0.0;
          top10 += row->top10 ? 1.0 : 0.0;
          mrr += row->rank > 0 ? 1.0 / static_cast<double> (row->rank) : 0.0;
          mean_rank += row->rank;
        }
      const double denom = static_cast<double> (view_rows.size ());
      out[view] = Json{
        { "eval_count", view_rows.size () },
        { "top1", top1 / denom },
        { "top5", top5 / denom },
        { "top10", top10 / denom },
        { "mrr", mrr / denom },
        { "mean_rank", mean_rank / denom },
      };
    }
  return out;
}

void
WriteTopkExamples (const std::filesystem::path &path,
                   const std::vector<EvalRow> &rows)
{
  std::ofstream out (path);
  out << "view,label_id,prompt,rank,target_score,top_score,top_label_id,"
         "top_label\n";
  int written = 0;
  for (const auto &row : rows)
    {
      if (row.view != "full_key" || row.top1)
        {
          continue;
        }
      out << row.view << ',' << CsvEscape (row.label_id) << ','
          << CsvEscape (row.prompt) << ',' << row.rank << ','
          << row.target_score << ',' << row.top_score << ','
          << CsvEscape (row.top_label_id) << ',' << CsvEscape (row.top_label)
          << '\n';
      ++written;
      if (written >= 50)
        {
          break;
        }
    }
}

int
Main (int argc, char **argv)
{
  Options opts = ParseArgs (argc, argv);
  std::filesystem::create_directories (opts.output_dir);

  const auto records = LoadRecords (opts);
  if (records.empty ())
    {
      throw std::runtime_error ("No label records loaded");
    }

  const auto model_path = ResolveModel (opts);
  cortext::AaitGgufConfig config;
  config.model_path = model_path.string ();
  config.context_length = 128;
  cortext::AaitGgufEncoder encoder (config);
  if (!encoder.IsRuntimeAvailable ())
    {
      throw std::runtime_error ("ES/AIST runtime unavailable: "
                                + encoder.Inspect ().runtime_error);
    }

  const auto start = std::chrono::steady_clock::now ();
  std::size_t encode_count = 0;
  std::vector<double> encode_ms;
  std::vector<LabelCentroid> centroids;
  std::vector<EncodedPrompt> eval_prompts;
  centroids.reserve (records.size ());

  for (const auto &record : records)
    {
      std::vector<std::vector<float>> train_embeddings;
      const int train_count = std::min<int> (
          opts.train_prompts_per_label, static_cast<int> (record.prompts.size ()));
      for (int i = 0; i < train_count; ++i)
        {
          const auto encode_start = std::chrono::steady_clock::now ();
          train_embeddings.push_back (EncodeText (encoder, record.prompts[i]));
          const auto encode_end = std::chrono::steady_clock::now ();
          encode_ms.push_back (Millis (encode_end - encode_start));
          ++encode_count;
        }
      LabelCentroid centroid;
      centroid.record = record;
      centroid.full = MeanNormalized (train_embeddings);
      centroids.push_back (std::move (centroid));

      int eval_added = 0;
      for (std::size_t i = static_cast<std::size_t> (train_count);
           i < record.prompts.size (); ++i)
        {
          if (opts.max_eval_prompts_per_label > 0
              && eval_added >= opts.max_eval_prompts_per_label)
            {
              break;
            }
          const auto encode_start = std::chrono::steady_clock::now ();
          EncodedPrompt encoded;
          encoded.label_id = record.label_id;
          encoded.prompt = record.prompts[i];
          encoded.embedding = EncodeText (encoder, record.prompts[i]);
          const auto encode_end = std::chrono::steady_clock::now ();
          encode_ms.push_back (Millis (encode_end - encode_start));
          ++encode_count;
          eval_prompts.push_back (std::move (encoded));
          ++eval_added;
        }
    }

  std::unordered_map<std::string, int> label_index;
  for (std::size_t i = 0; i < centroids.size (); ++i)
    {
      label_index[centroids[i].record.label_id] = static_cast<int> (i);
    }

  std::vector<EvalRow> rows;
  for (const auto &view : Views ())
    {
      std::vector<std::vector<float>> centroid_views;
      centroid_views.reserve (centroids.size ());
      for (const auto &centroid : centroids)
        {
          centroid_views.push_back (SliceView (centroid.full, view));
        }
      for (const auto &eval : eval_prompts)
        {
          const auto target = label_index.find (eval.label_id);
          if (target == label_index.end ())
            {
              continue;
            }
          const auto query = SliceView (eval.embedding, view);
          const auto [rank, scores] = RankLabels (
              query, centroids, centroid_views, target->second);
          EvalRow row;
          row.view = view;
          row.label_id = eval.label_id;
          row.prompt = eval.prompt;
          row.rank = rank;
          row.target_score = scores.empty () ? 0.0
                                             : Dot (query,
                                                    centroid_views[target->second]);
          row.top_score = scores.empty () ? 0.0 : scores.front ().first;
          if (!scores.empty ())
            {
              const auto &top = centroids[static_cast<std::size_t> (
                  scores.front ().second)];
              row.top_label_id = top.record.label_id;
              row.top_label = top.record.label;
            }
          row.top1 = rank <= 1;
          row.top5 = rank <= 5;
          row.top10 = rank <= 10;
          rows.push_back (std::move (row));
        }
    }

  std::sort (encode_ms.begin (), encode_ms.end ());
  const auto percentile = [&] (double q) -> double {
    if (encode_ms.empty ())
      {
        return 0.0;
      }
    const auto idx = std::min<std::size_t> (
        encode_ms.size () - 1,
        static_cast<std::size_t> (std::floor (
            q * static_cast<double> (encode_ms.size () - 1))));
    return encode_ms[idx];
  };
  const double mean_ms = encode_ms.empty ()
                             ? 0.0
                             : std::accumulate (encode_ms.begin (),
                                                encode_ms.end (), 0.0)
                                   / static_cast<double> (encode_ms.size ());

  WriteCentroids (opts.output_dir / "label_centroids.jsonl", centroids);
  WriteCases (opts.output_dir / "label_centroid_eval_cases.csv", rows);
  WriteTopkExamples (opts.output_dir / "label_centroid_failure_examples.csv",
                     rows);

  Json results{
    { "status", "ok" },
    { "model_path", model_path.string () },
    { "source_records", opts.source_records.string () },
    { "source_filter", opts.source_filter },
    { "label_count", centroids.size () },
    { "eval_prompt_count", eval_prompts.size () },
    { "train_prompts_per_label", opts.train_prompts_per_label },
    { "legacy_256d_centroids_compatible", false },
    { "reason_legacy_256d_incompatible",
      "Existing data/*_256.npy centroids were generated in the old 256d "
      "embedding space and cannot be compared to ES/AIST 1536d outputs." },
    { "contract",
      Json{ { "semantic_key", Json::array ({ 0, 768 }) },
            { "entity_key", Json::array ({ 768, 1536 }) },
            { "full_key", Json::array ({ 0, 1536 }) } } },
    { "metrics", SummarizeRows (rows) },
    { "encode_count", encode_count },
    { "mean_encode_ms", mean_ms },
    { "p50_encode_ms", percentile (0.50) },
    { "p95_encode_ms", percentile (0.95) },
    { "elapsed_ms", Millis (std::chrono::steady_clock::now () - start) },
  };
  std::ofstream results_out (opts.output_dir
                             / "label_centroid_eval_results.json");
  results_out << std::setw (2) << results << '\n';

  Json latency{
    { "encode_count", encode_count },
    { "mean_encode_ms", mean_ms },
    { "p50_encode_ms", percentile (0.50) },
    { "p95_encode_ms", percentile (0.95) },
  };
  std::ofstream latency_out (opts.output_dir / "label_centroid_latency.json");
  latency_out << std::setw (2) << latency << '\n';

  Json summary{
    { "label_count", centroids.size () },
    { "eval_prompt_count", eval_prompts.size () },
    { "source_filter", opts.source_filter },
    { "output_files",
      Json::array ({ "label_centroids.jsonl",
                     "label_centroid_eval_results.json",
                     "label_centroid_eval_cases.csv",
                     "label_centroid_failure_examples.csv",
                     "label_centroid_latency.json" }) },
  };
  std::ofstream summary_out (opts.output_dir / "label_centroid_summary.json");
  summary_out << std::setw (2) << summary << '\n';

  std::cout << "label_centroid_bench complete labels=" << centroids.size ()
            << " eval_prompts=" << eval_prompts.size ()
            << " mean_encode_ms=" << mean_ms << '\n';
  return 0;
}

} // namespace

int
main (int argc, char **argv)
{
  try
    {
      return Main (argc, argv);
    }
  catch (const std::exception &ex)
    {
      std::cerr << "error: " << ex.what () << '\n';
      return 1;
    }
}
