#include <cortext/models/aait_gguf_encoder.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using Json = nlohmann::json;

constexpr int kDbDim = 256;

struct Options
{
  std::filesystem::path source_records
      = "build/label_centroid_sources/label_centroid_source_records.jsonl";
  std::filesystem::path output_dir = "build/label_vector_db_bench";
  std::filesystem::path models_dir = "models";
  std::filesystem::path model_path;
  std::string source_filter = "salt.csv";
  int max_labels = 0;
  int prototype_candidates_per_label = 16;
  int max_prototypes_per_label = 8;
  int max_eval_prompts_per_label = 8;
  bool skip_eval = false;
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

struct LabelInfo
{
  LabelRecord record;
  std::vector<float> centroid;
  int prototype_begin = 0;
  int prototype_end = 0;
};

struct PrototypeRow
{
  int label_index = -1;
  int slot = 0;
  std::string prompt;
  std::vector<float> embedding;
};

struct EvalPrompt
{
  int label_index = -1;
  std::string prompt;
  std::vector<float> embedding;
};

struct EvalRow
{
  std::string variant;
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
      << "Usage: cortext_label_vector_db_bench [options]\n"
      << "  --source-records PATH            source records JSONL\n"
      << "  --output-dir PATH                output directory\n"
      << "  --models PATH                    models directory\n"
      << "  --model PATH                     explicit ES-AIST/AIST GGUF path\n"
      << "  --source-filter SOURCE|all       default: salt.csv\n"
      << "  --max-labels N                   deterministic cap, 0=all\n"
      << "  --prototype-candidates-per-label N default: 16\n"
      << "  --max-prototypes-per-label N     default: 8\n"
      << "  --max-eval-prompts-per-label N   default: 8, 0=all remaining\n"
      << "  --skip-eval                      write DB artifacts only\n";
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
      else if (arg == "--prototype-candidates-per-label")
        {
          opts.prototype_candidates_per_label = std::stoi (require_value ());
        }
      else if (arg == "--max-prototypes-per-label")
        {
          opts.max_prototypes_per_label = std::stoi (require_value ());
        }
      else if (arg == "--max-eval-prompts-per-label")
        {
          opts.max_eval_prompts_per_label = std::stoi (require_value ());
        }
      else if (arg == "--skip-eval")
        {
          opts.skip_eval = true;
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
  opts.prototype_candidates_per_label
      = std::max (1, opts.prototype_candidates_per_label);
  opts.max_prototypes_per_label = std::max (1, opts.max_prototypes_per_label);
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
Slice256 (const std::vector<float> &embedding)
{
  if (embedding.size () < static_cast<std::size_t> (kDbDim))
    {
      throw std::runtime_error ("Embedding smaller than 256d");
    }
  std::vector<float> out (embedding.begin (), embedding.begin () + kDbDim);
  Normalize (out);
  return out;
}

double
Dot (const std::vector<float> &a, const std::vector<float> &b)
{
  double out = 0.0;
  const auto n = std::min (a.size (), b.size ());
  for (std::size_t i = 0; i < n; ++i)
    {
      out += static_cast<double> (a[i]) * static_cast<double> (b[i]);
    }
  return out;
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

std::filesystem::path
ResolveModel (const Options &opts)
{
  if (!opts.model_path.empty ())
    {
      return opts.model_path;
    }
  const auto candidate
      = opts.models_dir / "ES-AIST-81M-preview-GGUF" / "ES-AIST-81M_q8_0.gguf";
  if (std::filesystem::exists (candidate))
    {
      return candidate;
    }
  throw std::runtime_error ("Could not resolve ES-AIST q8 model under "
                            + opts.models_dir.string ());
}

bool
SourceMatches (const std::string &filter, const std::string &source)
{
  return filter.empty () || filter == "all" || filter == "*" || filter == source;
}

std::vector<LabelRecord>
LoadRecords (const Options &opts)
{
  std::ifstream in (opts.source_records);
  if (!in.is_open ())
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
      const auto row = Json::parse (line);
      LabelRecord record;
      record.label_id = row.value ("label_id", "");
      record.source = row.value ("source", "");
      record.kind = row.value ("kind", "");
      record.label = row.value ("label", "");
      record.source_example_count = row.value ("source_example_count", 0);
      if (!SourceMatches (opts.source_filter, record.source))
        {
          continue;
        }
      if (row.contains ("prompts") && row["prompts"].is_array ())
        {
          for (const auto &prompt : row["prompts"])
            {
              if (prompt.is_string ())
                {
                  const auto value = prompt.get<std::string> ();
                  if (!value.empty ())
                    {
                      record.prompts.push_back (value);
                    }
                }
            }
        }
      if (record.label_id.empty () || record.prompts.empty ())
        {
          continue;
        }
      records.push_back (std::move (record));
      if (opts.max_labels > 0
          && static_cast<int> (records.size ()) >= opts.max_labels)
        {
          break;
        }
    }
  return records;
}

std::vector<float>
EncodeText256 (cortext::AaitGgufEncoder &encoder, const std::string &text)
{
  std::vector<float> embedding;
  encoder.EncodeText (text, embedding);
  Normalize (embedding);
  return Slice256 (embedding);
}

std::vector<int>
SelectFarthestFirst (const std::vector<std::vector<float>> &vectors,
                     int max_count)
{
  std::vector<int> selected;
  if (vectors.empty () || max_count <= 0)
    {
      return selected;
    }
  selected.push_back (0);
  while (static_cast<int> (selected.size ())
         < std::min<int> (max_count, vectors.size ()))
    {
      double best_min_distance = -1.0;
      int best_index = -1;
      for (int i = 0; i < static_cast<int> (vectors.size ()); ++i)
        {
          if (std::find (selected.begin (), selected.end (), i)
              != selected.end ())
            {
              continue;
            }
          double min_distance = std::numeric_limits<double>::infinity ();
          for (int selected_index : selected)
            {
              min_distance = std::min (
                  min_distance, 1.0 - Dot (vectors[static_cast<std::size_t> (i)],
                                           vectors[static_cast<std::size_t> (
                                               selected_index)]));
            }
          if (min_distance > best_min_distance)
            {
              best_min_distance = min_distance;
              best_index = i;
            }
        }
      if (best_index < 0)
        {
          break;
        }
      selected.push_back (best_index);
    }
  return selected;
}

std::pair<int, std::vector<std::pair<double, int>>>
RankScores (const std::vector<double> &scores, int target_index)
{
  std::vector<std::pair<double, int>> ranked;
  ranked.reserve (scores.size ());
  for (int i = 0; i < static_cast<int> (scores.size ()); ++i)
    {
      ranked.push_back ({ scores[static_cast<std::size_t> (i)], i });
    }
  std::sort (ranked.begin (), ranked.end (),
             [] (const auto &a, const auto &b) {
               if (a.first != b.first)
                 {
                   return a.first > b.first;
                 }
               return a.second < b.second;
             });
  int rank = static_cast<int> (ranked.size ());
  for (std::size_t i = 0; i < ranked.size (); ++i)
    {
      if (ranked[i].second == target_index)
        {
          rank = static_cast<int> (i) + 1;
          break;
        }
    }
  return { rank, ranked };
}

EvalRow
MakeEvalRow (const std::string &variant,
             const std::vector<LabelInfo> &labels,
             const EvalPrompt &query,
             const std::vector<double> &scores)
{
  const auto [rank, ranked] = RankScores (scores, query.label_index);
  EvalRow row;
  row.variant = variant;
  row.label_id = labels[static_cast<std::size_t> (query.label_index)]
                     .record.label_id;
  row.prompt = query.prompt;
  row.rank = rank;
  row.target_score = scores[static_cast<std::size_t> (query.label_index)];
  row.top_score = ranked.empty () ? 0.0 : ranked.front ().first;
  if (!ranked.empty ())
    {
      const auto &top = labels[static_cast<std::size_t> (ranked.front ().second)];
      row.top_label_id = top.record.label_id;
      row.top_label = top.record.label;
    }
  row.top1 = rank <= 1;
  row.top5 = rank <= 5;
  row.top10 = rank <= 10;
  return row;
}

void
EvaluateCentroid (const std::vector<LabelInfo> &labels,
                  const EvalPrompt &query,
                  std::vector<EvalRow> &rows)
{
  std::vector<double> scores (labels.size (), -2.0);
  for (std::size_t i = 0; i < labels.size (); ++i)
    {
      scores[i] = Dot (query.embedding, labels[i].centroid);
    }
  rows.push_back (MakeEvalRow ("centroid_256", labels, query, scores));
}

void
EvaluatePrototypeMax (const std::vector<LabelInfo> &labels,
                      const std::vector<PrototypeRow> &prototypes,
                      const EvalPrompt &query,
                      int max_slot,
                      std::vector<EvalRow> &rows)
{
  std::vector<double> scores (labels.size (), -2.0);
  for (const auto &prototype : prototypes)
    {
      if (prototype.slot >= max_slot)
        {
          continue;
        }
      auto &score = scores[static_cast<std::size_t> (prototype.label_index)];
      score = std::max (score, Dot (query.embedding, prototype.embedding));
    }
  rows.push_back (MakeEvalRow ("prototype_max_" + std::to_string (max_slot)
                                   + "_256",
                               labels, query, scores));
}

void
EvaluatePrototypeTopKMean (const std::vector<LabelInfo> &labels,
                           const std::vector<PrototypeRow> &prototypes,
                           const EvalPrompt &query,
                           int max_slot,
                           int top_k,
                           std::vector<EvalRow> &rows)
{
  std::vector<std::vector<double>> per_label (labels.size ());
  for (const auto &prototype : prototypes)
    {
      if (prototype.slot >= max_slot)
        {
          continue;
        }
      auto &scores = per_label[static_cast<std::size_t> (
          prototype.label_index)];
      scores.push_back (Dot (query.embedding, prototype.embedding));
    }

  std::vector<double> scores (labels.size (), -2.0);
  for (std::size_t i = 0; i < per_label.size (); ++i)
    {
      auto &label_scores = per_label[i];
      if (label_scores.empty ())
        {
          continue;
        }
      std::sort (label_scores.begin (), label_scores.end (),
                 std::greater<double> ());
      const int keep = std::min<int> (top_k, label_scores.size ());
      double sum = 0.0;
      for (int j = 0; j < keep; ++j)
        {
          sum += label_scores[static_cast<std::size_t> (j)];
        }
      scores[i] = sum / static_cast<double> (keep);
    }

  rows.push_back (MakeEvalRow ("prototype_top" + std::to_string (top_k)
                                   + "_mean_" + std::to_string (max_slot)
                                   + "_256",
                               labels, query, scores));
}

Json
SummarizeRows (const std::vector<EvalRow> &rows)
{
  std::map<std::string, std::vector<const EvalRow *>> by_variant;
  for (const auto &row : rows)
    {
      by_variant[row.variant].push_back (&row);
    }

  Json out = Json::object ();
  for (const auto &[variant, variant_rows] : by_variant)
    {
      double top1 = 0.0;
      double top5 = 0.0;
      double top10 = 0.0;
      double mrr = 0.0;
      double mean_rank = 0.0;
      for (const auto *row : variant_rows)
        {
          top1 += row->top1 ? 1.0 : 0.0;
          top5 += row->top5 ? 1.0 : 0.0;
          top10 += row->top10 ? 1.0 : 0.0;
          mrr += row->rank > 0 ? 1.0 / static_cast<double> (row->rank) : 0.0;
          mean_rank += row->rank;
        }
      const double denom = static_cast<double> (variant_rows.size ());
      out[variant] = Json{
        { "eval_count", variant_rows.size () },
        { "top1", denom > 0.0 ? top1 / denom : 0.0 },
        { "top5", denom > 0.0 ? top5 / denom : 0.0 },
        { "top10", denom > 0.0 ? top10 / denom : 0.0 },
        { "mrr", denom > 0.0 ? mrr / denom : 0.0 },
        { "mean_rank", denom > 0.0 ? mean_rank / denom : 0.0 },
      };
    }
  return out;
}

void
WriteRawF32Matrix (const std::filesystem::path &path,
                   const std::vector<PrototypeRow> &rows)
{
  std::ofstream out (path, std::ios::binary);
  if (!out.is_open ())
    {
      throw std::runtime_error ("Failed to open " + path.string ());
    }
  for (const auto &row : rows)
    {
      if (row.embedding.size () != kDbDim)
        {
          throw std::runtime_error ("Bad prototype dimension");
        }
      out.write (reinterpret_cast<const char *> (row.embedding.data ()),
                 static_cast<std::streamsize> (row.embedding.size ()
                                               * sizeof (float)));
    }
}

void
WriteRowsCsv (const std::filesystem::path &path,
              const std::vector<LabelInfo> &labels,
              const std::vector<PrototypeRow> &prototypes)
{
  std::ofstream out (path);
  out << "row_id,label_index,label_id,source,kind,label,prototype_slot,prompt\n";
  for (std::size_t i = 0; i < prototypes.size (); ++i)
    {
      const auto &prototype = prototypes[i];
      const auto &label = labels[static_cast<std::size_t> (
          prototype.label_index)];
      out << i << ',' << prototype.label_index << ','
          << CsvEscape (label.record.label_id) << ','
          << CsvEscape (label.record.source) << ','
          << CsvEscape (label.record.kind) << ','
          << CsvEscape (label.record.label) << ',' << prototype.slot << ','
          << CsvEscape (prototype.prompt) << '\n';
    }
}

void
WriteLabelCsv (const std::filesystem::path &path,
               const std::vector<LabelInfo> &labels)
{
  std::ofstream out (path);
  out << "label_index,label_id,source,kind,label,prototype_begin,"
         "prototype_end\n";
  for (std::size_t i = 0; i < labels.size (); ++i)
    {
      const auto &label = labels[i];
      out << i << ',' << CsvEscape (label.record.label_id) << ','
          << CsvEscape (label.record.source) << ','
          << CsvEscape (label.record.kind) << ','
          << CsvEscape (label.record.label) << ',' << label.prototype_begin
          << ',' << label.prototype_end << '\n';
    }
}

void
WriteCases (const std::filesystem::path &path,
            const std::vector<EvalRow> &rows)
{
  std::ofstream out (path);
  out << "variant,label_id,prompt,rank,target_score,top_score,top_label_id,"
         "top_label,top1,top5,top10\n";
  for (const auto &row : rows)
    {
      out << row.variant << ',' << CsvEscape (row.label_id) << ','
          << CsvEscape (row.prompt) << ',' << row.rank << ','
          << row.target_score << ',' << row.top_score << ','
          << CsvEscape (row.top_label_id) << ',' << CsvEscape (row.top_label)
          << ',' << (row.top1 ? 1 : 0) << ',' << (row.top5 ? 1 : 0)
          << ',' << (row.top10 ? 1 : 0) << '\n';
    }
}

void
WriteFailures (const std::filesystem::path &path,
               const std::vector<EvalRow> &rows)
{
  std::ofstream out (path);
  out << "variant,label_id,prompt,rank,target_score,top_score,top_label_id,"
         "top_label\n";
  int written = 0;
  for (const auto &row : rows)
    {
      if (row.top1)
        {
          continue;
        }
      out << row.variant << ',' << CsvEscape (row.label_id) << ','
          << CsvEscape (row.prompt) << ',' << row.rank << ','
          << row.target_score << ',' << row.top_score << ','
          << CsvEscape (row.top_label_id) << ',' << CsvEscape (row.top_label)
          << '\n';
      ++written;
      if (written >= 80)
        {
          break;
        }
    }
}

std::vector<int>
PrototypeVariants (int max_prototypes)
{
  std::vector<int> variants;
  for (int value : { 1, 2, 4, 8, 16 })
    {
      if (value <= max_prototypes)
        {
          variants.push_back (value);
        }
    }
  if (variants.empty () || variants.back () != max_prototypes)
    {
      variants.push_back (max_prototypes);
    }
  return variants;
}

int
Main (int argc, char **argv)
{
  const Options opts = ParseArgs (argc, argv);
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
  std::vector<double> encode_ms;
  std::size_t encode_count = 0;
  std::vector<LabelInfo> labels;
  std::vector<PrototypeRow> prototypes;
  std::vector<EvalPrompt> eval_prompts;
  labels.reserve (records.size ());

  for (const auto &record : records)
    {
      const int prompt_count = static_cast<int> (record.prompts.size ());
      int candidate_count
          = std::min (opts.prototype_candidates_per_label, prompt_count);
      if (!opts.skip_eval && prompt_count > 1 && candidate_count >= prompt_count)
        {
          candidate_count = prompt_count - 1;
        }
      candidate_count = std::max (1, candidate_count);

      std::vector<std::vector<float>> candidate_embeddings;
      candidate_embeddings.reserve (static_cast<std::size_t> (candidate_count));
      for (int i = 0; i < candidate_count; ++i)
        {
          const auto encode_start = std::chrono::steady_clock::now ();
          candidate_embeddings.push_back (
              EncodeText256 (encoder, record.prompts[static_cast<std::size_t> (
                                           i)]));
          const auto encode_end = std::chrono::steady_clock::now ();
          encode_ms.push_back (Millis (encode_end - encode_start));
          ++encode_count;
        }

      LabelInfo label;
      label.record = record;
      label.centroid = MeanNormalized (candidate_embeddings);
      label.prototype_begin = static_cast<int> (prototypes.size ());
      const int label_index = static_cast<int> (labels.size ());
      const auto selected = SelectFarthestFirst (
          candidate_embeddings, opts.max_prototypes_per_label);
      for (int slot = 0; slot < static_cast<int> (selected.size ()); ++slot)
        {
          const int selected_index = selected[static_cast<std::size_t> (slot)];
          PrototypeRow row;
          row.label_index = label_index;
          row.slot = slot;
          row.prompt = record.prompts[static_cast<std::size_t> (selected_index)];
          row.embedding = candidate_embeddings[static_cast<std::size_t> (
              selected_index)];
          prototypes.push_back (std::move (row));
        }
      label.prototype_end = static_cast<int> (prototypes.size ());

      if (!opts.skip_eval && prompt_count > candidate_count)
        {
          int eval_added = 0;
          for (int i = candidate_count; i < prompt_count; ++i)
            {
              if (opts.max_eval_prompts_per_label > 0
                  && eval_added >= opts.max_eval_prompts_per_label)
                {
                  break;
                }
              const auto encode_start = std::chrono::steady_clock::now ();
              EvalPrompt eval;
              eval.label_index = label_index;
              eval.prompt = record.prompts[static_cast<std::size_t> (i)];
              eval.embedding = EncodeText256 (
                  encoder, record.prompts[static_cast<std::size_t> (i)]);
              const auto encode_end = std::chrono::steady_clock::now ();
              encode_ms.push_back (Millis (encode_end - encode_start));
              ++encode_count;
              eval_prompts.push_back (std::move (eval));
              ++eval_added;
            }
        }

      labels.push_back (std::move (label));
    }

  WriteRawF32Matrix (opts.output_dir / "label_vector_db_256.f32", prototypes);
  WriteRowsCsv (opts.output_dir / "label_vector_db_rows.csv", labels,
                prototypes);
  WriteLabelCsv (opts.output_dir / "label_vector_db_labels.csv", labels);

  std::vector<EvalRow> rows;
  if (!opts.skip_eval)
    {
      const auto variants = PrototypeVariants (opts.max_prototypes_per_label);
      for (const auto &eval : eval_prompts)
        {
          EvaluateCentroid (labels, eval, rows);
          for (int variant : variants)
            {
              EvaluatePrototypeMax (labels, prototypes, eval, variant, rows);
              if (variant >= 3)
                {
                  EvaluatePrototypeTopKMean (labels, prototypes, eval, variant,
                                             3, rows);
                }
            }
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
  const auto elapsed_ms = Millis (std::chrono::steady_clock::now () - start);
  const auto db_bytes = static_cast<std::uint64_t> (prototypes.size ())
                        * static_cast<std::uint64_t> (kDbDim)
                        * static_cast<std::uint64_t> (sizeof (float));

  if (!opts.skip_eval)
    {
      WriteCases (opts.output_dir / "label_vector_db_eval_cases.csv", rows);
      WriteFailures (opts.output_dir / "label_vector_db_failure_examples.csv",
                     rows);
    }

  Json results{
    { "status", "ok" },
    { "model_path", model_path.string () },
    { "source_records", opts.source_records.string () },
    { "source_filter", opts.source_filter },
    { "label_count", labels.size () },
    { "prototype_row_count", prototypes.size () },
    { "prototype_candidates_per_label", opts.prototype_candidates_per_label },
    { "max_prototypes_per_label", opts.max_prototypes_per_label },
    { "eval_prompt_count", eval_prompts.size () },
    { "dim", kDbDim },
    { "slice", Json::array ({ 0, kDbDim }) },
    { "db_bytes", db_bytes },
    { "db_mebibytes", static_cast<double> (db_bytes) / (1024.0 * 1024.0) },
    { "skip_eval", opts.skip_eval },
    { "metrics", opts.skip_eval ? Json::object () : SummarizeRows (rows) },
    { "encode_count", encode_count },
    { "mean_encode_ms", mean_ms },
    { "p50_encode_ms", percentile (0.50) },
    { "p95_encode_ms", percentile (0.95) },
    { "elapsed_ms", elapsed_ms },
  };
  std::ofstream results_out (opts.output_dir
                             / "label_vector_db_eval_results.json");
  results_out << std::setw (2) << results << '\n';

  Json manifest{
    { "format", "row_major_f32" },
    { "vector_file", "label_vector_db_256.f32" },
    { "rows_file", "label_vector_db_rows.csv" },
    { "labels_file", "label_vector_db_labels.csv" },
    { "dim", kDbDim },
    { "slice", Json::array ({ 0, kDbDim }) },
    { "row_count", prototypes.size () },
    { "label_count", labels.size () },
    { "bytes", db_bytes },
    { "mebibytes", static_cast<double> (db_bytes) / (1024.0 * 1024.0) },
    { "distance", "cosine_on_l2_normalized_vectors" },
    { "aggregation",
      "benchmark reports centroid, prototype max, and prototype top-k mean; "
      "the DB artifact stores rows only" },
    { "production_behavior_changed", false },
    { "benchmark_only", true },
  };
  std::ofstream manifest_out (opts.output_dir
                              / "label_vector_db_manifest.json");
  manifest_out << std::setw (2) << manifest << '\n';

  Json latency{
    { "encode_count", encode_count },
    { "mean_encode_ms", mean_ms },
    { "p50_encode_ms", percentile (0.50) },
    { "p95_encode_ms", percentile (0.95) },
    { "elapsed_ms", elapsed_ms },
  };
  std::ofstream latency_out (opts.output_dir / "label_vector_db_latency.json");
  latency_out << std::setw (2) << latency << '\n';

  Json summary{
    { "label_count", labels.size () },
    { "prototype_row_count", prototypes.size () },
    { "eval_prompt_count", eval_prompts.size () },
    { "source_filter", opts.source_filter },
    { "output_files",
      Json::array ({ "label_vector_db_256.f32",
                     "label_vector_db_rows.csv",
                     "label_vector_db_labels.csv",
                     "label_vector_db_manifest.json",
                     "label_vector_db_eval_results.json",
                     "label_vector_db_eval_cases.csv",
                     "label_vector_db_failure_examples.csv",
                     "label_vector_db_latency.json" }) },
  };
  std::ofstream summary_out (opts.output_dir / "label_vector_db_summary.json");
  summary_out << std::setw (2) << summary << '\n';

  std::cout << "label_vector_db_bench complete labels=" << labels.size ()
            << " prototypes=" << prototypes.size ()
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
  catch (const std::exception &e)
    {
      std::cerr << "label_vector_db_bench failed: " << e.what () << '\n';
      return 1;
    }
}
