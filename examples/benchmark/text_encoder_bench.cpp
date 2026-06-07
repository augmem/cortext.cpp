#include <cortext/encoder/embeddinggemma.hpp>
#include <cortext/models/aait_gguf_encoder.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct Options
{
  std::string encoder = "gemma";
  std::filesystem::path models_dir = "models";
  int iterations = 50;
  int warmup = 5;
  int parallelism = 1;
  bool vary_text = false;
  bool ess_aist_compare_fallback = false;
  std::string text
      = "This is a short embedding benchmark sentence for cortext.";
};

struct GemmaResolved
{
  std::filesystem::path model_path;
  std::filesystem::path tokenizer_path;
};

void
PrintUsage ()
{
  std::cout
      << "Usage: cortext_text_encoder_bench [options]\n"
      << "  --encoder gemma|ess-aist\n"
      << "  --models-dir <path>\n"
      << "  --iterations <n>\n"
      << "  --warmup <n>\n"
      << "  --parallelism <n>\n"
      << "  --vary-text\n"
      << "  --ess-aist-compare-fallback\n"
      << "  --text <string>\n";
}

Options
ParseArgs (int argc, char *argv[])
{
  Options opts;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if (arg == "--encoder" && i + 1 < argc)
        {
          opts.encoder = argv[++i];
        }
      else if (arg == "--models-dir" && i + 1 < argc)
        {
          opts.models_dir = argv[++i];
        }
      else if (arg == "--iterations" && i + 1 < argc)
        {
          opts.iterations = std::stoi (argv[++i]);
        }
      else if (arg == "--warmup" && i + 1 < argc)
        {
          opts.warmup = std::stoi (argv[++i]);
        }
      else if (arg == "--parallelism" && i + 1 < argc)
        {
          opts.parallelism = std::stoi (argv[++i]);
        }
      else if (arg == "--vary-text")
        {
          opts.vary_text = true;
        }
      else if (arg == "--ess-aist-compare-fallback")
        {
          opts.ess_aist_compare_fallback = true;
        }
      else if (arg == "--text" && i + 1 < argc)
        {
          opts.text = argv[++i];
        }
      else if (arg == "--help" || arg == "-h")
        {
          PrintUsage ();
          std::exit (0);
        }
      else
        {
          throw std::runtime_error ("Unknown argument: " + arg);
        }
    }
  opts.iterations = std::max (1, opts.iterations);
  opts.warmup = std::max (0, opts.warmup);
  opts.parallelism = std::max (1, opts.parallelism);
  return opts;
}

std::optional<GemmaResolved>
ResolveGemma (const std::filesystem::path &models_dir)
{
  const char *model_path_override
      = std::getenv ("CORTEXT_EMBEDDINGGEMMA_MODEL_PATH");
  if (model_path_override != nullptr && *model_path_override != '\0')
    {
      const std::filesystem::path model (model_path_override);
      if (std::filesystem::exists (model))
        {
          return GemmaResolved{ model, {} };
        }
    }

  const char *backend_override = std::getenv ("CORTEXT_EMBEDDINGGEMMA_BACKEND");
  const std::string backend
      = backend_override != nullptr && *backend_override != '\0'
            ? std::string (backend_override)
            : std::string ("llama.cpp");

  std::vector<std::filesystem::path> roots{ models_dir };
  std::filesystem::path tokenizer_name = "tokenizer.model";
  bool require_tokenizer = true;
  std::vector<std::filesystem::path> model_candidates;

  if (backend == "onnx")
    {
      roots.push_back (models_dir / "embeddinggemma-300m-onnx");
      model_candidates = {
        std::filesystem::path ("onnx/model_q4.onnx"),
        std::filesystem::path ("onnx/model_quantized.onnx"),
        std::filesystem::path ("onnx/model.onnx"),
      };
    }
  else if (backend == "llama.cpp" || backend == "llama")
    {
      require_tokenizer = false;
      roots.push_back (models_dir / "llama_cpp");
      model_candidates = {
        std::filesystem::path ("mdbr-leaf-ir-q8_0.gguf"),
        std::filesystem::path ("embeddinggemma-300M-Q8_0.gguf"),
        std::filesystem::path ("embeddinggemma-300M-Q4_K_M.gguf"),
      };
    }
  else if (backend == "litert" || backend == "tflite")
    {
      tokenizer_name = "sentencepiece.model";
      roots.push_back (models_dir / "embeddinggemma-300m-litert");
      model_candidates = {
        std::filesystem::path ("embeddinggemma-300M_seq256_mixed-precision.tflite"),
      };
    }
  else
    {
      throw std::runtime_error ("Unsupported EmbeddingGemma backend: " + backend);
    }

  for (const auto &root : roots)
    {
      const auto tokenizer
          = require_tokenizer ? root / tokenizer_name : std::filesystem::path ();
      if (require_tokenizer && !std::filesystem::exists (tokenizer))
        {
          continue;
        }
      for (const auto &model_rel : model_candidates)
        {
          const auto model = root / model_rel;
          if (std::filesystem::exists (model))
            {
              return GemmaResolved{ model, tokenizer };
            }
        }
    }

  return std::nullopt;
}

double
ElapsedMillis (std::chrono::steady_clock::time_point start,
               std::chrono::steady_clock::time_point end)
{
  return std::chrono::duration<double, std::milli> (end - start).count ();
}

void
SetEnvValue (const char *name, const char *value)
{
#if defined(_WIN32)
  _putenv_s (name, value);
#else
  setenv (name, value, 1);
#endif
}

void
UnsetEnvValue (const char *name)
{
#if defined(_WIN32)
  _putenv_s (name, "");
#else
  unsetenv (name);
#endif
}

double
VectorNorm (const std::vector<float> &values)
{
  double norm_sq = 0.0;
  for (const float value : values)
    {
      norm_sq += static_cast<double> (value) * static_cast<double> (value);
    }
  return std::sqrt (norm_sq);
}

double
VectorChecksum (const std::vector<float> &values)
{
  double checksum = 0.0;
  for (std::size_t i = 0; i < values.size (); ++i)
    {
      checksum += static_cast<double> (values[i])
                  * static_cast<double> (i + 1);
    }
  return checksum;
}

void
EncodeOnce (cortext::Encoder &encoder, const Options &opts, int iteration,
            std::vector<float> &embedding)
{
  const std::string text = opts.vary_text
                               ? opts.text + " iteration "
                                     + std::to_string (iteration)
                               : opts.text;
  encoder.EncodeText (text, embedding);
}

struct VectorDiff
{
  double cosine = 0.0;
  double l2_diff = 0.0;
  double max_abs_diff = 0.0;
};

VectorDiff
CompareVectors (const std::vector<float> &a, const std::vector<float> &b)
{
  if (a.size () != b.size ())
    {
      throw std::runtime_error ("Cannot compare embeddings with different "
                                "dimensions");
    }
  double dot = 0.0;
  double a_norm_sq = 0.0;
  double b_norm_sq = 0.0;
  double diff_norm_sq = 0.0;
  double max_abs_diff = 0.0;
  for (std::size_t i = 0; i < a.size (); ++i)
    {
      const double av = static_cast<double> (a[i]);
      const double bv = static_cast<double> (b[i]);
      const double diff = av - bv;
      dot += av * bv;
      a_norm_sq += av * av;
      b_norm_sq += bv * bv;
      diff_norm_sq += diff * diff;
      max_abs_diff = std::max (max_abs_diff, std::abs (diff));
    }
  VectorDiff result;
  const double denom = std::sqrt (a_norm_sq) * std::sqrt (b_norm_sq);
  result.cosine = denom > 0.0 ? dot / denom : 0.0;
  result.l2_diff = std::sqrt (diff_norm_sq);
  result.max_abs_diff = max_abs_diff;
  return result;
}

std::unique_ptr<cortext::AaitGgufEncoder>
MakeEssAistEncoder (const Options &opts, std::string &resolved_model)
{
  const auto resolved = cortext::ResolveEssAistGgufModelPath (
      opts.models_dir);
  if (!resolved.has_value ())
    {
      throw std::runtime_error ("ESS-AIST GGUF assets not found under "
                                + opts.models_dir.string ());
    }
  cortext::AaitGgufConfig cfg;
  cfg.model_path = resolved->string ();
  cfg.context_length = 128;
  resolved_model = resolved->string ();
  return std::make_unique<cortext::AaitGgufEncoder> (std::move (cfg));
}

std::unique_ptr<cortext::Encoder>
MakeEncoder (const Options &opts, std::string &resolved_model,
             cortext::AaitGgufEncoder **ess_aist_encoder = nullptr)
{
  if (ess_aist_encoder != nullptr)
    {
      *ess_aist_encoder = nullptr;
    }
  if (opts.encoder == "gemma")
    {
      const auto resolved = ResolveGemma (opts.models_dir);
      if (!resolved.has_value ())
        {
          throw std::runtime_error (
              "EmbeddingGemma assets not found under "
              + opts.models_dir.string ());
        }
      cortext::EmbeddingGemmaConfig cfg;
      cfg.model_path = resolved->model_path.string ();
      cfg.tokenizer_path = resolved->tokenizer_path.string ();
      resolved_model = resolved->model_path.string ();
      return std::make_unique<cortext::EmbeddingGemmaEncoder> (std::move (cfg));
    }
  if (opts.encoder == "ess-aist")
    {
      auto ess = MakeEssAistEncoder (opts, resolved_model);
      if (ess_aist_encoder != nullptr)
        {
          *ess_aist_encoder = ess.get ();
        }
      return ess;
    }
  throw std::runtime_error ("Unsupported encoder: " + opts.encoder);
}

int
RunEssAistFallbackCompare (const Options &opts)
{
  if (opts.encoder != "ess-aist")
    {
      throw std::runtime_error (
          "--ess-aist-compare-fallback requires --encoder ess-aist");
    }

  std::string fallback_model;
  SetEnvValue ("CORTEXT_AAIT_DISABLE_FULL_GGML_GRAPH", "1");
  UnsetEnvValue ("CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH");
  auto fallback = MakeEssAistEncoder (opts, fallback_model);
  std::vector<float> fallback_embedding;
  const auto fallback_start = std::chrono::steady_clock::now ();
  fallback->EncodeText (opts.text, fallback_embedding);
  const auto fallback_end = std::chrono::steady_clock::now ();
  const std::string fallback_backend = fallback->KernelOpsBackend ();
  const std::string fallback_granularity = fallback->KernelOpsGranularity ();

  std::string full_graph_model;
  UnsetEnvValue ("CORTEXT_AAIT_DISABLE_FULL_GGML_GRAPH");
  SetEnvValue ("CORTEXT_AAIT_REQUIRE_FULL_GGML_GRAPH", "1");
  auto full_graph = MakeEssAistEncoder (opts, full_graph_model);
  std::vector<float> full_graph_embedding;
  const auto full_start = std::chrono::steady_clock::now ();
  full_graph->EncodeText (opts.text, full_graph_embedding);
  const auto full_end = std::chrono::steady_clock::now ();
  const std::string full_graph_backend = full_graph->KernelOpsBackend ();
  const std::string full_graph_granularity =
      full_graph->KernelOpsGranularity ();

  const VectorDiff diff = CompareVectors (fallback_embedding,
                                          full_graph_embedding);
  std::cout << std::fixed << std::setprecision (6);
  std::cout << "mode=ess_aist_full_graph_parity\n";
  std::cout << "fallback_model=" << fallback_model << "\n";
  std::cout << "full_graph_model=" << full_graph_model << "\n";
  std::cout << "embedding_dim=" << full_graph_embedding.size () << "\n";
  std::cout << "fallback_backend=" << fallback_backend << "\n";
  std::cout << "fallback_granularity=" << fallback_granularity << "\n";
  std::cout << "full_graph_backend=" << full_graph_backend << "\n";
  std::cout << "full_graph_granularity=" << full_graph_granularity << "\n";
  std::cout << "full_graph_used="
            << (full_graph->UsesFullTextGraphOps () ? "true" : "false")
            << "\n";
  std::cout << "fallback_norm=" << VectorNorm (fallback_embedding) << "\n";
  std::cout << "full_graph_norm=" << VectorNorm (full_graph_embedding)
            << "\n";
  std::cout << "fallback_checksum="
            << VectorChecksum (fallback_embedding) << "\n";
  std::cout << "full_graph_checksum="
            << VectorChecksum (full_graph_embedding) << "\n";
  std::cout << "cosine=" << diff.cosine << "\n";
  std::cout << "l2_diff=" << diff.l2_diff << "\n";
  std::cout << "max_abs_diff=" << diff.max_abs_diff << "\n";
  std::cout << "fallback_ms=" << ElapsedMillis (fallback_start, fallback_end)
            << "\n";
  std::cout << "full_graph_first_call_ms="
            << ElapsedMillis (full_start, full_end) << "\n";
  if (!full_graph->FullTextGraphError ().empty ())
    {
      std::cout << "full_graph_error="
                << full_graph->FullTextGraphError () << "\n";
    }
  return 0;
}

} // namespace

int
main (int argc, char *argv[])
{
  try
    {
      const Options opts = ParseArgs (argc, argv);
      if (opts.ess_aist_compare_fallback)
        {
          return RunEssAistFallbackCompare (opts);
        }

      cortext::AaitGgufEncoder *ess_aist_encoder = nullptr;
      std::string resolved_model;
      std::unique_ptr<cortext::Encoder> encoder =
          MakeEncoder (opts, resolved_model, &ess_aist_encoder);

      std::vector<float> embedding;
      std::chrono::steady_clock::time_point start;
      std::chrono::steady_clock::time_point end;
      if (opts.parallelism == 1)
        {
          for (int i = 0; i < opts.warmup; ++i)
            {
              EncodeOnce (*encoder, opts, i, embedding);
            }

          start = std::chrono::steady_clock::now ();
          for (int i = 0; i < opts.iterations; ++i)
            {
              EncodeOnce (*encoder, opts, i, embedding);
            }
          end = std::chrono::steady_clock::now ();
        }
      else
        {
          std::vector<std::unique_ptr<cortext::Encoder>> encoders;
          std::vector<std::vector<float>> embeddings (
              static_cast<std::size_t> (opts.parallelism));
          encoders.reserve (static_cast<std::size_t> (opts.parallelism));
          encoders.push_back (std::move (encoder));
          for (int i = 1; i < opts.parallelism; ++i)
            {
              std::string worker_model;
              encoders.push_back (MakeEncoder (opts, worker_model));
            }
          for (int worker = 0; worker < opts.parallelism; ++worker)
            {
              for (int i = 0; i < opts.warmup; ++i)
                {
                  EncodeOnce (
                      *encoders[static_cast<std::size_t> (worker)], opts, i,
                      embeddings[static_cast<std::size_t> (worker)]);
                }
            }
          std::atomic<int> next{ 0 };
          start = std::chrono::steady_clock::now ();
          std::vector<std::thread> threads;
          threads.reserve (static_cast<std::size_t> (opts.parallelism));
          for (int worker = 0; worker < opts.parallelism; ++worker)
            {
              threads.emplace_back ([&, worker] {
                for (;;)
                  {
                    const int iteration = next.fetch_add (
                        1, std::memory_order_relaxed);
                    if (iteration >= opts.iterations)
                      {
                        break;
                      }
                    EncodeOnce (
                        *encoders[static_cast<std::size_t> (worker)], opts,
                        iteration,
                        embeddings[static_cast<std::size_t> (worker)]);
                  }
              });
            }
          for (std::thread &thread : threads)
            {
              thread.join ();
            }
          end = std::chrono::steady_clock::now ();
          embedding = embeddings.front ();
        }

      const double total_ms = ElapsedMillis (start, end);
      const double mean_ms = total_ms / static_cast<double> (opts.iterations);
      const double embeds_per_sec = mean_ms > 0.0 ? 1000.0 / mean_ms : 0.0;
      double norm_sq = 0.0;
      double checksum = 0.0;
      for (std::size_t i = 0; i < embedding.size (); ++i)
        {
          const double value = static_cast<double> (embedding[i]);
          norm_sq += value * value;
          checksum += value * static_cast<double> (i + 1);
        }

      std::cout << std::fixed << std::setprecision (2);
      std::cout << "encoder=" << opts.encoder << "\n";
      std::cout << "modality=text\n";
      std::cout << "resolved_model=" << resolved_model << "\n";
      std::cout << "iterations=" << opts.iterations << "\n";
      std::cout << "warmup=" << opts.warmup << "\n";
      std::cout << "parallelism=" << opts.parallelism << "\n";
      std::cout << "throughput_mode="
                << (opts.parallelism == 1 ? "serial"
                                           : "parallel_frame_workers")
                << "\n";
      std::cout << "vary_text=" << (opts.vary_text ? "true" : "false")
                << "\n";
      std::cout << "embedding_dim=" << embedding.size () << "\n";
      std::cout << "embedding_norm=" << std::sqrt (norm_sq) << "\n";
      std::cout << "embedding_checksum=" << checksum << "\n";
      if (ess_aist_encoder != nullptr)
        {
          std::cout << "kernel_ops_backend="
                    << ess_aist_encoder->KernelOpsBackend () << "\n";
          std::cout << "kernel_ops_granularity="
                    << ess_aist_encoder->KernelOpsGranularity () << "\n";
          std::cout << "full_text_graph_used="
                    << (ess_aist_encoder->UsesFullTextGraphOps () ? "true"
                                                                   : "false")
                    << "\n";
          if (!ess_aist_encoder->FullTextGraphError ().empty ())
            {
              std::cout << "full_text_graph_error="
                        << ess_aist_encoder->FullTextGraphError () << "\n";
            }
        }
      std::cout << "total_ms=" << total_ms << "\n";
      std::cout << "mean_ms=" << mean_ms << "\n";
      std::cout << "embeddings_per_sec=" << embeds_per_sec << "\n";
      return 0;
    }
  catch (const std::exception &e)
    {
      std::cerr << "cortext_text_encoder_bench failed: " << e.what () << "\n";
      return 1;
    }
}
