#include <cortext/encoder/embeddinggemma.hpp>
#include <cortext/encoder/imagebind.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct Options
{
  std::string encoder = "gemma";
  std::filesystem::path models_dir = "models";
  int iterations = 50;
  int warmup = 5;
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
      << "  --encoder gemma|imagebind\n"
      << "  --models-dir <path>\n"
      << "  --iterations <n>\n"
      << "  --warmup <n>\n"
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
  return opts;
}

std::optional<GemmaResolved>
ResolveGemma (const std::filesystem::path &models_dir)
{
  const char *backend_override = std::getenv ("CORTEXT_EMBEDDINGGEMMA_BACKEND");
  const std::string backend
      = backend_override != nullptr && *backend_override != '\0'
            ? std::string (backend_override)
            : std::string ("onnx");

  std::vector<std::filesystem::path> roots{ models_dir };
  std::filesystem::path tokenizer_name = "tokenizer.model";
  bool require_tokenizer = true;
  std::vector<std::filesystem::path> model_candidates;

  if (backend == "onnx")
    {
      roots.push_back (models_dir / "embeddinggemma-300m-onnx");
      if (models_dir.filename () == "imagebind")
        {
          roots.push_back (models_dir.parent_path () / "embeddinggemma-300m-onnx");
        }
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
      if (models_dir.filename () == "imagebind")
        {
          roots.push_back (models_dir.parent_path () / "llama_cpp");
        }
      model_candidates = {
        std::filesystem::path ("embeddinggemma-300M-Q8_0.gguf"),
        std::filesystem::path ("embeddinggemma-300M-Q4_K_M.gguf"),
      };
    }
  else if (backend == "litert" || backend == "tflite")
    {
      tokenizer_name = "sentencepiece.model";
      roots.push_back (models_dir / "embeddinggemma-300m-litert");
      if (models_dir.filename () == "imagebind")
        {
          roots.push_back (models_dir.parent_path () / "embeddinggemma-300m-litert");
        }
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

std::filesystem::path
ResolveImageBindDir (const std::filesystem::path &models_dir)
{
  const std::vector<std::filesystem::path> roots = {
    models_dir,
    models_dir / "imagebind",
  };

  for (const auto &root : roots)
    {
      if (std::filesystem::exists (root / "text_encoder_int8.onnx")
          || std::filesystem::exists (root / "text_encoder.onnx"))
        {
          return root;
        }
    }

  throw std::runtime_error ("ImageBind models not found under "
                            + models_dir.string ());
}

double
ElapsedMillis (std::chrono::steady_clock::time_point start,
               std::chrono::steady_clock::time_point end)
{
  return std::chrono::duration<double, std::milli> (end - start).count ();
}

} // namespace

int
main (int argc, char *argv[])
{
  try
    {
      const Options opts = ParseArgs (argc, argv);

      std::unique_ptr<cortext::Encoder> encoder;
      std::string resolved_model;

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
          encoder = std::make_unique<cortext::EmbeddingGemmaEncoder> (
              std::move (cfg));
          resolved_model = resolved->model_path.string ();
        }
      else if (opts.encoder == "imagebind")
        {
          const auto resolved = ResolveImageBindDir (opts.models_dir);
          encoder
              = std::make_unique<cortext::ImageBindEncoder> (resolved.string ());
          resolved_model = resolved.string ();
        }
      else
        {
          throw std::runtime_error ("Unsupported encoder: " + opts.encoder);
        }

      std::vector<float> embedding;
      for (int i = 0; i < opts.warmup; ++i)
        {
          encoder->EncodeText (opts.text, embedding);
        }

      const auto start = std::chrono::steady_clock::now ();
      for (int i = 0; i < opts.iterations; ++i)
        {
          encoder->EncodeText (opts.text, embedding);
        }
      const auto end = std::chrono::steady_clock::now ();

      const double total_ms = ElapsedMillis (start, end);
      const double mean_ms = total_ms / static_cast<double> (opts.iterations);
      const double embeds_per_sec = mean_ms > 0.0 ? 1000.0 / mean_ms : 0.0;

      std::cout << std::fixed << std::setprecision (2);
      std::cout << "encoder=" << opts.encoder << "\n";
      std::cout << "resolved_model=" << resolved_model << "\n";
      std::cout << "iterations=" << opts.iterations << "\n";
      std::cout << "warmup=" << opts.warmup << "\n";
      std::cout << "embedding_dim=" << embedding.size () << "\n";
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
