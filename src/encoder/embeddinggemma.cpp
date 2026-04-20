#include "cortext/encoder/embeddinggemma.hpp"

#include "embeddinggemma_profile.hpp"
#include "../deep_llm/llama_cpp_support.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(CORTEXT_DISABLE_LITERT)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wc99-extensions"
#endif
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT)
#include <onnxruntime/onnxruntime_cxx_api.h>
#if defined(__APPLE__)
#include <onnxruntime/coreml_provider_factory.h>
#endif

#include "cortext/core/thread_config.hpp"
#endif

#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT) || !defined(CORTEXT_DISABLE_LITERT)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wc99-extensions"
#endif
#include "sentencepiece_processor.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_LLAMA_CPP)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wc99-extensions"
#endif
#include CORTEXT_GGML_BACKEND_HEADER_PATH
#include CORTEXT_LLAMA_HEADER_PATH
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

namespace cortext
{

namespace
{

using Clock = std::chrono::steady_clock;
constexpr std::size_t kTargetEmbeddingDim = 256;

enum class EmbeddingGemmaBackendKind
{
  Onnx,
  LlamaCpp,
  LiteRt
};

[[maybe_unused]] double
ElapsedMillis (Clock::time_point start, Clock::time_point end)
{
  return std::chrono::duration<double, std::milli> (end - start).count ();
}

std::string
GetEnvOrDefault (const char *name, const std::string &fallback = {})
{
  const char *value = std::getenv (name);
  if (value == nullptr || *value == '\0')
    {
      return fallback;
    }
  return value;
}

int
GetEnvInt (const char *name, int fallback)
{
  const std::string value = GetEnvOrDefault (name);
  if (value.empty ())
    {
      return fallback;
    }
  try
    {
      return std::stoi (value);
    }
  catch (...)
    {
      return fallback;
    }
}

EmbeddingGemmaBackendKind
ResolveBackendKind (const EmbeddingGemmaConfig &config)
{
  const std::string override = GetEnvOrDefault ("CORTEXT_EMBEDDINGGEMMA_BACKEND");
  if (!override.empty ())
    {
      if (override == "onnx")
        {
          return EmbeddingGemmaBackendKind::Onnx;
        }
      if (override == "llama.cpp" || override == "llama")
        {
          return EmbeddingGemmaBackendKind::LlamaCpp;
        }
      if (override == "litert" || override == "tflite")
        {
          return EmbeddingGemmaBackendKind::LiteRt;
        }
      throw std::runtime_error ("EmbeddingGemmaEncoder: unsupported backend override '"
                                + override + "'");
    }

  const std::filesystem::path model_path (config.model_path);
  const std::string ext = model_path.extension ().string ();
  if (ext == ".onnx")
    {
      return EmbeddingGemmaBackendKind::Onnx;
    }
  if (ext == ".gguf")
    {
      return EmbeddingGemmaBackendKind::LlamaCpp;
    }
  if (ext == ".tflite")
    {
      return EmbeddingGemmaBackendKind::LiteRt;
    }

  throw std::runtime_error ("EmbeddingGemmaEncoder: cannot infer backend from model path "
                            + config.model_path);
}

void
L2NormalizeInPlace (std::vector<float> &values)
{
  double sum_sq = 0.0;
  for (float value : values)
    {
      sum_sq += static_cast<double> (value) * static_cast<double> (value);
    }
  if (sum_sq <= 0.0)
    {
      return;
    }
  const double norm = std::sqrt (sum_sq);
  for (float &value : values)
    {
      value = static_cast<float> (static_cast<double> (value) / norm);
    }
}

void
StandardizeEmbeddingDim (std::vector<float> &values, std::size_t dim)
{
  if (values.size () > dim)
    {
      values.resize (dim);
    }
  else if (values.size () < dim)
    {
      values.resize (dim, 0.0f);
    }
}

void
FinalizeEmbedding (std::vector<float> &values)
{
  StandardizeEmbeddingDim (values, kTargetEmbeddingDim);
  L2NormalizeInPlace (values);
}

[[maybe_unused]] std::vector<int>
PadTokensInt (std::vector<int> ids, int max_length, int bos_id, int eos_id,
              int pad_id)
{
  if (max_length <= 0)
    {
      return {};
    }
  if (bos_id >= 0 && eos_id >= 0 && max_length >= 2)
    {
      const int max_len = max_length - 2;
      if (static_cast<int> (ids.size ()) > max_len)
        ids.resize (static_cast<std::size_t> (max_len));
      ids.insert (ids.begin (), bos_id);
      ids.push_back (eos_id);
    }
  else if (static_cast<int> (ids.size ()) > max_length)
    {
      ids.resize (static_cast<std::size_t> (max_length));
    }

  if (pad_id < 0)
    {
      pad_id = 0;
    }

  while (static_cast<int> (ids.size ()) < max_length)
    {
      ids.push_back (pad_id);
    }
  return ids;
}

std::string
ResolveExistingPath (const std::string &path)
{
  if (path.empty ())
    {
      throw std::runtime_error ("EmbeddingGemmaEncoder: model/tokenizer path is empty");
    }
  std::filesystem::path p (path);
  if (!std::filesystem::exists (p))
    {
      throw std::runtime_error ("EmbeddingGemmaEncoder: file not found: " + path);
    }
  return p.string ();
}

#if !defined(CORTEXT_DISABLE_LITERT)
void
ThrowIfLiteRtError (LiteRtStatus status, const std::string &context)
{
  if (status == kLiteRtStatusOk)
    {
      return;
    }
  throw std::runtime_error (context + ": " + LiteRtGetStatusString (status));
}
#endif

struct EmbeddingGemmaTextEncodeProfileState
{
  std::mutex mu;
  cortext::internal::EmbeddingGemmaTextEncodeProfileSnapshot snapshot;
};

EmbeddingGemmaTextEncodeProfileState &
GetEmbeddingGemmaTextEncodeProfileState ()
{
  static EmbeddingGemmaTextEncodeProfileState state;
  return state;
}

[[maybe_unused]] void
RecordEmbeddingGemmaTextEncodeProfile (
    double ensure_initialized_ms, double tokenize_ms, double tensor_create_ms,
    double run_ms, double copy_ms, double normalize_ms)
{
  auto &state = GetEmbeddingGemmaTextEncodeProfileState ();
  std::lock_guard<std::mutex> lock (state.mu);
  auto &snapshot = state.snapshot;
  ++snapshot.calls;
  snapshot.ensure_initialized_ms += ensure_initialized_ms;
  snapshot.tokenize_ms += tokenize_ms;
  snapshot.tensor_create_ms += tensor_create_ms;
  snapshot.run_ms += run_ms;
  snapshot.copy_ms += copy_ms;
  snapshot.normalize_ms += normalize_ms;
}

class EmbeddingGemmaBackend
{
public:
  explicit EmbeddingGemmaBackend (EmbeddingGemmaConfig config_in)
      : config_ (std::move (config_in))
  {
  }

  virtual ~EmbeddingGemmaBackend () = default;

  virtual void EncodeText (const std::string &text,
                           std::vector<float> &out_embedding)
      = 0;

protected:
  const EmbeddingGemmaConfig &Config () const { return config_; }

private:
  EmbeddingGemmaConfig config_;
};

#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT)

std::vector<int64_t>
PadTokens (std::vector<int> ids, int max_length, int bos_id, int eos_id,
           int pad_id)
{
  if (max_length <= 0)
    {
      return {};
    }
  if (bos_id >= 0 && eos_id >= 0 && max_length >= 2)
    {
      const int max_len = max_length - 2;
      if (static_cast<int> (ids.size ()) > max_len)
        ids.resize (static_cast<std::size_t> (max_len));
      ids.insert (ids.begin (), bos_id);
      ids.push_back (eos_id);
    }
  else if (static_cast<int> (ids.size ()) > max_length)
    {
      ids.resize (static_cast<std::size_t> (max_length));
    }

  if (pad_id < 0)
    {
      pad_id = 0;
    }

  std::vector<int64_t> out;
  out.reserve (static_cast<std::size_t> (max_length));
  for (int id : ids)
    {
      out.push_back (static_cast<int64_t> (id));
    }
  while (static_cast<int> (out.size ()) < max_length)
    {
      out.push_back (static_cast<int64_t> (pad_id));
    }
  return out;
}

std::vector<int64_t>
BuildAttentionMask (const std::vector<int64_t> &tokens, int pad_id)
{
  std::vector<int64_t> mask;
  mask.reserve (tokens.size ());
  for (int64_t token : tokens)
    {
      mask.push_back (token == pad_id ? 0 : 1);
    }
  return mask;
}

bool
TryAppendCoreMLProvider (Ort::SessionOptions &opts, std::string *error_out)
{
#if defined(__APPLE__)
  OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_CoreML (
      opts, 0 /* default: allow ANE/CPU/GPU */);
  if (status != nullptr)
    {
      if (error_out != nullptr)
        {
          *error_out = Ort::GetApi ().GetErrorMessage (status);
        }
      Ort::GetApi ().ReleaseStatus (status);
      return false;
    }
  return true;
#else
  if (error_out != nullptr)
    {
      *error_out = "CoreML execution provider only available on Apple platforms";
    }
  return false;
#endif
}

void
ConfigureSessionOptions (Ort::SessionOptions &opts, int threads)
{
  opts.SetIntraOpNumThreads (threads);
  opts.SetInterOpNumThreads (1);
  opts.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);
}

class EmbeddingGemmaOnnxBackend final : public EmbeddingGemmaBackend
{
public:
  explicit EmbeddingGemmaOnnxBackend (EmbeddingGemmaConfig config)
      : EmbeddingGemmaBackend (std::move (config))
  {
  }

  void EncodeText (const std::string &text,
                   std::vector<float> &out_embedding) override
  {
    const auto t_ensure_start = Clock::now ();
    EnsureInitialized ();
    const auto t_ensure_end = Clock::now ();

    const auto t_tokenize_start = Clock::now ();
    std::vector<int> token_ids_raw;
    auto encode_status = tokenizer_->Encode (text, &token_ids_raw);
    if (!encode_status.ok ())
      {
        throw std::runtime_error ("EmbeddingGemmaEncoder: tokenization failed: "
                                  + std::string (encode_status.ToString ()));
      }
    const auto t_tokenize_end = Clock::now ();

    const int max_length = Config ().max_length;
    if (max_length <= 0)
      {
        throw std::runtime_error ("EmbeddingGemmaEncoder: max_length must be > 0");
      }
    std::vector<int64_t> token_ids
        = PadTokens (std::move (token_ids_raw), max_length, bos_id_, eos_id_,
                     pad_id_);
    std::vector<int64_t> attention_mask
        = BuildAttentionMask (token_ids, pad_id_);

    std::array<int64_t, 2> shape{
      1, static_cast<int64_t> (token_ids.size ())
    };
    const auto t_tensor_start = Clock::now ();
    Ort::Value ids_tensor = Ort::Value::CreateTensor<int64_t> (
        memory_info_, token_ids.data (), token_ids.size (), shape.data (),
        shape.size ());
    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t> (
        memory_info_, attention_mask.data (), attention_mask.size (),
        shape.data (), shape.size ());

    std::vector<const char *> input_names;
    std::vector<Ort::Value> input_tensors;
    input_names.push_back (input_ids_name_.c_str ());
    input_tensors.push_back (std::move (ids_tensor));
    if (!attention_mask_name_.empty ())
      {
        input_names.push_back (attention_mask_name_.c_str ());
        input_tensors.push_back (std::move (mask_tensor));
      }
    const auto t_tensor_end = Clock::now ();

    std::vector<const char *> output_names{ output_name_.c_str () };
    const auto t_run_start = Clock::now ();
    auto outputs = session_->Run (Ort::RunOptions{ nullptr }, input_names.data (),
                                  input_tensors.data (), input_tensors.size (),
                                  output_names.data (), output_names.size ());
    const auto t_run_end = Clock::now ();
    if (outputs.empty ())
      {
        throw std::runtime_error ("EmbeddingGemmaEncoder: no output tensors");
      }

    auto &out = outputs.front ();
    if (!out.IsTensor ())
      {
        throw std::runtime_error ("EmbeddingGemmaEncoder: output is not a tensor");
      }

    const auto t_copy_start = Clock::now ();
    const float *data = out.GetTensorData<float> ();
    auto info = out.GetTensorTypeAndShapeInfo ();
    const auto shape_info = info.GetShape ();
    std::size_t copy_count = 0;
    if (shape_info.empty ())
      {
        copy_count = 1;
      }
    else if (shape_info.size () == 1)
      {
        copy_count = static_cast<std::size_t> (
            std::max<int64_t> (shape_info[0], 0));
      }
    else if (shape_info.size () == 2)
      {
        copy_count = static_cast<std::size_t> (
            std::max<int64_t> (shape_info[1], 0));
      }
    else
      {
        throw std::runtime_error (
            "EmbeddingGemmaEncoder: unexpected output rank for embedding tensor");
      }
    if (copy_count == 0)
      {
        throw std::runtime_error ("EmbeddingGemmaEncoder: empty output tensor");
      }

    out_embedding.assign (data, data + copy_count);
    const auto t_normalize_start = Clock::now ();
    FinalizeEmbedding (out_embedding);
    const auto t_normalize_end = Clock::now ();
    const auto t_copy_end = Clock::now ();

    RecordEmbeddingGemmaTextEncodeProfile (
        ElapsedMillis (t_ensure_start, t_ensure_end),
        ElapsedMillis (t_tokenize_start, t_tokenize_end),
        ElapsedMillis (t_tensor_start, t_tensor_end),
        ElapsedMillis (t_run_start, t_run_end),
        ElapsedMillis (t_copy_start, t_copy_end),
        ElapsedMillis (t_normalize_start, t_normalize_end));
  }

private:
  void EnsureInitialized ()
  {
    std::lock_guard<std::mutex> lock (init_mu_);
    if (initialized_)
      {
        return;
      }
    if (init_attempted_)
      {
        throw std::runtime_error (init_error_);
      }
    init_attempted_ = true;

    try
      {
        const std::string model_path = ResolveExistingPath (Config ().model_path);
        const std::string tokenizer_path
            = ResolveExistingPath (Config ().tokenizer_path);

        tokenizer_ = std::make_unique<sentencepiece::SentencePieceProcessor> ();
        auto load_status = tokenizer_->Load (tokenizer_path);
        if (!load_status.ok ())
          {
            throw std::runtime_error (
                "EmbeddingGemmaEncoder: failed to load tokenizer: "
                + std::string (load_status.ToString ()));
          }
        bos_id_ = tokenizer_->bos_id ();
        eos_id_ = tokenizer_->eos_id ();
        pad_id_ = tokenizer_->pad_id ();

        const int threads
            = Config ().num_threads > 0
                  ? Config ().num_threads
                  : static_cast<int> (core::GetEmbedThreadCount ());
#if defined(__APPLE__)
        std::string provider_error;
        Ort::SessionOptions coreml_opts;
        ConfigureSessionOptions (coreml_opts, threads);
        if (TryAppendCoreMLProvider (coreml_opts, &provider_error))
          {
            session_ = std::make_unique<Ort::Session> (env_, model_path.c_str (),
                                                       coreml_opts);
          }
        else
#endif
          {
            Ort::SessionOptions cpu_opts;
            ConfigureSessionOptions (cpu_opts, threads);
            session_ = std::make_unique<Ort::Session> (env_, model_path.c_str (),
                                                       cpu_opts);
          }

        Ort::AllocatorWithDefaultOptions allocator;
        if (session_->GetInputCount () < 1)
          {
            throw std::runtime_error ("EmbeddingGemmaEncoder: model has no inputs");
          }

        for (std::size_t i = 0; i < session_->GetInputCount (); ++i)
          {
            auto name = session_->GetInputNameAllocated (i, allocator);
            if (!name)
              {
                continue;
              }
            const std::string input_name = name.get ();
            if (input_name == "input_ids")
              {
                input_ids_name_ = input_name;
              }
            else if (input_name == "attention_mask")
              {
                attention_mask_name_ = input_name;
              }
          }
        if (input_ids_name_.empty ())
          {
            auto name = session_->GetInputNameAllocated (0, allocator);
            if (name)
              {
                input_ids_name_ = name.get ();
              }
          }

        for (std::size_t i = 0; i < session_->GetOutputCount (); ++i)
          {
            auto name = session_->GetOutputNameAllocated (i, allocator);
            if (!name)
              {
                continue;
              }
            const std::string candidate = name.get ();
            if (candidate == "sentence_embedding")
              {
                output_name_ = candidate;
                break;
              }
            if (output_name_.empty ())
              {
                output_name_ = candidate;
              }
          }
        if (output_name_.empty ())
          {
            throw std::runtime_error ("EmbeddingGemmaEncoder: output name missing");
          }

        initialized_ = true;
      }
    catch (const std::exception &e)
      {
        init_error_
            = std::string ("EmbeddingGemmaEncoder init failed: ") + e.what ();
        throw;
      }
  }

  Ort::Env env_{ ORT_LOGGING_LEVEL_WARNING, "cortext-embeddinggemma" };
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer_;
  Ort::MemoryInfo memory_info_
      = Ort::MemoryInfo::CreateCpu (OrtDeviceAllocator, OrtMemTypeCPU);
  std::string input_ids_name_;
  std::string attention_mask_name_;
  std::string output_name_;
  int bos_id_ = -1;
  int eos_id_ = -1;
  int pad_id_ = 0;
  std::mutex init_mu_;
  bool init_attempted_ = false;
  bool initialized_ = false;
  std::string init_error_;
};

#endif

class EmbeddingGemmaLlamaCppBackend final : public EmbeddingGemmaBackend
{
public:
  explicit EmbeddingGemmaLlamaCppBackend (EmbeddingGemmaConfig config)
      : EmbeddingGemmaBackend (std::move (config))
  {
  }

  ~EmbeddingGemmaLlamaCppBackend () override
  {
#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_LLAMA_CPP)
    if (batch_.token != nullptr || batch_.embd != nullptr)
      {
        llama_batch_free (batch_);
      }
    if (ctx_ != nullptr)
      {
        llama_free (ctx_);
      }
    if (model_ != nullptr)
      {
        llama_model_free (model_);
      }
#endif
  }

  void EncodeText (const std::string &text,
                   std::vector<float> &out_embedding) override
  {
    EnsureInitialized ();

#if !defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_LLAMA_CPP)
    (void)text;
    (void)out_embedding;
    throw std::runtime_error (
        "EmbeddingGemma llama.cpp backend unavailable: link libllama");
#else
    std::vector<llama_token> tokens = Tokenize (text);
    if (tokens.empty ())
      {
        out_embedding.assign (kTargetEmbeddingDim, 0.0f);
        return;
      }

    if (static_cast<int> (tokens.size ()) > Config ().max_length)
      {
        tokens.resize (static_cast<std::size_t> (Config ().max_length));
      }

    batch_.n_tokens = static_cast<int32_t> (tokens.size ());
    for (int32_t i = 0; i < batch_.n_tokens; ++i)
      {
        batch_.token[i] = tokens[static_cast<std::size_t> (i)];
        batch_.pos[i] = i;
        batch_.n_seq_id[i] = 1;
        batch_.seq_id[i][0] = 0;
        batch_.logits[i] = 1;
      }

    const int rc = llama_encode (ctx_, batch_);
    if (rc != 0)
      {
        throw std::runtime_error ("EmbeddingGemma llama.cpp encode failed: "
                                  + std::to_string (rc));
      }

    float *embedding = nullptr;
    switch (pooling_type_)
      {
      case LLAMA_POOLING_TYPE_CLS:
      case LLAMA_POOLING_TYPE_MEAN:
      case LLAMA_POOLING_TYPE_LAST:
        embedding = llama_get_embeddings_seq (ctx_, 0);
        break;
      case LLAMA_POOLING_TYPE_NONE:
        embedding = llama_get_embeddings_ith (ctx_, -1);
        break;
      default:
        throw std::runtime_error (
            "EmbeddingGemma llama.cpp backend does not support pooling type "
            + std::to_string (static_cast<int> (pooling_type_)));
      }

    if (embedding == nullptr)
      {
        throw std::runtime_error (
            "EmbeddingGemma llama.cpp backend returned no embedding");
      }

    out_embedding.assign (embedding, embedding + embedding_dim_);
    FinalizeEmbedding (out_embedding);
#endif
  }

private:
  static enum llama_pooling_type
  ResolvePoolingType ()
  {
    const char *pooling_env = std::getenv ("CORTEXT_LLAMA_CPP_POOLING");
    if (pooling_env == nullptr || std::string (pooling_env).empty ())
      {
        return LLAMA_POOLING_TYPE_UNSPECIFIED;
      }
    const std::string pooling = pooling_env;
    if (pooling == "cls")
      {
        return LLAMA_POOLING_TYPE_CLS;
      }
    if (pooling == "mean")
      {
        return LLAMA_POOLING_TYPE_MEAN;
      }
    if (pooling == "last")
      {
        return LLAMA_POOLING_TYPE_LAST;
      }
    if (pooling == "none")
      {
        return LLAMA_POOLING_TYPE_NONE;
      }
    if (pooling == "model")
      {
        return LLAMA_POOLING_TYPE_UNSPECIFIED;
      }
    throw std::runtime_error ("EmbeddingGemma llama.cpp backend: unsupported "
                              "pooling mode '" + pooling + "'");
  }

  std::vector<llama_token>
  Tokenize (const std::string &text) const
  {
#if !defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_LLAMA_CPP)
    (void)text;
    return {};
#else
    const bool add_special
        = vocab_ != nullptr && llama_vocab_get_add_bos (vocab_);
    const int32_t needed = llama_tokenize (
        vocab_, text.c_str (), static_cast<int32_t> (text.size ()), nullptr, 0,
        add_special, false);
    if (needed == INT32_MIN)
      {
        throw std::runtime_error (
            "EmbeddingGemma llama.cpp tokenization overflow");
      }
    const int32_t token_count = needed < 0 ? -needed : needed;
    std::vector<llama_token> tokens (static_cast<std::size_t> (token_count));
    const int32_t actual = llama_tokenize (
        vocab_, text.c_str (), static_cast<int32_t> (text.size ()),
        tokens.data (), token_count, add_special, false);
    if (actual < 0)
      {
        throw std::runtime_error (
            "EmbeddingGemma llama.cpp tokenization failed");
      }
    tokens.resize (static_cast<std::size_t> (actual));
    return tokens;
#endif
  }

  void EnsureInitialized ()
  {
    std::lock_guard<std::mutex> lock (init_mu_);
    if (initialized_)
      {
        return;
      }
    if (init_attempted_)
      {
        throw std::runtime_error (init_error_);
      }
    init_attempted_ = true;

    try
      {
        const std::string model_path = ResolveExistingPath (Config ().model_path);
#if !defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_LLAMA_CPP)
        throw std::runtime_error (
            "EmbeddingGemma llama.cpp backend unavailable: link libllama");
#else
        static std::once_flag backend_once;
        std::call_once (backend_once, [] () {
          internal::InstallLlamaCppLogFilter ();
          ggml_backend_load_all ();
          llama_backend_init ();
        });

        llama_model_params mparams = llama_model_default_params ();
        mparams.n_gpu_layers = 0;
        mparams.use_mmap = true;
        mparams.use_mlock = false;
        model_ = llama_model_load_from_file (model_path.c_str (), mparams);
        if (model_ == nullptr)
          {
            throw std::runtime_error (
                "EmbeddingGemma llama.cpp backend failed to load model");
          }

        vocab_ = llama_model_get_vocab (model_);
        if (vocab_ == nullptr)
          {
            throw std::runtime_error (
                "EmbeddingGemma llama.cpp backend model has no vocabulary");
          }

        llama_context_params cparams = llama_context_default_params ();
        cparams.n_ctx = static_cast<uint32_t> (Config ().max_length);
        cparams.n_batch = static_cast<uint32_t> (Config ().max_length);
        cparams.n_ubatch = static_cast<uint32_t> (
            GetEnvInt ("CORTEXT_LLAMA_CPP_UBATCH", Config ().max_length));
        cparams.n_seq_max = 1;
        cparams.n_threads
            = Config ().num_threads > 0 ? Config ().num_threads : 1;
        cparams.n_threads_batch = cparams.n_threads;
        cparams.pooling_type = ResolvePoolingType ();
        cparams.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        cparams.embeddings = true;
        cparams.offload_kqv = false;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx_ = llama_init_from_model (model_, cparams);
        if (ctx_ == nullptr)
          {
            throw std::runtime_error (
                "EmbeddingGemma llama.cpp backend failed to create context");
          }

        llama_set_embeddings (ctx_, true);
        llama_set_warmup (ctx_, false);
        pooling_type_ = ::llama_pooling_type (ctx_);
        embedding_dim_ = llama_model_n_embd (model_);
        if (embedding_dim_ <= 0)
          {
            throw std::runtime_error (
                "EmbeddingGemma llama.cpp backend invalid embedding dimension");
          }

        batch_ = llama_batch_init (Config ().max_length, 0, 1);
        if (batch_.token == nullptr || batch_.pos == nullptr
            || batch_.n_seq_id == nullptr || batch_.seq_id == nullptr
            || batch_.logits == nullptr)
          {
            throw std::runtime_error (
                "EmbeddingGemma llama.cpp backend failed to allocate batch");
          }

        std::vector<llama_token> warmup_tokens = Tokenize ("warmup");
        if (!warmup_tokens.empty ())
          {
            batch_.n_tokens = static_cast<int32_t> (warmup_tokens.size ());
            for (int32_t i = 0; i < batch_.n_tokens; ++i)
              {
                batch_.token[i] = warmup_tokens[static_cast<std::size_t> (i)];
                batch_.pos[i] = i;
                batch_.n_seq_id[i] = 1;
                batch_.seq_id[i][0] = 0;
                batch_.logits[i] = 1;
              }
            const int rc = llama_encode (ctx_, batch_);
            if (rc != 0)
              {
                throw std::runtime_error (
                    "EmbeddingGemma llama.cpp warmup failed: "
                    + std::to_string (rc));
              }
          }
        batch_.n_tokens = 0;
        initialized_ = true;
        return;
#endif
      }
    catch (const std::exception &e)
      {
        init_error_
            = std::string ("EmbeddingGemmaEncoder init failed: ") + e.what ();
        throw;
      }
  }

  std::mutex init_mu_;
  bool init_attempted_ = false;
  bool initialized_ = false;
  std::string init_error_;
#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_LLAMA_CPP)
  llama_model *model_ = nullptr;
  llama_context *ctx_ = nullptr;
  const llama_vocab *vocab_ = nullptr;
  llama_batch batch_{};
  enum llama_pooling_type pooling_type_ = LLAMA_POOLING_TYPE_UNSPECIFIED;
  int embedding_dim_ = 0;
#endif
};

class EmbeddingGemmaLiteRtBackend final : public EmbeddingGemmaBackend
{
public:
  explicit EmbeddingGemmaLiteRtBackend (EmbeddingGemmaConfig config)
      : EmbeddingGemmaBackend (std::move (config))
  {
  }

  ~EmbeddingGemmaLiteRtBackend () override
  {
#if !defined(CORTEXT_DISABLE_LITERT)
    if (input_buffer_ != nullptr)
      {
        LiteRtDestroyTensorBuffer (input_buffer_);
      }
    if (output_buffer_ != nullptr)
      {
        LiteRtDestroyTensorBuffer (output_buffer_);
      }
    if (compiled_model_ != nullptr)
      {
        LiteRtDestroyCompiledModel (compiled_model_);
      }
    if (options_ != nullptr)
      {
        LiteRtDestroyOptions (options_);
      }
    if (model_ != nullptr)
      {
        LiteRtDestroyModel (model_);
      }
    if (environment_ != nullptr)
      {
        LiteRtDestroyEnvironment (environment_);
      }
#endif
  }

  void EncodeText (const std::string &text,
                   std::vector<float> &out_embedding) override
  {
#if defined(CORTEXT_DISABLE_LITERT)
    (void)text;
    (void)out_embedding;
    throw std::runtime_error (
        "EmbeddingGemma LiteRT backend unavailable: rebuild without "
        "CORTEXT_DISABLE_LITERT");
#else
    EnsureInitialized ();

    std::vector<int> token_ids_raw;
    auto encode_status = tokenizer_->Encode (text, &token_ids_raw);
    if (!encode_status.ok ())
      {
        throw std::runtime_error ("EmbeddingGemmaEncoder: tokenization failed: "
                                  + std::string (encode_status.ToString ()));
      }

    const int max_length = Config ().max_length;
    if (max_length <= 0)
      {
        throw std::runtime_error ("EmbeddingGemmaEncoder: max_length must be > 0");
      }

    const std::vector<int> token_ids
        = PadTokensInt (std::move (token_ids_raw), max_length, bos_id_, eos_id_,
                        pad_id_);

    void *input_memory = nullptr;
    ThrowIfLiteRtError (
        LiteRtLockTensorBuffer (input_buffer_, &input_memory,
                                kLiteRtTensorBufferLockModeWrite),
        "EmbeddingGemmaEncoder: failed to lock LiteRT input tensor");
    std::memcpy (input_memory, token_ids.data (),
                 token_ids.size () * sizeof (int));
    ThrowIfLiteRtError (
        LiteRtUnlockTensorBuffer (input_buffer_),
        "EmbeddingGemmaEncoder: failed to unlock LiteRT input tensor");

    LiteRtTensorBuffer input_buffers[] = { input_buffer_ };
    LiteRtTensorBuffer output_buffers[] = { output_buffer_ };
    ThrowIfLiteRtError (
        LiteRtRunCompiledModel (compiled_model_, /*signature_index=*/0,
                                1, input_buffers, 1, output_buffers),
        "EmbeddingGemmaEncoder: LiteRT inference failed");

    size_t packed_size = 0;
    ThrowIfLiteRtError (
        LiteRtGetTensorBufferPackedSize (output_buffer_, &packed_size),
        "EmbeddingGemmaEncoder: failed to query LiteRT output size");
    out_embedding.assign (packed_size / sizeof (float), 0.0f);

    void *output_memory = nullptr;
    ThrowIfLiteRtError (
        LiteRtLockTensorBuffer (output_buffer_, &output_memory,
                                kLiteRtTensorBufferLockModeRead),
        "EmbeddingGemmaEncoder: failed to lock LiteRT output tensor");
    std::memcpy (out_embedding.data (), output_memory, packed_size);
    ThrowIfLiteRtError (
        LiteRtUnlockTensorBuffer (output_buffer_),
        "EmbeddingGemmaEncoder: failed to unlock LiteRT output tensor");

    FinalizeEmbedding (out_embedding);
#endif
  }

private:
  void EnsureInitialized ()
  {
    std::lock_guard<std::mutex> lock (init_mu_);
    if (initialized_)
      {
        return;
      }
    if (init_attempted_)
      {
        throw std::runtime_error (init_error_);
      }
    init_attempted_ = true;

    try
      {
#if defined(CORTEXT_DISABLE_LITERT)
        throw std::runtime_error (
            "EmbeddingGemma LiteRT backend unavailable: rebuild without "
            "CORTEXT_DISABLE_LITERT");
#else
        const std::string model_path = ResolveExistingPath (Config ().model_path);
        const std::string tokenizer_path
            = ResolveExistingPath (Config ().tokenizer_path);

        tokenizer_ = std::make_unique<sentencepiece::SentencePieceProcessor> ();
        auto load_status = tokenizer_->Load (tokenizer_path);
        if (!load_status.ok ())
          {
            throw std::runtime_error (
                "EmbeddingGemmaEncoder: failed to load tokenizer: "
                + std::string (load_status.ToString ()));
          }
        bos_id_ = tokenizer_->bos_id ();
        eos_id_ = tokenizer_->eos_id ();
        pad_id_ = tokenizer_->pad_id ();

        ThrowIfLiteRtError (
            LiteRtCreateEnvironment (/*num_options=*/0, nullptr, &environment_),
            "EmbeddingGemmaEncoder: failed to create LiteRT environment");
        ThrowIfLiteRtError (
            LiteRtCreateModelFromFile (model_path.c_str (), &model_),
            "EmbeddingGemmaEncoder: failed to load LiteRT model");
        ThrowIfLiteRtError (
            LiteRtCreateOptions (&options_),
            "EmbeddingGemmaEncoder: failed to create LiteRT options");
        ThrowIfLiteRtError (
            LiteRtSetOptionsHardwareAccelerators (options_,
                                                  kLiteRtHwAcceleratorCpu),
            "EmbeddingGemmaEncoder: failed to configure LiteRT accelerator");
        ThrowIfLiteRtError (
            LiteRtCreateCompiledModel (environment_, model_, options_,
                                       &compiled_model_),
            "EmbeddingGemmaEncoder: failed to compile LiteRT model");

        LiteRtSubgraph subgraph = nullptr;
        ThrowIfLiteRtError (
            LiteRtGetModelSubgraph (model_, 0, &subgraph),
            "EmbeddingGemmaEncoder: failed to get LiteRT subgraph");

        LiteRtParamIndex num_inputs = 0;
        LiteRtParamIndex num_outputs = 0;
        ThrowIfLiteRtError (
            LiteRtGetNumSubgraphInputs (subgraph, &num_inputs),
            "EmbeddingGemmaEncoder: failed to inspect LiteRT inputs");
        ThrowIfLiteRtError (
            LiteRtGetNumSubgraphOutputs (subgraph, &num_outputs),
            "EmbeddingGemmaEncoder: failed to inspect LiteRT outputs");
        if (num_inputs != 1 || num_outputs != 1)
          {
            throw std::runtime_error ("EmbeddingGemmaEncoder: expected 1 LiteRT "
                                      "input and 1 output tensor, got "
                                      + std::to_string (num_inputs) + " and "
                                      + std::to_string (num_outputs));
          }

        LiteRtTensor input_tensor = nullptr;
        LiteRtTensor output_tensor = nullptr;
        ThrowIfLiteRtError (
            LiteRtGetSubgraphInput (subgraph, 0, &input_tensor),
            "EmbeddingGemmaEncoder: failed to get LiteRT input tensor");
        ThrowIfLiteRtError (
            LiteRtGetSubgraphOutput (subgraph, 0, &output_tensor),
            "EmbeddingGemmaEncoder: failed to get LiteRT output tensor");
        ThrowIfLiteRtError (
            LiteRtGetRankedTensorType (input_tensor, &input_tensor_type_),
            "EmbeddingGemmaEncoder: failed to inspect LiteRT input tensor type");
        ThrowIfLiteRtError (
            LiteRtGetRankedTensorType (output_tensor, &output_tensor_type_),
            "EmbeddingGemmaEncoder: failed to inspect LiteRT output tensor type");

        LiteRtTensorBufferRequirements input_requirements = nullptr;
        LiteRtTensorBufferRequirements output_requirements = nullptr;
        ThrowIfLiteRtError (
            LiteRtGetCompiledModelInputBufferRequirements (
                compiled_model_, /*signature_index=*/0, /*input_index=*/0,
                &input_requirements),
            "EmbeddingGemmaEncoder: failed to get LiteRT input requirements");
        ThrowIfLiteRtError (
            LiteRtGetCompiledModelOutputBufferRequirements (
                compiled_model_, /*signature_index=*/0, /*output_index=*/0,
                &output_requirements),
            "EmbeddingGemmaEncoder: failed to get LiteRT output requirements");

        LiteRtTensorBufferType input_buffer_type = kLiteRtTensorBufferTypeUnknown;
        LiteRtTensorBufferType output_buffer_type
            = kLiteRtTensorBufferTypeUnknown;
        size_t input_buffer_size = 0;
        size_t output_buffer_size = 0;
        ThrowIfLiteRtError (
            LiteRtGetTensorBufferRequirementsSupportedTensorBufferType (
                input_requirements, 0, &input_buffer_type),
            "EmbeddingGemmaEncoder: failed to get LiteRT input buffer type");
        ThrowIfLiteRtError (
            LiteRtGetTensorBufferRequirementsSupportedTensorBufferType (
                output_requirements, 0, &output_buffer_type),
            "EmbeddingGemmaEncoder: failed to get LiteRT output buffer type");
        ThrowIfLiteRtError (
            LiteRtGetTensorBufferRequirementsBufferSize (input_requirements,
                                                         &input_buffer_size),
            "EmbeddingGemmaEncoder: failed to get LiteRT input buffer size");
        ThrowIfLiteRtError (
            LiteRtGetTensorBufferRequirementsBufferSize (output_requirements,
                                                         &output_buffer_size),
            "EmbeddingGemmaEncoder: failed to get LiteRT output buffer size");

        ThrowIfLiteRtError (
            LiteRtCreateManagedTensorBuffer (environment_, input_buffer_type,
                                             &input_tensor_type_,
                                             input_buffer_size, &input_buffer_),
            "EmbeddingGemmaEncoder: failed to create LiteRT input buffer");
        ThrowIfLiteRtError (
            LiteRtCreateManagedTensorBuffer (
                environment_, output_buffer_type, &output_tensor_type_,
                output_buffer_size, &output_buffer_),
            "EmbeddingGemmaEncoder: failed to create LiteRT output buffer");
#endif
        initialized_ = true;
      }
    catch (const std::exception &e)
      {
        init_error_
            = std::string ("EmbeddingGemmaEncoder init failed: ") + e.what ();
        throw;
      }
  }

  std::mutex init_mu_;
  bool init_attempted_ = false;
  bool initialized_ = false;
  std::string init_error_;
#if !defined(CORTEXT_DISABLE_LITERT)
  std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer_;
  LiteRtEnvironment environment_ = nullptr;
  LiteRtModel model_ = nullptr;
  LiteRtOptions options_ = nullptr;
  LiteRtCompiledModel compiled_model_ = nullptr;
  LiteRtTensorBuffer input_buffer_ = nullptr;
  LiteRtTensorBuffer output_buffer_ = nullptr;
  LiteRtRankedTensorType input_tensor_type_{};
  LiteRtRankedTensorType output_tensor_type_{};
  int bos_id_ = -1;
  int eos_id_ = -1;
  int pad_id_ = 0;
#endif
};

std::unique_ptr<EmbeddingGemmaBackend>
CreateBackend (const EmbeddingGemmaConfig &config)
{
  switch (ResolveBackendKind (config))
    {
    case EmbeddingGemmaBackendKind::Onnx:
#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT)
      return std::make_unique<EmbeddingGemmaOnnxBackend> (config);
#else
      throw std::runtime_error (
          "EmbeddingGemma ONNX backend unavailable: rebuild with "
          "CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT");
#endif
    case EmbeddingGemmaBackendKind::LlamaCpp:
      return std::make_unique<EmbeddingGemmaLlamaCppBackend> (config);
    case EmbeddingGemmaBackendKind::LiteRt:
      return std::make_unique<EmbeddingGemmaLiteRtBackend> (config);
    }

  throw std::runtime_error ("EmbeddingGemmaEncoder: unsupported backend");
}

} // namespace

struct EmbeddingGemmaEncoder::Impl
{
  explicit Impl (EmbeddingGemmaConfig config_in)
      : config (std::move (config_in)), backend (CreateBackend (config))
  {
  }

  EmbeddingGemmaConfig config;
  std::unique_ptr<EmbeddingGemmaBackend> backend;
};

EmbeddingGemmaEncoder::EmbeddingGemmaEncoder (EmbeddingGemmaConfig config)
    : impl_ (std::make_unique<Impl> (std::move (config)))
{
}

EmbeddingGemmaEncoder::~EmbeddingGemmaEncoder () = default;

void
EmbeddingGemmaEncoder::EncodeText (const std::string &text,
                                   std::vector<float> &out_embedding)
{
  if (!impl_ || !impl_->backend)
    {
      throw std::runtime_error ("EmbeddingGemmaEncoder not initialized");
    }
  impl_->backend->EncodeText (text, out_embedding);
}

void
EmbeddingGemmaEncoder::EncodeAudio (const float *pcm,
                                    std::size_t num_samples,
                                    std::vector<float> &out_embedding)
{
  (void)pcm;
  (void)num_samples;
  (void)out_embedding;
  throw std::runtime_error ("EmbeddingGemmaEncoder: audio encoding not supported");
}

void
EmbeddingGemmaEncoder::EncodeImage (const std::uint8_t *data, int width,
                                    int height, int channels,
                                    std::vector<float> &out_embedding)
{
  (void)data;
  (void)width;
  (void)height;
  (void)channels;
  (void)out_embedding;
  throw std::runtime_error ("EmbeddingGemmaEncoder: image encoding not supported");
}

namespace internal
{

#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT)
void
ResetEmbeddingGemmaTextEncodeProfile ()
{
  auto &state = GetEmbeddingGemmaTextEncodeProfileState ();
  std::lock_guard<std::mutex> lock (state.mu);
  state.snapshot = {};
}

EmbeddingGemmaTextEncodeProfileSnapshot
GetEmbeddingGemmaTextEncodeProfileSnapshot ()
{
  auto &state = GetEmbeddingGemmaTextEncodeProfileState ();
  std::lock_guard<std::mutex> lock (state.mu);
  return state.snapshot;
}
#endif

} // namespace internal

} // namespace cortext
