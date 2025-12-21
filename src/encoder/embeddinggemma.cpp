#include "cortext/encoder/embeddinggemma.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT)
#include <onnxruntime/onnxruntime_cxx_api.h>

#include "cortext/core/thread_config.hpp"
#include "runtime/components/sentencepiece_tokenizer.h"
#endif

namespace cortext
{

namespace
{
#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT)

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
  else
    {
      if (static_cast<int> (ids.size ()) > max_length)
        ids.resize (static_cast<std::size_t> (max_length));
    }

  if (pad_id < 0)
    pad_id = 0;

  std::vector<int64_t> out;
  out.reserve (static_cast<std::size_t> (max_length));
  for (int id : ids)
    out.push_back (static_cast<int64_t> (id));
  while (static_cast<int> (out.size ()) < max_length)
    out.push_back (static_cast<int64_t> (pad_id));
  return out;
}

std::vector<int64_t>
BuildAttentionMask (const std::vector<int64_t> &tokens, int pad_id)
{
  std::vector<int64_t> mask;
  mask.reserve (tokens.size ());
  for (int64_t t : tokens)
    mask.push_back (t == pad_id ? 0 : 1);
  return mask;
}

#endif
}

struct EmbeddingGemmaEncoder::Impl
{
  explicit Impl (EmbeddingGemmaConfig config_in)
      : config (std::move (config_in))
  {
  }

  EmbeddingGemmaConfig config;

#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT)
  Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "cortext-embeddinggemma" };
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  std::unique_ptr<litert::lm::SentencePieceTokenizer> tokenizer;
  Ort::MemoryInfo memory_info{ Ort::MemoryInfo::CreateCpu (OrtDeviceAllocator,
                                                          OrtMemTypeCPU) };

  std::string input_ids_name;
  std::string attention_mask_name;
  std::string output_name;

  int bos_id = -1;
  int eos_id = -1;
  int pad_id = 0;

  std::mutex init_mu;
  bool init_attempted = false;
  bool init_ok = false;
  std::string init_error;

  void
  EnsureInitialized ()
  {
    std::lock_guard<std::mutex> lock (init_mu);
    if (init_ok)
      return;
    if (init_attempted)
      throw std::runtime_error (init_error);
    init_attempted = true;
    try
      {
        const std::string model_path = ResolveExistingPath (config.model_path);
        const std::string tokenizer_path
            = ResolveExistingPath (config.tokenizer_path);

        auto tokenizer_or
            = litert::lm::SentencePieceTokenizer::CreateFromFile (
                tokenizer_path);
        if (!tokenizer_or.ok ())
          {
            throw std::runtime_error (
                "EmbeddingGemmaEncoder: failed to load tokenizer: "
                + tokenizer_or.status ().ToString ());
          }
        tokenizer = std::move (*tokenizer_or);
        const auto &proc = tokenizer->GetProcessor ();
        bos_id = proc.bos_id ();
        eos_id = proc.eos_id ();
        pad_id = proc.pad_id ();

        session = std::make_unique<Ort::Session> (env, model_path.c_str (),
                                                  opts);

        Ort::AllocatorWithDefaultOptions allocator;
        if (session->GetInputCount () < 1)
          throw std::runtime_error ("EmbeddingGemmaEncoder: model has no inputs");

        for (std::size_t i = 0; i < session->GetInputCount (); ++i)
          {
            auto name = session->GetInputNameAllocated (i, allocator);
            if (!name)
              continue;
            const std::string input_name = name.get ();
            if (input_name == "input_ids")
              input_ids_name = input_name;
            else if (input_name == "attention_mask")
              attention_mask_name = input_name;
          }
        if (input_ids_name.empty ())
          {
            auto name = session->GetInputNameAllocated (0, allocator);
            if (name)
              input_ids_name = name.get ();
          }
        if (session->GetOutputCount () == 0)
          throw std::runtime_error ("EmbeddingGemmaEncoder: model has no outputs");
        auto out_name = session->GetOutputNameAllocated (0, allocator);
        if (!out_name)
          throw std::runtime_error ("EmbeddingGemmaEncoder: output name missing");
        output_name = out_name.get ();

        init_ok = true;
      }
    catch (const std::exception &e)
      {
        init_error = std::string ("EmbeddingGemmaEncoder init failed: ") + e.what ();
        throw;
      }
  }
#endif
};

EmbeddingGemmaEncoder::EmbeddingGemmaEncoder (EmbeddingGemmaConfig config)
    : impl_ (std::make_unique<Impl> (std::move (config)))
{
#if defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT)
  auto threads = static_cast<int> (core::GetEmbedThreadCount ());
  if (impl_->config.num_threads > 0)
    threads = impl_->config.num_threads;
  impl_->opts.SetIntraOpNumThreads (threads);
  impl_->opts.SetInterOpNumThreads (threads);
  impl_->opts.SetGraphOptimizationLevel (
      GraphOptimizationLevel::ORT_ENABLE_BASIC);
#endif
}

EmbeddingGemmaEncoder::~EmbeddingGemmaEncoder () = default;

void
EmbeddingGemmaEncoder::EncodeText (const std::string &text,
                                   std::vector<float> &out_embedding)
{
#if !defined(CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT)
  (void)text;
  (void)out_embedding;
  throw std::runtime_error (
      "EmbeddingGemmaEncoder requires CORTEXT_ENABLE_EMBEDDINGGEMMA_ORT=ON");
#else
  if (!impl_)
    throw std::runtime_error ("EmbeddingGemmaEncoder not initialized");

  impl_->EnsureInitialized ();
  if (!impl_->session || !impl_->tokenizer)
    throw std::runtime_error ("EmbeddingGemmaEncoder not initialized");

  auto ids_or = impl_->tokenizer->TextToTokenIds (text);
  if (!ids_or.ok ())
    {
      throw std::runtime_error ("EmbeddingGemmaEncoder: tokenization failed: "
                                + ids_or.status ().ToString ());
    }

  const int max_length = impl_->config.max_length;
  if (max_length <= 0)
    throw std::runtime_error ("EmbeddingGemmaEncoder: max_length must be > 0");
  std::vector<int64_t> token_ids
      = PadTokens (std::move (*ids_or), max_length, impl_->bos_id,
                   impl_->eos_id, impl_->pad_id);
  std::vector<int64_t> attention_mask
      = BuildAttentionMask (token_ids, impl_->pad_id);

  std::array<int64_t, 2> shape{ 1, static_cast<int64_t> (token_ids.size ()) };
  Ort::Value ids_tensor = Ort::Value::CreateTensor<int64_t> (
      impl_->memory_info, token_ids.data (), token_ids.size (), shape.data (),
      shape.size ());
  Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t> (
      impl_->memory_info, attention_mask.data (), attention_mask.size (),
      shape.data (), shape.size ());

  std::vector<const char *> input_names;
  std::vector<Ort::Value> input_tensors;
  input_names.push_back (impl_->input_ids_name.c_str ());
  input_tensors.push_back (std::move (ids_tensor));
  if (!impl_->attention_mask_name.empty ())
    {
      input_names.push_back (impl_->attention_mask_name.c_str ());
      input_tensors.push_back (std::move (mask_tensor));
    }

  std::vector<const char *> output_names{ impl_->output_name.c_str () };
  auto outputs = impl_->session->Run (Ort::RunOptions{ nullptr },
                                     input_names.data (),
                                     input_tensors.data (),
                                     input_tensors.size (),
                                     output_names.data (),
                                     output_names.size ());
  if (outputs.empty ())
    throw std::runtime_error ("EmbeddingGemmaEncoder: no output tensors");

  auto &out = outputs.front ();
  if (!out.IsTensor ())
    throw std::runtime_error ("EmbeddingGemmaEncoder: output is not a tensor");

  const float *data = out.GetTensorData<float> ();
  auto info = out.GetTensorTypeAndShapeInfo ();
  const auto shape_info = info.GetShape ();
  std::size_t total = 1;
  for (auto dim : shape_info)
    {
      if (dim > 0)
        total *= static_cast<std::size_t> (dim);
    }
  if (total == 0)
    throw std::runtime_error ("EmbeddingGemmaEncoder: empty output tensor");

  out_embedding.assign (data, data + total);
#endif
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

} // namespace cortext
