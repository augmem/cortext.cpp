#include "cortext/generator/decoder_runner.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#if defined(CORTEXT_ENABLE_GEMMA_ORT)
#include <onnxruntime/onnxruntime_cxx_api.h>
#endif

namespace cortext
{

namespace
{

constexpr int kNumLayers = 30;
constexpr int kNumKVHeads = 2;
constexpr int kHeadDim = 256;
constexpr int kHiddenDim = 2048; // gemma-3n hidden dimension

} // namespace

// ============================================================================
// DecoderRunner Implementation
// ============================================================================

struct DecoderRunner::Impl
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "cortext-gemma-decoder" };
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  Ort::IoBinding *io_binding = nullptr;
  std::vector<std::string> output_names;
  bool available = false;

  void
  LoadModel (const std::string &models_dir)
  {
    std::filesystem::path model_path
        = std::filesystem::path (models_dir) / "onnx"
          / "decoder_model_merged_q4.onnx";

    if (!std::filesystem::exists (model_path))
      {
        throw std::runtime_error ("Decoder model not found: "
                                  + model_path.string ());
      }

    opts.SetIntraOpNumThreads (1);
    opts.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);

    session
        = std::make_unique<Ort::Session> (env, model_path.c_str (), opts);

    // Collect output names
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t num_outputs = session->GetOutputCount ();
    for (std::size_t i = 0; i < num_outputs; ++i)
      {
        auto name = session->GetOutputNameAllocated (i, allocator);
        output_names.push_back (name.get ());
      }

    available = true;
  }
#else
  std::vector<std::string> output_names;
  bool available = false;
#endif
};

DecoderRunner::DecoderRunner (const std::string &models_dir)
    : impl_ (std::make_unique<Impl> ())
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  impl_->LoadModel (models_dir);
#else
  (void)models_dir;
#endif
}

DecoderRunner::~DecoderRunner () = default;
DecoderRunner::DecoderRunner (DecoderRunner &&) noexcept = default;
DecoderRunner &DecoderRunner::operator= (DecoderRunner &&) noexcept = default;

DecoderOutput
DecoderRunner::Run (const std::vector<float> &inputs_embeds,
                    const std::vector<std::size_t> &embeds_shape,
                    const std::vector<float> &per_layer_inputs,
                    const std::vector<std::size_t> &per_layer_shape,
                    const std::vector<int64_t> &position_ids,
                    const std::vector<std::size_t> &position_shape,
                    const std::map<std::string, std::vector<float> > &past_kv,
                    const std::vector<std::size_t> &past_kv_shape)
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  if (!impl_->available)
    {
      throw std::runtime_error ("Decoder model not available");
    }

  Ort::MemoryInfo mem_info
      = Ort::MemoryInfo::CreateCpu (OrtDeviceAllocator, OrtMemTypeCPU);

  // Convert shapes to int64_t
  std::vector<int64_t> embeds_shape_i64 (embeds_shape.begin (),
                                         embeds_shape.end ());
  std::vector<int64_t> per_layer_shape_i64 (per_layer_shape.begin (),
                                            per_layer_shape.end ());
  std::vector<int64_t> position_shape_i64 (position_shape.begin (),
                                           position_shape.end ());
  std::vector<int64_t> past_kv_shape_i64 (past_kv_shape.begin (),
                                          past_kv_shape.end ());

  // Build input tensors
  std::vector<Ort::Value> input_tensors;
  std::vector<const char *> input_names;

  // inputs_embeds
  input_names.push_back ("inputs_embeds");
  input_tensors.push_back (Ort::Value::CreateTensor<float> (
      mem_info, const_cast<float *> (inputs_embeds.data ()),
      inputs_embeds.size (), embeds_shape_i64.data (),
      embeds_shape_i64.size ()));

  // per_layer_inputs
  input_names.push_back ("per_layer_inputs");
  input_tensors.push_back (Ort::Value::CreateTensor<float> (
      mem_info, const_cast<float *> (per_layer_inputs.data ()),
      per_layer_inputs.size (), per_layer_shape_i64.data (),
      per_layer_shape_i64.size ()));

  // position_ids
  input_names.push_back ("position_ids");
  input_tensors.push_back (Ort::Value::CreateTensor<int64_t> (
      mem_info, const_cast<int64_t *> (position_ids.data ()),
      position_ids.size (), position_shape_i64.data (),
      position_shape_i64.size ()));

  // past_key_values
  std::vector<std::vector<float> >
      past_kv_copies; // Keep alive during Run
  for (int layer = 0; layer < kNumLayers; ++layer)
    {
      std::string key_name
          = "past_key_values." + std::to_string (layer) + ".key";
      std::string value_name
          = "past_key_values." + std::to_string (layer) + ".value";

      auto key_it = past_kv.find (key_name);
      auto value_it = past_kv.find (value_name);

      if (key_it != past_kv.end ())
        {
          input_names.push_back (key_name.c_str ());
          past_kv_copies.push_back (key_it->second);
          input_tensors.push_back (Ort::Value::CreateTensor<float> (
              mem_info, past_kv_copies.back ().data (),
              past_kv_copies.back ().size (), past_kv_shape_i64.data (),
              past_kv_shape_i64.size ()));
        }

      if (value_it != past_kv.end ())
        {
          input_names.push_back (value_name.c_str ());
          past_kv_copies.push_back (value_it->second);
          input_tensors.push_back (Ort::Value::CreateTensor<float> (
              mem_info, past_kv_copies.back ().data (),
              past_kv_copies.back ().size (), past_kv_shape_i64.data (),
              past_kv_shape_i64.size ()));
        }
    }

  // Convert input_names to vector of const char* that stays valid
  std::vector<const char *> input_names_cstr;
  input_names_cstr.reserve (input_names.size ());
  for (std::size_t i = 0; i < 3; ++i)
    { // First 3 are string literals
      input_names_cstr.push_back (input_names[i]);
    }
  // For KV names, we need persistent strings
  std::vector<std::string> kv_names_storage;
  for (int layer = 0; layer < kNumLayers; ++layer)
    {
      kv_names_storage.push_back ("past_key_values." + std::to_string (layer)
                                  + ".key");
      kv_names_storage.push_back ("past_key_values." + std::to_string (layer)
                                  + ".value");
    }
  for (const auto &name : kv_names_storage)
    {
      input_names_cstr.push_back (name.c_str ());
    }

  // Output names
  std::vector<const char *> output_names_cstr;
  for (const auto &name : impl_->output_names)
    {
      output_names_cstr.push_back (name.c_str ());
    }

  // Run inference
  auto outputs = impl_->session->Run (
      Ort::RunOptions{ nullptr }, input_names_cstr.data (),
      input_tensors.data (), input_tensors.size (), output_names_cstr.data (),
      output_names_cstr.size ());

  // Extract results
  DecoderOutput result;

  // First output is logits
  if (!outputs.empty ())
    {
      auto &logits_tensor = outputs[0];
      const float *logits_data = logits_tensor.GetTensorData<float> ();
      auto shape_info = logits_tensor.GetTensorTypeAndShapeInfo ();
      auto shape = shape_info.GetShape ();

      std::size_t total_size = 1;
      for (auto dim : shape)
        {
          total_size *= static_cast<std::size_t> (dim);
          result.logits_shape.push_back (static_cast<std::size_t> (dim));
        }
      result.logits.assign (logits_data, logits_data + total_size);
    }

  // Extract present KV from remaining outputs
  for (std::size_t i = 1; i < outputs.size (); ++i)
    {
      const std::string &name = impl_->output_names[i];
      if (name.find ("present_key_values") != std::string::npos
          || name.find ("past_key_values") != std::string::npos)
        {
          auto &tensor = outputs[i];
          const float *data = tensor.GetTensorData<float> ();
          auto shape_info = tensor.GetTensorTypeAndShapeInfo ();
          auto shape = shape_info.GetShape ();

          std::size_t total_size = 1;
          for (auto dim : shape)
            {
              total_size *= static_cast<std::size_t> (dim);
            }

          result.present_kv[name].assign (data, data + total_size);

          // Store shape (same for all KV tensors)
          if (result.kv_shape.empty ())
            {
              for (auto dim : shape)
                {
                  result.kv_shape.push_back (static_cast<std::size_t> (dim));
                }
            }
        }
    }

  return result;

#else
  (void)inputs_embeds;
  (void)embeds_shape;
  (void)per_layer_inputs;
  (void)per_layer_shape;
  (void)position_ids;
  (void)position_shape;
  (void)past_kv;
  (void)past_kv_shape;
  throw std::runtime_error (
      "DecoderRunner: CORTEXT_ENABLE_GEMMA_ORT not defined");
#endif
}

bool
DecoderRunner::IsAvailable () const
{
  return impl_->available;
}

const std::vector<std::string> &
DecoderRunner::OutputNames () const
{
  return impl_->output_names;
}

// ============================================================================
// EmbedRunner Implementation
// ============================================================================

struct EmbedRunner::Impl
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "cortext-gemma-embed" };
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  bool available = false;
  std::size_t hidden_dim = kHiddenDim;

  void
  LoadModel (const std::string &models_dir)
  {
    std::filesystem::path model_path
        = std::filesystem::path (models_dir) / "onnx"
          / "embed_tokens_quantized.onnx";

    if (!std::filesystem::exists (model_path))
      {
        throw std::runtime_error ("Embed model not found: "
                                  + model_path.string ());
      }

    opts.SetIntraOpNumThreads (1);
    opts.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);

    session
        = std::make_unique<Ort::Session> (env, model_path.c_str (), opts);
    available = true;
  }
#else
  bool available = false;
  std::size_t hidden_dim = kHiddenDim;
#endif
};

EmbedRunner::EmbedRunner (const std::string &models_dir)
    : impl_ (std::make_unique<Impl> ())
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  impl_->LoadModel (models_dir);
#else
  (void)models_dir;
#endif
}

EmbedRunner::~EmbedRunner () = default;
EmbedRunner::EmbedRunner (EmbedRunner &&) noexcept = default;
EmbedRunner &EmbedRunner::operator= (EmbedRunner &&) noexcept = default;

std::vector<float>
EmbedRunner::Embed (const std::vector<int64_t> &token_ids,
                    const std::vector<std::size_t> &shape)
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  if (!impl_->available)
    {
      throw std::runtime_error ("Embed model not available");
    }

  Ort::MemoryInfo mem_info
      = Ort::MemoryInfo::CreateCpu (OrtDeviceAllocator, OrtMemTypeCPU);

  std::vector<int64_t> shape_i64 (shape.begin (), shape.end ());

  Ort::Value input = Ort::Value::CreateTensor<int64_t> (
      mem_info, const_cast<int64_t *> (token_ids.data ()), token_ids.size (),
      shape_i64.data (), shape_i64.size ());

  const char *input_names[] = { "input_ids" };
  const char *output_names[] = { "inputs_embeds" };

  auto outputs = impl_->session->Run (Ort::RunOptions{ nullptr }, input_names,
                                      &input, 1, output_names, 1);

  const float *data = outputs[0].GetTensorData<float> ();
  auto shape_info = outputs[0].GetTensorTypeAndShapeInfo ();
  auto out_shape = shape_info.GetShape ();

  std::size_t total_size = 1;
  for (auto dim : out_shape)
    {
      total_size *= static_cast<std::size_t> (dim);
    }

  // Update hidden_dim from actual output
  if (out_shape.size () >= 3)
    {
      impl_->hidden_dim = static_cast<std::size_t> (out_shape[2]);
    }

  return std::vector<float> (data, data + total_size);

#else
  (void)token_ids;
  (void)shape;
  throw std::runtime_error (
      "EmbedRunner: CORTEXT_ENABLE_GEMMA_ORT not defined");
#endif
}

std::size_t
EmbedRunner::HiddenDim () const
{
  return impl_->hidden_dim;
}

bool
EmbedRunner::IsAvailable () const
{
  return impl_->available;
}

} // namespace cortext
