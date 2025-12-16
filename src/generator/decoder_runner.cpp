#include "cortext/generator/decoder_runner.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <thread>

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
  std::unique_ptr<Ort::IoBinding> io_binding;
  Ort::MemoryInfo mem_info
      = Ort::MemoryInfo::CreateCpu (OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<std::string> output_names;
  std::vector<std::string> kv_input_names; // Persistent storage for KV names
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

    // Session options for memory optimization (matches Python)
    // Check environment for thread count, default to min(4, cpu_count)
    int cpu_count = static_cast<int> (std::thread::hardware_concurrency ());
    int intra_threads = std::min (4, std::max (1, cpu_count));
    const char *env_threads = std::getenv ("CORTEXT_ORT_INTRA_THREADS");
    if (env_threads)
      {
        intra_threads = std::atoi (env_threads);
      }
    opts.SetIntraOpNumThreads (intra_threads);
    opts.SetInterOpNumThreads (1);
    opts.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.EnableMemPattern ();
    opts.EnableCpuMemArena ();
    opts.SetExecutionMode (ExecutionMode::ORT_SEQUENTIAL);

    session
        = std::make_unique<Ort::Session> (env, model_path.c_str (), opts);

    // Create IOBinding for efficient inference
    io_binding = std::make_unique<Ort::IoBinding> (*session);

    // Collect output names
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t num_outputs = session->GetOutputCount ();
    for (std::size_t i = 0; i < num_outputs; ++i)
      {
        auto name = session->GetOutputNameAllocated (i, allocator);
        output_names.push_back (name.get ());
      }

    // Pre-build KV input names for reuse
    for (int layer = 0; layer < kNumLayers; ++layer)
      {
        kv_input_names.push_back ("past_key_values." + std::to_string (layer)
                                  + ".key");
        kv_input_names.push_back ("past_key_values." + std::to_string (layer)
                                  + ".value");
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

  auto &io = *impl_->io_binding;
  io.ClearBoundInputs ();
  io.ClearBoundOutputs ();

  // Convert shapes to int64_t
  std::vector<int64_t> embeds_shape_i64 (embeds_shape.begin (),
                                         embeds_shape.end ());
  std::vector<int64_t> per_layer_shape_i64 (per_layer_shape.begin (),
                                            per_layer_shape.end ());
  std::vector<int64_t> position_shape_i64 (position_shape.begin (),
                                           position_shape.end ());
  std::vector<int64_t> past_kv_shape_i64 (past_kv_shape.begin (),
                                          past_kv_shape.end ());

  // Bind inputs_embeds
  auto embeds_tensor = Ort::Value::CreateTensor<float> (
      impl_->mem_info, const_cast<float *> (inputs_embeds.data ()),
      inputs_embeds.size (), embeds_shape_i64.data (),
      embeds_shape_i64.size ());
  io.BindInput ("inputs_embeds", embeds_tensor);

  // Bind per_layer_inputs
  auto per_layer_tensor = Ort::Value::CreateTensor<float> (
      impl_->mem_info, const_cast<float *> (per_layer_inputs.data ()),
      per_layer_inputs.size (), per_layer_shape_i64.data (),
      per_layer_shape_i64.size ());
  io.BindInput ("per_layer_inputs", per_layer_tensor);

  // Bind position_ids
  auto position_tensor = Ort::Value::CreateTensor<int64_t> (
      impl_->mem_info, const_cast<int64_t *> (position_ids.data ()),
      position_ids.size (), position_shape_i64.data (),
      position_shape_i64.size ());
  io.BindInput ("position_ids", position_tensor);

  // Bind past_key_values using IOBinding
  std::vector<Ort::Value> kv_tensors; // Keep tensors alive during Run
  kv_tensors.reserve (kNumLayers * 2);
  for (int layer = 0; layer < kNumLayers; ++layer)
    {
      const std::string &key_name = impl_->kv_input_names[layer * 2];
      const std::string &value_name = impl_->kv_input_names[layer * 2 + 1];

      auto key_it = past_kv.find (key_name);
      auto value_it = past_kv.find (value_name);

      if (key_it != past_kv.end ())
        {
          kv_tensors.push_back (Ort::Value::CreateTensor<float> (
              impl_->mem_info,
              const_cast<float *> (key_it->second.data ()),
              key_it->second.size (), past_kv_shape_i64.data (),
              past_kv_shape_i64.size ()));
          io.BindInput (key_name.c_str (), kv_tensors.back ());
        }

      if (value_it != past_kv.end ())
        {
          kv_tensors.push_back (Ort::Value::CreateTensor<float> (
              impl_->mem_info,
              const_cast<float *> (value_it->second.data ()),
              value_it->second.size (), past_kv_shape_i64.data (),
              past_kv_shape_i64.size ()));
          io.BindInput (value_name.c_str (), kv_tensors.back ());
        }
    }

  // Bind outputs to CPU
  for (const auto &name : impl_->output_names)
    {
      io.BindOutput (name.c_str (), impl_->mem_info);
    }

  // Run inference with IOBinding
  impl_->session->Run (Ort::RunOptions{ nullptr }, io);

  // Get outputs from IOBinding
  auto outputs = io.GetOutputValues ();

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
  // Model outputs: present.N.key, present.N.value
  for (std::size_t i = 1; i < outputs.size (); ++i)
    {
      const std::string &name = impl_->output_names[i];
      if (name.find ("present.") != std::string::npos)
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
  std::unique_ptr<Ort::IoBinding> io_binding;
  Ort::MemoryInfo mem_info
      = Ort::MemoryInfo::CreateCpu (OrtDeviceAllocator, OrtMemTypeCPU);
  bool available = false;
  std::size_t hidden_dim = kHiddenDim;

  // Preallocated step buffers for single-token decode (B=1, L=1)
  std::vector<float> step_inputs_embeds;
  std::vector<float> step_per_layer_inputs;

  void
  LoadModel (const std::string &models_dir)
  {
    std::filesystem::path model_path
        = std::filesystem::path (models_dir) / "onnx"
          / "embed_tokens_int8.onnx";

    if (!std::filesystem::exists (model_path))
      {
        throw std::runtime_error ("Embed model not found: "
                                  + model_path.string ());
      }

    // Session options for memory optimization (matches Python)
    // Check environment for thread count, default to min(4, cpu_count)
    int cpu_count = static_cast<int> (std::thread::hardware_concurrency ());
    int intra_threads = std::min (4, std::max (1, cpu_count));
    const char *env_threads = std::getenv ("CORTEXT_ORT_INTRA_THREADS");
    if (env_threads)
      {
        intra_threads = std::atoi (env_threads);
      }
    opts.SetIntraOpNumThreads (intra_threads);
    opts.SetInterOpNumThreads (1);
    opts.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.EnableMemPattern ();
    opts.EnableCpuMemArena ();
    opts.SetExecutionMode (ExecutionMode::ORT_SEQUENTIAL);

    session
        = std::make_unique<Ort::Session> (env, model_path.c_str (), opts);

    // Create IOBinding for efficient inference
    io_binding = std::make_unique<Ort::IoBinding> (*session);

    // Preallocate step buffers: [1, 1, hidden_dim] and [1, 1, num_layers,
    // head_dim]
    step_inputs_embeds.resize (1 * 1 * kHiddenDim, 0.0f);
    step_per_layer_inputs.resize (1 * 1 * kNumLayers * kHeadDim, 0.0f);

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

EmbedOutput
EmbedRunner::Embed (const std::vector<int64_t> &token_ids,
                    const std::vector<std::size_t> &shape)
{
  // Delegate to EmbedPrefill for backward compatibility
  return EmbedPrefill (token_ids, shape);
}

EmbedOutput
EmbedRunner::EmbedPrefill (const std::vector<int64_t> &token_ids,
                           const std::vector<std::size_t> &shape)
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  if (!impl_->available)
    {
      throw std::runtime_error ("Embed model not available");
    }

  auto &io = *impl_->io_binding;
  io.ClearBoundInputs ();
  io.ClearBoundOutputs ();

  std::vector<int64_t> shape_i64 (shape.begin (), shape.end ());

  // Bind input
  auto input_tensor = Ort::Value::CreateTensor<int64_t> (
      impl_->mem_info, const_cast<int64_t *> (token_ids.data ()),
      token_ids.size (), shape_i64.data (), shape_i64.size ());
  io.BindInput ("input_ids", input_tensor);

  // Bind outputs to CPU
  io.BindOutput ("inputs_embeds", impl_->mem_info);
  io.BindOutput ("per_layer_inputs", impl_->mem_info);

  // Run with IOBinding
  impl_->session->Run (Ort::RunOptions{ nullptr }, io);

  // Get outputs
  auto outputs = io.GetOutputValues ();

  EmbedOutput result;

  // Extract inputs_embeds
  {
    const float *data = outputs[0].GetTensorData<float> ();
    auto shape_info = outputs[0].GetTensorTypeAndShapeInfo ();
    auto out_shape = shape_info.GetShape ();

    std::size_t total_size = 1;
    for (auto dim : out_shape)
      {
        total_size *= static_cast<std::size_t> (dim);
        result.embeds_shape.push_back (static_cast<std::size_t> (dim));
      }

    result.inputs_embeds.assign (data, data + total_size);

    // Update hidden_dim from actual output
    if (out_shape.size () >= 3)
      {
        impl_->hidden_dim = static_cast<std::size_t> (out_shape[2]);
      }
  }

  // Extract per_layer_inputs
  {
    const float *data = outputs[1].GetTensorData<float> ();
    auto shape_info = outputs[1].GetTensorTypeAndShapeInfo ();
    auto out_shape = shape_info.GetShape ();

    std::size_t total_size = 1;
    for (auto dim : out_shape)
      {
        total_size *= static_cast<std::size_t> (dim);
        result.per_layer_shape.push_back (static_cast<std::size_t> (dim));
      }

    result.per_layer_inputs.assign (data, data + total_size);
  }

  return result;

#else
  (void)token_ids;
  (void)shape;
  throw std::runtime_error (
      "EmbedRunner: CORTEXT_ENABLE_GEMMA_ORT not defined");
#endif
}

EmbedOutput
EmbedRunner::EmbedStep (int64_t token_id)
{
#if defined(CORTEXT_ENABLE_GEMMA_ORT)
  if (!impl_->available)
    {
      throw std::runtime_error ("Embed model not available");
    }

  auto &io = *impl_->io_binding;
  io.ClearBoundInputs ();
  io.ClearBoundOutputs ();

  // Single token input [1, 1]
  std::vector<int64_t> token_ids = { token_id };
  std::vector<int64_t> shape_i64 = { 1, 1 };

  // Bind input
  auto input_tensor = Ort::Value::CreateTensor<int64_t> (
      impl_->mem_info, token_ids.data (), token_ids.size (), shape_i64.data (),
      shape_i64.size ());
  io.BindInput ("input_ids", input_tensor);

  // Bind outputs to preallocated buffers for zero-copy
  std::vector<int64_t> embeds_shape = { 1, 1,
                                        static_cast<int64_t> (kHiddenDim) };
  std::vector<int64_t> per_layer_shape
      = { 1, 1, static_cast<int64_t> (kNumLayers),
          static_cast<int64_t> (kHeadDim) };

  auto embeds_out = Ort::Value::CreateTensor<float> (
      impl_->mem_info, impl_->step_inputs_embeds.data (),
      impl_->step_inputs_embeds.size (), embeds_shape.data (),
      embeds_shape.size ());
  io.BindOutput ("inputs_embeds", embeds_out);

  auto per_layer_out = Ort::Value::CreateTensor<float> (
      impl_->mem_info, impl_->step_per_layer_inputs.data (),
      impl_->step_per_layer_inputs.size (), per_layer_shape.data (),
      per_layer_shape.size ());
  io.BindOutput ("per_layer_inputs", per_layer_out);

  // Run with IOBinding
  impl_->session->Run (Ort::RunOptions{ nullptr }, io);

  // Return result pointing to preallocated buffers
  EmbedOutput result;
  result.inputs_embeds = impl_->step_inputs_embeds;
  result.embeds_shape = { 1, 1, static_cast<std::size_t> (kHiddenDim) };
  result.per_layer_inputs = impl_->step_per_layer_inputs;
  result.per_layer_shape = { 1, 1, static_cast<std::size_t> (kNumLayers),
                             static_cast<std::size_t> (kHeadDim) };

  return result;

#else
  (void)token_id;
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
