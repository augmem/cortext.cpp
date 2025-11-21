#include "cortext/encoder/imagebind.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(CORTEXT_ENABLE_IMAGEBIND_ORT)
#include <filesystem>
#include <onnxruntime_cxx_api.h>
#endif

namespace cortext
{

ImageBindEncoder::ImageBindEncoder (std::string models_dir)
    : models_dir_ (std::move (models_dir))
{
}

void
ImageBindEncoder::FillZeros (std::vector<float> &out, int dim)
{
  out.clear ();
  out.resize (static_cast<std::size_t> (dim), 0.0f);
}

void
ImageBindEncoder::EncodeText (const std::string & /*text*/,
                              std::vector<float> &out_embedding)
{
  // Tokenization and model-specific inputs are not wired yet; fallback.
  FillZeros (out_embedding, kDim);
}

void
ImageBindEncoder::EncodeAudio (const float * /*pcm*/,
                               std::size_t /*num_samples*/,
                               std::vector<float> &out_embedding)
{
  // Feature extraction and model-specific inputs are not wired yet; fallback.
  FillZeros (out_embedding, kDim);
}

void
ImageBindEncoder::EncodeImage (const std::uint8_t *data, int width, int height,
                               int channels, std::vector<float> &out_embedding)
{
#if defined(CORTEXT_ENABLE_IMAGEBIND_ORT)
  try
    {
      // Attempt a minimal inference path for an image encoder with input
      // shape [1,3,224,224] and a single float output "embedding".
      // If preconditions are not met, fall back to zeros.
      if (!data || width <= 0 || height <= 0 || channels < 3)
        {
          FillZeros (out_embedding, kDim);
          return;
        }

      // Locate model at models_dir_/image.onnx; if not found, fallback.
      std::string model_path = models_dir_ + "/image.onnx";
#if __cpp_lib_filesystem
      if (!std::filesystem::exists (model_path))
        {
          // Try alternate common name
          std::string alt = models_dir_ + "/image_encoder.onnx";
          if (std::filesystem::exists (alt))
            {
              model_path = alt;
            }
          else
            {
              FillZeros (out_embedding, kDim);
              return;
            }
        }
#endif

      Ort::Env env (ORT_LOGGING_LEVEL_WARNING, "cortext-imagebind");
      Ort::SessionOptions opts;
      opts.SetIntraOpNumThreads (1);
      opts.SetInterOpNumThreads (1);
      // If available, enable basic graph optimizations
      opts.SetGraphOptimizationLevel (
          GraphOptimizationLevel::ORT_ENABLE_BASIC);

      Ort::Session session (env, model_path.c_str (), opts);
      Ort::AllocatorWithDefaultOptions allocator;

      // Prepare input name 0
      char *in_name_c
          = session.GetInputNameAllocated (0, allocator).release ();
      std::string input_name (in_name_c ? in_name_c : "");
      if (in_name_c)
        {
          allocator.Free (in_name_c);
        }
      if (input_name.empty ())
        {
          FillZeros (out_embedding, kDim);
          return;
        }

      // Prepare output name 0
      char *out_name_c
          = session.GetOutputNameAllocated (0, allocator).release ();
      std::string output_name (out_name_c ? out_name_c : "");
      if (out_name_c)
        {
          allocator.Free (out_name_c);
        }
      if (output_name.empty ())
        {
          FillZeros (out_embedding, kDim);
          return;
        }

      // If the input is not 224x224, we will fallback for now (no resize).
      const int req_w = 224;
      const int req_h = 224;
      if (width != req_w || height != req_h)
        {
          FillZeros (out_embedding, kDim);
          return;
        }

      // Build CHW float input normalized to [0,1].
      const std::size_t chw = static_cast<std::size_t> (3 * req_h * req_w);
      std::vector<float> chw_data (chw, 0.0f);
      for (int y = 0; y < req_h; ++y)
        {
          for (int x = 0; x < req_w; ++x)
            {
              const int idx = (y * req_w + x) * channels;
              const std::uint8_t r = data[idx + 0];
              const std::uint8_t g = data[idx + 1];
              const std::uint8_t b = data[idx + 2];
              const std::size_t ofs = static_cast<std::size_t> (y * req_w + x);
              chw_data[0 * req_h * req_w + ofs]
                  = static_cast<float> (r) / 255.0f;
              chw_data[1 * req_h * req_w + ofs]
                  = static_cast<float> (g) / 255.0f;
              chw_data[2 * req_h * req_w + ofs]
                  = static_cast<float> (b) / 255.0f;
            }
        }

      std::vector<int64_t> input_shape{ 1, 3, req_h, req_w };
      Ort::MemoryInfo mem_info
          = Ort::MemoryInfo::CreateCpu (OrtDeviceAllocator, OrtMemTypeCPU);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float> (
          mem_info, chw_data.data (),
          static_cast<std::size_t> (chw_data.size ()), input_shape.data (),
          input_shape.size ());

      const char *in_names[] = { input_name.c_str () };
      const char *out_names[] = { output_name.c_str () };

      auto outputs = session.Run (Ort::RunOptions{ nullptr }, in_names,
                                  &input_tensor, 1, out_names, 1);
      if (outputs.size () != 1 || !outputs[0].IsTensor ())
        {
          FillZeros (out_embedding, kDim);
          return;
        }

      float *out_data = outputs[0].GetTensorMutableData<float> ();
      auto type_info = outputs[0].GetTensorTypeAndShapeInfo ();
      const auto out_dims = type_info.GetShape ();
      std::size_t out_size = 0;
      if (!out_dims.empty ())
        {
          out_size = 1;
          for (auto d : out_dims)
            {
              out_size *= static_cast<std::size_t> (d < 0 ? 1 : d);
            }
        }
      if (out_size == 0)
        {
          FillZeros (out_embedding, kDim);
          return;
        }
      out_embedding.assign (out_data, out_data + out_size);
      return;
    }
  catch (...)
    {
      // On any error, fall back to zeros.
      FillZeros (out_embedding, kDim);
      return;
    }
#else
  (void)data;
  (void)width;
  (void)height;
  (void)channels;
  FillZeros (out_embedding, kDim);
#endif
}

} // namespace cortext
