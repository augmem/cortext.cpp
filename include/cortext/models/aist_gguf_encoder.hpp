#pragma once

#include "cortext/encoder/encoder.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cortext
{

struct AistGgufConfig
{
  std::string model_path;
  bool shadow_only = true;
  bool use_semantic_for_retrieval = false;
  int n_gpu_layers = 999;
  int threads = 0;
  int context_length = 512;
  std::string tokenizer_gguf_path;
  std::string runtime = "gguf";
};

struct AistTensorInfo
{
  std::string name;
  std::vector<std::uint64_t> shape;
  std::uint32_t ggml_type = 0;
  std::uint64_t offset = 0;
};

struct AistModelInfo
{
  std::string model_path;
  std::string architecture;
  std::vector<std::string> metadata_keys;
  std::vector<std::string> tensor_names;
  std::vector<AistTensorInfo> tensors;
  int n_embd = 0;
  int n_embd_out = 0;
  std::string pooling_type;
  std::vector<std::string> supported_modalities;
  bool has_semantic_vector = false;
  bool runtime_available = false;
  std::string runtime_status;
  std::string runtime_error;
};

struct AistRuntimeStats
{
  std::size_t semantic_requests = 0;
  std::size_t semantic_cache_hits = 0;
  std::size_t semantic_cache_misses = 0;
};

std::optional<std::filesystem::path>
ResolveAistGgufModelPath (const std::filesystem::path &models_dir,
                          const std::filesystem::path &override_path = {});

AistModelInfo InspectAistGgufModel (const std::filesystem::path &model_path);

void NormalizeL2InPlace (std::vector<float> &values);

std::vector<float> TruncateAistMatryoshka (const std::vector<float> &values,
                                           std::size_t dim);

std::vector<float> SliceAistEmbedding (const std::vector<float> &values,
                                       std::size_t begin, std::size_t end);

/// @brief Native runtime for the AIST triembed GGUF encoder.
///
/// AIST is Cortext's required embedding model: a multimodal encoder placing
/// text, image, and audio in one retrieval space. The triembed GGUF export
/// runs on this built-in tensor runtime; no external inference library is
/// involved.
class AistGgufEncoder : public Encoder
{
public:
  AistGgufEncoder ();
  explicit AistGgufEncoder (AistGgufConfig config);
  ~AistGgufEncoder () override;

  bool Load (const AistGgufConfig &config);
  const AistModelInfo &Inspect () const;
  bool IsLoaded () const;
  bool IsRuntimeAvailable () const;
  bool UsesKernelOps () const;
  bool UsesFullTextGraphOps () const;
  std::string KernelOpsBackend () const;
  std::string KernelOpsGranularity () const;
  std::string FullTextGraphError () const;

  std::vector<int> DebugTokenizeText (const std::string &text) const;
  AistRuntimeStats DebugRuntimeStats () const;

  void EncodeText (const std::string &text,
                   std::vector<float> &out_embedding) override;
  void EncodeAudio (const float *pcm, std::size_t num_samples,
                    std::vector<float> &out_embedding) override;
  void EncodeImage (const std::uint8_t *data, int width, int height,
                    int channels, std::vector<float> &out_embedding) override;

private:
  class NativeRuntime;

  AistGgufConfig config_;
  AistModelInfo info_;
  std::unique_ptr<NativeRuntime> runtime_;
  bool loaded_ = false;
};

} // namespace cortext
