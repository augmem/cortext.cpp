#pragma once

#include "cortext/encoder/encoder.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cortext
{

enum class AaitAnchorAction
{
  CreateAnchor = 0,
  UpdateExistingAnchor = 1,
  SplitAnchor = 2,
  CloseAnchor = 3,
  Abstain = 4,
};

struct AaitGgufConfig
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

struct AaitTensorInfo
{
  std::string name;
  std::vector<std::uint64_t> shape;
  std::uint32_t ggml_type = 0;
  std::uint64_t offset = 0;
};

struct AaitModelInfo
{
  std::string model_path;
  std::string architecture;
  std::vector<std::string> metadata_keys;
  std::vector<std::string> tensor_names;
  std::vector<AaitTensorInfo> tensors;
  int n_embd = 0;
  int n_embd_out = 0;
  std::string pooling_type;
  std::vector<std::string> supported_modalities;
  bool has_semantic_vector = false;
  bool has_anchor_key = false;
  bool has_anchor_action_logits = false;
  bool has_anchor_confidence = false;
  bool has_salience_delta = false;
  bool has_bind_logits = false;
  bool runtime_available = false;
  std::string runtime_status;
  std::string runtime_error;
};

struct AaitOutput
{
  std::vector<float> semantic_vector;
  std::vector<float> anchor_key;
  std::vector<float> anchor_action_logits;
  std::vector<float> bind_logits;
  float anchor_confidence = 0.0F;
  float salience_delta = 0.0F;
  bool has_semantic_vector = false;
  bool has_anchor_key = false;
  bool has_anchor_action_logits = false;
  bool has_bind_logits = false;
  bool has_anchor_confidence = false;
  bool has_salience_delta = false;
};

struct AaitIngressInput
{
  std::string text;
  std::string recent_context;
  std::string active_context;
  std::vector<float> semantic_vector;
  std::vector<float> recent_context_vector;
  std::vector<float> active_anchor_state;
  std::vector<std::vector<float>> candidate_semantic;
  std::vector<std::array<float, 7>> candidate_features;
  int modality_id = 0;
  float time_delta = 0.0F;
  int source_id = 0;
};

struct AaitRuntimeStats
{
  std::size_t semantic_requests = 0;
  std::size_t semantic_cache_hits = 0;
  std::size_t semantic_cache_misses = 0;
};

struct EssAistEmbeddingViews
{
  std::vector<float> semantic_key;
  std::vector<float> subject_key;
  std::vector<float> event_key;
  std::vector<float> semantic_768_key;
  std::vector<float> entity_key;
  std::vector<float> full_key;
  std::vector<float> prefix_512;
  std::vector<float> prefix_768;
  std::vector<float> prefix_1024;
  std::vector<float> prefix_1536;
};

std::string AaitAnchorActionName (AaitAnchorAction action);

std::optional<std::filesystem::path>
ResolveAaitGgufModelPath (const std::filesystem::path &models_dir,
                          const std::filesystem::path &override_path = {});

std::optional<std::filesystem::path>
ResolveEssAistGgufModelPath (const std::filesystem::path &models_dir,
                             const std::filesystem::path &override_path = {});

AaitModelInfo InspectAaitGgufModel (const std::filesystem::path &model_path);

void NormalizeL2InPlace (std::vector<float> &values);

std::vector<float> TruncateAaitMatryoshka (const std::vector<float> &values,
                                           std::size_t dim);

std::vector<float> SliceAaitEmbedding (const std::vector<float> &values,
                                       std::size_t begin, std::size_t end);

EssAistEmbeddingViews BuildEssAistEmbeddingViews (
    const std::vector<float> &embedding);

class AaitGgufEncoder : public Encoder
{
public:
  AaitGgufEncoder ();
  explicit AaitGgufEncoder (AaitGgufConfig config);
  ~AaitGgufEncoder () override;

  bool Load (const AaitGgufConfig &config);
  const AaitModelInfo &Inspect () const;
  bool IsLoaded () const;
  bool IsRuntimeAvailable () const;
  bool UsesKernelOps () const;
  bool UsesFullTextGraphOps () const;
  std::string KernelOpsBackend () const;
  std::string KernelOpsGranularity () const;
  std::string FullTextGraphError () const;

  AaitOutput EncodeTextWithAnchors (const std::string &text);
  AaitOutput EncodeTextWithAnchors (const AaitIngressInput &input);
  std::vector<int> DebugTokenizeText (const std::string &text) const;
  AaitRuntimeStats DebugRuntimeStats () const;

  void EncodeText (const std::string &text,
                   std::vector<float> &out_embedding) override;
  void EncodeAudio (const float *pcm, std::size_t num_samples,
                    std::vector<float> &out_embedding) override;
  void EncodeImage (const std::uint8_t *data, int width, int height,
                    int channels, std::vector<float> &out_embedding) override;

private:
  class NativeRuntime;

  AaitGgufConfig config_;
  AaitModelInfo info_;
  std::unique_ptr<NativeRuntime> runtime_;
  bool loaded_ = false;
};

} // namespace cortext
