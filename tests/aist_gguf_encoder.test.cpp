#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cortext/models/aist_gguf_encoder.hpp>

#include <cmath>
#include <filesystem>
#include <vector>

namespace
{

std::filesystem::path
RepoRoot ()
{
  std::filesystem::path p = std::filesystem::current_path ();
  for (int i = 0; i < 6; ++i)
    {
      if (std::filesystem::exists (p / "CMakeLists.txt")
          && std::filesystem::exists (p / "models"))
        {
          return p;
        }
      if (!p.has_parent_path ())
        {
          break;
        }
      p = p.parent_path ();
    }
  return std::filesystem::current_path ().parent_path ();
}

std::filesystem::path
AaitModelPath ()
{
  const auto model = RepoRoot () / "models" / "AAIT-86M-GGUF"
                     / "AAIT-86M_q8_0.gguf";
  return model;
}

std::filesystem::path
EssAistModelPath ()
{
  return RepoRoot () / "models" / "ESS-AIST-81M-preview-GGUF"
         / "ESS-AIST-81M_q8_0.gguf";
}

std::filesystem::path
EsAistModelPath ()
{
  return RepoRoot () / "models" / "ES-AIST-81M-preview-GGUF"
         / "ES-AIST-81M_q8_0.gguf";
}

double
Norm (const std::vector<float> &v)
{
  double sum = 0.0;
  for (float value : v)
    {
      sum += static_cast<double> (value) * value;
    }
  return std::sqrt (sum);
}

} // namespace

TEST_CASE ("AAIT Matryoshka truncation normalizes output",
           "[aait][gguf][unit]")
{
  std::vector<float> values{ 3.0F, 4.0F, 0.0F, 10.0F };
  auto truncated = cortext::TruncateAaitMatryoshka (values, 2);
  REQUIRE (truncated.size () == 2);
  const double norm = std::sqrt (static_cast<double> (truncated[0])
                                     * truncated[0]
                                 + static_cast<double> (truncated[1])
                                       * truncated[1]);
  CHECK (std::abs (norm - 1.0) < 1.0e-5);
}

TEST_CASE ("AAIT action order is stable", "[aait][gguf][unit]")
{
  CHECK (cortext::AaitAnchorActionName (
             cortext::AaitAnchorAction::CreateAnchor)
         == "CREATE_ANCHOR");
  CHECK (cortext::AaitAnchorActionName (
             cortext::AaitAnchorAction::UpdateExistingAnchor)
         == "UPDATE_EXISTING_ANCHOR");
  CHECK (cortext::AaitAnchorActionName (
             cortext::AaitAnchorAction::SplitAnchor)
         == "SPLIT_ANCHOR");
  CHECK (cortext::AaitAnchorActionName (
             cortext::AaitAnchorAction::CloseAnchor)
         == "CLOSE_ANCHOR");
  CHECK (cortext::AaitAnchorActionName (cortext::AaitAnchorAction::Abstain)
         == "ABSTAIN");
}

TEST_CASE ("ES-AIST embedding views expose normalized signal slices",
           "[aait][gguf][unit]")
{
  std::vector<float> values (1536, 0.0F);
  values[0] = 3.0F;
  values[1] = 4.0F;
  values[512] = 6.0F;
  values[513] = 8.0F;
  values[1024] = 5.0F;
  values[1025] = 12.0F;

  const auto views = cortext::BuildEssAistEmbeddingViews (values);
  REQUIRE (views.semantic_768_key.size () == 768);
  REQUIRE (views.entity_key.size () == 768);
  REQUIRE (views.full_key.size () == 1536);
  REQUIRE (views.semantic_key.size () == 512);
  REQUIRE (views.subject_key.size () == 512);
  REQUIRE (views.event_key.size () == 512);
  REQUIRE (views.prefix_512.size () == 512);
  REQUIRE (views.prefix_768.size () == 768);
  REQUIRE (views.prefix_1024.size () == 1024);
  REQUIRE (views.prefix_1536.size () == 1536);

  CHECK (std::abs (Norm (views.semantic_768_key) - 1.0) < 1.0e-5);
  CHECK (std::abs (Norm (views.entity_key) - 1.0) < 1.0e-5);
  CHECK (std::abs (Norm (views.full_key) - 1.0) < 1.0e-5);
  CHECK (std::abs (Norm (views.semantic_key) - 1.0) < 1.0e-5);
  CHECK (std::abs (Norm (views.subject_key) - 1.0) < 1.0e-5);
  CHECK (std::abs (Norm (views.event_key) - 1.0) < 1.0e-5);
}

TEST_CASE ("AAIT GGUF metadata exposes anchor contract when model is present",
           "[aait][gguf][integration]")
{
  const auto model = AaitModelPath ();
  if (!std::filesystem::exists (model))
    {
      SKIP ("AAIT GGUF model not present");
    }

  const auto info = cortext::InspectAaitGgufModel (model);
  CHECK (info.architecture == "triembed");
  CHECK (info.n_embd == 1280);
  CHECK (info.n_embd_out == 1280);
  CHECK (info.has_semantic_vector);
  CHECK (info.has_anchor_key);
  CHECK (info.has_anchor_action_logits);
  CHECK (info.has_anchor_confidence);
  CHECK (info.has_salience_delta);
  CHECK (info.runtime_available);
  CHECK (info.runtime_status == "aait_native_triembed_runtime_available");
}

TEST_CASE ("AAIT encoder load executes native triembed inference",
           "[aait][gguf][integration]")
{
  const auto model = AaitModelPath ();
  if (!std::filesystem::exists (model))
    {
      SKIP ("AAIT GGUF model not present");
    }

  cortext::AaitGgufConfig config;
  config.model_path = model.string ();
  config.context_length = 64;
  cortext::AaitGgufEncoder encoder (config);
  CHECK (encoder.IsLoaded ());
  REQUIRE (encoder.IsRuntimeAvailable ());

  std::vector<float> embedding;
  encoder.EncodeText ("Jared came over yesterday.", embedding);
  REQUIRE (embedding.size () == 1280);
  double semantic_norm = 0.0;
  for (float value : embedding)
    {
      CHECK (std::isfinite (value));
      semantic_norm += static_cast<double> (value) * value;
    }
  CHECK (std::abs (std::sqrt (semantic_norm) - 1.0) < 1.0e-3);

  cortext::AaitIngressInput input;
  input.text = "Jared came over yesterday.";
  input.recent_context = "Jared is Gabe's best friend.";
  input.active_context = input.recent_context;
  const auto output = encoder.EncodeTextWithAnchors (input);
  CHECK (output.has_semantic_vector);
  CHECK (output.semantic_vector.size () == 1280);
  CHECK (output.has_anchor_key);
  CHECK (output.anchor_key.size () == 128);
  CHECK (output.has_anchor_action_logits);
  CHECK (output.anchor_action_logits.size () == 5);
  CHECK (output.has_anchor_confidence);
  CHECK (std::isfinite (output.anchor_confidence));
  CHECK (output.has_salience_delta);
  CHECK (std::isfinite (output.salience_delta));
}

TEST_CASE ("ESS-AIST GGUF executes embedding-only triembed runtime",
           "[aait][gguf][integration]")
{
  const auto model = EssAistModelPath ();
  if (!std::filesystem::exists (model))
    {
      SKIP ("ESS-AIST GGUF model not present");
    }

  const auto info = cortext::InspectAaitGgufModel (model);
  CHECK (info.architecture == "triembed");
  CHECK (info.n_embd == 1536);
  CHECK (info.n_embd_out == 1536);
  CHECK (info.has_semantic_vector);
  CHECK_FALSE (info.has_anchor_key);
  CHECK (info.runtime_available);

  cortext::AaitGgufConfig config;
  config.model_path = model.string ();
  config.context_length = 64;
  cortext::AaitGgufEncoder encoder (config);
  CHECK (encoder.IsLoaded ());
  REQUIRE (encoder.IsRuntimeAvailable ());

  std::vector<float> embedding;
  encoder.EncodeText ("Jared has a huge in-ground pool.", embedding);
  REQUIRE (embedding.size () == 1536);
  CHECK (encoder.UsesKernelOps ());
  CHECK (encoder.UsesFullTextGraphOps ());
  CHECK (encoder.KernelOpsGranularity () == "full_text_graph");
  const auto views = cortext::BuildEssAistEmbeddingViews (embedding);
  CHECK (views.semantic_key.size () == 512);
  CHECK (views.subject_key.size () == 512);
  CHECK (views.event_key.size () == 512);
  CHECK (views.full_key.size () == 1536);

  const auto output = encoder.EncodeTextWithAnchors (
      "Jared has a huge in-ground pool.");
  CHECK (output.has_semantic_vector);
  CHECK (output.semantic_vector.size () == 1536);
  CHECK_FALSE (output.has_anchor_key);
  CHECK_FALSE (output.has_anchor_action_logits);
}

TEST_CASE ("ES-AIST GGUF executes native text image and audio kernels",
           "[aait][gguf][integration][es-aist]")
{
  const auto model = EsAistModelPath ();
  if (!std::filesystem::exists (model))
    {
      SKIP ("ES-AIST GGUF model not present");
    }

  const auto info = cortext::InspectAaitGgufModel (model);
  CHECK (info.architecture == "triembed");
  CHECK (info.n_embd == 1536);
  CHECK (info.n_embd_out == 1536);
  CHECK (info.has_semantic_vector);
  CHECK_FALSE (info.has_anchor_key);
  CHECK (info.runtime_available);

  cortext::AaitGgufConfig config;
  config.model_path = model.string ();
  config.context_length = 64;
  cortext::AaitGgufEncoder encoder (config);
  CHECK (encoder.IsLoaded ());
  REQUIRE (encoder.IsRuntimeAvailable ());
  CHECK (encoder.UsesKernelOps ());

  std::vector<float> text_embedding;
  encoder.EncodeText ("Jared has a huge in-ground pool.", text_embedding);
  REQUIRE (text_embedding.size () == 1536);
  CHECK (std::abs (Norm (text_embedding) - 1.0) < 1.0e-3);
  const auto text_views = cortext::BuildEssAistEmbeddingViews (text_embedding);
  CHECK (text_views.semantic_768_key.size () == 768);
  CHECK (text_views.entity_key.size () == 768);

  constexpr int image_width = 32;
  constexpr int image_height = 24;
  constexpr int image_channels = 3;
  std::vector<std::uint8_t> image (
      static_cast<std::size_t> (image_width * image_height * image_channels),
      0);
  for (int y = 0; y < image_height; ++y)
    {
      for (int x = 0; x < image_width; ++x)
        {
          const std::size_t offset =
              static_cast<std::size_t> ((y * image_width + x)
                                        * image_channels);
          image[offset + 0] = static_cast<std::uint8_t> ((x * 255)
                                                        / image_width);
          image[offset + 1] = static_cast<std::uint8_t> ((y * 255)
                                                        / image_height);
          image[offset + 2] = 127;
        }
    }

  std::vector<float> image_embedding;
  encoder.EncodeImage (image.data (), image_width, image_height,
                       image_channels, image_embedding);
  REQUIRE (image_embedding.size () == 1536);
  CHECK (std::abs (Norm (image_embedding) - 1.0) < 1.0e-3);

  constexpr std::size_t audio_samples = 16000;
  std::vector<float> pcm (audio_samples, 0.0F);
  for (std::size_t i = 0; i < pcm.size (); ++i)
    {
      const double t = static_cast<double> (i) / 16000.0;
      pcm[i] = static_cast<float> (0.15 * std::sin (2.0 * 3.141592653589793
                                                    * 440.0 * t));
    }

  std::vector<float> audio_embedding;
  encoder.EncodeAudio (pcm.data (), pcm.size (), audio_embedding);
  REQUIRE (audio_embedding.size () == 1536);
  CHECK (std::abs (Norm (audio_embedding) - 1.0) < 1.0e-3);
}
