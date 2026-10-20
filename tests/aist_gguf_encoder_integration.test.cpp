// Compiled only when models/AIST-87M-GGUF is present (see CMakeLists.txt).
// Once built, every assertion is unconditional: resolution failure is a
// test failure, not a skip.

#include <catch2/catch_test_macros.hpp>

#include <cortext/models/aist_gguf_encoder.hpp>

#include "../src/models/aist_semantic_cache_internal.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
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
AistModelPath ()
{
  auto resolved = cortext::ResolveAistGgufModelPath (RepoRoot () / "models");
  REQUIRE (resolved.has_value ());
  return *resolved;
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

TEST_CASE ("AIST GGUF metadata exposes triembed contract",
           "[aist][gguf][integration]")
{
  const auto info = cortext::InspectAistGgufModel (AistModelPath ());
  CHECK (info.architecture == "triembed");
  CHECK (info.n_embd == 1280);
  CHECK (info.n_embd_out == 1280);
  CHECK (info.has_semantic_vector);
  CHECK (info.runtime_available);
  CHECK (info.runtime_status == "triembed_embedding_runtime_available");
}

TEST_CASE ("AIST GGUF executes native text image and audio kernels",
           "[aist][gguf][integration]")
{
  cortext::AistGgufConfig config;
  config.model_path = AistModelPath ().string ();
  config.context_length = 64;
  cortext::AistGgufEncoder encoder (config);
  CHECK (encoder.IsLoaded ());
  REQUIRE (encoder.IsRuntimeAvailable ());
  REQUIRE (encoder.UsesKernelOps ());
  CHECK (encoder.KernelOpsGranularity () != "none");

  std::vector<float> text_embedding;
  encoder.EncodeText ("Jared has a huge in-ground pool.", text_embedding);
  REQUIRE (text_embedding.size () == 1280);
  for (float value : text_embedding)
    {
      REQUIRE (std::isfinite (value));
    }
  CHECK (std::abs (Norm (text_embedding) - 1.0) < 1.0e-3);

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
  REQUIRE (image_embedding.size () == 1280);
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
  REQUIRE (audio_embedding.size () == 1280);
  CHECK (std::abs (Norm (audio_embedding) - 1.0) < 1.0e-3);
}

TEST_CASE ("AIST GGUF text graph memory stays bounded across lengths",
           "[aist][gguf][integration]")
{
  cortext::AistGgufConfig config;
  config.model_path = AistModelPath ().string ();
  config.context_length = 512;
  cortext::AistGgufEncoder encoder (config);
  REQUIRE (encoder.UsesKernelOps ());

  // Each text tokenizes to a distinct sequence length, as a real ingest
  // stream does. The longest goes first so it sizes the activation buffer.
  std::vector<std::string> texts;
  for (int words = 160; words >= 1; words -= 3)
    {
      std::string text;
      for (int w = 0; w < words; ++w)
        {
          text += (w % 2 == 0) ? "memory " : "garden ";
        }
      texts.push_back (text);
    }

  std::vector<float> embedding;
  encoder.EncodeText (texts.front (), embedding);
  const std::size_t after_longest
      = encoder.DebugRuntimeStats ().text_graph_buffer_bytes;
  REQUIRE (after_longest > 0);
  for (const std::string &text : texts)
    {
      encoder.EncodeText (text, embedding);
    }
  CHECK (encoder.DebugRuntimeStats ().text_graph_buffer_bytes
         == after_longest);

  // Shared activation memory must not leak state between lengths.
  std::vector<float> fresh;
  cortext::AistGgufEncoder fresh_encoder (config);
  fresh_encoder.EncodeText (texts[texts.size () / 2], fresh);
  encoder.EncodeText (texts.back () + "pool", embedding);
  encoder.EncodeText (texts[texts.size () / 2] + " ", embedding);
  REQUIRE (embedding.size () == fresh.size ());
  double max_abs_diff = 0.0;
  for (std::size_t i = 0; i < fresh.size (); ++i)
    {
      max_abs_diff = std::max (
          max_abs_diff,
          static_cast<double> (std::abs (embedding[i] - fresh[i])));
    }
  CHECK (max_abs_diff < 1.0e-5);
}

TEST_CASE ("AIST GGUF semantic cache stays within its byte budget",
           "[aist][gguf][integration]")
{
  cortext::AistGgufConfig config;
  config.model_path = AistModelPath ().string ();
  config.context_length = 64;
  cortext::AistGgufEncoder encoder (config);

  const std::string probe = "probe text that is encoded first";
  std::vector<float> embedding;
  encoder.EncodeText (probe, embedding);
  encoder.EncodeText (probe, embedding);
  REQUIRE (encoder.DebugRuntimeStats ().semantic_cache_hits == 1);

  // Fill past the budget with distinct texts; each entry costs at least its
  // embedding plus bookkeeping, so the probe must age out.
  using Cache = cortext::internal::AistSemanticCache;
  const std::size_t per_entry = Cache::EntryBytes (probe, embedding);
  const std::size_t fill = Cache::kDefaultCapacityBytes / per_entry + 2;
  for (std::size_t i = 0; i < fill; ++i)
    {
      encoder.EncodeText ("filler " + std::to_string (i), embedding);
    }
  const std::vector<float> before = embedding;
  encoder.EncodeText (probe, embedding);
  const auto stats = encoder.DebugRuntimeStats ();
  CHECK (stats.semantic_cache_hits == 1);
  CHECK (stats.semantic_cache_misses == fill + 2);
  CHECK (stats.semantic_requests == fill + 3);

  // The most recent text is still a hit.
  encoder.EncodeText ("filler " + std::to_string (fill - 1), embedding);
  CHECK (encoder.DebugRuntimeStats ().semantic_cache_hits == 2);
  CHECK (embedding == before);
}

TEST_CASE ("AIST native image tower output is pinned",
           "[aist][gguf][integration]")
{
  // Values recorded from the scalar single-threaded native image path; the
  // vectorized, row-parallel kernels must reproduce them.
  cortext::AistGgufConfig config;
  config.model_path = AistModelPath ().string ();
  cortext::AistGgufEncoder encoder (config);
  REQUIRE (encoder.IsRuntimeAvailable ());
  std::vector<std::uint8_t> pixels (768 * 512 * 3);
  for (std::size_t i = 0; i < pixels.size (); ++i)
    pixels[i] = static_cast<std::uint8_t> ((i * 2654435761u) >> 24);
  std::vector<float> v;
  encoder.EncodeImage (pixels.data (), 768, 512, 3, v);
  REQUIRE (v.size () == 1280);
  double sum = 0.0;
  for (float x : v)
    sum += x;
  CHECK (std::abs (v[0] - 0.0423770249) < 1.0e-6);
  CHECK (std::abs (v[1] - 0.0298241284) < 1.0e-6);
  CHECK (std::abs (sum - -0.047651531) < 1.0e-5);
}

namespace
{

// Last hidden states of tokens 1 and 10 of "The playlist is called Summer
// Vibes." from tools/reranker/aist_torch.py (fp32 PyTorch mirror).
const float kMirrorToken1[384] = {
    -0.03158F, -0.04251F, -0.00642F, -0.01585F, 0.04104F, 0.05901F, -0.03178F, 0.00041F,
    0.00773F, -0.02024F, -0.01730F, -0.06211F, -0.00256F, -0.02358F, 0.04406F, -0.03012F,
    0.04468F, -0.02351F, -0.02354F, 0.01368F, 0.02268F, -0.03879F, -0.02659F, -0.01199F,
    0.00672F, -0.02402F, 0.00532F, -0.01638F, 0.04582F, 0.02360F, -0.01771F, 0.01005F,
    0.04324F, 0.00121F, -0.04356F, -0.00271F, -0.00745F, -0.01633F, 0.03257F, -0.03047F,
    -0.00070F, 0.02437F, -0.01425F, 0.00768F, 0.00645F, 0.00817F, -0.04646F, 0.03138F,
    -0.01230F, 0.00280F, 0.04548F, -0.00913F, -0.00022F, 0.01269F, 0.03625F, 0.01403F,
    0.00369F, 0.03885F, -0.03477F, -0.02385F, 0.00357F, 0.01320F, -0.12360F, 0.04261F,
    -0.04829F, 0.00132F, -0.02089F, 0.00940F, 0.00139F, 0.04912F, -0.02649F, -0.02697F,
    -0.00706F, 0.05168F, -0.00180F, 0.00132F, 0.05384F, -0.03805F, -0.01788F, -0.02655F,
    -0.00511F, -0.07158F, 0.01046F, -0.02241F, 0.00958F, -0.02625F, 0.02986F, -0.01602F,
    0.00001F, 0.00713F, -0.02963F, -0.01158F, -0.06131F, 0.01941F, -0.00272F, 0.00540F,
    0.00033F, -0.00851F, 0.07741F, -0.02933F, -0.01402F, 0.01628F, 0.01302F, 0.00042F,
    0.01454F, -0.00587F, 0.02803F, -0.00016F, 0.03530F, -0.01056F, 0.02910F, -0.03038F,
    0.01480F, 0.00702F, 0.08646F, -0.06096F, 0.01464F, 0.03983F, 0.01457F, -0.02852F,
    -0.00055F, 0.00197F, 0.01267F, 0.02584F, 0.02706F, 0.03129F, -0.00046F, 0.03877F,
    0.02370F, -0.02933F, -0.04924F, -0.01456F, 0.09256F, -0.03712F, -0.04839F, -0.00163F,
    0.00223F, -0.00906F, -0.00923F, -0.04250F, -0.06194F, 0.03033F, 0.02190F, -0.01229F,
    0.00567F, -0.00532F, -0.00353F, -0.00204F, 0.02835F, 0.05599F, 0.02397F, -0.04687F,
    -0.00816F, -0.03069F, -0.02319F, 0.01045F, -0.02485F, 0.11375F, -0.03248F, 0.00945F,
    0.02142F, -0.01821F, 0.00497F, -0.02603F, 0.01710F, -0.06963F, -0.02017F, 0.02312F,
    0.07348F, 0.00088F, -0.05306F, 0.02008F, -0.06606F, 0.02976F, 0.04830F, -0.01276F,
    -0.01351F, 0.00671F, 0.01983F, -0.00283F, -0.01468F, -0.01857F, -0.00740F, -0.00827F,
    -0.01472F, 0.01428F, -0.00417F, -0.02325F, -0.02239F, -0.01984F, 0.01611F, -0.01578F,
    0.02023F, 0.03449F, 0.03226F, -0.06120F, 0.06273F, 0.02488F, -0.03010F, -0.01007F,
    0.00355F, 0.03911F, 0.02254F, 0.00727F, -0.05255F, -0.02147F, -0.02505F, 0.01149F,
    0.00439F, -0.05193F, -0.01429F, 0.00308F, -0.00473F, -0.03358F, 0.00263F, -0.01137F,
    -0.02564F, -0.05210F, -0.02453F, -0.02495F, 0.04128F, -0.00596F, -0.00788F, 0.08605F,
    0.01867F, 0.01416F, 0.00422F, 0.03620F, 0.03535F, -0.00474F, -0.00204F, 0.04100F,
    0.03695F, 0.00624F, -0.03671F, -0.05853F, 0.00544F, -0.01158F, 0.02060F, 0.00977F,
    -0.00319F, 0.02207F, -0.00691F, 0.00104F, 0.00711F, -0.04210F, 0.01436F, 0.02656F,
    -0.00527F, 0.01080F, 0.03799F, 0.03312F, 0.02021F, -0.01933F, 0.04251F, 0.03582F,
    -0.00634F, -0.07272F, 0.01622F, -0.02368F, -0.02019F, 0.00860F, -0.03097F, -0.00105F,
    0.00889F, -0.03572F, -0.01817F, 0.19428F, -0.02467F, 0.02737F, -0.05187F, -0.00586F,
    -0.01642F, 0.01176F, -0.00479F, 0.00326F, -0.02001F, -0.02069F, -0.02451F, 0.01043F,
    -0.04575F, -0.00240F, 0.01338F, -0.01873F, -0.00324F, 0.00705F, 0.03524F, -0.00837F,
    -0.02539F, 0.02496F, -0.00244F, 0.03093F, -0.01605F, -0.00006F, -0.03668F, -0.02360F,
    -0.01136F, -0.01784F, 0.01309F, 0.04249F, 0.01457F, 0.01130F, 0.02885F, -0.07131F,
    -0.01006F, 0.01877F, -0.07712F, -0.05620F, 0.01518F, -0.03523F, 0.01196F, -0.00373F,
    -0.00812F, -0.01599F, 0.03698F, 0.02997F, -0.04196F, 0.03007F, -0.00682F, 0.02083F,
    -0.01280F, 0.04397F, 0.00646F, -0.03760F, 0.02240F, 0.03060F, 0.02992F, 0.02860F,
    -0.00115F, -0.01072F, 0.03131F, 0.00589F, 0.00199F, 0.02004F, 0.03800F, 0.01592F,
    0.02930F, 0.02784F, -0.08811F, -0.04089F, -0.04171F, 0.05079F, 0.00001F, -0.06480F,
    0.00542F, -0.01060F, 0.02072F, -0.05008F, -0.02043F, -0.03693F, 0.02961F, -0.00936F,
    -0.01260F, -0.04559F, -0.02086F, 0.06071F, 0.00117F, 0.01887F, 0.01548F, -0.02700F,
    -0.00430F, 0.00921F, -0.00858F, 0.05251F, -0.03800F, -0.01510F, -0.00786F, -0.02913F,
    0.03647F, -0.01906F, -0.02343F, -0.05673F, -0.01482F, 0.00649F, 0.06897F, -0.01339F,
    -0.02785F, -0.01848F, 0.00123F, -0.02322F, 0.02926F, 0.01722F, 0.04320F, 0.02144F,
};

const float kMirrorLast[384] = {
    -0.00640F, -0.01228F, -0.02119F, 0.00673F, -0.00380F, 0.04312F, 0.02317F, -0.05253F,
    -0.02974F, -0.01037F, -0.01742F, 0.00074F, 0.01838F, 0.00626F, -0.01964F, -0.01672F,
    -0.00023F, -0.02368F, -0.03197F, 0.01414F, -0.02909F, -0.01141F, -0.00955F, -0.00240F,
    0.02782F, 0.01960F, 0.02215F, -0.00438F, -0.05721F, 0.02895F, 0.03439F, -0.02164F,
    0.02904F, 0.02511F, -0.02956F, -0.01531F, 0.01106F, -0.01953F, 0.00543F, -0.00486F,
    -0.00330F, -0.03288F, 0.00904F, 0.02183F, 0.03463F, 0.01083F, -0.02251F, 0.00859F,
    -0.02152F, -0.04717F, -0.02464F, 0.02779F, 0.00834F, -0.03639F, -0.01802F, 0.01600F,
    -0.03372F, 0.01511F, 0.01451F, -0.00566F, 0.00626F, -0.01912F, -0.02258F, -0.03090F,
    0.02240F, -0.03979F, 0.01349F, 0.03137F, 0.02209F, -0.02175F, 0.00706F, -0.02560F,
    -0.00039F, -0.01173F, 0.00962F, -0.01985F, 0.04141F, 0.01564F, -0.01326F, 0.00326F,
    -0.01284F, -0.01124F, 0.00229F, -0.01313F, -0.00375F, -0.00834F, 0.00268F, 0.00734F,
    -0.03833F, 0.04989F, 0.04595F, 0.01431F, 0.01338F, -0.04360F, 0.03857F, -0.01145F,
    0.04089F, 0.03368F, 0.02864F, -0.00668F, -0.03721F, -0.00326F, 0.02619F, 0.03265F,
    -0.03890F, -0.02227F, 0.02141F, 0.03960F, -0.03400F, 0.00323F, -0.01014F, 0.00727F,
    0.02522F, 0.01267F, 0.02402F, -0.00938F, 0.02849F, -0.01314F, -0.01571F, -0.04270F,
    0.00153F, 0.04704F, -0.00647F, -0.01771F, -0.02632F, 0.00353F, 0.00235F, 0.21252F,
    0.01411F, -0.00928F, -0.01046F, 0.00861F, 0.00857F, -0.00912F, 0.04708F, -0.03141F,
    -0.02092F, 0.00815F, -0.01087F, 0.03654F, -0.02622F, 0.00653F, 0.02990F, 0.01169F,
    0.02411F, -0.04392F, -0.00589F, 0.02137F, 0.02989F, 0.03366F, 0.03308F, 0.01524F,
    -0.01050F, -0.01476F, 0.02585F, 0.00698F, 0.05289F, 0.04136F, -0.00057F, -0.03878F,
    0.00185F, 0.03721F, 0.01901F, 0.01515F, -0.00408F, -0.01985F, -0.00284F, -0.01322F,
    0.00764F, -0.02771F, -0.01240F, 0.02910F, 0.00023F, 0.01318F, 0.00855F, 0.01075F,
    -0.02591F, 0.01003F, -0.03485F, -0.00081F, -0.00522F, 0.00285F, -0.01212F, 0.03891F,
    0.00825F, 0.03706F, 0.00683F, -0.01288F, 0.02835F, -0.00233F, 0.00637F, -0.02637F,
    0.00456F, -0.00072F, 0.01366F, 0.02048F, 0.01791F, -0.04688F, -0.00523F, 0.01747F,
    0.02590F, -0.00325F, -0.02911F, -0.00868F, -0.02048F, -0.02490F, 0.00536F, 0.01466F,
    0.01270F, 0.01509F, -0.00143F, 0.01811F, 0.02920F, -0.03080F, -0.03676F, 0.02091F,
    0.04290F, 0.00113F, 0.02315F, -0.01719F, 0.00916F, 0.01168F, -0.02623F, 0.34119F,
    0.03315F, -0.03767F, 0.01951F, 0.03055F, -0.02647F, -0.02446F, 0.05262F, 0.00971F,
    -0.04446F, -0.01511F, 0.00745F, 0.01125F, -0.01555F, 0.00614F, 0.00080F, -0.00051F,
    -0.00658F, 0.01183F, 0.01151F, -0.01451F, -0.02691F, -0.02822F, 0.01686F, -0.00040F,
    -0.04879F, -0.00338F, 0.02453F, -0.01240F, 0.03202F, 0.01456F, -0.00007F, -0.00886F,
    0.01523F, -0.02020F, 0.01642F, -0.00384F, -0.00180F, 0.01669F, 0.01201F, 0.01726F,
    -0.01166F, -0.00795F, 0.00280F, -0.03156F, -0.00774F, 0.02873F, -0.01496F, 0.01319F,
    -0.02767F, 0.00458F, 0.00525F, -0.01743F, -0.03046F, 0.00398F, -0.04319F, -0.00931F,
    0.00965F, 0.01152F, 0.00544F, -0.03270F, -0.01154F, -0.01251F, -0.01904F, -0.01956F,
    0.00882F, -0.02975F, 0.03303F, -0.01387F, 0.05489F, -0.01719F, 0.02869F, -0.03566F,
    0.00072F, -0.00818F, 0.00229F, 0.00581F, 0.00724F, -0.03572F, -0.01970F, -0.01334F,
    0.01793F, -0.02477F, -0.01634F, -0.01587F, 0.01433F, -0.00963F, 0.01414F, -0.01091F,
    0.00590F, 0.02073F, -0.00116F, 0.01586F, -0.03121F, 0.00138F, -0.01042F, 0.01057F,
    0.02280F, -0.00075F, 0.01038F, -0.06283F, -0.00243F, 0.01314F, -0.00283F, 0.00034F,
    0.05116F, 0.02786F, 0.01584F, -0.03720F, -0.01073F, -0.01917F, -0.01275F, 0.01817F,
    -0.01220F, 0.00911F, 0.04132F, 0.00637F, 0.00087F, 0.02531F, 0.00578F, -0.04097F,
    0.02080F, -0.01622F, -0.00231F, -0.02073F, -0.01147F, -0.02356F, 0.00194F, -0.01046F,
    -0.02717F, 0.04550F, -0.04399F, -0.03795F, -0.01900F, -0.02936F, -0.06415F, 0.05314F,
    -0.00410F, 0.01458F, 0.00071F, -0.03053F, -0.03394F, -0.01793F, -0.00374F, 0.01148F,
    0.03084F, 0.01523F, 0.02871F, -0.06141F, -0.03467F, -0.00690F, 0.00274F, -0.01690F,
    -0.02668F, 0.02603F, -0.00034F, 0.04278F, -0.01096F, 0.02273F, 0.01894F, -0.01570F,
};

double
MirrorCosine (const float *got, const float *want)
{
  double dot = 0.0, a = 0.0, b = 0.0;
  for (int i = 0; i < 384; ++i)
    {
      dot += static_cast<double> (got[i]) * want[i];
      a += static_cast<double> (got[i]) * got[i];
      b += static_cast<double> (want[i]) * want[i];
    }
  return dot / std::sqrt (a * b);
}

} // namespace

TEST_CASE ("AIST position features match the training mirror",
           "[aist][gguf][integration][late_interaction]")
{
  // Values from tools/reranker/aist_torch.py (PyTorch mirror the
  // late-interaction reranker trains on).
  cortext::AistGgufConfig config;
  config.model_path = AistModelPath ().string ();
  cortext::AistGgufEncoder encoder (config);
  REQUIRE (encoder.IsRuntimeAvailable ());

  std::vector<float> plain;
  std::vector<float> with_positions;
  cortext::EncoderPositions text;
  const std::string sentence = "The playlist is called Summer Vibes.";
  encoder.EncodeText (sentence, plain);
  encoder.EncodeTextPositions (sentence, with_positions, text);
  // positions come from the full ggml text graph, not the Eigen fallback
  INFO ("full graph error: " << encoder.FullTextGraphError ());
  CHECK (encoder.FullTextGraphError ().empty ());
  REQUIRE (with_positions.size () == plain.size ());
  for (std::size_t i = 0; i < plain.size (); ++i)
    CHECK (std::abs (with_positions[i] - plain[i]) < 1.0e-5);
  REQUIRE (text.tower == "text");
  REQUIRE (text.count == 11);
  REQUIRE (text.dim == 384);
  // Whole 384-d token states against the mirror: cosine absorbs per-element
  // q8 kernel drift across CPUs (up to ~6e-3 on x86 AVX2 vs arm64) but drops
  // far below the bound for a wrong layer, head or projection.
  CHECK (MirrorCosine (text.values.data () + 384, kMirrorToken1) > 0.99);
  CHECK (MirrorCosine (text.values.data () + 10 * 384, kMirrorLast) > 0.99);

  std::vector<std::uint8_t> pixels (768 * 512 * 3);
  for (std::size_t i = 0; i < pixels.size (); ++i)
    pixels[i] = static_cast<std::uint8_t> ((i * 2654435761u) >> 24);
  cortext::EncoderPositions image;
  std::vector<float> v;
  encoder.EncodeImagePositions (pixels.data (), 768, 512, 3, v, image);
  REQUIRE (v.size () == 1280);
  CHECK (std::abs (v[0] - 0.0423770249) < 1.0e-6);
  REQUIRE (image.tower == "image");
  REQUIRE (image.count == 144);
  REQUIRE (image.dim == 960);
  double total = 0.0;
  double patch70 = 0.0;
  for (std::size_t i = 0; i < image.values.size (); ++i)
    {
      total += image.values[i];
      if (i / 960 == 70)
        patch70 += image.values[i];
    }
  CHECK (std::abs (total - 91.2225) < 2.0e-2);
  CHECK (std::abs (patch70 - 0.052535) < 1.0e-3);
}
